# M6B Ladder Cumulative Audit Memo (M6B0-R + M6B1 + M6B2)

Auditor: opus reviewer, branch `tester/opus/m6b-ladder-cumulative-audit`, worktree `/tmp/wrf_gpu2_ladderaudit`.
Operational WRF SHA (post-audit, unchanged): `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37` (see `proof_operational_sha256.txt`).

## Part 1 — Cumulative schema integrity

- `SCHEMA_VERSION = "m6b0r-savepoint-v1"` has not been bumped despite three successive extensions of `VALID_BOUNDARIES`, `VALID_OPERATORS`, and the tolerance ladder. M6B1 added `advance_mu_t` + `advance_mu_t_pre/post`; M6B2 added `advance_w` + 4 new tridiag boundaries and 6 new tri_* fields. From the schema's own enforcement, `__post_init__` will reject any savepoint with `schema_version != SCHEMA_VERSION`, so old M6B0 savepoints in any persistence store would now silently still pass the version gate but with a stale field set. Recommend bumping to `m6b2-savepoint-v2` (or per-operator suffixed version) before M6B3 lands more fields.
- Tolerance ladder `schema_version = "m6b0r-tolerance-ladder-v1"` similarly unbumped.
- No field-name collisions in `tolerance_ladder.json` (24 unique keys). Coef-w fields (`a/alpha/gamma`) are intentionally distinct from Thomas fields (`tri_a/tri_alpha/tri_gamma`); both tracks tolerance entries at `abs=1e-11`, consistent.
- Pre-ladder legacy fields `cofrz/cofwr/cofwz/coftz/cofwt/rdzw` (M6B0 builder lookup) still live in the ladder. Currently consumed only by `scripts/diagnostic_first_bad_step_tracer.py`. Not strictly stale, but they live alongside the new ladder without provenance comment. Document in a follow-up cleanup.
- Tolerances tighter-than 1e-11 (per Stage-5 directive): all coef and Thomas-coefficient fields are at `abs=1e-11`. Laxer entries (intentionally, because the field's natural Pa- or m/s-scale would otherwise be impossible to meet):
  - `muts`, `mu`, `muave` — `abs=1e-8` (Pa-scale state).
  - `ww` — `abs=1e-9` (Pa s-1 vertical mass flux).
  - `mudf`, `theta`, `ph_tend` — `abs=1e-10`.
  - `tri_rhs`, `tri_fwd`, `tri_solution` — `abs=1e-10` (m s-1 w-component).
  All relative tolerances are `rel=1e-12`, ULP `8` (16 for `muts` with documented accumulation exception). No tolerance contradicts another. Per Critic Amendment #4 (precision fail-closed), `dtype="float64"` everywhere in the ladder — good.
- `acoustic_substep_index` and `rk_stage_index` enforce non-negative but allow `0`, and M6B0 alias boundaries (`coefficient_construction`, `acoustic_substep_start/end`, `rk_stage_end`) still live in `VALID_BOUNDARIES`. Not harmful but indicates accreting boundary set without a deprecation pass.

## Part 2 — `solve_em.F.patch` quality + hook ABI

The cumulative patch DOES NOT apply cleanly. Three quality defects:

1. Hunk 1 header `@@ -53,6 +53,15 @@` has wrong new-side count (should be `+53,16`: 3 context + 10 added + 3 context = 16). Standard `patch -p1` fails with `malformed patch at line 19` (see `proof_patch_dryrun.txt`, RC=2).
2. Three bare `@@` markers at patch lines 60, 65, 73 with no offsets/counts. Standard tooling cannot apply them.
3. M6B2 appended Thomas-solve hunks (`@@ -1533,...`, `@@ -1536,...`, `@@ -1546,...`) into `solve_em.F.patch`, but these target `module_small_step_em.F` (the file containing `advance_w`'s Thomas sweep at lines 1533-1550, confirmed in source tree). The patch header still says `--- a/dyn_em/solve_em.F`, so these hunks can never apply to the right file. Either split into `module_small_step_em.F.patch` or relabel.

The CPU vs GPU branch instrumentation IS consistently CPU-only per ADR-025 — both `#ifdef GPU_OPENACC` paths and the `#else` CPU path have `sp_calc_coef_w_pre/post` brackets. Good intent, but unverifiable until #1–#3 above are fixed.

Hook ABI status (latent M6B0-R RELINK bug carried forward):

| Hook | Args | Status |
|---|---|---|
| `sp_calc_coef_w_pre/post` | 0 | LATENT — cannot emit field arrays |
| `sp_small_step_prep_post`, `sp_advance_uv_post`, `sp_advance_w_rhs_ready`, `sp_advance_w_raw_w`, `sp_advance_w_rayleigh`, `sp_advance_w_ph_final`, `sp_calc_p_rho_post`, `sp_small_step_finish_post`, `sp_acoustic_substep_boundary`, `sp_rk_stage_boundary` | 0 | LATENT — same |
| `sp_advance_mu_t_pre/post` | typed (mu, mut, mudf, muts, muave, ww, theta, [ph_tend]) | ABI TYPED but bodies empty |
| `sp_advance_w_tridiag_fwd_pre/post`, `sp_advance_w_tridiag_back_pre/post` | typed (a/alpha/gamma/rhs or gamma/w) | ABI TYPED but bodies empty |

Net: even where M6B1/M6B2 typed the ABI, the wrapper file is still a shim (`program wrf_savepoint_instrumented` followed by `module savepoint_wrapper` with `contains` of empty subroutines). No in-timestep HDF5 emission happens; extraction is still Python/JAX transcription over wrfout slices. Each worker-report flagged this in "unresolved risks." The follow-up sprint `2026-05-25-m6b0r-fortran-hook-abi-followup` is queued. Not new news, but flag it: the patch's malformedness means it would not be applied today even if hook bodies were filled in.

## Part 3 — Helper code drift

Three new helpers, three different styles:

- `acoustic_wrf.py::calc_coef_w_wrf_coefficients` (lines 598–663): WRF-shaped, eta-coordinate hybrid mass denominators with Python loops over `k` doing `.at[kk].set(...)`. Lives next to legacy `_calc_coef_w` (ADR-023 runtime) which is unchanged.
- `mu_t_advance.py::advance_mu_t_wrf`: dataclass-of-inputs API, Python `for k in range(nz)` loop building per-level intermediates with `jnp.stack`, then JAX index-add.
- `tridiag_solve.py::thomas_forward_scan/back_scan/solve_scan`: stand-alone `jax.lax.scan` functions; takes raw arrays, no dataclass.

Per-WRF-citation pattern: all three cite WRF source lines in docstrings. Consistent in spirit.

Shared math that could be extracted (low priority; validation-only): mass denominators `mass_h = c1h*mut + c2h`, `mass_f = c1f*mut + c2f` are constructed in `calc_coef_w_wrf_coefficients` and re-constructed implicitly in M6B2's `_wrf_calc_coef_w` (in `m6b0r_wrf_savepoint_extract`, imported by m6b2). The duplication is in extractor (NumPy) vs helper (JAX) — defensible since one is the "WRF transcription" expected value and the other the "JAX under test" actual value. Recommend M6B3 not extract these into a shared helper, otherwise the same code generates both sides of the comparison and trivializes parity.

Operational wire-in check (validation-only invariant): `grep -rn` on `src/` confirms `mu_t_advance` and `tridiag_solve` are NOT imported anywhere in `src/`. `calc_coef_w_wrf_coefficients` is defined in `acoustic_wrf.py` but only referenced from the comparator script and the M6B0-R regression test, never from production runtime code (single definition, no internal call). Helpers are validation-only by construction. GOOD.

## Part 4 — Comparator script drift

Common pattern (all three):
1. Hardcode `JAX_PLATFORM_NAME=cpu` + `XLA_PYTHON_CLIENT_PREALLOCATE=false` defaults.
2. `load_tolerance_ladder()` + local `_threshold(entry, expected) = max(abs, rel*max(|expected|,1))`.
3. Per-field `max_abs_delta / tolerance / passed / location / expected_shape / actual_shape / units / dtype / *_threshold`.
4. Tier dispatch `column / patch16 / golden / all`; `--steps` for emission count.
5. JSON proof file + tee'd text.

Drift items:
- `_threshold` is duplicated verbatim in all 3 scripts (m6b0r line 63, m6b1 line 296, m6b2 line 221). Should be extracted to `gpuwrf.validation.savepoint_io` or a `savepoint_compare` helper. Risk: if the tolerance formula is ever revised, three diverging copies.
- `_field_compare` (the full per-field report dict) is duplicated in m6b1 (inline in `compare_step`) and m6b2 (extracted as helper). m6b0r does its own version inline. Same risk.
- Hardcoded source path `SOURCE_RUN = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")` in m6b1 is not CLI-parameterized. m6b2 inherits it via `from m6b0r_wrf_savepoint_extract import SOURCE_WRFOUT`. m6b0r does take `--savepoint-root` so its source is HDF5 not wrfout. If the canary wrfout path ever moves, M6B1+M6B2 break in lockstep. Recommend a single `--source-wrfout` flag with a sensible default.
- m6b2 does `sys.path.insert(0, str(ROOT / "scripts"))` then `from m6b0r_wrf_savepoint_extract import ...`. Cross-script import out of a non-package scripts directory is fragile. Should move shared functions to `gpuwrf.validation.wrf_extract` or similar.
- Kill-gate decision string is hard-coded per sprint: m6b1 → `PROCEED_TO_M6B2`, m6b2 → `PROCEED_TO_M6B3`. Acceptable for now but it's another copy-paste vector.
- Tolerance handling is identical across the 3 scripts — no divergence.

## Part 5 — Operational-compatibility classification compliance (Critic Amendment #1)

| Sprint | worker-report has classification section? | Notes |
|---|---|---|
| M6B0-R defect-analysis | NO | Predates Critic Amendment #1 (amendment landed 698fbde with M6B2 dispatch). Documented gap; not a violation. Fields involved: `a/alpha/gamma` are validation-only since the helper is not wired to runtime. |
| M6B1 advance_mu_t | NO | Post-dates amendment (M6B1 commit 8a0130e is after 698fbde). **VIOLATION** of Amendment #1. New fields `mu/mudf/muts/muave/ww/theta/ph_tend` (some are simultaneously operational state and validation comparators) and the new `mu_t_advance.py` helper need explicit classification. The worker-report mentions only "unresolved risks" + "does not alter production runtime semantics" — implicit but not in the required structured form. |
| M6B2 tridiag | YES | Full 9-row "operational-compatibility" table with validation-only / operational-approved-with-evidence / undecided. Compliant. Notable: `tridiag_solve.py` callable classified `undecided`; `lax.scan` choice classified `operational-approved-with-evidence`; HDF5 layout, dtype `float64`, hooks, fields, tolerance entries all classified `validation-only`. |

Aggregate "Undecided" items (input scope for M6-perf-design):
- `src/gpuwrf/dynamics/tridiag_solve.py` callable — Thomas vs PCR/batched-Thomas decision deferred.
- (M6B1 fields unclassified — manager should ask M6B1 worker to retroactively classify before M6B4, or have this audit's verdict count as the classification: validation-only for the helper, undecided for the runtime mu/muts/muave/ph_tend carry which the operational mode might drop per §14.5.1 invariant table.)

## Part 6 — Verdict

**Cumulative state quality: MIXED.**

Strengths:
- Three parity verdicts are mathematically clean (all max-abs deltas ≤ 6e-17 at patch/golden tier for Thomas; M6B1 PASS with `0/15` kill-gate; M6B0-R coef errors driven to zero).
- Tolerance ladder is internally consistent and tighter-than-or-equal-to `1e-11` for all numerically tight operators; only Pa-scale and m/s-scale fields are laxer with justification.
- Validation helpers are correctly isolated from production runtime (`grep -rn` confirms no `src/` consumer outside the helper file itself).
- Operational WRF SHA invariant has held (`1ec3815...` confirmed unchanged post-audit).
- 103/103 (M6B2) regression tests still pass per M6B2 proof; this audit's `pytest --collect-only` shows `604 tests collected` with `proof_no_touch.txt` capturing the unchanged state.

Drift / duplication / quality issues found:

1. **`solve_em.F.patch` is malformed** — wrong line counts in hunk 1, three bare `@@` markers (no offsets), and 3 hunks targeting the wrong file (`module_small_step_em.F`, not `solve_em.F`). This patch could never apply against canonical WRF today. (`proof_patch_dryrun.txt` RC=2.)
2. **Wrapper ABI still mostly 0-arg** — 10 of 16 hook subroutines accept zero args; only the M6B1/B2 hooks took typed args, and even those have empty bodies. No in-timestep HDF5 emission anywhere. Already tracked via `m6b0r-fortran-hook-abi-followup`, but it means the "savepoint harness" claim is still a Python-transcription harness, not a relinked WRF emission harness.
3. **Schema version not bumped** through 3 successive operator additions. `SCHEMA_VERSION="m6b0r-savepoint-v1"` and `tolerance_ladder schema_version="m6b0r-tolerance-ladder-v1"`.
4. **Critic Amendment #1 partial compliance** — M6B1 worker-report lacks the mandatory operational-compatibility classification section. M6B2 added it. M6B0-R predates amendment (excused).
5. **`_threshold` and `_field_compare` duplicated** verbatim across 3 comparator scripts. Hardcoded source-wrfout path in M6B1. Cross-script import (`from m6b0r_wrf_savepoint_extract import ...`) out of a non-package `scripts/` directory.
6. **Pre-M6B0-R legacy fields (`cofrz/cofwr/...`)** still live in the tolerance ladder with no provenance note; consumed only by a single diagnostic script. Not stale, but accreting.

Required cleanup before M6B4 (minimum bar to keep the ladder honest):

- [REQUIRED] Fix `solve_em.F.patch` hunk counts + bare `@@` markers, OR explicitly mark the patch file `EXPERIMENTAL — DO NOT APPLY` until the hook-ABI follow-up rewrites it. The patch is currently a documentation artifact, not an applicable patch; the README/contract should say so.
- [REQUIRED] Split Thomas-solve hunks out of `solve_em.F.patch` into `module_small_step_em.F.patch` (or relabel the umbrella patch as a directory-wide unified diff).
- [REQUIRED] M6B1 retroactive Amendment #1 classification — either patch the M6B1 worker-report OR have this audit memo serve as the classification (helper = validation-only; runtime mu/muts/muave/ww/theta/ph_tend = `undecided` for operational-mode carry inclusion).
- [RECOMMENDED] Bump `SCHEMA_VERSION` to `m6b2-savepoint-v2` and `tolerance_ladder` to `m6b2-tolerance-ladder-v2`. (Allows M6B3 to introduce `t_2ave` / `_save` fields under a fresh version.)
- [RECOMMENDED] Extract `_threshold` and `_field_compare` to `gpuwrf.validation.savepoint_compare`. Parameterize source-wrfout path via CLI.
- [DEFER to M6-perf-design] tridiag_solve operational-vs-PCR decision; mu_t_advance helper retire-or-promote decision.

**GO / WAIT / NO-GO for M6B3 verdict merge: GO-WITH-CONDITIONS.**

Rationale: M6B3 is a parallel scratch-state parity sprint over the same Python-transcription extraction lane. None of the issues found here are *correctness* defects in the parity verdicts already merged; they are quality / hygiene / coverage defects. The most important issue (malformed `solve_em.F.patch`) only matters once the hook-ABI follow-up tries to relink the patch against canonical WRF — which is a separate dispatched sprint, not M6B3. M6B3 can safely add `t_2ave/_save/ww/muave/muts/ph_tend` scratch-state parity using the same Python-transcription comparator pattern.

Conditions on the M6B3 merge:
1. M6B3's worker-report MUST include the Critic Amendment #1 operational-compatibility table (the M6B2 worker proved it can be done; no excuse for M6B3 to skip it).
2. If M6B3 also extends `solve_em.F.patch` or `savepoint_wrapper.F90`, this audit's defects #1–#2 above become blocking — either fix the patch in the same sprint or carry an explicit "patch is non-applicable WIP" annotation.
3. M6B3's new fields MUST get fresh ladder entries with documented tolerance choices; do NOT silently reuse `theta`/`ww`/`muave` entries if M6B3 is now treating them as scratch-state with different accumulation semantics.

After M6B3 merges, the manager should schedule a small "M6B-ladder-hygiene" cleanup sprint (1–2 hours) to address items #3, #5, #6 above before M6B4 dispatches.

## Part 7 — No regression

`pytest --collect-only` → `604 tests collected in 1.87s` (see `proof_no_touch.txt`). `git status --short` on `tests/ src/ external/ scripts/` is empty in this worktree. Operational WRF SHA `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37` unchanged (see `proof_operational_sha256.txt`).

## AGENT REPORT

Cumulative-state audit of B-direct ladder (M6B0-R + M6B1 + M6B2) finds the merged parity verdicts mathematically sound and the validation helpers correctly isolated from production runtime, but surfaces six hygiene defects: (1) `external/wrf_savepoint_patch/solve_em.F.patch` is malformed and cannot apply against canonical WRF (wrong hunk counts, bare `@@` markers, Thomas-solve hunks targeting the wrong file); (2) 10 of 16 savepoint-wrapper hooks still accept 0 args with empty bodies, so the harness remains a Python-transcription comparator rather than an in-timestep emission lane (known follow-up queued); (3) `SCHEMA_VERSION` and `tolerance_ladder schema_version` not bumped despite three extensions; (4) M6B1 worker-report omits the Critic Amendment #1 operational-compatibility classification section (M6B2 has it, M6B0-R predates the amendment); (5) `_threshold` / `_field_compare` duplicated across all three comparator scripts and hardcoded `/mnt/data/...` source path in M6B1; (6) pre-M6B0-R legacy `cofrz/cofwr/...` ladder entries accreting without provenance notes. Verdict: cumulative state quality MIXED; M6B3 merge is GO-WITH-CONDITIONS (M6B3 worker-report MUST include Amendment #1 classification; any further `solve_em.F.patch` extension is blocked on fixing the malformedness; new fields must get fresh ladder entries). Recommend a 1–2-hour M6B-ladder-hygiene cleanup sprint after M6B3 merges and before M6B4 dispatches. No code, schema, or tolerance was modified during this audit; operational WRF SHA `1ec3815...` unchanged; `pytest --collect-only` = 604 tests, no test files touched.
