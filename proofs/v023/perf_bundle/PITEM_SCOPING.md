# v0.23.0 Performance-Bundle — P-item Re-Verification vs v0.22.2 source

**Built:** 2026-07-02 by the analysis/planning worker (Opus), dispatched by manager 0:1.
**Task:** the roadmap's stated FIRST v0.23 step — a 1-pass RE-VERIFY of the GLM kernel levers
(`Performance_review_GLM_0.22.md`, read a STALE v0.20.2 checkout) against the CURRENT shipped
v0.22.2 source, so we know which levers are already shipped / moot / still valid before executing.
**Scope:** the SECONDARY performance track only (P0–P7 + K1). F1/F1b (batched mini-grid) is the
PRIMARY track, handled elsewhere — NOT touched here (no `domain_tree.py`/`nested_pipeline.py` batching).
**Constraints honoured:** analysis + planning only. No source edited, no GPU run, no lock taken.

## Ground truth established
- `origin/main` = **v0.22.2** (`53bd20bf`), the shipped code. Worktree `<USER_HOME>/src/wrf_gpu2_wt/v023-f1`
  is byte-clean == `origin/main` (`git diff --stat origin/main -- src/` empty). All findings below are
  verified against that worktree. The checkout the manager pane sits in is a stale v0.22.1 branch — NOT used.
- **GLM lever 4.1 (FUSED+AOT cheap-key warm-start) IS SHIPPED in v0.22.2.** Confirmed: `aot_cheap_key.py`,
  `aot_executable.py`, and `domain_tree.py` `_cheap_key_for` / `_verify_cheap_key_blob` /
  `load_domain_blob` / `_serialize_domain_blob` for `fused/<parent_name>` blobs are all wired. Matches the
  roadmap reconciliation. This is why P7's "ship-the-blob fallback" already partly exists.

---

## PER-ITEM VERDICTS

### P0 — M9 RRTMG re-solve bit-identical reduction (from the 2026-06-28 addendum)
NOT re-verified in this pass (it is an OUTPUT-boundary / radiation-timing investigation, not a GLM
kernel lever, and it lives in `nested_pipeline.py` / radiation carry — MED-HIGH complexity, MED risk).
Flagged as the highest OUTPUT-boundary impact item (M9-diag 23,219 ms dominant per the 384 A/B), but it
is a research question ("can the forecast scan's radiation be threaded out without a re-solve, bit-identically?")
requiring its own scoping spike, and touches the pipeline the F1 track owns. **Recommend a separate P0 scoping
task, sequenced after F1 plumbing lands (pairs with P3 pipeline default-on).** Out of scope for this kernel re-verify.

### P1 — Single-scan forecast as default  — **VALID, NOT DONE, CPU-gate-buildable-now**
- `run_forecast_operational_single_scan` EXISTS: `runtime/operational_mode.py:5715-5762`. It is ONE
  `jax.lax.scan` over `jnp.arange(1, steps+1)` with RRTMG gated by `jax.lax.cond` on the traced predicate
  `(step_index % cadence == 0)` — compile cost independent of forecast horizon.
- STILL opt-in only. The production default `_run_forecast_operational_jit` (`:5584-5641`) is the Python
  `while step<=steps:` loop that emits ONE `jax.lax.scan` per radiation interval (4@1h → 288@72h; ~30 min
  cold compile at 3h). Docstring `:5735-5736` still says "remains the validated default until this entry
  passes its own short-horizon equivalence gate (`proofs/perf/single_scan_equiv.json`)."
- **NO runtime knob** (no `GPUWRF_SINGLE_SCAN` env / no namelist flag) to select it yet.
- Gate driver `proofs/perf/single_scan_equiv.py` EXISTS and is **CPU-runnable** (`taskset -c 0-3`, no CUDA;
  device_get machine-precision field compare). But `single_scan_equiv.json` (the PASS artifact) does NOT exist.
- **STRONG SIGNAL:** a NEW `proofs/perf/segscan_equiv.json` (dated Jul 2) already proves the *segmented*
  path == single-scan path **bitwise-identical** on all fields at 0.2h & 0.6h incl. the radiation step.
  So the equivalence is very likely to pass; the remaining work is: run `single_scan_equiv.py` to produce the
  JSON, add a selection knob, flip default, warm GPU s/step confirm.
- Files/functions: `operational_mode.py:5715` (fn), `:5584` (current default), `:5507` (`run_forecast_operational`
  dispatch), `daily_pipeline.py:854` (`_default_forecast_fn`). Add env/flag dispatch in `run_forecast_operational`.
- **Complexity MED. Risk LOW-MED** (the `lax.cond` radiation schedule changes HLO → must be field-tolerance-green,
  not just intent-identical; the CPU gate catches it — segscan_equiv already bitwise-PASS is reassuring).
- **CPU-side NOW:** run the equivalence gate + wire the knob. **GPU-gated:** the final warm s/step "unchanged"
  confirmation + default-flip go-decision.

### P2 — `_safe_floors` gating completion + finite-guard 46→1 stacked reduction — **VALID, PARTIALLY done, CPU-buildable, ZERO bit-identity risk**
**Lever A — `_safe_floors` gating (bit-identical, gates proven 0/168 fire):** PARTIALLY done.
- GATED already (use `_floor_pos()` which honours `_safe_floors()`, default OFF via `GPUWRF_ADVANCE_W_SAFE_FLOORS`):
  `advance_w.py:408, :411, :453`.
- STILL UNGATED (fire every substep/stage on valid fp64 data; XLA emits abs+where anyway):
  - `advance_w.py:498-500` `safe_mass_h_mut`
  - `calc_p_rho.py:89` `safe_mass`, `:98-100` `safe_theta_ref`
  - `small_step_finish.py:17-19` `_safe_denominator` (called 4× at `:51-54`)
  - `rk_addtend_dry.py:194` `safe_base`, `:224` `safe_t`, `:299` `_dn_top_safe`
- Fix: wrap the 7 remaining sites in `_floor_pos()` / a gated helper. ~15–100 fewer micro-kernels/step.
- Note: `small_step_finish.py:60-87` has a traced-dtype dispatch that selects the base but is NOT a guard.

**Lever B — finite-guard 46→1 stacked reduction:** NOT truly done. The v0.22.2 "batching" is cosmetic.
- `finite_state_guard.py:81-83` `_jax_all_finite_flags` still does a **per-field** tuple generator
  `tuple(jnp.all(jnp.isfinite(v)) for v in values)` then `jnp.stack` — i.e. ~30–45 separate all+isfinite
  reductions per boundary call (one per active field of the ~51-field `PROGNOSTIC_STATE_FIELDS`), not one fused
  reduction. The docstring at `:143` claims "one batched device-side finite check" but the impl is per-field.
- Fix: a single `jnp.all(jnp.concatenate([jnp.ravel(jnp.isfinite(v)) for v in values]))` scalar reduction
  (fields have DIFFERENT shapes → must ravel+concatenate, NOT a naive `jnp.stack` of arrays), with the
  per-field first-bad-index lookup kept ONLY on the failure path. Call sites `nested_pipeline.py:1270, :1690`.
- **Complexity LOW. Risk LOW / bit-identical** (same boolean result; verify the failure diagnostic still reports
  first bad field+index). Fully CPU-buildable + unit-testable.

### P3 — 2-domain root fusion — **VALID, NOT DONE, GPU-gated**
- `domain_tree.py:1958` `_fusable_parent` still hard-returns `None` for the root (`if name == roots()[0]: return None`).
  Fusion is leaf-cascade-only (`:1554-1620` `_build_fused_cascade_program`; `:1967-1970` reject child-with-children /
  feedback). For a 2-domain d01→d02 nest the root advance + root→child force are BOTH eager Python dispatches → the
  fused cascade gives that common dev/validation case NO benefit.
- No 2-domain/root-fusion/"S4" work in the v0.22.x git history for `domain_tree.py`.
- Change shape: (1) allow root when it has exactly one fusable child in `_fusable_parent` (`:1958`); (2) let
  `_build_fused_cascade_program` handle root context (clock base / carry type); (3) fold the root cache key
  (`_operational_fused_cascade_factory` `:2002`). The eager root path (`:586-609`: advance → build_child_boundary_package
  → child advance) has clean dataflow order (child boundary reads the just-advanced parent), so fusing is safe.
- **Complexity MED. Risk MED** (root boundary/edge geometry into the fused jit). = the deferred S4.
- **GPU-gated:** needs a CPU determinant-matrix gate (like 4.1) THEN a GPU integration+bit-identity run. Payoff modest
  for 2-dom (4→1 dispatches/root step), the point is it is proportionally the most launch-bound case.

### P4 — `calc_coef_w` Python-loop → vectorized — **PARTIALLY VALID (Loop-1 only), CPU-buildable with an existing oracle**
- `acoustic_wrf.py:783-852` `calc_coef_w_wrf_coefficients` is PRODUCTION (imported by `operational_mode.py:150`;
  built once per RK stage, outside the substep scan → a COMPILE-cost lever, not a per-step launch lever). Two loops:
  - **Loop 1 (`:830-833`, `for kk in range(2, nz)`) = INDEPENDENT map** → trivially vectorizable with
    `jnp.arange`+scatter, bit-identical, ~86 scatters → 1. **This is the real P4 harvest. LOW risk.**
  - **Loop 2 (`:836-846`, `for k in range(1, nz)`) = SEQUENTIAL Thomas forward-sweep recurrence**
    (`alpha[k] = 1/(b - a[k]*gamma[k-1])`, `gamma[k]=c*alpha[k]`). CANNOT be done with `jnp.arange`+`where`
    (would break bit-identity). A `lax.scan` preserves it but gives only marginal compile benefit + MED-HIGH risk.
    **The roadmap's "sequential recurrence — verify" caution is correct; do Loop 1 only, LEAVE Loop 2.**
- Oracle EXISTS + CPU-runnable: `tests/test_m6b0r_calc_coef_w_fix.py` (WRF savepoint tiers column/patch16/golden,
  `max_abs_delta <= 1e-6`). Gate any change against it.
- **Complexity LOW (Loop 1 only). Risk LOW. CPU-side NOW** (compile-cost + oracle both CPU-measurable).

### P5 — Autotune-cache default-ON — **ALREADY DONE / HARVEST (moot)**
- **SHIPPED in v0.22.2.** The compile-cache import hook `compile_cache.py:479` calls
  `configure_autotune_cache(default_on=True)` AFTER configuring the persistent compile cache, so the autotune
  cache rides ON by default whenever the executable cache is on (the `_opt_in(default_on=True)` path in
  `xla_autotune.py:209-235`), preserving the `GPUWRF_XLA_AUTOTUNE_CACHE=0` opt-out. HARD-SAFE per-flag subprocess
  probe still in place. **BONUS:** `compile_cache.py:502` also `configure_parallel_compile(default_on=True)` →
  parallel compile is default-on too (corroborates P7a done).
- The GLM "doc/code mismatch" is now resolved (code matches the default-on docs).
- **No work. Move to HARVEST.** (Only residual: if a fresh GPU canary is ever run, record the actual cold-compile
  saving for the release notes — nice-to-have, not blocking.)

### P6 — Dead-code + idiom cleanups (4.13–4.21) — **VALID (all present), mostly bit-identical hygiene; ONE audit-misclassification corrected**
Per-item, verified present in v0.22.2:
| Item | Location | Present | Path | Disposition |
|---|---|---|---|---|
| 4.13 `mass=ones_like(theta_diffused)` per step | `orchestrator.py:115` | yes | **skeleton (c2, NOT production)** | low value; skeleton — verify pattern absent in `dynamics/core/` before bothering |
| 4.14 duplicate decouple fns + dead `_mass_couple_theta_before_advance` | `dynamics/core/acoustic.py:803,812-834` | yes | `:803` dead (no callers); `:812/:820` identical, aliased `:834`; `_decouple_theta_for_finish` IS production (`:1129`) | delete `:803` (zero-risk); collapse the duplicate to one name (keep `_decouple_theta_for_finish`) |
| 4.15 `w_solve_core` + `thomas_solve_scan` import | `dynamics/core/acoustic.py:44,789-800` | yes | dead on production (`advance_w_wrf` has its own inline Thomas; `w_solve_core` only oracle/M4) | mark validation-only / remove after checking oracle callers |
| 4.16 constants R_D=287.0/CP_D=1004.0/GRAVITY=9.80665 | `dynamics/vertical_implicit_solver.py:15-17` | yes | **DEAD on production** — used only by `build_epssm_column_coefficients` → `_mpas_recurrence_vertical_update` (`acoustic_wrf.py:944,1000,1026`), which is NOT reached by `operational_mode.py` (it imports only `calc_coef_w_wrf_coefficients`+helpers from acoustic_wrf; `run_acoustic_scan`/`_mpas` appear only in `integration/d02_replay.py` oracle). **GLM CORRECT; a parallel agent's "not dead" claim conflated reachable-within-acoustic_wrf with production — REJECTED.** | delete the dead constants (mis-transcription trap: they differ from production 9.81) after confirming the d02_replay/oracle path |
| 4.17 `jnp.pad`+slice vs Wave-A concatenate | `rhs_ph.py` (4×), `rk_addtend_dry.py` (8×) | yes | production | idiom align; per-STAGE not per-substep → small; needs savepoint oracle |
| 4.18 `flip(mask + zeros_like(tau_clear))` | `rrtmg_sw.py:1679,2431` | yes | production | → `broadcast_to`; verify shape; small |
| 4.19 `asymmetry_clear_orig = zeros_like` | `rrtmg_sw.py:2005,2184,2379` | yes | production | → `jnp.where`; small |
| 4.20 `max_event_tail=None` default (unbounded) | `domain_tree.py:321` | yes | production (library API) | default a bounded tail; `nested_pipeline` already caps its own path |
| 4.21 `donate_argnums=(0,)` only (namelist leaves not donated) | `operational_mode.py:5584,5715,5765,5828` | yes | production | INVESTIGATE-ONLY / MED risk (donation footgun; namelist reference semantics) — do NOT bundle with the zero-risk items |
- **Disposition:** 4.14/4.15/4.16 = pure dead-code deletion (zero risk, no oracle needed beyond a caller grep).
  4.17/4.18/4.19 = idiom changes → savepoint-oracle-gated but bit-identical, tiny. 4.13 = skeleton (verify first).
  4.20 = library-API host-RAM default. 4.21 = investigate-only, MED risk, keep separate.
- **All CPU-buildable. Complexity LOW. Ideal solo-with-proof hygiene sprint (split the zero-risk deletions from the
  oracle-gated idiom edits from the MED-risk 4.21).**

### P7 — Parallel cold-compile + lower XLA opt/fitting effort — **MOSTLY DONE; only the "lower-effort flags" wrapper remains**
- **P7a parallel cold-compile: ALREADY SHIPPED (v0.21+).** `aot_precompile.py:1053-1158` `prewarm_defused_nest`
  fans per-domain `_advance_chunk_fori` compiles across a `ProcessPoolExecutor`; `compile_cache.py:502`
  `configure_parallel_compile(default_on=True)`. **HARVEST.**
- **P7c fused-cascade AOT serialize/load ("ship-the-blob"): ALREADY SHIPPED (v0.21+).** `domain_tree.py:1637-1722`
  serialize under `fused/<parent_name>` + cheap_key/hlo_sha256, `:1809-1847` load-without-lower. **HARVEST.**
- **P7b lower XLA opt/fitting effort on the fused cold compile: STILL OPEN, CPU-buildable.** The exec flags
  (`jax_optimization_level`, `jax_memory_fitting_effort`, `XLA_FLAGS`, ...) ARE captured in the cheap_key
  (`aot_cheap_key.py:552-568` `_EXEC_JAX_CONFIG_FLAGS`/`_EXEC_ENV_FLAGS`, hashed `:593-613`), so a lower-effort cold
  blob is correctly addressed — but there is NO code that AUTO-lowers effort for the fused cold compile. Change shape:
  a config-context wrapper around the `@jax.jit` in `_build_fused_cascade_program` (cold-only, runtime-neutral by
  construction since it is in exec_key). **Complexity LOW-MED. Risk LOW-MED** (lower opt could regress warm s/step →
  MUST measure warm s/step on the canary; reject any lever that drops RAM but regresses warm speed).
- **CPU-side NOW:** build the wrapper + a CPU cold-compile-wall A/B. **GPU-gated:** the warm s/step no-regress confirm.

### K1 — BouLac O(nz²)→O(nz) spike — **VALID as a BLOCKED item, but the roadmap's PROPOSED spike is ALREADY ANSWERED (NO)**
- `physics/mynn_pbl.py`: default `_boulac_length_dense` (`:698`, dense `(B,nz,nz)` PE matrices) vs opt-in
  `_boulac_length_onz` (`:532`), dispatched by `_boulac_length` (`:838`) on `GPUWRF_MYNN_BOULAC_ONZ` (`:156`, default OFF).
  `GPUWRF_MYNN_BOULAC_FP32` (`:137`) opt-in default OFF.
- **The roadmap's stated K1 spike — "does a `lax.scan`-based O(nz) form dodge the slow compile WITHOUT the fp32 rewrite?"
  — is ALREADY TESTED and the answer is NO.** In-source comment `mynn_pbl.py:149-155`: the "Very slow compile"
  pathology "reproduced for **both a lax.scan and a Python-unrolled form**, BFC + cuda_async, autotune on/off … the
  data-dependent O(nz) search defeats the fusion that the pure-cumsum dense form enjoys." Kill-gate artifact
  `proofs/v022/compile_pathology/k1_boulac_onz_verdict.json`: dense = 51.5 s compile wall; O(nz) still compiling at
  727 s (manual stop), 15.2 GiB RSS. So the current `_boulac_length_onz` is a Python-unroll BECAUSE the scan tripped
  the same wall — the roadmap's premise ("scan may dodge it") is stale.
- Correctness is PROVEN (`proofs/perf/v015/boulac_nz_oracle.json` max_rel 2.4e-16 over 8 regimes;
  `boulac_nz_perf_ab.json` 1.30× / −387 MB in isolation). The blocker is purely the fusion-boundary compile pathology
  when fused into the operational forecast jit — escalated as "Fable fusion-boundary work" (`:155`).
- **Revised disposition:** K1 is NOT a cheap tiered ship. The remaining live question is NOT "scan vs unroll" (answered)
  but "**a fusion-BOUNDARY reformulation** — split the O(nz) search out of the operational jit boundary (separate jit /
  staged materialization) so the data-dependent search stops defeating fusion." That is genuinely HARD (kernel/perf-core,
  frontrunner+critic) and shares root cause with the fp32/ADR-031 work. **Recommend: do NOT re-run the scan spike; scope a
  narrow CPU compile-wall reproduction (can the pathology be reproduced on the CPU/XLA-CPU backend? unknown — no CPU harness
  exists today) to decide backend-specificity, else K1 waits for M1 (fp32).**
- **GPU-gated** for any real measurement; the CPU backend-specificity check is the only CPU-side probe.

---

## HARVEST LIST (already-done / trivially-bit-identical / moot)
1. **P5 autotune-cache default-ON — ALREADY SHIPPED** (`compile_cache.py:479`). Moot; only record the cold saving if a canary runs.
2. **P7a parallel cold-compile — ALREADY SHIPPED** (`aot_precompile.py:1053` + `compile_cache.py:502`). Moot.
3. **P7c fused-cascade AOT serialize/load ("ship-the-blob") — ALREADY SHIPPED** (`domain_tree.py:1637-1847`). Moot.
4. **GLM 4.1 FUSED+AOT cheap-key warm-start — ALREADY SHIPPED** (roadmap already reconciled). Confirmed.
5. **K1 "lax.scan dodges the slow compile" spike — ALREADY ANSWERED NO** (`mynn_pbl.py:149-155`, `k1_boulac_onz_verdict.json`).
   Do NOT re-run it. Reframe K1 to a fusion-boundary reformulation (hard) or park for M1.
6. **P4 Loop 2 (Thomas forward sweep)** — bit-identical vectorization is IMPOSSIBLE (sequential recurrence). Harvest only Loop 1.
7. **P6 4.16 constants** — confirmed DEAD on production (GLM correct); a parallel agent's "not dead" claim rejected.

## SANITY CROSS-CHECK NOTE (for the executing worker)
One background verifier initially claimed P6 4.16 constants are "not dead / used on production." I re-verified
directly: `build_epssm_column_coefficients` is reached ONLY via `_mpas_recurrence_vertical_update`
(`acoustic_wrf.py:944/1000/1026`), and `operational_mode.py` does NOT import/call that path (only
`calc_coef_w_wrf_coefficients`+helpers). The production w-solve is `dynamics/core/advance_w.py`. So 4.16 IS dead on
the production forecast path — matching the GLM audit. Treat 4.16 as safe-to-delete after a d02_replay/oracle caller grep.
