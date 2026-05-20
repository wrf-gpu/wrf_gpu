# Tester Report — M5-S1 Attempt 4 (Thompson Microphysics Column)

Tester: Claude Opus 4.7 xhigh (cross-AI tester, per `sprint-contract.md:7`).
Role: independent correctness/contract verifier. Read-only on code/artifacts/contracts; write-only on this report.
Sources read in mandated order: PROJECT_CONSTITUTION, AGENTS, conducting-blind-review skill, validating-physics skill, sprint-contract (incl. attempts 2/3/4 amendments), worker-report A4, BLOCKER, MANAGER-NOTE, diagnosis-report, gemini-third-opinion, ADR-005, ADR-006, attempt-4 artifacts under `artifacts/m5/`, integration diff `git log/diff main..HEAD`, JAX/Fortran source.

Mid-review manager injection received: "Gemini side-runner #2 (parity sanity) found a coefficient typo at `thompson_column.py:277-278`." Independently verified against the WRF snapshot — confirmed below in the Adversarial section.

---

## 1. Per-AC verdict table

ACs are numbered per `sprint-contract.md`. Attempt-4 binding fixes from the attempt-4 amendment are labelled `A4.1..A4.5`.

| AC | Subject | Verdict | Evidence |
|---|---|---|---|
| 1.1 | Public API `step_thompson_column(state, dt, *, debug=False)` | **pass** | `src/gpuwrf/physics/thompson_column.py:548-552` (`@partial(jax.jit, static_argnames=("dt", "debug"))`). |
| 1.2 | Sedimentation OUT of M5-S1 | **pass** | `wrf_thompson_harness_build.sh:42-70` patches the WRF source to zero `vtrk/vtnrk/vtik/vtnik/vtsk/vtgk/vtngk/vtck/vtnck` before the WRF sedimentation flux loops. JAX kernel has no sedimentation code (grep on `thompson_column.py`). ADR-006:44-48. |
| 1.3 | Debug HLO byte-identity | **pass** | `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` size = 0 bytes (`ls -l`). HLO sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` matches the SHA of the empty file. Stripped sibling at `thompson_column_debug_stripped.py:25-34` physically omits the two `_debug_checks` calls present in `thompson_column.py:533, 545`. |
| 1.4 | Zero in-trace allocations | **pass** | `grep -n 'jnp.array\|jnp.zeros\|jnp.empty' src/gpuwrf/physics/thompson_column.py thompson_constants.py thompson_saturation.py` → empty. Profile `temporary_bytes_per_step=0` in `artifacts/m5/thompson_profile.json:24`. |
| 1.5 | fp64 precision | **pass** | `thompson_column.py:75` `config.update("jax_enable_x64", True)`; same in stripped sibling. |
| 1.6 | Hash+eq pytree container | **pass** | `thompson_column.py:116-135` implements both `__eq__` and `__hash__`. |
| 2.1 | Tier-1 fixture from WRF source-of-truth | **pass** | `m5_generate_thompson_fixture.py:170-184` invokes the compiled harness binary; outputs are read from disk and packaged. No Thompson source/sink formulas in the Python generator. Structural anti-tautology guard satisfied. |
| 2.2.fix / Path A | WRF-faithful oracle | **pass** | Fortran harness `wrf_thompson_harness.f90:54-58` calls `thompson_init` then `mp_gt_driver` from the locally-patched `module_mp_thompson_nosed.o`. Build linkage in `wrf_thompson_harness_build.sh:81-86`. Fixture manifest `source: wrf-derived; source_commit: wrf-thompson-via-fortran-harness sha256=6a5b66e1e6d625b80...` (`analytic-thompson-column-v1.yaml:2-3`). |
| 2.3 | ≥3 scenarios | **pass** | `m5_generate_thompson_fixture.py:175` `("maritime_warm","cold_mixed_phase","precipitating")`. Fixture shape `[3,12]` per manifest variable entries. |
| 2.4 | Manifest schema | **pass with minor** | Manifest validates (`validate_fixture_manifest.py …: ok` per worker-report:81). **Minor finding:** `scenario:` text at `analytic-thompson-column-v1.yaml:6-7` says "sedimentation numerically suppressed with 1e30 m layer depths" — this is the attempt-3 narrative. Attempt 4 replaced that with the terminal-velocity-zeroing source patch and physical `dz=1000 m` (`wrf_thompson_harness.f90:40`). The manifest scenario text is stale and contradicts the actual mechanism documented in ADR-006:44-46. Not blocker-class, but should be corrected. |
| 2.5 | ≤200 KB | **pass** | `fixtures/samples/analytic-thompson-column-v1.npz` = 7026 bytes per manifest `files[0].bytes`. |
| 3.1 | Tier-1 parity engine | **pass on engine** | `src/gpuwrf/validation/tier1_thompson.py` exists; produces correct schema. |
| 3.2 | Tier-1 JSON schema | **pass** | `artifacts/m5/tier1_thompson_parity.json:1-39` contains `fixture_id`, `scenarios_tested=3`, `per_field_max_abs_err`, `per_field_max_rel_err`, `tolerances_met`, `pass`, plus a per-field break-out. |
| 3.3 | `pass: true` required | **fail** | `tier1_thompson_parity.json:14` `"pass": false`, `:38` `"tolerances_met": false`. Worst per-field abs errors: `qc=1.52e-4`, `qi=1.37e-4`, `qs=1.45e-4`, `T=0.0403 K`, `Ni=1.27e5`, `Nr=6.73e4`. Per-field rel errors include `qc=0.9999988`, `qs=1249.6`, `qg=9.83e8`, `qr=4.50e7`, `Nr=4.54e13` (`:26-36`). Worker filed `BLOCKER-m5-s1-attempt4-tolerance.md` per contract escape clause. |
| 4.1 | Tier-2 invariants (positivity, water budget, finite latent heat, NaN/Inf) | **pass** | `artifacts/m5/tier2_thompson_invariants.json:2-22`. Water residual `2.67e-12` ≤ `1e-8`; positivity violations `0`; max latent heating `2.73 K` < `100 K`; nan/inf violations `0`. |
| 4.2 | Tier-2 JSON `pass:true` | **pass** | `tier2_thompson_invariants.json:13`. |
| 5.1 | Gate definition implementation | **pass on engine** | `scripts/m5_gate_thompson.py` runs and emits required JSON. |
| 5.2 | Gate JSON schema | **pass** | `artifacts/m5/thompson_gate_result.json:1-9` contains required keys + `gate_status="FALLBACK"`, `rationale="correctness failed"`. |
| 5.3 | Gate path action | **mixed** | Reported `FALLBACK`, but the rationale is correctness-failure (Tier-1), not the ADR-005 launches/regs/local-mem performance-fallback path. ADR-001 per-scheme Triton fallback is for performance-class FALLBACK; here Tier-1 correctness is the real blocker. Gate-status semantics conflate two failure modes. Worker correctly flagged this in `worker-report.md:134`. Not blocker for the tester verdict — but reviewer must decide whether to keep the "FALLBACK" label or rename it. |
| 5.4 | Register/local-mem null acceptable | **pass** | `thompson_profile.json:18,19,21,23` null; `profiler_limitation` field documents perfmon block (`:21`). |
| 6.1 | HLO 0-byte diff | **pass** | See AC 1.3. |
| 7.1 | ADR-006 ≥1500 bytes + required tokens | **pass** | `.agent/decisions/ADR-006-thompson-jax-implementation.md` is 3.8 KB; contains `Decision:` (`:9`), `WRF source mapping:` (`:14`), `Sedimentation status:` (`:44`), `Gate dry-run:` (`:64`). |
| 7.2 | ADR-006 implementation record | **pass** | Sections cover WRF subroutines mapped (`:30-42`), sedimentation method (`:44-48`), kernel fusion (`:50-54`), tolerances (`:60-62`), gate (`:64-66`). |
| 8.1 | Spacetime budget inline | **pass** | `worker-report.md:113-122`. |
| 8.2 | Allocation audit | **pass** | `worker-report.md:123-129`. |
| 8.3 | HLO diff SHA-256 | **pass** | `worker-report.md:129` `e3b0c44298fc...85`. Sha is the empty-file sha256, consistent with 0-byte diff. |
| 8.4 | One-line docstring per helper | **pass** | Spot-checked `_clip_species:151`, `_air_properties:166`, `_rain_distribution:181`, `_cloud_distribution:194`, `_ice_distribution:205`, `_snow_moment_proxy:217`, `_graupel_distribution:230`, `_sublimation_prefactor:241`, `_finish:261`, `_saturation_adjustment:293`, `_warm_rain_collection:321`, `_rain_evaporation:356`, `_warm_rain:399`, `_instant_melt_freeze:405`, `_ice_sources:430`, `_debug_checks:513`, `_step_thompson_column_impl:530`, `step_thompson_column:550`. All have one-line docstrings. |
| 8.5 | Reviewer attests line-by-line | **blocked** | Tester's job is verdict + spot-checks; full line-by-line attestation is the reviewer's task. |
| 9 | Cross-AI tester duties | **executing now** | This report. Adversarial probes + Ni-deposition + sedimentation + HLO-unroll + coefficient-typo verified. |
| 10.1 | pytest count ≥ M4 baseline + new | **pass on count, fail on result** | Worker-report:88 records `1 failed, 397 passed`; total 398 (M4 baseline 384). The fail is `test_m5_thompson_tier1_parity_passes`, i.e. AC 3.3 cascade — not a new regression. |
| 11.1 | `validate_agentos.py` ok | **pass** | `worker-report.md:27` `{"errors": [], "ok": true, …}`. |
| 11.2 | M1-M4 oracles no regression | **fail (cascaded)** | All four `check_m1..m4_done.py` return `ok=false`. Root cause is the cascaded `pytest -q` failure on Tier-1. Not an M1-M4 regression in their own artifacts; entirely driven by AC 3.3. |
| 11.3 | No file >200 KB | **pass** | Largest committed M5 file is `thompson_column.py` ~24 KB; npz 7026 B. |
| 11.4 | pyproject.toml scipy | **n/a** | Path-A obviated need; no scipy required. |
| A4.1 | Process-order refactor | **pass on mechanism, partial on outcome** | Order rewritten at `thompson_column.py:529-545`: `_clip_species → debug → _warm_rain_collection → _ice_sources → _saturation_adjustment → _rain_evaporation → _instant_melt_freeze → _finish`. Mirrors the WRF checkpoint sequence in `diagnosis-report.md:13`. T max-abs error `0.040290844661740266 K` (`tier1_thompson_parity.json:18`) — better than the predicted `0.084 K` from the diagnosis probe and well under the contract's `<0.1 K` backstop. Caveat: WRF actually stages multi-process tendencies between checkpoints; JAX applies them sequentially. Not strict tendency-staging semantics, but the dominant T error is captured. |
| A4.2 | Ni-deposition fix | **pass on mechanism, partial on outcome** | `thompson_column.py:504-506`: `Ni` is incremented only when `ice_deposition < 0.0` (sublimation), matching WRF lines 2719-2727 per the comment at `:504-505`. Ni max-abs error collapsed from attempt-3 `1.414e6` to `126975.125` (`tier1_thompson_parity.json:16`) — 91% reduction. Sublimation gating verified by reading the `jnp.where(ice_deposition < 0.0, …, 0.0)` clause. |
| A4.3 | Real sedimentation bypass | **pass** | Build script `wrf_thompson_harness_build.sh:42-70` patches the WRF source pre-compile by injecting `vtrk/vtnrk/vtik/vtnik/vtsk/vtgk/vtngk/vtck/vtnck = 0.` immediately before the sedimentation flux loops at `module_mp_thompson.F.pre` line markers. Harness now passes physical `dz=1000 m` (`wrf_thompson_harness.f90:40`). This IS Gemini's Method A (zero terminal velocities); the `dz=1e30` hack is gone. ADR-006:44-48 documents it. **Side-finding:** manifest scenario text (`analytic-thompson-column-v1.yaml:7`) is stale and still says "1e30 m layer depths" — see AC 2.4. |
| A4.4 | Tighten Tier-1 tolerances to ADR-005 strict | **pass on mechanism, fail on parity** | Manifest has `abs=1e-10/rel=1e-8` for hydrometeors (`yaml:156-158, 168-170, …`), `abs=1e-3/rel=1e-6` for `Ni/Nr` (`yaml:228-229, 240-241`), `abs=1e-8/rel=1e-8` for `output_T` (`yaml:252-253`). Tolerances match ADR-005:27. Parity fails — worker filed BLOCKER per contract escape clause. |
| A4.5 | MORNING-REPORT.md preserved | **pass** | Local file present (`worker-report.md:93`). `git diff --name-status main...HEAD` does not show MORNING-REPORT.md (`worker-report.md:96-98`). Confirmed by my own `git log main..HEAD --stat`: MORNING-REPORT.md does not appear in the integration diff. |

Summary count: 27 pass, 5 fail (3.3, 5.3 mixed, 10.1 fail-on-result, 11.2 cascaded, A4.4 fail-on-parity — all driven by the same Tier-1 mass-partition residual; 5.3 is a labelling issue), 1 blocked (8.5 = reviewer's task), 1 minor (2.4 stale manifest text).

---

## 2. Independent verification of worker's load-bearing claims

**Claim 1: "T error 0.32K → 0.0403K via process-order refactor"** — **VERIFIED.**
`artifacts/m5/tier1_thompson_parity.json:18` reports `"T": 0.040290844661740266`. Attempt-3 baseline `T=0.3186 K` is cited in `diagnosis-report.md:3`. Reduction factor ≈ 7.9×. Better than the diagnosis probe's predicted 0.084 K — that extra ~2× improvement is plausibly explained by the Ni-deposition fix and sedimentation bypass (which the diagnosis probe did not include). The order refactor is visible at `thompson_column.py:529-545`.

**Claim 2: "Ni error 1.4M → 127k via deposition gating"** — **VERIFIED.**
`tier1_thompson_parity.json:16` reports `"Ni": 126975.12500000041`. Attempt-3 baseline `1.414e6` per `diagnosis-report.md:3`. Reduction factor ≈ 11.1×. Code-level gating at `thompson_column.py:504-506`: `Ni=jnp.maximum(0.0, state.Ni + jnp.where(ice_deposition < 0.0, ice_deposition / jnp.maximum(xmi, XM0I), 0.0))`. The `jnp.where(ice_deposition < 0.0, …, 0.0)` clause matches WRF lines 2719-2727 (`pni_ide` only set in sublimation branch).

**Claim 3: "real sedimentation bypass via locally-patched WRF Thompson object"** — **VERIFIED, not a `dz=1e30` hack in disguise.**
Mechanism trace:
- `wrf_thompson_harness_build.sh:42-70` runs an inline Python `replace()` against the WRF source snapshot to inject `vtrk(:)=0.` (and 8 more velocity arrays) at the marker `"!..Sedimentation of mixing ratio is the integral of v(D)*m(D)*N(D)*dD,"`.
- That patched `module_mp_thompson_nosed.F90` is compiled to `module_mp_thompson_nosed.o` with `nvfortran` (`:72-75`).
- The harness links against the patched object (`:81-86`).
- `wrf_thompson_harness.f90:40` sets `dz = 1000.0`, not `1.0e30`. Confirmed by reading the source.

This is the Method A bypass requested by the contract. The fixture-generated outputs are produced by WRF code that physically cannot move hydrometeors between vertical levels because the terminal velocities are zero, while still running every other source/sink and bookkeeping step.

Stale-text finding (not a contradiction of the mechanism): the manifest's `scenario:` text still references `1e30 m layer depths` — see AC 2.4.

**Claim 4: "1 kernel launch per step"** — **VERIFIED.**
`artifacts/m5/thompson_profile.json:16-17` `"kernel_launches": 1, "kernel_launches_per_step": 1`. Launch count comes from HLO `kernel_launches_per_step()` (script `m5_run_thompson.py:87,108`). Per ADR-005's `local≤256B AND reg≤128 AND launches≤10` GO criteria, this is a hard GO on the launch axis. Reg/local null per perfmon block (acceptable per `sprint-contract.md:225`).

**Claim 5: "0-byte HLO diff (debug vs stripped)"** — **VERIFIED.**
`ls -la artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` → `0 bytes` (also recorded in `worker-report.md:77`). HLO sha `e3b0c44…` is sha256 of an empty file, consistent. Production HLO 66474 B, stripped 66453 B — both materially identical after normalization in `m5_run_thompson.py:46-61`.

**Claim 6: "Tier-2 conservation + positivity + NaN/Inf pass"** — **VERIFIED.**
`tier2_thompson_invariants.json:13` `"pass": true`. Per-condition: `water_budget.relative_residual=2.67e-12` vs tolerance `1e-8`; `positivity.violations=0`; `finite_latent_heating.max_abs_delta_T_K=2.73 K` vs bound `100 K`; `nan_inf.violations=0`. All four conditions hold. Note this is the 10-step trajectory check per `sprint-contract.md:197-199`.

---

## 3. Adversarial probe

I picked the **Ni-deposition fix claim** because it is isolated and well-instrumented in the diagnosis report — making counterexample construction tractable.

**Counterexample attempt 1 (failed):** Could the 91% Ni reduction be explained by something other than the sublimation-gating refactor (e.g. by `_finish`'s lami-clip on Ni at `thompson_column.py:272-279`, which would lower Ni regardless of `_ice_sources`)? Inspection: `_finish:279` writes `Ni = jnp.where(qi <= R1, 0.0, jnp.minimum((ri / AM_I * lami**3.0 * OIG2) / rho, 999.0e3 / rho))` — this is a final-state ni clamp based on `lami^3`, not zero. Attempt 3 had the same `_finish` post-step. Difference in Ni between attempts 3 and 4 must come from upstream — which is exactly `_ice_sources:504-506`. Counterexample not found; claim holds.

**Counterexample attempt 2 (succeeded — Gemini #2's typo, independently confirmed):** Manager injected a finding from Gemini side-runner #2 that the JAX `_finish` lami-clip uses the wrong WRF constant. Verified against `module_mp_thompson.F.pre`:

- WRF source at line 688: `cie(2) = bm_i + mu_i + 1.` → with `bm_i=3, mu_i=0`, `cie(2)=4`.
- WRF source at line 695: `cig(2) = WGAMMA(cie(2))` → `gamma(4) = 3! = 6`.
- WRF lami-clip at lines 1928, 1931 (and identical pattern at 3106-3110, 4097-4099): `lami = cie(2)/5.E-6` and `lami = cie(2)/300.E-6` — i.e. `4./5.e-6` and `4./300.e-6`.
- JAX at `thompson_column.py:277-278`: `lami = jnp.where(xdi < 5.0e-6, 6.0 / 5.0e-6, lami)` and `lami = jnp.where(xdi > 300.0e-6, 6.0 / 300.0e-6, lami)`.

The literal `6.0` in JAX corresponds to `cig(2)=6`, not `cie(2)=4`. The two constants are easily confused (worker wrote `AM_I * 6.0 * OIG1` correctly on line 275 — that **is** `am_i*cig(2)*oig1` — and immediately mis-used `6.0` for `cie(2)` on the next two lines).

Numerical impact: lami at clamp is off by a factor `6/4 = 1.5`. The next line writes `Ni = (ri/AM_I * lami^3 * OIG2)/rho`, so the post-clamp Ni is off by `(1.5)^3 / 1 = 3.375×` at the lami-clipped levels. With a current Ni max-abs residual of `1.27e5` (vs WRF ~2e5 baseline at those levels per `diagnosis-report.md:37`), a 3.375× over-write would explain a substantial portion of the residual.

**This is a confirmed code bug that the worker missed and the manager surfaced via Gemini #2.** It is in-scope for attempt 4 (the worker modified `thompson_column.py` and the line is in `_finish`), it sits exactly in the lookup-table/moment-parity territory the BLOCKER attributes the residual to, and fixing it is a 2-character edit. I cannot tell without re-running parity whether this single fix closes the Ni gap, but it is a load-bearing reducible-debt item that the BLOCKER does not name.

**Escalation:** I escalate this as a tester-found defect that the reviewer should require fixing before A4 closes, regardless of the Path A/B/C decision on the broader table-parity work. It is not a tolerances-can't-be-met-yet item — it is a transcription typo that already passed three reviewer cycles and was caught only by an external sanity-checker.

---

## 4. Gemini's HLO-unroll compile-OOM concern

**Concern (Gemini third-opinion, summary in MANAGER-NOTE-FOR-REVIEWER.md:44, full text in `gemini-third-opinion.md:20-21`):** if WRF lookup tables get baked into the JAX trace, HLO may unroll into massive nested `select`/`conditional` trees → compile-OOM on the 4-core worker.

**Inspection of A4 artifacts:**
- `grep -n 't_Efrw\|tps_iaus\|tni_iaus\|t.*_qrfz\|lookup\|table' thompson_column.py` returns only 4 hits — all in code-comments documenting that the table is **omitted** and a proxy is used (lines 340-342, 436). No table arrays are constructed.
- `thompson_constants.py` is 150 lines of scalar Python floats — no array literals, no lookup tables. Only `WGAMMA`-precomputed scalar constants like `CCG3_NU12 = 6402373705728000.0`.
- `thompson_saturation.py` contains saturation helpers but no tables (no array literals in `grep -n 'jnp.array\|np.array' thompson_saturation.py thompson_constants.py`).
- HLO file sizes (`thompson_column_production.txt` = 66,474 B; `thompson_column_debug_stripped.txt` = 66,453 B) are unremarkable for a fused per-column physics kernel — no sign of combinatorial blowup. Compile succeeded on the 4-core workstation during `m5_run_thompson.py`.
- 1-launch trace fits in one fused HLO program; if tables had been inlined the launch count or HLO size would diverge.

**Verdict:** Gemini's HLO-unroll concern is **moot for the current A4 implementation** — no tables are baked into the JAX trace; the worker is using bounded analytic proxies (`thompson_column.py:343, 220-226, 233-236`) precisely to avoid the table-inlining failure mode. The concern is valid prospectively for whatever M5-S1.x table-export sprint follows: if tables are exported as JAX arrays and indexed inline, recompile time and trace size must be monitored at that point.

---

## 5. Tester's recommendation

**Tester decision: Accept-with-required-fixes — mapped to Path C of MANAGER-NOTE-FOR-REVIEWER.md.**

Required fixes (in priority order):

1. **(Blocker-class, tester-found)** Fix the `cie(2)` vs `cig(2)` typo at `thompson_column.py:277-278`. Replace `6.0/5.0e-6` and `6.0/300.0e-6` with `4.0/5.0e-6` and `4.0/300.0e-6`. Re-run `m5_run_thompson.py`; re-measure Tier-1 errors. This is not table-export work — it is a 2-line correctness edit and must land in attempt 4 (or attempt 5) before any close-and-defer decision. The same pattern at `thompson_column.py:272-289` should be re-read end-to-end for any other cie/cig swaps; I did not have time to grep every `lami*` clamp in the file.
2. **(Minor)** Correct the stale `scenario:` text in `fixtures/manifests/analytic-thompson-column-v1.yaml:6-7` to reflect the terminal-velocity zero-patch mechanism, not the attempt-3 `1e30 m` workaround.
3. **(Reviewer-discretion)** Reconcile `thompson_gate_result.json` semantics: `FALLBACK` currently labels a correctness failure rather than a performance failure. Either rename the bucket to `CORRECTNESS-FAIL` or document that ADR-005's GO/GRAY/FALLBACK assumes correctness as a precondition and that Tier-1 fail short-circuits to a separate state.
4. **(Defer to M5-S1.x)** Lookup-table/moment-parity work: `t_Efrw`, `tps_iaus`, `tni_iaus`, freezing tables, snow/graupel moments. Diagnosis estimate 12-24h; this is the legitimate residual the BLOCKER names.

After fix 1, re-measure Tier-1. If T/`Ni`/q-species absolute errors fall inside or close to ADR-005 strict tolerances, A4 can close as "Accept-clean". If they remain in the 1e-4 mass-partition band as predicted by diagnosis-report.md:23-31, then Path C remains correct: serial M5-S1.x for the table work, hold M5-S2 until M5-S1.x closes.

I am explicit that the reviewer (codex) is the binding authority on A/B/C and on the attempt-5 vs M5-S1.x mechanism. My testing call is that the work cannot ship in its current state because of fix-1, but that the architectural ACs (1-launch, 0-byte HLO, Fortran-harness oracle, Tier-2, no-sedimentation patch, hot-path discipline) are real and load-bearing and should not be re-litigated.

---

## 6. Independent take on the 3-AI debate (≤200 words)

Manager and Gemini both lean Path B; diagnosis was pre-result; codex is binding. My read:

The diagnosis correctly predicted that order+Ni+sedimentation would close the dominant T error and the dominant Ni error. Attempt 4 confirmed that prediction harder than predicted (0.32K → 0.04K). What is left is exactly the mass-partition residual the diagnosis named — and Gemini #2 just showed that *part* of that residual is a 2-character transcription typo, not 12-24h of table-export work. That changes the cost/benefit:

- **Path A (close+defer, parallel M5-S1.x)** stops being attractive because the BLOCKER under-counts the reducible debt — at least the cie/cig typo, and possibly more if the reviewer greps thoroughly.
- **Path B (full attempt-5 with tables now)** is heavier than needed: fix the typo first, then re-decide.
- **Path C (accept-with-required-fixes; M5-S1.x serial before M5-S2)** matches my testing call. The cie/cig fix is in-scope for the current attempt; the rest is M5-S1.x.

The architectural milestone-progress claim (Fortran-harness oracle, single fused JAX call, hot-path discipline) is real and worth preserving. The "physics is divergent" framing in Gemini #3 is sharpened — not refuted — by Gemini #2's typo finding: there are reducible bugs hiding inside what attempt 4 called table-parity debt.
