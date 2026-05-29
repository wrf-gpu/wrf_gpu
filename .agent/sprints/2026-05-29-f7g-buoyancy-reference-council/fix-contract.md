# Sprint Contract — F7G-fix: WRF-signed metric convention + once-per-stage pg_buoy_w / work-variable t_2ave staging → idealized cases PASS (F7 dycore close)

**Sprint ID**: `2026-05-29-f7g-buoyancy-reference-council` (fix phase)
**Frontrunner**: Opus 4.8 (in-process Agent subagent, high/max effort)
**Branch**: `worker/opus/f7d-pressure-mass-fix` (CONTINUE — carries F7D mass fix + F7F IC-rebalance + calc_p_rho_phi geopotential-term fix). The manager merges the chain once the idealized cases PASS.
**GPU**: YES — `taskset -c 0-3`; `cuda:0`; fp64.

## Binding spec
The authoritative resolution is the GPT-5.5 council findings: **`.agent/sprints/2026-05-29-f7g-buoyancy-reference-council/gpt-council-findings.md`** — read in full. It proved from WRF source that the persistent idealized-case failure is **NOT** an architecture flaw and **NOT** the "balance-IC-against-dycore-operators" idea (refuted — WRF already has the closed-form discrete inverse). It is two concrete bugs. (agy/Gemini independent review at `agy-council-findings.md` — read it too; fold in any material addition. If agy and GPT conflict on the root cause, STOP and report to the manager.)

## The two root causes (WRF-grounded)
1. **Sign-convention mismatch → the 19× `pg_buoy_w(grid%p)` artifact.** WRF uses SIGNED vertical metrics: `dnw(k)=znw(k+1)−znw(k)` (negative for eta 1→0), `rdnw=1/dnw` (negative). WRF's idealized-IC geopotential recurrence (`module_initialize_ideal.F:977-983`, `:1121-1129`, `:1305-1313`) and the pressure diagnostic `calc_p_rho_phi` (`module_big_step_utilities_em.F:1023-1030,:1082-1088`; `start_em.F:819-868`) are **exact discrete inverses by construction** with signed metrics. The JAX idealized path uses **positive `|dnw|`/`|rdnw|`** ("upward-positive" convention in `idealized.py`) while applying WRF formulas that require signed `rdnw` → flips the diagnosed `al/p` for a balanced `ph'` → 19×.
2. **Staging double-count → buoyancy over/under-forcing.** WRF builds `pg_buoy_w` **once per RK stage** from stage `grid%p`/`mu` (`module_em.F:1361-1368`, `module_big_step_utilities_em.F:2553-2572`), NOT recomputed each acoustic substep. WRF's `muave` and `t_2ave` are small-step **work** averages: for an RK1 fixed-mass rest thermal they are **zero** on the first substep (`module_small_step_em.F:1102-1108`, `:1138-1144`, `:1341-1344`). The initial θ′ must NOT be carried as a direct `c2a·alt·t_2ave` buoyancy source — doing so double-counts the thermal.

## Cardinal rule
WRF Fortran source is ground truth. Verify every change against the cited lines under `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`. If WRF disagrees with this contract, WRF wins — note it.

## Scope — GPT-5.5 §3 fix spec (implement in order)

1. **WRF-signed vertical-metric view.** Either adopt WRF-signed `dnw/rdnw/dn/rdn` (negative for normal eta) throughout the WRF-shaped operators, OR add explicit `wrf_dnw/wrf_rdnw/wrf_rdn` adapters and route ALL WRF-faithful operators through them: `diagnose_pressure_al_alt`/`calc_p_rho_phi`, `core/calc_p_rho.py`, `pg_buoy_w_dry`, the horizontal nonhydrostatic PGF, `advance_w_wrf`, and the idealized IC rebalance in `ic_generators/idealized.py`. **Do not mix positive metrics with unmodified WRF signs.** In the IC builder, make the equation visibly WRF-equivalent (`ph(k+1)=ph(k)−wrf_dnw(k)·(…)`), not hidden behind `abs(dnw)`.
2. **`start_em`-equivalent post-init recompute** (`start_em.F:819-868`): before the first RK tendency, derive stage `al` and `p` from `ph_1`, `mu_1`, `mub`, `alb`, signed `rdnw`, and the EOS. Store as the stage `grid%p` equivalent.
3. **pg_buoy_w once per RK stage**: compute `rw_tend_pg_buoy` from stage `p`/`mu` BEFORE the acoustic substep scan and carry it unchanged through all substeps. `calc_p_rho(step=iteration)` stays ONLY for substep pressure/density refresh + smdiv memory — it is NOT a per-substep `pg_buoy_w` source. Delete the F7F workaround comments/switches asserting live small-step pressure is the `pg_buoy_w` source.
4. **t_2ave / muave work-variable semantics**: after `small_step_prep` on an RK1 fixed-mass rest thermal, work-theta old/new are zero so `advance_w` sees `t_2ave=0`, `muave=0`; nonzero only from actual small-step evolution. Remove any path that seeds `t_2ave` from the initialized full θ′.

## Acceptance gates (all required for `F7G_COMPLETE` = F7 dycore close) — GPT-5.5 §4 checks

- **AC1 — algebraic round-trip**: for warm-bubble and Straka IC columns, `al_init = alt_full − alb`; build `ph'`; diagnose `al_calc` via the WRF-signed `calc_p_rho_phi`. Require `max_abs(al_calc − al_init) ≤ 1e-12` (fp64 interior).
- **AC2 — no 19× artifact**: balanced rest column → direct vertical `rw_tend` residual `max_abs ≤ 1e-10 m/s²`; for an explicitly-unbalanced analytic-buoyancy oracle, `pg_buoy_w(grid%p)/analytic ∈ [0.9,1.1]`. A 9×/19× ratio fails.
- **AC3 — small-step reference**: first acoustic substep from RK1 fixed-mass rest IC has `max_abs(muave)=0` and `max_abs(t_2ave)=0` before `advance_w` term B; `pg_buoy_w` is the fixed stage array, not recomputed from `calc_p_rho_step`.
- **AC4 — Skamarock warm bubble PASS**: finite to 500 s, thermal rises (centroid ≥ 500 m), θ′ transported, max|w| physical (≤30), symmetric, mass-conserving, no NaN.
- **AC5 — Straka density current PASS**: finite to 900 s, front ≈ 15 km (±~2 km), min θ′ ≈ −9..−10 K, max|w| O(10), mass drift ≤ 1e-8.
- **AC6 — no regression**: A/B/C/D/F gates hold (no-stub, flat-rest=0, analytic dipole, 300-step conservation, circulation, mass identities); report the d02 operational-dt audit (did the signed-metric fix improve F7D's first_critical step 5?). No test weakened/xfailed.

## Proof objects (into `proofs/f7g/`)
`signed_metric_roundtrip.json` (AC1), `pg_buoy_ratio.json` (AC2), `rk1_rest_staging.json` (AC3), `straka_density_current.json`+verdict+plots, `skamarock_warm_bubble.json`+verdict+plots (AC4/AC5), `signed_metric_fix.md` (the convention + before/after + WRF file:line), `regression_recheck.json` (AC6), `worker-report.md` (AGENTS.md format) ending `F7G_COMPLETE` or `F7G_PARTIAL` + precise gaps.

## Hard rules
1. `taskset -c 0-3`; `cuda:0`; fp64.
2. WRF source is ground truth; cite file:line in every changed operator docstring.
3. **No masking clamps/caps/sanitizers, no coefficient tuning, no synthetic pressure.** The fix is sign-convention + WRF work-variable staging. If the cases still misbehave after this, STOP and report with traces — do NOT add a 7th workaround.
4. **No performance work** (no fp32/fusion).
5. Commit incrementally on `worker/opus/f7d-pressure-mass-fix`; do not push; do not switch branches.
6. Files writable: `src/gpuwrf/dynamics/**` (acoustic_wrf.py, core/calc_p_rho.py, core/advance_w.py, core/acoustic.py, metric/grid helpers), `src/gpuwrf/contracts/grid.py` (metric view, if needed), `src/gpuwrf/runtime/operational_mode.py`/`operational_state.py`, `src/gpuwrf/ic_generators/idealized.py`, `scripts/**` (instrumentation, never weaken invariants), `tests/**` (add/fix, never weaken), `proofs/f7g/**`, this sprint folder.
7. Files NOT writable: governance, memory, skills, ADRs, plan, physics-scheme code, comparator scripts under `scripts/m6b6_*`.
8. If the cases still fail after the signed-metric + staging fix, mark `F7G_PARTIAL`, deliver AC1-AC3 (the round-trip + ratio + staging proofs) and report with traces — that would mean a third distinct cause and warrants WRF-savepoint instrumentation (M9) before more guessing.

## Forward pointer
- On AC4+AC5 PASS → manager runs the GPT-5.5 WRF-domain **pre-close code critique** of the whole dycore, then declares the **F7 dynamical-core milestone closed**, merges the f7d chain to `manager-2026-05-23`.
- Then **F7-perf** (XLA fusion + fp32 downcast + ≥10× recert) and **M9** (instrumented WRF savepoints → per-operator parity, the rigorous near-identical-RMSE-vs-real-WRF gate).
