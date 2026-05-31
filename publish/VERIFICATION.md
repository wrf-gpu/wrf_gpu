# v0.1.0 Verification & Proof Table (reproducible)

Status: the **binding proof contract** for v0.1.0, 2026-05-31. Every IN-SCOPE claim below
must reach **irrefutable PASS on the release commit**, and must be reproducible by anyone via
a single command. Limitations (the roadmap *gaps*) are explicitly OUT of scope here and are
documented in [`GPU_PORT_GAPS_TODO.md`](GPU_PORT_GAPS_TODO.md) — but everything IN scope must
be perfect, science-publication grade, not "mostly works".

**Master command (publishes the PASS/FAIL table):**
```bash
bash scripts/verify_all.sh        # regenerates/checks every row, emits proofs/PROOF_TABLE.md
```
Each row also has a standalone `scripts/verify/<row>.sh` that re-runs that single proof from
source (not from a cached JSON) and asserts the gate, so a reviewer can reproduce piecewise.

## Scope

- **IN scope (must be irrefutable + reproducible):** idealized dynamical-core benchmarks,
  operator parity vs pristine WRF v4 savepoints, **Canary 3 km (d02)** and **Canary 1 km (d03)**
  real-case correctness/stability/skill vs CPU-WRF, conservation/invariants, bit-reproducibility/
  restart, performance characterization, and an honest precipitation characterization.
- **OUT of scope (documented limitations, not blockers):** live multi-domain nesting, native
  WPS/real.exe initialization, prognostic Noah-MP, d01 cumulus, multi-GPU single-forecast,
  and — if the reuse-only corpus cannot support it — *all-season* TOST coverage (the available
  cases must still pass; seasonal breadth is a documented v0.2.0 gap).

## The proof table

| # | Claim (must be true for v0.1.0) | Gate | Proof object | Verify script | Status |
|---|---|---|---|---|---|
| 1 | Dycore: Skamarock warm bubble matches the benchmark reference | within published tol | `proofs/sprintU/close_gate/warm_bubble_verdict.json` | `verify/idealized_warmbubble.sh` | PASS (re-confirm on release commit) |
| 2 | Dycore: Straka density current matches the benchmark reference | within published tol | `proofs/sprintU/close_gate/density_current_verdict.json` | `verify/idealized_straka.sh` | PASS (re-confirm) |
| 3 | Operator parity vs pristine WRF v4 savepoints | per-operator tol | `proofs/f7/DYCORE_STATUS.md` + savepoints | `verify/savepoint_parity.sh` | re-confirm on release commit |
| 4 | **Canary 3 km (d02)**: finite & stable to 72 h, no blow-up, near-CPU-WRF, beats persistence on winds | all field-scores finite; U10/V10 beat persistence | `proofs/v010_validation/v010_d02_result.json` | `verify/d02_validation.sh` | re-run on FINAL code (post-HFX) |
| 5 | **Canary 1 km (d03)**: finite & stable to 24 h, near-CPU-WRF, beats persistence, **passes bounded gate** | T2 RMSE ≤ gate; beats persistence | `proofs/v010_validation/d03_summary_*.json` | `verify/d03_validation.sh` | **BLOCKED → needs HFX fix (#56)** |
| 6 | Equivalence (paired TOST) on all usable corpus cases, T2/U10/V10 | predeclared margins, honest n | `proofs/m20/*` | `verify/tost.sh` | run on achievable N; seasonal breadth → limitation |
| 7 | Conservation: dry-mass / water / energy budgets bounded; guards not load-bearing | bounded; guards-off finite | `proofs/.../conservation_*.json` | `verify/conservation.sh` | **run on FINAL code** |
| 8 | Reproducibility: deterministic re-run + restart-continuity | bitwise/within-tol identical | `proofs/v010_validation/{repeatability,restart_in_pipeline}.json` | `verify/repeatability.sh` | **run (currently NOT_RUN)** |
| 9 | Performance: roofline-grounded ~5.3×/7.8× vs 28-rank CPU-WRF d02 | provenance-backed | `proofs/perf/*` + `publish/runtime_optimization_analysis.md` | `verify/performance.sh` | PASS (re-confirm) |
| 10 | Precipitation: physically correct & functional (honest, not parity) | precipitates correctly; bias characterized | `proofs/thompson_perf/*` | `verify/precip.sh` | characterize honestly |
| 11 | Device residency: zero host↔device transfer inside the timestep loop | transfer count = 0 | `proofs/perf/fusion_transfer_audit.py` | `verify/device_residency.sh` | **run audit, emit count** |

## Release rule

`v0.1.0` is tagged **only when rows 1–11 are all PASS on the release commit** (row 6 = pass on
the achievable N with seasonal breadth documented; row 10 = honest characterization). The d02
and d03 validations MUST be re-run on the final post-HFX-fix code and tied to the release
commit — no claim may rest on a pre-fix proof. Every number in the paper traces to a row here.
