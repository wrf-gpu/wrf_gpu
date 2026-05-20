# Worker Report — ADR-007 Precision Policy

## objective

Produce ADR-007 for the precision-policy sprint, amend ADR-003, and generate precision benchmark proof objects for M2 column, M4 dycore, and M5 Thompson on RTX 5090 + 9950X (`.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:7`, `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:9`). The sprint explicitly forbids production downcast code, so all implementation work stayed in the benchmark script, ADRs, tests, and artifacts (`.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:13`, `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:20`).

## files changed

- `.agent/decisions/ADR-007-precision-policy.md`: new proof ADR with hardware context, speedup table, authorization matrix, 4x verdict, proof-object list, and review checklist (`.agent/decisions/ADR-007-precision-policy.md:1`, `.agent/decisions/ADR-007-precision-policy.md:14`, `.agent/decisions/ADR-007-precision-policy.md:22`, `.agent/decisions/ADR-007-precision-policy.md:40`, `.agent/decisions/ADR-007-precision-policy.md:55`).
- `.agent/decisions/ADR-003-dycore-precision.md`: amended to narrow the post-M5-S1 blanket fp64 lock through ADR-007, add a supersession note, and update the trigger language (`.agent/decisions/ADR-003-dycore-precision.md:12`, `.agent/decisions/ADR-003-dycore-precision.md:59`, `.agent/decisions/ADR-003-dycore-precision.md:109`, `.agent/decisions/ADR-003-dycore-precision.md:114`).
- `scripts/precision_bench.py`: new runner for CPU baselines, GPU precision cells, profiler probe, Gen2 CPU probe, and projected speedup summary (`scripts/precision_bench.py:32`, `scripts/precision_bench.py:433`, `scripts/precision_bench.py:470`, `scripts/precision_bench.py:493`, `scripts/precision_bench.py:538`).
- `tests/test_precision_bench.py`: new smoke test proving the precision bench CLI is callable and exposes the expected kernel set (`tests/test_precision_bench.py:12`, `tests/test_precision_bench.py:21`, `tests/test_precision_bench.py:24`).
- `artifacts/precision-bench/*`: new machine-readable proof objects for GPU/CPU timings, HLO, profiler limitation, Gen2 CPU probe, projected speedups, and validation summary (`.agent/decisions/ADR-007-precision-policy.md:101`, `.agent/decisions/ADR-007-precision-policy.md:105`, `.agent/decisions/ADR-007-precision-policy.md:108`, `.agent/decisions/ADR-007-precision-policy.md:109`).

## commands run

- `python scripts/precision_bench.py --run-all` exited 0 and generated the CPU/GPU precision cell artifacts plus `projected-speedups.json` (`artifacts/precision-bench/validation-summary.json:9`, `artifacts/precision-bench/validation-summary.json:10`, `artifacts/precision-bench/validation-summary.json:11`).
- `python scripts/validate_agentos.py` exited 0 with `ok=true`, 31 required files checked, and 13 skills checked (`artifacts/precision-bench/validation-summary.json:4`, `artifacts/precision-bench/validation-summary.json:5`, `artifacts/precision-bench/validation-summary.json:6`).
- `pytest -q` exited 0 with `399 passed, 1 skipped in 119.26s` after restoring existing gitignored external fixture payloads required by pre-existing tests (`artifacts/precision-bench/validation-summary.json:14`, `artifacts/precision-bench/validation-summary.json:15`, `artifacts/precision-bench/validation-summary.json:16`, `artifacts/precision-bench/validation-summary.json:19`).

## proof objects produced

- Profiler limitation proof: `ncu` and `nsys` were found, but `ncu` failed with `ERR_NVGPUCTRPERM`, so register count, occupancy, and local memory are intentionally null in benchmark records (`artifacts/precision-bench/profiler-probe.json:12`, `artifacts/precision-bench/profiler-probe.json:13`, `artifacts/precision-bench/profiler-probe.json:14`, `artifacts/precision-bench/profiler-probe.json:17`).
- Gen2 CPU probe: the Gen2 env and `wrf.exe` exist, but no `wrfinput*` files were found under the Gen2 WRF tree, so the exact Gen2 WRF 3 km timestep denominator remains unavailable (`artifacts/precision-bench/cpu-gen2-probe.json:2`, `artifacts/precision-bench/cpu-gen2-probe.json:9`, `artifacts/precision-bench/cpu-gen2-probe.json:10`, `artifacts/precision-bench/cpu-gen2-probe.json:11`).
- Projected speedup table: M2 column GPU cells are 0.278x/0.292x/0.327x vs sprint-local CPU FP64 for FP64/FP32/BF16 (`artifacts/precision-bench/projected-speedups.json:21`, `artifacts/precision-bench/projected-speedups.json:24`, `artifacts/precision-bench/projected-speedups.json:18`, `artifacts/precision-bench/projected-speedups.json:12`).
- Projected speedup table: M4 dycore GPU cells are 61.08x and 214.55x vs sprint-local CPU FP64 for FP64 and FP32, while BF16 is not applicable under the policy (`artifacts/precision-bench/projected-speedups.json:47`, `artifacts/precision-bench/projected-speedups.json:49`, `artifacts/precision-bench/projected-speedups.json:41`, `artifacts/precision-bench/projected-speedups.json:43`, `artifacts/precision-bench/projected-speedups.json:38`).
- Projected speedup table: M5 Thompson GPU cells are 0.175x/0.526x/0.386x vs sprint-local CPU FP64 for FP64/FP32/BF16; the FP32 cell is much faster than GPU FP64 but still launch-dominated on the tiny fixture (`artifacts/precision-bench/projected-speedups.json:72`, `artifacts/precision-bench/projected-speedups.json:74`, `artifacts/precision-bench/projected-speedups.json:66`, `artifacts/precision-bench/projected-speedups.json:68`, `artifacts/precision-bench/projected-speedups.json:60`, `artifacts/precision-bench/projected-speedups.json:62`).

## Authorization Matrix concise form

- `FP64-locked`: `state.mu`, `state.p`, `state.ph`, acoustic substep accumulator, and pressure-gradient accumulation (`.agent/decisions/ADR-007-precision-policy.md:57`, `.agent/decisions/ADR-007-precision-policy.md:58`, `.agent/decisions/ADR-007-precision-policy.md:59`, `.agent/decisions/ADR-007-precision-policy.md:60`, `.agent/decisions/ADR-007-precision-policy.md:61`).
- `FP32-OK`: non-acoustic `state.u`, `state.v`, `state.theta`, `state.qv`, hydrometeors `qc/qr/qi/qs/qg`, number fields `Ni/Nr`, `T`, non-acoustic advection tendencies, and Thompson source/sink arithmetic, all behind stability and operational RMSE gates (`.agent/decisions/ADR-007-precision-policy.md:62`, `.agent/decisions/ADR-007-precision-policy.md:63`, `.agent/decisions/ADR-007-precision-policy.md:65`, `.agent/decisions/ADR-007-precision-policy.md:66`, `.agent/decisions/ADR-007-precision-policy.md:67`, `.agent/decisions/ADR-007-precision-policy.md:68`, `.agent/decisions/ADR-007-precision-policy.md:69`, `.agent/decisions/ADR-007-precision-policy.md:70`, `.agent/decisions/ADR-007-precision-policy.md:71`, `.agent/decisions/ADR-007-precision-policy.md:72`, `.agent/decisions/ADR-007-precision-policy.md:73`, `.agent/decisions/ADR-007-precision-policy.md:74`, `.agent/decisions/ADR-007-precision-policy.md:76`, `.agent/decisions/ADR-007-precision-policy.md:77`).
- `needs-empirical-test`: `state.w` and `rho`; persistent BF16 state is not authorized because near-zero species and `q*rho` products remain a named risk (`.agent/decisions/ADR-007-precision-policy.md:64`, `.agent/decisions/ADR-007-precision-policy.md:75`, `.agent/decisions/ADR-007-precision-policy.md:28`).
- `BF16-OK`: bounded Thompson lookup/proxy coefficients only, not persistent conserved state (`.agent/decisions/ADR-007-precision-policy.md:29`, `.agent/decisions/ADR-007-precision-policy.md:78`).

## 4x verdict

ADR-007's verdict is that FP64-only is not feasible, and mixed precision is feasible under the proposed policy only if full-domain physics batching removes the M5 microfixture launch bottleneck (`.agent/decisions/ADR-007-precision-policy.md:22`, `.agent/decisions/ADR-007-precision-policy.md:49`, `.agent/decisions/ADR-007-precision-policy.md:117`). The immediate bottleneck path is Thompson/full physics, because the tiny M5 Thompson fixture is still slower than the CPU at FP32 despite a 3.0x GPU FP32-vs-FP64 improvement (`.agent/decisions/ADR-007-precision-policy.md:44`, `.agent/decisions/ADR-007-precision-policy.md:49`, `artifacts/precision-bench/m5_thompson-fp64.json:23`, `artifacts/precision-bench/m5_thompson-fp32.json:23`).

## ADR-003 amendment diff

ADR-003 now explicitly says ADR-007 narrows the blanket lock after M5-S1, names the still-locked fields/components, names the FP32 follow-on candidates, and records that no production dtype change is made by the amendment (`.agent/decisions/ADR-003-dycore-precision.md:12`). ADR-003 also marks its original authorization matrix as historical for M4/M5-S1 and points post-M5-S1 precision work to ADR-007 (`.agent/decisions/ADR-003-dycore-precision.md:59`). The trigger section now records M5-S1 `GO_CARRYFORWARD` as superseded by ADR-007 for post-M5-S1 precision authorization while keeping strict parity debt and operational RMSE gates in force (`.agent/decisions/ADR-003-dycore-precision.md:109`, `.agent/decisions/ADR-003-dycore-precision.md:114`).

## unresolved risks

- Exact Gen2 WRF 3 km CPU timing remains missing because no runnable `wrfinput*` files were present under the Gen2 WRF tree during this sprint (`artifacts/precision-bench/cpu-gen2-probe.json:7`, `artifacts/precision-bench/cpu-gen2-probe.json:11`).
- Register/local-memory/occupancy counters remain missing until NVIDIA perf-counter permission is enabled (`artifacts/precision-bench/profiler-probe.json:17`).
- The M5 Thompson FP32 result does not prove operational 4x speedup by itself; full-domain batching and M6/M7 operational RMSE validation are required before production downcast (`.agent/decisions/ADR-007-precision-policy.md:49`, `.agent/decisions/ADR-007-precision-policy.md:99`, `.agent/decisions/ADR-007-precision-policy.md:117`).

## next decision needed

Tester/reviewer should decide whether ADR-007's conditional feasibility wording is acceptable: it says mixed precision is feasible under the policy, but only if follow-on full-domain physics batching closes the M5 launch-bound microfixture gap (`.agent/decisions/ADR-007-precision-policy.md:22`, `.agent/decisions/ADR-007-precision-policy.md:117`).
