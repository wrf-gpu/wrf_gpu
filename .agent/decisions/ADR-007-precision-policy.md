# ADR-007 — Mixed Precision Authorization Policy

Date: 2026-05-20
Author: ADR-007 worker (codex gpt-5.5 xhigh)
Status: **PROPOSED for tester/reviewer acceptance.** No production code downcast is implemented by this ADR; follow-on scheme sprints must implement and validate the authorized rows.
Scope: RTX 5090 + Ryzen 9 9950X precision policy for M2 column, M4 dycore, and M5 Thompson evidence.

## Status

ADR-007 amends ADR-003's blanket M5-S1 fp64 production lock into a per-field authorization matrix. It does not modify production `src/gpuwrf` code. It authorizes follow-on implementation sprints to downcast named fields or arithmetic paths only where this ADR says `FP32-OK` or `BF16-OK`, and only with the operational RMSE gate below.

## Context

The sprint was triggered because the Stage-M4 Gemini review identified RTX 5090 FP64 throttling as a project-existential risk: the review states the RTX 5090 has a 1:64 FP64:FP32 rate, around 1.7-1.8 TFLOP/s FP64, while the Ryzen 9 9950X can reach roughly 2.2 TFLOP/s FP64 with AVX-512 (`.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:24`, `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:26`, `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:27`, `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:34`). That review concluded that a pure FP64 compute-heavy physics path cannot reach the 4x target (`.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:47`, `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:55`).

The binding validation philosophy changed the precision gate from per-cell parity to operational forecast impact: Tier-4 `U10`, `V10`, and `T2` RMSE is binding, and per-field downcast is gated by operational impact rather than tiny pointwise deltas (`/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:17`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:19`). Stability remains the floor: mass continuity and pressure-gradient paths remain FP64 while other paths are candidates (`/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:21`).

The M4 baseline has known limits. It has zero post-init transfers, zero temporary bytes, and 24 HLO-derived dycore launches (`.agent/decisions/MILESTONE-M4-CLOSEOUT.md:40`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:42`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:43`), but acoustic validation is still a reduced proxy and canonical `mu`/density mass continuity is not yet present (`.agent/decisions/MILESTONE-M4-CLOSEOUT.md:60`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:67`). M5 Thompson reached `GO_CARRYFORWARD`, one HLO-derived launch, and tier-2 finite/positivity/water-budget pass, but strict ADR-005 parity debt remains and register counters are null due to perfmon restriction (`artifacts/m5/thompson_gate_result.json:2`, `artifacts/m5/thompson_gate_result.json:3`, `artifacts/m5/thompson_gate_result.json:5`, `artifacts/m5/tier2_thompson_invariants.json:49`, `artifacts/m5/tier2_thompson_invariants.json:50`, `artifacts/m5/tier2_thompson_invariants.json:54`).

## Decision

Decision: **mixed precision is feasible under the proposed policy, but FP64-only is not feasible for the 4x target.** The named bottleneck path is full-domain physics, beginning with Thompson. The M5 microfixture is launch-dominated and still slower than the CPU at FP32, but it proves a material GPU FP64-to-FP32 wall-clock improvement on the same JAX kernel. M6/M7 must batch physics over operational domains, keep pressure/mass paths FP64, and validate `U10/V10/T2` operational RMSE before production downcast.

Production policy:

1. `mu`, pressure/geopotential fields, pressure-gradient accumulation, and acoustic accumulators remain `FP64-locked`.
2. Non-acoustic advection arithmetic, passive/thermodynamic tracers, and Thompson hydrometeor mass/number fields are `FP32-OK` for follow-on implementation sprints after stability-floor checks.
3. Persistent BF16 state is not authorized yet. BF16 is `needs-empirical-test` for hydrometeors, vapor, number concentrations, and thermodynamics because near-zero species and `q*rho` products can lose meaningful bits.
4. `BF16-OK` is limited to bounded non-conservative intermediates, such as lookup/proxy coefficient values, after M5-S1.x exports tables as device arrays and after the affected scheme proves no operational RMSE regression.
5. The binding production gate for all `OK` rows is operational RMSE: 24h and 72h GPU-vs-CPU deltas for `U10`, `V10`, and `T2` must stay below the CPU-vs-observation noise floor over a representative Canary sample.

## Evidence

Profiler limitation: `ncu` and `nsys` are installed, but the `ncu` probe failed with `ERR_NVGPUCTRPERM`, so register count, occupancy, and local memory are null in these artifacts (`artifacts/precision-bench/profiler-probe.json:12`, `artifacts/precision-bench/profiler-probe.json:13`, `artifacts/precision-bench/profiler-probe.json:14`, `artifacts/precision-bench/profiler-probe.json:17`). This satisfies the sprint's allowed null-counter path while preserving the raw profiler failure artifact (`.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:73`, `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:74`, `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:89`).

Exact Gen2 WRF CPU 3 km timing could not be produced: the Gen2 env and `wrf.exe` exist, but no `wrfinput*` candidates were found under the Gen2 WRF tree (`artifacts/precision-bench/cpu-gen2-probe.json:2`, `artifacts/precision-bench/cpu-gen2-probe.json:9`, `artifacts/precision-bench/cpu-gen2-probe.json:10`, `artifacts/precision-bench/cpu-gen2-probe.json:11`). Therefore the speedup table below uses sprint-local JAX CPU FP64 timings for the same kernels as a directly comparable denominator; the missing Gen2 denominator remains a validation gap, not hidden evidence (`artifacts/precision-bench/projected-speedups.json:3`, `artifacts/precision-bench/projected-speedups.json:4`).

Projected speedups, GPU wall time and speedup vs sprint-local CPU FP64:

| Kernel | CPU FP64 artifact | GPU FP64 | GPU FP32 | GPU BF16 |
|---|---|---:|---:|---:|
| M2 column | `artifacts/precision-bench/cpu-m2_column-fp64.json` | 26.645 us, 0.278x (`artifacts/precision-bench/projected-speedups.json:22`, `artifacts/precision-bench/projected-speedups.json:23`, `artifacts/precision-bench/projected-speedups.json:24`) | 25.375 us, 0.292x (`artifacts/precision-bench/projected-speedups.json:16`, `artifacts/precision-bench/projected-speedups.json:17`, `artifacts/precision-bench/projected-speedups.json:18`) | 22.665 us, 0.327x (`artifacts/precision-bench/projected-speedups.json:10`, `artifacts/precision-bench/projected-speedups.json:11`, `artifacts/precision-bench/projected-speedups.json:12`) |
| M4 dycore | `artifacts/precision-bench/cpu-m4_dycore-fp64.json` | 713.465 us, 61.08x (`artifacts/precision-bench/projected-speedups.json:47`, `artifacts/precision-bench/projected-speedups.json:48`, `artifacts/precision-bench/projected-speedups.json:49`) | 203.1115 us, 214.55x (`artifacts/precision-bench/projected-speedups.json:41`, `artifacts/precision-bench/projected-speedups.json:42`, `artifacts/precision-bench/projected-speedups.json:43`) | not applicable while pressure/acoustic/mass paths remain locked (`artifacts/precision-bench/projected-speedups.json:35`, `artifacts/precision-bench/projected-speedups.json:36`, `artifacts/precision-bench/projected-speedups.json:38`) |
| M5 Thompson | `artifacts/precision-bench/cpu-m5_thompson-fp64.json` | 141.431 us, 0.175x (`artifacts/precision-bench/projected-speedups.json:72`, `artifacts/precision-bench/projected-speedups.json:73`, `artifacts/precision-bench/projected-speedups.json:74`) | 46.995 us, 0.526x (`artifacts/precision-bench/projected-speedups.json:66`, `artifacts/precision-bench/projected-speedups.json:67`, `artifacts/precision-bench/projected-speedups.json:68`) | 64.1105 us, 0.386x (`artifacts/precision-bench/projected-speedups.json:60`, `artifacts/precision-bench/projected-speedups.json:61`, `artifacts/precision-bench/projected-speedups.json:62`) |

Interpretation:

- The M4 dycore benefits strongly from GPU execution even in FP64 and improves another 3.5x when the benchmark state is FP32 (`artifacts/precision-bench/m4_dycore-fp64.json:22`, `artifacts/precision-bench/m4_dycore-fp32.json:22`). This does not authorize BF16 dycore because the acoustic and mass-continuity evidence gaps remain (`.agent/decisions/MILESTONE-M4-CLOSEOUT.md:60`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:67`).
- The M5 Thompson microfixture improves 3.0x from FP64 to FP32 on GPU, but remains slower than the tiny CPU JAX baseline because the fixture has only 3 x 12 columns and one GPU launch (`artifacts/precision-bench/m5_thompson-fp64.json:21`, `artifacts/precision-bench/m5_thompson-fp64.json:23`, `artifacts/precision-bench/m5_thompson-fp32.json:21`, `artifacts/precision-bench/m5_thompson-fp32.json:23`, `artifacts/precision-bench/cpu-m5_thompson-fp64.json:21`). Full-domain batching is therefore the named bottleneck path for feasibility, not further microfixture timing.

## Authorization Matrix

`FP32-OK` means a follow-on implementation sprint may downcast the named persistent field or arithmetic path, provided it keeps FP64 accumulation where specified and passes stability plus operational RMSE gates. It does not mean this ADR changed production dtype code.

| Field / component | Verdict | Authorized target | Stability basis or empirical plan | Evidence |
|---|---|---|---|---|
| `state.mu` / column dry mass | `FP64-locked` | FP64 only | Canonical `mu` continuity is not implemented in M4; mass continuity is explicitly protected. | `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:67`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:70`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:21` |
| `state.p` | `FP64-locked` | FP64 only | Pressure-gradient and acoustic-adjacent path; catastrophic-cancellation-prone. | `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:16`, `.agent/decisions/ADR-003-dycore-precision.md:68`, `.agent/decisions/ADR-003-dycore-precision.md:69` |
| `state.ph` | `FP64-locked` | FP64 only | Geopotential/pressure-adjacent field; acoustic validation remains a proxy. | `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:60`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:65`, `.agent/decisions/ADR-003-dycore-precision.md:69` |
| Acoustic substep accumulator | `FP64-locked` | FP64 only | Sound-wave validation is deferred; no fp32 mass-residual proof exists. | `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:60`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:65`, `.agent/decisions/ADR-003-dycore-precision.md:68` |
| Pressure-gradient accumulation | `FP64-locked` | FP64 only | **Stability basis (catastrophic cancellation)**: pressure-gradient terms involve subtraction of nearly-equal pressures across adjacent vertical levels (typical Δp/p ~ 1e-3 to 1e-5 in lower troposphere). FP32 has ~7 decimal digits of mantissa precision; relative subtraction errors at Δp/p = 1e-5 cost ~5 of those, leaving 2 — operationally insufficient for hydrostatic/momentum balance. FP64 retains ~15 digits, leaving ~10 after the subtraction. Contract scope (`sprint-contract.md:16`) confirmed this with the user's validation philosophy that mass/pressure paths stay FP64. | `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:16`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:21`, `.agent/decisions/ADR-003-dycore-precision.md:68` |
| `state.u` / U wind | `FP32-OK` | FP32 storage and non-acoustic tendency arithmetic; FP64 acoustic/pressure accumulation | U10 is a binding Tier-4 output, so production merge requires 24h/72h U10 RMSE gate. | `artifacts/precision-bench/m4_dycore-fp32.json:22`, `artifacts/precision-bench/projected-speedups.json:41`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:17` |
| `state.v` / V wind | `FP32-OK` | FP32 storage and non-acoustic tendency arithmetic; FP64 acoustic/pressure accumulation | V10 is a binding Tier-4 output, so production merge requires 24h/72h V10 RMSE gate. | `artifacts/precision-bench/m4_dycore-fp32.json:22`, `artifacts/precision-bench/projected-speedups.json:43`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:17` |
| `state.w` / W wind | `needs-empirical-test` | Candidate FP32 only after sound-wave validation | **Concrete empirical test plan** (executes in M6 coupled-forecast sprint): (1) run paired 24h Canary-3km forecasts identical except `state.w` dtype = FP64 vs FP32, same IC/BC from Gen2 AIFS; (2) gate on `|U10/V10/T2 RMSE_fp32 - RMSE_fp64| < 0.10 K and 0.10 m/s` averaged across the domain at 6h, 12h, 24h leads; (3) inspect vertical-velocity column spectra at 4 sample points (sea, lee, ridge, peak) for spurious noise injection that could propagate into θ/q tendencies via vertical advection; (4) check Tier-2 water-budget residual stays ≤ 1e-10 fractional. Pass-all → reclassify to `FP32-OK`. Fail-any → keep `FP64-locked` and re-evaluate at M7. | `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:60`, `.agent/decisions/MILESTONE-M4-CLOSEOUT.md:65`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:17` |
| `state.theta` | `FP32-OK` | FP32 storage with FP64 conservation boundary | M4 FP32 dycore timing improves materially; production requires theta/tracer invariants and T2 RMSE gate. | `artifacts/precision-bench/m4_dycore-fp32.json:22`, `artifacts/precision-bench/m4_dycore-fp32.json:31`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:17` |
| `state.qv` / vapor | `FP32-OK` | FP32 storage/tendency; BF16 requires empirical test | Thompson tier-2 positivity and water budget pass in FP64; BF16 risk exists for `qv*rho` products. | `artifacts/m5/tier2_thompson_invariants.json:50`, `artifacts/m5/tier2_thompson_invariants.json:54`, `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:90` |
| `qc` cloud water | `FP32-OK` | FP32 persistent physics field | Non-negative hydrometeor with tier-2 positivity floor; operational RMSE gate remains binding. | `artifacts/m5/tier2_thompson_invariants.json:50`, `artifacts/precision-bench/m5_thompson-fp32.json:23`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:19` |
| `qr` rain water | `FP32-OK` | FP32 persistent physics field | Non-negative hydrometeor; water budget floor passes, BF16 near-zero behavior needs empirical test. | `artifacts/m5/tier2_thompson_invariants.json:54`, `artifacts/m5/tier2_thompson_invariants.json:56`, `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:90` |
| `qi` cloud ice | `FP32-OK` | FP32 persistent physics field | Non-negative hydrometeor; M5 tier-2 stability floor passes. | `artifacts/m5/tier2_thompson_invariants.json:49`, `artifacts/m5/tier2_thompson_invariants.json:50` |
| `qs` snow | `FP32-OK` | FP32 persistent physics field | Non-negative hydrometeor; M5 tier-2 stability floor passes. | `artifacts/m5/tier2_thompson_invariants.json:49`, `artifacts/m5/tier2_thompson_invariants.json:50` |
| `qg` graupel | `FP32-OK` | FP32 persistent physics field | Non-negative hydrometeor; M5 tier-2 stability floor passes. | `artifacts/m5/tier2_thompson_invariants.json:49`, `artifacts/m5/tier2_thompson_invariants.json:50` |
| `Ni` ice number | `FP32-OK` | FP32 persistent physics field | Number field is bounded non-negative in Thompson; BF16 still needs empirical test. | `artifacts/m5/tier2_thompson_invariants.json:50`, `artifacts/precision-bench/m5_thompson-fp32.json:23` |
| `Nr` rain number | `FP32-OK` | FP32 persistent physics field | Number field is bounded non-negative, but near-zero BF16 precision is a named sprint risk. | `artifacts/m5/tier2_thompson_invariants.json:50`, `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:90` |
| `T` / temperature | `FP32-OK` | FP32 physics/dycore thermodynamic storage; FP64 where used by locked accumulators | T2 is a binding operational metric; Thompson finite latent-heating floor passes. | `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:17`, `artifacts/m5/tier2_thompson_invariants.json:39`, `artifacts/m5/tier2_thompson_invariants.json:42` |
| `rho` density diagnostic | `needs-empirical-test` | Recompute from `p/T/qv`; do not persist BF16 | Used in `q*rho` products and denominators; test FP32 recompute against 24h/72h stability before storage downcast. | `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:90`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:21` |
| Non-acoustic advection tendencies | `FP32-OK` | FP32 arithmetic with FP64 conservation boundary | M4 FP32 dycore benchmark is 203.1115 us vs 713.465 us FP64 on GPU; launch count unchanged at 24. | `artifacts/precision-bench/m4_dycore-fp32.json:20`, `artifacts/precision-bench/m4_dycore-fp32.json:22`, `artifacts/precision-bench/m4_dycore-fp64.json:22` |
| Thompson source/sink arithmetic | `FP32-OK` | FP32 arithmetic; FP64 coupling boundary for mass/pressure | M5 Thompson GPU FP32 is 46.995 us vs 141.431 us FP64; tier-2 stability floor passes in current FP64 reference. | `artifacts/precision-bench/m5_thompson-fp32.json:23`, `artifacts/precision-bench/m5_thompson-fp64.json:23`, `artifacts/m5/tier2_thompson_invariants.json:49` |
| Bounded Thompson lookup/proxy coefficients | `BF16-OK` | BF16 intermediates only, not persistent conserved state | Values are bounded non-conservative coefficients; still require M5-S1.x exported-table proof and operational RMSE gate before production. | `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:100`, `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:101`, `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:19` |

## Alternatives

1. Keep ADR-003's blanket FP64 lock. Rejected because the hardware context says FP64 compute on RTX 5090 is not competitive with the 9950X for compute-heavy physics (`.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:34`, `.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md:37`).
2. Authorize broad BF16 persistent state immediately. Rejected because BF16 risks were explicitly named for `Nr` near zero and `qv*rho` products (`.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md:90`).
3. Move to a data-center GPU to preserve FP64 everywhere. Kept as fallback if full-domain mixed precision fails the 4x gate; it does not solve the policy issue on the target RTX 5090.
4. Replace compute-heavy physics with an ML hybrid emulator. Kept as fallback if full-domain FP32/BF16-safe physics cannot reach 4x or cannot pass operational RMSE.

## Consequences

Positive consequences:

- ADR-003 no longer blocks all downcast work after M5-S1. Hydrometeor and thermodynamic paths now have concrete FP32 follow-on authorization.
- The policy matches the user's validation framework: operational RMSE decides production fitness, while per-cell parity stays a bug-finding sanity check.
- Performance work now has named bottlenecks: full-domain physics batching, Thompson table export/dynamic gather, and launch amortization.

Costs and risks:

- The exact Gen2 WRF 3 km CPU denominator is still missing because runnable input files were not present under the Gen2 WRF tree (`artifacts/precision-bench/cpu-gen2-probe.json:7`, `artifacts/precision-bench/cpu-gen2-probe.json:11`).
- Register/local-memory/occupancy counters remain missing until perfmon permissions are enabled (`artifacts/precision-bench/profiler-probe.json:17`).
- M5 Thompson FP32 does not yet beat the CPU on the tiny analytic fixture, so the 4x feasibility claim depends on full-domain batching evidence in M6/M7 (`artifacts/precision-bench/projected-speedups.json:66`, `artifacts/precision-bench/projected-speedups.json:68`).

## Proof Objects

- Benchmark runner: `scripts/precision_bench.py`
- Sanity test: `tests/test_precision_bench.py`
- Speedup summary: `artifacts/precision-bench/projected-speedups.json`
- GPU artifacts: `artifacts/precision-bench/m2_column-{fp64,fp32,bf16}.json`, `artifacts/precision-bench/m4_dycore-{fp64,fp32,bf16}.json`, `artifacts/precision-bench/m5_thompson-{fp64,fp32,bf16}.json`
- CPU artifacts: `artifacts/precision-bench/cpu-m2_column-fp64.json`, `artifacts/precision-bench/cpu-m4_dycore-fp64.json`, `artifacts/precision-bench/cpu-m5_thompson-fp64.json`
- Profiler limitation proof: `artifacts/precision-bench/profiler-probe.json`
- Gen2 CPU probe: `artifacts/precision-bench/cpu-gen2-probe.json`
- ADR-003 amendment: `.agent/decisions/ADR-003-dycore-precision.md`

## Review

Pending tester and binding reviewer. Required review checks:

1. Each matrix verdict must cite either a stability argument or an empirical-test plan.
2. The 4x verdict must be read as: FP64-only is infeasible; mixed precision is feasible only if full-domain physics batching removes the M5 microfixture launch bottleneck.
3. ADR-003 must consistently cross-reference this ADR and must not imply production code was downcast in this sprint.
