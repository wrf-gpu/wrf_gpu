# Worker Report — c2-A1 JAX/WRF Dycore Architecture

Date: 2026-05-22
Worker: codex

## Objective

Freeze and prove the c2 architecture skeleton for a JAX/XLA WRF-compatible dycore: WRF map factors and hybrid-eta coefficients in `GridSpec.metrics`, base and boundary state separated from prognostic `State`, disabled-by-default stabilizer modules, nested scan carry for previous pressure and accumulators, and required proof objects.

## Outcome

Summary: architecture skeleton implemented; AC1-AC6 pass with proof objects, AC7 is partial analytic smoke only, and AC8 recommends continuing C with the warm-bubble harness as the next gate. AC1 final ACCEPT after spike absorption: ADR-020 and the ADR-002 amendment now include explicit total/perturbation state decomposition, terrain slope metrics, and a WRF-anchored well-balanced PGF requirement.

Recommendation: continue C implementation, with one explicit caveat: AC7 is a finite architecture smoke, not warm-bubble parity. The referenced `scripts/m6_warm_bubble_test.py` is absent in this worktree, so c2-A2 should first restore/build the warm-bubble harness before claiming physical integration progress.

Spike absorption update: the spike report is available on `origin/worker/codex/m6x-numerical-stability-spike`. It found flat warm-bubble nonfinite at step 76 (150 s), mountain nonfinite at step 36 (70 s), and unchanged flat failure after brute-force `smdiv=0.1` plus top Rayleigh sponge. c2-A1 now treats damping as secondary infrastructure and makes base-state/perturbation decomposition plus well-balanced PGF day-1 architecture commitments.

## AC Status

| AC | Status | Evidence |
|---|---|---|
| AC1 ADR/architecture | PASS final after spike absorption | `architecture.md`, `.agent/patches/2026-05-22-c2-adr-002-amendment.md`, `ADR-020-c2-dycore-architecture.md` |
| AC2 metrics | PASS | `proofs/metrics.json`; WRF `wrfinput_d02` map-factor shapes loaded |
| AC3 hybrid eta | PASS | `proofs/hybrid_eta.json`; analytic oracle max error 0.0; WRF coeffs loaded |
| AC4 damping/diffusion/limiter skeletons | PASS | `tests/test_m6x_c2_stabilizers.py` |
| AC5 scan carry | PASS with audit limitation | `proofs/scan_transfer_audit.md`; static JAXPR audit, final carry on GPU |
| AC6 limiter conservation | PASS | `proofs/limiter_conservation.json`; relative mass error 0.0 |
| AC7 integration | PARTIAL | `proofs/integration_warm_bubble.json`; finite analytic smoke only |
| AC8 decision gate | PASS | `manager-closeout.md` decision-gate draft recommends continue |

## Files Changed

- `src/gpuwrf/contracts/grid.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/dynamics/metrics.py`
- `src/gpuwrf/dynamics/hybrid_eta.py`
- `src/gpuwrf/dynamics/damping.py`
- `src/gpuwrf/dynamics/hyperdiffusion.py`
- `src/gpuwrf/dynamics/limiters.py`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/dynamics/orchestrator.py`
- `tests/test_m6x_c2_metrics.py`
- `tests/test_m6x_c2_hybrid_eta.py`
- `tests/test_m6x_c2_stabilizers.py`
- `tests/test_m6x_c2_scan.py`
- `tests/test_m3_state.py`
- `tests/test_m6_state_extension.py`
- `src/gpuwrf/contracts/precision.py`
- `scripts/precision_bench.py`
- `scripts/m6x_c2_generate_proofs.py`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/architecture.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/*`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/worker-report.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/manager-closeout.md`
- `.agent/patches/2026-05-22-c2-adr-002-amendment.md`
- `.agent/decisions/ADR-020-c2-dycore-architecture.md`

## Commands Run

- `sed`/`rg` reads of the required sprint prompt, sprint contract, scout report, constitution, AGENTS, ADRs, morning report, methodology review, bughunt report, current contracts/dynamics, and WRF `dyn_em` source anchors.
- `PYTHONPATH=src pytest -q tests/test_m6x_c2_metrics.py tests/test_m6x_c2_hybrid_eta.py tests/test_m6x_c2_stabilizers.py tests/test_m6x_c2_scan.py` -> `13 passed`
- `python -m py_compile src/gpuwrf/contracts/state.py src/gpuwrf/contracts/grid.py src/gpuwrf/contracts/precision.py src/gpuwrf/dynamics/metrics.py src/gpuwrf/dynamics/acoustic_wrf.py src/gpuwrf/dynamics/orchestrator.py scripts/m6x_c2_generate_proofs.py scripts/precision_bench.py`
- `PYTHONPATH=src pytest -q tests/test_m3_state.py tests/test_m6_state_extension.py tests/test_m4_acoustic.py tests/test_m4_rk3.py tests/test_m4_dycore_step.py` -> `15 passed`
- `PYTHONPATH=src python scripts/m6x_c2_generate_proofs.py`
- `python scripts/close_sprint.py .agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture` (fails pending independent `reviewer-report.md`, `tester-report.md`, and `memory-patch.md`)
- `git show origin/worker/codex/m6x-numerical-stability-spike:.agent/sprints/2026-05-22-m6x-numerical-stability-spike/worker-report.md` equivalent read via sibling worktree `/tmp/wrf_gpu2_spike`

## Proof Objects Produced

- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/metrics.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/hybrid_eta.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/scan_transfer_audit.md`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/limiter_conservation.json`
- `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/integration_warm_bubble.json`

## Unresolved Risks

- AC7 does not prove warm-bubble parity because the referenced warm-bubble script is missing.
- `scan_transfer_audit.md` is a static JAXPR audit plus executed GPU scan, not an Nsight trace; no GPU performance claim is made.
- The new acoustic module is a WRF-shaped scan skeleton, not WRF small-step numerics.
- Existing M4/M6 APIs still pass `GridSpec` as a static argument; c2 functions pass `DycoreMetrics` as a pytree to avoid WRF-sized static cache keys.

## Next Decision Needed

Manager/reviewer should decide whether the AC7 limitation is acceptable for c2-A1 architecture closeout. If accepted, dispatch c2-A2 to implement WRF small-step numerics and restore/build the warm-bubble harness before broader dycore work.
