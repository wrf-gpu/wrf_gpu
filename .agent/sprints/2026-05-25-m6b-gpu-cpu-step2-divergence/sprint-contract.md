# Sprint Contract — M6b GPU vs CPU Step-2 Divergence Isolation

## Objective

Before reboot, the V3 localization comparator on **GPU** showed:
- step 1: max_abs_delta = 0.0 (PASS)
- step 2-50: max_abs_delta = 1e+300 (NaN sentinel, all_fields_finite=False, largest_bad_field=mu)

But the prior CPU multi-step bisect on the same operational_mode reported 2/5/10 = 0.0 bitwise. And the V3 bounds-check itself (also on GPU) was finite through step 45 before the wind violation at step 46.

This is a contradiction. Three hypotheses:
- **(A) Sentinel coincidence**: both validation_wrappers AND operational explode at step 2 on GPU with the same NaN, comparator's max_abs_delta of NaN-NaN = NaN reported as 1e+300.
- **(B) GPU-shared bug**: validation_wrappers and operational both explode the same way on GPU but are bitwise on CPU. Real GPU-only numerical issue in dynamics/core/.
- **(C) Operational-only GPU bug**: only operational explodes on GPU; validation_wrappers stays finite. Reframe missed a path.

This sprint isolates which.

## Non-Goals

- NO modifications to `dynamics/core/`.
- NO modifications to operational composition.
- NO 1h forecast (this is a 10-step probe).
- NO remote push.

## File Ownership

Worktree **already created** at `/tmp/wrf_gpu2_gpucpu` on branch `worker/gpt/m6b-gpu-cpu-step2-divergence`.
Your FIRST command must be `cd /tmp/wrf_gpu2_gpucpu` — do everything else from there.

Write-only:
- `scripts/m6b_gpu_cpu_step2_probe.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/` — proofs + divergence_memo.md + worker-report.md

Read-only: validation_wrappers.py, operational_mode.py, dynamics/core/.

## Inputs

1. This sprint contract.
2. `.agent/sprints/2026-05-25-m6b-standalone-vs-comparator-bisect/worker-report.md` (the prior CPU 2/5/10 = 0.0 bitwise result).
3. `.agent/sprints/2026-05-25-m6b-honest-1h-canary-V3/` (the conflicting GPU finite-step-45 result).
4. The 20260521 IC at `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/`.

## Acceptance Criteria

### Stage 1 — Side-by-side step-2 probe on GPU AND CPU

For the 20260521 IC, run **four** 5-step probes:
1. validation_wrappers on **CPU** (JAX_PLATFORMS=cpu)
2. validation_wrappers on **GPU** (JAX_PLATFORMS=cuda)
3. operational_mode on **CPU**
4. operational_mode on **GPU**

For each: capture per-step max/min of theta, mu, u, v, w + finiteness check + max_abs_delta vs WRF reference (where available).

Write `proof_4path_step2_matrix.json` with the 4×5 matrix of (path × step) → {max_theta, min_theta, max_mu, finite_bool, ...}.

### Stage 2 — Discriminate the three hypotheses

Examine the matrix:

- If validation_wrappers GPU explodes at step 2 but CPU stays finite → **(B)**: GPU-shared bug in dynamics/core/. Most painful diagnosis. Open named-fix sprint on dynamics/core kernels.
- If both GPU paths explode at step 2 but CPU paths stay finite (matching CPU 2/5/10 bisect) → also **(B)**: GPU-shared. Most likely culprits: fused acoustic_loop/tridiag, numerical mode of jax.lax.scan on GPU vs CPU, jit precision settings.
- If validation_wrappers GPU stays finite but operational GPU explodes → **(C)**: operational-only GPU bug. Reframe was incomplete.
- If both CPU and GPU produce finite step-2 values on both paths → **(A)**: the prior comparator was reporting NaN sentinels. Re-investigate the comparator's max_abs_delta arithmetic. If a fused operator returns NaN for a single field even when most cells finite, `nanmax` vs `max` would differ.

### Stage 3 — Cross-check with V3 bounds path

The V3 bounds path ran 45 finite steps on GPU. What was different?
- Was V3 using validation_wrappers or operational_mode? (likely operational; confirm via grep on m6b_canary_1h_honest_v3.py)
- Did the comparator use a tighter or looser tolerance / different field ordering?
- Was the GPU warm vs cold? Was JIT cached vs first-compile?

Write `proof_v3_vs_comparator_diff.md` — a short diff summary.

### Stage 4 — Divergence memo

Write `divergence_memo.md`:
- **Verdict**: one of `(A)-SENTINEL-COINCIDENCE` | `(B)-GPU-SHARED-BUG` | `(C)-OPERATIONAL-ONLY-GPU` | `INSUFFICIENT-EVIDENCE`
- **Evidence**: cite the 4×5 matrix + V3-vs-comparator diff.
- **Recommended next sprint**: exact name + scope.
- **Severity for M6 close**: blocker / minor / non-blocker.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_gpucpu
export OMP_NUM_THREADS=4
export PYTHONPATH="src"
# CPU runs
JAX_PLATFORMS=cpu taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path validation --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/cpu_validation.json
JAX_PLATFORMS=cpu taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path operational --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/cpu_operational.json
# GPU runs
taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path validation --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/gpu_validation.json
taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path operational --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/gpu_operational.json
git add -A && git commit -m "[gpu-cpu step-2] $(date -u +%FT%TZ)"
```

## Handoff

Worker-report.md with `Summary:`, files changed, commands + truncated outputs, proof paths, the verdict, severity classification.
