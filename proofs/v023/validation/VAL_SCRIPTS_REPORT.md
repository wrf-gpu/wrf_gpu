# V023 Validation Scripts Report

Generated: 2026-07-03
Agent: codex
Branch: `worker/gpt/v023-validation-scripts`

## Objective

Author turnkey v0.23 close-validation scripts for gates A1 through C2, plus a
fresh paired-baseline wrfout field comparator. Per instruction, this pass was
CPU-only authoring and validation: scripts were written, parsed, compiled,
self-tested, and dry-run with `taskset -c 4-31`; no GPU forecast was launched.

## Files Changed

- `proofs/v023/validation/paired_baseline_field_compare.py`
- `proofs/v023/validation/v023_validation_common.sh`
- `proofs/v023/validation/a1_default_byte_identity.sh`
- `proofs/v023/validation/a2_canary_gate_fieldtol.py`
- `proofs/v023/validation/a3_m1_fp32_release_gates.sh`
- `proofs/v023/validation/a4_feature_branch_inert.sh`
- `proofs/v023/validation/b1_gate3_p1_p7b.sh`
- `proofs/v023/validation/b2_p0_m9_perf.sh`
- `proofs/v023/validation/b3_p3_fusion_perf.sh`
- `proofs/v023/validation/b4_f1_reconfirm.sh`
- `proofs/v023/validation/b5_k1_boulac.sh`
- `proofs/v023/validation/c1_full_suite.sh`
- `proofs/v023/validation/c2_skill_gate.sh`
- `proofs/v023/validation/results/`
- `proofs/v023/validation/VAL_SCRIPTS_REPORT.md`
- `proofs/v023/validation/VAL_SCRIPTS_DONE`

## Harness Reuse

- Reuses `scripts/compare_wrfout_grid.py` for field-level wrfout comparison.
- Reuses `scripts/with_gpu_lock.sh` for all non-dry-run GPU-touching gates.
- Reuses `scripts/run_regression_suite.py` for C1.
- Reuses `scripts/v020_skill_eval.py` for C2.
- Reuses `scripts/m7_nsys_stats_extract.py` when Nsight capture is enabled.
- Reuses the F1/G3 batch driver at `proofs/v023/gpu_gates/g3_batch_sweep_driver.py`
  when that driver exists in the branch under test.

## No Stored Digest

`paired_baseline_field_compare.py` compares fresh baseline and candidate wrfout
directories. It does not read or trust any stored digest. Tolerances are
declared before comparison via one of:

- `exact`: max absolute difference `0.0`.
- `autotune-floor`: default max absolute difference `1e-8`.
- `tolerance`: explicit tolerance JSON.

It fails on tolerance failures, baseline-only or candidate-only fields,
incompatible common fields, missing leads, incompatible leads, or unequal time
metadata.

## Gate Entrypoints

- A1: `a1_default_byte_identity.sh`
  - Fresh `BASE_REF` vs `HEAD` default paired run.
  - Uses fresh paired `autotune-floor` comparison by default.
- A2: `a2_canary_gate_fieldtol.py`
  - Canary field-tolerance wrapper around the paired comparator.
  - Includes CPU-only positive and negative controls.
- A3: `a3_m1_fp32_release_gates.sh`
  - Fresh fp64 default pair compare.
  - Requires explicit M1 closure evidence via `M1_KILLGATE_CMD` or
    `M1_KILLGATE_CLOSED_JSON` during real execution.
- A4: `a4_feature_branch_inert.sh`
  - Compares each `FEATURE_REFS` entry against `BASE_REF` with default settings.
  - Requires `A4_FAILCLOSED_CMD_<LABEL>` for real fail-closed evidence.
- B1: `b1_gate3_p1_p7b.sh`
  - P1/P7b off/on paired run, Nsight enabled by default.
- B2: `b2_p0_m9_perf.sh`
  - P0/M9 off/on paired run.
- B3: `b3_p3_fusion_perf.sh`
  - P3 fusion off/on paired run, defaulting to two domains.
- B4: `b4_f1_reconfirm.sh`
  - Optional F1 batch sweep reconfirmation when the F1 driver is present.
- B5: `b5_k1_boulac.sh`
  - Conditional K1 Boulac A/B gate. Skips unless K1 is required or an on-env is
    supplied.
- C1: `c1_full_suite.sh`
  - Full regression suite wrapper plus CPU snapshot comparison.
- C2: `c2_skill_gate.sh`
  - Skill gate wrapper using existing v020 skill evaluation.

## Commands Run

All commands were run CPU-only with GPU visibility disabled.

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  bash -n proofs/v023/validation/*.sh
```

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  python3 -m py_compile proofs/v023/validation/*.py
```

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  python3 proofs/v023/validation/paired_baseline_field_compare.py \
  --self-test \
  --out-json proofs/v023/validation/results/paired_baseline_field_compare_selftest.json
```

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  python3 proofs/v023/validation/a2_canary_gate_fieldtol.py \
  --self-test \
  --out-json proofs/v023/validation/results/a2_canary_gate_fieldtol_selftest.json
```

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  bash -lc 'source proofs/v023/validation/v023_validation_common.sh; \
  bash -c '\''set -euo pipefail; source "$V023_COMMON"; \
  test -n "$V023_ROOT"; test -x "$V023_GPU_LOCK"; echo child-source-ok'\'' '
```

```bash
taskset -c 4-31 env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES= \
  V023_RESULTS_DIR=proofs/v023/validation/results/dryrun \
  V023_WORK_ROOT=proofs/v023/validation/results/dryrun_work \
  bash proofs/v023/validation/a1_default_byte_identity.sh --dry-run
```

The same dry-run pattern was executed for A3, A4, B1, B2, B3, B4, B5, C1, and
C2. A4 used `FEATURE_REFS=TEST=HEAD`; B5 used
`K1_REQUIRED=1 K1_ON_ENV=GPUWRF_MYNN_BOULAC_ONZ=1`.

## Results

- Shell parse check: PASS.
- Python compile check: PASS.
- Comparator self-test: PASS.
- A2 canary self-test: PASS.
- Child-shell source smoke test: PASS.
- A1-C2 dry-run matrix: PASS, all scripts exited `0`.

## Proof Objects Produced

- `proofs/v023/validation/results/paired_baseline_field_compare_selftest.json`
- `proofs/v023/validation/results/a2_canary_gate_fieldtol_selftest.json`
- `proofs/v023/validation/results/a2_canary_gate_fieldtol_selftest_field_compare_selftest.json`
- `proofs/v023/validation/results/dryrun/`
- `proofs/v023/validation/results/dryrun_work/`

## Unresolved Risks

- The real GPU acceptance gates have not been executed, by instruction.
- A3 real execution needs M1 closure evidence supplied.
- A4 real execution needs actual feature refs and fail-closed commands.
- B4 real execution depends on the F1 batch driver existing in the branch under
  test.
- Performance pass/fail still depends on real Nsight and runtime artifacts from
  the GPU run.

## Next Decision Needed

Run the real validation gates on the GPU when the GPU lock is available, using
the authored scripts and supplying the A3/A4 evidence inputs.
