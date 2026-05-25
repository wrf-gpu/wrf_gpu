# Worker Report - M6B4 Acoustic Recurrence Parity

## objective

Compose the M6B0-R/B1/B2/B3 validated operators (`calc_coef_w`, `advance_mu_t`, Thomas solve, scratch state) into a validation-only acoustic substep and full acoustic loop for one RK stage, then prove sanitizer-off parity across column, patch16, and golden tiers.

## stage status

- Stage 1 wrapper/build: **PASS**. Added `sp_acoustic_substep_complete` and `sp_acoustic_loop_complete` stub ABIs, wired them in `solve_em.F.patch`, rebuilt the shim, and kept protected operational WRF at SHA-256 `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.
- Stage 2 synthetic dry-run: **PASS**. Clean full-boundary self-compare passed and 20x-tolerance perturbations were caught for every acoustic recurrence field.
- Stage 3 real extraction: **PASS**. Emitted 10 `acoustic_substep_complete` snapshots plus 1 `acoustic_loop_complete` snapshot per tier using the established Python/HDF5 validation lane over real Canary d02 wrfout.
- Stage 4 composition parity: **PASS**. Outcome `FIFTH-OPERATOR-COMPOSITION-PARITY-ACHIEVED` for column, patch16, and golden.
- Stage 5 kill gate: **PASS**. Substep-1 diverging field count across all tiers was `0`; threshold is `15`; decision `PROCEED_TO_M6B5`.
- Stage 6 operational compatibility: **done**. Critic Amendment #1 classification table is below.
- Stage 7 no regression: **PASS**. Required pytest set: `112 passed in 294.27s`.

## parity summary

| tier | substeps | savepoints | max observed delta | result |
|---|---:|---:|---:|---|
| column | 10 | 11 | `8.673617379884035e-19` (`w`) | PASS |
| patch16 | 10 | 11 | `5.551115123125783e-17` (`w`) | PASS |
| golden | 10 | 11 | `1.3877787807814457e-17` (`w`) | PASS |

The per-substep ladder uses a conservative linear growth rationale: 10x operator absolute tolerance plus 2x/substep roundoff headroom. No tolerance was tuned after seeing the comparison result.

## operational compatibility

| Item | Classification | Evidence |
|---|---|---|
| `sp_acoustic_substep_complete/loop_complete` hooks | **Validation-only** | Savepoint emission boundary only; wrapper bodies remain empty stubs and operational WRF SHA stayed unchanged. |
| `acoustic_loop.py` callable | **Validation-only** | New module is explicitly validation-only and is not wired into operational runtime; no Tier-4 operational evidence claimed. |
| New ladder entries (per-substep tolerances for `mut/u/v/w/ph/p`) | **Validation-only** | Comparator thresholds only; no runtime state API or precision policy change. |
| Schema v5 extension | **Validation-only** | Adds HDF5 savepoint boundary/operator names only; no operational state API change. |
| Full-state recurrence snapshot fields (`mu/mut/mudf/muts/muave/ww/theta/ph_tend/u/v/w/ph/p/t_2ave`) | **Validation-only** | Required for parity diagnosis; no field is approved for operational carry by this sprint. |
| Serial Thomas recurrence inside the validation loop | **Undecided** | Matches WRF-shaped parity composition, but operational solver choice remains deferred to M6-perf-design/ADR-026. |

Undecided items may not enter operational APIs without a follow-up Tier-4-backed decision.

## files changed

- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `external/wrf_savepoint_patch/HOOK_INVENTORY.md`
- `scripts/m6b4_acoustic_recurrence_compare.py`
- `src/gpuwrf/dynamics/acoustic_loop.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/tolerance_ladder.json`
- `tests/test_m6b4_acoustic_recurrence_parity.py`
- `.agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/.gitignore`
- `.agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_*.txt`
- `.agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_*.json`

## commands run

- `bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_build_rebuild.txt`
- `patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/solve_em.F.patch`
- `patch -p1 --dry-run -d /tmp/wrf_test_canonical < external/wrf_savepoint_patch/module_small_step_em.F.patch`
- `sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe | tee .agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity/proof_operational_sha256_post.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b4_acoustic_recurrence_compare.py --synthetic-dryrun`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b4_acoustic_recurrence_compare.py --tier column --substeps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b4_acoustic_recurrence_compare.py --tier patch16 --substeps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b4_acoustic_recurrence_compare.py --tier golden --substeps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b4_acoustic_recurrence_compare.py --tier all --substeps 10`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py -v`

## proof objects produced

- `proof_build_rebuild.txt`
- `proof_patch_dryrun.txt`
- `proof_operational_sha256_post.txt`
- `proof_synthetic_dryrun_m6b4.txt`
- `proof_synthetic_dryrun_m6b4.json`
- `proof_savepoint_acoustic_column.txt`
- `proof_savepoint_acoustic_patch16.txt`
- `proof_savepoint_acoustic_golden.txt`
- `proof_acoustic_recurrence_parity_column.json`
- `proof_acoustic_recurrence_parity_patch16.json`
- `proof_acoustic_recurrence_parity_golden.json`
- `proof_acoustic_recurrence_parity.txt`
- `proof_acoustic_recurrence_parity.json`
- `proof_kill_gate_status.txt`
- `proof_no_regression.txt`

## unresolved risks

- Direct relinked WRF in-timestep HDF5 emission remains incomplete. As in B1/B2/B3, the validation lane composes WRF-source-shaped Python/JAX formulas over real Canary d02 wrfout slices while wrapper bodies are stubbed.
- M6B4 proves composition of the four validated operator families, not the full `advance_uv`, `advance_w` RHS/geopotential update, `calc_p_rho`, boundary replay, or multi-RK-stage dycore coupling. Those remain M6B5+ scope.
- The validation loop carries WRF scratch/full-state fields for diagnosis only; no operational carry approval is implied.

## next decision needed

Proceed to M6B5 full dycore step parity: physics off, boundary off, sanitizer off, 10 steps.
