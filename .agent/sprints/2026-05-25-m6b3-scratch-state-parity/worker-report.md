# Worker Report - M6B3 Scratch-State Parity

## objective

Extend the B-direct savepoint ladder to WRF small-step scratch families (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, and `_save` fields), prove sanitizer-off JAX-vs-WRF-shaped parity across column / patch16 / golden tiers for 10 acoustic substeps, and classify all scratch carry as operational-undecided per PROJECT_PLAN 14.5.1 Critic Amendment #1.

## stage status

- Stage 1 wrapper/build: **done**. Added five scratch hook pairs, ten hooks total: `sp_t_2ave_update_pre/post`, `sp_ww_update_pre/post`, `sp_muave_update_pre/post`, `sp_ph_tend_accumulate_pre/post`, and `sp_substep_save_state_pre/post`. Instrumented shim rebuilt. Protected operational WRF stayed at SHA-256 `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37`.
- Stage 2 synthetic dry-run: **done**. Clean self-compare passed and 20x-tolerance perturbations were caught for every compared scratch field.
- Stage 3 extraction: **done** for column, 16x16 patch, and pinned golden tiers, 10 acoustic substeps each. Each tier emitted 100 scratch savepoints: five pre/post hook pairs per substep.
- Stage 4 parity: **PASS**. `proof_scratch_state_parity.json` reports all three tiers passed for `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `u_save`, `v_save`, `w_save`, `t_save`, `ph_save`, `mu_save`, and `ww_save`.
- Stage 5 kill gate: **PASS**. Substep-1 diverging field count is `0`, threshold `15`; decision `PROCEED_TO_M6B4`.
- Stage 6 operational compatibility: **done**. All scratch families are validation-required and operational-**Undecided**. No field is approved for operational mode.
- Stage 7 no regression: **PASS**. Required pytest set: `105 passed in 294.53s`.

## parity summary

| tier | steps | savepoints | max scratch delta | result |
|---|---:|---:|---:|---|
| column | 10 | 100 | `0.0` | PASS |
| patch16 | 10 | 100 | `0.0` | PASS |
| golden | 10 | 100 | `0.0` | PASS |

Outcome: `FOURTH-OPERATOR-FAMILY-PARITY-ACHIEVED`.

## operational-compatibility

| Field family | Validation classification | Operational classification | Evidence / note |
|---|---|---|---|
| `t_2ave` | Required | **Undecided** | Validation parity uses the contract rule `(t_old + t_new) / 2`; no Tier-4 ablation exists. |
| `ww` | Required | **Undecided** | Validation scratch only; no operational API change. |
| `muave` | Required | **Undecided** | Validation scratch only; defer to M6-perf-design ablation. |
| `muts` | Required | **Undecided** | Validation scratch only; defer to M6-perf-design ablation. |
| `ph_tend` | Required | **Undecided** | Validation accumulator only; no Tier-4 evidence for operational carry. |
| `_save` family | Required | **Undecided** | Validation snapshots only; no operational state insertion. |

Undecided fields may not enter operational state APIs without a follow-up Tier-4-backed decision.

## files changed

- `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90`
- `external/wrf_savepoint_patch/solve_em.F.patch`
- `scripts/m6b3_scratch_state_compare.py`
- `src/gpuwrf/dynamics/small_step_scratch.py`
- `src/gpuwrf/validation/savepoint_schema.py`
- `src/gpuwrf/validation/tolerance_ladder.json`
- `tests/test_m6b3_scratch_state_parity.py`
- `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/.gitignore`
- `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_*.txt`
- `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_*.json`
- `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/savepoints/*/manifest.json`
- `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/worker-report.md`

## commands run

- `bash external/wrf_savepoint_patch/build.sh 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_build_rebuild.txt`
- `sha256sum external/wrf_savepoint_patch/build/main/wrf.exe.instrumented | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_instrumented_sha256.txt`
- `sha256sum /home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_operational_sha256_post.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b3_scratch_state_compare.py --synthetic-dryrun 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_synthetic_dryrun_m6b3.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b3_scratch_state_compare.py --tier column --steps 10 --output .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_scratch_state_parity_column.json 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_savepoint_scratch_column.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b3_scratch_state_compare.py --tier patch16 --steps 10 --output .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_scratch_state_parity_patch16.json 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_savepoint_scratch_patch16.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b3_scratch_state_compare.py --tier golden --steps 10 --output .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_scratch_state_parity_golden.json 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_savepoint_scratch_golden.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b3_scratch_state_compare.py --tier all --steps 10 --output .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_scratch_state_parity.json 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_scratch_state_parity.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b3-scratch-state-parity/proof_no_regression.txt`

## proof objects produced

- `proof_build_rebuild.txt`
- `proof_instrumented_sha256.txt`
- `proof_operational_sha256_post.txt`
- `proof_synthetic_dryrun_m6b3.txt`
- `proof_synthetic_dryrun_m6b3.json`
- `proof_savepoint_scratch_column.txt`
- `proof_savepoint_scratch_patch16.txt`
- `proof_savepoint_scratch_golden.txt`
- `proof_scratch_state_parity_column.json`
- `proof_scratch_state_parity_patch16.json`
- `proof_scratch_state_parity_golden.json`
- `proof_scratch_state_parity.txt`
- `proof_scratch_state_parity.json`
- `proof_kill_gate_status.txt`
- `proof_no_regression.txt`
- `savepoints/column/manifest.json`
- `savepoints/patch16/manifest.json`
- `savepoints/golden/manifest.json`

## unresolved risks

- Direct relinked WRF call-site HDF5 emission remains incomplete. As in M6B1/M6B2, the local binary is the HDF5-linked CPU shim and extraction uses WRF-source-shaped transcription over real Canary d02 wrfout slices.
- `ph_tend` accumulation is represented as a validation-boundary accumulator in the scratch comparator; M6B4 still needs acoustic recurrence parity around the full `advance_w` RHS and update sequence.
- Scratch fields are high-risk carry-creep candidates. This sprint intentionally does not approve them for operational mode.

## next decision needed

Proceed to M6B4 acoustic recurrence parity. M6-perf-design / ADR-026 must later decide whether any M6B3 scratch family is retained, fused away, or dropped in operational mode using Tier-4 evidence.
