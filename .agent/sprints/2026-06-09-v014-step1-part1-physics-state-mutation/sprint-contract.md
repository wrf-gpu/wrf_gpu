# Sprint Contract: V0.14 Step-1 Part1 Physics-State Mutation

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Split the newly localized Step-1 mismatch inside WRF
`first_rk_step_part1`.

Trigger evidence:

- `proofs/v014/step1_rk1_source_boundary.json`
- verdict
  `STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`
- first material source-boundary mismatch:
  `after_first_rk_step_part1`, field `T_STATE`
- WRF vs JAX operational carry max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`
- WRF vs `_physics_step_forcing.state` max_abs `5.490142455570492`, RMSE
  `1.9174736017582765`
- RK1 `small_step_prep` continuity remains exact for `T_WORK` and `P_WORK`

The sprint must determine whether the source is:

- WRF part1 input already differs from the JAX state/carry;
- WRF `init_zero_tendency`, `phy_prep`, radiation, surface, PBL, cumulus,
  shallow-cumulus, SCM forcing, or FDDA changes `grid%t_2` / `T_STATE`;
- a WRF physics tendency leaf (`t_tendf`, `h_diabatic`, `rthraten`,
  `rthblten`, `rthcuten`, `rthften`, or related leaf) missing from JAX;
- a JAX `_physics_step_forcing` state/carry handoff mismatch;
- or a narrowly provable production bug.

## Method Rule

Use the fastest rigorous wall-clock method: extend the existing Step-1
source-boundary truth/comparator with internal `first_rk_step_part1` surfaces,
not a long validation run.

At planning time, explicitly re-check whether a savepoint/comparator is still
the right tool. For this sprint it is: one scratch WRF compile plus one step-1
truth run can classify many internal call boundaries in one pass and avoid
guessing across the 1800-line Fortran routine.

Accepted WRF surfaces include the smallest useful subset of:

- `part1_entry_before_init_zero_tendency`;
- `after_init_zero_tendency`;
- `after_phy_prep`;
- `after_pre_radiation_driver`;
- `after_radiation_driver`;
- `after_surface_driver`;
- `after_pbl_driver`;
- `after_cumulus_driver`;
- `after_shallowcu_driver`;
- `after_force_scm`;
- `after_fddagd_driver`;
- `part1_exit`.

The worker may emit fewer surfaces only if an earlier surface decisively
classifies the first material mismatch.

Accepted JAX comparisons:

- live-nest Step-1 carry/state before `_physics_step_forcing`;
- `_physics_step_forcing(...).state`, `.carry`, and `.dry_tendencies`;
- scheme-level adapter outputs if the first WRF-mutating surface maps to a
  specific adapter slot;
- the prior `after_first_rk_step_part1` comparator for continuity.

Forbidden comparisons:

- no WRF final truth vs JAX initial state;
- no JAX-vs-JAX-only conclusion;
- no one-cell/station proxy;
- no acoustic, TOST, Switzerland, FP32, or memory work.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No memory source work.
- No GPU.
- No Hermes or Telegram.
- No broad dycore rewrite or performance-regressing source change.

## Inputs

- `proofs/v014/step1_rk1_source_boundary.py`
- `proofs/v014/step1_rk1_source_boundary.json`
- `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`
- `proofs/v014/step1_t_p_operator_localization.py`
- `proofs/v014/step1_t_p_operator_localization.json`
- `proofs/v014/step1_live_nest_init_rerun.py`
- `proofs/v014/step1_live_nest_init_rerun.json`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/physics/**`
- `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/**`
- `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/**`
- scratch WRF file:
  `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/WRF/dyn_em/module_first_rk_step_part1.F`

Allowed scratch root:

- `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/**`

## Write Scope

Required repo files:

- `proofs/v014/step1_part1_physics_state_mutation.py`
- `proofs/v014/step1_part1_physics_state_mutation.json`
- `proofs/v014/step1_part1_physics_state_mutation.md`
- `.agent/reviews/2026-06-09-v014-step1-part1-physics-state-mutation.md`

Optional repo files:

- `proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`
- targeted source edits only if an exact, narrow, performance-compatible bug is
  proven:
  - `src/gpuwrf/runtime/operational_mode.py`
  - specific files under `src/gpuwrf/physics/**`
  - specific files under `src/gpuwrf/coupling/**`

Do not touch unrelated source, TOST outputs, Switzerland outputs, FP32 work,
memory source work, or old untracked artifacts.

## Required Work

1. Verify branch/head and that `c18795af` is an ancestor.
2. Reuse the existing Step-1 parser/comparator where practical; do not build a
   new broad framework unless it shortens the proof materially.
3. Emit or consume WRF full d02 internal part1 truth surfaces with enough fields
   to compare:
   - `T_STATE`, `P_STATE`, `PB`, `MU_STATE`, `MUB`, `MUT`;
   - `T_TENDF`, `H_DIABATIC`, `MU_TENDF`, `T_SAVE`, `T_OLD`;
   - `RTHRATEN`, `RTHRATENLW`, `RTHRATENSW`, `RTHBLTEN`, `RTHCUTEN`,
     `RTHFTEN` where present and useful;
   - `TH_PHY`, `T_PHY`, `P_PHY`, `PI_PHY` if needed to map physics-driver
     output to JAX state/tendency leaves.
4. Compare each WRF surface to the matching JAX surface, not to an unrelated
   time boundary.
5. Classify the first material mismatch. The result must say whether the issue
   is input-already-diverged, a specific WRF physics slot mutation, a missing
   JAX dry tendency leaf, a JAX state/carry handoff mismatch, or still blocked by
   one exact missing truth leaf.
6. If a source fix is made, rerun:
   - this sprint proof;
   - `proofs/v014/step1_rk1_source_boundary.py`;
   - `proofs/v014/step1_t_p_operator_localization.py`;
   - `proofs/v014/step1_live_nest_init_rerun.py`;
   and report before/after top residuals.

## Verdicts

Emit exactly one final verdict:

- `STEP1_PART1_MUTATION_LOCALIZED_<surface>_<field_or_leaf>`
- `STEP1_PART1_INPUT_ALREADY_DIVERGED_<field>`
- `STEP1_PART1_FIXED_<field_or_leaf>`
- `STEP1_PART1_BLOCKED_<specific_missing_truth_or_contract>`
- `STEP1_PART1_NO_REMAINING_DIVERGENCE`

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/step1_part1_physics_state_mutation.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_part1_physics_state_mutation.py
python -m json.tool proofs/v014/step1_part1_physics_state_mutation.json \
  >/tmp/step1_part1_physics_state_mutation.validated.json
git diff -- src/gpuwrf
```

If production source changes:

```bash
python -m py_compile src/gpuwrf/runtime/operational_mode.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_rk1_source_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_t_p_operator_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_init_rerun.py
```

## Acceptance Criteria

- JSON validates and records CPU-only execution.
- The proof names the exact WRF/JAX internal boundary and field/leaf for the
  first material mismatch or exact blocker.
- Any source fix is narrow and performance-compatible: no host/device transfer
  inside timestep loops, no CPU-WRF wrapper, no broad de-optimization.
- Production `src/gpuwrf/**` remains unchanged unless a concrete bug is proven.
- Review report includes objective, files changed, commands run, proof objects,
  unresolved risks, and next decision.

## Completion Signal

Notify the manager pane with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_PART1_PHYSICS_STATE_MUTATION DONE - see proofs/v014/step1_part1_physics_state_mutation.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
