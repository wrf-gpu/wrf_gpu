# V0.14 GPT Acoustic-Substep Candidate Verifier

Date: 2026-06-11
Worker: GPT-5.5 xhigh
Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

## Verdict

`NEED_FABLE_AFTER_RESET`

This candidate should not be accepted for the manager gate. It is stable enough to run the 2h Switzerland forecast, and several individual changes look directionally WRF-faithful, but the proof artifacts do not show the required h36->h37/h38 collapse. The operational residual is slightly worse than both prior baselines, and the WRF-native stage-boundary comparisons still show large p/ph increment mismatches.

## Findings

1. **Blocking: the h36->h37 Switzerland residual does not collapse.**

   From `proofs/v014/switzerland_acoustic_substep_blocker.json` after running the proof script's analysis path:

   | run | residual Pa/cell/h | net influx Pa/cell/h | dM Pa/cell/h |
   | --- | ---: | ---: | ---: |
   | CPU h36->h37 | `+5.178443877551032` | `-74.51517857142858` | `-69.33673469387755` |
   | old `ec4d6769` | `-32.686352040816345` | `-103.12997448979591` | `-135.81632653061226` |
   | HPG native-face `3d0b439c` | `-27.697448979591826` | `-102.84336734693878` | `-130.5408163265306` |
   | acoustic fix, no km | `-35.8594387755102` | `-103.31403061224489` | `-139.1734693877551` |
   | acoustic fix | `-35.94119897959182` | `-103.33431122448981` | `-139.27551020408163` |

   The script's collapse metrics are negative: `collapse_fraction_vs_old = -0.007140946777213886`, `collapse_fraction_vs_hypso = -0.017330577730950925`. That means the candidate very slightly worsens the budget error by this metric.

2. **Blocking: h36->h38 remains far from CPU.**

   The same proof JSON has `h36_h38.cpu.residual_pa_per_cell_h = -64.34387755102041` and `h36_h38.acoustic_fix.residual_pa_per_cell_h = -166.74681122448982`. The h38 net-influx difference is still `-20.199107142857144` Pa/cell/h. This is not a material residual collapse.

3. **Blocking: the WRF-native stage-boundary proof is real, but it fails the physics question.**

   The proof script compares JAX stage captures against WRF-native HPG dump calls `21601..21606`, not a JAX-vs-JAX self-compare. It also validates its eager replica against the jitted production `_physics_boundary_step`; for the final `sub4_dt18_bcfix` tag, max diffs are small (`u` about `5.91e-08`, `v` about `5.97e-08`, `p_perturbation` about `1.03e-08`, `ph_perturbation` about `4.62e-11`).

   However the native comparisons still have large increment errors. For `sub4_dt18_bcfix`:

   | comparison | p incr RMSE | ph incr RMSE | alt incr RMSE | note |
   | --- | ---: | ---: | ---: | --- |
   | `step1_stage1_vs_21602` | `6.66855` | `1.51467` | `0.00101993` | p/ph about `9.3x`/`6.8x` WRF increment scale |
   | `step1_final_vs_21604` | `4.76427` | `2.69105` | `0.00343084` | p/ph about `8.9x`/`6.1x`; alt about `57.5x` |
   | `step2_stage2_vs_21606` | `9.1671` | `3.21061` | `0.00149033` | p/ph about `20.6x`/`14.6x`; mu about `7.05x` |

   This supports "candidate path is being measured" but not "candidate matches WRF."

4. **Correctness risks in the diff are not small enough for local completion.**

   The following edits are plausible but not proven sufficient:

   - WRF constants in `src/gpuwrf/dynamics/acoustic_wrf.py`, `src/gpuwrf/coupling/boundary_apply.py`, and `src/gpuwrf/integration/d02_replay.py`.
   - Fresh stage omega threading in `src/gpuwrf/runtime/operational_mode.py`.
   - Work-delta `u/v` into `advance_w` in `src/gpuwrf/dynamics/core/acoustic.py`.
   - Stage-level `w_damping` in `src/gpuwrf/runtime/operational_mode.py`.
   - Real-case `diff_opt/km_opt` threading in `src/gpuwrf/integration/daily_pipeline.py`.

   But the remaining native p/ph mismatch is large enough that guessing another source edit here would be speculative. I did not modify source.

5. **Performance and architecture risk: not a host/device transfer regression, but not performance-cleared.**

   I did not see new explicit host/device transfers inside the timestep loop in the candidate diff. The added stage-level `w_damping` is traced JAX work, and `diff_opt=1/km_opt=4` intentionally enables more large-step diffusion work in the real forecast path. There are no profiler artifacts for the candidate, so performance impact is unresolved. The proof forecast log only establishes a 2h PASS, not throughput parity or no-regression.

6. **Broader constants surface remains inconsistent.**

   A repo search still finds `CP_D = 1004.0` and `GRAVITY_M_S2 = 9.80665` in non-acoustic modules and validation/oracle helpers, including `src/gpuwrf/physics/surface_constants.py`, `src/gpuwrf/dynamics/vertical_implicit_solver.py`, and several coupling/validation files. Some may be intentionally physics-scheme local, but the current candidate is not a comprehensive WRF-constant audit.

## Commands Run

- `sed -n '1,220p' PROJECT_CONSTITUTION.md`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,320p' .agent/sprints/2026-06-11-v014-gpt-acoustic-substep-verifier/sprint-contract.md`
- `sed -n '1,260p' .agent/skills/conducting-blind-review/SKILL.md`
- `sed -n '1,260p' .agent/skills/validating-physics/SKILL.md`
- `git status --short`
- `git diff --stat`
- `git diff -- src/gpuwrf/coupling/boundary_apply.py src/gpuwrf/dynamics/acoustic_wrf.py src/gpuwrf/dynamics/core/acoustic.py src/gpuwrf/integration/d02_replay.py src/gpuwrf/integration/daily_pipeline.py src/gpuwrf/runtime/operational_mode.py`
- `tail -n 160 /tmp/acoustic_gate_forecast.log`
- `tail -n 220 /tmp/acoustic_chain2.log`
- `ps -eo pid,ppid,stat,etime,cmd | rg 'switzerland_acoustic_substep_blocker|run_gpu_lowprio|wrf_gpu_validation_gpu.lock|python proofs/v014'`
- `ls -l /tmp/wrf_gpu_validation_gpu.lock /tmp/acoustic_gate_forecast.log /tmp/acoustic_chain2.log 2>/dev/null`
- `python -m py_compile src/gpuwrf/coupling/boundary_apply.py src/gpuwrf/dynamics/acoustic_wrf.py src/gpuwrf/dynamics/core/acoustic.py src/gpuwrf/integration/d02_replay.py src/gpuwrf/integration/daily_pipeline.py src/gpuwrf/runtime/operational_mode.py proofs/v014/switzerland_acoustic_substep_blocker.py`
- `python -m json.tool proofs/v014/switzerland_acoustic_substep_blocker.json >/tmp/v014_acoustic_json_validated.txt && wc -c proofs/v014/switzerland_acoustic_substep_blocker.json /tmp/v014_acoustic_json_validated.txt`
- `python proofs/v014/switzerland_acoustic_substep_blocker.py`
- `pytest -q tests/test_v014_hypsometric_opt2.py tests/dynamics/test_diffopt1_smagorinsky.py tests/dynamics/test_diffopt1_smagorinsky_integration.py tests/test_operational_namelist_cache_key.py tests/test_m6b0r_calc_coef_w_fix.py`
- Python JSON summarizers over `proofs/v014/switzerland_acoustic_substep_blocker.json`

No GPU process was started by me. The existing low-priority GPU chain appeared finished; no matching blocker/low-priority process was visible. The lock file existed but was empty.

## Proof Objects Used

- `/tmp/acoustic_gate_forecast.log`: 2h forecast PASS at `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_acoustic_substep_fix`.
- `/tmp/acoustic_chain2.log`: `CHAIN: nokm stashed`, `CHAIN: km gate exit=0`.
- `proofs/v014/switzerland_acoustic_substep_blocker.py`: WRF-native stage comparison and hourly-gate analyzer.
- `proofs/v014/switzerland_acoustic_substep_blocker.json`: stage comparisons plus hourly-gate analysis written by `python proofs/v014/switzerland_acoustic_substep_blocker.py`.
- `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_acoustic_substep_fix/wrfout_d01_2023-01-16_13:00:00`
- `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_acoustic_substep_fix/wrfout_d01_2023-01-16_14:00:00`
- `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_acoustic_substep_fix_nokm/wrfout_d01_2023-01-16_13:00:00`
- `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_acoustic_substep_fix_nokm/wrfout_d01_2023-01-16_14:00:00`

## Validation Results

- `py_compile`: passed for all changed source files and the proof script.
- JSON validation: passed; formatted validation output written to `/tmp/v014_acoustic_json_validated.txt`.
- Focused pytest subset: `20 passed, 7 skipped, 1 warning in 5.21s`. The warning was JAX failing to write a persistent cache entry under a read-only cache path.
- Proof analysis: ran successfully and added `hourly_gate` to `proofs/v014/switzerland_acoustic_substep_blocker.json`.

## Files Changed By GPT

- `.agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`
- `proofs/v014/switzerland_acoustic_substep_blocker.json` was updated by running its analyzer path to add the `hourly_gate` section.

I did not edit source files.

## Unresolved Risks

- The stage-boundary mismatch is still large enough that the next root cause is not identified by this candidate.
- The absent `gpu_output_acoustic_substep_fix_dt18` output means the proposed dt=18/substeps=4 forecast variant was not available for hourly-gate analysis. The stage compare already uses dt=18/substeps=4 and still fails p/ph closely enough that I did not start another GPU run.
- Candidate performance was not profiled.
- The WRF constants change is partial across the repo and needs a deliberate constants audit before broad acceptance.

## Next Decision Needed

Do not merge this candidate as the manager gate. Send Fable the narrowed handoff after reset:

1. Keep the proof harness; it is useful because it compares against WRF-native calls and validates replica-vs-jit.
2. Start from `sub4_dt18_bcfix` stage comparisons and localize the remaining p/ph increment mismatch, especially `step1_stage1_vs_21602` and `step2_stage2_vs_21606`.
3. Treat the hourly residual as the acceptance gate: `acoustic_fix` must improve materially over `old_ec4d6769` and `hypso_3d0b439c`, not just run stably.
4. Run/provide profiler or transfer audit before making any speed claim after `w_damping` and `diff_opt/km_opt` are enabled.
