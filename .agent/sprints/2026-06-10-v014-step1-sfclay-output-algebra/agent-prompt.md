You are GPT-5.5 xhigh, debugger/fixer for wrf_gpu2 v0.14.

Repo: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`
Base commit: `bdf68332 v014 fix sfclay thermo column inputs`

Read first:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-10-v014-step1-sfclay-output-algebra/sprint-contract.md`
5. `proofs/v014/step1_thermo_column_inputs.md`
6. `.agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md`
7. `src/gpuwrf/physics/surface_layer.py`

Mission:

Close, or reduce to one strictly narrower WRF-anchored blocker, the remaining
Step-1 `module_sf_mynn` surface-layer output algebra mismatch after the full
`sfclay_mynn` input tuple was fixed.

Important facts:

- TSK/ZNT/MAVAIL are fixed at the exact `sfclay_mynn` input hook.
- WRF `phy_prep` thermodynamic inputs are fixed/bounded:
  `th_phy` max_abs `6.71089752017906e-05 K`,
  `t_phy` `0.013577942721781255 K`,
  `p_phy` `0.015625 Pa`,
  `dz8w` `0.00018988715282830526 m`,
  `psfc` `0.015625 Pa`.
- Surface outputs remain red after fixed inputs:
  `UST` max_abs `0.01231782267117762`,
  `HFX` max_abs `27.09163832864155`,
  `QFX` max_abs `2.744275103194571e-07`,
  `BR` max_abs `2.0`.
- Strict Step-1 remains red: max_abs `847.1445725702908`, RMSE
  `9.56593990212596`.
- Do not run TOST, Switzerland, broad FP32, broad memory, Hermes, or Fable.
- CPU-only unless the manager later grants a short GPU probe.

Work style:

- Add a narrow WRF internal hook in `module_sf_mynn.F` / `SFCLAY1D_mynn`.
- Compare exact WRF internals (`thx/thgb/br/zol/psim/psih/ust/hfx/qfx`, plus
  necessary locals) against JAX `surface_layer_with_diagnostics` on the fixed
  input tuple.
- Rule out unit/sign/orientation before changing production code.
- If local and performance-compatible, fix `surface_layer.py` or a narrow
  surface constants/helper file.
- If not local, return one exact narrower blocker and the fastest next command.
- Keep reports compact.

Deliver:

- `proofs/v014/step1_sfclay_output_algebra.py`
- `proofs/v014/step1_sfclay_output_algebra.json`
- `proofs/v014/step1_sfclay_output_algebra.md`
- `proofs/v014/step1_sfclay_output_algebra_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-sfclay-output-algebra.md`
- focused tests if production code changes.

Run the acceptance gates from the sprint contract. If a gate is blocked, record
the exact blocker and fastest next command.

When done, print exactly:

`GPT STEP1_SFCLAY_OUTPUT_ALGEBRA DONE - see proofs/v014/step1_sfclay_output_algebra.md`

Then notify manager pane:

```bash
tmux send-keys -t 0:2 'GPT STEP1_SFCLAY_OUTPUT_ALGEBRA DONE - see proofs/v014/step1_sfclay_output_algebra.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
