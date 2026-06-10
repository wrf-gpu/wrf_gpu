You are Fable high, focused validation-debug worker for wrf_gpu2 v0.14.

Read exactly:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
5. `.agent/sprints/2026-06-10-v014-short-field-h1-residual/sprint-contract.md`
6. `proofs/v014/short_field_falsifier_h1_grid_compare.md`
7. `proofs/v014/short_field_falsifier_h1_grid_compare.json`
8. `proofs/v014/live_nest_base_source_fix.md`
9. `proofs/v014/step1_transient_adjust_base_fix.md`
10. `proofs/v014/step1_live_nest_theta_qv_wiring.md`

Verify `git log -1 --oneline`; expected base is `41468af4`.

Goal: give the manager a release-gate decision for the current short h1 Canary
d02 Field-Parity residual. Endpoint is one of:

- `PROCEED`: prove this is not a real release blocker and give the exact long
  GPU gate command/root requirements.
- `FIXED`: implement a local safe fix and prove it.
- `BLOCKED`: exact bug class and fastest next command, with ownership and
  whether GPU is needed.

Known evidence:

- Short GPU run root:
  `/mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T122005Z`.
- GPU h1 output:
  `/mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T122005Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-01_19:00:00`.
- CPU truth h1:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-01_19:00:00`.
- Comparator reported `PSFC` RMSE `323.115 Pa`, `P` RMSE `129.754 Pa`,
  `MU` RMSE `121.961 Pa`, `PBLH` RMSE `78.950 m`, `HFX` RMSE `38.186 W/m2`,
  `LH` RMSE `53.896 W/m2`, `PB` p99 `0.105 Pa` but max `249.875 Pa`, `MUB`
  p99 `18.194 Pa` and max `250.664 Pa`.
- Short run command used legacy runner
  `proofs/v0120/powered_tost_n15/run_one_case_v0120.py` and
  `/tmp/v0120_merged_run_root`.
- The selected `/tmp/v0120_merged_run_root/...` case symlinks inputs to
  `/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`,
  while comparator used CPU truth from
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/...`. Prove or dismiss
  stale/provenance mismatch first.

No GPU unless manager approves. Do not touch Switzerland CPU baseline. Do not
edit unrelated dirty files. If you need a GPU rerun, produce the exact command
and stop.

Required output:

- `.agent/reviews/2026-06-10-v014-short-field-h1-residual-fable.md`
- Optional `proofs/v014/short_field_h1_residual_classification.{py,json,md}`
  if useful.
- If code/tooling changed, include commands and proof.

Keep the top-level report short: verdict, bug class, evidence table, next
command. Then send:

```bash
tmux send-keys -t 0:2 'FABLE SHORT_FIELD_H1_RESIDUAL DONE - see .agent/reviews/2026-06-10-v014-short-field-h1-residual-fable.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
