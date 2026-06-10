You are GPT-5.5 xhigh, v0.14 post-EOS h1 residual adjudicator for wrf_gpu2.

Repo: /home/enric/src/wrf_gpu2
Branch context: manager branch worker/gpt/v013-close-manager.
Mode: CPU-only analysis on existing artifacts. Do not launch GPU, do not run WRF,
do not edit source code unless you discover a tiny, provably local comparator
metadata bug. Prefer analysis/proof artifacts only.

Objective
---------
Decide whether the v0.14 long GPU field-parity gates can start after the
EOS/theta fix and 1h GPU rerun, or whether a remaining blocker must be fixed
first.

You must return one of:
- PROCEED_72H_GATES: remaining h1 residuals are understood, bounded, and
  compatible with the current v0.14 release-gate/tolerance policy.
- NO_PROCEED_FIX_FIRST: a specific remaining bug class blocks 72h gates; provide
  the exact fastest next fix/proof command.
- INCONCLUSIVE_NEED_ONE_SHORT_RERUN: existing artifacts are insufficient, and
  one specific short rerun or comparator run is required before decision.

Current known state
-------------------
1. Fable fixed the mixed theta/EOS convention by making runtime State.theta
   moist theta_m for use_theta_m=1, production EOS qvf=1, writer dry T plus THM.
   Read:
   - .agent/reviews/2026-06-10-v014-eos-theta-semantics-fable.md
   - proofs/v014/eos_theta_semantics.md
2. Manager ran the 1h Canary d02 GPU falsifier after that fix:
   - run root: /mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T134205Z
   - GPU output:
     /mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T134205Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z
   - CPU truth:
     /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z
   - compare:
     /mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T134205Z/short_field_h1_grid_compare.{json,md}
   - resource CSVs:
     /mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T134205Z/resources/
3. Post-fix h1 headline metrics from the manager summary:
   - T RMSE improved old 1.457 K -> new 0.255 K; THM RMSE 0.242 K.
   - PSFC RMSE improved old 323.115 Pa -> new 124.299 Pa, bias -116.855 Pa.
   - P RMSE improved old 129.754 Pa -> new 39.090 Pa.
   - MU RMSE 98.291 Pa, bias +85.109 Pa.
   - PB/MUB static p99 mostly small but max ~250 Pa localized.
   - HFX RMSE 38.568 W/m2, LH 27.111 W/m2, PBLH 41.407 m.
   - SWDOWN/SWNORM/COSZEN unchanged timing-class residual.
4. Current release gate docs:
   - .agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md
   - .agent/decisions/V0140-STEP1-TOLERANCE-POLICY.md
   - proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md
   - proofs/v014/canary_file_inventory_gpt.md

Required work
-------------
1. Verify the post-fix compare JSON is parseable and inspect all common numeric
   h1 field stats, not only top 10.
2. For each core field class, decide whether the h1 result is:
   - green under existing/candidate tolerance;
   - known bounded/report-only but acceptable to start 72h stability gate;
   - an explicit blocker requiring fix before any 72h GPU gate.
   Mandatory fields to discuss: PSFC, MU, P, PB, MUB, PH, T, THM, U, V, W,
   QVAPOR, HFX, LH, PBLH, SWDOWN, SWNORM, COSZEN, GLW.
3. Check whether PB/MUB/PB max spikes are all spec-boundary/static/writer class
   or whether they indicate live interior drift. Use NetCDF indexing if useful,
   but keep output compact.
4. Check whether PSFC/MU/P residuals are within the existing hard dynamic
   candidate manifest or need a revised manifest before long gates. Do not
   invent weak thresholds; separate "safe to measure over 72h" from "release
   pass threshold already frozen".
5. Check whether radiation timing residual is enough to block 72h gates or can
   be report-only while long stability measures drift slope.
6. Compare current h1 to old pre-EOS h1 where useful:
   proofs/v014/short_field_falsifier_h1_grid_compare.{json,md}
7. Produce:
   - proofs/v014/post_eos_h1_residual_adjudication.md
   - proofs/v014/post_eos_h1_residual_adjudication.json
   Optional compact helper script:
   - proofs/v014/post_eos_h1_residual_adjudication.py

Output requirements
-------------------
Markdown structure:
- Verdict: one paragraph with one of PROCEED_72H_GATES,
  NO_PROCEED_FIX_FIRST, or INCONCLUSIVE_NEED_ONE_SHORT_RERUN.
- Evidence table: field, metric, old->new if available, class, gate implication.
- Boundary/static spike analysis for PB/MUB/PB-like fields.
- Radiation timing analysis.
- Manifest/tolerance implication: what is hard-pass now, what remains
  report-only, and whether that blocks starting 72h.
- Exact next manager command(s): either 72h GPU launch command family or exact
  next debug/fix command.
- Context-sparing handoff: max 10 bullets.

JSON must include:
{
  "verdict": "...",
  "can_start_72h_gates": true/false,
  "field_classes": {...},
  "blockers": [...],
  "report_only_nonblockers": [...],
  "next_commands": [...]
}

Validation:
- python -m json.tool proofs/v014/post_eos_h1_residual_adjudication.json >/tmp/post_eos_h1_residual_adjudication.validated.json
- If you create a Python helper: python -m py_compile proofs/v014/post_eos_h1_residual_adjudication.py

When done, notify the manager pane with delayed repeated enters:
tmux send-keys -t 0:2 'GPT POST_EOS_H1_ADJUDICATION DONE - see proofs/v014/post_eos_h1_residual_adjudication.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
