You are Fable 5 medium, focused debug/fix worker for wrf_gpu2 v0.14.

Work in this dedicated worker git worktree:
`/home/enric/src/wrf_gpu2/.claude/worktrees/fable-v014-venting-residual-fix`
(branch `worker/fable/v014-venting-residual-fix`; verify `git log -1` is the
advance_w-fixed merge tip before you start).

Read and follow:
`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-fable-venting-residual-fix/sprint-contract.md`

Task summary:
Close the remaining Switzerland/Gotthard h36 strong-flow dry-mass VENTING
residual so the 72h field gate can pass. The dominant per-substep (w,phi)
creator is already fixed and merged (b14b5f17: pg_buoy carried grid%p, w
cosine-Coriolis + curvature, WRF open top); the rw_tend lane is closed (stage
ph -96.5%) and the 2h short forecast is stable. What remains is a smaller
interior residual: hourly venting excess only improved to -26.6 Pa/cell/h vs
the CPU -74.5 reference. Find and fix the WRF-faithful interior root, or name
the exact next mismatching WRF term with proof.

Reuse the existing WRF-native oracle — do NOT rebuild it:
- `proofs/v014/wrf_native_advance_w_dump.py` + dumped truth at
  `/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump/awd_dumps/` (call
  21601 -> 21602).
- `proofs/v014/switzerland_acoustic_substep_blocker.py --stage-compare` (GPU
  interior increment rmse vs WRF call 21602).
- 2h open-top short forecast venting budget from
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable`.

Candidates to TEST (build your own ranked ledger; reject any the oracle does not
support): coupled work-theta `t_2` (53.7% rel, carries remaining stage p
through the EOS), `ph_tend` into the implicit solve (13.9%), `mu''` (15.5%),
`ww` (51% of a tiny field). Do not treat the manager's EOS/theta suspicion as
truth; prove it.

Hard constraints:
- No clamps, no masking, no tolerance changes, no JAX-vs-JAX self-acceptance.
- No host/device transfer inside the timestep loop.
- Do NOT run the long 72h GPU gate (the manager runs it after merge).
- Do not touch `/home/enric/src/wrf_pristine/WRF` in place; disposable
  instrumentation only. Do not touch `/home/enric/src/canairy_waves`.
- Do not use Hermes/Telegram or `ask-hermes`.
- Commit your source fix + proof artifacts on your worker branch.

The GPU is currently free; you may use it for the stage gate and the 2h short
forecast. Do not start any run longer than the 2h short forecast.

Write:
- `.agent/reviews/2026-06-11-v014-fable-venting-residual-fix.md`
- proof artifacts under `proofs/v014/` (stage-compare gate + short-forecast
  venting budget JSON).

End stdout exactly:
`FABLE V014_VENTING_RESIDUAL_FIX DONE - see .agent/reviews/2026-06-11-v014-fable-venting-residual-fix.md`
