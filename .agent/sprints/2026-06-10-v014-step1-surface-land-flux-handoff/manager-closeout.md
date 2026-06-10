Merge Decision: ACCEPT AND COMMIT as a v0.14 grid-parity narrowing sprint.

Manager assessment:
The sprint did not close strict Step-1, so it is not a release gate pass. It did
deliver the contract's acceptable endpoint and made the remaining problem much
more concrete. WRF's surface-layer output, NoahMP overlay, and MYNN input are now
bounded: `PRE_NOAHMP -> POST_NOAHMP` is the heat/moisture flux change point, and
`POST_NOAHMP -> MYNN` is exact for HFX/QFX/UST. The JAX proof path is now known
to be missing the relevant NoahMP configuration/state (`use_noahmp=False`,
`sf_surface_physics=None`, no NoahMP land/static state).

Proof objects:
`proofs/v014/step1_surface_land_flux_handoff.py`, `.json`, `.md`, and
`_wrf_patch.diff`; review at
`.agent/reviews/2026-06-10-v014-step1-surface-land-flux-handoff.md`. The prior
`step1_mynn_source_coupling.json` was rerun and only its metadata was refreshed.

Validation:
The worker ran py_compile, the new CPU proof, JSON validation, the prior MYNN
source-coupling proof, JSON validation, and `git diff --check`. The manager
spot-reran py_compile, JSON validation, and `git diff --check`.

Next sprint:
Escalate the whole remaining Step-1 NoahMP/land-state closure to Fable/Mythos in
tmux `0:1` after `/compact`. Endpoint: production fix plus strict Step-1 green
if possible; otherwise an exact WRF-anchored blocker that is narrower than
missing NoahMP configuration. Do not run TOST, Switzerland, broad FP32, or long
GPU validation before this parity blocker is closed or explicitly bounded.
