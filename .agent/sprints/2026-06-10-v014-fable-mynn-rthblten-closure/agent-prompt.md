You are Fable/Mythos, high-end hard-debug worker for wrf_gpu2 v0.14.

Read:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-10-v014-fable-mynn-rthblten-closure/sprint-contract.md`

Current base commit is `edaa4b1c` on branch `worker/gpt/v013-close-manager`.
Verify it with `git log -1 --oneline`.

Goal: close the current strict Step-1 grid-parity blocker as a whole endpoint,
not as a micro-run. The current verdict is:

`NOAHMP_STEP1_STRICT_RED_SURFACE_WATERPATH_CLOSED_NARROWED_TO_MYNN_EDMF_RTHBLTEN`.

The NoahMP/sfclay water-path bug is fixed and accepted. Remaining strict
Step-1 metric is max_abs `53.52301833555157`, RMSE `2.5444971494115354`, worst
Fortran cell `(i=20,j=7,k=2)`. The residual is PBL dominated: WRF `RTHBLTEN`
about `-1275.66` at the worst water cell, while WRF `RTHRATEN` is only about
`-0.914`. Land worst is also PBL dominated. RRTMG is real but secondary.

Important: earlier accepted proofs showed the MYNN driver source output is
WRF-faithful when fed WRF-equivalent inputs and WRF/WRF-pinned QKE. Reconcile
that with the current operational `RTHBLTEN` residual before changing code. The
bug may be in operational inputs/path, QKE/init, EDMF/mixing-length, vertical
metrics, dry/moist theta, source-leaf units, or writeback, not necessarily the
MYNN arithmetic core.

Endpoint:

1. strict Step-1 green, or
2. a WRF-anchored formal bound narrower than "MYNN-EDMF RTHBLTEN", with exact
   file/function/variable ownership and fastest next command.

If MYNN is closed/bounded and strict is still red, continue to secondary RRTMG
using the existing `proofs/v014/rrtmg_step1_forcing_parity.*` evidence. Do not
stop at a narrow note unless the blocker is exact and manager-actionable.

No GPU unless the manager explicitly approves. No TOST, Switzerland, FP32, or
memory edits. Preserve GPU-native performance structure.

Required handoff:

- `.agent/reviews/2026-06-10-v014-fable-mynn-rthblten-closure.md`
- new proof JSON/Markdown if you create one
- updated `proofs/v014/noahmp_step1_closure.*`
- focused tests if production changed
- delayed completion marker to tmux `0:2`:

`FABLE MYNN_RTHBLTEN_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-mynn-rthblten-closure.md`
