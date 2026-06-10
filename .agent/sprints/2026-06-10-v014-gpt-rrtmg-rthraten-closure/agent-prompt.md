You are GPT-5.5 xhigh, RRTMG/RTHRATEN closure worker for wrf_gpu2 v0.14.

Read:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/sprints/2026-06-10-v014-gpt-rrtmg-rthraten-closure/sprint-contract.md`

Verify base: `git log -1 --oneline` should be `649d8e0f` or a descendant on
branch `worker/gpt/v013-close-manager`.

Goal: close or formally bound the field-dominant strict Step-1 residual:
RRTMG clear-sky `RTHRATEN` / GLW. The MYNN lane is already formally bounded by
`proofs/v014/mynn_rthblten_step1_closure.*`; do not reopen broad MYNN unless a
new proof contradicts it.

Current evidence:

- `proofs/v014/mynn_rthblten_step1_closure.*`: RRTMG substitution collapses
  strict RMSE `2.5378 -> 0.5433` and p99 `16.63 -> 0.84`; operational source
  leaf reassembly matches runtime to `4.55e-13`.
- `proofs/v014/rrtmg_step1_forcing_parity.*`: current verdict
  `RRTMG_STEP1_RESIDUAL_LOCALIZED_TO_CLEAR_SKY_DERIVED_RRTMG_BOUNDARY`; GLW bias
  `17.44 W/m2`, mass-coupled `RTHRATEN` RMSE `2.488`, max `19.425`.

Endpoint:

1. production/proof fix that materially reduces RRTMG `RTHRATEN` and strict
   Step-1 field residual; or
2. WRF-anchored bound narrower than "clear-sky derived RRTMG boundary" naming
   exact derived quantity, owner, and fastest next command.

CPU-only unless the manager approves GPU. No TOST/Schweiz/FP32/memory edits. No
clamps, tolerance widening, CPU-WRF runtime dependency, or in-loop transfers.

Write:

- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-rthraten-closure.md`
- proof JSON/Markdown for any new localization/fix
- refreshed `proofs/v014/rrtmg_step1_forcing_parity.*` and
  `proofs/v014/noahmp_step1_closure.*` if rerun/corrected
- focused tests if production changed

Print completion marker to tmux `0:2`:

`GPT RRTMG_RTHRATEN_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-gpt-rrtmg-rthraten-closure.md`
