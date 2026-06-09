# v0.14 Empirical/Static Memory Map

Date: 2026-06-09
Worker: GPT-5.5 xhigh
Branch inspected: `worker/gpt/v013-close-manager`
Write scope: proof script/report JSON/MD plus this review only.

## Objective

Produce an implementation-ready empirical/static memory map for remaining
non-radiation memory risks on the exact current branch. Treat RRTMG column,
band, and optics tiling as prior fixed evidence. Do not edit production `src/`,
do not run TOST, Switzerland validation, GPU jobs, or FP32 source work.

## Outcome

Verdict:
`NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY`.

The smallest safe memory-only source sprint, if any, is WDM6 `slmsk`
shape-only cleanup: keep current values, replace the full-column broadcast with
a per-column mask, and prove exact WDM6 output equality. It is opt-in and small
(`0.075119 GiB` recoverable at 641x321x50 fp64), so it does not block long
validation.

The only material bit-identical cleanup is moisture transport velocity reuse
when active moisture advection matters. Static source arithmetic gives
`0.237621-0.620881 GiB` recoverable depending on construction-transient overlap,
but this path is inactive unless `use_flux_advection` and `moist_adv_opt != 0`.

The larger remaining risks are measurement-first or semantic/dycore work:
MYNN BouLac dense `(ncol,nz,nz)` matrices, non-radiation column tiling,
post-physics sparse/donated merge, moisture limiter workspace, PBL/surface
diagnostic threading, and acoustic carry split. None should interrupt current
grid-cell parity work.

## Files Changed

- `proofs/v014/empirical_memory_map.py`
- `proofs/v014/empirical_memory_map.json`
- `proofs/v014/empirical_memory_map.md`
- `.agent/reviews/2026-06-09-v014-empirical-memory-map.md`

No production `src/` files were edited.

## Commands Run

- `sed -n` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`, and the v0.14 sprint contract.
- `sed -n` reads of `.agent/skills/designing-gpu-state/SKILL.md`,
  `.agent/skills/designing-gpu-state/references/memory-layout-rules.md`, and
  `.agent/skills/maintaining-memory/SKILL.md`.
- `git status --short --branch`
- `sed -n`/`jq` reads of the v0.14 memory roadmap, grid-parity handoff,
  exact-branch memory preflight, and prior RRTMG proof JSONs.
- `rg` and `nl -ba | sed -n` source inspections for moisture transport, WDM6
  `slmsk`, non-radiation column adapters, MYNN BouLac dense arrays,
  post-physics replacements, PBL/surface prep, limiter workspace, and acoustic
  carry.
- `python -m py_compile proofs/v014/empirical_memory_map.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/empirical_memory_map.py`
- `python -m json.tool proofs/v014/empirical_memory_map.json >/tmp/empirical_memory_map.validated.json`

One initial generator run returned `source_patterns_ok=False` because the MYNN
BouLac dense-array regex expected a wrapped source phrase on one line. The proof
script pattern was corrected, then all required validation commands passed.

## Proof Objects Produced

- `proofs/v014/empirical_memory_map.json`
- `proofs/v014/empirical_memory_map.md`

Key JSON facts:

- `validation.source_patterns_ok: true`
- all candidates have `blocks_v014_long_validation_after_grid_parity: false`
- `recommendation.smallest_safe_memory_source_sprint`: WDM6 `slmsk` shape-only cleanup
- `recommendation.only_material_bit_identical_cleanup`: moisture transport velocity reuse

## Unresolved Risks

- This is static/source evidence plus prior proof reconciliation, not a fresh
  GPU peak measurement and not a full transfer audit.
- MYNN BouLac dense-array liveness remains unmeasured; one fp64 dense
  `(ncol,nz,nz)` matrix is `3.832597 GiB` at 641x321x50, but XLA fusion/liveness
  must be measured before any rewrite.
- Post-physics merge and moisture limiter estimates are source-static/inferred,
  not measured peaks.
- Acoustic carry split remains dycore-adjacent with a prior reverted attempt;
  do not start it before grid parity closes.

## Next Decision Needed

After grid-cell parity closes, either:

1. run the selected long-validation exact-branch memory preflight and proceed if
   it fits, or
2. authorize a very small WDM6 `slmsk` shape-only cleanup sprint first.

Do not start MYNN/PBL/acoustic semantic memory work until measurement or grid
attribution makes it the binding problem.
