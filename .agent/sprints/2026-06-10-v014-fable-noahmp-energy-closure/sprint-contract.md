# Sprint Contract: V0.14 Fable NoahMP Land-Tile Energy Closure

Date: 2026-06-10 WEST
Owner: Fable/Mythos in tmux `0:1`
Manager: `worker/gpt/v013-close-manager`
Base commit: `43accdc6`

## Objective

Close the current v0.14 strict Step-1 blocker as one whole hard task.

The previous sprint removed the broad "NoahMP disabled/missing land/static
state" blocker and proved a narrower WRF-anchored residual:
`NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_NOAHMP_LAND_TILE_ENERGY`.

Endpoint:

- preferred: strict Step-1 green in `proofs/v014/noahmp_step1_closure.py`
  (`max_abs <= 1.0e-3`, `rmse <= 1.0e-5`) after a production/proof fix; or
- acceptable fallback: exact WRF-anchored blocker narrower than NoahMP land-tile
  energy, with a ranked hypothesis table, proof artifacts, and fastest next
  command.

Do not return a micro-blocker.

## Current Evidence

Accepted artifacts:

- `.agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md`
- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- `proofs/v014/step1_mynn_source_coupling.{py,json,md}`
- `proofs/v014/step1_surface_land_flux_handoff.{py,json,md}`
- `.agent/sprints/2026-06-10-v014-fable-noahmp-step1-closure/manager-closeout.md`

Critical facts:

- Step-1 config now has `use_noahmp=True`, `sf_surface_physics=4`, and
  `inputs_have_noahmp_land=True`.
- Strict Step-1 is still red against pinned one-run WRF truth:
  max_abs `1489.5135568470864`, RMSE `13.2001844004901`.
- Worst strict cell starts at Fortran `i=66`, `j=37`, `k=3`.
- WRF exact SWDOWN/GLW swap does **not** collapse the land theta_flux residual.
- Land inputs match WRF hook print precision (`<=5e-9` for core land fields).
- MYNN kernel is exonerated with WRF inputs: raw `RTHBLTEN` corr `0.99993`.
- The likely chain is inside NoahMP land-tile surface energy/albedo/HFX:
  FVEG/LAI/SAI, CM/CH in/out, two-stream SAV/SAG/FSR/FSA albedo, SH/EV/GH/TRAD,
  T2MV/T2MB, and EFLXB terms.

## Required Work

1. Read governing docs and the accepted review/proof artifacts listed above.
2. Add or reuse a per-column WRF `noahmplsm` energy in/out hook on the pinned
   truth tree. Start with the strict worst cell and enough neighboring/strong
   land cells to distinguish a one-cell artifact from a systematic chain bug.
3. Compare WRF energy/albedo/flux internals against the JAX `physics.noahmp`
   solve at the same Step-1 boundary. Prefer a focused proof script over
   full-run iteration.
4. Fix the diverging JAX NoahMP chain if local. Keep WRF fidelity first; no
   clamps, tolerance widening, or CPU-WRF runtime dependency.
5. Rerun `proofs/v014/noahmp_step1_closure.py` and the contract gates.

## File Ownership

Allowed production files if needed:

- `src/gpuwrf/physics/noah_mp.py`
- `src/gpuwrf/physics/noahmp/**`
- `src/gpuwrf/physics/noahmp_coupler.py`
- `src/gpuwrf/coupling/noahmp_surface_hook.py`
- `src/gpuwrf/contracts/noahmp_state.py`
- focused tests under `tests/`

Allowed proof/review files:

- `proofs/v014/noahmp_step1_closure.*`
- new `proofs/v014/noahmp_land_tile_energy_closure.*`
- `.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`

Do not edit:

- RRTMG/radiation production files.
- TOST, Switzerland/Gotthard, Grid-Delta Atlas, FP32/memory lanes.
- unrelated dycore/runtime code unless the proof shows the NoahMP energy fix
  directly requires a narrow coupling change.

## Acceptance Gates

Required:

```bash
python -m py_compile proofs/v014/noahmp_step1_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/noahmp_step1_closure.validated.json
python -m json.tool proofs/v014/noahmp_land_tile_energy_closure.json >/tmp/noahmp_land_tile_energy_closure.validated.json
git diff --check
```

If production code changes, also run:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_noahmp_coupler.py tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py
```

Add focused tests for any touched NoahMP chain.

## Constraints

- CPU proof work preferred; no GPU needed unless the manager explicitly approves.
- Preserve GPU-native vectorized/JAX structure.
- No host/device transfer inside timestep loops.
- No dynamic-shape production arrays.
- No tolerance masking or stabilizing clamps.
- Keep output context-sparing.

## Handoff Requirements

Write:

- `.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`
- proof JSON/Markdown for the energy closure
- updated `noahmp_step1_closure` proof outputs
- focused tests if production changed

Handoff must include objective, files changed, commands run, proof objects,
unresolved risks, and next decision needed.

Completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'FABLE NOAHMP_ENERGY_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
