# Manager Closeout: V0.14 Step-1 Live-Nest Perturbation-State Init

Date: 2026-06-09 19:36 WEST

## Outcome

The sprint is closed with verdict
`STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP`.

The manager hypothesis is supported but not yet source-patch-ready: WRF
recomputes or adjusts `P/MU/W` before `first_rk_step_part1_call`, while current
JAX keeps raw `wrfinput_d02` perturbation leaves through raw child, live child,
boundary package, initial carry, halo entry, and `_physics_step_forcing`.

Proof-local WRF formula transcriptions reduce the current residuals:

- `P_STATE`: `69.96875` -> `3.9458582235092763` Pa via start-domain pressure
  recompute.
- `MU_STATE`: `13.256103515625` -> `0.047773029698646496` Pa via `press_adj`.
- `W_STATE`: `0.7605466246604919` -> `1.2992081932505783e-07` m/s via
  `set_w_surface`.

No production source edit was applied. `P_STATE` still needs one internal WRF
`start_domain` truth surface (`al/alt` plus pre/post `press_adj`) before a safe,
GPU-native source patch.

## Proof Objects

- `proofs/v014/step1_live_nest_perturb_state_init.py`
- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

## Merge Decision:

Commit and push the proof artifacts. Do not merge a source fix from this
sprint; it is a localization sprint only.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_live_nest_perturb_state_init.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_perturb_state_init.py`
- `python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json >/tmp/step1_live_nest_perturb_state_init.manager.validated.json`
- `git diff -- src/gpuwrf` with no output
- `git diff --check`

## Scope Changes

No production source, GPU validation, TOST, Switzerland, FP32 source work,
memory source work, or Hermes was used.

## Lessons

For this hard debug path, the useful pattern was a narrow task with permission
to challenge the manager hypothesis and return ranked alternatives. The worker
found the expected missing start-domain family but split it into exact W,
near-closed MU, and still-not-exact P rather than overclaiming a patch.

## Next Sprint

Open one narrow WRF savepoint/source sprint inside live-nest
`start_domain(nest,.TRUE.)`: emit surfaces after the hypsometric `P/al/alt`
recompute and immediately before/after `press_adj`, including `P_STATE`,
`MU_STATE`, `al`, `alt`, `alb`, `PH_STATE`, `PB`, `MUB`, `PHB`, `theta`, `qv`,
`HT`, and `HT_FINE`. Patch `d02_replay` only if that surface closes `P_STATE`
and preserves GPU-native execution.
