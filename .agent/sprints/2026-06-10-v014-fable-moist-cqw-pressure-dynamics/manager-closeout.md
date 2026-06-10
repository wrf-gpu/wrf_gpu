# Manager Closeout

Date: 2026-06-10 18:15 WEST
Manager: Codex
Worker: Fable/Mythos in tmux `0:1`

## Outcome

Closed as a production correctness fix for the v0.14 3D pressure-state dynamics
blocker.

Verdict:

`MOIST_CQW_GPU_H4_ACCEPT_DEFAULT_ON`

The old operational acoustic w-equation used the dry specialization
`dry_cqw`/`pg_buoy_w_dry`, so moist real cases relaxed `P+PB` onto a dry
hydrostatic column while CPU WRF relaxed onto the moist column. The accepted fix
implements WRF's moist `calc_cq` / `pg_buoy_w` water-mass loading and makes it
the default operational path; `GPUWRF_MOIST_CQW=0` remains a bisection escape
hatch.

## Proof Objects

- `proofs/v014/moist_cqw_pressure_dynamics_closure.py`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.json`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.md`
- `proofs/v014/moist_cqw_gpu_h4_validation.py`
- `proofs/v014/moist_cqw_gpu_h4_validation.json`
- `proofs/v014/moist_cqw_gpu_h4_validation.md`
- `.agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md`
- GPU run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z`

## Merge Decision:

Merge and push. The 3D pressure-state dry-vs-moist dynamics blocker is closed
for the v0.14 candidate. Long 72h Canary and Switzerland field-parity gates can
now run from the default-ON branch, while the all-field comparator must still
watch remaining static/base-state, surface, and radiation residuals.

## Key Numbers

- GPU h1-h4 rc `0`, harness `L2_D02_GREEN`.
- Peak VRAM `16921 MiB`.
- New GPU `P+PB(k0)` vs moist-column residual mean/RMSE:
  `-9.492/11.758 Pa`.
- CPU truth residual: `-13.349/13.444 Pa`.
- Previous PSFC-fix GPU baseline was dry-balanced:
  moist residual `-201.492/204.437 Pa`, dry residual `-5.990/9.268 Pa`.
- h1-h4 `P` RMSE improved `55.125 -> 22.642 Pa`.
- h1-h4 `U/V/U10/V10/T/W/PBLH` all improved; `QVAPOR` unchanged at the
  `1e-6` RMSE-delta level.

## Unresolved Risks

- The all-field h1-h4 comparator still reports `FAIL`, led by static `MUB/PB`
  edge/base-state differences and surface/radiation fields. That is outside
  this sprint's assigned dry-vs-moist pressure-state bug, but must remain visible
  during the 72h gates.
- `PH` improves only modestly (`44.729 -> 42.495` RMSE) and must be watched over
  72h stability; no h1-h4 instability or NaN signal was found.

## Next Sprint

Run the mandatory 72h Canary d02 CPU-vs-GPU field-parity/stability gate from the
fully fixed default-ON candidate with resource CSV logging. If h24 or final
shows renewed drift, send a bounded Fable high analysis sprint with the full
compare trajectory; otherwise proceed to Switzerland 72h GPU and Grid-Delta
Atlas.
