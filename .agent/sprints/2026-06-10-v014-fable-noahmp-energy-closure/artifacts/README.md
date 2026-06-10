# Artifacts

Primary artifacts are tracked in the repository instead of copied here:

- `.agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md`
- `proofs/v014/noahmp_land_tile_energy_closure.{py,json,md}`
- `proofs/v014/noahmp_step1_closure.{py,json,md}`
- `src/gpuwrf/physics/noahmp_coupler.py`
- `tests/test_noahmp_coupler.py`

The temporary WRF energy-hook truth run was emitted under
`/tmp/wrfgpu2_v014_noahmp_energy_pinned_onerun/`; the reproduction command and
provenance are recorded in the review and proof JSON.
