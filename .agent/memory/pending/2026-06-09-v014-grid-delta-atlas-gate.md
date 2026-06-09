# V0.14 Grid-Delta Atlas Gate

Date: 2026-06-09 16:08 WEST

Principal requirement: v0.14 TOST must not be station-only. The final validation
must also compare GPU-vs-CPU-WRF wrfouts over all paired cases, all lead times,
all grid cells, and all common numeric fields.

Decision file:

- `.agent/decisions/V0140-GRID-DELTA-ATLAS-GATE.md`

Required release artifacts:

- `proofs/v014/grid_delta_atlas/manifest.json`
- `proofs/v014/grid_delta_atlas/grid_delta_summary.json`
- `proofs/v014/grid_delta_atlas/GRID_DELTA_ATLAS.md`
- selected plots under `docs/assets/v014/grid_delta_atlas/`
- README section embedding the dashboard plot and linking the atlas report

The claim should be "near-equivalence under predeclared field envelopes and
stable bounded drift", not bitwise cell identity, unless the data actually prove
bitwise identity. Station TOST must be interpreted together with this atlas.
