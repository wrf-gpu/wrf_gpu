# M3 - GPU State And Grid Skeleton

Goal: create device-resident state, grid metadata, and halo contracts.

Deliverables:

- `GridSpec` contract
- `State` contract
- dummy timestep loop
- transfer audit

Acceptance gates:

- no hidden host/device transfers in dummy loop (audited; **zero**, not "low")
- state layout ADR accepted (ADR-002)
- `GridSpec` has named, machine-readable fields for: map projection, terrain/geog static-field provenance (source file, shape, units, checksum, projection transform, sanity check), vertical coordinate metadata, halo width and staggering, boundary-condition metadata (BC field names, update cadence, source dataset = AIFS per §11.6, interpolation policy)
- BC metadata schema frozen at M3 close

See `.agent/milestones/ROADMAP.md` M3 and `PROJECT_PLAN.md §7` for the full per-gate proof-object list.
