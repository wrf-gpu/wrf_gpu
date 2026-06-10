# Memory Patch Proposal

## Scope

v0.14 field-parity manager memory.

## Evidence

The fixed Canary LBC run h1/h4 outputs show `PSFC` remains a real
pressure-state residual: GPU vapor is present in `QVAPOR`, but the pressure
state/`PSFC` path is vapor-light by about one vapor-column load.

## Proposed Destination

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`

## Patch

Record that the Canary fixed run is characterization only until the
`PSFC` moist pressure-state lane is fixed or formally bounded. Do not launch
Switzerland GPU as release evidence while this lane is open. Keep the active
Canary run to h24/final only for slope and trajectory evidence unless a fix
requires a relaunch sooner.

## Reviewer Status

Reviewer Status:

ACCEPTED_BY_MANAGER_PENDING_ROADMAP_PATCH.
