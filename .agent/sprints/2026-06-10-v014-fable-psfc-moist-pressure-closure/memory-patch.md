# Memory Patch Proposal

## Scope

v0.14 field-parity manager memory and release checklist.

## Evidence

Fable proved WRF runtime `PSFC` equals `p_hyd_w(kts)` from `grid%p_hyd_w`, the
moist hydrostatic dry-mass-column integral. Production now uses that diagnostic
path. CPU proof residual is sub-Pa; post-fix h24 expected `PSFC` RMSE is
`64.202 Pa` rather than the old `296.060 Pa`.

## Proposed Destination

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`

## Patch

Record `PSFC` diagnostic as fixed after CPU gates, with short GPU h1/h4
validation pending. Record the deeper 3D pressure-state lane as the next
blocker: operational acoustic w-equation currently uses dry `cqw` /
`pg_buoy_w_dry`; moist `calc_cq` + `pg_buoy_w` threading must be fixed or
formally bounded before Switzerland GPU or field-parity release promotion.

## Reviewer Status

Reviewer Status:

ACCEPTED_BY_MANAGER_PENDING_ROADMAP_PATCH.
