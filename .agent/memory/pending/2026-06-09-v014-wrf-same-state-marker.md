# Memory Patch Proposal

## Scope

Project-memory update for v0.14 CPU-WRF same-state instrumentation: before
dynamic model fixes, first prove that a source-derived WRF marker is hitting the
intended model step, domain, native stagger, and history variable source.

## Evidence

- `proofs/v014/wrf_same_state_marker_savepoint.json` and `.md` prove h10 `d02`
  maps to WRF `grid%itimestep=6000`, with `current_timestr_before_step`
  `2026-05-02_03:59:54` and `lead_seconds_after_step=36000`.
- The accepted marker patch is preserved at
  `proofs/v014/wrf_same_state_marker_patch.diff`.
- The final post-marker comparison is exact for `T/P/PB` and within `2e-6`
  max_abs for `U/V/W/PH` against the provided CPU h10 wrfout.
- Earlier attempts showed two recurring traps: a marker immediately after
  `small_step_finish` proves mapping but not the final history state, and a
  post-RK hook gated on `rk_step == rk_order` does not emit because `rk_step`
  is `4` after the RK loop while `rk_order` is `3`.
- The accepted WRF history `T` source at the post-RK location is
  `grid%th_phy_m_t0`; `grid%t_1` and `grid%t_2` are THM-side state and can
  mislead wrfout-history comparisons.

## Proposed Destination

`.agent/memory/stable/recurring-gotchas.md` after independent review approves
the wording.

## Patch

Proposed addition:

- For WRF same-state dynamic localization, first prove the marker step/domain
  and native-stagger patch against CPU wrfout before interpreting terms. At the
  post-RK/history location, use `grid%th_phy_m_t0` for wrfout-history `T`; do
  not use `grid%t_1` or `grid%t_2` as history `T`, and do not gate post-RK hooks
  on `rk_step == rk_order`.

## Reviewer Status

Reviewer Status: pending. Do not apply to stable memory until the dynamic
term-localization sprint verifies the marker remains the correct anchor for the
next term emitters.
