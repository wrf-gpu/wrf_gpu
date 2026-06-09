# V0.14 Closeout Review: WRF Same-State Marker Savepoint

Verdict: `MARKER_GREEN`.

Findings:

- Green same-state marker was achieved only after using `grid%th_phy_m_t0` for history variable `T`. `grid%t_2` and `grid%t_1` are `THM`-side state for this output path and are not the wrfout `T` source.
- The early marker remains useful for step/index mapping but is not a same-state history marker. It can mislead dynamic localization if used for final h10 history comparisons.
- A post hook gated on `rk_step == rk_order` will not emit at this location because `rk_step` is `4` after the RK loop while `rk_order` is `3`.
- No GPU evidence was produced or claimed; all proof runs were CPU-only dmpar.

Evidence:

- Final comparison: `/mnt/data/wrf_gpu2/v014_same_state_wrf/compare_post_thphy_marker.json`.
- Archived attempt comparison: `/mnt/data/wrf_gpu2/v014_same_state_wrf/compare_archived_marker_runs.json`.
- Patch diff: `proofs/v014/wrf_same_state_marker_patch.diff`.
- Savepoint proof: `proofs/v014/wrf_same_state_marker_savepoint.json` and `.md`.

Recommendation:

Proceed with a V0.15 dynamic localization sprint using the thphy post-marker location as the reference. Add routine-boundary term-group emitters there, not at the early hook, and compare term patches to the green CPU h10 marker before any GPU-oriented work.
