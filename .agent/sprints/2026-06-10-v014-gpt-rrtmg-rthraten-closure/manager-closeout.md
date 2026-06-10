# Manager Closeout: V0.14 GPT RRTMG/RTHRATEN Closure

Merge Decision: ACCEPT AND COMMIT. This is a real production fix plus proof refresh. It closes the gross RRTMG temperature-input bug and materially reduces the field residual, but it does not authorize TOST yet.

What changed:

- `src/gpuwrf/coupling/physics_couplers.py::_rrtmg_column_inputs` now uses dry theta for grid-backed RRTMG temperature input, matching WRF `phy_prep`.
- `tests/test_v014_dry_source_leaf_wiring.py` proves grid-backed RRTMG gets dry temperature while no-grid fallback keeps the old moist-theta behavior.
- `tests/test_m5_rrtmg_gate.py` was corrected for current honest fallback semantics: RRTMG M5 can now pass Tier-1/2 but still fallback on launch-budget.
- RRTMG, MYNN, and NoahMP proof artifacts were regenerated and stale pre-fix phrasing removed.

Manager interpretation:

- Before/after is strong: GLW RMSE `17.52 -> 0.352 W/m2`; RTHRATEN RMSE `2.488 -> 0.365`.
- Strict Step-1 post-fix is still red: max_abs `55.9297`, RMSE `0.4997`, p99 `0.9529`.
- MYNN owns the strict worst-cell max/floor; remaining RRTMG accounts for `63.5%` of WRF-QKE RMSE variance in the decomposition but is now bounded rather than grossly wrong.

Release decision:

- Do not start powered TOST or Switzerland-GPU from this signal alone.
- Next required decision: record an explicit operational tolerance policy for the non-bitwise MYNN/RRTMG mass-coupled Step-1 gate, then run a short all-field rollout falsifier. If that shows no radical drift, start TOST/Switzerland-GPU with CSV resource logging.
