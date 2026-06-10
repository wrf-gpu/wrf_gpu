# Memory Patch

Reviewer Status: ACCEPT.

Durable memory for future managers:

- Do not reopen the NoahMP/sfclay water-path moist-theta bug unless the proof
  itself is invalidated. The fix is in `src/gpuwrf/coupling/noahmp_surface_hook.py`,
  not `surface_layer.py`.
- The operational state stores WRF moist potential temperature `theta_m`; physics
  adapters that need sensible/dry temperature must receive an explicit dry view
  and recouple on writeback where needed.
- The accepted water-path proof is
  `proofs/v014/surface_layer_theta_decoupling.*`.
- Current strict Step-1 blocker after this sprint:
  `NOAHMP_STEP1_STRICT_RED_SURFACE_WATERPATH_CLOSED_NARROWED_TO_MYNN_EDMF_RTHBLTEN`,
  max_abs `53.52301833555157`, RMSE `2.5444971494115354`, worst cell
  `(i=20,j=7,k=2)`.
- Next endpoint-sized sprint should attack MYNN-EDMF `RTHBLTEN`; RRTMG is
  secondary. Do not run TOST or Switzerland-GPU before that frontier is green or
  formally bounded.
