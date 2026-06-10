Merge Decision: ACCEPT AND COMMIT as a v0.14 grid-parity narrowing sprint.

Manager assessment:
This sprint did not close strict Step-1, so it is not a release gate pass. It did deliver the contract's acceptable endpoint: the broad MYNN/PBL source-coupling hypothesis is narrowed upstream of MYNN to the surface/land heat-moisture flux handoff. The proof also landed three scoped production fixes that are independently useful and WRF-anchored: MYNN receives `phy_prep` dry theta / hydrostatic pressure / rho / dz on grid-backed paths, dry-theta source leaves remain distinct from theta_m state, and first-call MYNN QKE initialization is ordered after surface fluxes.

Proof objects:
`proofs/v014/step1_mynn_source_coupling.py`, `.json`, `.md`, and `_wrf_patch.diff`; review at `.agent/reviews/2026-06-10-v014-step1-mynn-source-coupling.md`. Refreshed prior proof artifacts are retained because manager reran the contract gates under the changed adapter semantics.

Validation:
Manager reran py_compile, focused pytest (`13 passed, 1 skipped`), the new proof, three prerequisite proof scripts, JSON validation, and `git diff --check`.

Next sprint:
Open `v014-step1-surface-land-flux-handoff`. Endpoint: prove and fix the HFX/QFX/LH/TSK/GRDFLX handoff between WRF surface/land flux update and MYNN driver input, then rerun `step1_mynn_source_coupling.py` and strict Step-1.
