Extracted pristine WRF control and solar oracles reused from review-cadence RCAD02/RCAD03.
Control includes driver call gates and advance_ppt ordering; solar includes radconst/calc_coszen REAL32 routines.
source_hashes.json pins their WRF source; these check source semantics, not coupled forecast fidelity.

The cadence expressions are frozen from solve_em.F/module_physics_init.F; all three fixture files have SHA256 checks in fixture_hashes.json. Tests compile these checked-in sources without a local WRF checkout. source_hashes.json is archival provenance, not a requirement on a mutable runtime table directory. A missing Fortran compiler skips with its explicit prerequisite reason; numerical assertions and tolerances are unchanged.
