# V0.14 Nested-Pipeline Noah-MP Activation Proof (CPU-only)

Date: 2026-06-10 · Case: `20260501_18z_l2_72h_20260519T173026Z` · Verdict: **NOAHMP_NESTED_ACTIVATION_CPU_PROVEN**

Fix under proof: `nested_pipeline._load_domains` now reads per-domain
`sf_surface_physics`, wires Noah-MP (namelist bundle + seeded initial
carry) when it is 4, fails closed on unsupported options, and the wrfout
writer reads the EVOLVED land carry (`wants_carry`).

Run start: 2026-05-01T18:00:00+00:00 → WRF clock julian=120.75 yearlen=365.0

| domain | sf_surface_physics | use_noahmp | static/params/land non-null | n_land_cells | init land-mean TSK [K] |
|---|---|---|---|---|---|
| d01 | 4 | True | True | 523 | 300.95 |
| d02 | 4 | True | True | 768 | 294.88 |

Fail-closed (sf=2 stub): raised=True
Structural: {"operational_namelist_has_all_replaced_fields": true, "operational_carry_has_noahmp_slots": true, "initial_operational_carry_accepts_noahmp_seeds": true, "writer_wants_carry": true, "domain_tree_output_honours_wants_carry": true}

GPU used: NO (backend=cpu; JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES='').

GPU-only remainder (manager gates): State/initial-carry device build + noahmp_initial_rad RRTMG t=0 seed require a visible GPU (contracts/state.py State.zeros); covered by the manager GPU gates (memory preflight, Canary h1-h4, 72h rerun).

Problems: none
