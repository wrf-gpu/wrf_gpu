# Multi-GPU Readiness Path

Current state:
- `src/gpuwrf/contracts/halo.py` defines `HaloSpec` and `apply_halo`, but `apply_halo` is a single-GPU no-op.
- `GridSpec.halo_width` defaults to 2 and is validated in `[1, 4]`.
- No MPI, NCCL, `shard_map`, or collective-permute implementation exists.
- The operational path already calls `apply_halo` at RK/acoustic boundaries, so the call shape is present but the data movement is not.

Fields that need halo exchange:
- Dynamics/acoustic core: `u`, `v`, `w`, `theta`, `p`, `p_total`, `p_perturbation`, `ph`, `ph_total`, `ph_perturbation`, `mu`, `mu_total`, `mu_perturbation`.
- Moist/physics coupling if decomposition is horizontal: `qv`, `qc`, `qr`, `qi`, `qs`, `qg`, `Ni`, `Nr`, `Ns`, `Ng`, `qke`.
- Surface/PBL fields if physics uses neighbor-derived tendencies or post-physics boundary smoothing: `ustar`, `theta_flux`, `qv_flux`, `tau_u`, `tau_v`, `rhosfc`, `fltv`, `t_skin`, `soil_moisture`, `xland`, `lakemask`, `mavail`, `roughness_m`.
- Accumulation fields (`rain_acc`, `snow_acc`, `graupel_acc`, `ice_acc`) generally do not need per-step halo for column physics, but do need correct ownership and output stitching.
- Lateral boundary leaves (`u_bdy`, `v_bdy`, `theta_bdy`, `qv_bdy`, `ph_bdy`, `mu_bdy`) are not neighbor halos; each rank needs the relevant physical-domain boundary strips.

Payload estimate per full halo exchange, halo width 2:
- Current 3 km d02 shape `(44, 66, 159)`: ~4.36 MiB if all listed prognostic/coupling fields are exchanged.
- Derived full-domain 1 km shape `(44, 198, 477)`: ~13.07 MiB for the same field set.
- Largest contributors are FP64 vertical/pressure/geopotential fields: `w`, `ph`, `ph_total`, `ph_perturbation`, `p`, `p_total`, `p_perturbation`.

Expected communication overhead:
- If exchanged once per timestep, raw payload is probably manageable on a single node; the 1 km estimate is only ~13 MiB per exchange before protocol overhead.
- If exchanged inside every acoustic substep or after every tiny fused elementwise operator, communication will dominate. The multi-GPU design must exchange at numerically required operator boundaries only.
- The project needs a compiled-region map before multi-GPU work so halo boundaries do not undo the planned fusion.

Implementation path:
1. Freeze a machine-readable halo field registry with field, dtype, staggering, owner, halo depth, and exchange cadence.
2. Add a CPU/JAX reference halo pack/unpack test on a tiny decomposed grid, without GPU runtime.
3. Implement single-node multi-GPU proof with JAX `shard_map` or GSPMD-style explicit sharding first, because it keeps arrays inside XLA and reduces the risk of accidental host round-trips.
4. Use `collective_permute`/equivalent neighbor exchange for halos, then prove `d2h_inter_kernel_inside_window == 0`.
5. Move to explicit NCCL or MPI only if `shard_map` cannot express the required halo cadence, overlap, or multi-node topology.

Recommendation:
Start with `shard_map`/XLA collectives for a two-GPU single-node proof. Explicit NCCL/MPI is the long-term WRF-like rank-per-GPU shape, but it is higher risk in this JAX codebase because any Python-mediated exchange would threaten the zero-D2H invariant.
