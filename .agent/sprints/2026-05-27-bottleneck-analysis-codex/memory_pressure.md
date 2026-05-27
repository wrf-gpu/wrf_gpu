# Transient Memory And Allocator Pressure

Sources:
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/static_memory_model.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/live_vram_probe.json`

Measured 1 km synthetic full-domain probe:
- Shape: `(nz, ny, nx) = (44, 198, 477)`.
- Static State storage: 484,598,632 B = 462.149 MiB = 0.451 GiB.
- Tendencies: 168,610,464 B.
- Grid metrics: 8,332,800 B.
- Known resident State+Tendencies+Grid: 661,541,896 B = 631.849 MiB.
- Warm one-RK-step peak from `nvidia-smi`: 7,278 MiB on a 32,607 MiB RTX 5090.
- Transient estimate: 6,969,994,232 B, computed as peak process memory minus known resident bytes, including allocator/runtime overhead.

Where the 7.28 GB comes from:
1. Baseline process/runtime before allocation was already 2,470 MiB.
2. Full-domain synthetic resident State+Tendencies+Grid accounts for only ~632 MiB.
3. After cleanup, JAX reports only 13.9 MB `bytes_in_use`, but `peak_pool_bytes` remains 4,292,870,144 B and `pool_bytes` remains 4,292,870,144 B. The allocator is retaining a multi-GB pool.
4. The largest allocation seen after cleanup is 1,666,701,312 B, which points to large compiled temporary buffers or executable/runtime workspaces rather than persistent State.
5. The warm-step sampler stays at 7,278 MiB, so the peak is not persistent State size; it is transient buffers plus allocator retention.

Likely transient families:
- Pressure/geopotential acoustic temporaries: `p`, `p_total`, `p_perturbation`, `ph`, `ph_total`, `ph_perturbation`, coefficient arrays, and tridiagonal/PCR scratch.
- RK/acoustic save-family carry: `u_save`, `v_save`, `w_save`, `t_save`, `ph_save`, `mu_save`, `ww`, `ww_save`, `t_2ave`, `mudf`, `muave`, `muts`, `ph_tend`.
- Repeated dtype enforcement and state replacement paths that can materialize copies before XLA proves aliasing.
- Physics column adapters: `moveaxis`, mass/face interpolation, cloud fraction, density, and RRTMG/MYNN/Thompson column views.
- Output/land refresh is outside the compiled timestep loop but can keep process memory high across hourly segments.

Allocator-pressure opportunities:
1. Capture an XLA memory profile for `jit_run_forecast_operational` before editing. The current attribution is `nvidia-smi`-level and cannot prove which HLO owns the 1.67 GB largest allocation.
2. Strengthen aliasing/donation. `run_forecast_operational` donates State, but namelist leaves, tendencies, metrics, and save-family carries may still duplicate large arrays.
3. Fuse RK/acoustic field updates so XLA can shorten live ranges for pressure/geopotential and save-family arrays.
4. Avoid per-step `_enforce_operational_precision` copies if dtype invariants can be enforced at producer boundaries.
5. Keep physics adapters layout-stable instead of repeatedly materializing vertical-last column copies.
6. Revisit duplicate total/perturbation storage only through an ADR-backed operational-state design; the memory win is real, but those aliases are correctness-sensitive.

Risk:
Persistent State storage fits comfortably at the probed 1 km size. The risk is not static storage; it is transient peak, allocator retention, and lack of HLO-level attribution.
