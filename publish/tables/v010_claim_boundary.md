# v0.1.0 Claim Boundary — What the GPU port IS and IS NOT

Honest scope matrix for the wrf_gpu2 v0.1.0 release. v0.1.0 is a **validated
single-domain GPU replay forecast** for the Canary Islands (d02 3 km + d03 1 km
Tenerife), driven by replayed CPU-WRF/Gen2 lateral boundaries + hourly land
refresh. It is **NOT** a standalone WRF v4 replacement. The honesty is the point.

Sources: `publish/GPU_PORT_GAPS_TODO.md` (GPT-5.5 code-grounded P0/P1/P2 audit at
`5319b8d`), `.agent/decisions/POST-0.1.0-ROADMAP.md` (principal-directed
sequencing, 2026-05-31).

| Feature | WRF v4 has | v0.1.0 status | v0.2.0 target | proof / source path |
|---|---|---|---|---|
| **d02 3 km forecast** | RK3 split-explicit dycore + full physics on 3 km grid | **VALIDATED** — `D02_VALIDATED`, 3 real days, ≈ nightly CPU-WRF, stable 72 h, beats persistence on U10/V10 | maintained; close real-grid dycore parity (P0-6) | `proofs/v010_validation/v010_d02_result.json`; `publish/tables/v010_d02_validation.md` |
| **d03 1 km (Tenerife)** | 1 km island nest with own physics cadence | **BOUNDED_FAIL (honest)** — boundary-pump bias FIXED (v4 ~6.8 K → v5fix T2 3.01 K final / 2.43 K mean); residual daytime surface-flux warm bias | tighten via P0-6 + P1-3/P1-4 surface/PBL/radiation fidelity | `proofs/v010_validation/d03_summary_run24h_v5fix.json`; `publish/tables/v010_d03_status.md` |
| **Live multi-domain nesting** | `max_dom` tree, parent/child interp, child subcycling, optional feedback | **NOT PRESENT** — offline replay only; `GridSpec` is one grid; daily mode takes one `domain` (default d02) | **S1 / P0-1**: domain-tree runtime, parent→child interp, per-domain cadence | GPU_PORT_GAPS_TODO P0-1; ROADMAP item P0-1 |
| **Native init (WPS / real.exe)** | WPS + real.exe build wrfinput/wrfbdy from external analyses | **NOT PRESENT** — consumes pre-existing Gen2/CPU-WRF artifacts (IC, boundary strips, metrics, land) | **DEFERRED to AFTER v0.2.0** (highest risk); pragmatic default keeps `real.exe` | GPU_PORT_GAPS_TODO P0-2; ROADMAP item P0-2 (LAST) |
| **Prognostic Noah-MP** | Full prognostic LSM (soil T/moisture, skin, snow, canopy, energy/water budget) | **NOT PRESENT** — prescribed Noah-MP subset; land fields refreshed hourly from Gen2 corpus | **P0-3**: prognostic Noah-MP (or WRF-faithful operational subset), remove hourly corpus refresh | GPU_PORT_GAPS_TODO P0-3; ROADMAP item P0-3 |
| **d01 cumulus (Kain-Fritsch)** | `cu_physics=1` Kain-Fritsch on d01 (off d02–d05) | **NOT PRESENT** — physics hardwired Thompson→sfclay→MYNN→RRTMG; no `cu_physics` selector. (Does not affect current d02-only path: d02 has `cu_physics=0`) | **P0-4**: Kain-Fritsch for live d01 parent | GPU_PORT_GAPS_TODO P0-4; ROADMAP item P0-4 |
| **wrfout / wrfrst completeness** | Rich `wrfout` history + full-state `wrfrst` restart | **PARTIAL** — "Minimal WRF-compatible" writer (41 min vars, all readable); restart = project pickle checkpoints, NOT WRF `wrfrst` | **P0-5**: full Canary wrfout/wrfrst variable contract + restart-continuity proof | GPU_PORT_GAPS_TODO P0-5; `proofs/v010_validation/wrfout_inventory.json` (status PASS, 24/24 files, 41 vars) |
| **Conservation budgets** | Mass/moist/scalar conservation, pos-def/monotonic options, no hidden masking | **PARTIAL** — idealized dry-mass drift ≤ 1e-8 proven (warm bubble / Straka); runtime guards revert invalid moisture/theta; full coupled 24–72 h budgets NOT a closed WRF proof | **P0-7**: coupled budget diagnostics + thresholds + guard-engagement reporting + remove/justify masking | GPU_PORT_GAPS_TODO P0-7; `proofs/f7n/*_diagnostics.json` (idealized mass drift) |
| **Multi-GPU** | (WRF: MPI domain decomposition) | **NOT PRESENT** — single-GPU only; no `shard_map`/`pmap`/`Mesh`/`jax.distributed`; `contracts/halo.py` is a designed-in no-op with MPI-compatible call shape | **S1**: single-node domain decomposition (sharded stencils + halo exchange) | GPU_PORT_GAPS_TODO; ROADMAP item S1 (v0.2.0) |

## Notes

- **Single-GPU portability is unblocked today** (no roadmap item): pure JAX/XLA
  recompiles for Hopper (sm_90) / H100 / H200 with a standard `jax[cuda12]`
  install; expected faster (full-rate fp64, ~2–2.7× HBM bandwidth) and larger
  single-GPU domains. Speedup-vs-CPU is hardware-specific and must be re-measured
  on the actual box. Source: `.agent/decisions/POST-0.1.0-ROADMAP.md`.
- The Canary corpus namelist is a five-domain 9/3/1 km nest (`max_dom=5`,
  Thompson, MYNN, revised surface layer, Noah-MP, RRTMG, Kain-Fritsch on d01,
  `topo_shading=1`, `slope_rad=1`, specified/nested boundaries). v0.1.0 owns only
  the d02 replay path. Source: GPU_PORT_GAPS_TODO Audit Context.
