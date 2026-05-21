# 1 km Nest Risk Audit

## Verdict

The 1 km nest is not ruled out by persistent state size, but it remains a real RTX 5090 32 GB compile/HBM risk. M7-S2 must produce profiler-backed peak HBM, XLA temporary, compile-time, and transfer artifacts before any 1 km operational claim.

## Evidence Inputs

- M7 plan makes 1 km conditional on the RTX 5090 32 GB memory/compile gate and forbids a public 1 km claim if the gate fails (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:20`, `:22`, `:118`, `:123`).
- Gen2 L3 geometry is static and one-way nested with d03/d04/d05 as 1 km domains (`m7-milestone-plan.md:390`, `:407`, `:411`, `:414`).
- M6-S6 hit CUDA OOM during pinned d02 drift after reduced TSC work; Opus diagnosed it as likely XLA compilation buffer accumulation across shape recompiles, not true steady-state 32 GB saturation (`.agent/sprints/2026-05-21-m6-s6-tier3-tsc/manager-closeout.md:22`, `:24`).
- M6 spacetime evidence shows the persistent d02 state is small relative to HBM, but RRTMG memory analysis reported a large temporary allocation path (`artifacts/m6/spacetime_budget_d02.json` field `per_kernel.rrtmg.memory_analysis.temporary_bytes = 13221287152`).
- Terrain/static correctness is a first-order risk: wrong topography invalidates Canary evidence (`RISK_REGISTER.md:15`).

## Grid And State Estimate

M7 plan domain dimensions:

| Domain | dx | e_we | e_sn | e_vert | Cell estimate |
|---|---:|---:|---:|---:|---:|
| d01 | 9 km | 94 | 60 | 45 | 253,800 |
| d02 | 3 km | 160 | 67 | 45 | 482,400 |
| d03 | 1 km Tenerife | 94 | 76 | 45 | 321,480 |
| d04 | 1 km Gran Canaria | 70 | 61 | 45 | 192,150 |
| d05 | 1 km La Palma | 70 | 58 | 45 | 182,700 |

Using the current M6 `State` leaf set and ADR-007 dtype mix, the approximate persistent state bytes are:

| Domain | Persistent state |
|---|---:|
| d01 | 20.6 MiB |
| d02 | 39.0 MiB |
| d03 | 26.0 MiB |
| d04 | 15.6 MiB |
| d05 | 14.8 MiB |
| d01+d02+d03 | 85.6 MiB |
| d01+d02+d03+d04+d05 | 116.0 MiB |

Formula used: current state leaf shapes from `src/gpuwrf/contracts/state.py`, including staggered U/V/W, mass 3D leaves, surface leaves, and lateral-boundary leaves; bytes use the current precision matrix in `src/gpuwrf/contracts/precision.py`.

## Why This Is Still Risky

- Persistent state is not the limiting term. XLA temporaries, compiled HLO, fusion decisions, radiation tables, output buffers, and compile retries dominate the 32 GB risk.
- M6 d02 evidence already includes a 13.22 GB RRTMG temporary estimate in a single-kernel memory analysis path. That is for d02, before static nested scans over d01/d02/d03 and before any sibling 1 km domains.
- M6-S6 showed repeated OOM allocations up to 8 GiB after multiple shape recompiles. M7 nesting will introduce more static shapes unless S2 isolates compile processes and controls cache state.
- The M7 plan PASS gate is stricter than "does not OOM": peak HBM <= 26 GB, compile <= 45 min, no timestep-loop transfers, and no XLA OOM/retry instability.
- Terrain/geog provenance must be checked before memory success means anything. The 1 km domains are only useful if `geo_em` topography, coastline, land mask, and static fields match Gen2 provenance.

## Required M7-S2 Proof

- `rtx5090_1km_memory_audit.json` with hardware, driver, CUDA, jaxlib, domain shapes, state leaf count, persistent bytes, physics table bytes, boundary buffers, output buffers, peak HBM, compile time, HLO bytes, max temporary bytes, timestep-loop transfer bytes, and verdict.
- `nesting_compile_smoke.json` proving a static one-way d01/d02/d03 scan compiles for at least 12 model hours or records a concrete fail class.
- Transfer audit showing zero host/device bytes inside timestep loops.
- Terrain/static proof with `geo_em` path, projection, checksums, max elevation sanity, coastline/landmask sanity, and parent-child map consistency.

## Risk Classification

- d03-only 1 km: `HIGH`, because compile/XLA temporary behavior is unknown but domain size is modest.
- d03+d04+d05 sibling 1 km: `VERY_HIGH`, because static nested scans and output buffers multiply compile pressure.
- 3 km-only M7 v0 fallback: `LOWER_RISK`, already the required fallback if M7-S2 fails.

## Decision Needed

Dispatch M7-S2 as an audit sprint after S0/S1 state factories exist. Do not authorize tiling, streaming, public 1 km output, or dashboard work until S2 returns a PASS or a reviewed deviation/tiling plan.
