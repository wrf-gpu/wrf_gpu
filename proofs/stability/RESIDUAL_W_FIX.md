# Residual finite-but-unphysical w growth in the 24h coupled forecast — root cause + WRF-faithful fix

Branch `worker/opus/residual-w`, base `6c45f9c`. Real case: Gen2 d02
`20260521_18z_l3_24h` (nz=44, ny=66, nx=159; dt=10s, n_acoustic=10; fp64;
top_lid=True, epssm=0.5, w_damping=1, damp_opt=3, zdamp=5000, dampcoef=0.2; guards
OFF). All GPU/JAX; no WRF launched. Radiation-off (`no_rrtmg`) memory-light scan
surrogate is used for the per-step localization (Agent B proved radiation is
irrelevant to the w mode); the official gate runs full physics WITH radiation.

## What was wrong at base `6c45f9c`

Agent B's MYNN-w-roundtrip fix turned the 15h NaN detonation into a finite but
slowly-ramping **surface** w mode: `segscan_24h.json` had `all_finite=true` but
`physically_plausible=false`, w_min = **-1159 m/s** (theta_max = 494 K was a RED
HERRING — that is the legitimate stratospheric base-state theta at the model top
k43; it is flat 493-496 K for the whole run and not colocated with the w).

## Localization (proofs in this dir)

`scan_trace.py` (probe **fixed** for Agent A's radiation rewrite: `rrtmg_adapter`
no longer exists; the "radiation off" noop now zeroes `rrtmg_theta_tendency`, the
held-rate primitive) per-step ONE-compile trace, plus `surface_w_hotspot.py`:

| run | w@24h | level | shape |
|---|---|---|---|
| base (no_rrtmg) `scan_trace_no_rrtmg.json` | **1147 m/s** | k0 surface | LINEAR ramp ~47 m/s/hr |
| MYNN OFF (`norad_no_mynn`) | 14 m/s | k15 mid | FLAT, physical |
| after A2C fix only (`scan_trace_fixed_no_rrtmg.json`) | 146 m/s | k0 | linear ramp ~5 m/s/hr |
| after BOTH fixes (`scan_trace_fix2_no_rrtmg.json`) | **14.3 m/s** | k15 mid | FLAT, physical |

The mode is **entirely at the k0 surface face**: at 12h `surface_w_hotspot.json`
shows w@k0 = 73 m/s at the hotspot but w@k1 = 1.4 m/s and the whole interior column
≤14.3 m/s (p99.9 = 10 m/s). It is **MYNN-driven** (gone with MYNN off) and lives over
the steepest Canary volcanic cells (Mt. Teide, ht→2987 m; steepest 820 m per 3 km
cell ≈ 27% slope); `corr(|w@k0|, terrain steepness) = 0.55`.

## Two-stage root cause (both real, both WRF-faithfully fixed; NO masking/clamps)

**Stage 1 — MYNN momentum coupling overwrote the full C-grid wind via a non-identity
face→mass→face round trip.** `_state_from_mynn_output` did
`u = _mass_to_u_face(u_mass_after_mynn)`, re-interpolating (smoothing/shifting) the
WHOLE dynamics u/v field every step, not just the PBL increment. WRF instead ADDS
the A-grid PBL increment (RUBLTEN/RVBLTEN), averaged to the C-grid faces, onto the
existing C-grid wind and never replaces it
(`phys/module_physics_addtendc.F::add_a2c_u/add_a2c_v`, lines 2531-2582:
`lvar += 0.5*(rvar(i)+rvar(i-1))`, edge faces excluded). Fix: form the MYNN increment
on mass points and add ONLY that increment, A2C-averaged, onto the original C-grid
faces (`physics_couplers.py::_add_a2c_u_increment/_add_a2c_v_increment`). 8× reduction
(1147→146 m/s).

**Stage 2 — the dycore surface-w BC was fed COUPLED winds instead of decoupled.**
The kinematic terrain-following lower BC (`advance_w.py:274-303`,
WRF `module_small_step_em.F:1384`: `w = mx·u·dz/dx + my·v·dz/dy`) uses
`cf1·u(1)+cf2·u(2)+cf3·u(3)` with NO mass-factor division — WRF passes the
**decoupled** prognostic winds `grid%u_2/grid%v_2` there (`solve_em.F:1500-1501`).
The operational acoustic core was passing `uv_state.u/v` = the **coupled** small-step
perturbation work arrays (`small_step_prep` `u_work = (c1h·muu+c2h)·u/msf`,
~1e4-1e5× the physical wind). Over the steepest cells this produced an O(40×) surface
w@k0 once MYNN sustained a near-surface wind there. Fix: pass the decoupled stage
winds `uv_state.u_1/v_1` (= WRF `grid%u_2/v_2` at stage entry; the surface BC is slowly
varying) — `acoustic.py:566-584`. Verified: recomputing the kinematic BC from
decoupled final-state winds gives w_surface ≤ 2 m/s everywhere (`wsurf_check`), exactly
matching k1; the coupled feed gave the spurious 73 m/s at k0.

## Result

`scan_trace_fix2_no_rrtmg.json`: w FLAT at ~14.3 m/s @ k15 (mid-level) for the full
24h — identical to dycore-only / MYNN-off. No surface ramp, no migration to k0, no
top-of-column mode. The official full-physics segmented gate (`proofs/perf/segscan_24h.py`)
result is recorded in `proofs/perf/segscan_24h.json`.

## Files changed
- `src/gpuwrf/coupling/physics_couplers.py` — `_state_from_mynn_output` now uses WRF
  A2C incremental momentum coupling (`_add_a2c_u_increment`/`_add_a2c_v_increment`).
- `src/gpuwrf/dynamics/core/acoustic.py` — `acoustic_substep_core` passes decoupled
  `u_1`/`v_1` (not coupled `u`/`v`) to `advance_w`'s surface-w BC.
- `proofs/stability/scan_trace.py` — monkeypatch targets updated for the radiation
  rewrite; added `norad_strong_damp` probe variant.
- `proofs/stability/surface_w_hotspot.py` — new k0-hotspot / terrain-correlation probe.

## Probe commands (reproduce)
```
# fixed scan_trace (radiation off, full MYNN+surface+thompson), 24h:
PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  TF_GPU_ALLOCATOR=cuda_malloc_async OMP_NUM_THREADS=4 taskset -c 0-3 \
  python proofs/stability/scan_trace.py --hours 24 --variant no_rrtmg --out proofs/stability/scan_trace_fix2
# surface-w hotspot localizer (12h):
PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=4 taskset -c 0-3 \
  python proofs/stability/surface_w_hotspot.py --hours 12
# official acceptance gate (full physics incl. RRTMG, segmented):
PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
  taskset -c 0-3 python proofs/perf/segscan_24h.py --hours 24
```
