# Purdue-Lin microphysics (mp_physics=2) JAX port — Opus lane handoff

**Date:** 2026-06-04
**Branch:** `worker/opus/v060-lin-mp` (base `e998250`, trunk-0.9.0)
**Lane:** Opus implementer (CPU JAX fp64, cores 0-3, no GPU)

## Objective

Port WRF Purdue-Lin single-moment 6-class microphysics (`mp_physics=2`,
`phys/module_mp_lin.F`) to jit/vmap-able JAX, prove ISOLATED savepoint parity
against the UNMODIFIED pristine-WRF Lin scheme over all moist species
tendencies (qv,qc,qr,qi,qs,qg) + surface precip, and register mp=2 across the
shared dispatch/registry/namelist/scan-adapter surfaces.

## Result — PASS

`proofs/v060/lin_mp_savepoint_parity.json`: **6/6 cases PASS** vs the canonical
single-precision WRF Lin oracle, with all fields inside the predeclared
tolerances. Worst residuals vs the fp32 oracle:

| field | worst max_abs (any case) | tol |
| --- | --- | --- |
| theta | 4.5e-5 K | 1.0e-2 K |
| qv/qc/qr/qi/qs/qg | 1.8e-8 kg/kg (qv, rel ≤ 1.5e-6 ... 2.7e-4) | rel 1.0e-2 or abs 1.0e-7 |
| RAINNCV / SNOWNCV / GRAUPELNCV | 8.6e-8 mm | rel 1.5e-2 or abs 5.0e-4 mm |
| SR | 1.5e-8 | 1.0e-2 |

**Faithfulness floor (transparency):** a second oracle built from the SAME
unmodified source with `-fdefault-real-8` (double precision) is in
`proofs/v060/savepoints_lin_fp64`. The fp64 JAX port matches this fp64 oracle to
**~machine precision** (theta 1.1e-13 K, moisture ~7e-17 kg/kg, precip ~3e-17 mm)
across all 6 cases — proving the port is bit-faithful to the WRF Lin algorithm
and that the small fp32 residuals above are the reference's own single-precision
roundoff, NOT a port error. (Cases 3 and 5, which have no cloud water, already
matched the fp32 oracle to ~1e-13; cases 1/2/4/6 needed the satadj convergence
fix below.)

## Oracle (WRF-faithful, never self-compared)

- Driver: `proofs/v060/oracle/lin_oracle_driver.f90` — drives the public
  `lin_et_al` entry (which calls `clphy1d` + `satadj` internally), graupel active
  (`F_QG=.true.` ⇒ `gindex=1`), 6 single-column regimes (warm BL / mixed-phase /
  cold ice-snow / graupel core / subsaturated evap / clean) mirroring the WSM6
  oracle's regimes.
- Build: `proofs/v060/oracle/lin_build_and_run.sh` (fp32) +
  `lin_build_and_run_fp64.sh` (fp64). Both copy `module_mp_lin.F` +
  `module_mp_radar.F` VERBATIM from pristine WRF; the only project-authored
  Fortran is the column-builder driver + a minimal `module_wrf_error` logging
  stub (`module_wrf_error_lin.f90`, never on the scheme physics path).
- WRF source sha256 recorded in the savepoint dirs and in the parity JSON:
  - `module_mp_lin.F`  `bb9d0b99ab4ecd5e...`
  - `module_mp_radar.F` `ca069a12cc149313...`

## JAX port

- `src/gpuwrf/physics/lin_constants.py` — WRF model constants + the
  `module_mp_lin.F` PARAMETER block + `ggamma` (8-term Hastings polynomial gamma)
  + `parama1/parama2` (32-entry Bergeron tables), evaluated in fp64.
- `src/gpuwrf/physics/microphysics_lin.py` — `_lin_column` (one column) →
  `lin_run` (vmap+jit batch) → `lin_physics_tendency` (frozen `PhysicsTendency`).
  Theta-based (no Exner conversion of the moist update). Reproduces the WRF
  process order exactly.
- `src/gpuwrf/physics/_lin_graupel.py` — the T<0C / T>0C graupel process block
  (dry/wet growth `delta4` decision, Bigg freezing, dep/sub, melt, melt-evap).
- `src/gpuwrf/physics/_lin_update.py` — conservation depletion clamps + state
  update (cold/warm branches), the 20-iteration Newton `satadj`, and the
  ice/water melt-freeze + second satadj.

### Engineering notes (the data-dependent paths)

- **Adaptive sedimentation:** the WRF `notlast` Courant-subcycling loop (per
  substep: recompute fall speeds, find the active `[min_q,max_q]` span, pick
  `del_tv` from the 0.9·Δz/vt Courant limit, downward flux sweep, surface accum
  or deposit into `min_q-1`) is reproduced with `jax.lax.while_loop` + a masked
  `jax.lax.scan` flux sweep. Fully traceable, no host transfer in the column loop.
- **satadj:** `jax.lax.fori_loop` (20 iters) with a "freeze on convergence"
  emulation of the WRF `if(absft<0.01) go to 300` early exit.

### The one real bug found & fixed (not roundoff)

The satadj Newton loop initially returned the STALE `qvsbar` carry when a cell
converged, instead of the `qvsbar` recomputed from the converged `tsat` (which is
what WRF uses at label 300). This caused a ~5% condensation-amount error in the
warm saturation adjustment (qc↔qv exchange) at every saturated cloud-water cell —
visible as cases 1/2/4/6 failing even against the fp64 oracle. Fixed by always
adopting `qvsbar_new` (constant after `tsat` freezes). After the fix all 6 cases
match the fp64 oracle to machine precision.

## Registration (shared surfaces — merge with the WSM5/WSM3 lane expected)

- `contracts/physics_registry.py`: `ACCEPTED_MP_PHYSICS += 2`; `MP_SCHEMES[2]`
  (`linscheme`); `MP_MOIST_MEMBERS[2]` = 6-class; `MP_NUMBER_MEMBERS[2]=()`.
  `assert_registry_consistent()` passes.
- `coupling/physics_dispatch.py`: `_MP_ENTRIES[2]` →
  `microphysics_lin.lin_physics_tendency` (gpu_runnable, `mp_flat` convention).
- `coupling/scan_adapters.py`: `lin_adapter` (passes geometric level height `z`
  for the Courant sedimentation, in addition to `dz`) + `MP_SCAN_ADAPTERS[2]`.
- `io/namelist_check.py`: mp=2 accepted (description string updated).
- `runtime/operational_mode.py`: `_SCAN_WIRED_OPTIONS["mp_physics"]` + resolver
  error message include 2.
- `.agent/decisions/V0.6.0-S0-FROZEN-CONTRACT.md`: MP menu + accept-matrix
  extended; FROZEN-CONTRACT EXTENSION section appended.

## Integration smoke

`proofs/v060/scanwire_smoke.py` extended with a Purdue-Lin (mp=2) adapter smoke
(3-step scan on a synthetic State) + an mp=2 resolution-accept combo. Full
harness `all_pass: true` (`proofs/v060/scanwire_smoke.json` regenerated).

## Commands run

```
proofs/v060/oracle/lin_build_and_run.sh        # fp32 oracle, 6 savepoints
proofs/v060/oracle/lin_build_and_run_fp64.sh   # fp64 oracle, 6 savepoints
python3 proofs/v060/run_lin_parity.py          # 6/6 PASS
python3 proofs/v060/scanwire_smoke.py          # all_pass
```
(all `taskset -c 0-3`, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=true`)

## Unresolved risks / notes

- Parity is single-column isolated (the established v0.6.0 MP gate); a full
  multi-step coupled forecast vs CPU-WRF with mp=2 is a downstream gate (same as
  every other v0.6.0 scheme).
- The `lin_adapter`'s surface-precip unit is treated as mm directly (the WRF
  `flux*del_tv` = kg m⁻² = mm); cross-checked numerically against the oracle's
  RAINNCV (agree to ~1e-8 mm). The `lin_et_al` "m → mm" comment refers to the
  legacy `episp0`-form code; the active flux form already yields mm.
- The standalone `_satadj_cell`/`_satadj_col` in `microphysics_lin.py` are
  superseded by the inline satadj in `_lin_update.py` (kept for reference; not
  on the live path).
- `module_wrf_error_lin.f90` is a Lin-specific copy of the existing KF
  `module_wrf_error.f90` stub plus the `wrf_err_message` module buffer that
  `ggamma` references; the shared KF stub was left untouched.
