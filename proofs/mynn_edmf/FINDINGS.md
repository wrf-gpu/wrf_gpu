# MYNN-EDMF Mass-Flux Moisture Lane — Findings

Sprint: `worker/opus/mynn-edmf-moisture` (base `worker/opus/l1-rad-time` @ 5afd098)
Mode: CPU-only, GPU-free, no WRF rerun. fp64 JAX + linked pristine WRF objects.
Owner files: `src/gpuwrf/physics/mynn_pbl.py`, NEW `src/gpuwrf/physics/mynn_edmf.py`,
NEW `proofs/mynn_edmf/**`, NEW `tests/test_mynn_edmf_oracle.py`.

## Objective

Confirm/refute the pinpointed root cause: `mynn_pbl.py` transports qv as a **dry
eddy-diffusion solve + surface flux only**, missing the MYNN-EDMF **mass-flux (MF)
scalar transport** that WRF's `mynn_tendencies` applies (`s_awqv`, and in general
`sub_sqv`/`det_sqv`). If confirmed, port the MF qv transport, gated by a WRF column
oracle.

## What the operational config actually requires (resolved from WRF source)

For the d03 namelist (`bl_pbl_physics=5`, defaults from `Registry.EM_COMMON`):

| flag | value | consequence |
|---|---|---|
| `bl_mynn_edmf` | 1 | mass-flux ON |
| `bl_mynn_edmf_mom` | 0 | no momentum MF (`s_awu/s_awv` unused -> our plain-ED U/V is correct) |
| `bl_mynn_edmf_tke` | 0 | no TKE MF |
| `bl_mynn_mixqt` | 0 | "MIX WATER VAPOR ONLY (sqv)" path (not total-water) |
| `bl_mynn_cloudmix` | 1 | cloud mixing on |
| `bl_mynn_mixscalars` | 1 | scalar MF on |
| `env_subs` (hardcoded) | `.false.` | **`sub_sqv = det_sqv = 0`** |
| `bl_mynn_edmf_dd` (hardcoded) | 0 | **no downdraft -> all `sd_*` = 0** |

**Decisive simplification:** under this config the ONLY mass-flux moisture term
entering the qv solve is **`s_awqv1`** (the updraft total-water-minus-condensate
flux). Subsidence, detrainment, and downdraft are all identically zero. WRF refs:
`module_bl_mynnedmf.F:4316-4382` (qv solve), `:5603-6790` (`DMP_mf`),
`:6363-6382` (`s_awqv` assembly), `:343` (`env_subs=.false.`), `:337` (`edmf_dd=0`).

## Oracle method (true WRF, no rerun)

1. Extracted a real **d03 12z daytime-land column** (the exact equiv-T2 case
   `20260521_18z_l3_24h_20260522T133443Z`, point j=26 i=53, HFX=438.5 W/m2,
   PBLH=464 m, fltv=0.376) -> `column_d03_12z.json`.
2. **Fortran oracle** (`fortran_oracle/oracle.f90`) links the **pristine WRF
   objects** (`module_bl_mynnedmf.o` built 2026-05-29) and calls the REAL
   `DMP_mf` + `mynn_tendencies` on that column, for `edmf=1` and `edmf=0`.
3. **JAX port** (`mynn_edmf.py`) reproduces `DMP_mf` for the same column; compared
   to the Fortran s_aw/s_awqv arrays.

## Result 1 — JAX MF port is WRF-faithful (PASS)

`mf_oracle_compare.json`, predeclared tol = 5% relative max error:

| array | rel max err | tol | result |
|---|---:|---:|---|
| `s_aw` | 0.48% | 5% | PASS |
| `s_awqv` | 0.48% | 5% | PASS |
| `s_awqc` | 0.00% | 5% | PASS |

Residual is single-precision-WRF vs fp64-JAX plus plume tanh/condensation
nonlinearity. The surface-updraft interface (k=1) matches to 6 digits; the
plume-integrated levels (k=2-4) match to <0.5%. The plume activated through the
shallow PBL (ktop=4, maxmf=0.055), exactly as WRF.

## Result 2 — hypothesis CONFIRMED that the term is missing, with an HONEST CORRECTION

The single-step WRF oracle shows the MF arrays are nonzero where our code produces
zero. But the per-step lowest-level effect is tiny and **`s_awqv(kts)=0` always**:
the MF never directly sources qv at the lowest model level in one step -- it
reshapes the PBL above. fp32 WRF quantizes `Dqv` at ~3e-9, too coarse to resolve
the cumulative trend, so the characterization was done in fp64 JAX
(`integration_mf_vs_ed.json`, 2-hour single-column experiment):

- **The MF ventilates moisture UPWARD**: dries the lower PBL (k=0-6), moistens the
  upper PBL / entrainment zone (k=7-8). Column water is conserved.
- **Frozen-forcing**: MF makes the near-surface **drier** by -4.4e-4 kg/kg -- the
  OPPOSITE sign of the naive "missing-MF => our-run-too-dry" story.
- **Responsive-QFX closure** (surface vapor flux = ch*wspd*(qsfc-qv0)): MF **raises
  evaporation by +7-11%** (it lowers near-surface RH -> larger surface-air vapor
  gradient -> more QFX). **This matches the equiv-T2 signature**: GPU QFX ~= 60% of
  WRF, LH -8.7 W/m2.

**Verdict (honest):** the missing MF is **real** and **contributes** to the
QFX/LH suppression in the WRF-faithful direction. It is **NOT** the sole or even
dominant cause of the -1.74e-3 near-surface qv dry bias -- that bias is dominantly
coupled to the **land-side canopy vapor source** (EAH 1467 vs 1655 Pa, per the L2
Noah-MP / surface-layer lanes). The MF lever and the land-vapor lever stack.

## What landed

- `src/gpuwrf/physics/mynn_edmf.py` — fp64 JAX port of WRF `DMP_mf` (updraft
  init/area/excess, per-plume vertical scan with entrainment + `condensation_edmf`
  + `qsat_blend`, flux limiter, `s_aw`/`s_awqv`/`s_awqt`/`s_awqc`/`s_awthl`
  assembly). VERIFIED < 0.5% vs WRF.
- `src/gpuwrf/physics/mynn_pbl.py`:
  - `_diffusion_solve_with_mf` — WRF-faithful MF terms added to the implicit
    tridiagonal + explicit flux divergence (`:4326-4352`) + khdz MF stability
    floor (`:3990-3997`).
  - `_edmf_arrays_from_state` — builds MF arrays from the column state.
  - `step_mynn_pbl_column[_with_pblh]` gain `edmf` + `dx` static args.
    **Default `edmf=False`** => all 11 existing M5 MYNN tests stay bit-identical.
- `tests/test_mynn_edmf_oracle.py` — 2 tests pass (WRF-oracle gate + wiring +
  no-regression-when-off).

## Predeclared tolerances

- s_aw / s_awqv JAX-vs-WRF: rel max err <= 5% (PASS at 0.48%).
- Column-water conservation in the integration: ED and MF column integrals equal
  to 5 sig figs (19.6302) — PASS.

## What remains (handoff)

1. **Enable `edmf=True` in the operational coupler** (`physics_couplers.py`
   `mynn_adapter` — OTHER LANE's file, not edited here) and run the **GPU
   end-to-end d03 daytime remeasure** to confirm the QFX/LH/qair effect in the
   full coupled run. **This is the manager's GPU step.** Expect QFX/LH to rise
   toward WRF (+~7-11% from MF ventilation), partially closing the deficit; the
   residual near-surface dry bias should be chased on the land-vapor (EAH) lane.
2. Saturated-PBL plume condensation (qc>0, stratocu): this column was dry (qc=0),
   so `s_awqc` was verified =0 only; the condensation path (`condensation_edmf`)
   is ported and runs but is unexercised by a saturated oracle.
3. Skin temperature for the exact WRF superadiabatic guard is not carried in the
   standalone column (buoyancy-flux fallback used; equivalent for daytime land).
4. Downdraft/subsidence/detrainment intentionally omitted (zero in this config).

## Reproduce

```
# WRF Fortran oracle (needs wrfbuild conda gfortran + pristine WRF .o)
proofs/mynn_edmf/fortran_oracle/build_and_run.sh
# column extraction (needs canary_env netCDF4)
/home/enric/miniconda3/envs/canary_env/bin/python proofs/mynn_edmf/extract_column.py
proofs/mynn_edmf/emit_flat.py
# JAX verification + integration (CPU, fp64)
JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 python3 proofs/mynn_edmf/jax_oracle.py
JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 python3 proofs/mynn_edmf/integration_oracle.py
JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 PYTHONPATH=src taskset -c 0-3 python3 -m pytest tests/test_mynn_edmf_oracle.py -q
```
