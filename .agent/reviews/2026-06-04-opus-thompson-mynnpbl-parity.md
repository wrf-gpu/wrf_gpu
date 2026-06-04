# v0.9.0 — Thompson (mp=8) + MYNN-PBL (bl=5) isolated WRF-savepoint parity

Lane: `worker/opus/v090-thompson-mynnpbl-parity` (base `worker/opus/trunk-0.9.0` @ 7b7c26e)
Mode: CPU-only fp64 (JAX_PLATFORMS=cpu, JAX_ENABLE_X64=true), cores 0-3 only, no GPU.
Oracle: UNMODIFIED pristine WRF v4.7.1 (instrumented dumper `module_wrfgpu2_oracle.F`),
conda env wrfbuild. JAX-vs-WRF, never self-compare; predeclared fp64 tolerances.

## Objective

Close the isolated-WRF-savepoint VERIFICATION GAP for the two default operational
schemes previously only operationally-RMSE-validated:
- Thompson microphysics (`mp_physics=8`) — `src/gpuwrf/physics/thompson_column.py`
- MYNN PBL (`bl_pbl_physics=5`) — `src/gpuwrf/physics/mynn_pbl.py`

## WRF-faithful source provenance (sha256)

| module | path | sha256 |
|---|---|---|
| Thompson | `phys/module_mp_thompson.F` | `fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802` |
| MYNN-EDMF PBL | `phys/module_bl_mynnedmf.F` (= bl=5) | `6e4a7d5b35ce46f01591f2c1d58e545380d546e654b4a59ee1bcf99cfbce2d72` |

Note: WRF v4.7.1's `bl_pbl_physics=5` calls `module_bl_mynnedmf.F` (the task's
`module_bl_mynn.F` is the older v3 name; not present in v4.7.1).

## Oracle build

The pre-existing `/mnt/data/wrf_gpu2/physics_oracle/microphysics` oracle is
`itimestep=1` — a near-inactive step: ALL hydrometeors (qr/qi/qs/qg/ni/nr) are
identically zero in AND out; only a trace qc (~3.6e-8) evaporates. It does NOT
exercise warm-rain or ice/snow/graupel, so it cannot prove Thompson savepoint
parity per the contract. (The prior `analytic-thompson-column-v1` fixture is also
not a WRF oracle.)

The same `itimestep=1` is ALSO wrong for MYNN-PBL: `qke_in = 0` everywhere (cold
start). WRF MYNN re-initializes qke internally on a cold step
(`module_bl_mynnedmf.F90:620-691`: `IF MAXVAL(qke)<2e-4 -> INITIALIZE_QKE=.TRUE.`,
sets `qke=5*ust*max(...)` then calls `mym_initialize`), producing full diffusivities
within step 1. The JAX kernel takes `qke_in=0` at face value -> near-zero km/kh.
This is a cold-start path, NOT the operational mixing the kernel ports.

=> I re-ran the SAME instrumented WRF (oracle hooks already compiled into
`main/wrf.exe`) from the SAME real IC/BC (`20260428_18z_l3_24h`) for 6 h and captured
ONE late step (`WRFGPU2_ORACLE_STEP=1000` ~= model time 23z) where precipitation is
active and the PBL TKE is spun up, into `/mnt/data/wrf_gpu2/physics_oracle_v090`.
WRF pinned to cores 0-3 (cores 4-31 = live backfill, untouched).
Build script: `proofs/v090/oracle_build/run_thompson_active_oracle.sh`;
manifest: `proofs/v090/oracle_build/build_manifest.py` (verified to reproduce the
existing manifest exactly).

## Results

Oracle captured at **itimestep=1000** (model time 23z, 5 h into the forecast), grid 1
(91×44×57 interior = 5187 columns × 44 levels), dt=18 s. WRF run = SUCCESS COMPLETE.
The 20260428 case is synoptically quiet CONUS: warm-rain develops by 19z; ice/snow stay
trace; graupel never forms. Proof JSONs: `proofs/v090/thompson_savepoint_parity.json`,
`proofs/v090/mynn_pbl_savepoint_parity.json`.

### 1. Thompson (mp=8) — **FAIL** (localized, small-magnitude warm-rain divergence)

Hydrometeor activity at this step (out max): qc=2.24e-4, qr=4.61e-6, Nr=4.68,
qi=qs=qg=Ni=0 (ice/snow trace in, evaporated out; graupel never forms in this case).
So warm-rain (qc/qr/Nr) is GENUINELY exercised; ice/snow/graupel are NOT
significantly active in this synoptic case (an honest coverage limit, not a measured
ice/snow/graupel failure).

Per-field (frozen Phase-B transcription ladder, moist mask n=228228; fp64):

| field | pass | max abs | rel on active (median / p95 / max) | note |
|---|---|---|---|---|
| qv | PASS | 7.1e-9 | 1.1e-6 | exact |
| qc | FAIL | 5.4e-9 | 2.1e-5 / 1.4e-4 / 5.8e-4 | tiny (fails abs_tol=1e-9 only) |
| qr | FAIL | 7.5e-8 | 4.6e-3 / 2.0e-2 / 5.9e-2 | ~0.5-6 % warm-rain |
| nr (Nr) | FAIL | 0.40 | 1.3e-5 / 1.1e-1 / 7.6e-1 | rain NUMBER: long tail to ~10-76 % |
| qi/qs/qg/ni | PASS | ≤3e-11 | — | inactive/trace |
| th | FAIL | 4.3e-5 | 1.5e-7 | fp32-storage-precision artifact (B1) |
| water+precip closure | **PASS** | rel 3.0e-7 | — | total water conserved; surface precip 6.608e-4 vs WRF 6.608e-4 (4 sig figs) |

**Localization:** divergence is confined to the **warm-rain process rates** —
autoconversion qc→qr, rain accretion, and (dominantly) rain-number Nr evolution.
Mass and surface precip are near-exact; Nr has the largest relative spread.
Consistent with the documented B1 scope limits (`proofs/b1/coefficient_audit.md` §4:
unported cross-species collection tables; single-mode snow fall speed; no
cloud-water sedimentation). **Likely cause:** small transcription differences in the
warm-rain source/sink + rain-number tendency, exceeding the frozen ~1e-6 rel /
1e-3-number band but physically small (<~2 % mass, precip near-exact).
**Verdict: NOT savepoint-faithful to the frozen band on the warm-rain path; real,
small, localized divergence. Operationally close (water+precip conserved/near-exact).**

### 2. MYNN-PBL (bl=5) — **FAIL** (localized to the mixing-length closure)

The pre-existing itimestep=1 oracle was the WRONG comparison (cold-start qke=0 →
WRF re-inits qke internally; JAX km/kh ~5 orders too small — a cold-start path
artifact, see above). At the spun-up step (qke_in ∈ [1e-5, 1.31], mean 0.047) the
picture is completely different and physically meaningful:

| field | pass | WRF range | JAX range | note |
|---|---|---|---|---|
| qke (2·TKE) | ~ | [1e-5, 1.304] | [1e-5, 1.295] | **matches** (156/34575 viol) — TKE prognosis correct |
| pblh | ~ | [102, 857] | [85, 857] | **matches** (median 687.8 vs 688.4; 18/5487 viol) |
| exch_m (Km) | FAIL | [0, 86.5] | [0, 45.9] | right regime, peak ~½ WRF |
| exch_h (Kh) | FAIL | [0, 137] | [0, 56.5] | right regime, JAX/WRF median ratio ~0.33-0.48 |
| rublten/rvblten/rthblten/rqvblten | FAIL | (overlap) | (overlap) | same order, cell-wise > 10 % band |

**Localization (decisive):** PBLH matches and qke matches, so the inputs are right.
The divergence traces to the **master mixing length `el_pbl`**: JAX median is **0.37×
WRF** (JAX el systematically too short). Since Kh ∝ el·√qke·Sh and qke matches,
the diffusivity deficit (Kh ratio ~0.33) is dominantly the short mixing length,
which then propagates to the U/V/θ/qv tendencies. **Likely cause:** the JAX
`_mym_length_option2` master-length-scale blend (`el_t`/surface-layer/buoyancy
length combination, `mynn_pbl.py`) does not reproduce WRF `mym_length`
(`module_bl_mynnedmf.F90`) to savepoint precision. EDMF mass-flux on/off makes no
difference at this weakly-convective 23z step (plumes mostly inactive).
**Verdict: NOT savepoint-faithful; real, localized divergence in the mixing-length
closure. The kernel IS in the correct physical regime (qke + PBLH match, diffusivities
& tendencies right order of magnitude, ~½ WRF peak).**

## Scope-matrix decision

Neither scheme passes strict isolated-WRF-savepoint parity, so **neither is upgraded**
from "operational-RMSE-validated only" to "savepoint-proven." The honest labels stand.
Both divergences are real (not harness/units bugs — water/precip closure and
qke/PBLH matches rule those out) and cleanly localized for follow-up.

## Honest risk / caveats

- The 20260428 case never forms graupel and only trace ice/snow, so the Thompson
  ice/snow/graupel paths are UNTESTED by this oracle (warm-rain + Nr are tested). A
  convective/winter case would be needed to bind those. This is a coverage gap, not a
  measured failure.
- MYNN-PBL tested at one (weakly-convective, evening) spun-up step. A daytime
  strongly-convective step would stress the mixing length harder and exercise EDMF.
- Surface kinematic fluxes for MYNN were reconstructed from the oracle's own
  hfx/qfx/ust exactly as WRF BL_MYNN derives them (flt=hfx/(ρ·cpm), flq=qfx/ρ); the
  qke/PBLH match validates this reconstruction.

## Reproduce

```
# oracle build (instrumented pristine WRF, cores 0-3)
WRFGPU2_ORACLE_STEP=1000 RUN_HOURS=6 bash proofs/v090/oracle_build/run_thompson_active_oracle.sh
python3 proofs/v090/oracle_build/build_manifest.py /mnt/data/wrf_gpu2/physics_oracle_v090/microphysics --scheme thompson
python3 proofs/v090/oracle_build/build_manifest.py /mnt/data/wrf_gpu2/physics_oracle_v090/surface_mynn --scheme mynnedmf
# parity (CPU fp64, cores 0-3)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 \
  python3 proofs/v090/thompson_savepoint_parity.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 \
  python3 proofs/v090/mynn_pbl_savepoint_parity.py --oracle-dir /mnt/data/wrf_gpu2/physics_oracle_v090/surface_mynn
```
