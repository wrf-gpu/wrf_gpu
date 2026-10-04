# NSSL 2-moment (WRF `mp_physics=18`) single-column oracle

Gold Fortran oracle for the v0.23 F2 JAX port of the NSSL 2-moment
microphysics. It compiles the UNMODIFIED WRF source
`phys/module_mp_nssl_2mom.F` (25,163 lines; sha256
`29f42e76fbeca027a3ffb2ac805b148aaa1425c7f99ed1d8f19a90ebc1d015f8`, copied
verbatim from `<USER_HOME>/src/wrf_pristine/WRF`) and drives
`nssl_2mom_init` + `nssl_2mom_driver` exactly as WRF does for the DEFAULT
`mp_physics=18` configuration. No physics is stubbed or edited; the only
project Fortran is the column driver and two no-op shims
(`wrf_dm_on_monitor` logging shim returning `.FALSE.`, `wrf_error_fatal`
abort shim). Modeled on `proofs/v060/oracle/morrison_build_and_run.sh`.

Savepoints (schema `wrf-v023-f2-nssl2mom-column-savepoint-v1`):
`proofs/v022/f2_oracles/nssl_2mom/nssl_case_{1..6}.json` (fp32) and
`nssl_fp64_case_{1..6}.json` (fp64), plus `wrf_source_checksums.txt`,
`build_manifest_{fp32,fp64}.txt`, `verification.txt`.

Run: `./nssl2mom_build_and_run.sh [fp32|fp64]` (CPU-pinned `taskset -c 4-31`,
conda env `wrfbuild` gfortran 14.3, `OMP_NUM_THREADS=4`).

## 1. The default mp_physics=18 configuration (with WRF citations)

### 1.1 Active fields (Registry packages)

`Registry/Registry.EM_COMMON`:

| line | package | condition | fields |
|---|---|---|---|
| 3037 | `nssl_2mom` | `mp_physics==18` | moist: `qv,qc,qr,qi,qs,qg` |
| 3055 | `nssl2mconc` | `nssl_2moment_on==1` | scalar: `qndrop,qnr,qni,qns,qng`; state: `re_cloud,re_ice,re_snow` |
| 3058 | `nssl_hail` | `nssl_hail_on==1` | moist: `qh`; scalar: `qnh` |
| 3060 | `nssl_ccn_opt` | `nssl_ccn_on==1` | scalar: `qnn` |
| 3062 | `nssl_hailvol` | `nssl_density_on==2` | scalar: `qvolg,qvolh` |
| 3056/3057 | `nssl3mg/nssl3m` | `nssl_3moment>=1` | `qzr,qzg[,qzh]` — **inactive** (default `nssl_3moment=0`, line 2426) |
| 3063/3064 | ssat packages | `nssl_ssat_output>=1` | **inactive** (default 0, line 2428) |

Units (Registry.EM_COMMON:519-546): all `qn*` are number **mixing ratios**
`# kg(-1)`; `qvolg/qvolh` are particle volume mixing ratios `m(3) kg(-1)`.

### 1.2 Default resolution of the `-1` namelist flags

`share/module_check_a_mundo.F`, for `mp_physics==18` (all defaults are `-1`
in the Registry, lines 2422-2427):

- `nssl_ccn_on = 1` (lines 3449-3452)
- `nssl_2moment_on = 1` (lines 3454-3457)
- `nssl_hail_on = 1` (because `2moment_on==1`; lines 3459-3464)
- `nssl_density_on = 2` = graupel **and** hail volume (because `hail_on==1`;
  lines 3467-3470)
- `nssl_3moment = 0` (untouched)

So the default field set is: `qv,qc,qr,qi,qs,qg,qh` (moist) +
`qndrop,qnr,qni,qns,qng,qnh,qnn,qvolg,qvolh` (scalar) + `re_cloud/ice/snow`.

### 1.3 The init call (`phys/module_physics_init.F:4632-4678`)

```
nssl_params(1..15) = (nssl_cccn=0.5e9, nssl_alphah=0., nssl_alphahl=1.,
   nssl_cnoh=4.e5, nssl_cnohl=4.e4, nssl_cnor=8.e5, nssl_cnos=3.e6,
   nssl_rho_qh=500., nssl_rho_qhl=900., nssl_rho_qs=100.,
   ipelec=0, isaund=12, 0, 0, 0)        ! Registry.EM_COMMON:2410-2419,
                                        ! registry.elec:41-42; elec off
CALL nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0,
   nssl_density_on=.true., nssl_hail_on=.true., nssl_ccn_on=.true.,
   nssl_icdx=6, nssl_icdxhl=6, ccn_is_ccna=0)
```

- `ipctmp=5` = 2-moment for all hydrometeors (`nssl_ipconc`, physics_init
  4659-4667; `ipconc=8` only for 3-moment).
- `mixphase=0` is a **dead argument** (declared module line 1303, never
  read; `mixedphase` stays `.false.`, module line 524).
- `ccn_is_ccna` is `intent(inout)`: init **returns 1** because the module
  default `irenuc=5` (line 356, droplet renucleation with predicted
  activated CCN) forces `turn_on_ccna` (lines 1496-1514). Confirmed by the
  oracle: the dumped `CCN_IS_CCNA` scalar is 1.
- `ccn_conc = nssl_cccn/1.225` (physics_init 4669-4670) is only the qnn
  boundary-condition constant; the module computes the same value itself as
  `qccn = ccn/rho00` (lines 907, 2121). Dumped as scalar `QCCN =
  4.08163265e8 #/kg`.
- Init also reads namelist group `nssl_mp_params` from `./namelist.input`
  (`status='old'`, module lines 1447-1462) — the file MUST exist; the
  oracle provides an empty one (read fails with `iostat/=0` and all module
  defaults are kept, identical to a WRF run without that group).

### 1.4 Key module defaults active in this configuration

(all `phys/module_mp_nssl_2mom.F`)

- `irenuc=5` (l.356): CCN activation with prognostic activated-CCN (CCNA).
- `icenucopt=1` (l.380): Meyers/Ferrier primary ice nucleation.
- `imurain=1` (l.578): rain is gamma-of-DIAMETER (`alphar=0`, l.778).
- `ihlcnh=-1 -> 3` for `ipconc==5` (l.557, 1423-1431): graupel->hail
  conversion via wet growth.
- Sedimentation: `itfall=0` first-order upwind (l.279), `infall=4` hybrid
  number-fallout correction, Mansell (2010) methods I+II (l.288),
  `isfall=2` for snow (l.283), `do_accurate_sedimentation=.false.` (l.284,
  fall speeds reused across substeps), `vtmaxsed=70` m/s cap (l.312).
- `icdx=6/icdxhl=6`: Milbrandt & Morrison (2013) density-based fall speeds
  -> `bx(lh)=0.6, ax(lh)=157.71; bx(lhl)=0.593, ax(lhl)=179.36`
  (l.1591-1626).
- Thresholds: `qxmin = 1e-13 (qc,qi,qs), 1e-12 (qr,qh,qhl)` for 2-moment
  (l.2226-2241); `cxmin=1e-8` (l.619).
- WRF_CHEM undefined -> `wrfchem_flag=0` (l.202-206); NMM_CORE undefined ->
  `invertccn=.false.` (l.259-264); WRF_ELEC undefined (l.1458).

### 1.5 Internal species layout (what the JAX port must reproduce)

With this config `nssl_2mom_init` produces (l.1518-1554, 1762-1802,
1876-1894): `lhab=8`, `na=18`:

| index | species | WRF field | driver arg |
|---|---|---|---|
| 1 `lt` | theta | th | TH |
| 2 `lv` | vapor | qv | QV |
| 3 `lc` | cloud | qc | QC |
| 4 `lr` | rain | qr | QR |
| 5 `li` | ice | qi | QI |
| 6 `ls` | snow | qs | QS |
| 7 `lh` | **graupel** | **qg** | **QH** |
| 8 `lhl` | **hail** | **qh** | **QHL** |
| 9 `lccn` | background CCN (constant `qccn`) | — (see below) | — |
| 10-15 `lnc,lnr,lni,lns,lnh,lnhl` | numbers | qndrop,qnr,qni,qns,qng,qnh | CCW,CRW,CCI,CSW,CHW,CHL |
| 16 `lvh` | graupel volume | qvolg | VHW |
| 17 `lvhl` | hail volume | qvolh | VHL |
| 18 `lccna` | activated CCN | **qnn** | CN |

**CCN subtlety (default config):** because `lccna>1` and no `cna` argument
is passed, the WRF `qnn` field is loaded into `lccna` (activated CCN) and
`lccn` is held at the constant background `qccn` (driver l.2874-2895,
copy-out l.3469-3475). At `itimestep==1` the driver **zeroes the CN field**
(cold start, l.2717-2738).

## 2. Driver call semantics (JAX porter notes)

Call mirrored from `phys/module_microphysics_driver.F:2237-2303`
(`CASE (NSSL_2MOM)`); the oracle passes the same arguments except
chem/3-moment/ssat optionals that are inactive by default (equivalent:
`ipconc=5` -> `lzr=lzh=lzhl=0`; `nssl_ssat_output=0`; `wrfchem_flag=0`).

- **Orientation**: all 3-D arrays are `(i,k,j)`, memory dims
  `ims:ime,kms:kme,jms:jme`, processed tile `its:ite,kts:kte,jts:jte`.
  Internally the driver loops over j and packs 2-D slabs
  `an(its:ite, 1, kts:kte, 1:na)` (l.2843, 2860-2944). k=1 is the surface.
- **Units in/out**: q in kg/kg; numbers in #/kg; volumes in m3/kg. On entry
  (first inner loop iteration) indices 9..18 are multiplied by dry-air
  density `dn` -> #/m3, m3/m3 (`denscale`, l.3094-3104) and divided back on
  exit (l.3441-3449). All microphysics internals work with #/m3.
- **Temperature**: `th` is potential temperature; `t0 = th*pii` (l.2950).
  The scheme updates `an(:,:,:,lt)` and writes theta back (l.3457).
  `eqtset=1` (l.212). `tt`/`is_theta_or_temp` are the CCPP alternative,
  unused in WRF.
- **`dn` (air density)** is USED (unlike Morrison, which recomputes rho):
  number scaling, sedimentation, precip binding. WRF passes
  `rho = rho_dry*(1+qv)` (`dyn_em/module_big_step_utilities_em.F:4856`);
  the oracle computes the equation-of-state equivalent `p/(R_d*Tv)`.
- **Per-slab order of operations** (the port must match):
  1. pack `an`; `t0=th*pii`, `t00=380/p`; precompute ice-nucleation rate
     `t7` per cell (icenucopt=1 Meyers/Ferrier, l.3006-3078);
  2. density-scale indices 9..18 (first loop iteration only);
  3. `itimestep==1`: `calcnfromq` (l.3115-3117) — derives missing number/
     volume moments from mass via single-moment intercepts (module
     l.5298-5657) and folds sub-`qxmin` species back to vapor;
  4. cumulus tendencies (`cu_used==1` only; oracle passes 0);
  5. `sediment1d` (l.3144-3146) then bind surface precip (l.3153-3206):
     `RAINNCV = dtp*dn(k=1)*(xfall_rain + (xfall_snow + xfall_graupel +
     xfall_hail)*1000/xdn0(lr))` in mm (liquid-equivalent, includes ALL
     species); `SNOWNCV`, `GRPLNCV` (graupel only when `HAILNC` is passed),
     `HAILNCV` analogous; `SR = (SNOW+HAIL+GRPL)/(RAIN+1e-12)`;
  6. `nssl_2mom_gs` — the main gather/scatter process-rate routine
     (l.3219-3236);
  7. `NUCOND` — droplet nucleation + cloud condensation/evaporation
     (saturation adjustment lives HERE, not in gs; l.3246-3255,
     `flag_qndrop=.false.` without chem);
  8. `smallvalues` cleanup (l.3258-3263);
  9. diagnostics when `makediag` (= `diagflag .or. itimestep==1`):
     dbz via `radardd02` (l.3319-3351), effective radii via
     `calc_eff_radius` (l.3356-3421, clamped 2.51-50 um cloud /
     10.01-125 um ice / 25-999 um snow);
  10. de-scale numbers, unpack `an` back to the 3-D fields.
- **Sedimentation structure** (`sediment1d`, l.4425-4859): column-by-column
  (ix loop), species loop `il=lc..lhab`; per species compute all-moment
  fall speeds (`ziegfall1d`), cap at 70 m/s, courant
  `vtmax = max_k(vt*dz^-1)`; substeps: `ndfall = 1` if `dtp*vtmax < 0.7`,
  else `1 + Int(dtp*vtmax + 0.301)` for `dtp <= 20` s, else
  `Max(2, Int(dtp*vtmax/0.7)+1)` (l.4650-4658). Fall speeds are computed
  ONCE and reused across substeps (`do_accurate_sedimentation=.false.`).
  Per substep: `fallout1d` (first-order upwind, surface flux accumulated
  into `xfall`) for mass, then volume (`lvh/lvhl`), then number with the
  `infall=4` correction: number falls with number-weighted Vt, then is
  bounded below by the min of the mass-weighted and Z-weighted refluxed
  values (Mansell 2010 I+II; l.4706-4826). Cloud droplets do not sediment
  (`linfall(lc)=0`, `ido(lc)=1` but Vt tiny; snow uses `isfall=2` method).
- **Time-splitting**: none beyond sedimentation substeps for a plain WRF
  call (`ntmul/ntcnt/lastloop` absent -> `loopmax=1`, l.2636-2643).
- **RAINNC/RAINNCV are non-optional**; `RAINNCV/SNOWNCV/GRPLNCV/HAILNCV`
  are zeroed by the driver each call (l.2958-2963); `RAINNC` accumulates.

## 3. Oracle column + seeding

6 cases copied VERBATIM from the Morrison oracle
(`proofs/v060/oracle/morrison_oracle_driver.f90`): same KX=40 stretched-z
grid to 16 km, same thermodynamics, same hydrometeor mass seeds with
Morrison `QG` -> NSSL graupel `QH`. `DT=60 s`, `itimestep=1`, single call.
Extra NSSL fields:

- `CCW = MIN(QC/cwmas9, qccn_bg)`; `cwmas9 = 1000*(pi/6)*(18e-6)^3 =
  3.0536e-12 kg` (mass of a 9-um-radius droplet — the scheme's own
  cold-start assumption, `calcnfromq` module l.5340) and
  `qccn_bg = 0.5e9/1.225 #/kg` (the background CCN cap).
- `CRW = QR/5e-9`, `CCI = QI/1e-10`, `CSW = QS/2e-8`, `CHW = QH/5e-8`
  (identical mean-particle-mass seeding to the Morrison oracle NR/NI/NS/NG).
- `VHW = QH/500` (graupel starts exactly at its reference density
  `nssl_rho_qh=500`).
- `QHL = CHL = VHL = 0` (hail has no Morrison analogue) and `CN = 0`
  (the driver zeroes it at `itimestep==1` anyway; see 1.5).

With these seeds `calcnfromq`'s derivation branches are inert except on
trace cells (numbers are seeded everywhere mass is significant); its
sub-threshold cleanup still acts — deterministic scheme behavior the port
must reproduce.

## 4. fp32 vs fp64 (`-fdefault-real-8`)

The module is **NOT internally double-precision**: all prognostic state and
the bulk of the process arithmetic use default `REAL` (fp32 in a canonical
WRF build). Explicit `DOUBLE PRECISION` appears only in selected internals
(init-time gamma/incomplete-gamma lookup tables, `gamma_dp*` functions,
some accumulators/intermediates such as `dp1`, `proctot`, mass-budget
sums). The fp64 build (`-fdefault-real-8 -fdefault-double-8`) therefore
promotes the default-`REAL` state and arithmetic to 64-bit while keeping
the already-double internals at 64-bit; the source is unmodified.

Measured (see `verification.txt`): on significant cells the two builds
agree to 1e-7..2e-4 relative — a genuine physical band. Two categorical
threshold flips (expected fp32 detection-floor dust, same class the
Morrison oracle documents):

- case 5, `CSW` k=2 (Fortran): a fully-sublimating snow cell; fp64 zeroes
  the species (`qs -> 0`), fp32 leaves residual `qs = 1.8e-12 kg/kg` with
  `csw = 35 #/kg` — physically zero mass either way.
- case 6, `QC` k=1-2: marginal condensation onset at RH=1.01; fp32
  condenses ~0 (`qc` stays 1.4e-6), fp64 condenses to 5.0e-6 kg/kg
  (`dTH = 0.009 K`). Parity harnesses should use a dual-reference band on
  such onset cells.

## 5. Verification result (2026-07-02)

- All 12 savepoints: every value finite (gate 1 PASS).
- Activity: TH/QV evolve in all cases/both modes; nonzero RAINNCV in cases
  1,2,4,5 (mm/step, fp32|fp64): c1 2.08759e-2|2.087591e-2,
  c2 5.530706e-2|5.530707e-2, c3 1.292581e-3|1.292581e-3 (+SNOWNCV
  1.293e-3), c4 5.836652e-2|5.836654e-2 (+SNOWNCV 7.86e-4, GRPLNCV
  1.61e-3), c5 1.414414e-2|1.414415e-2 (+SNOWNCV 4.85e-4), c6 0 (clean
  condensation-only case, correct). Gate 2 PASS.
- fp32-vs-fp64 physical band: PASS with the two documented categorical
  flips above (gate 3).

**Known coverage gap (honest):** hail (`QHL/CHL/VHL`, `HAILNCV`) remains
identically 0 in all 6 cases — the Morrison-identical seeds contain no
hail and one 60 s step produces no graupel->hail wet-growth conversion.
The hail CODE PATH is active (arrays packed, sedimentation loop includes
`lhl`), but hail-specific process rates are not numerically exercised.
A hail-seeded supplementary case should be added when the JAX port needs
hail-process parity evidence.
