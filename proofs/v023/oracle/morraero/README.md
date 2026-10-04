# Aerosol-aware Morrison 2-moment (WRF mp_physics=40) single-column oracle

WRF-Fortran gold oracle for `phys/module_mp_morr_two_moment_aero.F`
(aerosol-aware Morrison; WRF namelist `mp_physics=40`), built VERBATIM from
the pristine tree `<USER_HOME>/src/wrf_pristine/WRF` — the scheme source is
never edited (sha256 provenance in the savepoint directory's
`wrf_source_checksums.txt`). Modeled 1:1 on the v0.6.0 base-Morrison oracle
(`proofs/v060/oracle/`): same 6-case sounding generator (copied as-is), same
build pattern (fp32 default + fp64 via `-fdefault-real-8`), same dump format
(`ES23.15` key=value -> JSON).

## Files

- `morraero_oracle_driver.f90` — driver (column builder + dump; never touches physics)
- `morraero_stub_modules.f90`  — `module_wrf_error` stub (identical to v0.6.0)
- `morraero_build_and_run.sh`  — build+run both modes (`fp32` default, `fp64` arg);
  CPU-pinned `taskset -c 4-31`, `OMP_NUM_THREADS=4`, conda env `wrfbuild`
- `dump_to_json.py`            — dump -> JSON converter (fail-closed: finiteness +
  required-column completeness; a truncated dump can never become a savepoint)
- `verify_savepoints.py`       — honest acceptance gate (see Verification)

Savepoints: `proofs/v022/f2_oracles/morrison_aero/morr_aero_case_{1..6}.json`
(fp32) and `morr_aero_fp64_case_{1..6}.json` (fp64), schema
`wrf-v023-f2-morraero-column-savepoint-v1`, plus `wrf_source_checksums.txt`
and `build_manifest.txt`.

## Entry points and call pattern (mp_physics=40, non-chem)

- Init: `MORR_TWO_MOMENT_INIT_AERO(morr_rimed_ice=0)` (graupel mode, IHAIL=0 —
  same choice as the base oracle).
- Tick: `MP_MORR_TWO_MOMENT_AERO(...)`, invoked exactly as WRF's
  `phys/module_microphysics_driver.F` `CASE (MORR_TM_AERO)` does with
  `WRF_CHEM` **undefined** (`-cpp`, macro not set — chem branches compiled out,
  matching the default WRF build). All-keyword call; the WRF-Chem-only
  OPTIONAL arguments (`qndrop`, `wetscav_on`, `rainprod`, `evapprod`,
  `QLSINK`, `PRECR/PRECI/PRECS/PRECG`) are omitted and `F_QNDROP=.FALSE.` is
  passed, exactly the values WRF uses for mp_physics=40 without chem. These
  optionals are pure diagnostics guarded by `PRESENT()`; omitting them changes
  no prognostic result. `DIAGFLAG=.FALSE., DO_RADAR_REF=0` skip the radar
  reflectivity block (same as the base oracle).

## The aerosol operating point (chosen values + rationale)

`aercu_opt=2, aercu_fct=1.0, no_src_types_cu=10, PBL=1`

- `aercu_opt=2` is THE aerosol-aware mode: the wrapper then sets `INUM=0,
  iinum=0` (droplet number NC becomes **prognostic**), `IACT=4`
  (Abdul-Razzak & Ghan 2000 activation from the prescribed aerosol),
  `INUC=2` (Liu & Penner 2005 aerosol ice nucleation). With any other value
  the scheme degenerates to constant-droplet base-Morrison behavior
  (NDCNST=250 cm-3) and the aerosol arrays are dead — that would be a
  synthetic happy-path, so 2 is the only honest choice.
- `no_src_types_cu` MUST be 10: the module hardwires `naer_cu=10` and the
  micro routine copies `maerosol/naer(K,1:10)`; any smaller value is
  out-of-bounds. (WRF requires 10 in the namelist for aercu_opt>0 too.)
- `aercu_fct=1.0` — WRF Registry default (aerosol multiplication factor).
- `PBL=1` — the dummy argument is declared but never referenced by the
  scheme; any value works. 1 = "a PBL scheme exists".

### Prescribed aerosol AEROCU (units ug/m3, WRF Registry `aerocu` order)

| idx | species (Registry)        | surface value (ug/m3) |
|-----|---------------------------|-----------------------|
| 1-4 | dust1..dust4              | 0.50 / 0.30 / 0.15 / 0.05 |
| 5   | sea salt                  | 0.50 |
| 6   | sulfate                   | 1.50 |
| 7   | BC hydrophobic            | 0.05 |
| 8   | BC hydrophilic            | 0.05 |
| 9   | OC hydrophobic            | 0.40 |
| 10  | OC hydrophilic            | 0.40 |

Vertical profile: `surface * max(exp(-z/2000 m), 0.005)` — a standard
continental-background scale-height decay with a free-troposphere floor so
the Liu-Penner ice-nucleation thresholds (`so4_num>=1e-10`,
`soot+dust>=1e-10` cm-3) stay reachable aloft. Per-case loading factor:
x1.0 default, x2.0 case 4 (polluted convective core), x0.5 case 3 (cold
clean airmass), x0.2 case 6 (clean condensation column) — exercises the
aerosol sensitivity across savepoints. Resulting activated droplet numbers
are physically sensible: ~185 cm-3 (clean case 6), ~440-560 cm-3
(background), ~870 cm-3 (polluted case 4); 0 in case 5 (no liquid cloud).
Sulfate 1.5 ug/m3 maps through the scheme's own mass-to-number relation
(`5.64259e13 * m^0.58`) to ~430 CCN cm-3 — typical continental.

The wrapper maps AEROCU -> internal modes as: mode1=SULFATE(idx6),
mode2=SEASALT(idx5), mode3..6=DUST1..4(idx1..4, x1.44),
mode7=OCPHO(idx9, x1.54), mode8=BCPHO(idx7, x1.37),
mode9=OCPHI(idx10, x1.25), mode10=BCPHI(idx8, x1.37). A JAX porter must
reproduce this index shuffle + the mass multipliers exactly.

### Droplet number NC (prognostic input, #/kg)

Seeded consistent with QC assuming a 10 um mean-radius droplet
(`NC = QC / 4.18879e-12 kg`) — gives ~360-430 cm-3 in-cloud, matching the
continental CCN levels above; 0 outside cloud. NC is INOUT: the scheme
activates/depletes it and writes it back (`NC(i,k,j)=nc1d(k)` in the
non-chem branch).

### Eddy diffusivity KZH (m2/s)

`KZH(z) = 1 + 59*exp(-((z-600)/500)^2)` — a YSU-like convective-BL `exch_h`
profile peaking at 60 m2/s near 600 m. The scheme derives its sub-grid
vertical-velocity st.dev. as `WVAR = KZH(i,K+1,j)/20`, clamped to
[0.1, 50] m/s (base Morrison uses a constant 0.5 m/s instead). WVAR feeds
both droplet activation (`DUM = W + WVAR`) and ice nucleation, and is
exported in WACT.

## Deviations from the base-Morrison oracle a JAX porter MUST know

1. **kme = kte+1 memory layout.** The wrapper reads `KZH(i,K+1,j)` for
   K=kts..kte. WRF always has `kme=kde=kte+1`; the base oracle used kme=kte
   (fine there — nothing indexes k+1). Here all 3D arrays carry one padding
   level (k=41) that the scheme never touches except KZH. Savepoints dump
   k=1..40 for all fields, except `KZH_IN` which is k=1..41.
2. **NC is a 12th prognostic column** (in+out); `INUM=0/iinum=0` disables
   the NDCNST constant-droplet path. Note the NC upper bound
   `NC<=(NANEW1+NANEW2)/RHO` applies only to IACT=2, NOT to IACT=4.
3. **Activation** = `mdm_prescribed_activate` (AR&G 2000, 10 modes,
   module-level per-mode constants from `dryrad_aer/density_aer/hygro_aer/
   dispersion_aer/num_to_mass_aer` DATA tables), NOT the base scheme's
   2-mode IACT=2 path.
4. **Ice nucleation** = `INUC=2` -> `mdm_prescribed_nucleati` (Liu & Penner
   2005: sulfate homogeneous freezing = mode 1 number, soot = mode 10,
   dust = modes 3-6, + Meyers mixed-phase), NOT the Cooper curve
   (INUC=0) of the base oracle. Guarded by the same
   `(QVQVS>=0.999 & T<=265.15) or QVQVSI>=1.08` trigger.
5. **Init constants differ from base Morrison:** `DCS=350e-6` (base:
   125e-6) — cascades into LAMMINI, CONS21, CONS22 and the ice->snow
   autoconversion threshold; plus the whole `*_pamdm` activation-constant
   block computed in `MORR_TWO_MOMENT_INIT_AERO`.
6. **Extra outputs** (INOUT, set because aercu_opt>0): EFCG/EFIG/EFSG
   (clamped effective radii for RRTMG, um), WACT (=WVAR+W), CCN1..7_GS
   (diagnostic CCN at 0.02/0.05/0.1/0.2/0.3/0.5/1.0 % supersaturation,
   #/m3, pure functions of naer + init-time `ccnfact_pamdm`).
7. **Inert-here extras:** NR_CU/QR_CU/NS_CU/QS_CU/CU_UAF/mskf_refl_10cm only
   matter inside the `diagflag` radar branch (skipped; all zeros here).
   qrcuten/qscuten/qicuten zeros as in the base oracle.
8. The `#if (WRF_CHEM == 1)` guards must stay compiled OUT (extra
   rainprod/evapprod arguments would otherwise appear mid-signature in
   `MORR_TWO_MOMENT_MICRO`).

## Build / run commands

```bash
bash proofs/v023/oracle/morraero/morraero_build_and_run.sh fp32
bash proofs/v023/oracle/morraero/morraero_build_and_run.sh fp64
taskset -c 4-31 python3 proofs/v023/oracle/morraero/verify_savepoints.py
```

## Verification results (2026-07-02, gfortran 14.3.0, VERDICT: PASS)

- All 12 savepoints finite (fail-closed at conversion + re-checked).
- Nontrivial physics in every case; RAINNCV > 0 in cases 1/2/4 (and 3/5/6):

| case | RAINNCV fp32 (mm/step) | RAINNCV fp64 (mm/step) |
|------|------------------------|------------------------|
| 1    | 1.793823577e-02        | 1.793823104e-02        |
| 2    | 5.255471915e-02        | 5.031818166e-02        |
| 3    | 2.134609735e-03        | 2.134609847e-03        |
| 4    | 6.398048252e-02        | 6.022270725e-02        |
| 5    | 5.013560411e-03        | 5.013546023e-03        |
| 6    | 1.106159561e-05        | 1.106105248e-05        |

- Aerosol path proven active: NC evolves with activation scaling with the
  prescribed loading; CCN diagnostics nonzero; outputs genuinely differ
  from the base-Morrison savepoints (dTH up to 0.31 K in case 4 — prognostic
  NC + aerosol ice nucleation at work), while untouched code paths produce
  bit-identical values to base (e.g. the case-2 QG flip value
  6.594518254e-06 appears identically in both oracles).
- fp32-vs-fp64: smooth cases (1,3,5,6) agree to <=1e-6 kg/kg / <=5e-3 K.
  Mixed-phase cases (2,4) show single-step process on/off threshold flips at
  individual levels (max |dQ| ~1e-4, RAINNCV rel 4.3%/5.9%). This is the
  reference scheme's OWN fp32 behavior: the accepted v0.6.0 base-Morrison
  gold pair diverges by the same amounts in the same cases (4.26%/5.85%);
  measured aero/base divergence ratios are 0.0-1.5 per field. The verifier
  enforces smooth bands OR <=2x the base-pair anchor, per case per field.
- Known reference artifacts (present bit-identically in the v0.6.0 base gold
  pair, NOT oracle defects): case 4 fp32 GRAUPELNCV = -9.219831e-11 mm
  (fp cancellation remnant, physically zero); EFIG/EFSG categorical values
  at trace-mass detection-flip levels.
