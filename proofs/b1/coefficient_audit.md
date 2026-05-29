# B1 Thompson coefficient audit (mp_physics=8)

WRF source-of-truth: `/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F`
(pristine WRF v4, gfortran serial). Config for the oracle: **mp_physics=8**, i.e.
`is_aerosol_aware = .FALSE.` (init line 468), `is_hail_aware = .FALSE.` (line 63),
so `nc = Nt_c` (fixed cloud droplet number, line 1842/3487) and graupel uses the
fixed `idx_bg1` density slot (line 1950). Comment at lines 18-20 confirms mp=8 is
the one-moment-cloud variant.

## 1. Prior cross-AI audit findings — RE-VERIFIED as already-correct

The two coefficient bugs a prior audit flagged are ALREADY fixed in the kernel at
the Gate-1 base:

| Coefficient | WRF def (file:line) | Value | Kernel | Status |
|---|---|---|---|---|
| `cie(2) = bm_i+mu_i+1` (ice slope clip) | `module_mp_thompson.F:688` | 4.0 | `thompson_constants.CIE2 = BM_I+MU_I+1.0 = 4.0`, used in `_finish` slope-clip (`thompson_column.py:371-372`) | CORRECT (not `6.0`) |
| graupel `cgg(11,·)` / `cge(11,·)` (sublimation/melt) | `cge(11,m)=0.5*(bv_g(m)+5+2*mu_g)` `:763`; used `:2697,2700,2808,2811` | `CGE11=2.8204808`, `CGG11=Γ(CGE11)` | `thompson_constants.CGE11/CGG11`; `T2_SUBL_QG=0.28*Sc3*sqrt(av_g)*CGG11`, `T2_MELT_QG` likewise — uses graupel (idx 11) values, NOT rain `crg(11)` | CORRECT |

Note `cig(2)=WGAMMA(cie(2))=Γ(4)=6.0` (`:695`), so the literal `6.0` in the
ice-slope `lami` formula (`_ice_distribution`, `_finish`, `_fall_speeds`) is the
correct `cig(2)`, not an error.

## 2. New sedimentation fall-speed coefficients — transcribed + verified

Added to `thompson_constants.py` for the sedimentation port; verified numerically
(`proofs/b1` coefficient check, all OK):

| Const | WRF (file:line) | mp=8 value |
|---|---|---|
| `AV_R` | `:143` | 4854.0 |
| `BV_R` | `:144` | 1.0 |
| `FV_R` | `:145` | 195.0 |
| `AV_I` | `:161` | 1493.9 |
| `BV_I` | `:162` | 1.0 |
| `AV_S` | `:147` (av_s) | 40.0 |
| `BV_S` | `:148` (bv_s) | 0.55 |
| `AV_G_MP8` | `am_g`/`av_g(idx_bg1)` `:155-156` | 143.204224 |
| `BV_G_MP8` | `bv_g(idx_bg1)` `:156` | 0.640961647 |
| `CRE3=bm_r+mu_r+1` → `CRG3=Γ(4)` | `:707,719` | 4 → 6 |
| `CRE6=bm_r+mu_r+bv_r+1` → `CRG6=Γ(5)` | `:710,719` | 5 → 24 |
| `CRE7=bm_r/2+mu_r+bv_r+1` → `CRG7=Γ(3.5)` | `:711,719` | 3.5 → 3.32335 |
| `CRE12=bm_r/2+mu_r+1` → `CRG12=Γ(2.5)` | `:716,719` | 2.5 → 1.32934 |
| `CIE3=bm_i+mu_i+bv_i+1` → `CIG3=Γ(5)` | `:689,696` | 5 → 24 |
| `CIE6=bm_i/2+mu_i+bv_i+1` → `CIG6=Γ(3.5)` | `:692,699` | 3.5 → 3.32335 |
| `CIE7=bm_i/2+mu_i+1` → `CIG7=Γ(2.5)` | `:693,700` | 2.5 → 1.32934 |

Rain mass fall speed (`module_mp_thompson.F:3618-3619`):
`vtr = rhof*av_r*crg(6)*org3 * lamr^cre(3) * (lamr+fv_r)^(-cre(6))` — transcribed
verbatim in `_fall_speeds`. Rain number speed (`:3626-3627`):
`vtnr = rhof*av_r*crg(7)/crg(12) * lamr^cre(12) * (lamr+fv_r)^(-cre(7))`.
Ice mass/number speeds (`:3681,3686`): `vti = rhof*av_i*cig(3)*oig2*ilami^bv_i`,
`vtni = rhof*av_i*cig(6)/cig(7)*ilami^bv_i`.

## 3. Empty-layer fall-speed inheritance

WRF fills layers without a hydrometeor with the speed from the layer above
(`vtrk(k)=vtrk(k+1)`, `:3630-3631,3689-3690,3729,3769-3770`) so a falling blob
keeps moving. Replicated in `_fill_down` (top→surface scan). Without this the
sediment flux stalls one layer below the source (verified bug → fix).

## 4. Honest scope limits (see status.json)

- **Cross-species collection (rain↔snow, rain↔graupel) is NOT ported**: the 15
  `tmr/tcr/tcs/tms/tnr_racs/sacr/racg/gacr` lookup tables (`qr_acr_qs`/`qr_acr_qg`,
  `:4091-4591`) are NOT in the extracted `thompson-tables-v1.npz` asset. Faithful
  re-creation requires extending `scripts/extract_thompson_tables.py` to dump them
  from a WRF build (nvfortran) — flagged, not fabricated.
- Snow fall speed uses the single-mode `av_s*Ds^bv_s` closure on the Field-moment
  mean diameter, not the full two-gamma `vts` integral (`:3711-3727`) — a faithful
  mp=8 approximation valid when the racs riming-boost is inactive.
- Cloud-water sedimentation (small, lowest 500 m only, `:3646-3667,3824-3837`) is
  not advected.
