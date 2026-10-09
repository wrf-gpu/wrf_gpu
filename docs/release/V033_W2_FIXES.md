# v0.3.3 W2 fixes — source-bound preparation

**Status: source/defaults reviewed; This source-dated candidate was held on external gates.**
Reviewed candidate `e83ec1fa7aca0b839357814ead52cb33bfdb8ea1` has `src/` tree
`fb50726fee1e701dbc9e874cd90bbcaed99aee86`, identical to measured `471efc475`.
Eight new defaults extend the previous 95 to 103; the resolved GPUWRF set
contains 101 controls. This is source/config acceptance, not release acceptance.
The 69d8a9a7d record remains pre-W2 evidence.

| Cause or uncovered difference | Original WRF source | W2 key |
|---|---|---|
| Urban soil-table values were retained instead of WRF's urban overrides. Too-small volumetric heat capacity increases the soil's diurnal swing and urban-night cooling; the soil solver itself matches the pristine REAL4 chain with shared coefficients. | `module_sf_noahmpdrv.F`, `TRANSFER_MP_PARAMETERS` :1694–1704: SMCMAX=.45, SMCREF=.42, SMCWLT=.40, SMCDRY=.40, CSOIL=3e6; derive FRZX after the override. | `GPUWRF_NOAH_URBAN_SOIL_PARAMS` |
| Slope-normal SWNORM was labelled SWDOWN in history. WRF restores horizontal SWDOWN after the land call and keeps SWNORM separate; this is an output convention change, not a change in radiation forcing. | Executable `module_surface_driver.F` :4590–4604; corresponding pristine source :4563–4581, slope-output restore/swap. | `GPUWRF_HISTORY_SWDOWN_HORIZONTAL` |
| MYNN TKE prediction omits the mass-flux `s_aw` stability floor; variance prediction also differs at the top-face extent. WRF applies the floor even when advective EDMF-TKE terms are disabled. | Executable `module_bl_mynnedmf.F90`, `mym_predict` :3059–3065; active interfaces kts+1…kte−1. | `GPUWRF_MYNN_PREDICT_SAW_FLOOR` |
| Canopy water capacity used total LAI/SAI instead of the exposed vegetation areas when snow buries short vegetation. | `module_sf_noahmplsm.F` :6116–6118 passes ELAI/ESAI to CANWATER; burial reduces those exposed areas. | `GPUWRF_NOAH_CANWATER_ELAI` |
| The land Q2 cap was applied before Noah and could be overwritten afterward. WRF caps the post-LSM result with the surface-call QV operand. | Executable `module_surface_driver.F` :4580–4581; pristine :4555–4556: `XLAND < 1.5`, then `min(Q2,1.05*QV_CURR)`. **Land only; water remains untouched.** | `GPUWRF_LAND_Q2_CAP_POSTLSM` |
| Remaining MYNN REAL primitive and interface-footprint differences. | Original DMP REAL `grav/tref` and mean-solve `kts+1:kte−1` stability floor; source packet `e2af8d5b2` (reviewed original `778aeee64`). | `GPUWRF_MYNN_SOURCE_PARITY2` |
| MYNN surface-layer psi lookup tables and stable-heat reciprocal were formed through the historical DOUBLE path. | Original MYNN `psi_init(psi_opt=0)` REAL tables, 4,004 nodes and 8,000 interpolation checks; REAL `1./1.1`. Source packet `7c96fdcef` (reviewed original `36fe022a0`). | `GPUWRF_MYNN_SFC_PSI_REAL` |
| Land QSFC history/coupling exported ENERGY's local vegetation blend instead of the final WRF INOUT writer. | `module_sf_noahmplsm.F` VEGE_FLUX :4073 then BARE_FLUX :4435; `module_sf_noahmpdrv.F` :1244 exports the last-written bare QSFC. ENERGY Q1 :645 is local. | `GPUWRF_NOAH_QSFC_WRF` |

Line numbers distinguish the executable build from the pristine source copy;
the named subroutine and operand sequence are the binding references. The
latest land-only Q2 correction supersedes the briefly proposed global scope.
All eight keys above are ON in the reviewed candidate defaults. Explicit
`=0` remains an opt-out, not the release recipe. Frozen A6 proof
`fb5a64e0a5b2493` establishes that measured W2bR's explicit controls equal
the candidate's fresh-process defaults exactly (101 keys, zero differences).
Its executed commit remains `5a2adb6e1`; only `_fast_defaults.py` differs in
the model source from `471efc475`. This identity proof must accompany the
frames; do not replace the executed hash with the candidate hash.

## Why removing an error exposed the urban cold term

The original RC had a near-surface k0 **warm PBL bias**. The WRF-faithful MYNN
set removes that warm term. At the original WRF τ48 land-night state, the
shipped PBL tendency is not a sufficient cold driver; a reverted MYNN group
restores warmth. That compensating warmth had masked the older cold
surface/urban-soil error. A bisect group that restores the old warm bias is
therefore a compensator, not the soil cause. The fix track retains the
faithful PBL correction and repairs urban soil coefficients; it does not
trade one bias against another. A smaller non-urban night residual remains
under independent attribution and is not automatically closed by the urban
fix.

## A/WD10 was not replicated

Under the pre-registered replicate rule `45f34e04ab5c`, the perturbed-IC GPU
member uses the same IC bytes as original CPU P0227. B/T2 is REPLICATED
(M−P +0.0147 K versus floor 0.0109 K; mean of the two GPU departures +0.0186 K),
and B/RH2* is REPLICATED (+0.170 versus 0.092). A/WD10 is **NOT REPLICATED**:
M−P −2.25° versus floor 7.59°, while the original candidate was +8.09°.
That wind-direction trigger is disclosed as one realization and remains
a roadmap/extended-verification item. This diagnostic does not erase the
original R32-60 DAMAGE verdict or assert a measured wind-direction fix.

## Rebase and refresh rule

When the manager freezes W2, place one docs/version commit on that exact
reviewed candidate. Preserve 69d8a9a7d measurements as explicitly dated
pre-W2 evidence; regenerate the release identity numbers and plots from the
new immutable W2 validation runs against the same original CPU references.
Do not transfer 69d8 PASS labels, source hashes, trace judgements or benchmark
numbers to a new tree without its receipts. The shipped Swiss 24-hour case
and its README must be refreshed again on the released numerical source.
The new [W2 galleries](evidence/v033/w2/index.html) contain the measured
W2bR/W0408/W0115 histories. W0115's 32 annex-real wind rows and station
WS10 OPEN result remain binding. Y0 is a pinned replay; distinct ZR/ZP
attribution, W2 Monica and
the W2 shipped-Swiss rerun are pending; old passes are not inherited.
