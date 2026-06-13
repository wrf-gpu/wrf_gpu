# v0.15 RAINNC cold-collection: scope verdict

**Verdict: BOUNDED-PARTIAL** — the dominant missing rain sink (rain-collecting-snow
+ rain-collecting-graupel below 0 C) is now WRF-faithful and closes the
column-integrated rain→graupel conversion to 3%; per-cell distribution and rain
number refinements remain for 0.16.

## Diagnosis (WRF v4 module_mp_thompson.F vs port)

Switzerland d01 is a **January Alpine** case (2023-01-15, T down to 212 K).
RAINNC +5.08 mm surplus is PROVEN non-chaotic
(`proofs/v015/falsifier_rainnc_report.json`, WRF internal variability 0.057 mm).
The port retained supercooled rain that WRF converts to graupel aloft below 0 C.

Audit of the full WRF cold collection-collision set vs the port:

| WRF process | what it does | port status before | this sprint |
| --- | --- | --- | --- |
| Bigg rain freezing `prg_rfz/pri_rfz` | rain→ice/graupel (T<0) | **present** (qrfz tables) | unchanged |
| snow/graupel riming `prs_scw/prg_gcw` | cloud-w→snow/graupel | present (v0.15 prior) | unchanged |
| **rain-collecting-snow `prr_rcs/prs_rcs/prg_rcs`** | rain+snow→graupel (T<0) | **MISSING** | **ADDED** |
| **rain-collecting-graupel `prr_rcg/prg_rcg`** | rain+graupel→graupel (T<0) | **MISSING** | **ADDED** |
| rain-collecting-ice `prr_rci/pri_rci/prg_rci` | rain+ice→graupel | MISSING | → 0.16 |
| snow-collecting-ice `prs_sci/pni_sci` | ice→snow accretion | MISSING | → 0.16 |
| rime-splinter `pri_ihm/pni_ihm` (Hallett-Mossop) | ice multiplication | MISSING | → 0.16 |

The aerosol-aware processes (`pna_*/pnd_*/pnb_*`) and the predicted-graupel-density
machinery (`qb`/`pbg_*`/`rho_g(NRHG)`) are **out of scope**: mp_physics=8 sets
`is_aerosol_aware=.FALSE.` and collapses `dimNRHG→1` (single density rho_g=400,
idx_bg=5), which the port already matches.

## What was added (bit-exact, not recomputed)

Tables loaded from the pristine WRF `.dat` files (gfortran unformatted, big-endian,
4-byte record markers) — `qr_acr_qsV2.dat` (12 records) + `qr_acr_qg_V4.dat`
(5 records, mp8 single-density plane). Extractor:
`proofs/v015/cold_collection_oracle/extract_collision_tables.py`. Fixture:
`data/fixtures/thompson-cold-collection-v1.npz` (73 MB; the warm-only `tcg_racg`
and the qrfz freezing tables already in `thompson-tables-v1.npz` are excluded).

`_cold_collection()` in `thompson_column.py` wires the exact WRF cold-branch
formulas (lines 2484–2548) and tendencies (3058–3120), gated to `T<T_0` (the
sub-freezing levels where WRF's `twet==temp`). Gate:
`GPUWRF_THOMPSON_COLD_COLLECTION` (default ON when the fixture is present).

## Evidence (`coldmix_validation_report.json`)

Cold mixed-phase WRF savepoint (184 cold rain+snow / 138 cold rain+graupel levels):

```
qr column rain sink:  WRF=-0.155   lane OFF=-0.028 (18%)   lane ON=-0.160 (103%)
qr per-cell mean_rel: OFF=12.81  ->  ON=0.87
qg per-cell mean_rel: OFF=0.55   ->  ON=0.30
```

Falsifier: with the lane OFF the port removes only 18% of WRF's column rain sink;
ON it matches to 3%. Warm precip oracle stays WRF-faithful (qr mean_rel<0.01).

## Honest remaining gap (0.16, est. 1 bounded sprint)

- **rain-collecting-ice (`prr_rci/pri_rci/prg_rci`)** + **snow-collecting-ice
  (`prs_sci`)**: additional rain/ice→graupel and ice→snow; needs `Ef_ri/Ef_si`
  efficiencies + the ice/snow distributions (no new lookup tables). ~+9% qg.
- **Rime-splinter `pri_ihm` (Hallett-Mossop)**: small ice-number source.
- **WRF joint-rate global ratio limiting**: WRF stages ALL rates from one
  pre-update state then applies with a single conservation ratio; the port
  applies freeze→riming→cold-collection sequentially. This is the source of the
  residual per-cell mean_rel and the rain-NUMBER over-depletion (nr column delta
  JAX −7.8e5 vs WRF −4.7e5). Refactoring to joint staging is the larger 0.16
  item and would tighten per-cell distribution.

The **bulk magnitude** that drives RAINNC accumulation is closed (103%); the
0.16 items refine vertical distribution and number, not the dominant sink.
