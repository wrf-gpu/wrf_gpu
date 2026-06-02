# P0-5 — Operational output completeness + restart (FINDINGS)

Sprint: **P0-5** (v0.2.0 Wave-1 P0-5a + Wave-5 P0-5b). Owner: Opus I/O lane.
Branch: `worker/opus/p0-5-io` from `worker/opus/v020-integration` @ d6ce779.
**GPU-FREE** (CPU JAX, `taskset -c 0-3`, `OMP_NUM_THREADS=2`); the single GPU is
reserved for the manager's equivalence remeasure.

File ownership respected: edited only the wrfout writer (`io/wrfout_writer.py`),
its `io/__init__.py` export, the writer's own test, plus NEW `io/restart.py`, NEW
`tests/test_p0_5_restart_full_carry.py`, and NEW `proofs/p0_5/**`. **Did NOT touch**
`runtime/operational_mode.py`, `coupling/physics_couplers.py`, `dynamics/**`, any
physics/nesting/conservation module, or the runtime-lane `runtime/checkpoint.py`.

---

## P0-5a — wrfout variable/metadata coverage

### Method

Authoritative WRF field set extracted from a real WRF-ARW run on disk:
`/mnt/data/canairy_meteo/runs/wrf_l3/20260428_18z_l3_24h_20260525T221139Z/wrfout_d02_2026-04-28_19:00:00`
(375 variables). For every field added, the **name, units, dimensions,
staggering, and description were copied from that reference** — no invented
fields. The proof `proofs/p0_5/wrfout_coverage_inventory.py` re-derives the
comparison and cross-checks metadata against the reference (`status: PASS`,
`proofs/p0_5/wrfout_coverage_inventory.json`).

### Starting point (already emitted before this sprint — 41 fields)

Coordinates/static: `Times XTIME XLAT XLONG XLAT_U XLONG_U XLAT_V XLONG_V HGT
LANDMASK LU_INDEX`. Prognostic: `U V W T QVAPOR P PB PH PHB MU MUB`. Hydrometeors:
`QCLOUD QICE QRAIN CLDFRA`. Surface map (already routed from the M9 diagnostics in
a prior sprint): `U10 V10 T2 Q2 PSFC SWDOWN GLW PBLH UST HFX LH TSK`. Precip:
`RAINC RAINNC RAINSH`.

### Gap closed — 31 fields ADDED (all real WRF wrfout variables)

Each field self-gates on a real source being present; an absent optional source
(operational diagnostics / Noah-MP land carry) leaves the field OUT of the file —
the writer never emits a fabricated quantity.

| WRF name | units | dims / stagger | Source in gpuwrf | Notes |
|---|---|---|---|---|
| `QSNOW` | kg kg-1 | XYZ | `State.qs` | Thompson prognostic snow |
| `QGRAUP` | kg kg-1 | XYZ | `State.qg` | Thompson prognostic graupel |
| `QNICE` | kg-1 | XYZ | `State.Ni` | ice number conc |
| `QNRAIN` | kg-1 | XYZ | `State.Nr` | rain number conc |
| `QKE` | m2 s-2 | XYZ | `State.qke` | twice TKE from MYNN |
| `ZNU` | – | (Time, bottom_top) | `GridSpec.eta_levels` midpoints | eta on mass levels |
| `ZNW` | – | (Time, bottom_top_stag) Z | `GridSpec.eta_levels` | eta on full levels |
| `MAPFAC_M` | – | XY | `metrics.msftx` | mass-grid map factor |
| `MAPFAC_U` | – | XY (west_east_stag) X | `metrics.msfux` | u-grid map factor |
| `MAPFAC_V` | – | XY (south_north_stag) Y | `metrics.msfvx` | v-grid map factor |
| `F` | s-1 | XY | `metrics.f` | Coriolis sin-lat |
| `E` | s-1 | XY | `metrics.e` | Coriolis cos-lat |
| `SINALPHA` | – | XY | `metrics.sina` | map rotation sine |
| `COSALPHA` | – | XY | `metrics.cosa` | map rotation cosine |
| `XLAND` | – | XY | landmask (1 land / 2 water) | WRF land/water flag |
| `P_TOP` | Pa | (Time,) | `metrics.p_top` | model-top pressure |
| `SNOWNC` | mm | XY | `State.snow_acc + ice_acc` | WRF SNOWNC = grid-scale snow+ice |
| `GRAUPELNC` | mm | XY | `State.graupel_acc` | grid-scale graupel |
| `QFX` | kg m-2 s-1 | XY | `diagnostics["QFX"]` | upward surface moisture flux (routed) |
| `GRDFLX` | W m-2 | XY | `diagnostics["GRDFLX"]` | ground heat flux (routed) |
| `TH2` | K | XY | T2·(P0/PSFC)^(Rd/cp) | 2-m potential temp (WRF TH2=T2/pi2) |
| `TSLB` | K | SOIL (4-layer) Z | `land_state.tslb` | soil temperature |
| `SMOIS` | m3 m-3 | SOIL Z | `land_state.smois` | total soil moisture |
| `SH2O` | m3 m-3 | SOIL Z | `land_state.sh2o` | liquid soil moisture |
| `SNOW` | kg m-2 | XY | `land_state.sneqv` | snow water equivalent |
| `SNOWH` | m | XY | `land_state.snowh` | physical snow depth |
| `CANWAT` | kg m-2 | XY | `land_state.canliq + canice` | canopy water |
| `SFROFF` | mm | XY | `land_state.sfcrunoff`*1e3 | surface runoff (m->mm) |
| `UDROFF` | mm | XY | `land_state.udrunoff`*1e3 | underground runoff (m->mm) |
| `ALBEDO` | – | XY | `land_state.albedo` | broadband albedo |
| `EMISS` | – | XY | `land_state.emiss` | surface emissivity |

Result with all optional sources present: **72 fields, 0 invented, metadata
WRF-faithful** (`wrfout_coverage_inventory.json` status PASS).

### Manager hook needed to fully populate the new operational fields

The writer accepts the new fields; routing them from the operational scan is a
small change the **L1 lane** owns in `runtime/operational_mode.py` /
`integration/daily_pipeline.py` (this lane must NOT edit those files):

1. **QFX / GRDFLX** — add to `M9Diagnostics` (already computed upstream: `QFX`
   from the Noah-MP `NoahMPFluxes.qfx` or bulk `surf.qfx`; `GRDFLX` from
   `NoahMPFluxes.grdflx`). Then add `("QFX","qfx")`, `("GRDFLX","grdflx")` to
   `_M9_OUTPUT_FIELDS` in `daily_pipeline.py` so `_surface_diagnostics_for_output`
   includes them. The writer's `_DIAGNOSTIC_SURFACE_FIELDS` set already accepts both.
2. **Soil/snow/land** (`TSLB/SMOIS/SH2O/SNOW/SNOWH/CANWAT/SFROFF/UDROFF/ALBEDO/EMISS`)
   — pass the prognostic `NoahMPLandState` carry to the writer:
   `prepare_wrfout_payload(..., land_state=carry.noahmp_land)` and
   `write_wrfout_netcdf(..., land_state=...)`. New optional kwarg; `None` →
   byte-identical to today (those fields simply absent).
3. **Microphysics extras / grid coords / precip partition** need NO hook — they
   read directly from the `State` leaves / `GridSpec` the writer already receives,
   so they appear as soon as the operational state carries them.

No regression risk: with `diagnostics=None` and `land_state=None` and a reduced
(synthetic/test) state, every new field self-gates off and the legacy output set
is preserved (existing writer + async-equiv tests stay green).

### Intentionally NOT emitted (303 WRF fields) — classified, not silent

`wrfout_coverage_inventory.json` lists all 303 with rationale. Groups:

- **AC\* accumulated land/energy/radiation budget terms** (Noah-MP ACCFLX family):
  diagnostic accumulators, not forecast fields — out of operational scope.
- **Radiation flux diagnostic tail** (`LWUP*/SWUP*/LWDN*/SWDN*` TOA/BOA clear-sky,
  `OLR`, `SWNORM`): RRTMG diagnostics; the **surface** forcing the product uses
  (`SWDOWN`, `GLW`) IS emitted.
- **Stochastic seed arrays** (`ISEEDARR*`): no stochastic physics in scope.
- **Ideal-run / bookkeeping flags** (`THIS_IS_AN_IDEAL_RUN`, `NEST_POS`, ...).
- **Lake / urban / crop / carbon** (`WATER_DEPTH`, `CROPCAT`, `LFMASS`, `GPP`, ...):
  options CUT from the Noah-MP scope (NOAH-MP-SCOPING.md).
- **SSO / GWD inputs** (`VAR_SSO`, `OA[1-4]`, `OL[1-4]`, ...): P1-7 GWD deferred.
- **Hybrid-eta coefficient + scalar metadata tail** (`C3H/C4H/.../FNM/FNP/DN/DNW/
  RDX/RDY/P00/T00`): the coordinate fields downstream tools need (`ZNU/ZNW/P_TOP`)
  ARE emitted.
- **Redundant / skin variants** (`SST/SSTSK/TMN/SNOALB/ALBBCK/Q2B/T2V/...`, extra
  map-factor variants `MAPFAC_UY/VX/MX/MY`): static or derived; primaries emitted.

These are candidates for a later 0.2.x pass if a downstream product needs them;
none is load-bearing for the Canary operational forecast product.

---

## P0-5b — wrfrst-equivalent restart + true-state resume fidelity

### Why a new module (vs the existing `runtime/checkpoint.py`)

`runtime/checkpoint.py` (runtime-lane owned) round-trips the prognostic `State`
(+ optional `runtime_state` / Noah-MP `land_state`) but the operational driver
advances an **`OperationalCarry`**, and the carry pieces OUTSIDE `State` are
exactly WRF's restart-relevant physics memory. A restart that drops them is not a
true bit-continuation. `gpuwrf.io.restart` serializes the **COMPLETE carry** — a
strict superset of `checkpoint.py`, built on the same fail-closed, host-numpy,
schema-versioned discipline, and it does **not** modify the runtime-lane module.

### Restart schema (`gpuwrf.io.restart`, FORMAT `gpuwrf-operational-restart` v1)

Pickle payload, every array leaf host-copied to numpy (process-independent,
GPU-free), each pytree stored as an explicit `{field_name: array}` dict with the
recorded field order so a schema change **fails closed** on read:

```
format, format_version
carry:
  state_field_order / state_fields            # all 53 State.__slots__
  scratch_field_order / scratch_fields         # 14 OperationalCarry scratch leaves
  noahmp_land_field_order / noahmp_land_fields  # 32 NoahMPLandState.__slots__ (or None)
  noahmp_rad                                    # held (SOLDN, LWDN, COSZ) (or None)
namelist, grid, step_index
noahmp_static                                   # optional read-only NoahMP inputs
metadata
```

### Full physics carry-state coverage (the bit-continuation guarantee)

| Carry component | Fields | Why it must persist |
|---|---|---|
| Prognostic `State` | 53 leaves (u/v/w/theta/qv, p/ph/mu totals+perturbations, qc/qr/qi/qs/qg + Ni/Nr/Ns/Ng, qke, surface-layer ustar/fluxes, land surface, precip accumulators, all `*_bdy` boundary forcing, lu_index) | the forecast state |
| WRF small-step scratch | `t_2ave ww mudf muave muts ph_tend` + `u_save v_save w_save t_save ph_save mu_save ww_save` | RK/acoustic transition state consumed across small-step stages (`runtime.operational_state`) |
| Held radiation | `rthraten` | WRF refreshes the radiative theta-tendency once per `radt` and ADDS `dt*rthraten` EVERY dynamics step in between; dropping it at a mid-`radt` restart silently loses up to one `radt` of radiative heating |
| Prognostic Noah-MP land | `noahmp_land` (32 leaves: 4-layer soil T/moisture/liquid, 3-layer snow, canopy big-leaf, runoff, albedo/emiss) | land-surface memory — the standalone-replacement land carry |
| Held surface radiation | `noahmp_rad` = (SOLDN, LWDN, COSZ) | held forcing the Noah-MP step reads between radiation calls |
| Run-static (optional) | `noahmp_static` (categories, soil geometry, parameter tables), `grid`, `namelist`, `step_index` | lets a resumed run rebuild the land driver without re-reading `wrfinput` |

> **W0AVG / NCA (Kain-Fritsch averaged vertical velocity / convective countdown):**
> these are P0-4 (d01 KF cumulus) carry that does NOT yet exist in `OperationalCarry`
> at this base commit. When the KF lane adds them to the carry, they are picked up
> by bumping the schema to v2 (append to `_CARRY_SCRATCH_FIELDS`). Flagged in
> *Unresolved risks*.

### CPU save/load bit-fidelity proof (`proofs/p0_5/restart_roundtrip.json`, PASS)

GPU-FREE (`proofs/p0_5/restart_roundtrip_proof.py`). State built from the frozen
field-shape contract + `initial_operational_carry` (NOT `State.zeros`, which
hard-requires a GPU). Every leaf gets a distinct deterministic non-zero pattern so
a dropped/swapped field cannot accidentally compare equal.

- **full_carry**: 102 leaves (53 State + 14 scratch + 32 land + 3 rad) **bit-identical**
  after write->read; step index round-trips (137). `bit_identical: true`.
- **landless_carry** (Noah-MP off): 67 leaves bit-identical; `has_noahmp_land: false`.
- **schema_drift_fails_closed**: corrupting a recorded field order raises
  `ValueError` on read (no silent mis-reconstruction).

pytest `tests/test_p0_5_restart_full_carry.py` (5 tests, all green) adds an explicit
**held-`rthraten`-survives-restart** continuity check (a non-zero held tendency is
preserved, not re-seeded to zero).

### Manager's GPU resume-continuity check (SPEC — needs a GPU forecast)

This lane delivers the schema + the CPU roundtrip fidelity. The end-to-end
"resume a forecast -> bit-continue vs uninterrupted" GPU proof is handed to the
manager. Recommended falsifiable gate:

1. Run an uninterrupted operational forecast of `N` steps on the GPU; record the
   final `OperationalCarry` (and the per-output-hour wrfout fields).
2. Run the SAME forecast to step `K` (`0 < K < N`, ideally landing **mid-`radt`**
   so the held `rthraten` is non-zero, and after >=1 Noah-MP physics step so the
   land carry has evolved), `write_restart(carry_K, namelist, grid, K, path)`.
3. `read_restart(path)` -> resume to step `N` with the SAME compiled scan.
4. **Gate (PASS):** the resumed final carry equals the uninterrupted final carry —
   exact bitwise for the same precision/compile, or within a predeclared
   round-off floor (`max |delta| <= a few ULP`) if XLA reassociates across the
   restart boundary; the wrfout fields at every hour >= K must match the
   uninterrupted run to the same tolerance. Choose `K` so it crosses a radiation
   refresh AND a microphysics-accumulator increment.
5. **Anti-cheat:** a restart that re-seeds scratch via `initial_operational_carry`
   instead of the saved scratch must FAIL this gate (proves the scratch coverage
   is load-bearing). Likewise zeroing the held `rthraten` at restart must produce
   a detectable theta discontinuity over the first `radt` interval.

Single-GPU, one job at a time; no in-loop host transfer (the restart save is at an
output boundary, the resume is a fresh device placement — both outside the scan).

---

## Commands run

```
git checkout -b worker/opus/p0-5-io worker/opus/v020-integration   # @ d6ce779
# coverage inventory + restart proofs (CPU, GPU-free):
PYTHONPATH=src JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 \
  python proofs/p0_5/wrfout_coverage_inventory.py        # status PASS
PYTHONPATH=src JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 \
  python proofs/p0_5/restart_roundtrip_proof.py          # status PASS
PYTHONPATH=src JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 \
  python -m pytest tests/test_m7_netcdf_writer.py tests/test_m7_wrfout_io_compat.py \
    tests/test_async_wrfout_equiv.py tests/test_p0_5_restart_full_carry.py -q   # all green
```

## Proof objects

- `proofs/p0_5/wrfout_coverage_inventory.json` — coverage vs WRF, metadata
  cross-check (PASS, 72 emitted / 0 invented / metadata faithful) + the
  reference file sha256.
- `proofs/p0_5/restart_roundtrip.json` — full-carry CPU bit-fidelity (PASS).
- `proofs/p0_5/restart_full.wrfrst`, `proofs/p0_5/restart_landless.wrfrst` —
  example restart artifacts.

## Unresolved risks

- **GPU resume-continuity gate is unproven** (no GPU here, by design) — spec above;
  manager owns it. Highest-value remaining check.
- **KF W0AVG/NCA not yet in the carry** at this base — when the P0-4 lane adds them
  to `OperationalCarry`, bump the restart schema to v2 by appending them to
  `_CARRY_SCRATCH_FIELDS`; the fail-closed reader will reject a v1 file against a
  widened carry until then (correct, safe).
- **L1 routing hook** for QFX/GRDFLX + `land_state=` is a separate one-line change
  the runtime/pipeline lane owns; until then those operational fields are
  present-in-schema but only populated on callers that pass them (the writer is
  ready and tested for both paths).
