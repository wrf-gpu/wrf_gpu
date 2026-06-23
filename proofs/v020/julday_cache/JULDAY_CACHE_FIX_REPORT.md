# v0.20.0 #91 — JAX compile cache HITS ACROSS DATES (julday) — FIX REPORT

**Verdict: FIXED (was a real bug; not a no-op).** The JAX persistent compile cache
now HITS when the same case runs on a different forecast DATE — out-of-the-box, no
config. Proven end-to-end on the real operational `_advance_chunk`, and proven
numerically inert (fp64_default byte-identical: traced vs baked clock).

Branch: `worker/cache/v020-julday` (off `worker/integration/v020-fp32 @ dcfe4cb8`).

---

## 1. Symptom — empirically CONFIRMED first (not assumed)

The forecast date was baked STATIC into the radiation/solar HLO, so a new date
minted a new XLA cache key → full recompile per date (cache MISS).

Root mechanism: three solar helpers in `coupling/physics_couplers.py`
(`_compute_coszen`, `_compute_solar_geometry`, `_solcon_for_time`) extracted
`julian, utc_minute = _time_utc_parts(time_utc)` as **Python floats** and used them
directly in `jnp` expressions (declination / equation-of-time / hour-angle /
eccentricity solar-constant). XLA folds Python-float literals into the HLO as
compile-time constants → date-dependent HLO. These funnel through the radiation
coupler inside the fused per-step `_advance_chunk`, and `OperationalNamelist`
carries `time_utc` / `noahmp_julian` / `noahmp_yearlen` in its **static aux**
(jit cache key, `operational_mode.py:758-765`), so each date → distinct compiled
program.

Pre-fix probe (`probe_julday_cache.py`, isolated cache, `_compute_solar_geometry`,
the exact coupler code path) — `probe_result_pre.json`:

| step          | date       | new cache entries | warm hit |
|---------------|------------|-------------------|----------|
| A_cold        | 2023-01-15 | **+1**            | no       |
| A_warm_again  | 2023-01-15 | 0                 | **yes** (harness valid) |
| B_other_date  | 2023-07-15 | **+1 (MISS)**     | no       |
| C_leap_year   | 2024-07-15 | **+1 (MISS)**     | no       |

→ Cross-date cache MISS confirmed.

---

## 2. Fix — surgical, moves the date scalars OUT of the static HLO → RUNTIME args

The date-derived scalars are now built ONCE on the host (outside jit) and threaded
into the compiled scan as a **traced argument** `clock_base` — exactly like the
existing traced `start_step` — so the compiled program is date-INDEPENDENT. Values
are byte-for-byte the old floats; only their **binding time** (runtime vs trace)
changes.

New seam:
- `physics_couplers.time_utc_clock_base(time_utc)` → traced `(julian, utc_minute)`.
- `physics_couplers._resolve_clock_parts(time_utc, clock_base)` — uses the traced
  pair when supplied, else legacy host extraction (back-compat).
- `operational_mode._ClockBase` NamedTuple (`rad_julian, rad_minute,
  noahmp_julian, noahmp_yearlen`) + `build_clock_base(namelist)` host builder.

`clock_base` threaded (default `None` = legacy host-extraction, fully back-compat):
- `physics_couplers.py`: the 3 solar helpers, `_solar_source_scale_for_time`, the
  3 column-input builders (`_rrtmg/_dudhia/_gsfc`), and the public radiation
  wrappers (`rrtmg_theta_tendency`, `rrtmg_sw/lw_theta_tendency`,
  `dudhia_sw/gsfc_sw_theta_tendency`, `rrtmg_radiation_diagnostics`).
- `operational_mode.py`: `_advance_chunk(_fori/_static_scan)` (new traced positional
  arg), `_physics_boundary_step(_with_limiter_diagnostics)`, `_physics_step_forcing`,
  `_refresh_noahmp_rad`, the SW/RRTMG radiation dispatch, `_NoahMPClock`
  (phenology julian/yearlen now traced when `clock_base` present), `_m9_snapshot` +
  `compute_m9_diagnostics`. Host loops `run_forecast_operational[_segmented/_with_m9]`
  build `clock_base` once and pass it.
- `domain_tree.py` (PRODUCTION 9-nest): `_operational_advance_factory` builds one
  `clock_base` per-domain; the fused cascade passes parent/child clock bases as
  **traced jit arguments** (NOT closed-over constants — that would re-bake them).
- `nested_pipeline.py` / `daily_pipeline.py`: M9 diagnostics pass `clock_base`.

LW-classic (`rrtm_lw`) and Held-Suarez are date-independent (no solar) — untouched.

### Documented residual (NON-default path)
GSFC SW (`ra_sw_physics=2`, NOT the default) still selects the ozone climatology
band `iprof` (a discrete int 1–5) via `_select_iprof(lat, julday)` → a static array
index `OZONE_GG[iprof-1]`, so the GSFC SW HLO still varies by season band. The
**default operational path is RRTMG (`ra_sw=4`/`ra_lw=4`)** and is fully
date-independent. Removing the GSFC band index would need a traced gather over the
small ozone table; left as a documented non-default residual.

---

## 3. Fix VERIFIED — end-to-end on the REAL operational `_advance_chunk`

`probe_advance_chunk_crossdate.py` builds the real v0.17 bigswiss d01 operational
state + namelist (full RRTMG SW+LW, fp64_default, `force_fp64=True`) from the
actual `wrfinput`, and lowers+compiles the production `_advance_chunk_fori` for
three dates against an isolated cache — `advance_chunk_crossdate_result.json`:

- **Lowered HLO sha256 IDENTICAL across dates**:
  A(2023-01-15) = B(2023-07-15) = C(2024-07-15, leap) = `49c769b4e531744d…`
  (`hlo_identical_across_dates_AB=true`, `…_AC_leap=true`). The compile cache keys
  on the HLO, so identical HLO ⇒ guaranteed cross-date hit.

| step          | date       | new entries | warm hit |
|---------------|------------|-------------|----------|
| A_cold        | 2023-01-15 | +1          | —        |
| A_warm        | 2023-01-15 | 0           | **yes**  |
| B_other_date  | 2023-07-15 | **0**       | **yes**  |
| C_leap        | 2024-07-15 | **0**       | **yes**  |

→ `cross_date_cache_hit=true`, `verdict=FIXED`.

Kernel-level confirmation (`probe_result_post.json`, `_compute_solar_geometry` with
traced `clock_base`): B and C both warm hits, 0 new entries.

---

## 4. Numerically INERT — fp64_default bit-identity preserved (HARD gate)

`probe_bit_identity.py` runs the real bigswiss `_advance_chunk` (fp64_default, RRTMG
radiation EVERY step) for the SAME date twice — once with `clock_base=None` (legacy
baked-literal path) and once with the traced `clock_base` (#91 fix) — and
byte-compares every State leaf — `bit_identity_result.json`:

> **64/64 State leaves BYTE-IDENTICAL** (`verdict: BIT_IDENTICAL`,
> `all_bit_identical: true`). bigswiss d01, fp64_default, RRTMG SW+LW, radiation
> fires on the cadence step, 3 steps. baked-literal arm == traced-clock_base arm,
> `np.array_equal` on every State leaf.

Identical values, only binding-time changed ⇒ the 963/963 fp64_default byte-identity
the integration achieved is preserved. fp32 modes are unaffected (the seam only
changes how the date scalars are passed; fp32 dtype handling is untouched).

Relevant unit tests (CPU, `clock_base=None` legacy host path): GREEN —
`test_v060_ra_sw_dudhia.py` 15 passed; `test_v013_ra_sw_gsfc.py` +
`test_rrtm_lw_operational_wiring.py` passed (exit 0). These exercise the SW solar
helpers via the host-extraction default, confirming no regression on the legacy path.

---

## 5. Out-of-the-box statement

**The JAX persistent compile cache now HITS across forecast dates with no config.**
A user (or the release gate) running the same case on any date gets the warm
out-of-the-box compile — the expensive cold compile is paid once, not per date.
Default operational path (RRTMG) is fully date-independent end-to-end.

## Files changed
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/domain_tree.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- proofs: `proofs/v020/julday_cache/probe_*.py` + `*_result*.json` + this report +
  `JULDAY_DONE`. (The brief named `proofs/v020/cache/`, but a `.gitignore` `cache/`
  rule makes any `cache/`-named dir un-committable, so the committed deliverables
  live in `proofs/v020/julday_cache/`.)
