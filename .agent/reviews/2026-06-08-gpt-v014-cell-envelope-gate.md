# v0.14 Cell-Envelope Gate Design

Objective: design a v0.14 validation gate that checks GPU-vs-CPU-WRF grid-cell values over retained wrfouts, not only station TOST pairs.

## Bottom Line

Build a CPU-only comparator named `v014_cell_envelope_v1`. It must read paired GPU and CPU `wrfout_d02_*` files, compare complete timestamp pairs over every finite grid cell, and emit per-case/per-field statistics plus a hard verdict only for fields with frozen tolerances.

Do not promote any "TOST-equivalent" claim from station scores alone. TOST is station skill evidence; the cell-envelope gate is field-parity evidence. Release claims need both, and a TOST pass must not mask a grid failure.

## Evidence Read

- `proofs/v0120/powered_tost_n15/manifest.json`
- `proofs/v0120/powered_tost_n15/case_*.json`
- `proofs/v0120/powered_tost_n15/pipeline_proofs/*`
- `proofs/v014/v10_grid_diagnostics.py`
- `proofs/v014/v10_grid_diagnostics.json`
- `docs/equivalence-demo.md`
- `docs/VALIDATION.md`
- `src/gpuwrf/io/wrfout_writer.py`
- runtime headers/logs under `/tmp/v0120_powered_tost_runs`, `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`, and `/mnt/data/wrf_gpu_validation/v0130_marathon`

Current runtime state observed without using GPU:

- CPU truth exists for all 15 manifest cases, 73 d02 wrfouts each.
- Repo case JSONs exist for 3 completed powered-TOST cases: `20260429`, `20260430`, `20260501`.
- Durable case JSON `cell_level` stores only `T2`, `U10`, `V10`, with aggregate stats and 0-6/6-12/12-24 h blocks.
- Current GPU wrfouts in `/tmp/v0120_powered_tost_runs` are retained for `20260501` only among those 3 JSON cases; `20260502` directory exists but had no d02 wrfout when inspected.
- Existing `proofs/v014/v10_grid_diagnostics.json` is stale relative to the 3 case JSONs: it reports 2 JSON-only cases and no spatial wrfout reads.
- Existing `M7L2D02Tier4RMSE` proof is final-hour only and only `T2`, `U10`, `V10` with loose thresholds (`3 K`, `7.5 m s-1`, `7.5 m s-1`). It is not this gate.

## Fields To Compare Now

The gate should have two field layers.

Hard-fail core, immediately usable because `docs/equivalence-demo.md` already freezes tolerances:

`T2`, `U10`, `V10`, `PSFC`, `RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`.

Full current-common inventory, to compute on every run now and promote to hard-fail as soon as per-field tolerances are frozen:

Geometry/static:
`XLAT`, `XLONG`, `XLAT_U`, `XLONG_U`, `XLAT_V`, `XLONG_V`, `CLAT`, `HGT`, `LANDMASK`, `LU_INDEX`, `XLAND`, `MAPFAC_M`, `MAPFAC_U`, `MAPFAC_V`, `MAPFAC_MX`, `MAPFAC_MY`, `MAPFAC_UX`, `MAPFAC_UY`, `MAPFAC_VX`, `MAPFAC_VY`, `F`, `E`, `SINALPHA`, `COSALPHA`, `ZNU`, `ZNW`, `DN`, `DNW`, `RDN`, `RDNW`, `FNM`, `FNP`, `C1H`, `C2H`, `C3H`, `C4H`, `C1F`, `C2F`, `C3F`, `C4F`, `CFN`, `CFN1`, `CF1`, `CF2`, `CF3`, `P_TOP`, `RDX`, `RDY`.

Dycore/base state:
`U`, `V`, `W`, `T`, `QVAPOR`, `P`, `PB`, `PH`, `PHB`, `MU`, `MUB`.

Surface/near-surface:
`T2`, `TH2`, `Q2`, `U10`, `V10`, `PSFC`, `TSK`, `PBLH`, `UST`, `HFX`, `LH`, `GLW`, `SWDOWN`.

Microphysics/cloud:
`CLDFRA`, `QCLOUD`, `QICE`, `QRAIN`, `QSNOW`, `QGRAUP`, `QKE`, `QNICE`, `QNRAIN`.

Precipitation/radiation:
`RAINC`, `RAINNC`, `RAINSH`, `SNOWNC`, `GRAUPELNC`, `SR`, `SWDNB`, `SWUPB`, `LWDNB`, `LWUPB`, `SWDNT`, `SWUPT`, `LWDNT`, `LWUPT`, `OLR`, `SWNORM`, `COSZEN`.

Metadata:
`Times` and `XTIME` are not scored as fields. They are exact timestamp/alignment checks.

Do not compare GPU-only fields as pass/fail until CPU truth contains them: current GPU file has `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW` that were not present in the inspected CPU truth file.

## Missing Writer/Runtime Fields

Current CPU truth contains more relevant fields than the inspected GPU wrfout. These should be explicit missing-field statuses, not silent skips.

Must add/fix before "all current operational wrfout cells" can be claimed:

- Clear-sky radiation fluxes: `SWDNBC`, `SWUPBC`, `LWDNBC`, `LWUPBC`, `SWDNTC`, `SWUPTC`, `LWDNTC`, `LWUPTC`.
  `wrfout_writer.py` has specs and the diagnostics allow-list, but `RADIATION_FLUX_DIAGNOSTIC_VARIABLES` / `OPERATIONAL_WRFOUT_VARIABLES` omit them, so `write_prepared_wrfout()` would skip them even if diagnostics supplied them. `daily_pipeline.py` also still routes all-sky only.
- Surface flux budget fields: `QFX`, `GRDFLX`.
  Writer specs exist and allow ADD-only diagnostics, but the inspected GPU wrfout did not contain them. The operational diagnostics map must route real values or the gate should mark `MISSING_GPU_FIELD`.
- Land/soil/snow/canopy fields already specced by writer but absent from inspected GPU wrfout: `TSLB`, `SMOIS`, `SH2O`, `SNOW`, `SNOWH`, `CANWAT`, `SFROFF`, `UDROFF`, `ALBEDO`, `EMISS`, `TSNO`, `SNICE`, `SNLIQ`, `ZSNSO`, `ISNOW`, `SNEQVO`, `CANLIQ`, `CANICE`, `SNOWC`.
  This is mostly a runtime source issue: the writer needs a real `land_state` at wrfout time.

Do not make the first v0.14 gate fail on all CPU-WRF-only Registry variables. The current Canary output contract is not full 375-variable WRF history coverage. The gate should inventory CPU-only names and require an explicit policy row before any becomes hard-fail.

## Tolerances

Use existing frozen core tolerances from `docs/equivalence-demo.md`:

| Field | Pooled RMSE tolerance |
|---|---:|
| `T2` | `1.5 K` |
| `U10` | `1.5 m s-1` |
| `V10` | `1.5 m s-1` |
| `PSFC` | `120 Pa` |
| `RAINNC` | `1.0 mm` |
| `T` | `1.5 K` |
| `U` | `1.8 m s-1` |
| `V` | `1.8 m s-1` |
| `W` | `0.30 m s-1` |
| `QVAPOR` | `1.0e-3 kg kg-1` |

All other fields must get per-field or per-family tolerances before they become hard-fail. A single tolerance is invalid because fields differ in units, scale, sparsity, staggering, and semantics:

- Static geometry needs shape/coordinate equality or very tight `max_abs` checks, not forecast RMSE.
- Categorical masks (`LANDMASK`, `LU_INDEX`, `XLAND`, later `ISNOW`) need exact/category checks.
- Base-state fields (`PB`, `PHB`, `MUB`, eta/map factors) should be near-exact schema/initialization checks.
- Dynamic pressure/mass fields (`P`, `PH`, `MU`) need Pa or geopotential tolerances and lead-time tracking.
- Sparse precipitation and hydrometeors need RMSE plus event/occurrence summaries; max error alone is misleading.
- Radiation/flux fields need W m-2 tolerances and diurnal lead stratification.

The implementation should therefore require a tolerance JSON with one row per hard-fail field:

```json
{
  "schema": "v014_cell_tolerances_v1",
  "frozen_utc": "...",
  "fields": {
    "T2": {"metric": "pooled_rmse", "rmse": 1.5, "units": "K", "source": "docs/equivalence-demo.md"},
    "LANDMASK": {"metric": "exact", "max_mismatch_fraction": 0.0, "units": "category"}
  }
}
```

No tolerance may be inferred from candidate output after the run. Fields with no frozen tolerance are `REPORT_ONLY`, never `PASS`.

## Case Corpus Policy

Use `proofs/v0120/powered_tost_n15/manifest.json` as the v0.14 corpus contract. It declares 15 cases and 72 h CPU truth, but the first gate should compare h1-h24 because that matches the powered TOST case JSONs and current GPU runs.

Status levels:

- `SMOKE`: 1 complete case with retained GPU wrfouts; useful for script validation only.
- `REGRESSION`: all completed case JSONs with durable GPU wrfouts; cannot support release equivalence.
- `PROMOTION`: all 15 manifest cases, d02, complete h1-h24 paired CPU/GPU wrfouts, no missing hard-required fields, no nonfinite values, and every hard-fail field within frozen tolerance.

Rules:

- A missing GPU directory or missing wrfout is `BLOCKED_MISSING_GPU_OUTPUT`, not pass.
- A skipped case remains in denominator unless the manifest is replaced before scoring.
- Replacement cases require a new manifest with reason and checksums before running.
- Do not rely on `/tmp` for promotion evidence. Copy or generate durable proof paths, or record a durable output root.
- Do not use `case_*.json` aggregate `cell_level` as the primary gate input. It is acceptable only as a fallback diagnostic when wrfouts are gone.

## Report Schema

Proposed output: `proofs/v014/cell_envelope_gate.json`.

```json
{
  "schema": "v014_cell_envelope_gate_v1",
  "generated_utc": "...",
  "cpu_only": true,
  "gpu_used_by_gate": false,
  "inputs": {
    "manifest": "proofs/v0120/powered_tost_n15/manifest.json",
    "case_json_dir": "proofs/v0120/powered_tost_n15",
    "cpu_root": "/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output",
    "gpu_root": "/tmp/v0120_powered_tost_runs",
    "domain": "d02",
    "lead_hours": [1, 24]
  },
  "field_policy": {
    "hard_fail_fields": ["T2", "U10", "V10", "PSFC", "RAINNC", "T", "U", "V", "W", "QVAPOR"],
    "report_only_fields": ["...full current-common fields without frozen tolerance..."],
    "missing_required_writer_fields": ["SWDNBC", "SWUPBC", "LWDNBC", "LWUPBC", "SWDNTC", "SWUPTC", "LWDNTC", "LWUPTC", "QFX", "GRDFLX"],
    "metadata_checks": ["Times", "XTIME", "shape", "dtype", "finite"]
  },
  "cases": [
    {
      "run_id": "...",
      "status": "PASS|FAIL|BLOCKED_MISSING_GPU_OUTPUT|BLOCKED_MISSING_CPU_OUTPUT",
      "n_common_leads": 24,
      "missing_gpu_fields": [],
      "missing_cpu_fields": [],
      "shape_mismatches": [],
      "nonfinite": [],
      "fields": {
        "V10": {
          "status": "PASS|EXCEEDS_TOL|REPORT_ONLY|MISSING_GPU_FIELD|MISSING_CPU_FIELD",
          "n": 251856,
          "rmse": 0.0,
          "bias": 0.0,
          "mae": 0.0,
          "p95_abs": 0.0,
          "p99_abs": 0.0,
          "max_abs": 0.0,
          "frac_abs_le_rmse_tol": 1.0,
          "tolerance": {"metric": "pooled_rmse", "rmse": 1.5, "units": "m s-1"},
          "by_lead": [],
          "by_block": {},
          "worst_cells": []
        }
      }
    }
  ],
  "summary": {
    "verdict": "PASS|FAIL|BLOCKED",
    "case_policy": "SMOKE|REGRESSION|PROMOTION",
    "n_cases_total": 15,
    "n_cases_compared": 0,
    "exceeding_fields": [],
    "blocked_cases": [],
    "tost_overlay": {}
  }
}
```

Minimum per-field stats: `n`, `bias`, `rmse`, `mae`, `p95_abs`, `p99_abs`, `max_abs`, `frac_abs_le_tol`, `by_lead`, `by_block` (`0-6h`, `6-12h`, `12-24h`), and worst-cell records with `lead_h`, `j`, `i`, `gpu`, `cpu`, `diff`, `lat`, `lon`, `hgt`, `landmask` when available.

## Commands

Inventory-only smoke, no GPU:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src \
  python proofs/v014/cell_envelope_gate.py \
    --manifest proofs/v0120/powered_tost_n15/manifest.json \
    --case-json-dir proofs/v0120/powered_tost_n15 \
    --cpu-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
    --gpu-root /tmp/v0120_powered_tost_runs \
    --domain d02 \
    --hours 24 \
    --mode inventory \
    --out proofs/v014/cell_envelope_inventory.json
```

Hard gate after tolerance freeze, no GPU:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src \
  python proofs/v014/cell_envelope_gate.py \
    --manifest proofs/v0120/powered_tost_n15/manifest.json \
    --case-json-dir proofs/v0120/powered_tost_n15 \
    --cpu-root /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output \
    --gpu-root /durable/path/to/v014_cell_outputs \
    --domain d02 \
    --hours 24 \
    --tolerances proofs/v014/cell_tolerances_v1.json \
    --require-cases 15 \
    --require-policy promotion \
    --out proofs/v014/cell_envelope_gate.json
```

Markdown rendering:

```bash
PYTHONPATH=src \
  python proofs/v014/render_cell_envelope_report.py \
    --input proofs/v014/cell_envelope_gate.json \
    --out proofs/v014/cell_envelope_gate.md
```

## Relationship To TOST

TOST answers: "Is GPU station skill statistically equivalent to CPU-WRF station skill for `T2`, `U10`, `V10`?"

The cell-envelope gate answers: "Are GPU fields close to CPU-WRF at all grid cells and all retained output times?"

They are complementary and neither subsumes the other. The current evidence shows why: stored powered-TOST case JSONs can have station V10 within margin while grid V10 RMSE is several m s-1. `v10_grid_diagnostics.py` already exists because station TOST and grid divergence can disagree.

Policy:

- TOST may include a `cell_envelope_summary` pointer, but must not override it.
- Cell-envelope failures explain whether a station TOST result is spatially localized, domain-wide, field-specific, lead-specific, or masked by station sampling.
- For an operational equivalence claim, require both `TOST_PASS` and `CELL_ENVELOPE_PASS`.
- If TOST passes but cell envelope fails, report `STATION_EQUIV_GRID_NOT_EQUIV`.
- If cell envelope passes but TOST fails, report `GRID_EQUIV_STATION_SKILL_NOT_EQUIV` and inspect obs representativeness or station extraction.

## Handoff

- objective: design the v0.14 all-cell validation gate.
- files changed: `.agent/reviews/2026-06-08-gpt-v014-cell-envelope-gate.md`.
- commands run: read-only `sed`, `jq`, `find`, `tail`, `ncdump`, `rg`, `git status`; no GPU command and no source edit.
- proof objects produced: this design report only.
- unresolved risks: no implementation yet; no frozen tolerances for the full common field set; current runtime output is not durable enough for promotion; missing clear-sky/land/QFX/GRDFLX fields need writer/runtime closure.
- next decision needed: freeze whether v0.14 promotion requires only the 10-field hard core or the full current-common field set with new per-field tolerances.
