# V0.14 Grid Comparison Framework

Date: 2026-06-08
Worker: GPT-5.5 xhigh validation-infrastructure
Scope: CPU-only comparator infrastructure. No GPU run. No model code edits.

## Objective

Implement the v0.14 fast complete CPU-WRF-vs-GPU wrfout grid comparator with concise top-level output and detailed JSON artifacts, then smoke it on retained Case 3 d02 wrfouts.

## Files Changed

- `scripts/compare_wrfout_grid.py`
- `proofs/v014/grid_comparison_framework_smoke.json`
- `proofs/v014/grid_comparison_framework_smoke.md`
- `proofs/v014/grid_comparison_method.md`
- `.agent/reviews/2026-06-08-v014-grid-comparison-framework.md`

No `src/` or model files were edited.

## Commands Run

```bash
python -m py_compile scripts/compare_wrfout_grid.py

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --vars T2 U10 V10 HGT Times \
    --out-json /tmp/wrfout_grid_subset.json \
    --out-md /tmp/wrfout_grid_subset.md

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --vars XTIME Times \
    --out-json /tmp/wrfout_grid_xtime.json \
    --out-md /tmp/wrfout_grid_xtime.md

/usr/bin/time -v env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
    --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
    --gpu-dir /tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
    --domain d02 \
    --out-json proofs/v014/grid_comparison_framework_smoke.json \
    --out-md proofs/v014/grid_comparison_framework_smoke.md \
    --progress 25

python -m json.tool proofs/v014/grid_comparison_framework_smoke.json >/dev/null
wc -l proofs/v014/grid_comparison_framework_smoke.md
```

Manager rerun after review: `PB`, `PHB`, `MUB`, and `MU0` were added to the
known static/base-state classification set, then the full smoke was rerun with
`taskset -c 24-31`, `--progress 50`, the same retained Case 3 d02 directories,
and the same output paths. Follow-up validation used
`python -m json.tool proofs/v014/grid_comparison_framework_smoke.json`,
`python -m py_compile scripts/compare_wrfout_grid.py`, and `git diff --check`
on the comparator/proof files.

## Proof Objects Produced

- `proofs/v014/grid_comparison_framework_smoke.json`
- `proofs/v014/grid_comparison_framework_smoke.md`
- `proofs/v014/grid_comparison_method.md`

The JSON validates with `python -m json.tool`. The markdown report is 51 lines, below the 80-line contract.

## Smoke Result

Case 3 d02 retained wrfouts:

- paired files: 24, leads h1-h24
- CPU files discovered: 73
- GPU files discovered: 24
- variables CPU/GPU/common: 375/104/100
- numeric compatible fields with stats: 99
- string metadata fields audited: `Times`
- dynamic fields: 37
- static/time-invariant fields: 61
- GPU-only fields: 4, `QNCCN`, `QNCLOUD`, `QNGRAUPEL`, `QNSNOW`
- CPU-only fields: 275
- incompatible common fields: 0
- verdict: `REPORT_ONLY_NO_TOLERANCE_MANIFEST`

Top smoke differences remain consistent with the existing v0.14 evidence: largest static/base-state failures are `C2F`, `C2H`, `C4F`, `C4H`, `MUB`, and `PHB`; dynamic failures include `PSFC`, `MU`, `P`, `PBLH`, and sparse `QNRAIN` outliers. The report keeps static/time-invariant fields separate from dynamic forecast fields.

## Runtime And Memory

Final full smoke under `/usr/bin/time -v`:

- elapsed wall time: 1:27.33
- user/system CPU: 85.49 s / 3.69 s
- max RSS: 446044 KB
- CPU share: 102%
- GPU used: no

The final manager rerun reported
`runtime.elapsed_seconds=86.87874836399897` in the JSON.

The script is variable-major and lead-streamed. It holds only the current variable's source arrays, finite absolute-difference chunks for exact p95/p99, optional split accumulators, and previous source arrays for time-invariance detection.

## Limitations

- No tolerance manifest was supplied, so the smoke is report-only and not an equivalence pass/fail gate.
- Static/time-invariant detection is exact over emitted wrfouts. Fields that are physically dynamic but unchanged over this 24 h case are classified as `time_invariant` to keep them out of dynamic RMSE.
- Spatial splits depend on CPU truth `HGT`, `LANDMASK`, `XLAT`, and `XLONG`; if those are absent, the JSON records split warnings and continues.
- Percentiles are exact but per-variable memory rises with the largest field and split masks. The Case 3 d02 max RSS was acceptable; Switzerland should re-record max RSS.
- The comparator identifies field/lead/region/cell symptoms. It does not localize first bad tendencies or prove whether emitted static mismatches are runtime-state or writer-payload defects.

## B4 Integration Recommendation

Make this comparator the v0.14 B4 grid gate. After each static metric/base-state or dycore fix, run it on retained Canary wrfouts and require:

- 24 paired d02 leads for the Case 3 smoke or the active B4 case set.
- All ten hard fields present and compatible: `T2`, `U10`, `V10`, `PSFC`, `RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`.
- Static/grid/base-state fields exact or covered by predeclared exceptions before dynamic RMSE is interpreted.
- A frozen tolerance manifest supplied before any pass/fail claim.
- The concise markdown attached to manager handoffs; JSON used by debug agents.

Do not resume TOST, FP32 landing, or speed claims from station evidence while this comparator still shows large static/grid or hard-field divergence.

## Switzerland Recommendation

Use the same script for Switzerland with `--domain d01` after a full GPU run emits 24 hourly wrfouts. Supply a frozen Switzerland tolerance manifest for the hard fields from `proofs/v014/switzerland_validation_plan.md`; fail coverage if any hard field is missing, incompatible, or lacks all 24 leads. Keep all extra common fields report-only until their tolerances are frozen before candidate scoring.

## Next Decision Needed

Manager should wire `scripts/compare_wrfout_grid.py` into the v0.14 B4 validation command set and define the first frozen tolerance manifest. The current smoke intentionally remains report-only.
