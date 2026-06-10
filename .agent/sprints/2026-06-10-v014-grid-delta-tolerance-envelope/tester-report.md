# Tester Report: V0.14 Grid-Delta Tolerance Envelope

Decision: PASS.

Required gates run by the worker and re-run or independently checked by the
manager:

```bash
python -m json.tool proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json >/tmp/v014_grid_delta_tolerance_candidate.validated.json
python scripts/build_grid_delta_atlas.py --help >/tmp/v014_build_grid_delta_atlas_help.txt
python scripts/compare_wrfout_grid.py --help >/tmp/v014_compare_wrfout_grid_help.txt
git diff --check
LC_ALL=C rg -n "[^\x00-\x7F]" proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json proofs/v014/grid_delta_atlas/TOLERANCE_MANIFEST_CANDIDATE.md .agent/reviews/2026-06-10-v014-grid-delta-tolerance-envelope.md || true
```

Manager consistency check:

```bash
python - <<'PY'
import json
d=json.load(open('proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json'))
fields=d.get('fields', {})
static=[]
for group in d.get('static_exactness_groups', {}).values():
    if isinstance(group, list):
        static += group
missing=sorted(set(static)-set(fields))
report=[k for k,v in fields.items() if v.get('gate')=='critical_report_only']
hard=[k for k,v in fields.items() if v.get('gate')=='hard_release_gate']
assert not missing
assert set(report) == {'P','PH','MU','RAINC'}
assert set(hard) == {'T2','U10','V10','PSFC','RAINNC','T','U','V','W','QVAPOR'}
PY
```

Optional offline parser smokes:

- One-hour old paired wrfout smoke over `T2/U10/V10` passed and confirmed
  `--tolerance-json` parser compatibility.
- Twenty-four-hour old red hard-field smoke over the ten hard fields returned
  `FAIL_TOLERANCE`, with failures in `PSFC`, `T`, `U`, `U10`, `V`, and `V10`.

No GPU, TOST, Switzerland, model forecast, or source test was run.
