# W1a separately pinned CWP evaluator

The engine is the byte-fixed v0.4.5 bundle with SHA256
`32ea753a8080ea271515e71845fba02ef63a9f4c7915915ddbebdbc45957c2f7`.
The CWP correction is an offline science operator, separate from that engine.
The manager must merge it after cr-w44-rc READY and ALISIOS must explicitly bind
this tuple before its first W1a value read. This receipt does not grant admission.

| Binding | Exact value |
| --- | --- |
| Evaluator Git commit | `adb4c33b8caa6e1cbb480660b8d2be06a94eca22` |
| `src/gpuwrf/optimize/cloud.py` SHA256 | `a630ae06aff0a76d3907af30446554cfdb81c2396618aecec337519289bc1645` |
| `src/gpuwrf/physics/surface_constants.py` SHA256 | `970cf54ab0b1eed596e48ae3b5047a5972b6bc7dcb707fb1e6a3eb332d0833a6` |
| Synthetic gate at that commit | `proofs/v45_cloudscore/CWP_CONSTANTS_GATE.json` |
| Synthetic gate SHA256 | `6b5654a4941c08bcc5c589a8c310d09e8cd406cbf6b6d3e596051725c9cfb360` |
| Synthetic revert receipt | `proofs/v45_cloudscore/cwp_constants/ROOT_REVERSION.json` |
| Synthetic revert receipt SHA256 | `9c1c6199ff0abcca941063ad4c59c8e2d54793559b8866ab09aeebc35160e477` |

The tested runtime source was commit `8a46f7b9aa3172e2527c710da572c4de5c3c962f`;
the evaluator pin above adds only receipts and notes. Synthetic original-WRF
control: p=80000 Pa, theta=290 K, qv=0.01 kg/kg, independent exponent 2/7.
Legacy/reference density ratio was 1.0000317511516545; corrected ratio is 1.0,
as is corrected CWP/reference. The single old-exponent revert produces one
genuine assertion failure, zero errors/skips. Affected suite: 101 PASS; full
ROOT collection: 21791 tests, zero errors. No W1a model or truth values were read
by this lane. Warm component cost is unchanged within noise (3.591/3.572 ms);
import/startup cost is unmeasured.

## Loading the operator independently of the engine

ALISIOS can unpack **only `src` from the exact evaluator commit** into a new,
private evaluator directory, outside the engine bundle, and use its existing
CPU Python environment with the optimize dependencies. For example, with an
empty `/path/to/evaluator` and the source repository available:

```bash
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu timeout 60 taskset -c 12,28 nice -n 19 bash -o pipefail -c 'git -C <USER_HOME>/src/wrf_gpu2 archive adb4c33b8caa6e1cbb480660b8d2be06a94eca22 src | tar -xf - -C /path/to/evaluator'
```

Run from that evaluator directory, with its `src` first on `PYTHONPATH`.
Do not invoke the bundle's Python consumer for offline evaluation: that loads
the bundle's older operator. The bundle remains the engine for model generation.

This model-only invocation reads an existing WRF history record and writes CWP
in g/m² to a **new** NPZ outside the bundle. Replace the three example arguments
(model file, zero-based record index, new output file). It reads no truth:

```bash
cd /path/to/evaluator
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/path/to/evaluator/src timeout 120 taskset -c 12,28 nice -n 19 <USER_HOME>/miniconda3/bin/python - /path/to/wrfout_d03 0 /path/to/new-model-cwp.npz <<'PY'
import hashlib
from pathlib import Path
import sys
import numpy as np

root = Path.cwd().resolve()
expected = {
    "src/gpuwrf/optimize/cloud.py": "a630ae06aff0a76d3907af30446554cfdb81c2396618aecec337519289bc1645",
    "src/gpuwrf/physics/surface_constants.py": "970cf54ab0b1eed596e48ae3b5047a5972b6bc7dcb707fb1e6a3eb332d0833a6",
}
for name, digest in expected.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"evaluator pin mismatch: {name}")

from gpuwrf.optimize import cloud
from gpuwrf.physics import surface_constants
from gpuwrf.io.netcdf_lock import Dataset
if (Path(cloud.__file__).resolve() != root/"src/gpuwrf/optimize/cloud.py" or
        Path(surface_constants.__file__).resolve() != root/"src/gpuwrf/physics/surface_constants.py"):
    raise RuntimeError("import did not use the pinned evaluator")
with Dataset(sys.argv[1], "r") as ds:
    columns = cloud.wrf_columns(ds, int(sys.argv[2]))
with open(sys.argv[3], "xb") as out:
    np.savez(out, cwp_g_m2=columns,
             evaluator_commit="adb4c33b8caa6e1cbb480660b8d2be06a94eca22",
             engine_bundle_sha256="32ea753a8080ea271515e71845fba02ef63a9f4c7915915ddbebdbc45957c2f7")
PY
```

The output metadata records the intended engine binding; the engine's existing
run/manifest receipt must establish that the supplied model file actually came
from that bundle. It is not inferred from the NetCDF filename.

For ALISIOS-owned scoring **after** its model-frame receipt, truth admission and
science-operator binding, the same pinned environment exposes the existing API:

```python
config = cloud.load_config(admitted_cloud_config, resolve_admitted_facade_path)
report, prediction, observation, keys = cloud.score_case(
    admitted_model_paths, config, admitted_issue_utc, admitted_horizon_hours)
```

`load_config` and `score_case` read truth; this lane has not invoked that path
with real W1a data. Keep the frozen threshold (10 g/m²), species, exact UTC slots,
daylight mask, regions and footprint contract. The model-only NPZ is a diagnostic;
pass the original model histories to `score_case` to preserve geometry and the
separate prognostic/PBL/active-radiation channels. No interpolation or diagnostic
substitution is introduced. This correction does not qualify a missing T3
sensitivity, a correlation contract, a forecast, or an engine bundle.
