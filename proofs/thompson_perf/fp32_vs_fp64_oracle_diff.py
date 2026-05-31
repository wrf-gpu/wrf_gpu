"""Direct fp32-vs-fp64 Thompson output diff on the REAL WRF-oracle pre-state.

Runs the full Thompson kernel on the actual operational oracle columns twice
(fp64 work dtype and fp32 work dtype, same inputs) and reports the per-field
max abs/rel difference -- the honest "how much does fp32 perturb the
microphysics" number, on real data, independent of the oracle's masking.

Also reports the difference RELATIVE TO each field's distance from the WRF
oracle post-state, i.e. is the fp32 perturbation small compared to the kernel's
existing (fp64) distance to WRF?

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    taskset -c 0-3 python proofs/thompson_perf/fp32_vs_fp64_oracle_diff.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROOF = Path("proofs/thompson_perf")

# We run each precision in a fresh subprocess so the GPUWRF_THOMPSON_FP32 env is
# read at import/trace cleanly (the kernel reads it inside the traced body).
_CHILD = """
import os, json, sys
import numpy as np
import jax.numpy as jnp
from gpuwrf.physics.thompson_column import step_thompson_column_with_precip
from gpuwrf.validation.tier1_thompson import _load_f64_oracle_arrays, _columns_from_oracle, ORACLE_DIR
pre, post = _load_f64_oracle_arrays(ORACLE_DIR)
col = _columns_from_oracle(pre)
out, precip = step_thompson_column_with_precip(col, 18.0)
fields = ("qv","qc","qr","qi","qs","qg","Ni","Nr","T")
res = {f: np.asarray(getattr(out, f), dtype=np.float64).tolist() for f in fields}
res["_precip_rain"] = float(np.sum(np.asarray(precip["rain"], dtype=np.float64)))
res["_precip_snow"] = float(np.sum(np.asarray(precip["snow"], dtype=np.float64)))
res["_precip_graupel"] = float(np.sum(np.asarray(precip["graupel"], dtype=np.float64)))
np.savez(sys.argv[1], **{f: np.asarray(res[f]) for f in fields},
         precip_rain=res["_precip_rain"], precip_snow=res["_precip_snow"], precip_graupel=res["_precip_graupel"])
"""


def _run(fp32: bool, npz_out: str):
    env = {"PYTHONPATH": "src", "OMP_NUM_THREADS": "4",
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.4", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "GPUWRF_THOMPSON_FP32": "1" if fp32 else "0"}
    import os
    full = dict(os.environ)
    full.update(env)
    subprocess.run([sys.executable, "-c", _CHILD, npz_out], env=full, check=True,
                   cwd=str(Path.cwd()))


def main() -> int:
    PROOF.mkdir(parents=True, exist_ok=True)
    f64_npz = str(PROOF / "_diff_fp64.npz")
    f32_npz = str(PROOF / "_diff_fp32.npz")
    _run(False, f64_npz)
    _run(True, f32_npz)
    a = np.load(f64_npz + ".npz") if Path(f64_npz + ".npz").exists() else np.load(f64_npz)
    b = np.load(f32_npz + ".npz") if Path(f32_npz + ".npz").exists() else np.load(f32_npz)

    fields = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T")
    rec = {"scope": "fp32-vs-fp64 Thompson output diff on real WRF-oracle pre-state",
           "n_columns": int(a["qv"].shape[0]), "per_field": {}}
    for f in fields:
        x = a[f].astype(np.float64)
        y = b[f].astype(np.float64)
        d = np.abs(x - y)
        denom = np.abs(x) + (1e-30 if f in ("Ni", "Nr") else 1e-15)
        rec["per_field"][f] = {
            "max_abs": float(np.max(d)),
            "max_rel": float(np.max(d / denom)),
            "mean_abs": float(np.mean(d)),
            "fp64_field_max": float(np.max(np.abs(x))),
        }
    rec["precip_total_mm"] = {
        "fp64": {k.replace("precip_", ""): float(a[k]) for k in ("precip_rain", "precip_snow", "precip_graupel")},
        "fp32": {k.replace("precip_", ""): float(b[k]) for k in ("precip_rain", "precip_snow", "precip_graupel")},
    }
    (PROOF / "fp32_vs_fp64_oracle_diff.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
