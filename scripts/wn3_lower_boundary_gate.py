"""Compare aux4's device-held fields with CPU WRF history (first 24h, 3 cases)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from netCDF4 import Dataset

from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.io.lower_boundary import load_lower_boundary, lower_boundary_history
from wn3_inventory import CASES, ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "cpu"
    result = {"platform": "cpu", "gate": "exact SST/SEAICE/VEGFRA/ALBBCK; exact ocean TSK; initial ALBBCK excluded (WRF land initialization, before aux4)",
              "source": "<USER_HOME>/src/wrf_pristine/WRF", "cases": [], "pass": True}
    for case in CASES:
        source = ROOT / f"wg_{case}" / "run/run"
        nml = parse_namelist(source / "namelist.input")
        rows = []
        for number, dt in ((1, 54), (2, 18), (3, 6)):
            dom = f"d{number:02d}"
            history = sorted(source.glob(f"wrfout_{dom}_*"))[:25]
            start = datetime.strptime(history[0].name[11:], "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
            with Dataset(source / f"wrfinput_{dom}") as ds:
                water = np.asarray(ds["XLAND"][0]) > 1.5
                shape = water.shape
                initial_sst = np.asarray(ds["SST"][0])
            boundary = load_lower_boundary(source, nml, dom, run_start=start, dt_s=dt, shape=shape)
            assert boundary is not None
            for file in history:
                t = datetime.strptime(file.name[11:], "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
                own_step = int((t - start).total_seconds() / dt)
                fields = lower_boundary_history(boundary, own_step)
                with Dataset(file) as ds:
                    errors = {key: float(np.max(np.abs(np.asarray(ds[key][0]) - np.asarray(value))))
                              for key, value in fields.items() if own_step or key != "ALBBCK"}
                    errors["water_TSK"] = float(np.max(np.abs(np.asarray(ds["TSK"][0])[water] - np.asarray(fields["SST"])[water])))
                    stale_error = float(np.max(np.abs(np.asarray(ds["SST"][0])[water] - initial_sst[water])))
                    # Hash only the oracle fields actually scored; input identities are in W1.
                    h = hashlib.sha256()
                    for key in ("SST", "SEAICE", "VEGFRA", "ALBBCK", "TSK"):
                        h.update(np.asarray(ds[key][0]).tobytes())
                ok = all(v == 0 for v in errors.values())
                result["pass"] &= ok
                rows.append({"domain": dom, "file": file.name, "own_step": own_step, "errors": errors,
                             "static_SST_max_water_error_K": stale_error, "oracle_arrays_sha256": h.hexdigest(), "pass": ok})
        result["cases"].append({"case": case, "aux4_sha256": {f"d{d:02d}": sha256(source / f"wrflowinp_d{d:02d}") for d in (1, 2, 3)}, "frames": rows})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "frames": sum(len(c["frames"]) for c in result["cases"]),
                      "static_water_SST_max_error_K": max(f["static_SST_max_water_error_K"] for c in result["cases"] for f in c["frames"])}))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
