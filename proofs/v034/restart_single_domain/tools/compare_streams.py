"""Compare two wrfout streams: identical file set, global attrs, every variable's dtype/shape/bytes."""
import hashlib, json, sys
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
control, candidate, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
names = sorted(p.name for p in control.glob("wrfout_*"))
cand_names = sorted(p.name for p in candidate.glob("wrfout_*"))
report = {"control_files": names, "candidate_files": cand_names, "file_set_equal": names == cand_names,
          "files": {}, "variables_compared": 0, "variables_different": []}
for name in names:
    if name not in cand_names:
        continue
    with Dataset(control / name) as a, Dataset(candidate / name) as b:
        a.set_auto_mask(False); b.set_auto_mask(False)
        attrs_equal = {k: a.getncattr(k) for k in a.ncattrs()} .__repr__() == {k: b.getncattr(k) for k in b.ncattrs()}.__repr__()
        keys_equal = list(a.variables) == list(b.variables)
        diff = []
        for key in a.variables:
            x = np.asarray(a.variables[key][...]); y = np.asarray(b.variables[key][...]) if key in b.variables else None
            report["variables_compared"] += 1
            if y is None or (x.dtype, x.shape, x.tobytes()) != (y.dtype, y.shape, y.tobytes()):
                diff.append(key)
        report["files"][name] = {"variables": len(a.variables), "global_attrs_equal": attrs_equal,
                                 "variable_names_equal": keys_equal, "different": diff}
        report["variables_different"] += [f"{name}:{k}" for k in diff]
report["byte_exact"] = (report["file_set_equal"] and not report["variables_different"]
                        and all(f["global_attrs_equal"] and f["variable_names_equal"] for f in report["files"].values()))
for extra in ("census.json",):
    a, b = control / extra, candidate / extra
    if a.exists() or b.exists():
        report[extra + "_equal"] = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
out.write_text(json.dumps(report, indent=1, default=str) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "files"}, default=str))
