"""Read staged history only; preserve exact negative counts by boundary ring.

This is an offline diagnostic, never a model operator or a positivity policy.
One horizontal layer at a time bounds memory; samples do not bound extrema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np


def inventory(path, fields):
    path = Path(path).resolve()
    # ALISIOS payload must first be copied through its guarded facade.
    protected = ("<DATA_ROOT>/alisios", "<DATA_ROOT>/server/work/alisios", "<DATA_ROOT>/server/work/src/alisios")
    if any(path == Path(root) or Path(root) in path.parents for root in protected):
        raise ValueError("use a facade-staged copy, never an ALISIOS payload root")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    rows = []
    with netCDF4.Dataset(path) as ds:
        for name in fields:
            if name not in ds.variables:
                raise ValueError(f"missing requested field {name}")
            var = ds.variables[name]
            if var.dimensions != ("Time", "bottom_top", "south_north", "west_east"):
                raise ValueError(f"unexpected dimensions for {name}: {var.dimensions}")
            nt, nz, ny, nx = var.shape
            y, x = np.ogrid[:ny, :nx]
            ring = np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))
            for t in range(nt):
                counts = np.zeros(int(ring.max()) + 1, dtype=np.int64)
                minimum, index = None, None
                for k in range(nz):
                    layer = np.asanyarray(var[t, k])
                    if np.ma.isMaskedArray(layer) and np.ma.getmaskarray(layer).any():
                        raise ValueError(f"masked/fill data in {name} frame {t} level {k}")
                    layer = np.asarray(layer)
                    if not np.isfinite(layer).all():
                        raise ValueError(f"nonfinite data in {name} frame {t} level {k}")
                    negative = layer < 0
                    counts += np.bincount(ring[negative], minlength=counts.size)
                    if negative.any():
                        j, i = np.unravel_index(layer.argmin(), layer.shape)
                        value = float(layer[j, i])
                        if minimum is None or value < minimum:
                            minimum, index = value, [k, int(j), int(i)]
                rows.append(dict(field=name, frame=t, shape=[nz, ny, nx],
                                 negative_count=int(counts.sum()),
                                 negative_by_ring=counts.tolist(),
                                 minimum=minimum, minimum_index_kji=index,
                                 interior_ge_5=int(counts[5:].sum())))
    return dict(path=str(path), sha256=digest.hexdigest(), rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--fields", nargs="+", default=["QCLOUD", "QRAIN", "QC_RAD"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = dict(schema="v45-bdyneg.ring-inventory.v1",
                  files=[inventory(p, args.fields) for p in args.files])
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
