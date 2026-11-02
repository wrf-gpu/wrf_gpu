"""Retain exact boundary operands from already staged single-frame histories.

Strips keep original dimensions and coordinates. They are not a cropped WRF
domain and do not contain the internal boundary records or RK stage inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np

FIELDS = ("QCLOUD", "QRAIN", "QC_RAD", "QVAPOR", "QICE", "QSNOW", "QGRAUP",
          "QNICE", "QNRAIN", "U", "V", "W", "MU", "MUB", "C1H", "C2H",
          "C1F", "C2F", "MAPFAC_M", "MAPFAC_MX", "MAPFAC_MY",
          "T", "P", "PB", "PH", "PHB")


def extract(source, destination, original):
    source = Path(source).resolve()
    if any(Path(p) in source.parents for p in
           ("<DATA_ROOT>/alisios", "<DATA_ROOT>/server/work/alisios", "<DATA_ROOT>/server/work/src/alisios")):
        raise ValueError("extract only from a facade-staged copy")
    arrays, fields = {}, {}
    with netCDF4.Dataset(source) as ds:
        assert len(ds.dimensions["Time"]) == 1, "single-frame histories required"
        attributes = {n: str(getattr(ds, n)) for n in
                      ("DT", "GRID_ID", "PARENT_GRID_RATIO", "SIMULATION_START_DATE") if n in ds.ncattrs()}
        arrays["Times"] = np.asarray(ds.variables["Times"][:])
        arrays["XTIME"] = np.asarray(ds.variables["XTIME"][:])
        for name in FIELDS:
            var = ds.variables[name]
            value = np.asanyarray(var[0])
            if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
                raise ValueError(f"masked {name}")
            value = np.asarray(value)
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite {name}")
            info = dict(shape=list(value.shape), dimensions=list(var.dimensions[1:]), dtype=value.dtype.str)
            if value.ndim < 2:
                arrays[name] = value
            else:
                width = 9 if name in ("U", "V") else 8
                info["width"] = width
                for side, strip in zip(("W", "E", "S", "N"),
                                       (value[..., :width], value[..., -width:],
                                        value[..., :width, :], value[..., -width:, :])):
                    arrays[name + "__" + side] = strip
                if name in ("QCLOUD", "QRAIN", "QC_RAD"):
                    indices = np.argwhere(value < 0)
                    arrays[name + "__negative_kji"] = indices
                    arrays[name + "__negative_values"] = value[tuple(indices.T)]
            fields[name] = info
    destination = Path(destination)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    # Read every saved array back before releasing a raw copy; no sampled byte proof.
    with np.load(destination, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for name, value in arrays.items():
            assert saved[name].dtype == value.dtype
            assert saved[name].shape == value.shape
            assert saved[name].tobytes() == value.tobytes(), name
    return dict(original=original, staged_source=str(source), attributes=attributes,
                fields=fields, arrays=len(arrays), output=str(destination),
                bytes=destination.stat().st_size,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                all_array_byte_readback=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-map", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records = [extract(path, output / (Path(original).name + ".npz"), original)
               for original, path in json.loads(Path(args.stage_map).read_text()).items()]
    (output / "MANIFEST.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(files=len(records), bytes=sum(r["bytes"] for r in records),
                          arrays=sum(r["arrays"] for r in records), readback="ALL BYTES EXACT")))
