#!/usr/bin/env python3
"""Distil a full CPU-WRF wrfout history into a COMPACT, shippable reference.

The CPU reference for the Switzerland equivalence test only needs the 10 fields
the comparator scores (T2/U10/V10/PSFC/RAINNC + U/V/W/T/QVAPOR) plus the
coordinate/time metadata the reader uses. A full hourly wrfout set is ~80 MB
per file (~2 GB for 24 h); the distilled set is small enough to publish as a
checksummed tarball (zlib-compressed NetCDF, only the scored fields).

The distilled files keep the SAME ``wrfout_d01_<timestamp>`` names and the same
variable shapes, so ``equivalence_switzerland_compare.py`` reads them with no
special-casing.

USAGE
=====
    python scripts/make_compact_reference.py \
        --src <DATA_ROOT>/wrf_gpu_switzerland/run_cpu \
        --dst <DATA_ROOT>/wrf_gpu_switzerland/cpu_reference_compact \
        --domain d01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

KEEP_VARS = ("Times", "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
             "T2", "U10", "V10", "PSFC", "RAINNC", "RAINC",
             "U", "V", "W", "T", "QVAPOR", "PH", "PHB", "HGT")


def distil(src: Path, dst: Path, domain: str) -> int:
    from netCDF4 import Dataset

    files = sorted(src.glob(f"wrfout_{domain}_*"))
    if not files:
        print(f"error: no wrfout_{domain}_* in {src}", file=sys.stderr)
        return 2
    dst.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    for f in files:
        out = dst / f.name
        with Dataset(f) as ds, Dataset(out, "w", format="NETCDF4") as od:
            # global attrs (real.exe/WRF metadata the reader may consult)
            od.setncatts({k: ds.getncattr(k) for k in ds.ncattrs()})
            present = [v for v in KEEP_VARS if v in ds.variables]
            # dimensions used by the kept vars
            needed_dims = set()
            for v in present:
                needed_dims.update(ds.variables[v].dimensions)
            for d in needed_dims:
                dim = ds.dimensions[d]
                od.createDimension(d, None if dim.isunlimited() else dim.size)
            for v in present:
                var = ds.variables[v]
                comp = dict(zlib=True, complevel=4) if var.dtype.kind == "f" else {}
                ov = od.createVariable(v, var.dtype, var.dimensions, **comp)
                ov.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                ov[:] = var[:]
        total_in += f.stat().st_size
        total_out += out.stat().st_size
    print(f"distilled {len(files)} files: {total_in/1e6:.1f} MB -> {total_out/1e6:.1f} MB "
          f"({100*total_out/max(total_in,1):.0f}%) into {dst}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Full CPU-WRF run dir")
    p.add_argument("--dst", type=Path, required=True, help="Output compact reference dir")
    p.add_argument("--domain", default="d01")
    a = p.parse_args(argv)
    return distil(a.src, a.dst, a.domain)


if __name__ == "__main__":
    raise SystemExit(main())
