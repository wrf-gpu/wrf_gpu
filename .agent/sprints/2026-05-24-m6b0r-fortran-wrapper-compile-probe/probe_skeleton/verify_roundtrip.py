"""verify_roundtrip.py — read /tmp/wrapprobe_test.h5 written by the Fortran probe.

Fails non-zero if shape, dtype, byte order, attributes, or values disagree.

Layout note (load-bearing for the M6B0-R worker):
  HDF5 preserves declared dataspace dims [d1, d2, d3] and writes the underlying
  memory contiguously.  Fortran is column-major (first index fastest), numpy is
  row-major (last index fastest), so a Fortran array declared arr(NX,NY,NZ)
  appears in numpy as a (NX,NY,NZ)-shaped buffer where numpy[a,b,c] equals
  fortran arr(c+1, b+1, a+1).  See the explicit reversed-axis comparison below.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np


PATH = "/tmp/wrapprobe_test.h5"
NX, NY, NZ = 4, 4, 4


def expected_array_numpy_view() -> np.ndarray:
    """Returns the array as h5py-on-disk will present it (axes reversed vs Fortran)."""
    arr = np.empty((NX, NY, NZ), dtype=np.float64)
    for a in range(NX):
        for b in range(NY):
            for c in range(NZ):
                # fortran (i,j,k) = (c+1, b+1, a+1)
                arr[a, b, c] = 100.0 * (c + 1) + 10.0 * (b + 1) + (a + 1)
    return arr


def _attr_to_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace").rstrip("\x00 ").strip()
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        flat = value.ravel()
        if flat.size == 1:
            return flat[0].decode(errors="replace").rstrip("\x00 ").strip()
    if isinstance(value, np.ndarray):
        return str(value.item()) if value.size == 1 else str(value)
    return str(value).strip()


def _attr_to_int(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.ravel()[0])
    return int(value)


def check(label: str, ok: bool) -> None:
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        check.failed = True  # type: ignore[attr-defined]


check.failed = False  # type: ignore[attr-defined]


def main() -> int:
    print(f"Opening {PATH}")
    with h5py.File(PATH, "r") as f:
        check("dataset 'data' present", "data" in f)
        dset = f["data"]
        check(
            f"shape == ({NX},{NY},{NZ}) got {dset.shape}",
            tuple(dset.shape) == (NX, NY, NZ),
        )
        check(f"dtype == float64 got {dset.dtype}", dset.dtype == np.float64)
        check(
            f"byteorder native/little got {dset.dtype.byteorder!r}",
            dset.dtype.byteorder in ("=", "<", "|"),
        )

        on_disk = dset[...]
        expected = expected_array_numpy_view()
        equal = np.array_equal(on_disk, expected)
        check(f"values bitwise equal to fortran fill (reversed-axis view)", equal)
        if not equal:
            diff = np.abs(on_disk - expected)
            print(f"        max abs diff = {diff.max()}")
            mismatches = np.argwhere(diff > 0)
            if mismatches.size:
                print(f"        first mismatch idx = {tuple(int(x) for x in mismatches[0])}")

        attrs = dict(dset.attrs)
        print(f"  attrs: {sorted(attrs)}")
        check(f"attr name == 'probe_field' (got {_attr_to_text(attrs.get('name'))!r})",
              _attr_to_text(attrs.get("name")) == "probe_field")
        check(f"attr units == 'arbitrary' (got {_attr_to_text(attrs.get('units'))!r})",
              _attr_to_text(attrs.get("units")) == "arbitrary")
        check(f"attr stagger == 'C' (got {_attr_to_text(attrs.get('stagger'))!r})",
              _attr_to_text(attrs.get("stagger")) == "C")
        check("attr rkstage == 1", _attr_to_int(attrs.get("rkstage", -99)) == 1)
        check("attr acstep == 0", _attr_to_int(attrs.get("acstep", -99)) == 0)
        check("attr schema_version == 1", _attr_to_int(attrs.get("schema_version", -99)) == 1)

    if check.failed:  # type: ignore[attr-defined]
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS — Fortran->HDF5->h5py round-trip clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
