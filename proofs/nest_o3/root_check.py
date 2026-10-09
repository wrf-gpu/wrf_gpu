"""RE03 root check: port wrf_cam_ozone_profile on d01 (the field the root stores in o3rad) vs pristine WRF
oznini/ozn_time_int/ozn_p_int on every d01 cell, same CPU-WRF history p_hyd/XLAT/time (in_<stamp>.bin)."""
import json
import sys
from pathlib import Path

import jax
import numpy as np

from gpuwrf.physics.wrf_cam_ozone import wrf_cam_ozone_profile

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from port_sint_check import load  # noqa: E402

XTIME_MIN = {"2026-02-28_12:00:00": 720.0, "2026-03-01_00:00:00": 1440.0}  # START 2026-02-28_00Z, GMT 0


def read_input(stamp):
    raw = (HERE / f"in_{stamp}.bin").read_bytes()
    hdr = np.frombuffer(raw[:48], np.int32)
    nx1, ny1, nz = (int(v) for v in hdr[:3])
    julday = int(np.frombuffer(raw[48:52], np.int32)[0])
    pos = 56
    lat = np.frombuffer(raw[pos:pos + 4 * nx1 * ny1], np.float32).reshape(ny1, nx1)
    pos += 4 * nx1 * ny1
    ph = np.frombuffer(raw[pos:pos + 4 * nx1 * ny1 * nz], np.float32).reshape(ny1, nz, nx1)
    assert pos + 4 * nx1 * ny1 * nz == len(raw)
    return julday, lat, np.ascontiguousarray(ph.transpose(0, 2, 1))  # (ny, nx, nz) bottom-to-top


if __name__ == "__main__":
    assert jax.devices()[0].platform == "cpu"
    res = {}
    for stamp in sys.argv[2:]:
        julday, lat, ph = read_input(stamp)
        port = np.asarray(jax.jit(lambda la, p: wrf_cam_ozone_profile(
            la, p, julian_day_1based=julday, utc_minute=XTIME_MIN[stamp]))(lat, ph))
        ref = np.moveaxis(load(stamp)["d01"], 0, -1)
        ulp = np.abs(port.astype(np.float64) - ref) / np.spacing(np.abs(ref)).astype(np.float64)
        res[stamp] = {"julday": julday, "cells": int(ref.size), "bitwise_frac": float((port == ref).mean()),
                      "max_ulp": float(ulp.max()), "max_rel": float((np.abs(port.astype(np.float64) - ref) / ref).max())}
        print(stamp, res[stamp])
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
