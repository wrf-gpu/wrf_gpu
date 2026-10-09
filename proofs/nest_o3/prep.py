"""RE03 input: d01 hydrostatic p_hyd (phy_prep REAL recursion) + XLAT from CPU-WRF 0227 history at one frame."""
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
f4 = np.float32


def p_hyd_field(ds):
    """phy_prep (module_big_step_utilities_em.F:4943-4970) vectorised, REAL, same association as RE01 extract.hydrostatic."""
    qtot = np.zeros(ds["QVAPOR"].shape[1:], f4)
    for name in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
        qtot = (qtot + np.asarray(ds[name][0], f4)).astype(f4)
    c1, c2, dnw = (np.asarray(ds[n][0], f4) for n in ("C1H", "C2H", "DNW"))
    mut = (np.asarray(ds["MU"][0], f4) + np.asarray(ds["MUB"][0], f4)).astype(f4)
    nz = qtot.shape[0]
    pw = np.zeros((nz + 1,) + mut.shape, f4)
    pw[nz] = f4(np.asarray(ds["P_TOP"][0]))
    for k in range(nz - 1, -1, -1):
        pw[k] = (pw[k + 1] - ((((f4(1) + qtot[k]).astype(f4) * ((c1[k] * mut).astype(f4) + c2[k]).astype(f4)).astype(f4))
                              * dnw[k]).astype(f4)).astype(f4)
    return (f4(0.5) * (pw[:-1] + pw[1:])).astype(f4)


def main(stamp, out):
    with Dataset(CPU / f"wrfout_d01_{stamp}") as d:
        ph = p_hyd_field(d)
        lat = np.asarray(d["XLAT"][0], f4)
        julday, gmt = int(d.getncattr("JULDAY")), float(d.getncattr("GMT"))
        xtime_min = float(np.asarray(d["XTIME"][0]))
    nz, ny1, nx1 = ph.shape
    julian = (julday - 1) + (gmt + xtime_min / 60.0) / 24.0
    shapes = {}
    for dom in ("d02", "d03"):
        with Dataset(CPU / f"wrfout_{dom}_{stamp}") as d:
            shapes[dom] = d["T"].shape[2:]
    hdr = np.asarray([nx1, ny1, nz, shapes["d02"][1], shapes["d02"][0], shapes["d03"][1], shapes["d03"][0],
                      16, 16, 92, 36, 3], np.int32)
    with open(out, "wb") as f:
        hdr.tofile(f)
        np.asarray([julday], np.int32).tofile(f)
        np.asarray([julian], np.float32).tofile(f)
        np.ascontiguousarray(lat).tofile(f)                       # Fortran (nx1, ny1)
        np.ascontiguousarray(ph.transpose(1, 0, 2)).tofile(f)     # Fortran (nx1, nz, ny1)
    print(stamp, "d01", nx1, ny1, nz, "d02", shapes["d02"], "d03", shapes["d03"], "julian", julian)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
