"""RE03 component check: port interp_sint_full(parent o3rad) vs pristine WRF interp_fcn force-down (d01->d02->d03)."""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.nesting.interp import build_sint_weights, interp_sint_full

NZ = 44
SHAPES = {"d01": (70, 120), "d02": (117, 267), "d03": (93, 111)}
EDGES = (("d01", "d02", 16, 16), ("d02", "d03", 92, 36))


def load(stamp):
    a = np.fromfile(Path(__file__).with_name(f"out_{stamp}.bin"), np.float32)
    out, pos = {}, 0
    for dom, (ny, nx) in SHAPES.items():  # Fortran (nx, nz, ny) -> numpy (ny, nz, nx) -> (nz, ny, nx)
        n = ny * NZ * nx
        out[dom] = a[pos:pos + n].reshape(ny, NZ, nx).transpose(1, 0, 2).copy()
        pos += n
    assert pos == a.size
    return out


if __name__ == "__main__":
    res = {}
    for stamp in sys.argv[1:]:
        o = load(stamp)
        for parent, child, ip, jp in EDGES:
            (pny, pnx), (cny, cnx) = SHAPES[parent], SHAPES[child]
            w = build_sint_weights(parent_grid_ratio=3, i_parent_start=ip, j_parent_start=jp, parent_ny=pny, parent_nx=pnx,
                                   child_ny=cny, child_nx=cnx)
            port = np.asarray(jax.jit(lambda f: interp_sint_full(f, w, parent_grid_ratio=3))(jnp.asarray(o[parent])))
            ref = o[child]
            d = np.abs(port.astype(np.float64) - ref)
            ulp = d / np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
            key = f"{stamp}_{parent}->{child}"
            res[key] = {"dtype": str(port.dtype), "shape": list(port.shape), "bitwise_equal_frac": float((port == ref).mean()),
                        "max_abs": float(d.max()), "max_rel": float((d / np.abs(ref)).max()), "max_ulp": float(ulp.max()),
                        "n_gt_1ulp": int((ulp > 1).sum())}
            print(key, res[key], flush=True)
    Path(Path(__file__).with_name("port_sint_check.json")).write_text(json.dumps(res, indent=1))
