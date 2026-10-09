"""RE03 repo fixture: pristine WRF o3rad force-down (d01 -> d02 -> d03) on real 0227, compact.

Inputs are PRISTINE parent fields cropped to the child footprint + 6-cell halo (interp_fcn_sint reads only cells around
the masked footprint) on a level subset (SINT is level-wise); expected child values are pristine, sampled on every 3rd
(d02) / 2nd (d03) row and column plus the three outermost rows/columns.  The builder proves crop == full-grid for the
port bitwise and records the full-resolution port-vs-pristine ulp numbers.
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.nesting.interp import build_sint_weights, interp_sint_full

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from port_sint_check import EDGES, SHAPES, load  # noqa: E402

LEVELS = np.asarray(list(range(0, 44, 4)) + [43], np.int32)
STEP = {"d02": 3, "d03": 2}
STAMP = "2026-02-28_12:00:00"


def port(parent, ip, jp, cny, cnx):
    pny, pnx = parent.shape[-2:]
    w = build_sint_weights(parent_grid_ratio=3, i_parent_start=ip, j_parent_start=jp, parent_ny=pny, parent_nx=pnx,
                           child_ny=cny, child_nx=cnx)
    return np.asarray(jax.jit(lambda f: interp_sint_full(f, w, parent_grid_ratio=3))(jnp.asarray(parent)))


def sample_mask(ny, nx, step):
    m = np.zeros((ny, nx), bool)
    m[::step, ::step] = True
    m[:3, :] = m[-3:, :] = m[:, :3] = m[:, -3:] = True
    return m


def main(out):
    o = load(STAMP)
    fx = {"levels": LEVELS, "ratio": np.int32(3)}
    meta = {"stamp": STAMP, "case": "wg_20260227_18z_a1 (START 2026-02-28_00Z)", "edges": {}}
    for parent, child, ip, jp in EDGES:
        (cny, cnx) = SHAPES[child]
        i0, j0 = ip - 6, jp - 6                              # 1-based start of the crop
        i1, j1 = ip + (cnx - 1) // 3 + 6, jp + (cny - 1) // 3 + 6
        crop = o[parent][:, j0 - 1:j1, i0 - 1:i1]
        full_port = port(o[parent], ip, jp, cny, cnx)
        crop_port = port(crop, ip - i0 + 1, jp - j0 + 1, cny, cnx)
        assert np.array_equal(full_port, crop_port), "crop changes the port SINT"
        ref = o[child]
        ulp = np.abs(full_port.astype(np.float64) - ref) / np.spacing(np.abs(ref)).astype(np.float64)
        mask = sample_mask(cny, cnx, STEP[child])
        jj, ii = np.nonzero(mask)
        fx[f"{child}_parent_crop"] = crop[LEVELS]
        fx[f"{child}_ip_jp"] = np.asarray([ip - i0 + 1, jp - j0 + 1], np.int32)
        fx[f"{child}_shape"] = np.asarray([cny, cnx], np.int32)
        fx[f"{child}_j"] = jj.astype(np.int32)
        fx[f"{child}_i"] = ii.astype(np.int32)
        fx[f"{child}_expected"] = ref[LEVELS][:, jj, ii]
        meta["edges"][f"{parent}->{child}"] = {"namelist_i_j_parent_start": [ip, jp], "crop_1based_i0_j0": [i0, j0],
                                                "full_res_max_ulp": float(ulp.max()),
                                                "full_res_bitwise_frac": float((full_port == ref).mean()),
                                                "n_sampled_cells": int(mask.sum())}
    np.savez_compressed(out, **fx)
    meta["sha256"] = __import__("hashlib").sha256(Path(out).read_bytes()).hexdigest()
    Path(out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), Path(out).stat().st_size)


if __name__ == "__main__":
    main(sys.argv[1])
