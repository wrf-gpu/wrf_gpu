"""Shared helpers: run the JAX Grell ports on an oracle tile and compare."""

from __future__ import annotations

import numpy as np

import grell_oracle_io as gio

COMPARE_3D = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "CUGD_TTEN", "CUGD_QVTEN",
              "CUGD_QCTEN", "CUGD_TTENS", "CUGD_QVTENS", "GDC", "GDC2")
COMPARE_2D = ("RAINCV", "PRATEC", "DRV_RAINCV", "DRV_PRATEC", "HTOP", "HBOT", "EDT_OUT",
              "XMB_SHALLOW", "KTOP_DEEP", "K22_SHALLOW", "KBCON_SHALLOW", "KTOP_SHALLOW")
APR_NAMES = ("APR_GR", "APR_W", "APR_MC", "APR_ST", "APR_AS", "APR_CAPMA", "APR_CAPME",
             "APR_CAPMI")


def tile_to_jax_inputs(tile, dtype):
    import jax.numpy as jnp
    f3, f2 = tile["f3"], tile["f2"]
    kx = tile["kx"]

    def m(name, nlev=None):  # (nx, kx+1, ny) -> (nlev, ny, nx)
        a = f3[name].transpose(1, 2, 0)
        if nlev is not None:
            a = a[:nlev]
        return jnp.asarray(a, dtype)

    d2 = lambda name: jnp.asarray(f2[name].T, dtype)
    return dict(
        u=m("u", kx), v=m("v", kx), w=m("w"), t=m("t", kx), q=m("q", kx), p=m("p", kx),
        pi=m("pi", kx), rho=m("rho", kx), p8w=m("p8w"), rthften=m("rthften", kx),
        rqvften=m("rqvften", kx), rthraten=m("rthraten", kx), rthblten=m("rthblten", kx),
        rqvblten=m("rqvblten", kx), ht=d2("ht"), xland=d2("xland"), gsw=d2("gsw"),
        kpbl=jnp.asarray(np.rint(f2["kpbl"].T).astype(np.int32)),
    )


def jax_out_to_oracle_layout(out):
    """JAX (kx, ny, nx)/(ny, nx) -> oracle (nx, kx, ny)/(nx, ny)."""
    res = {}
    for k in COMPARE_3D:
        if k in out:
            res[k] = np.asarray(out[k]).transpose(2, 0, 1)
    for k in COMPARE_2D:
        if k in out:
            res[k] = np.asarray(out[k]).T
    if "APR" in out:
        apr = np.asarray(out["APR"])
        for i, k in enumerate(APR_NAMES):
            res[k] = apr[:, :, i].T
    if "XF_ENS" in out:
        res["XF_ENS"] = np.asarray(out["XF_ENS"]).transpose(1, 0, 2)
        res["PR_ENS"] = np.asarray(out["PR_ENS"]).transpose(1, 0, 2)
    return res


def compare(oracle, jaxo, keys=None):
    """Per-field max-abs and max-rel errors (rel to the field's max-abs)."""
    rows = {}
    keys = keys or [k for k in jaxo if k in oracle]
    for k in keys:
        a = np.asarray(oracle[k], np.float64)
        b = np.asarray(jaxo[k], np.float64)
        if a.shape != b.shape:
            rows[k] = dict(shape_mismatch=[a.shape, b.shape])
            continue
        d = np.abs(a - b)
        scale = max(np.max(np.abs(a)), 1e-300)
        rows[k] = dict(max_abs=float(d.max()), max_rel=float(d.max() / scale),
                       scale=float(scale), finite=bool(np.isfinite(b).all()),
                       n_bad=int((d > 1e-9 * scale + 1e-300).sum()))
    return rows
