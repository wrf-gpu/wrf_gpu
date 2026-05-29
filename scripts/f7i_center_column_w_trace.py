#!/usr/bin/env python
"""F7I center-column 2Delta-z mode tracer.

Runs the warm bubble and records, per OUTPUT step, the bubble-center-column
vertical profile of w(k), ph'(k), theta'(k) plus a 2Delta-z mode indicator:

    zz_energy(t) = sum_k ( w(k) - 0.5*(w(k-1)+w(k+1)) )^2     (curvature energy)
    sign_alt(t)  = sum_k 1[ w(k)*w(k+1) < 0 ] / (nfaces-1)    (alternating fraction)

A 2Delta-z mode shows growing zz_energy and sign_alt -> ~1 localized at the
bubble center column.  We also dump the full center-column w profile at a few
times so the alternating-sign structure is directly visible.

Run:  PYTHONPATH=src taskset -c 0-3 python -u scripts/f7i_center_column_w_trace.py \
        --steps 1800 --stride 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("proofs/f7i/center_column_w_trace.json"))
    args = parser.parse_args(argv)

    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    # bubble center column index (x=10000m, dx=250m)
    ic = int(round(10000.0 / 250.0))
    nx = case.nx
    ic = min(max(ic, 0), nx - 1)
    print(f"center column ic={ic} (x={ (ic+0.5)*250.0 }m), nz={case.nz}", flush=True)

    n_marks = args.steps // args.stride
    stride = args.stride

    def diag(carry):
        s = carry.state
        wcol = s.w[:, 0, ic]              # (nz+1,) center-column w on faces
        phcol = s.ph_perturbation[:, 0, ic]
        thcol = s.theta[:, 0, ic] - 300.0
        # 2dz curvature energy on interior faces
        curv = wcol[1:-1] - 0.5 * (wcol[:-2] + wcol[2:])
        zz_energy = jnp.sum(curv * curv)
        # alternating-sign fraction across adjacent faces
        prod = wcol[:-1] * wcol[1:]
        sign_alt = jnp.mean((prod < 0.0).astype(jnp.float64))
        finite = (jnp.all(jnp.isfinite(s.w)) & jnp.all(jnp.isfinite(s.theta))).astype(jnp.float64)
        return {
            "finite": finite,
            "max_abs_w_center": jnp.max(jnp.abs(wcol)),
            "max_abs_w_global": jnp.max(jnp.abs(s.w)),
            "zz_energy_center": zz_energy,
            "sign_alt_center": sign_alt,
            "wcol": wcol,
            "phcol": phcol,
            "thcol": thcol,
        }

    @jax.jit
    def run(carry):
        def body(c, m):
            steps = m * stride + jnp.arange(stride, dtype=jnp.int32) + 1
            c, _ = jax.lax.scan(
                lambda cc, st: (_physics_boundary_step(cc, namelist, st, run_radiation=False), None), c, steps
            )
            return c, diag(c)

        marks = jnp.arange(n_marks, dtype=jnp.int32)
        final, d = jax.lax.scan(body, carry, marks)
        return final, d, diag(carry)

    print(f"tracer warm_bubble steps={args.steps} stride={stride} compiling...", flush=True)
    _final, d, d0 = run(carry0)
    d = jax.device_get(d)
    d0 = jax.device_get(d0)

    def rec_at(step, t, src, idx=None):
        get = (lambda k: float(np.asarray(src[k])) ) if idx is None else (lambda k: float(np.asarray(src[k])[idx]))
        getv = (lambda k: np.asarray(src[k]).tolist()) if idx is None else (lambda k: np.asarray(src[k])[idx].tolist())
        return {
            "step": step,
            "second": round(t, 3),
            "finite": get("finite") > 0.5,
            "max_abs_w_center": get("max_abs_w_center"),
            "max_abs_w_global": get("max_abs_w_global"),
            "zz_energy_center": get("zz_energy_center"),
            "sign_alt_center": get("sign_alt_center"),
            "wcol": getv("wcol"),
            "phcol": getv("phcol"),
            "thcol": getv("thcol"),
        }

    trace = [rec_at(0, 0.0, d0)]
    for m in range(n_marks):
        trace.append(rec_at((m + 1) * stride, (m + 1) * stride * float(case.dt_s), d, idx=m))

    for r in trace:
        print(
            f"  step={r['step']:5d} t={r['second']:7.1f} fin={r['finite']} "
            f"max|w|cen={r['max_abs_w_center']:.3e} max|w|glob={r['max_abs_w_global']:.3e} "
            f"zz_energy={r['zz_energy_center']:.3e} sign_alt={r['sign_alt_center']:.3f}",
            flush=True,
        )

    # print full center column w at a few key times
    for r in trace:
        if r["step"] in (0, trace[1]["step"], trace[len(trace) // 2]["step"], trace[-1]["step"]):
            wc = r["wcol"]
            print(f"  --- center w profile @t={r['second']}s: " + " ".join(f"{v:+.2f}" for v in wc), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"case": "warm_bubble", "center_col": ic, "trace": trace}, indent=2))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
