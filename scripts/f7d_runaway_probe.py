#!/usr/bin/env python
"""F7D diagnostic: trace max|w|, theta' bounds, max|p'|, and finiteness of the
idealized cases through the historical 80-100 s runaway window.  No clamps, no
masking -- pure observation of the operational dycore.

Records per-marker diagnostics inside a SINGLE jitted lax.scan (start_step is
NOT static), so the heavy operational-step graph compiles exactly once.

Run:  taskset -c 0-3 python -u scripts/f7d_runaway_probe.py --case warm_bubble
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
    build_density_current_numpy,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry


def _diag(carry, z_m) -> dict:
    """Per-step scalar diagnostics on the (k, 1, x) idealized slab."""
    state = carry.state
    w = state.w[:, 0, :]
    u = state.u[:, 0, :]
    theta = state.theta[:, 0, :]
    thp = theta - 300.0
    p = state.p_perturbation[:, 0, :]
    mu = state.mu_total[0, :]
    wpos = jnp.maximum(thp, 0.0)
    tot = jnp.sum(wpos)
    cz = jnp.where(tot > 1e-9, jnp.sum(wpos * z_m[:, None]) / jnp.maximum(tot, 1e-12), 0.0)
    finite = (
        jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(u))
        & jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(p))
        & jnp.all(jnp.isfinite(mu))
    ).astype(jnp.float64)
    return {
        "finite": finite,
        "max_abs_w": jnp.max(jnp.abs(w)),
        "max_abs_u": jnp.max(jnp.abs(u)),
        "theta_prime_min": jnp.min(thp),
        "theta_prime_max": jnp.max(thp),
        "max_abs_p_pert": jnp.max(jnp.abs(p)),
        "mass_total_pa": jnp.sum(mu),
        "pos_theta_cz": cz,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("warm_bubble", "density_current"), required=True)
    parser.add_argument("--end-seconds", type=float, default=200.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--epssm", type=float, default=None, help="override off-centering (WRF-cited; e.g. 0.5)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    case = build_warm_bubble_numpy() if args.case == "warm_bubble" else build_density_current_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    if args.epssm is not None:
        import dataclasses as _dc
        namelist = _dc.replace(namelist, epssm=float(args.epssm))
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    dt = float(case.dt_s)
    stride = max(1, int(round(args.stride_seconds / dt)))
    n_marks = int(round(args.end_seconds / args.stride_seconds))
    z_m = jnp.asarray(case.z_m)

    @jax.jit
    def run(carry):
        def stride_body(c, step):
            return _physics_boundary_step(c, namelist, step, run_radiation=False, debug=False), None

        def mark_body(c, mark_idx):
            base = mark_idx * stride
            steps = base + jnp.arange(stride, dtype=jnp.int32) + 1
            c, _ = jax.lax.scan(stride_body, c, steps)
            return c, _diag(c, z_m)

        marks = jnp.arange(n_marks, dtype=jnp.int32)
        final, diags = jax.lax.scan(mark_body, carry, marks)
        return final, diags, _diag(carry, z_m)

    print(f"case={args.case} dt={dt} device={setup.device} compiling+running...", flush=True)
    final, diags, diag0 = run(carry0)
    diags = jax.device_get(diags)
    diag0 = jax.device_get(diag0)

    trace = [{"second": 0.0, **{k: float(v) for k, v in diag0.items()}}]
    trace[0]["finite"] = bool(trace[0]["finite"] > 0.5)
    print(f"  t=    0.0  finite={trace[0]['finite']}  max|w|={trace[0]['max_abs_w']:.4e}  "
          f"theta'=[{trace[0]['theta_prime_min']:.3f},{trace[0]['theta_prime_max']:.3f}]", flush=True)
    for m in range(n_marks):
        second = (m + 1) * stride * dt
        rec = {"second": round(second, 3)}
        for k in diags:
            rec[k] = float(np.asarray(diags[k])[m])
        rec["finite"] = bool(rec["finite"] > 0.5)
        trace.append(rec)
        print(f"  t={second:7.1f}  finite={rec['finite']}  max|w|={rec['max_abs_w']:.4e}  "
              f"theta'=[{rec['theta_prime_min']:.3f},{rec['theta_prime_max']:.3f}]  "
              f"max|p'|={rec['max_abs_p_pert']:.4e}", flush=True)

    out = args.output or Path("proofs/f7d") / f"runaway_probe_{args.case}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case": args.case, "dt_s": dt, "device": setup.device,
                               "end_seconds": args.end_seconds, "stride_seconds": args.stride_seconds,
                               "trace": trace}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
