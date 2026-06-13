#!/usr/bin/env python
"""v0.15 S1 host-removal — can XLA command buffers capture the EDMF-shaped loop nest?

Microbench mimicking the production launch structure that defeats CUDA-graph
capture today: a vmapped ``lax.scan`` over NZ-2 levels whose body runs the
50-iteration ``lax.fori_loop`` condensation fixed point, on (B, NUP) fp64
leaves (B=16384 columns x NUP=8 plumes = the d01 EDMF shape).  ~42x50 = 2100
dependent micro-stages per call, each a ~us kernel: host-launch-bound by
construction, exactly like the real step.

Variants:
  nested    scan(42){ fori(50) }              -- today's EDMF lowering
  unroll16  scan(42){ 16x Python-unrolled }   -- the WRF-faithful niter cap shape
  flat      fully unrolled scan + 16x body    -- zero while ops (capture trivially)
  tridiag   nested + lax.linalg.tridiagonal_solve -- custom-call capture probe

For each variant: warm, then median ms/call over reps; plus introspection of the
post-optimization HLO: #while ops and #command_buffer computations, and whether
a while op ended up INSIDE a command_buffer computation (the decisive bit).

XLA_FLAGS is set by the caller; this script just records it.
Usage: probe_cmdbuf_capture.py --variant nested|unroll16|flat|tridiag --tag NAME
Artifact: proofs/perf/v015/cmdbuf_capture_<tag>_<variant>.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent

B, NUP, NZ = 16384, 8, 44
NITER_FULL, NITER_CAP = 50, 16

XLVCP, RCP, P1000MB, RVOVRD, GRAV = 2.5e6 / 1004.5, 0.2854, 1.0e5, 1.608, 9.81


def _qsat_like(t, p):
    # Horner chain like _qsat_blend (8 FMAs) -- realistic per-iteration ALU.
    xc = jnp.maximum(-80.0, t - 273.15)
    acc = 0.379534310e-11
    for c in (0.702620698e-8, 0.203154182e-5, 0.299291081e-3,
              0.264847430e-1, 0.143064234e1, 0.444006219e2, 0.609868993e3):
        acc = c + xc * acc
    es = jnp.minimum(acc, p * 0.15)
    return 0.622 * es / jnp.maximum(p - es, 1e-5)


def _cond_body(qc, qt, thl, p, exn):
    t = exn * thl + XLVCP * qc
    qs = _qsat_like(t, p)
    return 0.5 * qc + 0.5 * jnp.maximum(qt - qs, 0.0)


def make_fn(variant: str):
    def plume(qt0, thl0, p_col, w0):
        # per-plume column scan over levels, condensation per level
        def level_step(carry, k):
            w_p, qt_p, thl_p = carry
            pk = p_col[k]
            exn = (pk / P1000MB) ** RCP
            qc = jnp.zeros_like(qt_p)
            if variant in ("nested", "tridiag"):
                qc = lax.fori_loop(0, NITER_FULL, lambda _, q: _cond_body(q, qt_p, thl_p, pk, exn), qc)
            else:  # unroll16 / flat
                for _ in range(NITER_CAP):
                    qc = _cond_body(qc, qt_p, thl_p, pk, exn)
            thv = (thl_p + XLVCP * qc) * (1.0 + qt_p * (RVOVRD - 1.0) - RVOVRD * qc)
            wn = jnp.clip(w_p + 0.1 * GRAV * (thv / 300.0 - 1.0), 0.0, 3.0)
            ent = 0.1 * jnp.minimum(wn, 1.0)
            return (wn, qt_p * (1.0 - ent), thl_p * (1.0 - ent)), wn

        ks = jnp.arange(1, NZ - 1)
        if variant == "flat":
            (wf, qtf, thlf), ws = lax.scan(level_step, (w0, qt0, thl0), ks, unroll=int(NZ - 2))
        else:
            (wf, qtf, thlf), ws = lax.scan(level_step, (w0, qt0, thl0), ks)
        return wf + qtf + thlf, ws.sum(0)

    @jax.jit
    def fn(qt, thl, p, w):
        # vmap over (B, NUP) like the EDMF driver
        out, wsum = jax.vmap(jax.vmap(plume, in_axes=(0, 0, None, 0)), in_axes=(0, 0, None, 0))(qt, thl, p, w)
        if variant == "tridiag":
            d = out[..., None] * jnp.ones((1, 1, NZ))
            a = jnp.full_like(d, -0.1)
            bdiag = jnp.full_like(d, 2.0)
            c = jnp.full_like(d, -0.1)
            sol = lax.linalg.tridiagonal_solve(
                a.reshape(B * NUP, NZ), bdiag.reshape(B * NUP, NZ),
                c.reshape(B * NUP, NZ), d.reshape(B * NUP, NZ)[..., None])[..., 0]
            out = out + sol.reshape(B, NUP, NZ)[..., 0]
        return out, wsum

    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["nested", "unroll16", "flat", "tridiag"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    qt = 0.01 * jax.random.uniform(k1, (B, NUP), dtype=jnp.float64)
    thl = 290.0 + 10.0 * jax.random.uniform(k2, (B, NUP), dtype=jnp.float64)
    p = jnp.linspace(9.0e4, 2.0e4, NZ).astype(jnp.float64)
    w = 0.5 * jnp.ones((B, NUP), dtype=jnp.float64)

    fn = make_fn(args.variant)
    lowered = fn.lower(qt, thl, p, w)
    t0 = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0

    hlo = compiled.as_text()
    n_while = hlo.count(" while(")
    n_cmdbuf = sum(1 for ln in hlo.splitlines() if ln.strip().startswith("%command_buffer") or " command_buffer" in ln and "{" in ln)
    # decisive: a while op textually inside a command_buffer computation
    in_cb = False
    cur_cb = False
    for ln in hlo.splitlines():
        s = ln.strip()
        if s.startswith("%command_buffer") or (s.startswith("command_buffer") and "{" in s):
            cur_cb = True
        elif cur_cb and s.startswith("}"):
            cur_cb = False
        elif cur_cb and " while(" in s:
            in_cb = True

    out = compiled(qt, thl, p, w)
    jax.block_until_ready(out)
    samples = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(qt, thl, p, w))
        samples.append((time.perf_counter() - t0) * 1000.0)

    payload = {
        "schema": "V015CmdbufCapture",
        "variant": args.variant,
        "tag": args.tag,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "shape": {"B": B, "NUP": NUP, "NZ": NZ},
        "compile_s": round(compile_s, 2),
        "hlo_while_ops": n_while,
        "hlo_command_buffer_lines": n_cmdbuf,
        "while_inside_command_buffer": in_cb,
        "ms_per_call_median": round(statistics.median(samples), 3),
        "ms_per_call_min": round(min(samples), 3),
        "ms_per_call_all": [round(s, 3) for s in samples],
        "checksum": float(jnp.sum(out[0])),
    }
    fn_out = HERE / f"cmdbuf_capture_{args.tag}_{args.variant}.json"
    fn_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "ms_per_call_all"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
