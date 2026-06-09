#!/usr/bin/env python3
"""v0.14 Mythos memory lane: GPU measurement suite.

Run ONLY through the repo GPU wrapper (one GPU job at a time):

  scripts/run_gpu_lowprio.sh --cores 0-23 -- \
    python proofs/v014/mythos_memory_gpu_suite_260609.py

Produces proofs/v014/mythos_memory_gpu_suite_260609.json with:

1. MYNN BouLac materialization measurement: AOT-compile memory analysis of the
   whole-batch (untiled) vs leading-column-tiled MYNN step at the 641x321x50
   target geometry and a d03-1km-like geometry.  This is the measure-first
   evidence the empirical memory map required before non-radiation tiling work.
2. MYNN GPU tile-vs-untiled bit-identity execution at a bounded geometry.
3. Moisture transport-velocity duplicate-build measurement: compiled temp
   memory of the literal duplicated couple_velocities_periodic build vs the
   shared-build form at target geometry (answers whether XLA CSE already
   deduplicated the duplicate construction the roadmap flagged).
4. Active moisture limiter (moist_adv_opt=2) workspace measurement at target
   geometry (measure-first evidence for the limiter-workspace row).

No long validation, no TOST, no model-state semantics: synthetic inputs sized
like production, used purely for compiler/runtime memory accounting plus exact
equality checks of layout-identical computations.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs" / "v014" / "mythos_memory_gpu_suite_260609.json"

# Whole-batch reference path for the untiled measurements; the tiled variants
# re-enable tiling per-call by poking the module constants (test-only).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import gpuwrf.physics.mynn_pbl as mynn_pbl  # noqa: E402
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes  # noqa: E402
from gpuwrf.dynamics.flux_advection import (  # noqa: E402
    advect_moisture_scalars,
    advect_scalar_flux,
    couple_velocities_periodic,
)

GIB = 1024.0**3


def _mem_analysis_record(compiled) -> dict:
    try:
        ma = compiled.memory_analysis()
    except Exception as exc:  # pragma: no cover - backend-specific
        return {"ok": False, "reason": f"memory_analysis failed: {exc!r}"}
    if ma is None:
        return {"ok": False, "reason": "memory_analysis returned None"}
    rec = {"ok": True}
    for name in (
        "temp_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "generated_code_size_in_bytes",
    ):
        value = getattr(ma, name, None)
        if value is not None:
            rec[name] = int(value)
            rec[name.replace("_in_bytes", "_gib")] = round(int(value) / GIB, 6)
    return rec


def _device_kind() -> dict:
    dev = jax.devices()[0]
    return {
        "platform": dev.platform,
        "device_kind": getattr(dev, "device_kind", str(dev)),
        "jax_version": jax.__version__,
    }


def _mynn_struct(batch: int, nz: int):
    prof = jax.ShapeDtypeStruct((batch, nz), jnp.float64)
    surf = jax.ShapeDtypeStruct((batch,), jnp.float64)
    state = mynn_pbl.MynnPBLColumnState(
        u=prof, v=prof, w=prof, theta=prof, qv=prof, tke=prof, p=prof,
        rho=prof, dz=prof, km=prof, kh=prof, el=prof, qc=prof, qi=prof,
    )
    surface = SurfaceFluxes(
        ustar=surf, theta_flux=surf, qv_flux=surf, tau_u=surf, tau_v=surf,
        rhosfc=surf, fltv=surf, xland=surf,
    )
    return state, surface


def _mynn_compile_record(batch: int, nz: int, tile_cols: int | None) -> dict:
    """AOT-compiles the MYNN step at (batch, nz); tile_cols None = untiled."""

    mynn_pbl._MYNN_COLUMN_TILING = tile_cols is not None
    mynn_pbl._MYNN_COLUMN_TILE_COLS = int(tile_cols or 0)
    state, surface = _mynn_struct(batch, nz)

    def fn(s, sf):
        return mynn_pbl._tiled_mynn_step(s, 60.0, False, sf, True, 1000.0)

    try:
        compiled = jax.jit(fn).lower(state, surface).compile()
    except Exception as exc:
        return {"ok": False, "reason": f"compile failed: {exc!r}"}
    rec = _mem_analysis_record(compiled)
    rec["batch"] = batch
    rec["nz"] = nz
    rec["tile_cols"] = tile_cols
    return rec


def mynn_memory_measurements() -> dict:
    nz = 50
    target_batch = 641 * 321          # roadmap target geometry columns
    d03_batch = 313 * 313             # d03-1km-like geometry
    out = {"cases": []}
    for batch in (target_batch, d03_batch):
        untiled = _mynn_compile_record(batch, nz, None)
        tiled = _mynn_compile_record(batch, nz, 16384)
        case = {"batch": batch, "nz": nz, "untiled": untiled, "tiled_16384": tiled}
        if untiled.get("ok") and tiled.get("ok"):
            case["temp_delta_gib"] = round(
                (untiled["temp_size_in_bytes"] - tiled["temp_size_in_bytes"]) / GIB, 6
            )
        out["cases"].append(case)
    return out


def mynn_gpu_bit_identity(batch: int = 40000, nz: int = 50, tile_cols: int = 4096) -> dict:
    rng = np.random.default_rng(11)

    def mk(lo, hi, shape):
        return jnp.asarray(rng.uniform(lo, hi, shape))

    state = mynn_pbl.MynnPBLColumnState(
        u=mk(-15, 15, (batch, nz)), v=mk(-15, 15, (batch, nz)),
        w=mk(-0.5, 0.5, (batch, nz)),
        theta=290.0 + jnp.cumsum(mk(0.0, 0.2, (batch, nz)), axis=-1),
        qv=mk(1e-4, 1e-2, (batch, nz)), tke=mk(1e-3, 1.0, (batch, nz)),
        p=jnp.asarray(np.linspace(1000e2, 200e2, nz))[None, :] * jnp.ones((batch, 1)),
        rho=mk(0.4, 1.2, (batch, nz)), dz=mk(30, 400, (batch, nz)),
        km=mk(0, 5, (batch, nz)), kh=mk(0, 5, (batch, nz)), el=mk(1, 200, (batch, nz)),
        qc=mk(0, 1e-4, (batch, nz)), qi=mk(0, 1e-5, (batch, nz)),
    )
    surface = SurfaceFluxes(
        ustar=mk(0.05, 0.8, (batch,)), theta_flux=mk(-0.1, 0.3, (batch,)),
        qv_flux=mk(-1e-5, 1e-4, (batch,)), tau_u=mk(-0.5, 0.5, (batch,)),
        tau_v=mk(-0.5, 0.5, (batch,)), rhosfc=mk(1.0, 1.25, (batch,)),
        fltv=mk(-0.1, 0.3, (batch,)),
        xland=jnp.where(jnp.asarray(rng.uniform(size=batch)) < 0.5, 1.0, 2.0),
    )

    def run(tiling: bool, tile: int):
        mynn_pbl._MYNN_COLUMN_TILING = tiling
        mynn_pbl._MYNN_COLUMN_TILE_COLS = tile

        def fn(s, sf):
            return mynn_pbl._tiled_mynn_step(s, 60.0, False, sf, True, 1000.0)

        out_state, pblh = jax.jit(fn)(state, surface)
        return jax.device_get(out_state), np.asarray(pblh)

    ref_state, ref_pblh = run(False, 0)
    til_state, til_pblh = run(True, tile_cols)
    fields = {}
    all_same = bool(np.array_equal(ref_pblh, til_pblh))
    fields["pblh"] = {"bit_identical": bool(np.array_equal(ref_pblh, til_pblh))}
    for name in ref_state.__slots__:
        a = getattr(ref_state, name)
        b = getattr(til_state, name)
        if a is None:
            continue
        a = np.asarray(a)
        b = np.asarray(b)
        same = bool(np.array_equal(a, b))
        fields[name] = {
            "bit_identical": same,
            "max_abs": float(np.max(np.abs(a - b))),
        }
        all_same = all_same and same
    return {
        "batch": batch, "nz": nz, "tile_cols": tile_cols,
        "fields": fields, "all_bit_identical": all_same,
    }


def _advection_inputs(nz: int, ny: int, nx: int):
    rng = np.random.default_rng(3)

    def mk(lo, hi, shape):
        return jnp.asarray(rng.uniform(lo, hi, shape))

    u = mk(-20, 20, (nz, ny, nx + 1))
    v = mk(-20, 20, (nz, ny + 1, nx))
    mu = mk(40000.0, 70000.0, (ny, nx))
    theta = 300.0 + mk(-5, 5, (nz, ny, nx))
    qs = tuple(mk(0.0, 1e-2, (nz, ny, nx)) for _ in range(6))
    c1h = jnp.asarray(np.linspace(1.0, 0.9, nz))
    c2h = jnp.asarray(np.linspace(0.0, 5000.0, nz))
    dnw = jnp.asarray(np.full(nz, -1.0 / nz))
    rdnw = jnp.asarray(np.full(nz, float(nz)))
    fnm = jnp.asarray(np.full(nz, 0.5))
    fnp = jnp.asarray(np.full(nz, 0.5))
    msf = jnp.ones((ny, nx))
    return u, v, mu, theta, qs, c1h, c2h, dnw, rdnw, fnm, fnp, msf


def moisture_velocity_reuse_measurement(nz: int = 50, ny: int = 321, nx: int = 641) -> dict:
    """Duplicate vs shared couple_velocities_periodic build, compiled memory."""

    u, v, mu, theta, qs, c1h, c2h, dnw, rdnw, fnm, fnp, msf = _advection_inputs(nz, ny, nx)
    rdx = rdy = 1.0 / 3000.0
    kw = dict(c1h=c1h, c2h=c2h, dnw=dnw, rdx=rdx, rdy=rdy,
              msfuy=msf, msfvx=msf, msftx=msf, msfux=msf, msfvy=msf)

    def theta_part(vel, theta_, mu_):
        return advect_scalar_flux(theta_ - 300.0, vel, mut=mu_, c1=c1h,
                                  rdx=rdx, rdy=rdy, rdzw=rdnw, fzm=fnm, fzp=fnp)

    def moist_part(vel, qs_, mu_):
        return advect_moisture_scalars(
            qs_, None, vel, moist_adv_opt=0, is_final_rk_stage=True, mut=mu_,
            mu_old=mu_, c1=c1h, c2=c2h, rdx=rdx, rdy=rdy, rdzw=rdnw,
            fzm=fnm, fzp=fnp, dt=24.0,
        )

    def fn_duplicate(u_, v_, mu_, theta_, qs_):
        vel_a = couple_velocities_periodic(u_, v_, mu_, **kw)
        vel_b = couple_velocities_periodic(u_, v_, mu_, **kw)
        return theta_part(vel_a, theta_, mu_), moist_part(vel_b, qs_, mu_)

    def fn_shared(u_, v_, mu_, theta_, qs_):
        vel = couple_velocities_periodic(u_, v_, mu_, **kw)
        return theta_part(vel, theta_, mu_), moist_part(vel, qs_, mu_)

    structs = tuple(
        jax.ShapeDtypeStruct(np.asarray(a).shape, jnp.float64) for a in (u, v, mu, theta)
    ) + (tuple(jax.ShapeDtypeStruct((nz, ny, nx), jnp.float64) for _ in range(6)),)
    rec = {"nz": nz, "ny": ny, "nx": nx}
    for label, fn in (("duplicate_build", fn_duplicate), ("shared_build", fn_shared)):
        try:
            compiled = jax.jit(fn).lower(*structs).compile()
            rec[label] = _mem_analysis_record(compiled)
        except Exception as exc:
            rec[label] = {"ok": False, "reason": repr(exc)}
    if rec["duplicate_build"].get("ok") and rec["shared_build"].get("ok"):
        rec["temp_delta_gib"] = round(
            (rec["duplicate_build"]["temp_size_in_bytes"]
             - rec["shared_build"]["temp_size_in_bytes"]) / GIB, 6,
        )
        out_a = jax.jit(fn_duplicate)(u, v, mu, theta, qs)
        out_b = jax.jit(fn_shared)(u, v, mu, theta, qs)
        flat_a = jax.tree_util.tree_leaves(out_a)
        flat_b = jax.tree_util.tree_leaves(out_b)
        rec["value_bit_identical"] = bool(
            all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(flat_a, flat_b))
        )
    return rec


def limiter_workspace_measurement(nz: int = 50, ny: int = 321, nx: int = 641) -> dict:
    """Active moist_adv_opt=2 final-stage limiter workspace, compiled memory."""

    u, v, mu, theta, qs, c1h, c2h, dnw, rdnw, fnm, fnp, msf = _advection_inputs(nz, ny, nx)
    rdx = rdy = 1.0 / 3000.0
    kw = dict(c1h=c1h, c2h=c2h, dnw=dnw, rdx=rdx, rdy=rdy,
              msfuy=msf, msfvx=msf, msftx=msf, msfux=msf, msfvy=msf)

    def fn(u_, v_, mu_, qs_, opt: int):
        vel = couple_velocities_periodic(u_, v_, mu_, **kw)
        return advect_moisture_scalars(
            qs_, qs_ if opt else None, vel, moist_adv_opt=opt,
            is_final_rk_stage=True, mut=mu_, mu_old=mu_, c1=c1h, c2=c2h,
            rdx=rdx, rdy=rdy, rdzw=rdnw, fzm=fnm, fzp=fnp, dt=24.0,
        )

    structs = (
        jax.ShapeDtypeStruct(np.asarray(u).shape, jnp.float64),
        jax.ShapeDtypeStruct(np.asarray(v).shape, jnp.float64),
        jax.ShapeDtypeStruct((ny, nx), jnp.float64),
        tuple(jax.ShapeDtypeStruct((nz, ny, nx), jnp.float64) for _ in range(6)),
    )
    rec = {"nz": nz, "ny": ny, "nx": nx}
    for label, opt in (("plain_opt0", 0), ("monotonic_opt2", 2)):
        try:
            compiled = jax.jit(lambda a, b, c, d, _o=opt: fn(a, b, c, d, _o)).lower(*structs).compile()
            rec[label] = _mem_analysis_record(compiled)
        except Exception as exc:
            rec[label] = {"ok": False, "reason": repr(exc)}
    if rec["plain_opt0"].get("ok") and rec["monotonic_opt2"].get("ok"):
        rec["limiter_extra_temp_gib"] = round(
            (rec["monotonic_opt2"]["temp_size_in_bytes"]
             - rec["plain_opt0"]["temp_size_in_bytes"]) / GIB, 6,
        )
    return rec


def nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10.0, check=True,
        ).stdout.strip()
        used, total = (int(x.strip()) for x in out.split(","))
        return {"ok": True, "memory_used_mib": used, "memory_total_mib": total}
    except Exception as exc:
        return {"ok": False, "reason": repr(exc)}


def main() -> int:
    record = {
        "proof": "mythos_memory_gpu_suite_260609",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": _device_kind(),
        "gpu_before": nvidia_smi(),
        "constraints": {
            "wrapper": "scripts/run_gpu_lowprio.sh (flock held by parent)",
            "one_gpu_job": True,
            "long_validation": False,
            "tost": False,
        },
    }
    record["mynn_boulac_materialization"] = mynn_memory_measurements()
    record["mynn_gpu_tile_bit_identity"] = mynn_gpu_bit_identity()
    # Production-gate case: d03-1km-like ragged batch at the PRODUCTION tile
    # width (97969 = 5*16384 + 16049). CPU shows batch-width SIMD codegen ulps
    # in the tridiagonal solves; the GPU production path must be bit-exact.
    record["mynn_gpu_tile_bit_identity_production"] = mynn_gpu_bit_identity(
        batch=97969, nz=50, tile_cols=16384
    )
    record["moisture_velocity_reuse"] = moisture_velocity_reuse_measurement()
    record["moisture_limiter_workspace"] = limiter_workspace_measurement()
    record["gpu_after"] = nvidia_smi()
    OUT_JSON.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(OUT_JSON),
        "mynn_cases": [
            {k: c.get(k) for k in ("batch", "temp_delta_gib")}
            for c in record["mynn_boulac_materialization"]["cases"]
        ],
        "mynn_bit_identity": record["mynn_gpu_tile_bit_identity"]["all_bit_identical"],
        "velocity_reuse_delta_gib": record["moisture_velocity_reuse"].get("temp_delta_gib"),
        "velocity_value_identical": record["moisture_velocity_reuse"].get("value_bit_identical"),
        "limiter_extra_temp_gib": record["moisture_limiter_workspace"].get("limiter_extra_temp_gib"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
