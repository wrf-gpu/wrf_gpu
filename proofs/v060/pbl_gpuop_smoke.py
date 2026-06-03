"""v0.6.0 YSU + ACM2 PBL GPU-operational SMOKE (CPU JAX).

Drives the two ``jax.lax.scan``-traceable PBL adapters
(``coupling.scan_adapters.{ysu_pbl_adapter, acm2_pbl_adapter}``) -- the EXACT
``State -> State`` functions the operational scan PBL slot now calls -- through a
few steps on a small physically-reasonable C-grid State, JIT-COMPILED (proving the
kernel is device-traceable, not host-NumPy), and confirms:

* the adapter EXECUTED (wrote its u/v/theta/qv leaves),
* State stays finite / no NaN,
* loose conservation (total column water bounded; PBL mixing redistributes, does
  not create/destroy water at the column scale),
* the JIT trace succeeds (the host-NumPy single-column kernel could not be traced;
  a successful jit is the GPU-operational proof).

This is a CPU integration smoke (few steps), NOT a WRF-parity claim -- per-scheme
WRF savepoint parity lives in ``{ysu,acm2}_savepoint_parity_report.json`` and the
GPU multi-config forecast gate vs CPU-WRF is MANAGER-scheduled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import gpuwrf  # noqa: E402,F401  enables x64 at import
from gpuwrf.coupling.scan_adapters import acm2_pbl_adapter, ysu_pbl_adapter  # noqa: E402

# Reuse the b2 State builder from the existing scan-wire smoke.
import importlib.util  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "scanwire_smoke", Path(__file__).resolve().parent / "scanwire_smoke.py"
)
_sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)
_build_state = _sw._build_state

_WATER = ("qv", "qc", "qr", "qi", "qs", "qg")
_PBL_LEAVES = ("u", "v", "theta", "qv")


def _column_water(state) -> float:
    return float(sum(jnp.sum(jnp.asarray(getattr(state, q))) for q in _WATER))


def _all_finite(state, leaves) -> bool:
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(getattr(state, lf))))) for lf in leaves)


def _changed(before, after, leaf) -> bool:
    a = jnp.asarray(getattr(before, leaf))
    b = jnp.asarray(getattr(after, leaf))
    return float(jnp.max(jnp.abs(b - a))) > 0.0


def _smoke_pbl(name, adapter, *, steps=3, dt=60.0):
    state, grid = _build_state()
    # JIT the adapter (dt/grid static): a successful trace+exec is the
    # GPU-operational proof -- the prior host-NumPy single-column kernel could not
    # be traced at all (fail-closed in the scan).
    jitted = jax.jit(lambda s: adapter(s, dt, grid))
    w0 = _column_water(state)
    cur = state
    changed_any = False
    jit_ok = True
    try:
        for _ in range(steps):
            nxt = jitted(cur)
            changed_any = changed_any or any(_changed(cur, nxt, lf) for lf in _PBL_LEAVES)
            cur = nxt
    except Exception as exc:  # noqa: BLE001
        jit_ok = False
        return {
            "scheme": name,
            "kind": "pbl",
            "jit_traceable": False,
            "error": repr(exc),
            "pass": False,
        }
    w1 = _column_water(cur)
    finite = _all_finite(cur, _PBL_LEAVES)
    rel = abs(w1 - w0) / max(w0, 1e-12)
    theta_min = float(jnp.min(jnp.asarray(cur.theta)))
    theta_max = float(jnp.max(jnp.asarray(cur.theta)))
    return {
        "scheme": name,
        "kind": "pbl",
        "steps": steps,
        "dt_s": dt,
        "jit_traceable": bool(jit_ok),
        "executed_in_scan_adapter": bool(changed_any),
        "finite_no_nan": bool(finite),
        "column_water_before": w0,
        "column_water_after": w1,
        "column_water_rel_change": rel,
        "theta_band_K": [theta_min, theta_max],
        "conservation_ok": bool(rel < 0.1),
        "pass": bool(jit_ok and changed_any and finite and rel < 0.1
                     and 200.0 < theta_min and theta_max < 400.0),
    }


def _fail_open_checks():
    """Confirm _resolve_operational_suite now ACCEPTS YSU(1)/ACM2(7) PBL selections."""

    from gpuwrf.runtime.operational_mode import _resolve_operational_suite
    from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection

    class _NL:
        def __init__(self, **kw):
            self.mp_physics = kw.get("mp_physics", 8)
            self.bl_pbl_physics = kw.get("bl_pbl_physics", 5)
            self.sf_sfclay_physics = kw.get("sf_sfclay_physics", 5)
            self.cu_physics = kw.get("cu_physics", 0)
            self.sf_surface_physics = kw.get("sf_surface_physics", None)
            self.use_noahmp = kw.get("use_noahmp", False)

    results = {}
    for label, opt in (("ysu_bl1", 1), ("acm2_bl7", 7)):
        try:
            _resolve_operational_suite(_NL(bl_pbl_physics=opt))
            results[label] = True
        except UnsupportedSchemeSelection:
            results[label] = False
    return {
        "ysu_accepted": results["ysu_bl1"],
        "acm2_accepted": results["acm2_bl7"],
        "pass": bool(results["ysu_bl1"] and results["acm2_bl7"]),
    }


def run() -> dict:
    ysu = _smoke_pbl("YSU PBL (bl_pbl=1)", ysu_pbl_adapter)
    acm2 = _smoke_pbl("ACM2 PBL (bl_pbl=7)", acm2_pbl_adapter)
    fail_open = _fail_open_checks()
    per_scheme = [ysu, acm2]
    return {
        "proof": "v060-pbl-gpuop-smoke",
        "kind": "CPU JAX integration smoke (JIT-compiled adapter, few steps); NOT a "
                "WRF parity claim; per-scheme WRF savepoint parity in the lane reports; "
                "GPU multi-config forecast gate vs CPU-WRF is MANAGER-scheduled",
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "jax_platform": jax.default_backend(),
        "per_scheme_smoke": per_scheme,
        "fail_open_now_accepts_ysu_acm2": fail_open,
        "all_pass": bool(all(r["pass"] for r in per_scheme) and fail_open["pass"]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v0.6.0 YSU+ACM2 PBL GPU-op smoke")
    parser.add_argument("--out", type=Path, default=ROOT / "proofs" / "v060" / "pbl_gpuop_smoke.json")
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_pass"] else 1)
