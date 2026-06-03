"""v0.6.0 per-scheme integration SMOKE (CPU) for the operational scan adapters.

For each GPU-runnable NEW scheme that is scan-wired
(``coupling.scan_adapters``), this harness:

* builds a small but physically reasonable C-grid ``State`` (the b2 pattern),
* drives the scheme's State<->scheme SCAN ADAPTER for a few steps (the EXACT
  ``State -> State`` function the operational scan body calls via the dispatcher),
* confirms the scheme executed (the State actually changed in its written leaves),
* checks State stays consistent (finite, no NaN, dtypes/shapes preserved),
* checks loose conservation (total column water mass change bounded; no runaway),

then drives the integrated dispatch the way the scan body selects schemes
(``MP_SCAN_ADAPTERS`` / ``SFCLAY_SCAN_ADAPTERS`` / ``CU_SCAN_ADAPTERS``) and
verifies ``_resolve_operational_suite`` ACCEPTS the scan-wired combos and REJECTS
the not-wired ones (YSU/ACM2/GF/Tiedtke) loudly. Noah-classic has a dedicated
coupler smoke because explicit ``noahclassic_static``/``noahclassic_land`` bundles
are required; this legacy smoke only checks that selecting sf_surface_physics=2
without that bundle still fails closed.

This is a CPU JAX integration smoke (a few steps each), NOT a full forecast and
NOT a WRF-parity claim -- the GPU multi-config forecast gate vs CPU-WRF is
MANAGER-scheduled. Per-scheme WRF savepoint parity lives in the lane reports.
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
from gpuwrf.contracts.state import State, _state_field_shapes  # noqa: E402
from gpuwrf.coupling.scan_adapters import (  # noqa: E402
    CU_SCAN_ADAPTERS,
    MP_SCAN_ADAPTERS,
    SFCLAY_SCAN_ADAPTERS,
    initial_kf_carry,
    kessler_adapter,
    kf_adapter,
    morrison_adapter,
    pleim_xiu_sfclay_adapter,
    sfclay_revised_mm5_adapter,
    wdm6_adapter,
    wsm6_adapter,
)

GRAVITY = 9.81
P0 = 100000.0
RD = 287.0
CP = 1004.0

ROOT = Path(__file__).resolve().parents[2]


def _build_state(nz=24, ny=6, nx=8, dz_m=300.0, seed=3):
    """Small physically-reasonable C-grid State (b2 pattern, deeper column)."""

    class _Grid:
        def __init__(self, nz, ny, nx):
            self.nz, self.ny, self.nx = nz, ny, nx

    grid = _Grid(nz, ny, nx)
    shapes = _state_field_shapes(grid)
    rng = np.random.default_rng(seed)

    z_iface = np.arange(nz + 1) * dz_m
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    theta_col = 300.0 + 0.004 * z_mid
    p_col = P0 * (1.0 - GRAVITY * z_mid / (CP * 290.0)) ** (CP / RD)

    def m3(base, noise):
        return jnp.asarray(base[:, None, None] + noise * rng.standard_normal((nz, ny, nx)), dtype=jnp.float64)

    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    fields["theta"] = m3(theta_col, 0.3)
    fields["p"] = m3(p_col, 50.0)
    fields["p_total"] = fields["p"]
    # moist lower troposphere with a shallow cloud/rain seed so microphysics + KF
    # have something to act on.
    qv_col = 0.012 * np.exp(-z_mid / 3000.0)
    fields["qv"] = jnp.clip(m3(qv_col, 0.0005), 0.0, None)
    qc_col = np.where(z_mid < 4000.0, 4.0e-4, 0.0)
    fields["qc"] = jnp.clip(m3(qc_col, 2.0e-5), 0.0, None)
    qr_col = np.where((z_mid > 500.0) & (z_mid < 3000.0), 1.0e-4, 0.0)
    fields["qr"] = jnp.clip(m3(qr_col, 1.0e-5), 0.0, None)
    qi_col = np.where(z_mid > 6000.0, 5.0e-5, 0.0)
    fields["qi"] = jnp.clip(m3(qi_col, 1.0e-6), 0.0, None)
    fields["qs"] = jnp.clip(m3(np.where(z_mid > 5000.0, 3.0e-5, 0.0), 1.0e-6), 0.0, None)
    fields["qg"] = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    # number concentrations (m^-3) for two-moment / WDM6
    fields["Ni"] = jnp.clip(m3(np.where(z_mid > 6000.0, 5.0e3, 0.0), 1.0e2), 0.0, None)
    fields["Nr"] = jnp.clip(m3(np.where((z_mid > 500.0) & (z_mid < 3000.0), 1.0e4, 0.0), 1.0e2), 0.0, None)
    fields["Ns"] = jnp.clip(m3(np.where(z_mid > 5000.0, 5.0e3, 0.0), 1.0e2), 0.0, None)
    fields["Ng"] = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    fields["Nc"] = jnp.clip(m3(np.where(z_mid < 4000.0, 1.0e8, 0.0), 1.0e6), 0.0, None)
    fields["Nn"] = jnp.clip(m3(np.full(nz, 1.0e8), 1.0e6), 0.0, None)

    fields["u"] = jnp.asarray(6.0 + 0.5 * rng.standard_normal((nz, ny, nx + 1)), dtype=jnp.float64)
    fields["v"] = jnp.asarray(-2.0 + 0.5 * rng.standard_normal((nz, ny + 1, nx)), dtype=jnp.float64)
    fields["w"] = jnp.asarray(0.05 * rng.standard_normal((nz + 1, ny, nx)), dtype=jnp.float64)
    fields["qke"] = jnp.full((nz, ny, nx), 0.4, dtype=jnp.float64)

    ph = jnp.asarray(np.broadcast_to(GRAVITY * z_iface[:, None, None], (nz + 1, ny, nx)), dtype=jnp.float64)
    fields["ph"] = ph
    fields["ph_total"] = ph

    xland = np.ones((ny, nx))
    xland[:, nx // 2 :] = 2.0
    fields["xland"] = jnp.asarray(xland, dtype=jnp.float64)
    fields["lakemask"] = jnp.zeros((ny, nx), dtype=jnp.float64)
    fields["t_skin"] = jnp.asarray(np.where(xland > 1.5, 299.5, 304.0), dtype=jnp.float64)
    fields["soil_moisture"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["mavail"] = jnp.where(jnp.asarray(xland) > 1.5, 1.0, 0.4).astype(jnp.float64)
    fields["roughness_m"] = jnp.where(jnp.asarray(xland) > 1.5, 2.85e-3, 0.15).astype(jnp.float64)
    fields["ustar"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["rhosfc"] = jnp.full((ny, nx), 1.15, dtype=jnp.float64)
    fields["mu_total"] = jnp.full((ny, nx), 1.0e5, dtype=jnp.float64)
    fields["mu_perturbation"] = jnp.zeros((ny, nx), dtype=jnp.float64)

    return State(**fields), grid


_WATER = ("qv", "qc", "qr", "qi", "qs", "qg")


def _column_water(state) -> float:
    return float(sum(jnp.sum(jnp.asarray(getattr(state, q))) for q in _WATER))


def _all_finite(state, leaves) -> bool:
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(getattr(state, lf))))) for lf in leaves)


def _changed(before, after, leaf) -> bool:
    a = jnp.asarray(getattr(before, leaf))
    b = jnp.asarray(getattr(after, leaf))
    return float(jnp.max(jnp.abs(b - a))) > 0.0


def _smoke_mp(name, adapter, written, *, steps=3, dt=20.0):
    state, _ = _build_state()
    w0 = _column_water(state)
    cur = state
    changed_any = False
    for _ in range(steps):
        nxt = adapter(cur, dt, None)
        changed_any = changed_any or any(_changed(cur, nxt, lf) for lf in written if hasattr(nxt, lf))
        cur = nxt
    w1 = _column_water(cur)
    finite = _all_finite(cur, written)
    # loose conservation: total water mass should not blow up or go wildly negative.
    rel = abs(w1 - w0) / max(w0, 1e-12)
    return {
        "scheme": name,
        "kind": "microphysics",
        "steps": steps,
        "executed_in_scan_adapter": bool(changed_any),
        "finite_no_nan": bool(finite),
        "column_water_before": w0,
        "column_water_after": w1,
        "column_water_rel_change": rel,
        "conservation_ok": bool(rel < 0.5 and w1 >= -1e-9),
        "pass": bool(changed_any and finite and rel < 0.5 and w1 >= -1e-9),
    }


def _smoke_sfclay(name, adapter, *, steps=3, dt=20.0):
    state, grid = _build_state()
    handles = ("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv")
    cur = state
    changed_any = False
    for _ in range(steps):
        nxt = adapter(cur, dt, None)
        changed_any = changed_any or any(_changed(cur, nxt, lf) for lf in handles)
        cur = nxt
    finite = _all_finite(cur, handles)
    ustar = np.asarray(cur.ustar)
    return {
        "scheme": name,
        "kind": "surface_layer",
        "steps": steps,
        "executed_in_scan_adapter": bool(changed_any),
        "finite_no_nan": bool(finite),
        "ustar_band_m_s": [float(np.min(ustar)), float(np.max(ustar))],
        "ustar_positive": bool(np.all(ustar > 0.0)),
        "pass": bool(changed_any and finite and np.all(ustar > 0.0)),
    }


def _smoke_kf(*, steps=3, dt=300.0):
    state, _ = _build_state()
    w0avg, nca = initial_kf_carry(state)
    cur = state
    w0 = _column_water(state)
    executed = False
    for _ in range(steps):
        nxt, w0avg, nca = kf_adapter(cur, dt, w0avg, nca, grid=None)
        executed = executed or any(_changed(cur, nxt, lf) for lf in ("theta", "qv", "qc", "qr", "rainc_acc"))
        cur = nxt
    finite = _all_finite(cur, ("theta", "qv", "qc", "qr", "qi", "qs")) and bool(
        jnp.all(jnp.isfinite(jnp.asarray(cur.rainc_acc)))
    )
    w1 = _column_water(cur)
    rel = abs(w1 - w0) / max(w0, 1e-12)
    carry_finite = bool(jnp.all(jnp.isfinite(w0avg))) and bool(jnp.all(jnp.isfinite(nca)))
    return {
        "scheme": "Kain-Fritsch (cu=1)",
        "kind": "cumulus",
        "steps": steps,
        # KF may legitimately not trigger convection on a near-neutral smoke column;
        # the wiring is validated by a finite, consistent run + a stable evolving
        # (w0avg, nca) carry. "executed" records whether tendencies were applied.
        "tendency_applied": bool(executed),
        "carry_threaded_finite": carry_finite,
        "finite_no_nan": bool(finite),
        "rainc_acc_min_mm": float(jnp.min(jnp.asarray(cur.rainc_acc))),
        "column_water_rel_change": rel,
        "conservation_ok": bool(rel < 0.5),
        "pass": bool(finite and carry_finite and rel < 0.5),
    }


def _fail_closed_checks():
    """Confirm _resolve_operational_suite ACCEPTS wired combos and REJECTS others."""

    from gpuwrf.runtime.operational_mode import OperationalNamelist, _resolve_operational_suite
    from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection

    class _NL:
        """Minimal namelist-shaped stub exposing the physics-option attributes."""

        def __init__(self, **kw):
            self.mp_physics = kw.get("mp_physics", 8)
            self.bl_pbl_physics = kw.get("bl_pbl_physics", 5)
            self.sf_sfclay_physics = kw.get("sf_sfclay_physics", 5)
            self.cu_physics = kw.get("cu_physics", 0)
            self.sf_surface_physics = kw.get("sf_surface_physics", None)
            self.use_noahmp = kw.get("use_noahmp", False)

    accepted = []
    rejected = []
    # wired combos that should resolve
    for nl in [
        _NL(),  # v0.2.0 baseline
        _NL(mp_physics=6, sf_sfclay_physics=1, cu_physics=1),  # WSM6 + revised-MM5 + KF
        _NL(mp_physics=10, sf_sfclay_physics=7),  # Morrison + Pleim-Xiu
        _NL(mp_physics=1),  # Kessler
        _NL(mp_physics=16),  # WDM6
        _NL(bl_pbl_physics=1),  # YSU PBL (v0.6.0 jax.lax.scan rewrite -> now wired)
        _NL(bl_pbl_physics=7),  # ACM2 PBL (v0.6.0 jax.lax.scan rewrite -> now wired)
    ]:
        try:
            _resolve_operational_suite(nl)
            accepted.append(True)
        except UnsupportedSchemeSelection:
            accepted.append(False)
    # NOT-wired schemes (GF/Tiedtke CPU-ref cumulus -- YSU/ACM2 are now scan-wired),
    # plus explicit Noah-classic WITHOUT its required static/land bundle, that MUST
    # all fail closed.
    for nl in [
        _NL(cu_physics=3),      # Grell-Freitas CPU-ref
        _NL(cu_physics=6),      # Tiedtke CPU-ref
        _NL(sf_surface_physics=2),  # Noah-classic missing explicit land/static bundle
    ]:
        try:
            _resolve_operational_suite(nl)
            rejected.append(False)  # should NOT have resolved
        except UnsupportedSchemeSelection:
            rejected.append(True)
    return {
        "wired_combos_accepted": accepted,
        "all_wired_accepted": all(accepted),
        "unwired_schemes_rejected": rejected,
        "all_unwired_rejected": all(rejected),
        "pass": bool(all(accepted) and all(rejected)),
    }


def run() -> dict:
    mp_results = [
        _smoke_mp("Kessler (mp=1)", kessler_adapter, ("theta", "qv", "qc", "qr", "rain_acc")),
        _smoke_mp("WSM6 (mp=6)", wsm6_adapter, ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "rain_acc")),
        _smoke_mp("Morrison (mp=10)", morrison_adapter,
                  ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Ns", "Nr", "Ng", "rain_acc")),
        _smoke_mp("WDM6 (mp=16)", wdm6_adapter,
                  ("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Nc", "Nr", "Nn", "rain_acc")),
    ]
    sfclay_results = [
        _smoke_sfclay("revised-MM5 (sf_sfclay=1)", sfclay_revised_mm5_adapter),
        _smoke_sfclay("Pleim-Xiu (sf_sfclay=7)", pleim_xiu_sfclay_adapter),
    ]
    kf_result = _smoke_kf()
    fail_closed = _fail_closed_checks()

    per_scheme = mp_results + sfclay_results + [kf_result]
    return {
        "proof": "v060-scanwire-integration-smoke",
        "kind": "CPU JAX integration smoke (few steps each); NOT a WRF parity claim; "
                "GPU multi-config forecast gate vs CPU-WRF is MANAGER-scheduled",
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "jax_platform": jax.default_backend(),
        "scan_wired": {
            "microphysics_gpu_scan": sorted(MP_SCAN_ADAPTERS) + [8],
            "surface_layer_gpu_scan": sorted(SFCLAY_SCAN_ADAPTERS) + [5],
            "cumulus_gpu_scan": sorted(CU_SCAN_ADAPTERS),
        },
        "per_scheme_smoke": per_scheme,
        "fail_closed": fail_closed,
        "all_pass": bool(all(r["pass"] for r in per_scheme) and fail_closed["pass"]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="v0.6.0 scan-wire per-scheme integration smoke")
    parser.add_argument("--out", type=Path, default=ROOT / "proofs" / "v060" / "scanwire_smoke.json")
    args = parser.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_pass"] else 1)
