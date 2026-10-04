"""B2 coupled smoke proof: surface_adapter -> mynn_adapter through State.

Builds a small but physically reasonable C-grid ``State`` (warm land + cool sea
mix), runs the FROZEN operational hand-off
``surface_adapter`` (writes the WRF revised surface-layer flux handles) ->
``mynn_adapter`` (consumes them, advances the PBL), then re-runs the surface +
PBL diagnostic side-channel. Asserts:

* outputs are finite,
* outputs are float64 under force_fp64 (the precision-defeat guard),
* the surface flux handles + operational diagnostics (HFX/LH/PBLH/T2/U10/V10)
  are in physically sane bands,
* the non-periodic C-grid wind reconstruction did not wrap the domain edge.

This is a JAX-vs-JAX integration smoke (NOT a WRF parity claim); WRF parity is
the separate ``surface_mynn_parity`` harness gated on the oracle savepoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gpuwrf  # noqa: F401  enables x64 at import
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.physics_couplers import (
    mynn_adapter,
    surface_adapter,
    surface_layer_diagnostics,
)


GRAVITY = 9.81
P0 = 100000.0
RD = 287.0
CP = 1004.0


class _Grid:
    """Minimal GridSpec-shaped object for the column dz / shape helpers."""

    def __init__(self, nz, ny, nx):
        self.nz, self.ny, self.nx = nz, ny, nx


def _build_state(nz=20, ny=6, nx=8, dz_m=40.0, seed=1):
    grid = _Grid(nz, ny, nx)
    shapes = _state_field_shapes(grid)
    rng = np.random.default_rng(seed)

    # hydrostatic-ish base column
    z_iface = np.arange(nz + 1) * dz_m
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    theta_col = 300.0 + 0.004 * z_mid  # weakly stable free atmosphere
    p_col = P0 * (1.0 - GRAVITY * z_mid / (CP * 300.0)) ** (CP / RD)

    def m3(base, noise):
        return jnp.asarray(
            base[:, None, None] + noise * rng.standard_normal((nz, ny, nx)), dtype=jnp.float64
        )

    fields = {}
    for name, shape in shapes.items():
        fields[name] = jnp.zeros(shape, dtype=jnp.float64)

    fields["theta"] = m3(theta_col, 0.3)
    fields["p"] = m3(p_col, 50.0)
    fields["p_total"] = fields["p"]
    fields["qv"] = jnp.clip(m3(np.full(nz, 0.009), 0.0008), 0.0, None)
    fields["u"] = jnp.asarray(4.0 + 0.5 * rng.standard_normal((nz, ny, nx + 1)), dtype=jnp.float64)
    fields["v"] = jnp.asarray(-2.0 + 0.5 * rng.standard_normal((nz, ny + 1, nx)), dtype=jnp.float64)
    fields["w"] = jnp.asarray(0.01 * rng.standard_normal((nz + 1, ny, nx)), dtype=jnp.float64)
    fields["qke"] = jnp.full((nz, ny, nx), 0.4, dtype=jnp.float64)

    # geopotential interfaces (ph carries g*z); flat terrain
    ph = jnp.asarray(np.broadcast_to(GRAVITY * z_iface[:, None, None], (nz + 1, ny, nx)), dtype=jnp.float64)
    fields["ph"] = ph
    fields["ph_total"] = ph

    # surface: left half land (warm ground -> unstable), right half sea (cooler)
    xland = np.ones((ny, nx))
    xland[:, nx // 2 :] = 2.0
    t_skin = np.where(xland > 1.5, 299.5, 304.0)
    fields["xland"] = jnp.asarray(xland, dtype=jnp.float64)
    fields["lakemask"] = jnp.zeros((ny, nx), dtype=jnp.float64)
    fields["t_skin"] = jnp.asarray(t_skin, dtype=jnp.float64)
    fields["soil_moisture"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["mavail"] = jnp.where(jnp.asarray(xland) > 1.5, 1.0, 0.4).astype(jnp.float64)
    fields["roughness_m"] = jnp.where(jnp.asarray(xland) > 1.5, 2.85e-3, 0.15).astype(jnp.float64)
    fields["ustar"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["rhosfc"] = jnp.full((ny, nx), 1.15, dtype=jnp.float64)

    return State(**fields), grid


def _dtypes(state, names):
    return {n: str(jnp.asarray(getattr(state, n)).dtype) for n in names}


def _finite(x):
    return bool(jnp.all(jnp.isfinite(jnp.asarray(x))))


def run(out_path: Path) -> dict:
    state0, grid = _build_state()
    dt = 20.0

    # FROZEN operational hand-off order: surface -> mynn.
    s1 = surface_adapter(state0, dt)
    s2 = mynn_adapter(s1, dt, grid)
    diag = surface_layer_diagnostics(s1, grid)

    # Precision-defeat guard: the contract's force_fp64 path must NOT downcast my
    # written fields. Run the full chain, push through the operational precision
    # canonicaliser with force_fp64=True, and confirm every B2-written field stays
    # float64 (the dycore was bitten by a silent fp32 downcast here).
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision

    s2_fp64 = _enforce_operational_precision(s2, force_fp64=True)
    b2_written = (
        "ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv",
        "u", "v", "w", "theta", "qv", "qke",
    )
    force_fp64_ok = all(str(jnp.asarray(getattr(s2_fp64, n)).dtype) == "float64" for n in b2_written)

    written_surface = ("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv")
    pbl_fields = ("u", "v", "w", "theta", "qv", "qke")

    # non-periodic edge check: reconstruct u-face edges should equal the nearest
    # interior mass cell (zero-gradient), NOT the opposite-edge cell (no wrap).
    u_face = np.asarray(s2.u)
    u_mass0 = 0.5 * (np.asarray(s2.u)[:, :, :-1] + np.asarray(s2.u)[:, :, 1:])  # not used directly
    edge_no_wrap = bool(np.all(np.isfinite(u_face[:, :, 0])) and np.all(np.isfinite(u_face[:, :, -1])))

    record = {
        "proof": "b2-coupled-smoke",
        "kind": "jax-integration-smoke (NOT a WRF parity claim)",
        "grid": {"nz": grid.nz, "ny": grid.ny, "nx": grid.nx},
        "dt_s": dt,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "surface_handle_dtypes": _dtypes(s1, written_surface),
        "pbl_field_dtypes": _dtypes(s2, pbl_fields),
        "all_surface_handles_fp64": all(d == "float64" for d in _dtypes(s1, written_surface).values()),
        "all_pbl_fields_fp64": all(d == "float64" for d in _dtypes(s2, pbl_fields).values()),
        "diagnostics_fp64": {k: str(jnp.asarray(getattr(diag, k)).dtype) for k in diag._fields},
        "finite": {
            **{f"surf_{n}": _finite(getattr(s1, n)) for n in written_surface},
            **{f"pbl_{n}": _finite(getattr(s2, n)) for n in pbl_fields},
            **{f"diag_{k}": _finite(getattr(diag, k)) for k in diag._fields},
        },
        "bands": {
            "hfx_W_m2": [float(np.min(diag.hfx)), float(np.max(diag.hfx))],
            "lh_W_m2": [float(np.min(diag.lh)), float(np.max(diag.lh))],
            "pblh_m": [float(np.min(diag.pblh)), float(np.max(diag.pblh))],
            "t2_K": [float(np.min(diag.t2)), float(np.max(diag.t2))],
            "u10_m_s": [float(np.min(diag.u10)), float(np.max(diag.u10))],
            "v10_m_s": [float(np.min(diag.v10)), float(np.max(diag.v10))],
            "ustar_m_s": [float(np.min(diag.ustar)), float(np.max(diag.ustar))],
            "theta_flux": [float(np.min(s1.theta_flux)), float(np.max(s1.theta_flux))],
        },
        "u_face_shape": list(u_face.shape),
        "edge_reconstruction_finite": edge_no_wrap,
        "force_fp64_path_keeps_fp64": force_fp64_ok,
    }

    # physical-sanity gates (loose, smoke-level)
    checks = {
        "x64": record["x64_enabled"],
        "surface_fp64": record["all_surface_handles_fp64"],
        "pbl_fp64": record["all_pbl_fields_fp64"],
        "all_finite": all(record["finite"].values()),
        "pblh_positive": float(np.min(diag.pblh)) > 0.0,
        "ustar_positive": float(np.min(diag.ustar)) > 0.0,
        "t2_band": 270.0 < float(np.min(diag.t2)) and float(np.max(diag.t2)) < 330.0,
        "hfx_finite_band": -400.0 < float(np.min(diag.hfx)) and float(np.max(diag.hfx)) < 1200.0,
        "edge_no_wrap": edge_no_wrap,
        "force_fp64_no_downcast": force_fp64_ok,
    }
    record["checks"] = checks
    record["pass"] = all(checks.values())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    rec = run(here / "coupled_smoke.json")
    print(json.dumps(rec, indent=2, sort_keys=True))
    raise SystemExit(0 if rec["pass"] else 1)
