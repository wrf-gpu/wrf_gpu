#!/usr/bin/env python3
"""v0.22 F3 CAM-UW PBL source-present + idealized oracle gate.

This is deliberately fail-closed on missing pristine WRF CAM-UW source files and
deliberately does not claim full CAM-stack savepoint parity when no numerical
WRF savepoint fixture is present.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.physics_dispatch import resolve_physics_suite
from gpuwrf.coupling.scan_adapters import PBL_SCAN_ADAPTERS, camuw_pbl_adapter
from gpuwrf.physics.bl_camuw import camuw_columns

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "proofs/v022/feature_push/f3_camuw_pbl_oracle.json"
WRF_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF/phys")
WRF_SOURCES = [
    WRF_ROOT / "module_bl_camuwpbl_driver.F",
    WRF_ROOT / "module_cam_bl_eddy_diff.F",
    WRF_ROOT / "module_cam_bl_diffusion_solver.F",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _column_case() -> dict[str, Any]:
    n = 12
    z_mid = np.linspace(40.0, 6500.0, n)
    dz = np.full(n, 540.0)
    p = np.linspace(96000.0, 38000.0, n)
    pii = (p / 1.0e5) ** (287.0 / 1004.0)
    t = np.linspace(297.0, 255.0, n)
    theta = t / pii
    qv = np.maximum(0.012 * np.exp(-z_mid / 2200.0), 4.0e-5)
    qc = np.zeros(n)
    qi = np.zeros(n)
    qc[2:4] = 5.0e-5
    u = np.linspace(6.0, 16.0, n)
    v = np.linspace(1.0, 4.0, n)

    A = lambda x: jnp.asarray(np.stack([x, x]), jnp.float64)
    sc = lambda x: jnp.asarray([x, x], jnp.float64)
    out = jax.jit(camuw_columns)(
        A(u),
        A(v),
        A(t),
        A(theta),
        A(qv),
        A(qc),
        A(qi),
        A(p),
        A(pii),
        A(dz),
        A(z_mid),
        jnp.full((2, n), 0.025, jnp.float64),
        hfx=sc(200.0),
        qfx=sc(1.0e-4),
        ust=sc(0.4),
        wspd=sc(float(np.hypot(u[0], v[0]))),
        dt=60.0,
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    arrays = {k: np.asarray(v) for k, v in out.items()}
    finite = all(np.all(np.isfinite(a)) for a in arrays.values())
    checks = {
        "finite": bool(finite),
        "pblh_positive": bool(np.all(arrays["pblh"] > 0.0)),
        "pblh_below_top": bool(np.all(arrays["pblh"] < z_mid[-1])),
        "kvm_positive": bool(np.max(arrays["kvm"]) > 0.05),
        "kvm_bounded": bool(np.max(arrays["kvm"]) <= 1000.0 + 1.0e-9),
        "tke_nonnegative": bool(np.min(arrays["tke"]) >= 0.0),
        "surface_drag_decelerates_u": bool(arrays["u"][0, 0] < 0.0),
        "positive_hfx_warms_surface_theta": bool(arrays["theta"][0, 0] > 0.0),
    }
    return {
        "checks": checks,
        "metrics": {
            "backend": jax.default_backend(),
            "pblh_m": arrays["pblh"],
            "max_kvm_m2_s": float(np.max(arrays["kvm"])),
            "max_kvh_m2_s": float(np.max(arrays["kvh"])),
            "min_tke_m2_s2": float(np.min(arrays["tke"])),
            "max_tke_m2_s2": float(np.max(arrays["tke"])),
            "surface_u_tendency_m_s2": float(arrays["u"][0, 0]),
            "surface_theta_tendency_k_s": float(arrays["theta"][0, 0]),
        },
    }


def _adapter_case() -> dict[str, Any]:
    class Grid:
        nz = 6
        ny = 2
        nx = 2

    grid = Grid()
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in _state_field_shapes(grid).items()}
    p = jnp.broadcast_to(jnp.linspace(95500.0, 36000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 7000.0 * 9.80665, nz + 1)[:, None, None], (nz + 1, ny, nx))
    fields.update(
        theta=jnp.broadcast_to(jnp.linspace(296.0, 311.0, nz)[:, None, None], (nz, ny, nx)),
        p_total=p,
        ph_total=ph,
        mu_total=jnp.full((ny, nx), 90000.0),
        qv=jnp.full((nz, ny, nx), 7.0e-3),
        qc=jnp.full((nz, ny, nx), 2.0e-5),
        qi=jnp.zeros((nz, ny, nx)),
        qke=jnp.full((nz, ny, nx), 0.025),
        u=jnp.full((nz, ny, nx + 1), 6.0),
        v=jnp.full((nz, ny + 1, nx), 1.5),
        t_skin=jnp.full((ny, nx), 300.0),
        xland=jnp.ones((ny, nx)),
        mavail=jnp.full((ny, nx), 0.7),
        roughness_m=jnp.full((ny, nx), 0.08),
        ustar=jnp.full((ny, nx), 0.35),
        lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    state = State(**fields)
    after = camuw_pbl_adapter(state, 30.0)
    arrays = {leaf: np.asarray(getattr(after, leaf)) for leaf in ("theta", "qv", "qc", "qi", "u", "v", "qke")}
    checks = {
        "finite": bool(all(np.all(np.isfinite(a)) for a in arrays.values())),
        "theta_changed": bool(not np.allclose(np.asarray(state.theta), arrays["theta"])),
        "u_changed": bool(not np.allclose(np.asarray(state.u), arrays["u"])),
        "qke_changed": bool(not np.allclose(np.asarray(state.qke), arrays["qke"])),
        "dispatch_routes_bl9": bool(resolve_physics_suite({"bl_pbl_physics": 9, "sf_sfclay_physics": 1}).pbl.option == 9),
        "scan_adapter_registered": bool(PBL_SCAN_ADAPTERS.get(9) is camuw_pbl_adapter),
    }
    return {"checks": checks}


def main() -> int:
    missing = [str(path) for path in WRF_SOURCES if not path.exists()]
    source_hashes = {str(path): _sha256(path) for path in WRF_SOURCES if path.exists()}
    report: dict[str, Any] = {
        "feature": "F3 CAM-UW PBL bl_pbl_physics=9",
        "wrf_reference": {
            "required_sources": [str(path) for path in WRF_SOURCES],
            "missing": missing,
            "sha256": source_hashes,
        },
        "wrf_numerical_savepoint": {
            "status": "absent_fail_closed",
            "parity_claim": False,
            "reason": "No pristine-WRF CAM-UW single-column savepoint fixture is present in-tree; this gate only proves source presence plus idealized finite/plausible behavior.",
        },
        "landed": [
            "bl_pbl_physics=9 accepted in registry/catalog",
            "physics_dispatch routes option 9",
            "PBL_SCAN_ADAPTERS[9] registered",
            "JAX CAM-UW column finite on idealized moist boundary layer",
            "State adapter updates u/v/theta/qv/qc/qi/qke without host transfer",
        ],
        "scaffold": [
            "No full CAM cloud-number/dropmixnuc/sedimentation parity",
            "No persistent CAM residual-stress or previous-step kvh/kvm carry yet",
            "No pristine-WRF numerical CAM-UW savepoint parity claim",
        ],
    }
    if missing:
        report["gate"] = "FAIL_CLOSED"
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=_jsonable) + "\n")
        return 2

    column = _column_case()
    adapter = _adapter_case()
    report["column_case"] = column
    report["adapter_case"] = adapter
    all_checks = list(column["checks"].values()) + list(adapter["checks"].values())
    report["gate"] = "PASS" if all(all_checks) else "FAIL"
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=_jsonable) + "\n")
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
