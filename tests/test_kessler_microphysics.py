from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gpuwrf.physics.microphysics_kessler import kessler_physics_tendency


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "proofs" / "v060" / "kessler_savepoint_parity_report.json"


def test_kessler_adapter_returns_valid_physics_tendency():
    ncol, nlev = 1, 4
    theta = jnp.full((ncol, nlev), 300.0, dtype=jnp.float64)
    qv = jnp.full((ncol, nlev), 0.014, dtype=jnp.float64)
    qc = jnp.asarray([[0.0, 1.2e-3, 8.0e-4, 0.0]], dtype=jnp.float64)
    qr = jnp.asarray([[2.0e-4, 4.0e-4, 2.0e-4, 0.0]], dtype=jnp.float64)
    rho = jnp.full((ncol, nlev), 1.0, dtype=jnp.float64)
    pii = jnp.full((ncol, nlev), 0.98, dtype=jnp.float64)
    z = jnp.asarray([[100.0, 300.0, 650.0, 1100.0]], dtype=jnp.float64)
    dz8w = jnp.asarray([[200.0, 200.0, 350.0, 450.0]], dtype=jnp.float64)

    tendency = kessler_physics_tendency(theta, qv, qc, qr, rho, pii, z, dz8w, 30.0)
    tendency.validate_keys()

    assert set(tendency.state_replacements) == {"theta", "qv", "qc", "qr"}
    assert set(tendency.accumulator_increments) == {"rain_acc"}
    assert set(tendency.diagnostics) == {"rainncv"}
    assert tendency.accumulator_increments["rain_acc"].shape == (ncol,)
    assert jnp.all(tendency.state_replacements["qc"] >= 0.0)
    assert jnp.all(tendency.state_replacements["qr"] >= 0.0)


def test_kessler_savepoint_parity_report_passes():
    with open(REPORT, encoding="utf-8") as fh:
        report = json.load(fh)

    assert report["scheme"] == "Kessler warm rain (mp_physics=1)"
    assert report["oracle"]["source_unmodified"] is True
    assert report["oracle"]["full_wrf_exe"] is False
    assert report["overall_pass"] is True
    assert set(report["cases"]) == {"1", "2", "3", "4", "5"}
