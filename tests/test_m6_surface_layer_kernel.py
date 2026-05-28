from __future__ import annotations

from pathlib import Path
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.surface_constants import CP_D, P0_PA, R_D_OVER_CP
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics


class SurfaceState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _state_from_rows(rows):
    data = np.asarray(rows, dtype=np.float64)
    u, v, t, qv, p, dz, tsk, xland, znt, mavail, old_ust, _old_mol = data.T
    theta = t * (P0_PA / p) ** R_D_OVER_CP
    shape = (data.shape[0], 1)
    return SurfaceState(
        u=jnp.asarray(u.reshape(shape + (1,))),
        v=jnp.asarray(v.reshape(shape + (1,))),
        theta=jnp.asarray(theta.reshape(shape + (1,))),
        qv=jnp.asarray(qv.reshape(shape + (1,))),
        p=jnp.asarray(p.reshape(shape + (1,))),
        dz=jnp.asarray(dz.reshape(shape + (1,))),
        t_skin=jnp.asarray(tsk.reshape(shape)),
        xland=jnp.asarray(xland.reshape(shape)),
        roughness_m=jnp.asarray(znt.reshape(shape)),
        mavail=jnp.asarray(mavail.reshape(shape)),
        ustar=jnp.asarray(old_ust.reshape(shape)),
    )


def test_surface_layer_returns_fp64_finite_fluxes_and_diagnostics():
    rows = [
        [6.0, 2.0, 292.0, 0.008, 95500.0, 80.0, 296.0, 1.0, 0.08, 0.7, 0.0, 0.0],
        [9.0, -1.0, 294.0, 0.010, 100800.0, 60.0, 293.0, 2.0, 0.0015, 1.0, 0.0, 0.0],
    ]
    diag = surface_layer_with_diagnostics(_state_from_rows(rows))

    for value in (
        diag.fluxes.ustar,
        diag.fluxes.theta_flux,
        diag.fluxes.qv_flux,
        diag.fluxes.tau_u,
        diag.fluxes.tau_v,
        diag.fluxes.rhosfc,
        diag.fluxes.fltv,
        diag.u10,
        diag.v10,
        diag.t2,
        diag.q2,
    ):
        assert value.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(value)))
        assert value.shape == (2, 1)

    assert bool(jnp.all(diag.fluxes.ustar > 0.0))
    assert bool(jnp.all(diag.fluxes.rhosfc > 0.0))


def test_surface_layer_matches_wrf_sfclay_harness_when_available(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build = root / "scripts" / "wrf_sfclay_harness_build.sh"
    try:
        harness_path = subprocess.check_output([str(build)], cwd=root, text=True, timeout=60).strip().splitlines()[-1]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"WRF SFCLAY harness unavailable: {exc}")

    nm = subprocess.check_output(["nm", harness_path], text=True)
    assert "module_sf_sfclay_sfclay1d" in nm.lower()

    rows = np.asarray(
        [
            [6.0, 2.0, 292.0, 0.008, 95500.0, 80.0, 296.0, 1.0, 0.08, 0.7, 0.0, 0.0],
            [9.0, -1.0, 294.0, 0.010, 100800.0, 60.0, 293.0, 2.0, 0.0015, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    inp = tmp_path / "sfclay_input.dat"
    out = tmp_path / "sfclay_output.dat"
    with inp.open("w", encoding="utf-8") as handle:
        handle.write(f"{rows.shape[0]}\n")
        for row in rows:
            handle.write(" ".join(f"{item:.16e}" for item in row) + "\n")
    subprocess.check_call([harness_path, str(inp), str(out)], cwd=root, timeout=30)
    wrf = np.loadtxt(out, skiprows=1)

    diag = surface_layer_with_diagnostics(_state_from_rows(rows))
    cpm = CP_D * (1.0 + 0.8 * rows[:, 3])
    py = np.column_stack(
        [
            np.asarray(diag.fluxes.ustar).reshape(-1),
            np.asarray(diag.fluxes.theta_flux).reshape(-1) * np.asarray(diag.fluxes.rhosfc).reshape(-1) * cpm,
            np.asarray(diag.fluxes.qv_flux).reshape(-1) * np.asarray(diag.fluxes.rhosfc).reshape(-1),
            np.asarray(diag.u10).reshape(-1),
            np.asarray(diag.v10).reshape(-1),
            np.asarray(diag.th2).reshape(-1),
            np.asarray(diag.t2).reshape(-1),
            np.asarray(diag.q2).reshape(-1),
            np.asarray(diag.bulk_richardson).reshape(-1),
            np.asarray(diag.z_over_l).reshape(-1),
            np.asarray(diag.fm).reshape(-1),
            np.asarray(diag.fh).reshape(-1),
        ]
    )
    wrf_subset = wrf[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]]
    np.testing.assert_allclose(py, wrf_subset, rtol=2.0e-4, atol=2.0e-4)
