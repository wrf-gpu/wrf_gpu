from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gpuwrf.init.real_init.lateral_bc import generate_lateral_bc
from gpuwrf.init.real_init.types import RVOVRD, T0, RealInitConfig


def _config() -> RealInitConfig:
    return RealInitConfig(
        nz=2,
        p_top_pa=5000.0,
        hybrid_opt=2,
        etac=0.2,
        spec_bdy_width=2,
        interval_seconds=10,
    )


def _snapshot(valid_time: str, offset: float = 0.0) -> SimpleNamespace:
    nz, ny, nx = 2, 4, 5
    mass_shape = (nz, ny, nx)
    u = np.arange(nz * ny * (nx + 1), dtype=np.float64).reshape(nz, ny, nx + 1) + 1.0 + offset
    v = np.arange(nz * (ny + 1) * nx, dtype=np.float64).reshape(nz, ny + 1, nx) + 2.0 + offset
    theta = np.arange(np.prod(mass_shape), dtype=np.float64).reshape(mass_shape) * 0.1 + offset
    qv = np.full(mass_shape, 0.01 + 0.001 * offset, dtype=np.float64)
    ph = np.arange((nz + 1) * ny * nx, dtype=np.float64).reshape(nz + 1, ny, nx) + 3.0 + offset
    mu = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) + 100.0 + offset
    mub = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx) * 0.5 + 1000.0
    dynamics = SimpleNamespace(
        u=u,
        v=v,
        theta=theta,
        qv=qv,
        ph=ph,
        mu=mu,
        qc=np.full(mass_shape, 0.002 + 0.0001 * offset, dtype=np.float64),
    )
    base = SimpleNamespace(mub=mub)
    surface = SimpleNamespace(
        mapfac_uy=np.full((ny, nx + 1), 2.0, dtype=np.float64),
        mapfac_vx=np.full((ny + 1, nx), 4.0, dtype=np.float64),
    )
    vcoord = SimpleNamespace(
        c1h=np.array([0.25, 0.75], dtype=np.float64),
        c2h=np.array([10.0, 20.0], dtype=np.float64),
        c1f=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        c2f=np.array([5.0, 15.0, 25.0], dtype=np.float64),
    )
    return SimpleNamespace(
        valid_time=valid_time,
        dynamics=dynamics,
        base=base,
        surface=surface,
        vcoord=vcoord,
    )


def _mass_u(total_mass: np.ndarray) -> np.ndarray:
    ny, nx = total_mass.shape
    out = np.empty((ny, nx + 1), dtype=np.float64)
    out[:, 0] = total_mass[:, 0]
    out[:, 1:nx] = 0.5 * (total_mass[:, 1:] + total_mass[:, :-1])
    out[:, nx] = total_mass[:, -1]
    return out


def _mass_h(snapshot: SimpleNamespace) -> np.ndarray:
    total = snapshot.dynamics.mu + snapshot.base.mub
    c1h = snapshot.vcoord.c1h
    c2h = snapshot.vcoord.c2h
    return c1h[:, None, None] * total[None, :, :] + c2h[:, None, None]


def _stuff_3d(field: np.ndarray, width: int) -> dict[str, np.ndarray]:
    return {
        "bxs": np.moveaxis(field[:, :, :width], 2, 0),
        "bxe": np.moveaxis(field[:, :, -width:][:, :, ::-1], 2, 0),
        "bys": np.moveaxis(field[:, :width, :], 1, 0),
        "bye": np.moveaxis(field[:, -width:, :][:, ::-1, :], 1, 0),
    }


def test_generate_lateral_bc_couples_hybrid_mass_and_stuffs_wrf_order() -> None:
    config = _config()
    first = _snapshot("2026-05-21_18:00:00")
    second = _snapshot("2026-05-22_00:00:00", offset=1.5)

    lbc = generate_lateral_bc(config, [first, second])

    total = first.dynamics.mu + first.base.mub
    expected_u = first.dynamics.u * (
        first.vcoord.c1h[:, None, None] * _mass_u(total)[None, :, :]
        + first.vcoord.c2h[:, None, None]
    ) / first.surface.mapfac_uy[None, :, :]
    thm = (first.dynamics.theta + T0) * (1.0 + RVOVRD * first.dynamics.qv) - T0
    expected_t = thm * _mass_h(first)

    np.testing.assert_allclose(lbc.values["u"]["bxe"][0], _stuff_3d(expected_u, 2)["bxe"])
    np.testing.assert_allclose(lbc.values["t"]["bxs"][0], _stuff_3d(expected_t, 2)["bxs"])
    assert lbc.values["u"]["bxe"].shape == (1, 2, 2, 4)
    assert lbc.values["v"]["bys"].shape == (1, 2, 2, 5)
    assert lbc.values["mu"]["bxs"].shape == (1, 2, 4)
    assert lbc.valid_times == ("2026-05-21_18:00:00",)


def test_generate_lateral_bc_stacks_intervals_and_tendencies() -> None:
    config = _config()
    snapshots = [
        _snapshot("2026-05-21_18:00:00", offset=0.0),
        _snapshot("2026-05-22_00:00:00", offset=1.0),
        _snapshot("2026-05-22_06:00:00", offset=3.0),
    ]

    lbc = generate_lateral_bc(config, snapshots)

    def coupled_qc(snapshot: SimpleNamespace) -> np.ndarray:
        return snapshot.dynamics.qc * _mass_h(snapshot)

    expected_first = _stuff_3d(coupled_qc(snapshots[0]), 2)["bye"]
    expected_tendency = _stuff_3d(
        (coupled_qc(snapshots[2]) - coupled_qc(snapshots[1])) / 10.0,
        2,
    )["bye"]

    assert lbc.coupled_field_names == ("u", "v", "t", "ph", "qv", "mu", "qc")
    assert lbc.values["qc"]["bye"].shape[0] == 2
    np.testing.assert_allclose(lbc.values["qc"]["bye"][0], expected_first)
    np.testing.assert_allclose(lbc.tendencies["qc"]["btye"][1], expected_tendency)
    assert lbc.valid_times == ("2026-05-21_18:00:00", "2026-05-22_00:00:00")
