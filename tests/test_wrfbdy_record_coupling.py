"""Root wrfbdy leaves are decoupled with EACH record's own dry mass, as WRF real.exe coupled them.

WRF ``main/real_em.F:866-878, 1052-1064`` couples record k of wrfbdy with that record's
``mu_2 + mub`` (``couple``: u/v on ``calc_mu_staggered`` faces divided by msf, T/QV with
``c1h*mu + (c1h*mub + c2h)``, PH with c1f/c2f).  The port used to decouple every record with
the initial-condition mass, which scaled theta'/u/v/ph' by (c1*M(t)+c2)/(c1*M0+c2) (up to
2-4 % on the shipped Swiss case by 24 h).  The oracle below is an independent numpy
transcription of WRF ``couple`` inverted at the record mass; the vacuity guard proves the
case discriminates the IC-mass decoupling.
"""
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from netCDF4 import Dataset

SWISS = Path(__file__).resolve().parents[1] / "examples" / "switzerland_d01"
T0 = 300.0


def _wrf(path, name):
    with Dataset(path) as ds:
        return np.asarray(ds.variables[name][:], dtype=np.float64)


@pytest.fixture(scope="module")
def swiss_inputs():
    if not (SWISS / "wrfbdy_d01").exists():
        pytest.skip("shipped Swiss example unavailable")
    wi = SWISS / "wrfinput_d01"
    with Dataset(wi) as ds:
        assert int(ds.getncattr("USE_THETA_M")) == 1
    mu0, mub = _wrf(wi, "MU")[0], _wrf(wi, "MUB")[0]
    metrics = SimpleNamespace(
        c1h=_wrf(wi, "C1H")[0], c2h=_wrf(wi, "C2H")[0],
        c1f=_wrf(wi, "C1F")[0], c2f=_wrf(wi, "C2F")[0],
        msfuy=_wrf(wi, "MAPFAC_UY")[0], msfvx=_wrf(wi, "MAPFAC_VX")[0],
    )
    ny, nx = mu0.shape
    grid = SimpleNamespace(nx=nx, ny=ny, nz=int(metrics.c1h.shape[0]))
    return SimpleNamespace(mu0=mu0, mub=mub, metrics=metrics, grid=grid)


def _record(bdy, var, side, k, ntimes, interval):
    """Coupled wrfbdy strip of record k (k == ntimes: synthesized terminal record)."""

    b = _wrf(bdy, f"{var}_B{side}")
    if k < ntimes:
        return b[k]
    return b[ntimes - 1] + interval * _wrf(bdy, f"{var}_BT{side}")[ntimes - 1]


def _oracle(inputs, k, mass_mu=None):
    """WRF couple^-1 for record k at its own mass (or at ``mass_mu`` side strips if given)."""

    bdy = SWISS / "wrfbdy_d01"
    with Dataset(bdy) as ds:
        ntimes = ds.variables["MU_BXS"].shape[0]
        times = [bytes(t).decode() for t in np.asarray(ds.variables["Times"][:])]
    t0, t1 = (np.datetime64(t.replace("_", "T")) for t in times[:2])
    interval = float((t1 - t0) / np.timedelta64(1, "s"))
    m, mub, ny, nx = inputs.metrics, inputs.mub, inputs.grid.ny, inputs.grid.nx
    c1h, c2h, c1f, c2f = (x[None, :, None] for x in (m.c1h, m.c2h, m.c1f, m.c2f))
    mub_strip = {"XS": np.stack([mub[:, b] for b in range(5)]), "XE": np.stack([mub[:, nx - 1 - b] for b in range(5)]),
                 "YS": np.stack([mub[b, :] for b in range(5)]), "YE": np.stack([mub[ny - 1 - b, :] for b in range(5)])}
    out = {}
    for side in ("XS", "XE", "YS", "YE"):
        mu = mass_mu[side] if mass_mu is not None else _record(bdy, "MU", side, k, ntimes, interval)
        M = (mu + mub_strip[side])[:, None, :]                          # (bw, 1, tan) mass rows
        face = np.concatenate([M[:1], 0.5 * (M[1:] + M[:-1])], axis=0)  # normal faces, edge = row 0
        tang = np.concatenate([M[:, :, :1], 0.5 * (M[:, :, 1:] + M[:, :, :-1]), M[:, :, -1:]], axis=2)
        if side in ("XS", "XE"):
            cols = [b if side == "XS" else nx - b for b in range(5)]
            msf_n = np.stack([m.msfuy[:, c] for c in cols])[:, None, :]
            cols_t = [b if side == "XS" else nx - 1 - b for b in range(5)]
            msf_t = np.stack([m.msfvx[:, c] for c in cols_t])[:, None, :]
            u = _record(bdy, "U", side, k, ntimes, interval) * msf_n / (c1h * face + c2h)
            v = _record(bdy, "V", side, k, ntimes, interval) * msf_t / (c1h * tang + c2h)
        else:
            rows = [b if side == "YS" else ny - b for b in range(5)]
            msf_n = np.stack([m.msfvx[r, :] for r in rows])[:, None, :]
            rows_t = [b if side == "YS" else ny - 1 - b for b in range(5)]
            msf_t = np.stack([m.msfuy[r, :] for r in rows_t])[:, None, :]
            v = _record(bdy, "V", side, k, ntimes, interval) * msf_n / (c1h * face + c2h)
            u = _record(bdy, "U", side, k, ntimes, interval) * msf_t / (c1h * tang + c2h)
        th = _record(bdy, "T", side, k, ntimes, interval) / (c1h * M + c2h) + T0
        ph = _record(bdy, "PH", side, k, ntimes, interval) / (c1f * M + c2f)
        out[side] = dict(u_bdy=u, v_bdy=v, theta_bdy=th, ph_bdy=ph)
    return out, ntimes, interval


_SIDE = {"XS": "W", "XE": "E", "YS": "S", "YE": "N"}


def _leaf(leaves, name, k, side, shape):
    from gpuwrf.coupling.boundary_apply import SIDE_INDEX

    arr = np.asarray(jax.device_get(leaves[name]))[k, SIDE_INDEX[_SIDE[side]]]
    return arr[: shape[0], : shape[1], : shape[2]]


@pytest.fixture(scope="module")
def swiss_leaves(swiss_inputs):
    from gpuwrf.integration.d02_replay import load_wrfbdy_boundary_leaves
    from gpuwrf.io.gen2_accessor import Gen2Run

    leaves, meta = load_wrfbdy_boundary_leaves(
        Gen2Run(SWISS), swiss_inputs.grid, domain="d01", mu_total=swiss_inputs.mub + swiss_inputs.mu0,
        metrics=swiss_inputs.metrics, mub=swiss_inputs.mub,
    )
    return leaves, meta


@pytest.mark.parametrize("record", ["first", "last", "terminal"])
def test_root_wrfbdy_leaves_decoupled_with_record_mass(swiss_inputs, swiss_leaves, record):
    leaves, meta = swiss_leaves
    assert "per record" in meta["coupling"]
    with Dataset(SWISS / "wrfbdy_d01") as ds:
        ntimes = ds.variables["MU_BXS"].shape[0]
    k = {"first": 0, "last": ntimes - 1, "terminal": ntimes}[record]
    expected, _, _ = _oracle(swiss_inputs, k)
    for side, fields in expected.items():
        for name, want in fields.items():
            got = _leaf(leaves, name, k, side, want.shape)
            np.testing.assert_allclose(got, want, rtol=2e-6, atol=0.0, err_msg=f"{name} {side} record {k}")


def test_swiss_case_discriminates_ic_mass_decoupling(swiss_inputs, swiss_leaves):
    """Vacuity guard (E102): at the terminal record the IC-mass decoupling is far outside tolerance."""

    leaves, _ = swiss_leaves
    with Dataset(SWISS / "wrfbdy_d01") as ds:
        ntimes = ds.variables["MU_BXS"].shape[0]
    nx, ny = swiss_inputs.grid.nx, swiss_inputs.grid.ny
    mu0 = swiss_inputs.mu0
    ic_mu = {"XS": np.stack([mu0[:, b] for b in range(5)]), "XE": np.stack([mu0[:, nx - 1 - b] for b in range(5)]),
             "YS": np.stack([mu0[b, :] for b in range(5)]), "YE": np.stack([mu0[ny - 1 - b, :] for b in range(5)])}
    wrong, _, _ = _oracle(swiss_inputs, ntimes, mass_mu=ic_mu)
    right, _, _ = _oracle(swiss_inputs, ntimes)
    rel = max(float(np.max(np.abs(wrong[s]["u_bdy"] - right[s]["u_bdy"]) / (np.abs(right[s]["u_bdy"]).max())))
              for s in right)
    assert rel > 1e-3
    worst = max(float(np.max(np.abs((wrong[s]["theta_bdy"] - right[s]["theta_bdy"])))) for s in right)
    assert worst > 0.05  # K: the old leaves were this far off on the shipped case
    got = _leaf(leaves, "theta_bdy", ntimes, "YS", right["YS"]["theta_bdy"].shape)
    assert not np.allclose(got, wrong["YS"]["theta_bdy"], rtol=2e-6, atol=0.0)


def test_standalone_root_case_carries_record_mass_leaves(swiss_inputs):
    """Caller wiring (E11): the root ``build_replay_case(standalone=True)`` path used by the
    single-domain and nested release pipelines (nested_pipeline root, daily standalone) feeds
    the loader the wrfinput masses and stores the record-mass leaves on the State."""

    from gpuwrf.contracts import state as state_contract
    from gpuwrf.integration.d02_replay import build_replay_case

    original = state_contract._gpu_device
    state_contract._gpu_device = lambda: jax.devices()[0]
    try:
        case = build_replay_case(SWISS, domain="d01", standalone=True)
    finally:
        state_contract._gpu_device = original
    assert jax.devices()[0].platform == "cpu"
    with Dataset(SWISS / "wrfbdy_d01") as ds:
        ntimes = ds.variables["MU_BXS"].shape[0]
    expected, _, _ = _oracle(swiss_inputs, ntimes)
    leaves = {name: getattr(case.state, name) for name in ("u_bdy", "v_bdy", "theta_bdy", "ph_bdy")}
    for side, fields in expected.items():
        for name, want in fields.items():
            got = _leaf(leaves, name, ntimes, side, want.shape)
            np.testing.assert_allclose(got, want, rtol=2e-6, atol=0.0, err_msg=f"{name} {side}")
