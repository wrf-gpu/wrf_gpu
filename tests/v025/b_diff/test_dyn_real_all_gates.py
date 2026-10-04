"""review-realall C1 (item B): deletion-sensitive CPU gates for two GPUWRF_DYN_REAL_ALL hunks without a pytest.

(1) OperationalNamelist.__post_init__ makes DycoreMetrics + resident tendencies WRF REAL under the flag.
(2) rhs_ph REAL differences ph and phb separately (WRF literal, 93da0b755): on real-structured operands the
    REAL split stays far closer to the f64 reference than the REAL rounded-total association (E133). Reverting
    the split makes both arms identical and fails the ratio assertion.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__import__("gpuwrf").__file__).resolve().parents[2]
NATIVE = ("GPUWRF_DYN_FP32", "GPUWRF_DYN_RK_FP32", "GPUWRF_DYN_CARRY_FP32")


@pytest.fixture
def native(monkeypatch):
    for name in NATIVE:
        monkeypatch.setenv(name, "1")
    return monkeypatch


@pytest.mark.parametrize("flag", ("0", "1"))
def test_namelist_metrics_and_tendencies_follow_the_flag(native, flag):
    from gpuwrf.runtime import operational_mode as op

    from gpuwrf.contracts import state as state_contract

    native.setenv("GPUWRF_DYN_REAL_ALL", flag)
    native.setenv("GPUWRF_CARRY_REAL_ALL", flag)  # the v0.3 release pairs them; CARRY_REAL_ALL alone is refused
    native.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])  # CPU placement (E126)
    spec = importlib.util.spec_from_file_location("lw_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    grid = fixture._grid(ny=3, nx=3, nz=8)
    nml = op.OperationalNamelist.from_grid(grid)
    want = jnp.float32 if flag == "1" else jnp.float64
    leaves = [x for x in jax.tree.leaves(nml.metrics) if jnp.issubdtype(x.dtype, jnp.floating)]
    assert leaves and all(x.dtype == want for x in leaves), sorted({str(x.dtype) for x in leaves})
    assert nml.metrics.precision == ("fp32" if flag == "1" else "fp64")
    tend = [x.dtype for x in jax.tree.leaves(nml.tendencies) if jnp.issubdtype(x.dtype, jnp.floating)]
    if flag == "1":
        assert tend and all(d == jnp.float32 for d in tend), tend
    else:  # control: the default resident tendencies carry wide leaves (the REAL hunk is not vacuous)
        assert any(d == jnp.float64 for d in tend), tend


FIXTURES = Path("<USER_HOME>/wrf_gpu2_lanes/b-diff/rk_fixtures")  # b-diff real PROD RK fixtures (probe_rk.DATA)


@pytest.mark.parametrize("domain,dx", (("d01", 9000.0), ("d02", 3000.0)))
def test_rhs_ph_real_split_beats_rounded_total(native, domain, dx):
    """Reference = f64 arithmetic on the same REAL operands; review-realall CPU [M]: split/total rms
    d01 0.0174/0.213, d02 0.0125/0.599 (BD62 vs pristine REAL4: 0.0148/0.214, 0.028/0.599)."""
    import gpuwrf.dynamics.core.rhs_ph as rhs_mod
    from gpuwrf.dynamics.flux_advection import stage_omega_specified

    path = FIXTURES / f"{domain}.npz"
    if not path.is_file():
        pytest.skip(f"real RK fixture missing: {path}")
    z = np.load(path)
    g = lambda k: z[k].astype(np.float32).astype(np.float64)  # WRF REAL operands, held wide
    mut = (z["base_mub"].astype(np.float32) + z["state_mu_perturbation"].astype(np.float32)).astype(np.float64)
    px, py = np.pad(mut, ((0, 0), (1, 1)), mode="edge"), np.pad(mut, ((1, 1), (0, 0)), mode="edge")
    muu = (0.5 * (px[:, 1:] + px[:, :-1])).astype(np.float32).astype(np.float64)
    muv = (0.5 * (py[1:] + py[:-1])).astype(np.float32).astype(np.float64)
    m = {k[7:]: g(k) for k in z.files if k.startswith("metric_")}
    native.setenv("GPUWRF_DYN_REAL_ALL", "0")
    native.setenv("GPUWRF_CARRY_REAL_ALL", "0")  # the v0.3 release pairs them; CARRY_REAL_ALL alone is refused
    ww = np.asarray(stage_omega_specified(
        jnp.asarray(g("state_u")), jnp.asarray(g("state_v")), jnp.asarray(mut), c1h=m["c1h"], c2h=m["c2h"],
        dnw=m["dnw"], rdx=1 / dx, rdy=1 / dx, msfuy=m["msfuy"], msfvx=m["msfvx"], msftx=m["msftx"],
    )).astype(np.float32).astype(np.float64)
    dn, dnw = np.float32(m["dn"][-1]), np.float32(m["dnw"][-1])
    cfn, cfn1 = float(np.float32((np.float32(.5) * dnw + dn) / dn)), float(np.float32(-np.float32(.5) * dnw / dn))
    ops = dict(u=g("state_u"), v=g("state_v"), ww=ww, ph=g("state_ph_perturbation"), phb=g("base_phb"), w=g("state_w"),
               mut=mut, muu=muu, muv=muv, c1f=m["c1f"], c2f=m["c2f"], fnm=m["fnm"], fnp=m["fnp"], rdnw=m["rdnw"],
               msfty=m["msfty"], msfux=m["msfux"], msfvy=m["msfvy"])

    def run(dtype):
        a = {k: jnp.asarray(v, dtype) for k, v in ops.items()}
        return np.asarray(jax.jit(lambda a: rhs_mod.rhs_ph_wrf(
            **a, rdx=1 / dx, rdy=1 / dx, non_hydrostatic=True, advective_order=5, specified=True,
            cfn=cfn, cfn1=cfn1, top_lid=True))(a), np.float64)

    ref = run(jnp.float64)
    native.setenv("GPUWRF_DYN_REAL_ALL", "1")
    split = run(jnp.float32)
    native.setattr(rhs_mod, "dyn_real_enabled", lambda: False)  # REAL operands, rounded ph+phb total (E133 mutant)
    total = run(jnp.float32)
    inner = np.s_[1:ops["u"].shape[0]]
    e = lambda x: float(np.sqrt(np.mean((x[inner] - ref[inner]) ** 2)))
    assert split.dtype == np.float64 and np.isfinite(split).all()
    assert e(split) < 0.2 * e(total), (e(split), e(total))
