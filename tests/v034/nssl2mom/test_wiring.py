"""Deletion-sensitive tests of the mp=18 wiring (rv-nssl F2/F3/F4).

(a) State -> nssl2mom_adapter -> State vs the pristine-WRF fp64 oracle on an active column (case 4) with WRF's
    phy_prep density rho_d*(1+qv); the dry-density mutant (the pre-F4 adapter) must fail.  Template: rv-nssl
    adapter_probe.py (columns replicated on a 2x2 State).
(b) root lateral boundary routing of the NSSL extras: flow_only -> WRF flow_dep_bdy, have_bcs guard, mp gating,
    number scalars added to the transported species.
(c) mp_physics=18 with cu_physics /= 0 is refused pre-JAX (calcnfromcuten not ported).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_io as oio  # noqa: E402

RVOVRD = 461.6 / 287.0
LEAF = {"qv": "QV", "qc": "QC", "qr": "QR", "qi": "QI", "qs": "QS", "qg": "QH", "qh": "QHL",
        "Nc": "CCW", "Nr": "CRW", "Ni": "CCI", "Ns": "CSW", "Ng": "CHW", "Nh": "CHL", "Nn": "CN",
        "qvolg": "VHW", "qvolh": "VHL"}
FLOOR = {k: 1e-14 for k in ("qv", "qc", "qr", "qi", "qs", "qg", "qh")}
FLOOR.update({k: 1e-6 for k in ("Nc", "Nr", "Ni", "Ns", "Ng", "Nh", "Nn")})
FLOOR.update(qvolg=1e-17, qvolh=1e-17)


def _state(sp):
    from gpuwrf.contracts import state as sc
    from gpuwrf.contracts.state import State
    sc._gpu_device = lambda: jax.devices("cpu")[0]
    col = sp["columns"]
    nz, ny, nx = len(col["TH_IN"]), 2, 2
    st = State.zeros(types.SimpleNamespace(nz=nz, ny=ny, nx=nx), mp_physics=18)
    b = lambda v: jnp.asarray(np.broadcast_to(np.asarray(v, np.float64)[:, None, None], (nz, ny, nx)).copy())  # noqa: E731
    qv = b(col["QV_IN"])
    ph = np.concatenate([[0.0], np.cumsum(9.81 * np.asarray(col["DZ"], np.float64))])
    upd = dict(theta=b(col["TH_IN"]) * (1.0 + RVOVRD * qv), p_total=b(col["P"]), p_perturbation=jnp.zeros((nz, ny, nx)),
               ph_total=jnp.asarray(np.broadcast_to(ph[:, None, None], (nz + 1, ny, nx)).copy()),
               ph_perturbation=jnp.zeros((nz + 1, ny, nx)),
               w=jnp.asarray(np.broadcast_to(np.append(np.asarray(col["W"], np.float64), 0.0)[:, None, None],
                                             (nz + 1, ny, nx)).copy()))
    upd.update({leaf: b(col[name + "_IN"]) for leaf, name in LEAF.items()})
    upd.update({acc: jnp.zeros((ny, nx)) for acc in ("rain_acc", "snow_acc", "graupel_acc", "hail_acc")})
    return st.replace(_cast=False, **upd)


def _worst(new, precip, sp):
    col, worst = sp["columns"], {}
    for leaf, name in LEAF.items():
        ref = np.asarray(col[name + "_OUT"], np.float64)
        got = np.asarray(getattr(new, leaf), np.float64)[:, 0, 0]
        big = np.abs(ref) > FLOOR[leaf]
        worst[leaf] = float(np.max(np.abs(got - ref)[big] / np.abs(ref[big]))) if big.any() else 0.0
    for k in ("rainncv", "snowncv", "grplncv"):
        ref = sp["scalars"][k.upper()]
        worst[k] = abs(float(np.asarray(precip[k])[0, 0]) - ref) / abs(ref)
    return worst


def _run_adapter(sp):
    from gpuwrf.physics.nssl2mom.adapter import nssl2mom_adapter
    dt, first = float(sp["scalars"]["DT"]), bool(sp["scalars"]["ITIMESTEP"] == 1)
    fn = jax.jit(lambda st: nssl2mom_adapter(st, dt, first_step=first, return_precipitation=True))
    return jax.tree_util.tree_map(np.asarray, fn(_state(sp)))


def test_adapter_density_matches_oracle_and_dry_mutant_dies(monkeypatch):
    """(a) F4: the coupled adapter reproduces the oracle (which passes DN = rho_d*(1+qv)); the dry density dies."""
    sp = oio.load_case("fp64", 4)  # cold start, active rain/snow/graupel, dt 60
    new, precip = _run_adapter(sp)
    w = _worst(new, precip, sp)
    # remaining ~1e-4 is the shared pii/T convention (R_D_OVER_CP), rv-nssl F4
    assert max(w.values()) <= 1e-3, w
    from gpuwrf.coupling import physics_couplers as pc
    dry = pc.density_from_pressure_temperature
    monkeypatch.setattr(pc, "density_from_pressure_temperature", lambda p, T, qv: dry(p, T, qv) / (1.0 + qv))
    new_m, precip_m = _run_adapter(sp)
    wm = _worst(new_m, precip_m, sp)
    assert max(wm[k] for k in ("rainncv", "snowncv", "grplncv")) > 1e-3, wm


def test_root_flow_only_routing_of_nssl_extras():
    """(b) NSSL extras without boundary records take WRF flow_dep_bdy; have_bcs True fails closed; other mp unchanged."""
    from gpuwrf.coupling.boundary_apply import root_scalar_rk1_split
    from gpuwrf.runtime.operational_mode import _NSSL_NUMBER_SCALARS, _nssl_number_scalars, _nssl_root_flow_only
    nml18 = types.SimpleNamespace(mp_physics=18, moist_adv_opt=1, scalar_adv_opt=1)
    extras = ("qh", "qvolg", "qvolh") + _NSSL_NUMBER_SCALARS
    assert set(_nssl_root_flow_only(nml18)["flow_only"]) == set(extras)
    assert _nssl_root_flow_only(types.SimpleNamespace(mp_physics=8)) == {}
    assert _nssl_root_flow_only(types.SimpleNamespace()) == {}
    assert set(_nssl_number_scalars(nml18)) == {"Nc", "Ns", "Ng", "Nh", "Nn"}
    with pytest.raises(NotImplementedError):
        _nssl_number_scalars(types.SimpleNamespace(mp_physics=18, moist_adv_opt=1, scalar_adv_opt=2))
    species = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qvolg", "qvolh", "Ni", "Nr") + _NSSL_NUMBER_SCALARS
    cfg = types.SimpleNamespace(have_bcs_moist=False, have_bcs_scalar=False)
    relaxed, flow = root_scalar_rk1_split(cfg, species, **_nssl_root_flow_only(nml18))
    assert relaxed == ("qv",)
    assert set(extras) <= set(flow) and set(flow) == set(species) - {"qv"}
    with pytest.raises(NotImplementedError):  # without the NSSL routing the extras are unrepresented
        root_scalar_rk1_split(cfg, species)
    with pytest.raises(NotImplementedError):  # wrfbdy claims scalar records -> fail closed (no *_bdy leaves)
        root_scalar_rk1_split(types.SimpleNamespace(have_bcs_moist=False, have_bcs_scalar=True), species,
                              **_nssl_root_flow_only(nml18))
    with pytest.raises(NotImplementedError):  # qh keys on have_bcs_moist
        root_scalar_rk1_split(types.SimpleNamespace(have_bcs_moist=True, have_bcs_scalar=False), ("qv", "qh"),
                              **_nssl_root_flow_only(nml18))


def test_nssl_with_cumulus_is_refused_pre_jax():
    """(c) F3: calcnfromcuten (cu_used=1) is not ported -> mp=18 + cu/=0 refused; cu=0 and mp=8+cu=1 accepted."""
    from gpuwrf.io.namelist_check import NotOperationallyWiredError, validate_operational_namelist
    with pytest.raises(NotOperationallyWiredError, match="cu_physics"):
        validate_operational_namelist({"physics": {"mp_physics": [18], "cu_physics": [1]}})
    with pytest.raises(NotOperationallyWiredError):
        validate_operational_namelist({"physics": {"mp_physics": [18, 18], "cu_physics": [0, 3]}})
    validate_operational_namelist({"physics": {"mp_physics": [18], "cu_physics": [0]}})
    validate_operational_namelist({"physics": {"mp_physics": [8], "cu_physics": [1]}})
