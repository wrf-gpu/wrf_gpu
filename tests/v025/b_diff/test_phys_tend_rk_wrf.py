"""GPUWRF_PHYS_TEND_RK_WRF (D2/D3) vs unchanged pristine WRF REAL4 on real P0227 operands.

Operands: CPU-WRF P0227 d02 restart at 2026-03-01_00 (night): QVAPOR/QCLOUD/QICE/QNICE, the uncoupled
RQVBLTEN/RQCBLTEN/RQIBLTEN/RQNIBLTEN and RTHRATEN WRF wrote, MU_1/MU_2/MUB, C1H/C2H, MAPFAC_MX/MY.
Stage masses mu_1 + f*(MU_2 - MU_1) and the per-stage advective tendencies (mass-coupled centred
differences of the real field) are real-structured operands; the gate is the scalar UPDATE chain.
E200: every operand and result finite before any comparison.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import pristine_scalar_rk as wrf
from gpuwrf.runtime import operational_mode as op

RST = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/V33/p0227/run/wrfrst_d02_2026-03-01_00:00:00")
DT = 18.0
SPECIES = {"qv": ("QVAPOR", "RQVBLTEN"), "qc": ("QCLOUD", "RQCBLTEN"), "qi": ("QICE", "RQIBLTEN"),
           "Ni": ("QNICE", "RQNIBLTEN")}
# WRF routing of the PBL tendency into the RK sc_tend: 0 = moist_tend via update_phy_ten -> add_a2a
# (module_physics_addtendc.F:302-316), 1 = pbl_driver's direct scalar_tend(P_QNI) = RQNIBLTEN
# (module_pbl_driver.F:1866-1874).  Taken from the WRF source, not from the port.
ROUTE = {"qv": 0, "qc": 0, "qi": 0, "Ni": 1}
pytestmark = pytest.mark.skipif(not RST.exists(), reason="P0227 original restart not on this host")


DUST = 1e-36


def _band_ok(got, ref, q):
    """q = the magnitude of the largest addend of the final update (|q_old| or the |dt*tend/mass| terms)."""
    bound = 8 * float(np.finfo(np.float32).eps) * np.maximum(np.abs(q).astype(np.float64), np.abs(ref)) + DUST
    return bool(np.all(np.abs(got.astype(np.float64) - ref) <= bound))


def _band_report(got, ref, q):
    d = np.abs(got.astype(np.float64) - ref)
    bound = 8 * float(np.finfo(np.float32).eps) * np.maximum(np.abs(q).astype(np.float64), np.abs(ref)) + DUST
    i = np.unravel_index(np.argmax(d / bound), d.shape)
    return dict(worst=tuple(int(x) for x in i), err=float(d[i]), bound=float(bound[i]), ref=float(ref[i]), got=float(got[i]))


def _finite(*arrays):
    for a in arrays:
        if not np.all(np.isfinite(np.asarray(a, np.float64))):
            raise FloatingPointError("non-finite operand/result")


@dataclass
class _S:  # minimal State stand-in for the scalar helpers (getattr + replace)
    mu_total: object
    qv: object = None
    qc: object = None
    qi: object = None
    Ni: object = None
    qv_bdy: object = None
    qc_bdy: object = None
    qi_bdy: object = None
    Ni_bdy: object = None
    extra: dict = field(default_factory=dict)

    def replace(self, **kw):
        return replace(self, **kw)


@pytest.fixture(scope="module")
def operands():
    from netCDF4 import Dataset
    with Dataset(RST) as ds:
        ds.set_auto_mask(False)
        g = lambda n: np.asarray(ds.variables[n][0], np.float32)
        d = {n: g(v) for n, (v, _) in SPECIES.items()}
        d.update({f"R{n}": g(t) for n, (_, t) in SPECIES.items()})
        for n in ("MU_1", "MU_2", "MUB", "C1H", "C2H", "MAPFAC_MX", "MAPFAC_MY", "RTHRATEN"):
            d[n] = g(n)
        d["dx"] = float(ds.DX)
    _finite(*[v for v in d.values() if isinstance(v, np.ndarray)])
    nz, ny, nx = d["qv"].shape
    d.update(nz=nz, ny=ny, nx=nx)
    d["lib"] = wrf.build(Path("<USER_HOME>/wrf_gpu2_lanes/b-diff/BD99/pristine"))
    return d


def _stage_masses(o):
    dmu = o["MU_2"] - o["MU_1"]
    return [(o["MU_1"] + np.float32(f) * dmu).astype(np.float32) for f in (1 / 3, 0.5, 1.0)]


def _adv(o, q, mus):
    """Real-structured coupled advective tendencies per stage (centred x/y differences of mu*q)."""
    out = []
    for s, mu in enumerate(mus):
        mass = o["C1H"][:, None, None] * (mu + o["MUB"])[None] + o["C2H"][:, None, None]
        f = mass * q
        tx = np.zeros_like(f)
        tx[:, :, 1:-1] = -(f[:, :, 2:] - f[:, :, :-2]) * np.float32(5.0 / (2 * o["dx"]))
        ty = np.zeros_like(f)
        ty[:, 1:-1, :] = -(f[:, 2:, :] - f[:, :-2, :]) * np.float32((3.0 + s) / (2 * o["dx"]))
        out.append((tx + ty).astype(np.float32))
    return out


def _sc_other(o, q):
    """A real-structured frozen RK1 sc_tend (diffusion-like + relax-like ring term)."""
    mass = o["C1H"][:, None, None] * (o["MU_1"] + o["MUB"])[None] + o["C2H"][:, None, None]
    lap = np.zeros_like(q)
    lap[:, 1:-1, 1:-1] = (q[:, 2:, 1:-1] + q[:, :-2, 1:-1] + q[:, 1:-1, 2:] + q[:, 1:-1, :-2] - 4 * q[:, 1:-1, 1:-1])
    return (mass * lap * np.float32(1e-4)).astype(np.float32)


def _ring(shape):
    ny, nx = shape[-2:]
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    return np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))


def _ring_zero(width):
    """Test-local alternative mask: physics sc zero in rings < width (independent of the port helper)."""
    def mask(phys):
        return {n: jnp.where(jnp.asarray(_ring(t.shape) < width)[None], jnp.zeros_like(t), t) for n, t in phys.items()}
    return mask


def _port_chain(o, name, q, phys_raw, sc_other, adv, mus, *, nested, pd, owned, spec_zone=1, mask=None):
    ns = SimpleNamespace(precision="fp32", c1h=jnp.asarray(o["C1H"]), c2h=jnp.asarray(o["C2H"]))
    mub = o["MUB"]
    mut = o["MU_1"] + mub
    phys = op._coupled_physics_moist_tendf(jnp.asarray(mut), {name: phys_raw}, ns, real_glue=True)
    if mask is None:
        phys = op._physics_sc_outside_spec(phys, (name,) if owned else (), spec_zone, bounded=True)
    else:
        phys = mask(phys)
    # The state carries the species' boundary leaf (rk_update_scalar excludes the spec-zone advection on
    # every bounded species); ``owned`` is the separate spec_bdy_scalar overwrite (have_bcs_* / nested).
    origin = _S(mu_total=jnp.asarray(mut), **{name: jnp.asarray(q), f"{name}_bdy": 1})
    msfty = jnp.asarray(o["MAPFAC_MY"])
    others = {n: np.zeros_like(q) for n in op.NESTED_BOUNDARY_SCALAR_SPECIES}
    new = None
    for s, dt_rk in enumerate((DT / 3.0, 0.5 * DT, DT)):
        final = s == 2
        pd_species = (name,) if (pd and final) else ()
        scalar_origin = op._rk_update_scalar_pd(origin, pd_species, (phys, {name: sc_other}), dt_rk, ns) if pd_species else origin
        sc_stage = np.zeros_like(q) if pd_species else sc_other
        if nested:
            frozen = tuple(sc_stage if n == name else others[n] for n in op.NESTED_BOUNDARY_SCALAR_SPECIES)
            species, tends = op._nested_scalar_stage_tendencies((jnp.asarray(adv[s]),), (name,), frozen,
                                                                SimpleNamespace(spec_zone=spec_zone), msfty)
            species, tends = species[:1], tends[:1]
        else:
            species = (name,)
            tends = op._root_scalar_stage_tendencies((jnp.asarray(adv[s]),), species, origin, SimpleNamespace(spec_zone=spec_zone),
                                                     msfty, bounded=True, sc_tendencies={name: jnp.asarray(sc_stage)})
        tends = op._with_physics_sc(species, tends, phys, pd_species)
        stage = _S(mu_total=jnp.asarray(mus[s] + mub), **{name: jnp.asarray(q)})
        new = op._apply_moisture_large_step(stage, scalar_origin, q_tendencies=tends, dt_rk=dt_rk, metrics=ns, species=species)
    return np.asarray(getattr(new, name), np.float32)


def _wrf_chain(o, name, q, phys_raw, sc_other, adv, mus, *, nested, pd, owned, spec_zone=1):
    return wrf.chain(o["lib"], q=q, phys=phys_raw, sc_other=sc_other, adv=adv, mu1=o["MU_1"], mus=mus, mub=o["MUB"],
                     mut=o["MU_1"] + o["MUB"], c1=o["C1H"], c2=o["C2H"], msftx=o["MAPFAC_MX"], msfty=o["MAPFAC_MY"],
                     dt=DT, spec_zone=spec_zone, nested=nested, specified=not nested, pd=pd, owned=owned,
                     route=ROUTE[name])


def test_finite_gate_rejects_non_finite():
    with pytest.raises(FloatingPointError):
        _finite(np.array([1.0, np.nan], np.float32))
    with pytest.raises(FloatingPointError):
        _finite(np.zeros(3, np.float32), np.array([np.inf], np.float32))


# (nested, pd, owned, spec_zone).  Production (PROD/WN3 domain_tree, BoundaryConfig) is spec_bdy_width 5,
# spec_zone 1, relax_zone 4; spec_zone 5 binds the ring rule at the full boundary width as well.
CHAIN_CASES = [(True, 1, True, 1), (True, 0, True, 1), (False, 1, True, 1), (False, 1, False, 1),
               (True, 1, True, 5), (False, 1, False, 5)]


@pytest.mark.parametrize("nested,pd,owned,spec_zone", CHAIN_CASES)
@pytest.mark.parametrize("name", ["qv", "qc", "qi", "Ni"])
def test_scalar_chain_matches_pristine(operands, name, nested, pd, owned, spec_zone):
    o = operands
    q, raw = o[name], o[f"R{name}"]
    mus = _stage_masses(o)
    adv, sc = _adv(o, q, mus), _sc_other(o, q)
    kw = dict(nested=nested, pd=pd, owned=owned, spec_zone=spec_zone)
    got = _port_chain(o, name, q, raw, sc, adv, mus, **kw)
    ref = _wrf_chain(o, name, q, raw, sc, adv, mus, **kw)
    _finite(got, ref)
    # Per-cell REAL agreement, cancellation-aware: |got - ref| <= 8 eps32 * max(|q_old|, |ref|) + DUST.
    # The port associates (adv*msfty + sc) + phys and multiplies by 1/mass where WRF sums moist_tend
    # first and divides; DUST = 1e-36 covers XLA:CPU's flush of subnormal intermediates (E117) in
    # zero-valued cells (GPU .rn keeps them); no physical field is resolved below it.
    mass1 = o["C1H"][:, None, None] * (o["MU_1"] + o["MUB"])[None] + o["C2H"][:, None, None]
    mass3 = o["C1H"][:, None, None] * (mus[2] + o["MUB"])[None] + o["C2H"][:, None, None]
    addend = np.maximum.reduce([np.abs(mass1 * q), DT * np.abs(adv[2] * o["MAPFAC_MY"][None]), DT * np.abs(sc),
                                DT * np.abs(mass1 * raw)]).astype(np.float64) / mass3
    assert _band_ok(got, ref, addend), _band_report(got, ref, addend)
    active = raw != 0
    assert active.any()
    # Deletion sensitivity: without the physics sc_tend the active cells leave the band.
    dropped = _port_chain(o, name, q, np.zeros_like(raw), sc, adv, mus, **kw)
    _finite(dropped)
    assert not _band_ok(dropped[active], ref[active], addend[active])
    # Ring-rule sensitivity: every other candidate width (none, add_a2a's one ring, the spec zone) that
    # differs from WRF's on active cells must leave the band there.
    width = max(1 if ROUTE[name] == 0 else 0, spec_zone if owned else 0)
    ring = np.broadcast_to(_ring(q.shape)[None], q.shape)
    for alt_width in sorted({0, 1, spec_zone} - {width}):
        alt = _port_chain(o, name, q, raw, sc, adv, mus, **kw, mask=_ring_zero(alt_width))
        _finite(alt)
        cells = active & (ring >= min(width, alt_width)) & (ring < max(width, alt_width))
        assert cells.any(), (name, alt_width)
        assert not _band_ok(alt[cells], ref[cells], addend[cells]), (name, width, alt_width)


@pytest.mark.parametrize("name", ["qv", "qc", "Ni"])
def test_pd_preupdate_matches_pristine(operands, name):
    o = operands
    ns = SimpleNamespace(precision="fp32", c1h=jnp.asarray(o["C1H"]), c2h=jnp.asarray(o["C2H"]))
    q = o[name]
    mut = o["MU_1"] + o["MUB"]
    phys = op._coupled_physics_moist_tendf(jnp.asarray(mut), {name: o[f"R{name}"]}, ns, real_glue=True)
    phys = op._physics_sc_outside_spec(phys, (name,), 1, bounded=True)
    sc = _sc_other(o, q)
    got = op._rk_update_scalar_pd(_S(mu_total=jnp.asarray(mut), **{name: jnp.asarray(q)}), (name,),
                                  (phys, {name: sc}), DT, ns)
    total = np.asarray(phys[name], np.float32) + sc
    ref = wrf.pd_update(o["lib"], q=q, sc=total, mu1=o["MU_1"], mub=o["MUB"], c1=o["C1H"], c2=o["C2H"], dt=DT,
                        spec_zone=1, nested=True, specified=False)
    g = np.asarray(getattr(got, name), np.float32)
    _finite(g, ref)
    np.testing.assert_array_max_ulp(g, ref, maxulp=2)


def test_physics_sc_ring_rule():
    one = jnp.ones((3, 13, 14), jnp.float32)
    ring = _ring(one.shape)
    zero_rings = lambda t: sorted({int(r) for r in np.unique(ring[np.asarray(t)[0] == 0])})
    # add_a2a: moist species lose exactly ring 0 on a bounded domain, owned or not, whatever spec_zone is.
    assert zero_rings(op._physics_sc_outside_spec({"qc": one}, (), 5, bounded=True)["qc"]) == [0]
    assert zero_rings(op._physics_sc_outside_spec({"qv": one}, ("qv",), 1, bounded=True)["qv"]) == [0]
    # spec_bdy_scalar overwrite: an owned species loses rings < spec_zone.
    assert zero_rings(op._physics_sc_outside_spec({"qv": one}, ("qv",), 5, bounded=True)["qv"]) == [0, 1, 2, 3, 4]
    # pbl_driver scalar_tend(P_QNI): full mass tile unless the species is owned.
    assert op._physics_sc_outside_spec({"Ni": one}, (), 5, bounded=True)["Ni"] is one
    assert zero_rings(op._physics_sc_outside_spec({"Ni": one}, ("Ni",), 1, bounded=True)["Ni"]) == [0]
    # Periodic / unbounded: add_a2a adds everywhere.
    assert op._physics_sc_outside_spec({"qc": one}, (), 1, bounded=False)["qc"] is one


def test_held_radiation_decouple_matches_literal(operands):
    o = operands
    ns = SimpleNamespace(precision="fp32", c1h=jnp.asarray(o["C1H"]), c2h=jnp.asarray(o["C2H"]))
    mut, muts = o["MU_1"] + o["MUB"], o["MU_2"] + o["MUB"]
    import gpuwrf.kernels.dyn_carry_fp32 as carry_flags
    real_before = carry_flags.real_all_enabled
    carry_flags.real_all_enabled = lambda: True
    try:
        got = np.asarray(op._decouple_held_rthraten(jnp.asarray(o["RTHRATEN"]), jnp.asarray(mut), jnp.asarray(muts), ns))
    finally:
        carry_flags.real_all_enabled = real_before
    ref = wrf.decouple(o["lib"], r=o["RTHRATEN"], mut=mut, muts=muts, c1=o["C1H"], c2=o["C2H"])
    _finite(got, ref)
    np.testing.assert_array_max_ulp(got, ref, maxulp=2)
    assert float(np.abs(got - o["RTHRATEN"]).max()) > 0  # the round trip is not a no-op on changing mass


def test_pbl_scalar_rate_is_the_literal_rqniblten(operands):
    """RQNIBLTEN recovered from MYNN's post/entry Ni like the adapter's RQ?BLTEN: (post - entry) / dt in REAL."""
    o = operands
    entry = o["Ni"]
    post = (entry + np.float32(DT) * o["RNi"]).astype(np.float32)
    got = np.asarray(op._pbl_scalar_rate(jnp.asarray(post), jnp.asarray(entry), DT, jnp.float32))
    lit = ((post - entry) / np.float32(DT)).astype(np.float32)
    _finite(got, lit)
    assert got.dtype == np.float32
    # XLA may divide by the constant as a reciprocal multiply (1 ulp) and XLA:CPU flushes subnormal
    # differences (E117): band 2 eps |lit| + DUST instead of an ulp count.
    assert np.all(np.abs(got.astype(np.float64) - lit) <= 2 * float(np.finfo(np.float32).eps) * np.abs(lit) + DUST)
    bound = 4 * float(np.finfo(np.float32).eps) * np.maximum(np.abs(post), np.abs(entry)) / DT + DUST
    assert np.all(np.abs(got.astype(np.float64) - o["RNi"]) <= bound)
    assert np.any(o["RNi"] != 0)


def test_pd_family_species_selection():
    nml = SimpleNamespace(moist_adv_opt=1, scalar_adv_opt=0)
    assert op._pd_family_species(nml, ("qv", "qc", "Ni", "Nr")) == ("qv", "qc")
    nml = SimpleNamespace(moist_adv_opt=0, scalar_adv_opt=2)
    assert op._pd_family_species(nml, ("qv", "Ni")) == ("Ni",)
    nml = SimpleNamespace(moist_adv_opt=3, scalar_adv_opt=3)
    assert op._pd_family_species(nml, ("qv", "Ni")) == ()


def test_with_physics_sc_fails_closed_for_unadvected_species():
    t = (jnp.zeros(3),)
    with pytest.raises(NotImplementedError):
        op._with_physics_sc(("qv",), t, {"Ni": jnp.ones(3)}, ())
    assert op._with_physics_sc(("qv",), t, None, ()) is t


# ----------------------------------------------------------------------------------------------
# Whole-step dispatch on the real PROD step (eval_shape, CPU): hunks ran, wiring by object identity
# (phys_tend_dispatch_probe.py), once under this process's CPU-suite defaults and once in a fresh process
# with the v0.3 release defaults (MYNN cloudmix + QNI mixing, REAL glue) that the GPU arms run.
# ----------------------------------------------------------------------------------------------
import json
import os
import subprocess
import sys

import phys_tend_dispatch_probe as probe


@pytest.mark.parametrize("domain", ["d01", "d02"])
def test_step_dispatch_on_and_off(domain):
    probe.check(domain, probe.expected_raw_keys())


@pytest.mark.parametrize("domain", ["d01", "d02"])
def test_step_dispatch_release_defaults(domain):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GPUWRF_")}
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", GPUWRF_JAX_CACHE="0",
               PYTHONPATH=str(Path(__file__).resolve().parents[3] / "src"))
    cp = subprocess.run([sys.executable, str(Path(probe.__file__).resolve()), domain], env=env,
                        capture_output=True, text=True, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-4000:] + cp.stderr[-12000:]
    lines = [line for line in cp.stdout.splitlines() if line.startswith("PROBE_JSON ")]
    assert len(lines) == 1, cp.stdout[-4000:]
    res = json.loads(lines[0][len("PROBE_JSON "):])
    assert res["fast_defaults"] is None
    assert res["env"] == {"GPUWRF_MYNN_CLOUDMIX": "1", "GPUWRF_MYNN_QNI_MIXING": "1",
                          "GPUWRF_DYN_REAL_ALL": "1", "GPUWRF_ROOT_SCALAR_BDY_RK1": "1"}, res["env"]
    assert res["raw_keys"] == sorted(probe.RELEASE_RAW_KEYS) and res["spec_zone"] == 1
    if domain == "d01":
        assert res["owned"] == ["qv"]  # specified root, have_bcs_moist/scalar .false.: only QV relaxes
    else:
        assert {"qv", "qc", "qi", "Ni"} <= set(res["owned"])  # nested: every species with a boundary record
