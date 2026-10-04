"""#16 PD transport: GPUWRF_DYN_PD_SPECIES_LOOP runs the native advect_scalar_pd kernels with lanes over
(k, y, x) and the species loop inside the program (species-independent face geometry/velocity/Courant
terms once per cell).  Same expressions -> bitwise vs the per-(species, cell) kernels (CPU interpret;
GPU evidence tests/v025/b_core/pd_species_loop_evidence.json)."""
import jax
import numpy as np


def _pallas_names(fn, *args):
    names = []

    def walk(jaxpr):
        for e in jaxpr.eqns:
            if e.primitive.name == "pallas_call":
                names.append(str(e.params["name"]))
            for v in e.params.values():
                for sub in (v if isinstance(v, (list, tuple)) else [v]):
                    if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                        walk(sub.jaxpr)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return names


def test_pd_species_loop_is_bitwise(monkeypatch):
    assert jax.devices()[0].platform == "cpu"
    import test_root_scalar_stacking as t
    monkeypatch.setenv("GPUWRF_DYN_ADVECTION_FP32", "1")
    monkeypatch.setenv("GPUWRF_DYN_PD_FP32", "1")
    outs, names = {}, {}
    for flag in ("0", "1"):
        monkeypatch.setenv("GPUWRF_DYN_PD_SPECIES_LOOP", flag)
        jax.clear_caches()
        nml, origin, current = t._case()
        fn = lambda cur, org: t.op._scalar_transport_coupled_tendencies(  # noqa: E731
            cur, nml, rk_step=3, step_origin=org, species=t.SPECIES, advection_opt=int(nml.moist_adv_opt),
            transport_velocities=t.op._stage_transport_velocities(cur, nml))
        names[flag] = _pallas_names(fn, current, origin)
        outs[flag] = [np.asarray(a) for a in jax.jit(fn)(current, origin)]
    jax.clear_caches()
    assert "b_carry_pd_scale_fp32" in names["0"] and "b_core_pd_scale_sl_fp32" not in names["0"]
    assert {"b_core_pd_scale_sl_fp32", "b_core_pd_tend_sl_fp32"} <= set(names["1"])  # species loop dispatched
    for name, a, b in zip(t.SPECIES, outs["0"], outs["1"]):
        assert np.isfinite(b).all(), name
        np.testing.assert_array_equal(b, a, err_msg=name)
