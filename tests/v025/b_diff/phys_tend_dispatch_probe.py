"""Whole-step dispatch probe for GPUWRF_PHYS_TEND_RK_WRF on the real PROD step (eval_shape, CPU).

Hunks ran, and the PBL tendencies travel producer -> coupling -> ring mask -> every RK stage by object
identity.  Shared by the in-process CPU-suite test (legacy defaults, tests/conftest.py) and a fresh
release-defaults process (``python phys_tend_dispatch_probe.py <domain>``): the v0.3 release defaults are
captured at ``import gpuwrf``, so they cannot be switched inside one process.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "b_core")]
KEY = "GPUWRF_PHYS_TEND_RK_WRF"
RELEASE_RAW_KEYS = {"qv", "qc", "qi", "Ni"}  # MYNN cloudmix + QNI mixing (bl_mynn_cloudmix/mixscalars 1)


def trace_step(domain, flag, *, force_stacked=False):
    import jax
    import jax.numpy as jnp
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.domain_tree import DomainTree
    from prod_inputs import prod_domains

    hierarchy, bundles, _, _, _, carries = prod_domains()
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    nml, carry = tree.domains[domain].namelist, carries[domain]
    rec = {"pd": [], "couple": [], "decouple": [], "origins": [], "kf_in": [], "mynn": [], "ni_rate": [],
           "mask": [], "with_phys": [], "theta_barrier": [], "stacked": []}
    patched = []

    def spy(name, sink, capture):
        real = getattr(op, name)

        def wrapped(*args, **kwargs):
            out = real(*args, **kwargs)
            rec[sink].append(capture(args, kwargs, out))
            return out
        patched.append((name, real))
        setattr(op, name, wrapped)

    spy("_rk_update_scalar_pd", "pd", lambda a, k, out: (out, a[2]))
    spy("_coupled_physics_moist_tendf", "couple", lambda a, k, out: (a[1], out))
    spy("_decouple_held_rthraten", "decouple", lambda a, k, out: out)
    spy("_scalar_transport_coupled_tendencies", "origins", lambda a, k, out: k.get("step_origin"))
    spy("_scalar_transport_coupled_tendencies", "stacked", lambda a, k, out: (tuple(k.get("species", ())), k.get("step_origin")))
    spy("_moisture_coupled_tendencies", "origins", lambda a, k, out: k.get("step_origin"))
    spy("_kf_cadence_step", "kf_in", lambda a, k, out: a[0])
    spy("mynn_adapter_with_source_leaves", "mynn", lambda a, k, out: (a[0], out))
    spy("_pbl_scalar_rate", "ni_rate", lambda a, k, out: (a, out))
    spy("_physics_sc_outside_spec", "mask", lambda a, k, out: (a[0], out, k.get("bounded"), tuple(a[1]), int(a[2])))
    spy("_with_physics_sc", "with_phys", lambda a, k, out: a[2])
    real_barrier = jax.lax.optimization_barrier

    def barrier_spy(x):
        if sys._getframe(1).f_code.co_name == "_augment_large_step_tendencies":
            rec["theta_barrier"].append(x)
        return real_barrier(x)
    jax.lax.optimization_barrier = barrier_spy
    if force_stacked:
        # WN3 d03 (44x94x112 = 463,232 cells) takes the stacked moist+number limited stage
        # (<= _NESTED_SCALAR_BATCH_MAX_FIELD_CELLS); raise the threshold so PROD d02 exercises it.
        patched.append(("_NESTED_SCALAR_BATCH_MAX_FIELD_CELLS", op._NESTED_SCALAR_BATCH_MAX_FIELD_CELLS))
        op._NESTED_SCALAR_BATCH_MAX_FIELD_CELLS = 10 ** 12
    before = os.environ.get(KEY)
    os.environ[KEY] = flag
    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    try:
        jax.clear_caches()
        out = jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1, cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        jax.clear_caches()
        jax.lax.optimization_barrier = real_barrier
        for name, real in reversed(patched):
            setattr(op, name, real)
        if before is None:
            os.environ.pop(KEY, None)
        else:
            os.environ[KEY] = before
    assert jax.devices()[0].platform == "cpu"
    return nml, rec, [str(leaf) for leaf in jax.tree.leaves(out)]


def expected_raw_keys():
    from gpuwrf.kernels.phys_mynn_cloudmix import cloudmix_enabled
    from gpuwrf.kernels.phys_mynn_ni import ni_mixing_enabled

    keys = {"qv"}
    if cloudmix_enabled():
        keys |= {"qc", "qi"}
        if ni_mixing_enabled():
            keys.add("Ni")
    return keys


def check(domain, raw_keys):
    nml, on, avals_on = trace_step(domain, "1")
    assert int(nml.rad_rk_tendf) != 0 and int(nml.moist_adv_opt) in (1, 2)
    assert len(on["couple"]) == 1 and len(on["mynn"]) == 1 and len(on["decouple"]) == 1
    # Producer route: every WRF RQ?BLTEN the MYNN driver writes reaches the coupling, each from its own
    # MYNN source leaf (object identity); the Ni rate is formed from MYNN's own post/entry Ni.
    raw, coupled = on["couple"][0]
    mynn_in, mynn_out = on["mynn"][0]
    assert set(raw) == set(raw_keys) == set(coupled), sorted(raw)
    assert raw["qv"] is mynn_out.rqvblten
    if "qc" in raw_keys:
        assert raw["qc"] is mynn_out.rqcblten and raw["qi"] is mynn_out.rqiblten
    if "Ni" in raw_keys:
        assert len(on["ni_rate"]) == 1
        (ni_post, ni_entry, ni_dt, _), ni_rate = on["ni_rate"][0]
        assert ni_post is mynn_out.state.Ni and ni_entry is mynn_in.Ni and ni_post is not ni_entry
        assert raw["Ni"] is ni_rate and ni_dt == float(nml.dt_s)
    else:
        assert on["ni_rate"] == []
    # Consumer route: the coupled dict feeds the ring mask at every RK stage, the mask output feeds that
    # stage's scalar update, and the final stage's pre-update reads the same masked physics.
    assert len(on["mask"]) == int(nml.rk_order) == len(on["with_phys"])
    assert all(m[0] is coupled and m[2] is True for m in on["mask"])
    assert all(w is m[1] for w, m in zip(on["with_phys"], on["mask"]))
    owned, spec_zone = on["mask"][0][3], on["mask"][0][4]
    assert spec_zone == int(nml.boundary_config.spec_zone)
    # E41 seam: one output-only barrier on the assembled dry theta tendency per RK stage, ON only.
    assert len(on["theta_barrier"]) == int(nml.rk_order), len(on["theta_barrier"])
    assert len(on["pd"]) == 1  # the final stage only (E114: the path ran)
    assert on["pd"][0][1][0] is on["mask"][-1][1]
    # The limited final-stage transport (and only it) reads the pre-updated field; a wrapper may call
    # the moisture builder inside the stacked builder, so the final stage can record more than one origin.
    pd_at = [i for i, o in enumerate(on["origins"]) if o is on["pd"][0][0]]
    assert pd_at and pd_at == list(range(pd_at[0], len(on["origins"]))) and pd_at[0] > 0, pd_at
    if on["kf_in"]:  # d01 (cu_physics 1): KF reads the PBL-entry prognostics like WRF (RB256)
        assert on["kf_in"][0].theta is mynn_in.theta
        assert on["kf_in"][0].qv is mynn_in.qv
    nml, off, avals_off = trace_step(domain, "0")
    assert off["pd"] == [] and off["couple"] == [] and off["decouple"] == []
    assert off["ni_rate"] == [] and off["mask"] == [] and off["with_phys"] == [None] * int(nml.rk_order)
    assert off["theta_barrier"] == []
    if off["kf_in"]:
        assert off["kf_in"][0].theta is not off["mynn"][0][0].theta  # released: KF sees post-MYNN theta
    assert avals_on == avals_off  # no carry-signature change
    assert (domain == "d01") == bool(on["kf_in"])
    stacked = None
    if domain != "d01":
        nml, st, _ = trace_step(domain, "1", force_stacked=True)
        finals = [(sp, o) for sp, o in st["stacked"] if "Ni" in sp and any(n in sp for n in ("qv", "qc"))]
        assert len(finals) == 1, [sp for sp, _ in st["stacked"]]  # the stacked limited stage ran exactly once
        assert len(st["pd"]) == 1 and finals[0][1] is st["pd"][0][0]  # ... and reads the pd pre-update
        stacked = list(finals[0][0])
    return dict(domain=domain, raw_keys=sorted(raw), owned=list(owned), spec_zone=spec_zone, stacked_species=stacked,
                kf=bool(on["kf_in"]), stages=len(on["mask"]),
                fast_defaults=os.environ.get("GPUWRF_FAST_DEFAULTS"),
                env={k: os.environ.get(k) for k in ("GPUWRF_MYNN_CLOUDMIX", "GPUWRF_MYNN_QNI_MIXING",
                                                    "GPUWRF_DYN_REAL_ALL", "GPUWRF_ROOT_SCALAR_BDY_RK1")})


def cpu_pallas_interpret():
    """Release defaults route init kernels (RRTMG, Noah, ...) through Triton Pallas, which XLA:CPU runs only
    in interpret mode.  Must run before gpuwrf is imported; shapes/dtypes are unchanged and this probe
    checks wiring, never values."""
    import functools

    from jax.experimental import pallas as pl

    real = pl.pallas_call

    @functools.wraps(real)
    def interpreted(*args, **kwargs):
        kwargs["interpret"] = True
        return real(*args, **kwargs)
    pl.pallas_call = interpreted


if __name__ == "__main__":
    assert "gpuwrf" not in sys.modules
    cpu_pallas_interpret()
    print("PROBE_JSON " + json.dumps(check(sys.argv[1], RELEASE_RAW_KEYS)), flush=True)
