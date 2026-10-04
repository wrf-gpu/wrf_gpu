"""B29 finish gate with observed WRF live mass evolution.

The recorded WRF history pair supplies the live mass and physical endpoint.
Coupled finish work is constructed coherently from that endpoint and the saved
prep fields. This is an operator fixture spanning two observed history times,
not a claim to replay an internal acoustic substep. The live MU/MUTS and work
outputs feed both finish callers. A source-deletion
mutant must fail the SAME pre-registered U/V bounds in every domain/RK context.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types


def deletion_mutant(probe):
    source_path = Path(probe.finishmod.__file__)
    source = source_path.read_text()
    refresh = """    if _NATIVE_RK_FP32:
        # solve_em calls calc_mu_uv_1 on the LIVE MUTS before finish.
        prep = prep.replace(muus=_u_face_average_2d(muts), muvs=_v_face_average_2d(muts))
"""
    assert source.count(refresh) == 1, "B29 deletion must remove only the live refresh"
    module = types.ModuleType("b29_deleted_live_refresh")
    sys.modules[module.__name__] = module
    exec(compile(source.replace(refresh, ""), str(source_path), "exec"), module.__dict__)
    assert module._NATIVE_RK_FP32
    return module, hashlib.sha256(refresh.encode()).hexdigest()


def load_endpoints(probe, domain):
    """Recover the original WRF pair before casting the stored interpolation.

    The fixture manifest defines advanced=current+(current-previous)*dt/interval.
    Invert that lossless-fp64 construction before REAL conversion; do not scale
    or invent a new mass signal. Prep uses previous plus the original one-step
    interpolation, and the live endpoint is the recorded current WRF state.
    """
    np = probe.np
    meta = json.loads((probe.DATA/(domain+".json")).read_text())
    fraction = meta["pair"]["dt_s"]/meta["pair"]["snapshot_interval_seconds"]
    _, _, base, metrics = probe.load(domain, probe.jnp.float32)
    with np.load(probe.DATA/(domain+".npz")) as data:
        previous, entry, endpoint = {}, {}, {}
        for name in probe.STATE_FIELDS:
            current = data["ref_"+name]
            prior = current - (data["state_"+name]-current)/fraction
            previous[name] = probe.jnp.asarray(prior, probe.jnp.float32)
            entry[name] = probe.jnp.asarray(prior+(current-prior)*fraction, probe.jnp.float32)
            endpoint[name] = np.asarray(current, np.float32)
    def state(values):
        return probe.State.tree_unflatten(None, [values.get(n) for n in probe.State.__slots__])
    return (state(entry), state(previous), base, metrics), types.SimpleNamespace(**endpoint), meta


def live_work(probe, prepared, endpoint):
    """Mass/velocity/theta work describes the SAME observed physical endpoint."""
    np = probe.np
    def a(name):
        return np.asarray(getattr(prepared, name), np.float32)
    def u_face(mass):
        return np.concatenate((mass[:, :1], np.float32(.5)*(mass[:, :-1]+mass[:, 1:]), mass[:, -1:]), axis=1)
    def v_face(mass):
        return np.concatenate((mass[:1], np.float32(.5)*(mass[:-1]+mass[1:]), mass[-1:]), axis=0)
    def layer(c1, c2, mass):
        return a(c1)[:, None, None]*mass[None]+a(c2)[:, None, None]
    mu_work = endpoint.mu_perturbation-a("mu_save")
    muts = a("mut")+mu_work  # unchanged advance_mu_t caller invariant
    mu = a("mu_save")+mu_work
    mass_u = layer("c1h", "c2h", u_face(muts))
    mass_v = layer("c1h", "c2h", v_face(muts))
    mass_h = layer("c1h", "c2h", muts)
    mass_f = layer("c1f", "c2f", muts)
    work = types.SimpleNamespace(
        u=(mass_u*endpoint.u-layer("c1h", "c2h", a("muu"))*a("u_save"))/a("msfuy")[None],
        v=(mass_v*endpoint.v-layer("c1h", "c2h", a("muv"))*a("v_save"))/a("msfvx")[None],
        w=(mass_f*endpoint.w-layer("c1f", "c2f", a("mut"))*a("w_save"))/a("msfty")[None],
        theta_coupled_work=mass_h*(endpoint.theta-np.float32(300))-layer("c1h", "c2h", a("mut"))*a("t_save"),
        ph=endpoint.ph_perturbation-a("ph_save"), mu=mu,
        p=endpoint.p_perturbation, muts=muts, ww=a("ww_save"))
    coherent = np.array_equal(muts, a("mut")+mu_work) and np.array_equal(mu, a("mu_save")+mu_work)
    assert coherent
    delta = muts-a("muts")
    assert np.any(delta != 0) and np.any(mu_work != a("mu_work"))
    assert np.all(muts > 0) and all(np.all(np.isfinite(value)) for value in vars(work).values())
    evolution = dict(changed_cells=int(np.count_nonzero(delta)),
                     delta_rms_Pa=float(np.sqrt(np.mean(delta**2))),
                     delta_max_Pa=float(np.max(np.abs(delta))),
                     mu_muts_coherent=coherent,
                     physical_mu_endpoint=probe.compare(endpoint.mu_perturbation, mu),
                     live_total_endpoint=probe.compare(endpoint.mu_total, muts),
                     source="recorded WRF previous/current history pair; literal coupled finish work",
                     limitation="observed history-endpoint operator fixture, not internal acoustic savepoints")
    return types.SimpleNamespace(**{n: probe.jnp.asarray(v) for n, v in vars(work).items()}), mu_work, evolution


def check_case(probe, finish_path, mutant, inputs, endpoint, stage, domain):
    np, jax = probe.np, probe.jax
    prepared = probe.prep(inputs, stage)
    jax.block_until_ready(prepared)
    work, mu_work, evolution = live_work(probe, prepared, endpoint)
    names = ("u", "v", "w", "theta_coupled_work", "ph", "mu", "p", "muts", "ww")
    values = tuple(getattr(work, name) for name in names)
    def compiled(function):
        return jax.jit(lambda p, leaves: function(p, types.SimpleNamespace(**dict(zip(names, leaves)))))
    got = compiled(probe.finishmod.small_step_finish_wrf)(prepared, values)
    bad = compiled(mutant.small_step_finish_wrf)(prepared, values)
    jax.block_until_ready((got, bad))
    reference = probe.Oracle(finish_path, inputs[0].theta.shape)
    probe.oracle_inputs(reference, inputs)
    for name in ("muu", "muv", "muus", "muvs", "mut", "mu_save"):
        reference.set(name, getattr(prepared, name))
    reference.set("muts", work.muts)
    for name, value in (("u_2", work.u), ("v_2", work.v), ("w_2", work.w),
                        ("t_2", work.theta_coupled_work), ("ph_2", work.ph),
                        ("mu_2", mu_work), ("u_save", prepared.u_save),
                        ("v_save", prepared.v_save), ("w_save", prepared.w_save),
                        ("t_save", prepared.t_save), ("ph_save", prepared.ph_save),
                        ("ww", work.ww), ("ww1", prepared.ww_save),
                        ("h_diabatic", np.zeros_like(np.asarray(prepared.theta_work)))):
        reference.set(name, value)
    reference.run("calc_mu_uv_1", stage)
    reference.run("small_step_finish", stage)
    fields = ("u_2", "v_2", "w_2", "t_2", "ph_2", "mu_2")
    refs = [reference.field(name, np.asarray(getattr(got, attr)).shape)
            for name, attr in zip(fields, probe.FINISH_OUTPUT)]
    refs[3] += np.float32(300)
    checks = {name: probe.compare(ref, getattr(got, attr))
              for name, attr, ref in zip(fields, probe.FINISH_OUTPUT, refs)}
    negative = {name: probe.compare(ref, getattr(bad, attr))
                for name, attr, ref in zip(fields[:2], probe.FINISH_OUTPUT[:2], refs[:2])}
    endpoint_checks = {name: probe.compare(getattr(endpoint, attr), getattr(got, attr))
                       for name, attr in zip(fields, probe.FINISH_OUTPUT)}
    finite = all(np.all(np.isfinite(np.asarray(getattr(got, attr))))
                 for attr in probe.FINISH_OUTPUT)
    # Require each velocity to discriminate the deletion, rather than merely
    # one field somewhere across the matrix. Bounds come from compare unchanged.
    sensitive = all(not check["passed"] for check in negative.values())
    return dict(platform=jax.devices()[0].platform, domain=domain, rk_step=stage,
                evolution=evolution, checks=checks, endpoint_checks=endpoint_checks, deletion_checks=negative,
                all_finite=bool(finite), deletion_sensitive=sensitive,
                passed=finite and sensitive and all(c["passed"] for c in checks.values())
                and all(c["passed"] for c in endpoint_checks.values())
                and evolution["physical_mu_endpoint"]["passed"] and evolution["live_total_endpoint"]["passed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.argv = ["probe_rk", "--mode", "verify", "--arm", "native"]
    import probe_rk as probe

    assert probe.jax.devices()[0].platform == "gpu", probe.jax.devices()
    assert probe.prepmod._NATIVE_RK_FP32 and probe.finishmod._NATIVE_RK_FP32
    finish_path = probe.build(args.out/"pristine_finish")
    mutant, deletion_sha = deletion_mutant(probe)
    cases = []
    for domain in ("d01", "d02"):
        probe.CASE_DT, probe.CASE_DX = (54., 9000.) if domain == "d01" else (18., 3000.)
        inputs, endpoint, meta = load_endpoints(probe, domain)
        for stage in (1, 2, 3):
            case = check_case(probe, finish_path, mutant, inputs, endpoint, stage, domain)
            case["history_pair"] = meta["pair"]
            cases.append(case)
            (args.out/f"{domain}_rk{stage}.json").write_text(json.dumps(case, indent=2)+"\n")
            print(domain, stage, "PASS" if case["passed"] else "FAIL", case["evolution"], flush=True)
        probe.jax.clear_caches()
    result = dict(platform=probe.jax.devices()[0].platform,
                  source_commit=probe.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=probe.ROOT, text=True).strip(),
                  fixture_sha256={d: hashlib.sha256((probe.DATA/(d+".npz")).read_bytes()).hexdigest() for d in ("d01", "d02")},
                  bounds="unchanged tests/v025/b_core/bench_acoustic.py:compare",
                  deletion_sha256=deletion_sha, changed_model_files=False,
                  cache={k: probe.os.environ.get(k) for k in ("JAX_COMPILATION_CACHE_DIR", "GPUWRF_XLA_AUTOTUNE_CACHE_DIR")},
                  oracle_checks=sum(len(c["checks"]) for c in cases),
                  mutant_velocity_checks=sum(len(c["deletion_checks"]) for c in cases),
                  cases=cases, passed=all(c["passed"] for c in cases))
    (args.out/"receipt.json").write_text(json.dumps(result, indent=2)+"\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
