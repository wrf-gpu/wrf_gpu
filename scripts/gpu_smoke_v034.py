#!/usr/bin/env python3
"""v0.3.4 GPU smoke for the wave-O1 experimental schemes (lane o1-gpusmoke).

For every candidate scheme that is wired on THIS source tree (catalog IMPLEMENTED +
operational-suite admission), run the Swiss 42x42 d01 case (examples/switzerland_d01)
with the release defaults and that ONE option changed: cold compile + N root steps,
every carry leaf finite, then the E41 static gate on the scheme's private JAX cache
(scripts/gpu_smoke_e41.py; the opus-a19 gate tools are gone), cache deleted afterwards.

Orchestrator (inside the GPU lock, after ALISIOS posts GPU FREI)::

    source <USER_HOME>/wrf_gpu2_lanes/driver_bridge_59591/bridge.env
    scripts/with_gpu_lock.sh --label o1-gpusmoke -- \\
        python scripts/gpu_smoke_v034.py --out <USER_HOME>/wrf_gpu2_lanes/o1-gpusmoke/run --deadline 18:57

CPU dry run (plumbing check, 1 step, no GPU)::

    JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=0 PYTHONPATH=src taskset -c 9 \\
        python scripts/gpu_smoke_v034.py --out DIR --cpu-dry-run --steps 1 --schemes bl9_camuw

Each scheme runs in its own child process (private caches, per-scheme timeout <= 360 s,
never past the deadline); the GPU-lock fd 9 is passed through (E36/E215).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "switzerland_d01"
E41 = ROOT / "scripts" / "gpu_smoke_e41.py"

#: risk order (largest/most novel device programs first); key -> namelist edits
SCHEMES: dict[str, dict] = {
    "ra3_cam": {"lane": "o1-camrad", "catalog": [("ra_lw_physics", 3), ("ra_sw_physics", 3)],
                "replace": {"ra_lw_physics": 3, "ra_sw_physics": 3}, "drop": ("radiation_diagnostics",)},
    "mp18_nssl": {"lane": "o1-nssl", "catalog": [("mp_physics", 18)], "replace": {"mp_physics": 18}},
    "mp40_morraero": {"lane": "o1-morraero", "catalog": [("mp_physics", 40)], "replace": {"mp_physics": 40}},
    "cu5_grell3d": {"lane": "o1-grell", "catalog": [("cu_physics", 5)],
                    "replace": {"cu_physics": 5, "cumulus_cadence_steps": 1, "cudt_minutes": 0.0}},
    "cu93_grelldev": {"lane": "o1-grell", "catalog": [("cu_physics", 93)],
                      "replace": {"cu_physics": 93, "cumulus_cadence_steps": 1, "cudt_minutes": 0.0}},
    "cu4_sas": {"lane": "o1-sas", "catalog": [("cu_physics", 4)], "replace": {"cu_physics": 4}},
    "km3_smag3d": {"lane": "o1-smag3d", "catalog": [("km_opt", 3)], "replace": {"diff_opt": 2, "km_opt": 3}},
    "bl9_camuw": {"lane": "o1-camuw", "catalog": [("bl_pbl_physics", 9)], "replace": {"bl_pbl_physics": 9}},
    "sf3_ruc": {"lane": "o1-ruc", "catalog": [("sf_surface_physics", 3)], "replace": {}, "ruc": True},
}
#: optional control arm (release defaults, no change), only if time remains
CONTROL = "release_default"


def _utc_now():
    return _dt.datetime.now(_dt.timezone.utc)


def _deadline(hhmm: str):
    h, m = (int(x) for x in hhmm.split(":"))
    now = _utc_now()
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


# --------------------------------------------------------------------------- child


def _cpu_patches():
    """CPU dry run only: State on CPU, pure-JAX tridiagonal (E105), Pallas interpret."""

    import jax
    import jax.numpy as jnp
    from jax import lax
    from gpuwrf.contracts import state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]

    def thomas(dl, d, du, b):
        dl_t, d_t, du_t = (jnp.moveaxis(x, -1, 0)[..., None] for x in (dl, d, du))
        b_t = jnp.moveaxis(b, -2, 0)

        def fwd(carry, row):
            cp_prev, dp_prev = carry
            a_i, b_i, c_i, r_i = row
            den = b_i - a_i * cp_prev
            cp = c_i / den
            dp = (r_i - a_i * dp_prev) / den
            return (cp, dp), (cp, dp)

        _, (cp, dp) = lax.scan(fwd, (jnp.zeros_like(d_t[0]), jnp.zeros_like(b_t[0])), (dl_t, d_t, du_t, b_t))

        def bwd(x_next, row):
            x_i = row[1] - row[0] * x_next
            return x_i, x_i

        _, xs = lax.scan(bwd, jnp.zeros_like(b_t[0]), (cp, dp), reverse=True)
        return jnp.moveaxis(xs, 0, -2)

    jax.lax.linalg.tridiagonal_solve = thomas
    if os.environ.get("GPUWRF_FAST_DEFAULTS", "1") != "0":
        from jax.experimental import pallas as pl

        orig = pl.pallas_call
        pl.pallas_call = lambda *a, **k: orig(*a, **{**k, "interpret": True})


def _ruc_bundles(nml, state):
    """RUC static/land from the Swiss wrfinput (smoke-only Noah 4-layer -> RUC 9-level soil)."""

    import netCDF4
    import numpy as np
    from gpuwrf.coupling.ruc_surface_hook import build_ruc_bundles

    zs_ruc = (0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00)
    ds = netCDF4.Dataset(CASE / "wrfinput_d01")
    f = {n: np.asarray(ds[n][0]) for n in ("IVGTYP", "ISLTYP", "XLAND", "SEAICE", "TMN", "SHDMIN", "SHDMAX",
                                          "ALBBCK", "VEGFRA", "TSK", "TSLB", "SMOIS", "SNOW", "SNOWH",
                                          "CANWAT", "LAI", "SNOALB")}
    zs_noah = np.asarray(ds["ZS"][0], np.float64)
    mminlu, iswater, isice = str(ds.MMINLU), int(ds.ISWATER), int(ds.ISICE)
    ds.close()

    def interp(noah, deep, top=None):
        zz = np.concatenate([[0.0] if top is not None else [], zs_noah, [3.0]])
        out = np.empty((len(zs_ruc),) + noah.shape[1:])
        for j in range(noah.shape[1]):
            for i in range(noah.shape[2]):
                yy = np.concatenate([[top[j, i]] if top is not None else [], noah[:, j, i], [deep[j, i]]])
                out[:, j, i] = np.interp(zs_ruc, zz, yy)
        return out

    f["TSLB"] = interp(f["TSLB"], f["TMN"], top=f["TSK"])
    f["SMOIS"] = interp(f["SMOIS"], f["SMOIS"][-1])
    f["SNOWC"] = (f["SNOW"] > 0.0).astype(np.float32)
    p1 = np.asarray(state.p_total[0], np.float64)
    return build_ruc_bundles(f, mminlu=mminlu, iswater=iswater, isice=isice, dt=float(nml.dt_s),
                             zs=zs_ruc, p1=p1, frpcpn=True)


def _rebuild_carry(orig, state, nml, drop=()):
    """_initial_carry_for_run for the edited namelist + the _load_domains post-seeds."""

    from gpuwrf.kernels.dyn_carry_fp32 import real_carry
    from gpuwrf.runtime.operational_mode import _commit_to_operational_device, _initial_carry_for_run

    new = _initial_carry_for_run(state, nml)
    keep = {}
    for name in new.__dataclass_fields__:
        if name in drop:
            keep[name] = None
        elif getattr(new, name) is None and getattr(orig, name) is not None:
            keep[name] = getattr(orig, name)
    new = new.replace(**keep)
    return _commit_to_operational_device(real_carry(new))


def child(args) -> int:
    t_start = time.perf_counter()
    rec = {"scheme": args.child, "started_utc": _utc_now().isoformat(), "status": "started"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def dump():
        (out / "child.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")

    dump()
    import gpuwrf  # noqa: F401  (fast-default resolution first)
    import jax

    rec["jax"] = jax.__version__
    rec["platform"] = jax.devices()[0].platform
    rec["device"] = str(jax.devices()[0])
    if args.cpu_dry_run:
        assert rec["platform"] == "cpu"
        jax.config.update("jax_cpu_enable_async_dispatch", False)
        _cpu_patches()
    else:
        assert rec["platform"] == "gpu", f"GPU smoke must run on the GPU, got {rec['platform']}"
    import numpy as np
    from dataclasses import replace

    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime.domain_tree import DomainTree, run_operational_domain_tree
    from gpuwrf.runtime.operational_mode import _resolve_operational_suite

    config = NestedPipelineConfig(CASE, out / "stream", out / "proof", hours=1, max_dom=1)
    t0 = time.perf_counter()
    hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
    rec["load_s"] = time.perf_counter() - t0
    bundle = bundles["d01"]
    nml = bundle.namelist
    carry = carries["d01"]
    if args.child != CONTROL:
        spec = SCHEMES[args.child]
        edits = dict(spec["replace"])
        drop = tuple(spec.get("drop", ()))
        if spec.get("ruc"):
            static, land = _ruc_bundles(nml, carry.state)
            edits.update(use_noahmp=False, sf_surface_physics=3, noahmp_static=None, noahmp_energy_params=None,
                         noahmp_rad_params=None, ruc_static=static, ruc_land=land)
            drop += ("noahmp_land", "noahmp_rad", "land_history", "energy_accumulators", "history_diagnostics")
        nml = replace(nml, **edits)
        _resolve_operational_suite(nml)
        carry = _rebuild_carry(carry, bundle.state, nml, drop)
        bundles["d01"] = replace(bundle, namelist=nml)
        rec["namelist_edits"] = {k: v for k, v in edits.items() if isinstance(v, (int, float, bool, str))}
    rec["status"] = "loaded"
    dump()
    if args.trace_only:
        from gpuwrf.runtime.operational_mode import _physics_boundary_step

        rec["trace_eqns"] = {}
        for rad in (False, True):
            t0 = time.perf_counter()
            jaxpr = jax.make_jaxpr(lambda cc, si, r=rad: _physics_boundary_step(cc, nml, si, run_radiation=r))(
                carry, jax.numpy.asarray(1, jax.numpy.int32))
            rec["trace_eqns"]["radiation" if rad else "ordinary"] = [len(jaxpr.jaxpr.eqns), time.perf_counter() - t0]
        rec["status"] = "traced"
        rec["finite"] = None
        dump()
        return 0
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    carries = {"d01": carry}
    own = {"d01": 0}
    rec["step_s"] = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        res = run_operational_domain_tree(tree, root_steps=1, carries=carries, initial_own_steps=own,
                                          block_between=False, root_sync_cadence=0)
        jax.block_until_ready(res.carries)
        carries, own = res.carries, dict(res.own_steps)
        rec["step_s"].append(time.perf_counter() - t0)
        rec["status"] = f"stepped {own['d01']}"
        dump()
    c = carries["d01"]
    bad = []
    n_float = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(c):
        a = np.asarray(leaf)
        if a.dtype.kind == "f":
            n_float += 1
            if not np.all(np.isfinite(a)):
                bad.append(jax.tree_util.keystr(path))
    st = c.state
    rec.update(
        finite=not bad, nonfinite_leaves=bad, n_float_leaves=n_float,
        t_skin_range=[float(np.min(np.asarray(st.t_skin))), float(np.max(np.asarray(st.t_skin)))],
        theta_range=[float(np.min(np.asarray(st.theta))), float(np.max(np.asarray(st.theta)))],
        own_steps=own["d01"], process_s=time.perf_counter() - t_start,
        compile_plus_first_step_s=rec["step_s"][0] if rec["step_s"] else None,
        status="ok" if not bad else "nonfinite",
    )
    dump()
    return 0 if not bad else 1


# --------------------------------------------------------------------------- orchestrator


def _lane_merged(lane: str) -> bool:
    """Merge evidence on this tree: the lane branch tip is an ancestor of HEAD, or a
    'merge <lane>' commit exists (the catalog alone cannot tell an old scaffold from the lane)."""

    branch = f"lane/{lane}"
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", branch, "HEAD"],
                      capture_output=True).returncode == 0:
        return True
    log = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-400", "HEAD"], capture_output=True,
                         text=True).stdout
    return any(line.split(" ", 1)[-1].startswith(f"merge {lane}") for line in log.splitlines())


def _available(names, assume=False):
    """Schemes wired on this tree (catalog IMPLEMENTED for every listed key/code + lane merged)."""

    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    code = ("import json,sys; from gpuwrf.io.scheme_catalog import classify_scheme, SupportStatus; "
            "spec=json.loads(sys.argv[1]); print(json.dumps({k: all(classify_scheme(a, b).status is "
            "SupportStatus.IMPLEMENTED for a, b in v) for k, v in spec.items()}))")
    spec = {n: SCHEMES[n]["catalog"] for n in names}
    res = subprocess.run([sys.executable, "-c", code, json.dumps(spec)], env=env, capture_output=True, text=True,
                         timeout=300, cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-2000:])
    catalog = json.loads(res.stdout.strip().splitlines()[-1])
    return {n: {"catalog_implemented": catalog[n], "lane_merged": True if assume else _lane_merged(SCHEMES[n]["lane"]),
                "run": bool(catalog[n] and (assume or _lane_merged(SCHEMES[n]["lane"])))} for n in names}


def _du_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def orchestrate(args) -> int:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    deadline = _deadline(args.deadline)
    names = [s for s in (args.schemes.split(",") if args.schemes else SCHEMES) if s != CONTROL]
    avail = _available(names, assume=args.assume_available)
    summary = {"started_utc": _utc_now().isoformat(), "deadline_utc": deadline.isoformat(),
               "source_rev": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(),
               "git_status_dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                                                       capture_output=True, text=True).stdout.strip()),
               "cpu_dry_run": bool(args.cpu_dry_run), "steps": args.steps, "case": str(CASE),
               "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "1 (default)"),
               "available": avail, "schemes": {}}

    def save():
        (root / "GPU_SMOKE.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")

    queue = [n for n in names if avail[n]["run"]] + ([CONTROL] if args.control else [])
    for n in names:
        if not avail[n]["run"]:
            summary["schemes"][n] = {"status": "not_run: " + ("lane not merged" if avail[n]["catalog_implemented"]
                                                            else "not IMPLEMENTED on this tree"), **avail[n]}
    save()
    gates = []
    pass_fds = ()
    try:
        os.fstat(9)
        pass_fds = (9,)
    except OSError:
        pass
    for name in queue:
        remaining = (deadline - _utc_now()).total_seconds()
        if remaining < args.min_slot_s:
            summary["schemes"][name] = {"status": f"skipped_no_time ({remaining:.0f} s left)"}
            save()
            continue
        while _du_bytes(root) > args.max_bytes and gates:
            gates[0][0].wait()
            gates.pop(0)
        sdir = root / name
        cache = sdir / "cache"
        env = dict(os.environ)
        env.update(GPUWRF_JAX_CACHE_DIR=str(cache / "jax"), JAX_COMPILATION_CACHE_DIR=str(cache / "jax"),
                   GPUWRF_XLA_AUTOTUNE_CACHE_DIR=str(cache / "autotune"))
        if args.cpu_dry_run:
            env.update(GPUWRF_JAX_CACHE="0")
        else:
            env.pop("GPUWRF_JAX_CACHE", None)
            shared = root / "autotune_shared"
            shared.mkdir(parents=True, exist_ok=True)
            (cache / "jax").mkdir(parents=True, exist_ok=True)
            link = cache / "jax" / "xla_gpu_per_fusion_autotune_cache_dir"
            if not link.exists():
                link.symlink_to(shared, target_is_directory=True)  # share tunes across schemes (C34/E35)
        timeout = max(30, min(args.per_scheme_s, remaining - args.reserve_s))
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", name, "--out", str(sdir),
               "--steps", str(args.steps)] + (["--cpu-dry-run"] if args.cpu_dry_run else []) + (
                   ["--trace-only"] if args.trace_only else [])
        t0 = time.perf_counter()
        rc, status = None, None
        log = (sdir / "child.log")
        sdir.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, pass_fds=pass_fds,
                                    start_new_session=True, cwd=ROOT)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 15)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, 9)
                    proc.wait()
                status = f"timeout ({timeout:.0f} s)"
        child_rec = {}
        if (sdir / "child.json").exists():
            child_rec = json.loads((sdir / "child.json").read_text())
        entry = {"wall_s": time.perf_counter() - t0, "rc": rc, "timeout_s": timeout,
                 "status": status or child_rec.get("status", f"rc={rc}"), **{
                     k: child_rec.get(k) for k in ("platform", "device", "load_s", "step_s", "finite", "trace_eqns",
                                                   "nonfinite_leaves", "compile_plus_first_step_s",
                                                   "namelist_edits", "t_skin_range", "theta_range", "own_steps")}}
        if rc not in (0, 1) and status is None:
            entry["log_tail"] = log.read_text()[-1500:]
        summary["schemes"][name] = entry
        save()
        if (cache / "jax").exists() and not args.no_gate:
            gate_out = sdir / "e41"
            genv = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
            genv.pop("GPUWRF_GPU_LOCK_HELD", None)
            gcmd = ["taskset", "-c", args.gate_cpus, "nice", "-n", "19", "timeout", "900", sys.executable, str(E41),
                    str(cache / "jax"), str(gate_out)]
            gproc = subprocess.Popen(gcmd, env=genv, stdout=(sdir / "e41.log").open("w"), stderr=subprocess.STDOUT,
                                     cwd=ROOT)
            gates.append((gproc, name, cache, gate_out))
        elif not args.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)
        # harvest finished gates
        still = []
        for g in gates:
            if g[0].poll() is None:
                still.append(g)
            else:
                _finish_gate(g, summary, args)
        gates = still
        save()
    for g in gates:
        try:
            g[0].wait(timeout=max(10, (deadline - _utc_now()).total_seconds() + 900))
        except subprocess.TimeoutExpired:
            g[0].kill()
        _finish_gate(g, summary, args)
    summary["finished_utc"] = _utc_now().isoformat()
    save()
    print(json.dumps({k: {kk: v.get(kk) for kk in ("status", "finite", "compile_plus_first_step_s", "e41")}
                      for k, v in summary["schemes"].items()}, indent=1, default=str))
    return 0


def _finish_gate(g, summary, args):
    proc, name, cache, gate_out = g
    res = {"rc": proc.returncode}
    if (gate_out / "e41.json").exists():
        d = json.loads((gate_out / "e41.json").read_text())
        res.update({k: d.get(k) for k in ("hard_stack_gate", "max_stack_B", "conservative_flags", "n_entries", "unscanned")})
    summary["schemes"][name]["e41"] = res
    summary["schemes"][name]["cache_bytes"] = _du_bytes(cache)
    if not args.keep_cache:
        shutil.rmtree(cache, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", default=None)
    ap.add_argument("--schemes", default=None, help="comma list (default: all, risk order)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--deadline", default="18:57", help="UTC HH:MM; nothing starts or runs past it")
    ap.add_argument("--per-scheme-s", type=float, default=360.0)
    ap.add_argument("--reserve-s", type=float, default=45.0)
    ap.add_argument("--min-slot-s", type=float, default=150.0)
    ap.add_argument("--max-bytes", type=float, default=4.0e9, help="ROOT scratch cap before waiting on gates")
    ap.add_argument("--gate-cpus", default="9")
    ap.add_argument("--control", action="store_true", help="append the release-default control arm")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--assume-available", action="store_true", help="dry runs on lane trees: skip merge check")
    ap.add_argument("--trace-only", action="store_true", help="child: make_jaxpr of the step instead of running it")
    args = ap.parse_args(argv)
    if args.child:
        return child(args)
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
