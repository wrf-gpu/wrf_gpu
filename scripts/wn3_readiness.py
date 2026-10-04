"""Run the real 3-domain pipeline for two root steps, recording final carry."""
from __future__ import annotations

import argparse
from dataclasses import replace
import faulthandler
import hashlib
import json
from pathlib import Path
import signal
import time

import jax
import numpy as np
from gpuwrf.integration import nested_pipeline as pipeline


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--disable-sst", action="store_true", help="Disable ONLY aux4 after loading, for baseline regression")
    parser.add_argument("--emit-history", action="store_true", help="Write initial and final regression frames through the real writer")
    parser.add_argument("--dump-hlo", action="store_true", help="Record the full three-domain step graph for OFF identity")
    parser.add_argument("--lower-only", action="store_true", help="OFF proof: compare full step graphs and initial writer bytes, without a second CPU forecast")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stack_log = (args.output / "python_stack.log").open("w")
    faulthandler.register(signal.SIGUSR1, file=stack_log, all_threads=True)
    assert jax.devices()[0].platform == args.platform
    report = {"platform": jax.devices()[0].platform, "device": str(jax.devices()[0]), "input": str(args.input)}
    from gpuwrf.coupling import physics_couplers as pc
    column_inputs = pc._rrtmg_column_inputs
    report["radiation_consumer_flags"] = []

    def observed_inputs(state, grid=None, **kw):
        report["radiation_consumer_flags"].append({"shape": state.t_skin.shape,
            "topo_shading": int(kw.get("topo_shading", 0)), "slope_rad": int(kw.get("slope_rad", 0))})
        return column_inputs(state, grid, **kw)

    pc._rrtmg_column_inputs = observed_inputs
    if args.platform == "cpu":
        # The production zero allocator intentionally requires a GPU. For this
        # CPU readiness oracle change ONLY its target device; initial fields,
        # loaders, physics, dynamics and nested stepping remain production code.
        from gpuwrf.contracts import state as state_contract
        state_contract._gpu_device = lambda: jax.devices()[0]
        report["cpu_adapter"] = "State/Tendencies zero-allocation target only"
    original = pipeline.run_operational_domain_tree

    def capture(*a, **kw):
        result = original(*a, **kw)
        jax.block_until_ready(result.carries)
        report["own_steps"] = result.own_steps
        report["carry"] = {}
        for domain, carry in result.carries.items():
            leaves = []
            for path, leaf in jax.tree_util.tree_flatten_with_path(carry)[0]:
                arr = np.asarray(leaf)
                leaves.append({"path": jax.tree_util.keystr(path), "shape": arr.shape,
                               "dtype": str(arr.dtype), "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                               "finite": bool(np.isfinite(arr).all())})
            report["carry"][domain] = leaves
        (args.output / "carry.json").write_text(json.dumps(report, indent=2) + "\n")
        return result

    t0 = time.perf_counter()
    config = pipeline.NestedPipelineConfig(input_dir=args.input, output_dir=args.output / "history",
        proof_dir=args.output / "proof", scratch_dir=args.output / "scratch", hours=24,
        max_dom=3, emit_initial_history=True)
    hierarchy, bundles, metadata, start, dts, carries = pipeline._load_domains(config, ("d01", "d02", "d03"))
    if args.disable_sst:
        bundles = {name: replace(bundle, namelist=replace(bundle.namelist, lower_boundary=None))
                   if hasattr(bundle.namelist, "lower_boundary") else bundle
                   for name, bundle in bundles.items()}
        report["sst_update"] = 0
    else:
        report["sst_update"] = 1
    print("domains loaded", time.perf_counter() - t0, flush=True)
    report["terrain_radiation"] = {}
    for name, bundle in bundles.items():
        namelist = bundle.namelist
        assert int(namelist.topo_shading) == int(namelist.slope_rad) == 1, name
        assert namelist.radiation_static is not None, name
        fields = []
        for path, leaf in jax.tree_util.tree_flatten_with_path(namelist.radiation_static)[0]:
            arr = np.asarray(leaf)
            fields.append({"path": jax.tree_util.keystr(path), "shape": arr.shape,
                "dtype": str(arr.dtype), "finite": bool(np.isfinite(arr).all()),
                "min": float(arr.min()) if arr.size else None, "max": float(arr.max()) if arr.size else None})
        report["terrain_radiation"][name] = {"topo_shading": int(namelist.topo_shading),
            "slope_rad": int(namelist.slope_rad), "static": fields}
    (args.output / "terrain_radiation.json").write_text(json.dumps(report["terrain_radiation"], indent=2) + "\n")
    (args.output / "loaded.json").write_text(json.dumps({"dt": dts, "source": metadata}, indent=2) + "\n")
    tree = pipeline.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    if args.dump_hlo:
        from gpuwrf.runtime import operational_mode as op
        import jax.numpy as jnp
        report["step_hlo_sha256"] = {}
        for name, bundle in bundles.items():
            text = op._advance_chunk_fori.lower(carries[name], bundle.namelist,
                jnp.asarray(1, jnp.int32), op.build_clock_base(bundle.namelist),
                n_steps=1, cadence=int(bundle.namelist.radiation_cadence_steps)).compiler_ir("hlo").as_hlo_text()
            (args.output / f"{name}.hlo").write_text(text)
            report["step_hlo_sha256"][name] = hashlib.sha256(text.encode()).hexdigest()
        (args.output / "hlo_identity.json").write_text(json.dumps(report["step_hlo_sha256"], indent=2) + "\n")
    writer = None
    if args.emit_history:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        writer = pipeline._PerDomainWrfoutWriter(output_dir=config.output_dir,
            input_dir=args.input, run_start=start, bundles=bundles,
            output_cadence_steps={name: 1 for name in bundles}, dt_by_domain=dts)
        for name in bundles:
            writer(name, 0, carries[name])
    if args.lower_only:
        report["wall_seconds"] = time.perf_counter() - t0
        report["history_sha256"] = {p.name: digest(p) for p in sorted(config.output_dir.glob("wrfout*"))}
        report["scope"] = "full real-case 3-domain step HLO and initial full-history byte regression; no stepping claim"
        report["pass"] = True
        (args.output / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"identity_complete": True, "wall_seconds": report["wall_seconds"]}))
        return
    result = capture(tree, root_steps=2, carries=carries, block_between=False, root_sync_cadence=1)
    if writer is not None:
        for name in bundles:
            writer(name, result.own_steps[name], result.carries[name])
    report["wall_seconds"] = time.perf_counter() - t0
    report["pipeline"] = {"metadata": metadata, "start": start.isoformat(), "dt": dts}
    report["history_sha256"] = {p.name: digest(p) for p in sorted((args.output / "history").glob("wrfout*"))}
    report["pass"] = (report.get("own_steps") == {"d01": 2, "d02": 6, "d03": 18}
                       and all(l["finite"] for leaves in report["carry"].values() for l in leaves))
    (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("platform", "own_steps", "wall_seconds", "pass")}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
