"""Measured native ordinary root-step replay through the real PROD runtime.

This checks executable finiteness and actual work/guard counters. The separate
pristine-Fortran stage receipts remain the component fidelity authority.
"""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import statistics
import time


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_receipt(carries, jax, np):
    """Fingerprint host arrays before either arm commits the common carry."""
    domains = {}
    for domain, carry in sorted(carries.items()):
        leaves, _ = jax.tree_util.tree_flatten_with_path(carry)
        domains[domain] = [
            dict(path=jax.tree_util.keystr(path), shape=list(np.asarray(value).shape),
                 dtype=str(np.asarray(value).dtype),
                 sha256=hashlib.sha256(np.asarray(value).tobytes()).hexdigest())
            for path, value in leaves
        ]
    return domains


def export_fused_hlo(runtime, output):
    """Export the exact cached ordinary executable after the timed region."""
    programs = inspect.getclosurevars(runtime.fused_cascade).nonlocals["cache"]
    receipts = []
    for domain, program in sorted(programs.items()):
        if program is None:
            continue
        calls = inspect.getclosurevars(program).nonlocals["cached_calls"]
        for variant, (cached, cheap_key) in enumerate(calls.values()):
            if hasattr(cached, "runtime_executable"):
                executable = cached.runtime_executable()
            else:
                executable = inspect.getclosurevars(cached).nonlocals["le"]
            for number, module in enumerate(executable.hlo_modules()):
                text = module.to_string()
                path = output / f"{domain}_{variant}_{number}.optimized.hlo"
                path.write_text(text)
                receipts.append(dict(path=str(path), cheap_key=cheap_key,
                                     sha256=hashlib.sha256(text.encode()).hexdigest(),
                                     bytes=len(text.encode())))
    assert receipts, "no exact cached fused executable was available"
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--shared-inputs", type=Path)
    parser.add_argument("--create-shared-inputs", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--export-hlo", action="store_true")
    args = parser.parse_args()
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise SystemExit(125)
    if args.create_shared_inputs and (
        args.shared_inputs is None or not args.setup_only
        or os.environ.get("GPUWRF_DYN_ADVECTION_FP32", "0") != "0"
    ):
        parser.error("common seed creation requires --shared-inputs, --setup-only and advection OFF")
    if args.shared_inputs and args.shared_inputs.exists() and args.create_shared_inputs:
        parser.error("common inputs already exist; use them without --create-shared-inputs")
    if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise SystemExit("requires b-core GPU lock")
    if os.environ.get("GPUWRF_DYN_FP32") != "1":
        raise SystemExit("requires native acoustic flag")
    root = Path(__file__).resolve().parents[3]
    identity = json.loads((root / "measurement_source.json").read_text())
    for name, digest in identity["source_sha256"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"frozen source changed: {name}")
    os.umask(0o077)
    args.out.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The outer launcher waits for this exact application PID before releasing
    # its GPU lock, including on profiler errors or timeout cancellation.
    stat = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")",1)[1].split()
    (args.out/"app_pid.json").write_text(json.dumps(dict(pid=os.getpid(),starttime=stat[19]))+"\n")
    import jax
    import numpy as np
    from gpuwrf.diagnostics.census import GUARDS, EVENTS, WORK, resolved_namelist
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime.domain_tree import (
        DomainTree, NESTED_AOT_STATUS, _prepare_operational_domain_tree_runtime,
        run_operational_domain_tree,
    )
    from gpuwrf.runtime.operational_mode import _commit_to_operational_device

    if jax.devices()[0].platform != "gpu":
        raise SystemExit("requires GPU backend")
    os.environ["GPUWRF_CENSUS"] = "1" if args.census else "0"
    config = NestedPipelineConfig(args.input_dir, args.out/"unused_output",
                                  args.out/"proof", hours=3, max_dom=2)
    started = time.perf_counter()
    hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01", "d02"))
    print("PROD loaded", round(time.perf_counter()-started, 3), flush=True)
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    runtime = _prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)

    def replay(inputs, steps):
        result = run_operational_domain_tree(
            tree, root_steps=1, carries=inputs, initial_own_steps=steps,
            prepared_runtime=runtime, block_between=False, root_sync_cadence=0,
        )
        jax.block_until_ready(result.carries)
        return result

    namelists = {
        domain: resolved_namelist(bundles[domain].namelist, output_cadence_steps=0,
                                 output_set=())
        for domain in ("d01", "d02")
    }
    for domain, bundle in bundles.items():
        namelists[domain].update(top_lid=bool(bundle.namelist.top_lid),
            h_sca_adv_order=int(bundle.namelist.h_sca_adv_order),
            moist_adv_opt=int(bundle.namelist.moist_adv_opt),
            scalar_adv_opt=int(bundle.namelist.scalar_adv_opt))
    shared = None
    if args.shared_inputs and not args.create_shared_inputs:
        with args.shared_inputs.open("rb") as stream:
            pack = pickle.load(stream)
        assert pack["source"] == identity, "common seed came from a different source"
        assert pack["input_dir"] == str(args.input_dir.resolve())
        assert pack["census"] == args.census
        assert pack["run_start"] == run_start.isoformat()
        assert pack["namelists"] == namelists
        assert input_receipt(pack["carries"], jax, np) == pack["input_receipt"]
        inputs = _commit_to_operational_device(pack.pop("carries"))
        steps = pack["steps"]
        shared = dict(path=str(args.shared_inputs), sha256=file_sha256(args.shared_inputs), **pack)
        del pack
    else:
        seeded = replay(carries, {"d01": 0, "d02": 0})
        print("baseline first root step ready", round(time.perf_counter()-started, 3), flush=True)
        inputs, steps = seeded.carries, seeded.own_steps
        if args.create_shared_inputs:
            host_inputs = jax.device_get(inputs)
            pack = dict(schema="b_core.shared_prod_seed.v1", source=identity,
                        input_dir=str(args.input_dir.resolve()), census=args.census,
                        run_start=run_start.isoformat(), namelists=namelists,
                        steps=steps, input_receipt=input_receipt(host_inputs, jax, np),
                        carries=host_inputs)
            args.shared_inputs.parent.mkdir(parents=True, exist_ok=True)
            with args.shared_inputs.open("xb") as stream:
                pickle.dump(pack, stream, protocol=5)
            # Both baseline and flagged arms use the same fresh device placement.
            inputs = _commit_to_operational_device(pack.pop("carries"))
            shared = dict(path=str(args.shared_inputs), sha256=file_sha256(args.shared_inputs), **pack)
            del host_inputs, pack
        del seeded
    del carries
    jax.block_until_ready(inputs)
    print("ordinary shared inputs ready", steps, flush=True)
    first = replay(inputs, steps)
    record = dict(schema="b_core.real_prod.ordinary_rootstep.v1", source=identity,
        scope="one d01 + three d02 own-steps; genuine nested boundary forcing and physics",
        input_dir=str(args.input_dir), run_start=run_start.isoformat(),
        root_dt_s=dts["d01"], device=jax.devices()[0].device_kind,
        backend=jax.devices()[0].platform, shared_inputs=shared,
        initial_own_steps=steps, initialization_and_compile_s=time.perf_counter()-started,
        repeats=args.repeats, census=args.census, setup_only=args.setup_only,
        advection_fp32=os.environ.get("GPUWRF_DYN_ADVECTION_FP32", "0"),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        cache_dirs={name:os.environ.get(name) for name in (
            "GPUWRF_JAX_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR",
            "GPUWRF_XLA_AUTOTUNE_CACHE_DIR", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR", "TMPDIR")},
        domains={})
    for domain in ("d01", "d02"):
        before, after = inputs[domain], first.carries[domain]
        bad = [i for i, value in enumerate(jax.tree.leaves(after))
               if np.issubdtype(np.asarray(value).dtype, np.floating)
               and not np.all(np.isfinite(np.asarray(value)))]
        work = np.asarray(after.census.work)-np.asarray(before.census.work) if args.census else None
        guards = np.asarray(after.census.guards)-np.asarray(before.census.guards) if args.census else None
        expected_steps = 1 if domain == "d01" else 3
        data = dict(shape=list(after.state.theta.shape), nonfinite_leaf_indices=bad,
            work_delta={name:int(value) for name,value in zip(WORK,work,strict=True)} if args.census else None,
            guard_delta={name:{event:int(value) for event,value in zip(EVENTS,row,strict=True)}
                         for name,row in zip(GUARDS,guards,strict=True)} if args.census else None,
            namelist=namelists[domain])
        record["domains"][domain] = data
        sound_steps = int(bundles[domain].namelist.acoustic_substeps)
        stage_trips = [1, max(1,sound_steps//2), sound_steps]
        data.update(stage_trip_counts=stage_trips,
                    expected_acoustic_trips=sum(stage_trips)*expected_steps)
    (args.out/"setup.json").write_text(json.dumps(record,indent=2,default=str)+"\n")
    for domain, data in record["domains"].items():
        assert not data["nonfinite_leaf_indices"], (domain,data)
        if args.census:
            assert data["work_delta"]["steps"] == (1 if domain=="d01" else 3), data
            assert data["work_delta"]["acoustic_trips"] == data["expected_acoustic_trips"], data
    del first
    if args.setup_only:
        (args.out/"completed.json").write_text(json.dumps(record,indent=2,default=str)+"\n")
        print("cold setup PASS", round(time.perf_counter()-started, 3), flush=True)
        return
    for _ in range(2):
        replay(inputs, steps)
    cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    assert cudart.cudaProfilerStart() == 0
    samples = []
    try:
        for iteration in range(args.repeats):
            with jax.profiler.TraceAnnotation(f"BCORE:native_rootstep:{iteration}"):
                begin = time.perf_counter()
                replay(inputs, steps)
                samples.append(time.perf_counter()-begin)
    finally:
        cudart.cudaProfilerStop()
    record.update(warm_root_step_s=samples, median_root_step_s=statistics.median(samples),
                  ordinary_replay_extrapolation_s_per_fc_h=statistics.median(samples)*3600/dts["d01"],
                  aot_status=NESTED_AOT_STATUS)
    if args.export_hlo:
        record["optimized_hlo"] = export_fused_hlo(runtime, args.out)
    for name, digest in identity["source_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    (args.out/"completed.json").write_text(json.dumps(record,indent=2,default=str)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k not in ("source","domains","warm_root_step_s")}),flush=True)


if __name__ == "__main__":
    main()
