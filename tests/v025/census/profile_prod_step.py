"""Paired real-PROD d01 ordinary root-step replay, for node-traced counter attribution.

Run from a clean detached snapshot under census's GPU lock. Warm all variants
before cudaProfilerStart. This measures instrumentation, not forecast fidelity.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def field_path(value, path):
    """Name registered dataclass/slot children instead of opaque flat indices."""
    result = []
    for key in path:
        if hasattr(key, "key") and type(key).__name__ == "FlattenedIndexKey":
            names = tuple(getattr(value, "__dataclass_fields__", getattr(value, "__slots__", ())))
            name = names[key.key]
            value = getattr(value, name)
            result.append(name)
        elif hasattr(key, "name"):
            value = getattr(value, key.name)
            result.append(key.name)
        else:
            index = key.idx if hasattr(key, "idx") else key.key
            value = value[index]
            result.append(str(index))
    return ".".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--bisect", action="store_true", help="compare telemetry groups without capturing")
    parser.add_argument("--guard-bisect", action="store_true", help="split guard sites and try mask barriers")
    parser.add_argument("--barrier-only", action="store_true", help="compare only baseline and mask barriers")
    parser.add_argument("--candidate-barrier", action="store_true", help="barrier candidates before guard predicates")
    parser.add_argument("--tier-p", action="store_true", help="gate registered transparency bounds instead of exact bits")
    parser.add_argument("--drift-steps", type=int, default=10, help="ordinary trajectory checkpoint for Tier-P")
    args = parser.parse_args()
    if args.candidate_barrier:
        args.barrier_only = True
    if args.guard_bisect or args.barrier_only:
        args.bisect = True
    if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise SystemExit("requires census GPU lock")
    root = Path(__file__).resolve().parents[3]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise SystemExit("measurement snapshot must be clean")
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip():
        raise SystemExit("measurement snapshot must have detached HEAD")
    args.out.mkdir(parents=True, exist_ok=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    if jax.devices()[0].platform != "gpu":
        raise SystemExit("profiling requires the asserted GPU backend")
    from gpuwrf.diagnostics.census import initial_census, resolved_namelist, WORK
    from gpuwrf.diagnostics.census_transparency import compare_array
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime.operational_mode import _physics_boundary_step, build_clock_base
    import gpuwrf.runtime.operational_mode as operational
    import gpuwrf.diagnostics.census as census_module

    os.environ["GPUWRF_CENSUS"] = "0"
    names = ("d01",)
    config = NestedPipelineConfig(args.input_dir, args.out / "unused_output",
                                  args.out / "proof", hours=3, max_dom=1)
    hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, names)

    namelist = bundles["d01"].namelist
    clock = build_clock_base(namelist)
    bounds_path = Path(__file__).with_name("transparency_bounds.json")
    bounds = json.loads(bounds_path.read_text()) if args.tier_p else None
    if args.tier_p and args.drift_steps not in bounds["root_checkpoints_own_steps"]:
        raise SystemExit("drift checkpoint must match registered own-step count")

    @jax.jit
    def ordinary(carry, nml, step, clock_base):
        return _physics_boundary_step(carry, nml, step, run_radiation=False, clock_base=clock_base)

    step_values = {index: jnp.asarray(index + 1, jnp.int32) for index in range(max(2, args.drift_steps + 1))}
    jax.block_until_ready(step_values)

    def replay(mode, inputs, steps):
        result = ordinary(inputs["d01"], namelist,
                          step_values[steps["d01"]], clock)
        jax.block_until_ready(result)
        return {"d01": result}

    # The exact ordinary step; no outer forecast loop or inactive RRTMG branch.
    seeded = replay("off", carries, {"d01": 0})
    steps = {"d01": 1}
    variants = {"off": seeded,
                "on": {name: c.replace(census=jax.device_put(initial_census(), jax.devices()[0]))
                       for name, c in seeded.items()}}
    if args.barrier_only:
        variants["barrier_values" if args.candidate_barrier else "barrier_masks"] = variants.pop("on")
    elif args.guard_bisect:
        variants.update(no_dynamics=variants["on"], no_boundary=variants["on"], barrier_masks=variants["on"])
    elif args.bisect:
        variants.update(no_guards=variants["on"], no_rk=variants["on"])
    results = {"scope": "real PROD d01 ordinary root-step replay; d02 overhead extrapolation is enabler-only",
               "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "input_dir": str(args.input_dir), "run_start": run_start.isoformat(),
               "device": jax.devices()[0].device_kind, "repeats": args.repeats,
               "root_dt_s": dts["d01"], "initial_own_steps": steps,
               "domains": {}, "initial_metadata": meta,
               "static_radiation": False, "outer_forecast_loop": False}
    outputs = {}
    results["compile_and_first_s"] = {}
    results["backend_platform"] = jax.devices()[0].platform
    results["comparisons"] = {}
    original_dynamics = census_module.count_dynamics_guards
    original_boundary = census_module.count_boundary_guards
    original_rk = operational._rk_scan_step
    original_guard = census_module.count_guard

    def no_guard(census, *unused, **kwargs):
        return census

    def barrier_guard(census, name, candidate, *, lower=None, upper=None, rejected=None):
        if census is None:
            return None
        if args.candidate_barrier:
            candidate = jax.lax.optimization_barrier(candidate)
        finite = jnp.isfinite(candidate)
        outside = jnp.zeros_like(finite)
        if lower is not None:
            outside = outside | (candidate < lower)
        if upper is not None:
            outside = outside | (candidate > upper)
        outside = outside & finite
        repair = (~finite | outside) if rejected is None else rejected
        masks = jax.lax.optimization_barrier((~finite, outside, repair))
        counts = jnp.stack(tuple(jnp.sum(mask, dtype=jnp.uint64) for mask in masks))
        return census._replace(guards=census.guards.at[census_module.GUARDS.index(name)].add(counts))

    def uninstrumented_rk(carry, *positional, **kwargs):
        saved = carry.census
        result = original_rk(carry.replace(census=None), *positional, **kwargs)
        return result.replace(census=saved)

    for mode, inputs in variants.items():
        census_module.count_dynamics_guards = no_guard if mode in {"no_guards", "no_dynamics"} else original_dynamics
        census_module.count_boundary_guards = no_guard if mode in {"no_guards", "no_boundary"} else original_boundary
        census_module.count_guard = barrier_guard if mode.startswith("barrier_") else original_guard
        operational._rk_scan_step = uninstrumented_rk if mode == "no_rk" else original_rk
        # Each diagnostic variant retraces; globals must never reuse a stale JIT.
        if args.bisect:
            ordinary.clear_cache()
        if mode.startswith("barrier_"):
            fault = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf, -0.01, 0.1, 0.0, 0.01], jnp.float64)
            fault_seed = initial_census()
            guard_index = census_module.GUARDS.index("dynamics.qv")
            fault_seed = fault_seed._replace(guards=fault_seed.guards.at[guard_index].set(jnp.uint64(2**32)))
            fault_result = jax.jit(lambda c, v: barrier_guard(
                c, "dynamics.qv", v, lower=0.0, upper=0.05))(fault_seed, fault)
            measured = tuple(map(int, fault_result.guards[guard_index]))
            if measured != tuple(2**32 + count for count in (3, 2, 5)):
                raise SystemExit("barrier runtime fault/uint64 gate failed")
            results["barrier_fault_counts_above_2pow32"] = list(measured)
        started = time.perf_counter()
        outputs[mode] = replay(mode, inputs, steps)
        results["compile_and_first_s"][mode] = time.perf_counter() - started
        print(f"{mode}: {results['compile_and_first_s'][mode]:.3f}s", flush=True)
        if mode != "off":
            differences = []
            nonfinite = []
            scores = {}
            left_paths = jax.tree_util.tree_flatten_with_path(outputs["off"]["d01"])[0]
            right_paths = jax.tree_util.tree_flatten_with_path(outputs[mode]["d01"].replace(census=None))[0]
            for index, ((path, left), (_, right)) in enumerate(zip(left_paths, right_paths, strict=True)):
                left, right = np.asarray(left), np.asarray(right)
                if bounds is not None:
                    name = field_path(outputs["off"]["d01"], path)
                    scores[name] = compare_array(name, left, right, bounds)
                if np.issubdtype(right.dtype, np.floating) and not np.all(np.isfinite(right)):
                    nonfinite.append(field_path(outputs["off"]["d01"], path))
                if left.dtype != right.dtype or left.tobytes() != right.tobytes():
                    delta = left.astype(np.float64) - right.astype(np.float64)
                    differences.append({"leaf": index, "path": field_path(outputs["off"]["d01"], path),
                        "shape": list(left.shape), "dtype": str(left.dtype),
                        "different_cells": int(np.count_nonzero(left != right)),
                        "max_abs": float(np.max(np.abs(delta))),
                        "rms": float(np.sqrt(np.mean(delta * delta)))})
            host_census = jax.device_get(outputs[mode]["d01"].census)
            results["comparisons"][mode] = {"fields_bit_identical": not differences,
                "differences": differences, "nonfinite_fields": nonfinite,
                "actual_work": dict(zip(WORK, map(int, host_census.work))),
                "guards": {name: list(map(int, row)) for name, row in zip(
                    census_module.GUARDS, host_census.guards)}}
            if bounds is not None:
                results["comparisons"][mode].update(fields=scores, transparency_pass=all(
                    score["pass"] for score in scores.values()))
            print(f"{mode}: different leaves={len(differences)}", flush=True)
        (args.out / "bisect.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    census_module.count_dynamics_guards = original_dynamics
    census_module.count_boundary_guards = original_boundary
    operational._rk_scan_step = original_rk
    census_module.count_guard = original_guard
    if args.bisect:
        return
    if bounds is not None:
        current = {mode: result["d01"] for mode, result in outputs.items()}
        # Scalars are already resident; only the registered checkpoint reads fields.
        for ordinal in range(2, args.drift_steps + 1):
            for mode in ("off", "on"):
                current[mode] = ordinary(current[mode], namelist, step_values[ordinal], clock)
        jax.block_until_ready(current)
        fields = {}
        left_paths = jax.tree_util.tree_flatten_with_path(current["off"])[0]
        right_paths = jax.tree_util.tree_flatten_with_path(current["on"].replace(census=None))[0]
        for (path, left), (_, right) in zip(left_paths, right_paths, strict=True):
            name = field_path(current["off"], path)
            fields[name] = compare_array(name, np.asarray(left), np.asarray(right), bounds)
        results["drift"] = {"ordinary_own_steps": args.drift_steps,
            "scope": "static-radiation-false d01 trajectory; full cadence/LW2 proof separate",
            "fields": fields, "pass": all(score["pass"] for score in fields.values())}
        drift_census = jax.device_get(current["on"].census)
        results["drift"]["actual_work"] = dict(zip(WORK, map(int, drift_census.work)))
        results["drift"]["guards"] = {name: list(map(int, row)) for name, row in zip(
            census_module.GUARDS, drift_census.guards)}
        if results["drift"]["actual_work"]["steps"] != args.drift_steps:
            raise SystemExit("drift device-step count mismatch")
        results["bounds_sha256"] = hashlib.sha256(bounds_path.read_bytes()).hexdigest()
        del current
    for domain in names:
        off = outputs["off"][domain]
        on = outputs["on"][domain].replace(census=None)
        failures, nonfinite = [], []
        for index, (left, right) in enumerate(zip(jax.tree.leaves(off), jax.tree.leaves(on), strict=True)):
            left, right = np.asarray(left), np.asarray(right)
            if left.dtype != right.dtype or left.tobytes() != right.tobytes():
                failures.append(index)
            if np.issubdtype(left.dtype, np.floating) and not np.all(np.isfinite(left)):
                nonfinite.append(index)
        results["domains"][domain] = {"shape": list(off.state.theta.shape),
            "namelist": resolved_namelist(bundles[domain].namelist,
                                           output_cadence_steps=67, output_set=None),
            "fields_bit_identical": not failures, "nonfinite_leaf_indices": nonfinite,
            "different_leaf_indices": failures}
        transparency_failed = not results["comparisons"]["on"]["transparency_pass"] if bounds else bool(failures)
        drift_failed = not results["drift"]["pass"] if bounds else False
        if transparency_failed or drift_failed or nonfinite:
            (args.out / "setup.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
            raise SystemExit(f"{domain}: real PROD replay transparency/finiteness failed")
    del outputs, seeded, off, on
    for mode, inputs in variants.items():
        replay(mode, inputs, steps)
    results["source_hashes"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("src/gpuwrf/runtime/operational_mode.py", "src/gpuwrf/diagnostics/census.py")}
    (args.out / "setup.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    runtime = ctypes.CDLL("<USER_HOME>/miniconda3/lib/python3.13/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12")
    if runtime.cudaProfilerStart() != 0:
        raise SystemExit("cudaProfilerStart failed")
    try:
        for iteration in range(args.repeats):
            order = ("off", "on") if iteration % 2 == 0 else ("on", "off")
            for mode in order:
                with jax.profiler.TraceAnnotation(f"CENSUS:tree:{mode}:{iteration}"):
                    replay(mode, variants[mode], steps)
    finally:
        runtime.cudaProfilerStop()
    (args.out / "completed.json").write_text(json.dumps({"completed": True, "repeats": args.repeats}) + "\n")


if __name__ == "__main__":
    main()
