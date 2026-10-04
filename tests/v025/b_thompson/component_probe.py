"""Matched real PROD Thompson per-call probe; source tree selected by PYTHONPATH.

Compile and upload before cudaProfilerStart. The captured range has calls and
completion waits only; inputs remain on device and outputs are not copied to
host within the range. Use nsys node tracing to count ops and device time.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import thompson_column as tc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=["baseline", "candidate", "pair"], default="pair")
    parser.add_argument("--kind", choices=["full", "sedimentation"], default="full")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--compare-layout", action="store_true",
                        help="compare column kernels with original vs flattened full-body layout")
    parser.add_argument("--compare-native-real", action="store_true")
    parser.add_argument("--compare-prep", action="store_true")
    parser.add_argument("--compare-full-column", action="store_true")
    parser.add_argument("--compare-dpow", action="store_true",
                        help="full column kernel with libdevice DOUBLE pow vs integral-exponent products")
    parser.add_argument("--source-scopes", action="store_true",
                        help="attach original physics function scopes to HLO metadata")
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError("device probe requires GPU")
    assert jax.devices()[0].platform == "gpu", jax.devices()
    if args.source_scopes:
        for name in ("_clip_species", "_reset_mp8_graupel_number", "_warm_rain_collection",
                     "_cold_collection_rates", "_ice_sources_with_process_flags",
                     "_apply_cold_collection_rates", "_saturation_adjustment_with_condensation",
                     "_rain_evaporation", "_sedimentation", "_fall_speeds", "_sed_cloud_water",
                     "_cloud_water_fall_speed", "_instant_melt_freeze", "_finish",
                     "_clamp_rain_number", "_balance_ice_number", "_snow_moments",
                     "_graupel_distribution", "_rain_distribution", "_cloud_distribution",
                     "_ice_collection_rates", "_snow_terminal_velocity_wrf"):
            setattr(tc, name, jax.named_call(getattr(tc, name), name="thompson" + name))
    metadata = json.loads(args.input.with_suffix(".json").read_text())
    dt = metadata["dt"]
    arrays = np.load(args.input)
    column = tc.ThompsonColumnState(**{key: jnp.asarray(arrays[key])
                                       for key in tc.ThompsonColumnState.__slots__})
    jax.block_until_ready(column)
    function = tc._step_thompson_column_full_impl if args.kind == "full" else tc._sedimentation
    arms = ["baseline", "candidate"] if args.arm == "pair" else [args.arm]
    results, output, executables = {}, {}, {}
    # Each closure is compiled under its own trace-time gate setting.
    for arm in arms:
        os.environ["GPUWRF_THOMPSON_COLUMN_SED"] = ("1" if args.compare_layout or args.compare_native_real or args.compare_prep or args.compare_full_column
                                                  or args.compare_dpow
                                                  else str(int(arm == "candidate")))
        if args.compare_layout:
            os.environ["GPUWRF_THOMPSON_COLUMN_LAYOUT"] = str(int(arm == "candidate"))
        if args.compare_native_real:
            os.environ["GPUWRF_THOMPSON_COLUMN_LAYOUT"] = "1"
            os.environ["GPUWRF_THOMPSON_NATIVE_REAL"] = str(int(arm == "candidate"))
        if args.compare_prep:
            os.environ["GPUWRF_THOMPSON_COLUMN_LAYOUT"] = "1"
            os.environ["GPUWRF_THOMPSON_NATIVE_REAL"] = "1"
            os.environ["GPUWRF_THOMPSON_SED_PREP_FUSED"] = str(int(arm == "candidate"))
        if args.compare_full_column:
            os.environ["GPUWRF_THOMPSON_COLUMN_LAYOUT"] = "1"
            os.environ["GPUWRF_THOMPSON_NATIVE_REAL"] = "1"
            os.environ["GPUWRF_THOMPSON_SED_PREP_FUSED"] = "0"
            os.environ["GPUWRF_THOMPSON_FULL_COLUMN"] = str(int(arm == "candidate"))
        if args.compare_dpow:
            os.environ.update(GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_NATIVE_REAL="1",
                              GPUWRF_THOMPSON_SED_PREP_FUSED="0", GPUWRF_THOMPSON_FULL_COLUMN="1")
            os.environ["GPUWRF_THOMPSON_DPOW_PRODUCTS"] = str(int(arm == "candidate"))
        begin = time.monotonic()
        call = ((lambda s: function(s, dt, False)) if args.kind == "full"
                else (lambda s: function(s, dt)))
        lowered = jax.jit(call).lower(column)
        compiled = lowered.compile()
        hlo = compiled.as_text()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_name(f"{args.out.stem}_{arm}.hlo").write_text(hlo)
        output[arm] = compiled(column)
        jax.block_until_ready(output[arm])
        results[arm] = {"compile_and_first_call_s": time.monotonic() - begin,
                       "hlo_f64_tokens": hlo.count("f64["),
                       "wall_samples_ms": []}
        executables[arm] = compiled
    if len(arms) == 2:
        before = jax.tree.leaves(output["baseline"])
        after = jax.tree.leaves(output["candidate"])
        errors = []
        for leaf_index, (a, b) in enumerate(zip(before, after)):
            an, bn = np.asarray(a), np.asarray(b)
            if not np.isfinite(bn).all():
                raise AssertionError("nonfinite candidate")
            # Regression only, normalized by each output's maximum magnitude.
            diff = float(np.max(np.abs(an-bn)))
            scale = max(float(np.max(np.abs(an))), 1.e-30)
            errors.append(dict(leaf_index=leaf_index, max_abs=diff, normalized_max_abs=diff / scale))
            # The narrow rounding screen applies to unchanged-arithmetic
            # fusion/layout. REAL-vs-DOUBLE work is validated by the frozen WRF
            # gates; report its JAX regression deltas without treating them as
            # a substitute for that independent physics gate.
            if not args.compare_native_real and diff > 2.e-6 * scale:
                raise AssertionError(errors[-1])
        results["regression_only"] = errors
    cuda = ctypes.CDLL("libcudart.so") if args.capture else None
    if cuda:
        assert cuda.cudaProfilerStart() == 0
    for _ in range(args.calls):
        # Alternate arms inside one process to reduce thermal/order drift.
        for arm in arms:
            start = time.perf_counter()
            out = executables[arm](column)
            jax.block_until_ready(out)
            results[arm]["wall_samples_ms"].append((time.perf_counter()-start)*1000)
    if cuda:
        assert cuda.cudaProfilerStop() == 0
    for arm in arms:
        results[arm]["median_wall_ms"] = float(np.median(results[arm]["wall_samples_ms"]))
    event_counters = None
    if (args.compare_full_column or args.compare_dpow) and "candidate" in arms:
        from gpuwrf.kernels import phys_thompson_full as full
        # A separate diagnostic call reports device counters before masks.
        # It runs outside nsys's captured range and leaves the retained return
        # contract and measured executable unchanged.
        events = jax.jit(lambda s: full.full_column(s, dt, return_events=True)[2])(column)
        totals = np.asarray(jnp.sum(events, axis=0)).tolist()
        event_counters = dict(zip(("input_nonfinite", "inadmissible", "pre_mask_nonfinite"), totals))
        assert all(value == 0 for value in event_counters.values()), event_counters
    source_files = [Path(tc.__file__)]
    if "candidate" in arms:
        from gpuwrf.kernels import phys_thompson_sedimentation as kernel
        source_files.append(Path(kernel.__file__))
        if args.compare_full_column:
            source_files.append(Path(full.__file__))
    report = dict(work=metadata, kind=args.kind, calls=args.calls, results=results,
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_files},
                  cache_dir=os.environ.get("GPUWRF_JAX_CACHE_DIR"),
                  compare_layout=args.compare_layout,
                  compare_native_real=args.compare_native_real,
                  compare_prep=args.compare_prep,
                  compare_full_column=args.compare_full_column,
                  event_counters=event_counters,
                  event_scope="separate device diagnostic before retained fallback; outside timing capture",
                  source_scopes=args.source_scopes,
                  backend=jax.default_backend(), hardware=str(jax.devices()[0]),
                  oracle_scope="JAX regression and real workload only; WRF gates separate")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({arm: results[arm]["median_wall_ms"] for arm in arms}), flush=True)


if __name__ == "__main__":
    main()
