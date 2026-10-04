"""Real three-domain CPU writer gate; no forecast or GPU timing claim."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import types

import jax
from netCDF4 import Dataset
import numpy as np

from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline-source", type=Path, required=True)
    args = p.parse_args()
    assert not Path("/tmp/wrf_gpu2_quiet").exists()
    assert jax.devices()[0].platform == "cpu"
    state_contract._gpu_device = lambda: jax.devices()[0]
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    config = pipeline.NestedPipelineConfig(input_dir=args.input,
        output_dir=args.out / "candidate", proof_dir=args.out / "proof",
        scratch_dir=args.out / "scratch", hours=24, max_dom=3)
    names = ("d01", "d02", "d03")
    _, bundles, _, start, dts, carries = pipeline._load_domains(config, names)
    print("real domains initialized", time.monotonic() - started, flush=True)

    # Use the actual old materialization method with the same initialized
    # carries and physics context. This is writer regression, not F1 fidelity.
    source = args.baseline_source.read_text()
    old_class = next(n for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "_PerDomainWrfoutWriter")
    old_method = next(n for n in old_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "_materialize_and_submit")
    namespace = dict(vars(pipeline))
    exec(compile(ast.Module(body=[old_method], type_ignores=[]), str(args.baseline_source), "exec"), namespace)
    writers = {}
    for variant in ("baseline", "candidate"):
        dest = args.out / variant
        dest.mkdir()
        writer = pipeline._PerDomainWrfoutWriter(output_dir=dest,
            input_dir=args.input, run_start=start, bundles=bundles,
            output_cadence_steps={n: 1 for n in names}, dt_by_domain=dts)
        if variant == "baseline":
            writer._materialize_and_submit = types.MethodType(namespace["_materialize_and_submit"], writer)
        writers[variant] = writer

    report = {"platform": "cpu", "scope": "actual real-case writer with zero steps; positive-step controls reuse initialized carries",
        "cpu_adapter": "State zero-allocation target only", "initialized_fields": {}, "later_files": {},
        "baseline_materialize_sha256": hashlib.sha256(ast.unparse(old_method).encode()).hexdigest()}
    initial_dir = args.out / "initial"
    initial_dir.mkdir()
    for name in names:
        writers["candidate"](name, 0, carries[name])
        initial = next((args.out / "candidate").glob(f"wrfout_{name}_*"))
        # Keep separate initial-only directory for the D6 zero-hour comparator.
        (initial_dir / initial.name).hardlink_to(initial)
        with Dataset(initial) as emitted, Dataset(args.input / f"wrfinput_{name}") as inp:
            for field in ("PSFC", "T2", "TSK", "U10", "V10", "Q2"):
                report["initialized_fields"][f"{name}/{field}"] = np.array_equal(emitted[field][:], inp[field][:])
        for own_step in (1, {"d01": 67, "d02": 201, "d03": 603}[name]):
            for writer in writers.values():
                writer(name, own_step, carries[name])
        print(name, "initial and two positive-step writer controls emitted", flush=True)
    for path in (args.out / "baseline").glob("wrfout_*"):
        actual = args.out / "candidate" / path.name
        report["later_files"][path.name] = {"baseline": sha(path), "candidate": sha(actual)}
    report["pass"] = (all(report["initialized_fields"].values()) and
        all(v["baseline"] == v["candidate"] for v in report["later_files"].values()))
    report["wall_seconds"] = time.monotonic() - started
    report["utc"] = datetime.now(timezone.utc).isoformat()
    (args.out / "writer_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "initial_fields": len(report["initialized_fields"]),
        "later_files": len(report["later_files"]), "wall_seconds": report["wall_seconds"]}))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
