"""Run the accepted frozen Gate-2/L3a/L3b gates on a selected source tree.

Retained WRF inputs/outputs are read only. Tolerances and mutants remain those
of source commit 83c7ff126; copied runners carry SHA provenance. --interpret
explicitly uses the Pallas CPU interpreter for column-kernel development.
"""
import argparse
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "oracle_reference"
RETAINED = Path("<USER_HOME>/p1_cpu_20260923")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    path = REFERENCE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", choices=["gate2", "l3a", "l3b", "all"], default="all")
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--dtype", choices=["production", "fp64", "both"], default="both")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    if args.require_gpu:
        import jax
        assert jax.devices()[0].platform == "gpu", jax.devices()
        print(json.dumps({"asserted_platform": jax.devices()[0].platform,
                          "hardware": str(jax.devices()[0])}), flush=True)
    if args.interpret:
        import jax
        if jax.default_backend() != "cpu":
            raise RuntimeError("--interpret requires CPU backend")
        from gpuwrf.kernels import phys_thompson_sedimentation as kernel
        kernel.fill_down = functools.partial(kernel.fill_down, interpret=True)
        kernel.sediment_one_species = functools.partial(kernel.sediment_one_species,
                                                        interpret=True)
        if os.environ.get("GPUWRF_THOMPSON_SED_PREP_FUSED", "0") == "1":
            from gpuwrf.kernels import phys_thompson_prep as prep
            prep.sediment_and_finish = functools.partial(prep.sediment_and_finish,
                                                         interpret=True)
        if os.environ.get("GPUWRF_THOMPSON_FULL_COLUMN", "0") == "1":
            from gpuwrf.kernels import phys_thompson_full as full
            full.full_column = functools.partial(full.full_column, interpret=True)
    args.out.mkdir(parents=True, exist_ok=True)
    dtypes = ["production", "fp64"] if args.dtype == "both" else [args.dtype]
    cmp = REFERENCE / "p0_gate2_compare.py"
    g2 = REFERENCE / "gate2_tolerances.json"
    mp28 = REFERENCE / "p0c_mp28_tolerances.json"
    results = []
    if args.gate in ("gate2", "all"):
        runner = load("p0_gate2_compare")
        for dtype in dtypes:
            out = args.out / f"gate2_{dtype}.json"
            rc = runner.main([
                "--npz", str(RETAINED / "p0g2_r1/column/port_column.npz"),
                "--wrf-output", str(RETAINED / "p0g2_r1/run/OUTPUT.txt"),
                "--tolerances", str(g2), "--tolerances-sha256", sha(g2),
                "--repo", str(args.repo), "--state-dtype", dtype, "--out", str(out)])
            results.append(dict(gate="gate2", dtype=dtype, rc=rc, path=str(out)))
    for gate, runner_name, retained, manifest in [
        ("l3a", "p0c_ocol_runner", "p0cl3a_r1", "p0c_column_discriminators.json"),
        ("l3b", "p0l3b7c_runner", "p0l3b7c_r2", "p0c_aero_column_manifest_r7.json"),
    ]:
        if args.gate not in (gate, "all"):
            continue
        runner = load(runner_name)
        manifest_path = REFERENCE / manifest
        columns = json.loads(manifest_path.read_text())["columns"]
        for column in columns:
            for dtype in dtypes:
                out = args.out / f"{column}_{dtype}.json"
                tol = g2 if gate == "l3a" else mp28
                params = dict(compare_module=str(cmp), compare_sha256=sha(cmp),
                              manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
                              column=column, npz=str(RETAINED / retained / "columns" / f"port_column_{column}.npz"),
                              wrf_output=str(RETAINED / retained / "run" / f"OUTPUT_{column}.txt"),
                              tolerances=str(tol), tolerances_sha256=sha(tol),
                              state_dtype=dtype, out=str(out), repo=str(args.repo))
                if gate == "l3b":
                    params.update(gate2_tolerances=str(g2), gate2_tolerances_sha256=sha(g2))
                rc = runner.run(SimpleNamespace(**params))
                results.append(dict(gate=gate, column=column, dtype=dtype, rc=rc, path=str(out)))
                print(json.dumps(results[-1]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["rc"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
