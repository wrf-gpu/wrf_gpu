"""Run unchanged pristine-WRF savepoint gates in a private output directory.

The source gate and fixture SHA hashes are recorded. GPU mode asserts the
actual backend and avoids tests/v025's CPU-forcing conftest.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gates", default="energy,phenology,water,integration",
                        help="Explicit frozen gates; native water uses the separately registered REAL gate")
    args = parser.parse_args()
    if Path("/tmp/wrf_gpu2_quiet").exists():
        print("quiet active; defer savepoint validation", flush=True)
        return 125
    sys.path.insert(0, str(args.repo / "src"))
    import gpuwrf  # Configure import-time XLA flags before backend initialization.
    import jax
    assert jax.devices()[0].platform == args.backend, jax.devices()
    print("asserted savepoint backend", jax.devices()[0].platform, flush=True)
    source = args.repo / "proofs/noahmp"
    records = []
    for gate in args.gates.split(","):
        if gate not in ("energy", "phenology", "water", "integration"):
            raise ValueError(gate)
        if Path("/tmp/wrf_gpu2_quiet").exists():
            return 125
        filename = ("integration_step_gate.py" if gate == "integration"
                    else f"{gate}_savepoint_gate.py")
        path = source / filename
        spec = importlib.util.spec_from_file_location(f"frozen_{gate}_gate", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = args.out / gate
        out.mkdir(parents=True, exist_ok=True)
        for name in ("savepoints_energy.json", "savepoints_all.json"):
            target = out / name
            if not target.exists():
                target.symlink_to(source / name)
        # Redirect report output only. Functions, fixtures and tolerances stay
        # exactly those of the existing gate.
        module.HERE = out
        rc = module.main()
        fixtures = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                    for name in ("savepoints_energy.json", "savepoints_all.json")}
        records.append(dict(gate=gate, rc=rc, backend=args.backend,
                            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            fixtures_sha256=fixtures, tolerances=module.TOL))
    (args.out / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    return int(any(row["rc"] for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
