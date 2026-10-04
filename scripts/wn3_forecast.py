"""WN3 three-nest forecast using the existing asserted-GPU benchmark harness.

Run under with_gpu_lock --label wn3 and an arm timeout, from a detached snapshot.
Arguments are the bench harness arguments (--input-dir, --out, --hours, --tag).
Only the CLI domain count and lead-zero/full history are changed; reuse its
wall/compile/copy/VRAM observation wrappers and in-process real CLI execution.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

START_NS = time.monotonic_ns()
START_WALL = time.time()
ROOT = Path(__file__).resolve().parents[1]
os.environ["BENCH_SRC_REPO"] = str(ROOT)
os.environ["GPUWRF_FULL_WRFOUT_VARIABLES"] = "1"
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("wn3_bench", ROOT / "scripts/bench/s0_nested_harness.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)
harness.T0_NS, harness.T0_WALL = START_NS, START_WALL
harness.REC["schema"] = "wrf_gpu2.v025.wn3.nested_harness.v1"
harness.REC["cpu_affinity"] = sorted(os.sched_getaffinity(0))
harness.REC["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")

from gpuwrf import cli
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.coupling import physics_couplers as pc

load_domains = pipeline._load_domains
column_inputs = pc._rrtmg_column_inputs
harness.REC["radiation_consumer_flags"] = []


def observed_load(*args, **kwargs):
    result = load_domains(*args, **kwargs)
    harness.REC["terrain_radiation"] = {}
    harness.REC["runtime_namelist"] = {}
    for name, bundle in result[1].items():
        nml = bundle.namelist
        assert int(nml.topo_shading) == int(nml.slope_rad) == 1, name
        assert nml.radiation_static is not None, name
        harness.REC["terrain_radiation"][name] = {"topo_shading": int(nml.topo_shading),
            "slope_rad": int(nml.slope_rad), "statics_loaded": True}
        # Metadata-only audit: no device pull, no equation/dispatch changes.
        harness.REC["runtime_namelist"][name] = {
            field: value for field in nml.__dataclass_fields__
            if (value := getattr(nml, field)) is None or isinstance(value, (bool, int, float, str))
        }
        harness.REC["runtime_namelist"][name]["mass_shape"] = [bundle.grid.nz, bundle.grid.ny, bundle.grid.nx]
    return result


def observed_inputs(state, grid=None, **kwargs):
    harness.REC["radiation_consumer_flags"].append({"shape": state.t_skin.shape,
        "topo_shading": int(kwargs.get("topo_shading", 0)), "slope_rad": int(kwargs.get("slope_rad", 0))})
    return column_inputs(state, grid, **kwargs)


pipeline._load_domains = observed_load
pc._rrtmg_column_inputs = observed_inputs

original = cli.main


def wn3_cli(argv):
    args = list(argv)
    args[args.index("--max-dom") + 1] = "3"
    args.append("--emit-initial-history")
    # Optional extra CLI args (restart tests: checkpoint/resume), JSON list; default none.
    args += json.loads(os.environ.get("WN3_CLI_EXTRA", "[]"))
    harness.REC["cli_argv"] = args
    return original(args)


cli.main = wn3_cli

if __name__ == "__main__":
    raise SystemExit(harness.main())
