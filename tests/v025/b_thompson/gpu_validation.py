"""Frozen full-column and active precipitation/cold oracles in one short lease."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def check_quiet():
    if Path("/tmp/wrf_gpu2_quiet").exists():
        print("quiet benchmark active; defer validation, release GPU lease", flush=True)
        raise SystemExit(125)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise RuntimeError("GPU validation requires scripts/with_gpu_lock.sh")
    here = Path(__file__).resolve().parent
    check_quiet()
    subprocess.run(["timeout", "720", sys.executable, str(here / "run_oracles.py"),
                    "--repo", str(args.repo), "--out", str(args.out), "--require-gpu"], check=True)
    code = ("import jax; assert jax.devices()[0].platform == 'gpu', jax.devices(); "
            "print('asserted active-oracle platform', jax.devices()[0].platform, flush=True); "
            "import pytest; raise SystemExit(pytest.main(['-q', "
            "'tests/test_thompson_precip_oracle.py','tests/test_thompson_cold_collection_oracle.py',"
            "'--basetemp','<USER_HOME>/wrf_gpu2_lanes/b-thompson/pytest_active_gpu']))")
    check_quiet()
    subprocess.run(["timeout", "180", sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    main()
