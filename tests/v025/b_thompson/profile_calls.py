"""Capture matched Thompson arms under an already-held manager GPU lease."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def check_quiet():
    if Path("/tmp/wrf_gpu2_quiet").exists():
        print("quiet benchmark active; defer this arm, release GPU lease", flush=True)
        raise SystemExit(125)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=["full", "sedimentation"], default="full")
    parser.add_argument("--compare-layout", action="store_true")
    parser.add_argument("--compare-native-real", action="store_true")
    parser.add_argument("--compare-prep", action="store_true")
    parser.add_argument("--compare-full-column", action="store_true")
    parser.add_argument("--samples", nargs="+", default=["night_d01", "night_d02", "day_d01", "day_d02"])
    args = parser.parse_args()
    if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise RuntimeError("profile_calls requires scripts/with_gpu_lock.sh")
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = args.out / "raw"
    raw.mkdir(exist_ok=True, mode=0o700)
    probe = Path(__file__).resolve().with_name("component_probe.py")
    for sample in args.samples:
        check_quiet()
        common = [sys.executable, str(probe), "--input", str(args.inputs / f"{sample}.npz"),
                  "--kind", args.kind]
        if args.compare_layout:
            common.append("--compare-layout")
        if args.compare_native_real:
            common.append("--compare-native-real")
        if args.compare_prep:
            common.append("--compare-prep")
        if args.compare_full_column:
            common.append("--compare-full-column")
        subprocess.run(["timeout", "240", *common, "--arm", "pair", "--calls", "10",
                        "--out", str(args.out / f"{sample}_pair.json")], check=True)
        for arm in ("baseline", "candidate"):
            check_quiet()
            prefix = raw / f"{sample}_{arm}"
            cmd = ["timeout", "240", "nsys", "profile", "--trace=cuda",
                   "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
                   "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                   "--force-overwrite=true", "-o", str(prefix), *common,
                   "--arm", arm, "--calls", "20", "--capture",
                   "--out", str(args.out / f"{sample}_{arm}.json")]
            print("profiling", sample, arm, flush=True)
            with (raw / f"{sample}_{arm}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            check_quiet()
            subprocess.run(["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                            "--output", str(prefix)+".sqlite", str(prefix)+".nsys-rep"], check=True)


if __name__ == "__main__":
    main()
