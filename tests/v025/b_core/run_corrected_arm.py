"""Fresh review-repair receipts, including the full/evolving return comparison."""
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/evidence/BC18")
os.umask(0o077)
OUT.mkdir(mode=0o700, parents=True, exist_ok=False)

def run(*args, analysis=False):
    prefix = ["nice", "-n", "19"] if analysis else []
    subprocess.run([*prefix, *map(str, args)], check=True)

for domain in ("d01", "d02"):
    run(sys.executable, HERE/"verify_pristine.py", "--domain", domain,
        "--wrf-stages", "--payload", "evolving", "--out", OUT/(domain+"_oracle.json"))
for payload in ("full", "evolving"):
    run(sys.executable, HERE/"bench_acoustic.py", "--mode", "bench", "--domain", "both",
        "--payload", payload, "--out", OUT/("warm_"+payload+".json"))
    for domain in ("d01", "d02"):
        name = "nsys_"+payload+"_"+domain
        run("nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
            "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop", "--force-overwrite=true", "-o", OUT/name,
            sys.executable, HERE/"bench_acoustic.py", "--mode", "profile", "--domain", domain,
            "--payload", payload, "--reps", "11", "--batch", "32",
            "--out", OUT/(name+"_bench.json"))
        run("nsys", "export", "--type", "sqlite", "--force-overwrite=true",
            "-o", OUT/(name+".sqlite"), OUT/(name+".nsys-rep"), analysis=True)
        run(sys.executable, HERE/"profile_summary.py", OUT/(name+".sqlite"),
            "--dispatches", "352", "--out", OUT/(name+"_summary.json"), analysis=True)
run(sys.executable, HERE/"bench_acoustic.py", "--mode", "bench", "--domain", "both",
    "--family", "coef", "--out", OUT/"warm_coef.json")
