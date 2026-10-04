"""Warm profiler windows only; run after both independent WRF gates pass."""
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
OUT=Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/evidence")
for file in ("pristine_gpu_d01_stages.json","pristine_gpu_d02_stages.json"):
    if not json.loads((OUT/file).read_text())["passed"]:
        raise RuntimeError("refusing to profile an unvalidated native kernel")
for domain in ("d01","d02"):
    name="nsys_"+domain
    subprocess.run(["nsys","profile","--trace=cuda,nvtx","--cuda-graph-trace=node",
                    "--sample=none","--cpuctxsw=none","--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop","--force-overwrite=true","-o",str(OUT/name),
                    sys.executable,str(HERE/"bench_acoustic.py"),"--mode","profile",
                    "--domain",domain,"--reps","11","--batch","32","--out",str(OUT/(name+"_bench.json"))],check=True)
    subprocess.run(["nsys","export","--type","sqlite","--force-overwrite=true",
                    "-o",str(OUT/(name+".sqlite")),str(OUT/(name+".nsys-rep"))],check=True)
    subprocess.run([sys.executable,str(HERE/"profile_summary.py"),str(OUT/(name+".sqlite")),
                    "--dispatches","352","--out",str(OUT/(name+"_summary.json"))],check=True)
subprocess.run(["ncu","--set","full","--profile-from-start","off","--replay-mode","kernel",
                "--kernel-name","regex:b_core","--launch-count","4","--force-overwrite",
                "-o",str(OUT/"ncu_d02"),sys.executable,str(HERE/"bench_acoustic.py"),
                "--mode","profile","--domain","d02","--reps","1","--batch","1",
                "--out",str(OUT/"ncu_d02_bench.json")],check=True)
with (OUT/"ncu_d02_raw.csv").open("w") as stream:
    subprocess.run(["ncu","--import",str(OUT/"ncu_d02.ncu-rep"),"--page","raw","--csv"],
                   stdout=stream,check=True)
