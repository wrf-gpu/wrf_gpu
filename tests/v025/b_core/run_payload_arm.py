"""BC17: output only evolving leaves; independent oracle and same nsys metric."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
OUT=Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/evidence/BC17")
OUT.mkdir(parents=True,exist_ok=True)
for domain in ("d01","d02"):
    subprocess.run([sys.executable,str(HERE/"verify_pristine.py"),"--domain",domain,
                    "--wrf-stages","--payload","evolving","--out",str(OUT/(domain+"_oracle.json"))],check=True)
subprocess.run([sys.executable,str(HERE/"bench_acoustic.py"),"--mode","bench",
                "--domain","both","--payload","evolving","--out",str(OUT/"warm.json")],check=True)
for domain in ("d01","d02"):
    name="nsys_"+domain
    subprocess.run(["nsys","profile","--trace=cuda,nvtx","--cuda-graph-trace=node",
                    "--sample=none","--cpuctxsw=none","--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop","--force-overwrite=true","-o",str(OUT/name),
                    sys.executable,str(HERE/"bench_acoustic.py"),"--mode","profile",
                    "--domain",domain,"--payload","evolving","--reps","11","--batch","32",
                    "--out",str(OUT/(name+"_bench.json"))],check=True)
    subprocess.run(["nsys","export","--type","sqlite","--force-overwrite=true",
                    "-o",str(OUT/(name+".sqlite")),str(OUT/(name+".nsys-rep"))],check=True)
    subprocess.run([sys.executable,str(HERE/"profile_summary.py"),str(OUT/(name+".sqlite")),
                    "--dispatches","352","--out",str(OUT/(name+"_summary.json"))],check=True)
