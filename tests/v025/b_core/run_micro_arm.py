"""Small coordinated GPU arm; children inherit the approved GPU lock."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
OUT=Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/evidence")
commands=[
    ["verify_pristine.py","--domain","d01","--wrf-stages","--out",str(OUT/"pristine_gpu_d01_stages.json")],
    ["verify_pristine.py","--domain","d02","--wrf-stages","--out",str(OUT/"pristine_gpu_d02_stages.json")],
    ["bench_acoustic.py","--mode","bench","--domain","both","--out",str(OUT/"warm_micro.json")],
    ["bench_acoustic.py","--mode","bench","--domain","both","--family","coef","--out",str(OUT/"warm_coef.json")],
    ["run_profile_arm.py"],
]
for command in commands:
    subprocess.run([sys.executable,str(HERE/command[0]),*command[1:]],check=True)
