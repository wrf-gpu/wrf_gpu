# Installation

## Unit-Test Install

Anyone can run the CPU unit-test path. No NVIDIA GPU or reference data is required for the default public checks.

```bash
set -euo pipefail
git clone https://github.com/wrf-gpu/wrf_gpu.git
cd wrf_gpu
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
JAX_PLATFORMS=cpu pytest -q tests/ -k 'not gpu'
```

## Full GPU Install

Target hardware is an NVIDIA RTX 30/40/50-series GPU, with 12+ GB VRAM recommended. The v0.0.1 paper numbers were tested on an RTX 5090. For the current CUDA 13 JAX path, follow the official JAX CUDA installation guidance if this command changes: <https://docs.jax.dev/en/latest/installation.html>.

```bash
set -euo pipefail
git clone https://github.com/wrf-gpu/wrf_gpu.git
cd wrf_gpu
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade --pre jax jaxlib "jax-cuda13-plugin[with-cuda]" jax-cuda13-pjrt \
  -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
python -m pip install -e .
export WRF_GPU_REFERENCE_ROOT=<reference-data-root>
export JAX_PLATFORMS=cuda
python - <<'PY'
import jax
import jax.numpy as jnp
print(jax.devices())
x = jnp.ones((1024, 1024), dtype=jnp.float32)
print(float((x @ x).block_until_ready()[0, 0]))
PY
```

`WRF_GPU_REFERENCE_ROOT` must point at a local directory containing the reference-data layout used by replay and scoring scripts, for example `runs/wrf_l3`, `runs/wrf_l2`, and `artifacts/datasets/aemet_stations`.

## Reproducibility Install

Reviewer pin set for v0.0.1:

- Python 3.13.11
- JAX 0.10.0
- jaxlib 0.10.0
- CUDA toolkit 13.1.115
- NVIDIA driver 595.71.05
- Tested GPU: NVIDIA GeForce RTX 5090, 32607 MiB VRAM

```bash
set -euo pipefail
git clone https://github.com/wrf-gpu/wrf_gpu.git
cd wrf_gpu
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-frozen.txt
python -m pip install -e .
export WRF_GPU_REFERENCE_ROOT=<reference-data-root>
bash scripts/verify_reproducibility.sh
python - <<'PY'
import json
from pathlib import Path

def read(name):
    return json.loads((Path("proofs") / name).read_text())

speed = read("2026-05-27-m7-skill-fix-iter2__post_iter2_speedup.json")
skill = read("2026-05-27-m7-skill-fix-iter2__post_iter2_skill_diff.json")
print("speedup_proof", speed.get("selected_gpu_wall_s"), speed.get("selected_cpu_run_wall_s"))
print("skill_proof_variables", sorted(skill.get("variables", {}).keys()))
PY
```

The frozen requirements file was seeded from `python -m pip freeze` on the release-prep workstation and narrowed to the packages needed for the CPU tests and the documented CUDA/JAX runtime.
