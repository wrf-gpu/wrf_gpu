---
name: run-wrf-gpu
description: Use when a user asks to run, set up, or get their WRF or forecast case working with wrf_gpu.
---

# Run wrf_gpu for an End User

This skill is for operating `wrf_gpu`, not developing the port.

First read the repo-root `AI_OPERATOR.md` and follow it as the authoritative
runbook. Keep the user informed before installs, GPU checks, cold compiles, and
long integrations.

Six-step dispatcher:

1. Understand the request: identify the case directory, forecast hours, domain
   or `--max-dom`, requested output, and whether inputs are standalone
   `wrfinput`/`wrfbdy` or CPU-WRF replay `wrfout` files.
2. Set up the environment: Python 3.11 venv, `pip install --upgrade
   "jax[cuda13]"`, `pip install -e .` or `pip install -e ".[cuda]"`, then
   `python -c "import jax; print(jax.devices())"` and confirm CUDA appears.
3. Check GPU/driver: run `nvidia-smi` and confirm JAX lists a CUDA device.
4. Smoke-test the bundled case: set `GPUWRF_WRF_ROOT`, run
   `examples/switzerland_d01` for 1 hour with `python -m gpuwrf.cli run`, and
   confirm a `wrfout_d01_*` appears.
5. Prepare and run the user's case: use `GPUWRF_WRF_ROOT` for Noah-MP/RRTM/RRTMG,
   let `gpuwrf run` fail-closed on unsupported namelist choices, choose
   `--domain` for single-domain or `--max-dom N` for live nested, and narrate the
   cold compile plus ETA. Mention the default finite guard and nested VRAM
   preflight.
6. Deliver and verify: show the `wrfout` path, inspect with `ncdump -h`, and
   state any limitations honestly.

Do not claim the open 24 h/72 h forecast-skill gate is closed. Default precision
is fp64. Opt-in modes remain opt-in. No masking, clamps, or silent scheme
substitution.
