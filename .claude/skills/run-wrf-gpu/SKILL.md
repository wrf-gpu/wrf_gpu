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
   `wrfinput`/`wrfbdy` or CPU-WRF replay `wrfout` files. Raw `met_em` alone is
   upstream input, not directly runnable by the current CLI.
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

The v0.23.4 candidate is **not yet published and must not be recommended for
deployment ahead of that**: its production-faithful maxdom3 canary is 51.67%
slower than v0.23.3 — measured, root-caused (occupancy/latency-bound, not
bandwidth-bound), understood, but not fixed. The current published release
remains v0.23.3. The candidate accepts
one-way live nesting through d09 on its sealed one-hour
all-physics fixture (27/27 outputs finite; 9 domains × 102 fields within frozen
gates). Present that as a fixture-backed capability, not universal scheme
coverage or a speed claim. Large nine-domain cold compiles can take well over an
hour; the accepted one-hour stress replay used 5,744.407 s warm model wall. The
optional release-canary maxdom9 diagnostic was aborted inconclusive after
3:12:04 with zero completed calls and is not performance evidence.

Always disclose that nested execution currently uses a fixed 30-minute
radiation target instead of arbitrary namelist `radt`, and that nested
`topo_shading`/`slope_rad` currently bind disabled. Do not claim the broad
seasonal/configuration-independent 24-72 h forecast-skill gate is closed.
Also disclose that the single-domain daily pipeline currently drops requested
`moist_adv_opt`/`scalar_adv_opt` and runs `0/0` (the accepted nested path honors
them), and that omitted `time_step_sound` selects 10 acoustic substeps where
WRF derives 4 on the tracked fixture. Residual dry-mass behavior is open; do
not silently change either control or claim exact parity.
For long or strongly ice-sedimentation-sensitive runs, also disclose that the
static Thompson `NSED_MAX=16` cap and widespread exact-zero carried Ni remain
future fidelity work; the v0.23.4 late-Ni fix does not close those separate
items. AOT cheap-key fragmentation can cause a safe recompile when fused-phase
shape or terrain-staging provenance differs, so preserve the cache/staging
namespace and report fallback honestly rather than promising a warm hit.
Default precision is fp64, but v0.23.4's physics fixes intentionally change
ice-sensitive results relative to v0.23.3. Opt-in modes remain opt-in. No
masking, clamps, or silent scheme substitution.

For optimization, preserve `cuda_async`, the version-keyed warm cache, shipped
RRTMG tiling, fp64, timestep, and acoustic settings by default. Propose
`GPUWRF_BATCH_ENSEMBLE=B` only for same-date/same-geometry small-grid ensembles
after a VRAM check. H100/B200/GB300 performance is not release-benchmarked;
measure instead of extrapolating. Refuse or flag >~6000 m terrain, projected
VRAM overflow, unsupported schemes, and unrequested deep nesting. The full
decision table and current 5090 resource evidence are in `AI_OPERATOR.md`.

If running v0.23.4: disclose before a 3-domain
(`--max-dom 3`) or 9-domain (`--max-dom 9`) run that `d03`-activating
configurations measure ~51.67% slower wall-clock than v0.23.3 (correctness
unaffected; the regression is measured, root-caused as occupancy/latency-bound,
and understood, but not fixed). Single-domain and 2-domain (`d01`+`d02`)
configurations have no measured performance regression.
