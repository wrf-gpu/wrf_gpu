# v0.22 GPU-opener PREP (scoped while the 9-nest gate held the GPU, 2026-06-26)

Read-only scoping so the v0.22 GPU openers fire correctly the instant the GPU frees.
Consumed by: `proofs/v022/workflows/compile_pathology_r0.workflow.js` (R0 phase) + the K2 dt-ladder run.
These notes are SCOPING, not gospel — every GPU agent must re-confirm against live code before flipping anything.

## K2 — dt / n_sound CFL-first ladder (lossless; roadmap #96/K2)
- Knobs: `dt_s` = `src/gpuwrf/runtime/operational_mode.py:410` (OperationalNamelist, default 10.0s);
  `acoustic_substeps` = `:411` (default 10). Nested dt derived in
  `src/gpuwrf/integration/nested_pipeline.py:121-144` (`_dt_by_domain`) from root time_step × parent_grid_ratio chain;
  namelist built in `_make_namelist()` `:193-221`.
- CURRENT 9-nest canary (run_cadvariant): root d01 dt=54s, d02=18s, d03=6s (`aot_precompile.py:132-134`), n_sound=10 all.
  3-dom Canary same 54/18/6; Switzerland 128²: dt=10s, n_sound=10.
- CFL diagnostic: NO full max-Courant diagnostic exists. Only a mass-continuity stability fraction
  `TEMPORARY_MU_CONTINUITY_CFL_FRACTION=1.0e-3` at `dynamics/acoustic_wrf.py:68,662`. **K2 must ADD a max-Courant
  readout (per-domain max |U|·dt/dx, w·dt/dz) as the gate's CFL-headroom evidence — there is no existing number to trust.**
- Proposed ladder (CFL-first; each rung gated on max-Courant < target AND tiered-identity-or-Tier-O AND no finite/conservation break):
  - R1 baseline: dt_root=54s, n_sound=10
  - R2: dt_root=60s (→20/6.67), n_sound=9
  - R3: dt_root=72s (→24/8), n_sound=8
  - R4 aggressive: dt_root=90s (→30/10), n_sound=7
- Override mechanism — **VERIFY FIRST (Explore was contradictory):** `GPUWRF_ACOUSTIC_SUBSTEPS` may be plumbed
  (`v020_make_namelist.py` comment says "once plumbed"); grep `nested_pipeline.py` for it. dt_root likely needs a
  namelist-input edit (no env plumbing found). If neither env exists, the rung change is a small namelist-build edit,
  NOT a core-dycore edit — keep it config-level.
- K2 is **lossless** (same math, larger step / fewer substeps) → target tiered bit-identity is NOT expected (different dt
  changes results); acceptance = CFL-safe + stable + within strict operational tolerance (this is a real-runtime win, not an
  identity lever). Measure s/step win vs R1 and the skill delta on the canary.

## R0 — compile memory-fitting flag sweep (lower fused cold-compile RAM, ZERO runtime regression)
- Installed jax/jaxlib: pyproject pins only `jax>=0.4`; **the runtime is jax/jaxlib 0.10.0** (per project memory) — the
  R0 agent MUST `python -c "import jax; print(jax.__version__)"` and probe each flag's existence before use; do NOT assume.
- Currently-wired flags (only autotune+parallel-compile): `--xla_gpu_per_fusion_autotune_cache_dir`,
  `--xla_gpu_experimental_autotune_cache_mode`, `--xla_gpu_force_compilation_parallelism`
  (`src/gpuwrf/runtime/xla_autotune.py:150-210`).
- Candidate memory-fitting flags NOT yet wired (validate each against 0.10.0; some may not exist / may be no-ops):
  `jax_memory_fitting_effort`, `jax_optimization_level`, `jax_exec_time_optimization_effort`,
  `jax_compiler_enable_remat_pass`, plus XLA_FLAGS `xla_*memory*` / `xla_*scheduler*` / buffer-assignment knobs.
- Peak host RSS harness to REUSE: `getrusage(RUSAGE_SELF).ru_maxrss` (KiB ×1024) in
  `proofs/v021/canary_gate/perstep_timing_driver.py`.
- AOT blob validity check (Explore wrongly said "no cheap_key blob" — IT EXISTS): blob serialize/load is in
  `src/gpuwrf/runtime/aot_precompile.py` (`precompile()` ~`:151-218`) + cheap_key/cache resolve in
  `src/gpuwrf/runtime/compile_cache.py:93-256`. R0 validates each flag setting: compile → serialize blob → load →
  assert runtime output bit-identical to the fused baseline AND s/step unchanged (HARD-FAIL >1-2% regression).
- R0 outcome is binary: a setting with materially lower peak RAM + zero runtime regression + valid blob → adopt;
  else a clean IMPOSSIBILITY proof and ship-the-blob stays the answer (no "risk=0 by construction").
