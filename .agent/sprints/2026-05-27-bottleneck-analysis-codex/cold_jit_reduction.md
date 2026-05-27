# Cold-JIT Reduction Options

Sources:
- `.agent/sprints/2026-05-26-m7-gpu-profile-prep/worker-report.md`
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/pipeline_run_20260521.json`
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/integration/daily_pipeline.py`

Current evidence:
- Profile-prep wall-clock proof recorded 1h cold starts of 102.583 s and 106.176 s, then warm 1h runs of 5.707-5.873 s.
- The iter-2 24h pipeline had first two hourly segments of 294.318 s and 264.317 s, then steady hourly segments around 5.86-5.90 s. This makes cold compile/cache behavior the largest end-to-end operational cost in the 732.632 s run.
- The 1 km memory audit recorded 70.420 s cold compile-inclusive time for one warm RK-step probe.

Options, ranked:

1. Persistent JAX compilation cache
- Scope: initialize a persistent compilation cache in the runtime/profiling entry points and pin cache directory under `data/cache/` or another external non-git path.
- Expected saving: high for repeated daily runs with identical shapes/static args. It should remove most 100 s-class cold compilation after the first run on a given binary/JAX/CUDA signature.
- Risk: low correctness risk; medium ops risk from cache invalidation and disk growth.
- Proof needed: two clean-process runs with the same run key; second run must show cold segment reduced while numerical output stays bitwise or tolerance-equal.

2. Ahead-of-time export or precompile at service startup
- Scope: lower/compile `run_forecast_operational` for fixed `(grid, dt_s, acoustic_substeps, radiation_cadence_steps, hours=1.0)` before forecast scheduling.
- Expected saving: high for operational latency, even if total first-run compile still happens during a warmup phase.
- Risk: low numerical risk; medium implementation risk because static arg and donated-buffer signatures must exactly match production.
- Proof needed: startup precompile artifact plus forecast path demonstrating no compile inside the timed forecast.

3. Shape-stable hourly entry point
- Scope: keep the current daily pipeline's repeated 1h call shape, but avoid changing static args across hours. Audit land-refresh dtype/shape and radiation cadence so only one compiled executable is needed.
- Expected saving: medium-high. The first two slow hourly segments imply either multiple compiled variants or compile plus runtime warmup.
- Risk: low-medium; changes can accidentally alter radiation cadence or output timing.
- Proof needed: `JAX_LOG_COMPILES=1` or equivalent compile log proving one compilation for all 24 hourly segments.

4. Split expensive radiation cadence into a separately cached executable
- Scope: compile a non-radiation 1h segment and a radiation-step/block segment separately if the current `while` structure causes multiple variants.
- Expected saving: medium. It can reduce compile size and improve cache reuse, but may add launch boundaries.
- Risk: medium because operator fusion can regress.
- Proof needed: HLO/NSys compiled-region map and no D2H inter-kernel regression.

5. Lower optimization level for exploratory runs only
- Scope: use lower compiler optimization or shorter windows for diagnostics, not production.
- Expected saving: useful for developer iteration.
- Risk: production performance claims become invalid if made under lower optimization.
- Proof needed: separate debug-mode artifact; do not mix with operational timing.

Recommendation:
Run a dedicated compile-cache/precompile sprint before deeper kernel work. It offers the largest end-to-end operational saving with the least physics risk. It does not replace kernel optimization because warm steady-state still has launch/D2D overhead.
