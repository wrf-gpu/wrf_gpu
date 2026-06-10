# Worker Report: V0.14 Noah-MP In Standalone Nested Pipeline

Date: 2026-06-10
Worker: Fable high (isolated worktree `fable-noahmp-nested`)
Branch: `worker/fable/v014-noahmp-nested` (base `7c819067`)
Status: READY FOR MANAGER REVIEW — source fix + CPU proof complete; GPU gates
are the manager's (per contract).

## Objective

Fix the v0.14 release blocker: the standalone live-nested pipeline never
enabled an LSM (`_make_namelist` omitted `use_noahmp`/`sf_surface_physics`),
so land TSK was FROZEN whole-run on both domains
(proofs/v014/canary_h24_residual_adjudication.md). After this fix, a case
namelist with `sf_surface_physics=4` activates the proven prognostic Noah-MP
coupler per domain, with the land carry seeded before the first scan and the
wrfout writer reading the EVOLVED land carry.

## Files changed

- `src/gpuwrf/integration/nested_pipeline.py` (+~207)
  - `_domain_sf_surface_physics`: per-domain `&physics sf_surface_physics`
    (max-dom list/scalar), FAIL-CLOSED on anything outside `{0, 4}` — no
    silent bulk fallback (the frozen-land hazard is named in the error).
  - `_wrf_julian_yearlen`: WRF Noah-MP clock — `grid%julian` = 0-based
    FRACTIONAL day-of-year (frame/module_domain.F:2165, per the step-1
    closure finding; NOT `tm_yday`), leap-aware yearlen.
  - `_load_domains`: when option 4, builds `build_noahmp_land_state` +
    `build_noahmp_params` per domain and replaces the namelist with
    `use_noahmp=True, sf_surface_physics=4, noahmp_static/energy/rad/nroot/
    julian/yearlen` (mirrors the proven s6b/TOST wiring). Builds the initial
    `OperationalCarry` per domain via the SAME `_initial_carry_for_run` the
    domain-tree cold start used (bulk path bit-identical) and, under Noah-MP,
    seeds `noahmp_land` + `noahmp_rad = noahmp_initial_rad(...)` (real t=0
    held radiation; nocturnal LWDN cold-start mitigation) BEFORE the first
    `_advance_chunk` — no `None -> NoahMPLandState` promotion can occur inside
    a JAX scan, and the carry structure is stable across all 72h segments
    (subsequent segments reuse `result.carries`). Returns the carries as a new
    6th tuple element; `execute_nested_pipeline` passes them to segment 1.
  - `_noahmp_surface_diagnostics_for_output` + writer: the writer declares
    `wants_carry = True`, receives the full carry, and under Noah-MP routes
    `compute_m9_diagnostics(..., noahmp_land=carry.noahmp_land,
    noahmp_rad=carry.noahmp_rad)` so land TSK/HFX/LH and the LSM 2-m T2 are
    the prognostic overlay and SWDOWN/GLW report the held WRF-cadence
    radiation (L1 COSZEN-phase fix). Deliberately NOT best-effort (a wiring
    error fails at the first hourly output instead of silently regressing to
    frozen-looking fallbacks). Bulk path keeps the previous best-effort
    diagnostics unchanged.
  - Per-domain payload metadata: `land_surface` block with
    `sf_surface_physics`, `use_noahmp`, `noahmp_static_loaded`,
    `noahmp_energy_params_loaded`, `noahmp_rad_params_loaded`,
    `noahmp_land_seeded`, `noahmp_n_land_cells`, `noahmp_julian/yearlen`,
    provenance wrfinput path.
- `src/gpuwrf/runtime/domain_tree.py` (+9/−2, the narrowest carry mechanism)
  - `run_domain_tree_callbacks.maybe_output`: output callbacks that set a
    truthy `wants_carry` attribute receive the FULL carry; default callbacks
    keep the historical state-only payload (backward compatible; no change to
    `_advance_chunk`, physics ordering, cadence, or any timestep logic).
- `tests/test_v014_noahmp_nested_pipeline.py` (new, 17 tests, CPU-only)
- `proofs/v014/noahmp_nested_pipeline_activation.py` (new, CPU-only proof
  runner) + emitted `.json`/`.md`.

No timestep-loop host/device transfer added: all seeding happens at init; the
writer diagnostics run at the hourly output boundary exactly as before.

## Commands run (return codes)

- `python -m py_compile src/gpuwrf/integration/nested_pipeline.py src/gpuwrf/runtime/domain_tree.py` → rc 0
- `PYTHONPATH=src pytest -q tests/test_v013_tost_wrfbdy_fix.py` → rc 0 (3 passed)
- `PYTHONPATH=src pytest -q tests/test_v014_noahmp_nested_pipeline.py` → rc 0 (17 passed)
- `PYTHONPATH=src pytest -q tests/test_v0110_domain_tree.py tests/test_gwd_operational_wiring.py` → rc 0 (12 passed; no regression from the wants_carry/6-tuple changes)
- `python proofs/v014/noahmp_nested_pipeline_activation.py` (JAX_PLATFORMS=cpu,
  CUDA_VISIBLE_DEVICES='', cores 0-3) → rc 0, verdict
  `NOAHMP_NESTED_ACTIVATION_CPU_PROVEN`

All CPU-only. The running Canary 72h GPU job was not touched (verified the
single 23 GiB python compute process is the pre-existing run; this worker
initialized only the JAX CPU backend).

## Proof objects

- `proofs/v014/noahmp_nested_pipeline_activation.json` / `.md` — against the
  REAL case the 72h gate consumes
  (`/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`):
  - d01: sf=4, use_noahmp=True, statics/params/land non-null, n_land_cells=523,
    initial land-mean TSK 300.946 K;
  - d02: sf=4, use_noahmp=True, non-null, n_land_cells=768, initial land-mean
    TSK 294.879 K;
  - **corroboration**: those two seeds match EXACTLY the frozen land-mean TSK
    the h24 review measured in the running GPU output (d01 300.95 K, d02
    294.88 K) — the run is frozen at precisely this initial land state.
  - WRF clock julian=120.75 / yearlen=365 (0-based fractional convention);
  - fail-closed probe: sf=2 raises with the frozen-land reason;
  - structural: namelist/carry fields + writer/domain-tree `wants_carry` all
    verified;
  - gpu_used=false, backend=cpu.
- `tests/test_v014_noahmp_nested_pipeline.py` — pins the per-domain option
  resolution, fail-closed set `{0,4}`, WRF clock, and the carry opt-in payload
  semantics (default callbacks still get state-only).

## CPU-proof limitation (documented per contract)

The full `_load_domains` device build (State + initial carry +
`noahmp_initial_rad` RRTMG t=0 seed) requires a visible GPU
(`contracts/state.py::State.zeros`); a CPU one-step land probe through the
production loader is therefore not possible without violating the GPU lock.
Every CPU-provable element (namelist resolution, land/static/param builds on
the real case, clock, carry/writer structure, fail-closed) is proven above;
the coupler itself is separately step-1-closure-proven
(proofs/v014/noahmp_step1_closure.md, incl. the first_timestep threading fix,
which engages here automatically since the nested run starts at global step 1).

## Unresolved risks

1. **VRAM (mandatory preflight)**: Noah-MP adds a land carry + per-domain
   static/param bundles on BOTH domains plus one eager init-time RRTMG
   transient per domain; the previous green preflight did not include this
   path. Manager must rerun the exact-branch memory preflight before the h1-h4
   gate.
2. The d01 ocean noon mass dip (−98 Pa) and QVAPOR marginal growth may not
   fully collapse with the land fix (bounded RRTMG clear-sky / water-path
   moisture lanes); the 72h rerun discriminates.
3. `noahmp_julian` uses the WRF-faithful 0-based fractional convention; the
   older single-domain TOST runner used `tm_yday` (1-based). Effect is a
   ≤1-day phenology clock shift — not a behavior regression on this path
   (which previously had NO Noah-MP at all), but flagged for cross-driver
   consistency.
4. Mixed per-domain options (e.g. `4, 0`) are wired and tested structurally,
   but no real fixture exercises them; Canary/Switzerland are `4, 4`.

## Next decision needed

Manager review + merge, then the contract's GPU gates in order:
(1) exact-branch memory preflight with Noah-MP active on the nested path;
(2) Canary d02 h1-h4 gate (land TSK bias ≤2 K, land HFX bias ≤40 W/m² at
h2-h4, no 17-field regression); (3) full Canary 72h rerun; (4) Switzerland
only after (3) green/bounded. GPU gates can start as soon as the current
Canary 72h run finishes and the branch is merged.

## Commit

`worker/fable/v014-noahmp-nested` @ **`1326b6c9`**
("v014 activate noahmp in standalone nested pipeline", base `7c819067`) —
contains all source/test/proof changes plus this report.
