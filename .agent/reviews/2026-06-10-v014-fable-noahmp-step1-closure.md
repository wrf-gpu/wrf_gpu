# V0.14 Fable NoahMP Step-1 Closure — sprint review

- date: 2026-06-10
- owner: Fable/Mythos (tmux 0:1)
- contract: `.agent/sprints/2026-06-10-v014-fable-noahmp-step1-closure/sprint-contract.md`
- verdict: `NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_NOAHMP_LAND_TILE_ENERGY`

## Objective

Close the v0.14 Step-1 grid-parity blocker: the JAX Step-1 live-nest/source-capture
path was built with NoahMP disabled (`use_noahmp=False`, `sf_surface_physics=None`,
no NoahMP land/static state) while the WRF fixture runs `sf_surface_physics=4`.

## What was done

1. **Localized the drop point.** `proofs/v014/step1_live_nest_init_rerun.py::build_live_nest_step1_inputs`
   built the namelist with no surface/land configuration and a bare carry;
   `step1_rk1_p_state_source_split.py::apply_mythos_perturb_init` rebuilt the carry
   and dropped it again. Production (`operational_mode._physics_step_forcing`) already
   supports the full WRF chain (sfclay -> `noahmp_surface_step` overlay -> MYNN);
   **no production code change was needed** — the entire gap was Step-1 proof-builder
   configuration.
2. **Wired WRF-derived NoahMP state into the Step-1 builder** (the contract fix):
   - `build_noahmp_land_state(RUN_CASE3, "d02")` + `build_noahmp_params` (faithful
     WRF NOAHMP_INIT replica from `wrfinput_d02`; tables byte-identical between the
     case dir, pristine WRF, and the truth-producing tree);
   - `use_noahmp=True`, `sf_surface_physics=4`, WRF clock `julian=120.75`
     (0-based fractional day-of-year, `ESMF dayOfYear_r8 - 1`, frame/module_domain.F:2165 —
     NOT the 1-based `tm_yday` used by older proofs), `yearlen=365`;
   - `topo_shading=1`, `slope_rad=1` + `radiation_static` loaded from the case
     (previously 0/0/None against a WRF namelist that sets 1/1);
   - WRF-faithful **step-1 held radiation seeds**: Noah-MP forcing (SOLDN/LWDN/COSZ)
     and `carry.rthraten`, both computed eagerly from the step-1 entry state at the
     WRF forward interval midpoint `xtime + radt/2` (the lead-0 alternative is
     falsified by measurement, see below);
   - seeds re-derived from the PATCHED state in `apply_mythos_perturb_init` so the
     strict capture's carry matches WRF's actual start-of-step state.
3. **Re-emitted the strict WRF truth as ONE consistent rmol-pinned run.** The prior
   strict part2 truth (`wrf_truth`, 00:00Z emission) came from an UNPINNED build and
   carries the proven WRF uninitialized-`rmol` UB (~22% step-1 MYNN-source envelope
   across builds), making the 1e-3/1e-5 strict gate ill-posed against it. The current
   pinned binary re-emitted ALL step-1 surfaces (part2 strict target + MYNN driver
   boundary + surface/land handoff stages) in one run; outputs are **byte-identical
   across re-runs and across two pinned builds** (determinism proven). `wrf_truth` is
   now a symlink to `wrf_truth_pinned_onerun`; the old dir is retained as
   `wrf_truth_00z_prepin_unpinned_build`.
4. **Mirrored the production surface slot in the strict proof path**
   (`step1_mynn_source_coupling.build_step1_state`): sfclay scan adapter ->
   `noahmp_surface_step` land overlay -> MYNN, matching WRF
   `SFCLAY1D_mynn -> noahmplsm -> MYNN driver input`.
5. **Found + fixed an adjacent PRODUCTION bug on the same closure path** (contract
   Required Work item 3): under `use_noahmp=True` the surface layer runs INSIDE
   `noahmp_surface_adapter` (`surface_layer_with_diagnostics(state)`, sf_opt=5 is
   not in `SFCLAY_SCAN_ADAPTERS`), and `first_timestep` was never threaded into
   that call — so the just-merged WRF MYNN surface FIRST-CALL semantics
   (b89ec7bb: UST first guess, MOL=0, land QSFC, Li_etal_2010 z/L seed) never
   engaged on the Noah-MP path. Measured proof: post-overlay UST rmse was
   0.0865 (the pre-fix value) instead of the fixed 0.0295. Fix: `first_timestep`
   threaded `_physics_step_forcing` -> `noahmp_surface_step` ->
   `noahmp_surface_adapter` -> `surface_layer_with_diagnostics` (default `False`
   keeps every existing caller bit-identical; traced-flag safe). New focused
   test `tests/test_noahmp_coupler.py::test_first_timestep_threads_into_blend_sfclay`.

## Key measurements (WRF-anchored)

- Radiation seed convention pinned by hook truth: `+radt/2` seed SOLDN vs WRF SWDOWN
  rmse `2.76` W/m2 vs lead-0 seed rmse `56.45` W/m2 (lead-0 falsified; WRF samples
  the forward interval midpoint `xtime + radt/2` at the step-1 call).
- LWDN vs WRF GLW: uniform clear-sky bias `+17.44` W/m2 (p50 17.2, p1-p99 11.4-20.9;
  both sides cloud-free) — a systematic RRTMG-LW forcing offset feeding Noah-MP.
- JAX MYNN kernel re-exonerated vs the PINNED truth: WRF inputs + WRF init QKE ->
  raw RTHBLTEN strong-ratio median `0.9976`, corr `0.99993`. With OUR inputs and
  WRF QKE injected the sources drop to median `0.398` — the residual is entirely
  the surface input/flux boundary, not the kernel.
- Mass-coupled RTHRATEN seed vs WRF part2 (interior): max_abs `19.43`, rmse `2.49`
  (WRF field max `41.9`) — real but an order below the RTHBLTEN-side residual.
- Post-overlay UST boundary after the first_timestep fix: rmse `0.0865 -> 0.0289`
  (bias `-0.0765 -> -0.0254`) — the GPT b89ec7bb fix now engages on this path.
- Post-overlay MYNN flux boundary (land theta_flux/kinematic HFX): max_abs `0.252`
  K m/s, bias `-0.042`, rmse `0.062` — UNCHANGED by the sfclay fix and UNCHANGED
  (max 0.252, rmse 0.064) when the overlay is fed WRF's EXACT hook SWDOWN/GLW
  (causal radiation-swap split) -> the deficit is INSIDE the Noah-MP land tile.
- Land-input parity vs WRF PRE_NOAHMP hook: tslb1/smois1/sh2o1/tsk/vegfra/snow all
  match to hook print precision (max_abs <= 5e-9) -> not a land-state init bug.
  Diagnostic-level albedo-carry anomaly (+0.45) flags the two-stream albedo chain.
- Raw RTHBLTEN vs WRF: max_abs `0.0161`, strong-median `0.41`, corr `0.856`
  (with WRF inputs + WRF qke the kernel gives `0.998`/`0.99993`).
- POST_NOAHMP overlay structure verified: ZNT max_abs `4.6e-5`, land qv_flux rmse
  `1.4e-6`, TSK land rmse `2.4 K` (max `4.1 K`).

## Strict gate

- after-conv `T_TENDF` vs JAX dry `T_TENDF` (vs the PINNED one-run truth):
  max_abs `1489.5`, RMSE `13.20`, p95 `2.02`. NOT comparable 1:1 to the contract's
  `438.54/5.47`: that number was measured against the UNPINNED truth AND with the
  JAX side missing radiation + NoahMP entirely (two large signals absent on both
  sides of that comparison cancel differently). The honest decomposition: the JAX
  T_TENDF now CONTAINS RTHRATEN + NoahMP-driven RTHBLTEN (both were zero/bulk
  before), and the dominant remaining term is the land-tile HFX deficit
  (JAX mass-coupled RTHBLTEN max `1242` vs WRF `2523` at the strong cells;
  strict worst cell i=66 j=37 k=3: WRF `-2457.6` vs JAX `-968.1`).
- Release gate (max_abs <= 1e-3 AND rmse <= 1e-5): RED — contract's acceptable
  endpoint delivered instead: a WRF-anchored blocker strictly narrower than
  "NoahMP disabled/missing land state", with causal-split evidence and a ranked
  table (see `proofs/v014/noahmp_step1_closure.md`).

## Files changed

- `proofs/v014/step1_live_nest_init_rerun.py` — NoahMP/radiation enablement in the
  Step-1 builder (+ `wrf_step1_julian_yearlen`, `noahmp_step1_carry_seeds` helpers).
- `proofs/v014/step1_rk1_p_state_source_split.py` — carry seeds threaded through
  `apply_mythos_perturb_init` (re-derived from the patched state).
- `proofs/v014/step1_mynn_source_coupling.py` — production-mirrored surface slot
  (NoahMP overlay), pinned-truth hook root, refreshed verdict strings.
- `proofs/v014/step1_surface_land_flux_handoff.py` — verdict logic recognizes the
  enabled configuration (`...CLOSED_JAX_NOAHMP_ENABLED`).
- `proofs/v014/noahmp_step1_closure.py` (NEW) — the closure proof: truth provenance +
  determinism, config echo, radiation-seed-vs-hook, RTHRATEN-vs-part2, post-overlay
  flux boundary, strict gate, ranked hypotheses, fastest next command.
- PRODUCTION (first_timestep threading into the Noah-MP blend; default-inert):
  - `src/gpuwrf/runtime/operational_mode.py` (one call site: pass `first_timestep`
    into `noahmp_surface_step`)
  - `src/gpuwrf/coupling/noahmp_surface_hook.py` (`noahmp_surface_step` kwarg)
  - `src/gpuwrf/physics/noahmp_coupler.py` (`noahmp_surface_adapter` kwarg ->
    `surface_layer_with_diagnostics(state, first_timestep=...)`)
  - `tests/test_noahmp_coupler.py` (+1 focused test; 3 passed)

## Commands run

```bash
# one-run consistent rmol-pinned truth emission (WRF, ~32 s)
cd /tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/run && env \
  WRFGPU2_STEP1_PART2_SOURCE_LEAVES_SPLIT=1 ..._ROOT=.../wrf_truth_pinned_onerun \
  WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY=1 WRFGPU2_SOURCE_SAVE_BOUNDARY=1 \
  WRFGPU2_V014_MYNN_HOOK=1 WRFGPU2_V014_SURFACE_HANDOFF_HOOK=1 ... ./wrf.exe
# gates
python -m py_compile proofs/v014/step1_mynn_source_coupling.py proofs/v014/step1_surface_land_flux_handoff.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/noahmp_step1_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_surface_land_flux_handoff.py
python -m json.tool proofs/v014/*.json
git diff --check
JAX_PLATFORMS=cpu ... pytest -q tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py tests/test_noahmp_coupler.py  # 16 passed, 1 pre-existing skip
```

Proof verdicts on the final run: `noahmp_step1_closure` =
`NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_NOAHMP_LAND_TILE_ENERGY` (exit 0);
`step1_mynn_source_coupling` strict `1489.51 / 13.20` (identical metric in both
proofs — same capture, same pinned truth); `step1_surface_land_flux_handoff` =
`STEP1_SURFACE_LAND_FLUX_HANDOFF_CLOSED_JAX_NOAHMP_ENABLED` with
`use_noahmp=True, sf_surface_physics=4, inputs_have_noahmp_land=True` (the
contract's blocker configuration is gone).

## Proof objects

- `proofs/v014/noahmp_step1_closure.{py,json,md}` (NEW, primary)
- `proofs/v014/step1_mynn_source_coupling.{py,json,md}` (rerun vs pinned truth)
- `proofs/v014/step1_surface_land_flux_handoff.{py,json,md}` (config now enabled)

## Unresolved risks

- The strict target CHANGED with the truth re-anchor (pinned one-run set). All
  pre-existing `step1_*` artifacts that quote strict numbers vs the old
  `wrf_truth` (e.g. `step1_source_fidelity_closure.json` 1497.6/13.3,
  `step1_part2_source_leaves_split.json`) are now historical; their scripts read
  the symlinked pinned truth on the next rerun. The old dir is retained at
  `wrf_truth_00z_prepin_unpinned_build`.
- `/tmp` truth volatility: the pinned truth set lives under
  `/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/` and
  `/tmp/wrfgpu2_v014_surface_handoff_pinned_onerun/`; a reboot wipes it. It is
  reproducible from the archived WRF patch diffs + the documented one-run
  command (provenance + sha256 in `noahmp_step1_closure.json`).
- RRTMG GLW carries a uniform clear-sky `+17.4 W/m2` bias vs WRF (both sides
  cloud-free) and the RTHRATEN seed differs from WRF part2 at mass-coupled
  max_abs `19.4` — second-ranked lane, proven NOT the cause of the land flux
  deficit, but it must be closed for the 1e-3/1e-5 release gate.
- `noahmp_initial_rad` (production cold-start seed for operational runs) uses
  lead=0 solar time; the WRF-faithful step-1 convention is `+radt/2` (measured:
  rmse 56.4 vs 2.8 W/m2). Operationally this only affects the first radt
  interval; left unchanged (validated operational behavior) — flagged for the
  manager.

## Next decision needed

- Dispatch the Noah-MP land-tile ENERGY sprint (fastest next command in
  `noahmp_step1_closure.json`): per-column WRF `noahmplsm` energy in/out hook on
  the pinned tree (FVEG/LAI/SAI, CM/CH in+out, two-stream SAV/SAG/FSR/FSA albedo
  chain, SH/EV/GH/TRAD), column-diff vs `physics.noahmp` at the strict worst
  cell (i=66, j=37), fix the diverging chain, rerun
  `proofs/v014/noahmp_step1_closure.py`. Secondary lane (parallelizable, GPT):
  RRTMG GLW/RTHRATEN forcing parity hook.
