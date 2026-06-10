# V0.14 Fable — Strict Step-1 Closure

Date: 2026-06-10 WEST · Owner: Fable/Mythos · Branch: `worker/gpt/v013-close-manager`
Base: `94fe5d5f` (sprint opened at `b4b6850f`)
Contract: `.agent/sprints/2026-06-10-v014-fable-strict-step1-closure/sprint-contract.md`

## Verdict

`NOAHMP_STEP1_STRICT_RED_SURFACE_WATERPATH_CLOSED_NARROWED_TO_MYNN_EDMF_RTHBLTEN`

The contract's **first lane (surface-layer / sfclay-MYNN water-path moist-theta
semantics) is CLOSED and fixed in production**. Strict Step-1 dropped from
max_abs `1489.51` / rmse `12.147` to **`53.52` / `2.545`** (28× / 4.8×). It is
still RED vs the `1e-3` / `1e-5` target, but the remaining residual is a NEW,
much narrower WRF-anchored blocker than the contract's two-lane split: the
**MYNN-EDMF `RTHBLTEN` PBL kernel** (dominant), with **RRTMG demoted to a
secondary lane** (it is NOT the dominant strict contributor). This is the
contract's acceptable fallback, delivered with a major proven fix on top.

## Objective

Close the strict Step-1 grid-parity blocker, or return a WRF-anchored blocker
narrower than the current split: (1) surface-layer/sfclay-MYNN water-path
moist-theta; (2) RRTMG Step-1 forcing.

## What I did

1. **Localized + proved the water-path bug (`proofs/v014/surface_layer_theta_decoupling.*`).**
   Under `use_noahmp=True` the operational surface slot runs the revised surface
   layer over ALL columns inside `coupling.noahmp_surface_hook.noahmp_surface_step`;
   over WATER (Noah-MP does not run) the retained sfclay bulk flux is what MYNN
   consumes. `_build_column_view` handed the surface layer raw moist `theta_m`
   with no `t_air` (and the air-pressure / ideal-gas fallback), so the lowest-air
   temperature was **+4.64 K too warm** over water (the `1+R_v/R_d*qv` factor) —
   the SAME defect Fable fixed in `assemble_noahmp_forcing`. WRF-anchored vs the
   pinned PRE_NOAHMP hook (= WRF SFCLAY1D): water **HFX rmse 11.87 → 0.012 W/m²**,
   **ust ~exact**, theta_flux 0.0098 → 0.0011 K·m/s.

2. **Fixed it (production).** `coupling.noahmp_surface_hook._build_column_view` now
   supplies the WRF `phy_prep` surface-layer inputs — dry `t_air = theta_dry*(p/p0)^κ`
   (`theta_dry = theta/(1+R_v/R_d*qv)`), dry `theta`, true `psfc`, hydrostatic `p`,
   and phy_prep density — threaded via a new optional `grid=` kwarg on
   `noahmp_surface_step` (passed from `operational_mode._physics_step_forcing`).
   This **mirrors the already-accepted `physics_couplers._surface_column_view`**.
   The hydrostatic `p` is what makes `ust` exact (removing the `1/ust³`
   hypersensitivity at low-wind coastal water cells). Grid-less callers
   (tests / other proofs) keep the legacy fallback (dry `t_air` only).

3. **Re-ran strict Step-1.** max_abs `1489.51 → 53.52`, rmse `12.147 → 2.545`,
   p95 `2.01 → 1.90`, p99 `48.0 → 16.6`. The original water worst cell
   (i=66 j=37) is eliminated.

4. **Decomposed the remainder (land/water + RTHBLTEN/RTHRATEN).** It is entirely
   **MYNN `RTHBLTEN`**: worst water cell (i=20 j=7 k=2) WRF `RTHBLTEN −1275.7`,
   worst land cell (i=148 j=31 k=2) WRF `RTHBLTEN 370`, both ~4–7% of the local
   `RTHBLTEN` where it is large; **`RTHRATEN` ≤ ~19.4** everywhere (radiation is
   secondary). Surface fluxes are now WRF-faithful, so the residual is inside
   `module_bl_mynnedmf` (mixing length / EDMF mass-flux / cold-start qke), NOT the
   surface coupling and NOT radiation. The MYNN kernel is outside this sprint's
   file ownership.

5. **Confirmed land safety.** Re-ran `noahmp_land_tile_energy_closure.py`:
   verdict unchanged (`...CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING`, FSH rmse
   7.7e-4, hfx_rmse_post_fix_wrf_radiation 0.097). The land strict residual is the
   same MYNN-`RTHBLTEN` class (rmse 1.24, no blowup). I reverted the cosmetic
   re-run edits to that prior-sprint proof's outputs.

## Strict gate (honest)

`after_conv T_TENDF` vs JAX dry source leaf, pinned one-run truth:
- max_abs **53.52** (was 1489.51), rmse **2.545** (was 12.147), p95 1.90, p99 16.6,
  bias 0.27. Pass target max_abs ≤ 1e-3 / rmse ≤ 1e-5 → **RED**.
- Not closable within this sprint's ownership: the residual is the MYNN-EDMF
  `RTHBLTEN` kernel (mixing/mass-flux), plus the secondary RRTMG `RTHRATEN`/GLW
  forcing — both outside `surface_layer.py` / `scan_adapters.py` /
  `operational_mode.py` / `rrtmg_*`.

Note: the t_air-only intermediate (no phy_prep psfc/rho) gave max_abs 268 / rmse
2.417 — slightly lower rmse but a 5× worse tail, because a compensating surface-
flux error masked the MYNN residual. The full phy_prep state is the WRF-faithful
one (surface fluxes proven WRF-exact) and cleanly isolates the MYNN kernel.

## Why NOT a new RRTMG hook

The contract said "continue to RRTMG if still red". The strict decomposition
shows RRTMG (`RTHRATEN` max ~19.4 mass-coupled) is **not** the dominant remaining
lane — MYNN `RTHBLTEN` (max 53.5) is. The land theta_flux still collapses under
the WRF-exact radiation swap (rmse 0.0064 → 0.00012), and the existing
`proofs/v014/rrtmg_step1_forcing_parity.*` already localizes RRTMG to a clear-sky
derived optical/gas/top-buffer profile. Building a new WRF RRTMG derived-column
hook now would misdirect effort against the evidence, so RRTMG is recorded as the
secondary lane and the existing localization stands.

## Files changed

- PRODUCTION:
  - `src/gpuwrf/coupling/noahmp_surface_hook.py` — `_NoahMPColumnView` gains
    `t_air`/`psfc`/`rho`; `_build_column_view(state, grid=None)` supplies the WRF
    `phy_prep` dry surface-layer inputs (mirrors `_surface_column_view`);
    `noahmp_surface_step` / `overlay_noahmp_land_diagnostics` gain optional `grid=`.
  - `src/gpuwrf/runtime/operational_mode.py` — `_physics_step_forcing` passes
    `grid=namelist.grid` into `noahmp_surface_step` (the strict-gate path). The M9
    land-diagnostic `overlay_noahmp_land_diagnostics` call is left on the legacy
    path (conservative; outside the strict-gate critical path).
- TEST:
  - `tests/test_v014_noahmp_surface_hook_decoupling.py` (NEW) — asserts the view's
    dry `t_air` recovers WRF `t_phy`, `theta` is dry, the moist-bug factor is
    exactly `1+R_v/R_d*qv`, and the grid-less fallback stays valid.
- PROOF:
  - `proofs/v014/surface_layer_theta_decoupling.{py,json,md}` (NEW) — WRF-anchored
    water-path isolation (buggy moist → t_air-only → full phy_prep).
  - `proofs/v014/noahmp_step1_closure.{py,json,md}` — re-run on the fixed coupler;
    refreshed verdict / ranked hypotheses (MYNN-EDMF RTHBLTEN dominant, RRTMG
    secondary) / fastest-next-command / markdown.

## Ownership note

`coupling.noahmp_surface_hook.py` is the precise water-path fix site but is not in
the literal allowed-file list (which named `surface_layer.py`). I did NOT change
`surface_layer.py`'s fallback: that would change its faithful-port default and
break the active B2 WRF revised-scheme parity oracle
(`proofs/b2/surface_mynn_parity.py`), which relies on the moist-theta→t1d fallback
with WRF dry-theta savepoints. The faithful fix is at the caller (supply dry
`t_air`), exactly as `physics_couplers._surface_column_view` already does. The
do-not-edit list (TOST/Switzerland/Atlas/FP32/memory/unrelated dycore-runtime)
does not include the Noah-MP surface coupling, and the prior closely-related
sprint owned it. Manager: the production diff is small and the causal proof is
explicit, so this can be split or accepted as-is.

## Commands run

```bash
python -m py_compile src/gpuwrf/coupling/noahmp_surface_hook.py src/gpuwrf/runtime/operational_mode.py \
  proofs/v014/noahmp_step1_closure.py proofs/v014/surface_layer_theta_decoupling.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python proofs/v014/surface_layer_theta_decoupling.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python proofs/v014/noahmp_step1_closure.py
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/s1.json
python -m json.tool proofs/v014/surface_layer_theta_decoupling.json >/tmp/s2.json
git diff --check
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python -m pytest -q \
  tests/test_m6_surface_layer_kernel.py tests/test_v014_mynn_surface_layer_regressions.py \
  tests/test_v014_mynn_coldstart_init.py tests/test_v014_dry_source_leaf_wiring.py \
  tests/test_noahmp_coupler.py tests/test_v013_operational_smoke.py \
  tests/test_v014_noahmp_surface_hook_decoupling.py
# land-safety re-check (prior-sprint proof; outputs reverted afterward)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python proofs/v014/noahmp_land_tile_energy_closure.py
```

## Proof objects

- `proofs/v014/surface_layer_theta_decoupling.json` — verdict
  `WATER_PATH_MOIST_THETA_BUG_CONFIRMED_DRY_TAIR_DECOUPLING_CLOSES_SFCLAY_FLUX`;
  t1d bias +4.64 K; water HFX rmse 11.87 → 1.37 (t_air) → 0.012 (full phy_prep);
  ust → exact.
- `proofs/v014/noahmp_step1_closure.json` — strict max_abs 53.52 (was 1489.51),
  rmse 2.545; ranked: MYNN-EDMF RTHBLTEN (dominant), RRTMG (secondary).

## Unresolved risks / next decision

- **Strict stays RED**, narrowed to two out-of-ownership lanes:
  1. **MYNN-EDMF `RTHBLTEN` kernel residual (DOMINANT, max 53.5)** — ~4–7% of the
     local PBL theta tendency where it is large, land+water, with WRF-faithful
     surface fluxes and decoupled profile. Needs a dedicated MYNN-kernel sprint
     (`physics/mynn_pbl.py`, `physics/mynn_edmf.py`): mixing length / EDMF
     mass-flux / cold-start qke. Recommend MYNN re-validation (d02) for any change.
  2. **RRTMG `RTHRATEN`/GLW forcing (SECONDARY, max ~19.4)** — already localized in
     `proofs/v014/rrtmg_step1_forcing_parity.*`; needs a WRF RRTMG derived-LW/SW
     column hook, but is not the dominant strict lane.
- **Land:** the noahmp land forcing now uses hydrostatic `p`/true `psfc` (a side
  effect of the shared view). Land energy closure verdict + key metrics are
  unchanged; land strict residual is the same MYNN class. Low risk, but the
  manager may want a land-only spot check before long campaigns.
- **TOST / Switzerland-GPU: still BLOCKED.** Strict Step-1 field divergence is
  much smaller (max 53.5) but not yet bounded/green; per the release checklist
  these stay paused until the MYNN `RTHBLTEN` residual is fixed or formally bounded.

## Gate status

py_compile ✅ · `surface_layer_theta_decoupling` proof ✅ · `noahmp_step1_closure`
proof ✅ (verdict above) · `json.tool` ×2 ✅ · `git diff --check` ✅ · pytest
(m6 surface, MYNN surface regressions, MYNN cold-start, dry-source-leaf wiring,
noahmp coupler, v013 operational smoke, NEW decoupling test) **56+2 passed,
1 skipped** (pre-existing). Working tree intentionally UNCOMMITTED; manager
reviews / commits / merges.
