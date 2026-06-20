# v0.14 d01 LU16 Nonfinite Closure — Noah-MP WATER 1-based category gather

Date: 2026-06-10 · Worker: Fable (worker/fable/v014-noahmp-d01-lu16)
Fix commit: `80a693e2`

## Verdict

**ROOT-CAUSED + FIXED.** The 51 nonfinite d01 cells in the failing L2 nested
preflight (`<DATA_ROOT>/wrf_gpu_validation/v014_noahmp_l2_preflight_20260610T192315Z`)
are exactly the 51 **ISLTYP=1 (sand)** land cells of d01. LU_INDEX=16 was
coincident (sand occurs only under LU16 in this domain), **not causal**.

`water_hydro._category_index` indexed the frozen 1-based Noah-MP parameter
tables with `category - 1`. The tables (`tables._parse_soilparm`) are
`(ncat+1,)` with rows 1..ncat filled and **row 0 an all-zero placeholder**
(WRF's 1-based layout; `TRANSFER_MP_PARAMETERS` does `BEXP_TABLE(SOILTYPE)`
directly). Consequences:

1. **ISLTYP=1 (sand) → row 0 → SMCMAX=0** → `factr = max(0.01, smc/smcmax) = inf`
   → `wdf = dwsat·inf^(bexp+2) = 0·inf = NaN` in `_wdfcnd1` → NaN SMOIS/SH2O in
   all 4 layers on the **first** WATER call → NaN TG/TSK/HFX/LH/UST/T2/TH2 +
   LWUPB/LWUPT/OLR via the energy balance on subsequent steps → `rc=1`,
   `final_state_finite=false`.
2. **Every other soil category silently ran WATER with the PREVIOUS category's
   hydraulic parameters** (e.g. loamy-sand used sand's). Finite, so it never
   tripped a gate: the S4 water savepoint gate's binding criterion is mass
   conservation, which an off-by-one parameter row conserves perfectly.
3. Same helper gathered `ch2op` by `ivgtyp-1` in `_canwater` (IVGTYP=1 read the
   0.0 dummy row; rows 1..20 are uniform 0.1 → otherwise inert).

`noahmp_driver._gather_vec` and the phenology gathers already index 1-based;
WATER was the sole outlier.

## Evidence chain

1. **Cell identity (exact, 1:1).** In the failing wrfout_d01:
   `T2` nonfinite at 51 cells; domain has 51 land cells with ISLTYP=1; the
   sets are identical (`bad∧sand = 51`, `bad∧¬sand = 0`). Healthy LU16 cells
   (424) all have ISLTYP∈{2,6,8,9,12,13}. Discriminator = soil type, not LU.
2. **CPU repro** (`proofs/v014/noahmp_d01_lu16_nonfinite_repro.py`): production
   `noah_mp_step` on the real `wrfinput_d01` warm-start
   (`build_noahmp_land_state`) with frozen wrfinput-derived forcing, CPU fp64.
   Pre-fix: `smois` NaN at step 1 at 204 entries = 51 sand cells × 4 layers.
3. **Stage bisection** (`proofs/v014/noahmp_d01_lu16_bisect.py`): phenology,
   precip_heat, radiation+energy, phasechange all finite at sand cells; first
   NaN producer = stage 5 `noahmp_water_hydro` (inputs finite:
   smois=[0.0102,0.063,0.063,0.063], edir=1.2e-7, etran=0, no precip).
4. **Table layout proof**: loaded `load_noahmp_parameters` rows:
   `smcmax[0]=0` (dummy), `smcmax[1]=0.339` (sand), `smcmax[2]=0.421`.
   Pre-fix gather for ISLTYP=1 returned row 0.

## Fix (WRF-faithful, minimal)

`src/gpuwrf/physics/noahmp/water_hydro.py::_category_index`:
`clip(category - 1, 0, size-1)` → `clip(category, 0, size-1)` — the category
id IS the row index into the 1-based table, identical to
`noahmp_driver._gather_vec`. No masking, no clamps, no output-only patches.

## Proofs run (CPU)

| Proof | Result |
|---|---|
| 200-step CPU repro on real wrfinput_d01 (post-fix) | finite everywhere; sand TG cools 297.2→296.7 K over 1 h (18z evening), HFX small negative — physical |
| `tests/test_v014_noahmp_water_soil_category.py` (3 tests: gather-row identity vs driver gather; one-step WATER on the exact failing dry-sand config; all-category finite sweep) | **pass post-fix; all 3 fail pre-fix** |
| S4 water savepoint gate `proofs/noahmp/water_savepoint_gate.py` (real-WRF oracle) | conservation 11/11 PASS, finite, parity-constrained 8/11 — unchanged verdict vs pre-fix (savepoint columns move soil water ~1e-6/step → too parameter-insensitive to see the shift; that is HOW the bug survived) |
| Noah-MP test set (coupler, nested pipeline, surface hook, energy canopy, checkpoint, sh2o init, phenology, new water tests) | 47 pass + 8 energy savepoint pass (needs `WRF_PRISTINE_ROOT=<USER_HOME>/src/wrf_pristine/WRF` inside worktrees; path default is worktree-relative — pre-existing, not physics) |

## GPU confirmation

Bounded re-run of the SAME exact-branch 1h L2 nested preflight via
`scripts/run_gpu_lowprio.sh` from this worktree (commit `80a693e2`):
`<DATA_ROOT>/wrf_gpu_validation/v014_noahmp_l2_preflight_fix_20260610T205333Z`

**GREEN.** `rc=0`, preflight verdict `PASS_SHORT_GPU_PREFLIGHT`, pipeline
verdict `PIPELINE_GREEN`, `all_domains_finite=true`
(d01 `final_state_finite=true`, d02 `final_state_finite=true`). The 9
previously-nonfinite output fields (T2/UST/HFX/LH/TSK/TH2/LWUPB/LWUPT/OLR)
have **0** nonfinite cells in the new wrfout_d01; T2 over the 475 LU16 land
cells spans 285.4–301.0 K, all finite. Peak total VRAM 9783 MiB (vs ~10042
in the failing run) — no memory regression. Preflight JSON/MD:
`proofs/v014/exact_branch_memory_preflight.{json,md}` (this worktree).

## Residual risk / scope notes

- The off-by-one affected WATER hydraulic parameters for **all** soil
  categories since S4 — soil-moisture evolution in every prior Noah-MP run was
  mis-parameterized (one category off). Step-1/flux-level gates were
  insensitive (energy path gathers correctly); multi-hour soil drawdown
  (H4 land gate, TOST) should improve or shift slightly. Re-run H4 after merge.
- `_soil_param` still interprets a length-4 (==NSOIL) 1-D array as a per-layer
  field, not a category table — fine for real 20-row tables, a sharp edge for
  synthetic fixtures (documented in the new test).
