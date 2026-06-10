# V0.14 Canary h24 Residual Adjudication

Date: 2026-06-10
Author: Fable high (bounded debug analyst, tmux 0:3)
Sprint: `.agent/sprints/2026-06-10-v014-fable-canary-h24-residual/sprint-contract.md`
Run root: `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_moistcqw_20260610T171818Z`
CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
Proof: `proofs/v014/canary_h24_residual_adjudication.{json,md}`
GPU: untouched. Run: alive at h29 during analysis, not stopped.

## Decision: `FABLE_FIX_REQUIRED_AFTER_RUN`

Let the 72h run finish — there is no stability risk and the dynamics-lane
evidence (moist-cqw, PSFC diagnostic, LBC cadence) remains valid and green
against the manifests. But the run **cannot be promoted as the release
field-parity gate**, because the analysis found one new, proven, material
physics-wiring bug that must be fixed and the gate rerun:

**GPU land skin temperature is frozen for the entire run, on both domains.**

## The root cause (new, proven, not a dycore/kernel bug)

GPU land-mean `TSK` is constant to 0.01 K for 29 straight hours — d02 at
294.88 K, d01 at 300.95 K — while CPU truth swings 283 → 310 K through the
diurnal cycle. The fluxes prove the physics actually consumes the frozen skin
(not writer-only): land HFX bias is **+183 W/m² at night** (warm skin over cold
air) and **−308 W/m² at noon** (cold skin under strong sun), land T2 bias
+4.8 K night / −2.2 K day, daytime PBLH −100 m.

Mechanism, confirmed in code:

- `src/gpuwrf/integration/nested_pipeline.py::_make_namelist` builds
  `OperationalNamelist.from_grid(...)` **without** `use_noahmp`,
  `sf_surface_physics`, or `noahmp_static`.
- `OperationalNamelist.use_noahmp` defaults `False` → the land tile stays on
  the prescribed bulk surface path (`physics_couplers.surface_adapter`), which
  never advances `state.t_skin` over land. Ocean is unaffected (SST prescribed
  identically on both sides; ocean TSK bias is exactly 0).
- CPU truth runs Noah-MP (`sf_surface_physics=4`). The validated JAX Noah-MP
  coupler (`noahmp_surface_hook.noahmp_surface_step`, step-1-closure-proven)
  exists and is wired in `daily_pipeline`, but was **never threaded into the
  standalone nested pipeline** that the canary/TOST runs use since the v0.13
  KI-5 wrfbdy rerouting. The v0.12.0 final 24h nested gate ran this same
  pipeline, so it also had frozen land TSK (honest-record item for
  KNOWN_ISSUES).

Why every prior gate missed it: the short h1–h4 gates are evening/night hours
where the frozen-skin bias is smallest, the strict Step-1 proofs run at step 1
where frozen TSK equals the correct initial TSK, and the land fraction here is
only 7.3% (d02) / 9.5% (d01), so global aggregates dilute a 15 K land signal
~13×.

## What the formal h24 `FAIL` is actually made of

Only four fields fail the tolerance manifest; **no dynamics field fails at any
lead 1–24**. PSFC's worst lead (h18, RMSE 105.3 Pa) passes its 120 Pa limit.

| Rank | Field/family | Magnitude | Trend | Likely cause | Release relevance | Next action |
|---|---|---|---|---|---|---|
| 1 | Land surface family: TSK frozen; HFX/LH/T2/PBLH/TSK biases diurnally locked | land TSK ±15 K; land HFX −308…+183 W/m²; T2 manifest fail leads 10–12 (1.54–1.59 vs 1.5) | Periodic, repeats every diurnal cycle, no growth | **Frozen land TSK: nested pipeline never enables an LSM** (`use_noahmp=False` default) | **Blocking for release gate**; also blocks Switzerland GPU (land-dominated domain) | Fix sprint after run finishes (defined below), rerun Canary 72h |
| 2 | Diurnal mass excursion `PSFC/MU/P` (domain-wide incl. ocean) | PSFC RMSE 11→105 Pa@h18→35@h24→40@h29; d01 noon ocean bias −98 Pa | Diurnal, recovering; comparator "slope/h +2.0" is a linear fit aliasing the cycle | Frozen land TSK on both domains + bounded RRTMG clear-sky heating lane + ocean moisture lane | Within manifest (≤120 Pa) everywhere | Re-adjudicate after land fix; no dycore sprint on this signal |
| 3 | `QVAPOR` | RMSE 1.29e-3 at h24 vs 1e-3 limit (fail from h12); 1.35e-3 at h29 | Growth decelerating; bias −1.37e-4→−1.11e-4 shrinking after h24 | Water-path surface/PBL moisture lane (sustained LH −26 W/m², Q2 +1.6e-3, shallower GPU PBL traps moisture) | Marginal (≤35% over); watch | Re-check at h48/h72 final compare; if still growing post-land-fix, open water-path moisture-flux lane |
| 4 | `MUB/PB` static | max 250 Pa, RMSE 9.3/4.5 | Time-invariant | 5-cell nest-frame edge only; interior max 0.0078 Pa = fp32 wrfout quantum (verified this run) | Known static lane; manifest limit 0.2 Pa is interior-grade | Root-cause/annotate the nest-frame band or scope the manifest to interior before release |
| 5 | `T2` night leads 10–12 | 1.54–1.59 vs 1.5 | Recovers daytime | Direct frozen-TSK signature (stable night PBL over land) | Resolved by rank-1 fix | Covered by rank-1 gate |
| 6 | `QNRAIN` sparse | max 3122 #/kg, **p99 = 0 at every lead** | Episodic | Chaotic rain-cell placement in a number-concentration field | Report-only | Keep report-only; consider relative/number-field tolerance class |

Classification per the contract's question: this is **a missing-physics
configuration/wiring issue in the operational nested pipeline** (rank 1) plus
**known surface/radiation tolerance and static-edge lanes** (ranks 2–5) plus
**one comparator/tolerance-class artifact** (rank 6). It is **not** a new
kernel/dycore blocker: U/V/W/T RMSE saturate (U ~0.8 m/s flat from h9, W
~0.044, T peaks 0.71 K then declines), spot checks at h26–h29 show no renewed
drift (PSFC 40–43 Pa, P 22–30 Pa), and there is no LBC-cadence-style monotonic
growth anywhere (the old failure was 51–73 Pa/h).

## Concrete recommendation to manager

1. **Continue the run to h72** (`CONTINUE` in effect; decision label is
   `FABLE_FIX_REQUIRED_AFTER_RUN` because the gate cannot go green from this
   run). The 72h completion still buys: 72h stability evidence, the full
   diurnal-cycle envelope as a pre-fix baseline, and h48/h72 QVAPOR trend
   adjudication.
2. **Do not stop the run** — nothing here threatens the run, and the GPU job is
   the only source of the h48/h72 trend data.
3. **Hold Switzerland GPU until the land fix is merged.** The Gotthard d01
   domain is land-dominated alpine terrain; frozen land TSK there would
   invalidate the entire 72h campaign, far worse than 7% land here.
4. After the GPU frees: run the fix sprint below, then **rerun the Canary d02
   72h gate** from the fixed branch.

## Next sprint (endpoint-sized) — nested-pipeline land-surface activation

Objective: the standalone nested pipeline runs the same prognostic land surface
as CPU truth (Noah-MP, `sf_surface_physics=4`), so land TSK/HFX/T2 follow the
diurnal cycle.

Scope (frozen interfaces, single worker):

- `src/gpuwrf/integration/nested_pipeline.py`: per-domain, build `NoahMPStatic`
  from the domain `wrfinput` statics, seed the `noahmp_land` carry, pre-build
  energy/rad param bundles, thread `noahmp_julian`/`noahmp_yearlen` from run
  start, and set `use_noahmp=True` (or read `sf_surface_physics` from the case
  namelist, fail-closed on unsupported schemes). Mirror the existing
  `daily_pipeline` wiring; do not modify the coupler itself
  (`noahmp_surface_hook` is step-1-closure-proven, including the
  `first_timestep` threading fix).
- CPU-provable before GPU: the runner's `--setup-only` path on
  `JAX_PLATFORMS=cpu` proves namelist/static/carry construction for d01+d02;
  one CPU step-1 probe proves land `t_skin` advances (≠ wrfinput value) and
  land HFX magnitude is physical at night.
- Re-run the exact-branch memory preflight: Noah-MP adds a land carry on both
  domains; the previous green preflight (8858 MiB) did not include it on this
  path.

Proof gate (exact, falsifiable):

1. Short GPU h1–h4 gate (same harness as
   `proofs/v014/moist_cqw_gpu_h4_validation.md`): **d02 land-mean TSK bias
   |≤2 K| at h2–h4** (currently +7.7…+9.5 K) and **land HFX bias |≤40 W/m²|**
   (currently +116…+135), with no regression in the 17-field RMSE table
   (deltas ≥ −5% tolerated only for surface-family fields).
2. Full Canary d02 72h rerun: **T2 manifest green at all 72 leads (≤1.5)**,
   **per-lead land-mean TSK bias within ±3 K at every lead** (kills the frozen
   signature), PSFC worst-lead RMSE materially below 105 Pa, and QVAPOR
   re-adjudicated against its 1e-3 manifest with the diurnal envelope known.
3. Only after (2) is green or formally bounded: Switzerland 72h GPU.

## Unresolved risks

- The ocean component of the noon mass dip (d01 ocean −98 Pa) may not fully
  collapse with the land fix; the residual would then point at the bounded
  RRTMG clear-sky heating lane (atmospheric-tide forcing) or the water-path
  moisture lane. The 72h rerun will discriminate.
- QVAPOR may stay marginally over 1e-3 even post-fix (ocean-dominated). If so,
  it needs either a water-path LH/PBL-mixing sprint or a recorded
  tolerance-policy decision — not a silent gate change.
- VRAM: Noah-MP on both domains in the nested pipeline is untested at the 72h
  config; the memory preflight in the sprint scope is mandatory, not optional.

## Commands run / files changed

- Read-only netCDF probes against run-root GPU wrfouts and CPU truth (CPU
  cores 0–3, no JAX, no GPU): per-lead land/ocean splits, MUB/PB frame/interior
  split, h26–h29 spot compares, d01 noon attribution.
- Code recon (read-only): `nested_pipeline.py`, `operational_mode.py`,
  `noahmp_surface_hook.py`, `physics_dispatch.py`, `daily_pipeline.py`,
  `run_one_case_v0120.py`.
- New files: this report + `proofs/v014/canary_h24_residual_adjudication.{json,md}`.
  No source edits (fix interacts with the live GPU job's pipeline and needs its
  own gated sprint).
