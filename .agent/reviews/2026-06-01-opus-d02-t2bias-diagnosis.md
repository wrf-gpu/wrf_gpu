# d02 3 km T2 warm-bias diagnosis (post-P1-4a) — pressure-Exner AND a genuine theta-side physics drift

Date: 2026-06-01
Agent: Opus 4.8 MAX (final-verdict branch, main working tree)
Scope: DIAGNOSIS + fresh measurement. No production `src/` edits. New diag script
under `scripts/diag/`; proofs committed; findings here.

## TL;DR verdict

The d02 (primary product, 3 km) T2 warm bias is **NOT** explained by the
force_geopotential difference vs d03, and it is **NOT** a single-cause bias. On the
current post-P1-4a HEAD (d9f452e), a fresh 24 h d02 run of the representative MAM
case3 (20260521_18z) decomposes into **TWO comparable, independent causes**:

1. **The SAME ~+2.2-2.6 kPa diagnostic surface-pressure / Exner artifact as d03**
   — present even though d02 uses `force_geopotential=True`. Accounts for ~+1.85 K
   of the T2 bias (Exner pressure-only term) and ~1.6 K of the T2 RMSE.
2. **A GENUINE theta-side lower-troposphere physics warm bias** that d03 does NOT
   have — the GPU lower atmosphere over the open ocean runs +3 to +5 K too warm in
   the lowest ~10 model levels, with a too-shallow PBL, despite an IDENTICAL
   (corpus-refreshed) sea-surface temperature and near-zero surface-flux bias.
   Accounts for the residual ~+3.1 K theta-only term and ~2.2 K of residual T2 RMSE
   after the pressure artifact is removed.

This OVERTURNS the sprint's working hypothesis. The premise was: "d02 forces
geopotential, so it lacks the d03 free-drift mechanism; any remaining warm bias must
be theta-side physics." Reality: d02 has the pressure-Exner artifact too (so
`force_geopotential` is NOT what gates it), AND it additionally has a real theta-side
physics drift that d03's small boundary-dominated nest suppressed.

## Re-measurement (item 1): current d02 status post-P1-4a

Representative MAM case run: **case3 = 20260521_18z_l3_24h** (the validated
continuity anchor; case1 20260529 d02 corpus history is PURGED, 0 files; case2
20260509 available as a second case if needed). Full production d02 pipeline
(`_build_real_case` -> `execute_daily_pipeline`, domain=d02,
`force_geopotential=True`), 24 h, fp64, real wrfouts written, scored vs corpus L3
d02 truth.

Run: verdict PIPELINE_PARTIAL (main forecast OK; "PARTIAL" only because the
restart/repeat probes are NOT_RUN), all_finite=True, 24/24 wrfouts, wall 1451 s.

**Written-wrfout T2 (full domain), mean over 24 leads: RMSE 3.78 K, bias +3.44 K.**

| lead | T2 RMSE | T2 bias | U10 bias | V10 bias | Q2 bias | PBLH bias | PSFC bias | Ponly | THonly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1h (19Z) | 2.34 | +2.22 | -0.32 | +1.52 | ~0 | -122 | +2567 Pa | +2.12 | +1.05 |
| 6h (00Z) | 3.59 | +3.31 | +0.87 | +1.77 | +3e-4 | -127 | +2174 | +1.80 | +3.09 |
| 12h (06Z)| 3.98 | +3.69 | +0.68 | +1.48 | +4e-4 | -105 | +2247 | +1.87 | +3.49 |
| 24h (18Z)| 3.55 | +3.12 | +1.32 | +2.07 | +4e-4 | -65  | +2252 | +1.87 | +2.57 |
| MEAN     | 3.78 | +3.44 |       |       |        |       | +2227 | +1.85 | +3.10 |

vs the pre-P1-4a baseline for the SAME case3 L3
(`v010_d02_result_hfxfix_3case_VALIDATED.json`, HEAD d1c373b — the empirical HFX
z_t fix that P1-4a 0c4a4ce replaced): T2 RMSE 6h=1.88, 12h=2.14, 24h=1.11 K.

So the written-wrfout T2 RMSE roughly DOUBLED from the pre-P1-4a baseline. BUT this
comparison mixes two changes (P1-4a surface layer AND the write/score path). The
clean P1-4a-only re-measurement uses the baseline's own harness (`_advance_chunk` +
in-flight `compute_m9_diagnostics`) at the current HEAD:

**P1-4A-EFFECT (segmented `_advance_chunk` harness, current HEAD d9f452e, case3 L3,
in-flight `compute_m9_diagnostics` T2 — the baseline's own methodology):**

| lead | T2 RMSE | T2 bias | U10 RMSE/bias | V10 RMSE/bias |
|---|---:|---:|---|---|
| 6h  | 1.881 | +1.386 | 0.902 / +0.156 | 1.943 / +1.472 |
| 12h | 2.138 | +1.578 | 1.083 / +0.230 | 1.834 / +1.343 |
| 24h | 1.119 | +0.882 | 1.501 / +0.693 | 2.446 / +1.798 |

verdict=D02_VALIDATED, case passed=True. vs pre-P1-4a baseline (1.882/2.144/1.111):
**P1-4a is NEUTRAL on d02 — T2 matched to <0.01 K at every lead. The MYNN surface-
layer port neither helped nor hurt d02 T2.** Winds also unchanged.

### CRITICAL SECONDARY FINDING — the two integration paths DISAGREE

The SAME GPU model, SAME HEAD, SAME case, scored two ways, gives a 2x-different T2:

| path | T2 RMSE 6/12/24h | PSFC | verdict |
|---|---|---|---|
| `_advance_chunk` segmented harness (in-flight T2) | 1.88 / 2.14 / 1.12 | (matches corpus) | D02_VALIDATED PASS |
| `run_forecast_operational` production pipeline (written wrfout T2) | 3.59 / 3.98 / 3.55 | +2.6 kPa, +3-5K theta | the bias above |

Both T2 computations call the IDENTICAL `compute_m9_diagnostics` /
`surface_layer_diagnostics`, so the divergence is in the integrated STATE, not the
diagnostic. The difference is the integration path:
- The validation harness advances ONE continuous carry across the whole run with
  `_advance_chunk` (global step index, radiation gated by traced `step%cadence==0`).
- The production daily pipeline (and d03_replay) calls `run_forecast_operational(
  state, namelist, 1.0)` ONCE PER FORECAST HOUR — each call RE-INITIALIZES the
  operational carry (`initial_operational_carry`) with step restarting at 1, so
  radiation fires at intra-hour steps 180/360 every hour and any accumulated
  acoustic/PBL carry state is reset hourly.

The +2.6 kPa pressure inflation + lower-tropo theta warming appear on the PRODUCTION
`run_forecast_operational` per-hour path (the one that writes the operational wrfouts
AND the path `d03_replay`/`execute_daily_pipeline` use), but NOT on the validation
harness path. So: **the "D02_VALIDATED" proof was measured on a path that does NOT
match the operational wrfout-writing path. The operational product (the wrfouts a
user actually gets) is +2.6 kPa / +3.4 K T2 worse than the validated harness number
suggests.** This path discrepancy is itself a HIGH-PRIORITY bug.

Candidate mechanisms for the divergence (the two paths differ in three ways; not yet
isolated to one):
1. **Per-hour carry reset.** The pipeline calls `run_forecast_operational(state, nl,
   1.0)` once PER FORECAST HOUR; each call runs `initial_operational_carry(...)`
   fresh and restarts the global step at 1. The harness advances ONE continuous
   `_advance_chunk` carry. A per-hour reset of acoustic/integration carry state +
   intra-hour radiation timing could seed the drift.
2. **Guards on vs off.** Pipeline `_build_real_case` keeps `disable_guards=False`
   (theta-increment limiter + dry-mass floor every RK3 step ON); the harness sets
   `disable_guards=True`. (Note: guards should constrain theta MORE, so they are an
   unlikely source of EXTRA warming — but they do touch mu/theta and must be ruled
   out.)
3. **Diurnal clock.** Harness threads `time_utc` into the integrating namelist;
   `_build_real_case` does not (the clock is threaded only into the output-diagnostic
   recompute), so SWDOWN/radiation timing during integration may differ.

A focused path-equivalence check is the FIRST follow-up — it likely also explains the
d03 +2.6 kPa offset (d03 uses the same per-hour pipeline path), so fixing it may
collapse the pressure-Exner artifact on BOTH domains at once. A first attempt
(`scripts/diag/d02_path_equiv_check.py`, now fixed) is in the repo; the initial run
crashed on a `donate_argnums` buffer-reuse bug in the test harness (NOT the model) —
the script was corrected (independent device copies per path) but NOT re-run (the
two-path T2 evidence above — 3.78 K written-pipeline vs 1.12 K in-flight-harness from
the SAME `compute_m9_diagnostics` — already establishes the divergence; re-running
the 2x-heavy-compile test was not worth the GPU time for confirmation).

## Pressure-vs-physics split (item 2)

Method: validated against the d03 p1_4a wrfout — my Exner decomposition reproduces
GPT's d03 numbers exactly (PSFC +2606 Pa, Ponly +2.19 K, THonly -0.37 K), so the
method is sound.

**d02 surface pressure is NOT close to corpus — it is high by ~+2.2 to +2.6 kPa,
the SAME magnitude as d03**, despite `force_geopotential=True`:

- hour-1 P-pert bias: +2567 Pa (surface) growing to +3118 Pa aloft — a near-uniform
  full-column inflation. PB (base state) bias = 0.0 at every level. PH-pert
  (perturbation geopotential) anchored at 0 at the surface, bowed strongly LOW in
  the mid-column (k10 -806, k20 -1953, k30 -2135 m^2/s^2). This is the IDENTICAL
  signature the d03 bisection found (mid-column-depressed ph' -> al more negative ->
  EOS inflates p -> uniform +2.x kPa p' offset). It is present in the d02 operational
  path REGARDLESS of boundary geopotential forcing -> the +2.x kPa offset is a dycore
  perturbation-geopotential equilibration issue, NOT the `force_geopotential=False`
  free-drift the d03 review attributed it to.

**Decisive Exner knockout** (re-Exner the GPU T2 with corpus psfc, machine-exact, no
GPU): mean T2 RMSE collapses **3.78 K -> 2.16 K**. The pressure artifact removes
~1.62 K of RMSE; a **2.16 K PHYSICS residual remains** (hour1 0.55 K -> hour24
2.02 K, GROWING). Contrast d03, where the same knockout collapsed T2 to ~0.9 K
(pressure was ~all of it). In d02 the pressure artifact is only HALF.

VERDICT: **d02 T2 warm bias = pressure-Exner artifact (~half) + genuine theta-side
physics (~half, and growing with lead).** Not pressure-only (unlike d03), not
theta-only.

## The theta-side component localized (item 3)

The theta-only term is +3.1 K mean and is a LOWER-TROPOSPHERE COLUMN warming, not a
surface-diagnostic artifact:

theta-perturbation bias by model level (domain mean), over time:
```
lead            k0     k1     k3     k5     k10    k20    k30
hour1 (19Z)   +1.04  +1.46  +1.71  +1.67  +1.02  +0.39  +1.67
hour6 (00Z)   +3.08  +4.03  +5.15  +5.36  +2.54  +0.63  +2.34
hour16(10Z)   +3.64  +4.35  +5.21  +5.01  +2.53  +0.55  +2.30
hour24(18Z)   +2.56  +3.24  +4.01  +3.61  +1.67  +1.15  +0.82
```
The warming peaks at k3-k5 (~+5 K, a few hundred m), decays with height, and builds
in the first ~6 h then stays. It is concentrated in the PBL/lower troposphere.

Surface energy budget (GPU - corpus, land/sea x day/night aggregate over 24 h):
```
region      T2_K   TSK_K   HFX     LH      GLW    SW     PBLH    Q2(kg/kg)
land_all   +4.89   0.000   -21.6   +80.6   +27.2  +9.1   +8.6    -6e-4
land_day   +5.99   0.000   -31.2   +147.1  +27.7  +17.0  -5.1    -5e-4
sea_all    +3.49   0.000   -37.1   -40.7   +14.7  +37.7  -115.8  +5e-4
sea_night  +3.52   0.000   -36.4   -42.5   +17.0  -0.2   -136.1  +4e-4
```
Localization reading:
- **TSK bias = 0.000** everywhere (corpus-refreshed; no skin-temperature solve). The
  land-surface / Noah path is EXONERATED, same as d03.
- **HFX is LOW (negative), not high** (-22 land, -37 sea). NOT a sensible-heat
  over-flux. (The pre-P1-4a "HFX over-flux" story does not apply post-P1-4a.)
- **The dominant signal is over SEA (93% of the d02 grid is ocean).** Over the open
  ocean with IDENTICAL SST, the GPU lowest levels are +2.9 to +5.4 K too warm and the
  **PBL is too SHALLOW (sea PBLH -116 to -136 m; 280 m GPU vs 421 m corpus at hour6).**
  HFX/LH are ~slightly negative there, so this is NOT surface-flux driven.
- **GLW (downward longwave) is HIGH (+15-27 W/m^2)** everywhere, consistent with a
  warmer/moister lower column (a consequence/amplifier, likely not the seed).

Most likely responsible component: a **PBL / lower-troposphere thermal-mixing /
ventilation deficit** (MYNN PBL + the surface-layer coupling), NOT the surface flux
itself and NOT the LSM. The signature — a too-warm lowest few hundred metres trapped
under a too-shallow PBL over an ocean with correct SST and ~zero net surface-flux
bias — is a classic under-mixed / too-stable PBL that fails to vent a warm anomaly.
The warm anomaly's SEED is partly the +2.x kPa pressure inflation itself (which
warms the actual lower-air temperature through the EOS at hour 1, before any PBL
feedback), then the shallow PBL keeps it in. So the two causes are coupled: the
pressure/geopotential dycore error warms the lower air, and the PBL fails to mix it
out, letting theta drift up over the first 6 h.

## Why d02 differs from d03 (resolves the apparent paradox)

Both domains share the same surface-layer, PBL, and dycore code, and both have the
+2.x kPa pressure-Exner artifact. The difference in the THETA term is geometric:
d03 is a small 1 km Tenerife nest (93x75) whose interior is dominated by the forced
d02 boundary theta (u/v/w/theta/qv/mu/p ARE forced; only ph' is free) — the boundary
pins the interior theta, so the theta-only residual is ~0. d02 is a large 3 km domain
(159x66, ~93% open ocean) whose interior is many cells from the boundary, so the
interior lower-troposphere theta is free to drift under the PBL/dynamics, and it does
(+3-5 K). force_geopotential is a red herring for BOTH the pressure artifact (present
in d02 anyway) and the theta drift (a domain-size/interior-freedom effect).

## Proposed fix direction

Two coupled levers; both are upstream of the surface similarity functions (so they
are NOT a `surface_layer.py` fix — P1-4a's MYNN port is fine, it cannot move either):

1. **Pressure-Exner artifact (shared with d03).** The +2.x kPa near-uniform p'
   offset traces to the mid-column-depressed perturbation geopotential during the
   acoustic integration, present even with `force_geopotential=True`. This is a
   DYCORE perturbation-geopotential / pressure-diagnosis issue (acoustic_wrf
   `diagnose_pressure_al_alt` is a faithful EOS; the error is in the equilibrated
   ph' that feeds `al`). It is the SAME defect the d03 work is chasing; fixing it
   benefits d02 too. The cheap interim (re-reference the diagnostic surface pressure
   that T2's Exner uses to a hydrostatically-consistent psfc) removes ~1.6 K of d02
   T2 RMSE without touching the dycore — same stopgap proposed for d03 — but d02
   still has the bigger physics residual after that.
   (Owner: dynamics/runtime; out of this diag's scope.)

2. **The theta-side physics drift (d02-specific, the LARGER residual).** Target the
   PBL ventilation / lower-troposphere mixing over the ocean: the GPU PBL is too
   shallow (sea PBLH ~-130 m) and the lowest few hundred metres trap a +5 K warm
   anomaly. Audit the MYNN PBL mixing-length / TKE / entrainment over weakly-unstable
   sea columns vs WRF, and the surface-layer -> PBL flux handoff (theta_flux/fltv).
   The fact that the warming builds in the first 6 h then saturates suggests an
   equilibrium mixing deficit, not a runaway. This needs a PBL-parity oracle (MYNN
   `module_bl_mynn.F`) against a WRF column, analogous to the P1-4a surface-layer
   oracle. RECOMMEND building that PBL oracle next — it is the gating item for the
   d02 TOST-equivalence path and is currently the larger of the two T2-bias halves.

This determines the v0.2.0 TOST path for the primary product: **the d02 T2 gate
needs BOTH the shared dycore pressure-Exner fix AND a new PBL/lower-troposphere
mixing fix.** The surface-layer (P1-4a) is not the lever for either.

## Files

- `scripts/diag/d02_t2bias_diagnosis.py` — NEW. Two-phase (GPU run via the production
  d02 pipeline writing wrfouts; CPU score + Exner pressure/theta decomposition +
  land/sea day/night energy-budget table). Method validated against d03 p1_4a.
- `proofs/v010_validation/d02_t2bias_diag_case3.json` — per-lead + aggregate decomposition.
- `proofs/v010_validation/pipeline_run_d02_diag_case3.json` — the d02 24h run object.
- GPU wrfouts: `/tmp/v010_d02_diag_runs/d02_20260521_18z_l3_24h_20260522T133443Z_case3/`.

## Commands run

- d02 24h production run: `python scripts/diag/d02_t2bias_diagnosis.py run --run-id
  20260521_18z_l3_24h_20260522T133443Z --run-root .../wrf_l3 --hours 24 --tag case3`
  (detached `systemd-run --user --scope -p AllowedCPUs=0-3`, MEM_FRACTION 0.80,
  taskset 0-3).
- decomposition: `python scripts/diag/d02_t2bias_diagnosis.py score --gpu-dir ... --run-id ...`.
- Exner knockout + vertical-profile + budget: inline CPU-pinned python (OMP=4, taskset 0-3).
- P1-4a-effect harness: `python proofs/v010_validation/v010_d02_validate.py --execute
  --cases case3 --leads 6 12 24 --segment-steps 180` (current HEAD).
