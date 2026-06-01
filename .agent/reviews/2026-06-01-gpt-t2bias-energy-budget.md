# GPT-5.5 independent RCA: d03 p1_4a T2 warm bias

Date: 2026-06-01
Branch/worktree: `worker/opus/final-verdict`, `/home/enric/src/wrf_gpu2`
Mode: read-only analysis of existing wrfout/proof data; no GPU/model run launched.

## Verdict

The leading hypothesis is refuted by the saved p1_4a data: the GPU `TSK` field is exactly the
corpus `TSK` at every compared d03 cell-hour (`max_abs(TSK_gpu - TSK_cpu) = 0.0 K` over 167,400
cell-hours). The warm bias is therefore not caused by a warm Noah/LSM skin temperature in the
available p1_4a wrfout.

The single most likely proximate cause is a lower-atmosphere pressure/Exner-state error from the
dry dynamics / pressure-geopotential / nested-boundary state, upstream of the surface energy
budget. GPU perturbation pressure is too high by about 2.4-3.2 kPa through the column, with
`PSFC` too high by about 2.1-2.7 kPa. That pressure bias alone explains about 2.2-2.4 K of
lowest-model-level actual temperature warming, while potential-temperature bias is small
(land about +0.36 K, sea about -0.29 K over the 24 h aggregate).

This pressure/Exner-warm lower-air state exists at the first forecast output and over sea/night,
before the daytime land HFX excess can be causal.

## Inputs and paths

Confirmed from `scripts/d03_replay.py` and `proofs/v010_validation/pipeline_run_d03_p1_4a.json`:

- GPU p1_4a d03 wrfout:
  `/tmp/v010_d03_runs/d03_20260521_18z_l3_24h_20260522T133443Z_p1_4a/wrfout_d03_2026-05-21_19:00:00`
  through `wrfout_d03_2026-05-22_18:00:00`
- CPU-WRF corpus truth:
  `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfout_d03_*`
- Scored proof:
  `proofs/v010_validation/d03_validation_p1_4a.json`
- Final scored T2 proof result:
  `RMSE = 1.95885 K`, `mean_error = +1.52266 K`, final lead `2026-05-22T18:00:00Z`

Masks/method:

- Land/sea from corpus `LANDMASK > 0.5`; outer boundary excluded for land/sea decomposition.
- Interior land cells: 2,035. Interior sea cells: 4,608.
- Day/night defined per cell from corpus `SWDOWN`: day `> 50 W m-2`, night `<= 10 W m-2`.
- Actual model-level temperature computed as `T_actual = (T + 300) * ((P + PB)/100000)^(287/1004)`.

## Surface energy budget decomposition

GPU wrfout has `TSK`, `HFX`, `LH`, `GLW`, `SWDOWN`, `T2`, `Q2`, `PBLH`, but does not emit GPU
`TSLB`, `SMOIS`, or `GRDFLX`. Those three cannot be directly scored from the existing GPU wrfout.

Aggregate differences are GPU minus corpus:

| Mask | T2 K | TSK K | HFX W/m2 | HFX ratio | LH W/m2 | GLW W/m2 | SWDOWN W/m2 | PBLH m | Q2 kg/kg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Land, all hours | +1.86 | +0.00 | +86.9 | n/a | +104.7 | +23.0 | -31.9 | +27.5 | +0.00128 |
| Land, night | +0.81 | +0.00 | -7.5 | n/a | +1.9 | +16.1 | -0.4 | +59.1 | +0.00129 |
| Land, day | +2.75 | +0.00 | +166.9 | 1.57x | +191.6 | +28.9 | -58.6 | +0.8 | +0.00127 |
| Land, 10-15Z | +3.12 | +0.00 | +267.1 | 1.61x | +302.2 | +23.5 | -28.0 | -23.6 | +0.00079 |
| Sea, all hours | +1.46 | +0.00 | -13.8 | n/a | -24.4 | +15.3 | +7.1 | -7.9 | +0.00052 |
| Sea, night | +1.48 | +0.00 | -13.4 | n/a | -29.8 | +15.0 | -0.3 | -38.8 | +0.00066 |

Key points:

- The old "midday HFX about 3.7x corpus" result is no longer present after p1_4a. Current land
  daytime HFX is still high, but only about 1.57x over all daylight and about 1.61x for 10-15Z.
  The max hourly land mean ratio is about 1.67x at 14-15Z.
- The bias is not daytime-only. Sea T2 is about +1.46 K over all hours and +1.48 K at night.
  Land T2 has a persistent night component (+0.81 K) plus daytime amplification (+2.75 to +3.12 K).
- Land LH is much more excessive than HFX: midday LH is about 15x corpus, and Q2 is moist-biased.
  That refutes a "soil too dry / Bowen ratio too high" explanation. The GPU Bowen ratio is lower,
  not higher, than corpus over daytime land.
- PBLH is not a credible primary cause: land day mean PBLH bias is only +0.8 m and 10-15Z is
  -23.6 m, while sea/night biases persist.

## First divergence

At the first compared lead (`2026-05-21 19:00Z`), the surface energy terms do not support a
land-surface heating cause:

| Region | T2 K | TSK K | HFX W/m2 | LH W/m2 | GLW W/m2 | SWDOWN W/m2 | PBLH m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Land | +2.59 | +0.00 | -33.0 | +27.6 | +11.3 | -55.7 | -73.6 |
| Sea | +1.52 | +0.00 | -15.1 | -32.1 | +11.9 | -54.1 | -100.4 |

The first large state divergence is already in the lower atmospheric thermodynamic/pressure state,
not in `TSK` or a positive HFX excess.

## Pressure/Exner decomposition

Lowest-model-level actual temperature bias decomposes into a pressure-only contribution plus a
potential-temperature-only contribution:

| Region/mask | T0 actual K | Pressure-only K | Theta-only K | T2 K | PSFC Pa | bottom P+PB Pa | theta0 K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Land, first lead 19Z | +2.85 | +2.31 | +0.53 | +2.59 | +2312 | +2590 | +0.55 |
| Sea, first lead 19Z | +2.05 | +2.42 | -0.37 | +1.52 | +2660 | +2969 | -0.37 |
| Land, all hours | +2.58 | +2.22 | +0.36 | +1.86 | +2205 | +2481 | +0.39 |
| Sea, all hours | +2.05 | +2.34 | -0.29 | +1.46 | +2559 | +2865 | -0.29 |
| Land, day | +2.55 | +2.20 | +0.35 | +2.75 | +2179 | +2454 | +0.37 |
| Sea, night | +2.03 | +2.37 | -0.33 | +1.48 | +2590 | +2898 | -0.33 |

At 12Z, the pressure error is in `P`, not `PB`:

- Land bottom level: `P` bias `+2398.8 Pa`, `PB` bias approximately `0 Pa`.
- Sea bottom level: `P` bias `+2778.3 Pa`, `PB` bias `-0.04 Pa`.
- The positive `P` bias persists upward, reaching about `+3.1 to +3.2 kPa` around levels 20-40.

This is sufficient to explain the persistent all-domain T2 warm bias through the Exner factor,
especially over sea where LSM/soil cannot be the cause.

## Alternatives ruled in/out

- Noah/LSM skin temperature: ruled out for this p1_4a wrfout. `TSK` is exactly corpus at every
  compared grid point and hour.
- Soil temperature/moisture and ground heat flux: not directly scoreable because GPU wrfout does
  not emit `TSLB`, `SMOIS`, or `GRDFLX`. However, a warm skin/soil-memory cause is inconsistent
  with `TSK` equality and the sea/night bias.
- Downward longwave: GPU `GLW` is high (+15 to +29 W/m2), but the warm T2 and pressure/Exner bias
  are already present at the first lead and over sea/night. GLW is more likely a consequence or
  amplifier of the warm/moist column than the proximate cause.
- Shortwave/radiation imbalance: SWDOWN is often lower over land in the afternoon and is already
  lower at the first lead, so it cannot explain the warm bias.
- Soil too dry / high Bowen ratio: not supported. Land LH is massively high (about 15x corpus at
  midday) and Q2 is moist-biased.
- PBL under-mixing: not supported as the single cause. PBLH errors are mixed-sign and small during
  land daytime relative to the T2 bias; sea/night T2 remains warm.

## Most likely responsible component

Not `module_sf_mynn` similarity functions and not Noah skin-temperature solve. The responsible
WRF-equivalent component is the dry dynamical pressure/geopotential/mass state feeding surface
diagnostics: `solve_em` / split-explicit pressure update / nested-boundary pressure-geopotential
coupling. The data specifically points at perturbation pressure `P` and Exner conversion, with
secondary land daytime amplification from still-excessive turbulent flux diagnostics.

## Decisive knockout experiment

Run one pressure/Exner transplant diagnostic or replay:

Keep GPU winds, moisture, theta, `TSK`, radiation, and surface-layer code unchanged, but substitute
corpus `P/PB/PH/PSFC` or the derived Exner state into the lowest-column surface-diagnostic path
for the saved p1_4a hours, then re-score `T2/HFX/LH/Q2`.

Prediction if this RCA is correct:

- Sea T2 mean bias should collapse from about `+1.46 K` toward the theta-only residual
  (near `-0.3 K` to `0 K` depending on the diagnostic formula).
- Land all-hour T2 should drop by roughly `1.5-2 K`; remaining land daytime bias, if any, should
  track the residual HFX/LH surface coupling rather than the persistent sea/night bias.
- No change to `TSK` or soil fields is needed to get that collapse.

If the transplant does not materially reduce sea/night T2, then the pressure/Exner interpretation
is wrong and the next suspect becomes the T2 diagnostic/interpolation path itself.

## Commands run

- Read governance and local instructions:
  `sed -n '1,220p' PROJECT_CONSTITUTION.md`, `sed -n '1,220p' AGENTS.md`.
- Read local skills:
  `.agent/skills/validating-physics/SKILL.md`,
  `.agent/skills/conducting-blind-review/SKILL.md`.
- Read sprint/spec context:
  `.agent/sprints/2026-05-29-f7-acoustic-core/sprint-contract.md`,
  `.agent/sprints/2026-05-29-f7-sprint-c/sprint-contract.md`,
  `.agent/decisions/P1-4a-MYNN-PARITY-SPEC.md`.
- Confirm d03 paths and scoring:
  `sed -n '1,520p' scripts/d03_replay.py`,
  `proofs/v010_validation/d03_validation_p1_4a.json`,
  `proofs/v010_validation/pipeline_run_d03_p1_4a.json`.
- NetCDF variable inventory and all quantitative decomposition with:
  `OMP_NUM_THREADS=4 taskset -c 0-3 python - <<'PY' ...`.

Temporary proof extracts written outside the repo:

- `/tmp/gpt_t2bias_records.json`
- `/tmp/gpt_t2bias_column_rows.json`

## Handoff

Objective: identify the proximate physical cause of the persistent d03 p1_4a +T2 bias from existing
data only.

Files changed: `.agent/reviews/2026-06-01-gpt-t2bias-energy-budget.md`.

Commands run: listed above; all analysis Python was CPU-pinned with `OMP_NUM_THREADS=4 taskset -c
0-3`. No GPU/model job was launched.

Proof objects produced: this report plus `/tmp/gpt_t2bias_records.json` and
`/tmp/gpt_t2bias_column_rows.json`.

Unresolved risks: GPU wrfout lacks `TSLB`, `SMOIS`, and `GRDFLX`, so soil and ground-heat-flux
terms cannot be directly scored. The pressure error may originate in runtime dynamics or in wrfout
diagnostic mapping; the proposed Exner transplant distinguishes those paths.

Next decision needed: run the pressure/Exner transplant knockout before spending more effort on
Noah/LSM skin-temperature or MYNN similarity changes.
