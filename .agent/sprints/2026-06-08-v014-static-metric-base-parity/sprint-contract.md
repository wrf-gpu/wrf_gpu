# Sprint Contract: V0.14 Static Metric/Base-State Parity

Date: 2026-06-08
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Prove and, if the proof isolates a narrow bug, fix the CPU-WRF-vs-GPU mismatch
in static grid, vertical-coordinate, and base-state fields before any dycore,
radiation, FP32, Switzerland, or powered TOST campaign resumes.

The first target is not forecast skill. The target is a falsifiable answer to:
does the GPU runtime consume and emit the same static/metric/base payload as the
CPU-WRF truth before dynamics can diverge?

## Priority Context

Current project priority order:

1. Grid-cell parity across all comparable wrfout fields.
2. FP32 acoustic / mixed precision.
3. Remaining memory issues.
4. Powered TOST, only after the grid fields are no longer radically divergent.

TOST is paused after 3 durable cases. Do not resume it in this sprint.

## Evidence Already Available

- `proofs/v014/grid_cell_envelope.json`
- `proofs/v014/grid_cell_envelope.md`
- `.agent/reviews/2026-06-08-v014-grid-parity-attribution.md`
- `proofs/v014/wind_mass_divergence_probe.json`
- `proofs/v014/wind_mass_divergence_probe.md`

Current findings:

- Case 3 has 31 non-exact static/grid fields in emitted wrfouts.
- Largest static mismatches are `C2H/C2F` max 95,000 Pa, `C4H/C4F`
  approximately 26.7 kPa, `RDN` max 161.7, and `HGT` max 228 m.
- Dynamic divergence is broad: PSFC RMSE 525 Pa, PH 336 m2/s2, MU 274 Pa,
  P 228 Pa, U/V 4.61/5.83 m/s, U10/V10 2.07/2.52 m/s.
- Wind/Mass probe disfavors a pure 10 m diagnostic bug and a boundary-frame-only
  explanation.

## Scope

Allowed primary write scope:

- `proofs/v014/static_metric_base_parity.py`
- `proofs/v014/static_metric_base_parity.json`
- `proofs/v014/static_metric_base_parity.md`
- `.agent/reviews/2026-06-08-v014-static-metric-base-parity.md`

Allowed source write scope only after the proof isolates a narrow bug:

- `src/gpuwrf/init/real_init/vertical_coord.py`
- `src/gpuwrf/dynamics/metrics.py`

Read-only unless the manager approves a follow-up contract:

- `src/gpuwrf/contracts/grid.py`
- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/runtime/operational_mode.py`
- pressure-gradient, acoustic, diffusion, radiation, and surface-layer code

## Required Work Products

1. Static/base parity probe:
   `proofs/v014/static_metric_base_parity.py`
2. Machine report:
   `proofs/v014/static_metric_base_parity.json`
3. Human report:
   `proofs/v014/static_metric_base_parity.md`
4. Sprint review:
   `.agent/reviews/2026-06-08-v014-static-metric-base-parity.md`

## Required Questions

The probe must answer, separately:

- CPU wrfinput/wrfout static payload: are CPU truth wrfinput and first wrfout
  internally consistent for `XLAT/XLONG/HGT/MAPFAC*/F/E/SINALPHA/COSALPHA`,
  `ZNU/ZNW/DN/DNW/RDN/RDNW/FNM/FNP`, `C1*/C2*/C3*/C4*`, `P_TOP`,
  `PB/PHB/MUB`, `RDX/RDY`, `LANDMASK`, and `LU_INDEX`?
- GPU input payload: do GPU native-init inputs match CPU wrfinput for the same
  fields before any forecast step?
- GPU emitted payload: does GPU wrfout h1 reproduce the GPU in-memory/input
  payload, or is the mismatch introduced by writer mapping/reconstruction?
- Runtime metrics: do `DycoreMetrics` and `GridSpec.metrics` contain the CPU
  values for the vertical-coordinate and horizontal metric fields?
- Base-state fields: are `PB/PHB/MUB` mismatches true input/base-state
  mismatches, writer reconstruction artifacts, or forecast-step changes?

## Commands

CPU-only default:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
  python proofs/v014/static_metric_base_parity.py
python -m json.tool proofs/v014/static_metric_base_parity.json \
  >/tmp/static_metric_base_parity.validated.json
```

Optional short GPU smoke is allowed only if the CPU proof requires a fresh
zero-step or one-frame GPU writer artifact and no other GPU validation is
running. Use `scripts/run_gpu_lowprio.sh`; do not resume TOST.

## Acceptance Criteria

- The script runs CPU-only against current retained artifacts.
- The report separates input mismatch, runtime metric mismatch, writer mismatch,
  and dynamic forecast-step mismatch.
- Every static/base field named above is either exact, within a predeclared
  tolerance with a dtype reason, or listed as a blocker with exact max/RMSE/bias.
- If source is changed, fp64 default behavior is tested with:
  - the new static/base parity probe,
  - `proofs/v014/grid_cell_envelope.py`,
  - a focused import/compile gate for the touched modules.
- No dynamic operator fix starts until static metric/base payload is exact or
  explicitly root-caused as a harmless writer-only artifact.

## Sidecar Work

A separate read-only sidecar may design or prototype same-state tendency
localization, but it must not edit source or consume the GPU. Its output should
be a plan/probe under `proofs/v014/` and `.agent/reviews/`.

## Closeout

Close with:

- commands run
- proof objects produced
- exact root cause or narrowed suspect list
- whether source was changed
- whether same-state tendency localization is now warranted
- memory-patch recommendation status
