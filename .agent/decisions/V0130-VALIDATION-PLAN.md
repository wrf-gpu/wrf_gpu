# V0.13.0 Validation Plan - 3h Gate

Date: 2026-06-08
Owner: GPT-5.5 xhigh validation architect
Branch: `worker/gpt/v013-valplan`

## Positioning

This is a validation plan for a fast, GPU-native, GPU-scalable
WRF-compatible model. It does not claim a bit-true Fortran port and does not
claim a perfectly efficient rewrite.

The primary proof target is RUNS-confidence: implemented couplings run stably,
finish inside the intended resource envelope, and produce finite fields with no
NaNs or crashes.

The secondary proof target is gate-keeper equivalence: CPU-WRF and AEMET
comparisons are collected honestly and reported with the existing scorer. A
3h gate can prove the comparison path works and can produce rough equivalence
evidence, but it cannot close powered TOST.

## Assets And Current Truth

- Canary L2 9/3 km corpus: `/mnt/data/canairy_meteo/runs/wrf_l2`
- Pairable Canary L2 CPU-WRF truth: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`
  has the current `n=15` powered-TOST corpus, each with retained d01/d02
  wrfout through 72h.
- Canary L3 9/3/1 km corpus: `/mnt/data/canairy_meteo/runs/wrf_l3`
  has retained 24h pairable examples including
  `20260509_18z_l3_24h_20260511T190519Z` and
  `20260521_18z_l3_24h_20260522T133443Z`.
- AEMET station data:
  `/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations`.
- Pristine CPU-WRF oracle tree: `/home/enric/src/wrf_pristine/WRF`.
- Switzerland exists on disk but is not used in this 3h gate:
  `/mnt/data/wrf_gpu_switzerland_big/run_cpu` and
  `/mnt/data/wrf_gpu_switzerland_128/run_cpu` contain 24h CPU truth.

Known limit: 9/3/1 km plus GWD plus 2-way feedback for 24h is 32GB-VRAM
marginal and has OOMed around hour 14. This plan therefore runs a 9/3/1 km
2-way finite slice, not a false 24h claim.

## Common Setup

Create a unique output root before the run:

```bash
export OUT=/mnt/data/wrf_gpu_validation/v0130_plan_a_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
```

GPU jobs are serial. CPU work may run in parallel on cores 0-23, leaving the
remaining cores free. CPU-only commands must force `JAX_PLATFORMS=cpu`.
Use the versioned GPU wrapper `scripts/run_gpu_lowprio.sh` for all GPU jobs;
do not use a helper under `/tmp`. The powered TOST campaign can also be launched
through `scripts/run_powered_tost_n15.sh --detach --resume`, which writes
durable log/rc/runinfo files under `/mnt/data/wrf_gpu_validation/v0130_marathon`
by default. See `docs/GPU_RUNBOOK.md`.

## Tests

### A1 - 24h Canary L2 9/3 km, GWD, 2-Way Feedback

Type: PRIMARY RUNS-confidence, limited SECONDARY equivalence support

Resource: GPU, one serial job

Estimate: 45 min

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_GWD_NESTED=1 \
  python -m gpuwrf.cli run \
    --input-dir /mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z \
    --output-dir "$OUT/a1_canary_l2_24h_feedback_gwd" \
    --max-dom 2 \
    --hours 24 \
    --feedback \
    --proof-dir "$OUT/a1_canary_l2_24h_feedback_gwd/proofs" \
    --score
```

Pass criterion:

- Command exits 0.
- Proof reports pipeline green or equivalent success status.
- d01 and d02 each emit 24 hourly wrfout frames.
- T2, U10, V10, PSFC, QVAPOR, RAINNC, and key 3D wind/temperature fields are
  finite for every emitted frame.
- No OOM, NaN, crash, or host/device transfer regression is reported.

What it proves:

- The heaviest v0.13 configuration known to fit on the workstation runs for a
  full operational day with GWD and 2-way feedback.
- It is the highest-value RUNS-confidence gate in this plan.

### A2 - 6h Canary L3 9/3/1 km, GWD, 2-Way Feedback Slice

Type: PRIMARY RUNS-confidence

Resource: GPU, one serial job after A1

Estimate: 35 min

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_GWD_NESTED=1 \
  python -m gpuwrf.cli run \
    --input-dir /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z \
    --output-dir "$OUT/a2_canary_l3_6h_feedback_gwd" \
    --max-dom 3 \
    --hours 6 \
    --feedback \
    --proof-dir "$OUT/a2_canary_l3_6h_feedback_gwd/proofs" \
    --score
```

Pass criterion:

- Command exits 0.
- d01, d02, and d03 each emit 6 hourly wrfout frames.
- All core prognostic and diagnostic fields are finite.
- No OOM or compilation failure occurs.

What it proves:

- The maximum nesting depth exercises 1 km state, GWD, boundary coupling, and
  2-way feedback together through a bounded horizon that should fit in 32GB
  VRAM.
- It does not prove 24h 9/3/1 km plus 2-way plus GWD viability.

### A3 - Single-Case Powered TOST And AEMET Scorer Smoke

Type: SECONDARY gate-keeper equivalence, PRIMARY nested-run smoke

Resource: GPU, one serial job after A2; CPU scoring inside the runner

Estimate: 50 min

Command:

```bash
scripts/run_gpu_lowprio.sh --cores 0-23 -- env \
  PYTHONPATH=src \
  JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  GPUWRF_AEMET_ROOT=/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations \
  python proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py \
    --case 20260429_18z_l2_72h_20260524T204451Z \
    --allow-single \
    --resume
```

Pass criterion:

- Runner exits 0.
- It uses the fixed nested `max_dom=2` path and does not request impossible
  `wrfbdy_d02`.
- CPU-WRF, GPU-WRF, and AEMET pair counts are greater than 0 for T2, U10, and
  V10.
- The case report contains finite RMSE deltas, confidence inputs, and station
  pair metadata.

What it proves:

- The gate-keeper evidence pipeline is live: nested GPU forecast, retained
  CPU-WRF truth, and AEMET pairing can be joined for a real Canary case.
- It is a smoke of the powered TOST scorer, not a powered equivalence result.

### A4 - Community Dycore, Conservation, And Restart Gate

Type: PRIMARY RUNS-confidence and SECONDARY numerical-equivalence support

Resource: CPU, parallel lane

Estimate: 4 min

Command:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  PYTHONPATH=src \
  bash scripts/community_validation.sh
```

Pass criterion:

- `proofs/v013/community_validation.json` is produced.
- Straka density current and Skamarock warm bubble complete and stay finite.
- Conservation budgets pass the documented tolerances.
- Restart bit-identity passes.

What it proves:

- The cheap analytic and idealized gates that reviewers expect remain green
  while operational runs are being tested.

### A5 - Operational Physics-Suite Coupler Smoke

Type: PRIMARY RUNS-confidence

Resource: CPU, parallel lane

Estimate: 15 min

Command:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  PYTHONPATH=src \
  python proofs/v060/multicfg_operational_smoke.py \
    --steps 8 \
    --out "$OUT/a5_multicfg_operational_smoke.json"
```

Pass criterion:

- Report exits with all operational configurations passing.
- Every covered implemented scheme emits finite outputs through the operational
  coupler path.
- Unsupported or reference-only schemes are explicitly fail-closed rather than
  silently falling back.

Scheme coverage target:

- Microphysics: Thompson, WSM6, WDM6, Morrison, Kessler, Lin, WSM3, WSM5.
- PBL: MYNN, MYJ, YSU, ACM2, BouLac, MRF where wired.
- Surface layer: sfclayrev, MYNN, Janjic, GFS, old-MM5 where wired.
- LSM: Noah-MP and Noah-classic.
- Cumulus: KF, BMJ, GF, Tiedtke where wired.
- Radiation: RRTMG SW/LW through the operational smoke matrix.

What it proves:

- The implemented physics families are exercised through operational coupler
  code, not only isolated oracle scripts.
- This is intentionally short. It is a stability and wiring gate, not a full
  forecast-skill gate.

### A6 - V0.13 Focused Operational Wiring Tests

Type: PRIMARY RUNS-confidence

Resource: CPU, parallel lane

Estimate: 10 min

Command:

```bash
taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  PYTHONPATH=src \
  python -m pytest -q \
    tests/test_v013_myj_janjic_operational.py \
    tests/test_v013_mrf_operational.py \
    tests/test_v013_t3_surface_lsm_wiring.py \
    tests/test_v013_ra_sw_gsfc.py \
    tests/test_v060_ra_sw_dudhia.py \
    tests/test_rrtm_lw_operational_wiring.py \
    tests/test_gwd_operational_wiring.py \
    tests/test_v0110_boundary_feedback.py \
    tests/test_p0_1a_nesting.py
```

Pass criterion:

- Pytest exits 0.
- New or recently touched operational wiring for MYJ/Janjic, MRF, GFS/old-MM5,
  GSFC SW, Dudhia SW, RRTM LW, GWD, nesting, and feedback stays green.

What it proves:

- The v0.13 wiring points that are most likely to break operational runs are
  checked directly.

### A7 - Reference, Fail-Closed, And Fake-Mesh Proof Sweep

Type: PRIMARY RUNS-confidence and SECONDARY reviewer-support evidence

Resource: CPU, parallel lane

Estimate: 25 min if sequential; may be split across CPU lanes

Command:

```bash
taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_microphysics_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/myj_janjic_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/mrf_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_surface_lsm_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_radiation_oracle.py

taskset -c 0-23 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src \
  python proofs/v013/t3_cumulus_oracle.py

taskset -c 0-23 env \
  JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=true \
  XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  PYTHONPATH=src \
  python proofs/v013/multigpu_fakemesh.py
```

Pass criterion:

- Every script exits 0.
- WSM7-ref, MRF, MYJ/Janjic, GFS/old-MM5, GSFC, and cumulus carry-over status
  is recorded honestly as operational, reference-only, or fail-closed.
- Fake-mesh reports partition-invariant bit-identity.

What it proves:

- The reference and fail-closed side of the implemented suite is not drifting.
- Multi-GPU decomposition math is deterministic on fake mesh, without claiming
  real multi-GPU throughput.

## 3h Budget

GPU is serial. CPU lanes run concurrently and do not extend wall time if started
when A1 starts.

| Lane | Time window | Test | Estimate | Notes |
| --- | ---: | --- | ---: | --- |
| GPU | 00:00-00:45 | A1 L2 24h GWD 2-way | 45 min | Full-day heaviest fitting operational run |
| GPU | 00:45-01:20 | A2 L3 6h GWD 2-way | 35 min | 1 km finite slice |
| GPU | 01:20-02:10 | A3 single-case scorer smoke | 50 min | CPU-WRF plus AEMET path |
| GPU | 02:10-03:00 | Slack/retry/log packaging | 50 min | Reserved for compile variance or one failed retry |
| CPU | 00:00-00:04 | A4 community validation | 4 min | Can run in parallel |
| CPU | 00:00-00:15 | A5 multicfg operational smoke | 15 min | Can run in parallel |
| CPU | 00:15-00:25 | A6 wiring pytest | 10 min | Can run after A5 or parallel on separate lane |
| CPU | 00:25-00:50 | A7 oracle/fake-mesh sweep | 25 min | Can be split across CPU lanes |

Planned critical-path wall time: 2h10.

Budgeted wall time with explicit slack: 3h00.

Mandatory test count: 7.

## Acceptance Summary

The 3h gate passes only if:

- A1, A2, A4, A5, A6, and A7 pass their RUNS-confidence criteria.
- A3 completes and produces finite CPU-WRF/GPU-WRF/AEMET scoring artifacts for
  at least one pairable case.
- Any failed or not-equivalent secondary scorer result is reported as data, not
  hidden behind a pass/fail simplification.

The highest-value test is A1: it is the full-day, fitting nested operational
configuration with GWD and 2-way feedback.
