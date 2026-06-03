# v0.4.0 — 24h native-init FORECAST GATE (h2+ policy) — MILESTONE VERDICT

Worker: Opus 4.8 (MAX). Branch: `worker/opus/v040-fcstgate`
(worktree `/home/enric/src/wrf_gpu2/.claude/worktrees/v040-fcstgate-opus`),
parent `e17f27c` (= full v0.4.0 native-init + all V10 fixes). Date: 2026-06-03.

## Objective

Run the milestone-closing falsifiable proof: native real-init -> 24h GPU forecast
with the NATIVE wrfbdy as the ONLY LBC (no CPU-WRF replay) -> per-lead full-field +
core-surface comparison vs the CPU-WRF wrfout, scored under the principal's binding
**h2+ core-skill policy** (council-proven first-hour cold-start spin-up accepted as
inherent; core PASS judged from h2+; h1 reported non-blocking; **margins
UNCHANGED**). Verdict: does the native-init standalone forecast match CPU-WRF within
the gate thresholds across >=6 cases?

## VERDICT — FAIL (honest). Foundation NOT confirmed. STOP + reopen.

**`verdict = STABLE_BUT_CORE_FIELD_MISMATCH`, `foundation_confirmed = False`.**

The native-init forecast is **dynamically STABLE and PHYSICAL for the full 24h** in
every case (no blowup, no hour-0 detonation — the classic native-init failure mode
did NOT occur). But the standalone forecast **systematically DRIFTS away from
CPU-WRF over the forecast**, and the drift is **NOT a first-hour spin-up** — it is a
slow, monotonic/diurnal divergence that grows across leads h2..h24. The h2+ policy
therefore does NOT (and must NOT) rescue it: this miss is in the body of the
forecast, not in h1. This is exactly the task's defined STOP condition
("systematically misses h2+ (not just h1) -> STOP + report honestly; that reopens
the foundation, not masks it").

This is a TRUE finding reported without any clamp/mask/tol-loosening. The frozen
ADR-029 continuous-gate margins are unchanged (T2 0.2149 K / U10 0.2306 / V10 0.2752
m/s). The h2+ policy is wired correctly and is working as designed — it correctly
PASSES a case whose only large miss is the h1 spin-up (Case 1) and correctly FAILS a
case whose miss grows over the forecast (Cases 2, 3).

## h2+ policy — wired + unit-verified (commit `970ce5b`)

- `proofs/v040/s5_forecast_gate_exec.py`: the worst-over-leads envelope that DECIDES
  the blocking core PASS is now taken over leads `>= core_skill_first_lead`
  (default h2). The h1 lead is split into a reported `h1_spinup` diagnostic
  (`bias`/`rmse`/`abs_bias`/`within_margin`) that NEVER blocks. A case must have at
  least one h2+ scored lead per blocking core field (`core_has_h2plus_evidence`) for
  its h2+ PASS to count. Frozen `continuous_gate` margins untouched — only
  lead-inclusion changed (NOT a tolerance relaxation).
- `comparator.run_forecast_gate`: threads `core_skill_first_lead`, surfaces the
  policy + honest-claim string, and adds `reference_dir_overrides` + a backfill
  auto-fallback so cases whose corpus wrfout was purged from `run_dir` but retained
  in `wrf_l2_backfill_output` can still be scored vs CPU-WRF.
- Unit-verified on the known smoke numbers: V10 h2+ worst 0.208 (PASS, < 0.275)
  while h1 0.517 is reported non-blocking; the old all-lead policy failed.

## Case set (7 native-init forecasts; >=6 satisfied)

ONE GPU job throughout (verified free via nvidia-smi before launch; no orphan model
procs). First case paid the one-time ~22-min XLA cold compile; each subsequent case
re-traced (~15-20 min host-side compile per distinct bound configuration — a known
cost, NOT a hang; GPU at ~1% during compile, then ~2 s/forecast-hour). Per-case
result COMMITTED as it landed (hibernation-safe).

| case | CPU ref | scored leads | stable/phys 24h | h2+ core PASS |
|------|---------|--------------|-----------------|---------------|
| 20260428_l3 (..221139Z)     | oracle   | h1,h2        | YES | **YES** (1 h2+ lead) |
| 20260521_l3 (..072630Z)     | oracle   | h1..h8       | YES | **NO** (drift) |
| 20260521_l3 (..133443Z)     | oracle   | h1..h24      | YES | **NO** (drift) |
| 20260530_l3 (..050849Z)     | oracle   | h1 only      | YES | n/a (no h2+ ref) |
| 20260531_l3 (..125256Z)     | oracle   | h1 only      | YES | n/a (no h2+ ref) |
| 20260429_l2 (..204451Z)     | backfill | h1..h24      | (see report) | (see report) |
| 20260521_l2 (..133443Z)     | oracle   | h1..h19      | (see report) | (see report) |

(Cases 4-7 final numbers in `proofs/v040/forecast_gate_24h_report.json`; 5 cases
carry h2+ CPU references, 2 are h1-only stability+spin-up reports.)

## The drift signature (the real finding) — Case 3, full 24h trajectory

Case 3 (`20260521_l3_133443Z`, full 24h CPU coverage) is the decisive evidence.
**bias = mean(native - CPU), per lead hour:**

- **T2 (K):** h1 -0.05 -> rises monotonically to **+1.24 (h12)** -> swings down to
  **-0.90 (h21)** -> -0.75 (h24). RMSE peaks **3.7 K at h12**. A clear **diurnal
  amplitude/phase error**: the native-init forecast's surface-temperature diurnal
  cycle diverges from CPU-WRF (too warm by midday, too cold by evening). h1 bias is
  TINY -> this is NOT spin-up.
- **U10 (m/s):** h1 -0.10 -> grows monotonically to **+1.59 (h20-24)**, RMSE ~2.06.
- **V10 (m/s):** 0.42 (h1) -> dips ~0.31 (h12) -> grows to **+1.87 (h24)**, RMSE 2.55.
- **PSFC (Pa, descriptive):** h1 +42 -> -147 (h6) -> -191 (h18). Growing pressure
  drift (same family as the documented base-state hydrostatic offset).

Case 2 (same init time, different WPS cycle) shows the IDENTICAL drift onset over its
8 leads (T2 +0.96 / U10 +1.33 by h8), confirming the drift is **reproducible and
init-driven, not a one-off degenerate init**.

## Interpretation — this is the documented "damped-diurnal-T2" physics gap

The T2 diurnal signature (too-warm midday, too-cold evening, RMSE peaking at local
solar noon h12) is the SAME open physics blocker already in memory as the one
remaining v0.2.0 equivalence gap: **damped/phase-shifted diurnal T2, GPU vs CPU-WRF,
GPT-skeptic-confirmed REAL (not artifact)**. The native-init forecast inherits it and
— integrated over 24h with the native LBC instead of a replayed CPU LBC — it
accumulates into a ~1.2 K / ~1.6 m/s drift instead of staying within the looser
replay tolerances. The native-init machinery (IC dynamics/base, wrfbdy decoupling)
is NOT the proximate cause of the drift's GROWTH: the IC is stable and the early
leads (h1-h3) are close; the divergence accrues through the surface-flux / PBL /
radiation diurnal physics over the forecast, the same surface-physics gap the replay
path also carries but masks under looser scoring.

## Conservation + restart preconditions — PASS (cited, integrator unchanged)

The native-init gate reuses the SAME validated operational integrator
(`run_forecast_operational_segmented`); conservation + restart are properties of that
integrator, already proven:
- Conservation: `proofs/f7a2/conservation_long_run.json` — dry mass invariant
  (1.44e6 -> 1.44e6, exact, 300 steps, cuda:0). PASS.
- Restart: `proofs/p0_5/restart_roundtrip.json` — operational restart round-trip
  bit-identical (102 leaves, 0 mismatches; landless variant also bit-identical;
  schema drift fails closed). PASS.

## files changed / proof objects

- `proofs/v040/s5_forecast_gate_exec.py` — h2+ core-skill policy in the scoring.
- `src/gpuwrf/init/real_init/comparator.py` — `core_skill_first_lead` thread,
  `reference_dir_overrides` + backfill auto-fallback, policy in the result.
- `proofs/v040/run_forecast_gate_24h.py` — full-24h runner, per-case incremental
  commit, h2+ aggregation, backfill reference resolution.
- `proofs/v040/forecast_gate_24h_report.json` — **the deliverable**: per-case
  per-lead core-field deltas vs CPU-WRF, h2+ PASS/FAIL, h1 documented, full 24h
  trajectories, scoring policy, conservation/restart citations.

## commands run

```
# GPU verified free (only desktop procs), then detached full gate (ONE GPU job):
nohup setsid taskset -c 0-3 env PYTHONPATH=src:proofs/v040:. OMP_NUM_THREADS=4 \
  JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python proofs/v040/run_forecast_gate_24h.py --hours 24 \
  --out proofs/v040/forecast_gate_24h_report.json
```

## unresolved risks / next decision needed

- **v0.4.0 milestone does NOT close on this gate.** The native-init standalone
  forecast is STABLE for 24h and MATCHES CPU-WRF at the early leads (h1-h3), but
  DRIFTS over the forecast — it does not yet match CPU-WRF across all leads within
  the frozen margins. The honest claim is narrower than "standalone forecast matches
  CPU-WRF": it is "native-init produces a STABLE 24h forecast that tracks CPU-WRF for
  the first few hours, then accrues the documented diurnal-T2 / surface-physics
  drift." h1 is correctly not claimed as nowcast parity; but neither is h2+ across
  the full 24h.
- **Root cause is the surface/PBL/radiation diurnal physics, NOT the native-init
  assembly.** This is the SAME `damped-diurnal-T2` gap memory flags as the last open
  v0.2.0/equivalence blocker. v0.4.0 (native-init) is GATED on closing it: the
  native LBC removes the CPU-WRF replay crutch that was holding the replay path
  inside tolerance, so the underlying surface-physics drift is now exposed end to
  end. Fixing the diurnal-T2 surface-flux physics (the in-flight RRTMG-SW clear-sky
  SWDOWN / surface-flux work) is the prerequisite, then RE-RUN this gate.
- **Opposite-author debug:** this gate body + native-init assembly is Opus-authored;
  per the debug rule the diurnal-physics fix + re-run should route to GPT (or be
  bisected: disable radiation / disable PBL / freeze surface fluxes per-component on
  the Case-3 full-24h trajectory to localize whether T2 drift leads or lags the wind
  drift).
- The 2 h1-only cases (20260530, 20260531) contribute stability + h1 reports only;
  more retained-lead corpus (or backfill re-runs) would power the h2+ statistics, but
  the verdict is already conclusive from Cases 2 + 3 (8 and 24 h2+ leads).
- Conservation/restart cited from the validated integrator proofs (not re-run here);
  the gate changes only the IC/LBC source, not the integrator.
