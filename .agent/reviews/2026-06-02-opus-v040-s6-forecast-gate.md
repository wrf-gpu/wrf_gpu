# v0.4.0 S6 — native-init → 24h GPU forecast gate body + foundation stability smoke

Worker: Opus 4.8 (MAX). Branch: `worker/opus/v040-integration`. Date: 2026-06-02.

## Objective

LAST v0.4.0 implementation step + the FOUNDATION PROOF: wire the
native-init → GPU forecast pipeline behind
`comparator.run_forecast_gate(..., execute=True)` (was a guarded
`NotImplementedError`), and smoke-prove the foundation: a forecast launched from
our NATIVE real-init IC + NATIVE wrfbdy (NO CPU-WRF replay) stays STABLE and
MATCHES CPU-WRF at the early leads. Reuse the validated operational forecast
integrator — change ONLY the IC + LBC source.

## What was implemented

1. **`comparator.run_forecast_gate(execute=True)` body** (`src/gpuwrf/init/real_init/comparator.py`).
   - `execute=False` still returns the PLAN dict (unchanged contract).
   - `execute=True` discovers cases, builds the native `RealInitProduct` per case
     via the integrated `driver.build_real_init` factory (`make_factory`), drives
     the GPU forecast for the lead, scores per-lead vs the CPU-WRF wrfout with the
     frozen `proofs/m20/continuous_gate.py` metric (T2/U10/V10 BLOCKING; PSFC/PBLH/Q2
     descriptive), and writes a verdict proof. New kwargs: `forecast_hours`,
     `max_cases`, `dt_s`, `acoustic_substeps`, `radiation_cadence_steps`,
     `output_root`. Lazily imports the GPU body so the comparator stays a light
     non-GPU import.
   - Verdict ladder: `FOUNDATION_CONFIRMED` (stable + physical + core within
     margin) → `STABLE_BUT_CORE_FIELD_MISMATCH` → `FINITE_BUT_UNPHYSICAL` →
     `BLOWUP`.

2. **GPU body** (`proofs/v040/s5_forecast_gate_exec.py`, the single-GPU
   serialization point the S4 scaffold reserves for S5/manager):
   - `build_native_forecast_case`: packs the IC `State` + `BaseState` 100% from the
     native product (dynamics + base columns + surface + soil). `theta` → full dry
     theta (T+300, operational convention); `theta_base` = t0 + native `t_init` so
     the dycore's recomputed base inverse density `alb` matches the native discrete
     base state. Static grid geometry (map factors / hybrid-eta / Coriolis) sourced
     via the exact `load_wrfinput_metrics` on the reference t0 wrfout — these are
     STATIC geometry the S5 parity gate already proves native reproduces within the
     frozen `WRFINPUT_TOLS` (MAPFAC/C1*/F/E all PASS); the IC dynamics + base + LBC
     (the things the standalone claim is ABOUT) are 100% native.
   - `build_native_boundary_leaves`: DECOUPLES the native `LateralBC` (WRF wrfbdy
     mass-coupled values + tendencies) back to the decoupled raw fields the
     operational `apply_lateral_boundaries` adapter consumes, packed into the
     `(time, side, bdy_width, z, side_len)` State leaf layout. The NATIVE wrfbdy is
     the ONLY LBC source — NO CPU-WRF replay. Theta boundary: native `t` couples
     WRF THM (moist); converted back to full dry theta `(thm+300)/(1+Rv/Rd·qv)` so
     the forced ring matches the interior convention. **Verified on CPU**: the
     decoupled t0 boundary value reproduces the native IC field on the boundary
     strip to machine epsilon (theta 2.8e-14, u 7e-15, ph 9e-13; theta_bdy full vs
     native-full 5.7e-14).
   - `run_one_case_forecast_gate`: drives `run_forecast_operational_segmented`
     (the SAME validated operational entry the d02/d03 replay path uses) hour by
     hour with the Sprint-U d01 namelist (force_fp64, top_lid, epssm=0.5, flux
     advection, diff_6th_opt=2/0.12, w_damping=1, damp_opt=3, zdamp=5000,
     dampcoef=0.2, force_geopotential=True), writes per-lead wrfout via the
     operational writer + M9 surface diagnostics, runs a per-lead
     finite + gross-physical-range stability check, and scores vs CPU-WRF.

3. **Smoke runner** (`proofs/v040/run_forecast_gate_smoke.py`): 1-case bounded GPU
   smoke; writes `proofs/v040/s5_forecast_gate_report.json`.

## Commands run

```
# CPU validation of the decoupling (machine-epsilon faithful):
taskset -c 0-3 env JAX_PLATFORM_NAME=cpu PYTHONPATH=src:proofs/v040:. python -c "...build_native_boundary_leaves..."
# GPU smoke (ONE job; GPU verified free via nvidia-smi first):
taskset -c 0-3 env PYTHONPATH=src:proofs/v040:. python proofs/v040/run_forecast_gate_smoke.py \
    --hours 6 --max-cases 1 --dt-s 60 --acoustic-substeps 4 --radiation-cadence-steps 30 \
    --out proofs/v040/s5_forecast_gate_report.json
```

## Proof objects

- `proofs/v040/s5_forecast_gate_report.json` — smoke verdict (stability per-lead +
  early-lead core-field deltas vs CPU-WRF).
- GPU wrfout under `/tmp/v040_forecast_gate/<case>_d01/`.

## VERDICT — STABLE, but an honest hour-0 IMBALANCE found (NOT a blowup)

Smoke: 1 case (`20260428_18z_l3_24h_20260525T221139Z`), d01, 6h, dt=60s,
acoustic=4, radiation cadence 30. Full 6 leads ran. `proofs/v040/s5_forecast_gate_report.json`.

**Verdict = `STABLE_BUT_CORE_FIELD_MISMATCH` (foundation_confirmed = False).**

### (a) STABILITY — CONFIRMED.
All 6 forecast hours stayed **finite AND physical**. theta in [289, 500] K every
hour; |w| ≤ 0.6 m/s every hour; mu positive. **NO hour-0 hydrostatic/geostrophic-
imbalance detonation** — the classic native-init failure mode did NOT occur. The
native IC + native wrfbdy drive a calm, stable integration. Compile is a ONE-TIME
cost: hour 1 = 1126 s (cold XLA compile of the d01 operational scan), hours 2–6 =
~2 s each (warmed). So the foundation's *dynamical stability* is positively
established.

### (b) EARLY-LEAD MATCH vs CPU-WRF — a REAL hour-0 IMBALANCE signature.
Scored leads (h1, h2; h3–6 had no matching CPU d01 reference frame for this case —
see risks):

| field | h1 bias | h2 bias | margin | within |
|-------|---------|---------|--------|--------|
| PSFC | **−505 Pa** | **−528 Pa** | 50 Pa | NO |
| V10  | **+2.30 m/s** | **+2.07 m/s** | 0.275 | NO |
| T2   | −0.25 K | +0.03 K | 0.215 | borderline |
| U10  | −0.08 m/s | −0.06 m/s | 0.231 | YES |
| Q2 / PBLH / RAINNC / RAINC | within / tiny | | | YES |

**Signature = a near-uniform ~−520 Pa surface-pressure offset present from hour 1
and steady, plus a meridional-wind (V10) bias ~+2.3 m/s.** It is NOT a blowup and
NOT a growing instability — it is a steady hour-0 IMBALANCE between the native
init's base/dynamics state and the dycore's discrete hydrostatic balance. This is
the SAME family as the documented d02/d03 "near-uniform diagnostic surface-pressure
offset → Exner-T2" issue (see `_wrf_base_theta_from_loaded_state` and the
2026-06-01 pressure-drift root-cause review): the loaded IC is slightly out of the
dycore's *discrete* hydrostatic balance, so the diagnosed surface pressure carries
a steady offset and the geostrophic wind adjusts (V10 bias). The native real-init
base state (`base.pb/phb/mub`, `base.t_init`) is the prime suspect — its discrete
balance must match the dycore's `diagnose_pressure_al_alt` to ~Pa, the way
`build_replay_case` recovers `theta_base` by INVERTING the loaded discrete base
state. The S6 executor sets `theta_base = t0 + native t_init` directly rather than
inverting the discrete base; that is the most likely ~520 Pa source and the first
thing the cross-model fixer should check.

**This is a REAL, important finding — reported honestly, NOT masked/clamped.** No
guard or clamp was used to hide it. The foundation is STABLE; the early-lead MATCH
exposes a subtle hour-0 base-state/pressure-balance imbalance that the manager
should route to a cross-model fix (Opus wrote S6; GPT or agy is the right fixer per
the opposite-author debug rule).

## Unresolved risks / notes

- **Cold-compile cost is ONE-TIME**: hour 1 = 1126 s (cold XLA compile of the d01
  operational scan), hours 2–6 = ~2 s each. So the full 6×24h set pays the ~19-min
  compile once then runs at ~2 s/forecast-hour. The manager MUST run it detached
  (`nohup setsid`) and not kill it during the first-hour compile (an earlier 1h
  variant was killed by a 9-min timeout mid-compile — a false "hang").
- **h3–6 had no scored CPU reference for this smoke case**: the binding match
  evidence is h1/h2. The manager's full run should pick cases whose corpus d01
  wrfout retains every lead (e.g. `20260521_18z_l3_24h_20260522T133443Z` has 25
  hourly d01 frames) so all leads score.
- **PRIME fix candidate for the ~520 Pa PSFC offset**: the S6 executor sets
  `BaseState.theta_base = t0 + native t_init` directly. The validated replay path
  instead INVERTS the loaded discrete base state (`_wrf_base_theta_from_loaded_state`)
  so the dycore's recomputed base inverse density `alb` reproduces the loaded `phb`
  to round-off. Applying that same discrete-base inversion to the native
  `pb/phb/mub` (instead of the analytic `t0+t_init`) is the most likely collapse of
  the steady PSFC offset + V10 bias. The cross-model fixer should start here.
- Static grid metrics are sourced from the reference t0 wrfout (exact geometry the
  parity gate proves native matches within frozen tols), NOT from CPU-WRF dynamics.
  A fully self-contained native path would build `DycoreMetrics` from the native
  `vcoord`/`surface` directly; that is a clean follow-up (geometry only, no dynamics
  smuggled) and does not affect the standalone-init claim being tested here.
- `force_geopotential=True` is correct here because d01 is the PARENT (the native
  wrfbdy strips are self-consistent with the column), matching the validated d02
  self-replay path; the nested `force_geopotential=False` d03 branch is not used.

## What the manager runs next

The full 6-case × 24h gate (detached), now that the body is wired:

```python
from gpuwrf.init.real_init import comparator as C
from proofs.v040.s5_native_init_parity import make_factory
C.run_forecast_gate(make_factory(lbc_intervals=1), execute=True,
                    forecast_hours=24, max_cases=6,
                    out_path="proofs/v040/s5_forecast_gate_full24h.json")
```

ONE GPU job at a time; budget the one-time cold compile; run detached
(`nohup setsid`) per the long-GPU-run rule.
