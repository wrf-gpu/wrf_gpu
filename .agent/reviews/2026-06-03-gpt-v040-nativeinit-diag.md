# GPT v0.4.0 Native-Init Bias Diagnosis

## Objective

Diagnose the v0.4.0 fixed forecast-gate systematic signature
`PSFC negative bias + U10 positive bias` across 20260429 and 20260521, first
ruling out a scoring/measurement artifact, then localizing whether the error is
present at native-init t0 or grows during integration.

Branch/worktree:

- `/home/enric/src/wrf_gpu2/.claude/worktrees/v040-nativeinit-diag`
- `worker/gpt/v040-nativeinit-diag`
- Created from live `worker/gpt/v040-gatefix` tip `94655c1`; prompt-cited
  `f0c9692` is behind the current gatefix history.

## Apples-To-Apples Check

Result: scoring artifact ruled out for the scored fields.

Evidence:

- `proofs/m20/continuous_gate.py:146-157` reads same-name NetCDF fields,
  squeezes only leading `Time`, and does not alter 2-D `PSFC`, `U10`, `V10`,
  or `T2`.
- `proofs/m20/continuous_gate.py:163-180` compares complete finite gridpoint
  pairs with `native - CPU` bias; no land/water mask, destaggering, unit
  conversion, or tolerance change is involved.
- Per-lead NetCDF metadata in
  `proofs/v040/nativeinit_bias_diagnosis.json` confirms matching valid time,
  dimensions, shapes, and units: `PSFC` is `Pa`, `U10/V10` are `m s-1`, and
  `T2` is `K`.
- t0 CPU `wrfinput_d01` and CPU t0 `wrfout_d01` agree exactly for `U10`, `V10`,
  and `T2`, so the surface diagnostic time-zero reference is aligned.

Caveat, not the cause of this gate failure:

- `src/gpuwrf/io/wrfout_writer.py:1216-1237` can fall back to approximate
  projection-derived `XLAT/XLONG` when runtime state coordinates are missing.
  The gate scorer does not use `XLAT/XLONG` and pairs arrays by index, but this
  writer metadata issue should be fixed later.

## t0 Native Init vs real.exe

Native standalone init matches the oracle `real.exe` `wrfinput_d01` at roundoff.
The systematic forecast bias is not present at t0.

Mean bias / RMSE / max abs:

| case | field | mean bias | RMSE | max abs |
|---|---:|---:|---:|---:|
| 20260429 l2 | PSFC derived from native mass+qv | -0.658 Pa | 4.229 Pa | 69.803 Pa |
| 20260429 l2 | MU | +0.0027 Pa | 0.0211 Pa | 0.1975 Pa |
| 20260429 l2 | MUB | -0.0013 Pa | 0.0075 Pa | 0.0478 Pa |
| 20260429 l2 | MU+MUB | +0.0014 Pa | 0.0194 Pa | 0.1877 Pa |
| 20260429 l2 | PB | +0.0016 Pa | 0.0052 Pa | 0.0536 Pa |
| 20260429 l2 | PHB | -0.0288 | 0.0455 | 0.1930 |
| 20260429 l2 | T | -3.08e-05 K | 6.67e-05 K | 2.66e-04 K |
| 20260429 l2 | U staggered | +5.71e-07 m/s | 8.30e-06 m/s | 1.53e-04 m/s |
| 20260429 l2 | V staggered | -4.24e-08 m/s | 7.67e-06 m/s | 2.26e-04 m/s |
| 20260521 l2/l3 | PSFC derived from native mass+qv | +0.547 Pa | 5.208 Pa | 92.767 Pa |
| 20260521 l2/l3 | MU | +0.0028 Pa | 0.0225 Pa | 0.2461 Pa |
| 20260521 l2/l3 | MUB | -0.0013 Pa | 0.0075 Pa | 0.0478 Pa |
| 20260521 l2/l3 | MU+MUB | +0.0015 Pa | 0.0204 Pa | 0.2175 Pa |
| 20260521 l2/l3 | PB | +0.0016 Pa | 0.0052 Pa | 0.0536 Pa |
| 20260521 l2/l3 | PHB | -0.0288 | 0.0455 | 0.1930 |
| 20260521 l2/l3 | T | -3.09e-05 K | 6.69e-05 K | 3.12e-04 K |
| 20260521 l2/l3 | U staggered | +2.24e-06 m/s | 1.07e-05 m/s | 1.78e-04 m/s |
| 20260521 l2/l3 | V staggered | +4.74e-07 m/s | 1.36e-05 m/s | 5.93e-04 m/s |

## Lead-Time Localization

Mean bias is native-GPU wrfout minus CPU-WRF wrfout. Missing means the reference
frame was not present for that scored oracle.

| case | lead | PSFC Pa | U10 m/s | V10 m/s | T2 K | MU Pa | MU+MUB Pa |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260429 l2 | h0 | -0.658 | n/a | n/a | n/a | 0.000 | 0.000 |
| 20260429 l2 | h1 | -23.912 | -0.370 | -0.030 | -0.146 | -24.205 | -24.206 |
| 20260429 l2 | h2 | -56.864 | -0.293 | +0.371 | +0.026 | -56.932 | -56.934 |
| 20260429 l2 | h6 | -232.404 | +0.279 | +0.701 | +0.307 | -232.800 | -232.802 |
| 20260429 l2 | h12 | -205.163 | +1.579 | +0.382 | +0.926 | -225.968 | -225.969 |
| 20260429 l2 | h24 | -429.949 | +1.352 | -0.107 | -0.700 | -480.101 | -480.102 |
| 20260521 l2 | h0 | +0.547 | n/a | n/a | n/a | 0.000 | 0.000 |
| 20260521 l2 | h1 | +42.173 | -0.104 | +0.419 | -0.047 | +40.866 | +40.865 |
| 20260521 l2 | h2 | +3.099 | +0.092 | +0.465 | +0.322 | +1.882 | +1.881 |
| 20260521 l2 | h6 | -147.223 | +1.020 | +0.509 | +0.822 | -151.135 | -151.137 |
| 20260521 l2 | h12 | -80.976 | +1.338 | +0.307 | +1.236 | -93.678 | -93.679 |
| 20260521 l2 | h24 | missing | missing | missing | missing | missing | missing |
| 20260521 l3 072630 | h0 | +0.547 | n/a | n/a | n/a | 0.000 | 0.000 |
| 20260521 l3 072630 | h1 | +42.173 | -0.104 | +0.419 | -0.047 | +40.866 | +40.865 |
| 20260521 l3 072630 | h2 | +3.099 | +0.092 | +0.465 | +0.322 | +1.882 | +1.881 |
| 20260521 l3 072630 | h6 | -147.223 | +1.020 | +0.509 | +0.822 | -151.135 | -151.137 |
| 20260521 l3 072630 | h12 | missing | missing | missing | missing | missing | missing |
| 20260521 l3 072630 | h24 | missing | missing | missing | missing | missing | missing |
| 20260521 l3 133443 | h0 | +0.547 | n/a | n/a | n/a | 0.000 | 0.000 |
| 20260521 l3 133443 | h1 | +42.173 | -0.104 | +0.419 | -0.047 | +40.866 | +40.865 |
| 20260521 l3 133443 | h2 | +3.099 | +0.092 | +0.465 | +0.322 | +1.882 | +1.881 |
| 20260521 l3 133443 | h6 | -147.223 | +1.020 | +0.509 | +0.822 | -151.135 | -151.137 |
| 20260521 l3 133443 | h12 | -80.976 | +1.338 | +0.307 | +1.236 | -93.678 | -93.679 |
| 20260521 l3 133443 | h24 | -69.790 | +1.586 | +1.873 | -0.752 | -94.600 | -94.602 |

## Localized Root Cause

Bias origin: grows in forecast, not t0 init.

Localized defect class: post-t0 dry-column mass continuity in the runtime dycore
/ acoustic small-step path. `PSFC` follows `MU+MUB` one-for-one after h1, while
`MUB` is stable and native t0 `MU/MUB/PB/PHB/P/PH/T/U/V` match real.exe.

WRF-faithful references:

- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3790-3806`
  computes base `p_surf`, `PB`, and `MUB`.
- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3881`
  computes `MU_2 = MU0 - MUB`.
- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F:3935-4036`
  performs the hydrostatic pressure/geopotential integration.
- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1048-1063`
  narrows `advance_mu_t` loop bounds for specified/nested non-periodic real
  cases.
- `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1092-1107`
  computes `DMDT` and updates `MU/MUDF/MUTS/MUAVE`.

Current JAX suspect path:

- `src/gpuwrf/dynamics/mu_t_advance.py:76-100` documents and implements
  periodic/full-domain neighbor assumptions.
- `src/gpuwrf/dynamics/mu_t_advance.py:124-159` computes full-domain `DMDT`
  and updates all `MU`.
- `src/gpuwrf/dynamics/mu_t_advance.py:186-203` uses periodic neighbor rolls
  for theta fluxes.
- `src/gpuwrf/runtime/operational_mode.py:755-821` builds acoustic-core mass
  inputs.
- `src/gpuwrf/dynamics/core/acoustic.py:527-560` invokes `advance_mu_t_core`.
- `src/gpuwrf/dynamics/core/small_step_finish.py:60-76` writes evolved
  `mu_perturbation` back to state.
- `src/gpuwrf/dynamics/advection.py:15-18` shows the shared dycore halo helper
  still defaults to `edge_type="periodic"`.

## Proposed Fix

No model-code fix was applied in this diagnostic branch. A one-line or local
clamp would not be WRF-faithful.

Concrete follow-up sprint:

1. Implement WRF specified/nested non-periodic loop bounds and halo semantics for
   `advance_mu_t` dry-mass continuity and mass-coupled theta fluxes.
2. Add WRF small-step savepoint proof for `DMDT`, `MU`, `MUDF`, `MUTS`, `MUAVE`,
   `ww`, and theta flux terms before any forecast-gate claim.
3. Rerun the two-date v0.4.0 forecast gate only after the operator-level proof
   passes.

## Proof Objects Produced

- `proofs/v040/nativeinit_bias_diagnosis.py`
- `proofs/v040/nativeinit_bias_diagnosis.json`
- This handoff:
  `.agent/reviews/2026-06-03-gpt-v040-nativeinit-diag.md`

## Commands Run

```bash
git worktree add -b worker/gpt/v040-nativeinit-diag .claude/worktrees/v040-nativeinit-diag worker/gpt/v040-gatefix
git log -1 --oneline
sed -n '1,220p' PROJECT_CONSTITUTION.md
sed -n '1,220p' AGENTS.md
sed -n '1,220p' .agent/skills/managing-sprints/SKILL.md
sed -n '1,220p' .agent/decisions/V0.4.0-S0-PLAN.md
rg -n "PSFC|U10|T2|MU|MUB|advance_mu_t|metadata_run_dir|run_forecast_gate" src/gpuwrf/init/real_init proofs/v040 proofs/m20 src/gpuwrf/dynamics src/gpuwrf/runtime
nl -ba /home/enric/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F | sed -n '3770,4050p'
nl -ba /home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F | sed -n '1038,1125p'
taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src:proofs/v040:. python proofs/v040/nativeinit_bias_diagnosis.py --out proofs/v040/nativeinit_bias_diagnosis.json
```

GPU was not used.

## Files Changed

- Added `proofs/v040/nativeinit_bias_diagnosis.py`
- Added `proofs/v040/nativeinit_bias_diagnosis.json`
- Added `.agent/reviews/2026-06-03-gpt-v040-nativeinit-diag.md`

No runtime/model code was changed.

## Unresolved Risks

- The localized dycore fix still needs implementation and WRF savepoint proof.
- `XLAT/XLONG` runtime writer fallback should be fixed, even though it is not
  this scored-field artifact.
- Two 20260521 references lack some h12/h24 comparison frames in the available
  oracle set; the l2 and l3 133443 references still provide the required
  h1/h2/h6/h12 and h24 localization evidence.

## Next Decision

v0.4.0 cannot close yet. The standalone native real-init blocker should be
reclassified: native init itself matches `real.exe`, but the v0.4.0 forecast
gate remains blocked by post-t0 runtime dry-mass continuity drift. Open a
focused dycore mass-boundary sprint, not another native-init sprint.
