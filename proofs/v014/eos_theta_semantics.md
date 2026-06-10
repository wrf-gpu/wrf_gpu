# V0.14 EOS / State.theta Semantics Proof

Verdict: `FIXED_UNIFIED_MOIST_THETA_M_CONVENTION_GPU_RERUN_REQUIRED`. CPU-only. Companion JSON: `eos_theta_semantics.json`.

## Runtime convention (pre-fix falsifier files)

- wrfinput/wrfout variable `T` is DRY perturbation theta: wrfinput_d02 `T` ==
  CPU h0 `T` to 0.0e+00 K
  (vs THM: 5.49 K).
- CPU WRF runtime identity `THM == (T+300)(1+rvovrd qv)-300` to
  6.8e-05 K.
- GPU **d01** h1 `T` k0 bias vs dry +0.64 K,
  vs THM -4.02 K -> d01 State.theta was DRY.
- GPU **d02** h1 `T` k0 interior bias vs THM
  -0.38 K (vs dry
  +4.41 K) -> d02 interior was MOIST theta_m;
  boundary band5 vs dry +1.10 K -> the
  ring was forced DRY by the d01 parent (mixed-convention nest boundary, ~5 K edge error).
- Falsifier d02 init metadata: use_theta_m=1,
  theta_m_conversion_applied=True.

## Internal EOS self-consistency (rmse of EOS(p) - file P, interior, Pa)

| file | qvf=1 | qvf=1+0.608qv | qvf=1+1.608qv |
|---|---:|---:|---:|
| CPU d02 h1 (T dry) | 878.4 | 548.4 | 22.3 |
| GPU d01 h1 (T dry) | 330.1 | 5.1 | 547.5 |
| GPU d02 h1 (T moist) | 334.7 | 5.1 | 554.3 |

Every file is self-consistent with ONE factor applied to its own written T: CPU needs dry*1.608 (== theta_m*1, the WRF use_theta_m=1 EOS); pre-fix GPU needs written_T*(1+0.608qv) on BOTH domains -- but GPU d01 written T is DRY theta while GPU d02 interior written T is MOIST theta_m, so the single 0.608 kernel encoded TWO different wrong EOS forms (d01 vapor-light by ~1.0qv; d02 over-coupled by ~0.61qv on top of theta_m).

## Post-fix proofs

- Helper identity (machine precision): moist(qv=None) and dry(qv) forms both
  reproduce the analytic WRF EOS to rel
  6.7e-16.
- Ingest end-to-end: `build_replay_case(d01, standalone)` State.theta ==
  wrfinput THM to 6.1e-05 K
  (vs dry: 5.47 K);
  wrfbdy theta leaves moist and IC-consistent to
  3.1e-05 K.
- Writer round trip: `T` == dry theta to
  3.1e-05 K and `THM` == moist
  theta_m to 0.0e+00 K.
