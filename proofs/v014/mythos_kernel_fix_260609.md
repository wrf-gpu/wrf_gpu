# V0.14 Mythos Kernel Fix: live-nest start_domain P/MU/W

Verdict: `MYTHOS_KERNEL_FIX_START_DOMAIN_P_MU_W_CLOSED_FP32_LIBM_SINT_BLEND_BIT_EXACT`.

## Root Cause (bit-exact)

The CPU-WRF truth is gfortran `-O2` REAL(4) calling the scalar glibc float32 libm.
Three independent ulp sources explained the entire remaining base/perturbation gap;
each ulp is amplified ~50x through the fp32 hypsometric `AL/ALT` layer-thickness
division into the perturbation pressure:

| p_surf -> MUB candidate (WRF HT, WRF fp32 op order) | max_abs Pa | mismatched cells |
|---|---:|---:|
| `A_numpy_simd_fp32_exp_sqrtss` | 0.0546875 | 660/10494 |
| `B_float64_rounded_exp_sqrt` | 0.046875 | 3/10494 |
| `C_glibc_expf_sqrtss` | 0.046875 | 2/10494 |
| `D_glibc_expf_glibc_powf05` | 0.0 | 0/10494 |

`D` (glibc `expf` + glibc `powf(x,0.5)`) is **bit-exact**: gfortran compiles `(...)**0.5`
to a `powf` call, and NumPy's float32 SIMD `exp` is not glibc's `expf`.

| blended HT source | max_abs m vs WRF HT |
|---|---:|
| `fp64_sint_blend_previous_behaviour` | 2.682152282318384e-05 |
| `fp32_sint_blend_patched` | 4.547473508864641e-13 |

## Production Patch Result (vs WRF internal start_domain truth)

| Field | before patch | after patch | gate | pass |
|---|---:|---:|---:|---|
| `P_STATE` | 69.96875 | 0.0390625 | 1.0 | True |
| `MU_STATE` | 13.256103515625 | 4.547473508864641e-13 | 0.01 | True |
| `W_STATE` | 0.7605466842651367 | 5.551115123125783e-17 | 0.001 | True |

Base after patch: `MUB` 0.0, `PB` 0.0, `PHB` 4.547473508864641e-13, `HT` 4.547473508864641e-13 (truth-dump text precision).
All declared gates pass: `True`. libm provider: `glibc-libm-float32`.

## Strict Step-1 16-field comparison (one RK step, patched init)

Status: `COMPARISON_EXECUTED`; first divergent field: `T`.

| Field | max_abs | rmse |
|---|---:|---:|
| `T` | 0.15948589853104567 | 0.002242006507629072 |
| `P` | 975.1236470550566 | 135.95494353894753 |
| `PB` | 0.0 | 0.0 |
| `PH` | 63.82327410901786 | 17.415390008787018 |
| `PHB` | 4.547473508864641e-13 | 1.8923118348432175e-14 |
| `MU` | 14.007953430216503 | 0.7485993568714623 |
| `MUB` | 0.0 | 0.0 |
| `U` | 0.7838795148321176 | 0.02035888890860235 |
| `V` | 0.6215864259938341 | 0.02583415598943992 |
| `W` | 2.6401070776077424 | 0.44965939977129343 |
| `QVAPOR` | 0.000181070063263177 | 3.2924865537392053e-06 |
| `QCLOUD` | 0.0 | 0.0 |
| `QRAIN` | 0.0 | 0.0 |
| `QICE` | 0.0 | 0.0 |
| `QSNOW` | 0.0 | 0.0 |
| `QGRAUP` | 0.0 | 0.0 |

### Post-step residual attribution

- `P_wrf_eos_of_jax_post_step_state_vs_wrf_P`: max_abs 955.3671875, rmse 285.51252240353875
- `captured_jax_p_perturbation_vs_wrf_P`: max_abs 975.1236470550566, rmse 135.95494353894753
- `sanity_P_wrf_eos_of_wrf_post_step_state_vs_wrf_P`: max_abs 2.90625, rmse 0.07339449919424673

WRF-EOS pressure from the JAX post-step (PH, MU, theta) stays far from WRF P while the same diagnostic on WRF's own post-step state is ~0: the residual is REAL one-step dynamics state divergence (PH/MU evolve differently), not a pressure-diagnosis semantic gap. Theta/U/V p95 are tiny while P/PH/MU are broad, pointing at the acoustic/mass/vertical lane or one-step namelist parity (acoustic substep count, epssm, damping) rather than physics or horizontal advection. NOTE: the proof surrogate namelist hardcodes acoustic_substeps=10/epssm=0.5/damp_opt=3; namelist parity with the WRF case3 run must be frozen before instrumenting dycore substages.

## Ranked Hypotheses

- 1. The remaining p_surf->MUB gap was float32 libm provenance: WRF calls scalar glibc expf, and gfortran compiles (...)**0.5 to a glibc powf(x,0.5) call (1 ulp from sqrtf on rare inputs). Status: `PROVEN_BIT_EXACT`. glibc expf + glibc powf(x,0.5) reproduces WRF MUB exactly: max_abs 0.0 over 10494 cells; numpy SIMD fp32 exp leaves 660 cells (max 0.0546875 Pa); glibc expf + sqrtss leaves 2 cells (max 0.046875 Pa).
- 2. The residual production P gap after exact-libm base was the float64 SINT/blend terrain (<= 1 fp32 ulp HT error amplified ~50x through the AL/ALT layer-thickness division). Status: `PROVEN_AND_CLOSED`. fp64 SINT/blend HT max_abs 2.682152282318384e-05 m -> fp32 WRF-order SINT/blend 4.547473508864641e-13 m (truth-dump text precision).
- 3. Raw wrfinput perturbation P/MU/W leaves miss three WRF start_domain mutations (hypsometric P rederivation, press_adj MU, set_w_surface W). Status: `CONFIRMED_BY_PREDECESSORS_AND_CLOSED_IN_PRODUCTION`. P_STATE 69.96875 -> 0.0390625 Pa; MU_STATE 13.256103515625 -> 4.547473508864641e-13 Pa; W_STATE 0.7605466842651367 -> 5.551115123125783e-17 m/s.
- 4. float64-rounded transcendentals are a sufficient production formula. Status: `REFUTED_AS_PRIMARY_KEPT_AS_FALLBACK`. float64-rounded exp leaves 3 MUB cells at max 0.046875 Pa and (full chain, measured in scratch) P_STATE ~2.23 Pa > 1 Pa gate; it remains only the non-glibc fallback.

## Exclusions

- Terrain values, coefficients, cp constants, theta/QV, PH/MU time levels were excluded by predecessor proofs.
- No dycore, runtime, state-contract, boundary-ABI, wrfout, memory, or FP32 source was modified.
- The W branch keeps the WRF w_needs_to_be_set gate; inputs carrying real W are not overwritten.

## Files Changed

- `src/gpuwrf/integration/d02_replay.py` (fp32 WRF-libm base recompute; fp32 SINT/blend call sites;
  new `_wrf_live_nest_start_domain_perturb_init` wired into `build_replay_case`)
- `src/gpuwrf/nesting/interp.py` (dtype-parameterized SINT host reference; float64 default unchanged)

## Next Decision

Init-time P/MU/W is closed to WRF start_domain truth (bit-exact base, P 0.039 Pa). The strict Step-1 16-field one-RK-step comparison is now the authoritative post-init divergence frontier, and the attribution probe shows it is REAL one-step dynamics state divergence in the PH/MU/P (acoustic/mass/vertical) lane, not init and not pressure-diagnosis semantics. Next sprint: (1) freeze one-step namelist parity (acoustic substep count, epssm, damping) between the proof surrogate and the WRF case3 namelist; (2) rerun the existing RK1 substage comparator chain (/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth is still valid WRF-side) against JAX substages built from the NOW-CLOSED init state. No dycore source edit is justified before that localization.

Detailed metrics: `proofs/v014/mythos_kernel_fix_260609.json`.
