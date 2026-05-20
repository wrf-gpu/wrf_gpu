[gemini side-audit attempt-5] started at Wed May 20 14:44:26 UTC 2026
Read: [/home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py), [/home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_constants.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_constants.py), [/home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_saturation.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_saturation.py), [/home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre](file:///home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre), [/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/tester-a4-report.md](file:///home/enric/src/wrf_gpu2/.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/tester-a4-report.md)

### Suspect 1: Graupel Sublimation/Melting Exponent and Gamma Multipliers
1. **JAX file:line** — [thompson_column.py:462, 491](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py#L462) and [thompson_constants.py:89, 91](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_constants.py#L89)
2. **WRF file:line** — [module_mp_thompson.F.pre:2763-2764](file:///home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre#L2763), [2874-2875](file:///home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre#L2874), [2893-2894](file:///home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre#L2893)
3. **The mismatch** — Graupel sublimation/melting uses rain exponent `CRE11 = 3.0` and gamma multiplier `2.0` instead of `cge(11) = 2.82048` and `cgg(11) = 1.70425`.
4. **Severity** — `confirmed`
5. **One-line fix proposal** — Define `CGE11 = 2.8204808235` and `CGG11 = 1.7042533` in [thompson_constants.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_constants.py), adjust `T2_SUBL_QG` and `T2_MELT_QG` to use `CGG11` instead of `2.0`, and replace `ilamg**CRE11` with `ilamg**CGE11` in [thompson_column.py](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py).

### Suspect 2: Snow Sublimation Constant `c_snow`
1. **JAX file:line** — [thompson_column.py:489](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py#L489)
2. **WRF file:line** — [module_mp_thompson.F.pre:2859](file:///home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre#L2859)
3. **The mismatch** — Aggregate sublimation uses `c_snow` instead of the WRF melting constant `C_sqrd` when `T >= T_0`.
4. **Severity** — `dismissed`
5. **One-line fix proposal** — None; `c_snow` naturally clips to `C_SQRD` when `tempc >= 0.0` due to `jnp.maximum` capping.

### Suspect 3: Wet-Bulb Temperature `twet`
1. **JAX file:line** — [thompson_column.py:451](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py#L451)
2. **WRF file:line** — [module_mp_thompson.F.pre:2074](file:///home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre#L2074)
3. **The mismatch** — `twet` is approximated as `jnp.minimum(state.T, T_0)` instead of evaluating the iterative `compT_fr_The` function.
4. **Severity** — `dismissed`
5. **One-line fix proposal** — None; this is a known physical proxy simplification rather than a transcription typo.

---

- **Total suspects identified**: 3
- **Confirmed**: 1
- **Suspected**: 0

### Counterargument against my own answer
It could be argued that because this JAX kernel is a simplified column subset, unifying the graupel sublimation exponents with rain exponents was an intentional approximation to reduce compilation complexity and variable counts. However, since the graupel constants `T2_SUBL_QG` and `T2_MELT_QG` are explicitly defined, using the wrong rain multipliers and exponents (`3.0` and `2.0` instead of `2.82048` and `1.70425`) constitutes a transcription oversight that will introduce non-physical mass-transfer divergences relative to the WRF oracle.

### Confidence
High; the graupel coefficients were cross-referenced directly with WRF initialization arrays (`av_g`, `bv_g`, `cge`, `cgg`).

### Track record line
| 2026-05-20 | M5-S1 attempt-5 parallel side-audit | Identified 1 confirmed graupel exponent/multiplier mismatch and dismissed 2 other suspects. | Gemini successfully caught a major graupel coefficient mismatch that other workers missed. |

[gemini side-audit attempt-5] finished at Wed May 20 14:46:41 UTC 2026
