[gemini side-runner #2 — parity-numbers sanity] started at Wed May 20 14:10:16 UTC 2026
Acknowledge onboarding prefix. Proceeding directly to the parity-numbers sanity check report.

**Read**: `/home/enric/src/wrf_gpu2/artifacts/m5/tier1_thompson_parity.json`, `/home/enric/src/wrf_gpu2/artifacts/m5/tier2_thompson_invariants.json`, `/home/enric/src/wrf_gpu2/artifacts/m5/thompson_gate_result.json`, `/home/enric/src/wrf_gpu2/artifacts/m5/thompson_profile.json`, `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/BLOCKER-m5-s1-attempt4-tolerance.md`, `/home/enric/src/wrf_gpu2/.agent/sprints/2026-05-20-m5-s1-thompson-microphysics-column/diagnosis-report.md`, `/home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py`, `/home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`.

### 1. Diagnosis-budget verdict: holds
The attempt-4 parity results confirm the diagnosis error budget:
- **Process Order (predicted 55-65%):** Temperature max error fell by ~87% from `0.3186 K` to `0.0403 K` (`tier1_thompson_parity.json:18`), verifying the staging prediction (`diagnosis-report.md:11`).
- **Ni handling (predicted 5-10%):** `Ni` max absolute error fell by ~91% from `1.414e6` to `126975.125` (`tier1_thompson_parity.json:16`), matching the expected level (`diagnosis-report.md:35`).
- **Lookup Tables (predicted 20-30%):** The residual species errors remain concentrated in the hydrometeors (`qc`/`qi`/`qs` max absolute errors are `1.52e-4`, `1.37e-4`, and `1.45e-4` in `tier1_thompson_parity.json:19-23`), matching the lookup-table budget (`diagnosis-report.md:25`). There is no evidence of an unexpected fourth error source.

### 2. Tier-2/Tier-1 cross-check: consistent with table proxies
Tier-2 reports `water_residual = 2.67e-12` (`tier2_thompson_invariants.json:20`), proving total water is perfectly conserved at fp64 precision. The presence of species mass partition discrepancies of scale ~1.5e-4 indicates that mass is being shifted between hydrometeors differently. This is highly consistent with using table/moment proxies (e.g. `_snow_moment_proxy`, linear `t_Efrw` proxy, and omitted `tps_iaus/tni_iaus` autoconversion tables) as they dictate internal transfer rates.

### 3. Profile sanity: kernel doing real work
XLA successfully fused the JAX program, resulting in `temporary_bytes_per_step = 0` (`thompson_profile.json:24`) and `kernel_launches_per_step = 1` (`thompson_profile.json:17`). Correctness tests confirm it is not dead-code-eliminated: the temperature error dynamically dropped when changing process-order structure, and the conservation invariants are validated on the output state.

### 4. JAX scan: one suspect identified
In [thompson_column.py:277-278](file:///home/enric/src/wrf_gpu2/src/gpuwrf/physics/thompson_column.py#L277-L278), JAX caps the `lami` parameter when `xdi` goes out of bounds:
```python
lami = jnp.where(xdi < 5.0e-6, 6.0 / 5.0e-6, lami)
lami = jnp.where(xdi > 300.0e-6, 6.0 / 300.0e-6, lami)
```
However, `xdi` is defined using `4.0 / lami` (line 276). In WRF (`module_mp_thompson.F.pre:4096-4100`), the numerator is `cie(2)`. Since `cie(2) = bm_i + mu_i + 1. = 4.0` (`module_mp_thompson.F.pre:688`), WRF uses `4.0 / limit`. JAX mistakenly uses `6.0` (which is `CIG2`/`cig(2)`), causing clipped `lami` values to set `xdi` to out-of-bounds sizes.

### 5. Counterargument
The species errors could stem from a coefficient typo in a transfer rate rather than the table proxies. However, because the errors are distributed across multiple hydrometeors (`qc`/`qi`/`qs`) at the same scale (~1.5e-4), they represent coupled rate discrepancies consistent with approximate proxy equations rather than a single localized typo.

**Confidence**: **high**, as the `lami` clipping mismatch is mathematically and textually verified against WRF source.

| 2026-05-20 | M5-S1 attempt-4 parity-numbers sanity check | Verified error budget (holds); identified JAX-WRF `lami` clipping numerator mismatch | Useful side-runner. Found one mismatch in `_finish` and confirmed profile sanity. |

[gemini side-runner #2] finished at Wed May 20 14:11:40 UTC 2026
