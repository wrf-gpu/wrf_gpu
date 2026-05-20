# M5-S1 Attempt-3 Thompson Parity Diagnosis

Objective: identify the source of the `artifacts/m5/tier1_thompson_parity.json` gaps: `T=0.3186 K`, `qv=1.1249e-4`, `qi=9.11e-5`, `Ni=1.414e6`, `Nr=6.73e4`.

Important scope note: the prompt-relative WRF path `../wrf_gpu/.../module_mp_thompson.F.pre` is absent from `/tmp/wrf_gpu2_diagnosis`; I read the same snapshot at `/home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`.

Top suspects:

## 1. WRF tendency staging and call sequence mismatch

Probability contribution to the 0.3 K error: 55-65 percent.

Rationale: WRF does not apply the processes in the JAX order. The JAX public step runs `_saturation_adjustment`, then `_ice_sources`, then `_warm_rain`, then `_finish` (`src/gpuwrf/physics/thompson_column.py:501-506`). WRF stages rates and tendencies first, applies conservation and temperature tendency (`module_mp_thompson.F.pre:2917-3247`), updates the working state before condensation (`module_mp_thompson.F.pre:3250-3273`), then performs cloud condensation/evaporation (`module_mp_thompson.F.pre:3456-3558`), then rain evaporation (`module_mp_thompson.F.pre:3561-3638`), then sedimentation (`module_mp_thompson.F.pre:3653-4003`), then instant cloud-ice melt/cloud-water freeze (`module_mp_thompson.F.pre:4005-4031`), and only then writes final fields (`module_mp_thompson.F.pre:4033-4142`). That is a different numerical method, not just different floating-point ordering.

The max T/qv error is cold mixed-phase level 10. A read-only probe found JAX `qv=1.830572e-4`, WRF `qv=2.955486e-4`, difference `-1.124914e-4`; JAX `T=235.2241 K`, WRF `T=234.9055 K`, difference `0.31858 K`. The latent heat implied by the extra vapor deposition is `0.31748 K`, essentially the observed T error. The WRF final state there remains near liquid saturation and ice supersaturated, while JAX drives vapor far lower because it performs liquid saturation adjustment before ice deposition.

Quick evidence: a scratch reorder probe (`clip -> ice_sources -> warm_rain -> saturation_adjustment -> finish`) reduced max `T` error from `0.3186 K` to `0.0842 K` and max `qv` error from `1.12e-4` to `1.43e-5`. Some species errors worsened because this probe still did not implement WRF's simultaneous tendency staging, but it isolates the main thermal/qv error source.

Concrete diagnostic test under 30 min: add a temporary non-committed validation function that stages WRF-like tendencies through the same checkpoints as `module_mp_thompson.F.pre:2917-3273`, moves saturation adjustment after that checkpoint, and compares only `T/qv` on the existing fixture. Passing criterion: max `T` drops below `0.1 K` without broad tolerances.

Estimated fix effort: 10-18 hours for a real staged-tendency refactor; 2-4 hours for an isolation branch proving the order effect.

## 2. WRF lookup tables and moment formulas are replaced by proxies or omitted

Probability contribution to the 0.3 K error: 20-30 percent; higher for hydrometeor partition errors.

Rationale: attempt 3 made the oracle independent, but the JAX kernel still contains proxy physics where the harness runs full WRF tables. Rain collecting cloud water uses WRF `t_Efrw(idx, INT(mvd_c*1e6))` (`module_mp_thompson.F.pre:2260-2268`); JAX uses a bounded linear proxy (`src/gpuwrf/physics/thompson_column.py:340-345`). WRF snow moments use long polynomial moment reconstructions (`module_mp_thompson.F.pre:2090-2192` and `module_mp_thompson.F.pre:3371-3431`); JAX uses `_snow_moment_proxy` (`src/gpuwrf/physics/thompson_column.py:216-226`). WRF graupel number and volume state are derived from `qg/qb/rho_g` and `N0_exp` (`module_mp_thompson.F.pre:1291-1306`, `module_mp_thompson.F.pre:3351-3361`); JAX fixes a simple `ng=4.0e5*rho` proxy (`src/gpuwrf/physics/thompson_column.py:229-237`). WRF rain freezing uses generated tables for the main branch (`module_mp_thompson.F.pre:2658-2664`), while JAX implements only the fallback freeze-all branch (`src/gpuwrf/physics/thompson_column.py:407-415`). WRF cloud-ice deposition partitions positive deposition with `tpi_ide` and can autoconvert ice to snow via `tps_iaus/tni_iaus` (`module_mp_thompson.F.pre:2725-2741`); JAX omits those tables.

Concrete diagnostic test under 30 min: patch a throwaway harness copy to print, per fixture cell, the table indices and values for `t_Efrw`, `tpi_ide`, `tps_iaus`, `tni_iaus`, `t*_qrfz`, plus `smo0/smo1/smof/ng`. Run the existing three scenarios and compare those values to the JAX proxy values. This does not require changing the committed model; it is an oracle-audit print.

Estimated fix effort: 12-24 hours to export/bake the needed WRF tables and replace the warm-rain/freezing/deposition proxies; more if the team wants a general table-loader abstraction.

## 3. Number-concentration handling is wrong for deposition and incomplete for Ns/Ng

Probability contribution to the 0.3 K error: 5-10 percent for first-step T, 60-80 percent for the `Ni/Nr` reported errors.

Rationale: the `Ni` max error is mostly not a thermal root cause; it is a category/number update bug. JAX increases `Ni` during positive ice deposition as `ice_deposition / XM0I` (`src/gpuwrf/physics/thompson_column.py:475`). In WRF, `pni_ide` is set for sublimation when `pri_ide < 0`; for positive deposition WRF partitions mass with `tpi_ide` but does not create number in that line (`module_mp_thompson.F.pre:2719-2727`). That explains million-scale `Ni` growth in levels where WRF keeps `Ni` near the input 200k/kg. `Nr` also diverges where JAX evaporates rain to zero and `_finish` zeroes `Nr` when `qr <= R1` (`src/gpuwrf/physics/thompson_column.py:281-289`), while WRF keeps rain after its staged adjustment.

Concrete diagnostic test under 30 min: run a local monkeypatch that changes the JAX positive-deposition `Ni` increment to zero while leaving mass updates unchanged, then rerun tier-1 comparison. Expected result: `Ni` max error collapses substantially with little first-step `T` change; if mass errors change, the number error is feeding back through `_finish`.

Estimated fix effort: 3-6 hours for the immediate `Ni` deposition correction; 8-16 hours to align `Nc/Ns/Ng/Nr` handling with the WRF staged constraints.

## 4. Sedimentation is only numerically suppressed, not bypassed

Probability contribution to the 0.3 K error: below 5 percent.

Rationale: M5-S1 says sedimentation is out of scope. The harness does not bypass WRF sedimentation; it sets `dz=1.0e30` (`scripts/wrf_thompson_harness.f90:38`) and still calls full `mp_gt_driver` (`scripts/wrf_thompson_harness.f90:67-76`). WRF then runs the sedimentation velocity and flux-divergence path (`module_mp_thompson.F.pre:3653-4003`). Because the flux terms use `1/dz`, this should be negligible in mass, but it is still a semantic mismatch and can affect thresholds/precip accumulators.

Concrete diagnostic test under 30 min: build a patched throwaway WRF source/harness with the sedimentation block bypassed entirely, regenerate only the three fixture outputs, and diff against the current `dz=1e30` fixture. If the fixture-to-fixture diff is below `1e-8` for q and below `1e-4 K`, close this suspect.

Estimated fix effort: 4-8 hours for a clean explicit no-sedimentation harness; less for a throwaway proof patch.

## 5. Precision and initial-state interpretation are secondary, not primary

Probability contribution to the 0.3 K error: below 5 percent.

Rationale: the harness arrays are default Fortran `real` (`scripts/wrf_thompson_harness.f90:11-16`), while JAX enables fp64 (`src/gpuwrf/physics/thompson_column.py:75`). That can move branch thresholds and table indices, but the warm maritime scenario already matches to `2.35e-5 K`, so fp32/fp64 alone is not producing the 0.3 K cold mixed-phase error. Initial interpretation mostly matches: the harness sets `pii=1.0` (`scripts/wrf_thompson_harness.f90:35`), WRF computes `t1d=th*pii` (`module_mp_thompson.F.pre:1245-1246`), and writes `th=t1d/pii` (`module_mp_thompson.F.pre:1394`), so the synthetic `th` is effectively temperature. Density formulas also match (`module_mp_thompson.F.pre:1258`, `src/gpuwrf/physics/thompson_column.py:144-147`, `scripts/m5_generate_thompson_fixture.py:52-55`). Saturation iteration count is not the issue: both WRF and JAX use three Newton iterations (`module_mp_thompson.F.pre:3467-3472`, `src/gpuwrf/physics/thompson_column.py:301-305`).

Concrete diagnostic test under 30 min: cast the JAX state to fp32 on CPU and rerun tier-1 comparison. If max `T/qv` barely moves, retire the precision suspect. Separately, print `input_T`, harness `th`, and WRF `t1d` for one scenario to close the exner/temperature interpretation question.

Estimated fix effort: 2-4 hours if the team wants a fixture parity mode that casts JAX to WRF single precision; otherwise no model fix recommended here.

## Recommended next attempt strategy

Do not accept the current gap as irreducible fp ordering. It is mostly reducible model-order/table parity debt. Open an M5-S1.x exact-parity sprint with a narrow first gate:

1. Rework JAX into WRF-like staged tendencies and move saturation adjustment/rain evaporation to the WRF checkpoints.
2. Fix positive-deposition `Ni` handling immediately because it is isolated and high-signal.
3. Export or bake the WRF tables/moments needed by the active fixture cells, then replace the documented proxies.
4. Only after those pass, run the no-sedimentation patched harness to close the lower-probability sedimentation concern.

Decision: tighten one thing first, specifically the WRF staged tendency/order. The read-only reorder probe already showed most of the `T/qv` error is order-sensitive. Table work should follow because it likely controls the remaining species partition errors. The attempt-3 broad tolerances should stay marked non-final and should not be used as a GO physics-parity claim.

## Read-only commands/proofs used

- Read sprint contract, worker report, harness, generator, JAX code, manifest, ADR-006, and WRF source with `nl -ba`/`rg`.
- Parsed `artifacts/m5/tier1_thompson_parity.json` with `python -m json.tool`.
- Ran read-only JAX CPU probes with `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu` to locate max-error cells and test approximate process reordering. These probes wrote no committed artifacts.

## Handoff

Objective: diagnose origin of attempt-3 Thompson fixture parity errors.

Files changed: `.agent/sprints/2026-05-20-m5-s1-diagnosis/diagnosis-report.md`.

Proof objects produced: this diagnosis report.

Unresolved risks: table/proxy contribution is not yet quantified by exported WRF table values; sedimentation bypass has not been fixture-diffed.

Next decision needed: manager should open M5-S1.x for staged-order parity first, with table export as the second task.
