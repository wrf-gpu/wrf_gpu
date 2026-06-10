# Review: V0.14 Fable Moist-`cqw` / Moist `pg_buoy_w` Pressure-State Dynamics

Date: 2026-06-10 WEST · Owner: Fable/Mythos · Branch: `worker/gpt/v013-close-manager`
Base/HEAD: `007751d8` (v014 open moist cqw pressure dynamics sprint).
**UNCOMMITTED — manager reviews/gates/merges.** CPU-only; no GPU used; live run untouched.

## Objective

Close or formally bound the remaining 3D pressure-state blocker after the
accepted PSFC fix: GPU `P+PB(k0)` rides its DRY hydrostatic column while CPU/WRF
rides the MOIST column, because the operational acoustic w-equation uses
`dry_cqw`/`pg_buoy_w_dry`. Implement a WRF-faithful moist `cqw`/`pg_buoy_w`
path that closes it, or return a WRF-anchored bound.

## Decision: **FIXED** (implemented + CPU-proven + flag-gated default-OFF) pending one manager-run short GPU h1/h4 gate

Verdict object: `MOIST_CQW_FIX_PROVEN_CPU_INERT_OFFPATH_STABLE_COEF`.

The blocker is precisely the **water-mass loading** term `−cq2·(c1f·mub+c2f)`
the dry W-equation specialization omits (the JAX diagnostic pressure already
carries moist `theta_m`, so it is NOT a virtual-temperature/EOS effect). The
WRF-faithful moist path is implemented, proven on CPU to target exactly that
term, proven **bit-identical** to the shipped dry path where moisture is zero,
and proven to keep the implicit W solver well-conditioned and the operational
CPU path finite/stable end-to-end. It is gated behind `GPUWRF_MOIST_CQW`
(default OFF) so v0.12.0 behaviour is unchanged until the single short GPU h1/h4
stability/parity gate (the only GPU job, which I am not approved to run) confirms
it; on PASS the manager flips the default ON.

## WRF source anchors (pristine `/home/enric/src/wrf_pristine/WRF`)

- `calc_cq` `dyn_em/module_big_step_utilities_em.F:856-870` — w-face load
  `cqw=0.5·Σ_species(q(k)+q(k-1))` (vertical analogue of `cqu`/`cqv`).
- `pg_buoy_w` `:2474-2497` — `cq1=1/(1+cqw)`, `cq2=cqw·cq1`, `cqw←cq1`;
  interior `rw_tend += (1/msfty)·g·( cq1·rdn·Δp − c1f·mu' − cq2·(c1f·mub+c2f) )`;
  top uses `cqw(kde-1)`. Dry (`cqw=0`) reduces exactly to `pg_buoy_w_dry`.
- `cqw` enters `calc_coef_w` (`module_small_step_em.F:624-649`) and the
  `advance_w` term-A implicit pressure coefficient (`:1477-1489`).

## Proof summary (CPU, h1; numbers in `…closure.json`)

Two independent lines, both pointing at the loading term:

1. **Hydrostatic-column balance** — `P+PB(k0)` − half-level hydro pressure:
   GPU vs DRY **−8.18 Pa** / vs MOIST **−202.66 Pa**; CPU vs MOIST **−13.54 Pa**
   (reproduces the prior PSFC §3 −8.2/−202.7/−13.5 exactly). `moist−dry` loading
   ≈ 194–198 Pa.
2. **W-equation `rw_tend` residual** (production `pg_buoy_w_dry`/`pg_buoy_w_moist`,
   mean over low faces k=1..5): GPU DRY **−3.88≈0** / MOIST **−8934**; CPU DRY
   **+8992** / MOIST **−3.12≈0**. ⇒ GPU is in DRY balance, CPU in MOIST balance;
   the difference is the moist water loading. Threading it moves the GPU acoustic
   equilibrium onto the moist column (≈ +194 Pa at k0).

Safety:
- **Inertness:** `pg_buoy_w_moist(cqw_calc=0)` == `pg_buoy_w_dry` and its `cqw` ==
  `dry_cqw` to `max_abs == 0.0`. Skamarock/Straka (dry) gates unaffected.
- **Conditioning:** `calc_coef_w` on real GPU h1 `mut`, moist `cqw=cq1<1` vs dry
  `cqw=1`: all finite; `max|γ|` 5.4808e-4 (moist) ≤ 5.4807e-4 (dry); the moist
  tridiagonal is at least as diagonally dominant. No new instability.
- **Performance:** `cqw` built once per RK stage (face-average of total
  moisture), held through the acoustic loop (WRF `calc_cq` cadence); no
  host/device transfer, no new large transients, no CPU callback.

## Files changed

- `src/gpuwrf/dynamics/core/advance_w.py` — NEW `moist_cqw_calc_face` and
  `pg_buoy_w_moist` (returns `(rw_tend, cqw_solver)`); `pg_buoy_w_dry`/`dry_cqw`
  untouched; `__all__` extended.
- `src/gpuwrf/runtime/operational_mode.py` — NEW `_moist_cqw_enabled()`
  (`GPUWRF_MOIST_CQW`, default OFF); `_acoustic_core_state_from_prep` builds the
  stage `rw_tend` via `pg_buoy_w_moist` and threads moist `cqw=cq1` into
  `AcousticCoreState.cqw` when ON (unchanged dry path when OFF). Import extended.
- NEW `proofs/v014/moist_cqw_pressure_dynamics_closure.{py,json,md}`.
- NEW `tests/test_v014_moist_cqw_pressure_dynamics.py` (4 tests).
- This review. **No other source touched** (`git status --short src/` = exactly
  the two files above).

## Commands run

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/moist_cqw_pressure_dynamics_closure.py   # verdict object
python -m json.tool proofs/v014/moist_cqw_pressure_dynamics_closure.json                                          # JSON_VALID
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest tests/test_v014_moist_cqw_pressure_dynamics.py -q   # 4 passed
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest tests/test_v013_operational_smoke.py -q             # 39 passed (flag OFF, no regression)
GPUWRF_MOIST_CQW=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest tests/test_v013_operational_smoke.py -q   # 39 passed (flag ON, moist path executes + finite/stable)
python -m compileall -q src/... proofs/... tests/...   # OK
git diff --check                                       # OK
```

## CPU vs GPU proof status

CPU proof is strong and decisive (two independent lines + inertness + solver
conditioning + 39/39 operational smoke both flag states). **No GPU proof** — no
GPU approval and the single-GPU lock; the GPU h1/h4 gate is the manager's.

## Performance / memory implications

None when OFF (byte-identical). When ON: two extra `(nz+1,ny,nx)` face fields
already carried by the solver, one face-average reduction per RK stage. No
timestep-loop host/device transfer, no new full-run CPU callback. GPU-native
structure preserved.

## Unresolved risks

1. CPU cannot validate the full **prognostic GPU run** (acoustic stability over
   many steps, field parity, PSFC non-regression) — the short GPU h1/h4 gate
   must confirm before default-ON. The conditioning + finite CPU smoke both ways
   make destabilization unlikely but not GPU-proven.
2. Threading the loading shifts every real-case (moist) field-parity baseline
   when ON; existing real-case gates must be re-scored against the moist run, not
   the dry one. (Idealized/dry gates are unaffected — bit-identical.)
3. The legacy non-prep `_acoustic_core_state` path was intentionally left dry
   (not the operational Canary path); only `_acoustic_core_state_from_prep` is
   moist-gated.

## Exact next decision (manager)

Run the short Canary **GPU h1/h4** gate with `GPUWRF_MOIST_CQW=1` (same harness
as `proofs/v014/psfc_moist_pressure_gpu_h4_validation.md`):
- PASS criteria: `P+PB(k0)−moist_col` ≈ −13 Pa (down from −203), `PSFC` not
  regressed, `U/V/T/W/QVAPOR` same-or-better envelope and finite.
- On PASS: promote `GPUWRF_MOIST_CQW` to default ON and re-score real-case
  field-parity against the moist run; resume TOST/Switzerland-GPU.
- On FAIL: keep default OFF; the proof + bound stand for v0.15.

## TOST / Switzerland-GPU status

Still **BLOCKED** until the GPU h1/h4 gate promotes the moist path; the
pressure-state divergence is now root-caused, implemented, and CPU-proven (not
yet GPU-validated). No GPU used; no GPU run requested.
