# Opus v0.6.0 GF + Tiedtke GPU-batch + scan-wire handoff

## Objective

Close the rows-1-9 completeness gap flagged by the GPT audit: Grell-Freitas
(`cu_physics=3`) and modified-Tiedtke (`cu_physics=6`) existed only as faithful
host-NumPy *reference* ports (WRF-oracle-passing but `jit_vmap_native_kernel=false`,
not in the GPU scan). GPU-batch the column physics (jit/vmap), wire into the GPU
scan mirroring KF, and re-prove WRF-oracle parity on the batched device path.

## Verdict

- **Tiedtke (cu=6): DONE.** GPU-batched jit/vmap kernel, scan-wired, parity re-proven
  vs the unmodified-WRF oracle. `gf_tiedtke=PASS`.
- **Grell-Freitas (cu=3): NOT DELIVERED (honest carry-over).** A WRF-faithful
  jit/vmap rewrite of the GF closure is a multi-thousand-LOC re-derivation that this
  lane could not complete without high risk of silently breaking the (already-passing)
  WRF parity. Reported honestly per the no-fudge rule; the CPU-reference GF gate
  remains intact and PASSING; cu=3 stays fail-closed in the scan.

## Tiedtke — what was done

- **NEW `src/gpuwrf/physics/cumulus_tiedtke_jax.py`** (~900 LOC): a line-faithful
  *functional* port of the NumPy reference's `_cumastr_new` (WRF CUMASTRN) and its
  subroutines CUINI / CUBASE / CUBASMC / CUENTR / CUASC / CUDLFS / CUDDRAF / CUFLX /
  CUDTDQ / CUDUDV. The WRF algorithm is preserved EXACTLY — only the Python control
  flow and in-place NumPy mutation are replaced:
  - every vertical level sweep -> `jax.lax.fori_loop` (fixed trip count = KLEV);
  - every Fortran `IF(loflag)…CYCLE` / data-dependent branch -> `jnp.where` mask
    (no branch of the source is dropped);
  - in-place `_cuadjtq(kk)` -> a pure functional one-level update returning `(t,q)`;
  - carried scalars (`kcbot`/`kctop`/`ldcum`/`pmfub`/`zmfuu`/`zbuoy_acc`/…) thread
    through the loop carry.
  - The whole column traces to ONE XLA graph and `vmap`s across `(ny*nx)` columns
    with zero host transfer.
- `tiedtke_column_jax` + `step_tiedtke_column_jax` (frozen `PhysicsStepResult`).

### Tiedtke parity proof — `proofs/v060/tiedtke_gpubatch_savepoint_parity.json`

- `verdict: PASS`. Oracle = unmodified WRF `module_cu_tiedtke.F`
  (sha256 `3514aaaa…566999`, verified against the live source), single-column
  Fortran driver; **savepoints reused bit-identically** from the CPU-reference lane
  (`gf`/`tiedtke_case_*` are real WRF gold, not regenerated, not a self-compare).
- The JAX kernel reproduces the validated NumPy reference to **machine precision**
  (worst field abs ~1e-15; KTYPE exact on all 5 regimes), so it inherits the
  reference's WRF-faithfulness exactly.
- **Tolerance policy (honest):** the WRF oracle ran in REAL*4. The prior
  (GPT-reviewed) lane established the float32-vs-fp64 floor on the shallow-regime
  RTHCUTEN at ~4.2e-3 relative (at a tiny ~5e-8 absolute). Tightening the tendency
  gate below that floor would manufacture a false FAIL on a genuinely-equivalent
  result, so the tendency gate is held at the WRF-faithful 5e-3. The REAL tightening
  this lane adds: (1) `raincv` gate tightened 2x (1e-3 -> 5e-4) — RAINCV is the
  load-bearing convective output and the kernel matches it far inside that; (2) a
  NEW bit-level `batched_vs_single_abs=1e-12` invariant proving the `vmap`
  device-batch path is the SAME computation as the single column (measured 1.15e-15).
- Worst per-field residual vs oracle: RTHCUTEN 4.234e-3 (case-2 shallow, REAL*4
  floor) ; RQVCUTEN 8.2e-4 ; RQICUTEN 4.6e-4 ; RUCUTEN/RVCUTEN 2.9e-4 ; all PASS.

### Tiedtke scan wiring (mirrors KF)

- `coupling/scan_adapters.py`: `tiedtke_adapter` (vmap over columns, plain
  `State -> State`, no persistent carry); `CU_SCAN_ADAPTERS[6]=tiedtke_adapter` +
  `CU_STATELESS_SCAN_ADAPTERS={6:…}`. Tiedtke-specific inputs (P8W interface
  pressure, ZNU eta proxy, QFX from the B2 `qv_flux` handle; QVFTEN/QVPBLTEN=0 in
  the per-slot scan) assembled pure-`jnp`, no host transfer. RUCUTEN/RVCUTEN applied
  A2C onto the C-grid faces like the PBL adapter.
- `runtime/operational_mode.py`: cumulus dispatch routes the stateless Tiedtke vs
  the carry-threaded KF; `_SCAN_WIRED_OPTIONS["cu_physics"]=(0,1,6)`; cu=6 dropped
  from `_SCAN_UNWIRED_REASON`; KF carry only seeded for the stateful adapter.
- `proofs/v060/multicfg_operational_smoke.py`: `cu_tiedtke` is now a **RUN** config —
  end-to-end through the operational coupler it `compiles/finite/physical/active`.
  Regenerated `multicfg_smoke_report.json`: **15/15 RUN PASS**, **1/1 FAIL_CLOSED OK**
  (GF cu=3 correctly rejected).

## Grell-Freitas — honest non-delivery

`proofs/v060/gf_gpubatch_savepoint_parity.json` (`verdict: NOT_DELIVERED`,
scope = the GPU-batched kernel only; the CPU reference still PASSES the oracle).
Concrete blockers (all documented in the proof object): 170 control-flow sites,
118 explicit loops, 3 `while` loops incl. a data-dependent `while True` cap-increment
search in `cup_kbcon`, 20 `break`s, data-dependent `math.gamma(alpha)` in the
beta-PDF `get_zu_zd_pdf_fim` (4 draft types), the 16-member `cup_forcing_ens_3d`
closure ensemble, iterative downdrafts, and pervasive `ierr` short-circuiting across
~21 subroutines. Each is individually convertible (gamma -> `jax.scipy.special.gamma`;
break-searches -> masked `argmax`; while-true -> bounded `fori_loop` + mask; `ierr` ->
`jnp.where`), but the aggregate is a ~1500-2000 LOC faithful re-derivation — a
dedicated sprint, NOT a lane sub-task. The prior Opus GF handoff reached the same
conclusion. cu=3 stays ACCEPTED in the frozen S0 contract (physics_registry +
namelist_check) but FAIL-CLOSED in the scan (loud rejection, never a silent no-op).
A predeclared 2e-2 tolerance is recorded so any future GF vmap kernel is gated
honestly. **Not a v1.0.0 blocker: KF(1) + Tiedtke(6) cover the operational cumulus menu.**

## Frozen-contract note

cu_physics=3 and =6 were ALREADY accepted in `physics_registry.CU_SCHEMES` /
`ACCEPTED_CU_PHYSICS` and `io/namelist_check` (S0 frozen contract). This lane did
NOT extend the accept-matrix; it only changed the SCAN-WIRING (`_SCAN_WIRED_OPTIONS`):
cu=6 moved from unwired -> scan-wired; cu=3 unchanged (still unwired/fail-closed).

## Commands run (all `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3`)

- `python proofs/v060/run_tiedtke_gpubatch_parity.py --fail-on-parity-fail` -> exit 0, PASS.
- `python proofs/v060/multicfg_operational_smoke.py --steps 4` -> 15/15 RUN PASS, 1/1 FAIL_CLOSED OK.
- `pytest -q tests/test_tiedtke_cumulus_oracle.py tests/test_grell_freitas_cumulus.py` -> 5 passed (reference untouched).
- `assert_registry_consistent()` -> OK.

## Files changed

- NEW `src/gpuwrf/physics/cumulus_tiedtke_jax.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `proofs/v060/multicfg_operational_smoke.py` (+ regenerated `multicfg_smoke_report.json`)
- NEW `proofs/v060/run_tiedtke_gpubatch_parity.py`
- NEW proofs: `proofs/v060/tiedtke_gpubatch_savepoint_parity.json`,
  `proofs/v060/gf_gpubatch_savepoint_parity.json`

## Unresolved risks / honest caveats

- **GF cu=3 not GPU-batched** (above). KF + Tiedtke cover the menu; not a blocker.
- The Tiedtke adapter's `QVFTEN/QVPBLTEN=0` and `ZNU` eta-proxy are reasonable
  per-slot assemblers; the savepoint parity (the binding equivalence object) does
  NOT depend on them (it uses the real WRF inputs). A full coupled wrf.exe gate
  would exercise the assemblers — that is the v0.4.0 forecast-gate's job, not a
  per-scheme column gate.
- Oracle is the real WRF *module* (not a full wrf.exe run): `full_wrf_exe=false`,
  appropriate for a per-scheme column-equivalence gate.
- No GPU profiler artifact attached; the claim is GPU-RUNNABILITY (jit/vmap-traceable,
  zero host transfer in the column), not a measured speedup (`gpu_performance_claim=false`).
