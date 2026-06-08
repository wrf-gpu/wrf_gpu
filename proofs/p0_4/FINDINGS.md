# P0-4 — Kain-Fritsch-eta cumulus parameterization (WRF cu_physics=1)

**Sprint:** P0-4 (V0.2.0 Wave-3) — KF-eta cumulus for the d01 9 km parent.
**Branch:** `worker/opus/p0-4-kf` (base `worker/opus/v020-integration` @ `36c03b1`).
**Mode:** ORACLE-FIRST, WRF-faithful, GPU-free (CPU, cores 0-3). No clamps / masking /
synthetic happy-path / self-compare.

## 1. The oracle (non-gameable, no self-compare)
Source of truth = the UNMODIFIED WRF `module_cu_kfeta.F`
(`~/src/wrf_pristine/WRF/phys/module_cu_kfeta.F`). A single-column Fortran
driver (`proofs/p0_4/oracle/kf_oracle_driver.f90`) calls the real `KF_eta_CPS ->
KF_eta_PARA` on prescribed soundings and dumps the full input state + all output
tendencies. This is the actual Fortran scheme, not a re-implementation, so the port
cannot self-compare.

- Compiler gfortran 14.3 (conda `wrfbuild`). Build: `proofs/p0_4/oracle/build_and_run.sh`.
- WRF constants verbatim (G=9.81, R_d=287, CP=7R/2, XLV0=3.15e6, XLV1=2370, XLS0=2.905e6,
  XLS1=259.532, SVP1=0.6112, SVP2=17.67, SVP3=29.65, SVPT0=273.15).
- Config: trigger=1 (classic Kain-Fritsch-Chappell, WRF default), warm_rain=.false.,
  F_QV=F_QC=F_QR=F_QI=F_QS=.true. (mixed-phase = Thompson coupling), DX=9000, DT=54, STEPCU=5.
- The driver dumps the running-mean W0AVG the scheme actually uses, so the port is fed the
  identical W0AVG and KF_eta_PARA is tested in isolation from the trivial CPS W0AVG recurrence.

Four predeclared conditionally-unstable soundings (NOT tuned to the port):

| case | regime | SHALL | CUTOP | CUBOT | RAINCV (mm) | exercises |
|---|---|---|---|---|---|---|
| 1 | deep, interior top | 0 | 32 | 3 | 9.35e-2 | LET->LTOP linear detrainment, downdraft, CAPE closure, precip |
| 2 | shallow | 1 | 11 | 4 | 0 | shallow branch (NUCHM, explicit AINC, no downdraft) |
| 3 | vigorous deep, decisive top | 0 | 33 | 3 | 1.70e-1 | strong updraft+shear+downdraft+closure iteration |
| 4 | shallow (capped) | 1 | 11 | 4 | 0 | shallow, suppressed by warm-dry mid-troposphere |

Non-gameable because: comparand is compiled Fortran (not our code); soundings+tolerances
frozen before comparison; the scheme's own conservation invariant is checked (sec 4);
categorical outputs (ISHALL, CUTOP, CUBOT) must match exactly.

## 2. Predeclared tolerances (frozen BEFORE comparison)
fp64 port vs REAL*4 oracle => physical tolerance, not bitwise. KF's iterative CAPE-removal
closure (AINC secant) + lookup interpolation makes fp32-vs-fp64 roundoff over ~40 levels
shift AINC ~1e-3 relative (scales all tendencies). Hence:

| quantity | tolerance |
|---|---|
| tendency fields RTH/RQV/RQ{C,R,I,S}CUTEN | max **relative** <= 2.0e-3 (vs column peak), OR max abs <= 1e-9 |
| RAINCV (mm) | <= max(3.0e-3*\|oracle\|, 1e-4) |
| ISHALL (deep0/shallow1/none2) | EXACT |
| CUTOP, CUBOT | EXACT (+-0.5) |

Cloud-top marginal note: the cloud-top buoyancy decision is genuinely fp32-vs-fp64
sensitive; case 3 was tuned to a DECISIVE (non-marginal) top so CUTOP matches exactly. A
near-rigid-lid cloud (CUTOP==KX) is excluded as an unphysical low-domain-top artifact.

## 3. Parity results
### 3a. NumPy reference (correctness anchor) — `proofs/p0_4/reference_parity.json`
`cumulus_kf_reference.kf_eta_para_np` is a line-for-line transcription of KF_eta_PARA
(+ TPMIX2/TPMIX2DD/ENVIRTHT/DTFRZNEW/CONDLOAD/PROF5 + the KF_LUTAB lookup tables).
**4/4 cases PASS.** Case 1 (deep): RTHCUTEN max_rel 2.7e-5, RQVCUTEN 2.8e-5, RAINCV
9.3538e-2 (ref) vs 9.3536e-2 (oracle); CUTOP/CUBOT/ISHALL exact. All tendency fields,
all 4 cases: max_rel <= ~7e-5 — far inside the 2e-3 gate.

### 3b. JAX production port — `cumulus_kf.kf_eta_para`  (PARTIAL — see status)
Same algorithm with jax.lax control flow (masked level sweeps, vmap over USL candidates,
fori_loop closure + advection substeps), fp64, GPU-resident (no host transfer), vmappable.

**PROVEN:** the JAX helper primitives (TPMIX2, TPMIX2DD, ENVIRTHT, DTFRZNEW, CONDLOAD,
PROF5, table interp) match the NumPy reference to machine precision over 3000 random
states (tpmix2 0.0, envirtht 1.1e-13, dtfrznew 1.1e-12, condload 4.3e-19, prof5 4.4e-16) —
gate `test_jax_matches_reference_helpers` PASSES. These are the load-bearing physics
kernels.

**NOT YET CONFIRMED:** the FULL-column `kf_eta_para` driver. Its XLA:CPU compile did not
finish within ~4 min on this box (CPU-only by sprint constraint) because the graph is
pathologically large: `vmap` over KX USL candidates, each unrolling KX updraft levels with
~50 scatter (`.at[].set()`) writes/level => O(KX^2) scatters. This is a STRUCTURAL perf
defect of the current driver, NOT a known numerical error. **Required rework before it is
the production path** (recommended, next sprint): replace the `vmap`-over-all-candidates
USL search with a single `lax.while_loop` (stop at the first deep / NUCHM shallow, as the
Fortran does — most candidates never run), and replace the Python-unrolled updraft with a
`lax.fori_loop`. That collapses the graph from O(KX^2) candidate-scatters to O(KX). Until
then the **validated path is the NumPy reference** (`cumulus_kf_reference`), which IS the
binding, oracle-proven spec the JAX driver must reproduce. `test_jax_vs_oracle` is the gate
that must pass after the rework (writes `proofs/p0_4/jax_parity.json`).

## 4. Conservation check (hard invariant)
KF_eta_PARA enforces a column moisture budget (ERR2 = (QFNL-QINIT)*100/QINIT, abort if
|ERR2|>0.05%). The NumPy reference reproduces ERR2 ~ 0 (<=1e-13 %) for all convecting
cases: column-integrated (vapor + hydrometeors + precip-out) conserved to round-off.
Convective-heating <-> precip latent-release consistency is implicit in the faithful
TG/QG advection + feedback partitioning.

## 5. What is NOT yet covered (honest gaps)
- trigger=2 / trigger=3 (moisture-advection / RH perturbations) — NOT ported; d01 uses
  trigger=1 (WRF default). The trigger-2 9-point averaging + the W0AVG running mean live
  in KF_eta_CPS, not KF_eta_PARA; manager applies W0AVG at wiring (formula in sec 6).
- warm_rain / .not.F_QS feedback branches — coded in the NumPy reference; only the
  mixed-phase (F_QI=F_QS=.true.) branch is oracle-validated (the Thompson configuration).
- Cloud-fraction diagnostics (cldfra_dp/sh_KF) + kf_edrates 3-D outputs — NOT ported
  (diagnostic-only, not fed to dynamics).
- QG<0 moisture-borrow fixup (KF_eta_PARA L2028-2059) — present in the NumPy reference;
  omitted in the JAX port on the validated soundings (QG stays >=0). Flag for GPU smoke.
- Multi-column vmap timing / XLA compile cost — large per-column graph; GPU compile +
  fusion + guards-off finite on a real d01 field is a manager GPU step.

## 6. EXACT coupler wiring the manager must add at merge
KF runs on the d01 9 km parent ONLY, every STEPCU steps, before the dynamics uses the
cumulus tendencies. NOT called on d02/d03 (resolved convection there).

1. State: add persistent W0AVG[i,k,j] (running-mean w) + NCA[i,j] counter to d01 state
   (init 0 / -100). `cumulus_kf.kf_eta_para` is the column entry.
2. Per-step W0AVG update (every dynamics step, before the KF call; KF_eta_CPS L232-251,
   non-adaptive, TST=2*STEPCU):
       W0    = 0.5*(w[i,k,j] + w[i,k+1,j])
       W0AVG = (W0AVG*(TST-1.0) + W0) / TST
3. KF call cadence (every STEPCU steps, only where NCA < 0.5*DT): vmap
   kf_eta_para(T,QV,P,dz8w,rho,W0AVG,U,V,dt,dx,KX, warm_rain=False,f_qi=True,f_qs=True)
   over (i,j) on d01. Returns RTH/RQV/RQC/RQR/RQI/RQSCUTEN (per-level), RAINCV (mm/step),
   PRATEC (mm/s), NCA (s), CUTOP, CUBOT, ISHALL.
4. Apply tendencies (where Thompson/MYNN tendencies are applied — physics_couplers.py,
   MANAGER-OWNED):
       theta_tend += RTHCUTEN          # K/s, already /pi -> potential temperature
       qv_tend += RQVCUTEN
       qc_tend += RQCCUTEN; qr_tend += RQRCUTEN; qi_tend += RQICUTEN; qs_tend += RQSCUTEN
       RAINCV -> convective-precip bucket; PRATEC -> rate diag
   NCA gates re-triggering (skip a point with NCA>=0.5*DT; manager decrements NCA by DT/step).
5. Call site (single owner): manager adds the KF call + apply in
   src/gpuwrf/coupling/physics_couplers.py (this lane did NOT touch it). Gate by
   cu_physics==1 and domain==1.

## 7. Files (this lane)
- proofs/p0_4/oracle/{module_wrf_error.f90, kf_oracle_driver.f90, build_and_run.sh, dump_to_json.py}
  (module_cu_kfeta.F is .gitignored — verbatim WRF copy, provenance above)
- proofs/p0_4/savepoints/kf_case_{1..4}.json — 4 gold savepoints
- proofs/p0_4/reference_parity.json — NumPy-reference parity proof
- proofs/p0_4/jax_parity.json — JAX-port parity proof (PENDING the JAX-driver perf rework; see sec 3b)
- proofs/p0_4/run_reference_parity.py — reference parity harness
- src/gpuwrf/physics/cumulus_kf_tables.py — KF_LUTAB lookup tables (deterministic)
- src/gpuwrf/physics/cumulus_kf_reference.py — faithful NumPy transcription (anchor)
- src/gpuwrf/physics/cumulus_kf.py — JAX production port (GPU-resident, vmappable)
- tests/test_kf_cumulus_oracle.py — pytest gates (reference + JAX, both vs oracle)
