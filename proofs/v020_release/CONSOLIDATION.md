# v0.2.0 Release Consolidation — `worker/opus/v020-release`

**Date:** 2026-06-02
**Consolidation engineer:** Opus 4.8 (xhigh), manager-dispatched
**Branch:** `worker/opus/v020-release`
**Base:** `worker/opus/v020-tost-daytimefix` @ `c950c58` (equivalence-validated forecast: L1 + Thompson + nroot + MYNN-EDMF infra [edmf=OFF default]; dycore close-gate 2/2 PASS)
**Release tip:** `b900bb5`

## Objective
Build the v0.2.0 tag-ready branch by merging the four remaining validated, file-disjoint lanes onto the TOST-validated forecast base, without altering the d02/d03 replay forecast (so the TOST stays valid) and without wiring the inert hooks (so the release default == the TOST-validated behavior).

---

## 1. Merge results — ALL CLEAN, ZERO CONFLICTS

Lanes merged in order onto `worker/opus/v020-release` (created from `worker/opus/v020-tost-daytimefix`):

| Order | Lane branch | Merge commit | Result |
|------|-------------|--------------|--------|
| 1 | `worker/gpt/p0-7-conservation` | `3bbdb5d` | clean (ort), 4 files, +767 |
| 2 | `worker/gpt/p0-4-kf-jax` | `1a88963` | clean (ort), 18 files, +6227 |
| 3 | `worker/opus/p0-1a-nesting` | `c65996e` | clean (ort), 9 files, +2060 |
| 4 | `worker/opus/p0-5-io` | `b900bb5` | clean (ort), 11 files, +1876/−11 |

**Pre-merge disjointness check:** the union of files changed by the four lanes (vs their merge-base with the TOST base) contains **no file touched by more than one lane** (`git diff --name-only … | sort | uniq -d` → empty). The merges were therefore expected clean and were clean.

### `src/` files added/changed vs the TOST base (full list)
```
src/gpuwrf/diagnostics/conservation_budget.py   (new, p0-7)
src/gpuwrf/physics/cumulus_kf.py                 (new, p0-4 — JAX production port, vmappable)
src/gpuwrf/physics/cumulus_kf_reference.py       (new, p0-4 — NumPy anchor)
src/gpuwrf/physics/cumulus_kf_tables.py          (new, p0-4 — KF_LUTAB)
src/gpuwrf/nesting/__init__.py                   (new, p0-1a)
src/gpuwrf/nesting/boundary_construction.py      (new, p0-1a)
src/gpuwrf/nesting/interp.py                     (new, p0-1a)
src/gpuwrf/nesting/scheduler.py                  (new, p0-1a)
src/gpuwrf/io/__init__.py                         (+2 lines: additive export OPERATIONAL_WRFOUT_VARIABLES)
src/gpuwrf/io/restart.py                          (new, p0-5)
src/gpuwrf/io/wrfout_writer.py                    (extended, p0-5 — +367 new field specs/writers, additive)
```
**No dycore / coupling / operational-scan / daily_pipeline / dynamics / runtime file is touched.**
(`git diff --name-only worker/opus/v020-tost-daytimefix HEAD -- src/ | grep -iE 'dycore|coupl|operational|runtime|daily_pipeline|rk_scan|dynamics'` → NONE.)

---

## 2. Inherited end-gate (NOT re-run — by design)

The **dycore idealized close-gate (2/2 PASS)** and the **d02/d03 replay forecast behavior** are **inherited UNCHANGED** from `worker/opus/v020-tost-daytimefix`. Justification (structural, not re-measured):

- None of the four merges touches the dycore, the physics couplers, the operational scan (`_rk_scan_step` / `_physics_boundary_step_*`), `daily_pipeline.py`, or `operational_mode.py` (verified by the `git diff --name-only` filter above).
- The four lanes add **new, un-wired modules** only (`diagnostics/`, `nesting/`, `physics/cumulus_kf*`, `io/restart`) plus an **additive** extension to `io/wrfout_writer.py` and a **2-line additive export** in `io/__init__.py`.
- Because the forecast code path is byte-for-byte identical to the TOST base, the in-flight d02/d03 MAM GPU TOST run (owned by the parallel GPU job) and the dycore close-gate both apply to this branch as-is.

Per the contract, the GPU is owned by the parallel TOST run; this lane is **GPU-FREE** and the close-gate was **not** re-executed.

---

## 3. CPU sanity — ALL GREEN (`JAX_PLATFORM_NAME=cpu`, `taskset -c 0-3`, `PYTHONPATH=src`)

> Note: this is a worktree; the editable `gpuwrf` on `sys.path` resolves elsewhere, so all CPU checks were run with `PYTHONPATH=src` to bind to **this worktree's** merged source. Confirmed `gpuwrf.__file__` resolves into this worktree.

### Import smoke
```
JAX_PLATFORM_NAME=cpu PYTHONPATH=src taskset -c 0-3 python -c "import gpuwrf"   → OK
```
All merged-lane modules import cleanly:
`gpuwrf.diagnostics.conservation_budget`, `gpuwrf.physics.cumulus_kf{,_reference,_tables}`,
`gpuwrf.nesting{,.boundary_construction,.interp,.scheduler}`, `gpuwrf.io.{wrfout_writer,restart}` → all import; `compute_conservation_budget` and `kf_eta_para` present.

### Unit tests (each merged lane)
```
tests/test_conservation_budget.py        2 passed
tests/test_kf_cumulus_oracle.py          6 passed
tests/test_p0_1a_nesting.py             11 passed
tests/test_p0_5_restart_full_carry.py + tests/test_m7_netcdf_writer.py   8 passed, 1 skipped
─────────────────────────────────────────────────────────────────────────────
Combined run of all five files:        27 passed, 1 skipped in 3.82s
```

**The 1 skip is pre-existing and unrelated to the merge:** `tests/test_m7_netcdf_writer.py:129 — "Gen2 reference wrfout unavailable"` (a reference-data-availability skip in this worktree; netCDF4 1.7.4 is installed and the writer tests themselves pass). **No new failures.**

---

## 4. UN-WIRED HOOK SPECS (collected for the manager — DO NOT auto-wire)

Keeping these inert keeps the release default == the TOST-validated forecast. These are the careful manager follow-ups. Each spec is reproduced from the lane's own FINDINGS so the manager has one place to act from.

### 4.1 KF cumulus coupler activation (P0-4, FINDINGS §6)
Source: `proofs/p0_4/FINDINGS.md` §6. KF runs on the **d01 9 km parent ONLY**, every `STEPCU` steps, before dynamics consumes the cumulus tendencies. **NOT** called on d02/d03 (resolved convection). Owner file for the wiring: `src/gpuwrf/coupling/physics_couplers.py` (this lane did NOT touch it). Gate by `cu_physics==1 AND domain==1`.

1. **State:** add persistent `W0AVG[i,k,j]` (running-mean w) + `NCA[i,j]` counter to d01 state (init `0` / `-100`). `cumulus_kf.kf_eta_para` is the column entry point.
2. **Per-step `W0AVG` update** (every dynamics step, before the KF call; `TST = 2*STEPCU`, non-adaptive):
   ```
   W0    = 0.5*(w[i,k,j] + w[i,k+1,j])
   W0AVG = (W0AVG*(TST-1.0) + W0) / TST
   ```
3. **KF call cadence** (every `STEPCU` steps, only where `NCA < 0.5*DT`): vmap
   `kf_eta_para(T,QV,P,dz8w,rho,W0AVG,U,V,dt,dx,KX, warm_rain=False,f_qi=True,f_qs=True)` over (i,j) on d01. Returns `RTH/RQV/RQC/RQR/RQI/RQSCUTEN` (per-level), `RAINCV` (mm/step), `PRATEC` (mm/s), `NCA` (s), `CUTOP`, `CUBOT`, `ISHALL`.
4. **Apply tendencies** (where Thompson/MYNN tendencies are applied — `physics_couplers.py`, MANAGER-OWNED):
   ```
   theta_tend += RTHCUTEN          # K/s, already /pi -> potential temperature
   qv_tend    += RQVCUTEN
   qc_tend += RQCCUTEN; qr_tend += RQRCUTEN; qi_tend += RQICUTEN; qs_tend += RQSCUTEN
   RAINCV -> convective-precip bucket; PRATEC -> rate diag
   ```
   `NCA` gates re-triggering (skip a point with `NCA >= 0.5*DT`; manager decrements `NCA` by `DT` each step).
5. **Call site (single owner):** manager adds the KF call + apply in `physics_couplers.py`, gated by `cu_physics==1 and domain==1`.

> Honest status carried from the lane (FINDINGS §3b/§5): the NumPy reference port passes oracle parity; the JAX production driver is structurally complete and compiles, with `jax_parity.json` still **PENDING** the JAX-driver perf rework. Wiring KF is therefore both an activation step *and* gated on closing that JAX-parity proof — manager should not activate it into the operational default until §3b parity is green.

### 4.2 Conservation scan-carry (`ConservationDiagnosticsCarry`) (P0-7)
Source: `.agent/reviews/2026-06-02-gpt-p0-7-conservation.md` §"Manager Wiring Spec". The standalone module (`conservation_budget.py`) is **not** the operational proof until the manager wires the scan diagnostics. Do this in the **L1-owned runtime consolidation**, not in a lane branch. No in-loop host/device transfer needed (all carry fields scalar/small fixed arrays).

Add a device-resident carry alongside `OperationalCarry`, diagnostic-entry-point only:
```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ConservationDiagnosticsCarry:
    dry_mass_lbc_flux_kg: jax.Array      # scalar fp64, positive into domain
    water_lbc_flux_kg:    jax.Array      # scalar fp64, positive into domain
    mse_lbc_flux_j:       jax.Array      # scalar fp64, positive into domain
    qfx_accumulated_kg:   jax.Array      # scalar fp64, upward QFX source
    precip_step_kg:       jax.Array      # shape (steps,), optional proof trace
    guard_count:          jax.Array      # int64[n_guard_terms]
    guard_sum_magnitude:  jax.Array      # fp64[n_guard_terms]
    guard_max_magnitude:  jax.Array      # fp64[n_guard_terms]
    guard_signed_magnitude: jax.Array    # fp64[n_guard_terms]
```
Use a stable `GUARD_TERM_INDEX` tuple (≥26 terms): `theta_positive_definite_increment_limiter`, `dynamics_mu_guard`, `dynamics_q{v,c,r,i,s,g}_guard`, `boundary_{u,v,w,theta}_finite_or_origin`, `boundary_qv_guard`, `boundary_{p,ph,p_total,ph_total,p_perturbation,ph_perturbation}_finite_or_origin`, `boundary_mu_guard`, `advection_positive_definite_limiter`, `thompson_input_species_floor`, `thompson_final_species_floor`, `thompson_deposition_limiter`, `thompson_sedimentation_nonnegative_fixer`, `thompson_nstep_clip`.

Exact hook locations (verbatim from the review):
1. In `_physics_boundary_step_with_limiter_diagnostics`, capture `pre_step_state = carry.state` before `_rk_scan_step`.
2. After `_rk_scan_step`, before `_limit_guarded_dynamics_state_with_diagnostics`, compute the raw dycore budget if a per-step trace is desired.
3. Use the existing `limiter_diagnostics` for theta counts, **add** magnitude fields: `sum(abs(theta_after − theta_raw) * theta_mass)` and max abs K delta (don't rely on cell-count alone).
4. Around each `_valid_mixing_ratio` call, count `raw != guarded`; magnitude `sum(abs(guarded − raw) * layer_dry_mass_kg_species)` (separate indices per species, and dynamics vs boundary).
5. Around `_limit_guarded_mass_state`, count `raw_mu_total != guarded_mu_total`; magnitude `sum(abs(delta_mu)*A/g)` kg, signed `sum(delta_mu*A/g)`.
6. Around each `_finite_or_origin`, count nonfinite-raw replaced by origin; magnitude = sum abs field delta (kg-equiv for moisture/mass).
7. Surface evaporation: accumulate `sum(QFX_kg_m2_s * dt * A)` every step; current state carries kinematic `qv_flux = QFX/rhosfc`, so `QFX = qv_flux * rhosfc` after surface/Noah-MP updates, upward positive; accumulate only when surface flux actually refreshed/applied.
8. Precip per-step trace: difference in `(rain_acc+snow_acc+graupel_acc+ice_acc)` across Thompson call × area.
9. LBC fluxes accumulated in the boundary-application hook (signed change in integrated `MU_total` / water storage / MSE caused by the boundary update *before* guards — not the full-step dycore change).
10. Thompson internal guards must use a **side channel** in `thompson_column.py` (`_clip_species`, `_finalize_species`, deposition limiting, sedimentation nonneg fixers, `nstep` clip) — final-state differencing misses load-bearing internal repairs that cancel later.
11. After the scan: `compute_conservation_budget(final_state, grid, diagnostics={…})` for final and initial, then `compute_budget_closure(…, corrections={dry_mass_lbc_flux_kg, water_lbc_flux_kg, mse_lbc_flux_j})`. Host transfer + JSON only after the scan.

> Manager's closeout for P0-7 also needs: run **guards-on and guards-off** real d02/d03 GPU proofs and include the guard/limiter load-bearing table. Energy-release gate is **CPU-WRF-envelope based (±20%)**, not an absolute conservation threshold (MSE is a credibility diagnostic for this ARW-shaped system, not an absolute law).

### 4.3 P0-5 output-routing hooks (QFX / GRDFLX / soil → M9Diagnostics) (P0-5)
Source: `proofs/p0_5/FINDINGS.md` §"Manager hook needed…". The writer already **accepts** the new fields; routing them from the operational scan is an **L1-owned** change in `runtime/operational_mode.py` / `integration/daily_pipeline.py` (the p0-5 lane must NOT edit those). **No regression risk:** with `diagnostics=None` and `land_state=None` (and reduced/synthetic state) every new field self-gates off → byte-identical to today.

1. **QFX / GRDFLX** — add to `M9Diagnostics` (already computed upstream: `QFX` from `NoahMPFluxes.qfx` or bulk `surf.qfx`; `GRDFLX` from `NoahMPFluxes.grdflx`). Then add `("QFX","qfx")`, `("GRDFLX","grdflx")` to `_M9_OUTPUT_FIELDS` in `daily_pipeline.py` so `_surface_diagnostics_for_output` includes them. The writer's `_DIAGNOSTIC_SURFACE_FIELDS` set already accepts both.
2. **Soil/snow/land** (`TSLB/SMOIS/SH2O/SNOW/SNOWH/CANWAT/SFROFF/UDROFF/ALBEDO/EMISS`) — pass the prognostic `NoahMPLandState` carry to the writer: `prepare_wrfout_payload(…, land_state=carry.noahmp_land)` and `write_wrfout_netcdf(…, land_state=…)` (new optional kwarg; `None` → byte-identical to today).
3. **Microphysics extras / grid coords / precip partition** — **NO hook needed**; they read directly from `State` leaves / `GridSpec` the writer already receives, appearing as soon as the operational state carries them.

> P0-5 also has a SPEC-only **GPU resume-continuity** check (`proofs/p0_5/FINDINGS.md` §"Manager's GPU resume-continuity check") needing a GPU forecast (save mid-run, reload, compare to uninterrupted) — manager-owned, GPU.

---

## 5. Tag-readiness ledger

### IN (merged + CPU-green on this branch)
- P0-7 conservation **budget module** (standalone, CPU-controlled proof PASS) — instrumentation present, NOT yet scan-wired.
- P0-4 KF cumulus: NumPy reference + JAX port + tables + 4 gold savepoints + oracle tests green. Module present, NOT coupled.
- P0-1a nesting: scheduler/interp/boundary-construction + recorded parent→child oracle PASS + 11 unit tests green. Module present, NOT live-driven in the replay path.
- P0-5 IO: wrfout coverage extension (writer accepts +31 fields) + `wrfrst`-equivalent restart with CPU bit-fidelity roundtrip PASS + tests green. Writer/restart present, routing hooks NOT wired.
- All inherited from the TOST base: L1 + Thompson + nroot + MYNN-EDMF infra (edmf=OFF default).

### PENDING before tag (NOT in this lane's scope — manager/GPU owned)
- **TOST result** — the in-flight d02/d03 MAM GPU paired-delta TOST (parallel GPU job). The forecast code on this branch is identical to its base, so the result transfers; the branch is *not* tag-ready until that result lands and is labeled honestly.
- **Powered-n / single-season honesty** — corpus is MAM-only; release equivalence will be labeled **underpowered / single-season MAM**, never an unconditional "equivalence PASS" (per V0.2.0-PLAN Wave −1).
- **Boundary smoke** — a nesting boundary-construction smoke on the integrated branch (P0-1 live d01→d02→d03 path is a CORE v0.2.0 requirement per the plan; the recorded-oracle proof PASSes, but live-driven nesting in the replay path is not yet exercised here).
- **The three wirings** (§4.1 KF coupler, §4.2 conservation scan-carry, §4.3 P0-5 routing) — intentionally deferred; each is a careful manager follow-up and each has an open dependency:
  - KF: gated on closing the JAX-driver parity proof (`proofs/p0_4/jax_parity.json` PENDING) before operational activation.
  - Conservation: needs guards-on/off real d02/d03 GPU proofs for the operational closeout.
  - P0-5 routing: cheap/no-regression, but still a real-run output proof is wanted; plus the SPEC-only GPU resume-continuity check.

---

## 6. Commands run (CPU-only, GPU-free)
```
git checkout -b worker/opus/v020-release worker/opus/v020-tost-daytimefix
git merge --no-edit worker/gpt/p0-7-conservation        # clean
git merge --no-edit worker/gpt/p0-4-kf-jax              # clean
git merge --no-edit worker/opus/p0-1a-nesting           # clean
git merge --no-edit worker/opus/p0-5-io                 # clean
JAX_PLATFORM_NAME=cpu PYTHONPATH=src taskset -c 0-3 python -c "import gpuwrf"   # OK
JAX_PLATFORM_NAME=cpu PYTHONPATH=src OMP_NUM_THREADS=4 taskset -c 0-3 python -m pytest \
  tests/test_conservation_budget.py tests/test_kf_cumulus_oracle.py \
  tests/test_p0_1a_nesting.py tests/test_p0_5_restart_full_carry.py \
  tests/test_m7_netcdf_writer.py -q                       # 27 passed, 1 skipped (pre-existing data skip)
```

## 7. Unresolved risks
- Branch correctness rests on the **structural** disjointness argument for the inherited end-gate (verified: no dycore/coupling/scan touch). If the parallel TOST somehow exercises any of the new modules via an import side effect, that assumption would need re-checking — but all new modules are import-clean and inert by default (verified import + default-off behavior).
- The three un-wired hooks each carry an open dependency (above); activating any of them changes the operational default and would re-open the equivalence question.
- KF JAX-driver parity proof is PENDING — do not activate KF into the operational default until it is green.

## 8. Next decision needed
Manager: (1) confirm the in-flight TOST result transfers/labels honestly, (2) decide ordering of the three wirings (each with its dependency gate), and (3) run the boundary smoke on the integrated branch — then tag v0.2.0.
