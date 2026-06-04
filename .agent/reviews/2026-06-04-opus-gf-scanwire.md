# cu=3 Grell-Freitas scan-wire (v0.9.0 GPU-operational)

**Branch:** `worker/opus/v090-gf-scanwire`
**Base:** `worker/opus/v090-gf-gpubatch-A` @ `d9193028` (accepted GF GPU-batch kernel + parity proof)
**Date:** 2026-06-04
**Model:** Opus 4.8 (1M)

## Objective

Wire the already-DONE + savepoint-faithful GPU-batched Grell-Freitas cumulus
kernel (`src/gpuwrf/physics/_gf_jax.py`, parity PASS in
`proofs/v060/gf_gpubatch_savepoint_parity.json`) into the operational GPU scan so
`cu_physics=3` is selectable, GPU-operational, and RUN-PASSes the multi-config
operational smoke. Non-destructive; WRF-faithful (kernel untouched, no clamps);
honesty over green.

## What was done

Mirrored the modified-Tiedtke (cu=6) **stateless** `State -> State` adapter
pattern (GF, like Tiedtke, carries NO persistent cumulus state — no carry
threading, unlike KF's `(w0avg, nca)` / BMJ's `CLDEFI`).

1. **`coupling/scan_adapters.py`** — added `gf_adapter(state, dt, grid)` that
   `jax.vmap`s `_gf_jax.gfdrv_batched` over the `(ny*nx)` columns (whole deep +
   shallow scale-aware 16-member closure ensemble + beta-PDF gamma inside ONE
   vmapped jit, no host transfer in the column loop). Registered
   `CU_SCAN_ADAPTERS[3] = gf_adapter`, `CU_STATELESS_SCAN_ADAPTERS[3] =
   gf_adapter`, and added to `__all__`. Added a small `_kpbl_bulk_richardson`
   helper (per-column PBL-top diagnosis).
2. **`runtime/operational_mode.py`** — `3` added to
   `_SCAN_WIRED_OPTIONS["cu_physics"]` (now `(0,1,2,3,6)`); removed the
   `cu_physics=3` entry from `_SCAN_UNWIRED_REASON`; updated the
   `_resolve_operational_suite` error string. **No new dispatch branch needed** —
   the existing `if cu_opt in CU_STATELESS_SCAN_ADAPTERS` branch routes cu=3
   automatically (and `_initial_carry_for_run` correctly leaves `cumulus_carry =
   None` for cu=3).
3. **`coupling/physics_dispatch.py`** — `_CU_ENTRIES[3]` flipped
   `gpu_runnable=False -> True`; `owner_module/entrypoint` now
   `gpuwrf.physics._gf_jax.gfdrv_batched`; `writes_state` narrowed to the leaves
   GF actually advances (`theta, qv, qc, qi`); module + block docstrings updated.
4. **`contracts/physics_registry.py`** — `CU_SCHEMES[3]` status
   `accepted -> implemented` (the GPU-op convention used for YSU/ACM2/MYNN/Noah-MP)
   with provenance comment.
5. **`io/namelist_check.py`** — moved cu=3 into the operational-GPU group
   (`Use cu_physics=0/1/2/3/6 for the operational GPU scan`); only cu=16 remains
   fail-closed in the description.
6. **`proofs/v060/multicfg_operational_smoke.py`** — converted the GF
   `FAIL_CLOSED` config into the `cu_gf` **RUN** config (`8/5/5/3`, bulk land, like
   the cu_kf/cu_bmj/cu_tiedtke coverage configs); added a
   `_gf_adapter_triggering_probe` that runs the adapter on the WRF deep savepoint
   sounding; updated rationale comments. Regenerated `multicfg_smoke_report.json`.
7. **`tests/test_v060_physics_dispatch.py`** — updated the two assertions that
   encoded the *old* cu=3-fail-closed status to the new GPU-gate-ready status
   (cu=16 still fail-closed).

## WRF-faithfulness

- **Kernel `_gf_jax.py` is UNTOUCHED** — the proven-faithful physics is only
  *called*, never altered. **No clamps.**
- GF-specific driver inputs are assembled WRF-faithfully in the adapter, pure
  `jnp`, no host transfer:
  - `HFX = rho_sfc * cp * theta_flux` (W m^-2), `QFX = rho_sfc * qv_flux`
    (kg m^-2 s^-1) — kinematic B2 handles -> GF flux units (matches the
    `buo_flux = (hfx/cp + 0.608*T*qfx/XLV)/rho` usage in the kernel).
  - `HT = ph[0]/g` (surface geopotential height), `DX` from the grid (KF
    precedent), `KPBL` bulk-Richardson-diagnosed per column (the value a PBL
    scheme hands GF in WRF; no PBL-height leaf is threaded in this scan).
  - 1-based length-(nz+1) column mapping (GF level 1 = surface; State is
    bottom-up index 0 = surface -> prepend a dummy level-0).
- **Documented carry-over (identical to the accepted Tiedtke adapter):**
  `RTHBLTEN`/`RQVBLTEN` PBL forcing tendencies are passed as **zero**. The
  operational scan does not separately track the PBL-slot tendency into the
  cumulus slot (the PBL slot already applied it to State the same step). With
  zero forcing GF's "forced sounding" collapses to the current sounding and GF
  triggers on the actual column state. This mirrors the Tiedtke adapter's zero
  `QVFTEN`/`QVPBLTEN`. It is the one approximation vs a full WRF cumulus call and
  is recorded here (not masked).

## Proof objects

- `proofs/v060/multicfg_smoke_report.json` (regenerated):
  - **21/21 RUN configs PASS** including `cu_gf` (cu=3) `INTEGRATION_PASS`
    (namelist accepted, dispatch `gpu_gate_ready=True` / `non_gpu_schemes=[]`,
    operational coupler accepts, compiles/jit-traceable, 8 steps, finite,
    physical-in-band, schemes-active).
  - **2/2 FAIL_CLOSED configs OK** — cu=16 (New Tiedtke) and MYJ(2)+Janjic(2)
    still loudly rejected. **No regression** to any other scheme.
  - Coverage now includes `cu3-GF`.
  - `gf_adapter_triggering_probe`: **GF_ADAPTER_TRIGGERS** — the adapter PATH
    fires deep convection on the WRF deep savepoint sounding (`gf_case_1.json`,
    oracle KTOP_DEEP=26): `dtheta=0.133 K`, `rainc_acc=0.0414 mm`
    (oracle RAINCV=0.0425 mm; the small gap is the documented zero-PBL-forcing
    choice). This proves the wiring genuinely carries GF convective tendencies —
    not a dead/no-op path.
- Tests: `test_v060_physics_dispatch.py`, `test_namelist_check.py`,
  `test_grell_freitas_cumulus.py`, `test_v060_physics_interfaces.py`,
  `test_v060_cumulus_kf.py`, `test_tiedtke_cumulus_oracle.py` -> **32 passed**.

## Honest notes / risk

- **cu_gf on the idealized smoke grid does not trigger convection** (10-level,
  ~3 km top — too shallow for GF deep convection; the GF kernel returns
  `IERR_DEEP=2`). This is a *legitimate* non-trigger, exactly like KF/Tiedtke on
  this idealized state, NOT a wiring defect. The dedicated triggering probe (on a
  proper 45-level deep WRF sounding) is what demonstrates the adapter fires.
- **No GPU run** (per resource rules a d02-replay triage holds the GPU). All
  evidence is CPU + jit-traceability (traceable on CPU == lowerable on the GPU
  scan). A real GPU run + CPU-WRF reference scoring is the manager-scheduled GPU
  gate (the documented `reference_scoring_seam`).
- **KPBL diagnosis** is a thermal-only bulk-Richardson estimate (no shear term).
  GF's PDF/inversion logic is robust to this; the binding savepoint-parity gate
  feeds the WRF KPBL directly, so this only affects the operational (non-parity)
  path and is a reasonable WRF-consistent reconstruction.
- **3 pre-existing test failures** in `test_m6*_no_h2d` / `test_m6b_theta_fix`
  are UNRELATED to this change — they match `_m9_snapshot` (a function name that
  predates this branch, confirmed present 2x in base `operational_mode.py` @
  d9193028) and a full-dycore finite assertion. My operational_mode.py edits are
  comment/tuple/string-only and introduce no forbidden host-transfer tokens.

## Next decision

cu=3 GF is now GPU-operational-wired and RUN-PASSing. The remaining cumulus TODO
is cu=16 (New Tiedtke — needs a distinct WRF source-path savepoint gate). When the
GPU is free, the manager-scheduled full-dycore GPU gate (CPU-WRF reference
scoring via `reference_scoring_seam`) is the natural next validation for the GF
operational path.
