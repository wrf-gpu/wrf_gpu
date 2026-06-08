# Opus fix: surface-layer/PBL silent revised-MM5 substitution (GPT#1 MAJOR)

Date: 2026-06-08
Author: Opus-max debugger (opposite-model fix of GPT#1's finding)
Branch: `fix-mrf-sfclay-diagnostics` (base = v0.13 trunk tip `526f4d73`,
`worker/opus/v0120-integration`)

## Objective

GPT#1 found a MAJOR operational-honesty bug: the MRF PBL (bl=99) consumes
revised-MM5 surface forcing regardless of the SELECTED surface-layer scheme,
yet the resolver reported `gpu_gate_ready=True`. Diagnose the full extent and
fix faithfully — make pairings TRULY functional, or FAIL CLOSED (no silent
wrong-scheme substitution). Default config must be byte-unchanged.

## Diagnosis — full extent (broader than just MRF)

The bug is in `coupling/scan_adapters.py::_pbl_surface_forcing()` (line ~955),
which unconditionally calls `physics.surface_layer.surface_layer_with_diagnostics`
(= the **revised-MM5** `sf_sfclayrev` scheme) to assemble the per-cell surface
forcing (HFX/QFX/BR/PSIM/PSIH/U10/V10/ZNT) the PBL kernels consume — ignoring
which surface-layer scheme the namelist actually selected.

`_pbl_surface_forcing` is consumed by FOUR PBL scan adapters, not just MRF:
- **YSU (bl=1)** `ysu_pbl_adapter`
- **ACM2 (bl=7)** `acm2_pbl_adapter`
- **BouLac (bl=8)** `boulac_pbl_adapter`
- **MRF (bl=99)** `mrf_pbl_adapter`

Root cause: the frozen State B2 contract carries only the kinematic flux handles
(`ustar/theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv`) + `roughness_m`(ZNT). It does
NOT carry the stability functions PSIM/PSIH, the bulk Richardson BR, or U10/V10
that these PBL kernels also require. So instead of consuming the selected surface
scheme's output, the adapters re-derive **everything** from revised-MM5 — including
overriding HFX/QFX (line 987-990: `hfx = diag.hfx`, discarding State's `theta_flux`).

The two NON-affected PBLs are correct:
- **MYNN (bl=5)** reads `physics_couplers._surface_fluxes_from_state` — it consumes
  the SELECTED scheme's kinematic flux handles from State (MYNN only needs those),
  so it pairs correctly with any wired surface layer.
- **MYJ (bl=2)** re-runs the Janjic surface layer it is mandatorily paired with
  (sf=2, already enforced by the resolver's MYJ-pairing guard).

### Empirical confirmation the substitution is MATERIAL (not benign)

On a synthetic column the GFS surface scheme produces `theta_flux ≈ 0.076/0.027`
while revised-MM5 produces `≈ 0.268/0.071` — a ~3.5× difference in surface heat
flux. Selecting GFS under MRF/YSU silently fed the PBL the much larger revised-MM5
flux: a real, material wrong-result presented as gate-ready.

### WRF oracle confirms the faithful contract

`wrf_pristine/WRF/phys/module_physics_init.F` `pbl_select` FATAL-ERRORs unless the
surface layer satisfies the PBL's `isfc` requirement:
- YSU(1)/MRF(99)/Shin-Hong/GBM: `isfc==1` (sf ∈ {revised-MM5=1, old-MM5=91})
- ACM2(7): `isfc ∈ {1,7}` (sf ∈ {1, 91, Pleim-Xiu=7})
- BouLac(8): `isfc ∈ {1,2}` (sf ∈ {1, 91, Janjic=2})

and `dyn_em/module_first_rk_step_part1.F:594→1113` shows surface_driver writes
HFX/QFX/BR/PSIM/PSIH and pbl_driver reads those SAME fields by name. The selected
surface scheme determines the forcing; the PBL just consumes it.

In THIS reimplementation only the **revised-MM5 (sf=1)** forcing path is threaded
into these four PBL adapters (old-MM5 sf=91 forcing is NOT separately wired into
the PBL re-derivation). So the faithful, verifiable contract is narrower than WRF's
isfc set: **bl ∈ {1,7,8,99} is faithful ONLY with sf_sfclay_physics=1**.

## Fix — what was wired-truly-functional vs fail-closed, and why

**TRULY-FUNCTIONAL (kept gate-ready, real forcing = selected scheme):**
- bl ∈ {1,7,8,99} **+ sf=1 (revised-MM5)**. The PBL re-derives revised-MM5 forcing,
  which IS the selected scheme — no substitution. WRF-valid (isfc=1). Verified by the
  new test asserting the re-derived forcing equals the revised-MM5 adapter's output.
- bl=5 (MYNN) **+ any wired sf (0/1/3/5/7/91)** — MYNN consumes the SELECTED scheme's
  State flux handles; unchanged, gate-ready.
- bl=2 (MYJ) **+ sf=2** — re-runs its mandatory Janjic surface layer; unchanged.

**FAIL-CLOSED (would silently substitute revised-MM5):**
- bl ∈ {1,7,8,99} **+ sf ∈ {0,3,5,7,91}** (and the unpinned default sf=5). The real
  carry — threading each scheme's full forcing (PSIM/PSIH/BR/U10/V10) from the surface
  slot into the PBL — requires either new frozen State leaves or a surface→PBL forcing
  carry through the operational scan with per-scheme physics validation against
  pristine-WRF. Both are out of v0.13 scope. Per the principal's honest-fallback rule,
  these now `raise UnsupportedSchemeSelection` with a named, specific reason rather than
  running a different scheme than requested.

The check is added in `coupling/physics_dispatch.py::resolve_physics_suite` (the single
chokepoint; every `run_forecast_operational*` entry calls `_resolve_operational_suite`
→ `resolve_physics_suite`), mirroring the existing MYJ-pairing-violation pattern. New
module constants `_PBL_REQUIRES_REVISED_MM5_SFCLAY = {1,7,8,99}` and
`_REVISED_MM5_SFCLAY_OPTION = 1` with a full WRF-referenced rationale comment.

## Gate-ready map (before → after)

```
       bl=0   1   2   5   7   8  99            bl=0   1   2   5   7   8  99
sf=  0: GR  GR  FC  GR  GR  GR  GR     sf=  0: GR  FC  FC  GR  FC  FC  FC
sf=  1: GR  GR  FC  GR  GR  GR  GR     sf=  1: GR  GR  FC  GR  GR  GR  GR
sf=  2: FC  FC  GR  FC  FC  FC  FC     sf=  2: FC  FC  GR  FC  FC  FC  FC
sf=  3: GR  GR  FC  GR  GR  GR  GR  →  sf=  3: GR  FC  FC  GR  FC  FC  FC
sf=  5: GR  GR  FC  GR  GR  GR  GR     sf=  5: GR  FC  FC  GR  FC  FC  FC
sf=  7: GR  GR  FC  GR  GR  GR  GR     sf=  7: GR  FC  FC  GR  FC  FC  FC
sf= 91: GR  GR  FC  GR  GR  GR  GR     sf= 91: GR  FC  FC  GR  FC  FC  FC
```
GR=gpu_gate_ready, FC=fail-closed. The only changed cells: bl∈{1,7,8,99} × sf≠1
flip GR→FC (the silent-substitution pairings). bl=5/bl=2 columns and the default
(sf=5,bl=5) are unchanged.

## Default-unchanged proof (HARD INVARIANT)

A default-config (`bl=5` MYNN, `sf=5` MYNN-sfclay, Thompson/Noah-MP/RRTMG) physics
step is BYTE-IDENTICAL before/after. Ran `_physics_step_forcing` on the MRF test's
default namelist + state and hashed theta/u/v/qv/qc/qke/p/ph:
- baseline (526f4d73, change stashed): `ca729533300877f8083469f227e126d2364f8c201ad73424bc32db402c463f79`
- after fix: `ca729533300877f8083469f227e126d2364f8c201ad73424bc32db402c463f79`

Identical. Structurally: the new branch only triggers for `pbl∈{1,7,8,99} and sf≠1`;
the default (5,5) never enters it, and no dispatch/forcing logic was touched.

## Files changed

- `src/gpuwrf/coupling/physics_dispatch.py` — pairing fail-close + constants + rationale.
- `tests/test_v013_sfclay_pbl_pairing.py` (NEW) — 36 cases: no-silent-substitution
  fail-close for every re-deriving-PBL × non-sf1 pairing; faithful pairings stay
  gate-ready; MYNN/MYJ exemptions; GFS-vs-revised-MM5 flux-differs (substitution was
  material); default byte-unchanged.
- `tests/test_v060_physics_dispatch.py` — pinned `sf_sfclay_physics=1` in
  `test_nested_wrf_style_mapping_resolves` (was asserting the buggy YSU+default-sf=5
  gate-ready).
- `proofs/v060/scanwire_smoke.py` — corrected the gate-ready expectations (pin sf=1 for
  YSU/ACM2; add the silent-substitution pairings to the must-fail-closed list); also
  fixed a PRE-EXISTING breakage (the `_NL` stub lacked `ra_sw_physics` etc., so the
  script crashed at baseline) and moved GF(cu=3)/MYJ+Janjic out of fail-closed (wired in
  later sprints). Now `all_pass=True`.
- `proofs/v060/forecast_gate_harness.py` — updated the stale comment that claimed
  canonical combo_3 (ACM2+Pleim-Xiu) was scan-runnable; it now fails closed (the
  recommended SCAN_WIRED variant already uses MYNN, unaffected).

## Commands / proof objects

- `pytest tests/test_v013_t3_surface_lsm_wiring.py tests/test_v013_mrf_operational.py
  tests/test_v060_physics_dispatch.py tests/test_v013_sfclay_pbl_pairing.py` →
  **64 passed**.
- Broad sweep incl. YSU/ACM2/MYJ-Janjic operational + all resolver-touching suites
  (16 files) → **all green** (78 + 39 passed; idealized dycore-gate skipped on CPU).
- `proofs/v060/scanwire_smoke.py` → `all_pass: True`.
- Default byte-identity hash match (above).

All runs CPU-only: `JAX_PLATFORMS=cpu PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 taskset -c 0-11`.

## Unresolved risks / carry-over

- The real surface→PBL forcing carry for YSU/ACM2/BouLac/MRF + non-revised surface
  layers (old-MM5/Pleim-Xiu/GFS) is deferred (needs a forcing carry or new State leaves
  + per-scheme WRF-fixture validation). Tracked as fail-closed; honest, not silent.
- bl=99/8 + sf=91 (old-MM5) is WRF-valid (isfc=1) but fail-closed here because only the
  revised-MM5 forcing is threaded into the PBL re-derivation. A future sprint could make
  `_pbl_surface_forcing` dispatch to old-MM5 (its kernel returns the full forcing set)
  to widen the gate-ready set to the full WRF isfc=1 pairing — verifiable, but a
  separate validation task.
