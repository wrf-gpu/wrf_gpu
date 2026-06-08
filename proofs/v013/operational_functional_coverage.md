# v0.13 Operational-Functional Coverage Map

**Question:** does every physics option this port advertises as **OPERATIONAL**
(scan-wired: its State adapter is threaded into the operational scan) actually
**RUN** in the integration and **mutate its expected fields** (truly ran, not a
silent no-op)?

**Method:** new consolidated pytest `tests/test_v013_operational_smoke.py`. mp /
sf_sfclay / bl / cu / radiation are exercised through
`operational_mode._physics_step_forcing` — the EXACT per-step physics block the
operational scan runs — gated by the real fail-closed authority
`_resolve_operational_suite`. Land-surface is exercised through the EXACT coupler
steps the scan calls (`noahmp_surface_step` / `noahclassic_surface_step`). CPU
only, tiny idealized columns; whole module ~56 s on 4 cores. No masking / clamps /
self-compare / synthetic happy-path: a scheme that is rejected, crashes, or is
inert is reported as a real finding (xfail with a precise reason), never skipped.

**Authoritative operational set** (from `_SCAN_WIRED_OPTIONS` + the scan body,
excluding `0`=disabled):

| category | operational options |
|---|---|
| mp_physics | 1, 2, 3, 4, 6, 8, 10, 14, 16 |
| bl_pbl_physics | 1, 2, 5, 7, 8, 99 |
| sf_sfclay_physics | 1, 2, 3, 5, 7, 91 |
| cu_physics | 1, 2, 3, 6 |
| sf_surface_physics | 2 (Noah-classic, explicit hook), 4 (Noah-MP, `use_noahmp`) |
| ra_sw_physics | 1, 2, 4 |
| ra_lw_physics | 1, 4 |

REFERENCE-ONLY (oracle infra only, fail-closed in the scan; **not** covered here,
correctly): cu=5 (Grell-3D), cu=14 (KSAS), cu=16 (New-Tiedtke), sf_surface=1
(slab LSM), ra_lw=5 (GSFC/Goddard NUWRF LW).

## Coverage table (option → operational? → functional test → smoke result)

Legend: PASS = runs + mutates expected field; XFAIL = real "not fully functional"
finding (documented, fix turns it green).

### Microphysics (qv must change)

| opt | name | operational | pre-existing functional test | this smoke |
|---|---|---|---|---|
| 1 | Kessler | yes | `test_kessler_microphysics.py` (kernel) | PASS |
| 2 | Purdue-Lin | yes | lane parity only | PASS (newly covered integ) |
| 3 | WSM3 | yes | lane parity only | PASS (newly covered integ) |
| 4 | WSM5 | yes | lane parity only | PASS (newly covered integ) |
| 6 | WSM6 | yes | lane parity only | PASS (newly covered integ) |
| 8 | Thompson | yes | many (default suite) | PASS |
| 10 | Morrison | yes | lane parity only | PASS (newly covered integ) |
| **14** | **WDM5** | **yes (scan-wired)** | lane oracle only | **XFAIL — DEFECT** |
| 16 | WDM6 | yes | lane parity only | PASS (newly covered integ) |

### PBL (u or v must change; TKE schemes also update qke)

| opt | name | operational | pre-existing functional test | this smoke |
|---|---|---|---|---|
| 1 | YSU | yes | `test_v060_pbl_ysu.py` (kernel) | PASS (newly covered integ) |
| 2 | MYJ | yes (paired sf=2) | `test_v013_myj_janjic_operational.py` (`_physics_step_forcing`) | PASS |
| 5 | MYNN | yes | many (default) | PASS |
| 7 | ACM2 | yes | `test_v060_pbl_acm2.py` (kernel) | PASS (newly covered integ) |
| 8 | BouLac | yes | lane only | PASS (newly covered integ) |
| 99 | MRF | yes | `test_v013_mrf_operational.py` (`_physics_step_forcing`) | PASS |

### Surface layer (ustar or theta_flux must change)

| opt | name | operational | pre-existing functional test | this smoke |
|---|---|---|---|---|
| 1 | revised-MM5 | yes | `test_v060_sfclay_revised_mm5.py` (kernel) | PASS (newly covered integ) |
| 2 | Janjic Eta | yes (paired bl=2) | `test_v013_myj_janjic_operational.py` | PASS (via MYJ pair) |
| 3 | NCEP-GFS | yes | `test_v013_t3_surface_lsm_wiring.py` (adapter) | PASS (newly covered integ) |
| 5 | MYNN-sfclay | yes | many (default) | PASS |
| 7 | Pleim-Xiu | yes | `test_v060_sfclay_pleim_xiu.py` (kernel) | PASS (newly covered integ) |
| 91 | old-MM5 | yes | `test_v013_t3_surface_lsm_wiring.py` (adapter) | PASS (newly covered integ) |

### Cumulus (convective precip rainc_acc must appear, on an unstable column)

| opt | name | operational | pre-existing functional test | this smoke |
|---|---|---|---|---|
| 1 | Kain-Fritsch | yes | `test_kf_cumulus_oracle.py` (kernel) | PASS (newly covered integ) |
| 2 | BMJ | yes | lane only | PASS (newly covered integ) |
| 3 | Grell-Freitas | yes | `test_grell_freitas_cumulus.py` (kernel) | PASS (newly covered integ) |
| **6** | **Tiedtke** | **yes (scan-wired)** | `test_tiedtke_cumulus_oracle.py` (kernel) | **XFAIL — INERT** |

### Land surface (advance land carry + write surface flux)

| opt | name | operational | pre-existing functional test | this smoke |
|---|---|---|---|---|
| 2 | Noah classic | yes (explicit hook) | none (integ) | PASS (newly covered, via `noahclassic_surface_step`) |
| 4 | Noah-MP | yes (`use_noahmp`) | `test_noahmp_coupler.py` (adapter) | PASS (via `noahmp_surface_step`; MPTABLE-gated skip) |

### Radiation (finite, NONZERO RTHRATEN)

| opt | name | operational | pre-existing functional test | this smoke |
|---|---|---|---|---|
| ra_sw=1 | Dudhia SW | yes | `test_cdudhia_sw_operational_wiring.py` (`_physics_step_forcing`) | PASS |
| ra_sw=2 | GSFC/Chou-Suarez SW | yes | `test_v013_ra_sw_gsfc.py` (kernel) | PASS (newly covered integ) |
| ra_sw=4 | RRTMG SW | yes | many (default) | PASS |
| ra_lw=1 | classic AER RRTM LW | yes | `test_rrtm_lw_operational_wiring.py` (`_physics_step_forcing`) | PASS |
| ra_lw=4 | RRTMG LW | yes | many (default) | PASS |

## Tally

* **Total operational options covered:** 32 (9 mp + 6 bl + 6 sf_sfclay + 4 cu +
  2 sf_surface + 3 ra_sw + 2 ra_lw; sf=2 counted under the MYJ pair).
* **Pre-existing INTEGRATED functional coverage** (ran in `_physics_step_forcing`
  / coupler-step, asserting mutation): only the v0.13 wiring tests — MYJ pair
  (bl=2/sf=2), MRF (bl=99), Dudhia (ra_sw=1), classic-RRTM (ra_lw=1), and the
  default suite (mp=8, bl=5, sf=5, ra=4/4). The rest had only **per-scheme
  oracle/kernel parity** or **adapter-registration** tests, NOT an
  integration-step "did it actually run + mutate" check.
* **GAPS newly closed by this smoke** (first integrated functional coverage): mp
  1/2/3/4/6/10/16, bl 1/7/8, sf_sfclay 1/3/7/91, cu 1/2/3, sf_surface 2/4,
  ra_sw 2 — **~21 options**.
* **xfail GPU-only:** none. The whole operational surface runs functionally on CPU
  inside the budget (~56 s). Noah-MP is `skipif`-gated on the pristine WRF MPTABLE
  (present here → it runs), Noah-classic on the WRF-derived savepoint bundle
  (present here → it runs) — both ran PASS, neither is GPU-only.

## FAILURES — real "not fully functional" defects (each = a debug lane)

1. **mp_physics=14 (WDM5) — advertised-operational but UNROUTABLE.**
   WDM5 was merged at the v0.13 trunk tip (commit `baf3e2fe`, "WDM5 merged") and
   is fully scan-wired: it is in `ACCEPTED_MP_PHYSICS`, in
   `_SCAN_WIRED_OPTIONS["mp_physics"]`, and in `MP_SCAN_ADAPTERS[14]` (=
   `wdm5_adapter`). **BUT** it is MISSING from the physics-dispatch routable table
   `coupling/physics_dispatch.py::_MP_ENTRIES` (no key `14`). The operational
   fail-closed authority `_resolve_operational_suite` → `resolve_physics_suite`
   therefore **REJECTS** `mp_physics=14` with `UnsupportedSchemeSelection`
   ("mp_physics=14 is not a routable v0.6.0 scheme"). So a user selecting WDM5 in
   the operational forecast hits a hard fail-closed, not a run.
   **Fix (1 line):** add to `_MP_ENTRIES`:
   `14: _mp_entry(14, "gpuwrf.physics.microphysics_wdm5", "wdm5_physics_tendency", gpu=True),`
   (the module + entrypoint exist and the adapter already calls them).

2. **cu_physics=6 (modified Tiedtke) — scan-wired but INERT.**
   cu=6 is in `_SCAN_WIRED_OPTIONS["cu_physics"]` and `CU_SCAN_ADAPTERS[6]`, and it
   RUNS finite + JIT-traceable. But the operational adapter
   `coupling/scan_adapters.py::tiedtke_adapter` **hard-zeroes `QVFTEN` and
   `QVPBLTEN`** (the large-scale + PBL moisture-convergence forcing), a documented
   carry-over ("no separate forcing tracked into the cumulus slot here"). The
   Tiedtke closure triggers on that moisture convergence, so with it zeroed cu=6
   produces ZERO tendency / ZERO convective precip even on a strongly convective,
   near-saturated, warm-SST column. Direct kernel test confirms
   `tiedtke_column_jax` DOES trigger when fed nonzero `QVFTEN` (≈1e-6 → RAINCV≈2,
   RTHCUTEN≈1e-2), so the gap is the **operational coupling**, not the kernel.
   **Fix:** thread a real `QVFTEN` (advective moisture tendency) into
   `tiedtke_adapter` (e.g. from the resolved-scale moisture flux divergence), or
   re-scope cu=6 to REFERENCE-ONLY until that forcing is wired. As-is it is
   advertised operational but cannot influence a forecast.

## Reproduce

```
JAX_PLATFORMS=cpu PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 taskset -c 28-31 \
    python -m pytest tests/test_v013_operational_smoke.py -q
# => 36 passed, 2 xfailed in ~56 s
```
