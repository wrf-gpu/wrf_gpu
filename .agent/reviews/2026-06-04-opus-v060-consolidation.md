# v0.6.0 CONSOLIDATION — Opus lane review (2026-06-04)

**Branch:** `worker/opus/v060-consolidation`
**Final SHA:** `d34ad3e0b404357b04cafe2c580d8399a0629549`
**Base trunk:** `e998250` (trunk-0.9.0, the consolidated operational scan)
**Resource discipline:** CPU-only (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=true`), all compute pinned `taskset -c 0-3`; cores 4-31 untouched (live CPU-WRF backfill).

## Objective

Merge five VERIFIED, already-landed v0.6.0 scheme branches onto one consolidation
branch, unioning every scheme registration/adapter, then run the integration matrix
+ test suite and record per-scheme operational/parity/fail-closed status. Honesty
over green: a problem is a finding, not something to bury.

## What merged (in order)

| # | Branch | SHA | Payload | Conflicts |
|---|---|---|---|---|
| 1 | v060-close | c38a3c0 | README scope matrix + V0.6.0-CLOSE doc + close-proof gen + smoke | none (clean) |
| 2 | v060-myj | 4807e28 | MYJ PBL (bl=2) + Janjic Eta SL (sf=2) savepoint parity | namelist_check.py |
| 3 | v060-wsm-sm | dec11a1 | WSM5 (mp=4) + WSM3 (mp=3), GPU-scan-wired | physics_registry.py, test_namelist_check.py |
| 4 | v060-ysu-acm2-gpuop | 44aa8df | YSU (bl=1) + ACM2 (bl=7) GPU-op (scan/vmap) + proofs | physics_registry.py |
| 5 | v060-gf-tiedtke-gpu | 42534d8 | Tiedtke (cu=6) GPU-batched + wired; GF (cu=3) CPU-ref fail-closed | operational_mode.py, multicfg_smoke_report.json |

## Conflicts and how each was resolved (UNION, never drop)

1. **namelist_check.py SUPPORTED_OPTIONS (bl_pbl + sf_sfclay)** — HEAD (v060-close)
   carried "GPU-operational, scan-wired" annotations for YSU/MYNN/ACM2 + revised-MM5/
   MYNN-SL/Pleim-Xiu; the MYJ side added MYJ(2)/Janjic(2) with the mandatory 2<->2
   pairing. Resolved by union: kept the scan-wired annotations AND added MYJ/Janjic
   documented as parity-proven CPU references, **fail-closed** in the scan (not
   scan-wired), with the pairing rule.

2. **physics_registry.py ACCEPTED_* tuples (wsm branch)** — WSM branch added mp 3,4 but
   its copy of bl/sfclay predated MYJ. Resolved: `ACCEPTED_MP_PHYSICS=(0,1,3,4,6,8,10,16)`
   AND kept `bl/sf=...,2,...` (MYJ/Janjic). `tests/test_namelist_check.py` resolved to the
   same union.

3. **physics_registry.py PBL_SCHEMES (ysu-acm2 branch)** — that branch bumped YSU status
   to "implemented" (GPU-op) but lacked MYJ=2. Resolved: took the YSU "implemented" bump +
   GPU-op comment AND kept MYJ=2 ("accepted", parity-only). ACM2=7 already "implemented" on
   both sides.

4. **operational_mode.py `_SCAN_UNWIRED_REASON` + error string (gf-tiedtke branch)** —
   gf-tiedtke removed cu=6 from the unwired list (Tiedtke now wired) and improved cu=3/cu=16
   text; HEAD carried MYJ/Janjic unwired reasons + the correct mp menu. Resolved: kept
   MYJ(2)+Janjic(2) unwired reasons + GF(3)+New-Tiedtke(16) improved text; **dropped cu=6
   from unwired** (it is now in `_SCAN_WIRED_OPTIONS`, so listing it as unwired would be a lie);
   error string kept the correct mp menu `{0,1,3,4,6,8,10,16}` + correct cu menu `{0,1,6}`.
   Verified programmatically that **no scheme is both scan-wired AND in unwired-reason**.

5. **multicfg_smoke_report.json (gf-tiedtke branch)** — trivial stale-worktree-path conflict
   in a generated artifact. Resolved arbitrarily, then **regenerated** in this worktree.

`scan_adapters.py` and (mostly) `operational_mode.py` auto-merged cleanly: the consolidated
dispatch tables are single, de-duplicated, and correct (`MP_SCAN_ADAPTERS={1,3,4,6,10,16}`,
`CU_SCAN_ADAPTERS={1,6}`, `CU_STATELESS_SCAN_ADAPTERS={6}`, `PBL/SFCLAY_SCAN_ADAPTERS={1,7}`).

## Per-scheme status table (post-consolidation)

| Family | Option | Scheme | Status |
|---|---|---|---|
| mp_physics | 0 | passive qv | PASSIVE/OFF |
| mp_physics | 1 | Kessler | GPU-OPERATIONAL-WIRED |
| mp_physics | 3 | WSM3 | GPU-OPERATIONAL-WIRED |
| mp_physics | 4 | WSM5 | GPU-OPERATIONAL-WIRED |
| mp_physics | 6 | WSM6 | GPU-OPERATIONAL-WIRED |
| mp_physics | 8 | Thompson (default) | GPU-OPERATIONAL-WIRED |
| mp_physics | 10 | Morrison 2-moment | GPU-OPERATIONAL-WIRED |
| mp_physics | 16 | WDM6 | GPU-OPERATIONAL-WIRED |
| bl_pbl_physics | 1 | YSU | GPU-OPERATIONAL-WIRED |
| bl_pbl_physics | 2 | **MYJ** | **PARITY-PROVEN, FAIL-CLOSED** |
| bl_pbl_physics | 5 | MYNN (default) | GPU-OPERATIONAL-WIRED |
| bl_pbl_physics | 7 | ACM2 | GPU-OPERATIONAL-WIRED |
| sf_sfclay_physics | 1 | revised-MM5 | GPU-OPERATIONAL-WIRED |
| sf_sfclay_physics | 2 | **Janjic Eta** | **PARITY-PROVEN, FAIL-CLOSED** |
| sf_sfclay_physics | 5 | MYNN-SL (default) | GPU-OPERATIONAL-WIRED |
| sf_sfclay_physics | 7 | Pleim-Xiu | GPU-OPERATIONAL-WIRED |
| cu_physics | 1 | Kain-Fritsch | GPU-OPERATIONAL-WIRED |
| cu_physics | 3 | **Grell-Freitas** | **PARITY-PROVEN, FAIL-CLOSED** |
| cu_physics | 6 | Tiedtke | GPU-OPERATIONAL-WIRED |
| cu_physics | 16 | **New Tiedtke** | **FAIL-CLOSED** (not separately savepoint-gated) |
| sf_surface_physics | 2 | Noah classic | GPU-OPERATIONAL-WIRED (needs explicit static+land bundle) |
| sf_surface_physics | 4 | Noah-MP | GPU-OPERATIONAL-WIRED (use_noahmp=True) |
| ra_sw / ra_lw | 4 | RRTMG SW / LW | GPU-OPERATIONAL-WIRED (held-rate RTHRATEN) |

Counts: **19 GPU-operational-wired**, **4 fail-closed** (MYJ, Janjic, GF, New-Tiedtke),
7 passive/off, **0 unknown** (no scheme fell through the cracks). Source of truth:
`proofs/v060/consolidation_integration_matrix.json` (derived from the merged registries +
scan_adapters + `_SCAN_WIRED_OPTIONS`/`_SCAN_UNWIRED_REASON`).

Honesty note on cu=16: labeled fail-closed in the matrix. It shares Tiedtke's WRF source
path but has **no separate savepoint gate** of its own, so "parity-proven" is not strictly
true for cu=16 in isolation — it is correctly fail-closed.

## Integration matrix (`proofs/v060/multicfg_operational_smoke.py`)

`proofs/v060/multicfg_smoke_report.json` — **ALL PASS = True**, 20 configs:
- **RUN: 17/17 PASS** (real Canary Noah-MP baseline + Thompson/WSM3/WSM5/WSM6/Morrison/WDM6/
  Kessler MP, MYNN/YSU/ACM2 PBL, revised-MM5/MYNN-SL/Pleim-Xiu SL, KF + Tiedtke cumulus,
  Noah-MP + Noah-classic + bulk land) — finite, physical, schemes-active, jit-traceable.
- **FAIL-CLOSED: 3/3 OK** (GF cu=3, New-Tiedtke cu=16, MYJ+Janjic bl=2/sf=2 each loudly
  rejected by the operational coupler).

I ADDED coverage the consolidation needed: WSM3/WSM5 RUN configs + MYJ/Janjic + New-Tiedtke
FAIL_CLOSED configs.

## Honesty / cleanup fixes applied (GPT-completeness-audit items)

- **WSM stale MP-menu text in generated JSONs (CONFIRMED + FIXED).** `forecast_gate_readiness.json`,
  `scanwire_report.json`, and `multicfg_smoke_report.json` carried the OLD menu
  `mp_physics in {0,1,6,8,10,16}` (no WSM3/WSM5) and pre-YSU/ACM2/Tiedtke-wiring prose. Rewrote
  the stale prose in `forecast_gate_harness.py` + `gen_scanwire_report.py`, regenerated all three.
  No stale `{0,1,6,8,10,16}` menu remains anywhere in `proofs/`.
- **GF cu=3 documented PARITY-PROVEN CPU-reference, FAIL-CLOSED, post-0.9.0 carry-over** — README
  scope matrix, scanwire report, and matrix all state GF is CPU-reference only, loud fail-closed,
  faithful GPU-batch ≈ 2000-LOC closure-ensemble sprint. **NOT claimed operational.**
- **MYJ bl=2 documented parity-proven, scan-wiring carry-over, fail-closes LOUDLY** — verified in
  `_SCAN_UNWIRED_REASON` + integration smoke (the `pbl_myj_janjic_unwired` config asserts loud
  rejection, not silent skip).

## Other audit items checked (status recorded, not all fixed)

- **WDM6 proof path** — lives at `proofs/v060_wdm6/` (SEPARATE dir), unlike every other scheme under
  `proofs/v060/`. `overall_pass=True`, 6 cases. Path inconsistency is cosmetic; referenced by
  `gen_integration_report.py`. NOT moved (out of scope for honesty; flagging for the manager).
- **KF checksum sidecar** — `proofs/v060/kf_savepoint_parity_report.json` verdict=PASS (5 cases) but
  there is **NO `kf_wrf_source_checksums.txt` sidecar** (acm2/myjpbl/myjsfc/pxsfclay/sfclayrev1 all
  have one) and the KF report embeds no checksums. KF provenance traceability is weaker than the
  other schemes — a real (minor) gap, flagging for the manager.
- **Noah-classic static+land bundle** — requirement is enforced fail-closed: the scan rejects
  `sf_surface_physics=2` when `noahclassic_static`/`noahclassic_land` are absent
  (`_resolve_operational_suite` + the `land_noahclassic` RUN config supplies the real WRF NOAHMP_SFLX
  bundle). `noah_coupler_report.json` overall_pass=True. Good.

## Genuine problems / findings

1. **REAL BUG FIXED in the integration smoke RUN-path cumulus slot.** The gf-tiedtke smoke
   dispatched `if cu_opt in CU_SCAN_ADAPTERS: kf_adapter(...)` unconditionally — so the `cu_tiedtke`
   (cu=6) RUN config would have **silently run KF instead of Tiedtke**, not actually exercising the
   new scheme. Fixed to mirror `operational_mode._physics_boundary_step` (check
   `CU_STATELESS_SCAN_ADAPTERS` Tiedtke FIRST, then KF). After the fix, `cu_tiedtke` genuinely runs
   Tiedtke and PASSES.

2. **Semantic merge hazard caught by a test.** `SCHEME_STEP_SPECS` count assertion: v060-myj and
   v060-wsm-sm each independently set the literal to `21` (each adding 2 over a 19-base), so git
   merged `21`=='21' with NO conflict — but the union has BOTH pairs = **23**. Verified 23 specs, zero
   duplicates; updated the assertion + comment. (This is the kind of silent inconsistency parallel
   lanes produce; worth a manager note.)

## Test suite

CPU-only, `taskset -c 0-3`. All consolidation-relevant tests PASS:
- v060 dispatch/MYJ/Janjic/YSU/ACM2/Pleim-Xiu/revised-MM5/KF/WSM6/WSM3-WSM5/namelist/contracts/
  Noah-classic/Kessler/Noah-MP/C2-scan + WDM6 + Tiedtke-oracle + GF-cumulus: **all green**.

Two failures, BOTH unrelated to this consolidation (verified):
- `test_m6_operational_mode_no_h2d.py::test_operational_source_has_no_host_transfer_or_sanitizer_calls`
  — brittle source-grep forbidding the substring `snapshot(`; trips on the function name
  `_m9_snapshot`, which is on the **base trunk e998250** and untouched by my merges. **Pre-existing.**
- `test_m6b_operational_theta_fix.py::...acoustic_substep` — `State.zeros` raises
  "requires a GPU device"; cannot run on the mandated CPU-only sandbox. **Environment-gated, not a defect.**

## Deliverables (committed)

- consolidation branch `worker/opus/v060-consolidation` @ `d34ad3e`
- `proofs/v060/consolidation_integration_matrix.json` (per-scheme status + smoke results)
- `proofs/v060/multicfg_smoke_report.json` (regenerated: 17/17 RUN, 3/3 FAIL-CLOSED)
- `proofs/v060/forecast_gate_readiness.json`, `proofs/v060/scanwire_report.json` (regenerated, de-staled)
- README scope matrix + `.agent/decisions/V0.6.0-CLOSE.md` superseded-banner
- this review

## What the manager must still fold in (NOT merged here, by instruction)

- `worker/opus/v050-finish2` (v0.5.0)
- `worker/opus/v060-radiation` (RRTM-LW)
- `worker/opus/v060-lin-mp` (Lin, mp=2)
- `worker/opus/v060-bmj` (BMJ cumulus)

When folding those in, expect the SAME shared-dict conflicts (physics_registry ACCEPTED_*,
scan_adapters dispatch tables, operational_mode `_SCAN_WIRED_OPTIONS`/`_SCAN_UNWIRED_REASON`,
namelist_check SUPPORTED_OPTIONS, the `SCHEME_STEP_SPECS` count assertion, and the three generated
proof JSONs). Lin (mp=2) in particular must be unioned into `ACCEPTED_MP_PHYSICS` and the
`SCHEME_STEP_SPECS` count bumped accordingly — and watch the `21`-style literal merge hazard.

## Carry-overs (post-0.9.0)

- GF (cu=3): faithful GPU-batch closure-ensemble + beta-PDF gamma (~2000-LOC sprint).
- New-Tiedtke (cu=16): separate WRF-source savepoint gate + GPU-batch.
- MYJ (bl=2) + Janjic (sf=2): GPU-scan-wire the parity-proven CPU references.
- Noah-classic real-run static/land bundle assembly for canonical forecast-gate combo_2.
- KF checksum sidecar + WDM6 proof-path normalization (provenance hygiene).
