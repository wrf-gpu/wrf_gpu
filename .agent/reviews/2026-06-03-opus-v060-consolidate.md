# v0.6.0 INTERNAL CONSOLIDATION MERGE — Opus 4.8 MAX

**Date:** 2026-06-03
**Branch:** `worker/opus/v060-consolidate` (off `b75d7f9`)
**Worker:** Opus 4.8 MAX (frontrunner)

## Objective

Unify two sibling v0.6.0 feature branches — both off merge-base `382070f`
("wire 11 schemes into the operational scan") — into ONE v0.6.0 trunk whose
operational GPU scan runs the UNION of both scheme sets, with no capability
dropped from either side. WRF-faithful: no masking/clamp/self-compare/
synthetic-happy-path; no done-claim without a proof object.

## Two-sibling topology (verified)

```
            382070f  (merge-base: scan-wire 11 schemes; YSU/ACM2/Noah-classic FAIL-CLOSED)
           /        \
   a95c93c            9af2813   = worker/opus/v060-pbl-gpuop
  (Noah-classic        |          YSU(bl=1)+ACM2(bl=7) GPU-op rewrite:
   GPU-op +            |          pbl_ysu.py(+566), pbl_acm2.py(+608),
   scan-wire)          |          scan_adapters.py(+170 PBL adapters),
      |                |          physics_dispatch.py, operational_mode.py(+20),
   b75d7f9  <----------'          opened _resolve to bl_pbl{0,1,5,7}
  worker/opus/v060-multicfg-smoke
  (Noah trunk + the 16-config multi-config operational smoke)
```

`git merge-base 9af2813 a95c93c == 382070f` (confirmed).
`a95c93c` is an ancestor of `b75d7f9` (confirmed) — so the consolidate branch
based off `b75d7f9` already carries the Noah-classic GPU-op + scan-wire + smoke.
The merge brings in the PBL-GPU-op (YSU+ACM2) side.

**Merge command:**
```
git merge --no-ff worker/opus/v060-pbl-gpuop \
  -m "[v060] consolidate PBL-gpuop (YSU+ACM2) into the Noah trunk + smoke — unified v0.6.0 operational scan"
```

## Files conflicted + how resolved

`src/gpuwrf/runtime/operational_mode.py` **auto-merged cleanly** (complementary
regions): the PBL side opened `bl_pbl_physics` to `{0,1,5,7}` + routed the PBL
slot through `PBL_SCAN_ADAPTERS`; the Noah side added the `noahclassic_*`
namelist leaves, the `_explicit_noahclassic` gate, the Noah-classic land step
(inserted right before the PBL slot, WRF call order), and the M9 diagnostics
overlay. Both coexist; verified by inspection (imports both present; physics
step runs surface -> noahclassic land -> PBL dispatch -> KF cumulus).

`physics_dispatch.py` — **only the PBL side touched it** (YSU/ACM2 entries now
`gpu_runnable=True`); merged cleanly.

**Four textual conflicts, all complementary, resolved to the UNION:**

| File | Conflict | Resolution |
|---|---|---|
| `coupling/scan_adapters.py` | docstring "NOT wired here" block | Dropped the stale HEAD YSU/ACM2 "host-NumPy NOT-wired" bullet (PBL side wired them); kept Noah-side "Noah-classic wired in noahclassic_surface_hook" bullet. The PBL adapters / `PBL_SCAN_ADAPTERS` (PBL side) auto-merged in below the conflict. |
| `proofs/v060/gen_scanwire_report.py` | `NEW_SCHEME_STATUS` dict + `note` string | Dict auto-merged to the union (YSU/ACM2=True AND Noah-classic=True). Rewrote the `note` to the consolidated truth: **9 of 11 wired** (4 MP + 2 SL + 2 PBL + Noah-classic) + KF -> 10 adapters; only GF/Tiedtke fail-closed. |
| `proofs/v060/scanwire_smoke.py` | comment over the fail-closed list | Unified comment: GF/Tiedtke + Noah-classic-without-bundle fail-closed (YSU/ACM2 now scan-wired). The accept list (YSU,ACM2 added) and reject list auto-merged correctly. |
| `proofs/v060/scanwire_report.json` | generated counts/lists/combo rows | Hand-resolved to the union, then **regenerated** from the resolved generator (authoritative): 9 wired / 2 not-wired, overall_pass=True. |

No genuine logic incompatibility was found — the two siblings touched
orthogonal physics slots (PBL slot vs land slot) and complementary dispatch
metadata. This was a correctness merge with no papering-over.

## Stale-expectation finding fixed in the multi-config smoke (REAL, not masked)

The 16-config multi-config operational smoke (`multicfg_operational_smoke.py`,
authored on the Noah trunk before the PBL merge) had **two stale assumptions**
that the merge invalidated — fixed so the smoke reflects the merged reality:

1. Its `physics_body` PBL slot was **hardcoded to `mynn_adapter`** ("the only
   scan-wired PBL"). Updated to dispatch via `PBL_SCAN_ADAPTERS[bl_opt]` — the
   EXACT mirror of the merged `operational_mode._physics_boundary_step` PBL slot
   — so YSU(1)/ACM2(7) are actually exercised, not silently run as MYNN.
2. Two configs (`pbl_ysu_unwired`, `pbl_acm2_unwired`) were marked
   `expect="FAIL_CLOSED"` purely because YSU/ACM2 were absent. Flipped to
   `expect="RUN"` (renamed `pbl_ysu`, `pbl_acm2`), routed through fast bulk land
   (the PBL slot is the axis under test), with updated descriptions. The stale
   `sweep_rationale` string and `_expected_changed_leaves` comment were corrected.

## Before/after config-pass counts (the proof)

Multi-config operational smoke (CPU, JAX x64, 4 steps, cores 0-3, NO GPU):

| | RUN PASS | FAIL_CLOSED OK | all_pass |
|---|---|---|---|
| Before merge (b75d7f9) | 12/12 | 4/4 (YSU,ACM2,GF,Tiedtke) | True |
| **After consolidation** | **14/14** | **2/2 (GF cu=3, Tiedtke cu=6)** | **True** |

The two configs that were FAIL-CLOSED purely because YSU/ACM2 were absent now
`INTEGRATION_PASS`: `pbl_ysu` (8/1/1/0/bulk) and `pbl_acm2` (8/7/7/0/bulk) —
coupler accepts, jit-compiles (traceable == GPU-runnable), finite, all fields
physical, and **schemes-active** (`all_active: true`; u/v moved by the PBL
momentum mixing, theta/qv/qc/qr by MP, ustar/theta_flux by surface layer).

**14/16 PASS, exactly as predicted.** Scheme coverage now includes both
`bl1-YSU` + `bl7-ACM2` (PBL side) AND `land2-NoahClassic` + `land4-NoahMP`
(Noah side) — **no capability dropped from either sibling.**

## Confirmation: YSU/ACM2 integrate after merge

`proofs/v060/pbl_gpuop_smoke.py` (re-run on merged code, exit 0): YSU and ACM2
both execute in the scan adapter, jit-traceable, finite, conservation OK; the
resolver now accepts bl=1 and bl=7 (`fail_open_now_accepts_ysu_acm2.pass=True`).
`proofs/v060/scanwire_smoke` (via regenerated `scanwire_report.json`,
`overall_pass=True`): all 7 wired combos accepted incl. YSU/ACM2; all 3 unwired
(GF, Tiedtke, Noah-classic-without-bundle) correctly rejected.

## Noah side not broken by the merge

`proofs/v060/noah_coupler_smoke.py` (re-run, exit 0): Noah-classic land step
bounded soil water, water-tile land-carry unchanged. `land_noahclassic` config
in the multicfg smoke = INTEGRATION_PASS (t_skin advanced).

## Commands run

```
git worktree add -b worker/opus/v060-consolidate .claude/worktrees/v060-consolidate b75d7f9
git merge --no-ff worker/opus/v060-pbl-gpuop -m "[v060] consolidate ..."
# resolve 4 conflicts (scan_adapters.py docstring, 3 proof files)
JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONPATH=src \
  taskset -c 0-3 python3 proofs/v060/gen_scanwire_report.py        # regen -> 9 wired
  taskset -c 0-3 python3 proofs/v060/multicfg_operational_smoke.py --steps 4   # 14/14 + 2/2
  taskset -c 0-3 python3 proofs/v060/pbl_gpuop_smoke.py            # YSU/ACM2 integrate
  taskset -c 0-3 python3 proofs/v060/noah_coupler_smoke.py        # Noah-classic intact
```
All on CPU only (CUDA_VISIBLE_DEVICES="" forces CPU fallback) — the active
v0.4.0 GPU re-run was never contended; every smoke was tiny (4 steps).

## Proof objects produced / updated

- `proofs/v060/multicfg_smoke_report.json` — 14/14 RUN INTEGRATION_PASS +
  2/2 FAIL_CLOSED_OK (GF/Tiedtke), all_pass=True; pbl_ysu/pbl_acm2 now PASS.
- `proofs/v060/scanwire_report.json` (regenerated) — 9 wired / 2 not-wired,
  overall_pass=True.
- `proofs/v060/pbl_gpuop_smoke.json`, `noah_coupler_smoke.json` — both re-pass.

## Unresolved risks / carry-over

- **GF (cu=3) + Tiedtke (cu=6/16)** remain CPU-NumPy reference ports
  (`gpu_runnable=False`), correctly fail-closed in the coupler — documented
  GPU-batching TODO, the only 2 of the 11 new schemes not scan-wired.
- This is an **integration-wiring proof on a b2 atmospheric profile**, NOT a
  WRF/obs skill comparison. The full-dycore GPU multi-config forecast vs CPU-WRF
  (`reference_scoring_seam`) is MANAGER-scheduled (blocked on the v0.4.0
  forecast-gate reference-resolution fix). YSU/ACM2/Noah-classic per-scheme
  WRF savepoint parity = committed lane reports (`*_savepoint_parity_report.json`,
  `noah_coupler_report.json`).
- The PBL adapter re-derives interface pressure via a simple half-level
  assembler (`_pbl_surface_forcing`) and seeds ACM2 PBLH at 1000 m (overwritten
  by ACM2's own diagnosis); the operational forecast gate vs CPU-WRF refines
  these against true half-level pressure — noted in the PBL lane handoff.
- `pbl_gpuop_report.json` is the PBL lane's self-contained report; its
  "still_fail_closed_clean" block still lists Noah-classic (it describes the PBL
  lane in isolation, pre-Noah-merge). The AUTHORITATIVE consolidated status is in
  `scanwire_report.json` (Noah-classic = wired) + `multicfg_smoke_report.json`
  (land_noahclassic = PASS). Left as the PBL author wrote it (not my lane).

## Next decision needed

None blocking. v0.6.0 consolidation is complete and proven: the unified
operational scan runs MYNN+YSU+ACM2 PBL AND Noah-MP+Noah-classic LSM (+ the
existing MP{Thompson/WSM6/Morrison/WDM6/Kessler} + SL{1/5/7} + KF). Manager to
schedule the full-dycore GPU forecast gate vs CPU-WRF when the v0.4.0
reference-scorer path is fixed.
