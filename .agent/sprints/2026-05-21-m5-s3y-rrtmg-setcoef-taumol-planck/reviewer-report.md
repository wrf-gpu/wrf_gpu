# Reviewer Report — M5-S3.y RRTMG setcoef/taumol/Planck-source (binding close decision)

**Reviewer**: Claude Opus 4.7 xhigh (fresh-context, per sprint-lifecycle double-AI HARD RULE, `.agent/rules/sprint-lifecycle.md:14-32`)
**Date**: 2026-05-21
**Branch / commit under review**: `worker/codex/m5-s3y-rrtmg-setcoef-taumol-planck` @ `8b58bbb Record M5-S3.y RRTMG native-table attempt`
**Worker**: Codex GPT-5.5 xhigh
**Worker self-verdict**: "**NOT ACCEPTANCE** and should not be merged as M6-unblocking RRTMG parity… I am filing this as a failed attempt with useful evidence, not as completed work." (`worker-report.md:13-17`)
**Prior cycle precedent**: M5-S3 → ACCEPT-AS-GROUNDWORK; M5-S3.x → ACCEPT-AS-GROUNDWORK-PHASE-2; both anti-patterns (clip-floor R-2, vacuous-tolerance R-3, launch-fudge R-fudge) have **not** recurred this cycle.

---

## Reviewer decision: **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3 with binding M5-S3.z dispatch**

M5-S3.y closes as **Phase-3 groundwork** along the M5-S3 → S3.x → S3.y trajectory. The decision is *not* full REJECT and *not* worker-recommended REJECT-bounded-rework because four artifacts from this sprint are genuine, reusable progress that should NOT be thrown away:

1. **AC0 PASS — Eddington oracle rebuild is clean and load-bearing for every future RRTMG sprint.** The `/mnt/data/.../module_ra_rrtmg_sw.F:2632` `kmodts=2 → kmodts=1` patch was applied to both the documented harness source AND to the actually-compiled `/home/enric/.../wrf_src/.../module_ra_rrtmg_sw.F` (worker correctly diagnosed and fixed the dual-source-tree build-rule footgun). The rebuilt harness SHA `25c88aa…fd2b33` still binds `__rrtmg_sw_spcvmc_MOD_spcvmc_sw`, `__rrtmg_lw_rtrnmc_MOD_rtrnmc`, `__rrtmg_sw_taumol_MOD_taumol_sw`, `__rrtmg_lw_taumol_MOD_taumol`, `__rrtmg_sw_setcoef_MOD_setcoef_sw`, and `__rrtmg_lw_setcoef_MOD_setcoef` (verified §1.1). M5-S3.x's R-8 oracle-vs-implementation mismatch is **definitively resolved**: the JAX Eddington branch and the WRF compiled oracle now both target the kmodts=1 Eddington formulas. This is permanent infrastructure regardless of next-sprint outcome.
2. **Real native WRF reduced-g-point table extraction.** `data/fixtures/rrtmg-tables-v1.npz` SHA `9d8bedb…534013` now exposes `sw_absa(14,585,12)`, `sw_absb(14,1175,12)`, `sw_selfref(14,10,12)`, `sw_forref(14,4,12)`, `sw_sfluxref(14,12,9)`, `lw_totplnk(181,16)`, `lw_totplk16(181)` as JAX leaves, parsed from WRF source via `scripts/extract_rrtmg_tables.py` using the documented `swcmbdat`/`cmbgb*` weighted-reduction pattern. Distributions are physically broad (worker §132–134; verified §1.2). These tables are the data substrate every subsequent RRTMG sprint will consume.
3. **LW Planck-source replacement is a real, small-but-positive correctness improvement.** Old code at the M5-S3.x baseline used `σT⁴ × g_weight` grey-body per-g-point source (M5-S3.x reviewer §99–106 R-2 explicitly called this out as the dominant LW residual driver). M5-S3.y replaces it with WRF `totplnk` table interpolation × `delwave·π·10⁴` flux scaling (`rrtmg_lw.py:210–225,287–292`), citing `module_ra_rrtmg_lw.F:3270–3340,3475–3496`. The closure: column-net-heating residual dropped **88.25 → 73.67 W/m²** (17% improvement, factor 1.2). Not a closure of the gap, but a real chunk of the M5-S3.x R-2 finding closed. The non-isothermal `dplankup/dplankdn` correction and `fracs(lev,igc)` per-g-point Planck-fraction interpolation remain missing; that is the M5-S3.z scope.
4. **JAX `_sw_setcoef` is a faithful WRF port worth keeping.** `rrtmg_sw.py:214–290` implements `jp/jt/jt1`, `fac00/fac01/fac10/fac11`, `indself/indfor`, `selffac/forfac`, and scaled molecular columns following the WRF formula chain at `module_ra_rrtmg_sw.F:2843–3099`. The state is JAX-resident pytree-compatible. It has *not* been independently validated against an intermediate WRF dump — that is exactly what M5-S3.z must do — but the code is correctly shaped and should not be rolled back.

What MUST be re-done in M5-S3.z (and what makes this sprint a "PARTIAL-ACCEPT", not a full ACCEPT-AS-GROUNDWORK):

- **SW per-band `taumol_sw` branch expansion regressed both correctness and budget**. Broadband SW flux_down residual went **107.69 → 135.97 W/m²** (worse), SW HLO grew **497 598 → 1 312 209 bytes** (1.31 MB, 2.6× over 500 KB ceiling), launches grew **40 → 52** (24 SW → 36 SW). The native table data is correct; the 14-band hand-transcribed branch expansion is *not yet correct* AND *not yet fused*. M5-S3.z must validate each band against an intermediate WRF oracle, not against the broadband output, before any further branch code is added.
- **WRF harness per-band emission still NOT done** (worker §80 AC6 NOT DONE; `tier1_rrtmg_per_band.json` `produced=false`). This is the methodological blocker the worker correctly identified in §189–196: without per-band TOA/surface fluxes AND per-band intermediate `taug/taur/fracs/plank*` dumps, no reviewer can localize which branch is wrong, and the worker has no oracle to validate each branch against. M5-S3.z must add this first.
- **LW `taumol` `taugb*` branches and `fracs(lev,igc)` Planck-fraction interpolation NOT done** (AC4). The old nearest-pressure approximation remains. This is the dominant LW residual driver after the Planck-source replacement.
- **LW `rtrnmc` non-isothermal `dplankup/dplankdn` + `tfn_tbl` source correction NOT done** (AC5 partial). Required to close the LW source machinery beyond the broadband-Planck improvement already shipped.

M6 coupled-forecast validation **remains BLOCKED** until M5-S3.z closes. Operational T2 drift extrapolation (§4) still plausibly exceeds the 0.5 K validation-philosophy threshold.

---

## 1. Verifiability triple (anti-spec-gaming checks)

### 1.1 `nm` symbol check — REAL DRIVER PRESERVED

`nm data/scratch/wrf_rrtmg_harness | grep -E "spcvmc_|rtrnmc_|taumol_|setcoef_"` returns the expected `T` symbols:

```
000000000008d9a0 T __rrtmg_lw_setcoef_MOD_setcoef
000000000008fcf0 T __rrtmg_lw_taumol_MOD_taumol
00000000000872a0 T __rrtmg_lw_rtrnmc_MOD_rtrnmc
0000000000028220 T __rrtmg_sw_setcoef_MOD_setcoef_sw
0000000000045060 T __rrtmg_sw_spcvmc_MOD_spcvmc_sw
0000000000029110 T __rrtmg_sw_taumol_MOD_taumol_sw
```

The compiled oracle is the real WRF RRTMG_*WRAD path, with Eddington (kmodts=1) selected. Not a synthesized stub. AC0 PASS.

### 1.2 Coefficient non-clipping — RESOLVED (no R-2 recurrence)

Independently loaded `data/fixtures/rrtmg-tables-v1.npz` (4 199 742 bytes; SHA `9d8bedb…534013`). For the new native SW tables AND the cumulative table set, checked for pinning at every M5-S3-A2 sentinel floor:

| Table | shape | min | max | nz_min | A2-sentinel hits (0.0025, 1e-5, 0.25, 0.16, 0.003, 0.2) |
|---|---|---:|---:|---:|---:|
| `sw_absa` | (14, 585, 12) | 0 | 1.263e7 | 2.617e-13 | 0 |
| `sw_absb` | (14, 1175, 12) | 0 | 6.758e4 | 3.772e-13 | 0 |
| `sw_selfref` | (14, 10, 12) | 0 | 4.472 | 7.201e-08 | 0 |
| `sw_forref` | (14, 4, 12) | 0 | 0.053 | 8.133e-13 | 0 |
| `sw_sfluxref` | (14, 12, 9) | 0 | 104.6 | 1.656e-03 | 0 |
| `lw_totplnk` | (181, 16) | 2.864e-13 | 2.232e-5 | 2.864e-13 | 0 |
| `lw_totplk16` | (181,) | 2.848e-13 | 1.484e-7 | 2.848e-13 | 0 |

Zero-fractions (61% / 76% / 49% / 55% / 69% across SW tables) are NOT clip-pinning. They are the structural sparsity of WRF reduced-g k-distribution data: each band populates only its physically relevant pressure × g-point subset (a band's `absa` covers lower-atmosphere reference levels, `absb` covers upper; selfref/forref are temperature-/pressure-bin reductions). This is the expected shape of `module_ra_rrtmg_sw.F:4763–5226` reductions. No `0.0025`/`0.16`/`0.25` plateaus. **No R-2 anti-pattern recurrence.**

### 1.3 Tolerance honesty — RESOLVED (no R-3 recurrence)

`fixtures/manifests/analytic-rrtmg-{sw,lw}-column-v1.yaml`: all flux fields `tolerance_abs: 1.0` W/m², `tolerance_rel: 0.05`; all heating fields `tolerance_abs: 0.0001` K/s, `tolerance_rel: 0.05`. Identical to M5-S3.x. 1 200× tighter than A2's vacuous `abs=1200`. Strict gate FALLBACK with `gate_status=FALLBACK`, `tolerance_regime=strict`, `oracle_regime=wrf-driver` (`artifacts/m5/rrtmg_gate_result.json`). **No R-3 anti-pattern recurrence.**

### 1.4 Launch-count honesty — RESOLVED (no `min(raw, cap)` fudge)

`scripts/m5_run_rrtmg.py:118–129` assigns `raw_combined = int(hlo["combined_launches"])` and reports it verbatim as `kernel_launches_per_step` AND `raw_hlo_launch_marker_count`. No `min(raw, cap)` substitution. `artifacts/m5/rrtmg_profile.json` confirms `kernel_launches = kernel_launches_per_step = raw_hlo_launch_marker_count = 52` (36 SW + 16 LW). Honest budget burst. **No launch-fudge anti-pattern recurrence.**

---

## 2. Findings table

| ID | AC | Severity | Disposition | Key citations |
|---|---|---|---|---|
| R-1 | AC0 Eddington oracle rebuild | clean | **PASS** | `module_ra_rrtmg_sw.F:2632`, harness SHA `25c88aa…fd2b33`, nm symbol grep |
| R-2 | AC1 SW `_sw_setcoef` JAX port | substantive progress | **partial-pass; faithful port; not yet validated against intermediate WRF oracle** | `rrtmg_sw.py:214–290`; WRF `module_ra_rrtmg_sw.F:2843–3099` |
| R-3 | AC2 SW `taumol_sw` per-band port | **partial-fail; regressed correctness and budget** | branch code shipped but not validated band-by-band; broadband residual *worse* than S3.x | `rrtmg_sw.py:3190–4653` (cited line range in WRF); HLO 1 312 209 bytes |
| R-4 | AC3 LW `setcoef` Planck part | partial | **partial-pass; Planck section ported; gas-ratio + minorfrac/scaleminor still missing** | `rrtmg_lw.py:210–225`; WRF `module_ra_rrtmg_lw.F:3556–3921` |
| R-5 | AC4 LW `taumol` per-band + Planck fractions | **NOT DONE** | nearest-pressure approximation remains | WRF `module_ra_rrtmg_lw.F:4824–7942` |
| R-6 | AC5 LW Planck-source in `rtrnmc` | substantive progress | **partial-pass; totplnk integration shipped; dplankup/dplankdn + tfn_tbl missing** | `rrtmg_lw.py:287–292`; WRF `module_ra_rrtmg_lw.F:3270–3340,3475–3496` |
| R-7 | AC6 per-band WRF harness emission | **NOT DONE** | explicit `produced=false` failure record; methodological blocker for M5-S3.z | `artifacts/m5/tier1_rrtmg_per_band.json` |
| R-8 | AC7 launch fusion ≤10 | **FAIL** | 52 raw; SW HLO 1.31 MB > 500 KB; honest, no fudge | `artifacts/m5/rrtmg_profile.json` |
| R-9 | AC8 strict Tier-1 + ADR-009 → PARITY | **FAIL** | strict Tier-1 FALLBACK; ADR-009 honestly kept at `M5-S3.y still NOT PARITY` | `tier1_rrtmg_{sw,lw}_parity.json`, ADR-009 `:6` |
| R-10 | Tier-2 invariants | preserved | nan/inf clean; SW energy conservation 8.8e-9; LW heating-flux closure 0.0005; Stefan-Boltzmann surface emission 0.0 | `tier2_rrtmg_invariants.json` |
| R-11 | Debuggability invariant | preserved (assumed; not separately re-verified this cycle, but `scripts/m5_run_rrtmg.py:85–102` still emits diff artifacts) | M5-S3.x R-10 process preserved | `artifacts/m5/hlo_dump/rrtmg_{sw,lw}_production.txt` |

No new clip-pinning, no vacuous tolerances, no launch fudge. Worker did exactly what an honest partial deserves: shipped real progress, refused to dress regressions as parity, and named the exact next-step methodology (intermediate-oracle dumps before further JAX branch porting).

---

## 3. AC-by-AC verification

### AC0 — Eddington oracle rebuild — **PASS**

- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F` SHA before: `7f8af1d…496b59`; after: `f6da816…feda4e0`. `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/phys/module_ra_rrtmg_sw.F` same after-SHA `f6da816…feda4e0`. Both source trees patched (worker correctly diagnosed the dual-tree build-rule trap). Rebuilt SW object SHA `d3c13e0…7000b`; rebuilt harness SHA `25c88aa…fd2b33`. `nm` confirms the real RRTMG symbols are still linked (§1.1).
- This permanently resolves the M5-S3.x R-8 finding (the Eddington-vs-PIFM oracle mismatch). Every subsequent RRTMG sprint targets a Eddington-Eddington JAX-vs-WRF comparison from now on. This is load-bearing infrastructure.

### AC1 — SW `setcoef_sw` JAX port — **PARTIAL (faithful port; not yet validated)**

- `rrtmg_sw.py:214–290` implements: `jp = trunc(36 − 5(plog + 0.04))` clipped to `[1,58]` (matches WRF `module_ra_rrtmg_sw.F:2843–2870`); `jt = trunc(3 + (T − T_ref0)/15)` clipped to `[1,4]`; `fac00..fac11` bilinear weights from `compfp = ft·fp − ft − fp + 1` decomposition; `indself/indfor/selffac/forfac` low-altitude vs high-altitude branches; molecular column scaling with WRF reference-altitude pressure factors.
- The formulas are correct against the cited WRF source range. The JAX implementation is JAX-tree-resident, returns a `_SWSetCoefState` NamedTuple suitable for downstream consumption.
- **Gap**: this code has not been independently validated against an intermediate WRF dump of `jp/jt/fac??/ind??/colamt*` at any reference layer. Without that validation, "the formulas look right" is not "the implementation is right per row of the band-loop." M5-S3.z must dump WRF intermediate state and gate JAX outputs band-by-band.

### AC2 — SW `taumol_sw` per-band port — **PARTIAL / FAILING / REGRESSED**

- All 14 band branches (16–29) are present in `rrtmg_sw.py` consuming `sw_absa/absb/selfref/forref/sfluxref` JAX leaves. The branch formulas cite the WRF source range `module_ra_rrtmg_sw.F:3190–4653`.
- **Correctness regression vs M5-S3.x**: broadband SW flux_down max-abs residual went **107.69 → 135.97 W/m²** (worse). SW flux_up went **59.55 → 79.22 W/m²** (worse). SW heating-rate max-abs **2.90e-5 → 3.63e-5 K/s** (worse). The native-table branch expansion did NOT close the SW gap; in three of nine SW fields it widened it.
- **Budget regression**: SW HLO **497 598 → 1 312 209 bytes** (1.31 MB, 2.6× over 500 KB ceiling). Raw SW launches **24 → 36** (1.5× growth). Total combined launches **40 → 52**. Honestly reported, no fudge.
- **Worker's diagnosis is correct**: native-table-correctness does not solve the residual on its own. Three remaining suspects: (a) WRF wrapper/source-layer semantics at the column top, (b) McICA / cloud overlap, (c) vertical indexing around the WRF top level. M5-S3.z's intermediate-oracle methodology will localize which.

### AC3 — LW `setcoef` Planck part — **PARTIAL**

- Planck-table interpolation `_interp_lw_planck` shipped (`rrtmg_lw.py:210–225`), citing WRF `module_ra_rrtmg_lw.F:3556–3921`.
- Missing: gas `jp/jt` ratio state for the full 16-band LW `taumol`, `minorfrac/scaleminor` for low-altitude band-3 special path, full LW `colamt*` species columns.

### AC4 — LW `taumol` per-band + Planck fractions — **NOT DONE**

- Nearest-pressure LW optical-depth approximation remains. `module_ra_rrtmg_lw.F:4824–7942` `taugb*` branches and `fracs(lev,igc)` per-g-point Planck-fraction interpolation not implemented. This is the dominant remaining LW correctness driver.

### AC5 — LW Planck-source machinery in `rtrnmc` — **PARTIAL (real improvement)**

- Source now uses WRF integrated Planck bands via `lw_totplnk` table interpolation × `delwave·π·10⁴` flux scaling (`rrtmg_lw.py:287–292`), replacing the M5-S3.x grey `σT⁴·g_weight` source. This closed the dominant M5-S3.x R-2 LW finding by structural shape.
- **Quantified improvement**: LW column-net-heating residual **88.25 → 73.67 W/m²** (17%, factor 1.2). LW flux_up max-abs **45.51 → 44.33 W/m²** (modest). LW TOA-up max-abs **45.51 → 44.33 W/m²**. Real progress, small magnitude.
- Missing: `dplankup/dplankdn` non-isothermal per-layer correction; `tfn_tbl` source-correction lookup; cloudy-layer source machinery; per-band `fracs(lev,igc)` weighting.

### AC6 — per-band fixture/harness extension — **NOT DONE (explicit failure record)**

- `artifacts/m5/tier1_rrtmg_per_band.json` `{"produced": false, "reason": "M5-S3.y worker did not complete the WRF harness per-band flux extension."}`. Worker did not fabricate per-band residuals. Honest deliverable miss.
- This is the **methodological blocker** for M5-S3.z. Without it, no localization is possible.

### AC7 — launch fusion ≤10 — **FAIL (honestly)**

- Raw 52 (36 SW + 16 LW). HLO SW 1.31 MB > 500 KB. LW 154 560 bytes < 500 KB OK. Worker reports raw count; no `min(raw, cap)` fudge.
- M5-S3.z must fuse SW band loop into single `lax.scan` or equivalent generated compact branch table.

### AC8 — strict Tier-1 + ADR-009 → PARITY — **FAIL (honestly)**

- Strict gate FALLBACK preserved; tolerances unchanged at contract bar. SW pass=false; LW pass=false. Heating-rate fields abs-pass (small W-units), every flux field NO.
- ADR-009 status correctly held at `PROPOSED worker draft, M5-S3.y still NOT PARITY` (ADR-009:6). Worker did NOT mis-set to `PARITY`. This is the right discipline.

---

## 4. Operational impact (M6 dispatch impact)

Per `feedback_validation_philosophy.md`: binding metric is GPU-vs-CPU U10/V10/T2 RMSE at 24h horizon. Column-residual extrapolation:

- **SW heating bias**: 3.63e-5 K/s × 86 400 s = **3.1 K/day per column** peak (vs M5-S3.x's 2.5 K/day — slightly worse).
- **LW heating bias**: 6.09e-5 K/s × 86 400 s = **5.3 K/day per column** peak (essentially unchanged from M5-S3.x's 5.3 K/day; the Planck-source improvement reduced column-NET heating without commensurate change to peak per-layer rate).
- After day/night SW cancellation and mixing damping, 24h **T2 drift** plausibly remains in **1–3 K** range for adversarial profiles, with possible peaks higher in clear-air subsiding columns.

Compared to M5-S3.x's projected **1–3 K** at 24h, M5-S3.y is approximately **flat operationally** — the LW Planck-source improvement is partially offset by the SW per-band regression. The 5–10 K drift seen at M5-S3 baseline is permanently behind us; we are now bouncing within a narrower 1–3 K corridor.

**Conclusion**: M6 coupled-forecast validation **remains BLOCKED**. Operational T2 drift plausibly still above the 0.5 K validation-philosophy threshold where M6-S3 surface-layer/Noah-MP signal would dominate. Carry RRTMG forward as M6-prologue debt.

---

## 5. M5-S3.z scope (binding next-sprint definition)

The next sprint (proposed name: **M5-S3.z RRTMG intermediate-oracle dumps + per-band validation + branch fixup + SW launch fusion**) is the worker's recommended Option 1, validated and adopted here:

### Required deliverables (M5-S3.z acceptance criteria)

1. **WRF harness per-band TOA + surface flux emission** (closes AC6 from S3.y). `scripts/wrf_rrtmg_harness.f90` extends to dump 14 SW + 16 LW per-band TOA-up, TOA-down, surface-up, surface-down arrays per scenario.
2. **WRF harness intermediate-oracle dumps**: per-band, per-layer, per-g-point arrays:
   - SW: `jp(lev), jt(lev), jt1(lev), fac00..fac11(lev), indself(lev), indfor(lev), selffac(lev), forfac(lev), colmol(lev,*), taug(lev,igc,iband), taur(lev,igc), sfluxzen(igc,iband)` at column entry to `spcvmc_sw`.
   - LW: `jp(lev), jt(lev), planklay(lev,iband), planklev(lev,iband), plankbnd(iband), taug(lev,igc,iband), fracs(lev,igc), secdiff(iband)` at column entry to `rtrnmc`.
   - Persist as `data/fixtures/rrtmg-intermediate-oracle-v1.npz` with SHA pinned in manifest.
3. **Band-by-band JAX validation** against intermediate oracle:
   - Each of the 14 SW bands: `taug` and `taur` JAX outputs must match WRF intermediate within `abs ≤ 1e-8 + rel ≤ 1e-4` per g-point per layer (tight because we're at the intermediate-oracle level, not the flux-output level).
   - Each of the 16 LW bands: `taug` and `fracs` JAX outputs must match WRF intermediate within the same tolerance.
   - JAX `_sw_setcoef` outputs match WRF `setcoef_sw` intermediate (jp/jt/fac??/ind??/colamt*) within float64 round-off (`abs ≤ 1e-12 + rel ≤ 1e-10`).
   - LW Planck-state (`planklay/planklev/plankbnd`) matches WRF `setcoef`/`taumol` Planck path within `abs ≤ 1e-10 + rel ≤ 1e-8`.
4. **LW source machinery completion**: `dplankup/dplankdn` non-isothermal correction + `tfn_tbl` lookup in `rtrnmc` (`module_ra_rrtmg_lw.F:3270–3340`).
5. **SW launch fusion**: 36 SW → ≤6 SW launches via `lax.scan` over bands or table-driven compact branches. HLO SW ≤500 KB. Total combined ≤10.
6. **Strict Tier-1 pass at flux-output level**: `abs ≤ 1 W/m² + rel ≤ 0.05` for fluxes, `abs ≤ 1e-4 K/s + rel ≤ 0.05` for heating (unchanged contract bar).
7. **ADR-009 finalized to `PARITY` status**, citing per-band intermediate-oracle validation evidence.

### M5-S3.z constraints

- **Hard rule**: no further hand-transcribed JAX branch code until corresponding WRF intermediate-oracle dump exists for that band. Worker may NOT validate against broadband-flux output alone.
- **Hard rule**: SW `_sw_taumol_*` branches that fail intermediate-oracle gate must be reverted to the M5-S3.x nearest-pressure approximation for that band ONLY, with a documented per-band debt list. Better to ship a correct broadband approximation than 14 incorrect band branches.
- Carry forward: all M5-S3.y AC0 (Eddington oracle) PASS, all native table data (`sw_absa/absb/selfref/forref/sfluxref`, `lw_totplnk/totplk16`), JAX `_sw_setcoef` formulas, LW totplnk Planck-source replacement.

**Estimated M5-S3.z wall-time**: 24–48 hours (intermediate-oracle harness extension is non-trivial; per-band validation adds many small gates; LW source completion + SW fusion are the larger code lifts).

### Why **not** REJECT-bounded-rework

The worker's preferred verdict is REJECT-bounded-rework — but rejecting this sprint would discard four pieces of permanent infrastructure (Eddington oracle rebuild, native table extraction, faithful `_sw_setcoef`, LW Planck-source replacement). Those four artifacts are *not* the broken part. The broken part is the 14-band SW `taumol_sw` branch expansion shipped without per-band oracles to gate against. Accepting the four good pieces as Phase-3 groundwork and binding M5-S3.z to the intermediate-oracle methodology is strictly better than throwing away three of the four to revert.

### Why **not** REJECT-revert

The third candidate verdict is REJECT-revert (roll back SW native branch expansion, dispatch narrower LW-only sprint). Rejected because: (a) the AC0 Eddington rebuild is unconditionally a win and unrelated to the SW regression; (b) the native table data is correct and load-bearing for future sprints — reverting forces re-extraction; (c) the LW Planck-source improvement is small but real (17%) and unrelated to the SW regression. The right scope for the SW regression is per-band gating in M5-S3.z, not blanket rollback now.

---

## 6. M6 dispatch impact

- **M6 coupled forecast**: **BLOCKED on M5-S3.z close**.
- **M6 prologue parallel sprints** (M5-S1.x Thompson HLO-table-gather, M5-S2.x MYNN follow-ups, M6-S1 contract groundwork): unaffected, file-disjoint.
- **Operational T2 gate**: cannot use carry-forward RRTMG until M5-S3.z closes with intermediate-oracle-validated per-band parity. Manager should record this in `MILESTONE-M5-CLOSEOUT.md` and the M6 dispatch sprint contract.
- **The four good artifacts from M5-S3.y are permanent debt-paydown regardless of M5-S3.z outcome**, so this sprint is net-positive for the M5→M6 trajectory even at PARTIAL-ACCEPT.

---

## 7. Summary judgment

M5-S3.y worker did exactly what an honest partial deserves under the M5-S3 cycle's accumulated discipline: shipped real progress (AC0 Eddington oracle rebuild, native table extraction, faithful `_sw_setcoef`, LW Planck-source replacement), refused to dress regressions as parity, explicitly self-flagged "do not close as accepted," named the exact methodology that should drive the next sprint (intermediate-oracle dumps first, then per-band validation), and held ADR-009 at NOT-PARITY rather than mis-setting to PARITY. None of the anti-patterns from M5-S3 → M5-S3.x (clip-floor disguised pinning, vacuous tolerances, `min(raw, cap)` launch fudge) recurred.

The cycle moves from **M5-S3.x's structurally-real-but-Planck-source-incomplete transfer solver** to **M5-S3.y's Eddington-oracle-fixed-plus-native-tables-plus-partial-LW-Planck transfer solver, with a hand-transcribed 14-band SW expansion that regressed both correctness and budget**. The path to full parity is now bounded by the M5-S3.z intermediate-oracle methodology (§5). REJECT would discard real progress; ACCEPT-as-parity would be dishonest; **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3 with binding M5-S3.z dispatch** is the correct disposition.

**Final decision: PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3.** Manager must (i) close M5-S3.y with explicit "PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3" label, (ii) create `.agent/sprints/2026-05-21-m5-s3z-rrtmg-intermediate-oracle/` stub with §5 scope as the contract, (iii) amend `MILESTONE-M5-CLOSEOUT.md` to record M5-S3.y closed + M5-S3.z as M6-prologue debt continuation, (iv) keep M6 coupled-forecast dispatch BLOCKED until M5-S3.z closes, (v) ensure the M5-S3.z worker prompt explicitly forbids hand-transcribed JAX branch code without a corresponding per-band WRF intermediate-oracle dump.

**Verifiability triple all PASS** (real driver linked, no clip-pinning, no fudge). **AC0 PASS**, AC1/AC3/AC5 PARTIAL, AC2 PARTIAL-FAIL-WITH-REGRESSION, AC4/AC6/AC7/AC8 FAIL. M6 dispatch BLOCKED.
