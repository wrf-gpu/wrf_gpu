# Reviewer Report — M5-S3.x RRTMG Transfer-Solver Rewrite (binding close decision)

**Reviewer**: Claude Opus 4.7 xhigh (fresh-context, per sprint-lifecycle double-AI HARD RULE, `.agent/rules/sprint-lifecycle.md:14-32`)
**Date**: 2026-05-21 (post-A3 cycle)
**Branch / commit under review**: `worker/codex/m5-s3x-rrtmg-transfer-solver` @ `cbce2e5 M5-S3.x partial RRTMG transfer solver rewrite`
**Worker**: Codex GPT-5.5 xhigh
**Worker self-verdict**: "This pass does not meet M5-S3.x acceptance. It is useful groundwork but remains a blocker for M6 coupled validation… This report is intentionally not an acceptance closeout." (`worker-report.md:11-17`)
**Prior cycle precedent**: M5-S3 closed ACCEPT-AS-GROUNDWORK (`reviewer-a3-report.md:11`); R-2 (clip-floor disguised pinning) and R-3 (vacuous tolerances) were the two anti-patterns to guard against this cycle.

---

## Reviewer decision: **ACCEPT-AS-GROUNDWORK-PHASE-2**

M5-S3.x closes as **Phase-2 groundwork**: the new SW Eddington + Joseph-Wiscombe-Weinman delta-scaling + `vrtqdr_sw`-style adding solver and the LW reduced-g-point diffusivity recurrence are real, faithful, WRF-shape ports of the transfer-solver structure. The fabricated `tau_gas = vapor_path * 0.01 * log1p(gas_coeff)` curve identified in A3 §2 is verifiably gone. The clip-floor anti-pattern that fell A2 has NOT recurred (0 / 9 912 active SW absorption-coefficient values pinned to any A2 floor — see §1.2). Tolerances are NOT vacuous (output-side `abs=1.0 W/m² + rel=0.05` for fluxes, `abs=1e-4 K/s + rel=0.05` for heating — matching the AC1 contract bar exactly). The launch budget fails honestly (`raw_hlo_launch_marker_count == kernel_launches_per_step == 40` with no `min(raw, cap)` substitution).

The implementation does NOT meet strict Tier-1 because (a) full SW `setcoef_sw` + `taumol_sw` per-band gas optical-depth interpolation, (b) full LW `setcoef` + `taumol` Planck-fraction interpolation, and (c) full LW `rtrnmc` per-band Planck-source machinery are still missing. This is exactly what the worker discloses. The remaining work is well-scoped and should be carved out as a named **M5-S3.y** sprint (defined in §5 below).

M6 coupled-forecast validation **remains blocked** until M5-S3.y closes, because the operational T2 drift extrapolation (§4) still plausibly exceeds the validation-philosophy noise floor.

---

## 1. Verifiability triple (anti-spec-gaming checks)

### 1.1 `nm` symbol check — REAL DRIVER PRESERVED

`nm /tmp/wrf_gpu2_s3x/data/scratch/wrf_rrtmg_harness | grep -E "spcvmc_|rtrnmc_|taumol_|setcoef_|cldprmc_"` returns all 5 expected symbols, including:

- `__rrtmg_sw_spcvmc_MOD_spcvmc_sw` (T)
- `__rrtmg_lw_rtrnmc_MOD_rtrnmc` (T)
- `__rrtmg_sw_taumol_MOD_taumol_sw` (T)
- `__rrtmg_lw_taumol_MOD_taumol` (T)
- `__rrtmg_sw_setcoef_MOD_setcoef_sw` (T)
- `__rrtmg_lw_setcoef_MOD_setcoef` (T)
- `__rrtmg_sw_cldprmc_MOD_cldprmc_sw` (T)
- `__rrtmg_lw_cldprmc_MOD_cldprmc` (T)

A3's real-driver binding is preserved verbatim. The oracle remains the compiled WRF RRTMG_*WRAD path, not a synthesized stub.

### 1.2 Coefficient non-clipping — RESOLVED (no R-2 recurrence)

Independently loaded `data/fixtures/rrtmg-tables-v1.npz` (1 747 874 bytes; regenerated this cycle with `sw_cloud_liquid_asymmetry` / `sw_cloud_ice_asymmetry` arrays added). For all 9 912 active SW absorption-coefficient values:

- Pinned to `0.0025`: 0
- Pinned to `1e-5`: 0
- Pinned to `0.25`: 0
- Pinned to `0.16`: 0
- Pinned to `0.003`: 0
- Pinned to `0.2`: 0

Active distribution: `min=0, max=1.22e7, mean=2 008`. Real spectral spread, no clip-floor disguise.

The new SW cloud-asymmetry arrays show physically plausible values: liquid g ∈ [0.845, 0.917] mean 0.862; ice g ∈ [0.745, 0.933] mean 0.814. These are within the textbook range for Mie/geometric-optics cloud particles and are extracted from the real WRF source per `scripts/extract_rrtmg_tables.py` (worker-report §66).

### 1.3 Tolerance honesty — RESOLVED (no R-3 recurrence)

`fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml:169-252` and `fixtures/manifests/analytic-rrtmg-lw-column-v1.yaml:168-251`:

- All flux fields: `tolerance_abs: 1.0` W/m², `tolerance_rel: 0.05` (5 %).
- All heating fields: `tolerance_abs: 0.0001` K/s (≈8.6 K/day), `tolerance_rel: 0.05`.

These match the M5-S3.x sprint contract AC1 bar exactly. They are 1 200× tighter than A2's vacuous `abs=1200 / rel=15` and 10–25× tighter than the A2-reviewer's recommended carry-forward bound. Strict gate FALLBACK with `rationale="correctness failed"` (`artifacts/m5/rrtmg_gate_result.json:8`). No `min(raw, cap)` clamping in `scripts/m5_run_rrtmg.py`; `raw_hlo_launch_marker_count = kernel_launches_per_step = 40` (`rrtmg_profile.json:21,26`).

---

## 2. Findings table

| ID | AC | Severity | Disposition | Key citations |
|---|---|---|---|---|
| R-1 | AC1 SW Eddington + δ-scaling | substantive progress | **partial-pass; sound algebra** | `rrtmg_sw.py:192-274`; WRF `module_ra_rrtmg_sw.F:2647-2802` |
| R-2 | AC2 LW correlated-k | substantive progress | **partial-pass; sound recurrence, missing Planck source** | `rrtmg_lw.py:200-286`; WRF `module_ra_rrtmg_lw.F:3270-3522` |
| R-3 | AC3 real gas absorption | partial | **`log1p` curve gone; nearest-pressure interpolation in place of `setcoef`+`taumol`** | `rrtmg_sw.py:164-189`, `tests/test_m5_rrtmg_transfer_solver.py:39-42` |
| R-4 | AC4 cloud overlap / radii | partial | **fixed effective radii (10 / 30 µm); deterministic overlap; documented honestly** | `worker-report.md:88-95` |
| R-5 | AC5 strict Tier-1 | **fail, intended** | residuals reported honestly (§3) | `tier1_rrtmg_{sw,lw}_parity.json` |
| R-6 | AC6 HLO + launches | **pass HLO, fail launches honestly** | 497 598 / 136 941 bytes (under 500 KB); 40 raw launches | `rrtmg_gate_result.json:3-9` |
| R-7 | AC7 ADR-009 amend | pass | Joseph 1976 + Meador-Weaver 1980 + Mlawer 1997 cited, gaps disclosed | `ADR-009-rrtmg-jax-implementation.md:42-48,60-64` |
| R-8 | Eddington-vs-PIFM oracle mismatch | structural | **flagged honestly; orthogonal to most of the residual** | worker-report §53; WRF `kmodts=2` at `module_ra_rrtmg_sw.F:2632` |
| R-9 | Per-band residual table | deliverable miss | worker self-flagged at `worker-report:128`; carry into M5-S3.y | n/a |
| R-10 | Debuggability invariant | preserved | both diffs at 0 bytes | `hlo_dump/rrtmg_*_debug_vs_stripped.diff` |

No new R-2-in-disguise or R-3-in-disguise anti-patterns detected.

---

## 3. AC-by-AC verification

### AC1 — SW Eddington + delta scaling — **STRUCTURALLY SOUND; partial closure**

Cross-checked the JAX implementation against WRF `reftra_sw` (`/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:2563-2810`).

- **Delta scaling** (`rrtmg_sw.py:192-201`): `f = g²`; `τ' = (1 − f·ω)·τ`; `ω' = (1−f)·ω / (1 − f·ω)`; `g' = (g − f)/(1 − f)`. Matches Joseph-Wiscombe-Weinman 1976 (JAS 33, 2452–2459) and WRF `module_ra_rrtmg_sw.F:8603-8610` byte-by-byte algebra (modulo Fortran/JAX vector layout).
- **Eddington γ coefficients** (`rrtmg_sw.py:211-215`): `γ1 = (7 − ω(4+3g))/4`; `γ2 = -(1 − ω(4−3g))/4`; `γ3 = (2 − 3g·μ0)/4`; `γ4 = 1 − γ3`. These are the Meador-Weaver 1980 (JAS 37, 630–643) Eddington Table 1 row, and match WRF `module_ra_rrtmg_sw.F:2647-2651` (the `kmodts == 1` Eddington branch) verbatim.
- **Adversarial probe**: for `ω=0.95, g=0.85, μ0=0.5`: γ1 = 0.194375, γ2 = 0.094375, γ3 = 0.18125, γ4 = 0.81875. These match Eddington Table 1 by hand; JAX produces these.
- **Non-conservative reflectance/transmittance** (`rrtmg_sw.py:230-263`): the `zr1..zr5`, `zt1..zt5`, `zrk`, `zbeta = (γ1 − zrk)/(zrk + γ1)`, `zdend = 1/((1 − zbeta·exp(-2·zrk·τ))·zrkg)` structure matches WRF `module_ra_rrtmg_sw.F:2714-2802` exactly. The `mji-reformulated` zbeta variant (avoiding floating-point overflow) is preserved.
- **Conservative limit** (`rrtmg_sw.py:221-228`): `pref_cons = (zgt − za_cons·(1 − exp_mu))/(1 + zgt)`, `prefd_cons = zgt/(1 + zgt)`. Matches WRF `module_ra_rrtmg_sw.F:2696-2702`.
- **Vertical adding** (`_vertical_quadrature`, `rrtmg_sw.py:277-346`): faithful port of WRF `vrtqdr_sw` at `module_ra_rrtmg_sw.F:8108-8156`. Surface-layer coupling, upward pass (bottom-to-top), downward pass (top-to-bottom), and final `pfu/pfd` accumulation algebra all match.

**Caveat (R-8)**: the local WRF source pins `kmodts=2` (`module_ra_rrtmg_sw.F:2632`), which selects **PIFM, not Eddington**, in the compiled oracle. The sprint contract asked for Eddington explicitly (sprint-contract.md:11), so the JAX implements Eddington against a PIFM oracle. PIFM γ1 = (8 − ω(5+3g))/4 and γ2 = 3ω(1−g)/4; for `ω=0.95, g=0.85`, PIFM γ1 = 0.207 vs Eddington γ1 = 0.194 (≈6.5 % delta). This produces an unavoidable per-band Eddington-vs-PIFM bias even with a perfect port — typically 5–15 W/m² per band in mid-latitudes. The worker flagged this honestly at `worker-report.md:53`. This is a contract-vs-oracle definitional question, NOT an implementation bug.

### AC2 — LW correlated-k — **PARTIAL; structural recurrence correct, Planck-source machinery missing**

Cross-checked against WRF `rtrnmc` (`module_ra_rrtmg_lw.F:3085-3522`).

- **Diffusivity** (`rrtmg_lw.py:189-197`, constants `:31-83`): `secdiff = a0 + a1·exp(a2·pwvcm)`, clipped to `[1.50, 1.80]` for variable bands `{2,3,5,6,7,8,9}`, fixed `1.66` for bands `{1,4,10..16}`. `LW_DIFFUSIVITY_A0/A1/A2` constants exactly match WRF `data a0/a1/a2` at `module_ra_rrtmg_lw.F:3069-3081`; the `variable` mask matches WRF lines `3253-3261`.
- **Source recurrence** (`_source_recurrence_down/up`, `rrtmg_lw.py:200-221`): each layer `rad_new = rad·trans + source·(1 − trans)`. This is the standard layer-emission recurrence and matches the `rtrnmc` downward loop at `module_ra_rrtmg_lw.F:3320-3340` in algorithmic shape.
- **MISSING — Planck source**: WRF `rtrnmc` uses `plfrac = fracs(lev,igc)` (the per-g-point Planck fraction interpolated from `setcoef`/`taumol`) times `planklay(lev,iband)` (per-layer per-band Planck function), with `dplankup/dplankdn` correction for non-isothermal layers (`module_ra_rrtmg_lw.F:3323-3331`). The JAX implementation at `rrtmg_lw.py:256` uses `layer_source = σ·T⁴ × g_weights`, i.e. **grey-body × g-point quadrature weight**. This collapses both the per-band Planck fraction and the per-layer Planck-increment correction. Broadband column total ≈ correct (because `σT⁴ × Σweights = σT⁴`); per-band redistribution is **wrong by construction**, which is the dominant source of the LW residual at TOA (45.5 W/m² up, 75.6 W/m² down at one or both interfaces — see §3).

### AC3 — real gas absorption — **`log1p` curve verifiably gone; nearest-reference-pressure interpolation in place of full taumol**

- `grep -rn "log1p\|tau_gas = vapor_path" src/gpuwrf/physics/` returns **no matches**. The A3 §2 fabrication is removed.
- `tests/test_m5_rrtmg_transfer_solver.py:39-42` is a non-vacuous anti-regression test that fails if `log1p` or `vapor_path` reappears in `rrtmg_sw.py` source.
- Replacement (`rrtmg_sw.py:164-189`, `rrtmg_lw.py:159-186`): WRF-shaped molecular column construction (`colh2o`, `colco2`, `colo3`, `coln2o`, `colch4`, `colo2`) per `module_ra_rrtmg_sw.F:11084+` style, plus an `absorber` weighted-sum mixture passed through `_nearest_pressure_coefficients` (nearest-reference-pressure lookup into `tables.sw_absorption_coefficients`/`lw_absorption_coefficients`).
- **Gap**: this is not full `setcoef_sw` + `taumol_sw` (band-by-band lookup with T and pressure-interpolation factors `fac00/fac01/fac10/fac11`, per-species `colamt` paths through 14 SW + 16 LW `taumol_gb*` branches). The current approximation is "best WRF k-distribution row × molecular column" rather than "k(T,p,vmr) interpolated at each (band, g-point, layer)". This is the dominant SW residual driver after the Eddington-vs-PIFM bias.

### AC4 — cloud overlap — **partial; honestly disclosed**

- `sw_cloud_liquid_extinction`, `sw_cloud_liquid_ssa`, `sw_cloud_liquid_asymmetry`, `sw_cloud_ice_*` now present in `rrtmg-tables-v1.npz` (the asymmetry arrays are new this cycle, regenerated by `scripts/extract_rrtmg_tables.py:403-434`).
- Effective radii **fixed** at `r_e_liquid = 10 µm`, `r_e_ice = 30 µm` (matching harness defaults but not the WRF inflow `re_cloud`/`re_ice` fields). Snow/graupel folded into liquid/ice path partitions (`rrtmg_sw.py:367-368`).
- McICA / maximum-random / exponential-random overlap NOT implemented; the kernel uses deterministic cloud-fraction multiplication. This is documented as a gap (`worker-report.md:88-94`).

### AC5 — strict Tier-1 — **FAIL (honestly), tolerances NOT vacuous**

| Field | Max abs err | Max rel err | Pass? |
|---|---|---|---|
| **SW** flux_down | 107.69 W/m² | 1.00 | NO |
| SW flux_up | 59.55 W/m² | 1.13 | NO |
| SW toa_down | 67.04 W/m² | 0.078 | NO |
| SW toa_up | 33.06 W/m² | 0.111 | NO |
| SW surface_down | 58.95 W/m² | 0.986 | NO |
| SW surface_up | 14.57 W/m² | 0.986 | NO |
| SW column_absorbed | 111.54 W/m² | 0.481 | NO |
| SW heating_rate | 2.90e-5 K/s | 1.199 | **abs-pass** |
| **LW** flux_down | 75.56 W/m² | 0.703 | NO |
| LW flux_up | 45.51 W/m² | 0.198 | NO |
| LW toa_up | 45.51 W/m² | 0.198 | NO |
| LW surface_down | 9.85 W/m² | 0.027 | **rel-pass** |
| LW surface_up | 0.60 W/m² | 0.002 | **PASS** |
| LW toa_down | 0.0 | 0.0 | **PASS** |
| LW heating_rate | 6.15e-5 K/s | 19.94 | **abs-pass** |
| LW column_net_heating | 88.25 W/m² | 0.674 | NO |

**Improvement over A3** (per `reviewer-a3-report.md:34` and `reviewer-a2-report.md:83-89`):
- SW heating bias: 6.4e-4 → 2.9e-5 K/s (factor **22× smaller**)
- SW flux_down: 909 → 108 W/m² (factor **8.5×**)
- SW flux_up: 1 579 → 60 W/m² (factor **26×**)
- LW flux_down: 411 → 76 W/m² (factor **5.4×**)
- LW column_net_heating: 126 → 88 W/m² (factor **1.4×**)

This is substantive progress, not cosmetic relabeling. The new SW Eddington + δ-scaling + adding solver is closing a real chunk of the gap that A3's hand-rolled Beer-Lambert column could not.

### AC6 — HLO + launches — **HLO pass; launches fail honestly**

- `hlo_production_bytes_sw = 497 598` (under 500 KB ceiling, 0.5 % margin).
- `hlo_production_bytes_lw = 136 941` (well under).
- `kernel_launches_per_step = 40` (24 SW + 16 LW), AC bar ≤10. Fails by 4×.
- `raw_hlo_launch_marker_count == kernel_launches_per_step == 40` — honest, no fudge.
- Debuggability invariant: SW + LW `debug_vs_stripped.diff` both 0 bytes ✓.

### AC7 — ADR-009 amendment — **PASS**

`ADR-009-rrtmg-jax-implementation.md`:
- §"Implemented Formulas" lines 42-48 cite Joseph-Wiscombe-Weinman 1976 (delta scaling), Meador-Weaver 1980 (Eddington γ), and Mlawer et al. 1997 (correlated-k structure).
- §"Decision" lines 8-16 declares the result NOT accepted as full parity.
- §"Validation And Gate Status" lines 50-58 records strict Tier-1 fail + honest launch count.
- §"WRF Source Mapping" lines 18-40 cites the exact WRF source-line ranges for `setcoef`, `taumol`, `spcvmc`, `vrtqdr_sw`, `reftra_sw`, `rtrnmc`, `cldprmc`.
- The `kmodts=2` Eddington-vs-PIFM oracle mismatch is explicitly documented at line 31.

This is a substantive amendment, not a hand-wave.

---

## 4. Operational-impact extrapolation

Per validation-philosophy memory: the binding gate is GPU-vs-CPU U10/V10/T2 RMSE at the 24h horizon against the same CPU-vs-observation noise floor. The column-residual extrapolation:

- **SW heating bias**: 2.9e-5 K/s × 86 400 s = **2.5 K/day per column** (peak; mean likely 1–1.5 K/day across the 3 scenarios).
- **LW heating bias**: 6.15e-5 K/s × 86 400 = **5.3 K/day per column** (peak; LW persists 24h whereas SW is daylight-only, so net diurnal contribution is roughly equal).

After typical day/night SW cancellation and atmospheric mixing damping, the **24-hour T2 drift** is plausibly in the **1–3 K range** for adversarial profiles, with possibly higher peaks in clear-air subsiding columns where the radiation forcing is large but mixing is weak.

Compared to A3's estimate of **5–10 K T2 drift** (reviewer-a3-report.md:88), this is a **roughly 3–5× improvement** but still **above** the < 0.5 K threshold at which the M6-S3 surface-layer / Noah-MP signal would dominate the operational gate.

**Conclusion**: this M5-S3.x partial is **NOT M6 coupled-validation ready**. RRTMG remains carry-forward debt. But it is materially closer than A3's hand-rolled Beer-Lambert, and the remaining gap (`setcoef`+`taumol`+Planck-source) is **scoped and named**, not unbounded.

---

## 5. Required M5-S3.y scope (the remaining work)

The next sprint (proposed name: **M5-S3.y RRTMG `setcoef`+`taumol`+Planck-source port**) should close:

1. **SW `setcoef_sw` port**: pressure/temperature interpolation factors `fac00/fac01/fac10/fac11`, reference-pressure index `indfor/indself`, jp/jt lookups (`module_ra_rrtmg_sw.F:2843-3099`). Expose these as JAX-table-resident state.
2. **SW `taumol_sw` per-band port**: 14 band branches (bands 16–29) each computing `taug(ig) = colamt × k(jp,jt,ig) + selfref + forref` (`module_ra_rrtmg_sw.F:3190-4653`). Major code-size impact; mitigate with table-driven lookup over a JIT-unrolled band loop.
3. **LW `setcoef` port**: analog to SW (`module_ra_rrtmg_lw.F:3556-3921`).
4. **LW `taumol` per-band port**: 16 band branches each computing per-g-point `taug` AND `fracs(lev,igc)` (the Planck fraction needed for `rtrnmc`). (`module_ra_rrtmg_lw.F:4824-7942`).
5. **LW Planck-source machinery in `rtrnmc`**: `planklay(lev,iband)`, `planklev(lev,iband)`, `plankbnd(iband)` with `dplankup/dplankdn` per-layer non-isothermal correction (`module_ra_rrtmg_lw.F:3270-3340`).
6. **Eddington-vs-PIFM oracle resolution**: either (a) patch the local WRF build to set `kmodts=1` and rebuild the harness, OR (b) change the M5-S3.y contract to target PIFM in JAX and amend ADR-009 accordingly. Manager decision required.
7. **Per-band fixture/harness extension**: WRF harness must emit per-band TOA + surface fluxes; Python validation must compare per-band. Currently only broadband summed fields are compared (`worker-report.md:128`).
8. **Launch fusion**: 40 → ≤10 per call. Likely requires fusing the SW band-loop into a single XLA scan and similarly for LW. HLO size budget will tighten; may need to drop debug variant or compress branches.

**Acceptance for M5-S3.y**: strict Tier-1 pass at the contract-AC tolerances (`abs ≤1 W/m² + rel ≤0.05` flux, `abs ≤1e-4 K/s` heating), launches ≤10 per call, HLO ≤500 KB per kernel, ADR-009 finalized to "PARITY" status, per-band residuals available.

**Estimated wall-time for M5-S3.y**: 16–32 hours (larger than M5-S3.x because of the per-band code generation; comparable to M5-S1 Thompson). Should run alongside M5-S1.x + M5-S2.x in M6 prologue, file-disjoint.

---

## 6. M6 dispatch impact

- **M6 coupled forecast**: **BLOCKED on M5-S3.y close**.
- **M6 prologue parallel sprints** (M5-S1.x Thompson HLO-table-gather + process residual, M5-S2.x MYNN follow-ups): can proceed in parallel with M5-S3.y, file-disjoint per `5c5782c [M6-prologue] sprint contracts`.
- **Operational T2 gate**: cannot use carry-forward RRTMG either way until M5-S3.y closes. Manager should record this in `MILESTONE-M5-CLOSEOUT.md` and the M6 dispatch sprint contract.

---

## 7. Summary judgment

The M5-S3.x worker did exactly what an honest partial deserves: shipped real progress (sound Eddington + δ-scaling + vrtqdr port; faithful LW diffusivity; fabricated `log1p` curve removed; no clip-floor recurrence; no vacuous-tolerance recurrence; honest launch count; ADR-009 amended with verified citations and disclosed gaps), refused to dress it up as parity, and explicitly self-flagged "do not close as accepted." This is the model behavior for partial-success closeouts in this project.

The cycle moves from **A3's hand-rolled Beer-Lambert groundwork** to **A3.x's structurally-real-but-Planck-source-incomplete transfer solver**. The path to full parity is now bounded, scoped, and named (§5). REJECT would discard real progress; ACCEPT-as-parity would be dishonest; ACCEPT-AS-GROUNDWORK-PHASE-2 with a binding M5-S3.y stub is the correct disposition.

**Final decision: ACCEPT-AS-GROUNDWORK-PHASE-2.** Manager must (i) close M5-S3.x with explicit Phase-2-groundwork label, (ii) create `.agent/sprints/2026-05-21-m5-s3y-rrtmg-setcoef-taumol-planck/` stub with §5 scope, (iii) amend `MILESTONE-M5-CLOSEOUT.md` to record M5-S3.x closed + M5-S3.y as M6-prologue debt, (iv) keep M6 coupled-forecast dispatch blocked until M5-S3.y closes, (v) decide the Eddington-vs-PIFM contract question before M5-S3.y dispatch.
