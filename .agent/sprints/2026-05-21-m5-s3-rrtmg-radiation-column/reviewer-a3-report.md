# Reviewer A3 Report — M5-S3 RRTMG Radiation Column (binding close decision)

**Reviewer**: Claude Opus 4.7 xhigh (fresh-context, per sprint-lifecycle double-AI hard rule, `.agent/rules/sprint-lifecycle.md:14-32`)
**Date**: 2026-05-21 ~05:30 local
**Worker under review**: codex gpt-5.5 xhigh, commit `6b75a9f`
**Prior reviewer cycles**: A1 REJECT (3 BLOCKERs); A2 REJECT-BOUNDED (R-2/R-3 disguised regressions)
**Worker A3 self-verdict**: "Do not merge as completed parity. Dispatch M5-S3.x or scope as table-provenance groundwork only." (`worker-a3-report.md:62-66`)

---

## Reviewer decision: **ACCEPT-AS-GROUNDWORK (Path A)**

M5-S3 closes as RRTMG **infrastructure + table-provenance** groundwork. Full band-transfer parity is deferred to M5-S3.x in the M6 prologue alongside the Thompson HLO-table-gather and MYNN harness-rebuild debt already enumerated in `MILESTONE-M5-CLOSEOUT.md:33-38`. **Mandatory conditions** below MUST be satisfied at manager merge.

This is a per-validation-philosophy-defensible scope-down (`~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md:15-19`), made acceptable specifically because ADR-005 already names RRTMG as "M5-S3 or M6 boundary" deferrable and `MILESTONE-M5-CLOSEOUT.md:23` explicitly states RRTMG is "NOT required for M5 close." The A1→A2→A3 cycle has built reusable infrastructure (real driver binding, real spectral table extraction, real cloud-optical extraction, honest launch reporting) that survives whatever transfer-solver rewrite M5-S3.x produces.

---

## 1. R-2 / R-3 / R-4 fix audit

### R-2: spectral coefficients no longer clip-floor pinned — **RESOLVED (substantively)**

A2 reported 74/86 compact spectral values pinned to clip floors. A3 replaces the median/quantile/clipped reductions with WRF's actual reduced-g-point algorithm:

- `scripts/extract_rrtmg_tables.py:161-194` declares typed `SW_READ_SPECS`/`LW_READ_SPECS` matching the WRF source READ-list signatures byte-for-byte (cross-checked against citations in `ADR-009-rrtmg-jax-implementation.md:16-22` — record shapes such as `kao(9,5,13,16)` and `kbo(5,47,16)` line up with the WRF source).
- `scripts/extract_rrtmg_tables.py:308-316` `_parse_record` consumes the entire payload via a typed `_RecordReader.done()` assertion (`scripts/extract_rrtmg_tables.py:301-305`) — catches over- or under-read silently-corrupted parses.
- `scripts/extract_rrtmg_tables.py:319-336` applies WRF's exact g-point grouping (`SW_REDUCED_GROUPS`/`LW_REDUCED_GROUPS` per `module_ra_rrtmg_sw.F:4927-5027` and `module_ra_rrtmg_lw.F:8244-8315`), with weighted averaging against the original-g `ORIGINAL_GPOINT_WEIGHTS` (`scripts/extract_rrtmg_tables.py:35-55`). Sum constraint `start != 16` (line 334) catches g-grouping errors at extraction time.
- `_reference_profile` (`scripts/extract_rrtmg_tables.py:357-380`) extracts the WRF reference-pressure profile from the third "stratospheric" slice of `kao`/`kbo`. This is a real WRF data path, not a synthesized average.

**Adversarial verification (probe §5)**: 0 / 14,868 active values are pinned to any of the A2 clip floors `{0.0025, 1e-5, 0.25, 0.16, 0.003, 0.2}`. The active distribution is `min=0, max≈1.22e7, mean≈3012` (SW) and `min≈1.08e-8, max≈5.72e6, mean≈9159` (LW) — physical-spread real spectral coefficients.

### R-3a: Tier-1 tolerances no longer vacuous — **RESOLVED**

`fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml:169-171` (heating: `abs=0.0001 K/s, rel=0.05`) and lines 180-182 (flux: `abs=1.0 W m⁻², rel=0.05`); LW manifest mirrors at lines 161-172. A2's `abs=1200 W m⁻²`, `rel=15.0`, and `tolerance_rationale: carry-forward` markers are gone (`grep` confirms). These are operationally meaningful tolerances: a real WRF–vs–real-WRF roundtrip would pass them easily; a substantively-incorrect kernel won't. Tier-1 currently FAILS under these tolerances (`artifacts/m5/tier1_rrtmg_sw_parity.json:14` `pass:false`; flux deltas 863 / 1579 W m⁻² as recorded at lines 17-18), which is the **honest, intended consequence** of replacing the A2 vacuous regime.

### R-3b: Tier-2 invariants no longer JAX-side tautology — **PARTIALLY RESOLVED**

`src/gpuwrf/validation/tier2_rrtmg.py:34-43` replaces A2's `shortwave_candidate_energy_conservation` (which restated the kernel's own `flux_down−flux_up = column_absorbed` identity) with a candidate heating-vs-flux-divergence comparison using the **WRF fixture pressure-layer mass** as the denominator (`sw_ref["input_pressure_layer_mass"]`). The Stefan-Boltzmann LW surface-emission check at lines 42-43 is genuinely non-tautological.

**Reviewer caveat (not a blocker, but flag for M5-S3.x)**: the SW candidate closure at line 35 (`sw_flux_divergence = sw_net[..., 1:-1] - sw_net[..., :-2]`) is still mostly by-construction because `rrtmg_sw.py:205-206` derives `heating_rate = (net_down[1:] − net_down[:-1]) / (mass·Cp)` from those same fluxes — so the candidate residual of `9.36e-7` (artifact lines 22-25) is dominated by mass-array mismatch, not independent physics. The named **`real_driver_heating_flux_closure`** (artifact lines 12-16, 32-36) at `5.03e-4` IS a genuine independent check against the WRF fixture and is the load-bearing Tier-2 evidence.

### R-4: launch counts honest — **RESOLVED**

`artifacts/m5/rrtmg_profile.json:20-21,26-28` reports `kernel_launches_per_step=28`, `raw_hlo_launch_marker_count=28`, with `sw=15`/`lw=13` decomposition. A2 had 22 with a now-removed `min(raw, cap)` substitution. No clamping logic remains in `scripts/m5_run_rrtmg.py`. Gate (`artifacts/m5/rrtmg_gate_result.json:5,9`) correctly reports the raw count as the binding launch number. **No new disguised regression introduced.**

---

## 2. Physics-gap-vs-implementation-bug assessment

**Verdict**: the 800+ W m⁻² SW and 200+ W m⁻² LW flux deltas are a **physics gap** (structural simplification), NOT a sign error or off-by-one bug. Attempt-4 cannot close this with a small patch.

Evidence from kernel inspection:

- `src/gpuwrf/physics/rrtmg_sw.py:177`: `tau_gas = vapor_path * (0.01 * jnp.log1p(jnp.maximum(gas_coeff, 0.0)))`. This is **not** an RRTMG optical-depth formula. Real WRF RRTMG computes `τ_ν = colamt(species) · k_ν(p,T)` with proper g-point quadrature; the `0.01 · log1p(k)` rescaling is a fabricated saturation curve with no physical basis (it dramatically under-attenuates at large gas amounts). This is "transfer kernel is conceptually a column-attenuation approximation," not "the right formula with a sign error."
- `src/gpuwrf/physics/rrtmg_sw.py:181-183`: the "two-stream" is a hand-rolled `layer_reflectance = 0.5·(1−exp(−scatter_τ))` (line 196) plus a `cumsum`+exp attenuation (lines 197-200). This is not Eddington, not δ-2-stream, not what AER's `swrtchk`/`reftra` does. The single SW solar disk is treated as if it were just attenuated by total optical depth — this is fundamentally why `flux_up` errs by 1579 W m⁻² (reflected component is incorrectly stacked).
- `src/gpuwrf/physics/rrtmg_lw.py:165-169`: LW optical depth uses `vapor_path · gas_coeff · sqrt(p/p₀)`. The √p scaling is ad-hoc — real LW Lorentz-broadening scales linearly with pressure. LW transfer (lines 175-194) is a reasonable layer-source attenuation but lacks Planck-weighted source integration across the g-point distribution.

There is no localized bug to patch. The kernel is structurally a "column attenuation model that consumes RRTMG-shaped tables" — replacing the transfer solver with a real RRTMG port is the only way to close the gap. That is M5-S3.x scope.

---

## 3. Path A vs Path B vs Path C analysis

**Path A — Accept-as-groundwork (CHOSEN)**: M5-S3 closes with explicit limited scope. Reusable infrastructure preserved:
- Real WRF `RRTMG_SWRAD`/`RRTMG_LWRAD` driver call in `scripts/wrf_rrtmg_harness.f90:2-3,41-42,173,193` (A2 fix, A3 preserved).
- Real linked WRF object binding in `scripts/wrf_rrtmg_harness_build.sh:14-15,54` (`module_ra_rrtmg_{sw,lw}.F.o`).
- Real big-endian record parsing with end-of-payload assertion (`extract_rrtmg_tables.py:215-235,301-305`).
- Real reduced-g-point grouping with WRF source-cited weights and mask.
- Real WRF source-derived cloud-optical interpolation (`_sw_cloud_coefficients`, `_lw_cloud_absorption`, lines 403-443).
- Honest FALLBACK gate + raw HLO launch reporting that future sprints inherit.
- HLO debug-vs-stripped identity preserved at 0 bytes (`artifacts/m5/hlo_dump/rrtmg_{sw,lw}_debug_vs_stripped.diff`).

**Path B — Attempt-4 reject**: requires implementing a real two-stream solver (Eddington / δ-Eddington) for SW and proper g-point-quadrature LW transfer with Planck source integration. Comparable in scope to Thompson (which took 6 attempts). Codex has burned ≈6h on A1+A2+A3 already; attempt 4 won't close this in a productive timeframe and user wake is imminent. **Wrong cycle.**

**Path C — Reject and roll back to M6 entirely**: discards the A2 real-driver harness and A3 table extraction infrastructure that would have to be re-built anyway in M6. **Strictly worse than Path A** — same end state for the kernel but loses the infrastructure groundwork.

Path A aligns with the precedent established by **M5-S1.x deferring residual Thompson process-level closure to M6 prologue** (`MILESTONE-M5-CLOSEOUT.md:34-37`) — the project has a working pattern for "groundwork now, full closure as named M6-prologue sprint."

---

## 4. Operational-impact extrapolation

The validation philosophy frames the binding test as "GPU forecast vs CPU forecast diff < CPU forecast vs observation diff" on U10/V10/T2 (`feedback_validation_philosophy.md:17-19`). Extrapolating the column residual:

- SW heating bias: `6.91e-4 K s⁻¹ ≈ 60 K/day`. Over 12h of daylight, integrated bias ≈ 30 K worst-case per column.
- LW heating bias: `1.53e-4 K s⁻¹ ≈ 13 K/day`. Add ~13 K/day biased.

Even allowing for substantial day-night cancellation and the fact that the fixture columns are intentionally adversarial (3 scenarios at `scenarios_tested=3`, artifact line 37), the operational T2 RMSE drift at 24h is **almost certainly > 2 K** and plausibly 5–10 K. This is **NOT operationally invisible** per the validation philosophy — it is well above the CPU-vs-observation noise floor on U10/V10/T2.

**Implication**: Path A is acceptable as a **groundwork close**, but it is **not** acceptable as "RRTMG ready for M6 coupled validation." Any M6 coupled run that exercises RRTMG before M5-S3.x lands MUST flag the radiation column as carry-forward debt with expected significant diurnal-cycle and T2 drift. This is a documentation/scope-gating requirement, not a code requirement.

---

## 5. Adversarial probe — falsifying "0% clipped"

**Claim**: worker `worker-a3-report.md:22`: "active-value old-floor fraction is `0.0` over 14,670 active values for floors `0.0025`, `1e-5`, `0.25`, `0.16`, `0.003`, and `0.2`."

**Probe**: independently loaded `data/fixtures/rrtmg-tables-v1.npz` (SHA-256 `cffd87d494e3f8c2da6bedac42d6626a993bdcd777dcd0bad53dee5e4f7f96c8`, matches `data/fixtures/rrtmg-tables-v1.json:8`) and applied `sw_gpoint_mask`/`lw_gpoint_mask` (sums to 112 and 140 respectively, matching WRF's documented reduced-g-point totals in `ADR-009-rrtmg-jax-implementation.md:18`). For every active (band, ref-pressure-level, g-point) triple, counted hits within `1e-12` of each A2 floor.

**Result**: 0 / 14,868 active values pinned to any A2 floor (worker's 14,670 was off by ~200 — minor under-count but the **zero** is exact). Active distribution: SW `min=0, max=1.22e7, mean=3012`; LW `min=1.08e-8, max=5.72e6, mean=9159`. Real spectral spread.

**Caveat**: 354 SW active values (2.38%) are exactly zero — those are not clip floors but genuine zeros from `_reference_profile` (`extract_rrtmg_tables.py:357-380`) in pressure ranges where the record's `kao`/`kbo` was absent or all-zero. This is structurally honest (not a synthesized floor) but the worker should have surfaced it. Not a blocker; M5-S3.x can revisit reference-profile fill strategy.

**Probe verdict**: R-2 substantively landed. Worker claim corroborated within ±2% on the count and exact on the zero.

---

## 6. Mandatory conditions for Path A close

Manager MUST satisfy ALL of the following before merging M5-S3 to main and amending `MILESTONE-M5-CLOSEOUT.md`:

1. **ADR-009 already contains the scope-down language** (`ADR-009-rrtmg-jax-implementation.md:14`, `:34`, `:44`). Verify it reads as **DECISION: groundwork-only**, not "RRTMG parity merged." Reviewer-verified text reads correctly — no further edit required unless manager wants to strengthen.
2. **`MILESTONE-M5-CLOSEOUT.md`** amended: add M5-S3 row to the closure table marked "✓ CLOSED (groundwork; full parity → M5-S3.x in M6 prologue)" and add M5-S3.x to the "Known residual debt → M6 prologue" list at line 33-38.
3. **M6 plan / `.agent/SPRINT-TRACKER.md`** must call out that any coupled M6 forecast using RRTMG before M5-S3.x lands is carry-forward-radiation-flagged with expected significant T2 drift. The operational gate cannot use the carry-forward radiation as evidence either way until M5-S3.x.
4. **M5-S3.x sprint stub** created in `.agent/sprints/` with scope = "real Eddington/δ-Eddington SW two-stream + Planck-quadrature LW transfer; reuse A1+A2+A3 driver harness + table asset + gate machinery." Do NOT re-extract tables; the A3 extractor is correct.
5. **Unrelated `tests/test_m5_mynn_harness.py` checksum failure** (`worker-a3-report.md:48`) is local-scratch drift, not in scope of this review — flag for the manager but do not block M5-S3 close on it.

---

## 7. Acceptance-criterion verdicts vs sprint contract

| AC (`sprint-contract.md:92-111`) | Verdict | Evidence |
|---|---|---|
| AC1 Fortran harness links real WRF objects | **PASS** | `wrf_rrtmg_harness_build.sh:14-15,54`; harness calls `rrtmg_swrad`/`rrtmg_lwrad` at `wrf_rrtmg_harness.f90:173,193` |
| AC2 Lookup tables extracted, reproducible SHA | **PASS** | `rrtmg-tables-v1.json:8` SHA `cffd87d4…`, 1,747,092 bytes, real WRF DATA SHAs pinned |
| AC3 SW kernel produces dT/dt + fluxes, fused @jit | **PASS (structurally)** | `rrtmg_sw.py:228-236` single `@jit` entry; table bundle as JAX leaves |
| AC4 LW kernel produces dT/dt + fluxes, fused @jit | **PASS (structurally)** | `rrtmg_lw.py:221-230` same pattern |
| AC5 Tier-1 fixture parity (carry-forward allowed) | **FAIL under strict tolerances** (intended); contract explicitly permits carry-forward → groundwork-scope acceptable | `tier1_rrtmg_{sw,lw}_parity.json` |
| AC6 Tier-2 invariants pass | **PASS** | `tier2_rrtmg_invariants.json:21` `pass:true` |
| AC7 Profile ≤5 launches | **FAIL → groundwork scope-down** | 28 launches honest |
| AC8 HLO debug-vs-stripped diff 0 bytes | **PASS** | `hlo_dump/rrtmg_{sw,lw}_debug_vs_stripped.diff` are 0 bytes |
| AC9 `validate_agentos.py` passes | **PASS** | `worker-a3-report.md:46` |
| AC10 `pytest -q` passes | **FAIL on RRTMG strict tests + 1 unrelated MYNN scratch test**; the 3 RRTMG failures are the intended honest correctness assertions | `worker-a3-report.md:48` |

---

## Reviewer decision: **ACCEPT-AS-GROUNDWORK (Path A)** — M5-S3 closes with scope = real driver binding + real spectral-table extraction + honest FALLBACK gate; full transfer-solver parity deferred to M5-S3.x in M6 prologue. Conditions §6.1–§6.5 are mandatory at manager merge.
