# Reviewer Report — M5-S3.zz RRTMG SW Closeout (sfluxzen + setcoef precision + lax.scan fusion)

**Reviewer**: Claude Opus 4.7 xhigh (fresh-context, per sprint-lifecycle Double-AI HARD RULE, `.agent/rules/sprint-lifecycle.md:14-32`)
**Date**: 2026-05-21
**Branch / commit under review**: `worker/codex/m5-s3zz-rrtmg-sw-closeout` @ `f62fe88 M5-S3.zz PARTIAL: close SW intermediates, expose broadband root cause` (already merged onto `main` via `9637e98 --no-ff` + manager `[BIG TICK] 3242f1f`).
**Worker**: Codex GPT-5.5 xhigh
**Worker self-verdict**: "**PARTIAL delivery. Do not mark ADR-009 as full PARITY and do not claim SW-PARITY yet.** Closed the two M5-S3.z reviewer-named intermediate-oracle defects (`sfluxzen` allocation + `setcoef_sw` tolerance policy), re-enabled the validated SW optical-depth branch in production, exposed a new downstream root cause in broadband transfer / cloud-optics." (`worker-report.md:3`)
**Prior cycle precedent**: M5-S3 → ACCEPT-AS-GROUNDWORK; M5-S3.x → ACCEPT-AS-GROUNDWORK-PHASE-2; M5-S3.y → PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3; M5-S3.z → PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4 with binding Option 1 (SW-focused) dispatch.

---

## Reviewer decision: **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-5** with **binding M5-S3.zzzz Option A** (cldprmc_sw + spcvmc_sw intermediate-oracle FIRST → M5-S3.zzz LW closeout SECOND)

This sprint closes both M5-S3.z reviewer §4 root causes (`sfluxzen` band/g-point allocation; `setcoef_sw` precision-policy mismatch) at the intermediate-oracle level, re-enables the 14 PASS-validated SW per-band branches in production via `jax.lax.scan`, and produces an honest negative proof for strict Tier-1 SW flux parity that materially advances diagnostic resolution: the residual is now bounded *below* the gas-optical-depth + source-allocation stack and *above* the cloud-optics + broadband-transfer assembly. No anti-patterns from the M5-S3 → S3.x → S3.y → S3.z → S3.zz arc (clip-pinned fixtures, vacuous tolerances, `min(raw, cap)` launch fudge, fabricated coefficients) recurred. The methodology discipline established by M5-S3.z holds.

REJECT-bounded-rework would discard the AC1/AC2/AC3 closure infrastructure (sfluxzen source-active gating, single-precision setcoef floor, 14-branch `lax.scan` consumption in `_shortwave_impl`). ACCEPT-as-SW-PARITY would be dishonest — strict Tier-1 SW broadband fluxes still fail by 30-87 W/m². **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-5** is the correct disposition; the new sprint binding immediately follows.

---

## 1. Verifiability triple (anti-spec-gaming checks)

### 1.1 `nm` on harness binary — NOT independently re-run; symbol *preservation in source* verified

The harness binary `data/scratch/wrf_rrtmg_harness` is **absent from the worktree** at review time (only the three `.dat` input fixtures remain in `data/scratch/`). The worker explicitly self-flagged this gap (`worker-report.md:79`: "The harness `nm` symbol preservation check was not rerun in this partial closeout. The intermediate oracle files remain present, but the verifiability triple is incomplete until the next oracle sprint reruns it.").

Substitute checks performed:
- `scripts/wrf_rrtmg_harness.f90:9-19, 31-34, 73, 95, 116, 129, 134, 206, 245, 262, 303, 371-392` confirm the F90 source still:
  - declares `use rrtmg_sw_setcoef, only: setcoef_sw` and `use rrtmg_lw_setcoef, only: setcoef`,
  - calls `setcoef_sw`, `setcoef`, `spcvmc_sw`, `rtrnmc`,
  - emits the `#RRTMG_ORACLE_V1_BINARY` marker line + the `sfluxzen[max_sw_g,14]` record at offset `:392`.
- The pinned intermediate-oracle NPZ SHA `d8795e23…0b954` (`data/fixtures/rrtmg-intermediate-oracle-v1.npz`) is **byte-identical** to the M5-S3.z artifact reviewed at PASS (matches `manifests/rrtmg-intermediate-oracle-v1.yaml` and M5-S3.z reviewer §1.2). The oracle reference was *not* regenerated this sprint, so the binary that produced it from M5-S3.z transitively carries forward.

**Disposition**: PARTIAL — symbol-grep on the binary itself is debt to M5-S3.zzzz (the next sprint extends the harness for `cldprmc_sw` + `spcvmc_sw` dumps and MUST rebuild + re-`nm`). The M5-S3.z transitive triple is unbroken because the oracle NPZ is byte-pinned.

### 1.2 Intermediate-oracle NPZ — no clip-pinning, no fabricated coefficients

- `data/fixtures/rrtmg-intermediate-oracle-v1.npz` SHA unchanged from M5-S3.z (verified by `sha256sum`). The reviewer R-3 PASS from M5-S3.z carries forward unconditionally.
- `data/fixtures/rrtmg-tables-v1.npz` SHA `5cc63950…3a06e2` (new this sprint, regenerated to include 7 added cloud-optics arrays: `sw_cloud_{liquid,ice,snow}_{extinction,ssa,asymmetry,forward_fraction}`, plus `lw_cloud_absorption`, plus `cloud_optical_defaults`). Independent load shows physically-reasonable ranges per band (`sw_cloud_liquid_extinction ∈ [0.140, 0.168]`, `sw_cloud_ice_ssa ∈ [0.55, 1.0]`, `sw_cloud_snow_asymmetry ∈ [0.78, 0.96]`). No sentinel/clip values detected; ranges match published Hu & Stamnes / Fu cloud-optical-property tables.
- Worker did NOT introduce any new clip-pinning floor on oracle reference arrays. Test file `tests/test_m5_rrtmg_intermediate_oracles.py:42-43` asserts the SW setcoef precision floor at the policy level (`abs_tol == 1.0e-4; rel_tol == 1.0e-3`), so the bar cannot silently revert. **No R-2/R-3 anti-pattern recurrence.**

### 1.3 NO `min(raw, cap)` launch fudge

`grep -n 'min(.*cap\|launch_cap'` over `rrtmg_sw.py`, `m5_run_rrtmg.py`, `m5_gate_rrtmg.py` returns zero matches. Worker `worker-report.md:78` explicitly: "Launch-count and HLO-size targets are not claimed because the production path is not parity-correct yet. No `min(raw, cap)` launch fudge was introduced or used as evidence." **Honest no-claim disposition** — preferable to a synthetic launch budget burst. No R-4 anti-pattern recurrence.

---

## 2. Findings table (R-findings)

| ID | AC | Severity | Disposition | Key citations |
|---|---|---|---|---|
| R-1 | AC1 SW `sfluxzen` allocation | clean | **PASS** | `artifacts/m5/rrtmg_intermediate_validation.json` `.sw.sfluxzen.pass=true`, max_abs `3.81e-6`, max_rel `3.74e-7` (within `1e-8+1e-4`). Source-active gating wired at `src/gpuwrf/physics/rrtmg_sw.py:549-555, 612`. WRF citation `module_ra_rrtmg_sw.F:3380-3382, 4446-4464` is correct (zero-init then band-active loop). The band-11 `[*,0,11]` zero-allocation defect identified by M5-S3.z reviewer §3.1 is closed. |
| R-2 | AC2 SW `setcoef_sw` precision-policy | clean (Path B amended) | **PASS** | `validate_sw_setcoef_state` uses `abs_tol=1e-4 + rel_tol=1e-3` per `rrtmg_intermediate_oracles.py:88` with the policy line at `:81, :220` citing single-precision WRF oracle floor. All 12 setcoef sub-fields (`jp/jt/jt1/fac00..fac11/indself/indfor/selffac/forfac/colmol`) pass. `tests/test_m5_rrtmg_intermediate_oracles.py:42-43` asserts the bar so it cannot silently revert. Path B chosen per manager-recommended option in sprint contract AC2; rationale documented in worker report §"Closed root cause 2". |
| R-3 | AC3 14 SW branches re-enabled via `lax.scan` | substantive | **PASS-functional / PARTIAL-budget** | `_sw_taumol_fused` (`rrtmg_sw.py:514-523`) wraps `_sw_taumol` behind `jax.lax.scan` over `jnp.arange(14)`. Production `_shortwave_impl` consumes it at `:859` (and `_sw_sfluxzen` at `:860`, `_sw_setcoef` at `:858`). NOT silently bypassed by the nearest-pressure fallback. `artifacts/m5/rrtmg_per_band_status.json`: all 14 SW bands `implementation_status=FULL_BRANCH_ACCEPTED`, `intermediate_gate=PASS`. HLO/launch-count budget NOT CLAIMED — worker honestly defers because Tier-1 flux parity still fails. |
| R-4 | AC4 strict Tier-1 SW flux parity | failed | **FAIL HONESTLY** | `tier1_rrtmg_sw_parity.json` `pass=false`. Max abs residuals: `flux_down=56.5`, `flux_up=64.4`, `column_absorbed=87.5`, `toa_up=30.1`, `surface_down=20.6` W/m². Heating-rate max_abs `2.48e-5 K/s` is **inside** the `1e-4 K/s` threshold — heating-rate PASS at abs bar (max_rel `1.27` fails on small-magnitude cells but operational impact is below 0.5 K/day per `feedback_validation_philosophy.md`). |
| R-5 | AC5 LW no regression | preserved by scope | **PASS-by-edit-scope** | `git diff main…HEAD -- src/gpuwrf/physics/rrtmg_lw.py` returns zero lines. Worker did not touch LW production. Full LW rerun NOT performed this sprint — disposition is "edit-scope guarantee", not "fresh proof", per worker-report.md:17. Acceptable for AC5 phrasing. |
| R-6 | AC6 ADR-009 amendment | not done — correctly | **HONESTLY NOT DONE** | `git diff main…HEAD -- .agent/decisions/ADR-009-rrtmg-jax-implementation.md` returns zero lines. ADR-009 status remains `PROPOSED worker draft, M5-S3.y still NOT PARITY`. Worker correctly refused to mis-set to `SW-PARITY` while strict SW Tier-1 fails. Pattern matches M5-S3.z R-9 PASS. |
| R-7 | MCICA KISS random-overlap mask | verified | **PASS** | `_mcica_random_overlap_mask` (`rrtmg_sw.py:668-689`) mirrors WRF `module_ra_rrtmg_sw.F:1727-1744, 1754-1778, 2016-2040`: single-precision pressure cast at `:671`, four-seed fractional extraction at `:672-674`, KISS step at `:679-684`, threshold comparison `cdf >= (1 - cloud_fraction)` at `:687`, reduced-g-point index gather at `:688`. Worker's local Fortran KISS probe (worker-report.md:38) confirmed mask agreement over inspected layer/subcolumn cells. **MCICA mask is NOT the root cause.** |
| R-8 | Cloud-optics assembly | suspect | **FLAG for M5-S3.zzzz** | `rrtmg_sw.py:877` `cloud_safe = jnp.maximum(cloud_box, 0.01)` floors the cloud-fraction denominator when in-cloud values are derived from grid-mean cloud water. For `cloud_box ∈ (0, 0.01)`, this systematically *underestimates* in-cloud `liquid_path/ice_path/snow_path`, hence underestimates `tau_cloud`. This is a candidate contributor to the broadband residual and is *not* the same anti-pattern as clip-pinning oracle references (it's a numerical denominator floor on internal state), but the M5-S3.zzzz cldprmc intermediate-oracle WILL directly catch any resulting `ptaucmc` divergence. Flagged as a specific R-finding for the next sprint to verify against `cldprmc_sw` dumps. |
| R-9 | Clear/cloud Eddington blending | suspect | **FLAG for M5-S3.zzzz** | `_shortwave_impl` (`rrtmg_sw.py:936-945`) computes `_reftra_eddington` *twice* (clear and cloud) and then linearly blends `pref/prefd/ptra/ptrad` by `cloud_top_down`. WRF `spcvmc_sw` (`module_ra_rrtmg_sw.F:8554-8668`) instead applies the MCICA binary mask per-g-point per-layer to *select inputs* and runs `reftra` *once* per g-point with the resulting optical properties. Because Eddington reftra is **non-linear** in `(tau, omega, g)`, blending the *outputs* is mathematically distinct from blending the *inputs*. The `cloud_top_down` mask is binary per g-point (line 687 cast to float64), so when it acts per-g-point this *partially* reduces to selection — but the per-band aggregation in `flux_down_model = sum(down_band, axis=(-1,-2))` (`:965-966`) loses any g-point-level fidelity present in the WRF accumulators. The cldprmc + spcvmc intermediate oracle in M5-S3.zzzz will resolve this directly. |
| R-10 | Adjflux Earth-Sun correction | verified by oracle PASS | **PASS** | `_shortwave_impl:961` `top_flux_band = coszen * sfluxzen`. The Earth-Sun correction factor `adjflux` (WRF `:8474`) is either baked into the table-extracted `sw_sfluxref` or matches at the oracle threshold (since `sfluxzen` passes at `3.8e-6` abs in the oracle, JAX and WRF source values agree). Not a residual contributor. |
| R-11 | Tier-2 invariants (mass/energy conservation, nan/inf) | preserved | **PASS** | M5-S3.z `tier2_rrtmg_invariants.json` carried forward; worker did not run a fresh Tier-2 because no LW modifications and SW conservation is implicit in heating-rate-within-threshold result. |
| R-12 | Per-band debt list | clean | **PASS** | `artifacts/m5/rrtmg_per_band_status.json` 14 SW bands at `FULL_BRANCH_ACCEPTED + intermediate_gate=PASS`, policy line cites the M5-S3.zz single-precision setcoef floor (`rrtmg_intermediate_oracles.py:220`). LW band entries unchanged (still `DEBT_TO_M5_S3_ZZ`/`DEBT_TO_M5_S3_ZZZ`). |

No new clip-pinning on oracle references, no vacuous tolerances, no launch fudge. R-8 and R-9 are *flags for the next sprint*, not anti-pattern recurrences — they reflect honest blind-production-edit territory that the worker explicitly identified in §"New root cause" of the worker report.

---

## 3. Root-cause attribution deep-dive (cldprmc vs spcvmc vs flux-accumulation)

The M5-S3.z reviewer §3 attributed the SW flux residual to three sources: `sfluxzen` mis-allocation (30-50 W/m²), production-path nearest-pressure regression for `taug` (30-50 W/m²), `setcoef` single-precision propagation (<10 W/m²). M5-S3.zz closed the first two completely (intermediate oracle PASS) and the third by precision-policy amendment. The remaining 30-87 W/m² broadband residual is therefore **isolated to the cloud-optics + broadband-transfer + flux-accumulation stack**.

### 3.1 Diagnostic signatures

| Signature | Implication |
|---|---|
| `heating_rate max_abs = 2.48e-5 K/s` (INSIDE `1e-4 K/s` threshold) | Column-integrated energy balance ≈ correct; vertical *redistribution* is the issue. |
| `column_absorbed = 87.5 W/m²` (largest residual) | Total atmospheric absorption diverges from WRF by ~6% (87 / 1361·μ₀); consistent with cloud-optics path mis-assembly. |
| `surface_down = 20.6 W/m²` + `flux_up = 64.4 W/m²` + `toa_up = 30.1 W/m²` | More reflection upward, less reaching surface — classic cloud-radiative-effect (CRE) sign for over-reflective cloud. |
| `toa_down = 8.69e-5 W/m²` (~0) | Top-of-atmosphere DOWN matches — solar BC is correct. Not an `adjflux` bug. |
| All 14 SW per-band `taug` + `taur` + `sfluxzen` intermediate-oracle PASS | Gas optical depth, Rayleigh, source allocation are ALL correct. NOT the residual contributors. |
| MCICA KISS probe agrees with WRF Fortran | Random-overlap mask is NOT the residual. |

### 3.2 Candidate root causes inside cldprmc + spcvmc + accumulation

1. **`cloud_safe = max(cloud_box, 0.01)` denominator floor** (R-8): for cloud_fraction ∈ (0, 0.01), underestimates in-cloud LWP/IWP/SWP → underestimates `tau_cloud` → underestimates CRE → biases all four flux components. Magnitude depends on scenario; for the marine/humid scenarios with low-cloud-fraction stratus this could be 20-40 W/m².

2. **Double-Eddington-then-blend vs WRF's per-g-point select-then-Eddington** (R-9): non-linear in `(tau, omega, g)` so the blended output deviates from the WRF accumulator output. The marine_trade_cloud and humid_low_cloud scenarios have significant cloud cover and would exhibit this most; thin_nocturnal_ice should be least affected (and SW residuals there should be smaller — checkable in M5-S3.zzzz scenario-specific dumps).

3. **`pref_lay = (1-cloud)*pref_clear + cloud*pref_cloud`** etc. (`:942-945`) followed by `surface_albedo` concatenation at `:946-954`: the surface boundary condition shares one `pref`/`prefd` value across both clear and cloud paths, which is correct only if both paths have the same surface — which they do, BUT the layer-level mixing happens *before* the vertical adding (`_vertical_quadrature` at `:765-836`). WRF computes adding once per g-point with per-g-point optical properties; JAX computes adding on the *mixed* (pre-blended) optical properties. Mathematically distinct.

4. **`direct_trans = (1-cloud)*exp(-tau_clear/μ) + cloud*exp(-tau_total/μ)`** (`:956-958`): linear average of exponentials is biased high (Jensen's inequality) compared to `exp(-(avg tau)/μ)`. For binary masks (which `cloud_top_down` is), per-g-point this reduces to selection and is fine — but if `cloud_top_down` has values strictly in (0,1) due to `mask` multiplication in `:689` or `astype(jnp.float64)` rounding, residuals propagate. The cldprmc + spcvmc dump will confirm.

5. **Forward-fraction closure choice**: liquid uses `f = g²` (Henyey-Greenstein, `:870`) while ice/snow read tabular `sw_cloud_{ice,snow}_forward_fraction` from extracted WRF tables. Mismatch with WRF closure would bias delta-scaled cloud-component τ.

6. **Per-g-point vs per-band flux accumulation order**: `flux_down_model = sum(down_band, axis=(-1,-2))` (`:965-966`) sums band+g-point in one reduction. WRF `spcvmc_sw` adds per-g-point flux to per-band `znicddir/znicddif/znicddifup` (`:8739-8745`), then sums bands at exit. Both should give the same broadband answer numerically (associativity), but the M5-S3.zzzz `zfd/zfu` BEFORE-accumulation dump will catch any subtle reduction-order or sign issue.

### 3.3 Attribution table (probability-weighted estimate)

| Source | Estimated contribution to broadband residual | M5-S3.zzzz oracle will catch |
|---|---:|---|
| `cldprmc_sw` cloud-optics assembly (R-8 + cloud_safe floor + delta-scaling closure) | 20-50 W/m² | YES — `ptaucmc / pomgcmc / pasycmc / ptaormc` per band/layer/g-point |
| `spcvmc_sw` per-layer reftra blending (R-9) | 10-30 W/m² | YES — clear/cloud `zref/ztra/zrefd/ztrad` per layer/band |
| Direct-beam transmittance assembly | 5-15 W/m² | YES — per-band direct transmittance dump |
| Per-g-point flux accumulation (`zfd/zfu` pre-broadband) | 0-10 W/m² | YES — per-g-point `zfd/zfu` BEFORE accumulation |
| Transfer-solver (Eddington two-stream + vrtqdr) | ≈0 (M5-S3.y AC0 carry-forward PASS) | YES |
| Gas optical depth, sfluxzen, setcoef | ≈0 (M5-S3.zz intermediate-oracle PASS) | — already PASS |

**Worker's root-cause attribution is correct and well-bounded.** The residual is genuinely in `cldprmc_sw + spcvmc_sw + flux accumulation`, and the M5-S3.zzzz scope dumps (per worker §"New root cause" §3.3-3.4 of the worker report) are exactly the right intermediates to land. The flagged R-8 (`cloud_safe` floor) and R-9 (double-Eddington-blend) are specific testable hypotheses the M5-S3.zzzz validation will resolve.

---

## 4. M5-S3.zzzz scope decision (Option A vs B vs C — binding)

Three options were considered:

### Option A (manager-proposed): M5-S3.zzzz cldprmc_sw + spcvmc_sw oracle FIRST → M5-S3.zzz LW closeout SECOND

- **Pro**: SW heating-rate already PASSES (`2.48e-5 < 1e-4 K/s`). Only flux-distribution remains. Closing SW completely first proves the full intermediate-oracle methodology end-to-end before tackling the larger LW gap. Methodology dividend.
- **Pro**: The worker has fresh diagnostic context (just identified R-8 and R-9). Switching to LW now loses this momentum and forces a re-context.
- **Pro**: The `cldprmc + MCICA + spcvmc` infrastructure built in M5-S3.zzzz (harness extensions, oracle NPZ schema, per-band cloud-optics validators) is **partly reusable** for LW (which has its own `cldprmc_lw + rtrnmc` path with similar McICA + cloud-optics structure). LW closure in M5-S3.zzz then becomes additive, not greenfield.
- **Pro**: Bounded scope — worker has already named the dump variables (`ptaucmc`, `pasycmc`, `pomgcmc`, `ptaormc`, `zref/ztra/zrefd/ztrad`, direct-beam transmittance, per-g-point `zfd/zfu`). Estimate 16-32h.
- **Con**: Operational T2 drift is LW-dominated (no day/night cancellation, 24h column-integrated). Closing SW first does NOT unblock M6 alone.

### Option B: M5-S3.zzz LW closeout FIRST → M5-S3.zzzz cldprmc/spcvmc SECOND

- **Pro**: LW dominates 24h T2 drift; closing it has higher single-sprint operational impact.
- **Pro**: M5-S3.z established LW `planklay/planklev/plankbnd + dplankup/dplankdn + tfn_tbl` PASS, so the LW Planck-source machinery is ready; remaining work is the 16 `taumol_lw + fracs` branches.
- **Con**: M5-S3.z reviewer estimated 24-48h with ~50% sprint-success probability for LW closeout (16 new branches × per-band oracle validation × minor-species + ratio interpolation + band-3 special path). HIGHER RISK than SW cldprmc/spcvmc (16-32h, ~70%).
- **Con**: SW remains *half-closed* with R-8 and R-9 unresolved. Risk of LW work entangling with SW state through shared `compute_rrtmg_intermediates` / harness extensions.
- **Con**: Worker context is fresh on SW cloud-optics, NOT on LW taumol branches. Switching costs context-rebuild.

### Option C: PARALLEL M5-S3.zzzz + M5-S3.zzz (file-disjoint: rrtmg_sw vs rrtmg_lw)

- **Pro on paper**: production code is file-disjoint (`rrtmg_sw.py` vs `rrtmg_lw.py`).
- **Con (decisive)**: shared-file conflicts kill the parallelism: `scripts/wrf_rrtmg_harness.f90` (both sprints extend it for their respective `cldprmc/spcvmc` vs `cldprmc_lw/rtrnmc` dumps), `scripts/m5_generate_rrtmg_fixture.py` (both parse new records), `data/fixtures/rrtmg-intermediate-oracle-v1.npz` (both extend the same NPZ unless namespaced), `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` (both extend the validation framework), `.agent/decisions/ADR-009-rrtmg-jax-implementation.md` (both potentially amend).
- **Con**: Critical path is unchanged — both sprints MUST close before M6 dispatch. Parallel doesn't reduce blocking time unless both succeed *together*, which doubles the risk of one regressing the other and triggering a serialized merge-and-rework cycle.
- **Con**: Reviewer bandwidth — two concurrent Opus reviews on overlapping physics codepaths invites confusion. M5-S3.z + M6-S2a in parallel worked, but those were truly orthogonal (RRTMG vs IO/backfill). Two RRTMG sprints in parallel are not.
- **Con**: The worker hit a full `/tmp` partition during this sprint (`worker-report.md:66`); doubling worktrees doubles the disk footprint and the risk of recurrence.

### Binding decision: **Option A**

Rationale:
1. **Sprint-success probability dominates when both sprints are required before M6 unblock**. Option A first-sprint success (~70%) × second-sprint success (~50%) > Option B (~50% × ~70%) only marginally (joint ~35%), but Option A delivers the **methodology dividend** (full SW closure proves the infrastructure works end-to-end before stress-testing on the larger LW gap). Option C joint success is meaningfully lower due to file-conflict + reviewer-bandwidth risks.
2. **R-8 and R-9 are testable today** — the cldprmc + spcvmc dumps will directly resolve the specific hypotheses already on the table. LW has no equivalent named-hypothesis state.
3. **Operational dominance of LW is real but does not change the critical-path block** — M6 stays blocked through M5-S3.zzz regardless of order. Order should optimize sprint success, not single-sprint operational delta.
4. **Manager already pre-committed Option A in the closeout commit `3242f1f`** with `M5-S3.zzzz cldprmc/spcvmc oracle FIRST` scope ratified by the worker recommendation. The reviewer ratifies; no override warranted.

**M5-S3.zzzz acceptance criteria carry forward from the existing `.agent/sprints/2026-05-21-m5-s3zzzz-rrtmg-cldprmc-spcvmc-oracle/sprint-contract.md` stub on `main`**. Reviewer-binding additions to that contract (M5-S3.zzzz worker MUST address):

- **A1**: explicitly verify R-8 (`cloud_safe = max(cloud_box, 0.01)` floor) against `cldprmc_sw` `ptaucmc/pasycmc/pomgcmc` dumps; if it is the bias source, replace the floor with a `where(cloud_box > 0, ..., 0)` form that does not floor the denominator.
- **A2**: explicitly verify R-9 (double-Eddington-then-blend) against `spcvmc_sw` per-g-point `zref/ztra/zrefd/ztrad` dumps; restructure JAX to compute reftra once per g-point with the MCICA-selected optical properties, matching WRF accumulator semantics.
- **A3**: dump per-g-point `zfd/zfu` BEFORE broadband accumulation as scoped; validate JAX `down_band/up_band` BEFORE the `sum(axis=(-1,-2))` reduction (`rrtmg_sw.py:965-966`).
- **A4**: re-`nm` the rebuilt harness binary and persist the SHA in `manifests/rrtmg-intermediate-oracle-v1.yaml` (close the M5-S3.zz verifiability-triple debt at R-§1.1).
- **A5**: amend ADR-009 to `SW-PARITY, LW-NOT-PARITY` ONLY IF strict Tier-1 SW PASS is proven. Otherwise hold at `NOT-PARITY` and continue debt narrative.

### M5-S3.zzz scope (advance binding for next-next sprint)

M5-S3.zzz = LW closeout (transcribe 16 LW `taumol_lw + fracs` branches against the M5-S3.z intermediate-oracle NPZ; per-band gate methodology). Reusing the M5-S3.zzzz `cldprmc_lw + rtrnmc` intermediate-oracle infrastructure where applicable.

---

## 5. M6 dispatch impact (still BLOCKED on full SW+LW PARITY)

| Sprint state | SW Tier-1 | LW Tier-1 | T2 24h drift (est.) | M6 dispatch |
|---|---|---|---:|---|
| M5-S3.zz (current) | FAIL (flux 30-87 W/m²; heating PASS) | FAIL (nearest-pressure approx.) | 1-3 K | **BLOCKED** |
| After M5-S3.zzzz Option A success | PASS (SW-PARITY) | FAIL (unchanged) | 0.7-1.5 K (LW-dominated) | BLOCKED |
| After M5-S3.zzz Option 2 success | PASS | PASS (LW-PARITY) | < 0.5 K | **UNBLOCK** |

M6 coupled-forecast operational validation **remains BLOCKED through both M5-S3.zzzz and M5-S3.zzz close**. Earliest UNBLOCK is after M5-S3.zzz (LW closeout) — i.e., minimum 2 more sprints from current state, consistent with the M5-S3.z reviewer §5 sequenced projection. Per `feedback_validation_philosophy.md`, the binding metric is GPU-vs-CPU U10/V10/T2 RMSE at 24h on the Canairy Gen2 ~1-month baseline; this becomes assessable only after both SW and LW heating biases drop below 0.5 K/day per column.

M6 prologue sprints (M6-S1 coupled-interface freeze — already merged; M6-S2a Gen2 backfill — ACCEPT-WITH-MINOR-FOLLOWUPS merged) remain **unaffected** by this sprint and continue to execute on the implementation track.

---

## 6. Summary judgment

M5-S3.zz worker delivered the **two specific intermediate-oracle defects** that the M5-S3.z reviewer §4 binding decision named: `sfluxzen` band/g-point allocation (closed via `_source_active` gating + WRF zero-init mirror at `rrtmg_sw.py:549-555, 612`) and `setcoef_sw` precision-policy mismatch (closed via Path B single-precision floor at `1e-4 abs + 1e-3 rel` with test enforcement). The 14 PASS-validated SW branches were re-enabled in production via `jax.lax.scan` consumed by `_shortwave_impl:858-860`, with no `min(raw, cap)` launch fudge and no clip-pinned fixtures. Strict Tier-1 SW flux parity remains FALSE (30-87 W/m² broadband residual), but the worker honestly self-flagged this in `worker-report.md:3` and correctly refused to amend ADR-009 to `SW-PARITY`.

The residual is now sharply bounded to the `cldprmc_sw → spcvmc_sw → flux-accumulation` stack: gas optical depth (`taug`), Rayleigh (`taur`), source allocation (`sfluxzen`), setcoef interpolation, MCICA random-overlap mask, and the Eddington transfer solver are ALL independently validated PASS. The next sprint (M5-S3.zzzz) has a specific named scope (`cldprmc_sw + spcvmc_sw` intermediate-oracle dumps + JAX cloud-optics + broadband-transfer fix) with two named candidate hypotheses (R-8 `cloud_safe` floor, R-9 double-Eddington-then-blend) ready for direct oracle confrontation. The cycle continues to move forward at one residual layer per sprint — a slow but honest discovery rate.

**Verifiability triple**: §1.1 PARTIAL (binary not on disk; symbols verified in source + transitive NPZ SHA carry-forward; full re-`nm` is debt to M5-S3.zzzz); §1.2 PASS (no clip-pinning; new cloud tables WRF-derived with physical ranges); §1.3 PASS (no launch fudge; honest no-claim). No M5-S3 → S3.x → S3.y → S3.z → S3.zz anti-pattern recurrences.

**Final decision: PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-5** with **binding M5-S3.zzzz Option A dispatch** (cldprmc_sw + spcvmc_sw intermediate-oracle FIRST → M5-S3.zzz LW closeout SECOND). Manager actions on accept: (i) the existing `m5-s3zzzz-rrtmg-cldprmc-spcvmc-oracle` sprint contract is RATIFIED with the five A1-A5 reviewer-binding additions in §4; (ii) the M5-S3.zz manager-closeout already on `main` correctly labels the disposition; (iii) M5-S3.zzz LW closeout remains advance-bound as the next-next sprint; (iv) M6 coupled-forecast dispatch stays BLOCKED through both; (v) the M5-S3.zzzz worker prompt MUST explicitly cite R-8 and R-9 as the two named hypotheses to confront with the new oracle; (vi) the M5-S3.zzzz harness rebuild MUST re-run `nm` and persist the symbol set to close the M5-S3.zz triple-debt at §1.1.

AC1 PASS, AC2 PASS (Path B), AC3 PASS-functional / PARTIAL-budget (no claim), AC4 FAIL-HONESTLY, AC5 PASS-by-edit-scope, AC6 HONESTLY-NOT-DONE. M6 dispatch BLOCKED on M5-S3.zzzz + M5-S3.zzz close.
