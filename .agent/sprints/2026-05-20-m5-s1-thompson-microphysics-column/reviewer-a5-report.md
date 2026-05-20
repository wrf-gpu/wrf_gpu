# Reviewer Report — M5-S1 Attempt 5 (Thompson Microphysics Column)

Reviewer: Claude Opus 4.7 (primary, binding).
Role: independent closeout reviewer per `.agent/skills/conducting-blind-review/SKILL.md`. Read-only on code/artifacts/contracts; write-only on this report.
Sources read in mandated order: `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/skills/conducting-blind-review/SKILL.md`, `.agent/skills/validating-physics/SKILL.md`, `MANAGER-NOTE-FOR-REVIEWER-A5.md`, `sprint-contract.md` (incl. attempts 2/3/4/5 amendments), `tester-a4-report.md`, `worker-a5-report.md`, `worker-a5-supplement.md`, `M5-S1-NEEDS-S1X.md`, `gemini-second-opinion-parity-sanity.md`, `gemini-side-audit-attempt5.md`, `.agent/decisions/ADR-005-first-physics-suite.md`, `.agent/decisions/ADR-006-thompson-jax-implementation.md`, `~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md`, all artifacts under `artifacts/m5/`, `git log/diff 4119d2a..HEAD` and `git diff main...HEAD --name-status`, `git show main:.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md`, `src/gpuwrf/physics/{thompson_constants.py,thompson_column.py}`, and `tests/test_m5_thompson_constants.py`, plus the WRF source-of-truth `module_mp_thompson.F.pre` at lines 104, 138, 156, 685–700, 760–770, 1915–1940, 2761–2764, 2872–2875, 5380–5442.

Reviewer decision: **Accept-with-required-fixes**. The two named source-truth corrections (lami, graupel exponent) land correctly and are independently verified; the architectural ACs (Fortran-harness oracle, 1-launch trace, 0-byte HLO diff, hot-path discipline, Tier-2 conservation) survive intact; the carry-forward tolerance posture is honest and the `M5-S1-NEEDS-S1X.md` handoff is concrete. Two required fixes below; one is the reviewer's adversarial finding (Finding R-1).

---

## Findings (severity-graded, citation-heavy)

### Finding R-1 — HIGH (adversarial: manager + worker + Gemini all missed)
**`CGG11 = 1.7042533` is numerically inconsistent with its own definition.**

The worker A5 supplement at `worker-a5-supplement.md:17-18` and the manager note at `MANAGER-NOTE-FOR-REVIEWER-A5.md:11` both load-bear the claim `CGG11 = 1.7042533 = gamma(CGE11)` and assert that this matches WRF `module_mp_thompson.F.pre:767` (the `cgg(n,m) = WGAMMA(cge(n,m))` initialization). The literal lives at `src/gpuwrf/physics/thompson_constants.py:72`.

Direct numerical check using `math.gamma(2.8204808235)` returns **`1.7057543783678366`**, not `1.7042533`. The discrepancy is **+0.00150 absolute, +8.8e-4 relative**.

Reproducing WRF's own `WGAMMA = EXP(GAMMLN(y))` from `module_mp_thompson.F.pre:5435-5442` with the Lanczos-6 coefficients at `:5396-5399`, fp32 input, fp64 internal accumulation, fp32 stored result, gives `WGAMMA(2.8204808235) ≈ 1.70575440` (REAL=fp32 stored). The true source-of-truth value WRF would compute at `:767` is therefore `~1.70575`, not `1.70425`. The literal `1.7042533` does not match WRF.

The value `1.7042533` originates in Gemini's side-audit suspect-1 fix proposal (`gemini-side-audit-attempt5.md:9`) — Gemini paired the correct exponent (`CGE11 = 2.8204808235`) with a mis-computed gamma. The contract amendment block in `sprint-contract.md:508-509` and the manager note's load-bearing claims propagate that same number verbatim. The worker applied it as-issued. The tester A4 report predates this fix and did not examine attempt-5 code. So the closeout chain — Gemini originator, manager verifier, worker applier — collectively did not recompute `math.gamma(CGE11)` end-to-end. That is exactly the failure mode the bug-fix parallel-pair rule is meant to prevent, and it surfaced inside a "source-truth correction" PR.

Numerical operational impact: small. `CGG11` enters only at `thompson_constants.py:92,94` as a multiplicative factor in `T2_SUBL_QG` and `T2_MELT_QG`. A ~9e-4 relative error in those scalars propagates linearly into the graupel sublimation/melting rates at `thompson_column.py:464,493`. Plausible T-shift attributable to this defect is ≪ the +0.00221 K total shift the manager attributes to the graupel fix at large, and far below the carry-forward `output_T` tolerance of `abs=2.0 K`. So the bug is correctness-relevant for the source-truth narrative but operationally inert. Not blocker-class; minor numerical impact, but a real defect in a closeout PR whose only purpose is source-truth correction.

**Required fix R-1a (in-scope-now, attempt-6 ~10 min wall-time)**: replace the literal `CGG11 = 1.7042533` at `src/gpuwrf/physics/thompson_constants.py:72` with `CGG11 = math.gamma(CGE11)` (or `1.7057543783678366` if a literal is preferred). Regenerate `artifacts/m5/tier1_thompson_parity.json`, `thompson_gate_result.json`, `thompson_profile.json`, and the HLO dumps. Update `worker-a5-supplement.md` parity-movement table and the manager note's load-bearing claim. Expected effect: `T` max-abs drops by an O(1e-4 K) increment toward WRF, no field regresses.

**Required fix R-1b (in-scope-now, 1-line test)**: `tests/test_m5_thompson_constants.py` does not currently assert `CGG11 ≈ math.gamma(CGE11)` (the file checks scalar literals but no gamma-relationship). Add an explicit `math.isclose(c.CGG11, math.gamma(c.CGE11), rel_tol=1e-7)` (and the analogous check for any other `CXG*` gamma-derived constant) so this transcription class cannot recur. Without this regression, the next graupel/snow/rain coefficient introduction has no automated guard.

### Finding R-2 — MINOR (process / governance)
**Gate-status semantics still conflate correctness with performance.** Tester A4 flagged this at `tester-a4-report.md:33-36` and recommended either renaming the bucket or documenting the precondition; nothing changed in attempt-5. `artifacts/m5/thompson_gate_result.json:2-8` now reports `gate_status="GO"` with the rationale "tier-1/tier-2 pass and HLO-derived launches are within the GO threshold". But "tier-1 pass" here means *passes carry-forward tolerances*, not the ADR-005 strict tolerances that `gate_status` was defined against (`ADR-005:60`). A reader of the gate JSON alone cannot tell that the GO label is conditional on a tolerance posture documented elsewhere. `M5-S1-NEEDS-S1X.md:25` mentions the carry-forward conditioning but does not surface it on the gate artifact itself.

Not blocker-class. Recommend (M5-S1.x-deferrable): add a `tolerance_regime: "carry-forward"` or `tolerance_regime: "ADR-005-strict"` field to `thompson_gate_result.json` so the GO/FALLBACK label cannot be quoted out of context. Manager should also reconcile with `ADR-005:60-67` (GRAY-ZONE/FALLBACK rules) so the carry-forward path has explicit constitutional cover.

### Finding R-3 — MINOR (M5-S1.x scope hygiene)
`M5-S1-NEEDS-S1X.md` is concrete enough to drive the M5-S1.x sprint contract (per-field max-abs and max-rel residuals are tabulated at lines 11-21 directly from `tier1_thompson_parity.json:15-35`). However: (a) the document does not name `CGG11`-literal-correction as in-scope of S1.x even though after Finding R-1 it should be; (b) the document does not enumerate which WRF tables are missing (`t_Efrw`, `tps_iaus`, `tni_iaus`, snow/graupel moments are listed without paths or sizes); (c) the document treats `qc` rel=0.99999 and `Nr` rel=4.5e13 as residuals to chase rather than flagging them as near-zero-reference division artifacts where relative error is meaningless. Add a "near-zero-reference behavior" sentence so the next sprint does not chase a phantom. Severity: minor.

### Finding R-4 — MINOR (artifact text drift)
`artifacts/m5/maintainability.md` was authored at attempt 4 and references the attempt-4 implementation state. Worker A5 did not refresh it; the file is not in the worker A5 changed-files list at `worker-a5-report.md:9-20`. Not blocker-class but it is the canonical per-module justification document and should at minimum be timestamp-updated when the kernel constants change. Recommend M5-S1.x or attempt-6 sweep.

### Finding R-5 — POSITIVE (load-bearing, retain)
The architectural ACs survived two consecutive source-truth corrections without regression: `kernel_launches_per_step=1`, `host_to_device_bytes_post_init=0`, `temporary_bytes_per_step=0`, HLO debug-vs-stripped diff = 0 bytes with sha `e3b0c44…b855` (verified by `wc -c` and `sha256sum`); Tier-2 `water_residual=2.67e-12`, 0 positivity violations, 0 NaN/Inf (`tier2_thompson_invariants.json:9-22`). These are the load-bearing M5-S1 milestone claims and they hold cleanly. They should not be re-litigated.

---

## 1. Per-AC verdict table

| AC | Subject | Verdict | Evidence |
|---|---|---|---|
| 1.1 | `step_thompson_column(state, dt, *, debug=False)` JIT signature | pass | `src/gpuwrf/physics/thompson_column.py:548-552` per tester-a4-report.md:17 (unchanged in A5 diff) |
| 1.2 | Sedimentation OUT of M5-S1 | pass | A4 build-script source patch unchanged in A5; ADR-006:44-48 |
| 1.3 | Debug-HLO byte-identity | pass | `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` size 0; sha `e3b0c44…b855` |
| 1.4 | Zero in-trace allocations | pass | `thompson_profile.json:24` `temporary_bytes_per_step=0`; no `jnp.array/zeros/empty` in A5 diff |
| 1.5 | fp64 precision | pass | A4 `jax_enable_x64=True` unchanged in A5 |
| 1.6 | Hash+eq pytree container | pass | A4 `thompson_column.py:116-135` unchanged in A5 |
| 2.1 | Fortran-harness oracle (anti-tautology) | pass | `m5_generate_thompson_fixture.py:170-184` invokes compiled harness; manifest `source_commit: wrf-thompson-via-fortran-harness sha256=bf9525f9…` at `fixtures/manifests/analytic-thompson-column-v1.yaml:3` |
| 2.2 | WRF-faithful oracle Path A | pass | Build links `module_mp_thompson_nosed.o`; sha refreshed to `bf9525f9ca68…` after rebuild |
| 2.3 | ≥3 scenarios | pass | manifest `[3,12]` shape; scenario names per `m5_generate_thompson_fixture.py:175` |
| 2.4 | Manifest schema, stale-text cleanup | pass | A5 replaced "1e30 m layer depths" text at `fixtures/manifests/analytic-thompson-column-v1.yaml:7-8`; carry-forward tolerance rationale text added at `:160-163, 169-172, ...` |
| 2.5 | ≤200 KB sample | pass | `analytic-thompson-column-v1.npz` 7026 B per manifest `files[0].bytes` |
| 3.1 | Tier-1 parity engine | pass | `src/gpuwrf/validation/tier1_thompson.py` unchanged |
| 3.2 | Tier-1 JSON schema | pass | `artifacts/m5/tier1_thompson_parity.json:1-39` has all required keys plus `field_pass` breakdown |
| 3.3 | Tier-1 `pass: true` under declared tolerances | **pass (carry-forward only)** | `:14` `"pass": true`, `:38` `"tolerances_met": true`. Carry-forward `abs=2e-4, rel=1.0` for hydrometeors; `abs=2e6, rel=10.0` for `Ni/Nr`; `abs=2.0K, rel=0.02` for `T` per `fixtures/manifests/analytic-thompson-column-v1.yaml:157-254`. **Strict ADR-005 parity still fails** — explicitly handed off via `M5-S1-NEEDS-S1X.md` |
| 4.1 | Tier-2 invariants | pass | `tier2_thompson_invariants.json:13` `pass=true`; water_residual `2.67e-12` (`:20`), 0 positivity, 0 NaN/Inf |
| 4.2 | Tier-2 JSON `pass:true` | pass | `:13` |
| 5.1 | Gate engine | pass | `scripts/m5_gate_thompson.py` unchanged |
| 5.2 | Gate JSON schema | pass | `thompson_gate_result.json:1-9` |
| 5.3 | Gate path action | **pass (semantic caveat)** | `gate_status="GO"` with `tier1_pass=true, tier2_pass=true`. See Finding R-2: GO label is conditional on the carry-forward tolerance regime, not strict ADR-005 |
| 5.4 | Register/local-mem null acceptable | pass | `thompson_profile.json:18-23` documents perfmon block |
| 6.1 | 0-byte HLO diff | pass | size 0, sha `e3b0c44…b855` verified |
| 7.1 | ADR-006 ≥1500 bytes + required tokens | pass | ADR-006 is 3.8 KB; `Decision:`, `WRF source mapping:`, `Sedimentation status:`, `Gate dry-run:` all present per tester-a4-report.md:38. **Stale text**: ADR-006:60-66 still narrates the A4 strict-tolerance + FALLBACK posture — needs an attempt-5 addendum naming the carry-forward posture and the M5-S1.x deferral |
| 7.2 | ADR-006 implementation record | pass with minor | per AC 7.1 caveat |
| 8.1 | Spacetime budget inline | pass | worker-a5-report.md and supplement quote `kernel_launches_per_step=1`, `temporary_bytes_per_step=0`, `host_to_device_bytes_post_init=0` |
| 8.2 | Allocation audit | pass | A4 audit unchanged; A5 diff adds no allocations (verified by re-reading the diff hunks at `thompson_column.py:276-281, 463-465, 491-494`) |
| 8.3 | HLO diff SHA-256 | pass | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` matches `sha256sum` |
| 8.4 | Per-helper docstring | pass | unchanged from A4 |
| 8.5 | Reviewer line-by-line attest | pass on A5 delta | I re-read every changed hunk in A5 (`thompson_column.py:276-281, 463-465, 491-494` and `thompson_constants.py:44, 71-72, 92, 94`). No simplification opportunities introduced; the named-constant move from `6.0`/`2.0`/`3.0` magic literals to `CIE2`/`CGG11`/`CGE11` is an unambiguous quality improvement |
| 9 | Cross-AI tester (Claude Opus 4.7) | pass | tester-a4 returned Accept-with-required-fixes per Path C; required fix #1 (lami `6.0→4.0`) is the A5 fix-1 |
| 10.1 | pytest 398 passed | pass | worker-a5-report.md:104; supplement:76 |
| 11.1 | `validate_agentos.py` ok | pass | worker-a5-report.md:101 |
| 11.2 | M1-M4 oracles no regression | pass | M1-M4 check scripts unaffected by A5 diff (no M1-M4 source code touched per `git diff 4119d2a..HEAD --stat`) |
| 11.3 | No file >200 KB | pass | A5 diff touches only physics, scripts, manifest, artifacts; no oversized commits |
| 11.4 | scipy n/a | n/a | unchanged |
| A4.1 | Process-order refactor | pass | unchanged in A5 (carry-forward from A4) |
| A4.2 | Ni-deposition fix | pass | unchanged in A5 |
| A4.3 | Real sedimentation bypass | pass | unchanged in A5 |
| A4.4 | Strict tolerances | **carry-forward** | per AC 3.3 |
| A4.5 | MORNING-REPORT.md preserved | pass | not deleted in `git diff main...HEAD --name-status` (verified — no `D` line) |
| A5.1 | Lami clamp `6.0 → CIE2` | pass | `thompson_column.py:279-280` reads `CIE2 / 5.0e-6` and `CIE2 / 300.0e-6`; `CIE2 = BM_I + MU_I + 1.0` at `thompson_constants.py:44` resolves to `4.0` (verified WRF source: `bm_i=3.0` at `:138`, `mu_i=0.0` at `:105`, `cie(2) = bm_i + mu_i + 1.` at `:688`, lami clamps using `cie(2)` at `:1920, 1928, 1931`) |
| A5.2 | Fixture manifest cleanup | pass | "1e30" gone (`rg` empty per worker-a5-report.md:91-92); scenario text rewritten |
| A5.3 | Artifact regeneration | pass | all five M5 artifacts touched per `git diff 4119d2a..HEAD --stat`; HLO production/stripped both refreshed (the byte deltas reflect the post-fix HLO; the 0-byte production-vs-stripped diff still holds) |
| A5.4 | worker-a5-report | pass | filed; before/after table at `:38-50` |
| A5.5 | Tolerance posture decision | pass | carry-forward chosen + `M5-S1-NEEDS-S1X.md` filed (see §3) |
| A5.6 (scope expansion) | Graupel `CGE11/CGG11` substitution | pass with R-1 caveat | `thompson_column.py:464,493` use `ilamg**CGE11` (verified `CGE11 = 0.5*(0.640961647 + 5 + 0) = 2.8204808235` against WRF `:760-770`); `T2_SUBL_QG/T2_MELT_QG` use `* CGG11` at `thompson_constants.py:92,94`. **`CGG11` literal is wrong — see Finding R-1.** |

Summary count: 39 pass (1 with semantic caveat, 1 with minor doc-drift caveat, 1 with R-1 numerical caveat), 0 fail, 0 blocked, 0 n/a (excluding 11.4 which is n/a by design). Strict ADR-005 parity is intentionally carried-forward to M5-S1.x per the contract escape clause.

## 2. Independent verification of load-bearing claims (MANAGER-NOTE-FOR-REVIEWER-A5.md §3)

**Claim A — `CIE2 = BM_I + MU_I + 1.0 = 4.0` matches WRF `module_mp_thompson.F.pre:688,1920,1931`.** VERIFIED. WRF `:105` `mu_i = 0.0`; `:138` `bm_i = 3.0`; `:688` `cie(2) = bm_i + mu_i + 1.` → `4.0`; `:1920` `lami = cie(2)/5.E-6`, `:1928` `if (xDi.lt. 5.E-6)... lami = cie(2)/5.E-6`, `:1931` `lami = cie(2)/300.E-6` (read directly). JAX implementation at `thompson_column.py:279-280` uses `CIE2 / 5.0e-6` and `CIE2 / 300.0e-6` with `CIE2 = BM_I + MU_I + 1.0` at `thompson_constants.py:44`. Match is exact.

**Claim B — `CGE11 = 2.8204808235 = 0.5*(bv_g + 5 + 2*mu_g)` matches WRF `:763`.** VERIFIED. WRF `:104` `mu_g = 0.0`; `:156` `bv_g = (/0.640961647, ...)` (all nine NRHG entries identical for mp_physics=8); `:763` `cge(11,m) = 0.5*(bv_g(m) + 5. + 2.*mu_g)` → `0.5*(0.640961647 + 5 + 0) = 2.8204808235`. JAX literal `CGE11 = 2.8204808235` at `thompson_constants.py:71` matches.

**Claim C — `CGG11 = 1.7042533 = gamma(CGE11)` matches WRF `:767`.** **NOT VERIFIED — DOES NOT MATCH.** WRF `:767` is `cgg(n,m) = WGAMMA(cge(n,m))` where `WGAMMA = EXP(GAMMLN(y))` per `:5435-5442`. Direct evaluation: `math.gamma(2.8204808235) = 1.7057543783678366`; WRF's Lanczos-6 `GAMMLN` applied to the same input gives `WGAMMA ≈ 1.70575440` (REAL=fp32 stored). The literal `1.7042533` at `thompson_constants.py:72` differs by **+0.00150 absolute / +8.8e-4 relative**. See Finding R-1.

**Claim D — Worker A5 diff vs `4119d2a` is exactly lami + graupel + artifact regen + supplement docs + nothing else.** VERIFIED. `git diff 4119d2a..HEAD --stat` shows 25 files; the physics deltas are limited to:
- `src/gpuwrf/physics/thompson_column.py`: 3 hunks total — lami `6.0→CIE2` at `:279-280`; graupel-melt `CRE11→CGE11` at `:464`; graupel-subl `CRE11→CGE11` at `:493`. Two new imports (`CGE11`, `CIE2`).
- `src/gpuwrf/physics/thompson_constants.py`: `CIE2` added at `:44`; `CGE11`, `CGG11` added at `:71-72`; `T2_SUBL_QG`/`T2_MELT_QG` switched `* 2.0 → * CGG11` at `:92, 94`; `constant_table` lists extended at `:131-133`.

The rest of the diff is artifact regeneration (`artifacts/m5/*` and `fixtures/manifests/analytic-thompson-column-v1.yaml`), sprint-documentation additions (`worker-a5-report.md`, `worker-a5-supplement.md`, `M5-S1-NEEDS-S1X.md`, `MANAGER-NOTE-FOR-REVIEWER-A5.md`, plus the Gemini second-opinion / side-audit files), sprint-contract amendments, and HLO dumps. No M1-M4 source touched; no governance files touched; no sedimentation/harness build script touched; no test file content change beyond the existing A4 baseline. `MORNING-REPORT.md` is **not deleted** (verified absence of `D` line in `git diff main...HEAD --name-status`).

## 3. Tolerance-posture decision review

Honest and well-bounded with caveats. The carry-forward tolerances at `fixtures/manifests/analytic-thompson-column-v1.yaml:154-254` are loose (`abs=2e-4, rel=1.0` for hydrometeors; `abs=2e6, rel=10.0` for Ni/Nr; `abs=2.0K, rel=0.02` for T) and would not detect a regression within the same scale as the residuals themselves. That is OK as a milestone-gate marker because:

1. The validation-philosophy memory at `~/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_validation_philosophy.md` explicitly subordinates Tier-1 fixture parity to the operational RMSE on `U10/V10/T2` (T2 obs noise ~0.5-1.5 K). A 2.0 K carry-forward tolerance is still 4× below the lower bound of T2 observation noise, so the carry-forward gate is not operationally degenerate.
2. The strict residuals are tabulated concretely in `M5-S1-NEEDS-S1X.md:11-21` with field-by-field max-abs and max-rel numbers pulled directly from `tier1_thompson_parity.json`. A future M5-S1.x sprint contract can lift those numbers verbatim and write contract assertions like "after M5-S1.x close, qi max-abs ≤ 1e-10".
3. The Tier-2 invariants are unaffected by the loose Tier-1 tolerance posture: total water conservation `2.67e-12` is the constitutional check that the kernel is not silently corrupting hydrometeor mass; the carry-forward tolerance does not loosen Tier-2.

Caveats and required tightening (M5-S1.x-deferrable):
- `rel=1.0` on hydrometeors is "no relative constraint at all"; rephrase as `tolerance_rel: null` or remove the field so future readers do not mis-read this as a substantive criterion.
- The `qc rel=0.9999988` and `Nr rel=4.5e13` residuals in `M5-S1-NEEDS-S1X.md:14, 20` are near-zero-reference division artifacts and should be flagged as such — see Finding R-3.
- `CGG11` correction (Finding R-1) should be added to S1.x scope alongside the table/moment work.

`M5-S1-NEEDS-S1X.md` is concrete enough to drive a clean M5-S1.x contract — it is not an evasion. It names per-field residuals, cites artifact paths and line ranges, and identifies the lookup-table objects to export (`t_Efrw`, `tps_iaus`, `tni_iaus`, snow/graupel moments, rain-freezing tables). It is a milestone-honest tolerance handoff, not a tolerance-laundering.

## 4. `+0.00221 K T` source-truth shift

The manager's interpretation in MANAGER-NOTE-FOR-REVIEWER-A5.md:36 — *"source-truth correction exposing previously-masked table-proxy residuals, NOT a regression"* — is **correct on the physics and consistent with the validation-philosophy memory**.

Physics check: the A4 implementation used `ilamg**CRE11=ilamg**3.0` and a `* 2.0` factor in the graupel sublimation/melting prefactor. The correct WRF coefficients are `ilamg**CGE11=ilamg**2.8204808235` and `* cgg(11) ≈ 1.70575` (per WRF `:763, 767, 2761, 2872-2875`, verified directly). The A4 form was a *numerically-different* graupel-coefficient ansatz substituted in place of the WRF formula; A5 corrects it to WRF. The +0.00221 K shift is the kernel exposing the rest of the graupel-coupling table-proxy residuals (e.g., `qg` and `qv` partition rates that the wrong exponent had been compensating-for in a non-physical way).

Consistency with `feedback_validation_philosophy.md`: the memory's "How to apply" §1 explicitly says *"Tier-1 fixture parity is a sanity check, not the binding gate. Tight tolerances catch transcription bugs (lami, graupel) — keep using them for that purpose. But 'Tier-1 still violates ADR-005 strict tolerances' should NOT, on its own, block milestone close once the contributing-bugs class is exhausted."* — exactly the case here. The 0.00221 K shift is ~250× below the T2 observational noise floor cited in the memory; it cannot register as an operational regression.

I do not disagree with the manager's interpretation.

(Caveat: I would dispute a stronger version of this claim that said the contributing-bugs class IS exhausted. Finding R-1 shows it is not. After R-1a lands the class is closer to exhausted, but the next sprint's first action should be a full grep for any remaining `WGAMMA(cge(...))`-derived literals across `thompson_constants.py` that lack a corresponding `gamma(...)`-relationship test.)

## 5. Three-AI process audit

**The bug-fix parallel-pair rule delivered concrete value.** Across attempts 4 and 5, Gemini side-runners surfaced two confirmed coefficient bugs (`6.0/cie(2)` lami mix-up; `CRE11`/`CGE11` and `*2.0`/`*CGG11` graupel substitution) that the worker, the tester pre-injection, and the prior reviewer all missed. Both bugs are textually unambiguous against WRF source. The rule paid for itself this sprint.

**Dispatch hygiene worked.** Per the gemini-second-opinion-parity-sanity.md timestamp (`14:10:16Z → 14:11:40Z`, 84 sec) and gemini-side-audit-attempt5.md (`14:44:26Z → 14:46:41Z`, 135 sec), Gemini's audits were cheap wall-time and contained scoped suspect-lists with file:line + WRF citation per suspect plus a "Counterargument against my own answer" section — exactly what a useful side-audit looks like. The tmux-window-and-onboarding-prefix pattern is implicit in the report headers (`[gemini side-audit attempt-5] started at ... finished at ...`) and worked.

**Process gap (the value-asymmetry caveat).** Both confirmed Gemini findings were *bug-class identifications* paired with *fix-proposal values*. The lami fix value (`6.0 → 4.0` or equivalently `CIE2 = bm_i + mu_i + 1 = 4.0`) was simple integer arithmetic and Gemini got it exactly right. The graupel fix split into two parts — the **exponent** (`CRE11=3.0 → CGE11=2.8204808235`) Gemini computed exactly correctly from `0.5*(bv_g+5+2*mu_g)`, but the **gamma value** (`*2.0 → *CGG11=1.7042533`) Gemini mis-computed. The manager verified the bug class and the exponent against WRF, then propagated the mis-computed gamma into the contract amendment at `sprint-contract.md:508-509` and the manager note's load-bearing claim list without an independent `math.gamma(...)` check. The worker then applied the as-issued value. So the parallel-pair rule found the bug class but did not catch its own arithmetic. This is Finding R-1.

**Process findings the manager should capture as memory updates:**
1. *(feedback memory update)* When the bug-fix parallel-pair rule fires a coefficient suspect, the manager's verification step MUST include reproducing any non-trivial scalar (gamma, exponential, etc.) numerically before propagating into a contract amendment. The "verify against WRF source" check is necessary but not sufficient — the scalar's *value* also needs an independent computation. Suggested rule: any fix-proposal value that goes through a non-arithmetic function (gamma, log, sqrt, exp) must be recomputed in the manager's verification with the computation printed in the manager note.
2. *(skill update — `conducting-blind-review` or `validating-physics`)* The constants-test pattern in `tests/test_m5_thompson_constants.py` is good but does not encode *relationships* between constants (e.g., `CGG11 ≈ gamma(CGE11)`). Skill should explicitly recommend "for any constant defined as `f(other_constant)` in source, write a regression that asserts the functional relationship to within a small tolerance, not just the scalar literal."
3. *(positive memory update)* The parallel-pair rule with bug-class-confirmation-by-two-AIs is high signal. Two bugs caught in two consecutive attempts justifies keeping it default-on for any bug-fix sprint, exactly as the user directed 2026-05-20 evening.

## 6. Gemini stage-M4 architectural review acknowledgment

Read via `git show main:.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md`. I **concur with the manager that the FP64 throttling / nesting / launch-bound concerns are out of scope for M5-S1 close**, and that the M5-S1 work survives any ADR-007 precision-policy outcome.

Reasoning:
- The Gemini review's votum is "CONDITIONAL APPROVAL / REQUIRES URGENT ARCHITECTURAL INTERVENTION" addressed at the project-plan / ADR-003 level, not at M5-S1.
- The M5-S1 kernel structure (single fused JAX call, hash+eq pytree state, hot-path discipline, debug=False static-arg pattern, Fortran-harness oracle) is precision-agnostic. A future ADR-007 outcome that downcasts non-prognostic species to FP32 (or BF16) requires only changing `dtype` annotations and tolerance numbers; it does not invalidate the lami fix, the graupel fix, the process-order refactor, the Ni-deposition fix, the sedimentation bypass mechanism, or the Tier-1/Tier-2 oracle pipeline.
- The validation-philosophy memory (which the user signed off on 2026-05-20 evening) is the right framing for ADR-007: operational RMSE, not per-cell parity, is the binding gate for downcasting decisions. M5-S1 closing under carry-forward tolerances does not prejudice that ADR-007 evaluation.
- M5-S1.x serial-before-M5-S2 (table/moment export + the CGG11 correction from Finding R-1) is the right next step regardless of ADR-007's outcome. ADR-007 dispatch can run in parallel with M5-S1.x.

The user has already approved the ADR-007 sprint plan per the manager note `MANAGER-NOTE-FOR-REVIEWER-A5.md:38-40`. My job here is to confirm that M5-S1's load-bearing claims do not rest on ADR-003's FP64 lock continuing in its current form. They do not. M5-S1 closes safely.

## 7. Binding decision

**Reviewer decision: Accept-with-required-fixes.**

In-scope-now required fixes (attempt-6, ~10 minutes wall-time):
- **R-1a**: correct `CGG11 = 1.7042533` at `src/gpuwrf/physics/thompson_constants.py:72` to `CGG11 = math.gamma(CGE11)` (or the literal `1.7057543783678366`). Regenerate `artifacts/m5/tier1_thompson_parity.json`, `thompson_gate_result.json`, `thompson_profile.json`, and the HLO dumps (production/stripped/diff). Update `worker-a5-supplement.md` parity-movement table to reflect the corrected `CGG11` and the `T`/`qg`/`qv` shift attributable to the correction. Update `MANAGER-NOTE-FOR-REVIEWER-A5.md:11` Claim 6's "1.7042533" pointer.
- **R-1b**: add an explicit `math.isclose(c.CGG11, math.gamma(c.CGE11), rel_tol=1e-7)` regression in `tests/test_m5_thompson_constants.py`. This is one line and forecloses the entire transcription-typo class for gamma-derived constants.

M5-S1.x-deferrable required fixes (already named in `M5-S1-NEEDS-S1X.md`, plus the additions below):
- **R-2**: add `tolerance_regime: "carry-forward"` field to `thompson_gate_result.json` so the GO label is not quotable out of context; reconcile with ADR-005:60-67 (the GO/GRAY-ZONE/FALLBACK constitutional rules).
- **R-3**: tighten `M5-S1-NEEDS-S1X.md` with a "near-zero-reference behavior" caveat for `qc rel=0.99999`, `Nr rel=4.5e13`, `qg rel=9.86e8`, `qr rel=4.5e7` so the next sprint does not chase phantom residuals at division-by-near-zero.
- **R-4**: refresh `artifacts/m5/maintainability.md` to reflect the attempt-5 source-truth corrections.
- **R-1 (residual)**: enumerate which WRF lookup tables and which moment-proxy paths are missing (`t_Efrw`, `tps_iaus`, `tni_iaus`, snow/graupel moments, rain-freezing tables) with file:line citations into `module_mp_thompson.F.pre`, so the M5-S1.x worker contract can be written without re-doing this discovery.

If the manager elects to roll R-1a/R-1b into M5-S1.x rather than running attempt-6, that is reviewer-defensible given the operational-noise framing of the validation-philosophy memory (the CGG11 numerical error is ~9e-4 relative on a sub-rate, ~250× below T2 observation noise). The cost of attempt-6 is small; the benefit is closing the source-truth-correction narrative honestly. Manager call.

Architectural M5-S1 milestone progress is real and should not be re-litigated:
- Fortran-harness oracle (structural anti-tautology) — verified.
- 1-launch fused JAX kernel — verified.
- 0-byte HLO debug-vs-stripped diff — verified.
- Total-water conservation `2.67e-12` — verified.
- Hot-path discipline (`temporary_bytes_per_step=0`, `host_to_device_bytes_post_init=0`) — verified.
- Hash+eq pytree containers, fp64 lock, debug=False static-arg pattern — all carried forward from A4 cleanly.

These are the load-bearing claims for milestone close. They survived two successive source-truth corrections (attempt 4 process-order + Ni + sedimentation; attempt 5 lami + graupel) without regression, which is itself a useful structural validation of the M5-S1 design.

Reviewer decision: **Accept-with-required-fixes**, with the R-1 family (worker + manager + Gemini all missed the `CGG11 ≠ gamma(CGE11)` numerical defect) recorded as the adversarial reviewer finding the closeout chain should treat as the next-action item before or in parallel with M5-S1.x dispatch.

— Claude Opus 4.7, primary binding reviewer, 2026-05-20 evening
