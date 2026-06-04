# v0.9.0 RELEASE CANDIDATE — Opus-xhigh release worker handoff

- **Author:** Opus 4.8 (v0.9.0 release worker), 2026-06-04.
- **Branch:** `worker/opus/v090-release` (RC), built FROM `worker/opus/v090-release-trunk` @ `2162e04`.
- **RC tip:** `779ae7d`.
- **STATUS:** Release candidate built. **STOPPED before tagging** per protocol — the manager runs the mandatory cross-model gap-critic, then authorizes the tag (v0.9.0 + chain) + push. **One honest blocker surfaced (the naive-agent gate FAILs the strict criterion) — see §4; it is a gate-definition / writer-coverage question, NOT a forecast-correctness failure, but it must be arbitrated before the tag.**

---

## Objective

Consolidate the 7-branch-merged trunk + the three release-input branches into a single RC,
fill the README's 4 validation placeholders with REAL committed numbers, document the d03-1km
open issue, run the binding naive-agent gate, and STOP before tagging. WRF-faithful, honest,
no over-claim.

## 1. Consolidation — all branches folded cleanly (NO conflicts)

| Folded | SHA | How | Conflicts |
|---|---|---|---|
| `v090-validation-burst` | `5fec5ef` | ancestor of qke branch (auto via the qke merge) | none |
| `v090-qke-fp64-fix` | `2b9e346` | `git merge --no-ff` | none (orthogonal: precision.py / scripts / tests) |
| `v090-readme-rewrite` | `383b0d0` | `git merge --no-ff` | none (orthogonal: README / PROJECT_PLAN) |
| `v090-naive-gate-prep` ASSETS | `39e8d56` | cherry-pick of DATA files only via `git checkout <br> -- <files>` | n/a |

**Important consolidation judgement:** `v090-naive-gate-prep`'s merge-base with the trunk is the
PRE-merge `7b7c26e`; a full merge of it would have REVERTED ~19k lines of merged-trunk physics
(it shows `explicit_diffusion.py`, `_gf_jax.py`, etc. as deletions). I therefore folded in ONLY
its data/manifest artifacts (sample manifest, dry-check, staging namelist, review) and left the
RC's merged source untouched. The sample tarball sha256 (`1260f182…d17a`) was verified against
the dry-check; the CLI + `namelist_check` on the RC accept the sample.

**`python -c "import gpuwrf"` (PYTHONPATH=src): PASS** at every step (baseline, post-qke, post-readme, post-assets, final RC). qke→fp64 confirmed live in `precision.py` (`"qke": (FP64, False)`).

## 2. README placeholders — filled with REAL committed numbers (proofs cited inline)

1. **d02 (3 km) coupled skill** (`proofs/v090/d02_coupled_skill_72h.json`): finite/stable all
   72 h. **T2 within bar (3.0 K) 72/72** (mean 1.06 / max 1.42 / final 0.81 K); **V10 within bar
   (7.5) 72/72** (final 2.97); **U10 within bar 66/72** (final 4.00, max 8.04 episodic diurnal
   peak ~h22–30, recovers). HFX/PBLH within informational bands.
   - **HONEST CORRECTION I MADE vs the task brief:** the brief said "beats persistence 71/72" for
     d02 skill — that figure is the **U10** beats-persistence count, not a T2 skill score
     (T2 beats persistence only 3/72 because short-lead persistence is a very strong T2 baseline).
     I did NOT print "beats persistence 71/72" as a generic skill claim; I stated the per-field
     within-bar / RMSE numbers instead, which is what the proof actually supports.
   - **HONEST FLAG I added:** the proof's machine `status` is **`FAIL`** (the all-leads-within-bar
     predicate trips on the 6 U10 leads). I did NOT print an unqualified "GREEN" — I stated the
     finite/stable + final-hour-Tier-4-PASS + per-lead reality and the FAIL cause explicitly.
   - **Second honesty caveat I added (from the qke-fix review):** the production daily-pipeline
     path (hourly land-state refresh) on the only other data-available d02 config (`20260521`) is
     itself unstable in gated-fp32 (NaNs ~h1, qke canary). So the 72 h skill rests on the
     replay-harness path, not the production-pipeline path. README says so.
2. **d03 (1 km):** OPEN ISSUE (see §3).
3. **TOST n=15:** stated honestly as **prepared, NOT scored for v0.9.0**; no "TOST PASS" claimed;
   the d02 coupled-skill is the v0.9.0 operational equivalence; the formal n=15 TOST is the paper's
   powered analysis.
4. **Speedup:** **real-user-time WARM ≈ 2.16× (cons) / 2.41× / 2.59×** (72 h, gated-fp32,
   compile-inclusive, warm cache); **COLD first-launch ≈ 1.33×** (24 h); **1 km UNMEASURED/BLOCKED**.
   The **kernel/compute-only ≈ 5.3–7.84× ceiling is kept CLEARLY SEPARATE** and explicitly labeled
   NOT the headline (the OC-A risk). Compile-excluded real-user steady-state ≈ 2.33–2.79× (context);
   dt-matched floor ≈ 1.29× warm.

All 4 `«FILL FROM VALIDATION BURST»` markers removed; over-claim guards (OC-A…OC-E) verified intact
(no stray "TOST PASS"/"equivalence PASS" except the negations; flat-slab + fail-closed caveats present).

## 3. d03-1km gated-fp32 OPEN ISSUE — documented honestly (`docs/KNOWN_ISSUES.md` KI-1)

gated-fp32 d03 1 km NaNs after forecast hour 1 (qke the sole offending field, ~3036 cells over
~69 steep-terrain columns). The "qke fp32 overflow" hypothesis is **falsified** — qke→fp64 (now
shipped) reproduces the identical blow-up at tiny qke magnitudes, so it is a **dynamics-driven
structural instability over steep Tenerife terrain, not a precision-range problem**. 1 km is
**finite in full fp64** over the confirmed 0.3 h / 360-step window but fp64 is ~1:64-throttled so a
24 h fp64 validation is impractically slow. 3 km d02 IS stable + validated in gated-fp32 (the
primary operational resolution). Tracked for a post-0.9.0 numerics/stability sprint. README
limitations + KI-1 cite `d03_1km_validation.json`, `d03_1km_validation_qkefix.json`,
`d03_replay_finite_check.json`. No spin.

## 4. BINDING NAIVE-AGENT GATE — **FAIL on the strict criterion (NOT a correctness failure)**

Ran the documented gate command on the public sample via the GPU port CLI:
`gpuwrf run --namelist <DIR>/namelist.input --input-dir <DIR> --output-dir … --domain d02 --hours 1 --compare-cpu-dir <DIR>`. GPU lock claimed (cpu_cores_4_31 preserved) + released; cores 0-3.

**What PASSED:** namelist fail-closed validation; CLI acceptance; the 1 h GPU forecast RAN and is
**fully finite** (`all_finite=true`, 56 fields; T2 276–294 K, U10/V10 within ±7.4 m/s, PSFC, Q2 all
physical); **all 9 core spatial/vertical/soil dimensions match the CPU-WRF reference EXACTLY**
(west_east 159, south_north 66, bottom_top 44, all staggered, soil_layers_stag 4).

**What FAILED:** the strict every-dimension-equality compare → **FAIL** on 3 non-meteorological
sub-dimensions present in the CPU reference but absent from the GPU wrfout:
`seed_dim_stag` (SPPT/SKEBS/SPP stochastic-perturbation seed arrays — a feature v0.9.0 does not
implement), `snow_layers_stag` + `snso_layers_stag` (Noah-MP internal multi-layer snow diagnostics,
TSNO/SNICE/SNLIQ/ZSNSO). **Root cause:** the GPU writer emits a focused **64-variable** wrfout vs
the CPU **375-variable** catalog; the 3 mismatched dims exist only to dimension variables the writer
omits. This is a **writer-coverage / gate-definition** matter, not a grid error, transcription bug,
or instability.

The pipeline verdict is **PARTIAL** (not GREEN) only because the public sample bundles no AEMET
observations, so the obs-scoring sub-step was `NOT_RUN` — not a forecast failure.

**I did NOT relabel the FAIL as a PASS.** Proof: `proofs/v090/naive_agent_gate.json`
(+ `proofs/v090/naive_gate_run/`).

### Decision needed from the manager + cross-model critic BEFORE the tag

The strict gate as currently defined does not PASS on this RC. Two honest options:
- **(A)** Tighten the gate PASS criterion to the meteorological/spatial-dims + finiteness contract
  (which this RC passes cleanly) and document the GPU writer's focused 64-var scope explicitly (the
  seed/snow dims are out-of-scope writer coverage, already consistent with the README's fail-closed
  stochastic-perturbation + land-diagnostic boundaries).
- **(B)** Extend the GPU wrfout writer to emit the Noah-MP snow-layer diagnostics
  (TSNO/SNICE/SNLIQ/ZSNSO) so those dims appear, and keep the stochastic-seed dims as a documented
  out-of-scope omission.

Either way this is NOT a forecast-correctness blocker, but it IS an honest blocker to an unqualified
"naive-gate PASS" tag claim. My recommendation: **(A)** — it matches the existing README scope; or do
(B) as a small focused writer follow-up if the principal wants byte-for-byte dimension parity in the
public sample. The manager/critic should decide.

## Over-claims I caught / fixed

1. Task brief's d02 "beats persistence 71/72" was the U10 count, not a T2 skill claim → did not
   parrot it; used the real per-field numbers (§2).
2. d02 72h proof `status=FAIL` (U10 6-lead breach) → README does NOT call it an unqualified "GREEN".
3. Production-pipeline-path d02 instability (20260521) → surfaced as a README caveat so the 72h
   skill is not read as a production-pipeline result.
4. Speedup kernel 5.3–7.84× kept strictly separate from the 2.16× real-user headline (OC-A).
5. n=15 TOST stated as prepared-not-scored; no "TOST PASS" (OC-C).

## Risks / honest gaps for the critic

- **Naive-gate strict FAIL (the #1 item to arbitrate).** Gate-definition vs writer-coverage; not
  correctness. See §4.
- **d03 1 km gate OPEN** (gated-fp32 NaN, dynamics-driven). Carried over to a numerics sprint.
- **d02 skill is single-case / single-season (MAM)**, replay-harness path; the production daily
  pipeline is unstable in gated-fp32 over steep terrain on the data-available config. The 72h GREEN
  is the replay path.
- **n=15 TOST not scored.** Operational equivalence rests on the one d02 coupled-skill case.
- **qke→fp64** is asserted speedup-neutral on first principles (one small 3D field); not freshly
  re-timed (the prior GREEN gated-fp32 d02 cases are wrfout-purged).

## Ready for critic-then-tag?

**Yes, ready for the cross-model gap-critic** — the RC is consolidated, imports, README is filled
with real numbers and honest caveats, the open issue is documented, and the binding gate was run and
reported truthfully. **NOT ready for an unqualified tag** until the critic + manager arbitrate the
naive-gate strict-FAIL (option A or B). I did not tag, did not push.

## Files / proofs (committed on `worker/opus/v090-release`)

- Consolidated RC (merges + cherry-picked assets), `import gpuwrf` PASS.
- `README.md` (4 placeholders filled), `docs/KNOWN_ISSUES.md` (new), `PROJECT_PLAN.md` banner.
- `proofs/v090/naive_agent_gate.json` + `proofs/v090/naive_gate_run/{pipeline_payload,dimension_compare}.json`.
- This review.

## Resources

CPU pinned to cores 0-3 throughout (`taskset -c 0-3`); cores 4-31 (live CPU-WRF backfill pid 240156)
never touched; no `/mnt/data/canairy_meteo` writes. GPU lock claimed for the one gate run +
released, `cpu_cores_4_31` preserved. One GPU job.
