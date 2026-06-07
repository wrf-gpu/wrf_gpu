# v0.12.0 Pre-Release Gap-Analysis Critique (Opus, skeptical WRF-dev + release-eng)

READ-ONLY audit of `worker/opus/v0120-integration` @ 138262e (base v0.11.0 ac71ce8).
No code changed, no GPU used. CPU fast-test suites run under `JAX_PLATFORMS=cpu`.

**Overall verdict: FIX-THEN-SHIP.** The engineering is solid and the per-deliverable
docs (equivalence-demo.md, PERFORMANCE.md, scheme catalog) are notably honest. But the
**top-level README/KNOWN_ISSUES are still 100% v0.11.0 text** — they never describe the
v0.12.0 release — and the **shipped equivalence-demo proof object contradicts the
"PSFC now fixed" claim** because it was generated on pre-fix code and not re-run. Both
are gatekeeper-visible. Fix the four FIX-NOW items, then ship.

---

## FIX-NOW (undercuts the release claim — must fix before tag)

### FN-1. README is a v0.11.0 document tagged as v0.12.0
`README.md` title/status say **"Current status — v0.11.0"** (line 84), **"v0.11.0 is the
feature-complete release"** (line 86), and the entire "What v0.11.0 is", "Validation
(v0.11.0)", and "Honest boundaries — what v0.11.0 does NOT claim" sections describe
v0.11.0. `pyproject.toml` (line 7) and `src/gpuwrf/__init__.py` (line 33) are bumped to
`0.12.0`, but the README never describes the v0.12.0 deliverables. The only mentions of
"0.12.0" in README are forward-references ("fix→v0.12.0", "carried into v0.12.0").

A user/gatekeeper who clones the v0.12.0 tag reads a README that claims the release is
v0.11.0 and that the v0.12.0 work is still in the future. **Concretely missing from the
README:** the standalone auto-detect CLI (`run` → replay vs native-init), standalone
**live-nested `--max-dom`**, the persistent JIT compile cache, the **PSFC fix**, and the
**GPU-vs-CPU equivalence demo**. None of the five headline v0.12.0 features is in the
status/validation body.
**Fix:** Rewrite the status, validation, and known-issues sections for v0.12.0 (new
"Current status — v0.12.0" block; add the 5 deliverables; move v0.11.0 to the lineage
list). This is the principal-required "FINAL RELEASE WORKER always updates README" step.

### FN-2. Shipped equivalence-demo proof contradicts the "PSFC fixed" claim
The committed proof `proofs/v0120/equivalence_demo_20260509_d02.json` (generated
2026-06-07T06:24Z, commit 15e2be8) reports **PSFC EXCEEDS_TOL: pooled RMSE 707.8 Pa,
bias −703 Pa** (tol 120 Pa), and `docs/equivalence-demo.md` (lines 93, 102-109) describes
that ~590→700 Pa PSFC offset as "the prime follow-up." But the **PSFC fix landed AFTER**
the demo run: `93fed51 [v0120-psfc] FIX systematic PSFC offset` is committed at 07:36Z,
~70 min after the eqdemo (07:26Z), and the demo was **not re-run**. The fix IS genuinely
wired (`runtime/operational_mode.py::_psfc_from_state` L2660→L2780;
`io/wrfout_writer.py` p8w extrapolation L935-949/1018; `psfc_extrapolation_proof.json`
shows bias 328→−29 Pa on the same case at h1).

Net: the release's own self-serve equivalence proof shows the single largest field
failure (PSFC) that the release also claims to have fixed. A skeptic who runs the demo,
or reads the JSON/doc, sees a 700 Pa PSFC failure on the shipped code's predecessor.
**Fix (preferred):** re-run `scripts/equivalence_demo.py` on the same case with the
PSFC fix in place and replace the JSON + the doc's observed-result table. If GPU is
unavailable before tag, **add an explicit banner** to both the JSON (a `caveat` field)
and `docs/equivalence-demo.md` stating the run predates the PSFC fix (commit 93fed51),
that PSFC pooled RMSE is expected to drop to ≈30 Pa (within the 120 Pa tol) on the fixed
code per `psfc_extrapolation_proof.json`, and that a re-run is pending. Do NOT ship the
PSFC-fixed README alongside an unannotated PSFC-failing flagship proof.

### FN-3. Two un-reconciled speedup numbers a gatekeeper will collide
The README front page headlines **"~5× (band 5–8×)"** for d02 (and the per-watt/whole-Earth
framing). But the **self-serve equivalence demo on the same 20260509 d02 24 h case
measures 1.70×** (`speedup.speedup = 1.6990`; GPU 1408.6 s vs CPU 2393.2 s for 24 h).
Both are individually defensible — the 5× numerator is warm 15.35 s/fc-hr at dt=10 s vs
CPU clean-compute 83 s/fc-hr; the demo's 58.7 s/fc-hr GPU includes cold-compile/IO and
runs at the case's dt=6 s, and its CPU denominator is the mean-step "realistic" 99.7
s/fc-hr — but **nothing in the demo doc/README reconciles the gap**, so a skeptic who runs
the demo gets 1.70× and then reads "~5×" and concludes one of them is dishonest. The demo
doc (line 117-121) labels it "not the warm/fused kernel speedup quoted elsewhere," which
is a start, but does not explain *why* it is 3× lower or connect it to the headline.
**Fix:** Add one paragraph to `docs/equivalence-demo.md` (and a one-line footnote where
the README headline appears) reconciling the two: demo GPU run is cold-compile-amortized +
per-hour IO + dt=6 s matched-to-CPU, vs the warm dt=10 s steady-state numerator behind the
5× headline; cite `speedup_denominator.md`. Without this, the demo undercuts the headline.

### FN-4. KNOWN_ISSUES.md is titled and scoped to v0.11.0
`docs/KNOWN_ISSUES.md` line 1 = "# Known Issues — v0.11.0"; its changelog is v0.10.0→v0.11.0.
The README "Known issues (v0.11.0 → carried into v0.12.0)" table still lists **KI-6 (RRTMG
taug) as open with "fix → v0.12.0"** — but v0.12.0 IS the release being tagged, so either
KI-6 was fixed in this push (it was not — no taug fix in the commit log) and the label is
stale, or it is correctly still-open and the "fix → v0.12.0" wording is now self-referential
and wrong. Also the wind-divergence finding from the equivalence demo (U10/V10/U/V growing
to V~8 m/s by h19) is a **new, prominent, honestly-measured issue that is NOT in
KNOWN_ISSUES** (it overlaps KI-4 U10 but the demo's V-dominated column-wind drift is a
distinct and larger signal).
**Fix:** Re-title KNOWN_ISSUES to v0.12.0; correct KI-6's "fix→v0.12.0" to a real target
(or close it if fixed); add a KI for the equivalence-demo wind divergence (V column RMSE
→ ~11 m/s by h19) so it is listed, not hidden.

---

## CARRY-OVER (honest roadmap items, OK to ship with disclosure)

- **CO-1. Equivalence demo is replay-mode, not standalone native-init.** The demo's
  `init_mode = "cpu_wrf_replay"` and `docs/equivalence-demo.md` (lines 13-16) state this
  plainly (GPU borrows IC/LBC from the same CPU wrfout it compares against). This is
  honest and correctly framed as the operational replay use case — but it means the
  release's headline numerical-equivalence evidence is **not** from the new standalone
  native-init path. The standalone path is only smoke-proven (2 h finite, no RMSE). Keep
  as a documented limitation; do not let any README rewrite imply the equivalence demo
  validates the standalone path.
- **CO-2. Standalone live-nest is a 2 h finiteness smoke only.** `standalone_nest_smoke.json`
  is `PIPELINE_GREEN` d01→d02, both finite, no CPU wrfout — genuinely good and the carry_overs
  are listed honestly (one-way only, two-way OFF, w-relax OFF, no RMSE/speedup). Just ensure
  the README rewrite says "standalone live-nest proven finite for a 2 h smoke," not "validated."
  The 24 h nested RMSE numbers in the README are from the v0.11.0 **replay** nest, not standalone.
- **CO-3. Column T exceedance in the equivalence demo.** Beyond PSFC/winds, 3D `T` (θ′)
  also exceeds (pooled RMSE 2.04 K vs 1.5 K tol). The doc lists it under the wind-growth
  bucket; fine, but worth a one-line acknowledgement that θ′ also breaches, so the "T2 within
  tol" surface claim is not read as "temperature is equivalent" (surface T2 passes; 3D θ′ does not).
- **CO-4. README internal inconsistency: Grell-Freitas labeled "(ref)".** README scope-at-a-glance
  (line 113) tags **Grell-Freitas as "(ref)"** implying reference-only, but the catalog classifies
  `cu_physics=3` as **IMPLEMENTED** (scan-wired) and README line 136 lists it as GPU-operational.
  The "(ref)" annotation is wrong/misleading. Pre-existing v0.11.0 text; fix opportunistically
  during the README rewrite.
- **CO-5. Perf numbers DEFERRED (honestly).** `docs/PERFORMANCE.md` marks warm-cache hour-1,
  peak VRAM, and the CUDA-graph A/B as "deferred, not abandoned" due to the one-GPU lock behind
  a concurrent 24 h job. This is disciplined and honest (directional cache evidence + bit-identical
  cache mechanism stand). Fill in when the GPU frees; no blocker.
- **CO-6. KI-6 RRTMG taug, KI-7 free-running wide-domain, n=15 TOST.** All carried from v0.11.0,
  all documented, no over-claim. Fine to carry.

---

## CORRECTNESS-OF-EVIDENCE spot-checks (all PASS)

- **CPU fast tests:** `test_scheme_catalog_fail_closed.py` + `test_cli.py` +
  `test_namelist_check.py` = **59 passed** (`JAX_PLATFORMS=cpu`, taskset 0-3, 2.15 s).
- **Scheme catalog self-check:** `assert_catalog_consistent()` OK; counts =
  **43 implemented / 5 reference_only / 80 recognized_fail_closed / 3 out_of_scope codes
  + 14 out_of_scope feature switches** — exactly the task's stated matrix.
- **README scope table vs catalog:** MP {0,1,2,3,4,6,8,10,16}, CU {0,1,2,3,6}, PBL
  {0,1,5,7,8}, SFCLAY {0,1,5,7}, LSM {0,2,4}, RA {0,4} all match `_IMPLEMENTED`. No
  scheme is listed "implemented" in README that is not scan-wired in the catalog.
- **CLI flags vs quickstart:** `cli.py` defines `--input-dir/--output-dir/--domain/--hours/
  --scratch-dir/--max-dom`; README quickstart + Run sections use exactly these. Auto-detect
  (replay vs native-init) and `--max-dom>1` → nested driver are real in `_cmd_run`.
- **Equivalence demo methodology:** tolerances are hard-coded predeclared constants in
  `scripts/equivalence_demo.py` (FIELD_TOLERANCES, L113-123) matching the doc; verdict is
  data-driven (PASS iff pooled RMSE ≤ tol); framing explicitly disclaims bitwise + self-compare.
  Honest. Doc observed-result table matches the JSON pooled values exactly.
- **PSFC fix is real and wired** (operational_mode + wrfout_writer); proof shows bias
  328→−29 Pa. (The only problem is FN-2: the eqdemo proof was not re-run on it.)
- **Proof JSONs cited exist and say what the docs claim** (psfc, standalone smoke, nest smoke).

No fabricated/self-compare/happy-path evidence found. The dishonesty risk is entirely in
the **stale README/KNOWN_ISSUES (v0.11.0 text on a v0.12.0 tag)** and the **un-rerun PSFC
equivalence proof**, not in invented numbers.
