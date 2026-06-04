# Known Issues — v0.9.0

Honest, code-grounded list of the open issues shipped with v0.9.0. Each entry states
the symptom, what was ruled out, the current best understanding, the workaround, and
the tracked follow-up. No spin.

---

## KI-1 (OPEN, carried over) — d03 1 km gated-fp32 goes non-finite after forecast hour 1

**Severity:** ships as a documented OPEN ISSUE. The 1 km mandatory gate is **NOT met** in
the gated-fp32 ship precision. 3 km d02 is unaffected (stable + validated in gated-fp32).

**Symptom.** Running the 1 km Tenerife domain (`d03`, mass grid 75×93, `dt = 3 s`,
10 acoustic substeps) in the gated-fp32 operational ship mode goes **non-finite after
forecast hour 1**. The single offending field is `qke` (MYNN turbulent kinetic energy):
~3036 non-finite cells, localized to ~67–69 specific columns × 44 levels over the
**steepest Tenerife terrain gradients**. Every other prognostic field stays finite at
that point. Because the run NaNs before any wrfout is written, **no T2 / U10 / V10 /
PBLH / precip / prognostic-level RMSE could be scored** against the CPU-WRF 1 km
reference.

**This is NOT a precision-range problem (hypothesis falsified).** The validation-burst
root-cause guess was "qke overflows fp32 at 1 km." We promoted `qke` to fp64 in the
precision matrix (`src/gpuwrf/contracts/precision.py`; this change ships in v0.9.0 as a
harmless precision improvement) and re-ran. The d03 1 km forecast still NaNs at hour 1
with the **identical signature** — same ~3036 cells, same finite min/max
(2.33e-5 / 27.36) to the digit — and the divergence happens at **tiny qke magnitudes
(~0.04 – 0.13)**, nowhere near any fp32 *or* fp64 range limit. fp64 changed nothing.
The premise is falsified.

**Current best understanding — a dynamics-driven structural instability over steep
terrain, with qke as the canary.** Micro-step GPU probes show:

- `qke` is genuinely fp64 at every checkpoint after the promotion.
- On the last finite state, the real operational MYNN adapter produces a **fully finite**
  `qke` (max 0.071) and finite `dfm/dfh/km/kh/el/qkw` — so the MYNN physics is **not** the
  NaN source on a finite input.
- Yet a single *coupled* forecast step (dynamics acoustic/RK + boundary + MYNN-in-core)
  on that same finite state yields non-finite `qke`. The MYNN-in-core runs on the
  **dynamically-evolved intermediate** state, so the **dynamics** produces an unstable
  near-surface intermediate (extreme shear/θ over the 67–69 steep columns at `dt = 3 s`)
  that drives the in-core qke budget to NaN.
- `_mym_level2` `gm`/`gh`, `ustar`, `dz0` all stay finite through the blow-up — it is not
  a divide-by-tiny-`dz` in the level-2 gradients.

So `qke` is the most sensitive field that first goes non-finite; the underlying cause is a
coupled-step dynamics instability over steep 1 km terrain, **not** a precision contract,
**not** a clamp/masking issue, and **not** a MYNN-physics bug.

**Full fp64 is finite but impractical.** The 1 km domain runs **finite in full fp64** over
the confirmed window (0.3 h / 360 steps; proof
`proofs/v090/d03_replay_finite_check.json`). But fp64 is ~1:64-throttled on the RTX 5090
(GB202), so a full 24 h fp64 d03 validation (~hours of wall) cannot complete in a normal
sprint window — fp64 is precisely the slow path the gated-fp32 ship mode exists to replace.
Therefore **neither practical mode closes the 24 h 1 km gate today**.

**Workaround / what to use instead.** Use the **3 km d02** gated-fp32 path, which is stable
and validated to 72 h (see the README d02 coupled-skill result). The 1 km path is usable
only in full fp64 and only for short windows.

**Tracked follow-up — a real numerics/stability sprint (NOT a precision-contract sprint).**
Concrete leads (from `proofs/v090/d03_1km_validation_qkefix.json:carry_over`):

1. Audit whether the gated-fp32 dynamics (fp32 `u/v/θ` with fp64 `mu/p/ph`) produces an
   unstable near-surface intermediate over the 67–69 steep columns at `dt = 3 s` /
   10 acoustic substeps.
2. Try a 1 km-appropriate `dt` / more acoustic substeps / stronger near-surface
   vertical-implicit treatment.
3. Diff the gated-fp32 vs full-fp64 dynamics intermediate at the *same* step to isolate
   which fp32 dynamics field first diverges over those columns.
4. Cross-reference the ~69-column localization against the steepest d03 terrain gradients.

**Proof objects.**
- `proofs/v090/d03_1km_validation.json` — first gated-fp32 NaN, root-cause localization.
- `proofs/v090/d03_1km_validation_qkefix.json` — qke→fp64 premise falsified (identical
  signature), structural/dynamics conclusion, carry-over leads.
- `proofs/v090/d03_replay_finite_check.json` — full-fp64 finite over 0.3 h / 360 steps.
- `.agent/reviews/2026-06-04-opus-qke-fp64-fix.md`,
  `.agent/reviews/2026-06-04-opus-v090-validation-burst.md` — narrative.

---

## KI-2 (documented residual) — d02 near-surface westerly (U10) episodic under-prediction

**Severity:** within operational margins for the vast majority of the forecast; documented,
not a blocker.

The 72 h d02 (3 km) gated-fp32 coupled skill is finite and stable throughout, with
final-hour Tier-4 RMSE within all bars (T2 0.81 K, U10 4.00 m/s, V10 2.97 m/s). T2 and V10
are within bar at every one of the 72 leads. **U10 is within bar at 66/72 leads**: it
transiently breaches the 7.5 m/s bar (max 8.04 m/s) during the diurnal peak-wind window
(~h22–30) and then **recovers** to 3–4 m/s by 72 h. This is an *episodic near-surface
westerly under-prediction* during high-wind periods, consistent with the pre-existing
documented near-surface U-momentum bias (`proofs/f7/DYCORE_STATUS.md`, v0.4.0 carry-over);
T2/V10/HFX/PBLH are unaffected and it is **not** a runaway/degrading instability. This is
why the machine `status` in `proofs/v090/d02_coupled_skill_72h.json` is `FAIL` — the
all-leads-within-bar predicate trips on those 6 U10 leads only.

---

## KI-3 (scope, by design) — flat-slab diffusion; fail-closed schemes; n=15 TOST not scored

These are **scope boundaries**, not defects (see the README "Honest boundaries"):

- Both the constant-K and the new 2-D Smagorinsky diffusion paths are **flat-slab**
  (map-factor / coordinate-slope deformation terms dropped) — within tolerance for the
  Canary cases, not fully terrain-faithful. Terrain-slope diffusion is post-0.9.0.
- Schemes outside the GPU-operational subset **fail closed** with a named reason (they are
  recognized but not wired); v0.9.0 is not the full WRF v4 physics catalog.
- The **formal n=15 TOST equivalence has not been scored for v0.9.0**. The MAM corpus is
  prepared (forcing retained, CPU-WRF references assembled); the powered TOST is the paper's
  analysis. The v0.9.0 operational equivalence evidence is the d02 coupled-skill result. No
  "TOST PASS" is claimed.
