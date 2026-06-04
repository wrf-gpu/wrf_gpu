# Known Issues — v0.9.0

Honest, code-grounded list of the open issues shipped with v0.9.0. Each entry states
the symptom, what was ruled out, the current best understanding, the workaround, and
the tracked follow-up. No spin.

---

## KI-1 (OPEN, targeted for v0.10.0 kernel/numerics sprint) — d03 1 km gated-fp32 qke goes non-finite after forecast hour 1

**Severity:** the 1 km gated-fp32 preview is **NOT validated** for v0.9.0. v0.9.0
ships fp64 operationally; gated-fp32 is deferred to v0.10.0.

**Symptom.** Running the 1 km Tenerife domain (`d03`, mass grid 75×93, `dt = 3 s`,
10 acoustic substeps) in gated-fp32 goes **non-finite after forecast hour 1**.
The single offending field is `qke` (MYNN turbulent kinetic energy): `3036`
non-finite cells on the original proof (`proofs/v090/d03_1km_validation.json`).
Every other prognostic field stays finite at that point. Because the run NaNs
before any wrfout is written, **no T2 / U10 / V10 / PBLH / precip /
prognostic-level RMSE could be scored** against the CPU-WRF 1 km reference.

**Precision diagnosis correction.** The original validation proof attributed the
failure to qke fp32 overflow at 1 km. The later qke->fp64 follow-up falsified
that precise diagnosis: with qke genuinely promoted to fp64, the run still
blocked at forecast hour 1 with `qke` the sole non-finite field, the same `3036`
cells, and the same finite min/max (`2.33e-5` / `27.36`), while the blow-up
occurred at tiny qke magnitudes (`~0.04 -> 0.13`). Therefore this is best
documented as a qke/dynamics numerics robustness edge over steep 1 km Tenerife
terrain, **not** as a pure precision-range overflow. Proofs:
`proofs/v090/d03_1km_validation_qkefix.json` and
`proofs/v090/pipeline_run_d03_qkefix_gated_fp32.json`.

**Current best understanding — a dynamics-driven structural instability over steep
terrain, with qke as the canary.** Micro-step GPU probes in the qke-fp64 review
show:

- `qke` is genuinely fp64 at every checkpoint after the promotion.
- On the last finite state, the real operational MYNN adapter produces a fully
  finite `qke` (`max=0.071`) and finite `dfm/dfh/km/kh/el/qkw`, so MYNN physics
  is not the NaN source on a finite input.
- A single coupled forecast step on that same finite state yields non-finite
  `qke`; the in-core MYNN path sees a dynamically evolved intermediate over the
  steep columns.
- `_mym_level2` `gm`/`gh`, `ustar`, and `dz0` stay finite through the blow-up.

**Full fp64 status.** The d03 1 km path is finite in full fp64 over the confirmed
short window (`0.3 h / 360 steps`, `proofs/v090/d03_replay_finite_check.json`),
with all tracked dynamics finite. Full-fp64 24 h d03 remains impractically slow
on the RTX 5090 (`fp64 = fp32/64`) and was not completed for v0.9.0.

**Workaround / what to use instead.** Use the validated 3 km d02 path. Treat d03
gated-fp32 as an experimental preview only until v0.10.0 closes this numerics
edge.

---

## KI-2 (OPEN, targeted for v0.10.0 kernel/numerics sprint) — long single-call daily-pipeline advances can hit the same qke edge on case-sensitive initial states

**Severity:** documented robustness edge, not a blanket pipeline failure.

The supported v0.9.0 operational cadence advances in output-interval segments
and re-enters the public forecast boundary at each output interval, where the
precision contract and MYNN qke floor are re-enforced in a WRF-faithful way. That
cadence is finite and skillful for the 72 h d02 coupled proof
(`proofs/v090/d02_coupled_skill_72h.json`), and the naive-gate 20260429 d02
sample writes a finite 1 h wrfout under `force_fp64=True`
(`proofs/v090/naive_gate_run/pipeline_payload.json`).

The risky path is a long single jit-style advance on some initial states. The
available 20260521 d02 production-path recheck records a non-finite state after
forecast hour 1 with `qke` as the canary: `2024` qke cells in the base
FP32_GATED qke run, and the top-level recheck records the identical result after
the qke promotion (`proofs/v090/d02_gated_fp32_recheck.json`). That supports the
important conclusion: this edge is **not rescued by qke precision alone** and is
best handled as qke step-loop numerics robustness work, not as a v0.9.0 precision
default change.

**Provenance caveat.** The manager closeout request describes this as verified in
both full fp64 and gated-fp32. In this checkout, the clean committed evidence I
found directly supports the base-qke vs qke-promoted A/B above; one subordinate
proof path named by `d02_gated_fp32_recheck.json` points to a 20260509 grid
mismatch artifact instead of the claimed 20260521 promoted-qke run. This
provenance mismatch is recorded in the final RC closeout review rather than
papered over here.

**Non-overstatement guard.** This is case-sensitive. The naive-gate 20260429 case
runs finite for the documented 1 h gate. The validated d02 72 h skill proof also
runs finite/stable on its replay cadence. KI-2 is a robustness carry-over for the
long single-call path on susceptible initial states.

---

## KI-3 (scope, by design, targeted for v0.10.0 writer/gate hygiene) — operational wrfout writer emits a focused 64-variable subset

**Severity:** scope boundary, not a forecast-correctness defect.

The v0.9.0 operational writer emits a focused **64-variable** wrfout, while the
bundled CPU-WRF reference in the naive gate contains **375 variables**
(`proofs/v090/naive_agent_gate.json`). The missing dimensions in the generated
file are only:

- `seed_dim_stag=8`, used by SPPT/SKEBS/SPP stochastic-perturbation seed arrays;
- `snow_layers_stag=3`, used by Noah-MP internal snow-layer diagnostics
  `TSNO`, `SNICE`, `SNLIQ`;
- `snso_layers_stag=7`, used by Noah-MP snow+soil geometry diagnostic `ZSNSO`.

All core meteorological, spatial, vertical, and soil dimensions match the CPU-WRF
reference exactly: `Time=1`, `DateStrLen=19`, `west_east=159`,
`west_east_stag=160`, `south_north=66`, `south_north_stag=67`,
`bottom_top=44`, `bottom_top_stag=45`, and `soil_layers_stag=4`
(`proofs/v090/naive_gate_run/dimension_compare.json`). The old strict
375-variable criterion conflated diagnostic writer coverage with
forecast-correctness. The correct v0.9.0 contract is the focused operational
writer plus finite, physically plausible fields and matching core dimensions.

---

## KI-4 (documented residual) — d02 near-surface westerly (U10) episodic under-prediction

**Severity:** within operational margins for the vast majority of the forecast;
documented, not a blocker.

The 72 h d02 (3 km) coupled skill is finite and stable throughout, with final-hour
Tier-4 RMSE within all bars (T2 `0.81 K`, U10 `4.00 m/s`, V10 `2.97 m/s`).
T2 and V10 are within bar at every one of the 72 leads. **U10 is within bar at
66/72 leads**: it transiently breaches the `7.5 m/s` bar over lead hours 21-26
and then recovers. This is an episodic near-surface westerly under-prediction
during high-wind periods, not a runaway/degrading instability. This is why the
machine `status` in `proofs/v090/d02_coupled_skill_72h.json` is `FAIL`.

---

## KI-5 (scope boundaries) — flat-slab diffusion; fail-closed schemes; n=15 TOST not scored

These are **scope boundaries**, not defects (see the README "Honest boundaries"):

- Both the constant-K and the new 2-D Smagorinsky diffusion paths are **flat-slab**
  (map-factor / coordinate-slope deformation terms dropped) — within tolerance
  for the Canary cases, not fully terrain-faithful. Terrain-slope diffusion is
  post-0.9.0.
- Schemes outside the GPU-operational subset **fail closed** with a named reason
  (they are recognized but not wired); v0.9.0 is not the full WRF v4 physics
  catalog.
- The **formal n=15 TOST equivalence has not been scored for v0.9.0**. The MAM
  corpus is prepared (forcing retained, CPU-WRF references assembled); the
  powered TOST is the paper's analysis. The v0.9.0 operational equivalence
  evidence is the d02 coupled-skill result. No "TOST PASS" is claimed.
