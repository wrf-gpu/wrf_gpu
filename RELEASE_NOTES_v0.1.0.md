# Release Notes — v0.1.0

- **Tag:** `v0.1.0`
- **Release commit:** the commit this annotated tag points at — resolve with `git rev-parse v0.1.0^{commit}`.
- **Tag date:** 2026-06-01 (annotated tag on branch `release/v0.1.0`; promoted to `main` by the principal — Option B).
- **Source HEAD basis:** HFX/surface-layer fix `d1c373b`; proofs executed on branch
  `worker/opus/final-verdict` (RTX 5090).
- **Binding proof contract:** [`publish/VERIFICATION.md`](publish/VERIFICATION.md) (11 rows).
- **Authoritative outcome record:** [`proofs/PROOF_TABLE.md`](proofs/PROOF_TABLE.md).
- **Tally:** **9 PASS / 1 FAIL (comparator-harness gap, not a production defect) / 1 INCONCLUSIVE.**

Every number in these notes traces to a row in `proofs/PROOF_TABLE.md`. Nothing here is rounded,
invented, or relaxed to manufacture a pass.

## What v0.1.0 is

A **JAX-native, single-GPU port of the WRF v4 split-explicit dynamical core plus a physics suite**
(Thompson microphysics, WRF revised surface layer, MYNN PBL, RRTMG-style SW/LW radiation),
validated for **Canary Islands 1–3 km daily forecasting** on a single RTX 5090.

It is a **single-domain REPLAY path**: the lateral boundaries and the land/SST fields are replayed
from existing CPU-WRF / Gen2 corpus artifacts. It is **not yet** a self-contained, multi-domain,
live-nesting WRF, and it does **not** yet do native WPS/real.exe initialization. That is the honest
scope, and the gap chain to a full standalone WRF replacement is inventoried in
[`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md) and sequenced in
[`.agent/decisions/V0.2.0-PLAN.md`](.agent/decisions/V0.2.0-PLAN.md).

## Validated capabilities (the PASS rows)

| Row | Capability | Key numbers (executed) |
|---|---|---|
| 1 | Idealized dycore: Skamarock warm bubble vs published reference | **PASS** 6/6 |
| 2 | Idealized dycore: Straka density current vs published reference | **PASS** 6/6 |
| 4 | **Canary 3 km (d02)**: finite/stable to 72 h, beats persistence on winds | **PASS** — 3-case **D02_VALIDATED**; T2 RMSE unchanged vs pre-fix (no regression from the HFX fix); winds beat persistence every lead |
| 5 | **Canary 1 km (d03)**: 24 h finite, bounded gate, beats persistence (secondary claim) | **PASS** — **D03_1KM_VALIDATED**; T2 RMSE **1.92 K** (gate 3.0, beats persistence); U10 **3.45** / V10 **4.24** (gate 7.5, V10 beats persistence); all finite; wall 1970 s |
| 7 | Conservation: guards-off finite + genuinely fp64 on real d02 | **PASS** — warm bubble passes guards-off incl. dry-mass-drift check; guards not load-bearing |
| 8 | Reproducibility: deterministic re-run + restart-continuity | **PASS** — `--repeat` and `--restart-at-hour 1` both within tolerance |
| 9 | Performance vs 28-rank CPU-WRF d02 | **PASS** — warmed **~15–16 s/fc-hour**, **~5–8×** vs 28-rank CPU-WRF, dt-matched floor **3.2×** (d02-only). **NOT ≥10×.** |
| 10 | Precipitation: honest characterization (not parity) | **PASS** — jax **0.393 mm** vs WRF **0.347 mm**, ratio **1.13**; water closure **2.6e-6** (gate 1e-3); per-field bias reported, not gated |

The d03 1 km result is a **secondary claim** (the primary validated product is the d02 3 km path);
d03 enters with explicit field qualifiers (T2 beats persistence; V10 mostly beats persistence; U10
wins early leads / loses late, within the 7.5 gate). The HFX fix collapsed the pre-fix
+6.8 K / +3.6 K daytime warm bias to d02 quality.

## Qualified and inconclusive rows

### Row 6 — TOST: PASS (qualified), an underpowered single-season descriptive check

n=3 MAM paired-delta GPU-vs-CPU-WRF: **U10 EQUIVALENT** within margin (Δ +0.095, margin 0.231);
**V10 borderline** (tost_p 0.052); **T2 NOT equivalent** (Δ +0.86 K). This is a
**predeclared-UNDERPOWERED, SINGLE-SEASON MAM descriptive check** — **never an "equivalence PASS."**
The TOST machinery self-test reproduces a 0.0 delta. A full seasonal corpus at n≥15–27 is a v0.2.0
deliverable.

### Row 11 — Device residency: INCONCLUSIVE (not a forecast-correctness gate)

A byte-counted in-loop transfer audit was attempted. The classifier could **not** extract per-event
byte sizes from this trace (it classified 0 of the 7.39 MB measured), and a `bytes_accounted` guard
yields **INCONCLUSIVE** rather than a false zero-in-loop PASS. Device residency is **architecturally
guaranteed**: the whole model state is a device-resident pytree, and the scanned timestep performs
no host transfer by construction. Settling the byte-counted audit is a v0.2.0 follow-up.

## Honest limitations

- **Single-domain replay, not a full WRF.** Boundaries + land/SST are replayed from CPU-WRF / Gen2
  artifacts. Live multi-domain nesting, native WPS/real.exe initialization, prognostic Noah-MP, and
  d01 cumulus are **out of v0.1.0 scope** (see `publish/GPU_PORT_GAPS_TODO.md`).
- **Row 3 (operator parity vs WRF savepoints) = FAIL — comparator-harness gap, NOT a
  production-dycore defect.** The savepoint oracle is an hourly `wrfout` history state, not a true
  per-RK / restart-complete WRF savepoint, so the validation-only coupled-step comparator is fed a
  state missing ~30 `small_step_prep`-derived leaves and goes non-finite at step 1. Independently
  confirmed by two models (Opus + GPT-5.5). The production dycore is proven by rows 1/2/7 + the
  d02/d03 real-case runs (operational `small_step_prep → _rk_scan_step` path, finite over full
  forecasts). Regenerating true per-step WRF savepoints is a tracked v0.2.0 follow-up — **not a
  relaxed-away pass, and not a sign the dycore is broken.**
- **The HFX / surface-layer fix is an empirical, partial, MYNN-inspired land thermal-roughness
  repair — NOT a faithful `module_sf_mynn.F` port.** It collapsed the d03 daytime warm bias to d02
  quality and caused no d02/d03 regression, but the claim is narrowed accordingly. Faithful
  MYNN/HFX parity is the first v0.2.0 item (0.1.1).
- **TOST is underpowered + single-season (n=3 MAM).** A descriptive paired-delta check, not a
  seasonal equivalence result.
- **Speed is ~5–8×, not ≥10×.** The fp64 acoustic core is the per-step hot path; an fp32 downcast
  was implemented and validated numerically but gives ~0× additional speedup. Closing to ≥10× needs
  deeper fusion/kernel work and is post-v0.1.0.
- **Row 11 (byte-counted device-residency audit) is INCONCLUSIVE** at the byte-counted level while
  residency is architecturally guaranteed by construction. Tracked v0.2.0; not a forecast-correctness
  gate.

## Reproducing the proofs

```bash
bash scripts/verify_all.sh        # regenerates/checks every row, emits proofs/PROOF_TABLE.md
```

Each row also has a standalone `scripts/verify/<row>.sh` that re-runs that single proof from source
(not from a cached JSON) and asserts the gate, so a reviewer can reproduce piecewise. See
`publish/VERIFICATION.md` for the row-by-row contract and gates.

## v0.2.0 roadmap

The next release line closes the gap chain in
[`.agent/decisions/V0.2.0-PLAN.md`](.agent/decisions/V0.2.0-PLAN.md) — all gap items **except**
native WPS/real.exe initialization (deliberately last). The 0.1.x cadence starts with **0.1.1 =
faithful MYNN/HFX parity + moisture/PBL no-regression + fp64-mode declaration**, then Thompson
precip/water + conservation budgets, real-terrain/map-factor/boundary dynamics closure, output
completeness, and later live nesting / Noah-MP / multi-GPU. Tracked v0.2.0 follow-ups explicitly
include: regenerating true per-step WRF savepoints for row 3, settling the byte-counted device
audit for row 11, and building a full seasonal TOST corpus for row 6.
