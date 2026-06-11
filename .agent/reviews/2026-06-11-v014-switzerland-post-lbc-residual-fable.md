# V0.14 Switzerland Post-LBC Residual — Root Cause (Fable)

Date: 2026-06-11
Worker: Fable high, branch `worker/fable/v014-switzerland-post-lbc-residual`
Sprint: `.agent/sprints/2026-06-11-v014-switzerland-post-lbc-residual/sprint-contract.md`

## Verdict

**Root cause accepted** (acceptance-gate option 3): the post-LBC-clock-fix
Switzerland d01 72h FAIL is a **locally generated, regime-locked, domain-wide
divergence bias of the GPU solution in strong cross-Alpine flow** (the
2023-01-16 12Z – 2023-01-17 storm window, CPU |U|max 39 m/s at dx=3000 m).
The GPU's own winds vent dry mass through the (correctly forced) lateral
boundary at a steady ~30–50 Pa/cell/h; the interior subsides, warms
(+0.5–1 K/h low-mid troposphere), evaporates the widespread cloud deck,
shuts down precipitation, and by h72 the run sits in a wrong synoptic state:
domain-mean MU bias −1053 Pa, MU pattern correlation collapsed to r≈0,
T RMSE 7.1 K.

The driver is the **dry dynamics lane** — a 3 h `mp_physics=0` re-init probe
reproduces the venting unchanged (G6), exonerating microphysics/latent
heating entirely. Prime named suspect: the operational path's
`top_lid=True` rigid-lid top boundary condition (a documented workaround for
an open-top first-step instability), which reflects vertically propagating
mountain-wave energy that CPU-WRF radiates through its free p_top surface.

This is **not** a second LBC clock bug, **not** an interior mass-conservation
bug, **not** a writer/diagnostic issue, **not** the land refresh, **not**
guards/limiters, **not** chaotic amplification of accumulated drift, and the
microphysics number-concentration bursts are downstream symptoms, not the
driver. Proof chain below; proof object
`proofs/v014/switzerland_post_lbc_residual.{py,json}`.

## Evidence chain (all falsifications explicit)

### 1. The boundary forcing itself is correct (second-LBC-bug class falsified)

- `MU` ring0 matches same-hour CPU truth **bit-exact** at every probed lead
  (manager pre-check + re-verified at h36/h44/h60/h72: ring max abs 0.0).
- `U/V/T/QVAPOR/PH/W` ring0 all match at output times (max abs ≤ 1e-4).
- The relax-zone (rings 1–4) profile is the expected Davies shape: diffs ramp
  from 0 at ring0 to interior level at ring 4–5.
- `_rewindow_boundary_leaves` window logic re-checked against
  `interpolate_boundary_leaf`: with hourly records and hourly segments the
  in-segment forcing reproduces global-time linear interpolation exactly
  (also covered by `tests/test_daily_boundary_clock.py`).
- Namelist damping/diffusion parity confirmed: the daily pipeline hardcodes
  `epssm=0.5, w_damping=1, damp_opt=3, zdamp=5000, dampcoef=0.2,
  diff_6th_opt=2/0.12` (daily_pipeline.py:482–488), which **matches** this
  case's CPU `namelist.input` exactly.

### 2. Dry mass is conserved; the deficit is vented by the GPU's own winds (G1)

Column-integrated dry-mass budget with hybrid C1H/C2H coupling and map
factors, control surface at depth 8, trapezoid-in-time over hourly snapshots
(Pa/cell/h):

| window | CPU dM | CPU net-influx | CPU resid | GPU dM | GPU net-influx | GPU resid |
|---|---:|---:|---:|---:|---:|---:|
| h24→25 | −109.4 | −108.0 | −1.4 | −85.1 | −81.0 | −4.2 |
| h40→41 | −34.4 | −34.3 | −0.1 | −76.7 | −72.4 | −4.4 |
| h60→61 | −52.5 | −46.9 | −5.6 | −70.9 | −65.2 | −5.7 |
| h66→67 | +16.2 | +21.8 | −5.7 | −87.2 | −81.9 | −5.3 |
| h71→72 | +2.6 | +1.1 | +1.5 | −50.3 | −46.4 | −3.9 |

Both runs close to within ±8 Pa/cell/h (sampling error). **The GPU does not
destroy mass internally** — after h36 it genuinely keeps exporting 50–90
Pa/h through the boundary while CPU's net export shuts off. (A naive budget
without map factors mis-closes by ~200 Pa/h — the proof script uses the
correct hybrid/map-factor form.)

### 3. The divergence is locally generated, not accumulated drift (G2 — decisive)

A GPU run **re-initialized bit-true from CPU truth at h36**
(`/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable`,
truncated wrfout history so IC = `wrfout_d01_2023-01-16_12:00:00`, correct
solar clock via `run_start_label`) re-develops the venting immediately, **as
fast or faster than the full run** over the same valid window:

| valid h | probe MU rmse/bias | full-run MU rmse/bias |
|---|---:|---:|
| 37 (lead 1) | 63.6 / −54.4 | 67.1 / −37.6 |
| 40 (lead 4) | 219.7 / −187.3 | 169.1 / −124.9 |
| 42 (lead 6) | 321.4 / −278.3 | 260.6 / −207.8 |
| 44 (lead 8) | 381.8 / −337.0 | 315.0 / −262.9 |
| 48 (lead 12) | 332.5 / −299.6 | 261.4 / −225.3 |

Baseline: the calm-start full run drifted only 28–35 Pa MU RMSE over its
first 6 leads. Same model, same code, ~10× faster divergence when the IC sits
in the storm regime ⇒ **a systematic forcing error active in this regime**,
not chaos, not 36 h of accumulated physics drift.

### 4. The excess outflux is interior-wide, not a boundary-zone artifact (G3)

Probe-vs-CPU excess net outflux at lead 1 (h36→37) is the **same ~28–31
Pa/cell/h through control surfaces at depth 2, 8, and 20 cells** (depth 20 =
60 km inside a 384 km domain). The whole GPU wind field is slightly more
divergent than truth in this regime; the relax zone is innocent.

### 5. Thermal/moisture signature: subsidence, day and night (G4)

- Probe lead-1 (ONE hour from truth IC): dTheta ≈ +0.5…+1.4 K domain-wide at
  k≈6, +0.4–0.5 K through k0–20. The equivalent heating (~840 W/m² over that
  layer depth) is impossible for any physics flux — only **adiabatic descent**
  (≈1–2 cm/s net) can warm that deep that fast, and it is exactly the
  compensation of the measured lateral mass export.
- The full run's warm dome peaks at +14 K (k≈20) by h72 with PH bias +940 —
  hydrostatically consistent with a venting warm low.
- Probe hour 1 also annihilates the widespread condensate (domain-mean QCLOUD
  −84% vs CPU h37; QGRAUP −96%; peak values survive) → SWDOWN jumps +102 W/m²
  (GPU 277 vs CPU 175), GLW −32 W/m². **But the venting continues at full
  ~45 Pa/h after sunset (SWDOWN→0 from valid h41)**, so the radiation
  transparency is a *consequence* (cloud kill by subsidence drying), not the
  mass driver.
- Surface fluxes falsified as driver: probe lead-1 HFX is *lower* than CPU
  (−21 W/m² mean) while the air warms.

### 6. Guards/limiters and land refresh falsified

- The production theta increment limiter uses the wide [0, 500] K envelope
  (operational_mode.py:1058–1100, non-load-bearing by design); storm theta
  never approaches it. The mass guard only traps nonfinite/μ<1.
- Land refresh re-snaps TSK/SMOIS *from CPU truth* hourly — a damping,
  truth-ward forcing; it cannot create a warm low. (TSK exact in wrfout is
  the expected refresh artifact.)
- Writer-only class falsified: MU/PSFC/PH/T are prognostic and move together
  hydrostatically; the budget closes on the written fields.

### 7. Microphysics findings (real parity gaps, but downstream of the dynamics here)

- **QNICE ≡ 0 for the first ~5 h** while CPU nucleates 2×10⁶/kg within h1.
  Root cause measured: primary deposition nucleation needs `ssati ≥ 0.25`;
  on the shared IC neither run is supersaturated (max ssati 0.061); CPU's
  dynamics builds ssati to 0.43 by h1, the GPU only to 0.23–0.24 (its upper
  troposphere runs +0.15–0.2 K warm from the first hour — the known step-1
  radiation-dominant floor; at 210 K, ~0.5 K ≈ 12% saturation shift). The
  nucleation *code* is WRF-equivalent (gate, Cooper xnc, cap — verified
  against module_mp_thompson.F:2620–2630).
- **QNRAIN sparse bursts to 4.65×10⁷/kg** (CPU ceiling ~6×10³) sit in the
  melting band (271–278 K, k0–7). They do **not** violate the WRF mvd clamp
  (the band ceiling at the burst cell's qr is ~5×10⁷/kg, and `_finish`
  implements WRF 4040–4055 correctly); they come from the melt/shed number
  sources in a degraded state. Two genuine review flags for a follow-up
  sprint, neither load-bearing here:
  - `thompson_column.py:839`: `pnr_sml = smo0/rs * snow_melt * rho * 10**(−0.25(twet−T0))`
    carries an extra `*rho` relative to the WRF per-kg bookkeeping
    (module_mp_thompson.F:2793 + 3065 `*orho`) — O(rho)≈1 at the melting
    layer, so a ~±30% number bias, not the 10⁴ burst.
  - GPU rains over 2.5× more cells than CPU early (h12: 47k vs 18k rain
    cells) with lower peaks — a drizzle-spread bias that softens orographic
    precip maxima (RAINNC max 15.4 vs CPU 23.4 already at h36).

### 8. Why Canary passed and Switzerland fails

Canary d02 (subtropical, weak synoptics) never enters the regime: its
PSFC/P diurnal peaks *decay* day-over-day (102→17 Pa). Switzerland d01 is a
3-km Alpine domain hit by a 39 m/s cross-mountain jet with heavy orographic
snowfall — the regime where the GPU dycore's strong-flow solution bias
becomes a sustained, one-signed mass forcing. The verdicts are consistent:
same model floor, different regime gain.

## Named suspects for the term-level attribution

The constant day/night venting and the depth-independence bound the driver to
the **dynamics solution in strong cross-mountain flow**. Ranked suspects:

1. **`top_lid=True` (rigid lid) vs WRF's constant-pressure free top.** The
   operational real-case path zeroes w at the model top
   (daily_pipeline.py:461–472, a documented workaround for the open-top
   ~300 m/s first-step instability on real init,
   proofs/dycore_realinit/step4/5). CPU-WRF radiates vertically propagating
   mountain-wave energy through its material p_top surface; a rigid lid
   **reflects** it. In a 39 m/s cross-Alpine jet at dx=3 km this changes the
   wave-drag/divergence field domain-wide — mechanical, day/night-invariant,
   and regime-locked exactly like the measured signal. In calm regimes
   (Canary, first 36 h here) the lid is benign — matching the bounded
   verdicts everywhere else.
2. The known-open **one-RK-step acoustic/mass-lane residual** ("P~975 broad",
   2026-06-09 Mythos kernel-fix memory).
3. PBL momentum drag (MYNN) over steep terrain in extreme wind — exonerated
   as a heat source, not yet as a momentum contributor.

A 3 h genuine `mp_physics=0` re-init probe (G6, `run_nomp_driver.py`,
`gpu_output_nomp2`) splits dry-dynamics vs moist-coupling:

- venting persists without microphysics ⇒ dry dycore strong-flow bias
  (lid/acoustic/PGF/advection lane);
- venting collapses ⇒ moist coupling (saturation adjustment/latent placement
  or condensate-dynamics coupling) is required for the loss.

**RESULT (G6, decisive): the venting persists UNCHANGED without
microphysics.** With `mp_physics=0` from the same h36 truth IC, the MU bias
at valid h37/38/39 is −56.9/−102.1/−141.2 Pa vs −54.4/−99.4/−139.6 Pa with
Thompson — bit-for-bit-equal at the venting-magnitude level (slightly
*stronger* dry). Microphysics, its latent heating, and the
condensate/precipitation anomalies are **fully exonerated as drivers**; they
are downstream symptoms of the dry-dynamics divergence. The attribution
sprint goes to the **dry dycore strong-flow lane**: top boundary condition
(`top_lid=True` rigid lid, prime suspect), acoustic loop / w-ph solve, PGF,
advection over steep terrain.

## Side finding: daily path ignores case-namelist physics options

The first mp=0 attempt (editing `namelist.input` in the input dir) produced a
**bit-identical** run: `daily_pipeline._build_real_case` builds
`OperationalNamelist.from_grid(...)` without threading
`mp_physics/bl_pbl_physics/sf_sfclay_physics/cu_physics/ra_*` from the case
namelist — the replay daily path always runs the default suite
(Thompson/MYNN/MYNN-sfclay; only topo_shading/slope_rad/shadlen/gwd_opt are
read). For this case the defaults coincide with the CPU namelist, so there is
no parity break *here*, but any replay case with different physics would be
silently mis-dispatched despite the v0.12.0 "fail-closed scheme catalog"
policy. Recommend a follow-up: thread the physics family options from the
case namelist through `_build_real_case` (or fail loudly on mismatch).

## Smallest next fix sprint

1. **Bind the term**: with the G6 outcome choosing the lane, the cheapest
   decisive test is a **1 h h36 re-init A/B on the top boundary condition**
   (top_lid + damp-layer variants; if the open-top first-step instability
   blocks `top_lid=False`, fix/absorb that first — it is itself a WRF-parity
   gap), followed by the existing one-RK-step / substage comparator harness
   on the h36 storm state (PGF, acoustic loop, w/ph solve, advection) if the
   lid is exonerated. Acceptance: reproduce the ~30 Pa/h/cell
   domain-integrated divergence excess in ONE hour from the h36 IC and
   attribute ≥70% of it to a named term.
2. Re-run the 72h Switzerland gate only after that term-level fix; expect the
   h37+ venting class to collapse to the Canary-like bounded floor.
3. Independent (parallel-safe) cleanups now justified: `pnr_sml` rho
   bookkeeping check vs WRF, and the known dt parity note (GPU dt=10 s
   ceiling-limited vs CPU 18 s — smaller dt, not a suspect, but should be in
   the release notes).

## Objective / files / commands / proofs / risks (handoff)

- **Objective**: root-cause the post-LBC-clock Switzerland d01 72h residual
  end-to-end. Done: root cause accepted at mechanism level with the term-level
  attribution scoped as the smallest next sprint.
- **Files changed**: `proofs/v014/switzerland_post_lbc_residual.py` (proof
  generator), `proofs/v014/switzerland_post_lbc_residual.json` (proof object),
  this report. No model-code changes (none justified yet — the fix needs the
  term-level attribution first; changing tolerances was out of scope and was
  not done).
- **Commands run**: CPU probes over the run artifacts (netCDF budget/field
  analysis); three short GPU probes via `scripts/run_gpu_lowprio.sh`:
  (1) 12 h h36 re-init (`gpu_output`), (2) namelist-edit mp=0 attempt
  (`gpu_output_nomp`, ran Thompson bit-identically → exposed the dispatch
  gap), (3) genuine mp=0 via `run_nomp_driver.py` patched case_builder
  (`gpu_output_nomp2`); then the proof script
  `proofs/v014/switzerland_post_lbc_residual.py`.
- **Proof objects**: `proofs/v014/switzerland_post_lbc_residual.{py,json}`;
  probe artifacts under
  `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/`.
- **Unresolved risks**: (a) the named dycore/moist-coupling term is not yet
  isolated — the one-RK-step comparator sprint is required; (b) PBL momentum
  (MYNN drag in 39 m/s flow over 3-km Alps) is exonerated as a heat source
  but not fully excluded as a *momentum* contributor to the divergence bias;
  (c) the h36 probe shares the hourly-segment driver with the full run, so a
  per-segment-restart shock contribution cannot be fully separated from the
  regime-locked signal with these probes alone (the calm-regime baseline
  bounds it to ≤ ~12 Pa/h).
- **Next decision needed**: approve the one-RK-step attribution sprint
  (CPU-first, h36 storm state) as the follow-up.
