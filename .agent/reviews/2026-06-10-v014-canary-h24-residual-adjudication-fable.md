# V0.14 Canary h24 Residual Adjudication (Fable high)

Date: 2026-06-10 ~22:57 UTC
Run: `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z`
Run head: `5c2422ac` (contains LBC-cadence `53770411`, moist-cqw `7c819067`,
NoahMP activation `c2310c5b`, LU16 fix `22a2cc0c` — verified via merge-base).
Compare: `canary_d02_h24_intermediate_grid_compare.json` (24 paired leads, 100
numeric fields, CPU-only analysis). Run still in flight (d02 at ~h31 at review
time, on pace).

## 1. Verdict

**`PROCEED_BOUNDED_WITH_FOLLOWUP`** — let the 72h run finish; do not interrupt.
The comparator `FAIL` is driven entirely by 2 static-field hard failures
(`MUB`, `PB`) whose error lives **only in the 5-cell LBC boundary frame**
(interior RMSE ~3e-4 Pa, i.e. bit-noise; boundary max 250 Pa at one east-corner
column, constant in time). All **10 hard manifest fields pass** at h24
(`PSFC, QVAPOR, RAINNC, T, T2, U, U10, V, V10, W`), every compared field has
`finite_pair_fraction = 1.0`, and every large dynamic residual is
**diurnal-phased and recovering by lead 24**, not monotonic: PSFC/P RMSE peaks
at lead 18 (102/81 Pa) and falls to 38/36 by lead 24; LH/HFX/SW/TSK peak at
leads 19–21 and fall 2–3× by lead 24. Nothing here is an instability or a new
bug class; the two open signals (boundary-frame base-state seam; daytime land
surface-flux residual) are bounded, known classes that need a follow-up task
and an explicit tolerance/root-cause decision before tag — not a stop of the
running validation.

## 2. Evidence table

| Field/class | h24 signal | Likely cause class | Release impact | Recommended action |
|---|---|---|---|---|
| `MUB`/`PB` (static, only hard FAILs) | Interior RMSE 2.9e-4/3.7e-4, max 0.0078 Pa (exact). Boundary 5-cell frame RMSE 20.5/10.0, max 250.7/249.9 Pa at one column `[57,156]`/`[0,57,156]` (east edge). RMSE constant 9.28 every lead; GPU not time-invariant flag is jitter-level. | Live-nest d02 base-state construction in the LBC specified/relax zone differs from CPU-WRF's own d02 base state. Deterministic representation seam, not drift. | Blocks comparator GREEN verdict; does not threaten stability or interior parity. Must be root-caused or explicitly carved out (no silent tolerance change). | Follow-up task (§5). Do not touch the running run. |
| `QNRAIN` | RMSE 14.6 but `p99_abs = 0`; max 3122 at one cell `[4,24,42]` (CPU 3122, GPU 0, lead 20, land >1000 m). Episodic (spikes leads 11–12, 18–21, back to 0.24 by lead 24). | Rain number-concentration in displaced/missing individual showers; chaotic cell placement. `RAINNC` (mass) RMSE 0.049 mm, max 2.0 mm — precip mass is close. | None: report-only field, p99 exactly 0 = isolated cells. | Keep report-only. No action. |
| `LH`/`HFX`/`PBLH` | Diurnal: LH/HFX RMSE ≤16/10 through lead 13, peak 164/115 at lead 20, fall to 69/47 by lead 24. Ocean nearly exact (HFX ocean RMSE 7.7). Land daytime large: LH land bias +139 (worst 300–1000 m slope band +193), HFX land bias +80. PBLH bias is ocean-side (−66 m marine, land +14), plateaus at lead 15 then declines. | Known daytime land surface/PBL residual class (MYNN/RRTMG/NoahMP Step-1 lane: RRTMG field-dominant, MYNN worst-cell; SW cloud placement feeds slope fluxes). The accepted h1–h4 land gate only sampled evening hours; midday is where this class lives. | Not manifest-gated. Bounded and night-recovering at h24. Real signal for v0.15 surface/radiation lane; watch day-over-day growth at h72. | Carry as bounded known class; add the day-2/3 peak-growth check to h72 (§4). v0.15 lane unless h72 shows growth. |
| `SWDOWN`/`SWNORM`/`SWDNB` | Exactly 0 RMSE at all night leads 3–12 (clear-sky/radiation path honest). Daytime RMSE 34–93, max 947–951, bias +6; worst over land >1000 m (bias +106 — cloud over high terrain). | Cloud position/timing displacement (microphysics+PBL chaos), not a radiation transfer bug — nighttime exactness and prior RRTMG oracle bounds exclude a flux-code error of this size. | Not manifest-gated. Drives the correlated LH/HFX/TSK daytime residual. | Same as above: monitor day-over-day peak amplitude at h72. |
| `PSFC`/`P`/`MU`/`PH` | Diurnal, not monotonic: PSFC RMSE 52→11 (lead 6) →102 (lead 18) →38 (lead 24); bias +51 spin-up → −102 midday → −36. P parallels (peak 81). MU peak 75 at lead 18 → 35. PH peak 93 (lead 6) settling ~61; high-terrain bias −72. NOT boundary-localized (interior≈boundary). | Known ~O(100 Pa) midday surface-pressure/vapor-mass residual class (previously seen as "GPU surface p vapor-light ~ −210 Pa"; now ~ −102 peak post moist-cqw). | PSFC passes its 120 Pa manifest limit (overall 50.0). P/PH/MU are critical-report-only by accepted manifest. Recovering each night. | Watch lead-42/-66 midday peaks vs lead-18 at h72. No action now. |
| Core `T/QVAPOR/U/V/U10/V10/T2/TSK` | All PASS: T 0.52/1.5, T2 0.67/1.5, U 0.75/1.8, V 0.60/1.8, U10 0.80/1.5, V10 0.91/1.5, W 0.035/0.3, QVAPOR 0.000968/0.001. TSK ungated, RMSE 0.69, max 18.3 (one midday high-terrain cell), peak 1.46 lead 21 → 0.75 lead 24. U10/V10 saturate ~0.95/0.9 after lead 13. All finite. | Normal mesoscale divergence + the daytime classes above. | GREEN. **One at-risk field: QVAPOR** — per-lead RMSE rises monotonically to 0.00129 at lead 24 (growth decelerating, ~1.75e-5/h late). The 72h *overall* RMSE may end marginally above the 0.001 limit. | If QVAPOR fails at h72 by tail-saturation only (decelerating slope, all other fields green), treat as tolerance-calibration decision, not stability — needs explicit manager decision + recorded review per the no-silent-tolerance-change rule. |

## 3. Can Switzerland GPU start if h72 stays finite with no new worse signal?

**YES** — conditional on the h72 checklist below coming back green/bounded:
h72 all-finite, the 10 hard manifest fields still pass (or QVAPOR fails only
marginally by saturation with a recorded decision), day-2/3 diurnal peaks not
materially growing, and MUB/PB failure still confined to the boundary frame.
The MUB/PB seam is a live-nest d02 class; Switzerland is single-domain d01
driven by its own `wrfbdy`, so it does not inherit this seam mechanism —
but its comparator output should still be checked for an analogous
specified-zone base-state diff. The boundary-seam follow-up task can run in
parallel with the Switzerland GPU campaign (it is CPU-only analysis first).

## 4. Exact h72 checks for the manager

In `canary_d02_h72_grid_compare.{json,md}`:

1. **Finiteness**: `finite_pair_fraction = 1.0` for every compared field, all 72 leads.
2. **Manifest**: all 10 hard fields pass overall. Specifically `QVAPOR` overall
   RMSE vs 0.001 (h24 overall 0.000968, lead-24 already 0.00129 — expect
   marginal). Check its per-lead slope keeps decelerating (saturation), and
   `V10` (h24 0.91/1.5) stays under limit.
3. **Diurnal peak growth** (the real stability question): compare
   - `PSFC`/`P` RMSE at leads 18 vs 42 vs 66;
   - `LH`/`HFX`/`SWDOWN`/`TSK` peak RMSE at leads 19–21 vs 43–45 vs 67–69.
   Bounded = day-3 peak ≲ 1.3× day-1 peak. Monotonic day-over-day amplification
   ⇒ escalate before Switzerland.
4. **MUB/PB**: interior_excluding_5cell_frame RMSE still ~1e-4-class; max still
   in the boundary frame; RMSE still lead-constant (~9.28/4.52).
5. **PBLH** ocean bias plateaued (not past ~ −90 m); **PH** high-terrain bias
   not growing past ~ −100.
6. **QNRAIN** p99 still 0; **RAINNC** RMSE within 1.0 (h24: 0.049).
7. **Resources**: `resources/*gpu_usage.csv` VRAM plateau flat (no creep
   vs the ~21 GB h4-gate peak), process RSS stable, run `rc=0` with 73 d02 frames.

## 5. Follow-up fix task (whole-task prompt outline)

Single Fable/Mythos task, CPU-only first, after h72 lands (parallel to
Switzerland GPU is fine):

> **Title**: v0.14 MUB/PB boundary-frame base-state seam — root-cause and close.
> **Inputs**: h24+h72 compare JSONs in the 20260610T214731Z run root; CPU truth
> `wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`; GPU wrfout in
> run-root `gpu_output/`; nested d02 base-state/LBC construction code
> (`d02_replay`/interp/LBC specified-zone path).
> **Facts to honor**: interior exact to ~3e-4 Pa; failure only in the 5-cell
> frame; max 250 Pa at column `[57,156]`; constant in time; CPU MUB/PB
> time-invariant, GPU flagged not-invariant (quantify — likely jitter or
> LBC-update recompute).
> **Task**: (a) reproduce the frame-only diff with a CPU probe
> (`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`) reading both wrfout sets;
> (b) identify where the GPU writes specified-zone MUB/PB (own real-init base
> state vs d01-interpolated vs blended) and why the corner column lands 250 Pa
> off; (c) either fix the writer/base-state to match CPU-WRF's d02 base state
> in the frame (preferred if local), or produce an ADR-style tolerance
> carve-out for static-field boundary frames with manager decision +
> independent review (the no-silent-tolerance-change rule applies).
> **Proof gate**: rerun the grid comparator on existing h72 outputs (no new GPU
> run needed if fix is writer-side; short h4 GPU rerun if model-side) →
> `MUB`/`PB` tolerance PASS, or recorded carve-out; comparator verdict GREEN.
> **Do not** touch `surface_layer.py`, the daytime LH/HFX lane, or tolerance
> values for dynamic fields in the same task.

Secondary (v0.15 unless h72 shows day-over-day growth): daytime land LH/HFX/SW
residual lane — already-mapped MYNN/RRTMG/NoahMP Step-1 class; needs its own
sprint with the midday-hours land gate that the h1–h4 gate did not sample.

---
FABLE CANARY_H24_RESIDUAL_ADJUDICATION DONE - see .agent/reviews/2026-06-10-v014-canary-h24-residual-adjudication-fable.md
