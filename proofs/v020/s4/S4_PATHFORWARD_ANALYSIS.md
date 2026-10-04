# S4 fp32 — Math faithfulness, AceCast comparison, extension path

Independent analysis supporting `S4_FP32_ULTRACODE_REPORT.md` (STOP-discipline homework: prove fp32
is not a dead-end and chart the maximal faithful win). Inferences flagged [INFER]; recall [RECALL];
in-repo facts carry file:line.

## 1. Perturbation-authoritative fp32 is mathematically sound

Regime (b) stores p'/ph'/mu'/w fp32, rebuilds totals as `fp64_base + fp32_pert`
(`operational_mode.py:988-990`), base carried fp64 (`:968-978`), re-quantized 3×/timestep
(`:2020-2028`) but **never inside the acoustic substep scan** (`:2177-2190`).

Decisive quantity = base:perturbation ratio. fp32 absolute floor = ε₃₂·|value| (ε₃₂≈1.2e-7):
- Naïve fp32-**total** p (~1e5 Pa): floor ≈ 1.2e-2 Pa; forming p'=p−pb cancels 2–3 sig digits →
  ~1e-4 relative error on p' (~1e2 Pa). Catastrophic for the acoustic pressure-gradient coupling.
- Perturbation-authoritative fp32 p' (~1e2–1e3 Pa): floor ≈ 1e-5–1e-4 Pa, ~2–3 orders smaller; the
  total is fp64_base + fp32_pert so the large-number cancellation is **structurally avoided**.

This is WRF-ARW's *own* design reason for carrying p'/φ'/μ' as prognostics [RECALL]. **Verdict:**
perturbation-authoritative storage recovers ~2–3 digits a naïve fp32-total loses. Empirically
confirmed by the 1 h tolerance pass (report §3): wind diff ~0.001 m/s, PSFC ~0.07 Pa.

**Where it still loses (and the fix):** the six `force_fp64_island` brackets (EOS, both PGFs, vertical
implicit w/φ solve) are fp64-internal (`precision.py:21-49`) — CLOSED. But two tendency accumulators
are *unguarded*: `ph_tend` in `rhs_ph_wrf` (`rhs_ph.py:240..378`) and the coupled-work primes in
`small_step_prep` (`small_step_prep.py:273-277`). They accumulate the fp32 prime each stage. **Fix:**
force those *transient* accumulators to fp64 (same pattern as the islands) — zero resident-VRAM cost,
closes most of the residual. (No compensated summation exists on the acoustic path; the only Kahan sum
is the hydrostatic column integral `operational_mode.py:1169-1180` — extending it is optional once the
accumulators are fp64.)

## 2. AceCast / GPU-WRF comparison (STOP-discipline item C)

- **Stock WRF is single precision by default** (RWORDSIZE=4; `-r8` is the opt-in double flag). The CPU
  reference this project validates against is *itself* single precision for most arrays — so
  fp64_default is MORE precise than reference WRF, and fp32 on u/v/θ/qv/moisture moves *toward* WRF's
  own arithmetic.
- **AceCast (TempoQuest)** validates to *statistical* equivalence within tolerance, not bitwise.
- **ECMWF IFS** moved to single precision: ~40 % faster, no skill loss (Vana et al. MWR 2017) — the
  canonical proof operational NWP tolerates single precision.
- **GPU-RRTMG** literature: radiation accuracy unaffected; gas-optics path commonly kept double.
- **What's kept fp64 in practice** [INFER, no published AceCast dtype table]: radiative gas-optics
  band sums + conservation accumulators; bulk dynamics/microphysics run single (= WRF default).

**Our case is not fundamentally different from AceCast's success — it is more conservative** (explicit
`force_fp64_island` brackets AceCast runs in single). No physics/math reason fp32 fails here.

## 3. Extension path to maximal faithful win (ranked win/risk)

| Step | Action | Win | Risk |
|---|---|---|---|
| 1 | Harden regime (b)'s two accumulators to fp64; enable matrix's already-gated u/v/θ/qv | unlocks bulk of **35 %** state | LOW (u/v/θ/qv = WRF default single; islands protect) |
| 2 | Per-field 1 km finiteness gate for moisture/number species | rest of 35 % | MED (qke already went non-finite fp32→promoted fp64, `precision.py:233-257`; gate each) |
| 3 | **RRTMG transient — the real 1 km lever:** (a) tile harder (env `GPUWRF_RRTMG_*_COLUMN_TILE_COLS`, default 2048, zero code); (b) LW cloud-optics→fp32 keeping gas band-sum fp64 (SW two-stream is the in-repo template, ~7e-5 drift) | **direct 1 km capability** (~½ LW peak) | LOW (tiling) / MED (LW fp32 = prior OOM field; gate on heating-rate + finiteness) |
| 4 | Perturbation-fp32 on boundary leaves (w_bdy/p_bdy/ph_bdy/mu_bdy) | small extra state | MED (boundary cadence) |

**End state:** resident-state win capped ~41 % (dominated by the matrix, regime (b) a faithful ~6 %
add-on); the headline 1 km-capability prize lives in the RRTMG transient and is a separable,
higher-leverage attack. Sources: WRF Users Guide (RWORDSIZE/`-r8`); AceCast V&V docs; ECMWF single-
precision IFS (Vana 2017); GPU-RRTMG (GMD). Full reasoning + citations in the dispatch transcript.
