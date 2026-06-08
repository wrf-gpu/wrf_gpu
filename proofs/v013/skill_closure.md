# v0.13 Tier-1 #7 — Forecast-skill closure: WRF-faithful radiation `*_tendf` RK-cadence

**Author:** Opus 4.8 (1M) worker, `worker/opus/v013-skill-closure` (base `7fd92fd`).
**Mode:** CPU-only (`JAX_PLATFORMS=cpu`, cores 0-3). GPU untouched (#5 owns it).
**Owns / touched:** `src/gpuwrf/runtime/operational_mode.py` (dycore-forcing/cadence region only),
`proofs/v013/skill_closure.{py,md,json}`. No scheme files, no GPU.

---

## 1. What this sprint is and is not

The credibility gate (KI-9/KI-4) is the 24h `T2`/`U10`/`V10`-vs-CPU-WRF equivalence. The
front-loaded investigation (`.agent/reviews/2026-06-08-skill-closure-investigation.md`)
established the honest diagnosis, which I re-confirmed at the code level:

- **T2 already PASSES** (~0.484 K vs the 1.5 K bar); **QVAPOR already PASSES**.
- The `NOT_EQUIVALENT` verdict is driven by **wind error growth** (U10 final-lead RMSE
  8.06 m/s, bias −7.32 m/s; V grows ~3× faster than U).
- **The moisture-advection correctness gap (Rank 1) is already MERGED on trunk**
  (`584037d`, `moist_adv_opt`, default-off) — it is NOT this sprint's remaining work.
  The 3 GPT-flagged moisture-cadence refinements (Q1 acoustic-accumulated `ru_m/rv_m/ww_m`,
  Q3 physics-tendency folding) are part of the **same WRF-cadence-fidelity family** as the
  fix below and are GPU-bound for skill (see §6).
- The wind knob is **already at its lowest-error setting** per the v0.11 ablation suite
  (`proofs/v0110/wind_regression_debug.md`): the SHIPPED post-dycore dry-physics-delta
  cadence beats every alternative. The naive v0.11 `*_tendf` attempt **regressed** winds.

**The named dominant lever (#1) is the dycore/physics-coupling `*_tendf` cadence.** The
v0.11 attempt was correctly reverted; its root cause was precise and is the key to a
*faithful* fix:

> `proofs/v0110/wind_regression_debug.md`: the regression came from feeding the
> **aggregate** post-physics theta/`h_diabatic` state delta into `rk_addtend_dry` as if it
> were a raw WRF `R*TEN` source. Dropping the dry **momentum** tendf was **neutral**; the
> **theta** aggregate was the culprit. The residual-risk note: *"A future WRF-faithful
> raw-tendency adapter can populate `DryPhysicsTendencies`, but it must provide true
> per-source WRF tendency leaves and prove the mass/map/RK coupling."*

This sprint implements exactly that residual-risk item for the **one dry-physics theta
source that IS a genuine instantaneous WRF `R*TEN` rate** (not an implicit-solve integrated
delta): **radiation `RTHRATEN`**.

## 2. The change (default byte-identical, opt-in)

`src/gpuwrf/runtime/operational_mode.py`, +60 lines, dycore-forcing region only:

- New **static** namelist field `OperationalNamelist.rad_rk_tendf: int = 0` (threaded
  through `from_grid` / `tree_flatten` / `tree_unflatten`, mirroring `moist_adv_opt`).
- `rad_rk_tendf = 0` (DEFAULT): the v0.9 SHIPPED cadence — `theta += dt*RTHRATEN` as one
  Euler step BEFORE the dycore. **Byte-for-byte unchanged** (static gate, never traced).
- `rad_rk_tendf = 1`: route the SAME held rate through the **`t_tendf`** (mass-coupled)
  channel of `rk_addtend_dry` (`module_em.F:1770-1773`), so `RTHRATEN` is integrated by
  `advance_mu_t` at **every acoustic substep interleaved with the dynamics**, exactly as
  WRF (`module_first_rk_step_part2.F:392-394` feeds `t_tendf`). The implicit-solve
  PBL/surface/MP deltas **stay** on the post-dycore state-increment path (faithful for an
  implicit scheme; routing those aggregate deltas as sources is precisely what regressed
  v0.11 — this fix does NOT do that).

### Coupling algebra (machine-precision verified)

`rk_addtend_dry` folds `t_tendf/msfty` into `theta_tend`; `advance_mu_t` applies
`theta += msfty·dts·theta_tend` each substep (`mu_t_advance.py:195`). Supplying the
**coupled** source `t_tendf = (c1h·mut + c2h)·RTHRATEN` yields, per substep, a
coupled-theta increment `dts·(c1h·mut + c2h)·RTHRATEN` (the `msfty` cancels); after the
`small_step_finish` decouple by the column mass this is `dts·RTHRATEN` per substep →
`dt·RTHRATEN` over the full RK3 step. **Same net column heating as the lumped form; the
only difference is the within-step interleaving** — the WRF fidelity gain documented at
`physics_couplers.rrtmg_theta_tendency:1660-1665` ("the intervening dynamics/microphysics/
PBL would see a different temperature trajectory").

## 3. CPU gates (all PASS) — `proofs/v013/skill_closure.json`

| Gate | Result | Evidence |
|---|---|---|
| (1) **Default byte-identical** | PASS | `rad_rk_tendf=0` (default) vs explicit-0, 6-step forecast, ALL leaves bit-for-bit equal: **max_abs_leaf_diff = 0.0** |
| (2) **Net-heating equivalence** | PASS | lumped vs `t_tendf` deliver the same column-mean heating: **rel_diff = 6.07e-10** (< 1 ppm); within-step interleaving = 3.87e-12 max / 4.30e-13 mean (the intended cadence effect) |
| (3) **Conservation** | PASS | closed periodic box, `rad_rk_tendf=1`: **dry-mass rel drift 4.67e-16**, total-water rel drift **0.0** (radiation is a heat source, not mass/moisture) |
| (4) **Idealized unchanged** | PASS | Straka (density current) **PASS** + Skamarock (warm bubble) **PASS** through the operational dycore — radiation off there, so dry dynamics byte-unchanged → no destabilization |
| (5) **Finite / stable** | PASS | 24-step `rad_rk_tendf=1` run finite; mean warming tracks the seeded uniform heating (no runaway/spurious extrema) |

No-regression on the merged moisture wiring: `proofs/v013/moisture_advection_wiring.py`
still **5/5 PASS**. Namelist pytree + wiring tests: **44/44 PASS**
(`test_operational_namelist_cache_key`, `test_namelist_check`, `test_gwd_operational_wiring`).
Conservation budget: **2/2 PASS**.

## 4. Predicted skill delta (HONEST)

- **On the existing boundary-dominated demo cases: small.** Net heating is identical to ~1
  ppm; the cadence difference is the within-step interleaving (~6e-10 on a near-rest box).
  T2 already passes, so this is unlikely to move the headline T2 RMSE on those cases.
- **On the wind verdict (KI-9): plausibly favourable but unproven, and NOT a guaranteed
  win.** The faithful interleaving means the dynamics no longer advect a theta field that
  was already fully radiatively heated at step start (the lumped artifact). On longer/real
  advective integrations this is the documented "different temperature trajectory" and is
  the WRF-correct behaviour — but the v0.11 ablation is a standing warning that *any* theta
  cadence change can move winds in either direction. **The actual 24h U10/V10/T2-vs-CPU-WRF
  re-measure is GPU and is the manager's to run** (A/B `rad_rk_tendf` 0 vs 1 on the
  prod-failing case). My prediction: **a small, more-likely-favourable-than-not shift; it
  is NOT expected to close the 8.06 → 7.5 m/s gap on its own.**

This is deliberately a default-off, low-risk, WRF-faithful increment with a clean A/B knob —
not a skill claim. **I am not faking a skill number.**

## 5. Why this is the right CPU-provable contribution (and what hit a wall)

- Rank 1 (moisture advection): **already merged** — not remaining.
- Lever #1 momentum `*_tendf` (PBL/GWD): the dominant PBL momentum is an **implicit-solve
  integrated delta**, which cannot be faithfully expressed as a constant RK1-fixed source
  without re-deriving an instantaneous source — and the ablation showed routing momentum
  tendf was **neutral** anyway. GWD produces a clean explicit `rublten/rvblten`, but **GWD
  is gated OFF by default** (the nested-gate OOM, lever #4) so it does not move the default
  config. → I implemented the one **explicit** dry-theta source (radiation) faithfully.
- Lever #2 (MYNN `icloud_bl` cloud-PDF): EDMF is already ON by default and the ablation
  shows toggling it barely moves U10; the diagnostic SGS cloud-PDF (`cldfra_bl`/`qc_bl`)
  primarily feeds **radiation** (low skill leverage on clear-sky demo) and needs a per-
  column oracle + GPU. → deferred (see §6), correctly out of a CPU-provable scope.
- Lever #3 (moisture-cadence Q1/Q3): the same WRF-cadence-fidelity family; the acoustic-
  accumulated `ru_m/rv_m/ww_m` scalar-advection fluxes touch the acoustic core and are
  GPU-skill-bound. → deferred (see §6).

## 6. Residual / carry-over (precise)

1. **GPU A/B of `rad_rk_tendf` 0 vs 1** on the prod-failing 24h case (manager). This is the
   only way to know the wind-skill sign/magnitude. If favourable, flip the default to 1.
2. **Faithful momentum `*_tendf`** for an *explicit* scheme (GWD `rublten/rvblten` through
   `t_tendf`) — clean to implement once GWD is un-gated (lever #4 must land first).
3. **Moisture/theta acoustic-accumulated scalar-advection fluxes** (GPT Q1: `ru_m/rv_m/ww_m`
   from the acoustic substeps for scalar advection; Q3: fold PBL/cumulus moisture tendencies
   into the RK scalar tendency before the limiter). Touches the acoustic core; GPU-skill-bound.
4. **MYNN diagnostic SGS cloud-PDF** (`bl_mynn_cloudpdf`/`icloud_bl` → `cldfra_bl`/`qc_bl`):
   build a per-column oracle vs `module_bl_mynnedmf.F`, then couple to radiation; GPU-bound.

## 7. GPT cross-check — where I'd want it

The coupling algebra and the byte-identity/conservation/idealized gates are airtight on CPU.
The one judgement call worth a GPT skeptic before the manager spends GPU: **is routing ONLY
the explicit radiation rate through `t_tendf` (while keeping the implicit PBL/MP deltas
post-dycore) genuinely WRF's split** — i.e. does WRF treat `RTHRATEN` differently from the
implicit-PBL `RTHBLTEN` in `calculate_phy_tend`/`rk_addtend_dry`, or does it lump all
`RTH*TEN` into one `t_tendf`? My read of `module_first_rk_step_part2.F:392-394` +
`module_physics_addtendc.F` is that WRF accumulates all `RTH*TEN` into `t_tendf`, but our
PBL adapter returns an integrated state (not a raw `RTHBLTEN` rate), so the *faithful* choice
for our adapter contract is radiation-only-through-tendf. A GPT confirmation of that contract
distinction (and that no double-count exists with `_apply_physics_non_dry_updates`) would
de-risk the GPU A/B.

## Commands

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 python proofs/v013/skill_closure.py
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 python proofs/v013/moisture_advection_wiring.py
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 python -m pytest \
    tests/test_operational_namelist_cache_key.py tests/test_namelist_check.py \
    tests/test_gwd_operational_wiring.py tests/test_conservation_budget.py -q
```
