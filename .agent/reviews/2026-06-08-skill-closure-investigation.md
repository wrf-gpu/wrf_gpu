# Forecast-skill-closure investigation (v0.13 Tier-1 #7 PREP)

**Author:** Opus 4.8 (1M), worker. **Mode:** READ-ONLY diagnosis (`JAX_PLATFORMS=cpu`, no GPU context, no `src/`/`proofs/` edits). **Owns:** only this doc.
**Base inspected:** `worker/opus/v0120-integration` tip `21bc3ce` (the current checkout `worker/opus/v013-community-validation` is byte-identical for `runtime/operational_mode.py`, `dynamics/flux_advection.py`, `coupling/boundary_apply.py` — verified via `git diff`).
**Ground truth used (not aspirational docs):** the source itself + the v0.12.0 `docs/KNOWN_ISSUES.md` (KI-9 equivalence-demo numbers), the `proofs/v0110/wind_regression*` ablation suite, and the prior RCAs (`2026-06-01-opus-d02-t2bias-diagnosis.md`, `2026-06-01-gpt-t2bias-energy-budget.md`).

---

## TL;DR for the manager

1. **Moisture transport IS a real correctness gap, confirmed at the code level.** The operational dycore advects **only** `u/v/w/theta` (and a coupled `mu`/`ph`/`p`). Water vapour `qv` is updated **only** by physics (microphysics + PBL vertical mixing) and by the **lateral-boundary ring** (and only `qv` there — `qc/qr/qi/qs/qg` have **no** boundary leaf either). The condensate species `qc/qr/qi/qs/qg` receive **zero resolved-wind advection anywhere** in the interior. WRF, by contrast, flux-advects every moisture species every RK3 step (`solve_em.F:2282-2408 moist_variable_loop`). **The fix is a clean one-call hookup** (`advect_moisture_scalars`, already merged in `flux_advection.py:749`) into the existing flux-advection branch.

2. **The current dominant skill gap is NOT T2 and NOT moisture — it is WIND error growth.** On the v0.12.0 equivalence demo (KI-9) **T2 already PASSES (0.484 K vs 1.5 K bar) and QVAPOR already PASSES (5.67e-4 vs 1.0e-3)**; the `NOT_EQUIVALENT` verdict is driven by **U10/V10/U/V** drifting monotonically with lead (V reaches ~11 m/s by h19). The d02 coupled-skill proof shows the breach is concentrated at the **final lead** (U10 final-lead RMSE 8.06 m/s, bias **-7.32 m/s** westerly under-prediction = KI-4); U10 still beats persistence on 23/24 leads. The old "+2.9 K midday T2 / 3.7× HFX over-flux" framing in the task brief is **superseded** — post-P1-4a the HFX ratio is ~1.6× (not 3.7×) and the residual T2 warm bias was traced to a **pressure/Exner artifact + lower-tropo PBL under-mixing**, not the surface-layer code, and on v0.12.0 the surface-pressure fix + current config already bring T2 inside bar.

3. **Highest-leverage first fix = wire `advect_moisture_scalars` into the operational RK3 path** with `moist_adv_opt=2` (monotonic), CPU-oracle-validatable, low risk (additive, default byte-identical), and it is the one item that is unambiguously a *correctness* gap rather than chaotic error growth. It is unlikely to move T2/U10/V10 RMSE much on the small boundary-dominated demo cases, but it is **mandatory for credibility** ("we advect moisture") and is the prerequisite for honest precipitation/QVAPOR skill on larger/longer domains. The wind divergence (item 2) is the larger *skill* number but is **not** a single-line fix and largely needs GPU forecast iteration.

---

## 1. Operational moisture transport — REAL GAP (confirmed, exact hookup specified)

### What the operational path actually does (traced end-to-end)

`runtime/operational_mode.py`:

- `_rk_scan_step` (`:1940`) → per RK stage calls `compute_advection_tendencies` (`:1964`) then `_augment_large_step_tendencies` (`:1965`).
- `compute_advection_tendencies` (`dynamics/advection.py:262`) **does compute** a `qv` advection tendency (`:272 qv=base.qv + advect_mass_scalar(...)`), plus `theta` and `p`. **But this `tendencies.qv` is dead code in the operational path** (see below).
- `_augment_large_step_tendencies` (`:1665`): when `use_flux_advection=True` (the validated operational config) it **replaces** `u_t/v_t/w_t/th_t` with WRF flux-form advection (`advect_u_flux`/`advect_v_flux`/`advect_w_flux`/`advect_scalar_flux[_limited]`, `:1727-1789`). It then returns `tendencies.replace(u=u_t, v=v_t, w=w_t, theta=th_t)` (`:1926`) — **`tendencies.qv` is passed through untouched** and never recomputed in flux form.
- The acoustic core consumes only `theta_tend, mu_tend, ph_tend, u_tend, v_tend` and folds `w` (`_acoustic_substep_setup` `:1332-1378`). **`AcousticCoreState` (`dynamics/core/acoustic.py:104-145`) has no `qv` leaf and no scalar/moisture tendency leaf.** So `tendencies.qv` is never read.
- `small_step_finish_wrf` (`dynamics/core/small_step_finish.py:22`) returns `state.replace(u=…, v=…, w=…, theta=…, mu=…, ph=…, p=…)` — **`qv` (and every condensate) is NOT in the replace list**, so moisture passes through the entire dycore unchanged.
- `add_scaled_tendencies` (which *would* do `qv += dt*tendencies.qv`, `tendencies.py:16`) is **imported but never called** in `operational_mode.py` (only referenced in comments at `:1651/:1895/:1959`; the legacy forward-Euler was deliberately removed).

**Where moisture DOES change operationally** (`_physics_boundary_step_with_limiter_diagnostics` `:2704`):
- Physics: `thompson_adapter`/MP scan adapters (microphysics source/sink), MYNN/PBL vertical mixing (`_physics_step_forcing` `:2496`), applied post-dycore via `_apply_physics_non_dry_updates` (`:2481`, increments `qv/qc/qr/qi/qs/qg/...`).
- Lateral boundary: `apply_lateral_boundaries` (`coupling/boundary_apply.py:148`) writes **`qv` only**, **in the boundary strip only** (`:176`, `qv = max(_apply_3d(state.qv, state.qv_bdy, …), 0)`; docstring `:159` "writes u,v,w,theta,qv … in the boundary strip only. The interior beyond relax_zone is untouched"). There is **no** `qc_bdy/qr_bdy/...` leaf in `State` (only `qv_bdy` exists — `contracts/state.py:92/508`).

### Is moisture correctly advected by the resolved wind field? **NO.**

- **`qv`**: advected only at the boundary ring (via the time-interpolated parent `qv_bdy`). The **interior `qv` field is NOT transported by the resolved (u,v,w)** — it only diffuses/mixes vertically (PBL) and gets MP source/sink terms. Horizontal and resolved-vertical advection of the interior vapour field is **missing**.
- **`qc/qr/qi/qs/qg`**: **zero advection anywhere** — not in the dycore, not at the boundary. Hydrometeors are purely local to each column (MP + sedimentation only). This means cloud/rain water cannot be horizontally transported by the wind — physically wrong (e.g. precipitation falls/forms where it was generated, never advected downstream).

This matches the explicit honest note in `proofs/pd_monotonic/pd_monotonic_advection_proof.md:156-159`: *"The limiter is wired for THETA … Moisture species (qv/qc/...) ride the physics path, not this dry flux-form theta path; extending the operational limiter to moisture transport is a separate step."* It is a deferred step, **not** a WRF-faithful design choice — WRF advects all moisture species (`solve_em.F:2282-2408`, referenced in `flux_advection.py:729-746`).

### Exact one-call hookup

`advect_moisture_scalars(fields, fields_old, vel, *, moist_adv_opt, is_final_rk_stage, mut, mu_old, c1, c2, rdx, rdy, rdzw, fzm, fzp, dt)` already exists (`flux_advection.py:749`, exported `:1159`) and returns a tuple of **coupled** tendencies `d(mu*q)/dt`, the same contract as `advect_scalar_flux`. The hookup goes inside the `if bool(namelist.use_flux_advection):` block of `_augment_large_step_tendencies` (right after the theta flux-advection at `:1786-1789`), reusing the **already-built** `vel` (`couple_velocities_periodic`, `:1702`), `mu_total`, and `metrics`:

```python
# (proposed; OWNED by the dynamics/runtime lane, NOT this doc)
moist_fields = (haloed.qv, haloed.qc, haloed.qr, haloed.qi, haloed.qs, haloed.qg)
moist_old    = (step_origin.qv, step_origin.qc, ...) if use_limiter else None
q_tends = advect_moisture_scalars(
    moist_fields, moist_old, vel,
    moist_adv_opt=int(namelist.moist_adv_opt),         # NEW namelist field, default 0
    is_final_rk_stage=(int(rk_step) == int(namelist.rk_order)),
    mut=mu_total, mu_old=step_origin.mu_total,
    c1=metrics.c1h, c2=metrics.c2h, rdx=1/dx, rdy=1/dy,
    rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp, dt=float(namelist.dt_s),
)
```

The hard part is **plumbing the resulting coupled moisture tendency into a state update**, because the acoustic core does not carry moisture. WRF advances moisture in the **large step**, decoupled by `mu` after the acoustic loop (`solve_em.F` `rk_scalar_tend` → `q_new = (mu_old*q_old + dt*tend)/mu_new`). Two faithful integration options for the JAX path:

- **Option A (recommended, minimal-surface):** apply the coupled moisture tendency as a large-step forward-Euler **outside** the acoustic loop, exactly like WRF's scalar update: in `_rk_scan_step` after the acoustic stage, `q_new = (mu_old*q_old + dt_rk*q_tend) / mu_new` per species (final-stage uses the limiter via `is_final_rk_stage`). This keeps the acoustic core untouched. Add a new static `OperationalNamelist.moist_adv_opt: int = 0` (threaded through `from_grid`/`tree_flatten`/`tree_unflatten`, mirroring `scalar_adv_opt`); default `0` → the new code path is gated off and the operational program is **byte-identical** until enabled.
- **Option B (more invasive):** add a moisture-tendency leaf to `AcousticCoreState` and advance it per substep — heavier, not WRF's cadence for scalars; **not recommended**.

**Carry the `step_origin` (start-of-step) moisture** for the final-stage monotonic limiter — `rk1_reference` is already threaded into `_augment_large_step_tendencies` as `step_origin` (`:1971`), so `step_origin.qv/qc/...` are already available.

### Expected skill impact

- **Modest on the current demo cases, large on the credibility/general-domain axis.** On the v0.12.0 equivalence demo `QVAPOR` already PASSES (5.67e-4 < 1.0e-3) — because the Tenerife/Canary demo domains are small and **boundary-dominated** for `qv` (the `qv_bdy` ring pins vapour near the edges, and the interior is only a few cells from the boundary), and because the corpus cases are not strongly advective-moisture regimes. So expect **little movement on T2/U10/V10/QVAPOR RMSE for the existing single-case gates**.
- **The real wins are: (i) correctness/credibility** — a reviewer who greps the dycore will (correctly) flag "moisture is never advected"; (ii) **precipitation placement** — `qc/qr/qi` advection is required for rain to fall downstream of where it forms, which matters for `RAINNC`/`RAINC` skill on larger d01/d02 domains and longer integrations; (iii) **prerequisite for honest wide-domain / multi-day moisture and the powered TOST** (KI-5). Use `moist_adv_opt=2` (monotonic) — WRF's real-case default — to keep mixing ratios bounded/positive (the PD-monotonic proof shows the limiter is faithful and mass-conserving to ~1e-12).

---

## 2. T2 / U10 / V10 bias root causes (re-grounded on the CURRENT trunk)

> Important re-framing: the task brief cites a "+2.9 K midday T2 / HFX ~3.7× over-flux" story. That is **stale** (pre-P1-4a, ~v0.1.0). The evidence chain below shows the surface-layer over-flux was largely closed and the residual T2 bias re-attributed; on v0.12.0 **T2 is within bar** and **winds** are the binding gap.

### 2a. Daytime surface-flux over-flux (the historical "3.7×") — LARGELY CLOSED, not the current lever

- **Code:** `physics/surface_layer.py` (849 lines; a literal MYNN-SL port). HFX = `flhc*(thgb-thx)`, `flhc = cpm*rhox*ustar*KARMAN/psit` (`:771/:778`); the unstable-regime `psim/psih` and the `0.9*gz?oz?` thin-layer caps are at `:631-661`; `ustar` blend at `:681`; `psit` heat resistance at `:689`.
- **Status:** GPT's energy-budget RCA (`2026-06-01-gpt-t2bias-energy-budget.md`) found *"The old 'midday HFX about 3.7× corpus' result is no longer present after p1_4a. Current land daytime HFX is about 1.57× over all daylight, 1.61× for 10–15Z."* The opus d02 diagnosis (`2026-06-01-opus-d02-t2bias-diagnosis.md:186`) found HFX is actually **LOW (negative bias)** on d02 (mostly ocean), not an over-flux. On the v0.12.0 d02 coupled-skill proof, **HFX mean RMSE 62.6 W/m² is well within the 120 W/m² bar**.
- **Likely residual cause + WRF-faithful fix:** the land-daytime ~1.6× HFX is consistent with the slightly-too-warm lower column amplifying `(thgb-thx)`, i.e. a **consequence** of the pressure/PBL items below, not a `surface_layer.py` formula bug. The MYNN-SL port itself was diagnosed as **not the lever** (`2026-06-01-opus-d02-t2bias-diagnosis.md:220` "NOT a surface_layer.py fix"). **Do not re-tune `surface_layer.py`.** If anything is residual, it is the surface-layer→PBL flux handoff (`fltv`/`theta_flux`, `:782-787`) feeding a too-shallow PBL (see 2c).

### 2b. RRTMG-SW clear-sky SWDOWN / T2 bias — minor, isolated, not skill-binding

- **Code:** `physics/rrtmg_sw.py` (`_extend_with_wrf_top_layer`, ~`:589`); coupler `coupling/physics_couplers.rrtmg_sw_theta_tendency`; surface diagnostic `rrtmg_radiation_diagnostics`.
- **Status (KI-6):** the only known SW defect is an **intermediate `taug` per-band** mismatch in 4 UV bands from the top-layer convention (duplicates the ~190 hPa top layer instead of inserting a ~100 Pa model-top layer). **Integrated fluxes already pass <0.05% / within 1 W/m²**; forecast skill unaffected. The RCAs found **SWDOWN biased LOW** in the afternoon (`-28 to -59 W/m²` over land) — so it cannot cause a *warm* T2 (it would cool).
- **WRF-faithful fix:** KI-6 Fix B — make `_extend_with_wrf_top_layer` add a low-pressure (~100 Pa) model-top layer rather than duplicating the topmost input layer; or Fix A regenerate the oracle at the current convention. **Needs GPU re-validation** of the integrated-flux savepoint gate. **Low skill leverage** — treat as fidelity/completeness, not a T2-bias fix.

### 2c. The genuine T2/theta lever: pressure-Exner artifact + lower-tropo PBL under-mixing (DYCORE + PBL, not surface)

- **Pressure-Exner artifact:** a near-uniform **+2.x kPa** perturbation-pressure inflation (mid-column-depressed perturbation geopotential `ph'` → more-negative `al` → EOS inflates `p`), present on **both** d02 (`force_geopotential=True`) and d03. It alone explained ~+1.85–2.4 K of the historical T2 warm bias. **v0.12.0 partly addressed the diagnostic side**: the WRF-faithful `PSFC = p8w(kts)` surface-extrapolation fix (KNOWN_ISSUES "PSFC diagnostic offset CLOSED", `proofs/v0120/psfc_extrapolation_proof.json`, bias 328→−29 Pa; pooled PSFC RMSE 707.8→415.3 Pa) closed the *diagnostic* offset feeding T2's Exner. The **dynamical** `ph'` equilibration residual remains (now expresses as the lead-growing PSFC excess that *tracks the wind divergence* — KI-9). **Code:** `dynamics/acoustic_wrf.diagnose_pressure_al_alt` (faithful EOS — the error is in the equilibrated `ph'` that feeds `al`, i.e. the acoustic `w`–`ph` small-step coupling, `core/rhs_ph.py` + `core/advance_w.py`). **WRF-faithful fix direction:** fold the nested `ph'` boundary forcing INTO the acoustic small step coupled with `w` (the `force_geopotential=False` branch comment at `boundary_apply.py:186-217` explains why an end-of-step `ph'` overwrite detonates the `w`–`ph` resonance). This is a **dycore** sprint, GPU-iteration-bound.
- **Lower-tropo PBL under-mixing:** d02 (93% ocean) runs the lowest few hundred metres **+3–5 K too warm under a too-shallow PBL** (sea PBLH −116 to −136 m) with ~zero net surface-flux bias — a classic under-ventilated/too-stable MYNN-PBL signature (`2026-06-01-opus-d02-t2bias-diagnosis.md:182-203`). **Code:** `physics/mynn_pbl.py` (mixing length / TKE / entrainment over weakly-unstable sea columns). The recommended next instrument was a **MYNN-PBL parity oracle** vs `module_bl_mynn.F` (analogous to the surface-layer oracle). **WRF-faithful fix:** audit MYNN mixing-length/entrainment + the `icloud_bl=1` MYNN-EDMF cloud PDF (named in KI-4/KI-9 as the most likely improvement path). GPU-iteration-bound; needs the PBL oracle first (CPU-validatable per-column, then GPU coupled).

### 2d. U10/V10 wind divergence (KI-4 / KI-9) — the CURRENT binding skill gap

- **Symptom:** monotonic lead-time wind error growth; d02 U10 final-lead RMSE **8.06 m/s** (bar 7.5), **bias −7.32 m/s** (westerly under-prediction at high-wind final lead); 3D `V` grows ~3× faster than `U` (RMSE 0.17 m/s @h1 → ~11 m/s @h19). U10 beats persistence 23/24 leads — it's a **final-lead episodic breach**, not a runaway.
- **What the ablation suite already settled (`proofs/v0110/wind_regression*`):**
  | variant | U10 mean RMSE | U10 final | V10 mean | T2 mean |
  |---|---:|---:|---:|---:|
  | `dry_physics_post_dynamics_v090` (**SHIPPED default**) | **4.43** | **8.06** | **3.59** | 1.11 |
  | `no_dry_momentum_tendencies` (route through rk_addtend_dry) | 5.54 | 9.33 | 4.52 | 1.16 |
  | `no_dry_physics_tendencies` | 5.12 | 8.75 | 3.74 | 1.20 |
  | `no_mynn_edmf` | 5.50 | 9.15 | 4.47 | 1.22 |
  | `no_mynn_momentum_mf` | 5.54 | 9.33 | 4.52 | 1.16 |
  | `rrtmg_topo_slope_off` | 5.54 | 9.33 | 4.52 | 1.16 |
  **Reading:** the **best winds are the already-shipped config** (dry-physics applied post-dynamics, v0.9.0 cadence). Routing the aggregate dry-physics delta through `rk_addtend_dry` as RK-stage tendencies **degrades** winds (confirmed `5e8aabe` reverted; KNOWN_ISSUES conservation note). MYNN-EDMF/RRTMG-slope toggles barely move U10. So this is **intrinsic two-integrator error growth concentrated in the wind field**, with the lowest-error knob already selected.
- **Root-cause candidates still open (none a one-liner):** (i) the `*_tendf` source-tendency adapter (proper WRF RK-stage physics-momentum cadence) — but the naive version *hurt*, so a **faithful** `R*TEN`-based adapter is needed (deferred to v0.13 per KNOWN_ISSUES); (ii) MYNN-EDMF cloud PDF `icloud_bl=1` (the KI-4/KI-9 named path); (iii) the same pressure/`ph'` dycore residual (2c) feeding a mass/wind imbalance. **All GPU-forecast-iteration-bound.** First-hour V10 spin-up was separately diagnosed (`2026-06-03-gpt-v040-v10spinup.md`) as a real MYNN first-call QKE init gap → `NEEDS-COUNCIL`; the v0.11.0 qke cold-start seed (KI-1) partially addressed it.

---

## 3. Prioritized fix-plan (ranked by expected T2/U10/V10 RMSE leverage + validation tag)

| Rank | Fix | Subsystem | Expected skill leverage | Validation | Risk | Notes |
|---|---|---|---|---|---|---|
| **1** | **Wire `advect_moisture_scalars` into operational RK3** (Option A large-step update; new static `moist_adv_opt`, default 0 = byte-identical; recommend `=2` monotonic for real cases) | dynamics/runtime | **Low on existing single-case T2/U10/V10/QVAPOR** (boundary-dominated demo), **HIGH on correctness/credibility + precip placement + wide-domain/TOST prerequisite** | **CPU-oracle-validatable** (the `advect_moisture_scalars` + PD/mono kernels already have 14+7 CPU-jax parity tests vs WRF transcription; add a coupled-mass-conservation + qv-no-regression operational test). GPU only for final coupled real-case no-regression. | **Low** (additive, default-off, byte-identical until enabled) | **THE #1 highest-leverage first fix** — the only unambiguous *correctness* gap with a clean merged-kernel hookup; closes the reviewer-bait "moisture is never advected". |
| **2** | **Pressure/`ph'` dycore equilibration** — fold nested `ph'` boundary forcing into the acoustic `w`–`ph` small step (the historical +2.x kPa Exner artifact's *dynamical* root, now the lead-growing PSFC residual) | dynamics (acoustic core) | **High on T2 (historical ~1.6–2.4 K) AND on PSFC/wind divergence** (KI-9 PSFC now tracks wind drift) | **Needs GPU forecast iteration** (acoustic w–ph stability is the failure mode; CPU idealized gates for guardrails, but the real-case Exner collapse needs GPU). The CPU "Exner transplant knockout" (GPT proposed) can *quantify the ceiling* without a code change. | **High** (touches the acoustic core; prior `ph'`-overwrite attempts detonated `w`) | Largest *physical* T2 lever but architecturally hard; the diagnostic-side `PSFC=p8w` fix already banked the cheap half. |
| **3** | **MYNN-PBL lower-tropo mixing / `icloud_bl=1` EDMF cloud PDF** — the too-shallow ocean PBL + the KI-4/KI-9-named wind path | physics (mynn_pbl) | **Medium-High on the d02 theta residual + the wind final-lead breach** | **CPU-oracle-validatable per-column FIRST** (build a MYNN-PBL parity oracle vs `module_bl_mynn.F`, analogous to the surface-layer oracle), **then GPU coupled** for the skill delta | **Medium** | Recommended to **build the PBL oracle** (CPU) as the gating instrument before any coupled change; this is the larger of the two historical T2-bias halves and a leading wind candidate. |
| **4** | **Faithful `*_tendf` (R*TEN) source-tendency adapter** for RK-stage physics momentum (replace the post-dynamics aggregate-delta with WRF's per-stage source tendency — done *faithfully*, since the naive aggregate route hurt winds) | coupling/runtime | **Medium on U10/V10** (the correct WRF cadence *should* help where the aggregate delta drifts) | **Needs GPU forecast iteration** (the prior attempt `5e8aabe` was measured on GPU coupled skill and reverted) | **Medium-High** | Only attempt with the true `R*TEN` decomposition, not aggregate deltas; ablation already shows the wrong version regresses. |
| **5** | **RRTMG-SW `taug` UV-band top-layer fix (KI-6)** | physics (rrtmg_sw) | **Negligible on T2/U10/V10** (integrated fluxes already <0.05%) | **Needs GPU** (re-prove the integrated-flux savepoint gate) | **Medium** (top-layer convention change risks the flux gate) | Fidelity/completeness, **not** a skill-closure item; sequence after 1–4. |

### Single highest-leverage fix to do FIRST

**Rank 1 — wire `advect_moisture_scalars` into the operational RK3 path (Option A, `moist_adv_opt` default 0, recommend 2 for real cases).**

Rationale: it is the **only** candidate that is (a) an unambiguous *correctness* gap rather than chaotic error growth, (b) backed by an **already-merged, CPU-oracle-parity-proven kernel** (a clean one-call hookup, not a port), (c) **CPU-validatable end-to-end** (no GPU contention — respects the GWD-gate's GPU ownership), and (d) **low-risk** (additive, byte-identical until enabled). The two larger *skill-number* gaps (pressure/`ph'` dycore, MYNN-PBL/winds) are real but are GPU-iteration-bound, architecturally hard, and the wind knob is already at its lowest-error setting per the ablation suite — they belong to dedicated dycore/PBL sprints, not a quick skill-closure win.

**Honest caveat to flag to the manager:** Rank 1 will likely NOT move the headline T2/U10/V10 RMSE on the existing boundary-dominated demo cases — its value is correctness, precipitation placement, and unblocking wide-domain/TOST. If the manager's goal is specifically to move the **KI-9 `NOT_EQUIVALENT` wind verdict**, that is Rank 2–4 territory (GPU-bound), and the ablation evidence says there is **no cheap config knob left** — it needs the faithful `*_tendf` adapter and/or the dycore pressure fix and/or the MYNN-EDMF cloud PDF, each a real sprint.

---

## CPU-vs-GPU validation tags (summary)

- **CPU-oracle-validatable now (no GPU):** Rank 1 (moisture advection — kernels already CPU-parity-proven; add operational conservation/no-regression CPU tests), Rank 3's **PBL oracle build** (per-column vs `module_bl_mynn.F`), and the Rank 2 **Exner-transplant ceiling estimate** (re-Exner saved wrfout with corpus PSFC — quantifies the lever without code change).
- **Needs GPU forecast iteration:** Rank 2 (acoustic `w`–`ph` coupling — stability is the failure mode), Rank 3's coupled skill delta, Rank 4 (`*_tendf` adapter — measured on GPU coupled skill, prior attempt reverted), Rank 5 (RRTMG flux-gate re-prove).

## Files referenced (all absolute)

- `/home/enric/src/wrf_gpu2/src/gpuwrf/runtime/operational_mode.py` (RK scan, `_augment_large_step_tendencies` `:1665`, `_physics_boundary_step…` `:2704`, `_apply_physics_non_dry_updates` `:2481`)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/flux_advection.py` (`advect_moisture_scalars` `:749`, exported `:1159`; WRF `moist_variable_loop` ref `:729-746`)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/advection.py` (`compute_advection_tendencies` `:262`, the dead `qv` tendency `:272`)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/acoustic.py` (`AcousticCoreState` `:104` — no qv leaf)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/dynamics/core/small_step_finish.py` (`small_step_finish_wrf` `:22` — qv not in replace)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/coupling/boundary_apply.py` (`apply_lateral_boundaries` `:148`, qv-strip-only `:176`, force_geopotential=False `ph'` note `:186-217`)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/physics/surface_layer.py` (MYNN-SL; HFX `:778`, flhc `:771`, caps `:647-661`)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/physics/mynn_pbl.py` (PBL mixing — Rank 3)
- `/home/enric/src/wrf_gpu2/src/gpuwrf/physics/rrtmg_sw.py` (`_extend_with_wrf_top_layer` ~`:589` — KI-6)
- `/home/enric/src/wrf_gpu2/proofs/pd_monotonic/pd_monotonic_advection_proof.md` (moisture-advection deferral note `:156-159`; kernel parity)
- `/home/enric/src/wrf_gpu2/proofs/v0110/wind_regression_recovery/baseline/d02_coupled_skill.json` (+ `/home/enric/src/wrf_gpu2/proofs/v0110/wind_regression/*/d02_coupled_skill.json` ablation suite)
- `/home/enric/src/wrf_gpu2/docs/KNOWN_ISSUES.md` (KI-4 U10, KI-9 equivalence demo, PSFC fix, conservation note)
- `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-01-opus-d02-t2bias-diagnosis.md`, `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-01-gpt-t2bias-energy-budget.md`, `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-07-opus-v0120-differential-to-100pct.md`
