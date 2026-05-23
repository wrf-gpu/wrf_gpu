# M6.x Warm-Bubble Failure — Diagnostic Report

**Sprint:** `2026-05-23-m6x-warm-bubble-failure-diagnostic`
**Analyst:** Opus 4.7 (acting in tester/diagnostic role)
**Branch:** `tester/opus/m6x-warm-bubble-failure-diagnostic`
**Unification commit examined:** `e2391d3` ("Unify ADR-023 public scan path") + `d1f7d0c`.
**Probe artifact:** `probe_warm_bubble_vs_slice.json` (this folder)
**Diagnostic script:** `scripts/diagnostic_warm_bubble_vs_slice.py` (new — read-only on production code)

---

## §1. Reproduction

I rebuilt the warm-bubble setup independently (mirroring `scripts/m6_warm_bubble_test.py`) and drove the same `acoustic_substep_carry` path. Probe `a_3d_acoustic_substep_carry_baseline` confirms the failure with the unified parameters:

| step | t (s) | `jnp.max(w)` | `jnp.max(|w|)` | `theta_pert_max` | `p_pert_max` (Pa) | `mu_pert_max` (Pa) |
|----:|----:|----:|----:|----:|----:|----:|
|    0 |   0.0 | 0.0000 | 0.0000 |   1.979 |       0    |       0    |
|    1 |   0.25 | 0.0681 | 0.0681 |   1.975 | 4.4 × 10⁻¹¹ |       0    |
|    8 |   2.0 | 0.4827 | 0.4827 |   1.738 | 5.7 × 10²  |   149      |
|   80 |  20.0 | 0.6811 | 0.6811 |   4.783 | 1.1 × 10⁴  | 2.8 × 10³  |
|  800 | 200.0 | 0.3249 | 0.5843 |  40.879 | 1.3 × 10⁵  | 2.9 × 10⁴  |
| 2400 | 600.0 | **0.0409** | 0.9124 | **144.6** | 4.5 × 10⁵ | 8.8 × 10⁴ |

The harness's `jnp.max(state.w) = 0.0409` at 600 s reproduces exactly the worker's reported failure (`w_max_600s = 0.0409710985`). However the probe surfaces a previously-unreported piece of evidence: at 2400 substeps the **unsigned** w peak is 0.91 m/s (a strong downward jet — note `w_signed_max ≪ |w_min|`), `theta_pert_max = 144.6 K` (started at 1.98 K — bubble blowup by ~73×) and `p_pert_max = 4.5 × 10⁵ Pa` (≈ 4.4 atm pressure perturbation). The harness's published verdict mechanism samples `jnp.max(state.w)`, so it only sees the positive peak collapse to 0.04 and reports `FAIL_TARGETS_NOT_MET`. It does *not* see that the system has driven `theta_perturbation` and `p_perturbation` to non-physical magnitudes that would have crashed any post-substep diagnostic.

This is sustained only by the documented temporary `_mu_continuity_increment` `tanh` CFL limiter; without that bound the worker reported the run goes nonfinite at step 2.

**Confirmed:** the production path warm-bubble run fails with `w_max ≈ 0.04` at 600 s, and the underlying state has *also* diverged in θ′ and p′ — the failure is not a benign "bubble that just doesn't lift" but a slow blowup masked by the `tanh` mass limiter.

---

## §2. Slice oracle vs. warm-bubble harness — structural comparison

### Initialization paths

| Aspect | Slice oracle path (`mpas_column_slice.py`) | Warm-bubble harness (`m6_warm_bubble_test.py`) |
|---|---|---|
| θ′ profile | `delt·cos²(½π·r)` with `delt=3 K`, `radz=1500 m`, `zcent=2000 m` (`mpas_column_slice.py:67-79`) | Gaussian `2·exp(-r²/r₀²)`, `r₀=2000 m`, `zc=2000 m` (`m6_warm_bubble_test.py:115-119`) |
| Δz | 625 m (16 levels × 10 km column) | 100 m (40 levels × 4 km column) |
| dt_acoustic | 1.0 s | 0.25 s (dt=2 s ÷ 8 acoustic substeps) |
| Run length | 40 substeps (40 s) | 2400 substeps (600 s) |
| ρ_base | 1.0 (constant) | `pb/(R·θ_base)` ≈ 0.5–1.2 (vertically stratified) |
| Prognostic acoustic state | `rho_pp`, `rtheta_pp`, `rw_p` retained substep-to-substep (slice oracle lines 282-285, 339-355, 381-385) | `state.theta`, `state.w`, `state.p_perturbation`, `state.ph_perturbation`, `state.mu_perturbation` — but `state.p_perturbation` is overwritten by `diagnose_pressure_al_alt` (`acoustic_wrf.py:875-876`) — see §3, §7 |
| Horizontal coupling | none (single column) | full PGF + `mu_continuity` + smdiv |
| Mass-flux conversion | `w = rw_p / 1.35` with `ρ_base = 1` so `rw_p ≡ 1.35·w` | `rw_p = state.w · 1.35` with stratified ρ (the `1.35` constant is shared but its physical meaning differs across the two setups — see §5) |

Adjacent-layer θ′ differences differ by ~100× across the two setups: in the slice with 625 m layers the cos² profile gives `θ'[k+1]-θ'[k] ≈ 0.5 K` near the bubble's edge; in the harness with 100 m layers the Gaussian gives `≈ 0.04 K`. That difference matters for the implicit `cofwz·(ts[1:]-ts[:-1])` restoring term (§3).

### Why the slice trajectory PASSES while the harness FAILS

Probe `h_slice_oracle_vs_unified` re-runs the production-grade `test_mpas_slice_trajectory_rmse_under_production_target` test setup. The `acoustic_substep_carry`-driven trajectory MATCHES the slice oracle within ~38% peak amplitude after 40 substeps:

| step | slice oracle `max|w|` | unified `acoustic_substep_carry` `max|w|` |
|---:|---:|---:|
| 0 | 0.0000 | 0.0000 |
| 1 | 0.7431 | 0.7431 |
| 2 | 1.1801 | 1.1811 |
| 3 | 1.2763 | 1.2868 |
| 10 | 1.1748 | 1.2675 |
| 40 | 0.9774 | 1.3475 |

So the unified path matches the *toy* slice oracle's peak-then-oscillate behavior (`w` reaches ~1.3 m/s at step 3 then drifts), as the production-grade test confirms.

The reason the slice scenario looks "successful" while the warm-bubble harness "fails" is that the slice scenario only runs 40 substeps — it's stopped *at* the early peak of the gravity-wave oscillation, before the long-time decay that the 600 s harness captures (§3). The slice oracle is not a test of *sustained bubble lifting* — it tests that the recurrence reproduces the first ~1 minute of the toy oracle's gravity-wave response. Both the harness path and the slice oracle path, **if run for 600 s, decay**.

Direct evidence: probe `d_1d_shrunk_direct_recurrence` is the SAME column (single mid-bubble harness column), driven only by `_mpas_recurrence_vertical_update`, no `acoustic_substep_carry`, no mu update, no PGF. It reaches `|w|=0.475` at step 80 (20 s) and decays to `|w|=0.044` at step 2400 (600 s). The 100-m Gaussian harness column on the *pure* recurrence also fails to sustain bubble lifting — the bubble lifts briefly then the system oscillates around a near-zero `w`.

---

## §3. Buoyancy-term trace — does θ′ reach the recurrence's buoyancy RHS?

Probe `e_first_substep_rhs_decomposition` evaluates each component of the `_mpas_recurrence_vertical_update` RHS at the first substep (`dt = 0.25 s`, `epssm = 0.1`, harness initial condition with peak θ′ = 1.98 K):

| RHS term | `max|·|` over 3-D | center-column range | comment |
|---|---:|---:|---|
| `buoyancy_face[1:-1]` (= g·θ′/θ_base on faces) | 6.07 × 10⁻² m/s² | 0.023 → 0.061 | physically reasonable: g·2/300 = 0.065 |
| `dt·0.38·buoyancy_face[1:-1]` (rhs contribution) | 5.76 × 10⁻³ m/s | 0.0022 → 0.0058 | the actual term added to rhs |
| `-cofwz·((ts[1:]-ts[:-1]) + resm·(θ′[1:]-θ′[:-1]))` (implicit θ-pressure restoring) | **8.53 × 10⁻²** m/s | -0.085 → +0.085 | **≈ 15× larger than buoyancy** |
| `-cofwr·((rs[1:]+rs[:-1]) + resm·(ρpp[1:]+ρpp[:-1]))` (density coupling) | 0 | 0 | ρpp=0 at t=0 — see §7 wiring discussion |
| `cofwt·(ts + resm·θ′)` (geopotential restoring) | 6.03 × 10⁻³ m/s | 0.0023 → 0.0060 | comparable to buoyancy |

**Reading the magnitudes:**
- θ′ DOES reach the buoyancy term correctly: `buoyancy_face` reaches 0.061 m/s² at the bubble (g·θ′/θ_base for θ′ ≈ 1.85 K, θ_base = 300 K).
- The buoyancy term is in the right ballpark (not zero, not infinity, not a missing-factor 10⁰⁰).
- The DOMINANT term in the rhs is the **implicit θ-pressure restoring** `-cofwz·(ts diff)` at ~15× the buoyancy magnitude (and at different vertical locations — the restoring peaks at k=5 ≈ 500 m altitude where the bubble's θ′ vertical gradient is steepest, while buoyancy peaks at k=18 ≈ 1850 m near the bubble center).
- The density-coupling term `cofwr·(...)` is zero because `ρpp = state.p_perturbation / c_s²` and `state.p_perturbation = 0` initially. In a system where the rho perturbation persisted across substeps (the slice oracle behavior), this would build up over time. In the warm-bubble harness it is reset to ~0 every substep by the `diagnose_pressure_al_alt` overwrite (§7).

**The buoyancy is not missing a 30× factor**; it is correctly accounted for. The reason `w` does not amplify is *not* "buoyancy underpowered" — it is "the implicit restoring is overpowered relative to the available recurrence machinery, and the rho-feedback path that would normally compensate for it is severed."

---

## §4. Sign convention check

Probe data confirms the sign convention is consistent at the warm bubble:

- `theta_perturbation = state.theta - theta_base` is **positive** in the bubble (`theta_pert_max = 1.98 K`).
- `buoyancy_face = g·θ'/θ_base` is **positive** at faces above the bubble center (probe `e`: `buoyancy_face_at_center_column` max is 0.061 m/s² at the upper bubble edge, positive throughout).
- After substep 1, `w_signed_max = +0.0681 m/s` — **upward** motion at the bubble. So the sign is correct: warm parcel → positive buoyancy → upward acceleration.

The sign convention is NOT the failure mode. A sign error would have produced `w_signed_max = -0.068` at step 1; we see `+0.068`.

The R7 analytic linear-acoustic oracle test (`test_m6x_vertical_acoustic_oracle.py`) passes on the unified path — that test exercises the SAME `_mpas_recurrence_vertical_update` (with `non_hydrostatic=False, pressure_scale=1.0` → routes through the `pressure_scale>0` branch with `_calc_coef_w`+`_vertical_buoyancy_acceleration`+`_vertical_theta_transport`). A sign flip would break this oracle. It passes; therefore the buoyancy sign is correct.

---

## §5. `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38` audit

Probe `f_buoyancy_scale_sweep` runs the direct recurrence for 60 s (240 substeps) with different `buoyancy_scale` values:

| `buoyancy_scale` | `max|w|` at 60 s | `w_signed_max` at 60 s |
|---:|---:|---:|
| 0.38 (current value) | 0.357 | **+0.165** |
| 1.00 | 0.413 | +0.111 |
| 2.60 (= 1 / 0.38, decoupled magic) | 0.559 | +0.0001 |
| 2.63 (= 1 / `MPAS_OMEGA_TO_W_METRIC`⁻¹·1.0) | 0.562 | +0.00009 |

**Result:** increasing the buoyancy_scale does NOT increase sustained upward `w`. Larger scales actually drive `w_signed_max` **toward zero** (more violent oscillation about zero, not a stronger sustained jet). Setting `buoyancy_scale = 1.0` (the "un-scaled" value the contract asks about) gives `w_signed_max = +0.111` versus `+0.165` at `0.38` — *worse* sustained lift, not better.

**Where 0.38 comes from:** It is the *literal* multiplier the slice oracle uses for its `tend_rw[k]` (`mpas_column_slice.py:108`: `tend_rw = RHO_BASE_KG_M3 * 0.38 * buoyancy_faces`). Searching MPAS-A 5.3 source, the value `0.38` does not appear as a named constant — it is a placeholder amplitude in the slice oracle's synthetic tend_rw forcing, chosen so the toy gravity-wave oscillation reaches `w_max ≈ 1` over the slice's 40-step runtime. The unification worker copied it into the production recurrence as `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38` for consistency with the slice — which makes the unified-path test pass but doesn't ground the constant in physics.

**Verdict on §5:** Not a magic-number adjustment failure. `0.38` is documented as the slice-oracle value and changing it (to 1.0 or 2.6) does not move `w_signed_max` toward the [5,10] m/s target. The "magic" is real but the gap is not closed by adjusting it.

A related magic number that *does* matter is `MPAS_OMEGA_TO_W_METRIC = 1.35` (line 33 of `acoustic_wrf.py`). In real MPAS-A, the conversion between `rw_p` (perturbation contravariant mass flux) and `w` (geometric vertical velocity) is *per-column, per-layer*: roughly `rw = ρ_d · w · ∂z/∂η · (1/g)`. A column-constant `1.35` is appropriate only for the slice oracle where `ρ_base = 1` and the column is unit-height-normalized. In the 3-D harness with `ρ ≈ 0.5–1.2 kg/m³` and `∂z/∂η = 100 m`, the right scaling would be `O(10–25)` not `O(1.35)`. This is a second magic-number compromise that survives because: (a) the slice oracle test uses `ρ_base = 1`, hiding the scaling problem; (b) inside the recurrence, the `× 1.35` going in and `÷ 1.35` coming out for `w` cancels except in the buoyancy term where it leaves a net `0.281` factor — which has been compensated by `0.38` as a knob. So the two magic numbers are co-tuned to make the slice test pass, not derived from MPAS source.

---

## §6. `_mu_continuity_increment` audit

The temporary stabilizer is a `tanh` bound that caps |dmu| per substep at `TEMPORARY_MU_CONTINUITY_CFL_FRACTION × |μ_base| = 10⁻³ × ~9 × 10⁴ Pa ≈ 90 Pa`:

```python
max_delta = TEMPORARY_MU_CONTINUITY_CFL_FRACTION * jnp.maximum(jnp.abs(_base_mu(...)), 1.0)
return max_delta * jnp.tanh(raw_delta / max_delta)
```

Probe `j_mu_continuity_ablation_60s` (60 s harness with `mu_continuity={True,False}`) — both paths produce similar `|w|` magnitudes in the first 60 s, because the bubble is still mostly in the early gravity-wave phase where mu_continuity hasn't accumulated yet.

But the 600 s baseline (`a_3d_acoustic_substep_carry_baseline`) shows that `mu_pert_max` grows to **8.8 × 10⁴ Pa** by step 2400 — i.e. the limiter is *saturated* (each substep is hitting `max_delta`). The non-physical `theta_pert_max = 144 K` and `p_pert_max = 450,000 Pa` at the same point indicate the limiter is the only thing preventing immediate blowup, but the system is *already* blown up in θ′ and p′ space.

**Interpretation:** the limiter is not just a "performance smoother"; it is **bounding away a real numerical instability** in the unified mu-coupling path. The instability is driven by the same architectural gap that prevents bubble lifting: the lack of WRF small-step `t_2ave`/`muave`/`_save` scratch means the `mu` update sees raw (un-time-averaged) inflow tendency from a velocity field that is itself an over-coupled gravity-wave oscillation. Off-centering (`epssm > 0`) is not enough to damp it without the time-averaged intermediates.

A sign error or off-by-one in `mu_continuity_tendency` would produce **immediate nonfiniteness** (the limiter caps `dmu`, not `mu`, so a wrong-sign `dmu_dt` would still accumulate without bound after a few substeps). Probe `j` with `mu_continuity=False` runs fine for 60 s, so the mu coupling is *plausible* in sign and indexing — the failure is **architectural insufficiency** (missing scratch), not a wiring fault in `mu_continuity_tendency` itself.

---

## §7. Verdict

**`MIXED`** — there is one confirmed wiring bug *and* a genuine architectural gap; both are real, only the architectural gap drives the warm-bubble failure target.

### Wiring bug (real, fix it regardless)

Probe `g_overwrite_check`:

```
p_perturbation_after_recurrence_only:                      12.701 Pa
p_perturbation_after_acoustic_substep_carry:               4.37e-11 Pa
p_perturbation_from_diagnose_applied_to_post_recurrence:   4.37e-11 Pa
p_perturbation_from_diagnose_on_initial_state:             4.37e-11 Pa
```

`_mpas_recurrence_vertical_update` correctly computes `p_perturbation = ρ_next · c_s² ≈ 12.7 Pa` after substep 1. But the very next lines in `acoustic_substep_carry` (`src/gpuwrf/dynamics/acoustic_wrf.py:875-876`) **erase it**:

```python
final_pressure, final_al, final_alt = diagnose_pressure_al_alt(next_state, base_state, metrics)
final_state = _replace_pressure(next_state, final_pressure, base_state)
```

`diagnose_pressure_al_alt` (`acoustic_wrf.py:235-244`) computes total pressure from `base_state.theta_base` (NOT `state.theta`) and `alt` (which depends only on `state.mu_perturbation` — `phi_perturbation` is *not* included). With `mu_pert ≈ 0` (single column or first substep of the 3-D run), this yields `p_perturbation ≈ 0`. The conservative recurrence's density-derived pressure is discarded every substep.

WRF source anchor cited in the docstring (`module_big_step_utilities_em.F:1082-1087`) is `calc_p_rho_phi`, which uses **`state.theta`** (full θ, not `theta_base`) and **includes `ph_perturbation`** in `al`. The current implementation diverges from the cited source in both respects.

**Proposed fix (small):** in `acoustic_wrf.py:875-876` and `:838` (the pre-vertical-update overwrite), gate the post-vertical-update overwrite on `config.non_hydrostatic`:

```python
# After vertical_acoustic_update — only redo diagnostic al/alt; keep recurrence p_perturbation.
final_pressure_diag, final_al, final_alt = diagnose_pressure_al_alt(next_state, base_state, metrics)
if config.non_hydrostatic:
    final_state = next_state  # keep recurrence p_perturbation
else:
    final_state = _replace_pressure(next_state, final_pressure_diag, base_state)
```

Equivalently, fix `diagnose_pressure_al_alt` to use `state.theta` + include `ph_perturbation` in `al` (the literal WRF `calc_p_rho_phi`). The latter is the cleaner fix but is larger surface (about 30 lines).

**Estimated worker effort:** 2-4 hours for the gated fix + regression suite + a unit test that the recurrence-derived `p_perturbation` survives one substep of `acoustic_substep_carry` (probe `g` is a ready-made test fixture).

This fix alone will **not** raise `w_max_600s` to [5, 10] m/s. It will, however:
- Stop the slow blow-up in θ′ (144 K → bounded) and p′ (450 kPa → physical) seen in probe `a`.
- Make the harness's reported `w_max` more honest (it currently masks the underlying numerical instability).
- Close one path-split source flagged in the original reviewer report.

### Architectural gap (the real blocker)

Probe `d_1d_shrunk_direct_recurrence` is the cleanest read: a single mid-bubble harness column, driven only by `_mpas_recurrence_vertical_update` (no `acoustic_substep_carry`, no PGF, no mu update, no diagnostic-pressure overwrite). `w` rises to 0.475 m/s by step 80 (20 s, the buoyancy-driven transient) then decays back to 0.044 m/s by step 2400 (600 s). This is **gravity-wave oscillation**, not bubble lifting.

The recurrence as coded reproduces the slice oracle's first ~10 substeps of gravity-wave amplification but, crucially, does not reproduce the WRF small-step physics that *sustains* bubble lifting against the implicit pressure-restoring force. The slice oracle test passes (1.69 % slice RMSE) because both the slice oracle and the recurrence are running the SAME truncated gravity-wave model, not because either reproduces convective bubble dynamics.

Missing WRF small-step machinery (consistent with the ADR-021 fallback hypothesis):
1. **`t_2ave`** (`module_small_step_em.F:1340-1396`) — the time-averaged θ used in the buoyancy term. The current recurrence uses instantaneous `θ_perturbation` directly, which over-couples to the implicit pressure-restoring `cofwz·(ts[1:]-ts[:-1])` term and produces gravity-wave oscillation rather than convective rise.
2. **`ww`/`muave`** — substep-averaged vertical mass flux and column mass. Their absence means the `μ` update sees the raw (non-time-averaged) horizontal divergence of a gravity-wave-oscillating `u/v`, which is the source of the `_mu_continuity_increment` limiter saturation reported in §6.
3. **`_save` fields** — WRF small-step pattern `save then update` that prevents the pressure-θ-w coupling from re-injecting its own previous substep's response. The current code reads back from the state that was just written, creating spurious feedback.
4. **Proper mass-flux conversion** — `MPAS_OMEGA_TO_W_METRIC = 1.35` is a column-constant scalar; real MPAS uses `μ_d · ∂z/∂η · (1/g)`. Without this, the recurrence's buoyancy magnitude is co-tuned with `0.38` to fit the slice oracle only.

### Concrete recommendation for the manager

1. **Land the wiring fix** as a separate small sprint (~2-4 worker hours) so `acoustic_substep_carry` stops erasing the recurrence's `p_perturbation`. This is correct in isolation regardless of which architecture wins, and it makes the harness's `w_max` reporting honest about θ′ and p′ blow-up.

2. **Proceed with ADR-021 (WRF small-step prototype)** — already dispatched in parallel per `181f544`. The diagnostic confirms the conservative MPAS-recurrence path *as currently scoped* cannot reach the warm-bubble target even with the wiring fix; the missing scratch fields (`t_2ave`, `ww`, `muave`, `ph_tend`, `_save` family) are the gap.

3. **Update the warm-bubble harness's success metric**: report both `jnp.max(w)` AND `jnp.max|w|`, `theta_pert_max`, `p_pert_max`, and `mu_pert_max`. The current single-metric reporting hides the real failure mode (slow blow-up in θ′/p′ while `max(w)` happens to be small).

4. **Update the slice-oracle test**: extend it to 600 s (not 40 s) and assert `max|w|` stays bounded *and* `theta_pert_max` does not grow >5× initial. This would have caught the architectural gap earlier and would prevent any future "the slice test passes" claim from being read as "the recurrence reproduces bubble dynamics."

---

## §8. Confidence

**Confidence: HIGH**, with one explicitly-flagged limitation.

Justification:
- The wiring bug is **directly measured** (probe `g`): post-recurrence `p_perturbation` is 12.7 Pa; after `acoustic_substep_carry` it is 4 × 10⁻¹¹ Pa. The overwrite is in plain sight at `acoustic_wrf.py:875-876` and the diagnose function at `acoustic_wrf.py:230-244` demonstrably uses `theta_base` (not `state.theta`) and ignores `ph_perturbation` in `al`.
- The architectural gap is **directly measured** (probe `d`): the cleanest possible recurrence-only loop (no PGF, no mu coupling, no diagnose overwrite, single column from harness center) cannot sustain bubble lifting. It produces 0.475 m/s peak at 20 s then decays to 0.044 m/s by 600 s. Pure linear-acoustic recurrence reproduces gravity-wave oscillation but not convective lift, and no parameter sweep within the current recurrence machinery (`buoyancy_scale ∈ {0.38, 1.0, 2.6, 2.63}` per probe `f`, `epssm ∈ {0.0, 0.1, 0.3}` per probe `i`) raises the sustained `w_signed_max` toward the target.
- The buoyancy sign is correct, the buoyancy magnitude is roughly right (g·θ′/θ_base ≈ 0.06 m/s²; not 30× too small; not zero), and the slice-oracle production-grade test does pass — so this is not a one-line sign flip, a missing factor, or a typo. The deficiency is *structural*.
- The MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE = 0.38 audit (§5) closes the "maybe it's just a magic number" hypothesis: increasing it to 1.0 or 2.6 does not lift `w_signed_max`; it just makes the oscillation more violent.

**Limitation:** I did NOT exhaustively walk every line of MPAS `mpas_atm_time_integration.F:2146-2208` against `_mpas_recurrence_vertical_update`. There may be a more specific deviation from MPAS source (e.g., `coftz` vs `coftz·zz` zeta-coordinate factor; `cofwt[k-1]` vs `cofwt[k]` off-by-one) that, if matched, would close the gap without going to ADR-021. The parallel ADR-023-conservative-column-prototype worker dispatched in `181f544` should surface this if so. My current evidence supports MIXED with primary architectural finding, but a closer line-by-line MPAS comparison sprint could downgrade this to WIRING-BUG-WITH-FIX-PROPOSAL if such a deviation is found.

---

## §9. Open questions for the next sprint

1. **Line-by-line MPAS-A 5.3 vs `_mpas_recurrence_vertical_update`**: does the comparison expose a specific deviation (off-by-one, missing `zz[k]·zz[k-1]` zeta-factor, missing `wwAvg` accumulation, wrong `cofwt[k]` index) that would single-handedly explain the gap without invoking missing-scratch? The ADR-021 parallel sprint may produce this evidence directly.

2. **Big-step coupling**: the M6.x warm-bubble target of `w_max ∈ [5, 10] m/s` over 600 s was set based on what reference? If it is a WRF EM-CORE 5-min idealized squall-line target (which it appears to be), it assumes **periodic big-step (RK3) re-injection of theta, momentum, and microphysics tendencies** — which the harness does NOT do (`m6_warm_bubble_test.py` is pure small-step). Even a perfect WRF-faithful small-step implementation might not hit `[5,10] m/s` without RK3 coupling. The architectural assessment should test both:
   - "Does small-step alone hit the target?" (a question for ADR-021 prototype testing)
   - "Does small-step + RK3 hit the target?" (a question for the full dycore wiring)
   If the answer to the first is "no", the warm-bubble harness needs to either gain a big-step coupling or relax its `w_max` target.

3. **`MPAS_OMEGA_TO_W_METRIC = 1.35`**: what is the correct per-column, per-layer mass-flux ↔ vertical-velocity conversion in this code's vertical-coordinate convention? This is independent of the architecture choice and should be derived from `mu_d · ∂z/∂η · (1/g)` or the equivalent for the hybrid-η coordinate. The slice oracle's `ρ_base = 1` and unit-normalized column made this co-tunable with `0.38`; the harness's stratified column does not.

4. **Slice oracle's `tend_rw[k] = 0.38 · buoyancy_face`**: where does the `0.38` come from in MPAS source? If it cannot be cited from MPAS source, both the slice oracle and the recurrence are tuned to each other rather than to MPAS. The slice oracle should either cite a source line or document it as "synthetic placeholder for column-slice gravity-wave amplitude calibration only — not a portable physics constant."

5. **Should the warm-bubble harness change to use `jnp.max|w|` instead of `jnp.max(w)`**? The current single-positive-peak metric misses the negative-w jet (0.91 m/s downward at step 2400) and gives a misleadingly small "failure number" (0.04). Either way the answer is FAIL_TARGETS_NOT_MET, but for diagnostic transparency it should report both.

---

*Diagnostic complete. Per sprint contract, this report does NOT modify production code (the diagnostic script is read-only on production code and lives under `scripts/`). The manager should treat the §7 verdict as evidence-supported guidance for the next dispatch — fix the wiring bug as a small follow-on, proceed with ADR-021 prototype as the primary architectural answer for the warm-bubble target.*
