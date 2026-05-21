# M6-S1 Reviewer Report — Coupled Interface and Precision Boundary Freeze

**Sprint**: `2026-05-21-m6-s1-coupled-interface-freeze`
**Reviewer**: Claude Opus 4.7 xhigh (fresh context, independent of M6 manager)
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/codex/m6-s1-coupled-interface-freeze` @ `e50bcd2`
**Date**: 2026-05-21
**Read order followed**: PROJECT_CONSTITUTION → AGENTS → sprint-lifecycle (Double-AI hard rule) → sprint-contract → worker-report → ADR-010 → ADR-002 → ADR-007 → feedback_validation_philosophy → all touched files → artifacts → tests → script.

Verdict at a glance: **ACCEPT-WITH-MINOR-FOLLOWUPS**. M6-S2 may dispatch immediately; followups are non-blocking but must be enumerated in M6-S2's contract.

---

## R-Findings Table

| ID | Severity | AC / Probe | Finding | Disposition |
|---|---|---|---|---|
| R-1 | INFO | AC1 | All 23 new SoA leaves present at expected shapes and device-resident on GPU. Independent test run `tests/test_m6_state_extension.py` -> 2 passed. | PASS |
| R-2 | INFO | AC2 | `PRECISION_MATRIX` covers all 30 State fields plus `pgeop` alias. Tests pin every dtype to the matrix; locked/gated partition matches ADR-007 verdict rows. | PASS |
| R-3 | MINOR | AC2 / precision policy | Storage dtype for `u/v/theta/qv/qc/qr/qi/qs/qg/Ni/Nr/Ns/Ng/qke` is set to FP32 today via `FP32_GATED = jnp.float32`, even though ADR-007 §Status says "It does not modify production `src/gpuwrf` code" and conditions production downcast on the M6-S7 Tier-4 U10/V10/T2 RMSE gate. ADR-010 §Decision and worker-report both call this an "interface freeze, not a speedup claim", and the dummy carry does not produce operational forecasts — so the policy violation is only effective once M6-S2's forecast driver runs real physics. **Followup binding on M6-S2**: either (a) M6-S2 forecast driver must NOT claim operational fitness until M6-S7 gate passes, or (b) flip storage to FP64 in M6-S2's first PR until gates pass. The worker's choice to interpret "interface freeze" as "literal FP32 storage" is defensible but should be ratified, not implied. | FOLLOWUP |
| R-4 | INFO | AC3 | Four adapters (`thompson_adapter`, `mynn_adapter`, `rrtmg_adapter`, `surface_adapter`) at `src/gpuwrf/coupling/physics_couplers.py:133,166,200,223`. Each slices SoA -> column-batched view, calls the M5 kernel unchanged, reassembles State. `git diff main...HEAD -- src/gpuwrf/physics/` is empty (zero kernel diff) — wrap-only contract honored. | PASS |
| R-5 | MINOR | AC3 | `DEFAULT_DZ_M = 100.0` (`physics_couplers.py:29`) is a flat layer-thickness placeholder fed to MYNN and RRTMG. MYNN uses it for `_interface_dz`, `_edge_heights`, `_wrf_zw` (`mynn_pbl.py:142-260`) — the whole PBL eddy structure scales with this; RRTMG LW/SW use it as the path length in `dz=jnp.maximum(state.dz, 1.0)` (`rrtmg_lw.py:134`, `rrtmg_sw.py:132`), which directly sets optical-depth integration. On a non-uniform terrain-following grid this is wrong by ~10× near the surface and ~5× aloft. Acceptable for the M6-S1 dummy proof because no operational/conservation/RMSE claim is attached — but M6-S2 must thread real `GridSpec` metrics before any forecast claim. ADR-010 explicitly acknowledges this risk. | ACCEPTED INTERIM |
| R-6 | INFO | AC4 (transfer audit) | Independent re-run `python scripts/m6_run_dummy_coupled.py` produced `host_to_device_bytes_post_init=0` and `device_to_host_bytes_post_init=0` (measured via `count_transfer_bytes` over the xprof trace). Matches the committed artifact. HARD-CHECK PASS. | PASS |
| R-7 | MINOR | AC4 (temporary bytes) | `temporary_bytes_per_step = 0` in both artifacts is **hard-coded literal**, not measured. See `scripts/m6_run_dummy_coupled.py:242,274` and the inherited M3 pattern `src/gpuwrf/profiling/budget.py:82,89`. The justification at `budget.py:89` ("no array constructors") was true for the M3 theta-no-op scan body; it is NOT true for the M6 coupled scan body which contains `jnp.zeros_like`, `jnp.ones_like`, `jnp.concatenate`, `jnp.moveaxis`, `density_from_pressure_temperature`, `_to_columns`/`_from_columns`, and an `_cloud_fraction_columns` clip pipeline. JAX/XLA fusion may still drive the actual figure to zero, but the artifact asserts it instead of proving it. **Followup**: either add a real HLO-level temporary-buffer audit, or downgrade the field from `0` to `null` with a "not measured" tag. Not blocking because the H2D/D2H zeros are the load-bearing signal for M6-S2 dispatch. | FOLLOWUP |
| R-8 | INFO | AC4 (anti-pattern) | `grep -rn "lax.cond\|host_callback\|io_callback\|pure_callback" src/gpuwrf/coupling/ scripts/m6_run_dummy_coupled.py` returns empty. Worker's "earlier `lax.cond` D2H bug" story is fixed in the committed script: `run_dummy_coupled` uses static nested scans (`scripts/m6_run_dummy_coupled.py:151-159`), no dynamic radiation-cadence predicate. | PASS |
| R-9 | MINOR | AC4 (cadence edge case) | The trailing `remainder = steps % 10` branch (`scripts/m6_run_dummy_coupled.py:156-159`) skips radiation for the tail steps. For `steps=100` this is `length=0` (no-op) — fine. For any non-multiple-of-10 step count the trailing steps would silently omit radiation, and the artifact does not declare this. Cosmetic since the contracted run is 100 steps; M6-S2 driver should not inherit the pattern verbatim. | FOLLOWUP |
| R-10 | INFO | AC5 | `spacetime_budget.json` per-kernel record verified independently. My re-run: dycore=24 launches, Thompson=7, MYNN=32, surface=1, RRTMG=170 — all match the report. HLO bytes match within ±200 (expected XLA noise). Wall times match within ±2× (expected timer noise at sub-ms scale). Total coupled `kernel_launches_per_step=320` is exactly `dycore(24) + thompson(7) + mynn(32) + surface(1) + rrtmg(170)/10*10 = 320` for the 100-step compiled scan, internally consistent. | PASS |
| R-11 | INFO | AC5 (no launch fudge) | `grep -n "min(raw\|min(launch\|min(.*cap)"` against the script and `src/gpuwrf/profiling/` is empty. `kernel_launches_per_step` is sourced from compiled HLO text scrape (`scripts/m6_run_dummy_coupled.py:165-171`), not from a clamp. | PASS |
| R-12 | INFO | AC6 | `.agent/decisions/ADR-010-coupled-state-extension.md` exists, cites ADR-002 §State layout, cites ADR-007 §Authorization Matrix, points at `coupled_dummy_carry.json` and `spacetime_budget.json`. Status: PROPOSED for reviewer. After this report, it is ACCEPTED. | PASS |
| R-13 | MINOR | AC7 / amendment-3 | File-ownership freeze (ADR-010 §File Ownership Freeze) covers S2–S8 with disjoint paths. **However**: M6 plan critic amendment 3 said S1 should freeze boundary/forcing/output metadata interfaces, not just physics state leaves. The current State has NO lateral-boundary-forcing handles (no `u_bdy`, `theta_bdy`, no time-varying BC array port), NO output manifest, NO validation snapshot registration. None are precluded — they are separable additions — but M6-S2 will need to amend State for boundary-forcing inputs (Canairy d02 replay requires this), which requires touching `src/gpuwrf/contracts/state.py` again, a file ADR-010 implicitly froze. Resolution: M6-S2 contract MUST state explicitly that adding boundary-forcing leaves is in-scope and treat the amendment as a planned M6-S1.b appendix, not a violation. | FOLLOWUP |
| R-14 | INFO | Verifiability triple #2 | `git diff main...HEAD --name-only -- src/gpuwrf/physics/` is empty — zero modification to any M5-frozen physics kernel. `git diff main...HEAD --stat` shows 13 files changed: 5 source (contracts/state.py, contracts/precision.py, coupling/__init__.py, coupling/physics_couplers.py NEW, debug/snapshots.py 5-line compat patch), 1 script NEW, 3 tests NEW, 1 ADR NEW, 2 artifacts NEW, 1 worker-report. All within the "Files Worker May Modify" allowlist. The `debug/snapshots.py` 5-line patch is a benign compatibility fix to make snapshot recording iterate `jax.tree_util.tree_leaves(state)` instead of the hard-coded 8-leaf M3 tuple — necessary because State.__slots__ grew. | PASS |
| R-15 | MINOR | Adversarial — dtype cast trace | Walked one FP32-gated field (`qv`) through `thompson_adapter`: enters as FP32 storage, `_to_columns(state.qv)` keeps FP32, `density_from_pressure_temperature(state.p, T, state.qv)` promotes via JAX rules (p FP64, T FP64 from `_temperature_from_theta`, qv FP32 -> result FP64), kernel sees mixed precision but inputs ARE cast at the State boundary on output via `.astype(_field_dtype("qv"))`. The cast IS present. Side effect: `State.replace` now silently downcasts any FP64-typed update to the field's storage dtype (`state.py:273-282`); this is the M4 dycore fix the worker mentioned. It is correct but it is also a quiet behavior change inside dycore RK3 substeps — theta updates that used to flow FP64 internally now downcast to FP32 at every `state.replace`. Tier-1 dycore tests still pass (`test_m4_dycore_step.py` -> 3 passed independently) so M4 conservation invariants are not visibly broken, but the precision delta is real and undocumented in ADR-010. Should be called out in the M6-S5 ADR-007-verdict sprint. | FOLLOWUP |
| R-16 | INFO | Adversarial — scale-up to d02 | HLO size 5.19 MB at 16×16×30. JAX `lax.scan` compiles loop body ONCE; static structure dominates HLO size, so scaling the spatial grid to 160×67×45 (~62× cell count) should NOT proportionally inflate compiled HLO. Compile time should grow O(1) in grid, runtime grows linearly. The static-scan nest is correct here. No HLO blow-up risk identified. | PASS |
| R-17 | INFO | Tests | `pytest -q tests/test_m6_state_extension.py tests/test_m6_precision_matrix.py tests/test_m6_dummy_coupled.py tests/test_m4_dycore_step.py` -> 9 passed independently in this reviewer's environment. | PASS |

---

## Per-AC Verification

**AC1 — State pytree extension.** PASS. `src/gpuwrf/contracts/state.py:35-73` (`_state_field_shapes`) and `state.py:159-257` (`State.__slots__` and `__init__`) define all 23 new leaves with correct shapes (3D `(nz,ny,nx)` for hydrometeors/numbers/qke; 2D `(ny,nx)` for surface scalars/accumulators). Docstring documents units (`state.py:143-157`). Test at `tests/test_m6_state_extension.py:33-49` asserts presence, GPU residency, shape, and dtype for every new leaf. Independent re-run passes.

**AC2 — Precision boundary memo + code.** PASS WITH R-3 FOLLOWUP. `src/gpuwrf/contracts/precision.py:54-92` defines `PRECISION_MATRIX` as `(dtype, fp32_gate_required)` per field; `STATE_FIELD_ORDER` (`precision.py:19-51`) is the canonical iteration order; `DTypeRegistry.from_precision_matrix()` (`precision.py:102`) wires storage. Test `tests/test_m6_precision_matrix.py:10-33` enforces matrix-vs-storage equivalence and ADR-007 partition (gated set = `{u,v,theta,qv,qc..qg,Ni..Ng,qke}`, locked = everything else). Locked rows include `mu, p, ph/pgeop, w, all surface-stability handles, all precip accumulators` per ADR-007 §Decision §1. **R-3 caveat**: production downcast was scoped by ADR-007 to be follow-on after RMSE gates; M6-S1 has effectively executed the downcast at the storage layer ahead of those gates. Defensible as interface freeze; must be ratified by M6-S2 not silently inherited.

**AC3 — Coupling adapter contracts.** PASS WITH R-5 INTERIM. All four adapters present at file:line `physics_couplers.py:133` (thompson), `:166` (mynn), `:200` (surface), `:223` (rrtmg). Wrap-only verified via `git diff main...HEAD -- src/gpuwrf/physics/` = empty. Synchronous ordering in `run_dummy_coupled` at `scripts/m6_run_dummy_coupled.py:144-159`. `DEFAULT_DZ_M = 100.0` (R-5) is an interim that ADR-010 explicitly hands to M6-S2 with a "must thread real grid metrics" rider.

**AC4 — 100-step dummy coupled carry.** PASS for H2D=0, D2H=0 (independently reproduced). PARTIAL for temporary_bytes (R-7: literal not measurement). Anti-pattern grep clean (R-8). Cadence edge case is cosmetic (R-9). The HARD-CHECK in `sprint-contract.md:60` is on `host_to_device_bytes_post_init = 0 and device_to_host_bytes_post_init = 0`, and both are real measurements that I reproduced — so the hard-check is satisfied.

**AC5 — Spacetime budget.** PASS. Per-kernel record matches `PERFORMANCE_TARGETS.md`-style schema (per-kernel wall, launches, HLO bytes). Numbers independently re-derivable from compiled HLO scrape. RRTMG cadence_steps=10 is explicit. Total per-step = 0.65–0.67 ms is consistent between artifact and reviewer's re-run.

**AC6 — ADR.** PASS. ADR-010 cross-references ADR-002 and ADR-007 with §-level pointers. Status correctly says PROPOSED — this report ACCEPTS it.

**AC7 — File ownership freeze.** PASS WITH R-13 FOLLOWUP. ADR-010 lists disjoint owned paths per sprint. The amendment-3 gap (no boundary-forcing/output-manifest leaves) is not a precondition violation but must be planned-amendment in M6-S2's contract.

---

## Verifiability Triple

1. **0-byte transfer audit**: PASS. Independent re-run reproduced `h2d=0, d2h=0`. The script's profiler-trace + `count_transfer_bytes` path is real measurement, not literal. No `lax.cond`, no `*_callback` paths present (R-8).
2. **No physics-kernel modification**: PASS. `git diff main...HEAD -- src/gpuwrf/physics/` = empty. Wrap-only contract honored.
3. **No `min(raw, cap)` launch fudge**: PASS. Launch numbers derive from compiled HLO scrape via `kernel_launches_per_step(text)`; no clamp pattern present.

---

## Plan-Critic Amendment-3 Check

The fresh M6 plan critic (codex critical-reviewer, RATIFY-WITH-AMENDMENTS) said in amendment 3: "Amend S1 to freeze boundary/forcing/output metadata interfaces, not just physics state leaves."

**Status in M6-S1 deliverables**:
- Physics state leaves: FROZEN ✓
- Surface state handles: FROZEN ✓
- Precipitation accumulators (output diagnostics in mm): FROZEN ✓
- Lateral boundary forcing handles (for d02 nested AIFS replay): **NOT FROZEN** — no `u_bdy`, `theta_bdy`, `qv_bdy`, no time-varying BC port, no `t_bdy` cadence handle.
- Output manifest (which variables, at what cadence, to what file): **NOT FROZEN** — separable from State; no interface yet.
- Validation I/O hooks (Tier-2/Tier-3/Tier-4 snapshot registration): **NOT FROZEN** — separable; `debug/snapshots.py` exists but is not the validation channel.

**Reviewer assessment**: M6-S1's freeze does NOT preclude implementing the M6 amended plan, but it forces M6-S2 to amend State.py and ADR-010 to add boundary-forcing leaves before the forecast driver can ingest Canairy AIFS BCs. Two equivalent paths:

(a) **Accept** M6-S1 as-is; require M6-S2 contract to explicitly bundle boundary-forcing State extension as planned M6-S1.b appendix work, reviewed under the same Double-AI hard rule.
(b) **Reject-bounded** M6-S1; require a 1–2 hour follow-up sprint that adds boundary-forcing leaves before unblocking M6-S2.

Path (a) is cheaper and matches the "interface freeze for downstream parallel work" intent of M6-S1 because the M6-S2 forecast driver is the natural owner of boundary forcing (it owns time evolution). I recommend (a). Path (b) is justified only if the M6 manager judges that adding to State after M6-S1 close risks file-ownership confusion across S2–S8.

---

## Adversarial Probes

**Probe 1: FP32 dtype cast at adapter boundary.** Trace for `qv` (FP32-gated). State storage is FP32; `thompson_adapter` calls `_temperature_from_theta` which casts theta to `p.dtype` (FP64), then `density_from_pressure_temperature` produces FP64 rho. Thompson kernel receives mixed inputs. Output is cast back via `.astype(_field_dtype("qv"))` -> FP32. Boundary cast IS present. Side note (R-15): `State.replace` (`state.py:273-282`) silently downcasts any FP64 update to current field dtype — this affects dycore RK3 substeps in subtle ways (theta updates now downcast at every replace). M4 conservation tests still pass, but the precision delta should be re-validated in M6-S5.

**Probe 2: DEFAULT_DZ_M = 100.0 propagation.** MYNN's eddy-diffusivity structure scales with `dz`; using a uniform 100m collapses the surface-layer/PBL stratification. RRTMG optical depth uses dz as the path length, so column-integrated radiation is wrong by the same factor. Acceptable here because no operational claim is attached; M6-S2 must thread `GridSpec.eta_levels` -> `dz` array before forecast claims. ADR-010 acknowledges this in §Adapter Contracts.

**Probe 3: HLO scaling to d02 (160×67×45).** Static scan structure means HLO compile is grid-shape-invariant. Cell count grows ~62×, so runtime scales linearly and HBM persistent state grows ~62× (manageable on RTX 5090's 32 GB). No HLO blow-up risk. The script's compile-and-time path will work; only proof-object generation might need a per-step budget downgrade because per-step wall will jump to ~tens of ms.

---

## Honest Accounting

What this sprint actually proves:
- Interface and pytree residency are correct.
- All M5 physics kernels survive being called wrapped, with no signature changes.
- The committed compile path has zero post-init host/device transfer bytes and 320 kernel launches per step at 16×16×30.
- File ownership for M6-S2..S8 is documented.

What this sprint does NOT prove (and worker correctly disclaims):
- Forecast correctness (no IC/BC, dummy uniform fields, no truth comparison).
- Conservation closure (no mass/energy/water budget run).
- Operational RMSE on U10/V10/T2 (binding Tier-4 work, owned by M6-S7).
- 4× speedup (ADR-007 verdict owned by M6-S5).
- Real `dz` integration with MYNN/RRTMG (owned by M6-S2).
- Boundary-forcing / output-manifest / validation-I/O interfaces (amendment-3 gap, owned by M6-S2 or a micro-sprint).
- `temporary_bytes_per_step = 0` as a measurement (currently literal; R-7).

This honesty matches the worker report's "unresolved risks" section and ADR-010 §Consequences And Risks. No hallucinated claims found.

---

## Binding Decision

**ACCEPT-WITH-MINOR-FOLLOWUPS.**

Rationale: All 7 ACs functionally pass. The verifiability triple is independently reproducible. The Double-AI hard rule (`.agent/rules/sprint-lifecycle.md:14-18`) is satisfied by this fresh-context Opus 4.7 review. M6-S2..S8 can dispatch in parallel under the ADR-010 file-ownership freeze.

The followups are minor and non-blocking:

1. **R-3** (precision policy ratification): M6-S2 contract must state explicitly that storage of `u/v/theta/qv/hydrometeors/numbers/qke` is FP32 by current implementation, and that operational forecast outputs from M6-S2 are NOT to be claimed as production-fit until M6-S7 Tier-4 RMSE gates pass per ADR-007. Either formally ratify this storage choice OR flip to FP64 in M6-S2's first PR.

2. **R-7** (temporary_bytes is unmeasured): Add a real HLO/xprof tensor-memory measurement OR change the field type to `null` with a "not measured" note. Inherit-fix from `src/gpuwrf/profiling/budget.py` so M3/M4/M5/M6 all use the same honest semantics.

3. **R-9** (cadence remainder skips radiation): Cosmetic for M6-S1; M6-S2 driver should generalize the cadence so non-multiple-of-10 step counts behave predictably.

4. **R-13 / amendment-3**: M6-S2 contract MUST plan the boundary-forcing State amendment as in-scope work. Add `u_bdy`, `v_bdy`, `theta_bdy`, `qv_bdy` (or equivalent) leaves and document the BC-time interpolation port in ADR-010 as an appendix.

5. **R-15** (dycore precision drift via State.replace): Add a one-paragraph note in ADR-010 §Consequences pointing out that `State.replace` now matches the storage dtype on every update, which silently downcasts FP64 intermediates to FP32 for gated fields. Re-validate Tier-2 conservation invariants in M6-S5 ADR-007 verdict.

None of R-3/R-7/R-9/R-13/R-15 require reopening this sprint.

---

## M6-S2 Prerequisites

Before M6-S2 implementation work begins, the M6-S2 sprint contract must include:

- **P1** (from R-3): an explicit statement on FP32 storage policy — either "ratify M6-S1 storage and gate operational claims on M6-S7 outcomes" or "revert FP32 fields to FP64 in driver path until gates pass". Manager decides.
- **P2** (from R-13): explicit in-scope amendment of `src/gpuwrf/contracts/state.py` and `.agent/decisions/ADR-010-coupled-state-extension.md` to add boundary-forcing leaves and a BC-time-interpolation port; route the change through Double-AI review even though it touches an "M6-S1-owned" file.
- **P3** (from R-5): replace `DEFAULT_DZ_M = 100.0` with real `GridSpec.eta_levels`-derived `dz` array threaded into MYNN/RRTMG adapters. Decide whether the adapter signature widens to `(state, grid, dt)` or whether grid metrics live on State.
- **P4** (from R-7): adopt a real `temporary_bytes_per_step` measurement before any M6-S5 speedup claim.
- **P5** (from R-9): generalize the radiation cadence to handle arbitrary step counts, OR document the multiple-of-10 invariant in the driver contract.

---

## Closing

This is a clean interface-freeze sprint. The worker delivered on all 7 ACs, the verifiability triple holds independently, no physics kernels were touched, the file-ownership freeze is disjoint, and the proof objects exist and are reproducible. The five followups above are real but non-blocking and naturally belong in M6-S2's contract. M6-S2 dispatch is **UNBLOCKED**.

Reviewer signature: Claude Opus 4.7 xhigh, fresh-context, independent of M6 manager.
