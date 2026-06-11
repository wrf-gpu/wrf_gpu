# V0.14 GPT Switzerland Residual Root Cause

Date: 2026-06-11
Worker: GPT-5.5 xhigh
Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/gpt-switzerland-residual-root-cause`

## Verdict

`NARROWED_NO_FIX`

The manager/Fable interior `phi/p` interpretation is supported, but the best current evidence narrows it further: the first remaining interior wrong state is created during the single RK1 acoustic substep between WRF calls `21601` and `21602`. Boundary cadence/advection, specified stage omega, real-case `rhs_ph`, `rw_tend` vertical-CFL damping, and final wrapper pressure refresh are not the first creator.

After fixing the proof-harness `OperationalNamelist.emdiv` API mismatch, the GPU-backed surface-`w` discriminator rejects the known decoupled-vs-WRF-coupled lower-boundary wind-feed deviation: `p` and `ph` interior RMSE are unchanged at `1.126197518453275` and `0.43526395846317767`, with `0.0` improvement fraction.

The exact remaining target is the boundary between `advance_mu_t` outputs consumed by `advance_w` and `advance_w_wrf()` itself:

- `advance_mu_t` outputs: `ww_new`, `muave_new`, `muts_new`, `theta_coupled`;
- `advance_w_wrf()` terms other than the tested lower-boundary surface-wind feed: `rhs_seed`, phi vertical advection, explicit `rw_tend`, implicit pressure terms, Thomas solve, and final `ph_next`;
- `calc_p_rho_step` is downstream for `p/al`, but cannot be the first interior `ph` creator.

No WRF-faithful source fix is proven yet, so I did not edit model source or run an h36 forecast gate.

## Hypothesis Ledger

| Rank | Root class | Evidence for | Evidence against / narrowed | Cheapest next proof |
|---:|---|---|---|---|
| 1 | `advance_w_wrf()` ph/w implicit machinery or its immediate inputs | From identical h36 state, after one RK1 acoustic substep `ph` interior RMSE is already `0.4352639585` and `p` RMSE `1.126197518`; `mu` RMSE is only `0.0208960377`. `ph` is produced by `advance_w` in this interval. | Existing WRF dumps do not include pre/post `advance_w` work arrays, so cannot yet split input vs solver term. The surface-wind-feed discriminator rejected one local source deviation. | Add WRF dump around `advance_mu_t` and `advance_w` for call `21602`, or extend the proof harness to expose internal `advance_w_wrf()` terms. |
| 2 | `advance_mu_t` output consumed by `advance_w` | It produces `ww_new`, `muave`, `muts`, and theta work immediately before `advance_w`; these can bias `ph_next`. | Stage-boundary `mu` error is small relative to `p/ph`; specified stage omega before the acoustic loop is exact. | Dump/compare post-`advance_mu_t` `ww_new/muave/muts/theta_coupled` before `advance_w`. |
| 3 | Known non-WRF surface-`w` feed in `advance_w` | Source intentionally feeds decoupled `u_1/v_1`; WRF feeds coupled work `u_2/v_2`. This is a real source deviation in the suspected operator, and the discriminator shows it changes diagnosed surface `w` work. | Rejected as first `phi/p` creator: `wrf_coupled` and `current_decoupled` give identical `p/ph` interior RMSE and zero `ph_work_after_advance_w` / `p_work_after_calc_p_rho_step` delta. | Do not edit source for this candidate as an h36 fix unless a separate stability/faithfulness sprint wants to revisit it. |
| 4 | `calc_p_rho_step` / pressure refresh | `p/al/alt` amplify after `ph` changes; `alt` is large by stage end. | Interior `ph` is already wrong after RK1. Final stage3 raw and final wrapper interior `p/ph` were identical in prior proof, so wrapper refresh is not first creator. | If `ph_next` matches WRF but `p` does not, compare `calc_p_rho_step` inputs and outputs. |
| 5 | Real-case `rhs_ph` or stage `calc_ww_cp` | Historically plausible and previously stale reports ranked it high. | Current proof says specified stage omega interior RMSE is `5.79e-16`; new real-case `rhs_ph` port validation interior RMSE is `2.40e-11`. | Keep as regression watch only. |
| 6 | Boundary lane | Boundary band was genuinely wrong and is now more WRF-faithful. | h36 excess outflux worsens under the WRF-faithful boundary/advection path: `rhsph -27.204`, `speccad -30.682`, `advdeg -32.870 Pa/cell/h`; interior stage RMSE is unchanged by boundary fixes. | Do not spend next sprint here unless new interior proof contradicts this. |

Expected performance impact if fixed: likely neutral to small. A pure term-order/input correction in `advance_w` should not add host/device transfers. If the fix is coupled surface-`w`, it may affect stability and needs a short Canary/Switzerland smoke gate because the current source comments document prior terrain-related instability.

## Evidence Chain

1. WRF call `21601` and JAX h36 start are effectively identical for core fields: `mu/p/ph` max abs `0.0`; `alt` max abs `3.10e-06`; `al` max abs `5.35e-05`.
2. After RK1 stage 1, which is one acoustic substep at `dt=18/3=6 s`, interior increment RMSE vs WRF call `21602` is:
   - `mu`: `0.020896037745516495`
   - `p`: `1.1261975184532773`
   - `ph`: `0.4352639584631776`
   - `al`: `9.16e-05`
   - `alt`: `9.12e-05`
3. Current term exclusions:
   - specified stage omega vs WRF oracle, interior RMSE `5.79142152447787e-16`;
   - real-case `rhs_ph` port validation vs oracle, interior RMSE `2.3951502769070958e-11`;
   - `w_damping` active cells at this state: `0`.
4. Boundary fixes improve ring errors but do not change interior stage errors and do not close h36:
   - CPU truth residual `+5.178443877551032 Pa/cell/h`;
   - `rhsph` residual `-21.882908163265313`, excess `-27.203954081632645`;
   - `speccad` residual `-20.302933673469383`, excess `-30.68188775510204`;
   - `advdeg` residual `-21.064285714285717`, excess `-32.86951530612245`.
5. Existing WRF-native HPG dumps contain stage-boundary `p/ph/al/alt/php/mu/muu/muv` and HPG subterms, but not intra-`advance_w` work arrays or Thomas-solve terms. That is the current method limit.

## Proof Objects

- `proofs/v014/gpt_switzerland_residual_narrowing_summary.json`
- `proofs/v014/switzerland_advance_w_phi_discriminator.py`
- `proofs/v014/switzerland_advance_w_phi_discriminator.json`
- Existing evidence consumed:
  - `proofs/v014/switzerland_acoustic_substep_blocker.json`
  - `proofs/v014/switzerland_acoustic_continuation.json`
  - `proofs/v014/switzerland_stage3_wrapper_cadence.json`
  - `proofs/v014/gpt_stage3_wrapper_verifier.md`

## Commands Run

```bash
sed -n '1,220p' PROJECT_CONSTITUTION.md
sed -n '1,240p' AGENTS.md
sed -n '1,260p' /home/enric/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-gpt-switzerland-residual-root-cause/sprint-contract.md
sed -n '1,260p' .agent/skills/managing-sprints/SKILL.md
sed -n '1,280p' .agent/skills/validating-physics/SKILL.md
sed -n '1,260p' .agent/decisions/V0140-RELEASE-CHECKLIST.md
sed -n '1,260p' proofs/v014/gpt_stage3_wrapper_verifier.md
sed -n '1,260p' .agent/reviews/2026-06-11-v014-fable-stage3-wrapper-cadence.md
python -m json.tool proofs/v014/switzerland_stage3_wrapper_cadence.json
python -m json.tool proofs/v014/switzerland_acoustic_continuation.json
python -m json.tool proofs/v014/switzerland_acoustic_substep_blocker.json
python proofs/v014/switzerland_advance_w_phi_discriminator.py
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi pmon -c 1
python -m py_compile proofs/v014/switzerland_advance_w_phi_discriminator.py
python -m json.tool proofs/v014/gpt_switzerland_residual_narrowing_summary.json
git diff --check
rg -n "namelist\.(emdiv|smdiv)|getattr\(namelist, \"(emdiv|smdiv)" proofs/v014/switzerland_advance_w_phi_discriminator.py src/gpuwrf -S
```

The discriminator command did not complete in this worker because `State.zeros` requires a visible JAX GPU backend and `nvidia-smi` reported that it could not communicate with the NVIDIA driver. The manager then reran the proof outside the sandbox with GPU visible; that pre-fix run failed before comparison with `AttributeError: OperationalNamelist has no attribute emdiv` in `_run_one_substep`.

I patched the proof harness to use the production defaults from `_acoustic_scan`/`acoustic_substep_core`: `emdiv=0.01` and `smdiv=0.1` via `getattr`. The manager reran the fixed discriminator outside the sandbox on GPU. It completed and wrote `proofs/v014/switzerland_advance_w_phi_discriminator.json`; the WRF-coupled surface-`w` variant has identical `p/ph` interior RMSE and `0.0` improvement fraction, so this candidate is rejected as the first `phi/p` creator. No long GPU gate was started.

## Files Changed

- `.agent/reviews/2026-06-11-v014-gpt-switzerland-residual-root-cause.md`
- `proofs/v014/gpt_switzerland_residual_narrowing_summary.json`
- `proofs/v014/switzerland_advance_w_phi_discriminator.json`
- `proofs/v014/switzerland_advance_w_phi_discriminator.py`

No model source files were changed. No commit was made because there is no source fix.

## h36 Gate

Not run. The contract forbids long 72h GPU gates, and a short h36 forecast gate is not justified until a local source fix is proven. Current best h36 evidence remains the prior advdeg path: residual `-21.064285714285717 Pa/cell/h` vs CPU `+5.178443877551032`.

## Runtime / Performance Risk

The proof script is offline/proof-only and has no runtime path cost. A future fix inside `advance_w` should be performance-neutral if it only changes resident array math. The tested deliberate decoupled surface-`w` feed is not the h36 first creator, so it should not be changed as this blocker fix. Any later faithfulness/stability revisit of that source deviation needs a focused two-region smoke gate because source comments document prior Canary terrain instability.

## Next Manager Decision

- Add a minimal WRF dump around `advance_mu_t` and `advance_w` for call `21602`, or extend the proof harness to expose those same internal terms.
- First compare post-`advance_mu_t` inputs: `ww_new`, `muave_new`, `muts_new`, `theta_coupled`, `w`, `ph_tend`, `rw_tend`.
- If those match, split `advance_w_wrf()` terms: `rhs_seed`, `wdwn`, `rhs_after_phi_adv`, `w_pre_solve`, `w_fwd`, `w_solved`, `ph_next`.
- Do not spend the next sprint on the `current_decoupled` vs `wrf_coupled` surface-`w` variant; the GPU discriminator rejected it as the first `phi/p` creator.
- If `ph_next` matches WRF but `p` does not, move downstream to `calc_p_rho_step`.
- Do not rerun Switzerland 72h.
- Do not spend the next sprint on boundary cadence/advection unless new intra-acoustic evidence contradicts this narrowing.
- No Fable/Mythos escalation is needed before this single discriminator is run.

## Handoff

- objective: independently diagnose and narrow the remaining Switzerland/Gotthard h36->h37 dry-mass/PSFC blocker.
- files changed: this review, `proofs/v014/gpt_switzerland_residual_narrowing_summary.json`, `proofs/v014/switzerland_advance_w_phi_discriminator.py`, `proofs/v014/switzerland_advance_w_phi_discriminator.json`.
- commands run: listed above.
- proof objects produced: summary JSON, GPU-capable discriminator script, and GPU discriminator JSON.
- unresolved risks: exact `advance_w` subterm not split because current WRF dumps lack post-`advance_mu_t` and intra-`advance_w` terms.
- next decision needed: emit the minimal WRF intra-`advance_w` dump for call `21602` or build an equivalent JAX-side term splitter against WRF anchors.
