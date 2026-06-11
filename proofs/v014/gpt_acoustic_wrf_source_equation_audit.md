# GPT Acoustic WRF Source Equation Audit

Date: 2026-06-11
Worker: GPT-5.5 xhigh Runner B
Worktree: `/home/enric/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

## Verdict

The most likely remaining p/ph stage-increment mismatch source is the real-case geopotential tendency lane: WRF `rk_step_prep/calc_ww_cp` stage omega plus `rk_tendency/rhs_ph`, especially the specified-boundary/map-factor/higher-order horizontal-advection behavior that the current JAX implementation still documents as periodic/idealized. The dirty acoustic candidate fixes several plausible WRF-staging issues, but it leaves this lane structurally non-WRF for the Switzerland specified-boundary case, and the proof JSON shows p/ph already leading the failure at the first stage boundary.

This is an equation/source audit, not a new GPU proof. I did not edit source, run GPU, interact with Fable, use Hermes, or touch `/home/enric/src/canairy_waves`.

## Proof Context

Main proof object: `proofs/v014/switzerland_acoustic_substep_blocker.json`, tag `sub4_dt18_bcfix`.

Relevant verifier: `.agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md:15-43`.

The rejected candidate does not collapse the hourly gate:

| gate | key result |
| --- | ---: |
| CPU h36->h37 residual | `+5.178443877551032` Pa/cell/h |
| HPG native-face `3d0b439c` residual | `-27.697448979591826` Pa/cell/h |
| acoustic fix residual | `-35.94119897959182` Pa/cell/h |
| collapse vs old | `-0.007140946777213886` |
| collapse vs hypso | `-0.017330577730950925` |

The WRF-native stage comparison starts from the same state:

| field | base identity max abs vs WRF call 21601 |
| --- | ---: |
| `mu` | `0.0` |
| `p` | `0.0` |
| `ph` | `0.0` |
| `alt` | `3.1024941709034692e-06` |
| `al` | `5.345002354029127e-05` |

But the first stage already has p/ph mismatches out of scale with the WRF increment:

| comparison | field | JAX-vs-WRF increment RMSE, interior | WRF full increment RMSE | state RMSE, boundary band |
| --- | --- | ---: | ---: | ---: |
| `step1_stage1_vs_21602` | `mu` | `0.020896037745516495` | `0.4719360436558548` | `4.271408058360981` |
| `step1_stage1_vs_21602` | `p` | `6.668548991178234` | `0.8097546961307533` | `101.99669360089867` |
| `step1_stage1_vs_21602` | `ph` | `1.5146743040739983` | `1.7647076515968216` | `13.003568659113805` |
| `step1_final_vs_21604` | `p` | `4.764266613533931` | `0.5338511484290017` | `14.165135892033643` |
| `step1_final_vs_21604` | `ph` | `2.6910486370901605` | `0.42881522285982804` | `4.899549928359627` |
| `step2_stage2_vs_21606` | `p` | `9.167104987950042` | `0.446858712567686` | `91.40920613180404` |
| `step2_stage2_vs_21606` | `ph` | `3.210614796377422` | `0.21775792405383618` | `17.585634657547203` |

Interpretation: the initial p/ph state is not the bug. The first post-stage mismatch is p/ph-first, with `mu` still small relative to its own WRF increment. That points to the geopotential/pressure tendency and acoustic w/phi loop before it points to post-wrapper boundary relaxation or standalone `advance_mu_t`.

## WRF/JAX Source Comparison

| Lane | WRF source | Current JAX / dirty candidate source | Audit note |
| --- | --- | --- | --- |
| RK prep builds stage transport and `ww` | `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:658-672`; `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:640-782`, especially `MUU/MUV` at `696-708`, `divv` at `747-748`, `ww` recurrence at `775` | `src/gpuwrf/runtime/operational_mode.py:1857-1887`, `2415-2437`; `src/gpuwrf/dynamics/flux_advection.py:171-247` | Dirty candidate now feeds fresh `stage_velocities.rom` into `small_step_prep`, but `rom` is built by `couple_velocities_periodic`, with periodic face collapse and periodic face masses (`flux_advection.py:9-15`, `105-114`, `201-224`). WRF real specified domains use source loop bounds and non-periodic edge semantics. |
| `rk_tendency` computes `ph_tend` with `rhs_ph` | `/home/enric/src/wrf_pristine/WRF/dyn_em/module_em.F:668-680`; `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:1365-2178`; specified/order setup at `1432-1435`; vertical and gw terms at `1457-1507`; 2nd-order boundary trims at `1518-1584`; 6th-order specified/open degradation at `1768-2072` | `src/gpuwrf/runtime/operational_mode.py:1397-1429`; `src/gpuwrf/dynamics/core/rhs_ph.py:1-195` | This is the strongest mismatch. JAX `rhs_ph_wrf` states its scope as unit/periodic, `phi_adv_z==1`, 2nd-order horizontal advection, map factors and higher-order branches deferred (`rhs_ph.py:38-45`). The Switzerland case is specified-boundary real terrain; WRF `rhs_ph` uses map factors, configured `h_sca_adv_order`, and boundary/order degradation. |
| Small-step prep, `calc_p_rho`, `calc_coef_w` | `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1090-1136`; `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:125-290`, `438-568`, `570-652` | `src/gpuwrf/dynamics/core/small_step_prep.py:190-278`; `src/gpuwrf/dynamics/core/calc_p_rho.py:95-173`; `src/gpuwrf/runtime/operational_mode.py:1767-1791` | Candidate constants and `c2a`/`calc_coef_w` are directionally WRF-faithful, and base identity is exact for `p/ph/mu`. This lane can amplify ph error into p/al/alt, but it is less likely to be the first source after the first-stage p/ph signature. |
| `advance_mu_t` mass/theta/omega update | `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1393-1410`; `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:969-1175`, loop trims at `1048-1063`, divergence at `1094-1099`, `ww` update at `1109-1121` | `src/gpuwrf/dynamics/core/acoustic.py:668-695`; `src/gpuwrf/dynamics/mu_t_advance.py:81-86`, `245-389` | JAX has a specified/nested branch and matches the broad WRF staging. It remains relevant because it updates the small-step `ww`, but the first-stage `mu` increment error is much smaller than the p/ph error. |
| `advance_w` implicit w/ph update | `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:1500-1518`; `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F:1178-1469`, loop trims at `1265-1282`, `ph_tend` fold at `1312-1318`, small-step phi advection at `1340-1368`, surface w BC at `1383-1394`, ph finish at `1460-1464` | `src/gpuwrf/dynamics/core/acoustic.py:745-819`; `src/gpuwrf/dynamics/core/advance_w.py:229-532` | This is the second most plausible source. Dirty candidate fixed the work-delta `u/v` feed (`core/acoustic.py:745-775`), but `advance_w_wrf` still uses edge-padded terrain differences (`advance_w.py:372-401`) rather than WRF's specified/nested trimmed source loops. If `rhs_ph` and stage `ww` prove correct, this is next. |
| Post-RK pressure refresh `calc_p_rho_phi` | `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F:3049-3059`; `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F:953-1080`, LOG-form hypso opt 2 at `1033-1049`, EOS at `1061-1076` | `src/gpuwrf/runtime/operational_mode.py:1642-1676`; constants in `src/gpuwrf/dynamics/acoustic_wrf.py:37-56`, `src/gpuwrf/integration/d02_replay.py:164-167`, `src/gpuwrf/coupling/boundary_apply.py:62` | HPG native-face already fixed the real-case hypsometric opt 2 path, and the candidate fixed `cp=1004.5` / `g=9.81`. `step1_stage3_raw` and `step1_final` have identical interior p/ph increment errors, so the final wrapper is not the primary interior creator. Pressure refresh remains an amplifier, especially for `alt`. |

## Ranked Hypotheses

| Rank | Hypothesis | Why it fits | Why it may be wrong | Next falsifier |
| ---: | --- | --- | --- | --- |
| 1 | Real-case `rhs_ph` plus stage `calc_ww_cp`/transport velocity semantics are still non-WRF. | Fits p/ph-first signature from `step1_stage1_vs_21602`. Dirty candidate threads fresh omega, but fresh omega comes from `couple_velocities_periodic`, and `rhs_ph_wrf` is explicitly periodic/2nd-order/unit-map scoped. WRF Switzerland is specified-boundary and uses source loop trims, map factors, and configured order/degradation. | Need term-level `ww` and `ph_tend` WRF/JAX dumps; current proof compares stage states, not `rhs_ph` subterms. | Capture WRF `grid%ww` after `rk_step_prep/calc_ww_cp` and `ph_tend` immediately after `rhs_ph`, then compare against JAX `stage_velocities.rom` and `rhs_ph_wrf` for call 21602. |
| 2 | `advance_w` specified-boundary/terrain-surface implementation still diverges. | `advance_w` directly consumes `ph_tend` and writes ph. WRF trims specified/nested loops and uses terrain differences inside those bounds; JAX surface BC uses edge padding. Dirty work-delta `u/v` fix did not collapse the gate, so the broader source loop behavior remains plausible. | If `ph_tend` entering `advance_w` is already wrong, `advance_w` will merely propagate it. | If rank 1 `ww/ph_tend` matches, add first-substep captures around `advance_w` RHS, surface `w(1)`, solved `w`, and `ph` before `calc_p_rho_step`. |
| 3 | Pressure refresh / `calc_p_rho_phi` / EOS placement amplifies a correct ph into wrong p/alt. | `alt` becomes very large at `step1_final_vs_21604` (`0.0034308430124102103` vs WRF increment `5.7115380770404754e-05`). `calc_p_rho_step` and post-RK `calc_p_rho_phi` convert ph into p/al/alt. | Base `p/ph/mu` identity is exact, hypso opt 2 was already fixed, and first-stage ph is already wrong before final refresh can explain everything. | Compare JAX `ph` fed to `calc_p_rho_step` and post-stage `calc_p_rho_phi` against WRF. If ph matches but p does not, demote rank 1 and promote this. |
| 4 | Boundary/halo handling is an amplifier. | Boundary-band p errors are huge (`101.9967` at `step1_stage1_vs_21602`) and hourly mass residual is a budget/outflow symptom. | Interior increment errors are already large before final wrapper, and `step1_stage3_raw` vs final keeps identical interior p/ph increment errors. | Re-run stage compare with interior-only term captures and a boundary-strip mask; require the first divergence location before changing boundary code. |
| 5 | Standalone `advance_mu_t` mass/omega update mismatch. | `advance_mu_t` updates small-step `ww`, `mu`, and `theta`, all feeding `advance_w` and pressure. | JAX has a real specified/nested path; first-stage interior `mu` error (`0.020896`) is small while p/ph are already many times WRF increment scale. | Compare `dvdxi`, `dmdt`, `mu`, and post-`advance_mu_t` `ww` for the first acoustic substep after rank 1 is isolated. |
| 6 | Time-step cadence, w damping, diffusion, constants. | Candidate touched these because they were real WRF deviations: `cp/g`, fresh stage omega, stage-level `w_damping`, work-delta surface BC, and `diff_opt/km_opt`. | The current proof uses `dt=18`, `substeps=4`, `hypsometric_opt=2`; rejected candidate still worsened residual. These are not the most likely remaining root. | Keep as gate variables, not the next source edit. Re-test only after rank 1/2 term parity improves. |

## Next Falsifier Commands

These are CPU/file-inspection commands safe under the current no-GPU audit constraint:

```bash
python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path("proofs/v014/switzerland_acoustic_substep_blocker.json").read_text())["stage_compare"]["sub4_dt18_bcfix"]
print("config", r["config"])
print("base", {k: r["base_identity_vs_21601"][k].get("max_abs") for k in ("mu", "p", "ph", "alt", "al")})
for comp in ("step1_stage1_vs_21602", "step1_final_vs_21604", "step2_stage2_vs_21606"):
    c = r["comparisons"][comp]
    print(comp)
    for f in ("mu", "p", "ph", "alt", "php"):
        v = c[f]
        band = v["state_err"].get("band", {})
        print(f, "incr_i", v["incr_err"]["interior"]["rmse"], "wrf_full", v["wrf_incr"]["full"]["rmse"], "band", band.get("rmse"))
PY
```

```bash
nl -ba /home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F | sed -n '640,782p;1365,2178p'
nl -ba /home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F | sed -n '1178,1469p'
nl -ba src/gpuwrf/dynamics/core/rhs_ph.py | sed -n '1,210p'
nl -ba src/gpuwrf/dynamics/flux_advection.py | sed -n '1,120p;171,247p'
nl -ba src/gpuwrf/dynamics/core/advance_w.py | sed -n '229,532p'
```

The decisive next proof requires diagnostic instrumentation and a GPU run, so I did not run it in this audit. The next worker should add capture-only probes around WRF `calc_ww_cp`, WRF `rhs_ph`, JAX `_stage_transport_velocities`, and JAX `rhs_ph_wrf`, then run:

```bash
python proofs/v014/switzerland_acoustic_substep_blocker.py --stage-compare --tag sub4_dt18_terms --dt 18 --substeps 4 --steps 2 --capture-intra
python proofs/v014/switzerland_acoustic_substep_blocker.py --analyze
```

Acceptance discriminator:

1. If `stage_velocities.rom` differs from WRF `grid%ww` before `rhs_ph`, fix `calc_ww_cp` real-boundary semantics first.
2. If `rom` matches but `ph_tend` differs, implement WRF `rhs_ph` real-case branches and map-factor/order semantics.
3. If `ph_tend` matches but post-`advance_w` ph differs, move to `advance_w` loop bounds, terrain surface BC, and Thomas-solve subterms.
4. If ph matches but p/alt differs, move to `calc_p_rho_step` and `calc_p_rho_phi` refresh.

## Handoff

Objective: identify the most likely remaining p/ph stage-increment mismatch source after the rejected acoustic candidate by auditing WRF v4 source equations against current JAX and the dirty local candidate.

Files changed: `proofs/v014/gpt_acoustic_wrf_source_equation_audit.md`.

Commands run: source/proof inspection only with `rg`, `nl -ba`, `git diff`, and Python JSON summarizers; one initial summarizer failed on an older boundary-band key assumption, then the corrected summarizer above succeeded. No GPU command was run.

Proof objects produced: this report.

Unresolved risks: no term-level WRF/JAX `ww` or `ph_tend` arrays exist in the current proof JSON, so rank 1 is a source-and-signature inference rather than a direct term parity proof.

Next decision needed: instrument and compare `calc_ww_cp`/`rhs_ph` term arrays before another source fix is attempted.
