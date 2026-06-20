# GPT Acoustic Stage Pattern Miner

Date: 2026-06-11
Worker: GPT-5.5 xhigh Runner C
Worktree: `<USER_HOME>/src/wrf_gpu2/.claude/worktrees/v014-hpg-native-face-fix`

## Verdict

The remaining Switzerland h36->h37 acoustic blocker is most likely an interior pressure/geopotential acoustic-work refresh problem, with a real but secondary boundary-band component; it is not explained by the accepted HPG native-face fix, by km/diffusion threading, or by time-step cadence alone.

## Proof Objects Used

- Primary proof JSON: `proofs/v014/switzerland_acoustic_substep_blocker.json`
- Generated CPU-only summary: `proofs/v014/gpt_acoustic_stage_pattern_miner_summary.json`
- WRF HPG dump root: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_hpg_native_face/hpg_dumps`
- Hourly output root: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable`
- CPU truth root: `<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`

No source files were edited. No GPU work, Fable interaction, Hermes, Telegram, or `<USER_HOME>/src/canairy_waves` access was performed.

## Evidence Basis

The WRF-native dump inventory is complete for the target stage comparisons: calls `21601` through `21606` each have `24` text rank tiles and `24` binary rank tiles. The sample metadata for call `21601` reports domain IDs `1 129 1 129 1 45`, i.e. `128x128x44` mass dimensions, flags `nonhydro=T`, `top_lid=F`, `specified=T`, and `rdx=rdy=3.33333330e-4`.

The stage-compare proof is aligned at the start: for `sub4_dt18_bcfix`, `base_identity_vs_21601` has exact `mu/p/ph` identity and only tiny diagnostic `al/alt` differences. The eager replica is also production-representative: replica-vs-jit max diffs are about `5.9e-8` for `u/v`, `1.0e-8` for `p_perturbation`, and `4.6e-11` for `ph_perturbation`.

## Stage Pattern

All numbers below are increment errors for `sub4_dt18_bcfix`, split with the existing depth-8 interior mask. Means are `JAX increment - WRF increment`.

| stage comparison | variable | interior RMSE / mean | boundary-band RMSE / mean | full err / WRF-increment RMSE | reading |
|---|---:|---:|---:|---:|---|
| `step1_stage1_vs_21602` | `mu` | `0.0209 / -0.00014` | `4.27 / +0.153` | `4.38x` | mass error is almost entirely boundary-band at the first divergent stage |
| `step1_stage1_vs_21602` | `p` | `6.67 / +0.515` | `102.0 / -1.37` | `61.4x` | pressure is already wrong in the interior even when interior `mu` is near-zero |
| `step1_stage1_vs_21602` | `ph` | `1.51 / -0.362` | `13.0 / +0.728` | `3.65x` | geopotential increment is paired with the pressure error, opposite interior sign |
| `step1_stage1_vs_21602` | `alt` | `1.02e-3 / +2.44e-5` | `2.37e-3 / -1.20e-5` | `1.42x` | `alt/al` are no longer the dominant old HPG bug, but still participate |
| `step1_final_vs_21604` | `mu` | `0.442 / +0.00081` | `11.3 / -0.166` | `14.0x` | boundary mass error remains strong after end-of-step handling |
| `step1_final_vs_21604` | `p` | `4.76 / +2.65` | `14.2 / +1.24` | `15.0x` | pressure error has become a coherent positive interior bias |
| `step1_final_vs_21604` | `ph` | `2.69 / -2.17` | `4.90 / -1.45` | `7.79x` | geopotential has a coherent negative interior bias |
| `step2_stage2_vs_21606` | `mu` | `1.38 / +0.0110` | `4.77 / -0.0953` | `13.1x` | mass/divergence mismatch grows later, but it is not the first interior signal |
| `step2_stage2_vs_21606` | `p` | `9.17 / +2.42` | `91.2 / -0.0247` | `100x` | pressure remains severely wrong inside the depth-8 interior |
| `step2_stage2_vs_21606` | `ph` | `3.21 / -1.51` | `16.9 / -0.0763` | `39.7x` | geopotential remains paired with pressure, again negative in the interior |
| `step2_stage2_vs_21606` | `alt` | `1.49e-3 / +3.05e-4` | `2.24e-3 / +3.59e-4` | `44.7x` | `alt/al` errors are one-signed by this point, but too late/small to explain all p/ph |

Key localization: at `step1_stage1_vs_21602`, `mu` interior RMSE is only `0.0209`, while `p` interior RMSE is already `6.67` Pa and `ph` interior RMSE is `1.51`. A pure LBC or `advance_mu_t` mass-flux explanation would have to explain why the interior pressure/geopotential pair diverges before interior mass does.

## Variant Pattern

| variant | key pattern | interpretation |
|---|---|---|
| `sub4_dt18_bcfix` and `sub4_dt18_cpfix` | Identical mined values for the key p/ph/mu/alt stage rows. | The cp/g/constants path and the boundary candidate do not collapse the remaining p/ph lane. |
| `sub4_dt18_omfix` and `sub4_dt18_final` | Huge boundary-band p/ph contamination: at `step1_stage1`, `p` band RMSE is about `284` and `ph` band RMSE about `127`, while interiors stay close to the `bcfix` interior scale. | These tags prove a boundary-band failure mode exists, but also show that removing the boundary spike does not remove the interior p/ph mismatch. |
| `sub4_phys` (`dt_s=10`, substeps 4) | Still has severe p/ph errors: at `step1_stage1`, `p` interior/band RMSE is `8.15/96.5`; at `step2_stage2`, `p` is `14.3/100.7`. | Matching WRF `dt=18` is not the root; cadence can modulate but not eliminate the error. |
| `acoustic_fix` vs `acoustic_fix_nokm` hourly gate | h37 residuals are `-35.94` vs `-35.86` Pa/cell/h; excess outflux is `-28.819` vs `-28.799`. | `diff_opt=1/km_opt=4` threading is not the h36->h37 gate root. |

## Hourly Output Pattern

The h37/h38 net output mismatch is interior-heavy for mass and surface pressure, not just a lateral ring artifact.

| hour/run | field | interior mean / RMSE | boundary-band mean / RMSE | reading |
|---|---:|---:|---:|---|
| h37 `acoustic_fix` | `PSFC` | `-67.7 / 74.4` | `-35.5 / 65.1` | surface pressure loss is stronger in the interior |
| h37 `acoustic_fix` | `MU` | `-65.3 / 72.0` | `-34.2 / 64.4` | dry mass mirrors PSFC |
| h37 `acoustic_fix` | `PH` | `+39.4 / 50.5` | `+22.1 / 37.3` | geopotential error has the opposite sign to p-stage increments after integration |
| h38 `acoustic_fix` | `PSFC` | `-117.6 / 127.5` | `-66.2 / 101.1` | the interior mass/PSFC error grows by the next hour |
| h38 `acoustic_fix` | `MU` | `-114.4 / 124.0` | `-64.4 / 99.6` | dry-mass loss remains interior-dominant |
| h38 `acoustic_fix` | `PH` | `+70.3 / 86.3` | `+39.8 / 63.0` | p/ph coupling error continues to accumulate |

## Ranked Hypotheses

| rank | lane | evidence | why it fits | why it may be wrong | next falsifier |
|---:|---|---|---|---|---|
| 1 | Pressure/phi refresh placement or `calc_p_rho` work-state semantics | First divergent stage has near-zero interior `mu` error (`0.0209`) but large interior `p`/`ph` errors (`6.67`, `1.51`). By `step2_stage2`, interior `p` mean is `+2.42`, `ph` mean is `-1.51`, and `alt` mean is `+3.05e-4`. | The failure appears in the p/ph/alt diagnostic state before interior mass divergence is large. That points at how acoustic work arrays become persistent `state.p_perturbation`, `state.ph_perturbation`, `al`, and `alt`, or at the `calc_p_rho` denominator/reference path. | The current proof only observes stage boundaries, not each substep's internal pressure solve. A prior stage could feed bad `w/theta/ww` into pressure while `mu` still looks small. | Extend the proof harness to serialize per-substep `calc_p_rho` inputs/outputs (`mu_work`, `muts`, `ph_work`, `theta_work`, `c2a`, `alt`, `al`, `p`, `pm1`), then run: `python proofs/v014/switzerland_acoustic_substep_blocker.py --stage-compare --tag sub4_dt18_calcprho_trace --substeps 4 --dt 18 --steps 2 --capture-intra`. |
| 2 | `rhs_ph` or vertical implicit `advance_w` staging | `ph` interior bias is consistently negative while `p` interior bias is positive at final/stage2; `alt/al` then become one-signed. Existing fresh-`ww`, stage `w_damping`, and work-delta surface `u/v` changes did not collapse this. | Wrong geopotential tendency, wrong `ww` reference, or wrong vertical implicit coefficients can drive `ph` wrong first and force `calc_p_rho` to build a compensating pressure error. | `p` is already badly wrong at stage1; if `calc_p_rho` inputs are wrong before `advance_w`, this lane is downstream. | In the same trace, serialize `rhs_ph_stage`, `rw_tend_stage`, `advance_w` RHS, solved `w`, and `ph_next`. Re-run the same stage command with tag `sub4_dt18_rhsph_advw_trace` and compare first substep where `ph_work` diverges. |
| 3 | Boundary/halo application between stages | Boundary band dominates `mu/muu/muv`: at `step1_stage1`, `mu` band/interior RMSE ratio is `204x`, and `muu/muv` are about `212x/192x`. `omfix/final` show a huge boundary-band p/ph spike that `bcfix` largely removes. | There is unquestionably a boundary-band failure class, and it contaminates full-domain RMSEs. | The depth-8 interior still has large p/ph errors before interior mass divergence. h37/h38 PSFC and MU biases are stronger in the interior than in the band. | Save raw JAX captures or add configurable interior depth/ring summaries, then rerun `python proofs/v014/switzerland_acoustic_substep_blocker.py --stage-compare --tag sub4_dt18_bcfix_rings --substeps 4 --dt 18 --steps 2`. If depth-16/depth-24 p/ph errors remain large, boundary is secondary. |
| 4 | `advance_mu_t` / divergence / `ww` | Later stages do show interior `mu` growth: `step2_stage2` interior `mu` RMSE is `1.38`, and hourly `MU` is strongly negative in the interior. | A wrong divergence or `ww` path can eventually feed the mass budget and dry-mass gate. | It does not explain the first interior p/ph error, where interior `mu` is nearly clean. Fresh stage `ww` also did not collapse the JSON gate. | In the internal trace, compare `dvdxi`, `dmdt`, `mu_work`, `muts`, `muave`, and `ww` after the first acoustic substep. If those match while p/ph fails, demote this lane. |
| 5 | Time-step/substep cadence or diffusion variant | `sub4_phys` (`dt=10`) and `sub4_dt18_bcfix` both fail p/ph stage parity. The missing hourly `dt18` forecast exists as an availability gap, but stage compare already uses WRF `dt=18/substeps=4`. `km` vs `nokm` hourly residuals differ by only `0.0818` Pa/cell/h. | Cadence can affect hourly amplification and should eventually be measured. | It is not the first divergent mechanism in the WRF-native stage proof. | Only after the stage-origin lane is fixed, run the missing forecast gate: `python proofs/v014/switzerland_acoustic_substep_blocker.py --forecast-variant --hours 2 --outdir gpu_output_acoustic_substep_fix_dt18 --forecast-dt 18 --forecast-substeps 4 && python proofs/v014/switzerland_acoustic_substep_blocker.py --analyze`. |
| 6 | Large-step HPG face formula / old hypsometric option / constants-only EOS | `3d0b439c` already collapses native HPG face truth to roundoff-class relative error, and `base_identity_vs_21601` has exact `mu/p/ph`. `cpfix/bcfix` stage tables remain bad. | The old LOG-vs-linear bug and constants mismatch were real fixes. | They are insufficient for the remaining acoustic-stage p/ph increment mismatch. | Keep the accepted HPG/native-face fix, but do not spend the next sprint on another large-step HPG formula pass unless the internal trace shows `al/alt/c2a` diverging while `ph_work/theta_work/muts` match. |

## Fable Next 3 Actions

1. Extend the proof harness, not production source, so `--capture-intra` serializes per-substep pressure, `rhs_ph`, `advance_w`, and `advance_mu_t` summaries; run the `sub4_dt18_calcprho_trace` stage command above.
2. Add configurable ring/depth mining or save raw JAX captures so the same tag can be re-scored at depth 16 and 24; use this to bound the boundary contribution.
3. Defer the missing `dt18` hourly forecast until the first divergent p/ph substep is identified; current JSON already rules out km/diffusion and cadence as primary root causes.

## Commands Run

- `sed -n '1,240p' PROJECT_CONSTITUTION.md`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,260p' <USER_HOME>/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-gpt-acoustic-stage-mismatch-analysis/sprint-contract.md`
- `sed -n '1,260p' <USER_HOME>/src/wrf_gpu2/.agent/skills/validating-physics/SKILL.md`
- `sed -n '1,260p' <USER_HOME>/src/wrf_gpu2/.agent/sprints/2026-06-11-v014-switzerland-acoustic-substep-continuation/manager-handoff.md`
- `sed -n '1,260p' .agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`
- `find proofs/v014 -maxdepth 2 -type f | sort`
- `python -m json.tool proofs/v014/switzerland_acoustic_substep_blocker.json`
- CPU-only Python miners over `proofs/v014/switzerland_acoustic_substep_blocker.json`
- `find <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_hpg_native_face/hpg_dumps -maxdepth 2 -type f | sort`
- CPU-only Python inventory of WRF dump calls and rank-tile counts
- `sed -n` reads of `proofs/v014/switzerland_acoustic_substep_blocker.py`, `proofs/v014/switzerland_hpg_native_face_fix.py`, and relevant acoustic source files
- `git status --short`
- `git diff --stat`
- CPU-only netCDF split miner over existing h37/h38 output files, writing `proofs/v014/gpt_acoustic_stage_pattern_miner_summary.json`

## Files Changed

- Added `proofs/v014/gpt_acoustic_stage_pattern_miner_summary.json`
- Added `proofs/v014/gpt_acoustic_stage_pattern_miner.md`

## Unresolved Risks

- The current JSON does not contain raw JAX captures or per-substep internal arrays; the root lane is therefore localized to a class, not a single source line.
- The existing `--capture-intra` path records some dictionaries in memory but does not serialize enough to answer the next question directly.
- The hourly `acoustic_fix_dt18` output is absent, so the cadence forecast gate remains unmeasured; stage evidence says it should not be the first Fable action.

## Next Decision Needed

Send Fable after reset to instrument the first divergent acoustic substep, starting with pressure/geopotential refresh and `calc_p_rho` internals, while preserving the existing WRF-native stage proof as the acceptance oracle.
