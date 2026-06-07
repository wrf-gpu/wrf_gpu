# Naive-AI v0.11 critique — manager triage (2026-06-07, principal-forwarded)

Verdict accepted: publication-worthy NOW as a transparent research-artifact + AI-assisted-scientific-software-process preprint; NOT yet "full WRF replacement / near-perfect port." Framing must stay: JAX-native WRF-COMPATIBLE REIMPLEMENTATION (clean-room), NOT a Fortran-source port; honest about the skill gap. (README already frames this correctly — keep it.)

## ACT AT v0.12.0 RELEASE STEP (tonight, manager-direct, 0-GPU, after gwd8 GREEN)
- **#2 PUBLIC REPRODUCIBILITY/PRIVACY (release-critical):** before pushing to the org repo (~/src/wrf_gpu / wrfgpu remote), run `scripts/verify_reproducibility.sh` and SANITIZE any `/home/enric` path + email leak in the SHIPPED files (proofs/manifest/scripts). Public repo currently leaks 3 files. The .agent/ dev tree does NOT ship (curated subset) — do NOT churn the 728 dev-file hits. This session added many proof JSONs with absolute /home/enric + /mnt/data paths → sanitize the shipped ones.
- **#6 polish:** fix/trim the placeholder `src/gpuwrf/README.md` + `tests/README.md` (early-placeholder text damages first-review confidence). 
- **#5 framing (mostly done):** keep "WRF-compatible reimplementation, not a Fortran port" precise in README.
- **#4 partial (reflect v0.12.0 progress):** v0.12.0 ships the standalone native-init NESTED path (gate runs with NO CPU-wrfout) + prognostic Noah-MP — update any v0.11 "land replayed hourly" wording to reflect this; note the daily_pipeline REPLAY path still exists as one mode.
- **KNOWN_ISSUES limitations (honesty):** single-GPU / single-JAX-stack validation only (#7); forecast-skill gap T2/U10/V10 still open (#1); GWD operational-coupling gated-off (fp64 VRAM @hr7); TOST n=15 deferred (rc=2 daily_pipeline); compile-speed reverted→v0.13; RRTM-LW skeptic-pass→v0.13.

## ADD TO v0.13 ROADMAP (deep, multi-version)
- **#1 Forecast-skill closure (P1, credibility gate):** T2/U10/V10 RMSE regressions — surface-flux over-flux + RRTMG-SW clear-sky T2 bias + theta-guard/land-state. The blocker for any "operational/replacement" claim.
- **#2-full Outsider-runnable reproducibility (P1):** bundle the missing Thompson table assets + scripts so the FULL historical proof collection runs for an external reviewer (not just the 11 public tests); green verify_reproducibility end-to-end.
- **#3 Standard community validation (P2):** WRF/community idealized suite, closed-domain mass/energy budgets, bitwise-restart tests, larger multi-day case corpus.
- **#4-full Remove replay crutches (P1):** fully prognostic Noah-MP (currently a prescribed subset) + live boundary forcing + a clean WRF-compatible IC/BC ingest beyond the v0.12.0 native-init.
- **#7 Multi-hardware + independent reproduction (P2):** 2nd GPU / driver / JAX stack + independent repro (currently one RTX 5090 path).
- Novelty framing for the paper: "first source-open Python/JAX/XLA WRF-compatible regional replay+native-init prototype with whole-state GPU residency + proof-object validation on a consumer workstation" (NOT "first GPU WRF" — AceCAST/NCAR-WSM5/FahrenheitResearch exist).
