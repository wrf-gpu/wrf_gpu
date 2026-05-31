# Systems Invariants — device residency, restart, repeatability, speedup

Honest status of the operational-systems invariants. Several are **NOT_RUN** in the
v0.1.0 validation pipeline (they were not requested for the scored run) and are
marked as placeholders that need a dedicated probe before they can be claimed.

| Invariant | status | evidence | proof path |
|---|---|---|---|
| **24 h speedup vs CPU** | **PASS** (9.09×) | pipeline wall 1794.17 s vs CPU d02 24 h baseline 16305.31 s; target band [4,8]× exceeded | `proofs/v010_validation/speedup_vs_cpu_24h.json` |
| **All-finite over full run** | **PASS** | 24 h coupled real d02 (case3), full physics, **guards OFF**, fp64: `all_finite=true`, `physically_plausible=true`; final state ranges bounded (u ±23, v −18..+14, w −9.5..+1, θ 290–492 K) | `proofs/perf/coriolis_segscan_24h.json` |
| **Device residency (host-loop, length-independent peak)** | **PASS (partial)** | peak GPU mem after FULL 24 h run (10211 MB) ≈ peak after ONE 180-step segment (9048 MB) — host-loop + `block_until_ready` frees each segment's scratch, so peak memory is **independent of forecast length**. Confirms no per-step trajectory accumulation on device | `proofs/perf/coriolis_segscan_24h.json` |
| **Explicit H2D/D2H transfer count inside the step loop** | **[PLACEHOLDER: needs transfer-audit JSON]** | audit harness exists (`proofs/perf/fusion_transfer_audit.py`) but no committed transfer-count artifact was found; the GPU-kernel rule (no timestep-loop host/device transfer) is architecturally enforced via the segmented host-loop but lacks a counted proof object | `proofs/perf/fusion_transfer_audit.py` (script only) |
| **Restart-in-pipeline continuity** | **NOT_RUN** | `--restart-at-hour was not requested`; the restart probe compares final GPU wrfouts (project pickle checkpoints, not WRF `wrfrst`) and was not exercised in this run | `proofs/v010_validation/restart_in_pipeline.json` (status NOT_RUN) |
| **Repeatability (bitwise re-run)** | **NOT_RUN** | `--repeat was not requested`; no repeatability artifact for the scored run | `proofs/v010_validation/repeatability.json` (status NOT_RUN) |
| **wrfout completeness / readability** | **PASS** | 24/24 expected wrfout files written, all readable, 41 minimum variables present | `proofs/v010_validation/wrfout_inventory.json` (status PASS) |
| **Station scoring (obs-vs-GPU)** | **NOT_RUN** | `--score was not requested`; the scored validation used gridded GPU-vs-CPU-WRF truth, not direct station obs (0 joined rows) | `proofs/v010_validation/station_scores_20260521.json` (status NOT_RUN) |

## Notes

- **Why NOT_RUN, not FAIL.** The v0.1.0 d02/d03 validation runs scored gridded
  GPU-vs-CPU-WRF RMSE and did not enable the optional `--repeat` / `--restart-at-hour`
  / `--score` pipeline switches. These invariants are therefore *unmeasured for this
  release*, not *failed*. They should be exercised before any operational-readiness
  claim.
- **Bitwise reproducibility caveat.** The shipped acoustic-substep unroll is OFF by
  default precisely because it is not bitwise-identical (`publish/tables/optimization_refutations.md`);
  a repeatability probe should pin the default-safe configuration.
- **Restart = pickle, not wrfrst.** True WRF-compatible `wrfrst` restart is a v0.2.0
  deliverable (P0-5); v0.1.0 has project pickle checkpoints only
  (`publish/tables/v010_claim_boundary.md`).
