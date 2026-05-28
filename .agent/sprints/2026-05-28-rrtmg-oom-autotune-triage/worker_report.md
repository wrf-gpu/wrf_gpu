# Worker Report — RRTMG OOM Autotune Triage

Verdict: **RRTMG_OOM_PARTIAL** — full-radiation 1h harness is still blocked before RRTMG by a missing runtime helper import.

## Objective

Triage the ~922 MiB / RRTMG XLA autotune OOM class, apply an allowed memory/env fix, and verify full-radiation 1h harness, 24h pipeline, 100-step parity, and speedup.

## Files Changed

- `scripts/run_diagnostic_harness.py` — added process-start default `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
- `.agent/sprints/2026-05-28-rrtmg-oom-autotune-triage/root_cause.md`
- `.agent/sprints/2026-05-28-rrtmg-oom-autotune-triage/worker_report.md`
- `proofs/rrtmg_triage/**`

## Commands Run

- `taskset -c 0-3 python scripts/run_diagnostic_harness.py --hours 1 --jax-platform cuda --output proofs/rrtmg_triage/diagnostic_report_1h_full_radiation.json` — failed at import before RRTMG.
- `taskset -c 0-3 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 24 --output-dir /tmp/rrtmg_triage_24h --proof-dir proofs/rrtmg_triage --run-root /mnt/data/canairy_meteo/runs/wrf_l3 --domain d02` — logged XLA allocation failures, then blocked after hour 1 on nonfinite state.
- Direct `rrtmg_adapter` default allocator probe — reproduced 8.03 GiB BFC OOM.
- Direct `rrtmg_adapter` with `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` — passed with finite theta and active SW/LW diagnostics.
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` — passed.

## Proof Objects Produced

- `proofs/rrtmg_triage/triage_summary.json`
- `proofs/rrtmg_triage/pytest_dycore_100_steps.json`
- `proofs/rrtmg_triage/pipeline_run_20260521.json`
- `proofs/rrtmg_triage/pipeline_1h_autotune_off/pipeline_run_20260521.json`
- `proofs/rrtmg_triage/rrtmg_adapter_xla_cuda_async_probe.json`
- `proofs/rrtmg_triage/nvidia_smi_ac4_24h_default.csv`
- `proofs/rrtmg_triage/nvidia_smi_rrtmg_adapter_default_probe.csv`
- `proofs/rrtmg_triage/nvidia_smi_rrtmg_adapter_xla_cuda_async_probe.csv`

## Unresolved Risks

- Exact HLO pass name was not captured because no XLA dump was produced; evidence identifies XLA GPU compile/autotune/allocator behavior.
- AC3 is blocked by an import compatibility defect outside the writable sprint scope.
- AC4 is blocked by a nonfinite-state failure outside RRTMG memory tuning.
- GPU traces for direct RRTMG probes were partially contended by other workers; pass/fail evidence is valid, but peak attribution is imperfect.
- AC6 speedup is unmeasured because AC4 did not complete.

## Next Decision Needed

Approve a follow-up sprint to fix the diagnostic harness/runtime helper mismatch and the hour-1 nonfinite pipeline blocker, or broaden this sprint's writable scope beyond RRTMG/env tuning.
