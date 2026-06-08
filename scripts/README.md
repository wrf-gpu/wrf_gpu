# Scripts

These scripts are intentionally lightweight. They validate the AgentOS and create or check sprint artifacts. They should fail with clear JSON rather than assuming local GPU or agent tooling exists.

GPU validation entrypoints:

- `run_gpu_lowprio.sh` is the supported GPU mutex / low-priority wrapper. Use it instead of any `/tmp/wrf_gpu_run_lowprio.sh` helper.
- `run_powered_tost_n15.sh` is the durable foreground/detached launcher for the powered TOST campaign.

See `docs/GPU_RUNBOOK.md` for lock, log, rc, and hibernation-resume procedures.
