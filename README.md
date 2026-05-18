# wrf_gpu2

This repository is the bootstrap home for a GPU-native, WRF-compatible regional NWP project. The target is not a line-by-line WRF port. The target is a modern, professional-forkable system that can reproduce WRF-like physics behavior well enough for operational validation while exploiting whole-state GPU residency.

Initial operational target: Canary Islands 3 km and 1 km daily forecast runs on a Linux workstation with Ryzen 9 class CPU, 96 GB RAM, and an RTX 5090 32 GB GPU.

## What M0 Created

M0 creates the AgentOS: governance files, sprint contracts, role definitions, memory protocol, skill skeletons, validation scripts, and smoke tests. It intentionally does not implement dycore, physics, I/O, or GPU kernels.

## Validate AgentOS

```bash
python scripts/validate_agentos.py
pytest -q
python scripts/repo_status_snapshot.py
```

## Start A Sprint

```bash
python scripts/create_sprint.py backend-bakeoff-001
```

For a new milestone, the manager first writes a milestone plan and gets it reviewed. Then fill the generated sprint contract before any implementation work begins.

## Use Skills

Skills live under `.agent/skills/<skill-name>/SKILL.md`. Agents should read `PROJECT_CONSTITUTION.md`, then `AGENTS.md`, then only the relevant skill and references for the active sprint.

The old global `wrf-gpu-port` skill is explicitly out of scope for this repository.

## Do Not Do Yet

- Do not implement weather model code without a sprint contract.
- Do not lock the backend stack before the M2 bakeoff.
- Do not claim physics correctness without fixture or analytic-oracle evidence.
- Do not claim GPU performance without profiler artifacts.
- Do not expand beyond the Canary v0 scope without explicit approval.
