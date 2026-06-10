# Memory Patch Proposal

## Scope
No code memory patch in this sprint. This was a validation-gate and run-launch
sprint.

## Evidence
The gate update requires memory/resource logging for every long CPU and GPU
validation run. The Switzerland 72h CPU baseline was launched with
`scripts/monitor_resource_usage.sh --no-gpu`, producing process and system
memory CSVs under:

`/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/resources/`

## Proposed Destination
Keep the resource-CSV requirement in:

- `.agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `docs/GPU_RUNBOOK.md` for GPU jobs

## Patch
No source patch proposed. Future memory-related release proof is the final
exact-branch memory preflight plus the long-run resource CSVs for Switzerland
and Canary.

## Reviewer Status

Reviewer Status:

Accepted for this sprint: no code memory change was needed.
