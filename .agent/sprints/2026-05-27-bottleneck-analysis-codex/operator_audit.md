# Operator-Level Audit

Contract caveat: the requested Opus artifacts are missing in this worktree. `.agent/sprints/2026-05-27-m7-d2h-probe-opus/` contains only `sprint-contract.md`; there is no `top_3_suspects.md` or `operator_map.json`. The sibling codex D2H probe directory also contains only its contract. Therefore the labels `S1`, `S2`, and `S3` cannot be bound to exact Opus callsites here.

The audit below maps the current profile to the three strongest optimization classes that likely correspond to those parked fusion candidates. If Opus later defines different S1/S2/S3 meanings, these estimates must be remapped.

| Candidate | Real optimization win? | Evidence | Realistic saving estimate | Risk |
|---|---|---|---:|---|
| S1 assumed: RK/acoustic scan-carry housekeeping, save-family copies, dtype enforcement, finite guards | Yes, highest value | `loop_add_fusion_4`, `loop_multiply_fusion`, and `loop_subtract_fusion` together account for 546.861 ms of GPU kernel time and ~626k tiny launches in a warm 1h trace. CUDA launch API time is 3.815 s. | 1.5-3.0 s per warm forecast hour if 40-60% of tiny launches and related D2D copies disappear; ~35-70 s per 24h steady run. | Medium. Guards and save-family fields protect current finite behavior, so fusion must preserve Tier-2 invariant checks and M6B parity anchors. |
| S2 assumed: vertical solver coefficient/PCR sequence | Yes, but not first by wall clock | FP64+FP32 PCR family totals 213.759 ms of direct GPU kernel time per warm 1h trace. It is a real computational kernel, not just launch noise. | Direct 0.1-0.4 s per warm hour from solver specialization; up to ~5-12 s per 24h if coefficient temporaries and reverse scans also collapse. | High. Pressure/geopotential/acoustic paths are FP64-sensitive and require savepoint/Tier-4 proof. |
| S3 assumed: physics/boundary column layout and transient pack/unpack | Yes, medium value | `input_transpose_fusion_42` and `input_concatenate_fusion` total 16.533 ms direct kernel time, but they create layout traffic and transients. Physics adapters repeatedly `moveaxis` state fields to vertical-last columns. | Direct 0.1-0.3 s per warm hour; larger memory-pressure reduction possible if full-field tendency copies are also reduced. | Medium-high. Physics skill is already blocked, so layout changes need regression proof against station-skill and Tier-2 water/mass budgets. |

Decision:
- S1 is a real optimization win and should be the first implementation sprint after review.
- S2 is a real but correctness-sensitive optimization; prototype only behind a solver parity fixture.
- S3 is worth planning, but it should follow S1 unless a memory-profile sprint proves transposes dominate 1 km peak memory.

Missing evidence needed:
- Exact Opus `top_3_suspects.md` and `operator_map.json`.
- Nsight Compute metrics for achieved bandwidth, occupancy, and register spills.
- XLA HLO memory/profile dump for the current iter-2 operational one-hour scan.
