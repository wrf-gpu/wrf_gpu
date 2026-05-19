# Critical Review — ADR-002 State Layout

## Decision

Decision: Accept with required fixes

I accept the core M3 state-layout direction: SoA JAX pytrees, C-grid staggering, fp64 reference fields, separate tendencies, and a no-op single-GPU halo call shape are reasonable for the M3 skeleton. I do not accept the proposal as a locked ADR until the fixes below are applied, because the package overstates governance approval, treats synthetic Canary metadata as provenance, and claims more future halo stability than the evidence supports.

## Top Three Structural Concerns

1. The ADR frames an irreversible state/halo decision as manager-exercised even though the constitution and architecture policy still require explicit human approval.
2. `GridSpec.canary_3km_template()` records fake or placeholder static-field provenance while ADR-002 says the grid contract carries machine-readable Canary terrain/BC provenance.
3. The halo contract is a good single-GPU stub, but the proposal overclaims that future MPI/GPU-aware exchange can replace the body without dycore caller refactor.

## Findings

1. **Blocker — ADR-002 cannot be treated as locked without explicit human-approval provenance.** `PROJECT_CONSTITUTION.md:16` says irreversible architecture decisions require human approval, and `.agent/rules/architecture-decision-policy.md:13` repeats that rule. The proposal identifies ADR-002 as irreversible and says the manager exercises it via autonomy (`.agent/decisions/REVIEW-codex-ADR-002/proposal.md:7`). That conflicts with the M2 closeout precedent, which explicitly says the manager-autonomy directive does not silently amend the constitution (`.agent/decisions/MILESTONE-M2-CLOSEOUT.md:72`). Required fix: change ADR-002 status to "accepted by manager pending explicit user approval" or record the concrete approval artifact before M3 closeout.

2. **Major — Canary static-field provenance is not evidence-grade.** M3 requires named terrain/geog provenance fields (`.agent/milestones/ROADMAP.md:58`) and the project plan calls terrain/geog correctness a physics-validity risk requiring a provenance file, transform, checksum, and sanity check (`PROJECT_PLAN.md:153`). The implementation currently sets `source_path="data/static/canary_3km_terrain.nc"`, `sha256="analytic-m3-template"`, `coastline_sanity_check_passed=True`, and zero terrain height (`src/gpuwrf/contracts/grid.py:216`, `src/gpuwrf/contracts/grid.py:218`, `src/gpuwrf/contracts/grid.py:223`, `src/gpuwrf/contracts/grid.py:234`). Required fix: either point to a real external static-field object with checksum/sanity proof, or label this as an idealized M3 template and prevent ADR-002 from implying production Canary terrain provenance.

3. **Major — halo future-proofing is asserted, not demonstrated.** The ADR says `apply_halo(state, halo) -> state` can later be backed by MPI/GPU-aware exchange without changing dycore callers (`.agent/decisions/REVIEW-codex-ADR-002/proposal.md:17`), but `HaloSpec` currently carries only `width`, `fields_to_exchange`, and `edge_type` (`src/gpuwrf/contracts/halo.py:11`), and the implementation deletes the halo object and returns state (`src/gpuwrf/contracts/halo.py:28`). This does not yet model rank topology, neighbor direction, stagger-specific slab extents, corners, communicators/streams, or persistent pack buffers. Required fix: narrow the claim to "call-shape placeholder" and require a later halo ADR/experiment before M4/M6 code relies on exchange semantics.

4. **Major — review-sprint contract provenance is incomplete.** The role prompt required `/home/enric/src/wrf_gpu2/.agent/decisions/REVIEW-codex-ADR-002/sprint-contract.md` in the mandatory read order (`.agent/decisions/REVIEW-codex-ADR-002/role-prompts/critical-review.md:11`), but that file is absent from the decision folder. Required fix: add the missing review contract or explicitly record that the role prompt is the contract for this review.

5. **Minor — agent-success evidence is stale after the reject/fix cycle.** The sprint contract records attempt 2 after an attempt-1 reviewer rejection (`.agent/sprints/2026-05-19-m3-state-grid-halo-skeleton/sprint-contract.md:9`), and the worker report says attempt 2 addressed those reject items (`.agent/sprints/2026-05-19-m3-state-grid-halo-skeleton/worker-report.md:3`). `artifacts/m3/agent_success.json` still says `reviewer_rejections_before_handoff: 0` and `sprint_attempt: 1` (`artifacts/m3/agent_success.json:5`). Required fix: regenerate this proof object or remove it from the accepted evidence set.

6. **Minor — the HLO evidence does not prove the full State is carried through the scan body.** The compiled while tuple carries only the counter, `theta`, and `tendencies.theta` (`artifacts/m3/hlo_dump/dummy_loop.txt:92`, `artifacts/m3/hlo_dump/dummy_loop.txt:93`), while other fields are passed around the loop and copied at output (`artifacts/m3/hlo_dump/dummy_loop.txt:112`). This is acceptable compiler pruning for M3, but ADR-002 should not present it as evidence that every prognostic participates in timestep carry. Required fix: phrase the proof as API-level residency plus theta hot-path exercise; add a later real-dycore full-field carry check.

## Dissent

I dissent from the proposal's revisit rule that "outside these three, M4-M7 work on this layout without revisiting ADR-002" (`.agent/decisions/REVIEW-codex-ADR-002/proposal.md:52`). Static-field provenance and halo semantics are not mature enough to close the door that tightly. The layout can move forward, but the ADR should preserve an explicit escape hatch for M4 dycore boundary work and the first real Canary static-field ingestion.

## Closing Recommendation

Merge the ADR only as **Accept with required fixes**: update approval status, correct or re-scope static provenance, narrow the halo claim, restore review-contract provenance, and refresh stale agent-success evidence. If those are applied, no M3-S2 implementation sprint is required for the core SoA/C-grid/fp64 state layout.
