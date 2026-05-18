# Agent Instructions

Read order:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. Current sprint contract
4. Relevant skill under `.agent/skills`
5. Only the references needed for the task

For this repository, project-local skills under `.agent/skills` are authoritative and must be committed. Do not use the old global `wrf-gpu-port` skill; it describes a different failed legacy-port project with different architecture assumptions.

## Operating Rules

- Never implement model code without a sprint contract.
- Never mark work done without a proof object.
- Never modify stable memory, skills, rules, or contracts directly. Use the patch protocol.
- Do not edit the same core files as another active worker.
- Freeze interfaces before parallel work begins.
- Physics claims require WRF fixture, analytic oracle, conservation, or ensemble evidence.
- GPU performance claims require profiler artifacts and transfer audits.
- No host/device transfer inside timestep loops unless explicitly approved and documented.
- Architecture changes require an ADR.
- Branches should be purpose-named, for example `worker/gpt/m2-stencil-bakeoff`.
- User-facing reports must be concise, decision-oriented, and honest about missing evidence.

## Handoff Format

Every handoff must include:

- objective
- files changed
- commands run
- proof objects produced
- unresolved risks
- next decision needed, if any
