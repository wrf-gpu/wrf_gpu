# Agent Instructions

Read order:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. Current sprint contract
4. Relevant skill under `.agent/skills`
5. Only the references needed for the task

For this repository, project-local skills under `.agent/skills` are authoritative and must be committed. Do not use the old global `wrf-gpu-port` skill; it describes a different failed legacy-port project with different architecture assumptions.

## Current Operating Model (2026-05-29, principal-directed)

- **Manager + frontrunner = Opus 4.8.** The manager implements via in-process Opus subagents (high/max effort); the manager dispatches them directly, reviews the diff, runs the acceptance gates, and merges.
- **Verifier / critic / debugger = GPT-5.5 xhigh (codex, tmux).** Invoked sparingly — only before a milestone close, before committing to a major plan, or when stuck. Not a reflexive pair for every implementer sprint.
- **Third model = agy (Gemini 3.5), reactive only**, for stuck/architecture-tiebreak situations after Opus + GPT both failed.
- **Bigger steps**: prefer large coherent sprints / whole milestones with strong end-to-end falsifiable gates over many small chained sprints.
- **Endpoint**: a real WRF v4 GPU port that runs real WRF test fixtures with near-identical results / RMSE on all values, no shortcuts (no masking clamps, no JAX-vs-JAX self-compares, no synthetic happy-paths), GPU-efficient, with massive speedup on this workstation.
- Authoritative roadmap: `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` (M8–M23). Dycore rewrite gating Phase B is the active "F7" work.

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
