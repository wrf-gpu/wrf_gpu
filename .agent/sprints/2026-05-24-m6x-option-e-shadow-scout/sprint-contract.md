# Sprint Contract — M6.x Option E: Shadow GPU-WRF Benchmark Scout

## Objective

Evaluate two existing GPU-WRF implementations as **shadow benchmarks** — operational counterfactuals that protect Canary business continuity while the JAX rewrite under B-direct/savepoint-first is held to numerical truth. This is the Option E lane from `.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md`.

**This is a research scout sprint** — no implementation, no commitment, no dependency adoption. Deliverable: a memo comparing **AceCAST** (CUDA/OpenACC commercial WRF GPU port) and **FahrenheitResearch/wrf-gpu-port** (recent OpenACC patch claiming RTX 5090 dynamics-core port) on:
- License / availability / cost
- WRF version compatibility (Canairy is on WRF 4.7.1)
- Canary input compatibility (AIFS-driven d02 3km)
- Validation framing
- Performance evidence on Blackwell-class GPUs (RTX 5090)
- Risk: vendor lock / abandoned project / hidden physics simplifications

Scope is research only. The principal will choose whether to authorize E-lane execution post-scout.

## Non-Goals

- NO code edits anywhere.
- NO purchase / license commitments.
- NO build attempts (research only).
- NO promotion of E over B-direct (E remains parallel insurance).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_escout` on branch `scout/codex/m6x-option-e-shadow-scout`.

Write-only:
- `.agent/sprints/2026-05-24-m6x-option-e-shadow-scout/worker-report.md` (deliverable)
- `.agent/sprints/2026-05-24-m6x-option-e-shadow-scout/shadow_comparison.md` (the comparison memo)

Read-only everywhere else.

## Inputs

1. `.agent/decisions/manager-reflections/PLAN-REFLECTION-2026-05-24-post-consultation.md` (§Option E)
2. `.agent/decisions/blockers/M6-DYCORE-BLOCKER-MEMO.md`
3. `PROJECT_CONSTITUTION.md` (JAX primary; GPU-resident; WRF compatibility)
4. `.agent/references/cpu-wrf-baseline.md` (the Canairy WRF baseline)
5. Web research targets (mandatory citations):
   - AceCAST documentation: `acecast-docs.readthedocs.io`
   - FahrenheitResearch/wrf-gpu-port (GitHub)
   - NVIDIA Blackwell / CUDA capability listing
   - Any WRF mailing-list / NCAR discussion of GPU ports

## Acceptance Criteria

### Part 1: Shadow comparison memo (`shadow_comparison.md`)

| Dim | AceCAST | wrf-gpu-port |
|---|---|---|
| License | ? | ? |
| Cost | ? | ? |
| WRF version supported | ? | ? |
| GPU architecture support (RTX 5090 Blackwell sm_120) | ? | ? |
| Canary 3km d02 + AIFS input compatibility | ? | ? |
| Validation methodology | ? | ? |
| Performance evidence (specific numbers + source URL) | ? | ? |
| Active maintenance | ? | ? |
| Public source code availability | ? | ? |
| Vendor / community support | ? | ? |
| Operational risk (vendor lock / abandoned / physics simplification) | ? | ? |
| Integration cost into the Canairy Gen2 pipeline | ? | ? |

Every cell must cite a URL or "(not stated publicly, marked unknown)".

### Part 2: Strategic recommendation

In `worker-report.md`, ONE of:
- `RECOMMEND-PROCEED-WITH-ACECAST-EVAL` — propose a follow-on sprint to formally evaluate AceCAST (license terms + Canary 3km benchmark).
- `RECOMMEND-PROCEED-WITH-WRFGPUPORT-EVAL` — propose a follow-on sprint to build and benchmark FahrenheitResearch/wrf-gpu-port on the Canary case.
- `RECOMMEND-DEFER-E-LANE` — neither candidate worth pursuing now; document why; specify what would change the calculus.
- `RECOMMEND-BOTH-WITH-PRIORITY-X` — both worth following up; explain priority.

Plus one paragraph dissent against your recommendation.

### Part 3: Risk assessment for the savepoint-harness oracle pipeline

If AceCAST or wrf-gpu-port produces savepoints diverging from CPU WRF, can either be used as a **secondary oracle** that complements the CPU WRF savepoint extraction in M6B0? Answer with a yes/no per candidate and a 1-paragraph rationale.

## Validation Commands

None — research scout.

## Performance Metrics

N/A.

## Proof Object

- `shadow_comparison.md`
- `worker-report.md` (with recommendation + dissent + secondary-oracle assessment)
- Branch `scout/codex/m6x-option-e-shadow-scout`

Time budget: **3–6 hours**. Public research only.

## Risks

- Confabulation: every cell must cite a URL or be marked unknown.
- Spec-gaming: do not recommend a candidate without specific licensing/cost terms.
- Hidden agenda: this sprint must not push E to displace B-direct. E is parallel insurance.

## Handoff Requirements

When both deliverables exist + committed on branch `scout/codex/m6x-option-e-shadow-scout`: `/exit`. Wrapper sends AGENT REPORT to manager pane.
