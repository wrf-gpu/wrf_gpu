# Sprint Contract — M6b Failure Step-Back Critic (codex GPT-5.5)

## Objective

M6b honest 1h Canary forecast returned **BLOCKER** with two distinct failures:

1. **Theta-bounds failure at 10s** on all 3 real Gen2 d02 runs — operational-mode (strict-subset carry per §14.5.1 Amendment #1's Undecided default) diverges immediately, even though validation-mode M6B6 passed 0.0 bitwise on 10 timesteps with physics + boundary on.
2. **Nsight first-step D2H=53** — 53 device-to-host transfers in the first step alone, violating PROJECT_CONSTITUTION's no-H2D/D2H-in-timestep-loop rule.

Per principal directive ("if stuck, take a step back; ask GPT about ideas and plan changes"), this critic sprint stress-tests the right fix path **before** committing 30-60 min to an implementation sprint. The carry-creep trade was predicted in §14.5.1 + Critic Amendment #1 (validation/operational separation); now it has empirical data.

## Non-Goals

- NO code edits.
- NO ADR promotion.
- NO sub-sprint dispatch.
- NO re-opening of the B-direct ladder (its bitwise parity is real and locked).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_failcritic` on branch `critic/codex/m6b-failure-step-back`.

Write-only:
- `.agent/sprints/2026-05-25-m6b-failure-critic/reviewer-report.md`

Read-only everywhere else.

## Inputs (mandatory)

1. `.agent/sprints/2026-05-25-m6b-honest-1h-canary/worker-report.md` (the BLOCKER report)
2. All `.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_*.json/*.txt` (per-run finiteness + Nsight + RMSE)
3. `.agent/decisions/ADR-026-operational-mode-design-PROPOSED.md` (the operational-mode design that failed)
4. `src/gpuwrf/runtime/operational_mode.py` (the operational-mode entry point — find the carry shape it uses)
5. `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/worker-report.md` (validation-mode added WRF scratch; classified all Undecided for operational per Amendment #1)
6. `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/worker-report.md` (validation-mode 0.0 bitwise on 10 steps physics+boundary)
7. `PROJECT_PLAN.md §14.5 + §14.5.1 + §14.5.2`
8. `feedback_gpu_optimized_core_primacy.md` (memory — bitwise pilots OK if optimizable; if incompatible with GPU-optimized core, solution is wrong)

## Acceptance Criteria

`reviewer-report.md` (2000–4000 words) with **5 sections**:

### §1 — Diagnosis of the theta-bounds failure

Two competing hypotheses:
- **Hypothesis A (carry-required)**: operational mode is missing WRF scratch families (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `_save`). Without them, the small-step recurrence drifts; theta breaches bounds in <10s. **Action**: promote those fields from Undecided → Operational-required-with-Tier-4-evidence (the M6b failure IS the evidence).
- **Hypothesis B (composition-bug)**: operational mode has a localized composition bug (e.g., wrong RK stage interleaving, wrong physics tendency timing) that validation mode's per-operator parity didn't catch because validation tested operators in isolation. **Action**: bisect operational vs validation; find the divergence; localize fix.

Pick the more likely hypothesis based on evidence. Cite specific proof_*.json contents. Recommend the bisection strategy.

### §2 — Diagnosis of the D2H=53 failure

`grep -rn 'device_get\|jax.device_get\|.tolist()\|.item()\|float(.*\[\|host_callback' src/gpuwrf/runtime/operational_mode.py src/gpuwrf/runtime/cpu_wrf_baseline.py`. Document the offending calls. Are they inside the `lax.scan` body (loop) or outside (one-shot setup)? Recommend fix.

### §3 — Does Hypothesis A (carry expansion) cap max speed?

This is the principal directive's central concern. If we add `t_2ave/ww/muave/muts/ph_tend/_save` to operational carry:
- Carry size grows from ~14 fields to ~20 fields (≈ +40%)
- Memory traffic per timestep grows proportionally
- Max GPU-bandwidth-bound speedup is roughly 1/(carry_growth)
- Does this still leave room for ≥8-10× M7 target on RTX 5090?

Argue with specific numbers: 14 fields × n_cells × 8 bytes per timestep vs 20 fields × n_cells × 8 bytes. Per Canairy d02 (159×66×44 = 461 K cells), what's the bandwidth budget? RTX 5090 peak ≈ 1.8 TB/s. Estimate operational timestep memory traffic. Estimate launch overhead. **Is carry expansion fatal to the value-proposition or a marginal cost?**

### §4 — Recommended next sprint

ONE of:
- **`FIX-CARRY-EXPANSION`**: add the 6 scratch families to operational carry; promote to Operational-required-with-Tier-4-evidence; re-run M6b. Plus D2H lift. Single sprint.
- **`FIX-COMPOSITION-BUG`**: bisect operational vs validation to find the localized composition bug; fix; re-run M6b. Plus D2H lift. Single sprint.
- **`SPLIT-INTO-TWO-SPRINTS`**: do carry expansion + D2H lift in two separate sprints to keep change scope tight.
- **`PIVOT-OUT-OF-OPERATIONAL`**: validation-mode parity is so tight (0.0 bitwise) that operational-mode is the wrong frame; ship validation-mode as production (slow but correct) and pursue speed at M7. (Steelman this and reject if appropriate.)

Specify the sprint contract scope in 1-2 paragraphs.

### §5 — Updated kill gates

§14.5.2 said "if operational mode cannot beat 28-rank CPU WRF within 2 perf-design sprints despite passing Tier-4, the project re-opens whether savepoint-first was right." We now have:
- savepoint-first IS right (validation 0.0 bitwise)
- BUT operational mode fails Tier-4 first time tested

Does the §14.5.2 kill gate fire? Or is the carry-expansion fix the answer that was implied all along (per Amendment #1's "Tier-4 evidence" rule — M6b is the evidence)?

Recommend whether to update §14.5.2 OR keep it as written.

Plus one paragraph dissent against your recommendation.

## Validation Commands

None — read-only critic.

## Performance Metrics

N/A.

## Proof Object

- `reviewer-report.md` (2000–4000 words, file:line citations)
- Branch `critic/codex/m6b-failure-step-back`

Time budget: **60–90 min**.

## Risks

- Spec-gaming: every diagnosis cites proof_*.json. Memory bandwidth claims cite RTX 5090 specs.
- Premature carry-expansion: if Hypothesis B is more likely than A, don't recommend adding scratch families without bisection evidence.
- Sunk-cost: don't conclude that ladder work is wasted; it's the baseline.

## Handoff Requirements

Commit + `/exit`. Manager reads `reviewer-report.md`, then dispatches the recommended fix sprint.
