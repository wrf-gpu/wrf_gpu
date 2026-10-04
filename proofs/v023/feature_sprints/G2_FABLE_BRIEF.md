# G2 — Moving Nests + adaptive-Δt (Fable-5, X-High, CPU-ONLY)

## Why (intent)
v0.23 roadmap Table-E F-c. WRF's moving-nest capability: the inner nest domain FOLLOWS a feature (storm/vortex) — it re-centers as the feature moves. gpuwrf currently has a v0.22.0 fail-closed SCAFFOLD; build the REAL, WRF-faithful implementation. This is the hardest structural item on the v0.23 feature table (vortex-following forecasts) — that's why it's yours.

## The core to build
Moving nests: (a) a nest-motion driver — prescribed motion + optional feature-following/vortex-tracking per WRF's move criteria; (b) at each move: shift the nest grid, RE-INTERPOLATE the nest state from the parent + carry the overlap region, RE-DERIVE the boundary zone, RE-COUPLE the two-way feedback to the parent; (c) adaptive-Δt (CFL-driven timestep). Match WRF's moving-nest algorithm + feedback conventions.

## CPU-ONLY + validation endpoint (why you can prove this WITHOUT a GPU)
The GPU belongs to another lane — do NOT use it (run everything `JAX_PLATFORMS=cpu`). Everything here is CPU-validatable:
- **Fixture bit-identity/tolerance** vs a CPU-WRF moving-nest run (the moved-nest state after N moves).
- **Conservation**: mass + energy across each move (re-interpolation must conserve).
- **Analytic**: advect a known field through a nest move → moved-nest field matches the analytic solution.
- **Fail-closed**: any unsupported move config errors before compute with a named reason.
Once you have enough to act, ACT — don't over-plan. Only report work you can PROVE with these CPU oracles; show the result that proves it works.

## Do NOT
- Do NOT change the STATIC (non-moving) nest default path — it stays byte-identical; moving is opt-in.
- Do NOT use the GPU. No masking/clamp/nan_to_num, no synthetic happy-paths. Real WRF fixtures + conservation + analytic only.

## Deliverable
Commit on branch `worker/fable/g2-moving-nests`. Write `proofs/v023/feature_sprints/G2_REPORT.md` (the implementation + the CPU-validation proofs: fixture bit-identity, conservation, analytic) and `touch proofs/v023/feature_sprints/G2_DONE`. Report to pane `0:1` (`scripts/tmux_submit.sh 0:1 '<msg>'`) at: design-settled · impl-working · validation-passed. You MAY spawn GPT sub / 2nd-opinion agents to cross-check. If you hit a hard blocker, write `G2_QUESTION.md` + ping `0:1`.
