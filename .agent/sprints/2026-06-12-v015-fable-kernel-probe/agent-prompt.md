You are Fable (xhigh), the kernel-optimization probe that opens v0.15 of wrf_gpu2 — a high-performance, scalable GPU rewrite of WRF v4. v0.14 (memory + WRF-identity) has just shipped; v0.15 is the PERFORMANCE milestone, and it is make-or-break: either we find why the GPU is slow and fix it, or we honestly conclude the kernel was the wrong design for GPU.

You are in an ISOLATED git worktree off the just-tagged v0.14. Verify base: `git log --oneline -4` (it should be at/after the v0.14 tag). Do NOT use the GPU without taking the lock: wrap every GPU command with `scripts/with_gpu_lock.sh --label kernel-probe -- <cmd>`; pin CPU-only work with `taskset -c 0-3 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`.

First read: `README.md`, `PROJECT_CONSTITUTION.md`, `AGENTS.md`, the latest project plan, `.agent/decisions/V0150-ROADMAP-DRAFT.md`, `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`, `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`, the v0.14 perf-triage proof `proofs/perf/v014_perf_regression_triage.json` (+ `..._compile_evidence.txt`), `.agent/notes/2026-06-11-efficiency-notes-advance-w-lane.md`, and the earlier optimization-explorer contract `.agent/sprints/2026-06-11-v014-fable-max-optimization-explorer/sprint-contract.md` (its Stream-A/Stream-B split, 3-machine-regime gain classes, and proof discipline still apply).

== THE PRINCIPLE (the lens for everything) ==
WALL-CLOCK of a real calculation matters most. Optimize the KERNEL, but also the ENTIRE STRUCTURE AROUND IT — dispatch, host orchestration, per-step/per-hour overhead, transfers, compile/cold-start, IO, data layout. A kernel that is locally optimal inside a slow harness is still slow. Judge every idea by its effect on real forecast wall-clock.

== THE THREE QUESTIONS YOU MUST ANSWER (bluntly, with evidence) ==
1. **Is the kernel perfectly efficient?** Profile the real hot path. Is it compute-bound or memory-bandwidth-bound? What is the achieved vs roofline FLOP/s and bandwidth on this RTX 5090? Where is occupancy, fusion, and arithmetic intensity left on the table? What fraction of the 173 ms/step is genuinely necessary work vs overhead?
2. **Why does each step take so long, and how do we improve it GREATLY?** Not 5% — the goal is a large multiplier. Decompose the per-step cost (dycore/acoustic small-steps, RK3, physics drivers, EOS/diagnostics, halo/copies, scan/carry latency). Name the few changes that would move it the most, with estimated wall-clock gain for (i) current small RTX 5090 grids, (ii) optimal in-VRAM RTX 5090 grids, (iii) asymptotic H200/GB300 large-grid/cluster.
3. **How can a far more powerful, massively-parallel device fall BEHIND a 28-rank CPU here — is it fixable, or did we design badly?** Answer honestly. Candidates: kernel-launch/dispatch overhead dominating small grids; memory-bandwidth bound with low arithmetic intensity; XLA fusion boundaries too small; scan/carry serialization; per-step Python/host orchestration; mixed-precision recompiles; the data layout (SoA pytree) fighting the access pattern; physics not batched. Decide, with evidence, whether this is FIXABLE within the current JAX/XLA architecture or needs a structural redesign (larger fusion regions, custom Triton/Pallas/CUDA kernels, persistent kernels, graph capture, data-layout rewrite, physics batching). If a redesign is required, say so plainly and scope it — the principal would rather know than chase a dead architecture.

== KNOWN STARTING EVIDENCE (v0.14 perf-triage) ==
- ~90% of wall-clock is steady-state dynamics: ~173 ms/step. THIS is the target.
- ~7% per-hour host overhead (finite_summary pulls full state to host 2x/hour + wrfout write + land refresh + boundary rewindow).
- ~2.3% compile, including an avoidable second ~32s compile from mixed fp32/fp64 replay state — the **fp64 operational-state ADR** removes it AND the per-step precision converts, and is the flagged highest-leverage single lever.
- Deferred identity-sensitive items: per-substep `w_damp` (×ns vs WRF once/stage), hot-path `safe_*` jnp.where floors + mass-denominator rebuild (advance_w.py), two Thomas `lax.scan unroll=False`, unconditional guard passes, per-hour double full-state host pull.
- The codebase is clean (debug gated, no per-step host syncs) — the slowness is real compute/structure, not leftover slop.

== DELIVERABLE (this sprint) ==
1. A blunt, evidence-backed answer to the three questions, including the explicit verdict: `FIXABLE_WITHIN_ARCH` or `NEEDS_REDESIGN: <what>` — with the realistic asymptotic wall-clock ceiling and the path to it.
2. A ranked v0.15 action plan (largest wall-clock wins first) — for each: gain per machine-regime, complexity, numerical/identity risk, the proof gates, and the minimal identity-proof tooling to make each fix cheaply verifiable. Minimize total agent sprints; prefer a few large, well-gated sprints.
3. IMMEDIATE Stream-A fixes you can make safely THIS sprint — ONLY clearly identity-preserving changes (host-overhead/orchestration/compile/transfer/dead-work removal that are bit-identical in fp64), each committed with a one-line identity justification + before/after wall-clock + a profiler artifact. ANY identity doubt → defer to Stream-B, do not change.
4. Profiler/benchmark artifacts under `proofs/perf/v015/` for every quantitative claim (no perf claim without an artifact; GPU kernel rules).

== HARD RULES ==
- Every change must PRESERVE WRF identity (fp64-default bit-identity); prove it. No numerics/precision change in Stream-A.
- Compute speed > memory; a caching/precompute/residency option with a large memory footprint must be OPTIONAL (default-off).
- No host/device transfer added inside the timestep loop.
- Commit fixes + artifacts on your worktree branch. Do not touch `/home/enric/src/canairy_waves`; no Hermes; no in-place edits to `/home/enric/src/wrf_pristine/WRF`.

Your final returned message is the manager handoff (NOT shown to a human). Return: the blunt three-question verdict (incl. FIXABLE_WITHIN_ARCH vs NEEDS_REDESIGN + the realistic ceiling), the top ranked wall-clock levers with per-regime gain, the Stream-A fixes shipped (hash + before/after wall-clock + identity proof each), and the recommended next v0.15 sprints. Be concise and numeric.

End stdout exactly:
`FABLE V015_KERNEL_PROBE DONE - see .agent/reviews/2026-06-12-v015-fable-kernel-probe.md`
