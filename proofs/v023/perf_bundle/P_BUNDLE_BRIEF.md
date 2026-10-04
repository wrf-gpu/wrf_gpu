# P-BUNDLE SPRINT BRIEF — v0.23 CPU-side performance harvest (P2 + P4-Loop1 + P6 [+ P1-gate + P7b])

**Frontrunner:** GPT-5.5 xhigh. **Manager:** 0:1 (pane `0:1`). **Worktree:** off the v0.23 line (branch `worker/gpt/v023-perf-bundle`, base = the F1-integrated v0.23 HEAD). **Type:** perf-core → EACH change BIT-IDENTICAL + objectively gated. **NO GPU** (held by another lane; do NOT take `/tmp/wrf_gpu2_gpu.lock`) — everything here is CPU-buildable + CPU-oracle-gated.

## Read first (AUTHORITATIVE scope)
- `proofs/v023/perf_bundle/PITEM_SCOPING.md` — the Opus re-verify vs v0.22.2: exact sites, the traps, what's already-shipped. This is your spec.
- `proofs/v023/V0230_ROADMAP.md` Table A + the P-ITEM RE-VERIFY addendum.

## Scope — implement in THIS order, each with its bit-identity gate
1. **P6 dead-code / idiom hygiene (zero-risk FIRST):** delete 4.14 dead `_mass_couple_theta_before_advance`; 4.15 `w_solve_core` + its import; 4.16 dead constants in `vertical_implicit_solver.py` (**grep-verify no d02_replay/oracle caller first** — Opus confirmed dead on the production path). Then oracle-gated idiom edits 4.17/4.18/4.19. **DO NOT touch 4.21 donate_argnums — MED risk, investigate-only, leave it.**
2. **P2 launch-count cuts (biggest harvest, bit-identical):** (a) wrap the **7 still-ungated `_safe_floors` sites** — `advance_w.py:498-500`, `calc_p_rho.py:89/98-100`, `small_step_finish.py:17-19`, `rk_addtend_dry.py:194/224/299` — in the existing `_floor_pos()` gate; (b) collapse the per-field finite-guard reductions (`finite_state_guard.py:81-83`) into ONE reduction via **ravel + concatenate** — **NOT `jnp.stack`** (fields differ in shape → stack won't compile).
3. **P4 — Loop-1 ONLY:** vectorize the independent map at `acoustic_wrf.py:830-833` (`jnp.arange`+`where`). **DO NOT touch Loop-2 (`:836-846`) — it's a Thomas sequential recurrence; vectorizing BREAKS bit-identity.**
4. **P1 gate-run (quick):** run the existing CPU equivalence gate `proofs/perf/single_scan_equiv.py` (segmented==single-scan; `segscan_equiv.json` already bitwise-PASS). Wire a selection knob (env, **default OFF** — the default-flip needs a GPU warm-s/step confirm LATER, not this sprint).
5. **P7b (stretch, only if time):** a lower-XLA-effort config-context wrapper on the fused cold compile in `_build_fused_cascade_program` (flags already in cheap_key). CPU cold-compile A/B only; the warm-s/step no-regress confirm is GPU-gated (later).

## Acceptance gates (hard)
- **BIT-IDENTITY on the default path:** every P2/P4/P6 change must be **byte-identical** — prove via the existing CPU oracle (e.g. `tests/test_m6b0r_calc_coef_w_fix.py` for P4) and/or a before/after byte-compare on a small fixture. NO numeric change on the default path. No masking/clamp/`nan_to_num`.
- **NO SUITE REGRESSION:** the full relevant CPU test suite stays green (run what you touch + the nested/dycore subset).
- **MEASURE where CPU-visible:** launch-count (P2), HLO-size/scatter-count (P4), cold-compile time/RAM (P7b) — report the before/after deltas.
- If an item is NOT provably bit-identical or NOT gated → leave it out + flag it (do NOT widen a tolerance to force a pass).

## Deliverable
- Commits on the branch, one per P-item, clear messages.
- `proofs/v023/perf_bundle/P_BUNDLE_REPORT.md`: per item — files changed · bit-identity proof (oracle/byte-compare result) · measured delta · what's GPU-gated-for-later (P1-flip, P7b warm-confirm).
- `touch proofs/v023/perf_bundle/P_BUNDLE_DONE` as the VERY LAST step.

## Back-channel
A bit-identity gate FAILS (non-zero on the default path) or you're blocked → STOP, report to pane `0:1` via `scripts/tmux_submit.sh 0:1 '<msg>'` + append to `proofs/v023/perf_bundle/P_BUNDLE_QUESTION.md`, and keep the unblocked items moving. Begin now.
