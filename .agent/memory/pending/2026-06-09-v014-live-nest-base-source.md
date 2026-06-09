# Pending Memory: V0.14 Live-Nest Base Source

Status: pending promotion after direct grid-field/falsifier follow-up.

Lesson:

- Native live-nest base initialization is now ported for child replay/nested
  loading and closes HGT/PB/MUB/PHB source parity against CPU-WRF h0 as a
  validation oracle to formula-level residuals.
- Total-state parity improves materially on the target patch as well: P_TOTAL
  `1080.49` -> `33.43` Pa, MU_TOTAL `1038.05` -> `12.30` Pa, and PH_TOTAL
  `878.03` -> `0.0938`.
- This is not grid/V10 closure. Do not resume TOST or claim grid parity from a
  base-state proof alone; require an init-override/direct grid-field proof.
- Dynamic P/MU perturbation residuals remained visible in the source-fix proof
  and should remain on the active suspect list.
- For tmux Codex/Claude worker done markers, use delayed repeated Enter after
  `tmux send-keys` because a single Enter may leave the marker staged in the
  manager TUI.

Evidence:

- `proofs/v014/live_nest_base_source_fix.json`
- `proofs/v014/live_nest_base_source_fix.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-source-fix.md`
- `.agent/reviews/2026-06-09-v014-debug-method-critic.md`
