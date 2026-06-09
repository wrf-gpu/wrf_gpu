# Memory Patch

Scope:

Project-memory update for the v0.14 live-nest base hook sprint.

Evidence:

- `proofs/v014/live_nest_base_hook.json` verdict is
  `NATIVE_PORT_PLAN_READY`.
- No production source was edited.
- The missing d02 base-state split is now source-localized to WRF's live-nest
  initialization chain:
  `med_interp_domain` parent interpolation, generated
  `nest_interpdown_interp.inc`/`interp_fcn_sint`/`sint.F`,
  `blend_terrain`, and `start_domain_em`.
- Native `wrfinput_d02` differs from CPU-WRF h0 by about `1047` Pa `PB` and
  `1050` Pa `MUB` on the target patch.
- WRF base formulas on CPU-WRF h0 terrain reproduce h0 `PB/MUB/PHB` with
  residuals below `0.1` in their native units.
- CPU-WRF h0 is validation-only; production must compute the state natively.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-live-nest-base-hook.md`. After
the native source sprint lands and validates, condense the root-cause/fix into
stable memory.

Reviewer Status:

Pending. Do not promote to stable memory until the production source fix is
implemented or explicitly rejected.
