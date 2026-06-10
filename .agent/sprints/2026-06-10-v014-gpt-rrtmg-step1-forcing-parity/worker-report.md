# Worker Report: V0.14 GPT RRTMG Step-1 Forcing Parity

Summary: GPT-5.5 xhigh completed the CPU-only RRTMG localization sprint and
produced WRF-anchored proof artifacts without production source edits.

Objective:

- Localize the secondary Step-1 RRTMG GLW/RTHRATEN residual while Fable/Mythos
  worked on the primary NoahMP land-tile blocker.
- Keep ownership disjoint from `src/gpuwrf/**`, tests, TOST, Switzerland,
  Grid-Delta, FP32, memory, and NoahMP source/proof files.

Files produced:

- `proofs/v014/rrtmg_step1_forcing_parity.py`
- `proofs/v014/rrtmg_step1_forcing_parity.json`
- `proofs/v014/rrtmg_step1_forcing_parity.md`
- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-step1-forcing-parity.md`

Result:

- Verdict: `RRTMG_STEP1_RESIDUAL_LOCALIZED_TO_CLEAR_SKY_DERIVED_RRTMG_BOUNDARY`.
- GLW/LWDN vs WRF `PRE_NOAHMP`: bias `17.44070059852181 W/m2`,
  RMSE `17.520282676800505`, max_abs `22.521139408469537`.
- Mass-coupled `RTHRATEN` vs WRF part2: max_abs
  `19.425283200182427`, RMSE `2.4884141898276413`.
- SWDOWN midpoint convention remains close: RMSE `2.758969795939516 W/m2`.

Exonerated boundaries:

- clock/solar geometry;
- NoahMP/surface GLW handoff;
- gross thermodynamic state and cloud occupancy;
- layer ordering;
- flux-to-theta conversion;
- mass coupling.

Unresolved boundary:

- WRF-derived RRTMG clear-sky optical/gas/top-buffer profile or downstream
  kernel boundary. A temporary WRF RRTMG forcing hook is required to name the
  first divergent derived quantity before a production fix is responsible.

