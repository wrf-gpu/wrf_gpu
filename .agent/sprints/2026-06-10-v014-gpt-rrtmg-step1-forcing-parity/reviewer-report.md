# Reviewer Report: V0.14 GPT RRTMG Step-1 Forcing Parity

Decision: ACCEPT AS A BOUNDED SECONDARY-RESIDUAL LOCALIZATION.

The sprint stayed within the allowed file set and did not edit production
source, tests, validation launchers, memory, FP32, TOST, Switzerland, or
Grid-Delta code. The report is appropriately conservative: it does not claim a
production fix and does not generalize this Step-1 clear-sky residual into a
blanket RRTMG kernel failure.

What is proven:

- The Step-1 residual is real at the RRTMG output/handoff level:
  `GLW/LWDN` bias `17.44070059852181 W/m2`, mass-coupled `RTHRATEN` max_abs
  `19.425283200182427`.
- The main easy explanations are exonerated: clock/solar geometry,
  NoahMP/surface GLW handoff, gross thermodynamics, cloud occupancy, layer
  ordering, theta conversion, and mass coupling.
- The remaining named boundary is WRF-derived clear-sky RRTMG optical/gas/
  top-buffer profile or downstream kernel boundary.

What is not proven:

- The exact first divergent derived profile or kernel intermediate.
- A safe production patch.
- Final v0.14 RRTMG parity closure.

Recommended next action:

- Do not interrupt the active Fable/Mythos NoahMP lane for this secondary
  residual.
- Before final release, add a temporary WRF RRTMG forcing hook that dumps LW/SW
  derived profiles, optical inputs, fluxes, and heating arrays, then compare
  those columns before applying production code changes.

