# Manager Closeout: V0.14 GPT RRTMG Step-1 Forcing Parity

Merge Decision: ACCEPT AND COMMIT AS LOCALIZATION ONLY.

This sprint does not close v0.14 grid parity and does not provide a production
fix. It does, however, turn the secondary RRTMG GLW/RTHRATEN residual from an
open suspicion into a named WRF-anchored boundary:

`RRTMG_STEP1_RESIDUAL_LOCALIZED_TO_CLEAR_SKY_DERIVED_RRTMG_BOUNDARY`.

Accepted artifacts:

- `proofs/v014/rrtmg_step1_forcing_parity.py`
- `proofs/v014/rrtmg_step1_forcing_parity.json`
- `proofs/v014/rrtmg_step1_forcing_parity.md`
- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-step1-forcing-parity.md`

Manager decision:

- Do not block the active NoahMP-focused strict Step-1 attempt on this result.
- Do block final v0.14 strict release unless the RRTMG residual is closed,
  formally demoted by a recorded manager decision, or bounded by stronger WRF
  forcing-hook evidence.
- Next RRTMG sprint, when scheduled, should add a temporary WRF RRTMG forcing
  hook for derived LW/SW columns and compare exact profile/flux/heating arrays
  before production edits.

Roadmap update:

- `.agent/decisions/V0140-RELEASE-CHECKLIST.md` is updated so the active lane
  says "localized, not closed" rather than leaving RRTMG as a vague open item.

