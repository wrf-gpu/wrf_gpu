# Total-State to WRF Base/Perturbation Mapping

Sprint: `2026-05-27-m7-netcdf-writer`

WRF history files do not store several dynamic fields as a single total field. They store a perturbation field plus a base-state companion:

| Total physical quantity | WRF perturbation variable | WRF base variable | Writer identity |
|---|---|---|---|
| total pressure | `P` | `PB` | `P + PB == p_total` |
| total geopotential | `PH` | `PHB` | `PH + PHB == ph_total` |
| total dry-column mass | `MU` | `MUB` | `MU + MUB == mu_total` |

## Writer Rules

The writer accepts plain CPU objects, numpy arrays, or project state-like objects. For each pair it applies this fail-closed ordering:

1. If total and perturbation are present, write perturbation as-is and compute base as `base = total - perturbation`.
2. If total and base are present, compute perturbation as `perturbation = total - base`.
3. If perturbation and base are present, write both as-is.
4. If only a legacy total-style field is present, write that field as perturbation and write zero base. This preserves `perturbation + base == total` for old proof payloads, but M7 daily-pipeline integration should provide explicit perturbation/base inputs before making physical compatibility claims.

## Field Mapping

| GPU/source aliases read by writer | WRF output | Math |
|---|---|---|
| `state.p_total`, `state.p`, `state.P_total` | total pressure input | Used only to derive `P`/`PB`; not written directly. |
| `state.p_perturbation`, `state.P` | `P` | Written as perturbation pressure when present. |
| `state.pb`, `state.p_base`, `state.PB` | `PB` | Written as base pressure when present; otherwise `p_total - p_perturbation`. |
| `state.ph_total`, `state.ph`, `state.PH_total` | total geopotential input | Used only to derive `PH`/`PHB`; not written directly. |
| `state.ph_perturbation`, `state.PH` | `PH` | Written as perturbation geopotential when present. |
| `state.phb`, `state.ph_base`, `state.PHB` | `PHB` | Written as base geopotential when present; otherwise `ph_total - ph_perturbation`. |
| `state.mu_total`, `state.mu`, `state.MU_total` | total dry-column mass input | Used only to derive `MU`/`MUB`; not written directly. |
| `state.mu_perturbation`, `state.MU` | `MU` | Written as perturbation dry-column mass when present. |
| `state.mub`, `state.mu_base`, `state.MUB` | `MUB` | Written as base dry-column mass when present; otherwise `mu_total - mu_perturbation`. |

The round-trip proof and tests validate the six written fields `P`, `PB`, `PH`, `PHB`, `MU`, and `MUB`, and validate the three reconstruction identities above against the synthetic CPU state.
