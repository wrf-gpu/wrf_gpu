# v0.14 Moist-Theta Physics Consumer Audit

Date: 2026-06-10

Sprint: `.agent/sprints/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit/sprint-contract.md`

## Outcome

The NoahMP Step-1 issue is not isolated to one fallback temperature expression. Runtime `state.theta` is coupled moist potential temperature (`theta_m`), while many production physics adapters still treat it as dry potential temperature when forming `T`, `TH`, `THX`, `THV`, density, or dry theta tendencies.

The core compatibility rule should be:

```text
theta_dry = theta_m / (1 + (461.6 / 287.0) * qv_mixing)
theta_m   = theta_dry * (1 + (461.6 / 287.0) * qv_mixing)
```

`qv` here is WRF mixing ratio (`QVAPOR`, kg/kg), not specific humidity.

## Compatibility Table

| Consumer | Class | Evidence | Required action |
| --- | --- | --- | --- |
| LBC, feedback, dycore transport | `MUST_USE_MOIST_THETA` | `init/real_init/types.py:109`, `lateral_bc.py:236`, `boundary_apply.py:148`, `dynamics/advection.py:262` | Keep `state.theta` moist for storage, boundaries, and dynamics. Decouple only at physics boundaries. |
| Grid-backed MYNN and generic surface view | `ALREADY_DECOUPLED` | `physics_couplers.py:1313`, `:1404`, `:1648`; `operational_mode.py:3188` | Preserve this dry-view plus moist-writeback pattern. |
| NoahMP forcing and hook | `MUST_DECOUPLE_TO_DRY` | `noahmp_surface_hook.py:79`, `noahmp_coupler.py:161`, `:238` | Fable's active patch decouples `sfctmp` fallback, but the hook still needs a dry surface view for sfclay diagnostics, `th2/thx`, virtual temperature, and CH/CM seed terms. |
| Noah Classic | `MUST_DECOUPLE_TO_DRY` | `noahclassic_surface_hook.py:161`, `:181`, `:251` | Dry bottom-air `T`, `th2`, `thx`, and flux diagnostics before forcing assembly. |
| Thompson | `MUST_DECOUPLE_TO_DRY` | `physics_couplers.py:1155`, `:1187`; `thompson_column.py:180` | Feed dry-theta temperature and recouple returned dry theta with updated qv before updating `state.theta`. |
| Scan microphysics | `MUST_DECOUPLE_TO_DRY` | `scan_adapters.py:157`, `:176`, `:203`, `:222`, `:240`, `:258`, `:277`, `:312`; kernels convert theta*pi to T | Decouple theta before Kessler/Lin/WSM/WDM/Morrison calls and recouple dry theta outputs or tendencies. |
| Radiation | `MUST_DECOUPLE_TO_DRY` | `physics_couplers.py:965`, `:1067`, `:1826-2382`; `operational_mode.py:3171`, `:3207` | Build radiation `T`/rho from dry theta. Ensure dry `RTHRATEN` is converted before moist theta update in all runtime modes, not only source-leaf mode. |
| Surface-layer scan adapters | `MUST_DECOUPLE_TO_DRY` | `scan_adapters.py:352`, `:383`, `:403`, `:429`, `:456`, `:481` | Use the grid-backed dry bottom-air view instead of raw `state.theta`. |
| PBL scan adapters and MYJ | `MUST_DECOUPLE_TO_DRY` | `scan_adapters.py:971`, `:1063`, `:1077`, `:1091`, `:1126`; `myj_adapters.py:63`, `:176` | Feed dry `T`/theta and convert returned dry theta tendencies to moist tendency/writeback. |
| Cumulus scan adapters | `MUST_DECOUPLE_TO_DRY` | `scan_adapters.py:507`, `:588`, `:751`, `:819`, `:877` | Feed dry `T`, density, and virtual-theta inputs. Convert dry `RTHCUTEN` with qv tendencies before updating moist `state.theta`. |
| GWDO | `MUST_DECOUPLE_TO_DRY` | `physics_couplers.py:2475`, `:2488`; `gwd_gwdo.py:132` | Feed dry-theta temperature. No theta recoupling is needed because GWDO is momentum-only. |
| WRF output diagnostics | `MUST_DECOUPLE_TO_DRY` | `coupling/driver.py:1155`, `:1201` | Convert moist theta to dry theta before writing WRF-style `T` and before T2/Q2/U10/V10 diagnostics. |
| Real-init hydrostatic and no-grid fallbacks | `PROOF_ONLY_OR_ORACLE_INPUT` | `hydrostatic.py:64`, `:226`; `physics_couplers.py:1313`, `:1648` | Real-init already separates dry theta and qv factors. Treat no-grid raw fallbacks as analytic-only unless separately proved. |
| Dycore EOS qv factor | `UNCLEAR_NEEDS_TEST` | `dynamics/acoustic_wrf.py:135`, `:279`; `dynamics/core/rk_addtend_dry.py:127` | Do not change dycore as part of this physics fix. Add a separate savepoint/oracle gate for theta_m EOS qv-factor consistency. |

## Helper/API Shape

Put the conversion in one shared coupling-level helper near the existing `_temperature_from_theta` / `_theta_from_temperature` code, or in a small shared constants/helper module imported by adapters:

```python
RV_OVER_RD = 461.6 / 287.0

def theta_dry_from_moist(theta_m, qv_mixing):
    return theta_m / (1.0 + RV_OVER_RD * qv_mixing)

def theta_moist_from_dry(theta_dry, qv_mixing):
    return theta_dry * (1.0 + RV_OVER_RD * qv_mixing)

def temperature_from_dry_theta(theta_dry, pressure_pa):
    return _temperature_from_theta(theta_dry, pressure_pa)
```

For source terms, either finite-recouple with updated qv:

```text
theta_m_new = theta_dry_new * (1 + rvrd * qv_new)
```

or convert tendencies explicitly:

```text
dtheta_m = (1 + rvrd*qv_old) * dtheta_dry + rvrd * theta_dry_old * dqv
```

Keep these as elementwise JAX operations so they fuse and remain device-resident.

## Required Follow-Up Gates

- NoahMP strict Step-1 closure with `use_noahmp=True`: compare `SFCTMP/T_ML`, `HFX`, `LH`, `T2`, `Q2`, `CH`, `CM`, and `FLTV` against WRF savepoints after the Fable fix lands.
- CPU helper probe: construct `theta_dry`, `qv`, and `p`; derive `theta_m`; verify the dry helper recovers oracle `T` while raw `_temperature_from_theta(theta_m, p)` reproduces the warm bias.
- Adapter probes for NoahMP and Noah Classic bottom-air `T`, `th2/thx`, virtual temperature, and flux-handle fields.
- Radiation input parity for RRTMG SW/LW, Dudhia, GSFC, and classic RRTM, plus a tendency-application gate proving every dry `RTHRATEN` update is moist-converted.
- Microphysics, cumulus, PBL, MYJ, and GWDO smoke/parity cases initialized with `theta_m`; assert kernel inputs are dry-derived and final `state.theta` remains moist after qv changes.
- CPU-only execution for this audit sprint: `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=`.

## Notes

I observed an uncommitted active patch in `src/gpuwrf/physics/noahmp_coupler.py` that imports `R_D`, defines `RVOVRD`, and decouples the NoahMP `sfctmp` fallback. That is external work from the live Fable/Mythos fix, not an edit from this audit.

This audit wrote only:

- `proofs/v014/moist_theta_physics_consumer_audit.json`
- `proofs/v014/moist_theta_physics_consumer_audit.md`
- `.agent/reviews/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit.md`

Validation:

- `python -m json.tool proofs/v014/moist_theta_physics_consumer_audit.json >/tmp/moist_theta_physics_consumer_audit.validated.json`: PASS
- `git diff --check`: PASS
- Completion marker: attempted, but `tmux send-keys -t 0:2` failed with `Operation not permitted`; proof files are authoritative.
