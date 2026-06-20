# v0.11.0 GWD Status

Verdict: **documented namelist deviation for the current GPU d02/nest path.**

No `gravity_wave_drag.py` implementation was added. WRF GWD is active on the CPU-WRF 9 km parent, but it is not directly load-bearing for the current GPU d02/nest replay forecast path.

## Evidence

- Active contract: v0.11.0 Tier-2 item 11 from the task prompt.
- Branch/base: `worker/gpt/v0110-gwd` at `0cd101cfbf480ae41507ff76e904c6933e899b87`, matching `worker/opus/v0110-integration`.
- WRF source:
  - `~/src/wrf_pristine/WRF/phys/module_pbl_driver.F:2225` calls `gwdo` only when `GWD_opt == 1`.
  - `~/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part1.F:1245` passes `config_flags%gwd_opt` and the `VAR/CON/OA/OL` fields into the PBL driver.
  - `~/src/wrf_pristine/WRF/share/input_wrf.F:751` checks nonzero namelist `gwd_opt` against each domain's `wrfinput` `GWD_OPT`.
  - `~/src/wrf_pristine/WRF/share/module_check_a_mundo.F:913` allows child domains to set `gwd_opt=0`.
- Corpus case: `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l2rerun_l3_24h_20260522T231647Z`.
  - `namelist.input` has `gwd_opt = 1,` under `&dynamics`.
  - Generated `wrfinput` attributes are domain-specific:
    - d01: `DX=9000`, `DY=9000`, `GWD_OPT=1`.
    - d02: `DX=3000`, `DY=3000`, `GWD_OPT=0`.
    - d03/d04/d05: `DX=1000`, `DY=1000`, `GWD_OPT=0`.
  - d01 has nonzero GWD descriptors (`VAR_max=390.36 m`, `CON_nonzero=401`, `OA*_nonzero=401`).
  - d02-d05 have `GWD_OPT=0` and zero `CON/OA/OL` descriptor fields, even though `VAR` is present as terrain metadata.

## Decision

For the current GPU product, d01 effects are already baked into the CPU-WRF initial and lateral-boundary artifacts consumed by the d02/nest replay path. The GPU runtime does not integrate a live d01 parent, and the CPU-WRF nest baseline itself has `GWD_OPT=0` on d02-d05. Adding GWD now would therefore add an unvalidated tendency relative to the active WRF nest baseline.

This is not a claim that GWD is negligible for a future standalone live d01 parent. It is active on d01 and should be reopened when live parent or full multi-domain equivalence enters scope.

## Endpoint

- Implemented: no.
- Parity fixture: not applicable.
- Proof object: `proofs/v0110/gwd_status.json`.
- Status: endpoint (a) satisfied.
