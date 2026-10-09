# RE03 — nest radiation ozone from the parent (GPUWRF_NEST_O3_FROM_PARENT, v0.3.3)

Request: MAILBOX 2026-10-05T18:32:23Z / 18:47:45Z (manager; owner fid-q2, critic review-b). RE02 (`re02/`) measured
the seam: the port gave every domain its own CAM ozone column, WRF gives every nest the parent's field.

## WRF wiring (pristine V4.7.1)

| item | source | behaviour |
|---|---|---|
| CAM ozone | module_radiation_driver.F:1803 `IF (o3input .EQ. 2 .AND. id .EQ. 1)` | oznini/ozn_time_int/ozn_p_int(p_hyd) on **domain 1 only**, at its radiation calls; o3rad held between calls |
| nest o3rad | Registry.EM_COMMON:1264 `o3rad ikj misc 1 - rdf=(p2c)` | r = restart, d/f = down/force with interp_fcn p2c |
| force-down | inc/nest_forcedown_interp.inc:265–280; mediation_integrate.F:893 (imask_nostag 1) | after EVERY parent step: nest o3rad = interp_fcn SINT (mass point, shw 2) of the parent's held o3rad, level-wise, no re-interpolation to the nest pressure |
| chain | — | d03 receives d02's (already forced) field |

## Port (flag `GPUWRF_NEST_O3_FROM_PARENT=1`, default off)

- `OperationalCarry.o3rad` (new optional leaf, appended last; REAL `(z, y, x)`), seeded zeros by
  `nested_pipeline._load_domains` for nested runs only (`len(names) > 1`, RRTMG driver refresh present).
  Single-domain runs never get the leaf.
- `operational_mode._refresh_rrtmg_driver`: root (`boundary_config.force_geopotential` True) at its radiation call
  computes the CAM columns (`physics_couplers.rrtmg_ozone_columns`, the same `wrf_cam_ozone_profile` the OFF path uses),
  feeds them to RRTMG via `ozone_vmr_override` and stores them in o3rad; between calls the leaf is held (lax.cond
  `held` branch). A nest (live child, force_geopotential False) never recomputes: its radiation consumes the held o3rad.
- Force-down: `nesting/nest_o3.force_child_carry_o3rad` = `interp_sint_full` (pristine SINTB) of the parent o3rad on the
  child mass grid, in all three force sites: eager `domain_tree._operational_force`, the default fused leaf cascade
  (`_build_fused_cascade_program`; its source fingerprint now includes nest_o3.py) and `nested_pipeline._batched_force`
  (vmap for B > 1).
- `physics_couplers._rrtmg_column_inputs(_impl)` / `rrtmg_theta_tendency`: append-only kwarg `ozone_vmr_override`
  (REAL-cast on the REAL path); None = released code path.
- Legacy RTHRATEN refresh (no radiation_diagnostics) with an o3rad leaf raises NotImplementedError (fail closed).
- Restart: the product RestartStore (io.restart v2+ extra_fields) carries o3rad exactly; the WRF-NetCDF carry writer
  has no O3RAD variable and refuses a populated o3rad (UNSUPPORTED_CARRY_FIELDS, loud). The flag is in the
  trace-environment hash → AOT cheap key and restart identity split on it (no silent run/resume mismatch).
- Disclosed: `noahmp_initial_rad` (a port-side t=0 held surface-radiation seed, not a WRF call; WRF's first radiation
  call is itimestep 1) still uses the nest's own CAM column; it is replaced at the nest's first radiation call.
  Adding an OperationalCarry field changes `_CARRY_FIELD_ORDER`, so pre-existing pickle checkpoints fail closed
  (same as every earlier carry leaf, E78/E168).

## Gates

| gate | evidence | result |
|---|---|---|
| G1 pristine SINT | `re03_o3_force.F90` (oznini/ozn_time_int/ozn_p_int on all d01 cells from CPU-WRF 0227 history p_hyd/XLAT, then interp_fcn d01→d02→d03, unchanged libwrflib.a) at 2026-02-28_12Z and 03-01_00Z; `port_sint_check.py` full grids; fixture `tests/v025/fid_q2/fixtures/nest_o3_forcedown_0227_v1.npz` (sha 18105327…, crop == full grid bitwise) | port interp_sint_full vs pristine: d01→d02 84.9/85.1 % bitwise, d02→d03 87.6/87.4 %, max 4 ulp (rel ≤ 3.5e-7) at both times; test bound ≤ 8 ulp, ≥ 75 % bitwise, containing-cell copy rejected (> 1e-3) |
| G1b root CAM | `root_check.py`: port `wrf_cam_ozone_profile` on d01 vs pristine o3rad, all 369,600 cells | ≤ 4 ulp (rel ≤ 3.4e-7), 69.7 % / 83.1 % bitwise at 12Z / 00Z |
| G2 cadence | `test_root_refreshes_o3rad_only_at_its_radiation_calls` (real 0227 product carries, CPU) | non-radiation step: no RRTMG call, o3rad is the SAME held object; radiation step: o3rad == own CAM columns and RRTMG consumed exactly them |
| G3 force + consume | `test_forcedown_sets_child_o3rad_from_parent_and_child_radiates_with_it`, `test_fused_cascade_forces_child_o3rad_like_the_eager_force`, `test_fused_force_reads_the_post_step_parent` (critic b-core O3R: the stubbed parent step doubles o3rad, the forced child must be SINT(post-step parent) ≤ 4 ulp and differ from the pre-step force by > 50 %; kills the read-pre-step-parent mutant), `test_batched_carry_force_maps_each_case_like_the_single_force`, `test_override_reaches_both_rrtmg_column_states` | eager force: only o3rad differs from the released force; fused == eager ≤ 4 ulp; child radiation consumes the held forced field and never recomputes; own CAM differs > 1e-4 rel (RE02: up to 10 % in k0–9) |
| G4 OFF identity | `off_identity.py` candidate (b28200ff2 + this change) vs base export b28200ff2, CPU, product `_load_domains` on real 0227 (flag unset: d01+d02; flag=1: d01 alone) | ALL EQUAL: `_refresh_rrtmg_driver` jaxpr (source-stripped) d01/d02, carry leaves 167/154 byte-identical, eager `_operational_force` outputs 154 leaves and fused force-only outputs 321 leaves byte-identical, no o3rad leaf; single-domain with the flag ON: refresh jaxpr + 167 leaves identical (`g4_*.json`) |
| G5 restart | `tests/v025/fid_q2/test_nest_o3_restart.py` | pickle round trip bitwise, NetCDF writer refuses, identity splits on the flag |
| mutants | `mutants.py` (one fixture load; exec'd source edits + seed deletion) | 5/5 controls pass, 9/9 mutants killed at the targeted assertion: seed deleted, eager force hunk deleted, fused force hunk deleted, child runs own CAM, root uses the held field, root does not store, root refreshes every step, override not passed, override ignored in column inputs (`mutants.json`) |
| effect (RE02) | `re02/` pristine RRTMG on RE01 columns: own-column CAM vs parent copy | O3 ≤ 10.4 % (k0–9), GLW .051, OLR .042, SWDOWN .032 W/m², LW heating ≤ .074 K/d (≤ 0.6 % of the frozen bounds) — small, now WRF-exact |

Numbers [M] CPU unless stated. Not yet measured: GPU arm (needs a lock slot; the change is a held leaf + one SINT per
force-down, flag off by default).
