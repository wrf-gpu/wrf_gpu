# Manager Adjudication: EOS Moisture Factor Is Not Accepted Yet

Date: 2026-06-10 WEST

Fable round 1 returned `FIXED` with a change in
`src/gpuwrf/dynamics/acoustic_wrf.py` from `qvf = 1 + 0.608*qv` to
`qvf = 1 + (Rv/Rd)*qv`. This is **not accepted** until the theta convention is
settled.

## Manager Concern

The proof inverted EOS using CPU/GPU `wrfout` variable `T`, but operational
v0.14 evidence says the runtime state may store WRF `theta_m` when
`USE_THETA_M=1`, not dry theta:

- `proofs/v014/moist_theta_physics_consumer_audit.md` says boundary/feedback/
  dycore transport must keep `state.theta` as moist theta and physics adapters
  should decouple to dry views.
- `src/gpuwrf/integration/d02_replay.py` live-nest path applies
  `use_theta_m=1` dry-to-moist conversion and documents that
  `theta_full_out` is still moist.
- `src/gpuwrf/integration/d02_replay.py` start-domain perturb init documents
  WRF `qvf = 1` under `use_theta_m=1`.
- WRF `module_big_step_utilities_em.F` branches:
  `IF (use_theta_m .EQ. 1)` -> no qv factor; else `qvf = 1 + rvovrd*qv`.
- Current GPU writer writes `"T": state.theta - 300` and does not emit `THM`.
  If `state.theta` is `theta_m`, the GPU output `T` is mislabeled moist theta.

## Fresh Evidence

On CPU truth h1:

- `THM == T_dry * (1 + Rv/Rd*qv)` to max `6.8e-05 K`.
- CPU EOS with `T` needs `rvovrd`; CPU EOS with `THM` needs `qvf=1`.

On the pre-fix GPU h1:

- GPU output has no `THM`.
- `GPU_T - CPU_Tdry`: bias `+0.6705 K`, RMSE `1.457 K`.
- `GPU_T - CPU_THM`: bias `-0.3875 K`, RMSE `1.020 K`.
- This is consistent with GPU `T` being at least partly a moist-theta output,
  not proof that runtime dry-theta EOS should use `rvovrd`.

## Manager Decision

Do not merge the round-1 EOS patch or start 72h GPU gates until a second proof
answers:

1. What is the canonical runtime `State.theta` convention for the operational
   v0.14 Canary live-nest path at h1: dry theta or moist `theta_m`?
2. Given that convention and WRF `use_theta_m=1`, what exact qv factor belongs
   in `_pressure_from_theta_alt` / `_inverse_density_from_theta_pressure` for
   each production caller?
3. Does the writer need to decouple `state.theta` to dry `T` and/or emit `THM`
   for WRF-compatible field parity?
4. What is the smallest performance-compatible code change that is WRF-faithful
   and testable before the 72h gates?

Endpoint for the next worker: replace or ratify the round-1 patch with proof.
