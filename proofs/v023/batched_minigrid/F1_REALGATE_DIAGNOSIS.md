# F1 Real-Gate Diagnosis

Date: 2026-07-02

## Verdict

Root cause class: **B - XLA shape/fusion/accumulation-order change under
`jax.vmap`, not cross-lane contamination**.

The real CPU B=2 gate still fails strict byte identity, so
`F1_REALGATE_DONE` remains absent.  The failure is not accepted as a PASS; it is
classified as a non-contamination bitwise instability of the boundary-enabled
real dycore path under the vmap transform.

## Discriminating Tests

All commands used `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=true`,
`GPUWRF_NESTED_AOT=0`, and `GPUWRF_NESTED_FUSE=0`.

| Test | Result | Interpretation |
|---|---:|---|
| Singleton vmap, B=1, one lane vs standalone | `d02/carry.ph_tend max_abs_diff=5.820766091346741e-11`; `d01` exact | A second lane is not required to reproduce the divergence. This rules out batch-axis cross-lane contamination as the cause of the observed `ph_tend` mismatch. |
| Identical-IC B=2, lane0 vs lane1 | all compared fields byte-identical on `d01` and `d02` | Two lanes with the same inputs do not diverge from each other, so there is no evidence of shared scratch aliasing, host callback lane leakage, or axis-crossing reduction in this fixture. |
| Identical-IC B=2, lane0 vs standalone | same `d02/carry.ph_tend max_abs_diff=5.820766091346741e-11` pattern | Confirms the mismatch is lane-vs-standalone compilation shape/order, not lane-vs-lane contamination. |

The B=1 result is the decisive discriminator: a length-1 outer vmap has no
other lane to contaminate from, yet it reproduces the same `d02/ph_tend`
bitwise failure seen in the B=2 gate.

## Field Diffs From The Real Gate

Source artifact:
`proofs/v023/batched_minigrid/F1_CPU_B2_BIT_IDENTITY.json`

JSON verdict: `FAIL`; failure count: `40`; all field mismatches are on `d02`.
`d01` is byte-identical.

| Domain/field | max_abs_diff | dtype | shape | lanes |
|---|---:|---|---|---|
| `d02/carry.ph_tend` | `5.820766091346741e-11` | `float64` | `7x6x6` | lane0,lane1 |
| `d02/carry.state.ph_bdy` | `1.7763568394002505e-15` | `float64` | `2x4x5x7x7` | lane0,lane1 |
| `d02/carry.ph_save` | `1.7763568394002505e-15` | `float64` | `7x6x6` | lane0,lane1 |
| `d02/carry.state.w_bdy` | `8.881784197001252e-16` | `float64` | `2x4x5x7x7` | lane0,lane1 |
| `d02/carry.state.u_bdy` | `8.881784197001252e-16` | `float64` | `2x4x5x6x7` | lane0,lane1 |
| `d02/carry.state.ph_perturbation` | `8.881784197001252e-16` | `float64` | `7x6x6` | lane0,lane1 |
| `d02/carry.u_save` | `6.661338147750939e-16` | `float64` | `6x6x7` | lane0 |
| `d02/carry.state.u` | `6.661338147750939e-16` | `float64` | `6x6x7` | lane0,lane1 |
| `d02/carry.ww` | `6.175615574477433e-16` | `float64` | `7x6x6` | lane0,lane1 |
| `d02/carry.ww_save` | `5.93275428784068e-16` | `float64` | `7x6x6` | lane0,lane1 |
| `d02/carry.mu_save` | `2.636779683484747e-16` | `float64` | `6x6` | lane0,lane1 |
| `d02/carry.w_save` | `2.220446049250313e-16` | `float64` | `7x6x6` | lane0,lane1 |
| `d02/carry.state.w` | `2.220446049250313e-16` | `float64` | `7x6x6` | lane0,lane1 |
| `d02/carry.mudf` | `2.220446049250313e-16` | `float64` | `6x6` | lane0,lane1 |
| `d02/carry.state.mu_perturbation` | `2.1510571102112408e-16` | `float64` | `6x6` | lane0,lane1 |
| `d02/carry.state.mu_bdy` | `1.942890293094024e-16` | `float64` | `2x4x5x1x7` | lane0,lane1 |
| `d02/carry.state.v_bdy` | `1.6653345369377348e-16` | `float64` | `2x4x5x6x7` | lane0,lane1 |
| `d02/carry.v_save` | `1.1102230246251565e-16` | `float64` | `6x7x6` | lane0 |
| `d02/carry.muave` | `1.0408340855860843e-17` | `float64` | `6x6` | lane0,lane1 |
| `d02/carry.state.qv_bdy` | `1.734723475976807e-18` | `float64` | `2x4x5x6x7` | lane1 |
| `d02/carry.state.qv` | `1.734723475976807e-18` | `float64` | `6x6x6` | lane1 |

The largest relative difference observed in the follow-up numeric pass was for
`d02/carry.ph_tend`: about `4.9e-17` relative to a max reference magnitude of
about `1.189e6`; `np.allclose` was true for the listed non-identical fields.
That is only numeric context.  The F1 acceptance gate is byte identity, and it
still fails.

I did not find a documented nonzero fused-vs-eager dycore tolerance band to
claim this falls inside.  The nearest v021 fused/AOT proof records
`repeat_exact=True repeat_max_abs_diff=0`, and the local nested-domain tests
compare fused scheduler/value output with equality.  Therefore this diagnosis
does not convert the gate into a tolerance PASS.

## Localization

The real gate mismatch localizes to the boundary-enabled child-domain
`_advance_chunk` compiled under the outer vmap:

| Probe | Result |
|---|---|
| Parent `d01` `_advance_chunk` under vmap | byte-identical |
| `build_child_boundary_package` / forcedown-only under vmap | byte-identical |
| Child `d02` `_advance_chunk`, `run_boundary=False`, singleton vmap vs scalar | byte-identical |
| Child `d02` `_advance_chunk`, `run_boundary=True`, singleton vmap vs scalar, 1 step | 3 mismatching fields; worst `p_total/p=1.4551915228366852e-11`, `qv=8.67e-19` |
| Child `d02` `_advance_chunk`, `run_boundary=True`, singleton vmap vs scalar, 3 steps | 15 mismatching fields; worst `ph_tend=2.3283064365386963e-10` |
| Standalone `apply_lateral_boundaries` scalar vs singleton vmap on the same post-dycore state | byte-identical |

Boundary sub-toggle probes did not identify the forced boundary package as the
source.  Disabling the entire boundary path makes child `_advance_chunk`
byte-identical; disabling individual nested `ph`/`w` relax/spec toggles does not
remove the `ph_tend` mismatch.

Current localized statement:

- `build_child_boundary_package` is not the divergent operation.
- Standalone `apply_lateral_boundaries` is not the divergent operation.
- The divergence appears only when the real child `_advance_chunk` is compiled
  as a boundary-enabled integrated step under `vmap`.
- Given the B=1 and identical-IC B=2 results, the evidence supports XLA changing
  scalar operation grouping/fusion/accumulation order under the vmap-transformed
  shape, not a real batch-axis crossing bug.

## Required Follow-Up Decision

F1 does not have a passing bit-identical real-dycore gate.  The evidence says
this is not cross-lane contamination, so a contamination bug fix is not indicated
by these diagnostics.  The next project decision is whether the opt-in batched
path must remain a strict bit-identity feature, or whether it becomes an
explicit tolerance-gated mode with a documented nonzero dycore tolerance policy.

