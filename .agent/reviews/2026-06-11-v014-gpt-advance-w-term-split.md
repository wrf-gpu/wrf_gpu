# v0.14 GPT Advance-W Term Split Review

Verdict: **NARROWED_NO_FIX**.

I do not have a WRF-faithful local source fix. The new proof rejects the current manager boundary as the primary `p/ph` creator: replacing the available `advance_mu_t` dry-mass outputs consumed by `advance_w` with WRF call-21602-derived `mu/muts/muave` changes the first bad stage by only 0.11% in `p` and 0.0018% in `ph`. The remaining creator is still inside the `advance_w`/`calc_p_rho` term chain or in an `advance_w` input not exposed by the current HPG dumps.

## Objective

Independently test the h36 -> h37 Switzerland/Gotthard first-bad RK1 acoustic substep (`WRF call 21601 -> 21602`), specifically the proposed boundary between `advance_mu_t` outputs consumed by `advance_w` and internals of `advance_w_wrf()`. Implement a minimal source fix only if the evidence proves one.

## Proof Produced

New proof-only harness:

- `proofs/v014/switzerland_advance_w_term_split.py`
- `proofs/v014/switzerland_advance_w_term_split.json`

The harness builds the same h36/call-21601 stage context as the prior discriminator, applies a proof-only CPU allocator shim because this worker has no visible GPU, runs focused variants through `advance_w_wrf()`, stage finish, and WRF-call-21602 comparison, and writes a JSON ledger.

Baseline reproduction is exact enough to trust the harness:

- prior expected `p` interior RMSE: `1.1261975184532773`
- harness baseline `p` interior RMSE: `1.1261975184533854`
- prior expected `ph` interior RMSE: `0.4352639584631776`
- harness baseline `ph` interior RMSE: `0.4352639584631756`

## Term/Boundary Results

Ranked interior RMSE effects versus WRF call 21602:

| Variant | p RMSE | ph RMSE | p improvement | ph improvement |
| --- | ---: | ---: | ---: | ---: |
| `zero_ph_tend` | `1.02406587155389` | `0.322809005254585` | `9.07%` | `25.84%` |
| `dry_cqw_and_coefficients` | `1.1254482100842` | `0.435245613092974` | `0.067%` | `0.004%` |
| `wrf_call21602_mu_inputs` | `1.12493562350713` | `0.435256220049474` | `0.112%` | `0.0018%` |
| `baseline_current` | `1.12619751845339` | `0.435263958463176` | `0` | `0` |
| `calc_p_rho_no_smdiv` | `1.12619751845339` | `0.435263958463176` | `0` | `0` |
| `calc_p_rho_stage_mut_denominator` | `1.12619751845339` | `0.435263958463176` | `0` | `0` |
| `wrf_coupled_surface` | `1.12619751845339` | `0.435263958463176` | `0` | `0` |
| `zero_phi_adv` | `1.12749589406214` | `0.435778416157848` | `-0.115%` | `-0.118%` |
| `dry_cqw_and_dry_rw` | `0.764430734254639` | `0.46505091256469` | `32.12%` | `-6.84%` |
| `zero_rw_tend` | `0.76339042224408` | `0.465051108822985` | `32.22%` | `-6.84%` |
| `dry_recomputed_rw_tend` | `0.764449589795189` | `0.46505137452736` | `32.12%` | `-6.84%` |

Stop criterion result: `material_local_variant_found=false` because no variant improves both `p` and `ph` by >50%.

## Hypothesis Ledger

1. `advance_mu_t` dry-mass output consumed by `advance_w` is primary creator.
   - Evidence against: JAX `mu_new` vs WRF call-21602 `mu` has the known interior RMSE `0.02089603774551649`, but forcing WRF-derived `mu/muts/muave` into `advance_w` changes `p/ph` by only `0.00112049` / `0.00001778` improvement fractions.
   - Status: rejected as primary `p/ph` creator; still a secondary `mu` mismatch.

2. Surface `w` feed / WRF-coupled surface wind is primary creator.
   - Evidence against: `wrf_coupled_surface` is bit-identical to baseline at the WRF-scored stage boundary. This independently confirms the prior surface discriminator.
   - Status: rejected.

3. Moist `cqw` / coefficient choice is primary creator.
   - Evidence against: dry `cqw` coefficients improve `p` by only `0.0665%` and `ph` by `0.0042%`.
   - Status: rejected as primary.

4. `rw_tend` class is involved in `p` but not sufficient.
   - Evidence: zeroing or dry-recomputing `rw_tend` improves `p` by about `32%` but worsens `ph` by about `6.84%`.
   - Status: not a WRF-faithful fix; useful as a coupling clue.

5. `ph_tend` class is involved in `ph` but not sufficient.
   - Evidence: zeroing `ph_tend` improves `ph` by `25.84%` and `p` by `9.07%`, but leaves most of the error.
   - Status: not a WRF-faithful fix; points at a missing subterm split inside `advance_w` RHS/implicit solve.

6. `calc_p_rho_step` denominator or `smdiv` is primary creator.
   - Evidence against: stage-mut denominator and `smdiv=0` variants are identical to baseline at this first stage.
   - Status: rejected for this first creator.

## Interpretation

The available WRF HPG dump is enough to falsify the dry-mass boundary as the main `p/ph` creator, because `mu` is visible at call 21602 and mass is not changed by `advance_w`/`calc_p_rho`. It is not enough to prove the exact remaining term because it does not expose WRF-native `advance_w` internals between call 21601 and call 21602.

The strongest next boundary is now **inside `advance_w_wrf()`**, with priority on:

- `ph_tend` contribution into `rhs`
- `rw_tend` / vertical PGF-buoyancy contribution into the implicit vertical solve
- RHS after vertical phi advection but before Thomas sweep
- Thomas coefficients and solved `w`
- geopotential finish `ph_next`
- immediate `calc_p_rho_step` `p/al/alt` from that `ph_next`

## Commands Run

- `python -m py_compile proofs/v014/switzerland_advance_w_term_split.py`
- `python proofs/v014/switzerland_advance_w_term_split.py` (first run failed: `State.zeros requires a GPU device`; fixed with proof-only CPU allocator shim)
- `python -m py_compile proofs/v014/switzerland_advance_w_term_split.py && python proofs/v014/switzerland_advance_w_term_split.py` (second run failed on WRF `mu` dump `(129,129)` vs JAX mass `(128,128)`; fixed by cropping WRF fields to JAX mass domain)
- `python -m py_compile proofs/v014/switzerland_advance_w_term_split.py && python proofs/v014/switzerland_advance_w_term_split.py` (success)
- `python -m json.tool proofs/v014/switzerland_advance_w_term_split.json >/tmp/switzerland_advance_w_term_split.validated.json`
- Focused JSON summary commands for variant ranking and input-delta extraction
- Read-only source/proof inspection with `rg`/`sed` across `advance_w`, `advance_mu_t`, `calc_p_rho`, prior Switzerland proof scripts, and WRF pristine `module_small_step_em.F`

## Files Changed

- Added `proofs/v014/switzerland_advance_w_term_split.py`
- Added `proofs/v014/switzerland_advance_w_term_split.json`
- Added `.agent/reviews/2026-06-11-v014-gpt-advance-w-term-split.md`

No production source files were changed.

## Risks / Limits

- No GPU h36 or 72h gate was run. This worker has no visible CUDA device (`State.zeros` production constructor and JAX GPU backend are unavailable here).
- The CPU allocator shim is proof-local and only allows the existing replay constructor to allocate arrays on CPU; it does not modify production behavior.
- WRF call-21602 exposes `mu` but not all `advance_mu_t` outputs or any intra-`advance_w` arrays. The `wrf_call21602_mu_inputs` variant is therefore a strong falsifier for dry mass as primary creator, not a full proof that every `advance_mu_t` output is correct.

## Next Decision

Generate a short WRF-native call-21601 -> call-21602 intra-`advance_w` dump, not a long GPU gate. Required arrays:

- post-`advance_mu_t` inputs: `mu_2`, `muts`, `muave`, `ww`, `t_2`, `ph_tend`, `rw_tend`
- `advance_w` RHS before/after vertical phi advection
- implicit coefficients `a`, `alpha`, `gamma`
- Thomas RHS / solved `w`
- geopotential finish `ph`
- post-`calc_p_rho` `p`, `al`, `alt`

Then compare those arrays against the existing Python harness at the same h36 call boundary.
