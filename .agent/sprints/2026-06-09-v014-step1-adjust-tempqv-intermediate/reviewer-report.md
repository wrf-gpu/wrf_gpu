# Reviewer Report: V0.14 Step-1 Adjust-TempQV Intermediate Truth

Date: 2026-06-09

Decision: ACCEPT with a narrow next-step requirement. The sprint should be
closed as a valid proof sprint, not as a production source fix.

## Findings

- HIGH: The residual is classified at the right boundary. Exact WRF
  `adjust_tempqv` internals were captured for the previously identified
  interior worst cell, and the comparison names a material pressure/base-input
  mismatch rather than speculating about thermodynamics.
- HIGH: The mismatch pattern is specific: `p`, `mub_save`, `c3h`, `c4h`, and
  `p_top` match, while current `mub`, `pb_new_equiv`, and `p_new` are high in
  WRF by about `17.5 Pa`. This points to the current live-nest `mub`/base path
  after saved-state input, not to a broad QVAPOR or theta mapping issue.
- MEDIUM: The proof is one-cell by design. The next sprint should split the
  current-`mub` source and then apply a field-level proof before any source
  patch is accepted.

## Evidence

- `proofs/v014/step1_adjust_tempqv_intermediate.md`
- `proofs/v014/step1_adjust_tempqv_intermediate.json`
- WRF hook output:
  `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- Manager WRF log:
  `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/logs/wrf_run_mpirun_np28_manager.log`

## Required Next Sprint

Open a CPU-only proof sprint that instruments or reconstructs WRF current
`MUB`/`PB` immediately around `blend_terrain`, `adjust_tempqv`, and any
post-`start_domain_em` base recomputation, and compares the same quantities to
the JAX live-nest base-init proof values. Do not resume TOST or Switzerland
until this path is fixed or explicitly bounded.
