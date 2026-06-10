# Manager Closeout

Date: 2026-06-10 02:57 WEST
Manager: Codex
Worker: GPT-5.5 xhigh in tmux `0:3`

## Outcome

Closed as a valid narrowing fix, not a parity closure.

Verdict:

`STEP1_SFCLAY_FIRST_CALL_FIXED_NEXT_BLOCKER_TSK_ZNT_SURFACE_INPUTS`

The sprint fixed WRF MYNN surface-layer first-call semantics in production and
proved improved surface/MYNN boundary metrics. The strict Step-1 source-fidelity
gate remains red; the next active blocker is TSK/ZNT surface input sourcing.

## Proof Objects

- `proofs/v014/step1_sfclay_boundary_fix.py`
- `proofs/v014/step1_sfclay_boundary_fix.json`
- `proofs/v014/step1_sfclay_boundary_fix.md`
- `.agent/reviews/2026-06-10-v014-step1-sfclay-boundary.md`
- refreshed:
  - `proofs/v014/mynn_driver_source_output_fix.{py,json,md}`
  - `proofs/v014/step1_source_fidelity_closure.{py,json,md}`

## Manager-Rerun Evidence

- Surface tests: `2 passed, 1 skipped`.
- `step1_sfclay_boundary_fix.py` reproduced the new verdict.
- `step1_source_fidelity_closure.py` reproduced
  `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_TSK_ZNT_INPUTS`.
- `mynn_driver_source_output_fix.py` reproduced the MYNN kernel/init proof.
- JSON validation and `git diff --check` passed.

## Merge Decision:

Merge. This is a WRF-sourced, performance-compatible first-call fix and a
strictly narrower blocker. Do not start TOST, Switzerland, broad FP32, or long
GPU validation yet.

## Key Numbers

- UST RMSE improved `0.08667703917523994 -> 0.02954126268295198`.
- qv-flux RMSE improved `1.9833425562981398e-05 -> 1.442591864492997e-05`.
- strict after-conv `T_TENDF` max_abs `1497.6112467075195`, RMSE
  `13.296448784742802`.
- TSK max_abs remains `8.344940187890643 K`.
- ZNT max_abs remains `0.9737602076530456 m`.

## Next Sprint

Open a GPT-5.5 xhigh TSK/ZNT surface-input sprint:

- emit a tiny WRF Step-1 hook around `module_surface_driver/module_sf_mynn`;
- capture incoming `TSK/ZNT/UST/QSFC/MOL` and outgoing `UST/HFX/QFX/ZNT`;
- compare against JAX `_surface_column_view` inputs and diagnostics;
- fix TSK/ZNT sourcing if confirmed;
- rerun strict Step-1 source-fidelity proof.
