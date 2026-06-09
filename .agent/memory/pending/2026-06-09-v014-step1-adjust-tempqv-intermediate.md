# V0.14 Step-1 Adjust-TempQV Intermediate

Date: 2026-06-09 17:16 WEST

Sprint:
`.agent/sprints/2026-06-09-v014-step1-adjust-tempqv-intermediate`.

Proof:
`proofs/v014/step1_adjust_tempqv_intermediate.*`.

Verdict:
`STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

The manager reran the disposable WRF hook outside the Codex PMIx sandbox and
captured exact CPU-WRF `adjust_tempqv` internals for d02 Fortran cell
`{i:18,j:10,k:2}` / zero `{k:1,y:9,x:17}`.

Important deltas, WRF minus JAX:

- `p`: `0.0`
- `mub_save`: `0.0`
- `c3h`, `c4h`, `p_top`: `0.0`
- `mub`: `17.67503987130476 Pa`
- `pb_new_equiv`: `17.49400702366256 Pa`
- `p_new`: `17.49400702366256 Pa`
- `t_2_post`: `-0.00541785382188209 K`

Interpretation:

The remaining theta residual is not currently a proven `adjust_tempqv`
transcription bug. It is driven by a material current pressure/base-input
mismatch. The next debug sprint should split WRF and JAX current `MUB/PB` after
live-nest terrain/base blending and before `adjust_tempqv`; do not resume long
TOST/Switzerland validation until this path is fixed or explicitly bounded.
