# V0.14 Lat/Lon Writer Payload

Generated UTC: `2026-06-09T00:07:46.209646+00:00`

Verdict: `PASS`.

- Writer diagnostics with real `XLAT`/`XLONG` are selected over the projection fallback.
- With diagnostics absent, the writer still uses the projection fallback for synthetic/no-static callers.
- Emitted `XLAT` vs GPU-native wrfinput: RMSE `0.0`, max_abs `0.0`.
- Emitted `XLONG` vs GPU-native wrfinput: RMSE `0.0`, max_abs `0.0`.
- Static metric payloads checked here are unchanged by the lat/lon diagnostics path.
- Model numerics changed: `false` (writer-only host output payload).

Full tables and exact file paths are in `proofs/v014/latlon_writer_payload.json`.
