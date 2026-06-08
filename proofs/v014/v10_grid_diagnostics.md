# V10 Grid Diagnostics

Generated UTC: `2026-06-08T22:15:19.560056+00:00`

This is a diagnosis artifact, not an equivalence pass.

## Cases

## V10 Summary

- grid cases with V10 RMSE > 1.5 m/s: `3` / `3`
- grid V10 bias signs: `-, -, +`
- station TOST V10 outside ADR-029 margin: `1` / `3`

### 20260429_18z_l2_72h_20260524T204451Z

- source: `case_json_only`
- stored V10 grid RMSE: `4.183954121704864`; bias: `-2.4838981310329284`; p95 abs: `6.978271573781967`
- station V10 paired delta RMSE: `-0.21559807509640594` (margin `0.275`)
- note: GPU wrfout directory is not available; using stored aggregate stats only.

### 20260430_18z_l2_72h_20260520T191306Z

- source: `case_json_only`
- stored V10 grid RMSE: `3.0525760512908424`; bias: `-1.8751485808776087`; p95 abs: `5.415167272090912`
- station V10 paired delta RMSE: `0.5226565284451792` (margin `0.275`)
- note: GPU wrfout directory is not available; using stored aggregate stats only.

### 20260501_18z_l2_72h_20260519T173026Z

- source: `spatial_grid_wrfouts`
- V10 grid RMSE: `2.524` m/s; bias: `1.036` m/s; p95 abs: `5.474` m/s
- station V10 paired delta RMSE: `0.11816818702501841` (margin `0.275`)
- V10 by block:
  - `0-6h`: RMSE `1.578`, bias `-0.771`, p95 `2.647`
  - `6-12h`: RMSE `3.235`, bias `1.979`, p95 `6.612`
  - `12-24h`: RMSE `2.502`, bias `1.469`, p95 `5.442`
- V10 correlations:
  - `corr_dV10_dT2`: `0.003952303850801751`
  - `corr_dV10_dU10`: `-0.13797593988625842`
  - `corr_dV10_dPSFC`: `-0.2895051716197978`
