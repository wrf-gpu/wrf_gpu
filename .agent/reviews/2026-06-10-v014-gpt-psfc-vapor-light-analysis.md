# V0.14 GPT PSFC Vapor-Light Residual Analysis

Date: 2026-06-10. Sprint:
`.agent/sprints/2026-06-10-v014-gpt-psfc-vapor-light-analysis/sprint-contract.md`.
CPU-only; no GPU commands were run.

## 1. Verdict

**BLOCK** for v0.14 grid-parity promotion / Switzerland launch. The fixed Canary
run can continue as characterization, but it is not a green gate while `PSFC`
remains a real pressure-state residual. The h1-h4 evidence confirms the LBC
cadence drift is no longer the primary lane: `MU` improves from h1 to h4
(`58.079 -> 35.153 Pa` RMSE, bias `+52.861 -> +26.430 Pa`), while `PSFC`
stays negative and slowly worsens (`156.974 -> 186.741 Pa` RMSE, bias
`-154.941 -> -185.310 Pa`). This is the suspected vapor-light surface-pressure
floor, not a writer/comparator artifact.

## 2. Root-Cause Ranking

| Rank | Hypothesis | Evidence | Decision / next falsifier |
|---:|---|---|---|
| 1 | **GPU pressure/PSFC state is vapor-light / near dry-column only** | h1 CPU `PSFC-(P_TOP+MU+MUB)` mean `198.734 Pa`; independent `sum(QVAPOR*dp_dry)` mean `198.800 Pa`. GPU has vapor column `197.545 Pa`, but `PSFC-dry` mean is `-8.949 Pa`. | Proven as dominant. Fix pressure-state/PSFC moist load; do not bless as tolerance. |
| 2 | Hydrostatic/EOS reconstruction missing moist column load near surface | h1 GPU `P/PH` extrapolated `PSFC` bias is `-140.663 Pa`; `P_total(k0)` bias is `-137.804 Pa`, while dry-column bias is `+52.742 Pa`. | Likely production locus, broader than writer. Derive WRF-faithful moist pressure/p8w path. |
| 3 | Residual LBC cadence drift | h1-h4 `MU` residual shrinks; U/V/T/T2/U10/V10 remain far better than pre-fix. | Not the remaining h1-h4 PSFC floor. Keep h24/final as slope evidence only. |
| 4 | Writer artifact | GPU `PSFC` equals its own `P/PH` extrapolation to mean `4.9e-05 Pa`, std `0.00245 Pa`, max `0.0148 Pa`. | Exonerated for the large residual. |
| 5 | Comparator/pairing artifact | Direct NetCDF budget reproduces comparator h1 numbers; CPU-only comparator reports finite pair fraction `1.0`. | Exonerated. |
| 6 | Missing/failed moisture field | `QVAPOR` h1 passes (`1.8917e-4` RMSE); GPU vapor-column integral is physically present (`197.545 Pa`). | Exonerated as primary; moisture is present but not loaded into PSFC/pressure state. |
| 7 | `PB/MUB` static frame spikes | Static `MUB` h1 RMSE `9.276 Pa`, bias `-0.119 Pa`, p95 `0`; known 5-cell frame class. | Separate static-boundary issue; cannot explain domain-wide `PSFC` bias. |
| 8 | CPU exact `p8w` formula mismatch | CPU `PSFC - P/PH_extrap` mean is `14.279 Pa`; GPU residual is near zero. | Secondary diagnostic exactness gap, much smaller than the `~208 Pa` vapor-light miss. |

## 3. H1 Pressure Budget

All values are domain means in Pa unless noted. Diffs are GPU minus CPU.

| Quantity | CPU | GPU | Diff mean | Diff RMSE | Interpretation |
|---|---:|---:|---:|---:|---|
| `PSFC` | 101461.012 | 101306.071 | -154.941 | 156.974 | hard h1 failure |
| `MU` | 1730.231 | 1783.092 | +52.861 | 58.079 | dry perturbation mass residual, shrinking by h4 |
| `MUB` | 94532.046 | 94531.928 | -0.119 | 9.276 | static frame issue, small mean |
| dry column `P_TOP+MU+MUB` | 101262.278 | 101315.020 | +52.742 | 58.209 | GPU dry column is slightly high, not low |
| vapor proxy `PSFC-dry` | 198.734 | -8.949 | -207.684 | 210.405 | dominant missing moist load |
| independent `sum(QVAPOR*dp_dry)` | 198.800 | 197.545 | -1.254 mean | n/a | moisture column exists on GPU |
| `P/PH` extrapolated `PSFC` | 101446.733 | 101306.070 | -140.663 | 142.827 | pressure state is vapor-light too |
| `PSFC - P/PH_extrap` | +14.279 | +0.000049 | n/a | n/a | writer matches GPU state; CPU exact formula has small residual |

h1-h4 vapor lane: CPU vapor proxy rises `198.7 -> 208.0 Pa`; GPU proxy rises
`-8.9 -> -3.7 Pa`; vapor-proxy delta remains `-207.7 -> -211.6 Pa`.

## 4. Writer Decision

Writer is **exonerated for the large residual**. The emitted GPU `PSFC` is the
same field implied by GPU `P/PB/PH/PHB`, not a serialization or variable-routing
mistake. The problem is upstream: the pressure state being written is missing
the moist/vapor column contribution that CPU WRF carries in surface pressure.

There is a smaller CPU-vs-formula diagnostic gap (`~14 Pa`) in my simple
`P/PH` extrapolation. That should be captured in the fix proof, but it cannot
explain the `~155-185 Pa` PSFC miss or the `~208-212 Pa` vapor-proxy gap.

## 5. Exact Next Proof / Fix Sprint

Open a focused fix sprint:

```text
Sprint: V0.14 PSFC Moist Pressure-State Closure
Objective: close or formally bound the fixed-Canary PSFC vapor-light residual
without masking it in the comparator or adding a PSFC-only clamp.
Files in scope:
- src/gpuwrf/dynamics/acoustic_wrf.py
- src/gpuwrf/runtime/operational_mode.py
- src/gpuwrf/io/wrfout_writer.py
- proof-only scripts under proofs/v014/
Required proof:
- CPU-only NetCDF budget for h1-h4 and first available h24:
  PSFC, dry column, sum(QVAPOR*dp_dry), P/PH-extrapolated PSFC.
- An offline ablation that adds the WRF-derived vapor column load to GPU PSFC
  and reports the expected post-fix h1/h4 PSFC RMSE/bias without changing MU.
- Source-level audit against WRF `phy_prep`/`p8w` and `calc_p_rho_phi` before
  any production patch.
Acceptance:
- h1 PSFC RMSE <= 120 Pa without increasing MU/P/PH regressions.
- GPU PSFC must remain derivable from a documented WRF-faithful pressure path;
  no tolerance-only or output-only masking.
```

When h24 is available, run the existing CPU comparator for the trajectory:

```bash
RUN_ROOT=/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python scripts/compare_wrfout_grid.py \
  --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
  --gpu-dir "$RUN_ROOT/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z" \
  --domain d02 --init 2026-05-01T18:00:00+00:00 \
  --min-lead 1 --max-lead 24 \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --out-json "$RUN_ROOT/canary_d02_h24_grid_compare.json" \
  --out-md "$RUN_ROOT/canary_d02_h24_grid_compare.md"
```

## 6. Proposed Fix Direction

Do **not** merge a PSFC-only cosmetic correction until the pressure-state
semantics are proven. A diagnostic ablation should first show the bound:
`PSFC_corrected ~= dry_column + sum(qv*dp_dry)` would shift h1 GPU PSFC upward
by about `+206.5 Pa`, changing h1 bias from `-154.9 Pa` to roughly `+51 Pa`
(the remaining dry-column/MU residual). At h4 it should shift bias from
`-185.3 Pa` to roughly `+26 Pa`.

WRF-faithfulness requirement: the production fix must be tied to WRF's moist
surface pressure / `p8w` semantics and preserve dry `MU/MUB` as dry-air mass.
Expected signal if correct: PSFC loses the flat `-210 Pa` vapor floor, P/PH
near-surface extrapolation moves by the same moist-load sign, `QVAPOR` remains
unchanged, and `MU` remains governed by the already-improving dry-mass lane.

## 7. Context-Sparing Handoff

- objective: diagnose the fixed-Canary h1 `PSFC` residual after the LBC cadence fix.
- files changed: this report only.
- commands run: CPU-only direct NetCDF pressure budgets for h1 and h1-h4; CPU-only reads of h1/h4 comparator JSON/MD; source audit of writer/runtime/dycore state paths.
- proof objects produced: this report, with direct WRF fixture budget numbers from the fixed run root.
- verdict: `BLOCK` for promotion; active Canary may continue as characterization.
- root cause: GPU pressure/PSFC is vapor-light despite GPU `QVAPOR` column being present.
- writer/comparator: exonerated for the large residual.
- unresolved risk: exact WRF CPU `p8w` formula has a smaller `~14 Pa` gap in my simple extrapolation; fix proof must include source-level WRF anchoring.
- next decision needed: open the focused PSFC moist pressure-state closure sprint before Switzerland GPU or grid-parity closeout.
