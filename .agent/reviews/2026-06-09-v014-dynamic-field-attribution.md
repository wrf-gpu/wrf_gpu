# V0.14 Dynamic Field Attribution Review

## Objective

Produce a CPU-only wrfout attribution manifest for retained Case 3 using the V0.14 grid comparator evidence, without `src/` edits or GPU use.

## Outcome

- Probe implemented: `proofs/v014/dynamic_field_attribution.py`
- Proofs produced: `proofs/v014/dynamic_field_attribution.json`, `proofs/v014/dynamic_field_attribution.md`
- JSON validated and script compiled.
- No equivalence/pass claim is made.

## Commands Run

```bash
python -m py_compile proofs/v014/dynamic_field_attribution.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 python proofs/v014/dynamic_field_attribution.py
python -m json.tool proofs/v014/dynamic_field_attribution.json >/tmp/dynamic_field_attribution.validated.json
python -m py_compile proofs/v014/dynamic_field_attribution.py
```

## Selected Target

- First materially bad lead: `h1` (report-only threshold hits: `W`, `PSFC`).
- Same-state localization lead: `h10` / `2026-05-02T04:00:00+00:00`.
- Reason: highest h10-h14 primary localization score, with simultaneous PSFC/MU/P/PH/U/V/U10/V10/T/QVAPOR/W/PBLH signal.
- Selected mass-grid cells: `(9,13)`, `(25,39)`, `(41,14)`, `(49,17)`, `(32,53)`, `(27,143)`, `(44,15)`, `(38,14)`, `(39,11)`, `(36,73)`, `(22,37)`, `(30,50)`, `(38,76)`, `(13,89)`, `(16,35)`, `(36,11)`, `(22,45)`, `(19,36)`, `(41,17)`, `(38,145)`, `(23,49)`, `(47,14)`, `(50,26)`, `(27,146)`.
- Recommended vertical levels for first probe: `0`, `1`, `2`, `16`, `17`, `18`, `24`, `25`, `26`, `28`, `29`, `30`, `31`, `32`.

## Top Suspects

- `PSFC`: overall RMSE `525.288 Pa`, worst lead `h7`, first bad `h1`.
- `V`: overall RMSE `5.830 m/s`, worst lead `h13`, first bad `h2`.
- `P`: overall RMSE `228.122 Pa`, worst lead `h7`, first bad `h2`.
- `U`: overall RMSE `4.612 m/s`, worst lead `h12`, first bad `h4`.
- `V10`: overall RMSE `2.524 m/s`, worst lead `h11`, first bad `h3`.
- Selected h10 correlations: `corr(dU10,dU_k0)=0.996`, `corr(dV10,dV_k0)=0.996`, `corr(dPSFC,dP_k0)=0.716`, `corr(dV10,dP_k0)=-0.001`.

## Risks

- Wrfout-only attribution cannot identify the first failing tendency term.
- Static/base-state writer mismatches are excluded from dynamic ranking but remain a separate validation gate.
- The selected cells are a localization manifest, not proof that the bug originates at those cells.

## Next Target

Build CPU-WRF term savepoints for the selected h10 cells before relying on a JAX-only operator probe. First terms to expose: stage input parity, mass coupling, momentum advection, large-step PGF, Coriolis, source-tendency folding, acoustic U/V, MU/theta, W/PH, and boundary/spec-relax.
