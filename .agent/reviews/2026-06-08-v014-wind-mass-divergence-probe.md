# V014 Wind/Mass Divergence Probe

Date: 2026-06-08
Worker: GPT xhigh
Scope: CPU-only retained-wrfout anatomy. No GPU run. No `src` edits.

## Objective

Run the first V10/wind-mass anatomy probe recommended by the prior-attribution
sidecar on Case 3 retained wrfouts:

- GPU: `/tmp/v0120_powered_tost_runs/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`

## Files Changed

- `proofs/v014/wind_mass_divergence_probe.py`
- `proofs/v014/wind_mass_divergence_probe.json`
- `proofs/v014/wind_mass_divergence_probe.md`
- `.agent/reviews/2026-06-08-v014-wind-mass-divergence-probe.md`

## Commands Run

```bash
python -m py_compile proofs/v014/wind_mass_divergence_probe.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/wind_mass_divergence_probe.py
python -m json.tool proofs/v014/wind_mass_divergence_probe.json \
  >/tmp/wind_mass_divergence_probe.validated.json
```

## Proof Objects Produced

- `proofs/v014/wind_mass_divergence_probe.json`
- `proofs/v014/wind_mass_divergence_probe.md`

The JSON validates. The script compared all requested compatible fields with no
skips: `U`, `V`, `W`, `T`, `QVAPOR`, `P`, `PH`, `MU`, `MUB`, `PB`, `PHB`,
`U10`, `V10`, `T2`, `PSFC`.

## Key Findings

- V10 remains a true grid-field failure: pooled RMSE `2.524 m/s`, bias
  `+1.036 m/s`; h10-h14 RMSE worsens to `3.721 m/s`, bias `+2.785 m/s`.
- 3D winds are worse than 10 m diagnostics: native U RMSE `4.612 m/s`, V RMSE
  `5.830 m/s`; h10-h14 U/V RMSE `5.944/7.151 m/s`.
- Surface wind is nearly locked to low-level prognostic wind:
  `corr(dV10,dV_k0)=0.998`, `corr(dU10,dU_k0)=0.998`.
- Mass/pressure divergence is real: PSFC RMSE `525 Pa`, P RMSE `228 Pa`, PH
  RMSE `336 m2/s2`; `corr(dPSFC,dP_k0)=0.916`.
- Boundary-frame dominance is disfavored: V10 RMSE in the 5-cell frame is
  `2.541 m/s`; interior excluding that frame remains `2.519 m/s`.
- Static/base fields are compatible but not identical: MUB RMSE `58.8 Pa`, PB
  RMSE `28.6 Pa`, PHB RMSE `45.4 m2/s2`. This is a plausible contributor or
  wrfout/base reconstruction artifact, but not a full explanation because it is
  lead-invariant while the wind error peaks dynamically.
- Pure 10 m diagnostic bug is disfavored because the 3D wind field is already
  divergent and strongly coupled to U10/V10.

## Ranked Root-Cause Hypotheses

1. Favored: prognostic wind-column divergence with near-surface projection,
   coupled to mass/geopotential error, peaking around h10-h14.
2. Plausible contributor: static base-state or wrfout/grid-base reconstruction
   mismatch contributing to mass/geopotential residuals.
3. Plausible but unproven: surface/PBL or source-tendency cadence feedback
   amplifies a real low-level wind error after early leads.
4. Disfavored: boundary-frame forcing defect/regression as dominant cause.
5. Disfavored: pure 10 m diagnostic sign/formula bug.
6. Low priority unless contradicted by tendency evidence: old absent-Coriolis or
   post-step-only normal-boundary bug reappearing unchanged.

## Unresolved Risks

- This is wrfout anatomy only. It cannot identify the first bad tendency term or
  direction of causality between wind and mass.
- MUB/PB/PHB differences need a focused check to separate true model-state
  base mismatch from writer/reconstruction/reference artifacts.
- Only retained Case 3 wrfouts were available for full 3D spatial anatomy here;
  the prior three-case V10 sign changes still need cross-case confirmation once
  more retained wrfouts exist.

## Next Fix Probe

Run a CPU-only same-state tendency localization over h8-h14, sampled on
ocean/low-terrain interior cells where V10 and PSFC both fail. Split large-step
momentum and mass terms into PGF, Coriolis, advection, diffusion,
boundary/spec-relax, physics/source-tendency folding, and resulting
`ru`/`rv`/`mu` updates. Include an explicit base-state/writer parity subcheck for
`MUB`, `PB`, and `PHB` before attributing the static offsets to dynamics.
