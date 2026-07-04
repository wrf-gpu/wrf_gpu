# Release notes - wrf_gpu v0.22.0

v0.22.0 is a default-safe feature-integration release on top of v0.21.1.
The default fp64 fused+AOT forecast path remains bit-identical to v0.21.1;
new runtime behavior is opt-in, validation-only, or fail-closed.

## What changed

- Integrated the v0.22 feature push into the public mirror as curated source,
  tests, and documentation updates.
- Kept the v0.21.1 boundary-stability fix and default forecast behavior intact.
- Added opt-in K2 `time_step` / `n_sound` tuning support, AOT signature hardening,
  portable WRF-root lookup, async wrfout support, and corrected release framing.
- Added public tests for the new opt-in features and fail-closed scaffolds.

## Authoritative feature labels

| Area | Label | Public release status |
|---|---|---|
| G0 two-way nesting feedback | LANDED, validated, opt-in | Child-to-parent feedback and smoothing are available behind explicit selection. Default one-way nesting remains unchanged. |
| F1 3-D TKE / Smagorinsky | LANDED, validated, opt-in | 3-D turbulence support is present with small-grid coverage. Full WRF deformation-stress/moist parity remains follow-up. |
| G2 375-variable output stream | LANDED, validated, opt-in | Expanded WRF-history and auxhist output support is available when explicitly configured. Default output remains unchanged. |
| G1 data assimilation | LANDED, validated, opt-in | Nudging tendencies and a finite DFI path are present. Full raw-observation ingest and complete WRF DFI choreography remain follow-up. |
| E validation harness | LANDED, validated, opt-in | Small-grid validation harnesses are available for explicit release/feature checks. This is tooling, not a physics default. |
| F2 cumulus + microphysics + LSM bundle | RECOGNIZED-SCAFFOLD, fail-closed, default-off | New scheme targets are recognized and tested as unsupported where needed. They are not WRF-faithful operational GPU ports yet. |
| F3 CAM-UW PBL | RECOGNIZED-SCAFFOLD, fail-closed/experimental, default-off | CAM-UW selection is visible for follow-up work, but production WRF parity is not claimed. |
| G2 moving nests / adaptive timestep / global nests | RECOGNIZED-SCAFFOLD, fail-closed/experimental, default-off | Moving-nest state shifts and adaptive-dt planning are scaffolded. Runtime nest-weight rebuild, vortex-following choreography, and global/polar validation remain v0.22.x work. |
| G3 urban + lake | RECOGNIZED-SCAFFOLD, fail-closed, default-off | Urban/lake options are recognized with explicit unsupported paths. No faithful urban/lake physics kernel ships in v0.22.0. |

## Validation

- Default behavior was verified on the source release branch as bit-identical to
  a fresh v0.21.1 paired baseline.
- Focused CPU tests cover the public mirror sync, including the AOT signature,
  WRF-root portability, async output, boundary/state contracts, and v0.22
  feature/scaffold tests.
- No GPU validation was run for this public mirror sync; it is a source/version
  synchronization of the shipped v0.22.0 release.

## Operational notes

The scaffold rows are intentionally visible so unsupported selections fail loudly
instead of silently running an unfaithful substitute. Treat scaffolded schemes as
development targets for v0.22.x, not as production WRF-equivalent physics.

Use explicit configuration and small-grid validation before enabling any new
v0.22 feature in an operational run.
