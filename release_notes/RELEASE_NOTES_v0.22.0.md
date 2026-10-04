# wrf_gpu v0.22.0

v0.22.0 is a default-safe hygiene, opt-in performance, and decision-document
release on top of v0.21.1. It does not change the default forecast configuration:
the shipped fp64 fused+AOT path remains the v0.21.1 path unless an operator
explicitly changes namelist or environment settings.

## What Changed

- **v0.22 feature-push integration:** the release includes nine feature branches
  integrated at `9271dfc8`. The table below is authoritative: rows marked LANDED
  have small-grid validation; runtime features remain default-off unless
  explicitly selected. Rows marked SCAFFOLD are recognized, partial/experimental,
  fail-closed, default-off, and are not WRF-faithful operational implementations
  yet.
- **AOT signature hardening (#136):** strict treedef/leaf-count splitting is
  available behind `GPUWRF_AOT_STRICT_AVAL_SIGNATURE=1`. The release default
  keeps the v0.21.1 leaf-aval-only fused-call signature because the strict split
  changes the nested default trajectory and therefore is not bit-identical.
- **Portable WRF data lookup (#115):** physics table, Fortran-source oracle, and
  related data lookups honor `GPUWRF_WRF_ROOT` when set and retain the existing
  fallback behavior when unset.
- **Async history output (#101):** wrfout history writing can overlap with the
  next compute step while preserving byte-identical files versus the synchronous
  writer.
- **K2 dt / n_sound opt-in lever:** the normal WRF-style `time_step` / `n_sound`
  settings are documented as a CFL-gated single-domain lever. The short
  Switzerland-128 gate measured `18 s / 7` at `19.25 s/fc-h` versus `34.67
  s/fc-h` for `10 s / 10` (1.80x), finite/bounded and within the short
  operational band. This is not a default change, not a nested speedup claim, and
  not a 24-72 h skill claim.
- **Compile-wall correction:** the relevant wall is compile time. On the sound
  3-domain harness, peak compile RSS was about 22.32 GiB, not the old roughly
  60 GB premise; the AOT warm-start remains the practical answer and measured
  about 39x cold-to-warm on that harness.
- **ADRs:** the operational-relaxed acceptance-tier ADR is ratified for the
  §9 contract as written, including the 1 km Alpine case and hard rejection of
  `BOUNDED_GROWTH`. The K4 fp32-operational ADR remains a plan and proof inventory
  only; v0.22 ships no fp32 implementation/default flip.

## Authoritative Feature Labels

| Area | v0.22.0 label | What ships |
|---|---|---|
| Base release line | LANDED | K2 dt/n_sound opt-in single-domain lever, ADR §9 ratification, K4 fp32 plan ADR, AOT/WRF-root/async-output hygiene, compile-wall correction, and corrected paired-baseline canary gate. Defaults remain the v0.21.1 fp64 fused+AOT path. |
| G0 two-way nesting feedback | LANDED, validated, opt-in | Child-to-parent feedback with WRF copy_fcn-style area averaging plus sm121 smoothing. Small-grid gate passes finite/conservation/reflection checks. Default one-way nesting is unchanged. |
| F1 3-D TKE / Smagorinsky | LANDED, validated, opt-in | `diff_opt=2` / `km_opt=2,3,5` turbulence path with GPU oracle coverage on the small grid. Exact WRF deformation-stress and moist BN2 parity remain follow-up work, so users must opt in deliberately. |
| G2 375-variable output stream | LANDED, validated, opt-in | Full WRF-history variable set and auxhist stream support are available behind explicit config/env selection. Default wrfout payload is unchanged. |
| G1 data assimilation | LANDED, validated, opt-in | Resident analysis/obs/spectral nudging tendencies plus a finite DFI launch path. Raw obs ingest, full WRF backward+forward DFI choreography, and auxinput readers remain follow-up. |
| E validation harness | LANDED, validated | Small-grid operational-relaxed validation harness and per-feature synthetic smallest-Canary rollups. This is a validation tool, not a physics feature. |
| F2 cumulus + microphysics + LSM bundle | SCAFFOLD, fail-closed, default-off | New Tiedtke, NSSL/Morrison-aero, and RUC-style targets are cataloged/reference-aware where possible, but not operational GPU ports. Selection fails closed until WRF-faithful kernels and savepoint parity land. |
| F3 CAM-UW PBL | SCAFFOLD, fail-closed/experimental, default-off | CAM-UW option and idealized/source-present path are wired for exploration, but there is no pristine-WRF CAM-UW numerical savepoint parity claim. Treat as partial, not production WRF-faithful PBL. |
| G2 moving nests / adaptive timestep / global nests | SCAFFOLD, fail-closed/experimental, default-off | Moving-nest metadata/state-shift and adaptive-dt planner pass the small-grid gate. Runtime rebuild of nest/feedback weights, full vortex-following choreography, and polar/global validation remain v0.22.x work. |
| G3 urban/lake | SCAFFOLD, fail-closed, default-off | BEP/BEM urban and WRF lake options are recognized with source/provenance and explicit fail-closed errors. No faithful urban/lake physics kernel or WRF parity run ships in v0.22.0. |

## Validation

- CPU hygiene tests:
  - `tests/test_aot_executable.py::test_aval_signature_default_keeps_v0211_leaf_aval_key`
  - `tests/test_aot_executable.py::test_aval_signature_strict_mode_includes_treedef_and_leaf_count`
  - `tests/test_wrf_root_portability.py`
  - `tests/test_async_wrfout_equiv.py`
- K2 single-domain proof:
  `proofs/v022/k2_dt_ladder/k2_gate_summary.json`.
- K2 nested caution proof:
  `proofs/v022/k2_nest_ladder/K2_NEST_LADDER_SYNTHESIS.md`.
- Corrected release canary gate:
  `proofs/v022/release_prep/V0220_CORRECTED_CANARY_GATE.md`. The fresh
  matched v0.21.1 baseline digest is `9709039c...`, and the v0.22 default
  digest is also `9709039c...`. The old hardcoded `519cd3e5...` target is
  invalidated as a stale cold-compile/autotune artifact, not a source
  regression.
- Feature integration verdict:
  `proofs/v022/feature_push/V022_INTEGRATION_VERDICT.md`.

## Operational Notes

Defaults stay unchanged from v0.21.1. The corrected paired-baseline gate confirms
the v0.22.0 default digest matches the fresh v0.21.1 matched-env/AOT warm
baseline. To experiment with K2 or any v0.22 feature-push item, change the
namelist/environment deliberately and gate the run with CFL, finite/bounded state
checks, and skill validation. Do not infer nested safety from the single-domain
K2 result; the 3-domain steep-terrain ladder exposed vertical-CFL risk before
any higher rung could be evaluated.

The SCAFFOLD rows above are intentionally partial and default-off. They are
visible so unsupported selections fail loudly with named reasons rather than
silently running an unfaithful substitute.

For canary release gates, pair the candidate against a fresh v0.21.1 baseline
captured under matched env/AOT, or fall back to the ratified operational-identity
tier when same-HLO cold-compiled executable blobs serialize differently.
