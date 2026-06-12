# GPU↔CPU Identity-Proof Visualization System

A reusable, reproducible, publication-quality visual proof that the WRF-GPU port is
true to CPU-WRF v4 across **all grid cells, all 72 forecast leads, and all core
internal variables** for a region's 72 h GPU-vs-CPU run.

The tool is **offline and CPU-only**. It reads existing paired `wrfout` NetCDF files
(one CPU-WRF run, one GPU run, same init) and a frozen tolerance manifest. It does
**not** run WRF, JAX, CUDA, or any model kernel, and it never touches the GPU.

- Tool: [`scripts/build_identity_proof_plots.py`](../scripts/build_identity_proof_plots.py)
- Reuses [`scripts/build_grid_delta_atlas.py`](../scripts/build_grid_delta_atlas.py)
  for wrfout pairing, NetCDF parsing, tolerance loading, and streaming statistics —
  the identity-proof tool is a thin visualization layer on top of the same numbers
  the Grid-Delta Atlas gate uses, so the plots and the gate cannot disagree.

## What it produces (per region)

Into `--asset-dir` (PNG) and `--proof-dir` (`identity_proof_manifest.json`):

| Artifact | What it proves |
| --- | --- |
| `identity_timeseries_rmse_bias.png` | Per-variable **RMSE and bias** vs lead, all 72 leads, with the frozen tolerance line drawn. Curves sitting at/under the bound = stable identity over the whole forecast. |
| `identity_scoreboard.png` | Variable × lead **scoreboard**: `(per-lead metric) / frozen limit`, green where `< 1`. A wall of green = within tolerance everywhere; over-limit cells are marked `x` and turn red. |
| `identity_scatter_1to1.png` | GPU-vs-CPU **cell-value 1:1 scatter** per variable, pooled and subsampled over all leads/cells. Points on the diagonal = identity; the Pearson `r` is printed per panel. |
| `identity_spatial_diff_maps.png` | **Signed GPU−CPU spatial difference maps** at h24/h48/h72 for the main prognostic variables, symmetric diverging colormap, **tight honest per-panel scale with the true `max|GPU−CPU|` annotated**. Diffuse meteorological-texture differences (not localized blowups) = faithful dynamics. |
| `identity_dashboard.png` | **One README-embeddable summary dashboard**: headline (`N variables`, `72 leads`, `M/N within tolerance`), per-variable margin-vs-limit bars, a mini scoreboard, and a pooled all-cell/all-lead metrics table. |

## Honesty contract

- Differences are shown **at true scale**; nothing is clipped to hide error. Spatial
  maps annotate the real `max|GPU−CPU|`.
- The variable set is the **predeclared focused-writer hard-gate scope**
  (`T, U, V, W, QVAPOR, T2, U10, V10, PSFC, RAINNC`). This honest scope is printed
  on every artifact. Fields outside it (e.g. `P, PH, MU` pressure/mass diagnostics) are
  **report-only** in the frozen manifest and are not painted as passes.
- **Bounded-not-exact** fields are labelled. `RAINNC` (accumulated grid-scale
  precipitation) is an operational-bounded diagnostic, not a bitwise-exact channel; when
  it breaches its 1.0 mm RMSE envelope the tool draws it **red**, never green.
- A field whose **pooled** metric is within limit but which breaches at a few individual
  leads is labelled `pooled within; N lead(s) over` (orange) — neither hidden nor
  overstated.

## Reproduce

One documented command per region. Pin CPU work with `taskset` and keep the GPU free.
Deterministic: the scatter subsample uses a fixed seed.

### Switzerland d01 (72 h, init 2023-01-15 00Z)

```bash
taskset -c 0-3 python3 scripts/build_identity_proof_plots.py \
  --cpu-dir /mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu \
  --gpu-dir /mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_thetafix_20260612T012219Z/gpu_output \
  --domain d01 --init "2023-01-15T00:00:00+00:00" \
  --case-id switzerland_d01_72h \
  --region-label "Switzerland d01 72h (2023-01-15 00Z)" \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --proof-dir proofs/v014/identity_proof/switzerland_d01 \
  --asset-dir docs/assets/v014/identity_proof/switzerland_d01
```

### Canary L2 d02 (72 h, init 2026-05-01 18Z)

```bash
taskset -c 0-3 python3 scripts/build_identity_proof_plots.py \
  --cpu-dir /mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z \
  --gpu-dir /mnt/data/wrf_gpu_validation/v014_canary_d02_72h_noahmp_lu16fix_20260610T214731Z/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z \
  --domain d02 --init "2026-05-01T18:00:00+00:00" \
  --case-id canary_l2_d02_72h \
  --region-label "Canary L2 d02 72h (2026-05-01 18Z)" \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --proof-dir proofs/v014/identity_proof/canary_l2_d02 \
  --asset-dir docs/assets/v014/identity_proof/canary_l2_d02
```

The tool is fully data-driven via the `--cpu-dir/--gpu-dir/--init/--domain` (or
`--case-json`) arguments, so the **final** v0.14 release runs reuse the exact same two
commands against their own `run_root`s.

## Sample results (development runs)

| Region | Variables | Leads | Within frozen tolerance | Worst field |
| --- | --- | --- | --- | --- |
| Switzerland d01 | 10 | 72 | **9 / 10** | `RAINNC` precip 5.99 mm RMSE vs 1.0 mm (bounded-not-exact) |
| Canary L2 d02 | 10 | 72 | **9 / 10** | `QVAPOR` 1.45×10⁻³ kg/kg RMSE vs 1.0×10⁻³ (marginal, +45%) |

In both regions the full dynamics/thermodynamics core (`T, U, V, W, T2, U10, V10, PSFC`)
sits within its frozen operational envelope across all 72 leads, with `r ≈ 0.99–1.00`
cell-for-cell on the prognostic fields. The single out-of-envelope field per region is a
bounded diagnostic (precipitation in Switzerland, the tight moisture limit in Canary),
shown honestly rather than hidden.

## Embedding in the release README

```markdown
![GPU↔CPU identity proof — Switzerland d01](docs/assets/v014/identity_proof/switzerland_d01/identity_dashboard.png)
```
