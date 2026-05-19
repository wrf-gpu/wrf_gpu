# Fixture Storage Policy

This policy applies to M1 fixture manifests, sample slices, WRF-derived payloads, profiler artifacts, and candidate outputs.

## Naming

- Manifest files live under `fixtures/manifests/` and use `<fixture_id>.yaml`.
- `fixture_id` uses lowercase words joined by hyphens and ends with a version suffix such as `-v1`.
- External fixture payloads live under `data/fixtures/<fixture_id>/`.
- Large run products live under `data/runs/<run_id>/`.
- Profiler dumps live under `data/profiler_artifacts/<sprint_or_candidate>/`.
- Scratch files live under `data/cache/`.

`data/` is a repository symlink to `/mnt/data/wrf_gpu2/`. If the local symlink is unavailable, manifests may use an `external_uri` with a placeholder S3-style URI such as `s3://wrf-gpu2-fixtures/<fixture_id>/`, but local validation still requires checksums in the manifest.

## Checksums

- Every entry in `files` records `checksum_sha256`, `bytes`, and whether the `path` is external.
- Checksums are SHA-256 over the exact bytes consumed by the comparison harness.
- Repacking NetCDF, GRIB, Zarr, or NumPy payloads changes the checksum and requires a manifest update.
- Checksums are generated before review and must not be adjusted after looking at candidate output.

## What May Be Committed

- Fixture manifests in `fixtures/manifests/`.
- JSON-Schema files and validator code.
- Sample slices referenced by `sample_slice_path` when each file is no larger than 100 KB.
- JSON or CSV smoke arrays no larger than 100 KB when a later sprint explicitly owns them.
- Documentation describing storage, checksums, and validation commands.

## What Must Stay Out Of Git

- NetCDF, GRIB/GRIB2, Zarr, HDF5, raw binary dumps, and large NumPy `.npy` or `.npz` payloads.
- WRF run directories and candidate run outputs.
- `ncu`, `nsys`, SQLite, CUPTI, or other profiler dumps.
- Any file under `data/`, which points at external storage.

## Retention

- Manifests are retained in git for auditability.
- External fixture payloads are retained at least until all sprint branches that reference them are merged or rejected.
- Profiler artifacts are retained through the milestone closeout that consumes them.
- Scratch files under `data/cache/` may be removed after the producing sprint writes its proof objects.
