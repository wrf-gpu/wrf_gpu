# Sprint Contract: V0.14 WRF Same-State Marker Savepoint

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Create the first source-derived CPU-WRF same-state savepoint artifact for the
v0.14 grid-parity investigation, starting with a marker/step-mapping proof.

The first accepted output is not a full root cause. It must prove that an
instrumented WRF run can target the selected h10 `d02` state, selected native
indices, and selected patch bounds correctly. If and only if the marker is
green, emit the first routine-boundary savepoint layer around the requested WRF
term groups.

## Inputs

- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/same_state_savepoint_request.md`
- `proofs/v014/same_state_wrf_savepoint_feasibility.json`
- `proofs/v014/dynamic_field_attribution.json`
- `proofs/v014/base_state_writer_attribution.json`

## Required WRF Path Policy

- Do **not** patch `/home/enric/src/wrf_pristine/WRF` in place.
- Make a disposable copy under `/mnt/data/wrf_gpu2/v014_same_state_wrf/` if
  possible. If that is unavailable, use `/tmp/wrf_gpu2_v014_same_state_wrf/`
  and record the fallback.
- Record source provenance before patching: git head, `git describe`, dirty
  status, source path, executable hashes, compiler/build hints, and a patch
  hash.
- Treat `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF` as
  provenance/source comparison only unless a build is found.

## Write Scope

Repository write scope:

- `proofs/v014/wrf_same_state_marker_savepoint.json`
- `proofs/v014/wrf_same_state_marker_savepoint.md`
- `proofs/v014/wrf_same_state_marker_patch.diff`
- `.agent/reviews/2026-06-09-v014-wrf-same-state-marker-savepoint.md`

External scratch write scope:

- `/mnt/data/wrf_gpu2/v014_same_state_wrf/**`
- fallback `/tmp/wrf_gpu2_v014_same_state_wrf/**`

No edits to repo `src/`. No GPU. No Hermes. No source edits outside the
disposable WRF copy.

## Required Work

1. Prepare a disposable WRF instrumentation tree from
   `/home/enric/src/wrf_pristine/WRF`.
2. Prepare a disposable Case 3 run directory from
   `/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`.
3. Add an env-gated marker emitter in the disposable WRF copy. Prefer the
   smallest `solve_em.F` boundary hook that can record:
   - domain id;
   - WRF `Times` / current date string;
   - `grid%itimestep` or equivalent;
   - `dt`, `dx/dy` if available;
   - the target selected h10 mass cells and one native U/V/W/PH face context;
   - one or more marker field patches that can be compared to CPU h10 wrfout,
     using native indices and halo 8 from the request manifest.
4. Run a marker validation before any broad term emission:
   - proves the selected h10 marker corresponds to
     `2026-05-02_04:00:00` / lead h10 for `d02`;
   - proves zero-based request indices were converted to WRF one-based native
     indices correctly;
   - compares at least one selected marker patch against CPU h10 wrfout within
     serialization tolerance.
5. If marker validation is green and wall-clock remains reasonable, emit the
   first routine-boundary layer for the request term groups using raw
   little-endian f64/binary plus JSON sidecars or a compact NetCDF/HDF5/Zarr
   artifact. If full term emission is too large, emit a smaller first layer and
   explicitly name the next layer.

## First-Pass Term Groups

Use the request manifest's groups as the target taxonomy:

- `stage_input`
- `mass_coupling`
- `momentum_advection`
- `scalar_theta_mu_advection`
- `diffusion`
- `horizontal_pgf`
- `coriolis`
- `source_tendency_folding`
- `small_step_prep`
- `acoustic_uv`
- `mu_theta`
- `w_ph`
- `pressure_rho_refresh`
- `boundary_spec_relax`
- `final_stage_state`

The marker sprint may stop after `stage_input` and routine-boundary pre/post
snapshots if deeper arrays require a follow-up patch. It must say exactly what
was emitted and what was deferred.

## Commands / Validation

At minimum, run:

```bash
python -m json.tool proofs/v014/wrf_same_state_marker_savepoint.json \
  >/tmp/wrf_same_state_marker_savepoint.validated.json
```

If helper scripts are created under the scratch tree, record exact run commands
in the JSON/MD review. CPU-only execution must be explicit where applicable:

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ...
```

## Acceptance Criteria

- Original `/home/enric/src/wrf_pristine/WRF` remains unmodified by this sprint.
- Disposable WRF source path, patch diff/hash, run path, executable hash, and
  input checksums are recorded.
- Marker proof either:
  - `GREEN`: correct h10/domain/indices plus patch-vs-wrfout comparison, or
  - `BLOCKED`: concrete build/runtime blocker with logs and next command.
- If term savepoints are emitted, the artifact schema names routine, event,
  RK stage, acoustic substep, native stagger, shape, dtype, index origin,
  checksum, and source variable for every dataset.
- No JAX-vs-JAX self-truth and no equivalence/root-cause claim.

## Closeout

Close with:

- verdict: `MARKER_GREEN`, `MARKER_BLOCKED_<reason>`, or
  `TERM_LAYER_EMITTED`;
- files changed;
- exact WRF copy/run paths;
- commands run;
- proof objects and logs;
- unresolved risks;
- next sprint recommendation: full term emission, JAX same-state comparator,
  build fix, or source-path correction.
