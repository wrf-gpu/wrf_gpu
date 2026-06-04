# v0.9.0 naive-agent README gate — SAMPLE prep

- **Lane:** worker/opus/v090-naive-gate-prep (off worker/opus/trunk-0.9.0 @ 7b7c26e)
- **Date:** 2026-06-04
- **Scope:** SELECT + PACKAGE the public sample + manifest + no-GPU structural dry-check. NOT the README prose (separate lane), NOT the GitHub upload, NOT the GPU run (tag-time manager).

## Objective

Package a complete, self-contained Canary d02 case so the binding v0.9.0 gate can
run: a clean-room agent clones the tag, follows the README, downloads the asset,
runs ONE hour through the GPU port via `gpuwrf run ... --compare-cpu-dir ...`, and
dimension-compares against CPU-WRF.

## Selected case

`gpuwrf-canary-d02-sample`, from CPU-WRF/Gen2 backfill run
`20260429_18z_l2_72h_20260524T204451Z` (Canary Islands d02, 3 km nest).

- COMPLETE: 73 hourly d02 wrfout frames (full 72 h), output dir last modified
  2026-06-02 (2 days stable) — NOT the live-backfill case (live = `20260513...`,
  being written in `wrf_l2_backfill_staging/`; left untouched).
- d02 dims: west_east=159, south_north=66, bottom_top=44 (e_we=160/e_sn=67/e_vert=45 stag).
- Packaged the first 3 hourly frames (t=0/1h/2h) to keep the asset modest while
  satisfying the replay-boundary loader (needs >=2 history frames) and giving the
  1 h dim-compare its reference (t=1h) plus one margin frame (t=2h).
- Also packaged wrfinput_d02 + wrfbdy_d01 (Gen2Run inventory / metrics + bdy-width
  decode) + namelist + an in-asset README.

## Asset

- `release-assets/gpuwrf-canary-d02-sample.tgz`
- size **69,750,989 bytes (66.5 MiB)**
- sha256 **1260f182e47d84788f67a21aa1eb5426aacfe0e91d8ee684120b8e8052aad17a**
- Too large to commit; staged + sha-pinned in `proofs/v090/readme_case_manifest.json`.
  Per-file shas also in the manifest. `release-assets/.gitignore` keeps only the
  text artifacts (namelists + README) in git; the tarball + NetCDF binaries are ignored.

## The one substantive finding (and the honest fix)

The raw CPU-WRF backfill namelist sets `diff_opt=1` and `km_opt=4`. The GPU port's
fail-closed registry (`gpuwrf.io.namelist_check`) accepts `diff_opt in {0,2}` and
`km_opt in {0,1}` only, so the raw namelist is **BLOCKED** (verified: exit 2 with both
failures). ALL physics schemes pass (mp=8 Thompson, bl=5 MYNN, sfclay=5 MYNN-SL,
land=4 Noah-MP, ra_lw/ra_sw=4 RRTMG, cu=1/0 KF/none, sf_urban=0, diff_6th_opt=2,
w_damping=1, damp_opt=3).

The sample `namelist.input` is a faithful copy with EXACTLY ONE documented deviation:
`diff_opt 1->0`, `km_opt 4->0`. This is honest, not a shortcut:
`daily_pipeline._build_real_case` does **not** read diff_opt/km_opt from the namelist —
it builds `OperationalNamelist` with `diff_opt=0/km_opt=0` and applies horizontal
filtering via `diff_6th_opt=2/diff_6th_factor=0.12` (kept identical). So `0/0` is
exactly what the GPU port runs. The dimension gate is unaffected (dims fixed by
e_we/e_sn/e_vert). The verbatim original is preserved as
`namelist.input.cpu-wrf-original`. Both the file header and the manifest document this.

CAVEAT for the future value/RMSE gate: the CPU reference wrfout WAS produced with
diff_opt=1/km_opt=4 (3D-TKE), which the GPU port does not implement. The dimension
gate is fine; a value comparison must account for this scheme difference.

## Dry-check (no GPU) — all PASS

`proofs/v090/naive_gate_sample_drycheck.json`. Ran with `taskset -c 0-3`, GPU never touched.

- C1 sample namelist validates (all schemes supported) — PASS
- C2 `gpuwrf run` accepts the sample paths, passes cheap validation + the registry
  check, reaches the heavy `DailyPipelineConfig(...)` construction (cli.py:275) with
  the JAX pipeline import stubbed out so no GPU ran — PASS
- C2b `gpuwrf run --help` exposes --namelist/--input-dir/--compare-cpu-dir — PASS
- C3a namelist d02 grid consistent with wrfout d02 dims — PASS
- C3b all 3 packaged d02 frames have identical dims; `compare_wrfout_dimensions`
  returns PASS on a self-reference — PASS
- N1 negative control: raw CPU-WRF namelist (diff_opt=1/km_opt=4) is rejected (exit 2) — PASS
- N2 negative control: --namelist != <input-dir>/namelist.input rejected (exit 2) — PASS
- RT round-trip: extracted tarball is self-contained; namelist validates + CLI accepts — PASS

Note on environment: the CLI lives on `worker/opus/readme-runnability` (NOT trunk-0.9.0);
exercised via PYTHONPATH against the trunk-0.9.0 `gpuwrf` package. The main repo checkout
(`worker/opus/v040-s0`) has an OLDER namelist_check that only accepts cu_physics={0} and
lacks `physics_registry`; dry-checks were pinned to the worktree's trunk-0.9.0 src to
avoid that stale path.

## Exact tag-time gate command (requires a GPU)

```
gpuwrf run \
    --namelist        <DIR>/namelist.input \
    --input-dir       <DIR> \
    --output-dir      runs/canary_d02_sample \
    --domain          d02 \
    --hours           1 \
    --compare-cpu-dir <DIR>
```
`<DIR>` = the extracted `gpuwrf-canary-d02-sample` directory. PASS =
`runs/canary_d02_sample/proofs/dimension_compare.json` status PASS, exit 0.

## Remaining for the manager at tag time

1. Merge `worker/opus/readme-runnability` (gpuwrf CLI + README) into the tagged commit.
2. Upload `gpuwrf-canary-d02-sample.tgz` as a GitHub release asset on the v0.9.0 tag.
3. Pin the resolved URL + sha256 `1260f182...` in the README (replace the manifest
   `asset.download_url` placeholder).
4. Run the GPU gate command above; confirm dimension_compare PASS + exit 0. **This GPU
   run is the binding gate** — this lane did selection + packaging + a no-GPU structural
   dry-check only.
5. (Post-0.9.0) Extend the gate from dimension-compare to value/RMSE, accounting for the
   diff_opt/km_opt scheme difference.

## Risks

- **Asset not in git / not uploaded.** Mitigated by sha256 pinning (tarball + per-file)
  in the manifest; the tarball is reproducible from the named corpus paths.
- **CLI is on a different branch than trunk-0.9.0.** The gate cannot run until
  readme-runnability is merged to the tag. Flagged as tag-time step #1.
- **diff_opt/km_opt swap.** Honest and inert for the dimension gate; a future value gate
  must account for the CPU reference having used schemes the GPU port doesn't implement.
- **GPU run not performed here** (GPU held by another lane). The dry-check proves the
  asset reaches the GPU boundary cleanly, not that the 1 h forecast itself produces a
  matching-dim wrfout — that is the tag-time GPU gate.
```
