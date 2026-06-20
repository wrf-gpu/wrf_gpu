# Canary/Tenerife File Inventory for v0.14 Field Validation

Date: 2026-06-10
Agent: GPT-5.5 xhigh filesystem inventory
Scope: read-only source/data inspection; wrote only this Markdown and the paired JSON.

## Verdict

The best v0.14 Canary field-validation truth is already present: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output` holds 15 complete L2 d02 CPU-WRF 72h truths, each with 73 hourly `wrfout_d01_*` and 73 hourly `wrfout_d02_*` files and WRF rc=0 provenance in paired `.cpu_wrf_backfill` manifests. The "nightly" corpus is real, but it is a case-bank/teacher-shadow corpus with point artifacts, not retained full-field or restart coverage. I found no complete Canary/Tenerife checkpoint/restart corpus under the requested roots: no retained `wrfrst_d0*` files in the canonical run roots, representative namelists use `restart = .false.`, and the v0.14 savepoint/checkpoint material is single-step debug instrumentation rather than resumable 24h/72h field truth.

## Ranked Candidate Run Sets

| Rank | Root | Kind | Domains | Frame Counts | Hours / Date Range | Checkpoints Present | Manifests / READMEs | Recommended Use |
|---:|---|---|---|---|---|---|---|---|
| 1 | `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output` | CPU-WRF backfill truth | d01,d02 | 15 cases x 73 frames/domain = 1095 d01 + 1095 d02 | 72h per case; 2026-04-29_18 through 2026-06-02_18 across corpus | No `wrfrst_d0*` | paired `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/*/.cpu_wrf_backfill/*_manifest.json` | Primary Canary L2 d02 72h CPU truth corpus |
| 2 | `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z` | CPU-WRF backfill truth | d01,d02 | 73 d01 + 73 d02 | 2026-05-01_18 through 2026-05-04_18 | No `wrfrst_d0*` | manifest in paired input root | Best single mandatory v0.14 Canary d02 gate |
| 3 | `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z` | retained input/run dir | d01,d02 | wrfout moved to backfill output; inputs present | `run_hours = 72`, `max_dom = 2`; `wrfbdy_d01` has 12 boundary times | No; `restart = .false.` | `namelist.input`, `wrfinput_d01/d02`, `wrfbdy_d01`, READMEs, hidden backfill manifest | GPU/JAX input root paired with selected CPU truth |
| 4 | `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z` | retained CPU run | d01,d02 | 73 d01 + 73 d02 plus inputs | 2026-05-09_18 through 2026-05-12_18 | No `wrfrst_d0*` | READMEs/logs and case-bank manifest | Alternate complete d02 72h truth/input in one root |
| 5 | `<DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank` | nightly / case bank | d02,d03,d04,d05 point shadows | 54 case manifests; 29 `completed_teacher_shadow_only`; 156 parquet point artifacts | dense completed window 2026-04-29_18 through 2026-05-30_18 | None | per-case `manifest.yaml` and `*point_shadows.manifest.yaml` | Case selection and station/point evidence, not full-field truth |
| 6 | `<DATA_ROOT>/canairy_meteo/runs/wrf_l3` | retained L3 CPU runs | d01-d05 | two complete 24h cases: 25 frames/domain; d03=50 total across complete cases | 2026-05-09 and 2026-05-21 cases, 24h | No; `restart = .false.` | READMEs/logs and case-bank manifests | Secondary d03/Tenerife 24h evidence, not 72h gate |
| 7 | `<DATA_ROOT>/canairy_meteo/runs/surface_geo_v2_1` | research L3 runs | d01,d02,d03 | three 24h-ish d03 roots, 25 frames/domain each | 2026-05-13_18 through 2026-05-14_18 | No | surface_geo manifests/summaries plus READMEs/logs | Research/regression evidence only |
| 8 | `<USER_HOME>/src/wrf_gpu2/proofs/m20/tost_run/gpu_wrfout` | GPU/JAX proof output | d02 | historical d02 outputs: case1 72, case2 34, case3/4 24 each | mixed May 2026 lead windows | No | proof tree only | Historical provenance, not CPU truth |
| 9 | `<DATA_ROOT>/canairy_meteo/gate_gwd_nested_v013b/out` and `<DATA_ROOT>/canairy_meteo/gate_revalidate_gwd8/out` | GPU/JAX gate outputs | d01,d02,d03 | 24 frames/domain in each output root | 2026-05-31_19 through 2026-06-01_18 | No | no local manifest found in output root | Prior nested 24h GPU evidence only |

## Nightly Runs

Exact roots:

- `<DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank`
- `<USER_HOME>/src/canairy_meteo/Gen2/artifacts/datasets/wrf_case_bank`

These resolve to the same sampled device/inode for files such as `20260501_18z/manifest.yaml`; treat them as the same case-bank storage, not independent copies. The bank currently has 54 case manifest directories with statuses: 29 `completed_teacher_shadow_only`, 16 `failed`, 8 `running`, and 1 `preflight_ready_full_wrf_not_started`.

What it contains:

- Per-case `manifest.yaml`.
- `l2_d02_point_shadows.{csv,parquet}` and manifest sidecars.
- `l3_d03_point_shadows`, `l3_d03_tnf_point_shadows`, `l3_d04_gc_point_shadows`, `l3_d05_lp_point_shadows` artifacts for completed cases.
- Manifest links to WRF workdirs under `Gen2/runs/wrf_l2/...` and `Gen2/runs/wrf_l3/...`; `<USER_HOME>/src/canairy_meteo/Gen2/runs` is a symlink to `<DATA_ROOT>/canairy_meteo/runs`.

What is missing:

- Full `wrfout_d02_*` and `wrfout_d03_*` histories inside the case-bank root.
- `wrfrst_d02_*` or `wrfrst_d03_*`.
- `wrfinput_*`, `wrfbdy_*`, and `namelist.input` inside the case-bank root.
- Full checkpoint/restart coverage. The bank is point-shadow/thin-evidence storage.

The closest match to the "around 30 nightly runs" hypothesis is the 29 completed teacher-shadow case-bank entries, not 30 full-field d02/d03 restartable run sets.

## Repeated With Checkpoints

I did not find the hypothesized 15 full Canary/Tenerife runs repeated with WRF checkpoint/restart files under the requested roots.

Evidence:

- Canonical retained run roots under `<DATA_ROOT>/canairy_meteo/runs/wrf_l2`, `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output`, `<DATA_ROOT>/canairy_meteo/runs/wrf_l3`, `<DATA_ROOT>/canairy_meteo/runs/surface_geo_v2_1`, `<DATA_ROOT>/canairy_meteo/runs/terrain_sweep`, and `<DATA_ROOT>/canairy_meteo/runs/phys_sweep` have zero `wrfrst_d0*` files.
- Representative selected L2 namelist has `run_hours = 72`, `max_dom = 2`, `restart = .false.`, `restart_interval = 100000`.
- Representative L3 namelists have `run_hours = 24`, `max_dom = 5`, `restart = .false.`, `restart_interval = 100000`.
- The retained L3 d03 examples have 25 hourly frames and `wrfbdy_d01` with 4 boundary times, consistent with 24h orientation, not 72h resumability.

Checkpoint-like locations found:

- `<USER_HOME>/src/wrf_gpu2/proofs/v0120/powered_tost_n15/pipeline_proofs`: durable GPU/JAX pipeline proof directories for 3 cases, not 15. One case includes `restart_in_pipeline.json`, but this is runner/pipeline resume proof state, not WRF `wrfrst`.
- `<USER_HOME>/src/wrf_gpu2/proofs/v013/savepoints*` and `<USER_HOME>/src/wrf_gpu2/proofs/v060/*savepoint*`: per-scheme savepoint parity artifacts, not Canary field run sets.
- `<USER_HOME>/src/wrf_gpu2/proofs/v014/*savepoint*`: v0.14 proof manifests and scripts for same-state/source debugging, not a full run corpus.
- `<DATA_ROOT>/wrf_gpu2/v014_same_state_wrf`, `<DATA_ROOT>/wrf_gpu2/v014_step1_qvapor_precall_savepoint`, and `<DATA_ROOT>/wrf_gpu2/v014_full_pre_rk_savepoint_hook`: referenced by v0.14 proof files, outside the requested roots, and single-step/debug instrumentation rather than restartable 72h truth.

## Best v0.14 Gate Choices

Use Canary L2 d02 72h as the mandatory v0.14 Canary field-parity gate:

- CPU truth: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- Input/run dir: `<DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z`
- Domain: d02, 3 km, 159 x 66 x 44 mass grid, 73 hourly frames from 2026-05-01_18 to 2026-05-04_18.
- Provenance: hidden `.cpu_wrf_backfill/20260603T000612Z_manifest.json` reports WRF rc=0 and final counts d01=73,d02=73.
- This is already the h1 field-falsifier case, so the short-run and final 72h gate share the same input/provenance chain.

Do not use d03 as the v0.14 mandatory 72h gate unless a new CPU-WRF truth campaign is authorized. Existing d03 truths are useful, but retained complete examples are 24h-oriented, have no `wrfrst`, and are not cleanly resumable to 72h from retained files.

## Open Uncertainties

| Uncertainty | Fastest Command To Resolve |
|---|---|
| If the user meant a checkpointed 15-case corpus outside the three requested roots, it was not found as a primary-root artifact. | `find <DATA_ROOT> -path '*/artifacts/envs/*' -prune -o -type f \( -name 'wrfrst_d02_*' -o -name 'wrfrst_d03_*' -o -iname '*checkpoint*' -o -iname '*savepoint*' \) -printf '%p\n' \| sed -n '1,240p'` |
| Whether any deleted/thinned nightly full wrfout can be reconstructed without rerunning WRF. | `rg -n "safe_output_dir|delete|thin_gridded|teacher_shadow_only|wrfout" <USER_HOME>/src/canairy_meteo/Gen2/artifacts/datasets/wrf_case_bank <USER_HOME>/src/canairy_meteo/Gen2/reports/disk_retention_recon_20260522_v1.md` |
| Whether a d03 72h truth exists under a validation root outside the requested canairy roots. | `find <DATA_ROOT>/wrf_gpu_validation <DATA_ROOT> -maxdepth 5 -type f -name 'wrfout_d03_*' -printf '%h\n' \| sort \| uniq -c \| sort -nr \| sed -n '1,120p'` |

## Commands Run

Key exact commands run, with raw output summarized above:

```bash
pwd && rg --files -g 'PROJECT_CONSTITUTION.md' -g 'AGENTS.md' -g '.agent/**' -g '*SPRINT*' -g '*sprint*'
sed -n '1,220p' PROJECT_CONSTITUTION.md
sed -n '1,220p' AGENTS.md
find .agent -maxdepth 4 -type f | sort | sed -n '1,240p'
find . -maxdepth 4 \( -iname '*contract*' -o -iname '*sprint*' -o -iname '*v014*' -o -iname '*v0.14*' \) -print | sort | sed -n '1,240p'
find .agent/skills -maxdepth 3 -type f -name 'SKILL.md' -print 2>/dev/null | sort
sed -n '1,220p' .agent/SPRINT-TRACKER.md
sed -n '1,240p' .agent/decisions/V0140-FIELD-PARITY-RELEASE-GATE.md
sed -n '1,240p' .agent/decisions/V0140-VALIDATION-PLAN.md
sed -n '1,220p' .agent/skills/validating-physics/SKILL.md
sed -n '1,260p' proofs/v014/canary_cpu_truth_inventory.md 2>/dev/null || true
find proofs/v014 -maxdepth 2 -type f | sort | sed -n '1,240p'
find <USER_HOME>/src/canairy_meteo -maxdepth 3 \( -iname 'README*' -o -iname '*.md' -o -iname '*manifest*' -o -iname '*runinfo*' -o -iname '*inventory*' \) -print 2>/dev/null | sort | sed -n '1,240p'
find <DATA_ROOT>/canairy_meteo -maxdepth 5 \( -iname 'README*' -o -iname '*.md' -o -iname '*manifest*' -o -iname '*runinfo*' -o -iname '*inventory*' \) -print 2>/dev/null | sort | sed -n '1,240p'
python - <<'PY'
# walked <USER_HOME>/src/wrf_gpu2, <USER_HOME>/src/canairy_meteo, <DATA_ROOT>/canairy_meteo;
# grouped WRF files by parent dir and summarized wrfout/wrfrst/wrfinput/wrfbdy/time ranges.
PY
find <USER_HOME>/src/wrf_gpu2 <USER_HOME>/src/canairy_meteo <DATA_ROOT>/canairy_meteo \( -path '*/.git/*' -o -path '*/__pycache__/*' -o -path '*/.pytest_cache/*' -o -path '*/artifacts/envs/*' -o -path '*/node_modules/*' \) -prune -o -type f \( -name 'wrfrst_d01_*' -o -name 'wrfrst_d02_*' -o -name 'wrfrst_d03_*' \) -printf '%h/%f\n' 2>/dev/null | sed -E 's#/wrfrst_d0[123]_[0-9:_-]+$##' | sort | uniq -c | sort -nr | sed -n '1,240p'
find <USER_HOME>/src/wrf_gpu2 <USER_HOME>/src/canairy_meteo <DATA_ROOT>/canairy_meteo \( -path '*/.git/*' -o -path '*/__pycache__/*' -o -path '*/.pytest_cache/*' -o -path '*/artifacts/envs/*' -o -path '*/node_modules/*' \) -prune -o -type f \( -iname '*checkpoint*' -o -iname '*savepoint*' -o -iname '*restart*' \) -printf '%h/%f\n' 2>/dev/null | sed -E 's#/[^/]*(checkpoint|savepoint|restart)[^/]*$##I' | sort | uniq -c | sort -nr | sed -n '1,240p'
find <DATA_ROOT>/canairy_meteo/runs <USER_HOME>/src/canairy_meteo <USER_HOME>/src/wrf_gpu2/proofs -type f \( -name 'wrfout_d01_*' -o -name 'wrfout_d02_*' -o -name 'wrfout_d03_*' \) -printf '%h/%f\n' 2>/dev/null | sed -E 's#/wrfout_d0[123]_[0-9:_-]+$##' | sort | uniq -c | sort -nr | sed -n '1,180p'
find <USER_HOME>/src/wrf_gpu2/proofs/v014 -maxdepth 2 -type f \( -iname '*checkpoint*' -o -iname '*savepoint*' -o -iname '*restart*' -o -iname '*wrfrst*' \) -printf '%p\n' | sort
find <USER_HOME>/src/wrf_gpu2/proofs/v060 -maxdepth 2 -type f \( -iname '*checkpoint*' -o -iname '*savepoint*' -o -iname '*restart*' -o -iname '*wrfrst*' \) -printf '%p\n' | sort | sed -n '1,120p'
find <USER_HOME>/src/wrf_gpu2/.claude/worktrees/v0110-restart -maxdepth 6 -type f \( -iname '*wrfrst*' -o -iname '*restart*' -o -iname '*checkpoint*' -o -iname '*savepoint*' \) -printf '%p\n' | sort | sed -n '1,160p'
rg -n "checkpoint|savepoint|restart|wrfrst|repeated|n=15|15" proofs/v014 .agent/decisions .agent/reviews .agent/memory/pending -g '*.md' -g '*.json' | sed -n '1,240p'
sed -n '1,220p' proofs/v014/wrf_same_state_marker_savepoint.md 2>/dev/null || true
sed -n '1,220p' proofs/v014/step1_qvapor_precall_savepoint.md 2>/dev/null || true
sed -n '1,220p' proofs/v014/full_pre_rk_savepoint_hook.md 2>/dev/null || true
python - <<'PY'
# extracted path-like fields from proofs/v014/*savepoint*.json.
PY
find proofs/v0120/powered_tost_n15 -maxdepth 4 -type f 2>/dev/null | sed -n '1,240p'
find proofs/v013 -maxdepth 4 -type f \( -iname '*tost*' -o -iname '*n15*' -o -iname '*case_*.json' -o -iname '*restart*' -o -iname '*resume*' \) 2>/dev/null | sort | sed -n '1,240p'
find <DATA_ROOT>/wrf_gpu_validation -maxdepth 4 -type f \( -iname '*case_*.json' -o -iname '*tost*' -o -iname '*n15*' -o -iname '*resume*' -o -iname '*restart*' -o -name 'wrfout_d02_*' -o -name 'wrfout_d03_*' \) -printf '%p\n' 2>/dev/null | sort | sed -n '1,240p'
find <DATA_ROOT>/canairy_meteo -maxdepth 5 -type f \( -iname '*case_*.json' -o -iname '*tost*' -o -iname '*n15*' -o -iname '*resume*' -o -iname '*restart*' \) -printf '%p\n' 2>/dev/null | sort | sed -n '1,240p'
find <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank -maxdepth 4 -type f \( -name 'wrfout_d01_*' -o -name 'wrfout_d02_*' -o -name 'wrfout_d03_*' -o -name 'wrfinput_d01' -o -name 'wrfinput_d02' -o -name 'wrfinput_d03' -o -name 'wrfbdy_d01' -o -name 'namelist.input' -o -name 'manifest.yaml' -o -name '*point_shadows.manifest.yaml' \) -printf '%h/%f\n' 2>/dev/null | sed -E 's#/(wrfout_d0[123]_[0-9:_-]+|wrfinput_d0[123]|wrfbdy_d01|namelist.input|manifest.yaml|[^/]*point_shadows.manifest.yaml)$##' | sort | uniq -c | sort -nr | sed -n '1,240p'
find <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank -maxdepth 3 -type d | sort | sed -n '1,180p'
find <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank -maxdepth 5 -type f \( -name 'wrfout_d02_*' -o -name 'wrfout_d03_*' -o -name 'wrfrst_d02_*' -o -name 'wrfrst_d03_*' \) -printf '%p\n' 2>/dev/null | sed -n '1,240p'
for f in <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank/20260501_18z/manifest.yaml <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank/20260521_18z/manifest.yaml <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank/20260521_18z_l2rerun/manifest.yaml <USER_HOME>/src/canairy_meteo/Gen2/manifests/phase15d_wrf_evidence_bank_inventory_v1.yaml <USER_HOME>/src/canairy_meteo/Gen2/reports/disk_retention_recon_20260522_v1.md; do echo '###' $f; sed -n '1,160p' "$f" 2>/dev/null || true; done
stat -c '%N %F' <USER_HOME>/src/canairy_meteo/Gen2 <USER_HOME>/src/canairy_meteo/Gen2/runs <USER_HOME>/src/canairy_meteo/Gen2/artifacts/datasets/wrf_case_bank <DATA_ROOT>/canairy_meteo/runs <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank 2>/dev/null
find <DATA_ROOT>/canairy_meteo/runs -maxdepth 3 -type f -name 'manifest.json' -o -name 'runinfo*' -o -name '*.runinfo' -o -name '*.log' -o -name 'README*' -o -name '*.md' 2>/dev/null | sort | sed -n '1,240p'
python - <<'PY'
# summarized key candidate roots under wrf_l2, wrf_l2_backfill_output, wrf_l3, case_bank, and GPU proof dirs.
PY
find <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z -maxdepth 1 -type f -printf '%f %s\n' | sort | sed -n '1,220p'
find <DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z -maxdepth 1 -type f -printf '%f %s\n' | sort | sed -n '1,240p'
find <DATA_ROOT>/canairy_meteo/runs/wrf_l2 <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output -maxdepth 2 -type f \( -name 'manifest.json' -o -name '*manifest*' -o -name '*runinfo*' -o -name '*.json' \) -printf '%p\n' 2>/dev/null | sort | sed -n '1,220p'
find <DATA_ROOT>/canairy_meteo -path '*20260501_18z_l2_72h_20260519T173026Z*' -type f \( -iname '*manifest*' -o -iname '*runinfo*' -o -iname '*.json' -o -iname '*.log' \) -printf '%p\n' 2>/dev/null | sort | sed -n '1,240p'
sed -n '1,220p' <DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z/.cpu_wrf_backfill/20260603T000612Z_manifest.json
sed -n '1,160p' <DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z/wrf_stdout.log
sed -n '1,180p' <DATA_ROOT>/canairy_meteo/runs/nightly_scale_up.log
find <DATA_ROOT>/canairy_meteo/runs/wrf_l2 -maxdepth 3 -path '*/.cpu_wrf_backfill/*manifest*' -printf '%p\n' | sort
python - <<'PY'
# summarized all .cpu_wrf_backfill manifests and all wrf_l2_backfill_output case frame counts.
PY
rg -n "run_hours|max_dom|restart|history_interval|restart_interval|frames_per_outfile|input_from_file" <DATA_ROOT>/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z/namelist.input <DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z/namelist.input <DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/namelist.input
python - <<'PY'
# used netCDF4 to inspect representative wrfinput/wrfbdy/wrfout headers for d02/d03 dimensions, attrs, and Times.
PY
find <DATA_ROOT>/canairy_meteo/runs/wrf_l2 <DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output <DATA_ROOT>/canairy_meteo/runs/wrf_l3 <DATA_ROOT>/canairy_meteo/runs/surface_geo_v2_1 <DATA_ROOT>/canairy_meteo/runs/terrain_sweep <DATA_ROOT>/canairy_meteo/runs/phys_sweep <USER_HOME>/src/wrf_gpu2/proofs <DATA_ROOT>/canairy_meteo -type f \( -name 'wrfrst_d01_*' -o -name 'wrfrst_d02_*' -o -name 'wrfrst_d03_*' \) -printf '%p\n' 2>/dev/null | sed -n '1,240p'
find <USER_HOME>/src/canairy_meteo/Gen2/artifacts/datasets/wrf_case_bank/20260501_18z -maxdepth 1 -type f -printf '%f %s\n' | sort
find <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank/20260501_18z -maxdepth 1 -type f -printf '%f %s\n' | sort
python - <<'PY'
# counted case-bank manifest statuses and point artifact totals under /home and /mnt roots.
PY
stat -c '%n dev=%d inode=%i size=%s' <USER_HOME>/src/canairy_meteo/Gen2/artifacts/datasets/wrf_case_bank/20260501_18z/manifest.yaml <DATA_ROOT>/canairy_meteo/artifacts/datasets/wrf_case_bank/20260501_18z/manifest.yaml 2>/dev/null
```

Validation command:

```bash
python -m json.tool proofs/v014/canary_file_inventory_gpt.json >/tmp/canary_file_inventory_gpt.validated.json
```

## Context-Sparing Handoff

- Objective: inventory Canary/Tenerife/Canairy files relevant to v0.14 field validation across the requested roots.
- Files changed: `proofs/v014/canary_file_inventory_gpt.md`, `proofs/v014/canary_file_inventory_gpt.json`.
- Best gate: L2 d02 72h CPU truth at `<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`.
- Nightly finding: `wrf_case_bank` has 29 completed teacher-shadow case entries and point artifacts, not full `wrfout`/`wrfrst`.
- Checkpoint finding: no retained canonical `wrfrst_d0*` corpus under requested roots; v014 savepoints are debug/single-step, not restartable field truth.
- d03 finding: retained complete d03 examples are 24h-oriented; no 72h d03 truth or resumable d03 restart set found.
- Commands run: listed above; raw output summarized rather than pasted wholesale.
- Proof objects produced: this Markdown and paired JSON.
- Validation: JSON validation command recorded; run after file creation.
- Unresolved risk: a checkpointed 15-case corpus may exist outside the three requested roots, especially if the manager meant `<DATA_ROOT>/wrf_gpu2` or a different validation root.
- Next decision: proceed with d02 selected gate, or authorize a broader `<DATA_ROOT>` checkpoint search / new d03 72h CPU truth campaign.
