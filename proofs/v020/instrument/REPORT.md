# v0.20.0 GPU probes — BUILD-ONLY, ready-to-fire (Sprint C, Opus)

**Branch:** `worker/opus/v020-probes` · **Status:** all four probe families BUILT +
CPU-dry-run-validated; **NO GPU command was executed** (GPU held by the live 24 h corpus).
Every GPU command is wrapped in `scripts/with_gpu_lock.sh` and skipped under
`V020_DRYRUN=1`. **Collision-safe: only NEW files — zero edits to `src/gpuwrf` or any
other worker's area** (`git status`: all `??`). Fire S0 → G0 → P4 → skill-ladder the
instant the GPU frees, zero further coding.

## Files created (all NEW)
Shared: `scripts/v020_probe_common.sh` (sourced helpers + the `v020_run_gpu` choke-point
that wraps every GPU command in `with_gpu_lock.sh` and no-ops it under `V020_DRYRUN=1`).
S0: `v020_s0_instrument.sh`, `v020_cfl_probe.py`, `v020_nsys_extract.py`.
G0: `v020_g0_decision.sh`, `v020_run_metrics.py`, `v020_g0_verdict.py`.
P4: `v020_p4_cfl_ladder.sh`, `v020_make_namelist.py`.
Ladder: `v020_skill_ladder.sh`, `v020_blowup_check.py`, `v020_skill_eval.py`.
Evidence: `proofs/v020/instrument/DRYRUN_EVIDENCE_*` (the four CPU dry-run outputs).

## What each probe measures + EXACT GPU-locked fire command
(cwd = repo root; each script GPU-gates internally via `with_gpu_lock.sh`, so invoke bare.)

### S0 — instrument (roadmap step 0; ~1 session, no risk)
Measures: WARM nsys trace of the all-7 1 h forecast → host launch count, GPU kernel
instances+time, top kernel families (`loop_*_fusion` + `pcrGtsvBatch` tridiagonal = L5
fusion targets), in-loop memcpy audit (~0), GPU-active/wall host-bound proxy; PLUS the
per-domain realized-CFL headroom table (T0).
```
scripts/v020_s0_instrument.sh          # knobs: V020_DURATION=900  V020_HOURS=1
```
Artifacts: `<rundir>/{s0_launch_kernel_counts.json,s0_per_domain_cfl.json,nsys_all7.nsys-rep,S0_SUMMARY.txt}`.

### G0 — fp32-vs-fp64 decision (THE gate; ~1 session)
Measures: warm fp64-vs-fp32 A/B on the DRAM-bound single grid (v0.17 bigswiss 460×460×44
~211k cols, working set >> 96 MiB L2). fp64 = production default; **fp32 =
`JAX_ENABLE_X64=false` + `GPUWRF_THOMPSON_FP32=1`** = the §2 "aggressive" all-fp32 downcast
proxy (p'/ph'/mu'/w storage AND PCR solve → fp32), NO source edit. Reduces to warm ms/step
ratio, peak VRAM incl. transient, dominant-kernel wall delta, did-it-fit → GO/KILL verdict.
```
scripts/v020_g0_decision.sh            # knobs: V020_G0_HOURS=1  V020_G0_DURATION=1200
# AFTER the dtype/liveness audit, supply the measured transient fp32 share:
#   V020_G0_FP32_TRANSIENT_FRAC=0.62 [V020_G0_BOTTLENECK_MOVED=true] scripts/v020_g0_decision.sh
```
Artifacts: `<rundir>/{G0_VERDICT.json,g0_manifest.json,fp64_arm.json,fp32_arm.json}`.
Verdict ∈ {G0-SPEED-GO, G0-CAPABILITY-GO, G0-KILL} per FINAL_FP32_SPRINT_PLAN §2 + ROADMAP §8.4.

### P4 — CFL-first dt/n_sound ladder (cheap real win; medium, gated)
Measures: per dt rung (24→27→30→36 s; 45/54 opt-in high-risk) a 6 h all-7 forecast gated on
no-blow-up (hard) + realized per-domain/boundary-ring CFL (diagnostic). Raises dt only after
a clean pass; n_sound reduction only on a passing dt.
```
scripts/v020_p4_cfl_ladder.sh
# knobs: V020_P4_HOURS=6  V020_P4_DT_RUNGS="24 27 30 36"  (add "45 54" for high-risk)
#        V020_P4_FORCE_NSOUND=1  (only after the n_sound env override is plumbed — see risk 1)
```
Artifacts: `<rundir>/{ladder.jsonl,namelists/,*_blowup.json,*_cfl.json,P4_SUMMARY.txt}`.

### Skill / stability ladder (the gate every v0.20 change must clear)
Measures: candidate (fp32 / P4 / fused, set via env/namelist) vs an fp64 oracle across
1-substep→1-step→1h→6h→24h→120h on the all-7 nest; each rung gates HARD no-blow-up +
wind/temp/cloud divergence-growth skill (bounded=PASS, escalating=FAIL); never widens.
Wired through `proofs/perf/v015/fp32_oracles/divergence_growth_metric.py` (the module
`tests/test_fp32_divergence_growth_metric.py` pins).
```
scripts/v020_skill_ladder.sh                                                    # fp64 self-ladder
V020_LADDER_ENV="JAX_ENABLE_X64=false GPUWRF_THOMPSON_FP32=1" scripts/v020_skill_ladder.sh   # fp32
V020_LADDER_NAMELIST=/abs/namelist_dt30.input scripts/v020_skill_ladder.sh                    # P4 dt=30
# knobs: V020_LADDER_RUNGS="substep step 1 6 24 120"  V020_LADDER_ORACLE=<dir>  V020_LADDER_MAXHOURS=120
```
Artifacts: `<rundir>/{ladder.jsonl,*_blowup.json,*_skill.json,LADDER_SUMMARY.txt}`.

## CPU-dry-run results (all PASS; no GPU touched)
- divergence metric unit test: **8/8 PASS**.
- `v020_cfl_probe.py` on real all-7 wrfout: per-domain CFL OK; dt-tree fixed to walk
  `parent_id` (d03–d09 correctly 2 s under d02); acoustic CFL ≈0.068; **binding = d02
  vertical-advective, ~7.2× dt headroom**.
- `v020_nsys_extract.py` on real nsys CSVs: 289k host launches / 260k kernel instances;
  top families `loop_add/multiply/subtract_fusion` + `pcrGtsvBatch<double>`; memcpy ~0.
- `v020_blowup_check.py` on valid CPU-WRF: **PASS** (after fp-noise-floor slack on
  hydrometeors + wide P band — caught a too-strict q≥0 floor).
- `v020_skill_eval.py`: self-compare **GATE PASS** (divergence ~0); A/B also bounded-PASS.
- `v020_g0_verdict.py`: teeth verified — null→KILL, clean→SPEED-GO,
  speed-fail-but-fits→CAPABILITY-GO, both-fail→KILL.
- `v020_run_metrics.py` / `v020_make_namelist.py`: reduce / namelist-variant OK.
- S0/G0/P4/skill drivers `V020_DRYRUN=1`: all end-to-end OK (GPU skipped, gates exercised
  on existing wrfout/CSV, summaries written). `bash -n`+`py_compile`+`--help` clean on all 12.

## Residual risks / decisions for the manager
1. **P4 n_sound plumbing GAP.** `nested_pipeline.py:213` hardcodes `acoustic_substeps=10`
   with no env override. The **dt ladder is fully drivable today** (namelist `time_step`).
   For n_sound rungs, a dycore/integration worker must add
   `acoustic_substeps=int(os.environ.get("GPUWRF_ACOUSTIC_SUBSTEPS", 10))` at that site;
   my P4 driver already exports it per rung (fires instantly when it lands) and SKIPs
   honestly until then (`V020_P4_FORCE_NSOUND=1` to attempt). No fake-green.
2. **No global fp32 mode exists — G0 fp32 arm = `JAX_ENABLE_X64=false` (+Thompson fp32).**
   The faithful AGGRESSIVE proxy §2 asks for; touches NO production source. A GO funds the
   surgical S1–S8 per-field rewrite; a KILL ends it.
3. **`transient_fp32_fraction` is operator-supplied, not auto-measured.** Driver leaves it
   `-1` (→ NaN → conservative); manager sets `V020_G0_FP32_TRANSIENT_FRAC` from the
   HLO/liveness audit. A GO is never claimed on a guessed fraction.
4. **DRAM grid is 211k cols, not the plan's "130–147k".** `V020_BIG_INPUT` defaults to
   bigswiss 460×460×44 (available, firmly past L2). Override if a 130–147k grid is minted.
   `grid.fp64_oom_target` (capability arm) defaults True (a 1 km grid fp64 OOMs on);
   confirm from the v0.18 OOM proof or a dedicated capability sub-run.
5. **CFL/skill use CPU-WRF wrfout in dry-run; real runs use fresh GPU warm state.** Skill
   envelope is currently the field space-time std (variability proxy); supply a true
   fp64-GPU run-to-run envelope for a rigorous gate (ROADMAP §8.7).
6. **nsys `--gpu-metrics-devices=all` may hit `ERR_NVGPUCTRPERM`** (HW counters blocked).
   The trace still yields launch/kernel counts + timeline (the primary metrics); drop that
   one flag if it errors — the extractor does not depend on it.
