# M6b D2H Warmed Re-capture v2 Memo

Sprint: `2026-05-25-m6b-d2h-warmed-recapture-v2` (opus tester, parallel with M6b V3).
Branch: `tester/opus/m6b-d2h-warmed-recapture-v2` in worktree `/tmp/wrf_gpu2_d2h_v2`.

Inputs:

- `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/d2h_warmed_memo.md` (prior NO-GO: `d2h_inter_kernel = 20` on pre-lift `operational_mode.py`).
- `.agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/worker-report.md` (bisection localised the residual to `operational_mode.py:353-361` RK `lax.switch` and `:374-380` radiation cadence `lax.cond`).
- `.agent/decisions/ADR-027-d2h-invariant-clarification-PROPOSED.md` (constitutional invariant clarified as `d2h_inter_kernel == 0`).
- `src/gpuwrf/runtime/operational_mode.py` post-reframe (RK1 stages now statically sequenced; radiation cadence segmented outside the scan body).

Status: **VERDICT — GO-D2H-CLEAR.** The post-reframe `operational_mode.py` was independently re-profiled in this worktree under the warmed Nsight protocol (3 warmups + 5-step capture). The constitutional invariant `d2h_inter_kernel == 0` is satisfied. ADR-027 remains at PROPOSED status, now independently corroborated.

---

## Part 1 — Capture protocol

Orchestrator: `scripts/m6b_d2h_warmed_recapture.py` (the same script the prior sprint built; extended in v2 with a `GPUWRF_D2H_SPRINT_DIR` env-var override so v2 proofs land in the v2 sprint folder without overwriting the prior sprint's canonical artefacts).

Discipline applied (per the sprint contract):

1. Build a Gen2 d02 replay case (run `20260521_18z_l3_24h_20260522T072630Z`) producing operational shapes (mass `(44, 66, 159)`, staggered `(45, 67, 160)`).
2. Run **three** untimed warm-up calls of `run_forecast_operational(state, namelist, hours=50s/3600s)` — all outside the `cudaProfilerStart`/`Stop` window. First warmup compiles + stages constants; the second and third drain XLA settling.
3. `cudaProfilerStart` via `ctypes.CDLL("libcudart.so")`.
4. One profiled call with identical `(state shape, namelist tree, hours)` signature → guaranteed JIT cache hit.
5. `cudaProfilerStop`.

Measured wall times (`proof_warmed_call_log.json`):

| call | wall-time |
|---|---|
| warmup #1 (compile + first execution) | 79.60 s |
| warmup #2 (cache hit) | 0.062 s |
| warmup #3 (cache hit) | 0.125 s |
| profiled (cache hit, inside nsys window) | **0.126 s** |

The profiled call is 632× faster than the first warmup → JIT cache hit confirmed, no first-graph constant staging inside the profile window.

nsys invocation (cores 0-3, `OMP_NUM_THREADS=4`):

```bash
taskset -c 0-3 nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --output=.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/proof_warmed \
  env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
      OMP_NUM_THREADS=4 GPUWRF_CUDA_PROFILER_RANGE=1 \
      GPUWRF_D2H_SPRINT_DIR=.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2 \
  python scripts/m6b_d2h_warmed_recapture.py
```

---

## Part 2 — Trace summary (v2)

`proof_nsys_transfers_inside_loop.json` (canonical, v2 worktree, 5-step capture):

| metric | v2 warmed (5 steps, post-lift) | prior warmed (5 steps, pre-lift) | prior unwarmed M6b (1 step) |
|---|---|---|---|
| **total D2H** | **25** | 45 | 53 |
| **pre-kernel D2H** | **25** | 25 | 50 |
| **inter-kernel D2H** | **0** | 20 | 3 |
| post-kernel D2H | 0 | 0 | 0 |
| H2D | **0** | 0 | 0 |
| total D2D | 1 840 | 1 900 | 544 |
| total kernels | **3 675** | 3 770 | 730 |

The inter-kernel D2H count dropped from 20 → **0** between the prior sprint (pre-lift) and this re-capture (post-lift). All 25 D2Hs that remain are pre-kernel, i.e. XLA executable-boundary bookkeeping emitted before the first compute kernel launches — explicitly excluded from the constitutional invariant per ADR-027.

### D2H byte-cluster decomposition (v2)

All clusters are pre-kernel only (since `d2h_inter_kernel == 0`):

| bytes | count | likely emitter |
|---|---|---|
| 352 | 7 | XLA per-call argument-staging (small scalar table) |
| 83 952 | 5 | mass-surface field staging (matches `66*159*8`, post-lift `mu`/`p`-surface alias) |
| 8 | 4 | scalar ack (post-launch fusion-tail copy) |
| 85 224 | 3 | boundary face (matches `66*161*8` for west/east on `nx=159` staggered) |
| 84 480 | 3 | boundary face (matches `66*160*8` for north/south staggered v-edge) |
| 360 | 3 | small argument-staging scalar table |

`inter_kernel_d2h_clusters_by_prev_kernel` is `[]` — no inter-kernel D2H, no per-kernel attribution needed.

Cross-checked via `nsys stats --report cuda_gpu_mem_size_sum --format csv`: 25 D2Hs totalling 0.932 MB, 1 840 D2Ds totalling 165.8 MB. `cuMemcpyDtoHAsync_v2` called 25 times (1.2 % of API time). `cuMemcpyDtoDAsync_v2` called 1 840 times (14.4 % of API time). No `cuLaunchHostFunc` correlations to D2H ranges interleaved with kernels (verified by the SQL bucket split).

---

## Part 3 — Comparison to prior (pre-lift) finding

The prior warmed re-capture (`tester/opus/m6b-d2h-warmed-recapture`) recorded `d2h_inter_kernel = 20` over 5 steps and the inside-loop-fix bisection (`tester/codex/m6b-d2h-inside-loop-fix`) attributed them to two call sites in `operational_mode.py`:

| call site (pre-lift) | per-step | total over 5 steps |
|---|---|---|
| `lax.switch` over RK stage index (line 353-361) | 3 × 4 B | 15 |
| Radiation cadence `lax.cond` predicate (line 374-380) | 1 × 1 B | 5 |
| **subtotal** |  | **20** |

Reading the current post-reframe `operational_mode.py`:

- Lines 363-367: RK stages are **statically sequenced** — three direct calls to `advance_stage(carry, factor, acoustic_substeps)` with constant factors `1/3`, `1/2`, `1`. No `lax.switch`. The "Legacy test anchor for the prior dynamic form" comment documents the previous code.
- Lines 534-567: Radiation cadence is **segmented outside** the `jax.lax.scan` body in the Python-time `while step <= steps` loop. The scan body (`_scan_forecast_segment`, line 499-514) takes a static-Python `run_radiation: bool` and threads it as a captured constant — no per-step device-resident predicate.

Both lifts identified by the bisection sprint are in place in the post-reframe code. The v2 warmed re-capture independently confirms the warmed inter-kernel D2H drops from 20 → 0 as a result.

This matches the recording the prior `m6b-rk1-d2h-acceptance` worker produced (`proof_d2h_warmed_inter_kernel_zero.json`: 25 / 0 / 0 / 0). The v2 re-capture is an independent reproduction on the post-reframe worktree.

---

## Part 4 — Verdict

**GO-D2H-CLEAR.**

| gate | post-reframe finding | status |
|---|---|---|
| pre-kernel D2H ≤ 100 (ADR-027 perf threshold) | 25 | PASS (below absolute threshold) |
| **inter-kernel D2H == 0** (ADR-027 constitutional invariant) | **0** | **PASS** |
| H2D == 0 inside warmed window | 0 | PASS |
| operational sources untouched | yes (asserted by `test_v2_warmed_recapture_does_not_touch_operational_sources`) | PASS |
| `pytest --collect-only` (no regression) | 679 tests collected | PASS |

**Recommendation to manager:** the constitutional timestep-loop transfer invariant is satisfied for the operational acceptance path. ADR-027 was already at PROPOSED (promoted by the prior RK1+D2H acceptance worker) and this v2 re-capture provides **independent corroboration** on the post-reframe worktree. No fix sprint is needed for D2H. The remaining 25 pre-kernel D2Hs are within the ADR-027 performance threshold of ≤ 100 per warmed 5-step capture and are explicitly excluded from the constitutional invariant.

Operational acceptance for M6b remains gated on the **non-D2H** blockers tracked by other sprints (RK1 parity, theta bounds, Tier-4 RMSE) — D2H is not in their critical path.

---

## Part 5 — Proof objects

Under `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/`:

| file | what it records |
|---|---|
| `proof_warmed.nsys-rep` | warmed 5-step Nsight trace (canonical) |
| `proof_warmed.sqlite` | auto-exported SQLite mirror |
| `proof_warmed_call_log.json` | Python-side timing + protocol metadata |
| `proof_warmed_run.txt` | stdout of the orchestrator under nsys |
| `proof_warmed_trace_summary.txt` | `nsys stats --report cuda_gpu_mem_size_sum,cuda_api_sum --format csv` output |
| `proof_nsys_transfers_inside_loop.json` | parsed per-bucket D2H summary (pre/inter/post kernel) — the canonical machine-readable verdict |
| `proof_no_touch.txt` | `pytest --collect-only` tail (679 tests collected) |
| `d2h_warmed_memo_v2.md` | this memo |

Tests added:

| file | what it asserts |
|---|---|
| `tests/test_m6b_d2h_warmed_zero_v2.py` | 5 assertions: artifacts exist, 3-warmup protocol applied + cache-hit confirmed, H2D=0, **d2h_inter_kernel=0** (ADR-027 invariant), operational sources untouched. All 5 pass. |

Source files NOT modified:

- `src/gpuwrf/runtime/operational_mode.py` — read-only (verified by test). M6b V3 codex worker has the write-lock on this file in `/tmp/wrf_gpu2_m6b_v3`.
- `src/gpuwrf/runtime/validation_wrappers.py` — read-only.
- `src/gpuwrf/dynamics/core/` — read-only.
- Operational `wrf.exe` — read-only.

Script edits:

- `scripts/m6b_d2h_warmed_recapture.py`: added `GPUWRF_D2H_SPRINT_DIR` env-var override at the SPRINT module constant so the warmed re-capture's call-log and canonical summary JSON can be routed into the v2 sprint folder without overwriting the prior sprint's proofs. No semantic changes to the protocol itself (still 3 warmups + 1 profiled call, identical signature).

---

## AGENT REPORT

**Objective**: Re-run the warmed Nsight capture on the post-reframe `operational_mode.py` to determine whether the constitutional invariant (`d2h_inter_kernel == 0` per ADR-027) is satisfied, given that the standalone bisect honestly noted "D2H warmed zero was not newly proven" in the v2 worktree; the existing warmed summary still recorded nonzero inter-kernel D2H.

**Files changed** (worktree `/tmp/wrf_gpu2_d2h_v2`, branch `tester/opus/m6b-d2h-warmed-recapture-v2`):

- `scripts/m6b_d2h_warmed_recapture.py` — single-line addition of a `GPUWRF_D2H_SPRINT_DIR` env-var override so v2 proofs land in the v2 sprint dir (protocol unchanged).
- `tests/test_m6b_d2h_warmed_zero_v2.py` — NEW (5 assertions, all pass).
- `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/` — 8 proof artefacts + this memo.

NO operational sources modified. NO `wrf.exe` touched. NO files in `/tmp/wrf_gpu2_m6b_v3` (M6b V3 codex worker's worktree) touched.

**Commands run** (cores 0-3, `OMP_NUM_THREADS=4`):

- `taskset -c 0-3 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --output=.agent/sprints/.../proof_warmed env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 GPUWRF_CUDA_PROFILER_RANGE=1 GPUWRF_D2H_SPRINT_DIR=.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2 python scripts/m6b_d2h_warmed_recapture.py` — 5-step warmed capture, exit 0.
- `taskset -c 0-3 env PYTHONPATH=src GPUWRF_D2H_SPRINT_DIR=.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2 python scripts/m6b_d2h_warmed_recapture.py --parse-rep .agent/sprints/2026-05-25-m6b-d2h-warmed-recapture-v2/proof_warmed.nsys-rep` — canonical SQLite parse.
- `nsys stats --report cuda_gpu_mem_size_sum --report cuda_api_sum --format csv .../proof_warmed.nsys-rep` — human-readable summary.
- `taskset -c 0-3 env PYTHONPATH=src OMP_NUM_THREADS=4 pytest tests/test_m6b_d2h_warmed_zero_v2.py -v` → 5 passed.
- `taskset -c 0-3 env PYTHONPATH=src OMP_NUM_THREADS=4 pytest --collect-only` → 679 tests collected.

**Proof objects**: 8 files listed in Part 5.

**Key findings**:

1. **Warmed inter-kernel D2H = 0** on the post-reframe `operational_mode.py`. The two pre-lift emitters (RK `lax.switch` at line 353-361 and radiation `lax.cond` at line 374-380) are no longer present — RK is statically sequenced and radiation cadence is segmented in Python-time outside the scan body.
2. **25 pre-kernel D2Hs remain** (XLA executable-boundary bookkeeping; same byte clusters as the prior RK1+D2H acceptance capture: 352 B × 7, 83 952 B × 5, 8 B × 4, 85 224 B × 3, 84 480 B × 3, 360 B × 3). These are within the ADR-027 performance threshold of ≤ 100 and are explicitly excluded from the constitutional invariant.
3. JIT cache-hit confirmed: profiled call wall-time 0.126 s vs 79.6 s for the cold first warmup (632× speedup).
4. The v2 re-capture independently reproduces the prior `m6b-rk1-d2h-acceptance` result (`proof_d2h_warmed_inter_kernel_zero.json`) on the current post-reframe worktree, closing the standalone bisect's honest gap ("D2H warmed zero was not newly proven").
5. ADR-027 was already at PROPOSED (promoted by the prior acceptance worker). No status change is needed; the v2 capture provides additional corroborating evidence on a different worktree.

**Unresolved risks**:

- None for D2H. The 25 pre-kernel D2Hs are documented as expected XLA executable-boundary bookkeeping; a future ADR-026 follow-up could reduce them via persistent device-resident carries, but that is a performance optimisation, not a constitutional concern.
- Non-D2H M6b blockers (RK1 parity, theta bounds, Tier-4 RMSE) are tracked by other sprints and are not in this sprint's scope.

**Next decision needed**: manager folds GO-D2H-CLEAR into the M6b operational acceptance roll-up. No D2H fix sprint is required. M6b V3 codex worker continues independently on `/tmp/wrf_gpu2_m6b_v3` for the non-D2H acceptance gates.

**Time used**: ~45 minutes (within the 30-60 min budget).

**Branch**: `tester/opus/m6b-d2h-warmed-recapture-v2` in worktree `/tmp/wrf_gpu2_d2h_v2`. NO remote push.
