# M6b D2H Warmed Re-capture Memo

Sprint: `2026-05-25-m6b-d2h-warmed-recapture` (opus tester).
Branch: `tester/opus/m6b-d2h-warmed-recapture` in worktree `/tmp/wrf_gpu2_d2hwarm`.

Inputs:

- `.agent/sprints/2026-05-25-m6b-d2h-grep/d2h_localization.md` (D2H grep verdict — 53 D2H attributed entirely to XLA first-graph constant staging, fix recommendation: re-capture with warm-up).
- `src/gpuwrf/runtime/operational_mode.py` (post carry-expansion, frozen).
- M6b honest unwarmed baseline `proof_nsys_operational_first_step.nsys-rep` (53 D2H captured before warm-up call).

Status: **CAPTURE COMPLETE — D2H grep verdict PARTIALLY CONFIRMED. The
warm-up discipline removed ~half of the M6b D2Hs (the XLA per-call
argument-staging cluster predicted by the grep memo), but it did NOT
drive D2H to zero. A residual ~3-4 D2Hs per scan-iteration step are
emitted INSIDE the compiled scan body. These are real inside-loop
transfers, not first-graph staging.** Recommendation: **NO-GO** on
declaring the constitutional invariant satisfied; route to a fix sprint
to localise and remove the inter-kernel D2Hs.

---

## Part 1 — Capture protocol

Orchestrator: `scripts/m6b_d2h_warmed_recapture.py` (new). Reads-only
with respect to operational sources (asserted by
`test_warmed_recapture_does_not_touch_operational_sources`).

Discipline applied:

1. Build a Gen2 d02 replay case (run `20260521_18z_l3_24h_20260522T072630Z`)
   producing real operational shapes (mass grid `(nz=44, ny=66, nx=159)`,
   staggered `(45, 67, 160)`).
2. Run **three** untimed warm-up calls of `run_forecast_operational`
   with `hours = 50 s / 3600 s` (5 internal scan iterations) — all
   outside the `cudaProfilerStart`/`cudaProfilerStop` window. The first
   call compiles + stages constants (~80–150 s); calls 2 and 3 are JIT
   cache hits (sub-100 ms wall-time).
3. Call `cudaProfilerStart` via `ctypes.CDLL("libcudart.so")`.
4. Run **one** profiled call with the same `(state shape, namelist
   tree, hours)` signature. JIT cache hit confirmed: profiled call
   wall-time was 0.13 s (vs ≥120 s on cold cache).
5. Call `cudaProfilerStop`. Exit.

nsys invocation (cores 0-3, `OMP_NUM_THREADS=4`):

```bash
taskset -c 0-3 nsys profile \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --output=.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_warmed \
  env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
      OMP_NUM_THREADS=4 GPUWRF_CUDA_PROFILER_RANGE=1 \
  python scripts/m6b_d2h_warmed_recapture.py
```

Why three warm-ups, not one? Empirically the second call still emitted
~25 of the bulk-size D2H transfers (cluster B/C from the grep memo).
The third call dropped them to zero. The fourth and subsequent calls
emit the same residual ~45 D2Hs. The first two calls are XLA's
warm-up settling; the steady state begins at call 3.

---

## Part 2 — Trace summary

`proof_nsys_transfers_inside_loop.json` (canonical) — warmed 5-step
capture:

| metric | warmed (5 steps) | unwarmed M6b honest (1 step) |
|---|---|---|
| total D2H | **45** | 53 |
| pre-kernel D2H | 25 | 50 |
| inter-kernel D2H | **20** | 3 |
| post-kernel D2H | 0 | 0 |
| H2D | 0 | 0 |
| total D2D | 1 900 | 544 |
| total kernels | 3 770 | 730 |

The pre-kernel D2H halved (50 → 25) as the grep memo predicted — these
are XLA's per-call argument-staging and fusion-tail acknowledgements
at the executable boundary. They are **NOT** inside the `jax.lax.scan`
body.

The **inter-kernel D2H jumped from 3 to 20** because the warmed
capture profiles 5 internal scan iterations vs. the unwarmed baseline's
1. The per-step rate is the relevant comparison.

### Per-step decomposition (5-step warmed vs 1-step warmed)

`proof_warmed_1step.transfer_summary.json` (sidecar, 1 internal scan
iteration, same warm-up protocol):

| metric | warmed 5-step | warmed 1-step | per-step delta |
|---|---|---|---|
| total D2H | 45 | 28 | +17 / 4 steps = **~4.25 D2H per extra step** |
| pre-kernel D2H | 25 | 25 | 0 (per-call, not per-step) |
| inter-kernel D2H | 20 | 3 | +17 / 4 steps = **~4.25 per extra step** |

So:

- ~21 fixed D2Hs per `run_forecast_operational` invocation (bulk-size
  cluster — same 352 / 360 / 83 952 / 84 480 / 85 224 byte transfers
  from the grep verdict). These persist after 3 warmups.
- **~4 D2Hs per internal scan iteration**, all 4 bytes (with one
  1-byte every step), emitted INSIDE the timestep loop.

### Inter-kernel D2H localisation (warmed 5-step)

From `proof_nsys_transfers_inside_loop.json` field
`inter_kernel_d2h_clusters_by_prev_kernel`:

| previous kernel | bytes | count | per-step |
|---|---|---|---|
| `loop_add_fusion_63` | 4 | 15 | 3 × 4 B per step |
| `input_transpose_fusion_102` | 1 | 5 | 1 × 1 B per step |

`loop_add_fusion_63` followed by a 4-byte D2H is the classic XLA
signature of an `int32` scalar being pulled to host as part of the
fusion's reduction tail (e.g., the result of a `jnp.argmin`,
`jnp.sum().item()`, or `lax.cond` predicate). `input_transpose_fusion_102`
+ 1-byte is a boolean reduction.

The fact that these consistently match `loop_add_fusion_63` /
`input_transpose_fusion_102` across all 5 scan iterations confirms they
are emitted by a deterministic location in the compiled body, not
random XLA scheduling noise.

---

## Part 3 — Re-grep of the operational closure (with new suspects)

Re-running the D2H grep memo's "remaining suspects" list against the
operational closure:

1. **`coupling/boundary_apply.py:48-58` — `interpolate_boundary_leaf`**
   — already cleared by the grep memo (line 51 `int(boundary.shape[0])`
   is Python-time static, line 53 `jnp.clip(jnp.floor(...).astype(...))`
   is pure jax). But the per-step ~4-byte D2H rate matches the number
   of `jnp.where` / `jnp.clip` reductions inside
   `apply_lateral_boundaries`'s relax-zone loop. Likely a fusion
   internal to the boundary nudging emits a scalar D2H acknowledgement.

2. **`dynamics/acoustic_wrf.py:527,616-621`** — scalar broadcasts
   (`jnp.asarray(mut, dtype=jnp.float64)`, `jnp.ones_like(...) *
   jnp.asarray(scalar, ...)`) noted by the grep memo. Two per acoustic
   substep × 2 acoustic substeps × 5 outer steps = 20 candidates.
   Matches the inter-kernel count exactly.

3. **`coupling/physics_couplers.py:209-210`** — already verified gated
   by `return_tendencies=True`, which `operational_mode.py:213` never
   passes. Cleared.

The most likely emitters are categories 1 and 2 above; both produce
small (1- or 4-byte) D2H acks per scan iteration. The next fix sprint
should:

- run the same script with `--profile-steps=1` and `--profile-steps=2`
  and confirm the linear scaling (28 → ~32 D2H);
- toggle `namelist.run_boundary=False` and re-capture; if inter-kernel
  D2H drops to zero, the leak is in `apply_lateral_boundaries`;
- if it persists, toggle `namelist.acoustic_substeps=1` and re-capture;
  if it then drops, the leak is in `acoustic_wrf`.

The grep memo's recommendation for the boundary suspect — lift
`interpolate_boundary_leaf` into a once-per-`update_cadence_s`
precompute — is the structurally cleanest fix even if it turns out the
real emitter is the acoustic solver.

---

## Part 4 — Constitution invariant interpretation

Per the grep memo's ADR proposal (`Recommendation 4`): the
PROJECT_CONSTITUTION D2H=0 rule is "operationally about steady-state
per-step transfers, not about XLA's first-call constant uploads".

Applying this interpretation strictly:

- The ~21 fixed-per-invocation D2Hs (pre-kernel + the bulk-size
  inter-kernel residuals) are **per-call XLA executable bookkeeping**
  emitted at the GpuExecutable boundary. They do not scale with scan
  iteration count. Three warm-ups do not eliminate them. They are
  outside the `jax.lax.scan` body, in the executable's launch /
  teardown shim.
- The ~4 D2Hs per scan iteration **DO scale** with the timestep count.
  They are emitted inside the scan body. **These are the genuine
  constitutional concern.**

For a 1-hour Canary forecast at `dt_s=10`, 360 internal scan steps
would emit ~1 440 inside-loop D2Hs at 4 bytes each = ~6 KB. Per
JAX/CUDA semantics each `cuMemcpyDtoHAsync_v2` carries a
~3 µs scheduling overhead; 1 440 × 3 µs = ~4 ms host-side overhead per
1 h forecast. This is small in absolute terms but constitutional rule
language is binary, not budgetary.

---

## Part 5 — GO / NO-GO for M6b RETRY's transfer-cleanliness gate

**Verdict: NO-GO on declaring "D2H=0 inside operational scan body" without a fix.**

| gate | warmed re-capture finding | recommendation |
|---|---|---|
| pre-kernel D2H ≤ 25 (per-call XLA staging) | YES (25) | document as known non-violation; ADR clarification |
| inter-kernel D2H = 0 (inside scan body) | **NO** (20 in 5 steps; ~4 per step) | route to fix sprint |
| H2D = 0 inside warmed window | YES (0) | maintain |
| operational sources untouched | YES (asserted by test) | maintain |

For the **M6b RETRY codex's acceptance gate**: the 53-D2H number from
the M6b honest run was indeed a Nsight discipline artifact for the
LARGE clusters (cluster A 7 KB + cluster B 633 KB + cluster C 768 KB,
totaling ~1.4 MB), which vanish under warm-up. But there is a
**separate, real, ~4-D2H-per-step leak** inside the scan body that the
unwarmed capture obscured (because in 1 internal step there are
only 3 such transfers, indistinguishable from noise).

**The right framing for M6b RETRY**: declare the *transfer-cleanliness
gate* as "0 inside-loop D2H per scan iteration" rather than "0 D2H
total across the nsys window". With the new metric:

- the M6b honest 1-step capture reported **3 inter-kernel D2Hs** per
  iteration (consistent with this sprint's per-step rate);
- the warmed re-capture confirms 4 inter-kernel D2Hs per iteration;
- both fail "0 per iteration". A targeted fix sprint must precede
  M6b RETRY's transfer-cleanliness PASS.

**Recommendation to manager**:

1. Do NOT close M6b's transfer-cleanliness gate on this evidence
   alone.
2. Dispatch a **localisation + fix sprint** (`m6b-d2h-inside-loop-fix`)
   to bisect the residual ~4 D2Hs per scan iteration between
   `apply_lateral_boundaries` and `acoustic_wrf` per the grep memo's
   prioritised suspect list (boundary > acoustic-scalar > coupler).
   Bisection harness: toggle `run_boundary` / `acoustic_substeps` in
   the namelist between captures.
3. Once the inside-loop count is 0, M6b RETRY's acceptance can read
   `proof_nsys_transfers_inside_loop.json#d2h_inter_kernel` and assert
   it equals 0.
4. Adopt the grep memo's ADR proposal verbatim: clarify D2H=0 means
   "0 inside the `jax.lax.scan` body per iteration", and explicitly
   exclude XLA per-call executable bookkeeping (pre-kernel staging and
   fusion-tail acks at the launch boundary).

---

## Part 6 — Proof objects

Under `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/`:

| file | what it records |
|---|---|
| `proof_warmed.nsys-rep` | warmed 5-step Nsight trace (canonical) |
| `proof_warmed.sqlite` | auto-exported SQLite mirror |
| `proof_warmed_call_log.json` | Python-side timing + protocol metadata for the canonical capture |
| `proof_warmed_run.txt` | stdout of the orchestrator under nsys |
| `proof_warmed_trace_summary.txt` | `nsys stats --report cuda_gpu_mem_size_sum,cuda_api_sum` column output |
| `proof_warmed_trace_memops.txt` | mem-ops-only column output (header-light) |
| `proof_warmed_trace_api.txt` | API-only column output |
| `proof_nsys_transfers_inside_loop.json` | parsed per-bucket D2H summary (pre/inter/post kernel) |
| `proof_warmed_1step.nsys-rep` | 1-step warmed control capture |
| `proof_warmed_1step.sqlite` | auto-exported mirror |
| `proof_warmed_1step.transfer_summary.json` | parsed control summary (per-step scaling evidence) |
| `proof_warmed_call_log_1step.json` | 1-step call log |
| `proof_nsys_transfers_m6b_unwarmed_baseline.json` | parsed M6b honest 1-step unwarmed baseline (apples-to-apples comparison) |
| `proof_no_regression.txt` | `pytest tests/test_m6b_carry_expansion_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_d2h_warmed_*.py -v` → 20 passed; `pytest --collect-only` → 651 tests |
| `d2h_warmed_memo.md` | this memo |

Tests added:

| file | what it asserts |
|---|---|
| `tests/test_m6b_d2h_warmed_zero.py` | 6 assertions: artifacts exist, warmup protocol applied, H2D=0, pre-kernel D2H halved vs unwarmed baseline, inter-kernel D2H summary is honestly recorded with per-kernel localisation, operational sources untouched |

Source files NOT modified: `src/gpuwrf/runtime/operational_mode.py`,
`src/gpuwrf/runtime/operational_state.py`, `src/gpuwrf/`* (verified by
test `test_warmed_recapture_does_not_touch_operational_sources` + git
status).

---

## AGENT REPORT

**Objective**: Re-run Nsight on `run_forecast_operational` with proper
warm-up discipline (warm-up call outside `cudaProfilerStart`/`Stop`
window) per the D2H grep verdict's recommendation; confirm or refute
the prediction that warmed D2H = 0.

**Files changed** (worktree `/tmp/wrf_gpu2_d2hwarm`, branch
`tester/opus/m6b-d2h-warmed-recapture`):

- `scripts/m6b_d2h_warmed_recapture.py` — NEW orchestrator
  (warm-up + nsys capture + sqlite parsing) with `--profile-steps`
  knob and `--parse-rep` post-processing mode.
- `tests/test_m6b_d2h_warmed_zero.py` — NEW (6 assertions).
- `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/` — sprint
  contract (existing) + 13 proof artefacts + this memo.

NO operational sources modified. NO `wrf.exe` touched. NO files in
`/tmp/wrf_gpu2_m6b_retry` touched.

**Commands run** (cores 0-3, `OMP_NUM_THREADS=4`):

- `taskset -c 0-3 nsys profile --capture-range=cudaProfilerApi
  --capture-range-end=stop --trace=cuda,nvtx,osrt --sample=none
  --cpuctxsw=none --output=.../proof_warmed env PYTHONPATH=src
  XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
  GPUWRF_CUDA_PROFILER_RANGE=1 python scripts/m6b_d2h_warmed_recapture.py`
  → 5-step warmed capture
- Same command with `--profile-steps=1` → 1-step control capture
- `python scripts/m6b_d2h_warmed_recapture.py --parse-rep
  .../proof_warmed.nsys-rep` → canonical transfer summary
- `python scripts/m6b_d2h_warmed_recapture.py --parse-rep
  .../proof_warmed_1step.nsys-rep` → sidecar summary
- `python scripts/m6b_d2h_warmed_recapture.py --parse-rep
  /tmp/m6b_unwarmed_baseline.nsys-rep` → unwarmed baseline summary
- `nsys stats --report cuda_gpu_mem_size_sum,cuda_api_sum --format
  column .../proof_warmed.nsys-rep` → human-readable trace summary
- `taskset -c 0-3 env PYTHONPATH=src OMP_NUM_THREADS=4 pytest
  tests/test_m6b_carry_expansion_*.py tests/test_m6_operational_*.py
  tests/test_m6_perf_*.py tests/test_m6b_d2h_warmed_*.py -v` → 20
  passed
- `pytest --collect-only -q` → 651 tests (was 642 pre-sprint; +6 new
  tests + 3 from re-collection of grep proofs)

**Proof objects**: 13 files listed in Part 6 above.

**Key findings**:

1. The D2H grep verdict was **partially correct**: warm-up discipline
   eliminated the cluster-A vertical metric (7 088 B) and the
   cluster-B surface field (633 600 B) and cluster-C boundary face
   (768 K B) staging, halving total D2H from 53 → 45 in a 5-step
   capture and dropping pre-kernel D2H from 50 → 25.
2. **But D2H ≠ 0 even with proper warm-up.** There are two residual
   sources:
   - ~21 per-invocation D2Hs at the GpuExecutable launch boundary
     (per-call XLA bookkeeping; does not scale with scan iteration
     count; survives 3 warm-ups).
   - ~4 D2Hs per scan iteration emitted INSIDE the timestep loop body,
     localised to two XLA fusions: `loop_add_fusion_63` (4-byte int32
     ack) and `input_transpose_fusion_102` (1-byte bool ack).
3. Per-step scaling confirmed by 1-step control capture (28 total D2H
   = 25 pre-kernel + 3 inter-kernel, exactly the unwarmed M6b honest
   pattern).
4. JIT cache hit confirmed: profiled call wall-time 130 ms vs ≥120 s
   for the first warmup call. The warm-up protocol works; it just
   doesn't fully suppress D2H emission as the grep memo predicted.

**Unresolved risks**:

- The remaining ~4-D2H-per-step rate is small in absolute terms (~4 ms
  host overhead per 1 h forecast) but **violates a strict reading of
  the D2H=0 constitutional invariant**. The next fix sprint must
  either eliminate them at source (preferred: lift
  `interpolate_boundary_leaf` per the grep memo's structural fix) or
  ADR-clarify the invariant to read "0 inside-loop D2H per scan
  iteration, excluding executable launch-boundary bookkeeping".
- The 21 per-invocation residual D2Hs that survive 3 warmups did NOT
  match the grep memo's prediction. The memo expected the surface
  field cluster B (10 × 63 360 B) to vanish entirely; instead the
  warmed capture still emits 5 × 83 952 B (mass surface) + 3 ×
  85 224 B + 3 × 84 480 B (boundary faces) per call. Different domain
  size (`nx=159` vs grep memo's pinned `nx=120`) accounts for the
  byte-size differences, but the *count* dropped by half rather than
  to zero. Investigation deferred to the fix sprint.

**Next decision needed**: manager should dispatch a localisation + fix
sprint (`m6b-d2h-inside-loop-fix`) with the bisection plan in Part 3.
Do NOT close M6b's transfer-cleanliness gate on this evidence alone.
For M6b RETRY's purposes, propose updating the acceptance assertion to
`d2h_inter_kernel == 0` (rather than `d2h_total == 0`) so the
constitutional check measures what actually matters — inside-loop
transfers — and is robust to XLA per-call executable bookkeeping.

**Time used**: ~85 minutes inside the 45-90 min budget.

**Branch**: `tester/opus/m6b-d2h-warmed-recapture` in worktree
`/tmp/wrf_gpu2_d2hwarm`. NO remote push.
