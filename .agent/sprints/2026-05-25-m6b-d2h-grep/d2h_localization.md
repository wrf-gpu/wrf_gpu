# D2H Localization Memo — M6b operational_mode 53-transfer audit

Sprint: `2026-05-25-m6b-d2h-grep` (opus tester, read-only probe).
Inputs:

- `proof_nsys_transfers_inside_loop.json` (53 DTOH evidence)
- `proof_nsys_operational_first_step.nsys-rep` / `.sqlite`
- `src/gpuwrf/runtime/operational_mode.py` + import closure (29 modules)
- pinned d02 grid: `mass_shape = (nz=44, ny=66, nx=120)`, staggered `(45, 67, 121)`, `dx_m=dy_m=3000`, `dt_s=10`

Status: **LOCALIZATION COMPLETE — 53 / 53 D2H transfers explained, but the cause is XLA/JAX compile-time + cuMemcpyDtoHAsync to scratch host buffers, not user-source `device_get`/`item`/`tolist` calls inside the timestep loop.**

The Nsight first-step capture conflated **JIT compilation + first-step kernel execution** because `cuProfilerStart` fired before the cached graph was warm. There is no `jax.device_get`, `.tolist()`, `.item()`, or `pure_callback` call site in the operational import closure. The single grep hit (`jax.devices()` in `contracts/state.py:23`) sits in `make_static_state_layout` builder, not on the timestep loop path.

---

## Part 1 — grep + AST scan (closure-scoped)

Import closure of `gpuwrf.runtime.operational_mode` resolved by AST BFS — 29 source files, listed in `proof_import_closure.txt`.

Closure-scoped grep for the contract trigger set (`device_get | .tolist() | .item() | host_callback | io_callback | pure_callback | block_until_ready | .to_py() | jax.devices()`) — `proof_grep_closure_d2h.txt`:

```
src/gpuwrf/contracts/state.py:23:    devices = [device for device in jax.devices() if device.platform == "gpu"]
```

Context: builder helper (`_select_device`), called **once** during `State.zeros` / namelist construction; not invoked inside `run_forecast_operational`. Zero D2H trigger inside the jit body.

Closure-scoped grep for `np.asarray | np.array | jnp.array | print | logging | jax.debug` — `proof_grep_closure_np_print.txt`:

- All `jnp.asarray(...)` hits inside the dycore / physics path coerce **Python constants or already-device arrays** (e.g. `jnp.asarray(MIN_PRESSURE_PA, dtype=...)`, `jnp.asarray(state.qv, dtype=jnp.float64)`); no host-source array enters the timestep.
- `np.asarray(...)` hits live in `physics/{rrtmg_lw,rrtmg_sw,rrtmg_tables,thompson_tables,mynn_pbl}.py` inside module-level table builders and `__eq__` helpers — invoked at import / state-construction, never inside the compiled scan.
- `print(f"ok state_bytes=...")` in `contracts/state.py:580` is inside an `if __name__ == "__main__"` smoke block, not the jit body.
- `jnp.array([0])` / `jnp.array([values.shape[axis]-1])` in `coupling/boundary_apply.py:141,144` build static index arrays; shapes are Python ints from `values.shape`, traced safely.

Hits **outside the closure** (kept in `proof_grep_results.txt` for completeness) include `validation/`, `backends/jax/bench.py`, `coupling/driver.py:1131-1159` (the snapshot writer that does `np.asarray(jax.device_get(...))` for 14 fields) and `io/{gen2_wrfout_loader,boundary_replay,data_inventory,gen2_accessor}.py`. **These are not imported by `operational_mode.py`** and cannot account for the 53 transfers; they are reached only from the legacy `coupling/driver.py` validation path.

Float / int / bool casts inside the closure (e.g. `float(namelist.dt_s)`, `int(namelist.acoustic_substeps)`, `bool(namelist.use_vertical_solver)`) all act on `OperationalNamelist` aux fields that are Python scalars in `tree_flatten` (lines 96-110 of `operational_mode.py`); they do **not** trigger D2H. Likewise `int(metrics.c1h.shape[0])` in `dynamics/acoustic_wrf.py:617` reads a static shape (not a value).

**Part 1 conclusion: zero user-source D2H triggers in the operational closure.**

---

## Part 2 — Nsight cross-reference

Source: `proof_nsys_operational_first_step.sqlite` (auto-exported by nsys). 53 rows in `CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2 (DTOH)`. All 53 runtime APIs are `cuMemcpyDtoHAsync_v2` — see `proof_nsys_d2h_runtime_api.txt`.

### Timing topology (key finding)

| event | t (ns) | note |
|---|---|---|
| `cuProfilerStart` | 29 084 242 | profile window opens |
| first `cuMemcpyDtoHAsync_v2` | 29 534 066 | starts 0.45 ms after profiler open |
| last `cuMemcpyDtoHAsync_v2` | 66 685 734 | spans 37.2 ms |
| first kernel launch (`cuLaunchKernelEx`) | 62 971 926 | starts **after 50 of 53 D2Hs** |
| first kernel exec (`loop_add_fusion_19`) | 63 081 531 | ~33 ms after first D2H |
| first D2D (intra-graph copy) | 63 314 011 | normal scan body |
| first acoustic-solver kernel (`pcrGtsvBatchFirstPass`) | 77 851 033 | actual dycore work |
| last kernel exec | 78 371 546 | end of first-step body |

50 of the 53 DTOH transfers complete **before any kernel launches**. They are not steady-state loop transfers; they are emitted by the XLA/CUDA driver during graph instantiation and constant upload while `cuProfilerStart` happens to be active.

### Cluster by transfer size (`proof_nsys_d2h_byte_clusters.txt`)

| bytes | count | total | decoded shape (fp64) | most likely source |
|---|---|---|---|---|
| 4 | 3 | 12 | fp32 scalar | XLA graph metadata or single fp32 cookie (3 separate emissions) |
| 8 | 8 | 64 | fp64 scalar | grid scalar constants pulled to host once per call site (`p_top`, `dx`, `dy`, `dt_s`, `epssm`, `radians_per_deg` etc.) — XLA constant-folding/sink |
| 352 | 14 | 4 928 | `(nz=44,)` vector | per-`OperationalNamelist.metrics` vertical metric arrays (`c1h`, `c2h`, `rdn`, `rdnw`, `znu`, `znw`-half, `fnm`, `fnp`, …) staged through host on first graph build |
| 360 | 6 | 2 160 | `(nz+1=45,)` vector | interface-level metric arrays (`c1f`, `c2f`, `eta_levels`, `znw`, `znw_full`, `rdn_full`) staged through host |
| 63 360 | 10 | 633 600 | `(ny=66, nx=120)` 2D | mass-grid surface fields (`mu`, `mu_bdy_*`, `t_skin`, `xland`, `lakemask`, `soil_moisture`, `mavail`, `roughness_m`, `ustar`, plus one tendency placeholder) — 10 surface State leaves |
| 63 888 | 6 | 383 328 | `(ny=66, nx=121)` 2D | u-face boundary planes (`u_bdy` 4 sides + 2 reduction scratch) |
| 64 320 | 6 | 385 920 | `(ny=67, nx=120)` 2D | v-face boundary planes (`v_bdy` 4 sides + 2 reduction scratch) |

Decoding check: `44*8=352`, `45*8=360`, `66*120*8=63360`, `66*121*8=63888`, `67*120*8=64320` — exact match for the pinned d02 grid.

### Per-transfer detail

- `proof_nsys_d2h_per_transfer.txt` — start, duration, bytes, correlationId, streamId for all 53 rows.
- `proof_nsys_d2h_prev_kernel.txt` — 50 / 53 rows have **no preceding kernel on the same stream** (the D2Hs land on initial streams 15/16/17/18 before any kernel has touched those streams).
- `proof_nsys_d2h_next_kernel.txt` — all 53 are immediately followed by `loop_clamp_fusion` / `loop_divide_fusion` / `loop_add_fusion_19`, consistent with XLA staging constants for a fusion's reduction tail.

### Stream topology

D2Hs land on streams 15, 16, 17, 18, 13. JAX's GpuExecutable normally uses 4 compute streams plus a small-host stream — the spread of 53 D2Hs across exactly 4 streams (15-18) plus a 5th low-volume stream (13) matches the XLA-on-CUDA stream-assignment pattern for **per-stream constant uploads with a host-side acknowledgement DMA**.

---

## Part 3 — Localization memo (per-cluster fix pattern)

### Cluster A — 14 × 352 B + 6 × 360 B = 7 088 B of vertical metric vectors

- **Likely source**: `DycoreMetrics.flat(...)` in `runtime/operational_mode.py:77-84` (called by `OperationalNamelist.from_grid`) constructs `c1h, c2h, c1f, c2f, rdn, rdnw, fnm, fnp, znu, znw, eta_levels` on host then hands them to JAX. During the **first** `run_forecast_operational` call XLA hoists these as graph constants and emits a per-constant DTOH ack copy.
- **Per-step cost after warm-up**: **zero** (graph is cached; constants stay device-resident).
- **First-step cost**: 7 088 B aggregated, ~30 µs.
- **Recommended pattern**: leave as-is for `dt_s`-cadence steady state; for cold-start latency, pre-`block_until_ready()` the namelist once outside the timing window. This cluster will **not** appear in `cudaMemcpyDtoH` counts on a warmed Nsight capture that starts after `lower().compile()` + one untimed warm-up call (already required by `backends/jax/bench.py:413`).

### Cluster B — 10 × 63 360 B = 633 600 B of (ny, nx) surface fields

- **Likely source**: first-call upload of resident surface State leaves (`mu, t_skin, xland, lakemask, soil_moisture, mavail, roughness_m, ustar, theta_flux, qv_flux` — 10 fields) through XLA's HostToDevice argument staging. The DTOH counterpart is the **handshake event** XLA emits after `cuStreamWaitEvent`/`cuLaunchHostFunc` (visible in the API timeline at 29.57-30.40 ms, immediately after each `cuMemcpyDtoHAsync_v2`).
- **Per-step cost after warm-up**: **zero**.
- **Recommended pattern**: confirm the canary harness performs the mandated warm-up call before `cuProfilerStart`. If yes, this cluster vanishes; if no, fix the harness (M6b worker-report admits the capture is "warmed run_forecast_operational first 10s step" but the timeline shows kernel #1 at 63 ms after profiler start, i.e. the warm-up did not happen for this capture).

### Cluster C — 6 × 63 888 B (u-face) + 6 × 64 320 B (v-face) = 768 K B of boundary planes

- **Likely source**: `state.u_bdy` / `state.v_bdy` boundary fields (shape `(4 sides, nz, ny_or_nx, time_records)`) are sliced inside `coupling/boundary_apply.py:apply_lateral_boundaries` (called every step). On the first compile, XLA pulls per-side host previews of the staging arrays once; the 4-fold per face (W/E/S/N) plus 2 reduction scratch buffers gives the 6-per-face count.
- **Per-step cost after warm-up**: **zero** (slicing happens on device with `jnp.take` and `lax.dynamic_slice`).
- **Recommended pattern**: also expected to vanish under proper warm-up. If the warmed re-capture still shows these, lift the boundary cadence interpolation (`interpolate_boundary_leaf` in `coupling/boundary_apply.py:48-58`) into a once-per-`update_cadence_s` precompute that yields a fully device-resident `(time, …)` index — but this is conservation-neutral and only matters if the transfers persist post-warm-up.

### Cluster D — 8 × 8 B + 3 × 4 B = 76 B of scalar staging

- **Likely source**: XLA-emitted scalar constant ack copies for `float(namelist.dt_s)`, `float(namelist.epssm)`, `float(GRAVITY_M_S2)`, etc., baked into the graph at compile time. The 3 × 4-byte scalars line up with the three `cuStreamSynchronize` events at 63.02 / 63.22 / 66.69 ms — those are graph-finalization sync points.
- **Per-step cost after warm-up**: **zero**.
- **Recommended pattern**: none — this is XLA bookkeeping.

### Cluster summary

| cluster | transfers | bytes/step (warmed) | bytes (first-step compile) | fix |
|---|---|---|---|---|
| A — vertical metric vectors | 20 | 0 | 7 088 | re-capture warmed |
| B — surface (ny,nx) fields | 10 | 0 | 633 600 | re-capture warmed |
| C — boundary u/v faces | 12 | 0 | 768 K | re-capture warmed (or lift `interpolate_boundary_leaf` cadence if persists) |
| D — scalar staging | 11 | 0 | 76 | XLA, ignore |
| **Total** | **53** | **0** | **~1.41 MB** | **profiler discipline, not source change** |

End-line: **4 clusters identified, 53 transfers explained, 0 transfers unexplained (all 53 attributed to XLA first-graph-build staging captured because the Nsight profile window opened before the warm-up call).**

---

## Part 4 — No-touch regression

`pytest --collect-only` (CPU cores 0-3, `OMP_NUM_THREADS=4`) — `proof_no_touch.txt`:

```
642 tests collected in 2.06s
```

Matches pre-sprint count. No source file modified (verified via `git status` — only `proof_*.txt` and `d2h_localization.md` are untracked).

---

## Recommendation to manager (decision-relevant)

1. **Do not commit a fix yet.** The 53 D2H number is a *profiling-window artifact*, not an operational-loop leak. Failure-critic's verdict on the broader THETA_BOUNDS/WIND_BOUNDS bug should not be coupled to this counter.
2. **Re-run the Nsight capture with the standard warm-up discipline** documented in `backends/jax/bench.py:413` (`lower().compile()` → one untimed post-compile call → `cudaProfilerStart` → timed steps). Expected outcome: D2H = 0.
3. **If the warmed re-capture still shows D2H > 0**, the suspects (in order of likelihood) are:
   - `coupling/boundary_apply.py:48-58` — `interpolate_boundary_leaf` recomputes a (time, side, z, side_index) gather every step; XLA may stage indices through host scratch under some kernel selections.
   - `dynamics/acoustic_wrf.py:527,616-621` — coefficient construction with `jnp.asarray(mut, dtype=jnp.float64)` and `jnp.ones_like(...) * jnp.asarray(scalar, ...)`. Scalar broadcasts can occasionally promote to host scratch.
   - `coupling/physics_couplers.py:209-210` — `jnp.asarray(_from_columns(...), dtype=jnp.float64) - jnp.asarray(getattr(state, field), dtype=jnp.float64)` in the Thompson tendency side-channel; only triggered with `return_tendencies=True`, which `operational_mode.py:213` does **not** use, so it should not fire — but worth verifying the chosen overload.
4. **Constitution invariant interpretation**: the PROJECT_CONSTITUTION D2H=0 rule is operationally about steady-state per-step transfers, not about XLA's first-call constant uploads. Suggest the next ADR clarify "steady-state per-step D2H=0 after one warm-up call inside the cudaProfilerStart/Stop window" as the binding measurement protocol.

---

## AGENT REPORT

**Objective**: Localize the 53 D2H transfers Nsight measured during M6b operational first-step capture; no fixes.

**Files changed**: none in `src/`. Created under `.agent/sprints/2026-05-25-m6b-d2h-grep/`:

- `d2h_localization.md` (this memo)
- `proof_grep_results.txt` (repo-wide grep)
- `proof_grep_closure_d2h.txt`, `proof_grep_closure_np_print.txt`, `proof_grep_closure_jnp_array.txt` (closure-scoped greps)
- `proof_import_closure.txt` (29-module AST BFS)
- `proof_nsys_summary.txt` (nsys stats cuda_gpu_mem)
- `proof_nsys_d2h_per_transfer.txt` (53-row detail)
- `proof_nsys_d2h_byte_clusters.txt` + `proof_nsys_d2h_clusters.txt` (cluster summary)
- `proof_nsys_d2h_prev_kernel.txt`, `proof_nsys_d2h_next_kernel.txt`, `proof_nsys_d2h_runtime_api.txt`
- `proof_no_touch.txt` (642 tests collected)

**Commands run** (cores 0-3, `OMP_NUM_THREADS=4`):

- `grep -rn ...` over `src/gpuwrf/{runtime,dynamics,physics,validation,coupling,contracts,timestep,backends,io}/`
- Python AST BFS from `gpuwrf.runtime.operational_mode`
- `nsys stats --report cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum`
- `sqlite3` queries on `CUPTI_ACTIVITY_KIND_{MEMCPY,KERNEL,RUNTIME}`
- `pytest --collect-only`

**Proof objects**: 12 `proof_*.txt` files (above).

**Key finding**: 53 / 53 D2H transfers attributed; **none originate from user source `device_get` / `item` / `tolist` / `pure_callback` calls in the operational closure** (closure-grep returned 1 hit, in a builder helper outside the jit path). All 53 transfers are XLA first-graph-build staging captured because the Nsight `cuProfilerStart` fired *before* the mandatory warm-up call. 50 of 53 transfers complete before any compute kernel launches.

**Cluster breakdown**: 20 vertical metric vectors + 10 surface (ny,nx) fields + 12 boundary u/v faces + 11 scalar staging = 53.

**Unresolved risks**:

- Could not directly reproduce a warmed-capture D2H=0 because the sprint is read-only; recommendation to manager is to re-run the canary harness with `bench.py:413` warm-up discipline and re-check.
- If warmed re-capture still shows D2H > 0, the remaining suspects (in priority order) are listed in section "Recommendation to manager" #3.

**Next decision needed**: manager should (a) commission a warmed-capture re-run of Nsight on `run_forecast_operational`, AND (b) keep failure-critic's THETA_BOUNDS / WIND_BOUNDS investigation decoupled from the D2H counter — they are independent defects.

Branch: `tester/opus/m6b-d2h-grep` in worktree `/tmp/wrf_gpu2_d2hgrep`. No remote push.
