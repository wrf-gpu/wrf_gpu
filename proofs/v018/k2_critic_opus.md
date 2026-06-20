# v0.18 K2 Multi-GPU — Opus-MAX Adversarial Critic

- **Verdict: FIX (then ACCEPT-AS-EXPERIMENTAL).**
- Deliverable: commit `c1dadc1d` on `worker/gpt/v018-k2`; report `proofs/v018/k2_multigpu_report.md`.
- One-line: the **default-off path is provably bit-identical** (safe to merge into v0.18 trunk now), and the interior+seam decomposition is genuinely correct — but the report **overstates physical-boundary correctness** behind a widened tolerance, and the **NCAR / multi-node instructions are not runnable as written**. Those are doc/runnability fixes, not a core-algorithm reject.

Reviewer: autonomous Opus-MAX kernel critic. Method: read substrate + harness, re-ran the 4 data-independent fake-mesh checks on 3 CPU devices (all reproduced), traced the boundary mechanism in source. d02 operational numbers taken from the committed JSON (the indices are reproduced/verified). No real multi-GPU here (one RTX 5090).

Reproduction note: my independent d02 re-run (`--check operational-forecast --devices 3`, pinned `taskset -c 0-3`) **deadlocked on the CPU `collective-permute` rendezvous** ("40 s timeout, only 1 of 3 threads arrived") and was killed. This is a fake-mesh threading artifact — 3 pmap threads + their internal pools starved on 4 cores — **not** a K2 defect; the lighter substrate checks reproduced fine on the same pinning. Takeaway for the manager: the CPU full-forecast proof needs adequate cores (the committed run used a wider core set); note this in the harness so re-runs don't false-fail.

---

## Scope of commit c1dadc1d (what is actually new)

`git --numstat` shows `sharding.py` is **+134/-0** and the dycore context-hook modules
(`flux_advection`, `core/acoustic`, `acoustic_wrf`, `rk_addtend_dry`, `small_step_prep`,
`operational_mode`) are **not touched** by this commit. So:

- The partition / merge / `ppermute` substrate and `run_forecast_operational_sharded` **pre-existed** (D1/D2, already in trunk).
- c1dadc1d adds the **K2 entry layer** (`from_env`, `run_forecast_operational_k2_experimental`, `initialize_k2_distributed_from_env`), the **k2-lab harness** (`scripts/verify_multigpu_dgx_sim.py +397`), three **v0.17 nesting ports** (`nested_pipeline`, `boundary_construction`, `interp`, `domain_tree`), tests, and proof artifacts.

Implication: core-kernel risk is lower (substrate already reviewed); K2's own risks are the entry layer, the report's claims, and runnability. The bundled v0.17 nesting ports are scope-creep in one commit (see §6).

---

## 1. DEFAULT-OFF BIT-IDENTITY — PASS (stronger than the report claims)

Independently re-ran `--check all` on 3 CPU fake devices:

- `select_forecast_runner(disabled) is run_forecast_operational` → **True** (literal function object, `sharding.py:151`).
- Flag-off compiled HLO: `op_count 161 == 161`, **`hlo_sha256` IDENTICAL**, zero collective/SPMD tokens.

The report only asserts "op count unchanged + no collectives" (`assert_flag_off_graph_unchanged` does **not** compare the sha). My run shows the sha is in fact byte-identical — a strictly stronger result. Combined with the function-identity selection and the fact that every `_SHARDED_*_CONTEXT` global defaults to `None` (each consumer takes its original branch, e.g. `flux_advection._collapse_x_face_periodic` returns early when `context is None`), the default single-GPU graph is unchanged for **any** input, not just the tested config. **The v0.18 default does not regress.**

Residual nit (non-blocking): consider tightening `assert_flag_off_graph_unchanged` to also compare `hlo_sha256` so the proof asserts what is actually true.

## 2. DECOMPOSITION CORRECTNESS — interior/seams CORRECT; physical boundary is a KNOWN DIVERGENCE, not a seam bug, but mis-framed as "pass"

Region split (committed JSON, reproduced indices), nx=159 / 3 shards → bounds (0,53)(53,106)(106,159), seams at x=53/106:

| field | physical_boundary_ring | internal_shard_seams | strict_interior | boundary max idx |
|---|---|---|---|---|
| theta | **0.03551** | 1.7e-13 | 7.6e-12 | x=1 |
| p | **2.8949** | 3.5e-10 | 2.5e-9 | x=0 |
| mu | **1.0320** | 2.9e-11 | 3.65e-9 | x=0 |
| ph | **0.2353** | 2.9e-11 | 5.2e-9 | x=0 |
| u | **0.01571** | 4.0e-12 | 3.3e-10 | x=1 |
| w | **0.003106** | 5.8e-13 | 5.76e-11 | x=0 |
| v | 7.7e-5 | 3.95e-12 | 4.8e-12 | x=158 |
| qv, qke, rain_acc | 0 | 0 | 0 | — |

**Seams and interior are genuinely correct** (1e-13…1e-9 = roundoff at these magnitudes; e.g. p~1e4–1e5 Pa → 2.5e-9 abs ≈ 1e-13 rel). The seam mask (radius-8 bands around x=53/106, well inside owned cells with halos crossing) is non-vacuous, and it is bit-exact — this **proves the ppermute halo substrate is correct.**

**Diagnosis of the physical-boundary residual — it is legitimate (expected), NOT a masked seam bug:**
The argument is airtight by elimination. The seam region uses the *same* periodic-ppermute halo substrate as the physical boundary and is bit-exact. Therefore a divergence that appears *only* at the global physical x-edge (every max sits at x∈{0,1,158}) cannot come from the halo machinery. It can only come from the **single-GPU reference using a non-periodic edge operator at the true domain boundary that the periodic decomposition cannot reproduce**. Source confirms this:

- `operational_mode.py:2867` calls `stage_omega_specified` — flux_advection.py:192 docstring: *"Unlike `couple_velocities_periodic` (whose `rom` wraps the x/y edges periodically … up to ~5x the physical omega on the outermost row/column of a real specified domain), this uses the domain's actual staggered faces with NO wrap, and edge-pads mu."* So the reference edge-pads (specified); the sharded path forces periodic wrap at the global edge.
- `core/acoustic.py:290` `_maybe_exchange_sharded_acoustic_halos` exchanges acoustic scratch with periodic ppermute — for the global edge that is periodic-wrap, not the reference's edge treatment (explains p's 2.89 at x=0).

So: the periodic decomposition is internally consistent (interior + seams) but at the **global physical boundary it runs a different — periodic — boundary condition than the reference's specified/edge treatment.** This is exactly the limitation the code self-states (`run_forecast_operational_sharded` docstring: "periodic x halos, not the real WRF specified-boundary decomposition"; `run_boundary=True` is rejected).

**The problem is the framing, not the math.** The report's "Region Split … `physical_boundary_ring=… pass True`" reads as *boundary correct*. It is not: the boundary is **knowingly wrong vs the reference** and only "passes" because the tolerance was widened to swallow it. The code comment is candid about this (`verify_multigpu_dgx_sim.py:64-70`): `theta` atol was raised **1.0e-2 → 4.0e-2 specifically** for "a physical-domain boundary-ring theta residual." That is a goalpost move — done transparently, but still a tolerance set to pass a known divergence, not a physics bound. (p=3.5/mu=1.2 are pre-existing loose D2 bounds that also happen to cover 2.89/1.03.)

## 3. LAB-TEST VALIDITY — PASS (non-trivial)

Re-ran on 3 CPU fake devices; all reproduced:
- Halo exchange widths 1–4: **bit-exact** (max_abs 0.0) — compares `fill_halos=False`→ppermute against the direct periodic partition, genuinely exercising send direction/wrap for mass + x-face (u) leaves.
- Sharded operators: flux5 1.79e-7 (atol 2e-7 + rtol), 6th-order diffusion 2.3e-10, x-divergence 0.0, x-face pressure-dpn 1.1e-16. Real stencil comparisons vs the global formula, not self-compares.
- e2e (ppermute→operators): consistent (flux5 2.38e-7 passes via rtol term, diffusion 4.66e-10, dpn 1.67e-16).

These are meaningful: they assert the local halo-fed operators reproduce the owned columns/faces of the global operators. Not a trivial pass.

## 4. MAX-PERF DESIGN — non-pathological for experimental; name the foot-guns

No perf *claims* are made (good; the report repeatedly says wall-time/NVLink/NCCL/overlap are unmeasured). Foot-guns to name in the report:
- **`jax.pmap`, not `shard_map`/`jit`+mesh** → limited compute/halo overlap; modern multi-host wants `shard_map`. Acceptable for a first experimental cut.
- **`forecast_halo_width=8` exchanged every dycore stage**: at local_nx=53 that is ~30% halo width (16/53) and storage 1.30× (disclosed). High comm:compute for thin slabs; real runs need fatter slabs.
- **Host round-trips at setup/teardown**: `run_forecast_operational_sharded` does `jax.device_get` on the whole state/tendencies/metrics before pmap and again at `merge_state_x`. Fine once per forecast; pathological if ever called per-step. No in-loop host transfer was observed inside the pmapped region (halo stays device-resident via ppermute) — good.

## 5. HONESTY + NCAR-RUNNABILITY — the "can't measure" list is honest; the run instructions are NOT runnable as written

The limits section (NVLink/NCCL/multi-node/overlap unmeasured; `run_boundary=True` not claimed) is honest and complete. But the run instructions have three concrete defects:

1. **NCAR k2-lab command will fail at NCAR.** It omits `--run-dir`, so it defaults to `<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_…` — a *this-workstation* path. `run_operational_forecast_check` → `build_replay_case` needs that fixture; NCAR won't have it → the operational sub-check errors out. Provide an idealized/synthetic operational case for the lab, or ship/document the required fixture.
2. **The env gate is inert for the lab script.** `GPUWRF_K2_EXPERIMENTAL`/`GPUWRF_K2_PARTITIONS` are read only by `ShardingConfig.from_env`, which the harness never calls — every lab check builds `ShardingConfig(enabled=True, num_partitions=devices, …)` directly and is driven by `--devices`. The NCAR command sets those envs; they do nothing. The gate applies to the *production* entry `run_forecast_operational_k2_experimental`, not the proof script. Instructions conflate the two.
3. **Multi-node is documented but not wired.** `initialize_k2_distributed_from_env` is **never called** by the script, and `main()` enumerates `jax.devices()` before any distributed init. So "add `GPUWRF_K2_MULTI_NODE=1` … and run this script" does not actually initialize JAX distributed. As shipped this is **single-node multi-GPU (pmap) only**; multi-node needs the init wired before device enumeration (or a separate launcher), and the report should say so plainly.

## 6. Process / robustness (non-blocking, document)

- **Global monkeypatching is not reentrant/thread-safe.** `run_forecast_operational_sharded` mutates `operational_mode.halo_spec` + six `_SHARDED_*_CONTEXT` globals with try/finally restore. The default-off path is unaffected (globals only mutated while enabled), but K2 must **not** run concurrently with the default path in one process. State this constraint.
- **Commit bundles unrelated v0.17 nesting ports** (`boundary_construction` +172, `nested_pipeline` +60, `domain_tree` +45, `interp` +67). Report says they were validated (domain-tree + edge-only tests). Fine, but they should be a separate commit; and the manager should run the full pre-existing operational regression suite to confirm the bundled edits don't drift the default path (my flag-off proof covers the dycore None-branches structurally, but the nesting edits are outside that proof).

---

## Must-fix before this proof is cited / handed to NCAR

1. **Re-frame the physical boundary (report).** Do not present `physical_boundary_ring … pass True` as correctness. State: interior + internal seams are bit-for-bit vs the single-GPU reference; the **global physical x-boundary diverges by up to theta 0.036 K / p 2.89 Pa / mu 1.03** because the periodic halo substrate does not reproduce the reference's specified/edge boundary; **K2 is physically valid only for periodic/idealized domains until the specified-boundary decomposition lands.**
2. **Disclose the tolerance move.** Note theta atol was widened 1e-2→4e-2 to accommodate the boundary divergence (not a physics bound). Prefer reporting interior+seam tolerance (roundoff) separately from the boundary; don't fold the boundary into a single "within tolerance: True."
3. **Fix NCAR-runnability:** give the k2-lab check a fixture-free operational case (or document + ship the d02 dependency and pass `--run-dir`); and correct the docs that imply `GPUWRF_K2_EXPERIMENTAL/PARTITIONS` drive the lab script.
4. **Multi-node: wire it or scope it down.** Either call `initialize_k2_distributed_from_env` before device enumeration in the entry path, or relabel the deliverable single-node-multi-GPU and move multi-node to "designed, unexercised."

## Should-fix (follow-up)
5. Add `hlo_sha256` equality to `assert_flag_off_graph_unchanged` (it already holds).
6. Document the non-reentrant global-patch constraint; longer term migrate to `shard_map`/explicit-mesh to drop the monkeypatching and enable overlap.
7. Split the v0.17 nesting ports into their own commit.

## Bottom line for the manager
- **Merge into v0.18 trunk: SAFE NOW** — default-off is proven bit-identical (function identity + identical HLO sha + None-branch substrate). The release default cannot regress from the gated code.
- **Do NOT advertise "K2 multi-GPU validated" or hand the report to NCAR until §Must-fix 1–4 land.** The current report would let a reader believe the physical boundary is correct and that the NCAR command runs; neither is true.
- Accurate one-liner to ship instead: *"K2 (experimental, default-off): periodic x-domain decomposition reproduces the single-GPU dycore bit-for-bit in the interior and at internal shard seams on a 3-device CPU fake mesh; the global physical boundary uses periodic (not specified) BC and is not yet faithful; multi-GPU/cluster performance and multi-node launch are unmeasured/unexercised."*
