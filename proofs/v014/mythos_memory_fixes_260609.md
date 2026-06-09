# V0.14 Mythos Memory/FP32 Lane

Verdict: `MYTHOS_MEMORY_LANE_CLOSED_MYNN_TILING_MATERIAL_FIX_R0_LANDED_REST_MEASURED_OR_EXACT_DEFER`.

- Branch: `worker/mythos/v014-memory-fp32` @ `a32efce32852`
- CPU-only proof run; GPU evidence via scripts/run_gpu_lowprio.sh artifacts.

## Item Table

| Item | Status | Files | Memory effect | Gate | Recommendation |
|---|---|---|---|---|---|
| 1 exact-branch memory preflight (a32efce3 lineage) | `DONE_GPU_PASS` | proofs/v014/exact_branch_memory_preflight.py (resident-bridge allowlist only) | baseline peak compute VRAM 8169 MiB (nested L3 1h, 3 domains); final-tree rerun recorded in this proof | PASS_SHORT_GPU_PREFLIGHT + PIPELINE_GREEN + all finite + allocator re-exec + 0 OOM | MERGE |
| 2 moisture transport velocity reuse | `IMPLEMENTED_BIT_IDENTICAL__MEASURED_NON_MATERIAL` | src/gpuwrf/runtime/operational_mode.py | measured GPU compiled temp delta 0.0 GiB (XLA CSE already deduplicated); source-level guarantee retained | function-level exactness (this proof) + v013 five-gate wiring rerun + dynamics tests | MERGE (hygiene) |
| 3 non-radiation column tiling pilot (MYNN BouLac) | `MEASURED_MATERIAL_AND_FIXED` | src/gpuwrf/physics/mynn_pbl.py | compiled temp -11.53 GiB @641x321x50, -4.91 GiB @313x313x50 (untiled vs tile 16384) | GPU tile-vs-untiled bit identity (incl. ragged production tile); CPU non-tridiag exact + tridiag <=5e-13 codegen bound; MYNN suite; nested GPU preflight green | MERGE |
| 4 post-physics non-dry sparse/donated merge | `NON_MATERIAL_NOW` | none | static 1.3-2.6 GiB vs measured real-case peak 8.2/32 GiB and MYNN -11.5 GiB | exact-branch preflight headroom evidence | DEFER until a preflight shows pressure |
| 5 moisture limiter/species workspace | `MEASURED_DEFER` | none | active moist_adv_opt=2 limiter costs +1.90 GiB compiled temp at target geometry | GPU compile measurement (suite) | DEFER until active moisture advection is a validation target |
| 6 PBL/surface bottom-only prep / duplicate diagnostics | `DEFER_SEMANTIC` | none | 0.3-0.8 GiB static; not binding | surface->PBL contract proof required first (correctness, not memory) | DEFER to a PBL/surface correctness sprint |
| 7 acoustic scan carry split / evolving-only carry | `EXACT_DEFER_FAULT_SURFACE` | none | static ~1.56 GiB recoverable | open one-RK-step P/PH/MU dynamics divergence owns this fault surface; prior split attempt was reverted | DEFER until dynamics frontier closes; co-design with FP32 R2 |
| 8 small dycore mask/pad helper cleanup | `EXACT_DEFER_ADJACENT_ONLY` | none | 0.078-0.3 GiB | same acoustic fault surface; not worth standalone | DEFER; do adjacent to future acoustic work |
| 9 state total/perturbation/base alias reduction | `EXACT_DEFER_ADR_GATED` | none | 0.16-0.32 GiB | ADR + restart/wrfout/boundary parity required; high ABI risk | DEFER; needs ADR after grid parity |
| 10 FP32 mixed perturbation-authoritative acoustic | `R0_LANDED_DEFAULT_INERT__R1_EXACT_BLOCKER` | src/gpuwrf/contracts/precision.py, src/gpuwrf/runtime/operational_mode.py, tests/test_operational_namelist_cache_key.py, proofs/v014/fp32_acoustic_static_audit.py | none yet (contract only); future acoustic peak 1.5-2.3 GiB best case per roadmap | 5 cache-key tests green; audit: 0 timestep consumers; blocker = open fp64 P/PH/MU one-step divergence on the same files | MERGE R0; R1 after dynamics frontier closes |
| 11 newly discovered issues | `ONE_FOUND_AND_FIXED_VIA_ITEM_3` | see item 3 | MYNN BouLac dense materialization measured larger than mapped (11.5 GiB vs 'measure-first') | GPU suite measurement | covered by item 3 |

## GPU Proof Runs

| Run | Result |
|---|---|
| exact-branch nested preflight (final tree) | PASS_SHORT_GPU_PREFLIGHT — peak compute 8116 MiB, total 9265 MiB, 933.402 s |
| MYNN untiled vs tiled compiled temp | [{"batch": 205761, "untiled_temp_gib": 14.711055, "tiled_temp_gib": 3.178478, "temp_delta_gib": 11.532576}, {"batch": 97969, "untiled_temp_gib": 7.004383, "tiled_temp_gib": 2.092053, "temp_delta_gib": 4.912331}] |
| MYNN GPU tile bit identity (B=40000, tile=4096) | True |
| MYNN GPU tile bit identity, production tile (B=97969, tile=16384, ragged) | True |
| velocity reuse duplicate-vs-shared temp delta | 0.0 GiB (values identical: True) |
| limiter opt2-vs-opt0 extra temp | 1.904037 GiB |

## In-Process Proofs (CPU)

| Proof | Result |
|---|---|
| MYNN tile-vs-untiled CPU gate (non-tridiag exact; tridiag <= 5e-13 XLA:CPU codegen bound; tiles 128/1024, ragged) | True |
| moisture velocity reuse function-level exactness (opt=2 final stage) | True |
| FP32 R0 default-inert contract checks | True |
| v013 moisture wiring five-gate rerun | ALL_FIVE_GATES_PASSED |

## Deferred / Impossible (ranked, exact reasons)

1. Acoustic carry split + pad/mask helpers + FP32 R1/R2: the one-RK-step
   fp64 dynamics divergence (P/PH/MU lane, p95 WRF-EOS residual ~770 Pa,
   proofs/v014/mythos_kernel_fix_260609.json) owns exactly these files;
   editing them now would unfreeze the active root-cause fault surface and
   no WRF-anchored mixed-precision gate can pass until fp64 closes.
2. State alias reduction: ADR-gated ABI change, 0.16-0.32 GiB, not binding.
3. Moisture limiter workspace (+1.90 GiB measured): semantic FCT rewrite,
   only material when active moisture advection is a validation target.
4. Post-physics merge (static 1.3-2.6 GiB): non-material at measured
   real-case peak 8.2 GiB / 32 GiB after the MYNN fix.
5. PBL/surface bottom-only prep: correctness-unsafe without a
   surface->PBL contract proof; memory upside not binding.

## Final Merge Recommendation

`MERGE_NOW`

Details: proofs/v014/mythos_memory_fixes_260609.json;
GPU suite: proofs/v014/mythos_memory_gpu_suite_260609.json;
preflight: proofs/v014/exact_branch_memory_preflight.json.
