# Backend portability

The model retains its JAX interfaces and selects execution by backend capability.
CUDA uses the existing Pallas/Triton kernels. On other JAX backends, a local
Pallas facade lowers the same kernel bodies to ordinary JAX operations with
Pallas's interpreter. Tridiagonal solves use a JAX Thomas recurrence outside
CUDA. These paths retain the release's precision and WRF fidelity switches.

The interpreter is an execution implementation, **not an independent scientific
reference**. The portability gate exercises the same native kernels through the
facade and direct Pallas interpretation, within documented REAL tolerances, plus
finite release-default Swiss CPU steps. CUDA callable/source and release jaxpr
identity preserve the NVIDIA program. This work does not run new CUDA forecasts.
Original-WRF component fixtures supply separate scientific checks where available.

Legacy reference paths are not WRF-faithful; non-CUDA runs use the native kernels
through the facade. In particular, the old full acoustic composition has W/PH
ownership differences at specified and nested boundaries, stale `mu_work`/total reconstruction, and downstream pressure
differences; the old PD reference also differs. These are recorded diagnostics,
not portability acceptance references. No legacy equations are changed by this work.

| Backend | Dispatch | Qualification |
|---|---|---|
| NVIDIA CUDA | Original Pallas callable, Triton kernels and PTX rounding | CUDA program identity checked on CPU; prior GPU receipts apply to unchanged work |
| CPU | Pallas bodies lowered to JAX; JAX Thomas solves | Two release-default Swiss steps finite; targeted component checks; full forecast qualification pending |
| AMD ROCm | Portable JAX route; no NVIDIA setup. Experimental opt-in native Triton route: `GPUWRF_ROCM_TRITON_NATIVE=1` | Hardware untested; capability dispatch tested with simulated devices; native route only compiled on CPU (see below) |
| NVIDIA older than Ampere (cc < 8.0) | Experimental opt-in stock-Triton route: `GPUWRF_PRE_AMPERE_KERNEL_CALL=1` (CUDA 12 install) | One GTX 1080 Ti run (see below); unqualified |
| TPU | Portable JAX route; no NVIDIA setup | Hardware untested; DOUBLE islands and integer-64 support require qualification |
| Apple Metal | Portable JAX route when a compatible JAX plugin is installed | Hardware untested; required precision support unqualified |
| Intel / other PJRT GPU | Portable JAX route when a compatible plugin is installed | Hardware untested; required precision support unqualified |

Use an explicit JAX platform pin for reproducible backend selection, for example
`JAX_PLATFORMS=cpu gpuwrf run ...` or `JAX_PLATFORMS=rocm gpuwrf run ...` with an
installed matching JAX backend. This change does not install accelerator plugins
or emulate unavailable float64/integer64 support. A rejected plugin or unsupported
primitive remains an error; the model does not silently downcast DOUBLE islands.

## v0.5.0 compile check

Done on 11 November 2026 on the v0.5.0 model source (the same `src/` as the release), with the
two-domain 9/3 km Canary Islands case and the release defaults. These are compile results, not validations: accuracy and
speed on these GPUs are untested.

| Target | Full two-domain program | Notes |
|---|---|---|
| NVIDIA Ampere / Ada (A100, RTX 30, RTX 40; sm_80/86/89) | Compiles (standard install, no GPU needed) | Stack at most 32 B per kernel, no local-memory spill. Never run on these GPUs |
| NVIDIA RTX 5090 (sm_120) | Compiles | Control: the release GPU |
| AMD MI250X / MI300X (gfx90a/942) | Compiles with `GPUWRF_ROCM_TRITON_NATIVE=1` | No vector-register spill (small scratch in a few compiler fusions). Nothing executed on AMD hardware; control flow stays host-driven on ROCm |
| AMD RX 7900 (gfx1100) | Compiles with `GPUWRF_ROCM_TRITON_NATIVE=1` | The acoustic-coefficient kernel spills (26 vector registers, 108 B scratch), so it would be slower. Not executed |
| NVIDIA Pascal / Volta / Turing (sm_61/70/75) | Standard install refuses them (its compiler needs compute capability 8.0) | With `GPUWRF_PRE_AMPERE_KERNEL_CALL=1` and a CUDA 12 JAX install, all 127 kernel bodies of the nested program compile for all three; the acoustic-coefficient kernel spills (and one radiation kernel by 16 B on Volta) |

Real run on one GTX 1080 Ti (sm_61, 11 GB) with the pre-Ampere route: a 3-hour two-domain forecast finished with all
output finite (cold run 803 s including compilation, warm run 146 s, 5.1 GB host memory warm; one sample each). The warm
output equals the cold output byte for byte; that is a determinism check only. No fidelity comparison was run on that
card, and Volta and Turing have not run on hardware. The CLI still prints an outdated warning on this route that the
default kernels will fail; it can be ignored.

## NVIDIA assumptions and their boundaries

| Assumption | Location | Portable behavior |
|---|---|---|
| GPU-required allocation | `contracts/state.py` | State constructors use the selected JAX device; empty/unavailable backends still fail in JAX |
| `cuda_async`, pool sizing, NVIDIA VRAM queries | `cli.py`, `runtime/gpu_allocator.py`, `integration/nested_pipeline.py` | No allocator re-exec, CUDA defaults, or C-auto NVIDIA query on a non-CUDA target |
| NVIDIA VRAM/lock preflight | `runtime/gpu_preflight.py` | Explicit skipped/backend receipt on non-CUDA; CUDA locking and admission unchanged |
| `nvidia-smi` compute capability in cache tag | `runtime/compile_cache.py` | CPU/ROCm/TPU/Metal/Intel tags do not invoke NVIDIA tools |
| GPU command buffers, autotune, parallel compilation XLA flags | `runtime/xla_autotune.py` | No CUDA flag injection for non-CUDA targets; operator-supplied flags remain operator policy |
| Pallas/Triton and PTX inline assembly | `kernels/*.py`, `nesting/sint_kernel.py`, `coupling/boundary_apply.py` | Local facade selects JAX interpretation; rounding helpers use portable arithmetic |
| NVIDIA compiled thunks / cubin metadata | `runtime/aot_executable.py` | NVIDIA-only compaction/export/load transforms require a CUDA device |
| `CUDA_ROOT`, `ptxas`, `cuobjdump`, Nsight | Development profiling/build scripts, `backends/cuda_tile/`, `backends/jax/bench.py` | Outside the portable forecast path; no portable compiled-kernel resource, roofline or Nsight claim |

Bootstrap detection does not initialize JAX: explicit platform pins win, AMD
visibility/device nodes distinguish ROCm from the generic `gpu` alias, and NVIDIA
device nodes/visibility establish the default CUDA target. Kernel dispatch inspects
the actual PJRT client's platform version and device kind. An unknown accelerator
uses the portable route.

## Experimental native ROCm route (v0.5.0 groundwork)

`GPUWRF_ROCM_TRITON_NATIVE=1` lets an AMD ROCm device run the Pallas kernels as
native Triton instead of interpreting them. Without the variable ROCm keeps the
portable route and prints a one-time warning. The selection lives in
`gpuwrf.backends.vendor_ops`:

- `vendor()` returns `cuda`, `rocm` (admitted ROCm only) or `interp`.
- WRF-literal binary32/binary64 `+ - *` use AMDGCN instructions
  (`v_add_f32`, `v_sub_f32`, `v_mul_f32`, `v_add_f64`, `v_mul_f64`) through
  inline assembly, so the compiler cannot contract them into FMAs. Division uses
  the IEEE `/` sequence. This is the AMD counterpart of the PTX `.rn` spelling.
- CUDA call sites are unchanged; the CUDA program stays byte-identical.
- Optional schemes with NVIDIA-only typed memory or math externs (NTU, Jensen,
  P3, QNSE, Ferrier, Goddard, Milbrandt, CAM-MG) stay interpreted on ROCm.

Status: the whole production step compiles for gfx942, gfx90a and gfx1100 with
XLA-ROCm on a CPU host (no device). No AMD device has executed it, no WRF oracle
has run on AMD hardware, and ROCm command buffers cannot hold conditional/while
nodes, so control flow is host-driven. Treat it as unqualified. Evidence:
`proofs/o5_port/REPORT.md` (October 2026 development tree; the v0.5.0 result is in
[v0.5.0 compile check](#v050-compile-check)).

`GPUWRF_ROCM_INTERPRET_KERNELS=name1,name2` keeps individual kernels on the
interpreter while the others run natively (an escape for a kernel that fails on
one device).

## Experimental route for NVIDIA GPUs older than Ampere

XLA compiles Pallas kernels only on compute capability 8.0 and newer, and the
CUDA 13 assembler no longer accepts Pascal (sm_61) or Volta (sm_70). With a CUDA
12 JAX install and the stock `triton==3.6.*` package,
`GPUWRF_PRE_AMPERE_KERNEL_CALL=1` compiles every kernel with stock Triton for the
actual GPU and launches it through JAX's `triton_kernel_call`. On compute
capability 8.0 and newer, or without the variable, the compiled program does not
change. Setting the variable does change the cache key of saved compiled
programs, so the first run after setting it compiles once more. The product
preflight still asks for `GPUWRF_ALLOW_UNSUPPORTED_GPU=1` on these GPUs.

On an earlier development tree (October 2026), measured on one GTX 1080 Ti (sm_61)
with the release defaults: the bundled Swiss
case ran 24 hours at 4.15 s per forecast hour for the whole run (2.5 s in steady
state) and passed the D6 comparison against original CPU-WRF over all 24 hours;
the two-domain (9/3 km) Canary Islands case ran at 25.8 s per forecast hour in 11 GB. The frozen WRF
component oracles for MYNN (tridiag2, bitwise), Thompson, Kain-Fritsch and the
RRTMG shortwave pass on the same GPU. This is a single-box experiment, not a
qualified configuration.

## Parity and limits

`tests/test_portability.py` checks vendor selection, original CUDA callable identity,
CUDA-free lowering, state selection, and the hydrostatic recurrence. The companion
`tests/test_portability_kernels.py` compares direct stencils and real-operand dycore
operators with separate equations. Existing component gates cover radiation, PBL,
microphysics, land, cumulus and nesting.

New REAL comparisons reject NaN/Inf in both arms and bound maximum absolute error
by a documented number of float32 epsilons times the reference field scale (normally
64, coupled and boundary operators 8, hydrostatic 4). EOS perturbation pressure uses total pressure
as the cancellation scale. Random streams use exact equality. Existing component
gates retain their original tolerances; none are widened by backend dispatch.
CPU contraction, reduction order and subnormal handling differ from CUDA PTX RN32,
so interpreter success does not imply GPU byte equality.

`proofs/s1_port/FAMILY_INVENTORY.json` maps all 31 Pallas modules to component
checks and source-identity receipts. `proofs/s1_port/release_identity.py` adapts
the established release probe for CPU-only native-program tracing: initialization
uses the portable route, then the local facade selects the original callable for
the trace. Merely mocking `jax.default_backend()` no longer establishes CUDA
capability. This proof does not compile or execute a CUDA executable.

[M, CPU/source] The family checks cover 120 unique passing tests, with an observed
native factory in each of the 31 modules. The static audit covers 48 factory sites
and retains CUDA-equivalent source bytes and line counts. The CUDA-callable mutant
produces one genuine assertion failure and zero setup errors. These are portability
and source-identity results, not a new GPU or whole-forecast scientific validation.

Supplemental EOS and UV comparisons disable the CPU algebra-simplification pass locally to
avoid its known rewriting of divisions (reassociation); production flags are unchanged.
BouLac uses the release's WRF REAL constants in both arms. The acoustic reference
helper predates the specified/nested boundary ownership fix, so the retained diagnostic wrapper supplies
WRF's `advance_uv` update bounds from `module_small_step_em.F:735–751` before
downstream operators. Its wider coefficient/work outputs are compared at native
REAL storage precision, with finiteness checked before conversion. Those old-path
diagnostics still fail and are kept outside the portability acceptance suite.

CUDA returns the original `pallas_call` object, adds no wrapper to native kernel
tracing, and retains kernel source lines. [I, from program/source identity] The
facade introduces no additional native timestep launches, temporary arrays or
host/device copies. The rebased upstream initialization ownership guard is retained.
Capability checks occur at trace time;
source fingerprints change, so existing executable caches may need warming again.
Portable interpreter compile cost, memory use and speed require measurement for
each workload and device. No ROCm/TPU/Apple/Intel forecast validation is claimed.

[M, CPU] The smoke uses the bundled Swiss case (42×42 columns, 44 levels,
18 s timestep), release defaults and the normal operational advance wrapper,
without changing model dispatch. Two steps leave all 167 floating-point carry
leaves finite and invoke no NVIDIA tools. On shared core 29, load took 309.882 s,
the first step including cold trace/compile took 1376.216 s, and the second step
took 1.890 s. Process peak RSS including compilation was 6,359,104 KiB. These
are workload-specific CPU costs, not GPU timings or a CPU-WRF speed comparison.
Times use the monotonic clock; the host suspended during this arm, extending its
calendar duration. The receipt retains the existing REAL-context dtype warning.
The default event census was disabled (`null`), so this smoke establishes final
carry finiteness rather than a pre-mask event or trajectory-fidelity qualification.
The receipt is `proofs/s1_port/receipts/PORT32_cpu_smoke.json`.

The final gate seal is `proofs/s1_port/receipts/QUALIFICATION.json`. Required shared
consumers cover 252 cases: 251 executed passes and one previously qualified CAM
pass reused after exact program and all 80 constants matched. A clean tracked
archive passed 19 portability checks and 21 CLI checks; all 526 archived source
files matched their committed bytes. The model tree qualified here is
`8063c7f205647bf0caa67a6d121db9fbdd2ba00e`, composed on main `489a91483`.
Later integration changes go through the normal review of the combined release.
