# wrf_gpu

**A GPU-native, WRF-compatible regional weather model.** `wrf_gpu` runs a
standalone WRF v4 ARW forecast end-to-end on a single GPU: it reads a standard
WRF `namelist.input` plus prepared `wrfinput_*`/`wrfbdy_d01` real-data artifacts
(no runtime `real.exe` or CPU-WRF `wrfout` dependency), integrates a nonhydrostatic
split-explicit ARW dycore on the GPU, and writes a WRF-compatible `wrfout`
history file.

> 📖 **User's Guide:** the full, searchable HTML guide is live at
> **[wrf-gpu.github.io/wrf_gpu](https://wrf-gpu.github.io/wrf_gpu/)** (patterned after the WRF
> Users' Guide; source in [`docs/`](docs/)). In v0.23.4 the accepted live-nested
> path reaches nine domains; you can also
> **clone the repo and ask an AI agent to run your case** — see [`AI_OPERATOR.md`](AI_OPERATOR.md).

> 💡 **Ideas or feature requests?** Open a
> [**feature request**](https://github.com/wrf-gpu/wrf_gpu/issues/new?template=feature_request.yml)
> — no email needed, just a free GitHub account. Requests are collected in the repo's
> [Issues](https://github.com/wrf-gpu/wrf_gpu/issues) tab (see the
> [port-completion roadmap](docs/PORT_COMPLETION_ROADMAP.md) for what's already planned).

It is **not** a port of legacy WRF Fortran. It is a clean JAX rewrite that
targets the GPU memory hierarchy from day one and validates against WRF as an
**oracle** — proving cell-for-cell identity to CPU-WRF v4 rather than inheriting
WRF's architecture. The dynamical core runs in **fp64** (around the
pressure-gradient / buoyancy cancellation). The original operational target is
**Canary Islands daily forecasting** (3 km then 1 km) on a single-workstation
RTX 5090 — but its real strength is at the **opposite end of the spectrum: large
grids, big fp64-native GPU systems, and GPU clusters** (B200 / GB300 /
NVL72-class).

### What it is good for

- **Running real regional ARW forecasts on a GPU** from a standard WRF namelist —
  single-domain or live-nested (validated through d01→…→d09), with native
  init, restart, and a WRF-compatible `wrfout`.
- **Energy efficiency + modern-HPC fit (PROJECTED).** Past a certain level of
  parallelism, GPUs are inherently more energy-efficient per unit compute than
  CPUs. For serious/large workloads this rewrite is **projected** to run at
  **>3× the energy efficiency** of the CPU stack, to scale with GPU size, and to
  ride the trend of ever-faster, cheaper GPU compute — per kWh and per dollar.
  *(Projected from the device-bound kernel + architecture; not yet benchmarked at scale.)*
- **Capability the CPU stack cannot reach on one box.** **MEASURED:** a **1 km
  single domain fits one RTX 5090 bit-identically**, and the **all-7-island 1 km
  nested case runs end-to-end on one card**. **PROJECTED:** large single grids and
  **cluster / multi-GPU weak-scaling** — the throughput path (memory arithmetic +
  fake-mesh bit-identity proven; real multi-GPU throughput **not yet benchmarked**).
- **Batched ensembles fill the GPU on small grids (MEASURED).** A single small nest
  under-uses the card (it is launch/occupancy-bound); running **B independent
  same-geometry forecasts under one `jax.vmap`** recovers the idle throughput without
  touching the dycore. On the **Tenerife 3 km→1 km 2-nest on one RTX 5090**, warm total
  throughput rises from **4.51 M cells/s at B=1 (only 28.5 % GPU util)** to **5.63 M
  cells/s at B=4 — ≈97 % of a large saturated mountain grid's ceiling** (5.81 M cells/s,
  measured on an Alps 433²@1 km case), while the default unbatched path stays
  **byte-identical**. B=5 plateaus (VRAM-capped at ~26 GiB on the 5090); a larger card
  (B200) batches further. So the small-grid launch-bound penalty is **largely
  recoverable by batching**. *(opt-in `GPUWRF_BATCH_ENSEMBLE`; see [Performance](#performance).)*
- **A transparent, forkable research artifact.** Every claim has a proof object on
  disk; every architecture decision has a cross-model-reviewed ADR. It is built to
  be driven and extended by an AI manager agent (see [Use the manager](#use-the-manager-agent-driven-development)).

### What it is NOT

- **Not a universal WRF v4.** It covers the common operational ARW subset
  (the wired physics menu below); every unsupported namelist option **fails closed
  before any compute** with a named reason — it never silently substitutes a scheme.
- **Not proven for full 24 h/72 h forecast-skill equivalence.** The sealed
  Switzerland/Canary identity fixtures prove their dynamics/thermodynamics
  fields **cell-for-cell identical** to CPU-WRF, and v0.23.4 passes its specific
  24-hour SP2 chain gate; broad seasonal and
  configuration-independent T2/U10/V10 skill over 24–72 hours remains an **open
  credibility gate** (see [Boundaries](#boundaries--what-is-not-claimed)).
- **Nested radiation cadence is not yet namelist-faithful.** The nested pipeline
  currently targets a fixed 30-minute radiation interval instead of honoring an
  arbitrary `radt` value, and nested `topo_shading`/`slope_rad` currently bind
  disabled. The v0.23.4 one-hour nine-nest gate remains accepted, but do not
  extrapolate it into a longer-horizon radiation-fidelity claim.
- **Two accepted dynamics controls still have path-specific fidelity gaps.**
  The nested path honors `moist_adv_opt`/`scalar_adv_opt`, but the
  single-domain daily pipeline currently runs them as `0/0` even when the
  namelist requests otherwise. When `time_step_sound` is omitted, the current
  runtime selects 10 acoustic substeps instead of WRF's fixture-derived 4;
  the associated dry-mass behavior remains unresolved. Do not silently alter
  either control to manufacture parity.
- **Not a blanket single-card speedup story.** On tiny standalone geometries the
  GPU can still be launch/occupancy-bound — on a single-domain 129² grid it is
  **~2.3× SLOWER** than 24-rank CPU-WRF (host/launch-bound). The GPU advantage
  **grows with scale**: the v0.20.0 all-7-island 1 km nested fast path is
  **MEASURED ~1.07× faster than v0.19** and **~1.53× faster than the same-box
  12-rank CPU-WRF baseline**, byte-identical to v0.19 (1926/1926 vars,
  maxΔ=0). The broader value remains **capability** (1 km + scale), **fidelity**,
  **stability/reliability**, and **energy efficiency** (see [Performance](#performance)).
- **Not** DFI / FDDA / spectral-nudging / WRF-Chem / WRF-Fire / urban / lake.

**The current shipped release is v0.23.3.** v0.23.4 is **prepared, pending release publication** — it
closes the full nine-nest (`max_dom=9`) correctness chain (V10 wake-displacement scalar-transport
fix, Thompson late-Ni ice-sedimentation fix, and a full real nine-domain replay, all GREEN and
independently reviewed) and includes a validated dispatch-level performance fix for the 2-domain
(d01+d02) path (measured no regression, 0.960–1.018× v0.23.3). Landing the V10 fix also exposed a
real, measured **~51.67% performance regression** for 3-domain/9-domain configurations that engage
d03. That regression was investigated exhaustively: every quick-fix candidate (barrier, scan/map,
register-pressure/fusion-disable, B1/B2/B3 scheduling, roll substitution) was falsified or rejected
with evidence, and a Nsight Compute hardware measurement of the isolated dominant kernel confirmed
it is **occupancy/latency-bound, not memory-bandwidth-bound** (DRAM 20.3% of peak, occupancy 15.6%).
This rules out a reduced-precision fix and identifies concurrent sibling-domain scheduling as the
mechanistically-mapped next lever — design-scoped, not yet implemented. **The investigation is
closed; the regression itself is not fixed in this release.** See
[`RELEASE_NOTES_v0.23.4.md`](RELEASE_NOTES_v0.23.4.md) for the full, current status.

**v0.23.3** — a reliability point-release on top of v0.23.2 (numerically identical) — fixed a
`cuda_async` preflight false-negative that could refuse a launch on an otherwise-empty card under a
held `with_gpu_lock` (`rc=75`); a low free-VRAM reading is now advisory rather than fatal when the
lock is verifiably held, restoring the faster `cuda_async` allocator for locked production runs.
Full notes: [`RELEASE_NOTES_v0.23.3.md`](RELEASE_NOTES_v0.23.3.md).

**v0.23.1** — usability + AI-native onboarding, numerically identical to v0.23.0 — added a full
HTML [User's Guide](https://wrf-gpu.github.io/wrf_gpu/), an [AI operator skill](AI_OPERATOR.md),
and backward-compatible WRF-parity CLI ergonomics (forecast length and root domain default from
`namelist.input` when the flags are omitted; a `namelist-support` subcommand; a `--dry-run`
preflight). **v0.23.2** — numerically identical to v0.23.1 — fixed `GPUWRF_BATCH_INPUT_DIRS` to
accept comma-separated distinct-init dirs. Full notes:
[`RELEASE_NOTES_v0.23.1.md`](RELEASE_NOTES_v0.23.1.md),
[`RELEASE_NOTES_v0.23.2.md`](RELEASE_NOTES_v0.23.2.md).

**v0.23.0** — the performance + capability base — is a release on top of the
v0.22.2 nested-grid wall-clock line. It adds the opt-in **batched-ensemble** path
(`GPUWRF_BATCH_ENSEMBLE=B`) that fills the otherwise-idle GPU on small operational grids
(Tenerife 3/1 km reaches ~97 % of the large-grid throughput ceiling at B=4), default-path
launch/compile-count reductions (P0/P3 + the P-bundle, value-preserving), and a wave of
extended physics/nesting schemes that are **opt-in or reference-only/fail-closed** so the
default forecast is unchanged. The default is **numerically equivalent to v0.22.2** (the
value-preserving P0/P3 reductions match v0.22.2 within the GPU autotune floor,
self-control-verified). fp32-operational (M1) foundation is laid but is a separate
milestone, not enabled by default. Full notes:
[`RELEASE_NOTES_v0.23.0.md`](RELEASE_NOTES_v0.23.0.md). The priority
order remains explicit:

```text
STABILITY > IDENTITY > SPEED > MEMORY
```

v0.22.2 cuts the host-bound GPU-idle at the nested output boundary. Three
default-on, **byte-identical** host-work reductions — removing a redundant full
finite-summary, batching the per-leaf device→host pulls, batching the finite
guard, and a subset-aware payload build — measured **~9% faster nested
steady-state** (789 vs 865 s/fc-h on a 384² 2-nest, output byte-identical to
v0.22.1 across 12 frames × 107 variables). A default-on, **bit-identical** RRTMG
column-tile cap on the per-output radiation re-solve (the dominant ~23 s/boundary
transient) halves its VRAM (~5.1 → ~2.6 GiB) without changing any output byte,
fixing the convection-peak OOM that v0.22.1 hit on larger grids (a 433² 2-nest
now peaks 28.1 GiB < 32 over the afternoon-convection window with no OOM); the
main-forecast radiation cap is unchanged. Two opt-in levers go further:
`GPUWRF_NEST_OUTPUT_PIPELINE=1` overlaps the output materialization behind the
next GPU segment, and `GPUWRF_NESTED_M9_RADIATION_FROM_CARRY=1` skips the re-solve
(lossy, fewer fields). Default numerics and WRF-standard wrfout naming remain
byte-identical.

v0.22.1 keeps default numerics and WRF-standard wrfout naming unchanged while
fixing B200 pod output-path defects: nested leaf children now honor their own
`history_interval` on both eager and default fused paths, and
`GPUWRF_COLONFREE_OUTPUT=1` provides opt-in `HH-MM-SS` wrfout names for
S3/network-volume drains. v0.22.0 remains the base release that adds three
default-safe hygiene fixes: #136 opt-in strict fused JIT/AOT structural
signature, #115 portable `GPUWRF_WRF_ROOT` data lookup, and #101 byte-identical
async wrfout.

The only new speed lever is **opt-in**: operators may deliberately choose a
CFL-gated `time_step` / `n_sound` pair. The measured short single-domain
Switzerland-128 gate found `18 s / 7` at `19.25 s/fc-h` versus `34.67 s/fc-h` for
`10 s / 10` (1.80x), finite/bounded and within the short operational band. This
is **not** a default change, **not** a nested speedup claim, and **not** a 24-72 h
skill claim; the 3-domain steep-terrain ladder exposed vertical-CFL risk before
any higher nested rung could be evaluated.

v0.22.0 also corrects the compile-wall narrative: the wall is compile **time**,
and the AOT warm-start is the answer. The sound 3-domain harness measured about
22.32 GiB peak compile RSS, not the old roughly 60 GB premise, and AOT warm-start
measured about 39x cold-to-warm on that harness. The operational-relaxed
acceptance-tier ADR is ratified for its §9 contract (including the 1 km Alpine
case and `BOUNDED_GROWTH` hard rejection), while the K4 fp32-operational ADR is a
plan only.

The v0.21 fused+AOT foundation remains important background: after a one-time
cold compile, a fresh process loads the compiled GPU executable from disk via a
cheap metadata key and skips the multi-tens-of-minutes re-lower the old
persistent cache still paid. The AOT warm-start is on by default on the fused
runtime path. De-fuse remains an explicit low-host-compile-RAM fallback, not the
runtime default.

**Scope of the speed win — read this first.** v0.21.0 is faster at *getting a run
started* (compile / warm-start time), **not** at running the forecast itself. Warm
forecast throughput (seconds per forecast-hour) stays on the measured fused
runtime executable. De-fuse is the same bit-identical eager per-domain code path
the `GPUWRF_NESTED_FUSE=0` opt-out always used; only the XLA compile partitioning
differs, and it carries a documented runtime cost.

**Compile modes (the three knobs).** The nest default is fused + AOT:

```bash
# DEFAULT (no env): fused cascade + AOT cheap-key warm-start.
#   Higher runtime throughput; warm-start in SECONDS after the first compile.
export GPUWRF_NESTED_DEFUSE_COMPILE=1     # OPT-IN: lower host compile RAM, slower runtime
export GPUWRF_NESTED_PARALLEL_COMPILE=4   # OPT-IN with de-fuse: faster cold, more host RAM
export GPUWRF_NESTED_FUSE=0               # explicit eager/de-fuse debug path
```

The fused default runs no child processes. Parallel prewarm is strictly opt-in and
only applies when the de-fuse path is explicitly selected via
`GPUWRF_NESTED_DEFUSE_COMPILE=1` or `GPUWRF_NESTED_FUSE=0`
(`GPUWRF_NESTED_PARALLEL_COMPILE=0` is the explicit parallel-prewarm opt-out).

**Stability guardrails.** `GPUWRF_FINITE_CHECK` is enabled by default on the
nested forecast path. If a prognostic state first becomes `NaN` or `Inf`, the run
fails fast with the affected domain, field, level, step, simulation time, and
index instead of allowing corrupted values to propagate to output. Finite paths
are observational only; opt out only for explicit max-performance experiments:

```bash
export GPUWRF_FINITE_CHECK=0
```

**Dycore mechanism fix and v0.21.1 boundary closure.** The v0.21.0 dycore fix
addresses the diagnosed steep-terrain acoustic failure
mechanism: pathological dry-mass drain in the acoustic continuity loop (an acoustic
dry-mass-drain limiter) plus a singular `c2a`/`alt` denominator (a positive
physical floor). The fix is identity-preserving on normal regression cases and the
integrated CPU suite matches the v0.20.2 known-red baseline exactly (zero new
regressions). It stabilizes the 9-nest Canary gate-case past its step-67 divergence
window (all nine domains finite, MEASURED). The most-extreme 1 km Mont-Blanc
(~1042 m/cell) case needed the v0.21.1 point release, which fixed the
standalone-root specified-boundary physical-`W` boundary representation without
masking, `nan_to_num`, finite guards, clips, or clamps.

**AOT cheap-key warm-start — what the gate measured (9-nest, RTX 5090,
20240901 real case).** The persistent JAX cache stores the *compiled* executable
keyed by the *lowered* HLO, so even a warm hit had to re-lower (re-trace) the giant
nested module first — tens of minutes for the 9-nest. v0.21.0 serializes the
compiled per-domain executables to disk and indexes them by a **cheap key** (a fast
hash over the call metadata that fully determines the compiled program, computed
**without lowering**, scoped to the trace-import closure of the traced body so an
HLO-irrelevant source edit cannot shift it). A fresh process **loads the executable
directly and skips the re-lower**. MEASURED cold→warm re-confirm: **all 9 domains
`loaded=true source=aot_blob` cross-process, 0 fallback, 0 re-lower** (zero "very
slow compile" alarms in the warm log), **warm keys byte-match the cold keys**,
**finite integration (0 NaN)**, **warm peak host RSS 16.4 GB**, **load in seconds**.
A correctness backstop (`GPUWRF_AOT_VERIFY=1`, lower-once + HLO-digest compare,
fail-closed/quarantine) is available; the default is verify-off (a fresh load is
numerically inert — the cheap key only *locates* the blob; the loaded executable is
byte-identical to a cold compile).

```bash
export GPUWRF_NESTED_AOT=0      # opt OUT of AOT (default is on)
export GPUWRF_AOT_VERIFY=1      # fail-closed warm-load verification backstop (default off)
```

**Compile/cache operations.** The persistent compile cache now defaults to a
version/backend-keyed directory, so different `gpuwrf`, JAX/JAXLIB, and backend
combinations do not accidentally share one warm-cache-looking path. Autotune
cache support is default-on when the compile cache is on, with fail-open flag
probing. Prepared deployments can pack/unpack a warmed cache artifact for the
exact release/backend/JAX combination:

```bash
python -m gpuwrf.runtime.aot_precompile info
python -m gpuwrf.runtime.aot_precompile pack --out gpuwrf-jitcache.tar.gz
python -m gpuwrf.runtime.aot_precompile unpack gpuwrf-jitcache.tar.gz
```

**De-fuse is a RAM lever, not the runtime default.** The de-fuse sequential
opt-in lowers peak host RAM on the older 9-nest measurements but is **slower** to
compile cold than the fused single module (9 separate lowers; 9-nest cold ≈
**70–75 min** MEASURED, per-domain d01 ~15 min, d02–d06 ~3–8 min each, d07–d09
~13–18 min each). v0.22 corrected the 3-domain compile-memory premise: the sound
3-domain harness measured about 22.32 GiB peak RSS, so compile **time**, not the
old roughly 60 GB memory number, is the wall. The compile-time *win* is the
**AOT warm-start above** (skip the re-lower on every subsequent run), not the cold
compile. The fused path is the measured runtime path.

**Steep-terrain gate.** v0.21.0 adds an opt-in GPU regression gate for the
two-domain steep-terrain Thompson path. It uses the real nested API path and
requires both domains to remain finite:

```bash
GPUWRF_RUN_V021_STEEP_TERRAIN_GATE=1 \
scripts/with_gpu_lock.sh --label v021-steep-terrain -- \
  pytest -q tests/test_v021_steep_terrain_stability_gate.py
```

The ≥1 h-finite + all-fields CPU-match Canary gate is a **local** gate, not a 24 h
or OOM-proof default claim: in de-fuse mode the single-card 32 GB run can still hit
a GPU-VRAM OOM around the longer (~90 min) integration horizon (#123,
mitigated-not-eliminated), so v0.21.0 claims finite-past-step-67 stability plus the
AOT warm-start and compile/RAM levers — not a ≥1 h-OOM-proof default. The 3-domain
de-fuse path matches CPU-WRF to within tolerance (worst T2 2.96 K / 1.03%). B200 /
fp32 / VRAM work remains future milestone work.

Historical v0.22.1 notes:
[`RELEASE_NOTES_v0.22.1.md`](RELEASE_NOTES_v0.22.1.md). Earlier v0.22/v0.21
notes: [`RELEASE_NOTES_v0.22.0.md`](RELEASE_NOTES_v0.22.0.md),
[`RELEASE_NOTES_v0.21.1.md`](RELEASE_NOTES_v0.21.1.md) and
[`RELEASE_NOTES_v0.21.0.md`](RELEASE_NOTES_v0.21.0.md). The capability
narrative below (carried forward from v0.20.x where not explicitly changed)
still applies.

**v0.20.0 is a correctness, stability, capability, and reliability release.** It
is **bit-identical-safe by default** (the fp64 path is byte-for-byte unchanged)
and adds an honest, modest nest speedup, an opt-in fp32 capability mode, and a
compile cache that **just works across runs and across forecast dates**. On the
canonical all-7-island, 9-domain case the default fused nested path measures
**~668 s/forecast-hour warm** (range 645–680) on the reference GPU — **~1.07×
faster than v0.19** (713 s/forecast-hour) and **~1.53× faster than the 12-rank
CPU-WRF baseline** (1020 s/forecast-hour) — and is **byte-identical to v0.19
output (1926/1926 vars, maxΔ=0.000e+00)**, the gain coming entirely from a
numerics-free CUDA stream-ordered allocator. v0.18 remains the
feature-completeness baseline: every WRF v4 namelist scheme is classified and
handled, with no silent substitution or skipped scheme. See the [Scheme
triage](#scheme-triage--every-wrf-v4-scheme-classified).

*[MEASURED: COMBINED_SPEEDUP.md §5/§7 — ~668 / 1.07× / 1.53× / 1926-byte-identical.]*

> ### First run is slow on purpose, then a seconds-fast warm start
> The first forecast **JIT-compiles the GPU kernels** — a **~½–12 min one-time cold
> compile with no output before integration starts (on the n=1 reference system,
> scales with grid size)**; the large all-7 9-domain nest is a separate, larger
> **one-time compile (de-fuse sequential ~70–75 min cold, MEASURED 2026-06-25,
> RTX 5090)**. It is compiling, not hung.
>
> A **persistent, per-user on-disk JIT cache is on by default with zero config** —
> no flag, no setup — and it is now **version-keyed** (`gpuwrf` + JAX/JAXLIB +
> backend), so a stale older-release cache is never mistaken for a warm one.
> **New in v0.21.0: the warm start skips the re-lower.** Previously, even with a warm
> HLO cache, a fresh process had to *re-lower* the giant nested module (tens of
> minutes for the 9-nest) to find the right cached executable. v0.21.0 serializes the
> compiled per-domain executables and indexes them by a **cheap key** computed without
> lowering, so a fresh process **loads them directly and skips the re-lower** — the
> 9-nest gate measured **all 9 domains `loaded=true source=aot_blob` cross-process, 0
> re-lower, load in seconds** (warm peak host RSS 16.4 GB). A single-domain warm cache
> hit was already fast in v0.20 (`cold ~147 s → cache-hit ~29 s` on the d01 hour-1
> wrapper, cached executable bit-identical), and the cache also **hits across forecast
> dates** (re-running the same configuration on a new or leap-year date is a warm hit
> with 0 new cache entries, default path bit-identical). Set `GPUWRF_NESTED_FUSE=1` for
> the v0.20 fused path, or `GPUWRF_BITWISE=1` for the eager bitwise/debug path.
>
> *[MEASURED: 9-nest AOT cold→warm re-confirm `proofs/v021/blocker_9nest_cheapkey/FIX_REPORT.md` (9/9 loaded=true, 0 re-lower, RSS 16.4 GB); de-fuse cold + RSS `proofs/v021/canary_gate/GATE_RESULTS.md` + `HEADLINE_COMPILE_TABLE.md`. Cross-date warm hit: v0.20 `JULDAY_CACHE_FIX_REPORT.md` — cross-date HLO sha identical across 3 dates incl. leap; 0 new cache entries; 64/64 byte-identical.]*

> ### Optional fp32 mixed-precision mode (capability + VRAM, opt-in)
>
> v0.20.0 adds an **opt-in** perturbation-authoritative fp32 mode
> (`GPUWRF_ACOUSTIC_PRECISION_MODE=mixed_perturb_fp32_v020`). **The default stays
> fp64 (`fp64_default`) and is byte-for-byte unchanged** — the GPU all-7 9-domain
> fp64 output is **963/963 vars maxΔ=0.000e+00, byte-identical** across all 9
> domain files, and warm fp64 speed is unchanged (within noise).
>
> The value of fp32 is **capability + VRAM headroom + stability, NOT single-card
> speed**: it cuts whole-run VRAM by **−14.4%** (aggressive mode) and extends
> full-physics cell capability **~1.16×** (fits a 700² grid where fp64 caps at
> 650²; in a dynamics-only stress fp64 OOMs at 1M columns where fp32 still fits).
> On the single RTX 5090 it is **NOT a speedup** — the fp32/fp64 throughput-ceiling
> ratio is **≈0.91 (≈1, not ≈2)** and there is **no peak-VRAM win on small
> single domains** (peak is radiation-transient-bounded below ~384²).
>
> **Honest scope of the fp32 fidelity check:** the fp32 tolerance bands are checked
> at the **1 h forecast lead** (19/19 fields green) — real, but **NOT stringent**
> (it sits 2–3 orders of magnitude inside the eventual 24 h skill bands). fp32 is
> only truly stressed by the **24–120 h skill gate, which is future work, out of
> v0.20 scope**. Do not read the 1 h pass as 24–120 h skill proof.
>
> *[MEASURED: FP32_INTEGRATION_REPORT.md §4.1/§4.2 — 963/963 byte-identical fp64; 19/19 fields 1h tolerance-green. V0200-STATE — −14.4% VRAM / 1.16× cells. INCONCLUSIVE on single-card speed: T2T3 R∞ ratio ≈0.91.]*

---

## WRF-v4 identity — proven cell-for-cell against CPU-WRF v4

`wrf_gpu` is validated by a **reproducible, CPU-only identity-proof system**: it
compares a GPU `wrfout` against a CPU-WRF `wrfout` from the same init, over **all
grid cells, all 72 forecast leads, and all core prognostic variables**, against a
**frozen tolerance manifest** (read before comparison, never tuned). This
**cell-identity** method is the project's primary fidelity gate — a per-cell,
per-lead, per-variable proof, not an aggregate station-RMSE summary. Full method +
reproduce commands: **[docs/IDENTITY_PROOF.md](docs/IDENTITY_PROOF.md)**.

The result is **9 of 10 hard-gate fields within frozen tolerance with the full
dynamics/thermodynamics core cell-for-cell identical** (`r ≈ 0.99–1.00`). The one
out-of-envelope field is a **bounded diagnostic, drawn red, never painted green**:
accumulated precipitation `RAINNC`, which stays inside a bounded multiple of a
tight 1.0 mm bound (see the framing below the plots).

**The v0.18 default Thompson microphysics is strictly more WRF-faithful than
v0.17.** v0.18 adds Thompson cold-process fidelity (rci/sci cloud-ice collection,
cloud-water freezing, graupel-number diagnostics) and, during the v0.18 release
gate, fixed a warm-process regression that the cold-process work had introduced:
WRF's sparse-graupel melt-intercept override (`module_mp_thompson.F:2802-2806`) is
now transcribed verbatim, and the rci/sci ice-collection family is gated on WRF's
cold block `T < T_0` (`module_mp_thompson.F:2554`). Against the WRF mass oracle the
warm-process `qr`/`qg` errors drop by ~3–4 orders of magnitude vs both v0.17 and
the intermediate trunk; cell-level `qv` error reaches **1.2×10⁻¹³ (bit-exact
WRF)**. (Proof objects retained in the v0.18 release history; the v0.18 Thompson
process-oracle closeout is summarized in the CHANGELOG v0.18 section.)

**Switzerland d01 — 72 h, v0.18 (9/10, dynamics/thermo cell-for-cell):**
![GPU↔CPU identity proof — Switzerland d01 72 h (v0.18)](docs/assets/v018/identity_proof/switzerland_d01/identity_dashboard.png)

This dashboard is built from the **retained v0.18 72 h GPU run**
(`v018_rainnc_qvapor_switzerland_d01_72h_qcfz_20260616T115735Z`, default
Thompson/RRTMG/MYNN/Noah, 72 hourly `wrfout` leads) paired cell-for-cell against
the retained CPU-WRF truth (`v014_switzerland_72h_cpu_20260610T122909Z`), scored
against the **frozen** tolerance manifest. 10 fields scored, **9 within tolerance**,
the single miss being `RAINNC` (5.22 mm vs the 1.0 mm bound). (The v0.18
identity manifest is retained in the v0.18 release history.)

**Canary L2 d02 — 8 h, nested v0.18 (10/10, all fields within frozen tolerance):**
![GPU↔CPU identity proof — Canary L2 d02 8 h (v0.18)](docs/assets/v018/identity_proof/canary_l2_d02/identity_dashboard.png)

> **Plot provenance — stated plainly.** The Switzerland dashboard above is
> regenerated from **v0.18** run data
> (`v018_rainnc_qvapor_switzerland_d01_72h_qcfz_20260616T115735Z`, 72 leads). The
> **Canary L2 d02** dashboard is regenerated from a **fresh v0.18 nested-Canary 8 h
> GPU run** (run id `v018_canary_d02_8h_gpu_20260617T081455Z`, d01→d02 one-way nest,
> init 2026-05-01 18Z) paired cell-for-cell against CPU-WRF from the same init,
> scored against the **frozen** tolerance manifest: **10/10 fields within frozen
> tolerance** (worst field QVAPOR at 0.57× its tolerance limit). Both the GPU and CPU
> `wrfout` for this Canary pair are on disk and re-scorable.

> **Reading the per-field correlations honestly (v0.20.1).** On these 72 h identity
> dashboards the near-surface fields are **comparable, and Switzerland is actually
> *better* than Canary on every surface field** (T2, U10, V10, W) — terrain gives the
> Alpine surface fields more spatial variance, which *keeps* Pearson `r` high. The
> low `r` that shows up on **accumulator and low-variance fields** (e.g. RAINNC, the
> near-flat 2 m potential temperature θ) is a **metric effect — low-variance Pearson
> collapse, not a solver defect**: with a near-identical *absolute* error, a nearly
> flat field scores a collapsed `r`. v0.20.1 therefore reports a variance-normalized
> error (**nRMSE = RMSE/field-std**) alongside `r`, so a flat field is not scored as
> broken. (Hash-verified two-analysis consolidation; closes #119.)

> **The one red field — RAINNC — in plain terms.** Nine of ten gate fields are
> within tolerance; the single miss is **RAINNC**, the *total accumulated
> precipitation* summed over the 72 h forecast. Its **physics is correct** — the
> individual rain / ice / snow microphysics processes match WRF to ~1e-7
> (oracle-green). What does not close to the tight 1.0 mm bound is the *accumulated
> total*: precipitation placement is the **most chaotically-sensitive field** in any
> weather model, so tiny, physically-legitimate differences (down to floating-point
> operation order) move a shower one grid cell over or a few minutes earlier, and
> summed over 72 h that grows to a few-mm cell-by-cell difference in the total —
> even though the water budget and the physics are right (WRF compared against
> *itself* on a different compiler / core count would likely also exceed a 1.0 mm
> accumulated-precip bound). **RAINNC is a derived diagnostic** (a running counter)
> that does **not** feed back into the forecast; the prognostic fields that drive
> forecast skill — wind, temperature, and moisture (**QVAPOR, which passes and even
> improved in v0.18**) — are all within tolerance. We draw RAINNC red and carry it
> honestly rather than widen the frozen tolerance to paint it green.

Reproduce against any matching CPU/GPU `wrfout` pair (CPU-only, never touches the GPU):

```bash
taskset -c 0-3 python3 scripts/build_identity_proof_plots.py \
  --cpu-dir "$CPU_DIR" --gpu-dir "$GPU_DIR" \
  --domain d01 --init "2023-01-15T00:00:00+00:00" \
  --case-id switzerland_d01_72h --region-label "Switzerland d01 72h (v0.18)" \
  --tolerance-json proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json \
  --proof-dir build/identity_proof/switzerland_d01 \
  --asset-dir docs/assets/v018/identity_proof/switzerland_d01
```

> **Framing — read this first.** `wrf_gpu` is a **WRF-compatible reimplementation**
> (a clean JAX rewrite validated against WRF as an oracle), **not a Fortran-source
> port**, and a **transparent research artifact, not a full WRF replacement.** The
> cell-identity proofs above show the **dynamics/thermodynamics core is
> cell-for-cell identical** to CPU-WRF v4 over 72 h; the broader **24 h/72 h
> forecast-skill equivalence (T2/U10/V10) vs CPU-WRF is the credibility gate and is
> NOT claimed closed** — it is a hard dynamics-`ph'` / MYNN / `*_tendf` GPU problem
> and the dominant carry-over (KI-9). The earlier statistical-equivalence (TOST)
> framing is **superseded** by this more precise per-cell identity proof; **no
> "TOST PASS" / "statistically-proven equivalence" is claimed.**

---

## Quickstart

A fresh clone → install → **standalone GPU forecast** → `wrfout` in three steps.
Full walk-through (prerequisites, troubleshooting, output): **[docs/quickstart.md](docs/quickstart.md)**.

> **Prerequisite environment variables.** The table-using schemes (Noah-MP and
> RRTM/RRTMG) load lookup tables from a pristine WRF v4 install at runtime, so two
> paths must be set before you run a case that selects them:
>
> | Variable | Required for | What it points to |
> | --- | --- | --- |
> | `GPUWRF_WRF_ROOT` | Noah-MP, RRTM/RRTMG | Root of a pristine WRF v4 source/run tree — provides the RRTM/RRTMG `.F` sources and the `run/` tables (including `run/CCN_ACTIVATE.BIN` for Thompson aerosol). |
> | `GPUWRF_CANAIRY_ROOT` | the validation cases | The `met_em` / run corpus root for the validation cases (run inputs + CPU-WRF reference). |
>
> Optional (performance / scratch, all defaulted): `GPUWRF_JAX_CACHE_DIR` — the
> persistent JIT-cache location (the standard JAX `JAX_COMPILATION_CACHE_DIR` is
> also honored as an alias) — and `GPUWRF_TMPDIR` (scratch root, default
> `~/.cache/gpuwrf`).
>
> Without `GPUWRF_WRF_ROOT` set, the table-loading schemes (Noah-MP, RRTM/RRTMG)
> **fail closed with a clear named error** before any compute — they never silently
> substitute a scheme. Cases whose physics menu does not use those schemes run
> without it.

```bash
# 1. Clone + install (CUDA 13 GPU build of JAX, then the package)
git clone https://github.com/wrf-gpu/wrf_gpu.git && cd wrf_gpu
python -m venv .venv && . .venv/bin/activate     # or: conda create -n wrfgpu python=3.11
pip install --upgrade "jax[cuda13]"
pip install -e .
python -c "import jax; print(jax.devices())"     # should list a cuda device

# 2. Run the BUNDLED Switzerland 3 km case — real GFS-initialized inputs that ship
#    in the repo at examples/switzerland_d01 (wrfinput_d01 + wrfbdy_d01 +
#    namelist.input; native-init, no CPU wrfout needed). Its physics (RRTMG
#    radiation + Noah-MP) read WRF tables, so point GPUWRF_WRF_ROOT at your pristine
#    WRF v4 tree first (see the prerequisite box above):
export GPUWRF_WRF_ROOT=/path/to/your/WRF        # your pristine WRF v4 source/run tree
python -m gpuwrf.cli run \
    --input-dir   examples/switzerland_d01 \
    --output-dir  runs/switzerland_d01 \
    --domain      d01 \
    --hours       1 \
    --scratch-dir /tmp/gpuwrf_scratch           # any real (non-tmpfs) fast disk

# 3. Read the WRF-compatible history file
ncdump -h runs/switzerland_d01/wrfout_d01_*
```

> The bundled `examples/switzerland_d01` inputs are derived from public-domain
> NCEP **GFS** analysis (2023-01-15 00Z, 42×42 @ 3 km, 44 levels) via WPS/`real.exe`
> — freely redistributable. They need **only** `GPUWRF_WRF_ROOT`;
> `GPUWRF_CANAIRY_ROOT` is for the larger Canary validation corpus, not this case.

`run` **auto-detects** the input directory: a case with a CPU-WRF `wrfout` →
replay mode; a case with only `real.exe` outputs → **standalone native-init mode**
(assembles `wrfinput`/`wrfbdy` and integrates on the GPU, **no CPU-WRF
dependency**). Bring your existing WRF `namelist.input` — the supported matrix runs
as-is; unsupported options fail closed with a named reason
([docs/namelist-compatibility.md](docs/namelist-compatibility.md)).

For the bundled **three-domain live-nested** example (d01→d02→d03, down to the
1 km nest; the accepted capability extends through d09), add
`--max-dom N` — the parent builds each child's lateral boundary **live**, with no
pre-supplied `wrfbdy_d02`:

```bash
python -m gpuwrf.cli run --input-dir my_case --output-dir runs/nested \
    --max-dom 3 --hours 24 --scratch-dir /fast/nvme/gpuwrf_scratch
```

> Remember the **one-time cold compile** (no output) on the first run; later runs
> read the persistent JIT cache. Compile time **scales with domain size**: the
> bundled single-domain d01 case compiles in roughly **½–2 min**, while a large
> nested case can take **~8–12 min**. See the box at the top.

### Run JUST the current version without the full repo (VERIFIED)

You do not need the whole repository (proofs, agent infrastructure, validation
corpora) to *run* `wrf_gpu`. A **cone sparse-checkout of just `src` + the vendored
runtime data tables** is enough — the working tree shrinks from the full repo to a
source-only install. This path was **verified end-to-end fresh** (clone → sparse →
`pip install -e .` → `import gpuwrf` → `python -m gpuwrf.cli run --help`, all
succeeding) on a clean machine.

```bash
# Shallow, no-checkout clone, then cone-sparse-checkout only what `run` needs.
git clone --depth 1 --no-checkout \
    https://github.com/wrf-gpu/wrf_gpu.git wrf_gpu && cd wrf_gpu
git sparse-checkout init --cone
# In cone mode the root files (pyproject.toml, README.md, LICENSE_NOTES.md) come
# automatically; add the source package and the vendored runtime data tables
# (the Thompson/RRTMG tables are loaded at import — `src` alone is not enough):
git sparse-checkout set src data/fixtures
git checkout

# Install (CPU import works without a GPU; add the CUDA jaxlib to run on GPU):
python -m venv .venv && . .venv/bin/activate
pip install -e .                  # CPU import-check; or:
pip install -e ".[cuda]"          # GPU execution (CUDA 13 jaxlib)

# Verify:
python -c "import gpuwrf; print('wrf_gpu', gpuwrf.__version__)"
python -m gpuwrf.cli run --help   # the CLI is the entrypoint
```

`pip install` builds and installs the `gpuwrf` package and its runtime
dependencies (`jax`, `numpy`, `netCDF4`, `xarray`, …). A plain `pip install -e .`
pulls in CPU `jax` so the **import and namelist-validation paths work without a
GPU**; GPU execution needs the CUDA jaxlib (`pip install "jax[cuda13]"` or the
`cuda` extra). The vendored `data/fixtures/` tables (~147 MiB: Thompson + RRTMG)
are required at import — that is the only large data the runtime itself needs.

## Use the manager (agent-driven development)

This repository is built to be run and extended by an **AI manager agent**. The
checked-in manager operating manual is included with the source tree. To drive
the project this way:

1. **Clone the repo on an isolated machine or VM** (the agent runs commands and the
   GPU; isolate it).
2. **Start Claude Code or a GPT/codex agent in auto-permission mode** in the repo
   directory.
3. **Tell it: "you are now the manager."** From that point the shipped
   manager operating manual tells it **where everything is and what to do** — read
   order, evidence/proof-object rules, how to dispatch and gate sub-agents, the GPU
   lock, and the release protocol.

The manager assigns sprints, runs the acceptance gates, and merges — you steer it
at the milestone/decision level, not per-command.

## Performance

Measured on the reference RTX 5090 workstation vs same-box CPU-WRF.

> **v0.23.4 status (prepared, unreleased): 2-domain fine, d03 configurations measurably slower.**
> Landing v0.23.4's V10 correctness fix (see [Version history](#version-history)) made 8-species
> moist/number scalar transport genuinely active on the nested GPU path, where it had previously
> been a silent no-op — expensive under `d03`'s per-root-step subcycling. A dispatch-level fix
> (size-thresholded `jax.vmap` species batching) closes this **for 2-domain (d01+d02)
> configurations: MEASURED, no regression (ratio 0.960–1.018× v0.23.3)**. It does **not** close it
> for **3-domain (`maxdom3`) or 9-domain configurations that engage `d03`**, which carry a real,
> production-measured **51.67% regression** (v0.23.3 1.9668543059 s/root-step vs candidate
> 2.9831176877 s/root-step) — a component profile found 79.95% of the added d03 cost is inside the
> FCT limiter's own flux-renormalization computation (not a dispatch/batching problem), and a
> Nsight Compute hardware measurement of that isolated kernel confirmed it is **occupancy/latency-
> bound, not memory-bandwidth-bound** (DRAM throughput 20.3% of peak, achieved occupancy 15.6%,
> register-limited to 2 concurrent blocks of 24 possible). **This is a known, understood limitation,
> not fixed in this release** — every quick-fix candidate tried (barrier, scan/map, register-
> pressure/fusion-disable, B1/B2/B3 scheduling, roll substitution) was falsified or rejected with
> evidence, and the hardware measurement rules out reduced-precision fixes while identifying
> concurrent sibling-domain scheduling as the mechanistically-mapped next lever — design-scoped, not
> implemented or validated, with no established delivery schedule. If your case is single-domain or
> 2-domain, this does not affect you. If it needs 3+ domains including d03, expect roughly 1.5× the
> v0.23.3 wall-clock. Full detail: [`RELEASE_NOTES_v0.23.4.md`](RELEASE_NOTES_v0.23.4.md).

> **What v0.21.0 changed for performance.** v0.21.0 improves **compile and
> warm-start time** (version-keyed cache + the AOT cheap-key warm-start that skips
> the re-lower — 9-nest gate: all 9 domains `loaded=true source=aot_blob`
> cross-process, 0 re-lower, load in seconds, MEASURED) and **stability** (the
> 9-nest is now finite through the divergence window). It does **not** change warm
> forecast throughput — the s/forecast-hour numbers below are the measured fused
> runtime numbers and remain current for the default path. De-fuse is available as
> an explicit low-host-compile-RAM fallback with a documented runtime cost.

- **v0.20.0 makes the all-7 nested fast path faster than v0.19 and CPU,
  byte-identically.** The default fused all-7-island, 9-domain run measures
  **~668 s/forecast-hour warm** (range 645–680) on the reference GPU versus
  **713 s/forecast-hour for v0.19** and the canonical **12-rank CPU-WRF baseline
  at 1020 s/forecast-hour** — **~1.07× faster than v0.19** and **~1.53× faster
  than CPU**. The output is **byte-identical to v0.19 (1926/1926 vars,
  maxΔ=0.000e+00)**: the gain comes entirely from a **numerics-free CUDA
  stream-ordered allocator** (`cuda_async`, now the default), which carries
  essentially the whole improvement (~+44 s/forecast-hour); the async-output and
  host-RAM-guard levers are stability/neutral by design. Peak VRAM is **122 MiB
  leaner** than v0.19 and stays flat over 12 output groups (1 km fit, no OOM).
  *Honest framing:* this nest is **host-bound** (~7–8% GPU duty) and the headline
  is a **range, not a point** — measured under live CPU-corpus contention with
  ~±2–3% run-to-run noise; the relative cuda_async win and all identity/fit/VRAM
  gates are robust regardless. The bigger structural multipliers (acoustic
  substep config, fused megakernel, fp32-relaxed tolerance) are an explicit
  **future wave, NOT in v0.20**.
  *[MEASURED: COMBINED_SPEEDUP.md §4–§8.]*
- **fp64 default is byte-identical and same-speed as v0.19.** The fp64_default
  GPU all-7 9-domain output is **963/963 vars maxΔ=0.000e+00, byte-identical**;
  warm fp64 forecast-only time is **HEAD ≈ baseline** (within noise). v0.20 adds
  no numerics risk on the default path.
  *[MEASURED: FP32_INTEGRATION_REPORT.md §4.1/§4.1b.]*
- **fp32 mixed-precision is opt-in, capability/VRAM-only.** It reduces whole-run
  VRAM **−14.4%** (aggressive) and fits **~1.16×** more cells; on the single card
  it is **NOT a speedup** (fp32/fp64 throughput-ceiling ratio **≈0.91**), and its
  tolerance is checked only at the **1 h lead** (not the 24–120 h skill gate).
  *[MEASURED on VRAM/capability; INCONCLUSIVE on single-card speed: V0200-STATE; T2T3 R∞≈0.91.]*
- **The compile cache just works out of the box, including across dates.** On by
  default, persistent per-user, zero config; warm after the first run; and **new
  in v0.20** a warm hit **across forecast dates** (0 new cache entries on a new or
  leap-year date), default path bit-identical.
  *[MEASURED: JULDAY_CACHE_FIX_REPORT.md.]*
- **Single-domain scaling is measured and honest.** On a tiny single-domain 129²
  grid the GPU is **~2.3× slower** than 24-rank CPU-WRF (host/launch-bound). It
  pulls ahead at 1 km / large / nested scale. A parametrized single-domain
  scaling study (Swiss base, tiled) shows throughput saturating at a ceiling of
  **~9.6e6 cells/s (fp32) / ~1.06e7 cells/s (fp64)** by ~384²; fp32 **fits 512²
  (11.5 M cells) where fp64 OOMs** (capability), with **no single-domain speed
  win** from fp32.
  *[MEASURED: T2T3_REPORT.md G-series + Swiss-CPU-match.]*
- **The single-card ceiling, restated.** The tiny-nest all-7 is
  **GPU launch/occupancy-bound at ~674 s/forecast-hour** — an nsys trace shows many ~1.5 µs
  kernels with no hot-spot (a launch/occupancy limit, not a throughput limit). So
  **fp32 cannot move it** and **≥2× / 3× are NOT single-card reachable** for this
  tiny-nest geometry. The genuine speedup/scale levers are **algorithmic +
  multi-GPU**, not fp32.
  *[MEASURED: HISTORICAL §CASE-1 (~674 floor); prior nsys, v0.17.]*

**HEADLINE = correctness + stability + capability + reliability, plus a modest
measured nest speedup.** **MEASURED:** 1 km single domain fits one RTX 5090
bit-identically; the all-7 1 km nested case runs end-to-end on one card;
v0.20.0 default fused mode is **~1.07× faster than v0.19 / ~1.53× faster than the
12-rank CPU-WRF baseline, byte-identical to v0.19**; fp32 opt-in cuts VRAM
**−14.4%** and fits **~1.16×** more cells; the cache hits across runs **and dates**
out of the box. **PROJECTED / UNMEASURED:** larger single grids and higher
throughput on bigger GPUs (H200 — measurable via the parametrized harness),
**cluster / multi-GPU weak-scaling** (e.g. one domain across a GB300-NVL72), and
whole-Earth-at-1 km "fits one rack" (exact memory arithmetic; an SPMD shard_map +
collective-halo foundation is bit-identity-validated on a fake/CPU mesh, but
**real multi-GPU throughput is not benchmarked — no perfect-scaling claim**).
Detail: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

### Apples-to-apples vs AceCAST (EXPECTATION / PROJECTED — not measured)

`wrf_gpu` is not the first GPU WRF effort — commercial directive-based ports
(**AceCAST**, OpenACC WRF) and other open efforts (`FahrenheitResearch`) exist.
**No head-to-head benchmark has been run:** there is no AceCAST wall-clock on our
cases, our GPU, or our precision regime, and AceCAST is fp32-dominant
directive-accelerated Fortran whereas our dynamical core is fp64 by design. On a
**like-for-like basis — same GPU class, same precision regime, same domain size,
same physics — we EXPECT to land in the same ballpark as hand-tuned-CUDA/OpenACC
ports like AceCAST**, because the dominant cost is the same memory-bound stencil +
column-physics work and our fp64 core is already device-bound / near-roofline. This
is an **EXPECTATION / PROJECTED positioning note, not an established competitive
claim** — we do **not** claim parity with, or an advantage over, AceCAST. Reasoning
and what would turn it into a measured claim:
[`proofs/v018/acecast_reconciliation.md`](proofs/v018/acecast_reconciliation.md).

**The whole Earth at 1 km fits in a single rack (PROJECTED).** The global 1 km
50-level state — ~25 billion cells, ~4.3 TB (≈13 TB with solver working memory) —
fits in the HBM of one **NVIDIA GB300 NVL72**. This is **exact memory arithmetic,
a "where this is going" note, not a near-term capability**: the multi-GPU
domain-decomposition path is bit-identity-proven on a **CPU fake mesh only**
(`shard_map` + `lax.ppermute` halo); **real multi-GPU throughput is not yet
shipped**, and a global wall-clock figure is **not claimed**.

**Performance / identity env flags**:
`GPUWRF_NESTED_FUSE=0` / `GPUWRF_NESTED_DEFUSE_COMPILE=1` (explicit eager/de-fuse
path; lower host compile RAM, slower runtime),
`GPUWRF_NESTED_PARALLEL_COMPILE=N` (opt-in parallel prewarm for de-fuse — faster
cold, more host RAM; `=0` opts out; unset is sequential, no spawn),
`GPUWRF_NESTED_AOT=0` (opt out of the AOT cheap-key warm-start; on by default),
`GPUWRF_AOT_VERIFY=1` (fail-closed warm-load HLO-verify backstop; default off),
`GPUWRF_JAX_CACHE` / `GPUWRF_CACHE` / `GPUWRF_JAX_CACHE_DIR` (version-keyed compile
cache controls),
`GPUWRF_XLA_AUTOTUNE_CACHE` (XLA autotune cache, default-on with the compile cache,
fail-open flag probing),
`GPUWRF_MIN_FREE_VRAM_GIB` (grid-scaled free-VRAM preflight floor),
`GPUWRF_BITWISE=1` (eager non-fused bitwise/debug path),
`GPUWRF_NESTED_SYNC_MODE` (`root` default / `advance` / `segment`),
`GPUWRF_EDGE_ONLY_BOUNDARY` (ring-only boundary, **default on, bit-identical**),
`GPUWRF_JIT_BOUNDARY` (jit the boundary builder, default off),
`GPUWRF_ALLOCATOR` (live-nested GPU allocator: `cuda_async` **default** — the
CUDA stream-ordered pool, pooled but fragmentation-free; `platform` — the
synchronous cudaMalloc/cudaFree fallback used before v0.20; `bfc` — the XLA
default arena. An explicit `XLA_PYTHON_CLIENT_ALLOCATOR` overrides this. Choice
is numerics-free — it changes only where device buffers live, not the math),
`GPUWRF_FINITE_CHECK` (default on; fail-fast NaN/Inf state check at chunk/output
boundaries, opt out with `0` only for explicit max-performance experiments),
`GPUWRF_HOST_LEDGER` (per-phase host-time diagnostic),
`GPUWRF_ACOUSTIC_PRECISION_MODE` (`fp64_default` **default**, byte-identical;
`mixed_perturb_fp32_v020` — opt-in perturbation-authoritative fp32 for VRAM /
capability, **not** a single-card speedup; see the fp32 subsection above).

## System requirements & resource profile

Measured on the reference RTX 5090. Full detail: **[docs/resource-profile.md](docs/resource-profile.md)**.

| Resource | What to expect |
|---|---|
| GPU / VRAM + host RAM | The v0.23.4 accepted one-hour **nine-domain** all-physics replay peaks at **16,624 MiB VRAM** and **35,773,432 kB host RSS** on the reference RTX 5090 workstation. For that exact workload, use a 32 GiB-class GPU and at least 48 GiB free host memory (64 GiB installed is the practical reference envelope); smaller cases may need much less. The older 1 km-NESTED all-island AC1_FIT case (9/3/1, d03 520x280x45, ~145k columns) fits at ~18.1 GiB peak VRAM. Peak is transient working memory, not persistent fp64 State; every different grid/physics mix still needs the fail-closed preflight. |
| First-run compile | **~½–12 min** one-time cold JIT compile for ordinary single-domain/nested programs (no output during compile, scales with grid size). The **persistent, version-keyed on-disk cache** (default on, zero config) turns later runs into a fast cache read (**cold ~147 s → cache-hit ~29 s** d01 hour-1 wrapper), **including across forecast dates** (v0.20: 0 new cache entries on a new/leap-year date); cached executable bit-identical. The all-7 9-domain fused nest is a separate large one-time compile; the **AOT cheap-key warm-start** then loads the fused cascade executable cross-process and **skips the re-lower** (fused AOT gate proof in `proofs/v021/canary_gate/V0210_FUSED_AOT_GATE.md`). De-fuse remains an opt-in low-host-RAM path with slower runtime. |
| Scratch | A **real (non-tmpfs) NVMe scratch dir**, a few GiB free. Set via `--scratch-dir` / `$GPUWRF_SCRATCH`. Do **not** use a RAM disk. |
| Throughput | **v0.23.4 candidate is blocked:** the production-faithful maxdom3 canary is **51.67% slower** than v0.23.3 (2.9831176877 vs 1.9668543059 s/root-step); 2-domain configurations are unaffected (0.960–1.018×). The accepted full one-hour nine-nest replay used 5,744.407 s warm model wall, but that correctness stress workload is not a matched benchmark. Historical v0.20 figures remain in [docs/PERFORMANCE.md](docs/PERFORMANCE.md). |
| Runtime data | The vendored `data/fixtures/` tables (~147 MiB: Thompson + RRTMG) are loaded at import; a minimal run install needs `src` + `data/fixtures` (see the source-only quickstart above). |
| Toolchain | CUDA 13 + a JAX CUDA build that sees the GPU. |

## Version history

Newest first. Full per-release evidence is under [`proofs/`](proofs/) and the
`RELEASE_NOTES_v*.md` files.

| Version | Headline | Key proof / link |
|---|---|---|
| **v0.23.4** *(prepared, pending publication)* | **Nine-nest correctness closure; d03 performance regression measured, root-caused, investigation closed.** All five nine-nest correctness acceptance criteria GREEN (V10 wake-displacement scalar-transport fix, Thompson late-Ni ice-sedimentation fix, full real nine-domain replay: 27/27 outputs exact, 177,497,511 values finite, zero tolerance failures), 3× independent Opus review. A dispatch-level scalar-batching perf fix is validated for **2-domain (d01+d02): MEASURED no regression (0.960–1.018×)**. **3-domain/9-domain configurations using d03 carry a real, measured 51.67% regression** vs v0.23.3 (79.95% of the added cost is inside the FCT flux-limiter's flux-renormalization computation). Every candidate fix tried (barrier, scan/map, register-pressure/fusion-disable, B1/B2/B3 scheduling, roll substitution) was falsified or rejected with evidence; a hardware Nsight Compute measurement then confirmed the kernel is **occupancy/latency-bound, not bandwidth-bound** (DRAM 20.3% of peak, occupancy 15.6%), ruling out reduced-precision fixes and identifying concurrent sibling-domain scheduling as the mechanistically-mapped next lever (design-scoped, not implemented or validated). This regression is understood down to the hardware root cause and ships as a known limitation, not fixed in this release. | [`RELEASE_NOTES_v0.23.4.md`](RELEASE_NOTES_v0.23.4.md), `.agent/decisions/VERSION-SPRINT-LEDGER.md` |
| **v0.23.3** | **cuda_async preflight false-negative fix; numerically identical to v0.23.2.** The nested-GPU VRAM preflight could see `cuda_async`'s own reserved pool as "used" and refuse an otherwise-empty card (`rc=75`). Under a verifiably-held `with_gpu_lock`, a low reading is now advisory, not fatal; the hard gate is unchanged with no lock held. Restores the faster `cuda_async` allocator for locked production runs. No dynamics/physics/numerics change. | [`RELEASE_NOTES_v0.23.3.md`](RELEASE_NOTES_v0.23.3.md) |
| **v0.23.2** | **Batched-ensemble distinct-init CLI ergonomics; numerically identical to v0.23.1.** `GPUWRF_BATCH_INPUT_DIRS` now accepts a **comma**-separated list of the B distinct-day input dirs (the old `:`-only separator still works), with an actionable count-mismatch error. Documents + verifies the distinct-init contract (nested one-way, exactly B same-geometry dirs; only the day differs). No dynamics/physics change; batched vmap math unchanged. New CPU unit tests. | [`RELEASE_NOTES_v0.23.2.md`](RELEASE_NOTES_v0.23.2.md) |
| **v0.23.1** | **Usability + AI-native onboarding; numerically identical to v0.23.0.** Full HTML [User's Guide](https://wrf-gpu.github.io/wrf_gpu/); an [AI operator skill](AI_OPERATOR.md) (`AI_OPERATOR.md` + a Claude Code skill) so an agent can run a user's case; backward-compatible WRF-parity CLI (`--hours`/`--domain` default from `namelist.input`; `--domains-from-namelist`; `gpuwrf namelist-support`; `--dry-run`; clearer `GPUWRF_WRF_ROOT` errors; effective-values in the run payload). No dynamics/physics change; CLI covered by CPU unit tests. | [`RELEASE_NOTES_v0.23.1.md`](RELEASE_NOTES_v0.23.1.md), [`docs/WRF_PARITY_ROADMAP.md`](docs/WRF_PARITY_ROADMAP.md) |
| **v0.22.2** | **Nested-grid wall-clock — host-bound GPU-idle reduction; default byte-identical.** Default-on byte-identical host-work cuts at the nested output boundary (removed a redundant full finite-summary, batched the ~70 device→host pulls + the finite guard, subset-aware payload build) measure **~9% faster nested steady-state** (789 vs 865 s/fc-h on a 384² 2-nest, output byte-identical to v0.22.1). A default-on **bit-identical** M9 RRTMG radiation column-tile cap (512 cols, output re-solve only; main forecast stays 1024) halves the per-output VRAM transient (~5.1 → ~2.6 GiB), fixing the 433² convection-peak OOM. Opt-in `GPUWRF_NEST_OUTPUT_PIPELINE` (async output overlap) / `GPUWRF_NESTED_M9_RADIATION_FROM_CARRY` (skip re-solve, lossy) / `GPUWRF_NEST_PERF_TIMERS`. No new physics; open 24–120 h skill gate carried. | [`RELEASE_NOTES_v0.22.2.md`](RELEASE_NOTES_v0.22.2.md) |
| **v0.22.1** | **Nested d02 cadence + opt-in colon-free wrfout names.** Point release on v0.22.0 for B200 pod output-path defects. Leaf children honor their own `history_interval` on eager and default fused paths; fused flat leaf subtrees fall back to the eager split path when a child cadence is not parent-ratio aligned. `GPUWRF_COLONFREE_OUTPUT=1` writes `HH-MM-SS` names for S3/network drains; default stays WRF-standard `HH:MM:SS`. No numerics, masking, clamp, or schema change. | [`RELEASE_NOTES_v0.22.1.md`](RELEASE_NOTES_v0.22.1.md), `proofs/v022/d02_cadence/REPORT.md` |
| **v0.22.0** | **Default-safe hygiene + opt-in K2 lever + ADR release.** Default forecast behavior stays v0.21.1. Adds #136 opt-in treedef/leaf-count AOT signature hardening, #115 `GPUWRF_WRF_ROOT` data-root portability, and #101 byte-identical async wrfout. Documents K2 `time_step` / `n_sound` as an **opt-in single-domain** CFL-gated lever (1.80x on Switzerland-128 short gate; no nested/default speed claim), corrects the compile-wall story (time is the wall; sound 3-domain peak RSS about 22.32 GiB; AOT about 39x), ratifies the operational-relaxed ADR §9, and carries K4 fp32 as a plan only. The corrected canary gate pairs v0.22 against a fresh matched v0.21.1 baseline (`9709039c... == 9709039c...`); the old `519cd3e5...` byte target was a stale cold-compile/autotune artifact. | [`RELEASE_NOTES_v0.22.0.md`](RELEASE_NOTES_v0.22.0.md), `proofs/v022/release_prep/V0220_CORRECTED_CANARY_GATE.md` |
| **v0.21.1** | **Mont-Blanc extreme-terrain boundary stability fix.** Point release off v0.21.0: restores standalone-root specified-boundary cadence and WRF-faithful physical-`W` zero-gradient boundary semantics. d01 max `|W|` bounded over the 2 h proof window, no masking / `nan_to_num` / finite guard / clamp. | [`RELEASE_NOTES_v0.21.1.md`](RELEASE_NOTES_v0.21.1.md), `proofs/v022/nest_dycore/MONTBLANC_W_BOUNDED_FIX_SUMMARY.md` |
| **v0.21.0** | **Stability + compile-cache speed; fused runtime default.** Dycore boundary fix (acoustic mass-drain limiter + positive `c2a`/`alt` floor, WRF-faithful, identity-preserving) makes the **all-7 9-domain Canary nest finite through the old step-67 divergence window** (MEASURED, all 9 domains). The nest default is **fused + AOT cheap-key warm-start**: a fresh process loads the serialized fused cascade executable cross-process and **skips the multi-tens-of-minutes re-lower**. De-fuse is an explicit low-host-compile-RAM fallback (`GPUWRF_NESTED_DEFUSE_COMPILE=1` / `GPUWRF_NESTED_FUSE=0`) with documented runtime cost; parallel de-fuse prewarm remains opt-in (`GPUWRF_NESTED_PARALLEL_COMPILE=N`). Also: default-on fail-fast finite guard (`GPUWRF_FINITE_CHECK`, first-bad `{domain,field,level,step,sim-time,index}`), an opt-in steep-terrain GPU gate, a version-keyed compile cache, and XLA autotune-cache default-on with fail-open probing. Full CPU suite A/B = **zero new failures** vs baseline. **Historical limitation:** the most-extreme 1 km Mont-Blanc (~1042 m/cell) case needed v0.21.1; long-horizon 9-nest GPU-VRAM limits are mitigated-not-eliminated. | `proofs/v021/canary_gate/V0210_FUSED_AOT_GATE.md`, [`RELEASE_NOTES_v0.21.0.md`](RELEASE_NOTES_v0.21.0.md) |
| **v0.20.2** | **Training-subset cloud-validation fields.** Output-only patch: the opt-in `MINIMAL_TRAINING_SET` gains OLR, RAINC, and SWDNB for satellite/cloud-validation workflows. Default full output is byte-identical; no HLO/compile/cache change. | [`RELEASE_NOTES_v0.20.2.md`](RELEASE_NOTES_v0.20.2.md) |
| **v0.20.1** | **Reliability + I/O readiness + honesty refresh; fp64 default path byte-identical.** Hardens the nested GPU path against OOM (#123, **two modes**): co-resident headroom **SOLVED** by a launch-time **fail-closed preflight** (card-relative free-VRAM gate exiting **before** the ~50-min compile; GPU-validated fail-closed + happy-path); solo `cuda_async` fragmentation **MITIGATED, not fixed** via the **bit-identical** RRTMG column-tile cap 2048→1024 (CPU `max_abs=0.0`; GPU-measured largest alloc **0.432→0.271 GiB, −37%**) + a shipped reproducer — **carried limitation: can still OOM the full fp64 nest; no OOM-proof / fp32-nest / 24 h large-nest claim**. Makes the **nested** compile cache hit across forecast dates (#114, `_DateClockAux` treedef date-invariance, **bit-identical, GPU-CONFIRMED**: cold DATE_A Δ≈6497 → warm different-date DATE_B Δ=2; **warm saves the ~50-min recompile but still pays ~38-min module load+link → ~30-min net, not "instant"**; CPU treedef tests 9/9). Adds **opt-in compact training output** (#122, `GPUWRF_TRAINING_OUTPUT_SUBSET`, 36-var zlib-lossless; **default byte-identical**; real-nest validated, CPU 10/10). Lands paid-B200 **I/O-readiness tooling** (manifest/WRF-dimension validator + block drain/resume/stop-pull, 22/22 CPU tests; **carried: real-S3 path not yet exercised**). Honesty refresh (source: S3 paper/identity honesty pass): **TH2 step-1 divergence is a metric artifact not a bug** (I1), nRMSE reported alongside `r` (I2), d01 mean-r coverage caveat (I3), and **#119 closed** — "Switzerland worse than Canary" is a presentation/metric artifact (Swiss surface fields comparable/better; no model bug). | [`RELEASE_NOTES_v0.20.1.md`](RELEASE_NOTES_v0.20.1.md), [`proofs/v013/rrtmg_column_tile.json`](proofs/v013/rrtmg_column_tile.json) |
| **v0.20.0** | **Correctness + stability + capability + reliability; modest measured nest speedup.** Default fused all-7 9-domain nest is **~1.07× faster than v0.19 / ~1.53× faster than 12-rank CPU-WRF (~668 vs 713 vs 1020 s/forecast-hour), byte-identical to v0.19 (1926/1926, maxΔ=0)** — gain from a numerics-free `cuda_async` allocator (now default). Adds **opt-in fp32 mixed-precision** (`mixed_perturb_fp32_v020`) for **−14.4% VRAM / ~1.16× cell capability** (fp64 default byte-identical, 963/963 maxΔ=0; fp32 is **not** a single-card speedup, tolerance checked at 1 h only). **Compile cache now hits across forecast dates** (#91 — 0 new cache entries on new/leap dates, default path 64/64 byte-identical), zero config. GPU-vs-CPU all-7 24 h identity: **core EXCELLENT — T corr 0.9999 (RMSE 0.69 K), PH/PSFC 0.9997, U 0.991, V 0.968, QVAPOR 0.964; surface diagnostics looser (most parameterization-sensitive) — T2 0.944 (RMSE 0.78 K), TH2 0.878, U10 0.855, V10 0.852 mean corr; on the inner 1 km Alpine nests d06/d07 the 10 m winds spread to corr ~0.48–0.65 / RMSE ~4–5 m/s** — the expected most-sensitive field on complex terrain, **byte-identical to validated v0.19 (not a v0.20 regression)**, divergence grows with lead time. The **TH2 part of #119 is resolved in v0.20.1 as a metric artifact, not a bug** (recomputed TH2 reproduces the stored field to ~10⁻³ K on both runs and its absolute error equals T2's ~0.78 K — low-variance Pearson collapse over the shared near-surface T2 offset; the θ field's ~4× smaller spread collapses `r` at near-identical absolute error); identity now also reports **nRMSE = RMSE/field-std alongside `r`** so a near-flat field is not scored as broken, and the **d01 "mean r over domains" carries a coverage caveat** (d01 = only 2 short leads ≤0.67 h, inflating the surface-wind mean by ~+0.018). The 10 m wind part is genuine lead-amplified chaos (paper-Q5). | `proofs/v020/lowhang/COMBINED_SPEEDUP.md`, `proofs/v020/fp32_integration/FP32_INTEGRATION_REPORT.md`, `proofs/v020/julday_cache/JULDAY_CACHE_FIX_REPORT.md`, `proofs/v020/benchmark/T2T3_REPORT.md`, `proofs/v020/validation/identity/` |
| **v0.19.0** | **Fast all-7 nested fusion + terrain-blend fidelity.** Default fused nesting plus the restored fast `_advance_chunk` loop body makes the all-7-island `max_dom=9` case **1.43x faster than the 12-rank CPU-WRF baseline** (713 vs 1020 s/forecast-hour; best segment 683). The one-time fused compile remains large (historical v0.19: ~41 min first
segment, cached; v0.21 fused AOT warm-start supersedes this). The live-nest terrain/base-state fix closes the HGT/MUB/PB/PHB red-field class; all 9 domains write finite `wrfout` and the established grid comparator reports 102 fields/domain, 0 tolerance failures. | [`proofs/v019/release_prep/gate_summary.json`](proofs/v019/release_prep/gate_summary.json), [`proofs/v019/release_prep/grid_compare_summary.json`](proofs/v019/release_prep/grid_compare_summary.json), [`RELEASE_NOTES_v0.19.0.md`](RELEASE_NOTES_v0.19.0.md) |
| **v0.18.3** | **max_dom=9 compile fix + nested `history_interval` cadence fix, bit-identical.** The all-7-island `--max-dom 9` nest compiled forever (`jit__advance_chunk` constant-folding static `s64[nz]` Thompson scan-index arrays across 9 domain shapes) → now all 9 domain-shape compiles complete **bounded** (≤409 s cold / ≤22 s warm), integrate (~85 % util), and write output. Also fixes the nested pipeline ignoring the namelist `history_interval` (was hardcoded hourly). Default numerics bit-identical (26/26 `wrfout` exact, `max_abs_diff 0.0`); hourly gates unchanged. | [`proofs/v018/maxdom9_fix/report.md`](proofs/v018/maxdom9_fix/report.md), [`RELEASE_NOTES_v0.18.3.md`](RELEASE_NOTES_v0.18.3.md) |
| **v0.18.2** | **1 km nested VRAM-efficiency fix, bit-identical.** The AC1_FIT 9/3/1 all-island nested case now fits the reference RTX 5090 (**OOM near 31.8/32 GiB → 18.1 GiB peak**) via radiation column-tile defaults 16384→2048 plus tiled MYNN cold-start. Default numerics unchanged: 26/26 `wrfout` fields exact, MYNN cold-start `qke`/`pblh` diffs 0.0. Warm steady-state utilization is ~85–88%; full-run aggregate is lower because it includes the one-time load/cold-JIT prefix. Restores Thompson aero+cold runtime fixture tables. | [`proofs/v018/oom_fix/fix_report.md`](proofs/v018/oom_fix/fix_report.md), [`RELEASE_NOTES_v0.18.2.md`](RELEASE_NOTES_v0.18.2.md) |
| **v0.18.0** | **FEATURE-COMPLETENESS + scheme triage.** Classifies and handles **every WRF v4 namelist scheme**: **50 operational** / **23 reference-only-with-real-oracle** / **33 documented-boundary or proven-irrelevant** (State = 67 leaves; no scheme/leaf dropped). Default **Thompson microphysics is strictly more WRF-faithful than v0.17** (cold-process additions + a warm-process melt/cold-gate fix → cell `qv` bit-exact WRF). **Perf-neutral vs v0.17** (default case, dual-confirmed). Adds **experimental, default-OFF K2 multi-GPU** domain decomposition (periodic-BC bit-exact; specified-BC not yet faithful — lab-only). | [`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md), git tag `v0.18.0` |
| **v0.17.0** | **PERFORMANCE + ceiling.** Closes the live-nested GPU host-orchestration holes — the **all-7 island nest (`--max-dom 9`) now forecasts at all** (previously recompiled forever → 0 output); default config **bit-identical to v0.16**. Adds an **opt-in fused fast-mode** (`GPUWRF_NESTED_FUSE=1`: util 56→96 %, **~1.27–1.30× vs 12-rank CPU**, tolerance-PASS not bitwise, ~38 min one-time compile). Answers speedup plainly: tiny-nest all-7 is **launch/occupancy-bound (~674 s/hr, nsys-grounded)** — **fp32 cannot move it, ≥2×/3× not single-card reachable**. Value = **capability** (1 km fits one card + scale), not single-card tiny-nest speed. | git tag `v0.17.0` |
| **v0.16.0** | **STABILITY + 1 km-unlock.** Proves **24 of 25 L2 physics schemes run coupled-green** on a real Switzerland d01 case (25th = Noah-classic, scope-carry → `ALL_GREEN_OR_CARRIED`). Adds **aerosol-aware Thompson** (`mp_physics=28`, WRF-module oracle PASS). Ships a **chunked MYNN BouLac** that makes a **1 km single domain fit one RTX 5090 bit-identically** (dense OOMs at ≈18.8 GiB; chunked fits at 18.25 GiB). **fp32 make-or-break CONCLUDED** (Opus + independent GPT): valid-numerics ceiling **~1.1×**, 0 % VRAM-peak reduction. | git tag `v0.16.0` |
| v0.15.0 | **Final fp64 kernel + WRF-fidelity.** Delivers the project's **final fp64 GPU kernel** (adversarially confirmed near-optimal, device-bound). Lands **MYNN-EDMF condensation `niter` 50→16** + **Thompson cold-collection**, fixes the **MUB/PB nest-base-state seam** (250.7 → 0.0078 Pa), re-closes both 72 h gates **9/10 within frozen tolerance**, dynamics/thermo cell-for-cell. **~parity total-wall** (0.99×/1.04×). | [`proofs/v015/finalgates/`](proofs/v015/finalgates/), [`proofs/perf/v015/kernel_characterization.md`](proofs/perf/v015/kernel_characterization.md) |
| v0.14.0 | Memory + WRF-identity: root-causes Switzerland venting (stratospheric-theta masking clamp), lands advance_w WRF-faithfulness + physics-`tendf` fold + 2D Smagorinsky on the default path, and **first closes both 72 h GPU-vs-CPU field-parity gates** with the reproducible identity-proof system. | [`proofs/v014/`](proofs/v014/) |
| v0.13.0 | Lifts the single-GPU VRAM ceiling (**RRTMG VRAM-floor chunking**, SW −88.6 % / LW −43.6 %), turns **GWD on by default on the nested 1 km path**, adds **MYJ+Janjic**, multi-GPU fake-mesh sharding, moisture flux-advection into RK3, clear-sky diagnostics (all opt-in/default-off). | [`proofs/v013/`](proofs/v013/), [`proofs/v0130/`](proofs/v0130/) |
| v0.12.0 | Standalone out-of-box CLI + live-nested `--max-dom`, **persistent JIT cache**, fail-closed scheme catalog, WRF-faithful PSFC fix, runnable equivalence demo. | [`proofs/v0120/`](proofs/v0120/) |
| v0.11.0 | Live multi-domain nesting, WRF restart bit-identity, conservation budgets closed, MYNN-EDMF, topographic/slope radiation, terrain-slope diffusion, Kain-Fritsch/BMJ/Tiedtke/Grell-Freitas cumulus. | [`proofs/v0110/`](proofs/v0110/) |
| v0.9.0–v0.10.0 | Consolidated standalone forecast system; removed a faithful Thompson sedimentation inefficiency. | [`proofs/v090/`](proofs/v090/), [`proofs/v0100/`](proofs/v0100/) |
| v0.1.0–v0.6.0 | Single-domain replay → native metgrid (v0.3.0) → native real-init proven equivalent to `real.exe` at t=0 (v0.4.0) → expanded operational physics menu (v0.6.0). | git tag history |
| v0.2.0 | Intended stable paper-claims baseline (accessible via git tag; never formally re-tagged). | git tag `v0.2.0` |

## Scope at a glance — implemented / fail-closed / out-of-scope

A high-level summary of what runs, what is recognized-but-refused (loudly, before
any compute), and what is a deliberate boundary. Full per-scheme support table:
**[docs/namelist-compatibility.md](docs/namelist-compatibility.md)**; open issues:
**[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**.

| Area | Implemented (runs) | Fail-closed (recognized, refused with a named reason) | Out-of-scope / roadmap boundary |
|---|---|---|---|
| **Init** | Native real-init (`wrfinput`/`wrfbdy` from met_em, no `real.exe`); WRF restart | — | — |
| **Dynamics** | Nonhydrostatic ARW, RK3 + split-explicit acoustic, flux-form advection, constant-K (`diff_opt=2`/`km_opt=1`) + 2-D Smagorinsky (`diff_opt=1`/`km_opt=4`) horizontal diffusion | 3-D TKE / full Smagorinsky (`km_opt=2/3/5`) → use `km_opt=1` or `4` | Global/periodic grids; opt-in moving/adaptive drivers exist but still need end-to-end operational validation |
| **Microphysics** | Kessler, Purdue-Lin, WSM3/5/6/7, Thompson, **aerosol-aware Thompson (mp=28)**, Morrison, SBU-YLin, WDM5/6/7, Goddard GCE | Aerosol-coupled Morrison (mp=40), NSSL, and the rest of the WRF MP tail (recognized, real-oracle or documented-boundary) | WRF-Chem |
| **PBL / sfc** | YSU, MYJ, MYNN-EDMF, ACM2, BouLac, GFS, GBM-TKE, MRF, **Shin-Hong (operational, TKE-diagnostic follow-up)**; MYNN-SL, revised-MM5, Pleim-Xiu, Janjic-Eta, NCEP-GFS sfclay | CAM-UW (`bl=9`); reference-only PBL tail (real oracle) | — |
| **Cumulus** | Kain-Fritsch, BMJ, Tiedtke (needs active flux-form moisture advection for RQVFTEN), Grell-Freitas (scale-aware), New-Tiedtke (`cu=16`) | SAS/Grell-family reference-only and documented-boundary CU tail (real oracle or named reason) | — |
| **Radiation** | RRTMG SW + LW; Dudhia SW + classic RRTM LW (`ra_lw=1`); clear-sky `…C` flux diagnostics (opt-in). Topographic shading/slope correction are available standalone; nested execution currently disables them and fixes cadence at 30 minutes. | Reference-only RA tail (real oracle); `ra_*={14,24}` compiled-out (BUILD-gated, like WRF) | — |
| **Land** | Noah classic, Noah-MP (prognostic), Pleim-Xiu LSM, thermal-diffusion slab | RUC LSM (reference-only, real oracle staged); **CLM4 (`sf_surface_physics=5`) / CTSM (`6`) — documented architecture boundary, fail-closed (no oracle claimed)** | Full Noah-MP snow-layer diagnostics in wrfout (KI-3) |
| **Nesting** | One-way live d01→…→d09 on the accepted one-hour fixture, per-domain subcycling, restart; GWD (`gwd_opt=1`) default-on on nested | Arbitrary nested `radt` is not honored; nested topo/slope radiation binds disabled | Two-way 24 h equivalence and broader long-horizon/configuration coverage remain open (KI-11) |
| **Output** | Focused 104-variable `wrfout` (core met/spatial/vertical/soil + radiation-flux + Noah-MP snow-layer) | — | Full 375-variable wrfout; auxhist streams (KI-3) |
| **Multi-GPU** | `shard_map` + `lax.ppermute` halo sharding, single-GPU default = zero overhead; **experimental K2 domain-decomposition (default-OFF, periodic-BC only)** | — | Real multi-GPU throughput (needs DGX/NVLink; fake-mesh bit-identical only); K2 specified-BC not yet faithful |
| **Data assim.** | Lateral-BC relaxation | — | DFI, FDDA, grid/obs/spectral nudging |
| **Other** | — | — | Urban (BEP/BEM), lake, fully aerosol-coupled MP / WRF-Chem (rejected, not roadmap) |

These are **boundaries and a roadmap, not hidden gaps**: every unsupported
namelist selection is rejected before any compute with a specific named reason —
the port never silently substitutes or skips a scheme. The full per-scheme
classification is in the [Scheme triage](#scheme-triage--every-wrf-v4-scheme-classified)
below.

### GPU-operational physics menu (scan-wired, WRF-oracle-gated)

These are the schemes the operational scan actually dispatches; the wiring is in
[`src/gpuwrf/runtime/operational_mode.py`](src/gpuwrf/runtime/operational_mode.py)
(`_SCAN_WIRED_OPTIONS`) and
[`src/gpuwrf/coupling/scan_adapters.py`](src/gpuwrf/coupling/scan_adapters.py); the
namelist-accepted matrix is in
[`src/gpuwrf/contracts/physics_registry.py`](src/gpuwrf/contracts/physics_registry.py).

| Family | Namelist key | GPU-operational options (scan-wired) |
|---|---|---|
| Microphysics | `mp_physics` | 0 passive, 1 Kessler, 2 Purdue-Lin, 3 WSM3, 4 WSM5, 6 WSM6, 8 Thompson, 10 Morrison, 13 SBU-YLin, 14 WDM5, 16 WDM6, 24 WSM7, 26 WDM7, **28 aerosol-aware Thompson** (QNWFA/QNIFA prognostics; WRF-module oracle PASS), 97 Goddard GCE |
| PBL | `bl_pbl_physics` | 1 YSU, 2 MYJ (mandatory Janjic pairing), 3 GFS, 5 MYNN-EDMF (DMP mass flux + cloud-aware moisture/thermodynamics), 7 ACM2, 8 BouLac, 11 Shin-Hong, 12 GBM-TKE, 99 MRF |
| Surface layer | `sf_sfclay_physics` | 1 revised-MM5, 2 Janjic-Eta (paired with MYJ), 3 NCEP-GFS, 5 MYNN-SL, 7 Pleim-Xiu, 91 old-MM5 |
| Cumulus | `cu_physics` | 1 Kain-Fritsch, 2 BMJ (fp64), 3 Grell-Freitas (scale-aware), 6 Tiedtke (needs flux-form moisture advection for RQVFTEN), 16 New-Tiedtke |
| Radiation | `ra_sw_physics` / `ra_lw_physics` | RRTMG SW + LW (`=4`); Dudhia SW (`ra_sw=1`) + classic RRTM LW (`ra_lw=1`); Held-Suarez idealized radiation (`ra_lw=31`); clear-sky `…C` flux diagnostics (opt-in). Topo/slope radiation is supported standalone but currently disabled on the nested runtime, whose cadence is fixed at 30 minutes. |
| Land surface | `sf_surface_physics` | 1 thermal-diffusion slab, 2 Noah classic (explicit static/land bundle), 4 Noah-MP (`use_noahmp=True`), 7 Pleim-Xiu LSM |
| Diffusion | `diff_opt`, `km_opt` | constant-K and 2-D Smagorinsky (incl. terrain-slope + map-factor deformation terms; WRF formula parity, max residual `3.78e-15`) |
| GWD | `gwd_opt` | 1 gravity-wave drag — **default-ON on the nested 1 km path** (`GPUWRF_GWD_NESTED=0` forces off) |
| Advection | `moist_adv_opt`, `scalar_adv_opt` | Moisture flux-advection into RK3 + PD/monotonic moisture limiter (both opt-in, default-off = byte-identical). The accepted nested path honors these values; the single-domain daily pipeline currently drops requested values and runs `0/0`. |

`mp_physics=0`, `bl_pbl_physics=0`, `sf_sfclay_physics=0`, `cu_physics=0`, and
`ra_*=0` are accepted as "disabled" slots.

### Scheme triage — every WRF v4 scheme classified

The current registry classifies every WRF v4 namelist scheme into one of three
buckets, with no scheme silently dropped (State = **67 leaves**; set-union and
dispatch consistency are machine-checked):

| Class | Count | Meaning |
|---|---|---|
| **Operational** | **51** | Scan-wired into the GPU forecast loop, WRF-oracle-gated. (mp 15, cu 6, bl 10, sfclay 7, sf_surface 5, ra_lw 4, ra_sw 4) |
| **Reference-only-with-real-oracle** | **25** | A real WRF v4 scheme with a staged oracle, **not** scan-wired; operational selection fails closed with a named reason. (mp 2, cu 8, bl 5, sf_surface 2, ra_lw 4, ra_sw 4) |
| **Recognized fail-closed** | **32** | Valid WRF codes not yet implemented; refused by name rather than substituted. (mp 21, cu 3, sfclay 2, sf_surface 2, ra_lw 2, ra_sw 2) |

The live counts and complete per-code table are maintained in
[`docs/PORT_COMPLETION_ROADMAP.md`](docs/PORT_COMPLETION_ROADMAP.md) and
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md).

## Boundaries — what is NOT claimed

- **v0.23.4 d03 performance regression is real and NOT yet resolved (prepared/unreleased).**
  3-domain (`maxdom3`) and 9-domain configurations that engage `d03` measure a real **51.67%**
  slower wall-clock than v0.23.3. The investigation is closed: every quick-fix candidate was
  falsified or rejected with evidence, and a Nsight Compute hardware measurement confirmed the
  isolated dominant kernel is **occupancy/latency-bound, not memory-bandwidth-bound** (DRAM 20.3%
  of peak, occupancy 15.6%) — this rules out a reduced-precision fix and identifies concurrent
  sibling-domain scheduling as the mechanistically-mapped next lever (design-scoped, not yet
  implemented). 2-domain (d01+d02) configurations are unaffected (measured 0.960–1.018×). This is
  a known, understood, unfixed limitation, not an active open question — see
  [Performance](#performance) and [`RELEASE_NOTES_v0.23.4.md`](RELEASE_NOTES_v0.23.4.md).
- **Not a universal WRF v4.** Standard regional ARW configs only; the common
  operational subset above. Every other scheme is classified
  (reference-only-with-oracle or documented-boundary) and fails closed with a named
  reason.
- **24 h/72 h forecast-skill equivalence is NOT closed — the credibility gate.**
  On the runnable equivalence demo (24 h d02), the verdict is `NOT_EQUIVALENT`:
  short-lead fields track CPU-WRF within tolerance, but by 24 h the run diverges,
  **dominated by lead-time wind divergence** (3D V pooled RMSE 8.13 m s⁻¹ vs a
  1.8 m s⁻¹ bar). PSFC is improved (707.8 → 415.3 Pa) but still out of bar, its
  residual driven by that same dynamical divergence. **Neither the winds nor PSFC
  are equivalent at 24 h.** Off-by-default fidelity levers (moisture flux-advection
  into RK3, MYJ+Janjic, clear-sky diagnostics) move toward this gap but do **not**
  close it — hard dynamics-`ph'` / MYNN / `*_tendf` GPU work, no cheap knob. This
  is the gate for any "operational / replacement" claim. See
  [docs/equivalence-demo.md](docs/equivalence-demo.md) (KI-9).
- **Cell-identity proof passes 9/10 with one bounded miss.** The Switzerland 72 h
  cell-identity proof closes with **9/10 hard-gate fields within frozen tolerance**
  and the dynamics/thermo core cell-for-cell identical; the one out-of-envelope
  field is accumulated `RAINNC` (**5.22 mm vs the 1.0 mm bound**, class-c). This is a
  derived accumulated-precip diagnostic with no expected forecast-skill impact,
  drawn **red** in the dashboard, **not** an identity failure; the frozen limit is
  unchanged (no goalpost moving, no tolerance widening).
- **K2 multi-GPU is EXPERIMENTAL and default-OFF.** The K2 domain-decomposition
  path (`GPUWRF_K2_EXPERIMENTAL=1`) is **lab-tested only**: with the gate unset the
  default single-GPU graph is **bit-identical** (no collectives emitted). With
  the gate set, the **periodic-BC**
  decomposition reproduces the single-GPU reference **bit-for-bit at roundoff** on
  interior + internal shard seams — but the **physical (specified) boundary is NOT
  yet faithful** (periodic vs WRF specified BC diverge by design at the true domain
  edge; the boundary ring is *excluded* from the pass gate, not hidden behind a
  loosened tolerance). Do **not** enable K2 specified-BC multi-GPU for production.
- **No statistical-equivalence (TOST) claim.** The cell-identity proof above
  **supersedes** the earlier TOST framing as the primary fidelity gate. The
  station-RMSE TOST campaign is underpowered at the available corpus (n=15;
  n≈27 for full power) and is **not run / not claimed**; deferred (KI-5).
- **Single-card speedup is measured but modest and scale-dependent.** v0.20.0
  default fused all-7 nesting is **~1.07× faster than v0.19 / ~1.53× faster than
  the same-box 12-rank CPU-WRF baseline** and **byte-identical to v0.19**, but on
  tiny single-domain geometries the GPU is **host/launch-bound and slower than
  CPU** (~2.3× slower at 129²). The tiny-nest all-7 is compute-bound at **~674
  s/forecast-hour**: **≥2× / 3× are NOT single-card reachable** and **fp32 cannot
  move it** (fp32 single-card speed ratio ≈0.91, INCONCLUSIVE-to-none). The
  remaining scale levers are **algorithmic + multi-GPU**, not broad fp32.
- **Multi-GPU throughput is PROJECTED, not measured.** An SPMD
  `shard_map` + collective-halo (`lax.ppermute`) foundation exists and is
  **bit-identity-validated on a fake/CPU mesh**, and the parametrized scaling
  harness lifts 1:1 to bigger GPUs — but this workstation has **one physical RTX
  5090**. Real multi-GPU throughput (e.g. one domain across a GB300-NVL72),
  NVLink/NCCL bandwidth, and collective overlap are **UNMEASURED**; the
  whole-Earth memory note stays **PROJECTED**. We do **not** claim it scales
  perfectly on GB300, and **no per-watt / per-kWh claim is made**.
- **Shin-Hong PBL (`bl_pbl_physics=11`) is operational with a TKE-diagnostic
  follow-up.** It is scan-wired and operational despite a ~28.5 % residual in the
  diagnostic TKE field, which was source-traced as **non-driving** (the dynamics
  tendencies never read it); the TKE-oracle upgrade is a documented follow-up, not
  a masked failure.
- **Not full two-way nesting.** One-way live nesting is proven over a 24–72 h
  window; the two-way feedback path is finite/stable but its 24 h real-GPU
  equivalence vs CPU-WRF is **untested** (KI-11).
- **fp32 is opt-in, capability/VRAM-only.** The standalone default path runs pure
  fp64 (byte-identical). The opt-in `mixed_perturb_fp32_v020` mode buys VRAM
  (−14.4%) and cell capability (~1.16×), **not** single-card speed (ratio ≈0.91),
  and its fidelity is verified only at the 1 h lead (24–120 h skill is future
  work).
- **Free-running open-lateral-boundary stability.** Free-running without
  lateral-boundary relaxation on wide domains (nx≈160+) can go unstable beyond
  ~14 h. The validated operational path uses boundary forcing (KI-7).
- **Apples-to-apples vs AceCAST is an EXPECTATION, not a benchmark.** No
  head-to-head run exists; see the [Performance](#apples-to-apples-vs-acecast-expectation--projected--not-measured)
  note. No competitive claim is made.
- **Not** DFI / FDDA / spectral-nudging / adaptive-Δt; **aerosol-coupled Morrison
  (`mp=40`) and NSSL fail closed**; **not urban (BEP/BEM) / lake / WRF-Chem /
  WRF-Fire / WRF-Hydro** (rejected, not roadmap).
- **v0.2.0 paper tag not formally re-released.** All prior releases remain
  accessible via git tags on the org repo; v0.2.0 stays accessible for paper claims.

A code-grounded, prioritized inventory of the remaining gap to a complete WRF v4
replacement lives in
[`docs/GPU_PORT_GAPS_TODO.md`](docs/GPU_PORT_GAPS_TODO.md) and the roadmap table
below.

## Roadmap — remaining work toward a complete WRF v4 port

> 🗺️ **Full port-completion roadmap:** every scheme/feature still reference-only,
> fail-closed, or out-of-scope — the 25 reference-only + 32 fail-closed physics codes, the
> 3-D turbulence closures (`km_opt=2/3/5` = 3-D TKE / 3-D Smagorinsky / SMS-3DTKE), data
> assimilation, coupled models, and output/grid gaps — is enumerated with a per-item "how to
> close it" in [`docs/PORT_COMPLETION_ROADMAP.md`](docs/PORT_COMPLETION_ROADMAP.md). Its
> explicit target is to drive the reference-only and fail-closed columns to zero. Regenerate the
> live status with `python -m gpuwrf.cli namelist-support`.

v0.18 is **feature-complete on scheme classification** — every WRF v4 namelist
scheme is operational, reference-only-with-oracle, or documented-boundary. What
remains is **fidelity, robustness, statistical closure, and performance/scale**, not
"missing schemes." Consolidated, prioritized ledger, sorted by importance for an
*optimal complete* port. Complexity: **S** ≈ 1–2 focused sprints · **M** ≈ 3–5 ·
**L** ≈ 5–10 · **XL** ≈ 10+.

| # | Item — remaining delta vs official WRF v4 | Cmplx | Detail |
|---|---|---|---|
| **Tier 1 — fidelity (blocks an operational replacement claim)** | | | |
| 1 | **24 h/72 h forecast-skill closure (T2/U10/V10)** — the credibility gate; cell-identity proven, broad skill-equivalence open. Hard dynamics-`ph'`/MYNN/`*_tendf` work. | L | KI-9; docs/equivalence-demo.md |
| 2 | **RAINNC bounded accumulated-precip residual** — 5.22 mm RMSE vs 1.0 mm bound (class-c, no skill impact expected); diffuse Thompson staging + coupled accumulated-precip propagation, no single bounded missing process. | M | KI-9 (RAINNC) |
| 3 | **MYNN PBL completeness** — EDMF mass flux wired; `icloud_bl=1` cloud PDF and `cloudmix` partial. Tied to the residual near-surface wind-skill gap. | M | GPU_PORT_GAPS P1-4 |
| 4 | **Shin-Hong PBL TKE-diagnostic** — operational; diagnostic TKE field ~28.5 % residual (non-driving, source-traced); oracle upgrade follow-up. | S | GPU_PORT_GAPS (PBL TKE) |
| 5 | **Moisture advection into RK3 + cadence fidelity** — wired opt-in (default-off); cadence refinements + operationalizing on the default path remain. | M | GPU_PORT_GAPS P1-6; KI-10 |
| 6 | **RRTMG SW taug top-layer convention fix** — 4 UV bands fail intermediate oracle; tier-1 fluxes faithful; pre-existing. | S | KI-6 |
| **Tier 2 — nesting / output completeness** | | | |
| 7 | **Broader multi-domain nested equivalence** — v0.23.4 closes one frozen one-hour d01-d09 fixture; longer horizons/configurations, arbitrary nested `radt`, and two-way 24 h real-GPU equivalence remain open. | L | GPU_PORT_GAPS P0-1; KI-11 |
| 8 | **Full `wrfout` variable coverage** — focused 104-variable writer vs WRF's 375. Blocks downstream tools. | M | GPU_PORT_GAPS P0-5; KI-3 |
| **Tier 3 — correctness / robustness debts** | | | |
| 9 | **Free-running open-lateral-boundary stability** — wide domains (nx≈160+) can blow up without boundary relaxation beyond ~14 h. | M | KI-7 |
| 10 | **U10 episodic under-prediction** — final-lead breach on the validated d02 case (tied to MYNN cloud PDF). | S–M | KI-4 |
| 11 | **CLM4/CTSM land-surface** — documented architecture boundary (fail-closed, no oracle); a faithful port needs the CLM/CTSM column model, a v1.0 boundary. | XL | docs/namelist-compatibility.md (LSM family) |
| **Tier 4 — statistical / release closure** | | | |
| 12 | **Powered n≈27 TOST scoring** — corpus prepared, not scored; superseded as the primary gate by cell-identity but still a paper-equivalence item. | S–M | KI-5; ADR-029 |
| 13 | **v0.2.0 stable paper-release tag** — intended stable baseline never formally re-tagged. | S | `V0.2.0-PLAN.md` |
| **Tier 5 — performance / scale** | | | |
| 14 | **Real multi-GPU throughput** — K2 domain-decomposition periodic-BC bit-exact (experimental, default-off); specified-BC decomposition not yet faithful; DGX/NVLink cluster required for real throughput. | L | `contracts/halo.py` |
| 15 | **fp32-physics islands fast-mode** — compact explicit-fp64-island restructuring (~1.5–1.6×, still < 2×) as an optional fast-mode. | XL | GPU_PORT_GAPS (fp32 islands) |
| **Tier 6 — breadth beyond the wired set** | | | |
| 16 | **Scan-wire the reference-only-with-oracle tail** — 25 scheme codes have staged oracles but fail closed operationally; wiring each is incremental. | XL | docs/PORT_COMPLETION_ROADMAP.md |
| 17 | **FDDA / grid+obs / spectral nudging** — none (only lateral-BC relaxation). | M–XL | GPU_PORT_GAPS P1-1 |
| 18 | **Map-projection / grid generality** — Lambert/Mercator/Polar + hybrid-eta C-grid only; the opt-in moving driver needs validation and global/periodic grids remain open. | M | GPU_PORT_GAPS P2-1 |

**Critical path to a *complete operational* port:** item **1** (skill closure) is
the gate; **2–6** are the highest-value fidelity levers (where the remaining
wind/T2 skill lives); **7–8** complete nesting + output; the perf/scale items
(14–15) and breadth (16–18) are real but lower-leverage than the skill + fidelity
tier.

## Core goals (immutable)

1. **GPU-native architecture.** Whole-state device residency after init. No
   host/device transfers inside the timestep loop without an ADR. Fused
   timestep-scale kernels, not micro-kernel launch storms.
2. **Operational skill parity with CPU WRF v4** on Canary L2/L3 cases — proven
   cell-for-cell on the dynamics/thermo core; 24–72 h T2/U10/V10 forecast-skill
   equivalence is the open credibility gate.
3. **Performance vs CPU WRF** on the same workstation, re-certified after every
   correctness fix (no stale speedup claims). The headline is the
   command-to-finish wall-clock ratio; kernel-level ratios are reported separately.
4. **Validation against WRF, not bitwise reproducibility.** Tiered pyramid: micro
   fixture / savepoint parity → physical invariants → short-run / convergence →
   cell-identity proof against CPU-WRF.
5. **Forkable and auditable.** Every claim has a proof object on disk. Every
   architecture decision has an ADR with cross-model review.

## Where to look first (in this order)

| When you want to… | Read |
|---|---|
| Install and run your first forecast | [`docs/quickstart.md`](docs/quickstart.md) |
| Run the bundled real-data case (no download) | [`examples/switzerland_d01/`](examples/switzerland_d01/) |
| Compare the GPU port to CPU-WRF yourself | [`docs/equivalence-switzerland.md`](docs/equivalence-switzerland.md) |
| Run JUST the current version without the full repo | [Run JUST the current version](#run-just-the-current-version-without-the-full-repo-verified) above |
| Size a machine (VRAM / compile / scratch / energy) | [`docs/resource-profile.md`](docs/resource-profile.md) |
| Know which namelist options run vs fail-closed | [`docs/namelist-compatibility.md`](docs/namelist-compatibility.md) |
| See every WRF v4 scheme's classification | [Scheme triage](#scheme-triage--every-wrf-v4-scheme-classified), [`docs/namelist-compatibility.md`](docs/namelist-compatibility.md) |
| Understand the project scope | [`PROJECT_CONSTITUTION.md`](PROJECT_CONSTITUTION.md), [`CHANGELOG.md`](CHANGELOG.md) |
| See the WRF-v4 cell-identity proof + how to reproduce it | [`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md), `docs/assets/v018/identity_proof/` |
| Understand the performance (v0.20 all-7 ~1.07× vs v0.19 / ~1.53× vs CPU + capability/cache) | [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md), [`proofs/v020/lowhang/COMBINED_SPEEDUP.md`](proofs/v020/lowhang/COMBINED_SPEEDUP.md), [`proofs/v020/benchmark/T2T3_REPORT.md`](proofs/v020/benchmark/T2T3_REPORT.md) |
| Read the AceCAST positioning (PROJECTED) | [`proofs/v018/acecast_reconciliation.md`](proofs/v018/acecast_reconciliation.md) |
| Run & verify the GPU-vs-CPU equivalence demo | [`docs/equivalence-demo.md`](docs/equivalence-demo.md) — `scripts/equivalence_demo.py` |
| Run long GPU validation reliably | [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md) — `scripts/run_gpu_lowprio.sh` |
| Check current known issues | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) |
| Reproduce the proof collection on CPU | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — `scripts/verify_reproducibility.sh` |
| See the full WRF v4 gap inventory | [`docs/GPU_PORT_GAPS_TODO.md`](docs/GPU_PORT_GAPS_TODO.md) |
| See prior release proofs | [`proofs/`](proofs/) (`v019`, `v018`, `v017`, `v016`, `v015`, `v014`, `v013`, `v0120`, `v0110`, `v090`, `v0100`) |

## Known issues (v0.23.4, prepared/unreleased)

Full detail with symptom / ruled-out / workaround / follow-up in
**[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**. v0.23.4 resolves the SP2 mass-flux
seam, S2 boundary retention, deterministic V/V10 wake displacement, and
Thompson late-Ni balance, but carries the limits listed here.

| ID | Summary | Severity |
|---|---|---|
| **v0.23.4 d03 performance regression (measured, root-caused, not fixed)** | 3-domain (`maxdom3`) and 9-domain configurations that engage `d03` measure a real, production-faithful **51.67% regression** vs v0.23.3 (1.9668543059 → 2.9831176877 s/root-step); 79.95% of the added d03 cost is inside the FCT limiter's flux-renormalization computation (not dispatch/batching). Every quick-fix candidate (barrier, scan/map, register-pressure/fusion-disable, B1/B2/B3 scheduling, roll substitution) was falsified or rejected with evidence. A Nsight Compute hardware measurement confirmed the isolated kernel is **occupancy/latency-bound, not bandwidth-bound** (DRAM 20.3% of peak, occupancy 15.6%), ruling out a reduced-precision fix and identifying concurrent sibling-domain scheduling as the mapped next lever (design-scoped, unimplemented). **2-domain (d01+d02) configurations are unaffected** (measured 0.960–1.018×). The investigation is closed; the regression is shipped as a known, understood limitation. | Known limitation, understood |
| **Nested radiation cadence** | The nested runtime targets a fixed 1,800 s cadence rather than arbitrary namelist `radt` (the accepted fixture requests `radt=9`). The one-hour gate is accepted; longer-horizon radiation fidelity is not inferred. | Disclosed fidelity gap |
| **Nested terrain radiation** | `topo_shading=1` and `slope_rad=1` are recognized, but currently bind disabled in the nested runtime. Standalone support does not make them a nested capability. | Disclosed implementation gap |
| **Single-domain daily advection binding** | The daily pipeline currently drops requested `moist_adv_opt`/`scalar_adv_opt` and runs `0/0`; the accepted nested path does honor them. Do not claim a daily-pipeline run used the requested limiter without explicit evidence. | Disclosed path-specific fidelity gap |
| **Acoustic substep / dry-mass mismatch** | With `time_step_sound` omitted, pristine WRF derives 4 acoustic substeps on the tracked fixture while the current runtime selects 10. A four-substep discriminator improved the initial interior U/V comparison but did not pass the terminal gate; residual dry-mass behavior remains open. Defaults are unchanged. | Deferred dynamics fidelity work |
| **Thompson sedimentation debt** | v0.23.4 fixes the late-Ni ordering defect. The static `NSED_MAX=16` cap and widespread exact-zero carried Ni remain separate future work; neither is masked or claimed closed. | Deferred correctness work |
| **AOT cache-key fragmentation** | Shape-only fused-phase leaves and namespace-specific terrain provenance can prevent otherwise reusable executables from sharing a key. This is compile/warm-start debt, not a numerical workaround. | Performance debt |
| **K2 dt/n_sound remains opt-in** | The short single-domain gate measured a real 1.80x forecast-hour win, but the 3-domain steep-terrain ladder exposed vertical-CFL risk before any higher nested rung could be evaluated. Defaults stay unchanged; do not treat the single-domain result as a nested/default speedup. | Opt-in only |
| **Opt-in de-fuse trades host-RAM for GPU-VRAM/runtime (#123 mitigated-not-eliminated)** | The explicit de-fuse path cuts host compile-RAM on older 9-nest measurements but keeps nine resident per-domain executables and is slower at runtime; on a single 32 GB card the de-fuse 9-nest can still hit a GPU-VRAM OOM around the ~90 min integration horizon (#123, mitigated by the RRTMG-transient cap + fail-closed preflight, **not OOM-proof**). Use the fused default for runtime throughput and VRAM-stable long integration. B200 / fp32 / VRAM work remains future milestone work. | Mitigated, not fixed |
| **AOT warm-start ships verify-off by default** | The AOT cheap-key warm-start is default verify-off (a fresh load is numerically inert — the cheap key only locates the blob; the loaded executable is byte-identical to a cold compile). `GPUWRF_AOT_VERIFY=1` is the fail-closed backstop (lower-once + HLO-digest compare, quarantine on mismatch). v0.22 keeps the v0.21.1 default leaf-aval fused-call signature for bit identity and exposes the treedef/leaf-count structural-signature split as `GPUWRF_AOT_STRICT_AVAL_SIGNATURE=1`. | Documented scope |
| **fp32 1 h-only fidelity** | The opt-in fp32 mode (`mixed_perturb_fp32_v020`) is tolerance-checked **only at the 1 h lead** (19/19 fields green) — real but **not stringent**; the **24–120 h skill gate is future work, out of v0.20 scope**. fp64 stays the byte-identical default. | Documented scope |
| **High resident host RAM (~36 GB)** | A v0.20 run holds **~36 GB of host RAM** resident even at only ~9.6 GB VRAM, because the Noah-MP/physics constant tables are currently baked into the compiled executable (static aux) rather than passed as runtime arguments. This is **precision-independent** and **does not affect correctness, single-run stability, or results** — but it limits running two instances on a 64 GB box and reduces large-grid / pod-density headroom. Root-caused; the host-RAM-reduction fix (tables as runtime args, est. **−5 to −15 GB**) is **NOT shipped in v0.20.1 — deferred** (it was resequenced after the reliability/readiness work). v0.20.1 does separately remove the residual per-date **nested** recompile via the #114 cache fix. | Root-caused, host-RAM fix deferred |
| **fp32 mixed-precision OOMs on the deep all-7 nest** | The opt-in fp32 mode is **single-domain / capability-only**; on the full all-7 9-domain nest it can OOM on the recurring RRTMG radiation transient (allocator fragmentation) and is **~5× slower than fp64 there anyway**. The nest default is **fp64 + `cuda_async`**, the bounded path (~12.6 GB peak). v0.20.1 **reduces** the radiation transient (bit-identical RRTMG cap 2048→1024, largest alloc 0.432→0.271 GiB) and adds a fail-closed headroom preflight, but **does not make the nest OOM-proof** — solo `cuda_async` fragmentation can still OOM the full fp64 nest (#123 carried limitation). **fp32-on-nest remains out of scope** (not added in v0.20.1). | Documented scope; #123 mitigated not fixed |
| **RAINNC residual** | Accumulated `RAINNC` is **5.22 mm RMSE vs the 1.0 mm bound** (class-c) on the Switzerland 72 h cell-identity proof — a bounded, derived accumulated-precip diagnostic with **no expected forecast-skill impact**; **no tolerance widening**, drawn red. Diffuse Thompson staging + coupled accumulated-precip propagation; no single bounded missing process. | Bounded acceptance |
| **CLM4/CTSM boundary** | CLM4 (`sf_surface_physics=5`) / CTSM (`6`) are a **documented architecture boundary** — recognized, **fail-closed** with a named reason, **no oracle claimed**. A faithful port needs the CLM/CTSM column model; a v1.0 boundary. | Scope boundary |
| **K2 multi-GPU experimental** | The K2 domain-decomposition path is **EXPERIMENTAL, default-OFF, lab-only**: periodic-BC bit-exact on interior + shard seams, **physical specified-BC not yet faithful** (boundary ring excluded from the pass gate, not hidden). Default single-GPU graph bit-identical (no collectives). Not for production. | Experimental |
| **Shin-Hong PBL11 TKE** | Shin-Hong (`bl_pbl_physics=11`) is operational despite a ~28.5 % diagnostic-TKE residual, source-traced as **non-driving** (dynamics tendencies never read it); TKE-oracle upgrade is a documented follow-up. | Documented follow-up |
| **CPU suite xfail debt** | The full CPU test suite carries **38 documented non-strict xfail tests**, all **pre-existing** (each fails identically on tag `v0.17.0`; **zero v0.18-introduced regressions**, verified). They run and surface an XPASS if they start passing. Triage + per-test disposition recorded at the v0.18 release. | Carried test-debt |
| **KI-9** | **The credibility gate.** v0.23.4 closes its specific 24-hour SP2 chain and one-hour d01-d09 fixture, but broad seasonal/configuration-independent **24–72 h T2/U10/V10 skill equivalence** remains open. The older d02 demo is not silently reclassified by these narrower gates. | Documented gap |
| **Fused fast-path caveat** | v0.20 default fused all-7 nesting is **~1.53× faster than 12-rank CPU-WRF / ~1.07× faster than v0.19 (byte-identical to v0.19)** and all-fields tolerance-green, but it is tolerance-green rather than bitwise-vs-eager and pays a large one-time compile before cache warm. Use `GPUWRF_BITWISE=1` or `GPUWRF_NESTED_FUSE=0` for eager bitwise/debug comparisons. | Carried caveat |
| **KI-4** | d02 **U10** episodic final-lead under-prediction (8.06 m/s vs 7.5 m/s bar); within bar at all other leads, beats persistence 23/24. Tied to KI-9. | Documented residual |
| **KI-3** | Operational `wrfout` is a focused **104-variable** subset (vs WRF's 375). | Scope boundary |
| **KI-5** | Powered TOST campaign not run; **superseded by cell-identity as the primary gate**. No TOST PASS claimed. | Scope boundary |
| **KI-6** | RRTMG SW intermediate `taug` top-layer convention differs in 4 UV bands; integrated fluxes pass tier-1 (< 0.05% rel). Pre-existing. | Isolated |
| **KI-7** | Free-running (`run_boundary=False`) on **wide domains** (nx≈160+) can go unstable beyond ~14 h. Validated path uses boundary forcing. | Robustness edge |
| **KI-10** | Moisture-advection cadence refinements (opt-in path; physics-tendency folding not yet WRF-cadence-exact). Default-off → no shipped-behavior impact. | Fidelity refinement |
| **KI-11** | 2-way nesting equivalence vs CPU-WRF untested (only finite/stable proven). | Scope boundary |

## Layout

```
.
├── PROJECT_CONSTITUTION.md          immutable end goal
├── ARCHITECTURE_PRINCIPLES.md       backend / runtime principles
├── VALIDATION_STRATEGY.md           validation pyramid
├── PRECISION_POLICY.md              FP64/FP32/BF16 rules
├── docs/                            user-facing references
├── fixtures/                        manifest schemas + analytic samples + Canary slice
├── data/fixtures/                   vendored runtime tables (Thompson + RRTMG)
├── src/gpuwrf/                      implementation code
│   ├── contracts/                   frozen State / grid / physics_registry
│   ├── coupling/                    scan adapters + physics dispatch
│   ├── runtime/                     operational forecast loop
│   ├── physics/                     scheme kernels
│   ├── io/                          namelist check + wrfout/wrfinput I/O
│   └── integration/                 daily pipeline / native init
├── scripts/                         CLIs, validators, identity-proof builder
├── tests/                           pytest suite
└── proofs/                          per-milestone proof objects (JSON + reports)
```
