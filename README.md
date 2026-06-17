# wrf_gpu

**A GPU-native, WRF-compatible regional weather model.** `wrf_gpu` runs a
standalone WRF v4 ARW forecast end-to-end on a single GPU: it reads a standard
WRF `namelist.input`, assembles its own initial/boundary state from `met_em`
forcing (no `real.exe`, no CPU-WRF dependency), integrates a nonhydrostatic
split-explicit ARW dycore on the GPU, and writes a WRF-compatible `wrfout`
history file.

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
  single-domain or live-nested (d01→d02→d03, down to the 1 km nest), with native
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
- **A transparent, forkable research artifact.** Every claim has a proof object on
  disk; every architecture decision has a cross-model-reviewed ADR. It is built to
  be driven and extended by an AI manager agent (see [Use the manager](#use-the-manager-agent-driven-development)).

### What it is NOT

- **Not a universal WRF v4.** It covers the common operational ARW subset
  (the wired physics menu below); every unsupported namelist option **fails closed
  before any compute** with a named reason — it never silently substitutes a scheme.
- **Not proven for full 24 h/72 h forecast-skill equivalence.** The
  dynamics/thermodynamics core is proven **cell-for-cell identical** to CPU-WRF;
  the broader T2/U10/V10 forecast-skill equivalence is the **open credibility gate**
  (see [Boundaries](#boundaries--what-is-not-claimed)).
- **Not a single-card speedup story.** On tiny-nest geometries the GPU is
  launch/occupancy-bound and runs at ~parity with same-box CPU-WRF; an opt-in
  fast-mode reaches ~1.3×. The value is **capability** (1 km + scale) and
  **energy efficiency**, not raw single-card speed (see [Performance](#performance)).
- **Not** DFI / FDDA / spectral-nudging / WRF-Chem / WRF-Fire / urban / lake.

**v0.18 is the feature-completeness release.** Every WRF v4 namelist scheme is now
classified and handled: **50 operational** (oracle-gated), **23
reference-only-with-real-oracle** (recognized, validated against a real WRF
oracle, fail-closed if selected operationally), and **33 documented-boundary or
proven-irrelevant** (recognized, fail-closed with a named reason). No scheme is
silently substituted or skipped. See the [Scheme triage](#scheme-triage--every-wrf-v4-scheme-classified).

> ### First run is slow on purpose, then fast
> The first forecast **JIT-compiles the GPU kernels** — a **~8–12 min one-time cold
> compile with no output before integration starts (on the n=1 reference system)**.
> It is compiling, not hung. A **persistent on-disk JIT cache** (on by default)
> makes **every later run a fast cache read** (`cold ~147 s → cache-hit ~29 s` on
> the d01 hour-1 wrapper); the cached executable is **bit-identical** to the cold
> one. The opt-in fused fast-mode (`GPUWRF_NESTED_FUSE=1`, below) carries a
> separate, larger **one-time compile (~38 min on the n=1 reference system)** —
> also cached after the first run.

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
WRF)**. Proof: `proofs/v018/integration_report.md` (F1 closeout),
`proofs/v018/thompson_process_oracle.json`.

**Switzerland d01 — 72 h, v0.18 (9/10, dynamics/thermo cell-for-cell):**
![GPU↔CPU identity proof — Switzerland d01 72 h (v0.18)](docs/assets/v018/identity_proof/switzerland_d01/identity_dashboard.png)

This dashboard is built from the **retained v0.18 72 h GPU run**
(`v018_rainnc_qvapor_switzerland_d01_72h_qcfz_20260616T115735Z`, default
Thompson/RRTMG/MYNN/Noah, 72 hourly `wrfout` leads) paired cell-for-cell against
the retained CPU-WRF truth (`v014_switzerland_72h_cpu_20260610T122909Z`), scored
against the **frozen** tolerance manifest. 10 fields scored, **9 within tolerance**,
the single miss being `RAINNC` (5.22 mm vs the 1.0 mm bound). Manifest:
`proofs/v018/identity_proof/switzerland_d01/identity_proof_manifest.json`.

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
  --proof-dir proofs/v018/identity_proof/switzerland_d01 \
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

For a **live-nested** forecast (d01→d02→d03, down to the 1 km nest), add
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
succeeding) on a clean machine; evidence:
`proofs/v018/quickstart_minimal_source_verified.txt`.

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

This repository is built to be run and extended by an **AI manager agent** — the
shipped skill `.agent/skills/managing-sprints` is the operating manual. To drive
the project this way:

1. **Clone the repo on an isolated machine or VM** (the agent runs commands and the
   GPU; isolate it).
2. **Start Claude Code or a GPT/codex agent in auto-permission mode** in the repo
   directory.
3. **Tell it: "you are now the manager."** From that point the shipped
   `managing-sprints` skill tells it **where everything is and what to do** — read
   order (`PROJECT_CONSTITUTION.md` → `AGENTS.md` → the sprint contract → the
   relevant `.agent/skills`), the evidence/proof-object rules, how to dispatch and
   gate sub-agents (Opus ↔ GPT critic for kernel/perf-core work), the GPU lock, and
   the release protocol.

The manager assigns sprints, runs the acceptance gates, and merges — you steer it
at the milestone/decision level, not per-command.

## Performance

Measured on the reference RTX 5090 workstation vs same-box CPU-WRF.

- **v0.18 is perf-neutral vs v0.17.** The default-scheme standard case does **not**
  regress: two independent same-session measurements bracket zero (a committed GPT
  series shows v0.18 **−1.05 %**, i.e. slightly faster; an Opus canonical series
  shows **+0.54 %**, prose-only, within v0.17's own ±0.5–1 % repeat spread). The
  v0.18 Thompson cold-process additions are **warm-free** (XLA fuses the extra ops;
  ablation = +0.07 % noise) and stay **default-ON**; a transient program-shape
  regression (the mp=8 conditional-leaf carry narrowing 81→74) was reverted
  **bit-identically** (`292a4431`). Proof: `proofs/v018/perf_neutrality_FINAL.md`,
  `proofs/v018/perf_rootcause_opus.md`.
- **Default config runs at ~parity total-wall** on the 72 h gates (Switzerland
  0.99×, Canary 1.04× vs 24/28-rank CPU-WRF; forecast-only ~1.05–1.20×; warmed
  steady-state ~1.5×). This is a **GeForce fp64 hardware law** (1/64 fp32
  throughput), not a defect — the acoustic core is deliberately fp64 around
  cancellation.
- **Opt-in fused fast-mode** (`GPUWRF_NESTED_FUSE=1`) lifts the canary all-7
  (9/3/1 km) nested run to **~1.27–1.30× vs the same-box 12-rank CPU-WRF
  (MEASURED)**, GPU utilization 56 → 96 %. It is **tolerance-PASS vs CPU but NOT
  bitwise** (XLA FMA-contraction → a different valid trajectory) and carries a
  **~38 min one-time fused compile** (cached) — so it ships **opt-in**, and a
  24 h/72 h-tolerance check vs WRF v4 is the operator's gate for their case.
- **The ceiling** (profiler + independent GPT-5.5 cross-confirmed): after v0.17's
  host-orchestration fixes, the tiny-nest all-7 is **GPU-compute-bound (~674
  s/forecast-hour)** — an nsys trace shows **many ~1.5 µs kernels with no
  hot-spot**, i.e. a **launch/occupancy limit, not a throughput limit**. So **fp32
  cannot move it** (broad fp32 is ~1.1×, proven), and **≥2× / 3× are NOT
  single-card reachable** for this tiny-nest geometry (a 12-rank CPU is competitive
  on ~55 k tiny columns). The genuine speedup/scale levers are **algorithmic +
  multi-GPU**, not fp32.

**HEADLINE = CAPABILITY, not single-card tiny-nest speedup.** **MEASURED:** 1 km
single domain fits one RTX 5090 bit-identically; the all-7 1 km nested case runs
end-to-end on one card; the ~1.27–1.30× opt-in fast-mode is real. **PROJECTED /
UNMEASURED:** large single grids + **cluster / multi-GPU weak-scaling** and
whole-Earth-at-1 km "fits one rack" (exact memory arithmetic; multi-GPU throughput
**not benchmarked**). Detail: [docs/PERFORMANCE.md](docs/PERFORMANCE.md),
[`proofs/v017/analyze_hostgap_arm.py`](proofs/v017/analyze_hostgap_arm.py),
[`proofs/v018/perf_neutrality_FINAL.md`](proofs/v018/perf_neutrality_FINAL.md).

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

**Opt-in performance env flags** (all default to the bit-identical path):
`GPUWRF_NESTED_FUSE=1` (fused cascade fast-mode; ~1.3×, not bitwise),
`GPUWRF_NESTED_SYNC_MODE` (`root` default / `advance` / `segment`),
`GPUWRF_EDGE_ONLY_BOUNDARY` (ring-only boundary, **default on, bit-identical**),
`GPUWRF_JIT_BOUNDARY` (jit the boundary builder, default off),
`GPUWRF_HOST_LEDGER` (per-phase host-time diagnostic).

## System requirements & resource profile

Measured on the reference RTX 5090. Full detail: **[docs/resource-profile.md](docs/resource-profile.md)**.

| Resource | What to expect |
|---|---|
| GPU / VRAM | NVIDIA GPU with **≥ 31 GiB free VRAM** for the nested 72 h fp64 case (RTX 5090 / 32 GiB reference). 72 h gate peaks: **22.9 GiB** (Switzerland d01) / **29.8 GiB** (Canary L2 d02, nested); d01 9 km standalone peaks **≈ 4.7 GiB**; the 1 km single domain fits in a fresh process at **18.25 GiB** (chunked BouLac). Peak is transient working memory, not persistent fp64 State, so **fp32 does not reduce it** — the VRAM levers are algorithmic + multi-GPU. |
| First-run compile | **~8–12 min** one-time cold JIT compile (no output during compile). The **persistent on-disk cache** (default on since v0.12.0) turns later runs into a fast cache read (**cold ~147 s → cache-hit ~29 s** d01 hour-1 wrapper); cached executable is **bit-identical**. The opt-in `GPUWRF_NESTED_FUSE=1` adds a separate **~38 min one-time** fused compile (cached). |
| Scratch | A **real (non-tmpfs) NVMe scratch dir**, a few GiB free. Set via `--scratch-dir` / `$GPUWRF_SCRATCH`. Do **not** use a RAM disk. |
| Throughput | **~Parity** default (Switzerland 0.99×, Canary 1.04× total-wall vs 24-rank CPU; forecast-only ~1.05–1.20×; warmed steady-state ~1.5×); **~1.27–1.30×** with the opt-in fused fast-mode (vs 12-rank CPU); **perf-neutral vs v0.17** on the default case. No multi-× single-card speedup is claimed. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md). |
| Runtime data | The vendored `data/fixtures/` tables (~147 MiB: Thompson + RRTMG) are loaded at import; a minimal run install needs `src` + `data/fixtures` (see the source-only quickstart above). |
| Toolchain | CUDA 13 + a JAX CUDA build that sees the GPU. |

## Version history

Newest first. Full per-release evidence is under [`proofs/`](proofs/) and the
`RELEASE_NOTES_v*.md` files.

| Version | Headline | Key proof / link |
|---|---|---|
| **v0.18.0** | **FEATURE-COMPLETENESS + scheme triage.** Classifies and handles **every WRF v4 namelist scheme**: **50 operational** / **23 reference-only-with-real-oracle** / **33 documented-boundary or proven-irrelevant** (State = 67 leaves; no scheme/leaf dropped). Default **Thompson microphysics is strictly more WRF-faithful than v0.17** (cold-process additions + a warm-process melt/cold-gate fix → cell `qv` bit-exact WRF). **Perf-neutral vs v0.17** (default case, dual-confirmed). Adds **experimental, default-OFF K2 multi-GPU** domain decomposition (periodic-BC bit-exact; specified-BC not yet faithful — lab-only). | [`proofs/v018/integration_report.md`](proofs/v018/integration_report.md), [`proofs/v018/scheme_count_no_clobber.json`](proofs/v018/scheme_count_no_clobber.json), [`proofs/v018/suite_triage.md`](proofs/v018/suite_triage.md), [`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md) |
| **v0.17.0** | **PERFORMANCE + ceiling.** Closes the live-nested GPU host-orchestration holes — the **all-7 island nest (`--max-dom 9`) now forecasts at all** (previously recompiled forever → 0 output); default config **bit-identical to v0.16**. Adds an **opt-in fused fast-mode** (`GPUWRF_NESTED_FUSE=1`: util 56→96 %, **~1.27–1.30× vs 12-rank CPU**, tolerance-PASS not bitwise, ~38 min one-time compile). Answers speedup plainly: tiny-nest all-7 is **launch/occupancy-bound (~674 s/hr, nsys-grounded)** — **fp32 cannot move it, ≥2×/3× not single-card reachable**. Value = **capability** (1 km fits one card + scale), not single-card tiny-nest speed. | [`proofs/v017/analyze_hostgap_arm.py`](proofs/v017/analyze_hostgap_arm.py), [`proofs/v017/run_all7_hostgap_arm.sh`](proofs/v017/run_all7_hostgap_arm.sh) |
| **v0.16.0** | **STABILITY + 1 km-unlock.** Proves **24 of 25 L2 physics schemes run coupled-green** on a real Switzerland d01 case (25th = Noah-classic, scope-carry → `ALL_GREEN_OR_CARRIED`). Adds **aerosol-aware Thompson** (`mp_physics=28`, WRF-module oracle PASS). Ships a **chunked MYNN BouLac** that makes a **1 km single domain fit one RTX 5090 bit-identically** (dense OOMs at ≈18.8 GiB; chunked fits at 18.25 GiB). **fp32 make-or-break CONCLUDED** (Opus + independent GPT): valid-numerics ceiling **~1.1×**, 0 % VRAM-peak reduction. | [`proofs/v016/coverage/`](proofs/v016/coverage/), [`proofs/v016/coverage_map.json`](proofs/v016/coverage_map.json) |
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
| **Dynamics** | Nonhydrostatic ARW, RK3 + split-explicit acoustic, flux-form advection, constant-K (`diff_opt=2`/`km_opt=1`) + 2-D Smagorinsky (`diff_opt=1`/`km_opt=4`) horizontal diffusion | 3-D TKE / full Smagorinsky (`km_opt=2/3/5`) → use `km_opt=1` or `4` | Moving/global nests; adaptive Δt |
| **Microphysics** | Kessler, Purdue-Lin, WSM3/5/6/7, Thompson, **aerosol-aware Thompson (mp=28)**, Morrison, SBU-YLin, WDM5/6/7, Goddard GCE | Aerosol-coupled Morrison (mp=40), NSSL, and the rest of the WRF MP tail (recognized, real-oracle or documented-boundary) | WRF-Chem |
| **PBL / sfc** | YSU, MYJ, MYNN-EDMF, ACM2, BouLac, GFS, GBM-TKE, MRF, **Shin-Hong (operational, TKE-diagnostic follow-up)**; MYNN-SL, revised-MM5, Pleim-Xiu, Janjic-Eta, NCEP-GFS sfclay | CAM-UW (`bl=9`); reference-only PBL tail (real oracle) | — |
| **Cumulus** | Kain-Fritsch, BMJ, Tiedtke (needs active flux-form moisture advection for RQVFTEN), Grell-Freitas (scale-aware) | New-Tiedtke + the reference-only/​documented-boundary CU tail (real oracle or named reason) | — |
| **Radiation** | RRTMG SW + LW with topographic shading + slope correction; Dudhia SW + classic RRTM LW (`ra_lw=1`); clear-sky `…C` flux diagnostics (opt-in) | Reference-only RA tail (real oracle); `ra_*={14,24}` compiled-out (BUILD-gated, like WRF) | — |
| **Land** | Noah classic, Noah-MP (prognostic), Pleim-Xiu LSM, thermal-diffusion slab | RUC LSM (reference-only, real oracle staged); **CLM4 (`sf_surface_physics=5`) / CTSM (`6`) — documented architecture boundary, fail-closed (no oracle claimed)** | Full Noah-MP snow-layer diagnostics in wrfout (KI-3) |
| **Nesting** | One-way live d01→d02→d03, per-domain subcycling, restart; GWD (`gwd_opt=1`) default-on on nested | — | Two-way feedback + radiation/w-relax in loop — finite/stable but 24 h equivalence untested (KI-11) |
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
| Cumulus | `cu_physics` | 1 Kain-Fritsch, 2 BMJ (fp64), 3 Grell-Freitas (scale-aware), 6 Tiedtke (needs flux-form moisture advection for RQVFTEN) |
| Radiation | `ra_sw_physics` / `ra_lw_physics` | RRTMG SW + LW (`=4`) with topo shading (`topo_shading=1`) + slope-corrected surface radiation (`slope_rad=1`); Dudhia SW (`ra_sw=1`) + classic RRTM LW (`ra_lw=1`); Held-Suarez idealized radiation (`ra_lw=31`); clear-sky `…C` flux diagnostics (opt-in) |
| Land surface | `sf_surface_physics` | 1 thermal-diffusion slab, 2 Noah classic (explicit static/land bundle), 4 Noah-MP (`use_noahmp=True`), 7 Pleim-Xiu LSM |
| Diffusion | `diff_opt`, `km_opt` | constant-K and 2-D Smagorinsky (incl. terrain-slope + map-factor deformation terms; WRF formula parity, max residual `3.78e-15`) |
| GWD | `gwd_opt` | 1 gravity-wave drag — **default-ON on the nested 1 km path** (`GPUWRF_GWD_NESTED=0` forces off) |
| Advection | `moist_adv_opt`, `scalar_adv_opt` | moisture flux-advection into RK3 + PD/monotonic moisture limiter (both opt-in, default-off = byte-identical) |

`mp_physics=0`, `bl_pbl_physics=0`, `sf_sfclay_physics=0`, `cu_physics=0`, and
`ra_*=0` are accepted as "disabled" slots.

### Scheme triage — every WRF v4 scheme classified

v0.18 closes the scheme gap by **classifying every WRF v4 namelist scheme** into
one of three buckets, with no scheme silently dropped (State = **67 leaves**, set-
union integrity proven across all family branches):

| Class | Count | Meaning |
|---|---|---|
| **Operational** | **50** | Scan-wired into the GPU forecast loop, WRF-oracle-gated. (mp 15, cu 5, bl 10, sfclay 7, sf_surface 5, ra_lw 4, ra_sw 4) |
| **Reference-only-with-real-oracle** | **23** | A real WRF v4 scheme, validated against a real WRF oracle, **not** scan-wired; selecting it operationally **fails closed** with a named reason (never a silent fallback). (cu 9, bl 4, ra_lw 4, ra_sw 4, sf_surface 2) |
| **Documented-boundary / proven-irrelevant** | **33** | Recognized WRF option, **fail-closed** as a documented architecture boundary (e.g. CLM4/CTSM, CAM-UW) or proven-irrelevant tail. (mp 23, cu 3, ra_lw 2, ra_sw 2, sf_surface 2 [CLM4/CTSM], bl 1 [CAM-UW]) |

Proof object (set-union integrity, no-clobber, per-family counts):
[`proofs/v018/scheme_count_no_clobber.json`](proofs/v018/scheme_count_no_clobber.json)
(`checks.all_green=true`), independently re-verified by the v0.18 integration
critic ([`proofs/v018/integration_honesty_critic_opus.md`](proofs/v018/integration_honesty_critic_opus.md)).
The full per-code support table is in
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md).

## Boundaries — what is NOT claimed

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
  unchanged (no goalpost moving, no tolerance widening). Proof:
  `proofs/v018/rainnc_qvapor_status.json`.
- **K2 multi-GPU is EXPERIMENTAL and default-OFF.** The K2 domain-decomposition
  path (`GPUWRF_K2_EXPERIMENTAL=1`) is **lab-tested only**: with the gate unset the
  default single-GPU graph is **bit-identical** (no collectives emitted,
  `proofs/v018/k2_flag_off_graph.json`). With the gate set, the **periodic-BC**
  decomposition reproduces the single-GPU reference **bit-for-bit at roundoff** on
  interior + internal shard seams — but the **physical (specified) boundary is NOT
  yet faithful** (periodic vs WRF specified BC diverge by design at the true domain
  edge; the boundary ring is *excluded* from the pass gate, not hidden behind a
  loosened tolerance). Do **not** enable K2 specified-BC multi-GPU for production.
  Proof: `proofs/v018/k2_multigpu_report.md`.
- **No statistical-equivalence (TOST) claim.** The cell-identity proof above
  **supersedes** the earlier TOST framing as the primary fidelity gate. The
  station-RMSE TOST campaign is underpowered at the available corpus (n=15;
  n≈27 for full power) and is **not run / not claimed**; deferred (KI-5).
- **Not a single-card speedup release.** Default total-wall is ~parity; the opt-in
  fused fast-mode reaches ~1.27–1.30× (tolerance-PASS, not bitwise). On tiny-nest
  geometries the GPU is **launch/occupancy-bound**, so **≥2×/3× are NOT
  single-card reachable** and **fp32 cannot move it** (the valid-numerics fp32
  ceiling is ~1.1×, proven + cross-confirmed). The genuine speedup/scale levers are
  **algorithmic + multi-GPU**, not fp32. v0.18 is **perf-neutral vs v0.17** on the
  default case.
- **Multi-GPU throughput unmeasured.** The `shard_map` + `lax.ppermute` halo
  sharding is bit-identical on a CPU fake mesh, but this workstation has one
  physical RTX 5090 — real multi-GPU throughput / NVLink-NCCL bandwidth / collective
  overlap are **UNMEASURED**; the whole-Earth memory note stays **PROJECTED**. **No
  per-watt / per-kWh claim is made.**
- **Shin-Hong PBL (`bl_pbl_physics=11`) is operational with a TKE-diagnostic
  follow-up.** It is scan-wired and operational despite a ~28.5 % residual in the
  diagnostic TKE field, which was source-traced as **non-driving** (the dynamics
  tendencies never read it); the TKE-oracle upgrade is a documented follow-up, not
  a masked failure. See `proofs/v018/schemes_critic_opus.md`.
- **Not full two-way nesting.** One-way live nesting is proven over a 24–72 h
  window; the two-way feedback path is finite/stable but its 24 h real-GPU
  equivalence vs CPU-WRF is **untested** (KI-11).
- **fp64-only standalone.** The standalone CLI path forces pure fp64; there is no
  fp32 standalone path (gated-fp32 is an experimental ADR-007 preview, no faster on
  this memory-bound workload).
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
| 2 | **RAINNC bounded accumulated-precip residual** — 5.22 mm RMSE vs 1.0 mm bound (class-c, no skill impact expected); diffuse Thompson staging + coupled accumulated-precip propagation, no single bounded missing process. | M | `proofs/v018/rainnc_qvapor_status.json` |
| 3 | **MYNN PBL completeness** — EDMF mass flux wired; `icloud_bl=1` cloud PDF and `cloudmix` partial. Tied to the residual near-surface wind-skill gap. | M | GPU_PORT_GAPS P1-4 |
| 4 | **Shin-Hong PBL TKE-diagnostic** — operational; diagnostic TKE field ~28.5 % residual (non-driving, source-traced); oracle upgrade follow-up. | S | `proofs/v018/schemes_critic_opus.md` |
| 5 | **Moisture advection into RK3 + cadence fidelity** — wired opt-in (default-off); cadence refinements + operationalizing on the default path remain. | M | GPU_PORT_GAPS P1-6; KI-10 |
| 6 | **RRTMG SW taug top-layer convention fix** — 4 UV bands fail intermediate oracle; tier-1 fluxes faithful; pre-existing. | S | KI-6 |
| **Tier 2 — nesting / output completeness** | | | |
| 7 | **Full multi-domain nested equivalence** — 24 h one-way proven; two-way feedback + radiation-in-loop + w relaxation + 5-domain long-run equivalence remain (2-way 24 h real-GPU equivalence untested). | L | GPU_PORT_GAPS P0-1; KI-11 |
| 8 | **Full `wrfout` variable coverage** — focused 104-variable writer vs WRF's 375. Blocks downstream tools. | M | GPU_PORT_GAPS P0-5; KI-3 |
| **Tier 3 — correctness / robustness debts** | | | |
| 9 | **Free-running open-lateral-boundary stability** — wide domains (nx≈160+) can blow up without boundary relaxation beyond ~14 h. | M | KI-7 |
| 10 | **U10 episodic under-prediction** — final-lead breach on the validated d02 case (tied to MYNN cloud PDF). | S–M | KI-4 |
| 11 | **CLM4/CTSM land-surface** — documented architecture boundary (fail-closed, no oracle); a faithful port needs the CLM/CTSM column model, a v1.0 boundary. | XL | `proofs/v018/lsm_family_status.json` |
| **Tier 4 — statistical / release closure** | | | |
| 12 | **Powered n≈27 TOST scoring** — corpus prepared, not scored; superseded as the primary gate by cell-identity but still a paper-equivalence item. | S–M | KI-5; ADR-029 |
| 13 | **v0.2.0 stable paper-release tag** — intended stable baseline never formally re-tagged. | S | `V0.2.0-PLAN.md` |
| **Tier 5 — performance / scale** | | | |
| 14 | **Real multi-GPU throughput** — K2 domain-decomposition periodic-BC bit-exact (experimental, default-off); specified-BC decomposition not yet faithful; DGX/NVLink cluster required for real throughput. | L | `proofs/v018/k2_multigpu_report.md`; `contracts/halo.py` |
| 15 | **fp32-physics islands fast-mode** — compact explicit-fp64-island restructuring (~1.5–1.6×, still < 2×) as an optional fast-mode. | XL | `proofs/v016/fp32_verdict/` |
| **Tier 6 — breadth beyond the wired set** | | | |
| 16 | **Scan-wire the reference-only-with-oracle tail** — 23 schemes validated against a real oracle but fail-closed operationally; wiring each is incremental. | XL | `proofs/v018/scheme_count_no_clobber.json` |
| 17 | **FDDA / grid+obs / spectral nudging** — none (only lateral-BC relaxation). | M–XL | GPU_PORT_GAPS P1-1 |
| 18 | **Map-projection / grid generality** — Lambert/Mercator/Polar + hybrid-eta C-grid only; no moving/global nests. | M | GPU_PORT_GAPS P2-1 |

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
| Run JUST the current version without the full repo | [Run JUST the current version](#run-just-the-current-version-without-the-full-repo-verified) above (`proofs/v018/quickstart_minimal_source_verified.txt`) |
| Size a machine (VRAM / compile / scratch / energy) | [`docs/resource-profile.md`](docs/resource-profile.md) |
| Know which namelist options run vs fail-closed | [`docs/namelist-compatibility.md`](docs/namelist-compatibility.md) |
| See every WRF v4 scheme's classification | [Scheme triage](#scheme-triage--every-wrf-v4-scheme-classified), [`proofs/v018/scheme_count_no_clobber.json`](proofs/v018/scheme_count_no_clobber.json) |
| Understand the project scope | [`PROJECT_CONSTITUTION.md`](PROJECT_CONSTITUTION.md), [`CHANGELOG.md`](CHANGELOG.md) |
| See the WRF-v4 cell-identity proof + how to reproduce it | [`docs/IDENTITY_PROOF.md`](docs/IDENTITY_PROOF.md), `docs/assets/v018/identity_proof/`, [`proofs/v018/identity_proof/`](proofs/v018/identity_proof/) |
| Understand the performance (~parity + opt-in fast-mode + ceiling + perf-neutral vs v0.17) | [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md), [`proofs/v018/perf_neutrality_FINAL.md`](proofs/v018/perf_neutrality_FINAL.md), [`proofs/v017/analyze_hostgap_arm.py`](proofs/v017/analyze_hostgap_arm.py) |
| Read the AceCAST positioning (PROJECTED) | [`proofs/v018/acecast_reconciliation.md`](proofs/v018/acecast_reconciliation.md) |
| Run & verify the GPU-vs-CPU equivalence demo | [`docs/equivalence-demo.md`](docs/equivalence-demo.md) — `scripts/equivalence_demo.py` |
| Run long GPU validation reliably | [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md) — `scripts/run_gpu_lowprio.sh` |
| Check current known issues | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md), [`proofs/v018/suite_triage.md`](proofs/v018/suite_triage.md) |
| Reproduce the proof collection on CPU | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — `scripts/verify_reproducibility.sh` |
| See the full WRF v4 gap inventory | [`docs/GPU_PORT_GAPS_TODO.md`](docs/GPU_PORT_GAPS_TODO.md) |
| See prior release proofs | [`proofs/`](proofs/) (`v018`, `v017`, `v016`, `v015`, `v014`, `v013`, `v0120`, `v0110`, `v090`, `v0100`) |

## Known issues (v0.18.0)

Full detail with symptom / ruled-out / workaround / follow-up in
**[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**. The v0.18 release carries only the
items below; everything resolved in prior releases is dropped.

| ID | Summary | Severity |
|---|---|---|
| **RAINNC residual** | Accumulated `RAINNC` is **5.22 mm RMSE vs the 1.0 mm bound** (class-c) on the Switzerland 72 h cell-identity proof — a bounded, derived accumulated-precip diagnostic with **no expected forecast-skill impact**; **no tolerance widening**, drawn red. Diffuse Thompson staging + coupled accumulated-precip propagation; no single bounded missing process. | Bounded acceptance |
| **CLM4/CTSM boundary** | CLM4 (`sf_surface_physics=5`) / CTSM (`6`) are a **documented architecture boundary** — recognized, **fail-closed** with a named reason, **no oracle claimed**. A faithful port needs the CLM/CTSM column model; a v1.0 boundary. | Scope boundary |
| **K2 multi-GPU experimental** | The K2 domain-decomposition path is **EXPERIMENTAL, default-OFF, lab-only**: periodic-BC bit-exact on interior + shard seams, **physical specified-BC not yet faithful** (boundary ring excluded from the pass gate, not hidden). Default single-GPU graph bit-identical (no collectives). Not for production. | Experimental |
| **Shin-Hong PBL11 TKE** | Shin-Hong (`bl_pbl_physics=11`) is operational despite a ~28.5 % diagnostic-TKE residual, source-traced as **non-driving** (dynamics tendencies never read it); TKE-oracle upgrade is a documented follow-up. | Documented follow-up |
| **CPU suite xfail debt** | The full CPU test suite carries **38 documented non-strict xfail tests**, all **pre-existing** (each fails identically on tag `v0.17.0`; **zero v0.18-introduced regressions**, verified). They run and surface an XPASS if they start passing. Triage + per-test disposition: [`proofs/v018/suite_triage.md`](proofs/v018/suite_triage.md). | Carried test-debt |
| **KI-9** | **The credibility gate.** Cell-identity proven (dynamics/thermo core cell-for-cell), but the broader **24 h/72 h forecast-skill equivalence** is open — equivalence demo 24 h d02 `NOT_EQUIVALENT`, dominated by **lead-time wind divergence** (3D V pooled RMSE 8.13 m/s). Hard dynamics-`ph'`/MYNN/`*_tendf` GPU work, no cheap knob. | Documented gap |
| **Perf ceiling** | Tiny-nest single-card is launch/occupancy-bound; **≥2×/3× not single-card reachable**, fp32 ceiling ~1.1×. Opt-in fuse ~1.27–1.30× (not bitwise). **Perf-neutral vs v0.17.** Value = capability + scale. | Documented next-lever |
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
