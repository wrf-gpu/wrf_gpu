# Checkpoint branches and short forecast windows

Available in the merged v0.4.4 development build. CPU host/branch admission
is qualified; three-nest GPU null-resume, null-branch and edited-branch gates
remain pending. These are gpuwrf checkpoint/workflow extensions. They do not
make native CPU-WRF restart files interchangeable with gpuwrf checkpoints.

A branch starts from a verified gpuwrf synchronized checkpoint, keeps every
nest's model clock and held physics, and writes subsequent history to a new
directory. Checkpoints are trusted local Python artifacts; they are not a file
interchange format. Use the same installed source, inputs, static namelist,
environment and enabled parameter families as the parent run. Branching requires
batch size one. Regional stream journals and domain decomposition are currently
unsupported. The parallel-case launcher (`scripts/run_parallel_cases.sh`)
refuses `--minutes`, `--end-time`, `--branch-from` and `--branch-manifest`,
and all checkpoint and resume options; run the parent and its branches
directly with `gpuwrf run`.

The checkpoint also binds runtime data by content: WRF land/soil/vegetation/
urban/radiation/gas tables, package resources and selected microphysics assets.
Replacing a table at the same path is an undeclared change and is refused.
Use the traced parameter controls for supported sensitivity experiments;
editing an underlying table is outside the branch manifest contract. Keep
runtime tables immutable during a forecast, like the model inputs.

Write a parent checkpoint with the usual `--checkpoint-dir` and
`--checkpoint-interval-steps`. Checkpoints created before the branch schema was
introduced are refused; produce a new checkpoint with this build.

For a short Swiss parent run (GPU execution), the shipped root timestep is
18 seconds, so 18 minutes is 60 root steps:

```bash
gpuwrf run --input-dir examples/switzerland_d01 --domain d01 --minutes 18 \
  --output-dir runs/swiss-parent --checkpoint-dir runs/swiss-checkpoints \
  --checkpoint-interval-steps 60
```

Choose the exact completed generation path inside `runs/swiss-checkpoints`
for a later branch; this example does not select a generation automatically.
A window shorter than the one-hour history cadence need not write a new
history frame; the requested checkpoint retains endpoint state.

To continue without changing the state, create `empty.json`:

```json
{"schema_version": 1}
```

Then fork a generation into an empty output directory:

```bash
gpuwrf run --input-dir case --domains-from-namelist \
  --branch-from checkpoints/generation-EXACT-NAME \
  --branch-manifest empty.json --output-dir branch-null --hours 6
```

Durations are endpoints **since the input issue time**, including for a branch.
For example, branching at hour three with `--hours 6` advances three more hours.
A branch may choose a shorter endpoint than its parent's originally planned
forecast, provided the endpoint is after the checkpoint time. Ordinary
`--resume-checkpoint` keeps its existing same-stream and `--extend-run` rules.
Child checkpoints are written only when `--checkpoint-dir` and a cadence are
requested; their provenance retains the full parent identity and manifest digest.
Later ordinary resumes keep that child identity and its own history journal.

The resolved output policy is part of restart identity: changing variable
selection, compression, history cadence or effective writer settings is an
undeclared change and refuses. Branches create their own history journal;
ordinary resume continues the existing journal. Branching does not waive
source, static namelist, runtime table or output-policy identity checks.
Ordinary `--resume-checkpoint` needs the parent's `--checkpoint-dir` including
its `history-<stream>/owner.json` and journal receipts; copying only generation
directories is insufficient, and `--branch-from` is the route to a new directory.

`--minutes` and `--end-time` are alternatives to `--hours`:

```bash
gpuwrf run --input-dir case --domains-from-namelist \
  --output-dir short-run --minutes 18
```

For a 54 second root timestep, 18 minutes is 20 complete root steps; 10 minutes
is refused. An absolute endpoint can be `--end-time 2026-02-27T00:18:00Z`.
Short windows use the native d01 or nested driver. The normal history cadence
still applies: a window shorter than the history interval may produce no new
history frames. Request a checkpoint cadence to retain its endpoint state.

## Declared state and land changes

The manifest names each nest explicitly. An omitted nest is carried unchanged.
Each field accepts exactly one operation: `set`, `add`, or `multiply`. A numeric
operand applies uniformly. A spatial operand is an exact-shape NumPy `.npy`
array referenced by `{"path":"relative.npy","sha256":"FULL_SHA256"}`. Its
hash is checked at admission and again before applying it; no broadcasting of
spatial arrays, pickled arrays, NaN or infinity is accepted.

For example, this changes QV on all three nests:

```json
{
  "schema_version": 1,
  "domains": {
    "d01": {"state": {"qv": {"multiply": 0.99}}},
    "d02": {"state": {"qv": {"multiply": 0.99}}},
    "d03": {"state": {"qv": {"multiply": 0.99}}}
  },
  "waivers": {
    "d01": {"hydrostatic_consistency": "Controlled unbalanced QV sensitivity experiment"},
    "d02": {"hydrostatic_consistency": "Controlled unbalanced QV sensitivity experiment"},
    "d03": {"hydrostatic_consistency": "Controlled unbalanced QV sensitivity experiment"}
  }
}
```

The shared input consistency checks apply to edited fields and their dependencies.
Thermodynamic edits currently require an explicit hydrostatic-consistency waiver
because no balanced MU/PH recomputation is provided. A waiver records a research
choice; it does not establish WRF fidelity. Finite/schema failures cannot be
waived. Supersaturation or range waivers, if needed, must also be declared with
their own reasons. Unchanged checkpoint data is retained exactly, including
supersaturated model states.

State field names are internal carry names: `u`, `v`, `w`, `theta`, `qv`, `qc`,
`qr`, `qi`, `qs`, `qg`, `qh`, `Nc`, `Nn`, `Ni`, `Nr`, `Ns`, `Ng`, `Nh`,
`qvolg`, `qvolh`, `nwfa`, `nifa`, `qke`, `p_perturbation`,
`ph_perturbation`, and `t_skin`. Only leaves already present in the selected
scheme may be edited. `theta` is full moist potential temperature in kelvin,
including when WRF's history uses dry theta. Editing pressure or geopotential
perturbations also applies that increment to the corresponding total array.
Dry mass MU, terrain and persistent base states remain fixed.

The `land` group addresses the Noah-MP carry: `smois`, `tslb`, `sh2o`, `tsno`,
`snice`, `snliq`, `snowh`, `sneqv`, `canliq`, `canice`, `tv`, `tg`, `tah`,
`lai`, `sai`, and `t_skin`. Layer arrays retain their stored layer order.
All other scratch arrays, held tendencies, accumulators and prognostics retain
their parent bytes at the branch point.

## Traced parameter changes

The manifest's optional `parameters` object maps a supported family to per-nest
SHA256 value digests. For v0.4.4, `surface_controls` follows its existing
contract; `mynn_parameters` ships as **experimental**, qualified only by its
component and short two-domain GPU root checks. Three-nest D6, station results
and 24–72-hour drift are not qualified by that result. See
[family status](runtime-parameters.md).

The schema name `diffusion_parameters` does not imply release support:
diffusion/damping overrides, including `c_s`, are **not available in v0.4.4**
and are refused after their GPU gate failed. They may come in a later release after GPU qualification; no date or
commitment. Do not include them as a branch's enabled parameter family.

Supply supported new values through the family tool's normal traced input
option. The family must already be enabled with the same names, shapes and
dtypes; changing static namelist options is refused.
The requested run must match the declared digests exactly. See the parameter
family documentation for supported names and qualification status.

## Branch series: many value variants from one spin-up

For a series that differs only in surface-control values (for example z0mvt
per nest), run the parent once with a neutral control file that enables the
family on **every nest any variant will touch** (factor 1 reproduces the table
values byte for byte). Give every nest explicitly: `--set z0mvt=1` without a
`dNN:` prefix writes d01 only.

```bash
gpuwrf surface-params basis --input-dir case --max-dom 3 --output basis.npz
gpuwrf surface-params expand --basis basis.npz --output neutral.npz \
  --set d01:z0mvt=1 --set d02:z0mvt=1 --set d03:z0mvt=1
gpuwrf surface-params expand --basis basis.npz --output z0x2_d03.npz \
  --set d01:z0mvt=1 --set d02:z0mvt=1 --set d03:z0mvt=2
gpuwrf aot-key --input-dir case --surface-controls neutral.npz \
  --compare-surface-controls z0x2_d03.npz        # every program SHARED = no recompile
gpuwrf run --input-dir case --domains-from-namelist --hours 12 --surface-controls neutral.npz \
  --output-dir parent --checkpoint-dir ckpt --checkpoint-interval-steps 800
```

Then list the variants in a series file and run them from the parent
generation with `scripts/branch_series.py SERIES.json` (format in the script).
It writes each variant's manifest (the control digest of every changed nest),
refuses a control file whose nests, names, shapes or dtypes differ from the
parent's (that would be another program; the CLI refuses it as an undeclared
`namelists` change), and checks that the parent control file reproduces the
checkpoint's recorded digests. `--mode process` runs one CLI process per
variant; `--mode inprocess` runs all variants in one process and reuses imports
and loaded programs; it stops at the first failure. Receipts go to
`OUT_ROOT/series_receipt.json` (it keeps each variant's `wall_clock_total_s` and
`wall_clock_forecast_only_s`, and in-process JAX counters including cache-disabled
and fresh compiles). On CUDA, `--mode inprocess` needs the verified GPU lock
(`scripts/with_gpu_lock.sh`: lock fd and holder token); the runner never forces
the VRAM preflight and refuses `--force-gpu-run` / `GPUWRF_FORCE_GPU_RUN`.
`cli_args` go through the real `gpuwrf run` parser with abbreviations disabled.
Only `--domains-from-namelist`, `--max-dom`, `--domain`, `--emit-initial-history`,
`--aot-prefetch` and `--runtime-parameters` may appear there. Branching supports
one-way nesting only: a parent or branch with two-way feedback is refused. The
parallel launcher still refuses branches.
Seed variants are not available: SKEBS/SPPT seeds live in the checkpointed
carry and `nens`/`iseed_*` are part of the static namelist identity.

## Cost and qualification

Branch admission runs once on host checkpoint arrays before restoring them to
the device. It adds no step carry leaves or numerical operators. It retains
host checkpoint storage and allocates edited array copies during admission;
device restore has the ordinary restart allocation cost. Default runs keep the
unchanged Swiss and two-nest Canary Islands forecast programs. CPU structural
tests do not establish a GPU forecast result. GPU comparisons of continuation,
an empty branch and a water-vapour-edited branch on three Tenerife nests are
still pending.

On **input views** from the original three-domain Tenerife case, host admission measured a null
median of 0.00568 s; QV admission took 5.2833 s first call and a later median
of 0.24284 s [M]. Traced peak allocations were 1.35 MiB null / 302.53 MiB QV,
with 1.106 GiB process RSS including imports/input arrays. These are neither
complete-checkpoint timings nor GPU restore time/VRAM measurements. Runtime
table hashing and output-policy binding add host work once per checkpoint-
enabled run; no isolated complete-run cost is established.

The original CPU checks, source-identity evidence and open GPU gates are in
the [BRANCH report](../proofs/w44_branch/BRANCH.md),
[output-policy delta](../proofs/w44_branch/OUTPUT_POLICY_DELTA.json) and
[runtime-data contract](../proofs/w44_branch/RUNTIME_DATA_BINDING.md).

## CPU-WRF comparison outputs

Use original CPU-WRF's own restart workflow as its continuation reference.
Match the parent checkpoint valid time, forcing and edits; compare null and
edited arms separately. A manifest waiver is a research choice, not CPU-WRF
configuration. For atmospheric/land response, WRF's default history already
holds the winds, temperature, pressure and geopotential, moisture, QKE, the
2 m and 10 m fields, soil and snow state and canopy water, so no
`iofields_filename` line is needed for them. Check that every extra species of
the chosen microphysics is present in both outputs. Hourly history
does not preserve all held physics/scratch for a restart proof, and a sub-hour
window may produce no comparison frame. Use checkpoint/native restart
evidence for the exact seam, then matched history times for forecast response.
Return to [Tune WRF for your region](regional_optimization.md).
