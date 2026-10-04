export const meta = {
  name: 'v022-compile-pathology-r0',
  description: 'v0.22 GPU openers: 3-gate compile-pathology + R0 flag-sweep + K2 dt/n_sound CFL ladder + synthesis',
  whenToUse: 'Launch when the GPU is FREE (9-nest bit-identity gate done). Big coding fan-out per the user dispatch rule (ultracode). One launch fires ALL GPU openers in deterministic sequence (they serialize on the single GPU lock anyway). Verified paths/rungs/flags live in proofs/v022/GPU_OPENER_PREP.md.',
  phases: [
    { title: 'Compile-pathology diagnostic' },
    { title: 'R0 flag-sweep' },
    { title: 'K2 dt/n_sound ladder' },
    { title: 'Synthesize' },
  ],
}

// READY-TO-FIRE SKELETON (the user 2026-06-26 ultracode-dispatch rule). Authored pre-compaction.
// CONTEXT: .agent/decisions/V022-ROADMAP.md (§"RECOMMENDED v0.22 SEQUENCE" step 1 + the K-lever list)
//          .agent/decisions/V022-EXECUTION-STATE.md (the live state).
// HARD CONSTRAINT: every GPU command MUST go through scripts/with_gpu_lock.sh — agents SERIALIZE on the
//   single GPU (this workflow's value is deterministic sequencing + structured collection + synthesis,
//   NOT parallel GPU speedup). Do NOT kill 0:2's corpus wrf.exe. Measure-first; flip no default without a gate.
// CRITIC: per the model rule, each lever's verdict needs a GPT/critic pass before it becomes a v0.22 work-item.

const DIAG_SCHEMA = {
  type: 'object',
  required: ['gate', 'verdict', 'evidence'],
  properties: {
    gate: { type: 'string' },
    verdict: { type: 'string', enum: ['CRACKED', 'STILL-PATHOLOGICAL', 'INCONCLUSIVE'] },
    cold_compile_s: { type: 'number' },
    peak_rss_gb: { type: 'number' },
    runtime_s_per_step: { type: 'number' },
    identity: { type: 'string' },
    evidence: { type: 'string' },
  },
}

const R0_SCHEMA = {
  type: 'object',
  required: ['feasible', 'best_setting', 'evidence'],
  properties: {
    feasible: { type: 'boolean' },
    best_setting: { type: 'string' },
    cold_peak_rss_gb: { type: 'number' },
    runtime_regression_pct: { type: 'number' },
    aot_blob_valid: { type: 'boolean' },
    identity_tier: { type: 'string' },
    evidence: { type: 'string' },
  },
}

const K2_SCHEMA = {
  type: 'object',
  required: ['best_rung', 'cfl_safe', 'evidence'],
  properties: {
    best_rung: { type: 'string' },           // e.g. "dt_root=72s n_sound=8"
    cfl_safe: { type: 'boolean' },            // max-Courant readout stayed < target on every rung adopted
    max_courant_observed: { type: 'number' }, // the worst per-domain |U|·dt/dx (and w·dt/dz) at best_rung
    runtime_speedup_x: { type: 'number' },    // s/step vs the R1 baseline rung
    stable: { type: 'boolean' },              // finite + conservation + bounded over the test horizon
    skill_delta: { type: 'string' },          // canary skill vs baseline (operational tolerance check)
    evidence: { type: 'string' },
  },
}

const SYNTH_SCHEMA = {
  type: 'object',
  required: ['summary', 'work_items', 'impossibility_proofs'],
  properties: {
    summary: { type: 'string' },
    work_items: { type: 'array', items: { type: 'string' } },
    impossibility_proofs: { type: 'array', items: { type: 'string' } },
  },
}

// Phase 1 — the 3 INDEPENDENTLY-FALSIFIABLE compile-pathology kill-gates (critic B2: a shared SYMPTOM-class,
// NOT one proven root). Each may PASS or stay permanently opt-in/closed unless its OWN full-pipeline gate passes.
phase('Compile-pathology diagnostic')
const GATES = [
  { key: 'onz_boulac', prompt:
    'Compile-pathology kill-gate (a): O(nz) BouLac dense->O(nz) as default in the FULL operational jit pipeline. '
    + 'Reproduce the XLA "Very slow compile" fusion pathology (data-dependent first-crossing search; see '
    + 'proofs/perf/v015/boulac_nz_optimization.json). Measure full-pipeline cold compile wall + peak RSS + runtime '
    + 's/step + tiered-identity vs the frozen v0.14 manifest. PASS = full-pipeline compile <~120s AND no runtime '
    + 'regression AND tiered-identity holds. Run GPU work ONLY via scripts/with_gpu_lock.sh. Return the schema.' },
  { key: 'fp32_boulac', prompt:
    'Compile-pathology kill-gate (b): fp32-BouLac mixed fp32/fp64 fusion stall in the full pipeline '
    + '(see proofs/perf/v015/fp32_definitive_verdict.json). Measure cold compile wall + whether it stalls + runtime. '
    + 'PASS = compiles bounded + runs. Do NOT assume fp32-operational (K4) fixes the O(nz) gate — that is a separate '
    + 'hypothesis. GPU via with_gpu_lock.sh. Return the schema.' },
  { key: 'fused_rss', prompt:
    'Compile-pathology kill-gate (c): the fused-cascade cold-compile RSS (the ~60GB fused build). This IS R0 below — '
    + 'coordinate: measure the fused 3-dom + 9-nest cold-compile peak host RSS as the baseline the R0 flag-sweep tries '
    + 'to lower. GPU via with_gpu_lock.sh. Return the schema.' },
]
const diag = await parallel(GATES.map(g => () =>
  agent(g.prompt, { label: `diag:${g.key}`, phase: 'Compile-pathology diagnostic', schema: DIAG_SCHEMA })))

// Phase 2 — R0 compile-TRILEMMA: low-RAM compile + AOT cache + ZERO runtime regression via a memory-fitting flag sweep.
// Candidates: jax_memory_fitting_effort/level, jax_optimization_level, jax_exec_time_optimization_effort,
// jax_compiler_enable_remat_pass, + XLA_FLAGS buffer-assignment/scheduler memory knobs. Gate = NO significant runtime
// regression (hard-fail >1-2%) + valid AOT blob + Tier-S bitwise OR Tier-P field identity (flags can change numerics).
phase('R0 flag-sweep')
const r0 = await agent(
  'R0 compile-trilemma memory-fitting FLAG SWEEP on the fused 3-dom + 9-nest cold compile. For each setting record: '
  + 'cold wall, peak host RSS (VmHWM), serialized-blob validity (load + bit-identical), and runtime s/step (MUST stay '
  + '== fused baseline). PASS = a setting with materially lower peak RAM AND zero runtime regression AND a valid AOT '
  + 'blob. This keeps the SAME runtime executable; the only risk is "no setting helps enough" -> then it is a clean '
  + 'IMPOSSIBILITY proof + ship-the-blob stays the answer. GPU via with_gpu_lock.sh. Measure-first; flip no default. '
  + 'FIRST probe each flag against the installed jaxlib (python -c "import jax;print(jax.__version__)" then test the '
  + 'flag is accepted) — proofs/v022/GPU_OPENER_PREP.md lists the candidate flags + the getrusage(ru_maxrss) RSS '
  + 'harness in perstep_timing_driver.py + the AOT-blob path in aot_precompile.py to reuse. '
  + 'See V022-ROADMAP.md R0 + critic B1 (no "risk=0 by construction"). Return the schema.',
  { phase: 'R0 flag-sweep', schema: R0_SCHEMA })

// Phase 3 — K2 dt/n_sound CFL-first ladder (lossless real runtime win, roadmap #96). Independent of compile-pathology.
// There is NO existing max-Courant diagnostic (GPU_OPENER_PREP.md) -> the FIRST step is to add a per-domain Courant
// readout, then climb the ladder one rung at a time, each rung gated on CFL-safe + finite + conservation + bounded +
// within-strict-operational-tolerance. Lossless (same math, bigger step / fewer substeps) -> bit-identity NOT expected;
// this is a real s/step win, not an identity lever. n_sound via GPUWRF_ACOUSTIC_SUBSTEPS if plumbed (verify), dt via a
// namelist-build edit (config-level, NOT a core-dycore edit).
phase('K2 dt/n_sound ladder')
const k2 = await agent(
  'K2 dt/n_sound CFL-first ladder. COST GUARD: do the ladder SWEEP on the FAST Switzerland 128x128 single-domain case '
  + '(dt=10s, n_sound=10 baseline) for quick iteration, then VALIDATE only the best safe rung on the 3-dom Canary '
  + '(dt 54/18/6). Do NOT run the ladder on the all-7 9-nest (each step ~40min — far too expensive for a sweep). '
  + 'Read proofs/v022/GPU_OPENER_PREP.md for the rung philosophy + knob locations. STEP 1: add a per-domain max-Courant '
  + 'readout (max |U|*dt/dx and w*dt/dz) to '
  + 'the integration as evidence — there is NO existing CFL diagnostic to trust. STEP 2: climb the ladder scaled to the '
  + "case's baseline dt — Switzerland 128 sweep: dt 10->12->15->18s with n_sound 10->9->8->7; the 3-dom-Canary validation "
  + 'rung scales the root dt 54->.. by the same factor. Each rung GATED on CFL-safe (max-Courant < target) AND '
  + 'finite AND conservation AND bounded AND within strict operational tolerance vs the R1 baseline. Report the best safe '
  + 'rung + its s/step speedup vs R1 + the worst Courant observed. Lossless lever: do NOT expect bit-identity; do NOT '
  + 'flip a default — this is a measurement + a recommended config. n_sound via GPUWRF_ACOUSTIC_SUBSTEPS (verify it is '
  + 'plumbed in nested_pipeline.py), dt via a namelist-build edit (config-level). GPU via with_gpu_lock.sh; do NOT kill '
  + '0:2 corpus. Return the schema.',
  { phase: 'K2 dt/n_sound ladder', schema: K2_SCHEMA })

// Phase 4 — synthesize into v0.22 work-items + impossibility proofs (manager reviews; each needs a GPT critic before adopting)
phase('Synthesize')
const synth = await agent(
  'Synthesize the 3 compile-pathology kill-gate verdicts + the R0 flag-sweep + the K2 dt/n_sound ladder into: '
  + '(1) a short honest summary, (2) concrete v0.22 work_items (e.g. "K1 BouLac O(nz) default — compile cracked, proceed '
  + 'to identity+stability gate"; "K2 adopt dt_root=72 n_sound=8 — N% s/step win, CFL-safe"), (3) any impossibility_proofs '
  + '(the XLA mechanism that blocks it, or the CFL rung that broke). Be brutally honest per the north-star; do NOT claim a '
  + 'single-card multiplier that was not measured. Inputs: '
  + JSON.stringify({ diag: diag.filter(Boolean), r0, k2 }) + '. Return the schema.',
  { phase: 'Synthesize', schema: SYNTH_SCHEMA })

return { diag: diag.filter(Boolean), r0, k2, synth }
