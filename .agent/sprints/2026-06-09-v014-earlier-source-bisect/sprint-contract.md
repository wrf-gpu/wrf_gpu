# Sprint Contract: V0.14 Earlier-Source Bisection

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Bisect the source of the bad h10 d02 carry before d02 completed step 5997. The
previous sprint proved the final partial parent/child subcycle is downstream of
the error; this sprint must decide whether `T/P/PB/MU/MUB` are already wrong at
native load / initial carry, become wrong during an earlier replay segment, or
are blocked behind a missing source hook.

## Inputs

- `proofs/v014/previous_step_handoff_bisect.json`
- `proofs/v014/previous_step_handoff_bisect.md`
- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/base_state_writer_attribution.json`
- `proofs/v014/static_metric_base_parity.json`
- `proofs/v014/jax_h10_prestep_carry_producer.py`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- Native L2 run directory under `/tmp/v0120_merged_run_root` or
  `/mnt/data/canairy_meteo/runs/wrf_l2`

## Write Scope

Repository write scope:

- `proofs/v014/earlier_source_bisect.py`
- `proofs/v014/earlier_source_bisect.json`
- `proofs/v014/earlier_source_bisect.md`
- `.agent/reviews/2026-06-09-v014-earlier-source-bisect.md`

External scratch write scope:

- `/mnt/data/wrf_gpu2/v014_earlier_source_bisect/**`
- fallback `/tmp/wrf_gpu2_v014_earlier_source_bisect/**`

Default rule: no production `src/` edits and no WRF source edits. If a deeper
hook is required, emit a blocked JSON naming the exact hook and source file
needed; do not add the hook in this sprint.

GPU use is allowed only for targeted replay/snapshot probes if CPU replay is
not practical. Do not run TOST, Switzerland validation, broad validation
campaigns, or FP32 source work.

Do not use Hermes, Telegram, `ask-hermes`, or any human-notification bridge in
this sprint. If blocked, record the blocker in the sprint artifacts.

## Required Work

1. Load the native L2 domains through the same producer path used for the h10
   checkpoint producer.
2. Capture compact d02 target-leaf snapshots for `T/P/PB/MU/MUB` at the
   earliest reachable surfaces, including:
   - immediately after native domain load / initial carry construction;
   - after the first replay segment;
   - coarse segment boundaries before d02 step 5997, using a context-sparing
     schedule such as d02 `0, 600, 1200, ... 5400, 5997` if practical;
   - any narrower boundary where the first bad transition appears.
3. Compare each snapshot against the appropriate CPU-WRF/native truth available
   from existing proof artifacts. If exact same-step CPU-WRF truth is missing,
   classify what can still be proven without inventing a tolerance, and name
   the exact WRF savepoint/hook needed for the next sprint.
4. Specifically decide whether `PB/MUB` are already wrong at initial/base-state
   construction or become wrong during replay.
5. Classify the result as exactly one of:
   - `BAD_AT_NATIVE_LOAD_OR_INITIAL_CARRY`
   - `BAD_AFTER_FIRST_REPLAY_SEGMENT`
   - `DRIFTS_BETWEEN_SEGMENTS_<range>`
   - `BASE_STATE_SPLIT_DEFINITION_MISMATCH`
   - `EARLIER_SOURCE_BLOCKED_<reason>`
   - `REPRODUCER_MISMATCH_<reason>`
6. State the next decision narrowly: source-changing fix sprint target,
   smaller hook/savepoint sprint, or escalation after repeated failed attempts.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/earlier_source_bisect.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/earlier_source_bisect.py
python -m json.tool proofs/v014/earlier_source_bisect.json \
  >/tmp/earlier_source_bisect.validated.json
```

If GPU replay is required, record the exact command and environment in JSON,
including `CUDA_VISIBLE_DEVICES`, JAX backend, allocator settings, peak VRAM if
available, and why CPU replay was not practical.

## Acceptance Criteria

- JSON validates and top-level output is compact.
- No source fix is landed.
- The proof separates native-load/base-state mismatch from replay-time drift.
- The proof does not use JAX-vs-JAX self-comparison as a CPU-WRF truth
  substitute.
- If blocked, the JSON names the exact missing artifact/API/hook and the next
  command needed.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, GPU use if any, and next decision.
