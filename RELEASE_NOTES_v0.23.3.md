# wrf_gpu v0.23.3 — cuda_async preflight false-negative fix

**Type:** reliability point-release. **No dynamics, no physics, no default numerical
change** — every forecast is byte-for-byte identical to v0.23.2. This release only
changes *when the nested-GPU preflight fails*.

## The bug
Nested GPU chunks of an overnight production run died immediately with
`rc=75: nested GPU preflight failed: free VRAM 3.23 GiB is below resolved threshold
24.00 GiB` — on an **empty card** (the `with_gpu_lock` wrapper read 27.9 GiB free at
the same moment). This was the first night on the new `GPUWRF_ALLOCATOR=cuda_async`
default.

## Root cause
The preflight measures free VRAM with `nvidia-smi`. `cuda_async` (JAX/XLA's
stream-ordered allocator) reserves its device memory pool during backend init —
**before** the preflight's `nvidia-smi` read. So the "used" VRAM the preflight
observed was **our own allocator pool**, not another process. The check could not
tell "another job is hogging the GPU" from "our own cuda_async pool is warming up,"
and fail-closed on the latter.

## The fix
When the GPU lock is **verifiably held** (`scripts/with_gpu_lock.sh` owns the card
exclusively *and* already checked free VRAM pre-launch, before this process
allocated its pool), a low free-VRAM reading is downgraded to an **advisory**
(logged, recorded in `payload["advisories"]`) instead of failing `rc=75`. Rationale:
under an exclusive lock there is no other process to contend with, so any "used" VRAM
is ours. If our pool is genuinely too large for the run, allocation fails later with
a real OOM — surfaced honestly at that point, not pre-empted by a false-negative.

**Unchanged:** with no lock held, a low free-VRAM reading is still a hard `rc=75`
fail-closed (genuine contention). `--force-gpu-run` / `GPUWRF_FORCE_GPU_RUN` still
override everything.

## Impact
Restores the faster `cuda_async` allocator for locked production runs (the `platform`
fallback was the only workaround, and it is measurably slower). Operators can return
`gpu_allocator` to `cuda_async`.

## Validation
- Unit: `tests/test_gpu_preflight.py` — 8/8, incl. a regression guard reproducing the
  incident (lock-held + 3 GiB free → advisory PASS) and the preserved gate
  (no-lock + low VRAM → FAIL).
- GPU: a `cuda_async` nested run under `with_gpu_lock` passes the preflight (advisory)
  and completes finite. (Gates the tag.)

## Files
`src/gpuwrf/runtime/gpu_preflight.py`, `tests/test_gpu_preflight.py`.
