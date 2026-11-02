# C boundary-negative diagnostic launcher

This is an operand-acquisition/debug job, not a model correction or release
fidelity/performance result. No file under `src/` changes.

## Frozen work

- Model: unchanged v0.4.4 tag `fcf12bc229`, src tree `a22dea711202a48d00c86eafa34391ff2cb60fe2`.
  Use the original canonical RC bundle path, whose source equals the sealed
  v0.4.4 release bundle. A private bundle-consumer overlay receives every write.
- Original C input directory is read in place. All eight file hashes are pinned
  in BD08; namelist SHA starts `5963ad19`. Start January 11 00:00 UTC,
  dt=27/9/3 s, three nests, icloud=1, rad_diag=1, original physics and compile mode.
- Native CLI changes only forecast duration (3 h), private output/scratch paths
  and final checkpoint controls. Checkpoint at root step400; no earlier checkpoint
  boundary is added. Original-input directories are OS-mounted read-only.
- The paid cache seed contains only C step AOT keys/aliases and auxiliary caches
  from the original C/R layer. R stepping AOTs and redundant stepping JCT files are
  excluded. About354 MB unique is copied after GA_START into the private layer;
  aliases link only our own copied inode, never the donor or the sealed bundle.

## Manager admission

`prepare.py --commit <commit> --deploy <ROOT path> --cache-seed <manifest>` exports
committed helper blobs, resolves a full commit, and seals tools/dependencies.
The generated `entry.sh` and its SHA are identical in these two plans:

- `plan.json`: native window step, cap20 min. Model cap840 s, complete flow1170 s,
  outer timeout1200 s, MemoryMax16G/Swap0. These native limits stay unchanged.
- `CPU_SMOKE_PLAN.json`: manager V046 2026-10-30 23:51Z CPU-only slot32, cap65 min,
  model cap3600 s, complete flow3900 s, MemoryMax32G/Swap0. All scratch on HOME;
  CPU bubblewrap mounts **all <DATA_ROOT> read-only**. No de-fuse flag is set.

The native entry requires GA_START/GA_END covering the complete20 min and ending
by the proposed 2026-10-31 18:45Z ceiling, GA_GATE_ID=`v45-bdyneg-night`, a manager Nightly-DONE receipt at the
generated path, an unexpired lane GPU lease, the main-pinned official wrapper,
and a manager CPU lease for3,19,13,29 aged>=30 s. It checks source/input hashes
before creating DATA. Initial DATA admission is floor430GiB+authorized5GiB.
The manager books the window and writes the Nightly-DONE receipt; workers do not.
The Oct31 ceiling is a proposal, not a granted window. Expired deployments are
preserved unchanged; the V046 successor is `launch_v6_v046`, with a fresh one-shot
DATA path. Its entry clears the environment before the first Python process and
sets umask077. Scientific code and compilation mode remain v0.4.4.

## Execution and evidence

The stdlib controller starts the actual ALISIOS registry event, supervises an
owned model process group, then releases the official GPU lock before running
the separately supervised CPU post phase. The original bundle consumer checks
the real sealed bundle and classifies its private writes. The model wrapper
asserts the backend in the actual CLI process after allocator setup.

Post-review bootstrap correction: the original v3 CPU path used a single bwrap,
but the GPU path nested that inside the bundle consumer's bwrap. This host
rejects nested unprivileged namespaces. The corrected CPU and GPU paths both
use the real consumer as the sole namespace owner; read-only payload mounts
and cwd are options of that same bwrap. No inner bwrap remains. Host integration
tests run the real consumer, preserve an opaque token fixture, and prove that
re-inserting the old nested bwrap fails with the observed namespace denial.
The model package and all compile-mode/physics flags remain unchanged.

The checkpoint is read by the original authenticated reader with `device=False`.
Post checks domain steps/dt/input hashes and finite arrays, retains actual coupled
boundary records, state fields, authoritative mu_total, scratch muts and input
c1h/c2h, and reads every compressed array back byte-for-byte. Whole-domain
dimensions remain intact. History-derived inverse numerators are never operands.

For nested domains, unchanged pristine-WRF spec_bdy_final runs on these actual
inputs at the existing BD95 rounding-debug gate (normal-result bytes; E117
subnormals separately counted; whole operands finite first). Nonzero normal
mismatches fail the post phase. A real-cache E41 scan records PASS/FAIL/INCOMPLETE
from JSON; rc2 is never blindly converted to PASS, and explicit FAIL is red.

Saved-C boundary equality is GPU regression/determinism evidence only. A normal
checkpoint does not guarantee transient pre-microphysics/PD operands. Root PD
amplitude and unobserved events remain OPEN even if nested boundary debug passes.

Registry END occurs after teardown and actual cleanup. Ordinary successful
readback removes all own raw/checkpoint/cache files; small verified operands and
receipts stay on HOME. If post fails before a complete compact receipt, the sole
verified checkpoint is held explicitly for its next CPU consumer (raw_deleted=false),
not silently destroyed. No foreign process or original ALISIOS/bundle file is
modified. Disk figures are labelled sampled named-inode accounting; unlinked-file
attribution is not fabricated as zero. The genuine5GiB cap/global floor is enforced.

Mode-specific ROOT admission preserves the20GiB global floor: CPU smoke adds
3GiB for its local compilation cache; native adds256MiB for compact operands
and receipts because its large working files are on DATA. Tests cover the real
22GiB failure class. Owned overlay work directories may be mode000: cleanup
restores only owner rwx bits under the verified owned tree and removes its
ownership marker last, so partial cleanup remains safely retryable.

## CPU smoke and review

Run the generated CPU plan through the actual manager `night_seq_v9.sh` with
SEQ_SMOKE=1. The same CLI requests3 h; only `_steps_by_domain` bounds actual work
to1/3/9 steps. Physics, inputs and fused compilation stay unchanged. Reserve20GiB
on HOME substitutes for the native430GiB DATA reserve. Private registry start/end
uses the live parser; all GPU-native gates are NOT-APPLICABLE.

The initial16GiB attempt hit a proven cgroup OOM. The first manager-granted32GiB
attempt compiled both programs but exceeded the too-short840s CPU phase cap.
Both were red, fully reaped/cleaned, and logged BD10/BD11. The final32GiB/60min
CPU grant is explicit; the native20min window plan never expands. Temporary paid
CPU cache remains only through the final dependent readback and is then removed.

Host controls exercise real private registry writes, admission, read-only child
write rejection, explicit-FD inheritance, and success/failure/timeout teardown.
Removing the input mount is killed by a genuine AssertionError (child exit42),
not by a setup exception. Full archive collection and exact sequencer receipt are
required before READY-FOR-CRITIC; the lane scientific gate is distinct from
launcher readiness.

`registry_schema_probe.py` evaluates the hash-verified controller's native
registry metadata expressions through live `register.main`, intercepting only
the final append. It checks successful and failed terminal schemas and rejects
an invalid forcing enum before append. The exact CPU smoke separately exercises
real private-atlas state transitions. `delta_evidence.py` identifies unchanged
model/post/consumer code and CPU arguments; it never claims an older smoke is
a newly executed call.
