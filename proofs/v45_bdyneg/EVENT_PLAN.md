# January boundary-negative attribution plan

Source base: `7b7346352af4de52de7df1ff18d3e6222125c4e1`.
No production correction is proposed before the event operands identify a mismatch.

## Smallest next evidence

The cross reports 793 negative **values across three variables**, not 793 unique
physical cells. QC_RAD is a held solver-input condensate diagnostic:
`physics_couplers._rrtmg_history_from_results` stores `sw_state.qc`; it may reflect
an earlier radiation call than the frame's QCLOUD. Count and attribute it separately.

1. Manager obtains a guarded facade copy of WATER_INVENTORY.json; extract domain,
   frame, minimum and indices if present. The cross alone gives none of those.
2. If ring indices are absent, stage only the affected history files and use
   `ring_inventory.py`. It reads one horizontal layer at a time and reports
   counts for **every ring**, the actual minimum and its k,j,i. Missing/fill/nonfinite
   data fail; an absent requested field is never silently treated as zero.
3. For the first affected step, retain entry, after force-down, post-RK,
   post-microphysics, post-final-boundary and radiation-input samples. Reuse existing
   taps/retained operands where available; a new observer can change GPU fusion
   (E242), so an instrumented result needs identity evidence.

## Distinguishing mechanisms

- Ring 0 after nested final-boundary pin: retain the two coupled boundary records,
  dtbc, parent cadence, c1h/c2h and final child muts. Compare original
  `bdy_interp1` REAL subtraction/DOUBLE reciprocal/REAL assignment, then unmodified
  `spec_bdy_final`, with the port on the **same operands**. Positive records do
  not prove the time-interpolated REAL result nonnegative.
- Ring 1..4: final `spec_bdy_final` cannot own these cells. Retain scalar input,
  complete residual stencil, frozen RK1 boundary tendency and RK mass/advection
  operands. Reuse `pristine_boundary.py` and the existing pristine scalar-chain
  wrapper; retain whole-domain bounds (E218), never reinterpret a crop as a domain.
- Parent interpolation: retain the actual coupled parent 5x5 SINT stencil and
  child subcell registration before the two-record construction. Compare original
  `sint.F`; distinguish an inherited negative from a newly negative result.
- Root flow-dependent boundary: retain ru_m/rv_m and the copied first interior
  scalar. Copy/select can propagate a negative but cannot create a new sign.
- Microphysics skips **spec_zone** at specified/nested boundaries, not the full
  spec_bdy_width=5. Its zeroing of interior cells cannot justify an upstream negative.

Original WRF sources and hashes are pinned in BD01. A separate original Swiss
reference already disproves universal nonnegativity at ring 0 (422 endpoints).
It is context, not attribution of this January case or a general F2 waiver.

## Gate and GPU work

Keep original signs/values if the same-operand pristine result reproduces the
event. A blanket clip is unsupported by `spec_bdy_final` (module_bc.F:2066-2216).
If an actual mismatch is identified, freeze the smallest correction and actual
numeric operand fixture once. The regression must kill removal of that correction
with an AssertionError, and verify unchanged inactive-path bytes/graph.

Current change is offline proof tooling only: no device launches, intermediate
model arrays, host transfers, synchronization, model compile or readiness cost.
BD02 measures the scanner at 1.744 s / 81,812 KiB RSS over five small original-WRF
files; this is **CPU diagnostic cost**, not GPU forecast performance.

No GPU run is needed for an offline-only conclusion. If a production correction
is needed, request a manager-booked window and pre-register a sealed launcher
against its final commit: actual-operand component OFF/ON (private cache per arm),
GPU backend assertion, original-WRF comparison, warmed repeated changed-artifact
timing, compiled temporary/output bytes and launch/copy counts. Cold model setup
gets a separate <=60 min arm and real-executable E41 gate; measured GPU arm <=30
min, always timeout and official v45-bdyneg lock. Run the exact launcher through
SEQ_SMOKE before admission. The manager owns any affected composed RW1–RW4 gate;
do not re-run already-proven unchanged science.

## Actual-frame follow-up (BD05–BD07)

Five selected actual C frames contain 104 negative values, all ring 0, none
in rings 1..4. This does not assert the exact rings of the remaining 689 values.
The largest d02 QCLOUD sample is at (k=0,j=62,i=266), 03:00 UTC; native dt is
27/9/3 s and simulation start is January 11 00:00 UTC. The issue-directory date
must not replace that native clock. Exact strips and readback hashes are in BD06.

Preferred acquisition is an existing authenticated checkpoint. A new diagnostic,
if needed, uses the original C inputs/argv and v0.4.4 bundle, stops at exactly
3 h (400 root steps at dt=27), and checkpoints only at the end (interval=400).
That adds no earlier segment boundary to the natural history schedule. Verify
the loaded program identities and that the repeated d02 03:00 boundary arrays
match the retained C strips before treating its records as this event's operands.
Do not start until the input/launch manifest and manager window are available.

The numerical check is a **rounding debug comparison**, using the existing
BD95 operator gate, not a new forecast tolerance:

- Verify source/input hashes, global dimensions, actual completed own steps,
  boundary configuration, represented species, finite records/masses and units.
- Use authoritative checkpoint mu_total and actual coupled two-record leaves;
  never infer records or authoritative mass from the saved history output.
- Run unmodified original-WRF spec_bdy_final via pristine_boundary.py, REAL4,
  and the exact bdy_interp1 REAL-difference/DOUBLE-reciprocal/REAL-rate convention.
- Compare the full spec zone at the frozen BD95 normal-result byte gate; report
  subnormal cases and FTZ behavior separately under the existing E117 policy.
  Reject nonfinite values over whole operands before selecting comparison cells.
- Confirm whether the observed negative is reproduced on those actual operands.
  A hypothetical clip must fail the original-value assertion if WRF is negative;
  a genuine correction must fail its gate when the correction is deleted.
- A normal checkpoint retains boundary leaves, not necessarily transient PD or
  pre-microphysics operands. It cannot by itself close root PD amplitude or
  every F2 event. Preserve that remaining scope explicitly.
