# H1 result: INHERITED_RECORD

The exact d02 QCLOUD event at `2026-01-11_03:00:00`, zero-based `(0,62,266)`,
reproduces. **H1 is refuted as creation at the final boundary pin:** the upper
mass-coupled boundary record is already negative. Pristine WRF propagates the
same sign and produces the same normal float32 value on these actual operands.

| Captured quantity | Value |
|---|---:|
| Lower coupled QC record | 0 |
| Upper coupled QC record | −2.384185791015625e−7 = −2⁻²² |
| Parent cadence / endpoint dtbc | 27 s / 27 s |
| Retained REAL4 tendency | −8.830317810293309e−9 |
| Endpoint increment / numerator | −2.384185791015625e−7 / −2.384185791015625e−7 |
| Positive hybrid mass denominator | 97465.890625 |
| Original C, new GPU and pristine result | −2.4461745563170156e−12 kg/kg |
| Result float32 bits | `ac2c225d` |

The numerator equals the captured upper record exactly. Replacing endpoint
interpolation by that upper record still gives the same negative value. The
data therefore locate the negative upstream of temporal interpolation/final
division. They do not identify which operation created the upper record.
The force-down producer couples the parent scalar and interpolates it into this
upper record (`nesting/boundary_construction.py:708–724`); a same-operand pristine
spatial-interpolation comparison is still needed to classify its creation.

The targeted CPU replay runs unchanged pristine `spec_bdy_final` over the full
44×117×267 domain with actual boundary records, mass and coefficients. Its output
spec zone starts at zero, ensuring the reference must write it. It gives zero
normal-byte mismatches, one negative and no subnormal references on ring0.
A no-op reference fails with128 numerical mismatches; a blanket-clip mutant
fails with one genuine AssertionError. Source/routine hashes match the native
post's pristine build. The completed post's16 species/domain comparisons also
have zero normal-byte mismatches; those unchanged comparisons are reused.
Original-C equality is regression evidence; pristine same-operand replay supplies
the operator-fidelity evidence. See [BD32](receipts/BD32_H1_ACTUAL.json).

## Root-PD disposition

**OPEN_NO_SAME_STEP_PD_OPERANDS.** The reported d01 event is step267 at
02:00:09. The retained snapshot is step400 at03:00:00. That snapshot has one
root QCLOUD negative, `(0,0,73) = −7.568344935325776e−24`, also ring0; it is a
different event. Neither snapshot-final water nor original final histories
provide step267's pre/post-PD scalar, complete residual stencil/tendencies,
acoustic `ru_m/rv_m`, copied inward source and pre-microphysics water.
The v0.4.4 normal scratch serialization lists14 carried fields; the retained
compact leaves contain no such same-step stage operands. Root `*_bdy` presence
alone does not establish that QCLOUD uses a specified-boundary pin.

No root-PD attribution, global F2 waiver, or blanket clamp follows from this
result. Preserve the compact operands and original strips. A root-stage capture
needs a separate admission and a source/identity plan; no follow-up GPU run is
started here. See [BD33](receipts/BD33_ROOT_PD_DISPOSITION.json).

## Acquisition and disposition

The manager-run native arm completed with model/post rc0, verified checkpoint,
all-array BYTE_EXACT readback and actual GPU backend. Its134.82 s total and
2,103,635,968 B sampled peak are acquisition wall and named-disk accounting,
not a forecast benchmark or VRAM figure. Owned raw was deleted after compact
verification; retained NPZ hashes, child teardown and actual tree absence were
rechecked and fsynced into the native raw-disposition journal (BD34).
Original inputs, donor caches and CPU-WRF references remain preserved.

E41 remains **INCOMPLETE**, with hard-stack PASS. The scientific finding does
not convert that to native qualification. No production arithmetic, kernel,
allocation, transfer, precision or physics setting changes are proposed; this
proof adds no timestep cost. Temporary oracle builds/snapshots are deleted.
