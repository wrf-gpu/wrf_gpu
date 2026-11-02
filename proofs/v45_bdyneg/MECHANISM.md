# Boundary negatives: evidence and remaining operands

The h3 replay reproduced the d02 event and classified it as **INHERITED_RECORD**:
its upper coupled QC record is already negative and pristine WRF's final pin
propagates it exactly. H1 creation at the final pin is refuted. Upstream record
creation and Root-PD remain **OPEN**. There is no model fix in this lane.
See [the actual-operand result](H1_RESULT.md).

The five retained history frames contain 104 negative samples: d01 QCLOUD1,
d02 QCLOUD2/QRAIN96/QC_RAD4, d03 QCLOUD1. Every sample is on ring0, has a
nonnegative final inward neighbor and a positive history-reconstructed
`C1H*(MU+MUB)+C2H`. Same-frame face velocity indicates outflow for the root
sample, 1/2 d02 QCLOUD, 60/96 d02 QRAIN, 1/4 d02 QC_RAD and 0/1 d03 QCLOUD.
These are measured final-history observations, not stage-transport measurements.
QC_RAD is a held radiation-input diagnostic, not another transported water field.
See [BD23](receipts/BD23_HISTORY_MECHANISM.json) for per-sample values and hashes.

The positive reconstructed denominator excludes a denominator sign flip in that
history reconstruction. It does not authenticate history mass as `state.mu_total`.
Final face velocity and final inward water cannot reconstruct `flow_dep_bdy`:
the routine consumes stage/acoustic transport and water before microphysics.
Microphysics subsequently skips the outer specified cell, but may change its
inward neighbor. The original first-root CPU checkpoint has no negative nested
endpoint numerators or water fields; it does not reproduce the later events.
Its carried `mu_total` and scratch `muts` differ on ring0 at376/576/305 cells
in d01/d02/d03, with maxima0.296875/0.015625/0.015625 Pa. These are different
captured operands; substituting scratch mass in the operator comparison is not
justified. See BD29; this first-root observation does not classify January mass.

The expanded BD07 history fingerprint is also non-universal: of96 d02 QRAIN
samples,28 inferred inverse numerators are exact powers of two and37 are within
one float32 ULP. This is consistent with cancellation in some samples, not
proof of a specific operator or a recovered boundary record (BD28).

Pristine `share/module_bc.F:2066–2216` evaluates `spec_bdy_final` as REAL
`(bdy + dtbc*bdy_tend)/(c1*mu+c2)`, without a positivity clamp. Its Y sides own
corners. `solve_em.F:4714–4733` applies the final nested moist-species pin.
The shipped B45 root policy already uses flow-dependent boundaries for QC..QG
and number species when no moist boundary files exist. Historical B45 therefore
does not by itself identify a current bug. Original CPU-WRF Swiss evidence
already contains ring0 negatives; it is a counterexample to universal CPU-WRF
nonnegativity, not a classification of the January events.

The unchanged-v0.4.4 three-hour replay retained actual two-record `*_bdy`,
authoritative `mu_total`, scratch `muts` and hybrid coefficients. At the target,
the lower record is0, upper−2⁻²² and denominator97465.890625; pristine and port
both give−2.4461745563170156e−12. A fresh CPU reference replay rejects both a
no-op oracle and a blanket clip. History-derived inverse numerators are not
substituted for these authenticated records. Root step267 stage operands remain
absent; the final step400 snapshot cannot close that event's PD attribution.

No timestep arithmetic, device allocation, kernel launch or model source changes.
Launcher changes affect host admission, namespace ownership, environment and
cleanup only. Native acquisition accounting and operator fidelity are recorded
in H1_RESULT; forecast performance/VRAM are not inferred from those figures.
E41 remains INCOMPLETE despite hard-stack PASS.
