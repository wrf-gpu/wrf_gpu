# launch_v6: ALISIOS hypothesis, stop rule and raw-disposition proposal

Status: proposal for ALISIOS and cr-w44-launchers delta acceptance; **not a GO**.
Authority: manager MAILBOX 2026-10-31T01:09:45Z, ALISIOS agreement in principle
for one diagnostic of at most20 min, its own GO after the critic delta, GPU end
strictly before19:15Z. The existing sealed ceiling is18:45Z and remains unchanged.

Binding: `launch_v6_v046/plan.json` SHA
`fe42ac3c520c8e884a49fc2f0b5f7a8c0d12454c8c02d9a4c4cd6de2ccf0a9e8`,
seal `4652583ebb2fb07f6790012b4e6dc156343bf816fc20b4beea7af3f867772eb9`,
entry `92526edf588ad8f4eb7d6a5a3872308ed7f8b90f740a3637d094c87b95f62650`.
Original v0.4.4 source/input/physics/compile mode and all caps stay fixed.

## Hypothesis

H1: the d02 QCLOUD ring0 negative at `2026-01-11_03:00:00`, zero-based
`(k,j,i)=(0,62,266)`, original value `-2.4461745563170156e-12 kg/kg`, is created
by REAL4 endpoint cancellation in the final specified-boundary pin, and the
unmodified pristine-WRF operator gives the same negative on the same operands.

At this child step1200, parent cadence and endpoint `dtbc` are both27 s. Using
the actual two retained coupled records, evaluate WRF's REAL subtraction,
DOUBLE reciprocal-time multiplication, REAL retained tendency, then REAL
`bdy + dtbc*bdy_tend` and division by `c1h*mu_total+c2h`. H1 requires finite,
nonnegative endpoint records, a positive denominator, a newly negative numerator,
and the frozen BD95 normal-result byte agreement with pristine `spec_bdy_final`.
Retain scratch `muts` separately; do not substitute it for the declared mass.

Actual h3 target reproduction is checked against the retained original C strip.
That identity is GPU regression evidence; the pristine-Fortran same-operand
comparison supplies operator fidelity. If an endpoint record is already negative,
the final pin can propagate it: classify **INHERITED_RECORD**, which refutes H1
as a creation mechanism and leaves its upstream source open. If the exact target
does not reproduce, classify **EVENT_NOT_REPRODUCED** and do not attribute the
original event from a different negative. These decisions cover this nested
event only. A normal checkpoint does not retain root transient PD/pre-MP operands.

## Stop rule

- One unchanged C replay, exactly3 forecast hours: own steps400/1200/3600 at
  dt27/9/3 s. One final checkpoint at root step400. No retry, extension to later
  frames, physics/compile-mode change, extra GPU arm or blanket clamp in this GO.
- Model phase cap840 s; complete flow1170 s; outer entry1200 s. The sequencer
  needs its usual2 min admission reserve outside the20 min entry cap. Existing
  manager GA/Nightly-DONE/leases/official lock and resource checks remain binding:
  DATA start435 GiB, floor430, own peak5 GiB; ROOT20 GiB+256 MiB; native RAM16G,
  swap0. No new DATA before GA. This packet grants no window or resource change.
- Missing/changed inputs, wrong backend, nonfinite operands, checkpoint/hash/
  readback failure, explicit E41 FAIL, oracle normal-byte mismatch or supervisor
  failure yields a failed/incomplete diagnostic and no further GPU work. Finish
  bounded CPU evidence retention/cleanup and record the failure honestly.
- After the one replay and bounded CPU post, close as **H1_SUPPORTED**,
  **INHERITED_RECORD**, **ORACLE_MISMATCH**, **EVENT_NOT_REPRODUCED**, or
  **FAILED_INCOMPLETE**. No automatic follow-up even if the answer is inconclusive.
  H1 support is not a global F2 waiver or a root-PD diagnosis.
- Keep the sealed18:45Z end ceiling. A later window would require a newly reviewed
  seal/plan; this addendum does not extend it toward the ALISIOS19:15Z hard limit.

## Raw-disposition proposal

Owner/deletion executor: wrf_gpu2/v45-bdyneg, using the sealed controller's owned
tree cleanup. Target is **only**
`<DATA_ROOT>/wrf_gpu2/scratch/v45-bdyneg/diagnostic_20261031_v6`.
This includes our history/scratch/checkpoint and private cache/overlay copies.
Original ALISIOS inputs, original C captures/cache donor, sealed bundles and
CPU-WRF references remain outside the cleanup target.

For a completed extraction, the controller verifies the authenticated checkpoint,
retains full boundary operands and coefficients in compact NPZs, verifies every
array byte-for-byte, and publishes `POST.json` before cleanup. It reaps its owned
children, verifies the ownership marker, restores owner permissions only within
the owned tree, and removes that tree with the marker last. This remains valid
when the retained comparison reveals a scientific failure: successful capture
verification is distinct from hypothesis success. Small original-operand NPZs,
source/checkpoint manifests, logs and verdicts stay on HOME.

After entry completion, the worker performs CPU-only disposition verification:
read `RESULT.json` and `POST.json`, rehash every retained NPZ, check model/post
owned-child teardown and the exact raw-tree absence, then append and fsync one
record to `<USER_HOME>/wrf_gpu2_lanes/v45-bdyneg/RAW_DISPOSITION_JOURNAL.jsonl`.
Record run ID, exact tree path, seal/plan hashes, actual model/post return codes,
verification outcome, RESULT/POST/checkpoint-manifest hashes, retained paths/
sizes/hashes, and observed deletion state. The journal timestamp is the recording
time; an exact unlink time is not inferred. Granularity is one owned run tree;
retained scientific products have individual SHA256 hashes. The existing
controller performs the deletion; the worker confirms and journals its result.

If model success has no verified compact extraction, the controller holds the
tree/checkpoint for CPU recovery. Journal **HOLD_FOR_VERIFIED_EXTRACTION** with
`raw_deleted=false`; no verified-cleanup claim or new GPU run. Failed incomplete
model runs already removed by the controller are journaled as
**FAILED_WITHOUT_VERIFIED_CAPTURE**, with their actual deletion state and retained
failure logs. No scientific-success claim is made for such a run. Every held
tree stays charged to the existing budget and needs an explicit recovery/
disposition before closure. We do not wait for ALISIOS to delete our own raw.

This is a packet/procedure addendum; it changes no sealed executable, cap or
model program. A journal rehearsal on the completed BD22 CPU run demonstrates
the success verification/record format without deleting any new files. Its row
is explicitly CPU evidence. Native disposition remains pending the future GO.
