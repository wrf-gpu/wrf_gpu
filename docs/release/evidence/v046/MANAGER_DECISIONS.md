# v0.4.6 candidate decisions and evidence

This is a compact public-facing draft of the decisions recorded during the
candidate. The original full reports and raw verdicts remain separate.

- **RW4 target-specific key comparison**, 1 November 2026: replay on three
  dates plus shifted input, pairwise CPU-target key equality, and reuse of
  sealed GPU keys satisfy RW4 as in v0.4.5. CPU-target and GPU-target keys are
  not compared directly. Original decision SHA256:
  `9f918fdf80f831165349a438b33e68d78cb61e05d9481dc4088d2e5c4460fbdf`.
- **RW3 QGRAUP trace disclosure**, 2 November 2026: the frozen A3 floor passes
  with five FLOOR-LIMITED and zero REAL FAIL, while final-frame QGRAUP output
  integrity remains false. The manager retains the v0.4.5 trace disposition;
  the 71–72 h cause remains open. Original decision SHA256:
  `bcb6a102173c66801561b9280c6d700ac3f014c737f2ee596a87b84548cb65db`.
- **Pre-existing CAM-MG resume failure**, 31 October 2026: the missing-manifest
  failure reproduces on the published v0.4.5 source and is not classified as a
  v0.4.6 regression. It remains a real pre-existing failure and was dispositioned
  as not release-blocking. The reader-inventory literal correction was reviewed.

The numerical [release notes](../../../../release_notes/RELEASE_NOTES_v0.4.6.md)
identify raw D6, floor, output integrity, regression comparisons, speed and
memory separately. E41 native INCOMPLETE and the verdict-reducer defect are
explicitly disclosed there.
