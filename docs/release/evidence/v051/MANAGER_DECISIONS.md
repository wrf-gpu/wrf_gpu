# v0.5.1 decisions and evidence

A compact public summary of the decisions recorded during the v0.5.1 release candidate (16 November 2026).
The full reports and raw verdicts remain separate; SHA256 values identify them.

- **72-hour Tenerife check, fidelity:** PASS with disclosure. The frozen floor rule passes with 0 real failures
  and 6 floor-limited rows (d03 accumulated rain, hours 67–72). The only output-integrity failure is the known d03 graupel
  trace class at hour 72, so the v0.5.0 rule for that class applies (disclose, cause open). Manager decision SHA256
  `eefdf6c3f0684254e52dd76e6a541c446eb3a9e6024ac5eed88330d8b4035fdc` and its addendum (correcting an unproven explanation of
  the rain difference) SHA256 `a22942d7a25aed20b50a3b7240465f672bdb97ba43a8c77f5312b360d6ae4b52`; gate disposition SHA256 `94c597781517af5f35bf2b3a13b2ecceb2411be1805e57eaaa490808cbcf0e89`.
- **Operational user's verdict on the release-candidate checks:** pass with disclosures, with one open point on the rain
  difference, SHA256 `b995ee212b40fc0d35fbc964e6cacbb2b1ed412eb7e30bae6edf22e90aa49ec4`.
- **Rain difference resolved as a compile effect:** all six programs of the 72-hour case are the same lowered programs in
  v0.5.0 and v0.5.1, the start frame is byte-identical, and the runs differ from the first forecast hour on. The
  +3.8 to +19.1 % is one pair of compiles, not a bound; no same-build twin was run. Operational user's decision SHA256
  `4df8ed3c37bab66959eb3a33af1fea50e0b94d1a277f5c5ad4e5a4dcd2d1d237`.
- **Compiled resources:** same scanner rule as v0.5.0 (INCOMPLETE accepted only for compact AOT metadata, every other gate
  passing). The nine-nest program keeps the two known input-transpose fusions disclosed in v0.5.0.
- **Nine-nest qualification (17 November 2026): undecided (UNENT), no switch of the operated nightly.** Operated setup → v0.5.1 off
  56/60 cells green (four elevation cells with one yellow case, three from the winter case), off → on 60/60 green.
  - Verdict SHA256 `d86ded6bedd1d8b66a33c4450d229d28278dc8041d554fc894de652febe77ac8`.
  - Evaluator eval_qual17 v1.6 `035f3c79977cefab92a15c91650e289a348d58d320a35eab7a785b2cbea94d1e`.
  - Criteria addendum (N033 cap per arm) `e1fea2e3317c8e37a092a0afa0059d81d71303c32bb0e6ca97af99549598fefb`.
  - TECH_SUMMARY `5fcdab3deb7a322db4a1b25a92dff772f7d651e5c82e20092e0025a4808a1546`, TECH_CHECK
    `bb175912faca9590a703b94ee8b150372814837641d0d2521f3eeb025c62afbe`.
  - Both operated-setup (N033) arms ran as second attempts (a2) after clean pre-start space refusals of the first attempts.
  - Dew point at elevation and the winter case are not cleared for serving; a transfer test against station measurements is pending.
