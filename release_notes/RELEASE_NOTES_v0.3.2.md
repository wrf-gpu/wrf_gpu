# v0.3.2 — layout stability, faster startup and long-run throughput

Numerical freeze: internal revision `3e194296c`. Packaging version 0.3.2 is
assigned separately without changing the hot model. Original-WRF validation,
regression identity and timing evidence have distinct scopes.

## Changes

- **Nested layout pin, default on.** A stable layout boundary reduced the
  measured ordinary device budget by about 0.91 ms/root in its registered
  component gate. The final composed run supplies the release result; the
  component gain is not multiplied into other improvements.
- **Momentum fusion.** The accepted momentum update reduces device work behind
  the retained interfaces. Two-domain PROD is bitwise versus v0.3.1. The
  three-nest fused cascade has small child differences, handled by the
  pre-registered Tier-P original-CPU gate rather than a blanket bitwise claim.
- **Startup and cache paths.** Dormant aerosol/BMJ initialization is lazy, and
  Pallas configuration paths are canonicalized. The measured relocation probe
  recorded 1,185/1,185 cache hits, zero misses and both required AOT hits. A new
  version/geometry still needs its first compile.
- **EOS metadata pruning remains off.** The N=3 test was bitwise but its −0.14%
  aggregate change did not exceed the 1.17% repeat spread. No gain is claimed.

The v0.3.1 WRF-order, snow/water, open-top/boundary and writer fixes remain.
No new physics relaxation or masking correction is introduced for speed.

## Measured performance [M]

The accepted matched 24-hour Canary pair is **3.669 → 3.305 s/forecast-hour**,
whole process, a **B/A ratio of 0.901 (9.9% less wall time)**. The new process
took 79.319 s; steady-segment median is 2.625 s/hour and pre-step work 15.008 s.
The host was contended, so the absolute new rate is disclosed as conservative;
it is not a quiet-host timing result. The original unpaired S1 attempt had 11.9%
A/A variation and is invalid for release speed claims. Historical v0.3.1's
3.604 s/hour isolated run is not the matched baseline.

Four independent WN3 Tenerife three-nest forecasts, **24 h each on one RTX
5090**, give **5.507 s/case-forecast-hour whole-run aggregate (22.4× CPU-WRF
3×4-core throughput)**. The measured all-four-running steady window is
**4.857 s/case-hour (25.4×)**. Start-up/load and complete history are included;
output was not compressed. Different duration prevents using the old six-hour
7.77 s/case-hour as a pure code-speedup denominator.

Warm per-PID allocation is 5,032 MiB, reported host high-water approximately
5.6 GB/case. The common 0227 run retained 75/75 history hashes against cold and
validation output. That is a regression control, not independent fidelity.

Timestamped E158 board energy is **0.678 Wh (2.441 kJ)/case-hour [M]**.
Adding the explicit 4/12 × owner-reported 200 W host-share assumption gives
**0.780 Wh (2.808 kJ) [I]**, versus CPU **6.86 Wh (24.68 kJ) [I]**. No idle
subtraction is used for the headline, and no whole-node/package-power measurement
is asserted. See [measurement evidence](../docs/release/V0.3.2.md).

## Validation and disclosures

- PROD 24 h: **48/48 history files bitwise versus v0.3.1**, with zero original
  CPU-WRF D6 failures at 6 and 24 h. Its numerical errors therefore retain the
  v0.3.1 values: worst 24-hour PSFC RMSE 1.550/1.547 Pa on d01/d02.
- WN3 0227 24 h: d01 bitwise; d02/d03 differ at round-off level and all three
  domains pass the frozen original-CPU D6 limits. d03 worst hourly RMSE is T2
  0.144 K, U10 0.337 m/s, V10 0.391 m/s, PSFC 8.657 Pa and RAINNC 0.060 mm.
  PSFC is 8.66 Pa versus v0.3.1's 4.49 Pa, below the frozen 120 Pa limit.
- All-frame raw integrity retains the same registered R2/R5 fields and frames
  as the earlier candidate: d01 QICE 3/25, QSNOW 5/25; d03 CANICE 3/25,
  QSNOWXY 4/25. Registered-exception acceptance is distinct from a raw clean scan.
- Actual compiled-program hard stack checks pass; conservative flags remain
  documented and source/program-scoped. The independent critic returned
  **RELEASE-OK**: actual PROD/WN3 programs have STACK at most 32 B;
  conservative CALL/kInput flags remain, with 40 root callers, no nested
  calls and a 475-instruction leaf bound. This is not a blanket flag-free claim.

Swiss nested winter and bundled-demo measurements remain explicitly v0.3.1;
they are not rebranded as new v0.3.2 runs. Longer 72-hour Tenerife and 162-hour
Canary studies remain v0.3.0 evidence, including the published long-lead frame
failures. Dedicated glacier runtime and MPI/multi-GPU domain decomposition are
still unimplemented.

## Best-case scaling [I]

The ensemble and giant-domain figures use the **new 24-hour measured anchors**.
They show a configurable 32-member × 72-hour ensemble and the estimated largest
single square 44-level domain per GPU, with explicit 1 km/6-second timestep.
All completion times and giant-domain capacities are inferred, including
72-hour durations based on the measured CPU/RTX rates.

The best-case assumptions are warm programs, adequate host resources, steady
saturation and output bandwidth. Start-up per case stays unscaled but amortizes
over 72 h. Gross allocated bytes/cell include pools/context/scratch; a linear
capacity estimate does not establish that a giant domain fits or passes physics.
H100/B200/B300 transfer historical B200 bandwidth efficiency to a new kernel
mix; every data-centre number is [I].

**B200 measured on earlier versions; H100/B200/B300 extrapolated to v0.3.2,
best case, not measured.** [The method](../docs/release/V032_BEST_CASE_SCALING.md)
and normalized, source-hashed inputs record every assumption. No data-centre
or multi-GPU execution is claimed.
