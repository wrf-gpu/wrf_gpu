# v0.4.6 — WRF initialization and graupel entry mass

**2 November 2026.** The real-world checks below include a disclosed QGRAUP
trace disposition. Raw failures and the open scientific cause are not
relabelled as numerical passes.

[Known issues](../KNOWN_ISSUES.md) · [v0.4.5 notes](RELEASE_NOTES_v0.4.5.md) ·
[Decision and evidence summary](../docs/release/evidence/v046/MANAGER_DECISIONS.md)

## What changes

- **Noah-MP initial emissivity.** The first radiation step uses the WRF
  `landuse_init` seasonal `SFEM` emissivity instead of a uniform 0.97 seed.
- **Thompson graupel entry mass.** Retain WRF's raw active entry mass below
  `R1` in the entry spectra. This changes the compiled forecast program; it
  does not resolve the final-hour QGRAUP trace described below.
- **Offline cloud diagnostics.** The cloud-water-path evaluator uses the shared
  WRF thermodynamic constant. This is an offline diagnostic change, not a
  forecast improvement claim.
- **Bundle handling and tooling.** Relocated bundle checks and offline wave
  tooling receive host-side updates. The replay checks below use the
  actual sealed GPU bundle.

Existing v0.4.5 qualifications continue to apply. Diffusion/damping overrides
including `c_s` and the nine unqualified LSM controls remain refused; MYNN
cloud-PDF modes 0/1 remain CPU-qualified development options pending coupled
GPU qualification. The separate A2 entrainment/denominator investigation is
**NOT_APPLICABLE to this release** and supplies no release or calibration claim.

## Release checks

Model pin `80a627e28fe3`; original CPU-WRF is the fidelity reference.
GPU-to-GPU comparisons below are regression or determinism evidence only.

| Check | Result and scope |
|---|---|
| RW1 warm cache | PASS: byte-exact short replays and N72/A3 first-three-hour prefix, zero new compiled artifacts. d01 startup to lead-1 output: 30.844 s (v0.4.6) vs 34.687 s (v0.4.5); see measurement scope below |
| RW2 cross-date | PASS: sealed programs reused across WN3 dates and shifted input |
| RW4 bundle replay | PASS: at least three WN3 dates, including 48-hour input; target-specific key policy below |
| Swiss 24 h | PASS against original CPU-WRF, 25 frames, output integrity PASS |
| WN3 0227, 72 h | All three domains and 73 frames each. Raw D6 FAIL; frozen A3 floor PASS: 5 FLOOR-LIMITED, 0 REAL FAIL. QGRAUP output-integrity result remains false and is separately disclosed below |
| Speed, memory | Three same-session 24-hour N/O pairs PASS, with values below |
| Compile proxy | Cold-minus-warm WN3 three-hour proxy 473.13 s versus 489.90 s in v0.4.5 (ratio 0.9658); fresh-empty versus seeded autotune state differs |
| Compiled resources | Stack hard gate and frozen policy PASS. Native E41 scanner reports INCOMPLETE; exact loaded executable coverage and manager-accepted policy are retained. This is not a measured full-HLO PASS |

The RW1 startup measurement is the d01 lead-1 output file modification time
minus arm start, for the candidate and v0.4.5 runs. It is a single measured
startup comparison, not a fidelity measure or a general speed claim. Source:
`SHORT_VERDICT.json`, SHA-256
`2de3fa19c217150d4dffc55a1766e27ff3dabd783a3937997665ffdbd5b20fa1`.

The 72-hour raw score has five d03 RAINNC exceedances at leads 68–72. Each is
FLOOR-LIMITED under the unchanged `C > limit OR C/X >= 0.75` original CPU-twin
rule; none is REAL FAIL. At lead 72, RMSE is 1.907328 mm against the raw 1.0 mm
limit, with CPU-twin C 2.111858 mm. The original raw report remains FAIL.

### Final-frame graupel trace

At d03 lead 72, the original CPU-WRF has three positive QGRAUP cells out of
454,212, with maximum 6.57752064620043e-12 kg/kg. Both the v0.4.6 candidate
and retained v0.4.5 GPU field are zero everywhere and are byte-identical for
this array; that comparison is regression evidence, not a CPU-WRF fidelity
claim. The retained-entry fix `720d8d2937` is present in both Thompson PTX
kernels embedded in the GPU executable actually loaded by N72. Native SASS and
per-cell predicate visits were not independently re-audited for this v0.4.6
executable. The process responsible between hours 71 and 72 remains **OPEN**.

Output integrity therefore remains false. The manager accepts the disclosed
trace class using the v0.4.5 precedent; the RW3 outcome is **MANAGER_DECISION**,
not an unqualified numerical PASS. See the [decision summary](../docs/release/evidence/v046/MANAGER_DECISIONS.md).

### RW4 key interpretation

The replay covers three dates and the shifted input, with zero new compiled
artifacts. CPU-target `aot-key` values agree pairwise; the GPU runtime hits the
same sealed GPU keys. CPU-target and GPU-target values are not directly
comparable. The manager applies the same operational check used in v0.4.5;
the decision and its SHA are recorded in the [evidence summary](../docs/release/evidence/v046/MANAGER_DECISIONS.md).

## Matched timing and memory

N is v0.4.6 and O is v0.4.5. The same-session WN3 0227 cohort contains N1/O1,
N2/O2 and N3/O3, each 24 hours. The 72-hour O arm did not run, so there is no
matched 72-hour old-release comparison. All counted timing arms loaded their
sealed programs without new compiled artifacts.

| Metric | v0.4.5 O mean | v0.4.6 N mean | N/O |
|---|---:|---:|---:|
| d03 stepping median, s per forecast hour | 5.7870 | 5.7987 | 1.0020 |
| Process start to last output, s per forecast hour | 6.9017 | 6.9484 | 1.0068 |
| Sampled own-process VRAM peak, MiB | 5,255.3 | 5,234 | 0.9959 |
| Host peak RSS, KiB | 6,280,868 | 6,282,764 | 1.0003 |

The stepping repeat spread is 0.0416; the registered upper bound is 1.0436,
below the 1.10 stop threshold. No speedup is established. N72 alone measured
6.455 s per forecast hour, 5,394 MiB sampled process VRAM and 6,336,444 KiB
host peak RSS; it is not a same-session 72-hour version comparison.

The inherited v0.4.5 verdict reducer mistakenly compares the six 24-hour arm
names with a counted list that also contains N72, so its memory field reports
MISSING. The independent six-arm calculation above is the manager-designated
result; the reducer error is disclosed rather than silently patched.

The `icloud=0` GPU replay is checked, but a fresh CPU-WRF fidelity result is
not claimed because its matching original CPU twin is unavailable. Nine-nest
qualification and the five-domain Q9 cold-compile timeout remain open.
