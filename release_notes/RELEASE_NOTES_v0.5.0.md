# v0.5.0 — Terrain and surface experiment toolkit

Released 13 November 2026.

v0.5.0 adds tools for running many cheap, short, controlled forecast variants: branch many surface variants
from one spin-up, perturb members reproducibly, patch land use, roughness or terrain with consistency checks,
extract the evaluation fields while the forecast runs, and keep the 10 m wind maximum of WRF's AFWA package.
With every new option switched off, the forecast programs are the v0.4.6 programs.

[Known issues](../KNOWN_ISSUES.md) · [v0.4.6 notes](RELEASE_NOTES_v0.4.6.md) ·
[Decision and evidence summary](../docs/release/evidence/v050/MANAGER_DECISIONS.md)

## What changes

All new options are **off by default**. Each has its own page with commands, limits and refusals.

| Option | What it does | Status |
|---|---|---|
| [Branch series](../docs/checkpoint-branching.md#branch-series-many-value-variants-from-one-spin-up) | Spin up once, then run N surface-control variants (for example `z0mvt` per nest) from the same checkpoint; value variants share the compiled programs | GPU-checked on a nine-nest Canary Islands case: unmodified branch and null resume equal the continuous run byte for byte; variants change only their nests |
| [Stochastic physics in nests](../docs/stochastic.md) | SKEBS, SPPT and SPP on every nest as in WRF (domain 1 makes the patterns, nests receive them), native restart of the stochastic state | GPU-checked against original CPU-WRF on a three-nest Canary Islands case (9/3/1 km): seeds exact, pattern fields and forecast fields within limits on all three domains. `rand_perturb=1` has no CPU-WRF reference (WRF 4.7.1 crashes) |
| [Seeded boundary-layer noise](../docs/stochastic.md#seeded-boundary-layer-temperature-and-moisture-noise) | `gpuwrf perturb --thermo-config`: small reproducible θ/qv noise in the input files; the model program is unchanged | Input preparation only; not a WRF option |
| [Static-field patches](../docs/static_patch.md) | `gpuwrf patch`: land use, soil, land/sea, roughness on a mask, terrain via geo_em + `real.exe`, with cross-field and nest-boundary checks | Tool checked on CPU against real cases. GPU-checked with one land-use patch (barren land on part of the 1 km nest of the Tenerife case, 3 hours, started from an empty compile cache): finite output, all files written. Compile keys, checked on CPU: the domain-1 and domain-2 keys are unchanged, the 1 km nest and fused nest-program keys change, as predicted. On the GPU both the domain-1 program and the fused nest program were compiled (empty cache); the domain-1 program is identical to the unpatched one (same program hash), the fused nest program differs. Reuse of the unpatched domain-1 binary from a filled cache was not exercised. A patched forecast has not been compared with CPU-WRF; soil, roughness and terrain patches have not run on the GPU. Land-use and terrain variants compile their own programs |
| [In-run extract](../docs/extract.md) | Per-nest NetCDF of surface, cloud (cloud fraction maxima, LWP/IWP, cloud base), inversion and point profiles written while the forecast runs; optional run without wrfout files | GPU-checked on nine nests: the forecast is byte-identical with the extract on or off. The extract equals an offline recomputation from full history (checked on CPU) |
| [AFWA `WSPD10MAX`](../docs/ONLINE_DIAGNOSTICS.md#afwa-severe-10-m-maximum-wind-wspd10max) | `&afwa afwa_diag_opt=1, afwa_severe_opt=1`: maximum 10 m wind since the last history write, raised under heavy rain as in WRF | Checked bit for bit against the original WRF code on real columns; GPU-checked on the two-domain 9/3 km Canary Islands case (3 hours, three on/off pairs): the GPU kernel matches the WRF code exactly, warm cost with the option on vs off 0.998×. Switching it on makes a new compiled program, so other fields differ from an off run by up to 0.87 K (2 m temperature) and 0.78 m/s (10 m wind) after 3 hours; compare AFWA-on runs only with an AFWA-on twin. Not compared with a full CPU-WRF forecast. Not an observed gust |
| [Several members per GPU](../docs/ensemble.md) | Run 2–3 nine-nest members at once on one 32 GB GPU | Measured: 3 concurrent members 1.60× the throughput of one, 2 members 1.25×; bytes equal to serial runs |
| [Shared compile store](../docs/aot_content_store.md) | **Experimental, off by default.** Compiled programs addressed by the lowered program, so builds whose program is identical can share one binary | Shares the forecast-step binaries as designed, but runs from two builds are **not** byte-identical yet: about 4,100 small start-up/output programs outside the store recompile and the GPU autotuner picks differently. Use a same-build twin for A/B comparisons |

Research setting, not release-qualified: **85 vertical levels** (`e_vert=85`, inversion-focused `eta_levels`) runs and
stays within the CPU-WRF limits on a 3-hour three-nest Canary test, but costs 1.7× per step at equal time step, and the Thompson
column kernel spills registers above 64 levels. Use 45 levels for production.

## Portability (compile-checked, not validated)

v0.5.0 was compile-checked for GPUs other than the RTX 5090. These are build checks, not validations: accuracy and speed on
these GPUs are untested and will follow in a later release.

- **NVIDIA Ampere and Ada (A100, RTX 30, RTX 40):** the full two-domain forecast program compiles with the standard install.
  It has not been run on these GPUs yet.
- **AMD (MI250X, MI300X, RX 7900):** the full program compiles with the experimental ROCm route
  (`GPUWRF_ROCM_TRITON_NATIVE=1`). Nothing has run on AMD hardware. On RX 7900 one dynamics kernel runs out of registers and
  spills, so it would be slower.
- **Older NVIDIA (Pascal, Volta, Turing, for example GTX 10, V100, RTX 20):** the standard install does not support them
  because its compiler needs Ampere or newer. The experimental route (`GPUWRF_PRE_AMPERE_KERNEL_CALL=1` with a CUDA 12 JAX
  install) compiles every model kernel for all three generations. On a GTX 1080 Ti it ran a 3-hour, two-domain forecast to
  completion with finite output. That is one run on one card; Volta and Turing have not been run on hardware.

Details: [docs/portability.md](../docs/portability.md#v050-compile-check).

## Release checks

Model pin `4e13626e1429`; original CPU-WRF is the fidelity reference. GPU-to-GPU comparisons are regression or
determinism evidence only.

| Check | Result and scope |
|---|---|
| Default programs vs v0.4.6 | Lowered forecast programs (all constants, all GPU kernels) identical to v0.4.6 for the root step, the nest force-down and the nest step of the 9/3 km Canary case with every new option off. The compiled binaries are new (fresh compile) |
| RW1 warm cache | PASS: byte-exact 3-hour replays (WN3 0227, Swiss, two-domain 9/3 km Canary case) and the first three hours of the Swiss 24-hour run against the cold 3-hour run, zero new compiled programs. d01 startup to lead-1 output: 32.424 s (v0.5.0) vs 30.844 s (v0.4.6), one measurement each, not the same session. The first three hours of the 72-hour run equal the cold 3-hour run byte for byte |
| RW2 cross-date | PASS: sealed programs reused for WN3 0502 and for 0227 started from 48-hour input (also with `icloud=0`), zero new compiled programs, one program key per program |
| RW4 bundle replay | PASS: three WN3 dates (0502, 0614, 0227 from 48-hour input) replayed from the sealed bundle with zero new compiled programs; CPU-target program keys agree pairwise. Switching `icloud`, or patching land use on the 1 km nest, gives new keys for the affected programs, as designed |
| Swiss 24 h | PASS against original CPU-WRF, 25 frames, output integrity PASS |
| Tenerife 9/3/1 km, 72 h | PASS, disclosed: against original CPU-WRF, 73 paired frames per domain; frozen floor rule PASS with 0 real failures and 6 floor-limited rows, all d03 accumulated rain (`RAINNC`) at hours 67–72 (v0.4.6: 5). Output integrity passes on d01 and d02; on d03 it fails only on the known graupel trace at hour 72 (see below) |
| Speed, memory | PASS: same-session 24-hour pairs against v0.4.6, d03 stepping median 0.9992, forecast-only time 1.0007, VRAM 1.006, host RSS 0.996 (stop at > 10 %). Whole-run time per forecast hour is inconclusive by the registered rule: data-disk write latency exceeded 50 ms in 5 of the 6 arms (pair ratios 0.999–1.002). Table below |
| Compile proxy | PASS on the same basis: cold-minus-warm WN3 three-hour proxy 624.42 s (v0.5.0) and 623.17 s (v0.4.6), both started from an empty cache on the same host on 13 November, ratio 1.002 (check limit 1.10). The compiled program set is the same (all 1,237 autotuning keys and the forecast-program hashes are equal). The earlier figures were not comparable with this basis: the v0.4.6 reference of 473.13 s ran after the Swiss and two-domain checks had already compiled about two thirds of the small start-up programs and three quarters of the autotuning results it needed. The 12 November v0.5.0 cold run on that same primed basis (592.65 s, ratio 1.25) was slowed only from the third run of its session on (identical programs compiled 14–36 % slower, the first two runs matched v0.4.6 within 3 %); the cause of that slowdown is not identified. The earlier host-load ruling is replaced by this measurement |
| Compiled resources | Short-check executables (programs d01, d02 and the fused nest program): stack hard gate PASS, coverage PASS, no flagged fusion; the native scanner reports INCOMPLETE only for compact AOT metadata (accepted rule below). 72-hour run (same three programs): stack hard gate PASS, coverage PASS, no flagged fusion, INCOMPLETE only for compact AOT metadata, each compact executable bound to a scanned one by its exact program |

### Compiled-resource disclosures (opt-in programs)

The release gate covers the default programs. Opt-in features make new programs; their scanner results:

- **Stochastic physics on:** one input-transpose fusion with 600 optimized HLO instructions (rule: ≤ 500), 6 transposes;
  stack ≤ 32 B, no large non-leaf callee. Accepted and disclosed for the opt-in program; splitting it is deferred to the
  performance release. The acceptance is bound to this fusion (program, size, at most one flagged entry), not to a cache
  file name, because cache file names change between runs.
- **AFWA wind maximum on:** one input-transpose fusion with 664 optimized HLO instructions in the domain-1 step program
  (rule: ≤ 500); stack ≤ 32 B, no large non-leaf callee. Accepted and disclosed for the opt-in program; splitting it is deferred
  to the performance release.
- **Nine-nest programs (default and z0 branch series):** two known input-transpose fusions (564 and 522 instructions)
  on the nest program, stack ≤ 160 B. Accepted as in the v0.4.6 nine-nest runs.
- **85 levels (research):** `thompson_full_column` uses 600 B stack (limit 256 B). Not release-qualified.
- Compact AOT executables without rich metadata are reported INCOMPLETE by the scanner, as in v0.4.6; accepted only
  when every other gate passes and each compact executable matches a scanned one.

### Final-frame graupel trace

At d03 lead 72 of the WN3 0227 72-hour check, v0.4.5 and v0.4.6 had a QGRAUP trace in original CPU-WRF (a few cells around
1e-12 kg/kg) where the GPU field is zero; the responsible process between hours 71 and 72 is still **OPEN**. Thompson
microphysics is unchanged in v0.5.0. The disposition was fixed before the run: if the 72-hour check shows the same trace class
and it is the only output-integrity failure, it is disclosed as in v0.4.6; any other deviation stops the release.
Result: the same trace class. On d03 at hour 72 original CPU-WRF has QGRAUP up to 6.58e-12 kg/kg where the GPU field is
zero; it is the only output-integrity failure (d01 and d02 pass, no missing variables, all frames paired). Disclosed as
decided; the cause remains open.

## Matched timing and memory

N is v0.5.0 and O is v0.4.6 (canonical bundle). The same-session WN3 0227 cohort contains N1/O1, N2/O2 and N3/O3, each
24 hours, run alternately on 13 November. All counted arms loaded their sealed programs without new compiled artifacts.

| Metric | v0.4.6 O mean | v0.5.0 N mean | N/O |
|---|---:|---:|---:|
| d03 stepping median, s per forecast hour | 5.5017 | 5.4970 | 0.9992 |
| Forecast-only time, s (24 h) | 139.43 | 139.53 | 1.0007 |
| Process start to last output, s per forecast hour | 6.7699 | 6.7732 | 1.0005 (inconclusive, see below) |
| Sampled own-process VRAM peak, MiB | 5,234 | 5,266 | 1.0061 |
| Host peak RSS, KiB | 6,336,661 | 6,309,759 | 0.9958 |

The stepping repeat spread is 0.0018; no speedup or slowdown is established. The whole-run row is reported but not
counted: the registered rule excludes arms whose data-disk write latency exceeded 50 ms, which left one arm. N72 alone
measured 6.112 s per forecast hour (process start to last output), a d03 stepping median of 5.624 s, 5,362 MiB sampled
process VRAM and 6,336,092 KiB host peak RSS; its first output appeared 21.7 s after process start (17.4 s of it loading
the sealed programs). There is no 72-hour v0.4.6 arm, so this is not a version comparison.

## Upgrade notes

- New compile caches: the version and the source fingerprint changed, so v0.4.6 caches are not reused. The first
  run per grid compiles cold as usual; a v0.5.0 bundle is provided.
- AFWA `WSPD10MAX` on makes a different compiled program. Compare a run with AFWA on only against a twin with AFWA on
  (same program); differences to an AFWA-off run include compile-level differences in all fields.
- Checkpoints written by v0.4.6 resume only with unchanged options; a checkpoint with stochastic physics on needs
  the stochastic state in it (older active checkpoints are refused).
- Branching supports one-way nesting only; the parallel launcher still refuses branches.
