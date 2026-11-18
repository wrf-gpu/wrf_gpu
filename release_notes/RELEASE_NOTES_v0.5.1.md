# v0.5.1 — Bare-ground roughness control for fixed grids

Released 17 November 2026.

v0.5.1 adds one experimental option that is off by default: a bare-ground roughness control (`z0bare`). It is read from a static
file per grid, and the file is locked to the grid it was built for. A second new option writes the roughness length the model
carries into the history files. With both options off, the forecast programs are the v0.5.0 programs.

[Known issues](../KNOWN_ISSUES.md) · [v0.5.0 notes](RELEASE_NOTES_v0.5.0.md) ·
[Decision and evidence summary](../docs/release/evidence/v051/MANAGER_DECISIONS.md)

## What changes

All new options are **off by default**.

| Option | What it does | Status |
|---|---|---|
| [`z0bare` surface control](../docs/surface_params.md#experimental-bare-ground-roughness-z0bare) | `--surface-controls FILE.npz` sets the bare-ground roughness length that Noah-MP uses for the ground below and between vegetation, per nest. The value is a traced input: changing it needs no new compile, but switching the control on compiles its own program | **Experimental and not WRF-faithful by design**: WRF has no such control, it uses one fixed ground roughness. GPU-run with the release build in the nine-nest qualification (two cases, 72 hours): the switch works as expected; a benefit at stations is not shown. Qualification result: undecided, see below |
| [Grid-locked control files](../docs/surface_params.md#grid-locked-static-control-files-z0bare-nightly-path) | `scripts/z0bare_grid_controls.py build/verify` writes one static file per grid. Its lock covers latitude, longitude, land use, vegetation type, land mask, terrain height and the projection and nest attributes. `gpuwrf run` refuses a file whose lock does not match the input, at the command line and again at the model's runtime boundary | Checked on CPU: a changed land-use cell or a 1 m terrain change is refused; one file passes on a summer and a winter case on the same grid |
| [`--history-znt`](../docs/surface_params.md#optional-carried-znt-history) | Adds the carried roughness length ZNT (m) to the history files | Output only: no change to the compiled forecast programs or their cache keys |

## Portability

Unchanged from v0.5.0: see [Portability (compile-checked, not validated)](RELEASE_NOTES_v0.5.0.md#portability-compile-checked-not-validated).

## Release checks

Model pin `c07aec178d42`. Original CPU-WRF is the fidelity reference; GPU-to-GPU comparisons are regression or
determinism evidence only.

| Check | Result and scope |
|---|---|
| Default programs vs v0.5.0 | With the new options off, the lowered forecast programs (all constants, all GPU kernels) equal v0.5.0. Checked for the root step, nest force-down and nest step of the 9/3 km Canary case, and for all six programs of the 9/3/1 km Tenerife case used in the 72-hour check: root step, both force-downs, both nest steps, and the fused 3 km + 1 km program. The two programs the 72-hour run loads carry the same program hash in v0.5.0 and v0.5.1. The compiled binaries are new (fresh compile) |
| RW1 warm cache | PASS: byte-exact 3-hour replays (WN3 0227, Swiss, two-domain 9/3 km Canary case) with zero new compiled programs. The first hours of the Swiss 24-hour run and of the 72-hour run equal their cold 3-hour runs byte for byte |
| RW2 cross-date | PASS: sealed programs reused for WN3 0502 and for 0227 started from 48-hour input (also with `icloud=0`); one program key per program |
| RW4 bundle replay | PASS: three WN3 dates (0502, 0614, 0227 from 48-hour input) replayed from the sealed bundle; CPU-target program keys agree pairwise. Switching `icloud`, or patching land use on the 1 km nest, gives new keys for the affected programs, as designed |
| Swiss 24 h | PASS against original CPU-WRF, 25 frames, output integrity PASS |
| Tenerife 9/3/1 km, 72 h | PASS, disclosed: against original CPU-WRF, frozen floor rule PASS with 0 real failures and 6 floor-limited rows, all d03 accumulated rain (`RAINNC`) at hours 67–72 (v0.5.0: 6). Output integrity fails on d03 only on the known graupel trace at hour 72. Both items below |
| Speed, memory | PASS: same-session 24-hour pairs against v0.5.0, d03 stepping median 0.989–1.020, host RSS 1.002, VRAM 0.996 (stop at > 10 %) |
| Compile time | Cold-minus-warm WN3 three-hour proxy 522.6 s, 0.88× the v0.5.0 figure of 592.6 s. That v0.5.0 figure was measured on a slowed session (see the v0.5.0 notes), so the ratio is not a speed-up claim |
| Compiled resources | Programs d01, d02 and the fused nest program: stack hard gate PASS, coverage PASS, no flagged fusion; the scanner reports INCOMPLETE only for compact AOT metadata, as in v0.5.0 |

### Accumulated rain on the 1 km nest: two compiles of the same program

On the 1 km nest at hours 67–72, the GPU-vs-CPU-WRF difference in accumulated grid-scale rain (`RAINNC`) is larger than in
v0.5.0 in all six hours: +3.8 to +19.1 % (RMSE 1.10–2.18 mm against 1.06–1.91 mm). This comes from compiling, not from a
model change:
- all six forecast programs of this case have the same lowered program in v0.5.0 and v0.5.1;
- the start frame (hour 0) is byte-identical on all three domains, and the runs differ from the first forecast hour on;
- the compiled binaries differ, because a fresh compile of the same program does not give the same binary (the GPU
  autotuner can choose differently).

The +3.8 to +19.1 % was observed in **one** pair of compiles, on one case. It is not an upper bound: no twin from the same build
was run, so the usual spread between two compiles is not measured. Compare rain between builds only with a same-build twin. All
six rows stay inside the frozen floor rule (CPU-WRF's own twin spread is at least 0.75 of the GPU difference).
v0.4.6 and v0.5.0 also ran the same program, and their rows moved only by −1.4 to +0.1 %.

### Final-frame graupel trace

The same trace class as in v0.4.5, v0.4.6 and v0.5.0: on d03 at hour 72, original CPU-WRF holds QGRAUP up to 6.6e-12 kg/kg
where the GPU field is exactly 0. It is the only output-integrity failure. Thompson microphysics is unchanged; the responsible
process between hours 71 and 72 is still **open**.

## Nine-nest qualification

Arms on two frozen cases (9 July and 7 January), 72 hours, nine nests:
- the operated nightly setup;
- v0.5.1 with `z0bare` off;
- v0.5.1 with `z0bare` on.

Stations are sampled with the operational station operator. Criteria and evaluation were frozen before the runs.

**Result: undecided (UNENT); the operated nightly is not switched.**
- Operated setup → v0.5.1 with `z0bare` off: 56 of 60 cells green, nothing red. Four elevation cells have one yellow case, three of them
  from the winter case (7 January): wind direction by day (hours 7–24), relative humidity by day (hours 7–24) and night T2 (hours 49–78);
  the fourth is one wind-direction cell of the summer case.
- v0.5.1 `z0bare` off → on: 60 of 60 cells green; the switch check passes.
- Winter scatter between the operated setup and v0.5.1 off is about twice the summer value at every elevation band; the means stay green.
  The cause stays open without a twin of the same engine.
- The nightly stays on v0.3.3. Dew point at elevation and the winter case are not cleared for serving; a transfer test against station
  measurements (including winter) is pending.
- Both operated-setup arms ran as second attempts after clean refusals before start (not enough disk space at the first try).
- Source: verdict `RND_VERDICT_QUAL17_V051_UNENT_20261117T100648Z.json` (SHA256 `d86ded6b…`), evaluated with eval_qual17 v1.6 (`035f3c79…`;
  N033 cap per arm by criteria addendum `e1fea2e3…`).

## Upgrade notes

- New compile caches: the version changed, so v0.5.0 caches are not reused. The first run per grid compiles cold as usual.
- Control files from v0.5.0 research runs carry no grid lock. They are still accepted, but nothing checks their grid: rebuild
  them with `scripts/z0bare_grid_controls.py build` before using them with v0.5.1.
- Restarts: a checkpoint written with `--history-znt` resumes only with `--history-znt`, and one written without it resumes
  only without it. `--history-znt` runs one member at a time (B=1).
