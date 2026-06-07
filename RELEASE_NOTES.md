# Release notes — wrf_gpu v0.12.0

**v0.12.0 makes wrf_gpu a true out-of-the-box standalone GPU forecast system.**
It runs a real-data WRF case end-to-end on a single GPU — including a live-nested
run down to the 1 km inner nest — with **no CPU-WRF `wrfout` dependency**, reads a
standard WRF `namelist.input`, and writes a WRF-compatible `wrfout`.

This release builds on the v0.11.0 capability set (live multi-domain nesting,
restart continuity, conservation-closed budgets, MYNN-EDMF, topographic/slope
radiation, terrain-slope diffusion, KF/BMJ/Tiedtke/Grell-Freitas cumulus,
gravity-wave drag, the recompile fix), all of which carries forward unchanged.

## Headline features

- **Standalone out-of-box CLI.** `python -m gpuwrf.cli run --input-dir <case>`
  auto-detects native-init vs replay: a case with only `real.exe` outputs
  (`wrfinput_<domain>` + `wrfbdy_d01` + met_em) runs in **standalone native-init
  mode** with no CPU-WRF artifact. This also fixes the production AIFS-pull crash
  (a JAX `donate` input-aliasing bug, closed at two layers, plus a disk-scratch
  fix so scratch is never placed on `/tmp` tmpfs). Proof:
  `proofs/v0120/standalone_native_init_smoke.json` (2 h `PIPELINE_GREEN`).
- **Standalone live-nested `--max-dom`.** `--max-dom N` runs a standalone
  live-nested forecast (d01→d02→d03, down to the 1 km nest) from `real.exe`
  outputs alone — the parent builds each child's lateral boundary **live**, with
  no CPU-WRF `wrfout` and no pre-supplied `wrfbdy_d02`. `--max-dom` defaults to 1
  (single-domain); nested is explicit opt-in. Proof:
  `proofs/v0120/standalone_nest_smoke.json` (2 h, both domains finite).
- **Persistent JIT cache (on by default).** A persistent on-disk XLA compilation
  cache turns the multi-minute cold compile into a ~10 s disk read on every later
  run: measured **cold ~147 s → cache-hit ~29 s** (d01 hour-1 wrapper). The cached
  executable is keyed by HLO + backend + flags and is **bit-for-bit identical** to
  the cold one — **zero numerics change**. After a `jax`/`jaxlib` upgrade the key
  changes and the first run pays one cold compile again (stale entries are
  ignored, never wrong).
- **Fail-closed scheme catalog + validator.** Every unsupported namelist option is
  rejected **before any compute** with a specific named reason — the port never
  silently substitutes or skips a scheme (three honest outcomes:
  *implemented* / *recognized-WRF-not-yet-implemented* / *invalid*).
- **WRF-faithful PSFC fix.** The surface-pressure diagnostic now uses the
  WRF-faithful `PSFC = p8w(kts)` extrapolation from the total-geopotential faces
  (per `module_surface_driver.F` / `module_big_step_utilities_em.F`) instead of
  the old `p0`-based value, closing a systematic ~29 Pa diagnostic offset (proof:
  `proofs/v0120/psfc_extrapolation_proof.json`).
- **Runnable GPU-vs-CPU equivalence demo.** `scripts/equivalence_demo.py` lets a
  skeptic run and check, field-by-field, that the GPU port reproduces a retained
  CPU-WRF forecast under the same ICs/LBCs — emitting an honest verdict from the
  data (see the honest framing below).

The standalone path is **fp64-only** (no fp32 standalone path is reachable through
the CLI; gated-fp32 is an experimental ADR-007 preview and is no faster on this
memory-bound workload).

## Honest equivalence framing — read this before quoting numbers

The runnable equivalence demo's verdict on the default **24 h d02** case is
**`NOT_EQUIVALENT`** (6 of 10 fields exceed the predeclared operational
tolerances). This is the honest current state, reported as-is, on the
**post-PSFC-fix** re-run (proof:
`proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`).

| Field | pooled RMSE | tol | verdict |
|---|---|---|---|
| T2 | 0.484 K | 1.5 K | PASS |
| U10 | 2.237 m/s | 1.5 | EXCEEDS |
| V10 | 2.441 m/s | 1.5 | EXCEEDS |
| PSFC | 415.3 Pa | 120 | EXCEEDS |
| RAINNC | 0.501 mm | 1.0 | PASS |
| T (θ′) | 2.040 K | 1.5 | EXCEEDS |
| U | 3.167 m/s | 1.8 | EXCEEDS |
| V | 8.130 m/s | 1.8 | EXCEEDS |
| W | 0.126 m/s | 0.30 | PASS |
| QVAPOR | 5.67×10⁻⁴ kg/kg | 1.0×10⁻³ | PASS |

- **Winds dominate the verdict.** U10/V10/T/U/V start within (or near) tolerance at
  short lead and grow monotonically with lead time; the 3D meridional wind **V** is
  essentially identical at h1 and drifts to ~11 m/s by h19 (~3× faster than U).
  **Winds are not equivalent at 24 h.** T2, W, QVAPOR and RAINNC stay inside
  tolerance for the full 24 h.
- **PSFC is improved but still out of bar, and the residual is now dynamical.** The
  WRF-faithful surface-extrapolation fix dropped PSFC pooled RMSE from
  **707.8 → 415.3 Pa**, closing the systematic ~29 Pa diagnostic offset. The
  residual excess **tracks the developing wind/mass divergence** over the run — it
  is **not** a fixed diagnostic offset. **PSFC is not equivalent at 24 h either.**

Do **not** read this release as "PSFC fixed" or "winds equivalent." The honest
summary: **short-lead fields track CPU-WRF within tolerance; by 24 h the run is
`NOT_EQUIVALENT`, driven by wind divergence.** Full numbers and framing:
[`docs/equivalence-demo.md`](docs/equivalence-demo.md), tracked as **KI-9**.

## Speedup (three distinct numbers, none dishonest)

All numbers are one RTX 5090 vs 28-rank CPU-WRF on the same workstation, both
fp64, same d02 3 km grid, computed per forecast-hour (same model time). Full
reconciliation: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

- **Warm kernel (apples-to-apples): ~5×** (band 5–8×, strict dt-parity floor
  ~3.2×) — compute-only per forecast-hour
  ([`proofs/perf/speedup_denominator.md`](proofs/perf/speedup_denominator.md)).
- **Warm real-user wall: ~2.5×** — full command-to-finish wall, persistent cache
  warm, includes IO + case build.
- **Equivalence-demo real-user: ~4.26× warm-cached / ~1.70× cold** — the 24 h d02
  demo run (GPU 561.3 s warm vs 1408.6 s cold; CPU 2393.2 s d02 solver). The
  cold-vs-warm gap is entirely the persistent JIT cache + IO/case-build, **not** a
  numerics change.

## Known issues (carried + new)

See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) for full detail.

| ID | Summary | Severity |
|---|---|---|
| KI-3 | Focused **64-variable** `wrfout` (vs WRF's 375); missing only stochastic-seed + Noah-MP snow-layer diagnostics. | Scope boundary |
| KI-4 | d02 **U10** episodic final-lead under-prediction (8.06 vs 7.5 m/s bar); within bar at all other leads, beats persistence 23/24. | Documented residual |
| KI-5 | Powered **n=15 TOST** not yet scored (corpus prepared); **no TOST PASS is claimed**, n=15 underpowered. | Scope boundary |
| KI-6 | RRTMG SW intermediate `taug` differs in 4 UV bands; integrated fluxes pass tier-1 (< 0.05% rel). Pre-existing; carried to v0.13.0. | Isolated |
| KI-7 | Free-running (`run_boundary=False`) on **wide domains** (nx≈160+) can go unstable beyond ~14 h. Operational path uses boundary forcing. | Robustness edge |
| **KI-9** | Equivalence demo: **24 h d02 `NOT_EQUIVALENT`**, dominated by **lead-time wind divergence** (3D V pooled RMSE 8.13 m/s); residual PSFC excess driven by the same divergence (PSFC improved 707.8 → 415.3 Pa). | Documented gap |

## Deliberately deferred to v0.13.0 (deliberate scope boundaries, not silent gaps)

- **Gotthard / Switzerland operational suite** — v0.12.0 ships the standalone port
  + the AIFS / 1 km-nest path only.
- **Scheme scan-wiring of the reference-only families** — MYJ PBL + Janjic-Eta
  sfclay, Dudhia SW, classic RRTM LW, New-Tiedtke cumulus are recognized and
  parity-proven but **fail closed** if selected operationally.
- **Full two-way nesting** — feedback + radiation-in-loop + in-loop `w` relaxation
  + 5-domain long-run equivalence (one-way 24 h is proven via the v0.11.0
  replay-boundary proof).
- **fp32 standalone path** — gated-fp32 operational mode (ADR-007), pending
  evidence it helps on this memory-bound workload.
- **Full 375-variable `wrfout`** (KI-3), **RRTMG SW `taug` UV-band fix** (KI-6),
  and the **`*_tendf` source-tendency adapter** for RK-stage physics.

## Pending before tag (manager fills)

- **Standalone nested 24 h 1 km proof** — <<MANAGER-FILL: Lane A in flight; replace
  with the proof result/verdict if it lands before tag, else state it remains a
  smoke-only (2 h) demonstration carried to v0.13.0.>>
- **Powered n=15 TOST** — <<MANAGER-FILL: pending GPU campaign; do NOT claim a TOST
  PASS — record "not scored" if the campaign has not completed.>>
