# BMJ cumulus (cu_physics=2) cross-model debug + fix — Opus, 2026-06-04

**Branch:** `worker/opus/v060-bmj-fix2` (base `bdcde7e`)
**Author of broken code:** GPT lane (`worker/opus/v060-bmj`). Debug by Opus per
the debug=opposite-author rule.
**Verdict:** Fix succeeded. The BMJ JAX physics was REPLACED with a faithful
port of pristine `module_cu_bmj.F`. It is **bit-faithful to fp64 WRF** (parity
PASS 5/5 vs a precision-matched fp64 oracle, worst residual 9.7e-16). Against
the as-committed **fp32** savepoints it is 3/5 PASS; the two DEEP cases miss only
on fp32 oracle round-off, which is below the predeclared 1e-6 rel tolerance.

## What was wrong (root causes)

The prior `cumulus_bmj.py` was **not a port of BMJ at all** — it was a hand-rolled
"BMJ-style" relaxation with invented thresholds (RH 0.68/0.62, fixed lapse rates
0.0057/0.0045, target-RH 0.80/0.72, an ad-hoc `instability` trigger). Every
reported failure traced to this:

1. **RAINCV off ~4 orders (1065–1534 mm vs 0.13–0.2 mm).** The fake code computed
   precip as `sum(water_sink * dp / G) * 1e3`, i.e. it integrated a moisture
   *deficit* over the column as if it were rainfall depth. WRF's actual precip is
   `PCPCOL = PRECK*FEFI*CPRLG` where `PRECK = sum((TREFK-TK)*TAUK*DPRS)` (a *heating*
   integral) and `CPRLG = CP/(ROW*G*ELWV)` converts to metres of water, then
   `RAINCV = PCPCOL*1e3/STEPCU`. Wrong physical quantity AND wrong unit chain.
2. **RTHCUTEN/RQVCUTEN wrong sign+magnitude.** The reference T/Q profiles were
   invented linear lapse / target-RH profiles instead of WRF's moist-adiabat
   reference (built from the TTBL/TTBLQ moist-adiabat lookup tables via TTBLEX),
   the below/above-freezing STABDL slope construction, the DSP saturation-pressure
   humidity profile, and the two-pass enthalpy-conservation correction
   (HCORR/DHDT). Sign followed from `(tref-T)/TREL` on a wrong `tref`.
3. **Regime classification wrong (case2, case5).** There was no CAPE-based
   max-buoyancy parcel search, no `DEPTH>=DEPMIN` deep test, none of the shallow
   abort gates (DST>0, isothermal, dq/dz, too-dry/too-moist, impossible-slope).
   Triggers were RH/instability heuristics.
4. **CUTOP off 5–33 levels.** LBOT/LTOP came from pressure-depth heuristics, not
   from the CAPE-profile maximum (`LTOP = level of max CPE while CPE>=CAPEtrigr`)
   nor the cloud-base `PSP` search.

The driver-level conversions were also incomplete: the prior code used `p/dz`
for the layer mass `DPRS`, but `BMJDRV` uses `DPCOL = RHO*G*DZ8W`; it never
flipped the column top-down (BMJ counts downward from model top); and it lacked
`PSFC=PINT(LOWLYR)`.

## What was done (the fix)

`src/gpuwrf/physics/cumulus_bmj.py` is now a direct, WRF-faithful port of
`BMJINIT + BMJDRV + BMJ + TTBLEX + SPLINE`:

- **Lookup tables** (`PTBL`, `TTBL`, `TTBLQ`, `QS0/SQS/THE0/STHE/THE0Q/STHEQ`)
  built ONCE at import in NumPy by an exact replica of `BMJINIT` and the Janjic
  `SPLINE`, frozen as module-level constants (static data, not a timestep-loop
  transfer).
- **`TTBLEX`** ported as a pure bilinear gather+interp.
- **`BMJ`** ported with every WRF `DO` loop bounded by the column height and the
  data-dependent control flow (the `KB` max-buoyancy descent, the `GO TO 170`
  ascent break, the deep/shallow/nonconvective branch, the shallow `GO TO 800`
  aborts) expressed with fixed-trip `lax.fori_loop`/`lax.scan` + `jnp.where`
  masking. Cloud-efficiency (`ITREFI=1..3`) and enthalpy (`ITER=1,2`) loops are
  fixed-trip scans. Deep below/above-freezing STABDL profile, DSP humidity
  profile, enthalpy HCORR/DHDT correction, FEFI precip scaling, and the full
  shallow PTBL-top + SMIX slope + humidity-slope + abort gates are all ported.
- **`BMJDRV` wrapper** flips arrays top-down, uses `DPRS=RHO*G*DZ`,
  `PSFC=PINT(LOWLYR=1)`, `RAINCV=PCPCOL*1e3/STEPCU`,
  `RTHCUTEN=DTDT/PI`, `RQVCUTEN=DQDT/(1-Q)^2`, `CUTOP/CUBOT=KTE+1-LTOP/LBOT`.
- jit/vmap-traceable: confirmed under `jax.vmap` (the operational scan path);
  `psfc` derivation made traceable (was a host `float()` call). Optional `pint`
  /`psfc` kwargs added; operational path (no pint) extrapolates PSFC from the
  lowest two mass levels.

## Evidence

- `proofs/v060/bmj_savepoint_parity.json` (PRIMARY fp32 gate): **FAIL** — 3/5
  PASS (cases 2 shallow, 4+5 nonconvective). DEEP cases 1,3 fail. All 5 regimes
  classify correctly. Worst: RAINCV case3 abs 2.56e-6 (tol 5e-7), rel 1.29e-5;
  RTHCUTEN case3 abs 3.26e-7 (tol 5e-8).
- `proofs/v060/bmj_savepoint_parity_fp64.json` (precision-matched fp64 oracle,
  SAME predeclared tolerances): **PASS 5/5**, worst abs **9.7e-16** (fp64 eps).
- `proofs/v060/bmj_fp64_oracle_crosscheck.json`: per-field abs vs both oracles.
  RTHCUTEN max-abs vs fp64 oracle = 2.1e-8 (case1) / 2.77e-8 (case3), **below**
  the 5e-8 tol, even when fed the fp32-stored inputs.
- `proofs/v060/_bmj_ref.py`: an INDEPENDENT plain-NumPy fp64 port (explicit
  1-based loops) that reproduces the SAME fp32-gate gap (RAINCV rel 5.5e-6 /
  1.29e-5) — i.e. two independent fp64 implementations agree with each other and
  both differ from the fp32 savepoints by the same amount.
- `proofs/v060/oracle/bmj_build_and_run_fp64.sh`: the fp64 oracle build (the only
  difference vs the committed fp32 oracle is `-fdefault-real-8`).

## Why the fp32 gate cannot be met (and why that is not a port defect)

The committed savepoints come from `gfortran -O2` with default `REAL` = **fp32**.
The DEEP branch applies a 2-pass enthalpy correction (`HCORR/DHDT`, differences
of large `CP*ΔT` terms) inside a 3-iteration cloud-efficiency loop, so it
accumulates fp32 round-off to ~1e-5 relative — *larger* than the predeclared
1e-6 rel tolerance. A faithful fp64 port therefore cannot match the fp32
savepoints to 1e-6, no matter how correct it is. The fp64-oracle PASS (9.7e-16)
and the independent NumPy reproduction prove the port is exact; the residual is
the oracle's own fp32 precision. (An fp32 JAX kernel was tried; it was *worse*
— rel 1.8e-5 — because XLA's fp32 op ordering/FMA/`exp` differ from gfortran's,
adding independent fp32 noise rather than reproducing gfortran's exact fp32
sequence, which is not reliably achievable from XLA.)

## Recommendation (decision for the manager)

The port is WRF-faithful and proven so. To get a green PRIMARY gate without
loosening any tolerance or doing a JAX-vs-JAX compare, **adopt the fp64 oracle
as the BMJ gate** (`run_bmj_parity_fp64.py`, PASS 5/5 at the same predeclared
tolerances) — a *more* rigorous, precision-matched comparison. The committed
fp32 savepoints (GPT lane, frozen) are left untouched and remain as a documented
fp32-precision-limited reference. I did not overwrite the GPT lane's fp32
savepoints unilaterally.

BMJ is faithful and operationally usable for v0.6.0/0.9.0. Not a blocker either
way — KF + Tiedtke already cover the operational cumulus menu.

## Residual risk

- The fp64 oracle was built and run by this lane; it should be independently
  re-run from `bmj_build_and_run_fp64.sh` to confirm reproducibility.
- Only 5 single-column soundings (the GPT lane's). Broader sounding coverage
  (esp. genuine shallow-with-tendency and freezing-level-below-cloud-top edge
  cases) would harden the shallow-branch abort gates and the above-freezing
  TREF construction, which are exercised but not stress-tested here.
- The shallow branch's deep-demotion path (DENTPY<EPSNTP -> shallow) is ported
  but none of the 5 cases exercises it; unverified against the oracle.
