# NSSL 2-moment (mp_physics=18) JAX port — conventions

Source of truth: pristine `<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F`
(sha256 29f42e76…, 25,163 lines). Only the WRF-default mp=18 configuration is ported
(ipconc=5, hail on, CCN on with irenuc=5 → activated CCN `lccna`, graupel+hail volume,
icdx=icdxhl=6, all other module defaults). Branches that are dead for that configuration
(3-moment `lz*`, `ipelec`, `mixedphase`, chem, `ipconc<5`, …) are omitted, but every branch
REACHABLE in that configuration is ported even if no oracle case exercises it.

## Oracle / stages

`proofs/v034/oracle/nssl2mom` (pristine + print-only instrumented copy, E26 bit-identity
gate) → savepoints `proofs/v034/f2_oracles/nssl_2mom/nssl{,_fp64}_case_{1..14}.json`.
Driver stages (all `an` values density-scaled: indices 9..18 in #/m3 or m3/m3):

| stage | after | extra columns |
|---|---|---|
| S0 | pack + `t0/t7/t00/t77` + denscale | t0 t7 t00 t77 pn wn dn1 dz2d |
| S1 | `calcnfromq` (itimestep==1 only; S1==S0 otherwise) | |
| S2 | `sediment1d` | `xfall[il]` (surface flux accumulators) |
| S3 | `nssl_2mom_gs` | t0..t9 |
| S4 | `NUCOND` | t0 t9 ssat |
| S5 | `smallvalues` | |
| S6 | `radardd02` | dbz2d |

gs-internal (DEV only, lane dir `<USER_HOME>/wrf_gpu2_lanes/o1-nssl/oracle_v2/build_<mode>_instr/case_<c>.txt`,
loader `tests/v034/nssl2mom/oracle_io.py::load_gs_dump`): `G1` = every gs local after the
collection-efficiency loop (Fortran line 16326), `G2` = after the qlimit/deposition block
(line 19695). gs part A = lines 13715–16326 (S2 → G1), part B = 16327–19695 (G1 → G2),
part C = 19696–25151 (G2 → S3).

Cases: 1–6 = v023 (cold start itimestep=1, dt 60); 7 hail-seeded cold start; 8–12 warm
start (itimestep=2, CN = activated CCN) dt 18/54; 13–14 hail-seeded warm start.

## Code conventions

* Package `src/gpuwrf/physics/nssl2mom/`. Pure JAX (`jax.numpy`, `lax`), no numpy at run time
  except for constants. Functions are vectorised over an arbitrary point shape; column routines
  take `(..., nz)` with k=0 = surface.
* **Fortran species indices verbatim**: `indices.py` (`LT=1 … LCCNA=18`, `LN`, `LVOL`). State
  stacks `an` have leading axis `NA+1 = 19`, index 0 unused. gs-local species-indexed arrays are
  Python dicts keyed by the Fortran index (`qx[LR]`), multi-index → tuple keys (`vtxbar[(LR, 1)]`),
  mirroring `oracle_io.load_gs_dump`.
* **Constants**: `constants.get_constants(mode)` (frozen pristine-init values, Fortran-indexed
  arrays, REAL values already in the build's REAL dtype). Do not re-type constants by hand.
  Large init tables are recomputed at the use site (`mathfun.gaminterp`, `tabqvs(ltemq)` →
  `exp(cawbolton*(temq-273.15)/(temq-cbwbolton))` with `temq = 163.15 + (ltemq-1)*fqsat` in REAL).
* **Precision**: `indices.Prec` — `R` = default REAL (float32 in WRF, float64 in the
  `-fdefault-real-8` oracle), `D` = DOUBLE PRECISION (always float64). Mirror Fortran typing:
  a `DOUBLE PRECISION` local is computed in float64 and rounded to `R` exactly where Fortran
  assigns it to a REAL; a `d0` literal promotes the expression. Python float literals are weakly
  typed (they take the array dtype) — that matches Fortran default-REAL literals.
* **Literal order** (E133): keep Fortran association and operation order, no algebraic
  simplification, keep divisions as divisions; `x**2` (integer power) → `x**2` (lax.integer_pow),
  `x**2.0`/`x**(1./3.)` → real power. `Int(x)` → `jnp.trunc(x).astype(int32)`; `Nint` → round half
  away from zero; `Float(i)` → `.astype(R)`; `Min/Max` → `jnp.minimum/maximum`; `Sign(a,b)` →
  `jnp.where(b >= 0, abs(a), -abs(a))`.
* **Control flow**: data-dependent `IF` → `jnp.where` with BOTH branches computed; guard
  denominators/logs in the untaken branch (`jnp.where(cond, x, 1)`) so no NaN/Inf is produced.
  Sequential overwrites in one Fortran loop body become sequential `where`s. Static config flags
  are Python `if`s. Iteration loops with data-dependent exit → `lax.fori_loop` with a `done` mask.
* **Tests** `tests/v034/nssl2mom/test_<stage>.py`, CPU only, `jax.config.update("jax_enable_x64", True)`,
  eager (no jit needed for 40-point columns). Gate: fp64 port vs fp64 oracle on all 14 cases,
  every species/level, `|port-ref| <= 1e-10*|ref| + atol` (atol tiny, per-variable floor documented);
  fp32 port vs fp32 oracle reported as a band. Tests must be deletion-sensitive (E39): include
  one mutant check (perturb one rate/coefficient → test fails).
* Run on core 25 only: `JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 taskset -c 25 nice -n 19 python -m pytest
  --basetemp <USER_HOME>/wrf_gpu2_lanes/o1-nssl/pytest_<you> -q tests/v034/nssl2mom/test_<stage>.py`.
