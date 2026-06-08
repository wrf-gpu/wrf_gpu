# Adversarial skeptic review — classic RRTM longwave JAX kernel

- **Reviewer**: Opus 4.8 (adversarial skeptic, independent of the kernel/oracle author)
- **Trunk base**: `6d051f8` (READ-ONLY review; no code changes)
- **Kernel under audit**: `src/gpuwrf/physics/ra_lw_rrtm_jax.py` (`solve_rrtm_lw_column_jax`,
  16-band / 140-g-point JIT/vmap-traceable JAX; opt-in `ra_lw_physics=1`)
- **Host reference kernel**: `src/gpuwrf/physics/ra_lw_rrtm.py` (`solve_rrtm_lw_column`,
  NumPy, pristine-WRF-parity-proven by `proofs/v060/run_rrtm_lw_parity.py`)
- **Oracle**: `proofs/radiation/rrtm_lw_oracle.py` + savepoints `proofs/v060/savepoints_fp64/`
- **WRF ground truth**: `/home/enric/src/wrf_pristine/WRF/phys/module_ra_rrtm.F:RRTMLWRAD`
- **Constraint honored**: `JAX_PLATFORMS=cpu`, no GPU context, cores 0-3.

## VERDICT: SOUND (within the operational envelope) — with TWO carry-over findings

The kernel **math is faithful**. Across an aggressive adversarial sweep, the JAX kernel
reproduces the WRF-parity-proven host kernel **to fp64 round-off (rel ≤ ~3e-13)** in every
physically-realizable column where layer pressures stay positive — including regimes the 7
committed savepoints do NOT cover (low-psfc to 700mb, shallow 8-level / deep 60-level grids,
hot 342K surface, cold polar, strong inversion, high cirrus, very thick multi-species cloud,
surface fog, bone-dry, super-humid). No band-index, g-point-accumulation, RTRN-sweep,
laytrop, or Planck defect was found.

The two findings below are **not JAX porting bugs** — they are a faithfully-ported WRF design
fragility plus a robustness asymmetry — but they are real, currently unproven, and worth a
carry-over because the proofs systematically avoid them.

---

## Oracle integrity check (the author could not self-give this) — CLEAN

The concern was a shared author error self-confirming via the oracle. Refuted:

- The oracle reference is **genuinely pristine WRF**, not a JAX-vs-JAX self-compare.
  `proofs/v060/oracle/rrtm_lw_oracle_driver.f90` compiles the **unmodified**
  `module_ra_rrtm.F:RRTMLWRAD` (`rrtm_lw_build_and_run.sh`), and the committed
  `rrtm_lw_wrf_source_checksums.txt` SHA-256 `9f54e70d…` **exactly matches** the on-disk
  pristine `/home/enric/src/wrf_pristine/WRF/phys/module_ra_rrtm.F`.
- The fp64 savepoints are a real `-fdefault-real-8` recompile reading `RRTM_DATA_DBL`
  (distinct `RRTM_DATA` checksum confirms a separate double-precision asset), so the fp64
  verdict is a true double-precision Fortran reference.
- The fp32 "rel ~3e-4" is the **Fortran REAL\*4 oracle's own** single-precision round-off,
  NOT a tolerance hiding a JAX error: the fp64 oracle vs fp64-JAX agrees to ~1e-13, so the
  fp32 gap is entirely WRF's single-precision dust through the long sequential
  per-layer/g-point accumulation. The fp32 claim is honest.

## Algorithmic cross-check vs WRF — CLEAN

Verified line-by-line against `module_ra_rrtm.F` and confirmed numerically equal to the
WRF-proven host kernel:

- **Band-index / g-point mapping**: `NGC=[8,14,16,14,16,8,12,8,12,6,8,8,4,2,2,2]` (Σ=140),
  `NGB`, `NGB_START` are **imported directly from the host module** (ra_lw_rrtm_jax.py:43-58)
  — identical to the parity-proven host; no re-derivation, so no divergence possible.
- **`jnp.take` clamped 1-based gathers** (`_row`/`_interp4`/`_binary_lower`,
  ra_lw_rrtm_jax.py:223-263): match host `_row`/`_interp4`/`_binary_lower`; the clamp only
  bites where the host would clamp too. No off-by-one found.
- **laytrop / layswtch / laylow** (`_setcoef_jax`, ra_lw_rrtm_jax.py:489-526): the per-layer
  `plog>4.56` mask is contiguous-from-bottom on every tested profile (verified
  `lead == sum(mask)`), so the scalar-count → mask vectorization is valid. The layswtch
  round-off `+1` correction (line 522-523) matches WRF SETCOEF:4362-4364, with an added
  `< nlayers` bound that mirrors the host and is unreachable in practice.
- **Band 9 `ioff` persistent scalar** (the riskiest control-flow, ra_lw_rrtm_jax.py:852-874):
  the `>=`-formulation `where(idx+1>=layswtch, 2ng, where(idx+1>=laylow, ng, 0))` is provably
  equivalent to WRF's equality-triggered carry (TAUGB9:5569-5570) given
  `laylow ≤ layswtch ≤ laytrop`; numerically matches host on deep/shallow/low-psfc columns.
- **One-angle RTRN** (`_rtrn_jax`, ra_lw_rrtm_jax.py:970-1083): `lax.scan` directions, the
  `bglev` PFRAC(above)·PLVL(current-top) recurrence, surface emissivity reflection
  (`radlu0 = raduemit + (1-semis)·radld`), level-vs-layer indexing
  (`plvl_g[idx]`/`play_g[idx]`, `bglev0 = pfrac[n-1]·plvl[n]`), and `WTNUM`/`fluxfac`/`heatfac`
  scaling all match WRF RTRN:6195-6602 and the host kernel. OLR taken at the **model top**
  (`totuflux[nz]`, line 1093), not the buffer top — correct. The clear-sky stream is threaded
  but feeds only the unused htrc/fnetc diagnostics; it cannot affect the emitted
  heating/GLW/OLR (all from the all-sky `radld`/`radlu`).
- **GASABS itr index** (`_gasabs_jax`, ra_lw_rrtm_jax.py:948-953): `tff` Pade + `INT(5e3·tff+0.5)`
  match GASABS:6748-6755; the `clip(…,0,5000)` only protects the upper boundary.
- **Planck** (`_planck_band`, ra_lw_rrtm_jax.py:959-967): `idx=INT(T-159)` clamp [1,180] and
  `frac=T-INT(T)` match WRF RTRN:6334-6394 and host `_planck_value`.

---

## FINDING 1 (MEDIUM, carry-over) — buffer layer count hardcoded to ptop=5000 Pa; not grid-aware

`_nbuf()` (ra_lw_rrtm_jax.py:298-299, and identically host ra_lw_rrtm.py:562-563) sizes the
above-model-top buffer from a **hardcoded `DEFAULT_PTOP_PA = 5000.0`**:

```python
def _nbuf() -> int:
    return _nint(DEFAULT_PTOP_PA * 0.01 / DELTAP_MB)   # = nint(50/4) = 13, ALWAYS
```

The buffer interfaces are then built `pz[l] = pz[l-1] - DELTAP_MB (=4mb)` for 13 layers
(ra_lw_rrtm_jax.py:341-343), bridging the model top to TOA. WRF's `module_ra_rrtm.F:3965`
sizes this identically as `NLAYERS = kte + int(p_top/4) + 1`, **but from the namelist `p_top`**,
whereas this kernel ignores the grid's `vertical.top_pressure_pa` entirely — the coupler
`_rrtm_lw_column_inputs` (physics_couplers.py:1868) never plumbs grid top pressure to the
kernel.

Consequence (measured, host == JAX exactly, so faithful but unvalidated):
the 13×4 = 52 mb buffer span is correct **only when the model-top interface ≈ 50 mb**. Sweeping
the model top (probe_ptop.py):

| model-top P8W | buffer reaches | OLR (W/m²) | status |
|---|---|---|---|
| 284 mb (ztop 10km) | 236 mb (should be ~0) | 310 | undershoots TOA |
| 112 mb (ztop 16km) |  64 mb | 282 | undershoots TOA |
|  54 mb (ztop 20km) |   6 mb | 274 | ~correct |
|  21 mb (ztop 25km) | **−27 mb** | 272 | negative-pressure (see Finding 2) |

For ptop > 5000 Pa the buffer stops short of TOA, leaving one oversized top layer and biasing
the model-top→TOA downward flux; for ptop < 5000 Pa it overshoots into negative pressures.
**The 7 oracle savepoints all hardcode `p_top=5000.0` (driver line 46), so this is never
exercised.** Severity is MEDIUM because the standard operational config uses ptop=5000 Pa
(so production today is fine), but any non-default ptop (or a high-top domain) silently gets a
mis-sized RRTM buffer with no proof coverage.

**Suggested fix**: pass the grid's `vertical.top_pressure_pa` into the kernel and size
`nbuf = nint(ptop_mb / DELTAP_MB)` from it (a static int given the grid), and add an oracle
savepoint at a non-5000 ptop (regenerate the Fortran oracle with a matching `p_top`).

## FINDING 2 (LOW–MEDIUM, carry-over) — defensive clamps convert a WRF/host crash into SILENT finite garbage

When the buffer (Finding 1) drives layer pressures negative — model-top interface below
~52 mb, e.g. a high-top or aggressively-low-ptop column — the prepared atmosphere becomes
unphysical (negative `pavel`, negative `coldry`, `plog = log(neg) = NaN`). I verified host and
JAX produce **byte-identical** prepared `pavel`/`coldry` here (`max|Δ| = 0.0`), confirming the
port is faithful. But the **outcomes diverge**:

- **Host / WRF**: raw indexing → `IndexError` (band-2 `corr1/corr2`, size 201; garbage index
  686336) — i.e. a hard, visible failure.
- **JAX kernel**: `jnp.maximum(pavel, 1e-300)` in `plog` (ra_lw_rrtm_jax.py:476) and
  `jnp.clip(ifp, 0, 200)` in `_gb2` (line 590-591) **swallow the NaN/out-of-range index and
  emit a plausible-looking finite result** (e.g. ztop=25km → JAX `glw=371.3, olr=271.8`, all
  finite; host crashes).

This is the project's explicitly-forbidden "masking clamp" pattern: on a pathological column an
operational `ra_lw_physics=1` run would not crash or NaN-propagate — it would inject **silent,
physically-meaningless** LW heating. The full-wired-path proof part (C) only checks
`|RTHRATEN| < 1e-2 K/s` and finiteness, which this garbage **passes**, so the sanity gate would
not catch it.

**Suggested fix**: once Finding 1 is fixed the negative-pressure regime is largely removed; in
addition, replace the silent clamps with an explicit guard that the prepared `pavel`/`pz` are
strictly positive (assert in validation mode, or NaN-propagate rather than clip in operational
mode) so a mis-configured column fails loudly instead of producing quiet garbage.

---

## Falsification artifacts (CPU, cores 0-3, JAX_PLATFORMS=cpu)

Scratch (not committed): `/tmp/rrtm_audit/skeptic_probe.py` (15 adversarial profiles, host-vs-JAX),
`/tmp/rrtm_audit/probe_lowpsfc.py` (negative-pressure root cause), `/tmp/rrtm_audit/probe_ptop.py`
(buffer-vs-ptop sweep). Trunk `6d051f8` modules extracted to `/tmp/rrtm_trunk` for isolated
CPU import. Worktree and repo were not modified.

Headline numbers: positive-pressure envelope max divergence **2.7e-13** (bone_dry); negative-
pressure regime = host crash / JAX silent-finite. fp64 oracle vs JAX = ~1e-13 across all 7
committed cases; fp32 ~3e-4 = genuine Fortran REAL\*4 round-off.
