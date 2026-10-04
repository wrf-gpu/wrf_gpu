# F2 Question — mp_physics=18 (NSSL 2-moment) faithful JAX port scope

**RESOLVED (0:1 ruling, 2026-07-03): mp18 = OWN MILESTONE, not part of F2.**
F2 finalized with mp18 at reference-only (oracle + state contract + runbook
delivered as the port sprint's launchpad). Kept below as the decision record.

## Status of the rest of F2 (not blocked)

This question covers ONLY the NSSL mp=18 **JAX kernel**. Everything else in the
F2 family is delivered and proven — see `F2_REPORT.md` (RUC integrated+validated,
New-Tiedtke ported+validated, Morrison-aerosol oracle+port at machine precision,
NSSL oracle built+verified).

## The blocker, with measured evidence

GPT's original F2 blocker for mp=18 was "no local single-column oracle". That is
now SOLVED: `proofs/v022/f2_oracles/nssl_2mom/` (fp32+fp64 × 6 regimes,
checksummed unmodified `phys/module_mp_nssl_2mom.F`, WRF-default mp=18 config
cited to `module_check_a_mundo.F`/`module_physics_init.F`; builder + README at
`proofs/v023/oracle/nssl2mom/`).

What remains is the faithful port itself. Measured size of the oracle-exercised
call path inside the 25,163-line module:

| routine        | LOC     | role |
|----------------|---------|------|
| `nssl_2mom_gs` | ~12,533 | all process rates (2-moment, 7 classes + CCN) |
| `nucond`       | ~2,296  | saturation adjustment + droplet nucleation |
| `setvtz`/`ziegfall1d`/`sediment1d` | ~2,800 | fall speeds + hybrid number-fallout sedimentation |
| `calcnfromq`, `smallvalues`, pack/scale plumbing | ~1,300 | init/cleanup |
| **total**      | **~19k** | vs ~3.6k for the whole New-Tiedtke core |

For calibration from this sprint: the New-Tiedtke port (3.6k LOC Fortran) took a
full focused session to reach machine-precision parity (including fp32-literal
and `amax1` fidelity archaeology); the mp40 port was tractable in one agent-run
ONLY because it is a delta on an already-proven base Morrison port. NSSL has no
existing base to lean on. A faithful machine-precision port is a **dedicated
multi-day milestone (own sprint contract)**, and rushing it through subagents
would produce exactly the plausible-but-unverified physics the project
constitution forbids (no proof object → no done claim).

## What F2 delivers for mp=18 instead (proven, committed)

* Real single-column oracle (the former hard blocker) — BUILT + VERIFIED.
* Honest seam: mp=18 flipped `recognized_fail_closed` → `REFERENCE_ONLY`
  (namelist-accepted for single-column oracle comparison; operational scan
  fail-closes with a named reason; endpoint stub raises, never silently wrong).
* Frozen state contract for the future port
  (`physics.microphysics_nssl2mom`): moist `qv..qg(=NSSL QH graupel),qh(=QHL
  hail)`, numbers `Nn,Nc,Nr,Ni,Ns,Ng,Nh`; **documented gap:** `qvolg`/`qvolh`
  volume scalars have no State substrate (needs a small ADR before wiring).
* Porter's runbook in `proofs/v023/oracle/nssl2mom/README.md` (driver call
  semantics, per-slab order, density scaling, sedimentation substeps, fp64
  promotion behavior).
* Known oracle gap to fix in the port sprint: current seeds never exercise the
  hail (QHL) process rates → add a hail-seeded supplementary case then.

## Decision requested

Schedule the NSSL mp=18 faithful port as its own milestone/sprint contract
(recommended), rather than counting it against F2. If instead F2 must include
it, say so explicitly and I will start it as the next long-running task — but
it will not be quick, and I will not fake it.
