# Stability localization: the 9-24h coupled guards-OFF non-finite blowup

Task #36 (diagnostic, READ-ONLY on src). Branch `worker/opus/stability-localize`,
base `177c734`. Real case: Gen2 d02 `20260521_18z_l3_24h` (nz=44, ny=66, nx=159; dt=10s,
n_acoustic=10; fp64; top_lid=True, epssm=0.5, w_damping=1, damp_opt=3, zdamp=5000,
dampcoef=0.2; guards OFF). All GPU/JAX; no WRF launched.

## Surgical root cause

**The MYNN PBL coupler's `w` face->mass->face round-trip corrupts the rigid-lid top
`w` boundary, seeding an explosive top-of-column `w` acoustic instability that goes
non-finite at step 5413 = 15.04h.**

- Field that blows first: **`w` (vertical velocity), at the model-top level k=43**
  (topmost interior level; nz=44, top face k=44). `u` is dragged along at k=43; `theta`/`v`
  join only at the final 1-2 steps.
- Onset: **step 5413, lead 15.04h** (deterministic; identical with or without
  Thompson/surface/radiation).
- Growth curve (explosive, ~1.3-1.5x/step doubling, NaN in ~11 steps / ~110s):
  w@k43 = 19 -> 22 -> 26 -> 31 -> 38 -> 49 -> 67 -> 100 -> 160 -> 213 -> 370 -> NaN;
  u@k43 -> 962; theta -> 744 at last finite step. Before ~step 5400, w sits at ~10-14 m/s
  at mid-level k17-18, perfectly stable 1h..14h, then suddenly migrates to the top and erupts.

## Bisection (component-disable; "bisection before theory")

| variant | physics on | result |
|---|---|---|
| dycore-only (`dycore_only_walk_bndy.jsonl`) | none (run_physics=False) | **STABLE to 24h** (w~14@k15, u 31->20 decaying) |
| `no_rrtmg` (`scan_trace_no_rrtmg.json`) | Thompson+surface+MYNN | NON-FINITE @ **15.04h** (step 5413) |
| `norad_no_mynn` (`scan_trace_norad_no_mynn.json`) | Thompson+surface | **STABLE to 17h** (w~14@k15, no top migration) |
| `norad_only_mynn` (`scan_trace_norad_only_mynn.json`) | MYNN only | NON-FINITE @ **15.04h** (step 5413, identical) |
| fix=preserve-w (`fix_validate.json`) | Thompson+surface+MYNN, MYNN.w restored | **FINITE to 17h** (top eruption gone) |

=> MYNN is **necessary** (remove it -> stable) and **sufficient** (MYNN-alone -> identical
blowup). Radiation (the lumped-cadence heating, GPT P0-2), Thompson, surface, the
periodic-advection LAM mode (ROOT_CAUSE Mode B), and the open-top w solve are all RULED
OUT: dycore-only with rigid lid + real boundaries is stable for the full 24h, and the
onset step is bit-identical whether or not the other physics run.

## Mechanism (direct probe, `mynn_w_roundtrip.log`)

MYNN solves u/v/theta/qv (NOT w). But the coupler reconstructs w anyway
(`physics_couplers.py:777` `w=_mass_to_w_face(w_mass)`), after reading it via `_w_mass`
(face->mass average). `face(nz+1) -> mass(nz) -> face(nz+1)` is NOT identity:

- BEFORE MYNN: top w-face `w[k=44] = 0.0` exactly (rigid lid).
- AFTER one MYNN call: `w[k=44] = 1.04e-5` (nonzero). The pure round-trip with NO MYNN
  physics reproduces the identical 1.04e-5, proving it is the staggering reconstruction,
  not the turbulence calculation.

So every step MYNN re-introduces a small nonzero w at the model-top face that
`top_lid=True` must keep at exactly 0. ~1e-5/step is tiny but persistent; over 5400
steps it seeds and feeds the top-of-column w acoustic mode (the same mode the open-top
solve excited instantly on step 1 in `proofs/dycore_realinit/`) until it crosses a
threshold and erupts. The rigid lid that makes the dycore-only run stable is silently
defeated by MYNN every step.

## Best-hypothesis fix (for the coupler-owning agent; NOT applied here)

`src/gpuwrf/coupling/physics_couplers.py::_state_from_mynn_output` (line 777): do NOT
reconstruct/round-trip `w` — MYNN does not change it. Replace `w=_mass_to_w_face(w_mass)`
with keeping the input `w` (the dynamics already set the correct diagnostic surface/top
w faces and enforced the rigid lid). Validated in `fix_validate.json`: preserving w
removes the 15h NaN (finite to 17h, u behaves like dycore-only). The same face->mass->face
round-trip also smooths u/v every step (lines 775-776); those are edge-corrected by
boundaries so they are not the NaN trigger, but reconstructing only the levels MYNN
actually changed (or skipping the round-trip for unchanged fields) is the clean fix.

CAVEAT on the validation probe: the crude `out.replace(w=state.w)` restore left a FINITE
but unphysical w growth at the k=0 surface face (->814 m/s by 17h) because it also
restored the dycore's k0 terrain-BC face from the MYNN input; the production fix (don't
round-trip w at all) avoids both the top-BC corruption and this surface artifact.

It is NOT precision-sensitive: w is already fp64-locked through MYNN (the parallel P0-1
fp32 fix is merged; `_output_dtype`), and the round-trip corruption is a staggering/BC
error independent of dtype.

## Probe scripts (reproduce)

```
# dycore-only 24h (run_physics=False) -> STABLE
PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 taskset -c 2-3 \
  python proofs/stability/dycore_only_walk.py --hours 24 --run-boundary 1
# single-compile per-step onset trace, any variant (no recompile, bounded mem):
PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 taskset -c 2-3 \
  python proofs/stability/scan_trace.py --hours 24 --variant no_rrtmg
  python proofs/stability/scan_trace.py --hours 17 --variant norad_no_mynn
  python proofs/stability/scan_trace.py --hours 17 --variant norad_only_mynn
# mechanism probe (top w-face corruption):
PYTHONPATH=src taskset -c 2-3 python proofs/stability/mynn_w_roundtrip.py
# fix-direction validation (MYNN preserve-w):
PYTHONPATH=src taskset -c 2-3 python proofs/stability/fix_validate.py --hours 17
```

NOTE on method: `_advance_chunk` does NOT recompile on varying `start_step` when
run_physics=False (cache hits, `recompile_probe.log`), but with physics ON it
re-specializes per distinct start_step (each host-loop segment recompiles ~180s). The
`scan_trace.py` machinery sidesteps this by running the whole forecast as ONE
`jax.lax.scan` (one compile, bounded memory) and emitting per-step extrema as the
stacked scan output. RRTMG's ~8 GiB per-step transient OOMs under GPU contention, so the
memory-light bisection variants keep radiation OFF (radiation is irrelevant to the blowup).
