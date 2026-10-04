# F1 Implementation Report - Real CPU Gate Final

Date: 2026-07-02

## Status

`PASS` for the corrected real-dycore CPU B=2 F1 gate.

The strict byte-identical-vs-standalone expectation was replaced after the real
diagnosis proved root cause class **B**: XLA shape/fusion/accumulation order
changes under `jax.vmap`, not cross-lane contamination.  The accepted CPU
correctness property is now:

- contamination-free vmap orchestration;
- per-lane real-dycore differences within the measured fused-vs-eager band on
  the same fixture.

Proof script:

- `proofs/v023/batched_minigrid/f1_cpu_b2_gate.py`

Proof JSON:

- `proofs/v023/batched_minigrid/F1_CPU_B2_BIT_IDENTITY.json`

Diagnosis:

- `proofs/v023/batched_minigrid/F1_REALGATE_DIAGNOSIS.md`

## Gate Fixture

The gate builds a tiny analytic two-domain `d01 -> d02` nest using real
`GridSpec`, `State`, `OperationalNamelist`, `OperationalCarry`, `DomainTree`,
`_advance_chunk`, and `build_child_boundary_package`.  It runs two root steps,
child ratio 3, and output cadence on both domains on the JAX CPU backend.

The corrected gate runs:

- two different standalone eager lanes;
- B=1 singleton-vmap diagnostic lane;
- B=2 eager batched lanes, `GPUWRF_BATCH_ENSEMBLE=2` and de-fused/bitwise mode;
- B=2 identical-IC lanes for contamination detection;
- B=2 fused batched lanes, `GPUWRF_NESTED_FUSE=1`, to measure the fused-vs-eager
  tolerance band on the same fixture.

## Result

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
  GPUWRF_NESTED_AOT=0 python proofs/v023/batched_minigrid/f1_cpu_b2_gate.py
```

Result:

```text
PASS F1 real CPU B=2 contamination-free tolerance gate: F1 max_abs=5.82076609135e-11 <= fused_band=1.78624759428e-09; F1 max_rel=1.99494331303e-14 <= fused_band=1.43511125826e-12
```

Global maxima:

| Comparison | Max absolute | Field | Max relative | Field |
|---|---:|---|---:|---|
| F1 B=2 eager batched vs standalone | `5.820766091346741e-11` | `lane0/d02/carry.ph_tend` | `1.994943313031931e-14` | `lane0/d02/carry.ww` |
| Fused-vs-eager band, same fixture | `1.786247594282031e-09` | `lane1/d02/carry.ph_tend` | `1.4351112582583804e-12` | `lane1/d01/carry.ww` |
| Default fused B=2 batched vs standalone | `1.786247594282031e-09` | `lane1/d02/carry.ph_tend` | `1.4351112582583804e-12` | `lane1/d01/carry.ww` |

The full JSON includes max absolute and max relative differences per lane,
domain, and field under:

- `comparisons.f1_batched_eager_vs_standalone`
- `comparisons.fused_vs_eager_band`
- `comparisons.f1_default_fused_batched_vs_standalone`

## Contamination Checks

All corrected checks passed:

- singleton-vmap lane reproduces B=2 lane0 byte-identically;
- identical-IC B=2 lane0 and lane1 are byte-identical for all compared fields;
- identical-IC B=2 output lane digests are byte-identical;
- B=2 eager and B=2 fused output schedules match standalone schedules;
- F1 max absolute and max relative differences are within the measured global
  fused-vs-eager band.

## Implementation Files

- `src/gpuwrf/runtime/domain_tree.py`
  - `GPUWRF_BATCH_ENSEMBLE` fixed-size resolver.
  - Per-domain `DomainTree.batch_namelists`.
  - Outer `jax.vmap` wrappers around `_advance_chunk` and forcedown.
  - Batched root fused-cascade path.
- `src/gpuwrf/integration/nested_pipeline.py`
  - Homogeneous batch loading/stacking.
  - Per-lane namelist clock threading.
  - Per-lane output writer slicing.
- `tests/test_v023_batched_minigrid.py`
  - Pytest wrapper for the corrected proof gate.

## Remaining Work

F1 CPU correctness is done: opt-in, contamination-free, tolerance-not-bitwise.
The remaining F1 value gate is GPU G3 later: throughput and VRAM across swept
batch sizes.
