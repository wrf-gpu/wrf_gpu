# Worker Report

Summary:

The sprint produced the missing h10/d02 JAX pre-step `OperationalCarry`
checkpoint and reran the same-surface pre-halo comparison against Boole's green
CPU-WRF target. The comparison ran and returned `JAX_MISMATCH_T`.

Objective:

Produce a CPU-loadable full carry at completed step 5999, immediately before
`d02` step 6000/h10, then run `proofs/v014/jax_h10_prestep_carry.py`.

Files changed:

- `proofs/v014/jax_h10_prestep_carry_producer.py`
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry_producer.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`
- The canonical `proofs/v014/jax_h10_prestep_carry.{json,md}` and review were
  then rerun by the manager with the produced checkpoint, updating the older
  blocker verdict to `JAX_MISMATCH_T`.

Commands run:

- `python -m py_compile proofs/v014/jax_h10_prestep_carry_producer.py`
- `WRFGPU2_H10_PRODUCER_ALLOW_GPU=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry_producer.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry_producer.py`
- `python -m json.tool proofs/v014/jax_h10_prestep_carry_producer.json >/tmp/jax_h10_prestep_carry_producer.validated.json`

Proof objects produced:

- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/jax_h10_prestep_carry_producer.md`
- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`
- `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

Result:

The checkpoint is CPU-loadable via
`gpuwrf.runtime.checkpoint.read_checkpoint_with_runtime_state`, contains an
`OperationalCarry`, paired `OperationalNamelist`, grid shape
`159 x 66 x 44`, step index `5999`, and SHA256
`0896e4a272cbeaa85d1bb969ecae82b047e75a028df45a87ddab4f4572af8dde`.

Unresolved risks:

- The comparison covers Boole's selected h10 patch, not the full grid.
- The producer uses private proof/runtime helpers and does not add a public
  checkpoint API.
- The first mismatch is `T`, but WRF's accepted history `T` source is
  `grid%th_phy_m_t0`; the next sprint must check JAX history/source semantics
  before assuming a numerical theta operator bug.

Next decision needed:

Open a T history/source-attribution sprint before any production dycore fix.
