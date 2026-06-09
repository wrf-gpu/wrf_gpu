# Tester Report

Decision: pass for the producer sprint proof gate.

The manager reran validation after taking over the worker artifacts. The
redundant second GPU producer process was terminated only after the checkpoint,
JSON, markdown, and review artifacts were already present and stable; no model
or TOST process remained active afterward.

Commands rerun by the manager:

- `python -m json.tool proofs/v014/jax_h10_prestep_carry_producer.json >/tmp/jax_h10_prestep_carry_producer.manager.validated.json`
- `python -m py_compile proofs/v014/jax_h10_prestep_carry_producer.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src WRFGPU2_H10_PRESTEP_CARRY=/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl python proofs/v014/jax_h10_prestep_carry.py`
- `python -m json.tool proofs/v014/jax_h10_prestep_carry.json >/tmp/jax_h10_prestep_carry.manager.validated.json`
- `python -m py_compile proofs/v014/jax_h10_prestep_carry.py proofs/v014/jax_h10_prestep_carry_producer.py`

Observed result:

- Producer JSON validates.
- Canonical h10 compare JSON validates.
- Both proof scripts compile.
- Canonical rerun exits zero and prints `JAX_MISMATCH_T`.
- GPU compute is free after terminating the redundant second producer process.

Coverage limits:

- No TOST run was started.
- No Switzerland run was started.
- No FP32 source change was tested.
- No production model source was edited.
