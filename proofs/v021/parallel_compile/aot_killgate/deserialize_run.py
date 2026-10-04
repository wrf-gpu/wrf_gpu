#!/usr/bin/env python3
"""AOT KILL-GATE step 2/2 -- DESERIALIZE + EXECUTE + byte-compare, FRESH process.

Reads the blob + metadata ``serialize_one.py`` wrote, in a FRESH process (no
trace / lower / compile of the body), deserializes the PJRT executable, executes
it on the SAME saved inputs, and proves the result is BYTE-IDENTICAL to a freshly
``jax.jit``-compiled reference of the same ``_advance_chunk_fori`` on the same
inputs. Prints the deserialize wall + execute wall (the "warm in seconds" claim)
and ``BYTE_IDENTICAL=True/False`` (+ per-array max_abs_diff if not).

KILL CONDITION: ``BYTE_IDENTICAL=False`` on the real module => AOT is dead.
SECONDARY KILL: deserialize wall is large (erases the warm win) -- reported, the
manager judges against the ~30 min re-lower it replaces.

ENV PARAMETERS
--------------
* ``AOTKG_OUT``     dir holding the blob/meta/inputs (default ``<this dir>/out``).
* ``AOTKG_INPUT`` / ``AOTKG_DOMAIN`` / ``AOTKG_N_STEPS`` / ``AOTKG_RADT_STEPS`` /
  ``AOTKG_DT_S`` -- MUST match the serialize run (the fresh reference rebuilds the
  same real carry/namelist). Defaults match ``serialize_one.py``.
* ``AOTKG_SKIP_FRESH_REF=1`` -- skip rebuilding the fresh-compiled reference and
  compare against the saved ``out_ref_host`` only (use if a second real-State
  build is too costly; weaker, but still proves deserialize+execute runs).
* ``GPUWRF_FORCE_FP64`` / ``JAX_ENABLE_X64`` -- set by the manager's run wrapper.

Run (manager, under the GPU lock, in a FRESH process after serialize_one)::

    scripts/with_gpu_lock.sh -- env \
        PYTHONPATH=<worktree>/src JAX_ENABLE_X64=true GPUWRF_FORCE_FP64=1 \
        python proofs/v021/parallel_compile/aot_killgate/deserialize_run.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_OUT = _HERE / "out"

OUT = Path(os.environ.get("AOTKG_OUT", str(_DEFAULT_OUT)))
INPUT = os.environ.get(
    "AOTKG_INPUT", "<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init"
)
DOMAIN = os.environ.get("AOTKG_DOMAIN", "d01")
N_STEPS = int(os.environ.get("AOTKG_N_STEPS", "2"))
RADT_STEPS = int(os.environ.get("AOTKG_RADT_STEPS", "100"))
DT_S = float(os.environ.get("AOTKG_DT_S", "18.0"))
SKIP_FRESH_REF = os.environ.get("AOTKG_SKIP_FRESH_REF", "0").strip().lower() in (
    "1", "true", "yes", "on",
)
RUN_DATE = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)


def _fail(msg: str, code: int = 2) -> int:
    sys.stderr.write(f"deserialize_run: {msg}\n")
    return code


def main() -> int:
    blob_path = OUT / "advance_chunk.xlaexec"
    meta_path = OUT / "meta.pkl"
    inputs_path = OUT / "inputs.pkl"
    for p in (blob_path, meta_path, inputs_path):
        if not p.is_file():
            return _fail(f"missing artifact {p} -- run serialize_one.py first")

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    import gpuwrf  # noqa: F401,E402  - configures x64 + the version-keyed cache
    from gpuwrf.runtime import operational_mode as om  # noqa: E402

    try:
        import jaxlib  # noqa: E402
        import jaxlib._jax as _jax  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        return _fail(f"cannot import jaxlib._jax ({exc}); jax {jax.__version__} API drift")

    dev = jax.devices()[0]
    print(f"# deserialize_run  jax={jax.__version__}  x64={jax.config.jax_enable_x64}")
    print(f"# device={dev}  kind={getattr(dev, 'device_kind', '?')}")

    blob = blob_path.read_bytes()
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    with open(inputs_path, "rb") as fh:
        in_host = pickle.load(fh)

    in_tree = meta["in_tree"]
    out_tree = meta["out_tree"]
    fingerprint = meta.get("fingerprint", {})

    # ----- fingerprint guard (fail-closed on mismatch; SIGILL-risk per AOT_PLAN) -
    live_fp = {
        "jaxlib_version": getattr(jaxlib, "__version__", "?"),
        "jax_version": jax.__version__,
        "device_kind": str(getattr(dev, "device_kind", None)),
        "platform": str(getattr(dev, "platform", None)),
        "compute_capability": _compute_capability(dev),
        "x64": bool(jax.config.jax_enable_x64),
    }
    fp_match = all(fingerprint.get(k) == live_fp.get(k) for k in live_fp)
    print(f"# fingerprint saved={json.dumps(fingerprint)}")
    print(f"# fingerprint live ={json.dumps(live_fp)}")
    print(f"# fingerprint_match={fp_match}")
    if not fp_match:
        sys.stderr.write(
            "deserialize_run: WARNING fingerprint MISMATCH -- a real load path MUST "
            "fail-open to compile here (SIGILL risk). Proceeding for the gate "
            "experiment so the manager sees the failure mode.\n"
        )

    # ----- DESERIALIZE (no trace/lower/compile) --------------------------------
    if not hasattr(dev.client, "deserialize_executable"):
        return _fail(
            f"dev.client has no deserialize_executable; jax {jax.__version__} API drift"
        )
    try:
        device_list = _jax.DeviceList((dev,))
    except Exception as exc:  # noqa: BLE001
        return _fail(f"jaxlib._jax.DeviceList((dev,)) failed ({exc}); API drift")

    t0 = time.perf_counter()
    try:
        le = dev.client.deserialize_executable(blob, device_list, None)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"deserialize_executable raised ({type(exc).__name__}: {exc}) -- "
            "AOT load KILLED on the real blob.",
            code=1,
        )
    deserialize_wall = time.perf_counter() - t0
    print(f"# DESERIALIZE wall {deserialize_wall:.3f}s  blob {len(blob)} bytes "
          f"({len(blob) / (1024 * 1024):.3f} MB)")

    # ----- SELECT exactly the runtime buffers the executable expects -----------
    # XLA const-folds / DCEs some inputs at compile time, so a naive tree_flatten
    # supplies MORE buffers than the executable wants (the GPU gate: "supplied 130
    # but expected 123"). JAX's own executor keeps only kept_var_idx
    # (pxla.py:373). We replicate that EXACTLY: supply [in_host[i] for i in
    # kept_var_idx], in order. If serialize could not capture kept_var_idx (API
    # drift) we fall back to the naive flatten and let the buffer-count mismatch
    # surface clearly.
    kept_var_idx = meta.get("kept_var_idx")
    expected = meta.get("expected_runtime_buffers")
    kvi_source = meta.get("kept_var_idx_source")
    if kept_var_idx is not None:
        try:
            selected = [in_host[i] for i in kept_var_idx]
        except IndexError as exc:
            return _fail(
                f"kept_var_idx out of range for {len(in_host)} input leaves ({exc}); "
                "meta/blob mismatch -- re-run serialize_one.",
                code=1,
            )
        print(f"# input-buffer select: naive={len(in_host)} -> kept={len(selected)} "
              f"(dropped {len(in_host) - len(selected)})  source={kvi_source}")
    else:
        selected = list(in_host)
        sys.stderr.write(
            "deserialize_run: WARNING kept_var_idx unavailable in meta "
            f"(source={kvi_source}); using naive flatten of {len(selected)} buffers -- "
            "expect a buffer-count mismatch if XLA dropped any inputs.\n"
        )
    if expected is not None and len(selected) != expected:
        sys.stderr.write(
            f"deserialize_run: WARNING selected {len(selected)} buffers but serialize "
            f"recorded expected_runtime_buffers={expected}; proceeding (execute will "
            "surface the truth).\n"
        )

    # ----- EXECUTE on the SAME saved inputs (kept-filtered) --------------------
    in_bufs = [jax.device_put(np.asarray(x), dev) for x in selected]
    # warm/throwaway run to realize the executable, then a timed run.
    try:
        warm = le.execute(in_bufs)
        for b in warm:
            getattr(b, "block_until_ready", lambda: None)()
    except Exception as exc:  # noqa: BLE001
        return _fail(
            f"le.execute raised on first run ({type(exc).__name__}: {exc}) -- "
            f"supplied {len(in_bufs)} buffers. AOT execute KILLED on the real blob.",
            code=1,
        )
    t1 = time.perf_counter()
    out_bufs = le.execute(in_bufs)
    for b in out_bufs:
        getattr(b, "block_until_ready", lambda: None)()
    execute_wall = time.perf_counter() - t1
    print(f"# EXECUTE wall {execute_wall:.3f}s  ({len(out_bufs)} output leaves)")

    out_aot = jax.tree_util.tree_unflatten(out_tree, out_bufs)
    out_aot_flat = jax.tree_util.tree_leaves(out_aot)

    # ----- REFERENCE: a FRESH normally-jit-compiled _advance_chunk_fori ---------
    ref_source = "fresh_jit"
    if SKIP_FRESH_REF:
        ref_source = "saved_out_ref_host"
        ref_flat = [np.asarray(x) for x in meta["out_ref_host"]]
    else:
        try:
            ref_flat = _fresh_reference(jax, jnp, np, om)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(
                f"deserialize_run: fresh reference build failed ({type(exc).__name__}: "
                f"{exc}); falling back to saved out_ref_host\n"
            )
            ref_source = "saved_out_ref_host(fallback)"
            ref_flat = [np.asarray(x) for x in meta["out_ref_host"]]

    # ----- byte-compare every output array -------------------------------------
    aot_host = [np.asarray(x) for x in out_aot_flat]
    byte_identical = True
    diffs = []
    if len(aot_host) != len(ref_flat):
        byte_identical = False
        diffs.append(
            {"leaf": -1, "error": f"leaf count {len(aot_host)} != ref {len(ref_flat)}"}
        )
    else:
        for i, (a, r) in enumerate(zip(aot_host, ref_flat)):
            a = np.asarray(a)
            r = np.asarray(r)
            if a.shape != r.shape or a.dtype != r.dtype:
                byte_identical = False
                diffs.append(
                    {"leaf": i, "shape_a": list(a.shape), "shape_r": list(r.shape),
                     "dtype_a": str(a.dtype), "dtype_r": str(r.dtype)}
                )
                continue
            same = np.array_equal(a, r)
            if not same:
                byte_identical = False
                max_abs = float(np.max(np.abs(a.astype(np.float64) - r.astype(np.float64))))
                diffs.append({"leaf": i, "max_abs_diff": max_abs, "shape": list(a.shape)})

    verdict = {
        "BYTE_IDENTICAL": bool(byte_identical),
        "reference": ref_source,
        "fingerprint_match": bool(fp_match),
        "deserialize_seconds": round(deserialize_wall, 4),
        "execute_seconds": round(execute_wall, 4),
        "blob_bytes": len(blob),
        "blob_mb": round(len(blob) / (1024 * 1024), 3),
        "naive_input_leaves": len(in_host),
        "supplied_runtime_buffers": len(selected),
        "expected_runtime_buffers": expected,
        "dropped_input_leaves": len(in_host) - len(selected),
        "kept_var_idx_source": kvi_source,
        "n_output_leaves": len(aot_host),
        "n_diff_leaves": len(diffs),
        "diffs": diffs[:20],  # cap the dump
        "jax_version": jax.__version__,
    }
    print("\n# KILL-GATE VERDICT")
    print(json.dumps(verdict, indent=2))
    out_json = OUT / "killgate_verdict.json"
    with open(out_json, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"\n# wrote {out_json}")
    print(f"# RESULT: BYTE_IDENTICAL={byte_identical}  "
          f"deserialize={deserialize_wall:.3f}s  execute={execute_wall:.3f}s")
    return 0 if byte_identical else 1


def _fresh_reference(jax, jnp, np, om):
    """Rebuild the SAME real carry + a FRESH jit-compiled _advance_chunk_fori and
    return its flat output arrays (host). Independent of the saved blob."""
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.integration.d02_replay import build_replay_case
    from gpuwrf.io.radiation_static import load_radiation_static

    replay = build_replay_case(INPUT, domain=DOMAIN, standalone=True)
    radiation_static, _ = load_radiation_static(
        replay.run, DOMAIN, grid=replay.grid, metrics=replay.metrics
    )
    namelist = OperationalNamelist.from_grid(
        replay.grid,
        tendencies=replay.tendencies,
        metrics=replay.metrics,
        dt_s=DT_S,
        acoustic_substeps=4,
        radiation_cadence_steps=RADT_STEPS,
        force_fp64=True,
        radiation_static=radiation_static,
    )
    namelist = dataclasses.replace(namelist, time_utc=RUN_DATE)
    carry = om._committed_initial_carry_for_run(replay.state, namelist)
    clock_base = om.build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)
    start = jnp.asarray(1, dtype=jnp.int32)
    jfn = jax.jit(om._advance_chunk_fori, static_argnames=("n_steps", "cadence"))
    out = jfn(carry, namelist, start, clock_base, n_steps=int(N_STEPS), cadence=cadence)
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]


def _compute_capability(dev) -> str | None:
    val = getattr(dev, "compute_capability", None)
    return str(val) if val is not None else None


if __name__ == "__main__":
    sys.exit(main())
