#!/usr/bin/env python3
"""AOT KILL-GATE step 1/2 -- SERIALIZE one REAL de-fuse `_advance_chunk` executable.

WHY (the go/no-go this proves)
------------------------------
The de-fuse compile-wall fix (AOT_PLAN.md) rests on one unproven claim: that the
PJRT executable for a REAL `_advance_chunk_fori` module (RRTMG SW+LW + Noah-MP,
the huge constant-folding-heavy graph) can be `serialize()`d to a blob and, in a
FRESH process, `deserialize_executable()`d + executed WITHOUT any re-trace /
re-lower / re-compile, byte-identically. The cross-domain-parallel-compile
workflow only proved this on TOY modules. This script + ``deserialize_run.py``
close that gap on a real module.

This script builds ONE real single-domain `_advance_chunk_fori` (the SMALLEST real
body -- bigswiss d01, tiny ``n_steps`` -- so the GPU compile is ~minutes, NOT the
~50 min 9-nest), compiles it via the SAME machinery the runtime + the parallel-
prewarm use, then serializes the compiled executable + the metadata a fresh
process needs to rebuild the call (in_tree / out_tree / flat in_avals / a
target-fingerprint / the concrete input arrays). Everything is written to
``out/`` for ``deserialize_run.py``.

NUMERICS
--------
Inert. This compiles the IDENTICAL HLO the runtime would build for the same
carry/namelist/clock_base/n_steps/cadence (``_advance_chunk_fori`` is a bare
``@jax.jit`` no-donate). The serialized blob IS that compiled executable's bytes.

ENV PARAMETERS
--------------
* ``AOTKG_INPUT``       input dir for ``build_replay_case`` (default the v017
                        bigswiss single-domain GPU init).
* ``AOTKG_DOMAIN``      domain label (default ``d01``).
* ``AOTKG_N_STEPS``     traced loop count (default 2; keep tiny for a fast compile).
* ``AOTKG_RADT_STEPS``  radiation cadence steps (default 100; large => the radiation
                        cond is traced but rarely fires -- HLO unchanged, fast).
* ``AOTKG_DT_S``        timestep seconds for the namelist (default 18.0, d01-3km).
* ``AOTKG_OUT``         output dir (default ``<this dir>/out``).
* ``GPUWRF_FORCE_FP64`` / ``JAX_ENABLE_X64`` -- set by the manager's run wrapper.

Run (manager, under the GPU lock)::

    scripts/with_gpu_lock.sh -- env \
        PYTHONPATH=<worktree>/src JAX_ENABLE_X64=true GPUWRF_FORCE_FP64=1 \
        python proofs/v021/parallel_compile/aot_killgate/serialize_one.py
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

# --------------------------------------------------------------------------- #
# Env parameters (all defensive: clear error if a required input is missing).
# --------------------------------------------------------------------------- #
INPUT = os.environ.get(
    "AOTKG_INPUT", "<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init"
)
DOMAIN = os.environ.get("AOTKG_DOMAIN", "d01")
N_STEPS = int(os.environ.get("AOTKG_N_STEPS", "2"))
RADT_STEPS = int(os.environ.get("AOTKG_RADT_STEPS", "100"))
DT_S = float(os.environ.get("AOTKG_DT_S", "18.0"))
OUT = Path(os.environ.get("AOTKG_OUT", str(_DEFAULT_OUT)))

# A fixed run date (the carry is date-independent post-#91, but pin it so the
# deserialize side rebuilds the SAME reference).
RUN_DATE = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)


def _fail(msg: str, code: int = 2) -> "int":
    sys.stderr.write(f"serialize_one: {msg}\n")
    return code


def main() -> int:
    if not Path(INPUT).is_dir():
        return _fail(
            f"input dir not found: {INPUT!r} (set AOTKG_INPUT to a wrfinput run dir)"
        )

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    import gpuwrf  # noqa: F401,E402  - configures x64 + the version-keyed cache
    from gpuwrf.runtime import operational_mode as om  # noqa: E402
    from gpuwrf.runtime.operational_mode import OperationalNamelist  # noqa: E402
    from gpuwrf.runtime.compile_cache import cache_entry_count  # noqa: E402
    from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
    from gpuwrf.io.radiation_static import load_radiation_static  # noqa: E402

    dev = jax.devices()[0]
    print(f"# serialize_one  jax={jax.__version__}  x64={jax.config.jax_enable_x64}")
    print(f"# device={dev}  kind={getattr(dev, 'device_kind', '?')}")
    print(f"# input={INPUT}  domain={DOMAIN}  n_steps={N_STEPS}  radt_steps={RADT_STEPS}")

    # ----- build the REAL single-domain operational namelist + committed carry --
    t_build = time.perf_counter()
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
    print(f"# built real carry in {time.perf_counter() - t_build:.1f}s; "
          f"theta {carry.state.theta.shape} {carry.state.theta.dtype}")

    # The EXACT positional args the runtime + the parallel-prewarm pass to
    # `_advance_chunk_fori` (see operational_mode._advance_chunk and
    # aot_precompile._compile_one_domain_worker). n_steps/cadence are static_kwargs.
    pos_args = (carry, namelist, start, clock_base)
    static_kwargs = {"n_steps": int(N_STEPS), "cadence": int(cadence)}

    # ----- compile (warms the persistent cache too; we want the AOT blob) -------
    cache_before = cache_entry_count()
    t_lower = time.perf_counter()
    lowered = jax.jit(
        om._advance_chunk_fori, static_argnames=("n_steps", "cadence")
    ).lower(*pos_args, **static_kwargs)
    t_compile = time.perf_counter()
    compiled = lowered.compile()
    compile_wall = time.perf_counter() - t_compile
    lower_wall = t_compile - t_lower
    cache_after = cache_entry_count()
    print(f"# lower {lower_wall:.1f}s  compile {compile_wall:.1f}s  "
          f"cache {cache_before}->{cache_after}")

    # Run the reference once now (so deserialize_run can byte-compare without
    # needing to re-derive it -- though it ALSO rebuilds a fresh ref independently).
    out_ref = compiled(*pos_args)

    # ----- serialize the compiled PJRT executable ------------------------------
    try:
        xla_exec = compiled._executable.xla_executable  # private; pinned jax 0.10
    except AttributeError as exc:
        return _fail(
            f"compiled._executable.xla_executable missing ({exc}); jax {jax.__version__} "
            "API drift -- AOT path needs a version guard. KILL-GATE cannot serialize."
        )
    if not hasattr(xla_exec, "serialize"):
        return _fail(
            f"xla_executable ({type(xla_exec).__name__}) has no .serialize(); "
            f"jax {jax.__version__} API drift. KILL-GATE cannot serialize."
        )
    blob = xla_exec.serialize()
    blob = bytes(blob)
    blob_bytes = len(blob)

    # ----- flatten the call structure + capture the concrete inputs ------------
    in_flat, in_tree = jax.tree_util.tree_flatten(pos_args)
    out_ref_flat, out_tree = jax.tree_util.tree_flatten(out_ref)
    # Concrete host arrays for the SAME inputs (so deserialize_run executes on
    # byte-identical data) + their avals for a structural sanity check.
    in_host = [np.asarray(x) for x in in_flat]
    in_avals = [
        {"shape": tuple(np.shape(x)), "dtype": str(np.asarray(x).dtype)} for x in in_flat
    ]
    out_ref_host = [np.asarray(y) for y in out_ref_flat]

    # ----- CAPTURE the EXACT runtime-buffer convention -------------------------
    # XLA const-folds / DCEs some inputs at compile time, so the executable
    # expects FEWER runtime buffers than a naive tree_flatten supplies (the GPU
    # gate hit "supplied 130 but expected 123"). JAX's own executor does
    #   args = [x for i, x in enumerate(args) if i in kept_var_idx]   (pxla.py)
    # i.e. it keeps ONLY the indices in ``kept_var_idx`` (which already includes
    # any const_args at the front). We persist that index set so deserialize_run
    # supplies EXACTLY the kept buffers in order -- replicating JAX's convention
    # instead of a naive flatten. ``num_const_args`` is recorded for sanity (it is
    # 0 for the bare @jax.jit _advance_chunk_fori; kept_var_idx already accounts
    # for it). Defensive: if the private attr is absent (jax API drift) we record
    # None and deserialize_run falls back to the naive flatten (and reports the
    # mismatch clearly rather than guessing).
    kept_var_idx, num_const_args, kvi_source = _extract_kept_var_idx(compiled)
    expected_runtime_buffers = len(kept_var_idx) if kept_var_idx is not None else None
    print(f"# kept_var_idx source={kvi_source}  naive_leaves={len(in_flat)}  "
          f"expected_runtime_buffers={expected_runtime_buffers}  "
          f"num_const_args={num_const_args}")

    fingerprint = {
        "jaxlib_version": _jaxlib_version(),
        "jax_version": jax.__version__,
        "device_kind": str(getattr(dev, "device_kind", None)),
        "platform": str(getattr(dev, "platform", None)),
        "compute_capability": _compute_capability(dev),
        "x64": bool(jax.config.jax_enable_x64),
    }

    meta = {
        "in_tree": in_tree,
        "out_tree": out_tree,
        "in_avals": in_avals,
        "static_kwargs": static_kwargs,
        "fingerprint": fingerprint,
        # The EXACT runtime-buffer convention (the 130-vs-123 fix): the sorted
        # kept-input indices into the naive tree_flatten, the const-arg count, and
        # how we got them. deserialize_run supplies [in_flat[i] for i in kept].
        "kept_var_idx": kept_var_idx,
        "num_const_args": num_const_args,
        "kept_var_idx_source": kvi_source,
        "expected_runtime_buffers": expected_runtime_buffers,
        # The reference outputs (host) so deserialize_run can compare even if it
        # cannot rebuild a fresh ref (it tries to, but this is the fallback truth).
        "out_ref_host": out_ref_host,
    }

    # ----- write everything to out/ --------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    blob_path = OUT / "advance_chunk.xlaexec"
    meta_path = OUT / "meta.pkl"
    inputs_path = OUT / "inputs.pkl"
    summary_path = OUT / "serialize_summary.json"

    blob_path.write_bytes(blob)
    with open(meta_path, "wb") as fh:
        pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(inputs_path, "wb") as fh:
        pickle.dump(in_host, fh, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "ok": True,
        "input": INPUT,
        "domain": DOMAIN,
        "n_steps": N_STEPS,
        "radt_steps": RADT_STEPS,
        "lower_seconds": round(lower_wall, 3),
        "compile_seconds": round(compile_wall, 3),
        "blob_bytes": blob_bytes,
        "blob_mb": round(blob_bytes / (1024 * 1024), 3),
        "cache_entries_before": cache_before,
        "cache_entries_after": cache_after,
        "cache_entries_written": cache_after - cache_before,
        "n_input_leaves": len(in_flat),
        "n_output_leaves": len(out_ref_flat),
        "expected_runtime_buffers": expected_runtime_buffers,
        "dropped_input_leaves": (
            len(in_flat) - expected_runtime_buffers
            if expected_runtime_buffers is not None else None
        ),
        "num_const_args": num_const_args,
        "kept_var_idx_source": kvi_source,
        "fingerprint": fingerprint,
        "blob_path": str(blob_path),
        "meta_path": str(meta_path),
        "inputs_path": str(inputs_path),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n# SERIALIZE SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\n# wrote {blob_path} ({summary['blob_mb']} MB), {meta_path}, {inputs_path}")
    print(f"# next: python deserialize_run.py  (FRESH process)")
    return 0


def _extract_kept_var_idx(compiled):
    """Return ``(kept_var_idx_sorted, num_const_args, source)`` for ``compiled``.

    XLA drops const-folded / DCE'd inputs, so the executable expects only the
    inputs whose flatten index is in ``kept_var_idx`` (pxla.py:373 filters
    ``[x for i, x in enumerate(args) if i in self.kept_var_idx]``). ``kept_var_idx``
    already includes any const_args at the front (pxla.py:883-884). We read it off
    the ``MeshExecutable`` (preferred ``_kept_var_idx``; fallback the
    ``ExecuteReplicated`` on ``unsafe_call``). On any API drift we return
    ``(None, num_const_args, "unavailable:<why>")`` so the consumer falls back to a
    naive flatten and reports the mismatch instead of guessing.
    """
    me = getattr(compiled, "_executable", None)
    num_const_args = None
    if me is not None:
        nca = getattr(me, "_num_const_args", None)
        if nca is None:
            nca = getattr(me, "num_const_args", None)
        if nca is not None:
            try:
                num_const_args = int(nca)
            except (TypeError, ValueError):
                num_const_args = None
    if me is None:
        return None, num_const_args, "unavailable:no_executable"
    kvi = getattr(me, "_kept_var_idx", None)
    source = "MeshExecutable._kept_var_idx"
    if kvi is None:
        uc = getattr(me, "unsafe_call", None)
        kvi = getattr(uc, "kept_var_idx", None)
        source = "unsafe_call.kept_var_idx"
    if kvi is None:
        return None, num_const_args, "unavailable:no_kept_var_idx_attr"
    try:
        kept = sorted(int(i) for i in kvi)
    except (TypeError, ValueError) as exc:
        return None, num_const_args, f"unavailable:bad_kept_var_idx({exc})"
    return kept, num_const_args, source


def _jaxlib_version() -> str:
    try:
        import jaxlib

        return jaxlib.__version__
    except Exception:  # noqa: BLE001
        return "?"


def _compute_capability(dev) -> str | None:
    """Best-effort SM compute capability (GPU only); None on CPU / unknown."""
    for attr in ("compute_capability",):
        val = getattr(dev, attr, None)
        if val is not None:
            return str(val)
    return None


if __name__ == "__main__":
    sys.exit(main())
