#!/usr/bin/env python3
"""Build the real FAST-v025 advance-w Thomas input savepoint on CPU.

The arrays come from ``real_state.load_real_snapshot`` and the already accepted
``build_cancellation_map.build_context`` stage construction.  This script
contains no model equations.  Its compact manifest is committed; the ~12 MiB
array payload stays under ``<DATA_ROOT>`` for the future authorized native child.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cpu_guard  # noqa: E402,F401  MUST precede jax/gpuwrf imports
import numpy as np  # noqa: E402

from build_cancellation_map import build_context  # noqa: E402
from real_state import (  # noqa: E402
    assert_cpu_only,
    default_run_dir,
    load_real_snapshot,
)


SCHEMA = "wrf_gpu2.v025.m0.pallas_fast_savepoint.v1"
DEFAULT_PAYLOAD = Path(
    "<DATA_ROOT>/wrf_gpu2/v025/m0/pallas/advance_w_fast_v1.npz"
)
DEFAULT_MANIFEST = REPO / "proofs/v025/m0/pallas_fast_savepoint_manifest.json"
CANCELLATION_ENVELOPE_P99_REL = 3.410948495142528e-06


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _callable_ast_sha256(function: Any) -> str:
    source = textwrap.dedent(inspect.getsource(function))
    canonical = ast.dump(
        ast.parse(source), annotate_fields=True, include_attributes=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json_no_replace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to replace savepoint manifest: {path}")
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def build(
    *,
    run_dir: Path,
    payload_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Create one immutable real-state array payload and its proof manifest."""

    import jax
    from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan

    cpu = assert_cpu_only()
    run_dir = Path(run_dir).resolve()
    histories = sorted(run_dir.glob("wrfout_d01_*"))
    if len(histories) < 2:
        raise RuntimeError("real Pallas savepoint requires two FAST histories")
    previous = load_real_snapshot(run_dir, wrfout_name=histories[0].name)
    current = load_real_snapshot(run_dir, wrfout_name=histories[-1].name)
    context = build_context(current, previous, domain="d01")
    arrays = {
        "a": np.asarray(jax.device_get(context["a"]), dtype=np.float64),
        "alpha": np.asarray(
            jax.device_get(context["alpha"]), dtype=np.float64
        ),
        "gamma": np.asarray(
            jax.device_get(context["gamma"]), dtype=np.float64
        ),
        "rhs": np.asarray(
            jax.device_get(context["prep"].w_work), dtype=np.float64
        ),
    }
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    if len({tuple(shape) for shape in shapes.values()}) != 1:
        raise RuntimeError(f"Thomas savepoint shapes differ: {shapes}")
    if tuple(arrays["rhs"].shape) != (45, 70, 120):
        raise RuntimeError(
            f"real FAST Thomas shape is {arrays['rhs'].shape}, expected (45,70,120)"
        )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("real FAST Thomas savepoint contains non-finite values")

    payload_path = Path(payload_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(payload_path):
        raise RuntimeError(f"refusing to replace savepoint payload: {payload_path}")
    temporary = payload_path.with_name(
        f".{payload_path.name}.{os.getpid()}.tmp.npz"
    )
    try:
        np.savez(temporary, **arrays)
        os.link(temporary, payload_path)
        directory_fd = os.open(payload_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema": SCHEMA,
        "status": "CPU_REAL_FAST_SAVEPOINT_GREEN_NATIVE_PALLAS_MISSING",
        "device_touched": False,
        "native_pallas_verdict": "MISSING",
        "payload_path": str(payload_path.resolve()),
        "payload_bytes": payload_path.stat().st_size,
        "payload_sha256": _sha256_file(payload_path),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
                "all_finite": bool(np.isfinite(value).all()),
            }
            for name, value in arrays.items()
        },
        "source": {
            "state_source_is_real": True,
            "run_dir": str(run_dir),
            "previous_wrfout": previous.provenance["wrfout"],
            "previous_wrfout_sha256":
                previous.provenance["wrfout_sha256"],
            "current_wrfout": current.provenance["wrfout"],
            "current_wrfout_sha256": current.provenance["wrfout_sha256"],
            "stage_pair": context["stage_pair"],
            "context_builder":
                "scripts.v025.build_cancellation_map.build_context",
            "rhs": "context['prep'].w_work",
            "coefficients": "context['a'], context['alpha'], context['gamma']",
        },
        "production_oracle": {
            "fqname":
                "gpuwrf.dynamics.tridiag_solve.thomas_solve_scan",
            "ast_sha256": _callable_ast_sha256(thomas_solve_scan),
            "wrf_source":
                "dyn_em/module_small_step_em.F:1533-1550",
        },
        "correctness_envelope": {
            "metric": "p99_rel",
            "maximum":
                CANCELLATION_ENVELOPE_P99_REL,
            "source":
                "proofs/v025/m0/cancellation_map.json advance_w fp32 arm",
        },
        "cpu_platform": cpu,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json_no_replace(Path(manifest_path), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir())
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = build(
        run_dir=args.run_dir,
        payload_path=args.payload,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
