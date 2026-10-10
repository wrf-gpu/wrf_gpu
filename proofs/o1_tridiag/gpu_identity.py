"""GPU-lowering byte-identity + CPU FFI-removal proof (lane o1-tridiag, v0.3.4).

Run (CPU only), with the PRE-fix blobs taken from <base_ref> (default ``main``; run
this *after* committing so the fixed code is in HEAD and the pre-fix code is the
merge base / main):
    JAX_PLATFORMS=cpu PYTHONPATH=<tree>/src python proofs/o1_tridiag/gpu_identity.py <tree> <out.json> [base_ref]

Claim 1 (GPU untouched): with the active backend simulated as CUDA
(``jax.default_backend`` monkeypatched to return "cuda" at trace time, the only
input the CPU guard reads -- E115/E156), the location-stripped lowered HLO of
``solve_tridiagonal`` and ``solve_tridiagonal_xla`` is **identical** before and
after the fix for every tested shape/dtype, and still contains the cuSPARSE
``gtsv`` custom call.

Claim 2 (CPU deadlock source removed): on the real CPU backend the pre-fix
lowering contains ``lapack_{s,d}gtsv_ffi`` (the FFI that deadlocks, E105); the
fixed lowering contains none and lowers to the pure-JAX Thomas ``while`` scan.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _lower_text(fn, specs, platform: str | None) -> str:
    jax.clear_caches()
    wrapper = lambda *args: fn(*args)  # noqa: E731 - fresh object: no stale trace cache
    if platform == "cuda":
        with mock.patch.object(jax, "default_backend", lambda *a, **k: "cuda"):
            lowered = jax.jit(wrapper).trace(*specs).lower(lowering_platforms=("cuda",))
    else:
        lowered = jax.jit(wrapper).trace(*specs).lower()
    text = str(lowered.as_text())
    # Strip anything that is a source location / name, keep pure structure.
    text = "\n".join(
        line for line in text.splitlines() if "loc(" not in line and "jax.result_info" not in line
    )
    return text


def main() -> int:
    tree = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2])
    base_ref = sys.argv[3] if len(sys.argv) > 3 else "main"  # ref holding the PRE-fix blobs
    base_dir = out.parent / "base_src"
    (base_dir / "physics").mkdir(parents=True, exist_ok=True)
    (base_dir / "dynamics").mkdir(parents=True, exist_ok=True)
    for rel, dst in (
        ("src/gpuwrf/physics/tridiagonal_solver.py", base_dir / "physics/tridiagonal_solver.py"),
        ("src/gpuwrf/dynamics/vertical_implicit_solver.py", base_dir / "dynamics/vertical_implicit_solver.py"),
    ):
        blob = subprocess.run(
            ["git", "-C", str(tree), "show", f"{base_ref}:{rel}"], check=True, capture_output=True
        ).stdout
        dst.write_bytes(blob)

    base_ts = _load_module("base_tridiagonal_solver", base_dir / "physics/tridiagonal_solver.py")
    base_vis = _load_module("base_vertical_implicit_solver", base_dir / "dynamics/vertical_implicit_solver.py")
    from gpuwrf.physics import tridiagonal_solver as new_ts
    from gpuwrf.dynamics import vertical_implicit_solver as new_vis

    cases = [
        ("f32_vectors", (6, 9), jnp.float32, "solve_tridiagonal"),
        ("f32_batch3d", (2, 3, 9), jnp.float32, "solve_tridiagonal"),
        ("f64_vectors", (6, 9), jnp.float64, "solve_tridiagonal"),
        ("xla_helper_f64", (9, 6), jnp.float64, "solve_tridiagonal_xla"),
    ]
    report = {"tree": str(tree), "base_ref": base_ref, "head": subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "cases": {}}
    ok = True
    for name, shape, dtype, fn_name in cases:
        old_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", dtype == jnp.float64)
        try:
            specs = [jax.ShapeDtypeStruct(shape, dtype) for _ in range(4)]
            base_fn = getattr(base_ts if fn_name == "solve_tridiagonal" else base_vis, fn_name)
            new_fn = getattr(new_ts if fn_name == "solve_tridiagonal" else new_vis, fn_name)
            cpu_old = _lower_text(base_fn, specs, None)
            cpu_new = _lower_text(new_fn, specs, None)
            gpu_old = _lower_text(base_fn, specs, "cuda")
            gpu_new = _lower_text(new_fn, specs, "cuda")
        finally:
            jax.config.update("jax_enable_x64", old_x64)
        entry = {
            "shape": list(shape),
            "dtype": str(dtype),
            "fn": fn_name,
            "cpu_old_has_gtsv": "gtsv" in cpu_old,
            "cpu_new_has_gtsv": "gtsv" in cpu_new,
            "cpu_new_has_while": "stablehlo.while" in cpu_new,
            "gpu_old_has_gtsv": "gtsv" in gpu_old,
            "gpu_new_has_gtsv": "gtsv" in gpu_new,
            "gpu_identity": gpu_old == gpu_new,
        }
        entry["verdict"] = bool(
            entry["cpu_old_has_gtsv"]
            and not entry["cpu_new_has_gtsv"]
            and entry["cpu_new_has_while"]
            and entry["gpu_old_has_gtsv"]
            and entry["gpu_new_has_gtsv"]
            and entry["gpu_identity"]
        )
        ok = ok and entry["verdict"]
        report["cases"][name] = entry
    report["pass"] = ok
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
