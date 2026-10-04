#!/usr/bin/env python3
"""CPU-only P0 proof for M9 radiation output-subset reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ALL_M9_ATTRS = (
    "t2",
    "u10",
    "v10",
    "psfc",
    "swdown",
    "glw",
    "pblh",
    "tsk",
    "swdnb",
    "swupb",
    "lwdnb",
    "lwupb",
    "swdnt",
    "swupt",
    "lwdnt",
    "lwupt",
    "swnorm",
)
REQUESTED_ATTRS = ("swdown", "swdnb", "swupb", "swdnt", "swupt", "swnorm")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cpu_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_PLATFORM_NAME"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_ENABLE_COMPILATION_CACHE"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{repo / 'src'}:{repo}"
    env.setdefault("GPUWRF_WRF_ROOT", "<USER_HOME>/src/wrf_pristine/WRF")
    return env


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _hlo_counts(text: str) -> dict[str, int]:
    return {
        "chars": len(text),
        "dynamic_update_slice": text.count("dynamic-update-slice"),
        "while": text.count("while"),
        "tuple": text.count("tuple"),
        "reduce": text.count("reduce"),
        "divide": text.count("divide"),
        "subtract": text.count("subtract"),
    }


def _inner(repo: Path, label: str, mode: str, out_dir: Path) -> None:
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)

    from tests.test_rrtm_lw_operational_wiring import _grid, _namelist, _state  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        build_clock_base,
        compute_m9_diagnostics,
    )

    grid = _grid(ny=2, nx=3, nz=8)
    state = _state(grid)
    namelist = _namelist(grid)
    clock_base = build_clock_base(namelist)

    if mode == "full":
        attrs = ALL_M9_ATTRS

        def fn(s):
            diag = compute_m9_diagnostics(
                s,
                namelist,
                jnp.asarray(0.0, dtype=jnp.float64),
                clock_base=clock_base,
            )
            return tuple(getattr(diag, attr) for attr in attrs)

    elif mode == "subset":
        attrs = REQUESTED_ATTRS
        from gpuwrf.runtime.operational_mode import compute_m9_selected_diagnostics  # noqa: PLC0415

        def fn(s):
            return compute_m9_selected_diagnostics(
                s,
                namelist,
                jnp.asarray(0.0, dtype=jnp.float64),
                clock_base,
                attrs,
            )

    else:
        raise ValueError(f"unknown mode: {mode}")

    lowered = jax.jit(fn).lower(state)
    hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    hlo_path = out_dir / f"P0_{label}_{mode}.hlo"
    hlo_path.write_text(hlo_text, encoding="utf-8")

    values = jax.jit(fn)(state)
    jax.block_until_ready(values)
    arrays = {attr: np.asarray(value) for attr, value in zip(attrs, values, strict=True)}
    np.savez(out_dir / f"P0_{label}_{mode}_fields.npz", **arrays)
    summary = {
        "label": label,
        "mode": mode,
        "attrs": list(attrs),
        "hlo": _hlo_counts(hlo_text),
        "fields": {
            name: {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            }
            for name, arr in arrays.items()
        },
    }
    (out_dir / f"P0_{label}_{mode}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _compare_fields(out_dir: Path) -> dict[str, Any]:
    import numpy as np

    before = _load_npz(out_dir / "P0_before_full_fields.npz")
    after = _load_npz(out_dir / "P0_after_subset_fields.npz")
    fields: dict[str, Any] = {}
    ok = True
    for attr in REQUESTED_ATTRS:
        lhs = before[attr]
        rhs = after[attr]
        equal = bool(np.array_equal(lhs, rhs))
        same_bytes = lhs.tobytes() == rhs.tobytes()
        ok = ok and equal and same_bytes
        fields[attr] = {
            "array_equal": equal,
            "byte_identical": same_bytes,
            "shape": list(lhs.shape),
            "dtype": str(lhs.dtype),
            "before_sha256": hashlib.sha256(lhs.tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(rhs.tobytes()).hexdigest(),
        }
    result = {
        "verdict": "PASS" if ok else "FAIL",
        "requested_attrs": list(REQUESTED_ATTRS),
        "fields": fields,
    }
    (out_dir / "P0_CPU_FIELD_COMPARE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not ok:
        raise SystemExit("P0 CPU field compare failed")
    return result


def _compare_default_full_fields(out_dir: Path) -> dict[str, Any]:
    import numpy as np

    before = _load_npz(out_dir / "P0_before_full_fields.npz")
    after = _load_npz(out_dir / "P0_after_full_fields.npz")
    fields: dict[str, Any] = {}
    ok = True
    for attr in ALL_M9_ATTRS:
        lhs = before[attr]
        rhs = after[attr]
        equal = bool(np.array_equal(lhs, rhs))
        same_bytes = lhs.tobytes() == rhs.tobytes()
        ok = ok and equal and same_bytes
        fields[attr] = {
            "array_equal": equal,
            "byte_identical": same_bytes,
            "shape": list(lhs.shape),
            "dtype": str(lhs.dtype),
            "before_sha256": hashlib.sha256(lhs.tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(rhs.tobytes()).hexdigest(),
        }
    result = {
        "verdict": "PASS" if ok else "FAIL",
        "attrs": list(ALL_M9_ATTRS),
        "fields": fields,
    }
    (out_dir / "P0_DEFAULT_FULL_FIELD_COMPARE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not ok:
        raise SystemExit("P0 default full CPU field compare failed")
    return result


def _write_hlo_summary(out_dir: Path) -> dict[str, Any]:
    before = json.loads((out_dir / "P0_before_full_summary.json").read_text(encoding="utf-8"))
    after = json.loads((out_dir / "P0_after_subset_summary.json").read_text(encoding="utf-8"))
    before_hlo = before["hlo"]
    after_hlo = after["hlo"]
    summary = {
        "before": before_hlo,
        "after": after_hlo,
        "delta_after_minus_before": {
            key: int(after_hlo[key]) - int(before_hlo[key]) for key in before_hlo
        },
        "before_attrs": before["attrs"],
        "after_attrs": after["attrs"],
        "verdict": "PASS",
    }
    (out_dir / "P0_HLO_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _outer(out_dir: Path) -> None:
    repo = _repo_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="p0_m9_base_") as tmp:
        base = Path(tmp) / "base"
        _run(["git", "worktree", "add", "--detach", str(base), "origin/main"], cwd=repo)
        try:
            _run(
                [
                    sys.executable,
                    str(script),
                    "--inner",
                    "--repo",
                    str(base),
                    "--label",
                    "before",
                    "--mode",
                    "full",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=base,
                env=_cpu_env(base),
            )
            _run(
                [
                    sys.executable,
                    str(script),
                    "--inner",
                    "--repo",
                    str(repo),
                    "--label",
                    "after",
                    "--mode",
                    "subset",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=repo,
                env=_cpu_env(repo),
            )
            _run(
                [
                    sys.executable,
                    str(script),
                    "--inner",
                    "--repo",
                    str(repo),
                    "--label",
                    "after",
                    "--mode",
                    "full",
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=repo,
                env=_cpu_env(repo),
            )
        finally:
            _run(["git", "worktree", "remove", "--force", str(base)], cwd=repo)
    _compare_default_full_fields(out_dir)
    _compare_fields(out_dir)
    _write_hlo_summary(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--mode", choices=("full", "subset"))
    parser.add_argument("--out-dir", type=Path, default=_repo_root() / "proofs/v023/perf_bundle")
    args = parser.parse_args()
    if args.inner:
        if args.repo is None or args.label is None or args.mode is None:
            raise SystemExit("--inner requires --repo, --label, and --mode")
        _inner(args.repo.resolve(), args.label, args.mode, args.out_dir.resolve())
    else:
        _outer(args.out_dir.resolve())


if __name__ == "__main__":
    main()
