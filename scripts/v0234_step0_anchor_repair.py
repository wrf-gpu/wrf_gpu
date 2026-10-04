"""CPU-only dual cold-load proof for the H5 Step0 anchor repair.

The pinned replay base and repaired H5 candidate are loaded independently from
the same real fixture and runtime authority.  Every floating d03 state array is
hashed and compared byte-for-byte.  The historical retained Step0 carry is
also compared in both processes.  No timestep, model compile, WRF/MPI, GPU
command, or GPU query is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any

from scripts import v0234_v10_rootcause_step200 as step200


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "COLD_STEP0_ANCHOR_REPAIR.json"
BLOCKER = SPRINT / "COLD_STEP0_ANCHOR_REPAIR_BLOCKER.json"
BASE_COMMIT = "1bfee4e94be5c614051563678d62e5293bb5426a"
BASE_SRC_TREE = "5a6298fba90e76c3cafe674e1413d38efb694532"
CANDIDATE_COMMIT = "3b81fb5b093639e70c12cce87d602c45b326b18b"
CANDIDATE_SRC_TREE = "e627605f6a8bc0dc23f5c474be4bb532b99297c1"
EXPECTED_SOURCE_DELTA = (
    "src/gpuwrf/integration/nested_pipeline.py",
    "src/gpuwrf/runtime/operational_mode.py",
)
EXPECTED_RETAINED_MISMATCHES = ("mavail", "qke", "roughness_m")
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": str(step200.RUNTIME_AUTHORITY),
}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def _array_sha256(np: Any, array: Any) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _state_arrays(np: Any, state: Any) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for name in sorted(dir(state)):
        if name.startswith("_"):
            continue
        value = getattr(state, name, None)
        if value is None or callable(value) or not hasattr(value, "shape"):
            continue
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            arrays[name] = array
    return arrays


def _manifest(np: Any, arrays: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        rows[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_sha256(np, value),
            "finite": bool(np.all(np.isfinite(value))),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
        }
    return {
        "array_count": len(rows),
        "arrays": rows,
        "manifest_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _worker(source_root: Path, expected_commit: str, expected_tree: str, output: Path) -> int:
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise RuntimeError(f"worker environment mismatch: {actual_env!r}")
    if sorted(os.sched_getaffinity(0)) != [13, 14, 15, 29, 30, 31]:
        raise RuntimeError("worker CPU affinity changed")
    head = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD:src/gpuwrf")
    if head != expected_commit or tree != expected_tree:
        raise RuntimeError(f"worker source mismatch: {head}/{tree}")
    if str(source_root / "src") not in sys.path:
        raise RuntimeError("worker source root is absent from sys.path")

    import jax

    jax.config.update("jax_cpu_enable_async_dispatch", False)
    import numpy as np

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"worker backend changed: {jax.default_backend()}")
    import gpuwrf
    import gpuwrf.contracts.state as state_contract

    gpuwrf_path = Path(gpuwrf.__file__).resolve()
    if source_root.resolve() not in gpuwrf_path.parents:
        raise RuntimeError(f"gpuwrf imported from wrong tree: {gpuwrf_path}")
    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    from gpuwrf.integration.nested_pipeline import (
        NestedPipelineConfig,
        _load_domains,
        domain_names_for,
    )

    load_root = output.parent / f"load-{expected_commit[:8]}"
    load_root.mkdir()
    config = NestedPipelineConfig(
        input_dir=step200.INPUT_DIR,
        output_dir=load_root / "unused-output",
        proof_dir=load_root / "unused-proof",
        hours=0,
        max_dom=3,
        feedback=False,
    )
    names = domain_names_for(3)
    _hierarchy, bundles, metadata, run_start, dt_by_domain, carries = _load_domains(
        config, names
    )
    if names != ("d01", "d02", "d03") or dt_by_domain != {
        "d01": 54.0,
        "d02": 18.0,
        "d03": 6.0,
    }:
        raise RuntimeError("worker hierarchy changed")
    current_arrays = _state_arrays(np, jax.device_get(carries["d03"]).state)
    with step200.RETAINED_STEP0.open("rb") as stream:
        retained = pickle.load(stream)
    retained_arrays = _state_arrays(np, retained.state)
    common = sorted(set(current_arrays) & set(retained_arrays))
    if set(current_arrays) != set(retained_arrays):
        raise RuntimeError(
            "cold/retained floating state inventory changed: "
            f"cold_only={sorted(set(current_arrays) - set(retained_arrays))} "
            f"retained_only={sorted(set(retained_arrays) - set(current_arrays))}"
        )
    mismatches = [
        name
        for name in common
        if not np.array_equal(current_arrays[name], retained_arrays[name])
    ]
    mismatch_rows = {}
    for name in mismatches:
        current = np.asarray(current_arrays[name], dtype=np.float64)
        old = np.asarray(retained_arrays[name], dtype=np.float64)
        delta = current - old
        mismatch_rows[name] = {
            "cold_sha256": _array_sha256(np, current_arrays[name]),
            "retained_sha256": _array_sha256(np, retained_arrays[name]),
            "rms": float(np.sqrt(np.mean(delta * delta, dtype=np.float64))),
            "max_abs": float(np.max(np.abs(delta))),
        }
    loaded = bundles["d03"].namelist
    metadata_namelist = metadata["domains"]["d03"]["namelist"]
    payload = {
        "source": {
            "commit": head,
            "src_gpuwrf_tree": tree,
            "gpuwrf_path": str(gpuwrf_path),
        },
        "fixture": {
            "run_start": run_start.isoformat(),
            "domain_order": list(names),
            "dt_by_domain": dt_by_domain,
            "moist_adv_opt": int(loaded.moist_adv_opt),
            "scalar_adv_opt": int(loaded.scalar_adv_opt),
            "metadata_moist_adv_opt": metadata_namelist.get("moist_adv_opt"),
            "metadata_scalar_adv_opt": metadata_namelist.get("scalar_adv_opt"),
        },
        "cold_state": _manifest(np, current_arrays),
        "retained_state": _manifest(np, retained_arrays),
        "retained_comparison": {
            "compared_arrays": len(common),
            "mismatched_arrays": mismatches,
            "mismatch_metrics": mismatch_rows,
        },
        "runtime": {
            "backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    }
    step200._atomic_json(output, payload)
    return 0


def _run_worker(
    source_root: Path,
    expected_commit: str,
    expected_tree: str,
    output: Path,
) -> None:
    environment = dict(os.environ)
    environment.update(REQUIRED_ENV)
    environment["PYTHONPATH"] = f"{source_root / 'src'}:{ROOT}"
    subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source-root",
            str(source_root),
            "--expected-commit",
            expected_commit,
            "--expected-tree",
            expected_tree,
            "--worker-output",
            str(output),
        ),
        check=True,
        env=environment,
        timeout=1200,
    )


def _parent() -> int:
    stage = "startup"
    temporary: Path | None = None
    base_worktree: Path | None = None
    worktree_added = False
    try:
        actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"parent environment mismatch: {actual_env!r}")
        if sorted(os.sched_getaffinity(0)) != [13, 14, 15, 29, 30, 31]:
            raise RuntimeError("parent CPU affinity changed")
        if _git(ROOT, "status", "--porcelain"):
            raise RuntimeError("dual cold-load proof requires a clean worktree")
        if _git(ROOT, "rev-parse", f"{BASE_COMMIT}:src/gpuwrf") != BASE_SRC_TREE:
            raise RuntimeError("pinned base source tree changed")
        if _git(ROOT, "rev-parse", f"{CANDIDATE_COMMIT}:src/gpuwrf") != CANDIDATE_SRC_TREE:
            raise RuntimeError("repaired candidate source tree changed")
        source_delta = tuple(
            _git(
                ROOT,
                "diff",
                "--name-only",
                BASE_COMMIT,
                CANDIDATE_COMMIT,
                "--",
                "src/gpuwrf",
            ).splitlines()
        )
        if source_delta != EXPECTED_SOURCE_DELTA:
            raise RuntimeError(f"candidate source delta changed: {source_delta}")
        authority = step200._assert_model_and_fixture_authority(require_clean=True)

        stage = "worktree"
        temporary = Path(tempfile.mkdtemp(prefix="v0234-step0-anchor-repair-"))
        base_worktree = temporary / "pinned-base"
        subprocess.run(
            (
                "git",
                "-C",
                str(ROOT),
                "worktree",
                "add",
                "--detach",
                str(base_worktree),
                BASE_COMMIT,
            ),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        worktree_added = True
        base_output = temporary / "base.json"
        candidate_output = temporary / "candidate.json"

        stage = "base_cold_load"
        print("STEP0_ANCHOR base cold load", flush=True)
        _run_worker(base_worktree, BASE_COMMIT, BASE_SRC_TREE, base_output)
        stage = "candidate_cold_load"
        print("STEP0_ANCHOR candidate cold load", flush=True)
        _run_worker(ROOT, _git(ROOT, "rev-parse", "HEAD"), CANDIDATE_SRC_TREE, candidate_output)
        base = json.loads(base_output.read_text())
        candidate = json.loads(candidate_output.read_text())

        stage = "decision"
        base_arrays = base["cold_state"]["arrays"]
        candidate_arrays = candidate["cold_state"]["arrays"]
        exact_rows = {
            name: {
                "base_sha256": base_arrays[name]["sha256"],
                "candidate_sha256": candidate_arrays[name]["sha256"],
                "exact": base_arrays[name] == candidate_arrays[name],
            }
            for name in sorted(set(base_arrays) & set(candidate_arrays))
        }
        inventory_exact = set(base_arrays) == set(candidate_arrays)
        state_exact = inventory_exact and all(row["exact"] for row in exact_rows.values())
        base_mismatches = tuple(base["retained_comparison"]["mismatched_arrays"])
        candidate_mismatches = tuple(
            candidate["retained_comparison"]["mismatched_arrays"]
        )
        checks = {
            "candidate_and_fixture_authority_exact": True,
            "only_preregistered_two_source_files_changed": True,
            "base_and_candidate_state_inventory_exact": inventory_exact,
            "base_and_candidate_all_floating_state_arrays_byte_exact": state_exact,
            "base_retained_mismatch_set_exact": (
                base_mismatches == EXPECTED_RETAINED_MISMATCHES
            ),
            "candidate_retained_mismatch_set_exact": (
                candidate_mismatches == EXPECTED_RETAINED_MISMATCHES
            ),
            "base_and_candidate_retained_mismatch_metrics_exact": (
                base["retained_comparison"]["mismatch_metrics"]
                == candidate["retained_comparison"]["mismatch_metrics"]
            ),
            "all_cold_and_retained_arrays_finite": all(
                row["finite"]
                for arm in (base, candidate)
                for manifest in (arm["cold_state"], arm["retained_state"])
                for row in manifest["arrays"].values()
            ),
            "pinned_base_options00_candidate_options11": (
                (
                    base["fixture"]["moist_adv_opt"],
                    base["fixture"]["scalar_adv_opt"],
                )
                == (0, 0)
                and (
                    candidate["fixture"]["moist_adv_opt"],
                    candidate["fixture"]["scalar_adv_opt"],
                )
                == (1, 1)
            ),
        }
        proof = {
            "schema": "gpuwrf.v0234.h5-cold-step0-anchor-repair.v1",
            "verdict": (
                "PINNED_BASE_AND_REPAIRED_CANDIDATE_COLD_STEP0_BYTE_EXACT__"
                "HISTORICAL_THREE_LEAF_DRIFT_AUTHORIZED"
                if all(checks.values())
                else "COLD_STEP0_ANCHOR_REPAIR_RED"
            ),
            "scope": (
                "CPU-only dual cold domain load; no model lower/compile/timestep, "
                "WRF/MPI execution, GPU command, or GPU query"
            ),
            "authority": authority,
            "source_comparison": {
                "pinned_base_commit": BASE_COMMIT,
                "pinned_base_src_gpuwrf_tree": BASE_SRC_TREE,
                "candidate_commit": CANDIDATE_COMMIT,
                "candidate_src_gpuwrf_tree": CANDIDATE_SRC_TREE,
                "changed_files": list(source_delta),
                "diff_sha256": hashlib.sha256(
                    subprocess.check_output(
                        (
                            "git",
                            "-C",
                            str(ROOT),
                            "diff",
                            "--binary",
                            BASE_COMMIT,
                            CANDIDATE_COMMIT,
                            "--",
                            "src/gpuwrf",
                        )
                    )
                ).hexdigest(),
            },
            "pinned_base": base,
            "repaired_candidate": candidate,
            "cold_state_identity": {
                "inventory_exact": inventory_exact,
                "array_count": len(exact_rows),
                "all_exact": state_exact,
                "rows": exact_rows,
            },
            "expected_historical_retained_mismatches": list(
                EXPECTED_RETAINED_MISMATCHES
            ),
            "checks": checks,
            "gpu_commands": 0,
            "gpu_queries": 0,
            "model_compiles": 0,
            "model_steps": 0,
            "wrf_or_mpi_executions": 0,
        }
        proof["proof_sha256"] = step200._canonical(proof)
        step200._atomic_json(OUT, proof)
        print(
            json.dumps(
                {
                    "verdict": proof["verdict"],
                    "proof_sha256": proof["proof_sha256"],
                    "checks": checks,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if all(checks.values()) else 3
    except Exception as exc:
        blocker = {
            "schema": "gpuwrf.v0234.h5-cold-step0-anchor-repair-blocker.v1",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_commands": 0,
            "gpu_queries": 0,
            "model_compiles": 0,
            "model_steps": 0,
            "wrf_or_mpi_executions": 0,
        }
        blocker["proof_sha256"] = step200._canonical(blocker)
        step200._atomic_json(BLOCKER, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4
    finally:
        if worktree_added and base_worktree is not None:
            subprocess.run(
                ("git", "-C", str(ROOT), "worktree", "remove", str(base_worktree)),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-tree")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        if any(
            value is None
            for value in (
                args.source_root,
                args.expected_commit,
                args.expected_tree,
                args.worker_output,
            )
        ):
            parser.error("worker mode requires source/commit/tree/output")
        return _worker(
            args.source_root.resolve(),
            args.expected_commit,
            args.expected_tree,
            args.worker_output.resolve(),
        )
    if any(
        value is not None
        for value in (
            args.source_root,
            args.expected_commit,
            args.expected_tree,
            args.worker_output,
        )
    ):
        parser.error("parent mode forbids worker arguments")
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
