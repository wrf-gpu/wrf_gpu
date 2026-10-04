"""CPU-only proof generator for the v0.23.4 S2 boundary-retention gate.

The proof has two independent arms:

* replay the immutable step-1 retained-data split and correct its historical
  field-side sign bookkeeping without changing the sealed artifact;
* exercise the current production coupled-boundary relax operator against a
  literal NumPy transcription of WRF v4.7.1 ``relax_bdytend_core``.

The historical arm must run before JAX/gpuwrf is imported because its sealed
authority explicitly proves an offline NumPy-only import surface.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-21-v0234-gpt-s2-boundary"
HISTORICAL_SPRINT = (
    REPO / ".agent/sprints/2026-07-18-v0234-dycore-internal-split-kimi"
)
HISTORICAL_ANALYSIS = HISTORICAL_SPRINT / "internal-split-analysis.json"
BASELINE_COMMIT = "5193db05cd67cfc79292b0206ca43b1d3c68b2e8"
ASSIGNED_BASE = "1ef7181674b3bec67066654a240fb3575d654285"
LEDGER_COMMIT = "d541a9ffad965b426b7bbf8a830b95c411bf3e8b"
HISTORICAL_CANONICAL_SHA256 = (
    "65421e6c7ef823c821e43e3d28a460df56656e4b30e8d638dd1ec88895788996"
)
DEFAULT_CONFIRM = (
    REPO
    / ".agent/sprints/2026-07-20-v0234-gpt-drift-perf-bisect"
    / "BOUNDARY_DEFAULT_CONFIRM.json"
)
DEFAULT_CONFIRM_EXPECTED_SHA256 = (
    "3435c40ebde20b6466b9dc376aac5ac8cc49f3e29618f0a7283b1ed984f7328a"
)

BOUNDARY_PATH = "src/gpuwrf/coupling/boundary_apply.py"
OPERATIONAL_PATH = "src/gpuwrf/runtime/operational_mode.py"
PIPELINE_PATH = "src/gpuwrf/integration/nested_pipeline.py"

FOCUSED_TESTS = (
    "tests/test_v0234_s2_boundary_revalidation.py",
    "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py",
    "tests/test_v014_specified_bdy_cadence.py",
    "tests/test_daily_boundary_clock.py",
    "tests/test_v0234_event_aware_fusion.py",
    "tests/test_v0234_b2_bounded_terminal_gate.py",
    "tests/test_v024_nested_runtime_reuse.py",
)
NUMPY_ISOLATED_TESTS = ("tests/test_v0234_dycore_internal_split_kimi.py",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object, *, omit: str | None = None) -> str:
    if omit is not None and isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != omit}
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_self_hashed(path: Path, payload: dict[str, Any]) -> None:
    out = dict(payload)
    out["proof_sha256"] = canonical_digest(out, omit="proof_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def file_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_text(commit: str, path: str) -> str:
    return run_git("show", f"{commit}:{path}").stdout


def git_blob(commit: str, path: str) -> str:
    return run_git("rev-parse", f"{commit}:{path}").stdout.strip()


def git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return run_git(
        "merge-base", "--is-ancestor", ancestor, descendant, check=False
    ).returncode == 0


def ast_function_digest(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return hashlib.sha256(
                ast.dump(node, annotate_fields=True, include_attributes=False).encode()
            ).hexdigest()
    raise KeyError(f"function not found: {function_name}")


def ast_smallest_node_digest(
    source: str, function_name: str, required_fragments: tuple[str, ...]
) -> str:
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    matches: list[tuple[int, ast.AST]] = []
    for node in ast.walk(function):
        segment = ast.get_source_segment(source, node) or ""
        if all(fragment in segment for fragment in required_fragments):
            matches.append((len(segment), node))
    if not matches:
        raise KeyError(
            f"node not found in {function_name}: required={required_fragments}"
        )
    node = min(matches, key=lambda item: item[0])[1]
    return hashlib.sha256(
        ast.dump(node, annotate_fields=True, include_attributes=False).encode()
    ).hexdigest()


def source_interaction_audit(approved_head: str) -> dict[str, Any]:
    current_boundary = git_text(approved_head, BOUNDARY_PATH)
    baseline_boundary = git_text(BASELINE_COMMIT, BOUNDARY_PATH)
    current_operational = git_text(approved_head, OPERATIONAL_PATH)
    baseline_operational = git_text(BASELINE_COMMIT, OPERATIONAL_PATH)
    current_pipeline = git_text(approved_head, PIPELINE_PATH)
    baseline_pipeline = git_text(BASELINE_COMMIT, PIPELINE_PATH)

    function_names = (
        "_nested_frozen_wrf_boundary_active",
        "_specified_bdy_relax",
        "_nested_frozen_bdy_relax",
    )
    operational_functions = {}
    for name in function_names:
        before = ast_function_digest(baseline_operational, name)
        after = ast_function_digest(current_operational, name)
        operational_functions[name] = {
            "baseline_ast_sha256": before,
            "current_ast_sha256": after,
            "identical": before == after,
        }

    node_specs = {
        "relax_bundle_additive_fold": (
            "_augment_large_step_tendencies",
            ("bdy_relax is not None", "bdy_relax.ru", "bdy_relax.rv"),
        ),
        "rk1_bundle_hoist": (
            "_rk_scan_step",
            ("nested_frozen_relax =", "_nested_frozen_bdy_relax"),
        ),
        "rk_stage_bundle_reuse": (
            "_rk_scan_step",
            ("bdy_relax =", "nested_frozen_relax", "_specified_bdy_relax"),
        ),
    }
    operational_nodes = {}
    for label, (function_name, fragments) in node_specs.items():
        before = ast_smallest_node_digest(
            baseline_operational, function_name, fragments
        )
        after = ast_smallest_node_digest(
            current_operational, function_name, fragments
        )
        operational_nodes[label] = {
            "baseline_ast_sha256": before,
            "current_ast_sha256": after,
            "identical": before == after,
        }

    changed_raw = run_git(
        "diff",
        "--name-status",
        f"{BASELINE_COMMIT}..{approved_head}",
        "--",
        "src/gpuwrf",
    ).stdout.splitlines()
    changed_files = []
    for row in changed_raw:
        if not row.strip():
            continue
        status, path = row.split("\t", maxsplit=1)
        changed_files.append({"status": status, "path": path})

    current_default_one = (
        'os.environ.get("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", "1")'
        in current_pipeline
    )
    baseline_default_zero = (
        'os.environ.get("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", "0")'
        in baseline_pipeline
    )

    direct_paths = {
        OPERATIONAL_PATH,
        PIPELINE_PATH,
        "src/gpuwrf/runtime/domain_tree.py",
        "src/gpuwrf/dynamics/core/rk_addtend_dry.py",
        "src/gpuwrf/dynamics/explicit_diffusion.py",
    }
    classified = []
    for row in changed_files:
        path = row["path"]
        if path in direct_paths:
            category = "runtime_or_adjacent_dynamics_interaction_surface"
        elif path.startswith("src/gpuwrf/physics/") or path.endswith(
            "physics_couplers.py"
        ):
            category = "upstream_physics_state_only"
        else:
            category = "non_boundary_operator_source"
        classified.append({**row, "classification": category})

    boundary_blob_equal = git_blob(BASELINE_COMMIT, BOUNDARY_PATH) == git_blob(
        approved_head, BOUNDARY_PATH
    )
    functions_equal = all(
        row["identical"] for row in operational_functions.values()
    )
    nodes_equal = all(row["identical"] for row in operational_nodes.values())

    return {
        "baseline_commit": BASELINE_COMMIT,
        "approved_head": approved_head,
        "baseline_is_ancestor": git_is_ancestor(BASELINE_COMMIT, approved_head),
        "assigned_base_is_ancestor": git_is_ancestor(ASSIGNED_BASE, approved_head),
        "boundary_operator": {
            "path": BOUNDARY_PATH,
            "baseline_blob": git_blob(BASELINE_COMMIT, BOUNDARY_PATH),
            "current_blob": git_blob(approved_head, BOUNDARY_PATH),
            "whole_file_identical": boundary_blob_equal,
            "baseline_file_sha256": hashlib.sha256(
                baseline_boundary.encode()
            ).hexdigest(),
            "current_file_sha256": hashlib.sha256(
                current_boundary.encode()
            ).hexdigest(),
        },
        "operational_wrapper_functions": operational_functions,
        "operational_fold_and_freeze_nodes": operational_nodes,
        "pipeline_default_promotion": {
            "baseline_explicit_default_zero": baseline_default_zero,
            "current_explicit_default_one": current_default_one,
            "classification": "accepted-path selection promotion; rollback =0 retained",
        },
        "changed_production_files_since_prior_s2": classified,
        "gate": bool(
            boundary_blob_equal
            and functions_equal
            and nodes_equal
            and baseline_default_zero
            and current_default_one
        ),
    }


def _combined_rms(components: dict[str, np.ndarray]) -> float:
    count = sum(array.size for array in components.values())
    sse = sum(
        float(np.square(array, dtype=np.float64).sum(dtype=np.float64))
        for array in components.values()
    )
    return math.sqrt(sse / count)


def historical_retained_replay() -> dict[str, Any]:
    # Keep this import and authentication before any JAX/gpuwrf import.
    from scripts import v0234_dycore_internal_split_kimi as legacy

    auth, _cpu_analysis, _gpu_analysis, _runtime = legacy.authenticate()
    historical, historical_row = legacy.read_self_hashed(HISTORICAL_ANALYSIS)
    if historical_row["canonical_self_hash"] != HISTORICAL_CANONICAL_SHA256:
        raise RuntimeError("historical S2 analysis canonical hash drifted")

    hgt = legacy.cpu.load_hgt()
    gpu, control, members, const = legacy.load_all_states(step=1)
    replay = legacy.s2_relax_split(gpu, control, members, const, hgt)
    historical_s2 = historical["s2_relax_bdy_dry_split"]

    c1h, c2h = const["c1h"], const["c2h"]
    muu_ctl = legacy.mu_faces(control["mut"], axis=1)
    muv_ctl = legacy.mu_faces(control["mut"], axis=0)
    muu_gpu = legacy.mu_faces(const["gpu_mu_total"], axis=1)
    muv_gpu = legacy.mu_faces(const["gpu_mu_total"], axis=0)
    ru_ctl = (
        legacy.mass_weight(c1h, c2h, muu_ctl)
        * control["u_sp1"]
        / const["mapfac_uy"][None, :, :]
    )
    rv_ctl = (
        legacy.mass_weight(c1h, c2h, muv_ctl)
        * control["v_sp1"]
        / const["mapfac_vx"][None, :, :]
    )
    ru_gpu = (
        legacy.mass_weight(const["gpu_c1h"], const["gpu_c2h"], muu_gpu)
        * gpu["u_sp1"]
        / const["gpu_mapfac_uy"][None, :, :]
    )
    rv_gpu = (
        legacy.mass_weight(const["gpu_c1h"], const["gpu_c2h"], muv_gpu)
        * gpu["v_sp1"]
        / const["gpu_mapfac_vx"][None, :, :]
    )
    delta = {
        "u": gpu["u_save"] - control["u_save"],
        "v": gpu["v_save"] - control["v_save"],
    }
    d_field = {"u": ru_gpu - ru_ctl, "v": rv_gpu - rv_ctl}

    # Correct sign: relax(dF, 0) is already the field-side image
    # -fcx*dF + gcx*lap(dF); the historical helper negated it once more.
    field_side = {
        "u": legacy.wrf_relax_tendency(
            d_field["u"], np.zeros_like(d_field["u"]), legacy.DT_D03, "u"
        ),
        "v": legacy.wrf_relax_tendency(
            d_field["v"], np.zeros_like(d_field["v"]), legacy.DT_D03, "v"
        ),
    }
    corrected_target_side = {
        name: delta[name] - field_side[name] for name in ("u", "v")
    }
    closure = {
        name: delta[name] - field_side[name] - corrected_target_side[name]
        for name in ("u", "v")
    }

    # An independent finite-difference identity fixes the sign without using
    # any retained target (which was not captured in the old arm).
    rng = np.random.default_rng(230421)
    f0 = rng.normal(size=(2, 10, 11))
    df = rng.normal(scale=0.01, size=f0.shape)
    t0 = rng.normal(size=f0.shape)
    dt = rng.normal(scale=0.01, size=f0.shape)
    r00 = legacy.wrf_relax_tendency(f0, t0, legacy.DT_D03, "u")
    direct_field = (
        legacy.wrf_relax_tendency(f0 + df, t0, legacy.DT_D03, "u") - r00
    )
    direct_target = (
        legacy.wrf_relax_tendency(f0, t0 + dt, legacy.DT_D03, "u") - r00
    )
    direct_both = (
        legacy.wrf_relax_tendency(f0 + df, t0 + dt, legacy.DT_D03, "u")
        - r00
    )
    field_formula = legacy.wrf_relax_tendency(
        df, np.zeros_like(df), legacy.DT_D03, "u"
    )
    target_formula = legacy.wrf_relax_tendency(
        np.zeros_like(dt), dt, legacy.DT_D03, "u"
    )
    sign_oracle_max_abs = max(
        float(np.max(np.abs(direct_field - field_formula))),
        float(np.max(np.abs(direct_target - target_formula))),
        float(np.max(np.abs(direct_both - direct_field - direct_target))),
    )

    delta_rms = _combined_rms(delta)
    field_rms = _combined_rms(field_side)
    target_rms = _combined_rms(corrected_target_side)
    closure_max_abs = max(float(np.max(np.abs(v))) for v in closure.values())
    corrected_field_metrics = legacy.cpu.metrics_with_bands(field_side, hgt)
    corrected_target_metrics = legacy.cpu.metrics_with_bands(
        corrected_target_side, hgt
    )

    exact_subtree = replay == historical_s2
    gate = bool(
        exact_subtree
        and closure_max_abs <= 1.0e-12
        and sign_oracle_max_abs <= 1.0e-12
        and field_rms / delta_rms <= 0.001
        and 0.998 <= target_rms / delta_rms <= 1.002
    )

    result = {
        "historical_analysis": historical_row,
        "savepoints_revalidated": auth["savepoints_revalidated"],
        "wrf_source_sha256": auth["wrf_source_sha256"],
        "oracle_inputs": auth["oracle_inputs"],
        "historical_s2_subtree_replayed_exactly": exact_subtree,
        "historical_s2_subtree_sha256": canonical_digest(historical_s2),
        "replayed_s2_subtree_sha256": canonical_digest(replay),
        "sealed_metrics": {
            "delta_rms": historical_s2["delta_relax"]["rmse"],
            "historical_field_side_rms": historical_s2["field_side_image"][
                "rmse"
            ],
            "historical_target_side_rms": historical_s2[
                "target_side_image_E"
            ]["rmse"],
        },
        "corrected_sign_audit": {
            "issue": (
                "historical s2_relax_split negated relax(dF,0), although "
                "relax(dF,0) already is the field-side image"
            ),
            "delta_rms": delta_rms,
            "field_side_rms": field_rms,
            "target_side_rms": target_rms,
            "field_side_fraction": field_rms / delta_rms,
            "target_side_rms_ratio": target_rms / delta_rms,
            "pointwise_closure_max_abs": closure_max_abs,
            "independent_sign_oracle_max_abs": sign_oracle_max_abs,
            "field_side_metrics": corrected_field_metrics,
            "target_side_metrics": corrected_target_metrics,
            "frozen_bounds": {
                "pointwise_abs_max": 1.0e-12,
                "field_side_fraction_max": 0.001,
                "target_side_rms_ratio_min": 0.998,
                "target_side_rms_ratio_max": 1.002,
            },
        },
        "gate": gate,
    }

    # Encourage release before the production-JAX arm starts.
    del gpu, control, members, const, hgt, replay
    return result


def _ring_target(leaf: np.ndarray, *, z: int, ny: int, nx: int) -> np.ndarray:
    """Independent WRF registration: S/N sides own all corners."""

    array = np.asarray(leaf)
    out = np.zeros((z, ny, nx), dtype=array.dtype)
    for b in range(5):
        out[:, :, b] = array[0, b, :z, :ny]
        out[:, :, nx - 1 - b] = array[1, b, :z, :ny]
        out[:, b, :] = array[2, b, :z, :nx]
        out[:, ny - 1 - b, :] = array[3, b, :z, :nx]
    return out


def _wrf_relax(
    field: np.ndarray,
    target: np.ndarray,
    *,
    dt: float,
    spec_zone: int = 1,
    relax_zone: int = 4,
) -> np.ndarray:
    """Literal WRF v4.7.1 ``relax_bdytend_core`` on a coupled field."""

    field = np.asarray(field)
    target = np.asarray(target)
    _z, ny, nx = field.shape
    out = np.zeros_like(field)
    residual = target - field
    for b in range(spec_zone, relax_zone):
        loop = b + 1
        linear = (spec_zone + relax_zone - loop) / (relax_zone - 1)
        fcx = 0.1 / dt * linear
        gcx = 1.0 / dt / 50.0 * linear
        # Y sides own diagonal/corner cells.
        for row, inward in ((b, 1), (ny - 1 - b, -1)):
            for i in range(b, nx - b):
                im1, ip1 = max(i - 1, 0), min(i + 1, nx - 1)
                fls0 = residual[:, row, i]
                lap = (
                    residual[:, row, im1]
                    + residual[:, row, ip1]
                    + residual[:, row - inward, i]
                    + residual[:, row + inward, i]
                    - 4.0 * fls0
                )
                out[:, row, i] += fcx * fls0 - gcx * lap
        # X sides trim cells already owned by Y.
        for col, inward in ((b, 1), (nx - 1 - b, -1)):
            for j in range(b + 1, ny - 1 - b):
                jm1, jp1 = max(j - 1, 0), min(j + 1, ny - 1)
                fls0 = residual[:, j, col]
                lap = (
                    residual[:, jm1, col]
                    + residual[:, jp1, col]
                    + residual[:, j, col - inward]
                    + residual[:, j, col + inward]
                    - 4.0 * fls0
                )
                out[:, j, col] += fcx * fls0 - gcx * lap
    return out


def _leaf(
    rng: np.random.Generator,
    *,
    z: int,
    side: int,
    scale: float,
) -> np.ndarray:
    return rng.normal(scale=scale, size=(2, 4, 5, z, side)).astype(np.float64)


def _distance_to_edge(shape: tuple[int, ...]) -> np.ndarray:
    ny, nx = shape[-2:]
    yy, xx = np.indices((ny, nx))
    dist2 = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
    return np.broadcast_to(dist2, shape)


def _ownership_counts(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    spec = np.zeros((ny, nx), dtype=np.int32)
    relax = np.zeros((ny, nx), dtype=np.int32)
    b = 0
    spec[b, b:nx] += 1
    spec[ny - 1 - b, b:nx] += 1
    spec[b + 1 : ny - 1 - b, b] += 1
    spec[b + 1 : ny - 1 - b, nx - 1 - b] += 1
    for b in range(1, 4):
        relax[b, b : nx - b] += 1
        relax[ny - 1 - b, b : nx - b] += 1
        relax[b + 1 : ny - 1 - b, b] += 1
        relax[b + 1 : ny - 1 - b, nx - 1 - b] += 1
    return spec, relax


def current_production_oracle() -> dict[str, Any]:
    # Imported only after historical_retained_replay has authenticated the
    # NumPy-only historical arm.
    import jax
    import jax.numpy as jnp

    from gpuwrf.coupling.boundary_apply import (
        BoundaryConfig,
        specified_relax_dry_tendencies,
    )

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"production oracle backend is {jax.default_backend()}, not cpu")
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("production oracle requires JAX_ENABLE_X64=true")

    rng = np.random.default_rng(23420260721)
    nz, ny, nx = 3, 14, 13
    side = max(ny + 1, nx + 1)
    mu_total = 900.0 + rng.normal(size=(ny, nx))

    base_arrays = {
        "u": rng.normal(size=(nz, ny, nx + 1)),
        "v": rng.normal(size=(nz, ny + 1, nx)),
        "w": rng.normal(size=(nz + 1, ny, nx)),
        "theta": 300.0 + rng.normal(size=(nz, ny, nx)),
        "ph_perturbation": rng.normal(size=(nz + 1, ny, nx)),
        "mu_perturbation": rng.normal(size=(ny, nx)),
    }
    base_leaves = {
        "u_bdy": _leaf(rng, z=nz, side=side, scale=900.0),
        "v_bdy": _leaf(rng, z=nz, side=side, scale=900.0),
        "w_bdy": _leaf(rng, z=nz + 1, side=side, scale=900.0),
        "theta_bdy": _leaf(rng, z=nz, side=side, scale=900.0),
        "ph_bdy": _leaf(rng, z=nz + 1, side=side, scale=900.0),
        "mu_bdy": _leaf(rng, z=1, side=side, scale=10.0),
    }
    field_delta = {
        name: rng.normal(scale=0.01, size=value.shape)
        for name, value in base_arrays.items()
    }
    target_delta = {
        "u_bdy": _leaf(rng, z=nz, side=side, scale=0.01),
        "v_bdy": _leaf(rng, z=nz, side=side, scale=0.01),
        "w_bdy": _leaf(rng, z=nz + 1, side=side, scale=0.01),
        "theta_bdy": _leaf(rng, z=nz, side=side, scale=0.01),
        "ph_bdy": _leaf(rng, z=nz + 1, side=side, scale=0.01),
        "mu_bdy": _leaf(rng, z=1, side=side, scale=0.01),
    }

    def reference(*, alter_field: bool, alter_target: bool) -> SimpleNamespace:
        arrays = {
            name: value + (field_delta[name] if alter_field else 0.0)
            for name, value in base_arrays.items()
        }
        leaves = {
            name: value + (target_delta[name] if alter_target else 0.0)
            for name, value in base_leaves.items()
        }
        return SimpleNamespace(
            **{name: jnp.asarray(value) for name, value in arrays.items()},
            **{name: jnp.asarray(value) for name, value in leaves.items()},
            mu_total=jnp.asarray(mu_total),
        )

    msfuy = 1.0 + 0.01 * rng.random((ny, nx + 1))
    msfvx = 1.0 + 0.01 * rng.random((ny + 1, nx))
    msfty = 1.0 + 0.01 * rng.random((ny, nx))
    c1h = 0.9 + 0.01 * np.arange(nz)
    c2h = 40.0 + np.arange(nz)
    c1f = 0.9 + 0.01 * np.arange(nz + 1)
    c2f = 40.0 + np.arange(nz + 1)
    metrics = SimpleNamespace(
        c1h=jnp.asarray(c1h),
        c2h=jnp.asarray(c2h),
        c1f=jnp.asarray(c1f),
        c2f=jnp.asarray(c2f),
        msfuy=jnp.asarray(msfuy),
        msfvx=jnp.asarray(msfvx),
        msfty=jnp.asarray(msfty),
    )
    cfg = BoundaryConfig(
        update_cadence_s=6.0,
        force_geopotential=False,
        nested_frozen_wrf_boundary_bundle=True,
    )
    dt_full = 2.0

    def run(ref: SimpleNamespace) -> dict[str, np.ndarray]:
        bundle = specified_relax_dry_tendencies(
            ref,
            6.0,
            metrics,
            dt_full,
            cfg,
            include_nested_w=True,
            coupled_boundary_leaves=True,
        )
        return {
            name: np.asarray(getattr(bundle, name))
            for name in ("ru", "rv", "t", "ph", "mu", "w")
        }

    ref_base = reference(alter_field=False, alter_target=False)
    out_base = run(ref_base)
    out_field = run(reference(alter_field=True, alter_target=False))
    out_target = run(reference(alter_field=False, alter_target=True))
    out_both = run(reference(alter_field=True, alter_target=True))
    out_repeat = run(ref_base)

    muu = 0.5 * (
        np.concatenate((mu_total[:, :1], mu_total), axis=1)
        + np.concatenate((mu_total, mu_total[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu_total[:1, :], mu_total), axis=0)
        + np.concatenate((mu_total, mu_total[-1:, :]), axis=0)
    )
    mass_u = c1h[:, None, None] * muu[None] + c2h[:, None, None]
    mass_v = c1h[:, None, None] * muv[None] + c2h[:, None, None]
    mass_h = c1h[:, None, None] * mu_total[None] + c2h[:, None, None]
    mass_f = c1f[:, None, None] * mu_total[None] + c2f[:, None, None]

    targets = {
        "ru": _ring_target(base_leaves["u_bdy"][1], z=nz, ny=ny, nx=nx + 1),
        "rv": _ring_target(base_leaves["v_bdy"][1], z=nz, ny=ny + 1, nx=nx),
        "t": _ring_target(base_leaves["theta_bdy"][1], z=nz, ny=ny, nx=nx),
        "ph": _ring_target(
            base_leaves["ph_bdy"][1], z=nz + 1, ny=ny, nx=nx
        ),
        "w": _ring_target(
            base_leaves["w_bdy"][1], z=nz + 1, ny=ny, nx=nx
        ),
        "mu": _ring_target(base_leaves["mu_bdy"][1], z=1, ny=ny, nx=nx),
    }
    current = {
        "ru": mass_u * base_arrays["u"] / msfuy[None],
        "rv": mass_v * base_arrays["v"] / msfvx[None],
        "t": mass_h * (base_arrays["theta"] - 300.0),
        "ph": mass_f * base_arrays["ph_perturbation"],
        "w": mass_f * base_arrays["w"],
        "mu": base_arrays["mu_perturbation"][None],
    }
    expected = {}
    for name in ("ru", "rv", "t", "ph", "w", "mu"):
        value = _wrf_relax(current[name], targets[name], dt=dt_full)
        if name in ("t", "ph", "w"):
            value = value / msfty[None]
        if name == "mu":
            value = value[0]
        expected[name] = value

    comparisons = {}
    for name, oracle in expected.items():
        got = out_base[name]
        abs_error = np.abs(got - oracle)
        comparisons[name] = {
            "shape": list(got.shape),
            "dtype": str(got.dtype),
            "max_abs_error": float(abs_error.max(initial=0.0)),
            "rmse": float(np.sqrt(np.mean(np.square(abs_error)))),
            "within_rtol_3e_13_atol_3e_13": bool(
                np.allclose(got, oracle, rtol=3.0e-13, atol=3.0e-13)
            ),
        }

    decomposition = {}
    closure_max = 0.0
    target_nonzero = False
    deterministic = True
    support_green = True
    for name in out_base:
        field_side = out_field[name] - out_base[name]
        target_side = out_target[name] - out_base[name]
        delta_both = out_both[name] - out_base[name]
        closure = delta_both - field_side - target_side
        max_abs = float(np.max(np.abs(closure), initial=0.0))
        closure_max = max(closure_max, max_abs)
        target_nonzero = target_nonzero or bool(np.any(target_side != 0.0))
        deterministic = deterministic and np.array_equal(
            out_base[name], out_repeat[name]
        )
        array3 = out_base[name] if out_base[name].ndim == 3 else out_base[name][None]
        dist = _distance_to_edge(array3.shape)
        ring0_zero = bool(np.count_nonzero(array3[dist == 0]) == 0)
        b4_zero = bool(np.count_nonzero(array3[dist == 4]) == 0)
        interior_zero = bool(np.count_nonzero(array3[dist >= 4]) == 0)
        support_green = support_green and ring0_zero and b4_zero and interior_zero
        decomposition[name] = {
            "closure_max_abs": max_abs,
            "field_side_rms": float(np.sqrt(np.mean(np.square(field_side)))),
            "target_side_rms": float(np.sqrt(np.mean(np.square(target_side)))),
            "ring0_exact_zero": ring0_zero,
            "b_dist_4_exact_zero": b4_zero,
            "interior_ge_4_exact_zero": interior_zero,
            "repeat_bit_identical": bool(
                np.array_equal(out_base[name], out_repeat[name])
            ),
        }

    ownership = {}
    ownership_green = True
    for label, shape in {
        "mass": (ny, nx),
        "u": (ny, nx + 1),
        "v": (ny + 1, nx),
    }.items():
        spec, relax = _ownership_counts(*shape)
        row = {
            "spec_max_owners": int(spec.max()),
            "relax_max_owners": int(relax.max()),
            "spec_relax_overlap_cells": int(np.count_nonzero((spec + relax) > 1)),
        }
        row["exactly_once"] = bool(
            row["spec_max_owners"] <= 1
            and row["relax_max_owners"] <= 1
            and row["spec_relax_overlap_cells"] == 0
        )
        ownership_green = ownership_green and row["exactly_once"]
        ownership[label] = row

    gate = bool(
        all(row["within_rtol_3e_13_atol_3e_13"] for row in comparisons.values())
        and closure_max <= 1.0e-12
        and target_nonzero
        and deterministic
        and support_green
        and ownership_green
    )
    return {
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "fixture": {
            "seed": 23420260721,
            "mass_shape": [nz, ny, nx],
            "staggering": {
                "u": [nz, ny, nx + 1],
                "v": [nz, ny + 1, nx],
                "w_ph": [nz + 1, ny, nx],
            },
            "lead_seconds": 6.0,
            "dt_full": dt_full,
            "coupled_boundary_leaves": True,
            "include_nested_w": True,
        },
        "wrf_literal_oracle": comparisons,
        "target_field_decomposition": decomposition,
        "combined_closure_max_abs": closure_max,
        "target_side_nonzero": target_nonzero,
        "repeat_all_leaves_bit_identical": deterministic,
        "support_gate": support_green,
        "corner_ownership": ownership,
        "frozen_tolerances": {
            "wrf_oracle_rtol": 3.0e-13,
            "wrf_oracle_atol": 3.0e-13,
            "decomposition_max_abs": 1.0e-12,
        },
        "gate": gate,
    }


@contextlib.contextmanager
def _temporary_env(name: str, value: str | None):
    old_present = name in os.environ
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old_present:
            assert old is not None
            os.environ[name] = old
        else:
            os.environ.pop(name, None)


def runtime_retention_gate(approved_head: str) -> dict[str, Any]:
    from gpuwrf.integration.nested_pipeline import _make_namelist
    from gpuwrf.runtime.aot_cheap_key import static_config_hash
    from gpuwrf.runtime.operational_mode import (
        _nested_frozen_wrf_boundary_active,
        _rk_scan_step,
    )
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid

    grid = build_flat_grid(nx=8, ny=8, nz=3, dx_m=1000.0)
    grid = dataclasses.replace(grid, bc=dataclasses.replace(grid.bc, source="AIFS"))

    def configured(value: str | None):
        with _temporary_env("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", value):
            return _make_namelist(
                grid=grid,
                tendencies=object(),
                metrics=grid.metrics,
                dt_s=2.0,
                parent_dt_s=6.0,
                run_start=datetime(2026, 7, 21, tzinfo=timezone.utc),
                radiation_static=None,
                cu_physics=0,
            )

    unset = configured(None)
    explicit_on = configured("1")
    rollback = configured("0")
    source = inspect.getsource(_rk_scan_step)
    hoisted_once = bool(
        source.count("_nested_frozen_bdy_relax(") == 1
        and source.index("nested_frozen_relax =") < source.index("def advance_stage")
    )

    confirm = json.loads(DEFAULT_CONFIRM.read_text())
    confirm_hash_ok = sha256_file(DEFAULT_CONFIRM) == DEFAULT_CONFIRM_EXPECTED_SHA256
    confirm_source = confirm["source_sha"]
    src_changes_after_confirm = run_git(
        "diff",
        "--name-only",
        f"{confirm_source}..{approved_head}",
        "--",
        "src/gpuwrf",
    ).stdout.splitlines()
    confirm_correctness = confirm["correctness"]
    retained_confirm_green = bool(
        confirm_hash_ok
        and confirm["classification"] == "SCIENTIFIC_PASS__POST_PROOF_TIMEOUT"
        and confirm["configuration"]["frozen_wrf_boundary_bundle_environment"] is None
        and confirm["configuration"]["resolved_frozen_wrf_boundary_bundle"]
        == {"d01": False, "d02": True}
        and confirm_correctness["all_theta_finite"]
        and confirm_correctness["matches_explicit_bundle_on_terminal_states"]
        and confirm_correctness["matches_explicit_bundle_on_terminal_carries"]
        and git_is_ancestor(confirm_source, approved_head)
    )

    unset_hash = static_config_hash(unset)
    explicit_hash = static_config_hash(explicit_on)
    rollback_hash = static_config_hash(rollback)
    config_green = bool(
        _nested_frozen_wrf_boundary_active(unset)
        and _nested_frozen_wrf_boundary_active(explicit_on)
        and not _nested_frozen_wrf_boundary_active(rollback)
        and unset_hash == explicit_hash
        and unset_hash != rollback_hash
    )

    return {
        "fresh_configuration": {
            "unset_active": bool(_nested_frozen_wrf_boundary_active(unset)),
            "explicit_on_active": bool(
                _nested_frozen_wrf_boundary_active(explicit_on)
            ),
            "rollback_active": bool(_nested_frozen_wrf_boundary_active(rollback)),
            "unset_static_config_hash": unset_hash,
            "explicit_on_static_config_hash": explicit_hash,
            "rollback_static_config_hash": rollback_hash,
            "gate": config_green,
        },
        "rk1_freeze_and_reuse": {
            "nested_relax_call_count_in_rk_scan_source": source.count(
                "_nested_frozen_bdy_relax("
            ),
            "hoisted_before_advance_stage": hoisted_once,
        },
        "retained_default_unset_confirmation": {
            "artifact": file_row(DEFAULT_CONFIRM),
            "expected_sha256": DEFAULT_CONFIRM_EXPECTED_SHA256,
            "source_sha": confirm_source,
            "source_is_ancestor_of_approved_head": git_is_ancestor(
                confirm_source, approved_head
            ),
            "terminal_states_sha256": confirm_correctness[
                "terminal_states_sha256"
            ],
            "terminal_carries_sha256": confirm_correctness[
                "terminal_carries_sha256"
            ],
            "matches_explicit_on_state": confirm_correctness[
                "matches_explicit_bundle_on_terminal_states"
            ],
            "matches_explicit_on_carry": confirm_correctness[
                "matches_explicit_bundle_on_terminal_carries"
            ],
            "src_changes_after_confirmation": src_changes_after_confirm,
            "gate": retained_confirm_green,
        },
        "gate": bool(config_green and hoisted_once and retained_confirm_green),
    }


def parse_junit(path: Path, *, required: tuple[str, ...]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    names = sorted(
        testcase.attrib.get("name", "") for testcase in root.iter("testcase")
    )
    required_present = {name: name in names for name in required}
    return {
        "artifact": file_row(path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "required_tests_present": required_present,
        "gate": bool(
            tests > 0
            and failures == 0
            and errors == 0
            and all(required_present.values())
        ),
    }


def run_pytest_group(
    *,
    targets: tuple[str, ...],
    junit: Path,
    log: Path,
    required: tuple[str, ...],
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *targets,
        f"--junitxml={junit}",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout)
    parsed = parse_junit(junit, required=required) if junit.exists() else {
        "gate": False,
        "error": "JUnit artifact missing",
    }
    return {
        "command": command,
        "returncode": completed.returncode,
        "log": file_row(log),
        "junit": parsed,
        "gate": bool(completed.returncode == 0 and parsed["gate"]),
    }


def run_focused_tests(
    *,
    model_junit: Path,
    model_log: Path,
    numpy_junit: Path,
    numpy_log: Path,
) -> dict[str, Any]:
    # The historical discriminator deliberately asserts that neither JAX nor
    # gpuwrf is imported in its process. Pytest imports every selected test
    # module during collection, so mixing it with current production tests is
    # an invalid harness: those modules necessarily import JAX before any test
    # runs. Keep the historical suite in its own fresh interpreter.
    numpy_group = run_pytest_group(
        targets=NUMPY_ISOLATED_TESTS,
        junit=numpy_junit,
        log=numpy_log,
        required=(
            "test_e_decomposition_closure",
            "test_no_jax_or_gpuwrf_imported_by_discriminator",
        ),
    )
    model_group = run_pytest_group(
        targets=FOCUSED_TESTS,
        junit=model_junit,
        log=model_log,
        required=(
            "test_coupled_forcedown_relax_consumes_records_without_second_mass_weight",
            "test_complete_candidate_on_ordinary_cpu_step_is_finite_and_interface_stable",
            "test_fresh_production_child_defaults_to_accepted_bundle_with_rollback",
        ),
    )
    return {
        "harness_correction": (
            "historical NumPy-only import-discipline suite isolated from "
            "production JAX test collection"
        ),
        "numpy_only_group": numpy_group,
        "current_model_group": model_group,
        "tests": (
            int(numpy_group["junit"].get("tests", 0))
            + int(model_group["junit"].get("tests", 0))
        ),
        "failures": (
            int(numpy_group["junit"].get("failures", 0))
            + int(model_group["junit"].get("failures", 0))
        ),
        "errors": (
            int(numpy_group["junit"].get("errors", 0))
            + int(model_group["junit"].get("errors", 0))
        ),
        "skipped": (
            int(numpy_group["junit"].get("skipped", 0))
            + int(model_group["junit"].get("skipped", 0))
        ),
        "gate": bool(numpy_group["gate"] and model_group["gate"]),
    }


def resource_attestation() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "cpu_affinity": affinity,
        "required_cpu_affinity": [13, 14, 15, 29, 30, 31],
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "gpu_commands_queries_locks": 0,
        "wrf_executions": 0,
        "mpi_executions": 0,
        "gate": bool(
            affinity == [13, 14, 15, 29, 30, 31]
            and os.getpriority(os.PRIO_PROCESS, 0) == 15
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("JAX_PLATFORMS") == "cpu"
            and os.environ.get("OMP_NUM_THREADS") == "1"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-head", required=True)
    parser.add_argument(
        "--output", type=Path, default=SPRINT / "S2_REVALIDATION_PROOF.json"
    )
    parser.add_argument(
        "--junit", type=Path, default=SPRINT / "FOCUSED_MODEL_TESTS.xml"
    )
    parser.add_argument(
        "--test-log", type=Path, default=SPRINT / "FOCUSED_MODEL_TESTS.log"
    )
    parser.add_argument(
        "--numpy-junit",
        type=Path,
        default=SPRINT / "FOCUSED_NUMPY_TESTS.xml",
    )
    parser.add_argument(
        "--numpy-test-log",
        type=Path,
        default=SPRINT / "FOCUSED_NUMPY_TESTS.log",
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    head = run_git("rev-parse", "HEAD").stdout.strip()
    approved = run_git("rev-parse", args.approved_head).stdout.strip()
    if head != approved:
        raise SystemExit(f"HEAD {head} != approved head {approved}")
    if not git_is_ancestor(ASSIGNED_BASE, approved):
        raise SystemExit("approved head does not descend from assigned base")

    source_audit = source_interaction_audit(approved)
    historical = historical_retained_replay()
    production = current_production_oracle()
    runtime = runtime_retention_gate(approved)
    tests = (
        run_focused_tests(
            model_junit=args.junit,
            model_log=args.test_log,
            numpy_junit=args.numpy_junit,
            numpy_log=args.numpy_test_log,
        )
        if args.run_tests
        else {"gate": False, "status": "not run"}
    )
    resources = resource_attestation()

    gates = {
        "G0_authority_and_interaction_surface": bool(source_audit["gate"]),
        "G1_historical_replay_and_corrected_sign": bool(historical["gate"]),
        "G2_current_production_wrf_oracle": bool(production["gate"]),
        "G3_current_runtime_retention": bool(runtime["gate"]),
        "G4_focused_regressions": bool(tests["gate"]),
        "resource_attestation": bool(resources["gate"]),
    }
    all_green = all(gates.values())
    verdict = "S2_GREEN" if all_green else "S2_OPEN"

    payload = {
        "schema": "gpuwrf.v0234.s2-boundary-retention-revalidation.v1",
        "verdict": verdict,
        "approved_head": approved,
        "assigned_base": ASSIGNED_BASE,
        "baseline_s2_commit": BASELINE_COMMIT,
        "ledger_authority": {
            "commit": LEDGER_COMMIT,
            "path": ".agent/decisions/V0234-LIVE-CORRECTNESS-LEDGER-2026-07-19.md",
            "note": "ledger is absent from assigned base; read immutably from this commit",
        },
        "contract": file_row(SPRINT / "CONTRACT.md"),
        "source_interaction_audit": source_audit,
        "historical_retained_replay": historical,
        "current_production_oracle": production,
        "runtime_retention": runtime,
        "focused_tests": tests,
        "resource_attestation": resources,
        "efficiency_and_completeness_audit": {
            "significant_new_production_inefficiency_found": False,
            "finding": (
                "The production relax stencil remains the accepted static JAX "
                "scatter implementation and is byte-unchanged. B1/B2 affect "
                "runtime construction/scheduling, not the boundary operator."
            ),
            "historical_methodology_issue_corrected": (
                "field-side sign bookkeeping corrected in this proof without "
                "mutating historical evidence"
            ),
        },
        "gates": gates,
        "terminal": {
            "all_green": all_green,
            "s2_target_side_localization_retained": bool(
                historical["gate"] and production["gate"] and runtime["gate"]
            ),
            "model_source_fix_required": False,
            "gpu_required": False,
            "next_decision": (
                "none for S2; manager may advance to the pinned deterministic V10/V gate"
                if all_green
                else "inspect first false gate; no downstream acceptance advancement"
            ),
        },
        "commands": {
            "proof_invocation_argv": sys.argv,
            "focused_test_targets": list(FOCUSED_TESTS),
            "numpy_isolated_test_targets": list(NUMPY_ISOLATED_TESTS),
        },
    }
    write_self_hashed(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "proof_sha256": json.loads(args.output.read_text())[
                    "proof_sha256"
                ],
                "verdict": verdict,
                "gates": gates,
                "corrected_field_side_fraction": historical[
                    "corrected_sign_audit"
                ]["field_side_fraction"],
                "corrected_target_side_ratio": historical[
                    "corrected_sign_audit"
                ]["target_side_rms_ratio"],
                "production_closure_max_abs": production[
                    "combined_closure_max_abs"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not all_green:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
