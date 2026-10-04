#!/usr/bin/env python3
"""Reproduce the retained-output evidence around the V10 late-Ni failure.

This is deliberately a CPU/read-only discriminator.  It does not load a model
carry, initialize JAX, query a GPU, or run WRF.  The proof separates what the
retained hourly/20-minute frames can establish from the operation-local facts
that require a fresh exact carry.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any

import netCDF4
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-late-ni-rootcause"
DEFAULT_OUTPUT = SPRINT / "RETAINED_PRECURSOR_PROOF.json"

CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
GPU_NAMESPACE = (
    CASE_ROOT
    / "corrected_ni_rca_max_22c2bd7a"
    / "v0234_gpt_v10_rootcause_30b5cf89ff6027c4_full18h"
)
GPU_OUTPUT = GPU_NAMESPACE / "gpu-output"
CPU_OUTPUT = CASE_ROOT / "run/wrf"
BLOCKER = GPU_NAMESPACE / "step9000-blocker.json"
NAMELIST = CASE_ROOT / "config/namelist.input"

SINT_ORACLE = ROOT / "scripts/v0234_wrf_sint_source_oracle.py"
WRF_SINT = Path("<USER_HOME>/src/wrf_pristine/WRF/share/sint.F")
BOUNDARY_CONSTRUCTION = ROOT / "src/gpuwrf/nesting/boundary_construction.py"
BOUNDARY_APPLY = ROOT / "src/gpuwrf/coupling/boundary_apply.py"
OPERATIONAL_MODE = ROOT / "src/gpuwrf/runtime/operational_mode.py"
SCALAR_ADVECTION = ROOT / "src/gpuwrf/dynamics/flux_advection.py"
SCALAR_DIFFUSION = ROOT / "src/gpuwrf/dynamics/explicit_diffusion.py"
TERMINAL_RUNNER = ROOT / "scripts/v0234_h5_terminal_step9000.py"

HISTORICAL_COMPARATOR = (
    ROOT
    / ".agent/sprints/2026-07-17-v0234-final-holistic-fable5-xhigh2"
    / "amendment-01-late-window-corner-discriminator.json"
)

THRESHOLDS = (1.0e-30, 1.0e-20, 1.0e-10, 1.0, 10.0, 100.0, 1000.0, 10000.0)
CORNER_NAMES = ("southwest", "southeast", "northwest", "northeast")
CORNER_INDEX = ((1, 1), (1, -2), (-2, 1), (-2, -2))


class ForensicsFailure(RuntimeError):
    """Raised when an authority or retained-input invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticated_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ForensicsFailure(f"missing authority/input file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = deepcopy(payload)
    canonical.pop("canonical_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_variable(path: Path, name: str) -> np.ndarray:
    with netCDF4.Dataset(path, "r") as dataset:
        if name not in dataset.variables:
            raise ForensicsFailure(f"{name} absent from {path}")
        value = np.asanyarray(dataset.variables[name][0])
    if np.ma.isMaskedArray(value):
        if np.any(value.mask):
            raise ForensicsFailure(f"masked values in {name}: {path}")
        value = value.data
    return np.asarray(value)


def array_stats(value: np.ndarray, *, thresholds: tuple[float, ...] = ()) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    stats: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        "nonzero_count": int(np.count_nonzero(array)),
    }
    if np.any(finite):
        finite_value = array[finite]
        stats["finite_min"] = float(np.min(finite_value))
        stats["finite_max"] = float(np.max(finite_value))
        stats["finite_max_abs"] = float(np.max(np.abs(finite_value)))
    if thresholds:
        stats["abs_count_above"] = {
            format(threshold, ".0e") if threshold < 1.0 else format(threshold, "g"): int(
                np.count_nonzero(np.abs(array[finite]) > threshold)
            )
            for threshold in thresholds
        }
    return stats


def hydrometeor_stats(path: Path) -> dict[str, Any]:
    return {
        "QNICE": array_stats(_read_variable(path, "QNICE"), thresholds=THRESHOLDS),
        "QICE": array_stats(_read_variable(path, "QICE")),
    }


def _load_sint_oracle():
    spec = importlib.util.spec_from_file_location("v0234_wrf_sint_source_oracle", SINT_ORACLE)
    if spec is None or spec.loader is None:
        raise ForensicsFailure(f"cannot import independent SINT oracle: {SINT_ORACLE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_to_child_ni(path: Path) -> dict[str, Any]:
    """Reconstruct the d02->d03 coupled QNI target from the retained 17:00 frame.

    GPU state is fp64 but wrfout stores these variables as fp32.  The principal
    reconstruction therefore promotes the retained values before the source-
    literal SINT operation.  A native-fp32 reconstruction is retained as a
    rounding-envelope cross-check; both must stay finite.
    """

    ni = _read_variable(path, "QNICE")
    mu_total = _read_variable(path, "MU") + _read_variable(path, "MUB")
    c1h = _read_variable(path, "C1H")
    c2h = _read_variable(path, "C2H")
    mass_h = c1h[:, None, None] * mu_total[None, :, :] + c2h[:, None, None]
    coupled = ni * mass_h
    oracle = _load_sint_oracle()

    reconstructions: dict[str, Any] = {}
    for label, dtype in (("promoted_fp64", np.float64), ("native_fp32", np.float32)):
        child = oracle.sint_full(
            coupled.astype(dtype),
            ratio=3,
            i_parent_start=92,
            j_parent_start=36,
            child_ny=93,
            child_nx=111,
        )
        strips = oracle.wrf_sides(child, width=5, side_len=111)
        reconstructions[label] = {
            "child_full": array_stats(child),
            "boundary_strips": array_stats(strips),
            "south_relax_ring1_k0_x1_coupled_qni": float(strips[2, 1, 0, 1]),
        }

    if any(
        item["child_full"]["nonfinite_count"] != 0
        or item["boundary_strips"]["nonfinite_count"] != 0
        for item in reconstructions.values()
    ):
        raise ForensicsFailure("retained d02->d03 SINT reconstruction became nonfinite")

    return {
        "geometry": {
            "parent": "d02",
            "child": "d03",
            "ratio": 3,
            "i_parent_start_1based": 92,
            "j_parent_start_1based": 36,
            "child_mass_shape": [44, 93, 111],
            "boundary_width": 5,
        },
        "retained_output_precision_caveat": (
            "wrfout QNICE/MU/MUB/C1H/C2H are fp32 snapshots of an fp64 model carry; "
            "this proves a finite, tiny retained-frame target but is not an exact carry replay"
        ),
        "parent_mass_h": array_stats(mass_h),
        "parent_coupled_qni": array_stats(coupled),
        "reconstructions": reconstructions,
    }


def d03_corner_health(path: Path) -> dict[str, Any]:
    mu_perturbation = _read_variable(path, "MU")
    mu_total = mu_perturbation + _read_variable(path, "MUB")
    u = _read_variable(path, "U")
    v = _read_variable(path, "V")
    w = _read_variable(path, "W")
    corners: dict[str, Any] = {}
    for name, (y_index, x_index) in zip(CORNER_NAMES, CORNER_INDEX, strict=True):
        y_window = slice(0, 4) if y_index > 0 else slice(-4, None)
        x_window = slice(0, 5) if x_index > 0 else slice(-5, None)
        corners[name] = {
            "index_yx": [y_index, x_index],
            "mu_perturbation_pa": float(mu_perturbation[y_index, x_index]),
            "mu_total_pa": float(mu_total[y_index, x_index]),
            "u_k0_m_s": float(u[0, y_index, x_index]),
            "v_k0_m_s": float(v[0, y_index, x_index]),
            "lowest_level_local_uv_max_abs_m_s": float(
                max(
                    np.max(np.abs(u[0, y_window, x_window])),
                    np.max(np.abs(v[0, y_window, x_window])),
                )
            ),
        }
    return {
        "corners": corners,
        "corner_mu_perturbation_max_abs_pa": float(
            max(abs(item["mu_perturbation_pa"]) for item in corners.values())
        ),
        "corner_mu_total_min_pa": float(min(item["mu_total_pa"] for item in corners.values())),
        "corner_mu_total_max_pa": float(max(item["mu_total_pa"] for item in corners.values())),
        "corner_local_lowest_uv_max_abs_m_s": float(
            max(item["lowest_level_local_uv_max_abs_m_s"] for item in corners.values())
        ),
        "global_w_max_abs_m_s": float(np.max(np.abs(w))),
    }


def source_topology() -> dict[str, Any]:
    namelist = NAMELIST.read_text(encoding="utf-8")
    boundary = BOUNDARY_APPLY.read_text(encoding="utf-8")
    operational = OPERATIONAL_MODE.read_text(encoding="utf-8")
    diffusion = SCALAR_DIFFUSION.read_text(encoding="utf-8")
    terminal_runner = TERMINAL_RUNNER.read_text(encoding="utf-8")
    required = {
        "namelist_parent_geometry": all(
            token in namelist
            for token in (
                "i_parent_start = 1, 16, 92",
                "j_parent_start = 1, 16, 36",
                "parent_grid_ratio = 1, 3, 3",
            )
        ),
        "namelist_scalar_adv_opt_1": "scalar_adv_opt = 1, 1, 1" in namelist,
        "scalar_relax_then_spec": all(
            token in boundary
            for token in (
                "def nested_scalar_boundary_tendencies(",
                "_scatter_relax_tendency(",
                "_scatter_spec_scalar_tendency(",
                "tendencies.append(tendency + spec_tendency)",
            )
        ),
        "scalar_boundary_frozen_at_rk1": all(
            token in operational
            for token in (
                "nested_scalar_boundary_tendencies(",
                "nested_frozen_scalar = (",
                "if nested_frozen_bundle and lead_seconds is not None",
            )
        ),
        "scalar_diff6_excludes_rings_0_to_2": all(
            token in diffusion
            for token in (
                "if bool(specified_or_nested):",
                "owned = (yy >= 3) & (yy <= ny - 4) & (xx >= 3) & (xx <= nx - 4)",
            )
        ),
        "terminal_segment_1134_checked_d03_10206_finite": all(
            token in terminal_runner
            for token in (
                'ROOT_STEP_GATE = 1000',
                'output_cadence = {"d01": 67, "d02": 200, "d03": 200}',
                'segment_steps = min(output_cadence["d01"], target - segment_start)',
                'assert_state_finite_at_boundary(',
            )
        ),
    }
    if not all(required.values()):
        raise ForensicsFailure(f"source topology assertion failed: {required}")
    return {
        "assertions": required,
        "failure_cell_y1_x1_semantics": (
            "south-owned relaxation ring 1: scalar advection and RK1-frozen boundary relaxation "
            "are active; specified overwrite is ring 0 and sixth-order scalar diffusion is zero"
        ),
        "last_completed_segment_derivation": (
            "after the sealed d01 step1000/d03 step9000 gate, 67-root-step segments end at "
            "d01 1067/d03 9603 and d01 1134/d03 10206; the runner materializes and finite-checks "
            "every domain after each completed segment before entering the segment that fails at "
            "the d03 step10400 output alarm"
        ),
    }


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    )
    return completed.stdout.strip()


def build_proof() -> dict[str, Any]:
    blocker = json.loads(BLOCKER.read_text(encoding="utf-8"))
    expected_exception = (
        "non-finite prognostic state detected: domain=d03 field=Ni level=0 "
        "step=10400 sim_time_s=62400 first_index=(0, 1, 1)"
    )
    if blocker.get("exception") != expected_exception:
        raise ForensicsFailure(f"unexpected V10 blocker: {blocker.get('exception')!r}")

    frame_paths = {
        "gpu_d02_1600": GPU_OUTPUT / "wrfout_d02_2025-03-01_16:00:00",
        "gpu_d02_1700": GPU_OUTPUT / "wrfout_d02_2025-03-01_17:00:00",
        "gpu_d03_1600": GPU_OUTPUT / "wrfout_d03_2025-03-01_16:00:00",
        "gpu_d03_1700": GPU_OUTPUT / "wrfout_d03_2025-03-01_17:00:00",
        "cpu_d02_1600": CPU_OUTPUT / "wrfout_d02_2025-03-01_16:00:00",
        "cpu_d02_1700": CPU_OUTPUT / "wrfout_d02_2025-03-01_17:00:00",
        "cpu_d03_1700": CPU_OUTPUT / "wrfout_d03_2025-03-01_17:00:00",
        "cpu_d03_1800": CPU_OUTPUT / "wrfout_d03_2025-03-01_18:00:00",
    }

    hydrometeors = {
        name: hydrometeor_stats(path)
        for name, path in frame_paths.items()
    }
    d03_zero_checks = {
        name: (
            hydrometeors[name]["QNICE"]["nonzero_count"] == 0
            and hydrometeors[name]["QICE"]["nonzero_count"] == 0
        )
        for name in ("gpu_d03_1600", "gpu_d03_1700", "cpu_d03_1700", "cpu_d03_1800")
    }
    if not all(d03_zero_checks.values()):
        raise ForensicsFailure(f"unexpected retained d03 ice activation: {d03_zero_checks}")

    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.gpt-late-ni.retained-precursor-proof.v1",
        "verdict": "LATE_NI_RETAINED_PRECURSOR_NARROWED_EXACT_CARRY_REQUIRED",
        "branch": _git_value("branch", "--show-current"),
        "accepted_v10_model_commit": blocker["candidate_model_commit"],
        "method": {
            "type": "read-only retained real-fixture NetCDF plus source-literal CPU SINT",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "model_dispatches": 0,
            "resource_policy": (
                "taskset 13,14,15,29,30,31; nice 15; ionice class 3; "
                "one OpenMP/BLAS thread"
            ),
        },
        "failure": {
            "exception": blocker["exception"],
            "guarded_bad_step": 10400,
            "guarded_bad_sim_time_s": 62400,
            "last_retained_checked_d03_step": 10200,
            "last_retained_checked_sim_time_s": 61200,
            "last_completed_segment_finite_d03_step": 10206,
            "last_completed_segment_finite_sim_time_s": 61236,
            "guarded_ni_onset_interval_d03_steps": "(10206,10400]",
            "scratch_first_onset_interval": (
                "not bounded by the prognostic-only segment/output guard; scratch may precede "
                "the Ni interval"
            ),
            "first_guarded_field": "Ni",
            "first_guarded_index_zyx": [0, 1, 1],
            "guard_limit": (
                "the output-boundary guard checks prognostic State only; it neither identifies the "
                "first bad dispatch nor excludes an earlier scratch leaf"
            ),
        },
        "authorities": {
            "blocker": authenticated_file(BLOCKER),
            "namelist": authenticated_file(NAMELIST),
            "wrf_sint": authenticated_file(WRF_SINT),
            "independent_sint_oracle": authenticated_file(SINT_ORACLE),
            "historical_old_tree_comparator": authenticated_file(HISTORICAL_COMPARATOR),
            "current_sources": {
                path.relative_to(ROOT).as_posix(): authenticated_file(path)
                for path in (
                    TERMINAL_RUNNER,
                    BOUNDARY_CONSTRUCTION,
                    BOUNDARY_APPLY,
                    OPERATIONAL_MODE,
                    SCALAR_ADVECTION,
                    SCALAR_DIFFUSION,
                )
            },
            "retained_frames": {
                name: authenticated_file(path) for name, path in frame_paths.items()
            },
        },
        "source_topology": source_topology(),
        "retained_hydrometeors": hydrometeors,
        "d03_ice_zero_checks": d03_zero_checks,
        "d02_to_d03_ni_reconstruction": parent_to_child_ni(frame_paths["gpu_d02_1700"]),
        "d03_1700_corner_health": {
            "gpu": d03_corner_health(frame_paths["gpu_d03_1700"]),
            "cpu_wrf": d03_corner_health(frame_paths["cpu_d03_1700"]),
        },
        "findings": {
            "upstream_activation": (
                "GPU d02 changes from exact-zero QICE/QNICE at 16:00 to finite QICE/QNICE at "
                "17:00; CPU WRF activates the same species but in only 248 cells. GPU d02 has a "
                "much broader mostly tiny QNICE tail, while its meaningful maximum is comparable "
                "to CPU WRF."
            ),
            "child_snapshot": (
                "GPU d03 remains exact-zero QICE/QNICE through its last retained 17:00 frame; "
                "CPU WRF d03 remains exact-zero through 18:00."
            ),
            "finite_seed": (
                "Source-literal d02-to-d03 coupling/SINT of the retained GPU frame intersects only "
                "a finite approximately 1e-9 coupled-QNI tail at the d03 footprint; the exact "
                "reported south ring1 k0 x1 snapshot target is zero."
            ),
            "old_signature_absent_at_1700": (
                "Current d03 ring1 corners have positive approximately 96 kPa total dry mass, "
                "approximately +1.3 kPa perturbation mass, and single-digit local lowest-level "
                "winds. The old-tree pre-detonation -10.6 to -12.3 kPa corner mass-pump signature "
                "is not visible at the last retained frame."
            ),
        },
        "hypothesis_effect": {
            "H1_nested_scalar_or_boundary": (
                "raised, but narrowed to amplification/transport after a finite tiny seed; direct "
                "nonfinite parent-boundary injection at 17:00 is excluded"
            ),
            "H2_thompson_first_nonfinite": (
                "open only as a same-step producer/consumer question; retained frames contain no "
                "d03 ice activation and cannot place the first Thompson operation"
            ),
            "H3_earlier_scratch_first": (
                "open and co-leading because the terminal guard does not inspect scratch leaves"
            ),
            "H4_boundary_clock_or_ownership": (
                "open; failure cell is south-owned relax ring1 where scalar advection and frozen "
                "boundary relaxation meet"
            ),
            "H5_precision_or_scheduling": (
                "deprioritized: both fp64-promoted and fp32 retained-frame SINT reconstructions "
                "are finite, but only exact-carry replay can test long-window realization"
            ),
        },
        "exclusions": [
            "No retained evidence supports reopening MYNN dissipative heating, moist-theta finish, or the native step-15 Thompson first-ice event.",
            "The historical old-tree step-9314 event is not assumed causal for this current-tree step-10400 event.",
            "Retained output cannot distinguish scalar advection, boundary relaxation, Thompson, or an earlier acoustic scratch operation inside (10200,10400].",
            "No clamp, sanitizer, tolerance relaxation, or output-only mask is authorized by this proof.",
        ],
        "next_discriminator": {
            "required": "fresh exact current-tree carry and complete-leaf health around the onset",
            "minimum_arm": (
                "one continuous accepted Step0 trajectory to the root-aligned d03 step10206, "
                "identity-authenticated against retained 17:00 output, then production-aligned "
                "root/d02-triplet health with atomic last-green/first-red complete carries; stop "
                "at first red"
            ),
            "capture": [
                "all prognostic and persistent scratch leaves",
                "Ni/Nr coupled scalar advection and PD limiter intermediates",
                "Ni boundary target/relax/spec tendencies at ring1",
                "Thompson Ni input/output/source terms if scalar state remains finite on entry",
                "dry mass, mass flux, pressure/geopotential, and RK/acoustic stage identity",
            ],
            "directional_decision": {
                "scratch_first": "any scratch nonfinite/unbounded while all prognostic scalars remain finite",
                "transport_or_boundary_first": "Ni becomes nonfinite during scalar merge/update from finite Thompson input",
                "thompson_first": "finite Ni enters Thompson and a named source expression first becomes nonfinite",
            },
        },
    }
    proof["canonical_sha256"] = canonical_sha256(proof)
    return proof


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the rebuilt canonical payload with --output instead of writing",
    )
    args = parser.parse_args()
    proof = build_proof()
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if canonical_sha256(existing) != existing.get("canonical_sha256"):
            raise ForensicsFailure("existing proof canonical hash is invalid")
        if existing != proof:
            raise ForensicsFailure("rebuilt proof differs from retained proof")
    else:
        write_atomic(args.output, proof)
    print(json.dumps({"output": str(args.output), "verdict": proof["verdict"], "canonical_sha256": proof["canonical_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
