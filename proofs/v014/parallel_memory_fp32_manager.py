#!/usr/bin/env python3
"""Parallel v0.14 memory/FP32 manager proof bundle.

CPU-only by default.  This script records the file collision map, proves the
WDM6 slmsk memory cleanup is exact against the previous full-column layout, and
refreshes the existing FP32 acoustic feasibility blockers without touching the
locked production dycore/runtime/nesting paths.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT_JSON = ROOT / "proofs" / "v014" / "parallel_memory_fp32_manager.json"
OUT_MD = ROOT / "proofs" / "v014" / "parallel_memory_fp32_manager.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.contracts.state import State, _state_field_shapes  # noqa: E402
from gpuwrf.coupling.scan_adapters import (  # noqa: E402
    GRAVITY_M_S2,
    P0_PA,
    R_D_OVER_CP,
    _apply_mp_replacements,
    _mp_in,
    _output_dtype,
    _rho_from_state,
    wdm6_adapter,
)
from gpuwrf.physics.microphysics_wdm6 import wdm6_physics_tendency  # noqa: E402


LOCKED_PATTERNS = (
    "src/gpuwrf/dynamics/**",
    "src/gpuwrf/runtime/operational_mode.py",
    "src/gpuwrf/integration/d02_replay.py",
    "src/gpuwrf/nesting/**",
    "src/gpuwrf/boundary*",
    "src/gpuwrf/contracts/state.py",
    "live-nest/base-state/init/restart/boundary/carry files",
)

MEMORY_CANDIDATES = (
    {
        "name": "wdm6_slmsk_shape_only_cleanup",
        "files": ["src/gpuwrf/coupling/scan_adapters.py"],
        "status": "implemented",
        "collision": False,
        "reason": "single allowed adapter file; no dycore/runtime/nesting/boundary/state edit",
    },
    {
        "name": "moisture_transport_velocity_reuse",
        "files": [
            "src/gpuwrf/runtime/operational_mode.py",
            "src/gpuwrf/dynamics/flux_advection.py",
        ],
        "status": "blocked_by_lock",
        "collision": True,
        "reason": "runtime and dynamics are locked by the active fp64 grid-parity debug",
    },
    {
        "name": "post_physics_sparse_merge",
        "files": [
            "src/gpuwrf/runtime/operational_mode.py",
            "src/gpuwrf/coupling/physics_couplers.py",
            "src/gpuwrf/coupling/scan_adapters.py",
        ],
        "status": "blocked_by_lock_and_measure_first",
        "collision": True,
        "reason": "needs locked runtime and changes coupling liveness/donation semantics",
    },
    {
        "name": "moisture_limiter_workspace_reduction",
        "files": [
            "src/gpuwrf/dynamics/flux_advection.py",
            "src/gpuwrf/runtime/operational_mode.py",
        ],
        "status": "blocked_by_lock",
        "collision": True,
        "reason": "dynamics/runtime locked; active scalar path requires conservation proof",
    },
    {
        "name": "acoustic_carry_split_or_pad_cleanup",
        "files": [
            "src/gpuwrf/dynamics/core/acoustic.py",
            "src/gpuwrf/dynamics/core/small_step_prep.py",
            "src/gpuwrf/dynamics/core/small_step_finish.py",
            "src/gpuwrf/runtime/operational_mode.py",
        ],
        "status": "blocked_by_lock",
        "collision": True,
        "reason": "same fault surface as current P/MU/W live-nest fp64 debug",
    },
    {
        "name": "state_total_perturbation_base_alias_reduction",
        "files": [
            "src/gpuwrf/contracts/state.py",
            "init/restart/wrfout/boundary compatibility paths",
        ],
        "status": "blocked_by_lock_and_adr",
        "collision": True,
        "reason": "state contract and compatibility surfaces are locked and ADR-required",
    },
)

FP32_CANDIDATES = (
    {
        "name": "R0 precision-mode scaffold",
        "files": [
            "src/gpuwrf/runtime/operational_mode.py",
            "namelist/cache/static-aux consumers",
        ],
        "status": "source_blocked",
        "collision": True,
        "reason": "runtime source is locked; default-inert scaffold still changes cache/static surface",
    },
    {
        "name": "R1 explicit base-state plumbing",
        "files": [
            "src/gpuwrf/dynamics/core/small_step_prep.py",
            "src/gpuwrf/dynamics/core/small_step_finish.py",
            "src/gpuwrf/runtime/operational_mode.py",
            "boundary/restart/init/carry staging",
        ],
        "status": "source_blocked",
        "collision": True,
        "reason": "direct overlap with active live-nest P/MU/W perturbation-state initialization debug",
    },
    {
        "name": "R2 perturbation-authoritative acoustic storage",
        "files": [
            "src/gpuwrf/dynamics/core/acoustic.py",
            "src/gpuwrf/dynamics/core/advance_w.py",
            "src/gpuwrf/dynamics/core/calc_p_rho.py",
            "src/gpuwrf/dynamics/core/rk_addtend_dry.py",
        ],
        "status": "source_blocked",
        "collision": True,
        "reason": "dycore production source is locked during fp64 grid-parity localization",
    },
    {
        "name": "R3 CPU scalar/one-column probes",
        "files": ["proofs/v014/fp32_acoustic_probes.py"],
        "status": "refreshed",
        "collision": False,
        "reason": "proof-only CPU NumPy probe; no production source mutation",
    },
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_snapshot() -> dict[str, Any]:
    status = _run_cmd(["git", "status", "--short", "--branch"])
    base = _run_cmd(["git", "merge-base", "--is-ancestor", "131b27cd", "HEAD"])
    return {
        "branch": _run_cmd(["git", "branch", "--show-current"])["stdout"].strip(),
        "head": _run_cmd(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "base_131b27cd_is_ancestor": base["returncode"] == 0,
        "status_short": status["stdout"].splitlines(),
        "dirty": any(line and not line.startswith("##") for line in status["stdout"].splitlines()),
    }


def _line_hits(path: str, pattern: str) -> dict[str, Any]:
    full = ROOT / path
    text = full.read_text(encoding="utf-8")
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line):
            hits.append({"line": lineno, "text": line.strip()})
    return {
        "path": path,
        "sha256": _sha256(full),
        "pattern": pattern,
        "count": len(hits),
        "hits": hits,
    }


def _candidate_collision_map() -> dict[str, Any]:
    return {
        "locked_patterns": list(LOCKED_PATTERNS),
        "active_primary_debug_surface": (
            "live-nest raw child -> live child perturbation-state initialization "
            "for P_STATE/MU_STATE/W_STATE"
        ),
        "memory_candidates": list(MEMORY_CANDIDATES),
        "fp32_candidates": list(FP32_CANDIDATES),
        "source_audit": {
            "wdm6_slmsk_adapter": _line_hits("src/gpuwrf/coupling/scan_adapters.py", r"slmsk"),
            "moisture_velocity_runtime_sites": _line_hits(
                "src/gpuwrf/runtime/operational_mode.py", r"vel = couple_velocities_periodic\("
            ),
            "small_step_base_recovery_prep": _line_hits(
                "src/gpuwrf/dynamics/core/small_step_prep.py",
                r"state\.(p_total|ph_total|mu_total).*state\.(p_perturbation|ph_perturbation|mu_perturbation)",
            ),
            "small_step_base_recovery_finish": _line_hits(
                "src/gpuwrf/dynamics/core/small_step_finish.py",
                r"state\.(p_total|ph_total|mu_total).*state\.(p_perturbation|ph_perturbation|mu_perturbation)",
            ),
        },
    }


def _synthetic_wdm6_state() -> State:
    grid = SimpleNamespace(nz=10, ny=3, nx=4)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    rng = np.random.default_rng(14016)
    fields = {
        name: jnp.zeros(shape, dtype=jnp.float64)
        for name, shape in _state_field_shapes(grid).items()
    }
    z_iface = np.arange(nz + 1, dtype=np.float64) * 450.0
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    theta_col = 292.0 + 0.0045 * z_mid
    pressure_col = 100000.0 * np.exp(-z_mid / 8500.0)

    def m3(base: np.ndarray, noise: float, *, floor: float | None = None) -> jax.Array:
        value = base[:, None, None] + noise * rng.standard_normal((nz, ny, nx))
        if floor is not None:
            value = np.maximum(value, floor)
        return jnp.asarray(value, dtype=jnp.float64)

    fields["theta"] = m3(theta_col, 0.15)
    fields["p"] = m3(pressure_col, 8.0, floor=5000.0)
    fields["p_total"] = fields["p"]
    fields["p_perturbation"] = fields["p"]
    fields["qv"] = m3(0.012 * np.exp(-z_mid / 3500.0), 1.0e-5, floor=1.0e-8)
    fields["qc"] = m3(np.where((z_mid > 600.0) & (z_mid < 2800.0), 3.5e-4, 1.0e-8), 2.0e-6, floor=0.0)
    fields["qr"] = m3(np.where((z_mid > 600.0) & (z_mid < 2400.0), 6.0e-5, 0.0), 8.0e-7, floor=0.0)
    fields["qi"] = m3(np.where(z_mid > 3500.0, 2.5e-5, 0.0), 4.0e-7, floor=0.0)
    fields["qs"] = m3(np.where(z_mid > 3000.0, 2.0e-5, 0.0), 4.0e-7, floor=0.0)
    fields["qg"] = m3(np.where((z_mid > 2300.0) & (z_mid < 4200.0), 1.5e-5, 0.0), 4.0e-7, floor=0.0)
    fields["Nn"] = m3(np.full(nz, 8.0e7), 5.0e5, floor=1.0e4)
    fields["Nc"] = m3(np.where(z_mid < 3200.0, 5.0e7, 1.0e5), 2.0e5, floor=1.0e3)
    fields["Nr"] = m3(np.where(z_mid < 2600.0, 2.0e6, 1.0e3), 1.0e4, floor=1.0)
    fields["ph"] = jnp.asarray(
        np.broadcast_to(GRAVITY_M_S2 * z_iface[:, None, None], (nz + 1, ny, nx)),
        dtype=jnp.float64,
    )
    fields["ph_total"] = fields["ph"]
    fields["ph_perturbation"] = fields["ph"]
    fields["mu"] = jnp.full((ny, nx), 90000.0, dtype=jnp.float64)
    fields["mu_total"] = fields["mu"]
    fields["mu_perturbation"] = fields["mu"]
    fields["u"] = jnp.asarray(4.0 + rng.standard_normal((nz, ny, nx + 1)) * 0.05, dtype=jnp.float64)
    fields["v"] = jnp.asarray(-1.0 + rng.standard_normal((nz, ny + 1, nx)) * 0.05, dtype=jnp.float64)
    fields["w"] = jnp.asarray(rng.standard_normal((nz + 1, ny, nx)) * 0.01, dtype=jnp.float64)
    fields["qke"] = jnp.full((nz, ny, nx), 0.3, dtype=jnp.float64)
    xland = np.ones((ny, nx), dtype=np.float64)
    xland[:, nx // 2 :] = 2.0
    fields["xland"] = jnp.asarray(xland, dtype=jnp.float64)
    fields["lakemask"] = jnp.asarray(xland > 1.5, dtype=jnp.float64)
    fields["mavail"] = jnp.where(fields["xland"] > 1.5, 1.0, 0.5).astype(jnp.float64)
    fields["roughness_m"] = jnp.where(fields["xland"] > 1.5, 0.002, 0.12).astype(jnp.float64)
    fields["t_skin"] = jnp.where(fields["xland"] > 1.5, 299.0, 302.0).astype(jnp.float64)
    fields["soil_moisture"] = jnp.full((ny, nx), 0.25, dtype=jnp.float64)
    fields["ustar"] = jnp.full((ny, nx), 0.25, dtype=jnp.float64)
    fields["rhosfc"] = jnp.full((ny, nx), 1.15, dtype=jnp.float64)
    fields["lu_index"] = jnp.asarray(xland, dtype=jnp.int32)
    return State(**fields)


def _legacy_wdm6_adapter(state: State, dt_s: float) -> State:
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    slmsk_2d = jnp.where(jnp.asarray(state.xland) < 1.5, 1.0, 0.0)
    slmsk = _mp_in(jnp.broadcast_to(slmsk_2d[None, :, :], state.theta.shape), ny, nx, nz)
    tend = wdm6_physics_tendency(
        mp(state.theta),
        mp(state.qv),
        mp(state.qc),
        mp(state.qr),
        mp(state.qi),
        mp(state.qs),
        mp(state.qg),
        mp(state.Nn),
        mp(state.Nc),
        mp(state.Nr),
        mp(pii),
        mp(rho),
        mp(state.p),
        mp(dz),
        float(dt_s),
        slmsk,
    )
    tend.validate_keys()
    next_state = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    nn = tend.diagnostics.get("Nn")
    if nn is not None:
        nn3d = jnp.moveaxis(jnp.asarray(nn).reshape(ny, nx, nz), -1, 0)
        next_state = next_state.replace(Nn=nn3d.astype(_output_dtype(state, "Nn")))
    return next_state


def _wdm6_shape_equivalence() -> dict[str, Any]:
    state = _synthetic_wdm6_state()
    dt_s = 12.0
    old_state = _legacy_wdm6_adapter(state, dt_s)
    new_state = wdm6_adapter(state, dt_s)
    jax.block_until_ready((old_state.theta, new_state.theta))

    leaf_rows = []
    all_exact = True
    all_finite = True
    for name in State.__slots__:
        old = np.asarray(getattr(old_state, name))
        new = np.asarray(getattr(new_state, name))
        exact = bool(np.array_equal(old, new))
        finite = bool(np.all(np.isfinite(new))) if np.issubdtype(new.dtype, np.floating) else True
        max_abs = float(np.max(np.abs(old.astype(np.float64) - new.astype(np.float64)))) if old.size else 0.0
        all_exact = all_exact and exact
        all_finite = all_finite and finite
        if (not exact) or max_abs != 0.0 or name in {
            "theta",
            "qv",
            "qc",
            "qr",
            "qi",
            "qs",
            "qg",
            "Nn",
            "Nc",
            "Nr",
            "rain_acc",
            "snow_acc",
            "graupel_acc",
        }:
            leaf_rows.append(
                {
                    "leaf": name,
                    "shape": list(new.shape),
                    "dtype": str(new.dtype),
                    "exact": exact,
                    "finite": finite,
                    "max_abs": max_abs,
                }
            )

    nz, ny, nx = state.theta.shape
    slmsk_2d = jnp.where(jnp.asarray(state.xland) < 1.5, 1.0, 0.0)
    old_slmsk = _mp_in(jnp.broadcast_to(slmsk_2d[None, :, :], state.theta.shape), ny, nx, nz)
    new_slmsk = jnp.asarray(slmsk_2d).reshape(ny * nx)
    target_nx, target_ny, target_nz = 641, 321, 50
    target_old_bytes = target_nx * target_ny * target_nz * 8
    target_new_bytes = target_nx * target_ny * 8
    if not all_exact:
        raise AssertionError("WDM6 slmsk shape cleanup was not exact against legacy adapter layout")
    if not all_finite:
        raise AssertionError("WDM6 slmsk shape cleanup produced non-finite output")

    return {
        "verdict": "WDM6_SLMSK_SHAPE_CLEANUP_EXACT",
        "cpu_only": True,
        "jax_platforms": [device.platform for device in jax.devices()],
        "synthetic_grid": {"nz": nz, "ny": ny, "nx": nx, "ncol": ny * nx},
        "dt_s": dt_s,
        "legacy_slmsk_shape": list(old_slmsk.shape),
        "new_slmsk_shape": list(new_slmsk.shape),
        "legacy_slmsk_unique": sorted(float(x) for x in np.unique(np.asarray(old_slmsk))),
        "new_slmsk_unique": sorted(float(x) for x in np.unique(np.asarray(new_slmsk))),
        "preserved_value_semantics": "kept existing adapter 1.0 land / 0.0 water values; no 1/2 land-sea semantic change",
        "all_state_leaves_exact": all_exact,
        "all_state_leaves_finite": all_finite,
        "reported_leaf_rows": leaf_rows,
        "target_641x321x50_fp64_bytes": {
            "old_full_column": target_old_bytes,
            "new_per_column": target_new_bytes,
            "saved_bytes": target_old_bytes - target_new_bytes,
            "saved_mib": (target_old_bytes - target_new_bytes) / 1024**2,
            "saved_gib": (target_old_bytes - target_new_bytes) / 1024**3,
        },
    }


def _load_fp32_probe_summary() -> dict[str, Any]:
    path = ROOT / "proofs" / "v014" / "fp32_acoustic_probes.py"
    spec = importlib.util.spec_from_file_location("fp32_acoustic_probes_current", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = module.build_results()
    abs_probe = results["absolute_total_cancellation"]
    pert_probe = results["perturbation_form_preservation"]
    recurrence = results["one_column_recurrence_sensitivity"]["errors_vs_fp64_reference"]
    memory = results["memory_savings"]
    return {
        "source": str(path.relative_to(ROOT)),
        "source_sha256": _sha256(path),
        "cpu_only": True,
        "absolute_total_millipascal_recurrent_delta_pa": abs_probe[
            "millipascal_recurrent_recovered_delta_pa"
        ],
        "absolute_total_millipascal_fresh_delta_pa": abs_probe[
            "millipascal_fresh_recovered_delta_pa"
        ],
        "fp32_ulp_at_90100_pa": abs_probe["fp32_ulp_at_total_pa"],
        "perturbation_millipascal_delta_pa": pert_probe["millipascal_recovered_delta_pa"],
        "perturbation_millipascal_relative_error": pert_probe["millipascal_relative_error"],
        "one_column_p_error_ratio_absolute_over_perturbation": recurrence[
            "p_error_ratio_absolute_over_perturbation"
        ],
        "one_column_ph_error_ratio_absolute_over_perturbation": recurrence[
            "ph_error_ratio_absolute_over_perturbation"
        ],
        "shape_only_memory_savings_mib": {
            "core_candidate_set": memory["core_candidate_set"]["saving_mib"],
            "prep_carry_candidate_set": memory["prep_carry_candidate_set"]["saving_mib"],
            "core_plus_prep_candidate_set": memory["core_plus_prep_candidate_set"]["saving_mib"],
        },
        "source_work_verdict": "FP32_SOURCE_WORK_INFEASIBLE_WITH_CURRENT_LOCKS",
        "source_work_blocker": (
            "R0/R1/R2 touch locked runtime, dycore, boundary/restart/init/carry "
            "surfaces while primary debug targets live-nest P/MU/W perturbation-state initialization."
        ),
    }


def _build_results() -> dict[str, Any]:
    collision_map = _candidate_collision_map()
    wdm6 = _wdm6_shape_equivalence()
    fp32 = _load_fp32_probe_summary()
    recommendation = "MERGE_NOW" if wdm6["all_state_leaves_exact"] else "DO_NOT_MERGE"
    return {
        "proof": "v0.14 parallel memory/FP32 manager",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "environment": {
            "cpu_only": True,
            "gpu_used": False,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "numpy_version": np.__version__,
        },
        "git": _git_snapshot(),
        "collision_map": collision_map,
        "memory_fix": wdm6,
        "fp32_refresh": fp32,
        "source_edits": [
            {
                "path": "src/gpuwrf/coupling/scan_adapters.py",
                "change": "WDM6 slmsk uses per-column vector instead of full vertical broadcast",
                "locked_path": False,
                "exact_output_proof": "memory_fix.all_state_leaves_exact",
            }
        ],
        "validation_requirements": {
            "source_edit_minimum": [
                "python -m py_compile src/gpuwrf/coupling/scan_adapters.py proofs/v014/parallel_memory_fp32_manager.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/parallel_memory_fp32_manager.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q tests/test_wdm6_savepoint_parity.py",
                "python -m json.tool proofs/v014/parallel_memory_fp32_manager.json >/tmp/parallel_memory_fp32_manager.validated.json",
                "git diff --check",
                "git diff -- src/gpuwrf",
            ],
            "gpu_validation": "not run; not required for shape-only CPU exact proof and avoided to yield to primary manager",
        },
        "recommendation": recommendation,
        "unresolved_risks": [
            "WDM6 adapter still preserves the pre-existing 1.0/0.0 slmsk values; correcting WRF 1/2 land-sea semantics would be a separate non-bit-identical physics fix.",
            "Exact-branch memory preflight remains a short audit/capped observation, not a full validation campaign.",
            "FP32 acoustic source work remains blocked until the live-nest P/MU/W fp64 grid-parity surface is fixed or released.",
        ],
    }


def _write_markdown(results: dict[str, Any]) -> None:
    memory = results["memory_fix"]
    fp32 = results["fp32_refresh"]
    saved = memory["target_641x321x50_fp64_bytes"]
    lines = [
        "# v0.14 Parallel Memory/FP32 Manager",
        "",
        f"- Verdict: `{results['recommendation']}`",
        f"- Branch: `{results['git']['branch']}`",
        f"- HEAD: `{results['git']['head'][:12]}`",
        f"- CPU-only: `{results['environment']['cpu_only']}`",
        f"- GPU used: `{results['environment']['gpu_used']}`",
        "",
        "## Collision Map",
        "",
        "- Implemented source edit: `src/gpuwrf/coupling/scan_adapters.py` only.",
        "- Locked paths avoided: `src/gpuwrf/dynamics/**`, `runtime/operational_mode.py`, `integration/d02_replay.py`, `nesting/**`, boundary/carry/init/restart/state-contract files.",
        "- Moisture velocity reuse, acoustic carry split, limiter workspace reduction, state aliasing, and all FP32 source work collide with the active fp64 grid-parity lock.",
        "",
        "## Memory Fix",
        "",
        f"- WDM6 `slmsk` old shape: `{memory['legacy_slmsk_shape']}`.",
        f"- WDM6 `slmsk` new shape: `{memory['new_slmsk_shape']}`.",
        f"- Exact old-layout vs new-layout State leaves: `{memory['all_state_leaves_exact']}`.",
        f"- Target 641x321x50 fp64 transient saving: `{saved['saved_mib']:.3f} MiB` (`{saved['saved_gib']:.6f} GiB`).",
        "- Semantics note: this preserves the existing 1.0 land / 0.0 water adapter values; it is not a land-sea physics correction.",
        "",
        "## FP32 Status",
        "",
        f"- Absolute-total fp32 1 mPa recovered delta: `{fp32['absolute_total_millipascal_recurrent_delta_pa']}` Pa.",
        f"- Perturbation-form fp32 1 mPa recovered delta: `{fp32['perturbation_millipascal_delta_pa']}` Pa.",
        f"- One-column pressure error ratio absolute-total32 / perturbation32: `{fp32['one_column_p_error_ratio_absolute_over_perturbation']:.3e}`.",
        f"- One-column geopotential error ratio absolute-total32 / perturbation32: `{fp32['one_column_ph_error_ratio_absolute_over_perturbation']:.3e}`.",
        "- Source verdict: `FP32_SOURCE_WORK_INFEASIBLE_WITH_CURRENT_LOCKS`.",
        "",
        "## Next Gate",
        "",
        "Merge only the WDM6 shape-only cleanup after validation. Reopen FP32 R0/R1 only after the primary manager releases the live-nest P/MU/W perturbation-state initialization lock.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = _build_results()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_markdown(results)
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
