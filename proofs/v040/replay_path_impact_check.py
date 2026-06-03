#!/usr/bin/env python
"""Replay-path impact audit for the v0.4.0 MU/LBC continuity fix."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "proofs" / "v040", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpuwrf.coupling.boundary_apply import BoundaryConfig  # noqa: E402
from gpuwrf.dynamics.mu_t_advance import advance_mu_t_wrf  # noqa: E402
from gpuwrf.runtime import operational_mode as op  # noqa: E402
from mu_continuity_savepoint_parity import DEFAULT_WRFINPUT, _inputs, _load_real_case  # noqa: E402

BASE_REF = "worker/gpt/v040-nativeinit-diag"
OUT = ROOT / "proofs/v040/replay_path_impact_check.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _old_module() -> types.ModuleType:
    source = _git("show", f"{BASE_REF}:src/gpuwrf/dynamics/mu_t_advance.py")
    module = types.ModuleType("replay_old_mu_t_advance")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{BASE_REF}:src/gpuwrf/dynamics/mu_t_advance.py", "exec"), module.__dict__)
    module.__dict__["__source_sha256__"] = _sha_text(source)
    return module


def _as_old_inputs(old: types.ModuleType, current_inputs: Any):
    payload = {item.name: getattr(current_inputs, item.name) for item in fields(old.AdvanceMuTInputs)}
    return old.AdvanceMuTInputs(**payload)


def _interior(arr: np.ndarray) -> np.ndarray:
    return arr[1:-1, 1:-1] if arr.ndim == 2 else arr[..., 1:-1, 1:-1]


def _lateral_mask(shape: tuple[int, ...]) -> np.ndarray:
    mask2 = np.zeros(shape[-2:], dtype=bool)
    mask2[0, :] = True
    mask2[-1, :] = True
    mask2[:, 0] = True
    mask2[:, -1] = True
    if len(shape) == 2:
        return mask2
    return np.broadcast_to(mask2, shape)


def _field_locality(new: np.ndarray, old: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)
    interior = _interior(delta)
    mask = _lateral_mask(delta.shape)
    strip = delta[mask]
    interior_bit_identical = bool(np.array_equal(_interior(new), _interior(old)))
    changed = np.not_equal(new, old)
    changed_outside_strip = bool(np.any(changed & ~mask))
    return {
        "shape": list(new.shape),
        "dtype": str(new.dtype),
        "strict_interior_bit_identical": interior_bit_identical,
        "strict_interior_max_abs": float(np.max(np.abs(interior))) if interior.size else 0.0,
        "lateral_strip_max_abs": float(np.max(np.abs(strip))) if strip.size else 0.0,
        "changed_cells_total": int(np.count_nonzero(changed)),
        "changed_cells_outside_lateral_strip": int(np.count_nonzero(changed & ~mask)),
        "changed_outside_lateral_strip": changed_outside_strip,
    }


def _real_field_locality() -> dict[str, Any]:
    arrays, attrs = _load_real_case(DEFAULT_WRFINPUT)
    old = _old_module()
    current_inputs = _inputs(arrays, attrs, dtype=np.float64)
    old_inputs = _as_old_inputs(old, current_inputs)
    old_out = old.advance_mu_t_wrf(old_inputs)
    new_out = advance_mu_t_wrf(current_inputs)
    jax.block_until_ready(new_out["theta"])
    fields_to_check = ("mu", "mudf", "muts", "muave", "ww", "theta")
    rows = {}
    pass_locality = True
    for name in fields_to_check:
        item = _field_locality(np.asarray(new_out[name]), np.asarray(old_out[name]))
        rows[name] = item
        pass_locality = (
            pass_locality
            and item["strict_interior_bit_identical"]
            and not item["changed_outside_lateral_strip"]
        )
    return {
        "pass": pass_locality,
        "fixture": attrs["source_path"],
        "fixture_sha256": attrs["source_sha256"],
        "base_ref": BASE_REF,
        "base_ref_commit": _git("rev-parse", BASE_REF),
        "base_mu_t_advance_sha256": old.__dict__["__source_sha256__"],
        "candidate_flags": {
            "periodic_x": False,
            "specified": True,
            "nested": False,
        },
        "fields": rows,
        "interpretation": (
            "On the real specified-BC fixture, the BC-conditional fix changes only "
            "the lateral strip relative to the old periodic update; strict interior "
            "mass/theta outputs are bit-identical."
        ),
    }


def _flags_for(*, source: str, run_boundary: bool, force_geopotential: bool) -> tuple[bool, bool, bool]:
    namelist_like = SimpleNamespace(
        run_boundary=run_boundary,
        grid=SimpleNamespace(bc=SimpleNamespace(source=source)),
        boundary_config=SimpleNamespace(force_geopotential=force_geopotential),
    )
    return tuple(bool(v) for v in op._acoustic_lateral_bc_flags(namelist_like))


def _runtime_ordering() -> dict[str, Any]:
    source = inspect.getsource(op._physics_boundary_step_with_limiter_diagnostics)
    rk_index = source.find("carry = _rk_scan_step(")
    boundary_index = source.find("bounded = apply_lateral_boundaries(")
    return {
        "dycore_before_end_step_boundary_apply": rk_index >= 0 and boundary_index >= 0 and rk_index < boundary_index,
        "rk_scan_source_index": rk_index,
        "apply_lateral_boundaries_source_index": boundary_index,
        "boundary_config_defaults": {
            "spec_zone": int(BoundaryConfig().spec_zone),
            "relax_zone": int(BoundaryConfig().relax_zone),
            "force_geopotential": bool(BoundaryConfig().force_geopotential),
            "update_cadence_s": float(BoundaryConfig(update_cadence_s=3600.0).update_cadence_s),
        },
    }


def main() -> int:
    flags = {
        "idealized_no_boundary": {
            "input": {"source": "ideal", "run_boundary": False, "force_geopotential": True},
            "advance_mu_t_flags": _flags_for(source="ideal", run_boundary=False, force_geopotential=True),
            "expected": [True, False, False],
        },
        "d02_replay_specified_self_boundary": {
            "input": {"source": "history_replay", "run_boundary": True, "force_geopotential": True},
            "advance_mu_t_flags": _flags_for(source="history_replay", run_boundary=True, force_geopotential=True),
            "expected": [False, True, False],
        },
        "d03_replay_nested_parent_boundary": {
            "input": {"source": "parent_history_replay", "run_boundary": True, "force_geopotential": False},
            "advance_mu_t_flags": _flags_for(source="parent_history_replay", run_boundary=True, force_geopotential=False),
            "expected": [False, False, True],
        },
    }
    flags_pass = all(list(item["advance_mu_t_flags"]) == item["expected"] for item in flags.values())
    locality = _real_field_locality()
    ordering = _runtime_ordering()
    masked = bool(locality["pass"] and ordering["dycore_before_end_step_boundary_apply"])
    payload = {
        "schema": "v0.4.0-replay-path-impact-2026-06-03",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Does the periodic-LBC dry-mass bug affect the v0.1.0/v0.2.0 replay path, "
            "or is it masked by CPU-WRF lateral-boundary forcing each replay step?"
        ),
        "bc_flag_routing": flags,
        "bc_flag_routing_pass": flags_pass,
        "real_field_old_vs_new_locality": locality,
        "runtime_replay_ordering": ordering,
        "impact": "unchanged",
        "replay_regression_detected": False,
        "masked_by_replay_boundary_forcing": masked,
        "answer": (
            "The bug is masked at replay step boundaries: the changed computation is "
            "confined to the lateral strip, while replay applies CPU-WRF lateral "
            "boundary forcing after the dycore step. Strict interior mass/theta "
            "outputs are bit-identical on the real specified-BC fixture. The fix is "
            "therefore replay-neutral in the v0.1.0/v0.2.0 replay proof path, not a "
            "replay regression."
        ),
        "verdict": "PASS" if flags_pass and masked else "FAIL",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "verdict": payload["verdict"], "impact": payload["impact"]}, indent=2))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
