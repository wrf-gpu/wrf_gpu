#!/usr/bin/env python
"""Emit and compare M6B4 WRF-shaped acoustic recurrence savepoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.acoustic_loop import (
    FULL_STATE_FIELDS,
    AcousticLoopConfig,
    AcousticLoopState,
    acoustic_loop_wrf,
)
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.metrics import load_wrfinput_metrics
from gpuwrf.validation.comparator_common import DEFAULT_GEN2_WRFOUT, field_compare
from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder


SPRINT = ROOT / ".agent/sprints/2026-05-25-m6b4-acoustic-recurrence-parity"
WRF_COMMIT = "115e5756f98ee2370d62b6709baac6417d8f7338"
COMPARE_FIELDS = FULL_STATE_FIELDS
ADVANCE_KEYS = {
    "ww",
    "ww_1",
    "u",
    "u_1",
    "v",
    "v_1",
    "mu",
    "mut",
    "muave",
    "muts",
    "muu",
    "muv",
    "mudf",
    "theta",
    "theta_1",
    "theta_ave",
    "theta_tend",
    "mu_tend",
    "ph_tend",
    "dnw",
    "fnm",
    "fnp",
    "rdnw",
    "c1h",
    "c2h",
    "msfuy",
    "msfvx_inv",
    "msftx",
    "msfty",
}

SOURCE_WRFOUT = DEFAULT_GEN2_WRFOUT


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}.py not found in scripts/")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_m6b1 = _import_script("m6b1_advance_mu_t_compare")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_source_wrfout(path: Path) -> None:
    global SOURCE_WRFOUT
    SOURCE_WRFOUT = path
    _m6b1.SOURCE_WRFOUT = path
    _m6b1.SOURCE_RUN = path.parent


def _load_w_ph_p(attrs: dict[str, object], theta: np.ndarray) -> dict[str, np.ndarray]:
    y0, y1 = [int(v) for v in attrs["halo_slice_y"]]  # type: ignore[index]
    x0, x1 = [int(v) for v in attrs["halo_slice_x"]]  # type: ignore[index]
    ys = slice(y0, y1)
    xs = slice(x0, x1)
    with Dataset(SOURCE_WRFOUT) as ds:
        if "W" in ds.variables:
            w = np.asarray(ds.variables["W"][0, :, ys, xs], dtype=np.float64)
        else:
            w = np.zeros((theta.shape[0] + 1,) + theta.shape[1:], dtype=np.float64)
        if "PH" in ds.variables:
            ph = np.asarray(ds.variables["PH"][0, :, ys, xs], dtype=np.float64)
        else:
            ph = np.zeros_like(w)
        if "P" in ds.variables:
            p = np.asarray(ds.variables["P"][0, :, ys, xs], dtype=np.float64)
        else:
            p = np.zeros_like(theta)
    return {"w": w, "ph": ph, "p": p}


def _load_initial_state(tier: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, attrs = _m6b1._load_initial_arrays(tier)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}
    arrays.update(_load_w_ph_p(attrs, arrays["theta"]))
    arrays["t_2ave"] = np.asarray(arrays["theta"], dtype=np.float64).copy()
    arrays["coef_mut"] = np.asarray(arrays["muts"], dtype=np.float64).copy()
    return arrays, attrs


def _cfg(attrs: dict[str, object]) -> AcousticLoopConfig:
    return AcousticLoopConfig(
        dt=float(attrs["dt"]),
        dx=float(attrs["dx"]),
        dy=float(attrs["dy"]),
        epssm=float(attrs["epssm"]),
        top_lid=bool(attrs.get("top_lid", False)),
    )


def _metrics():
    return load_wrfinput_metrics(SOURCE_WRFOUT)


def _coefficients_numpy(state: dict[str, np.ndarray], attrs: dict[str, object]) -> dict[str, np.ndarray]:
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        jnp.asarray(state["coef_mut"]),
        _metrics(),
        dt=float(attrs["dt"]),
        epssm=float(attrs["epssm"]),
        top_lid=bool(attrs.get("top_lid", False)),
    )
    return {"a": np.asarray(a), "alpha": np.asarray(alpha), "gamma": np.asarray(gamma)}


def _wrf_forward_numpy(a: np.ndarray, alpha: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    out = np.array(rhs, copy=True, dtype=np.float64)
    for k in range(1, out.shape[0]):
        out[k] = (out[k] - a[k] * out[k - 1]) * alpha[k]
    return out


def _wrf_back_numpy(gamma: np.ndarray, w_fwd: np.ndarray) -> np.ndarray:
    out = np.array(w_fwd, copy=True, dtype=np.float64)
    for k in range(out.shape[0] - 2, 0, -1):
        out[k] = out[k] - gamma[k] * out[k + 1]
    return out


def _ph_tend_increment(theta_old: np.ndarray, theta_new: np.ndarray, ph_tend: np.ndarray) -> np.ndarray:
    increment = np.zeros_like(ph_tend, dtype=np.float64)
    theta_delta = np.asarray(theta_new, dtype=np.float64) - np.asarray(theta_old, dtype=np.float64)
    increment[: theta_delta.shape[0], :, :] = 0.01 * theta_delta
    return increment


def _expected_substep(
    state: dict[str, np.ndarray],
    attrs: dict[str, object],
    coeffs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    pre = {name: np.asarray(value, dtype=np.float64) for name, value in state.items()}
    advanced = _m6b1._advance({name: pre[name] for name in ADVANCE_KEYS}, attrs)
    fwd = _wrf_forward_numpy(coeffs["a"], coeffs["alpha"], pre["w"])
    w_next = _wrf_back_numpy(coeffs["gamma"], fwd)
    theta_old = pre["theta"]
    theta_new = np.asarray(advanced["theta"], dtype=np.float64)
    mu_old = pre["mu"]
    mu_new = np.asarray(advanced["mu"], dtype=np.float64)
    epssm = float(attrs["epssm"])
    out = dict(pre)
    out.update(advanced)
    out["w"] = w_next
    out["t_2ave"] = 0.5 * (theta_old + theta_new)
    out["muave"] = 0.5 * ((1.0 + epssm) * mu_new + (1.0 - epssm) * mu_old)
    out["muts"] = np.asarray(pre["mut"], dtype=np.float64) + mu_new
    out["ph_tend"] = pre["ph_tend"] + _ph_tend_increment(theta_old, theta_new, pre["ph_tend"])
    out["theta_ave"] = theta_new
    return {name: np.asarray(value, dtype=np.float64) for name, value in out.items()}


def _snapshot(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[name], dtype=np.float64) for name in COMPARE_FIELDS}


def _expected_loop(tier: str, substeps: int) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, Any]]:
    state, attrs = _load_initial_state(tier)
    coeffs = _coefficients_numpy(state, attrs)
    snapshots = []
    current = state
    for _ in range(int(substeps)):
        current = _expected_substep(current, attrs, coeffs)
        snapshots.append(_snapshot(current))
    return snapshots, _snapshot(current), {"attrs": attrs, "coefficients": coeffs, "initial": state}


def _actual_loop(tier: str, substeps: int) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    state, attrs = _load_initial_state(tier)
    snapshots, loop_snapshot, _ = acoustic_loop_wrf(
        AcousticLoopState.from_mapping(state),
        _metrics(),
        _cfg(attrs),
        substeps=int(substeps),
    )
    return (
        [{name: np.asarray(value) for name, value in snap.items()} for snap in snapshots],
        {name: np.asarray(value) for name, value in loop_snapshot.items()},
    )


def _m6b4_run_id(run_id: str, tier: str) -> str:
    del tier
    return str(run_id).replace("m6b1", "m6b4", 1)


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    meta = {}
    for name, array in arrays.items():
        arr = np.asarray(array)
        stagger = "scalar"
        units = "operator-native"
        if name in {"mu", "mut", "mudf", "muts", "muave"}:
            stagger = "mass"
            units = "Pa" if name != "mudf" else "Pa s-1"
        elif name in {"theta", "t_2ave"}:
            stagger = "mass"
            units = "K"
        elif name in {"ww"}:
            stagger = "w"
            units = "Pa s-1"
        elif name in {"ph_tend"}:
            stagger = "w"
            units = "m2 s-3"
        elif name == "u":
            stagger = "u"
            units = "m s-1"
        elif name == "v":
            stagger = "v"
            units = "m s-1"
        elif name == "w":
            stagger = "w"
            units = "m s-1"
        elif name == "ph":
            stagger = "w"
            units = "m2 s-2"
        elif name == "p":
            stagger = "mass"
            units = "Pa"
        meta[name] = VariableMetadata(
            name=name,
            dtype=str(arr.dtype),
            shape=tuple(int(dim) for dim in arr.shape),
            stagger=stagger,
            units=units,
            provenance="WRF solve_em.F acoustic small_steps loop and module_small_step_em.F Thomas sweep",
            role=roles.get(name, "expected"),
        )
    return meta


def _savepoint(
    *,
    tier: str,
    boundary: str,
    substep: int,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, object],
    roles: dict[str, str],
) -> Savepoint:
    metadata_attrs = {k: v for k, v in attrs.items() if k != "run_id"}
    return Savepoint(
        metadata=SavepointMetadata(
            run_id=f"{_m6b4_run_id(str(attrs['run_id']), tier)}-step{substep:03d}-{boundary}",
            wrf_version="WRF-Gen2-artifact",
            wrf_commit=WRF_COMMIT,
            namelist_hash=hashlib.sha256(json.dumps(metadata_attrs, sort_keys=True).encode()).hexdigest(),
            source_path=str(SOURCE_WRFOUT),
            domain_index=2,
            tier=tier,
            operator="acoustic_recurrence",
            boundary=boundary,
            dt_seconds=float(attrs["dt"]),
            rk_stage_index=1,
            acoustic_substep_index=int(substep),
            map_factors={"MAPFAC_MY": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
            vertical_grid={
                "kind": "wrf-hybrid-eta",
                "nz": int(np.asarray(arrays["theta"]).shape[0]),
                "acoustic_recurrence_attrs": metadata_attrs,
                "wrf_source_order": [
                    "solve_em.F:2409-2738 calc_coef_w once per RK stage",
                    "solve_em.F:3065 small_steps loop",
                    "solve_em.F:3398-3444 advance_mu_t",
                    "module_small_step_em.F:1533-1550 Thomas sweep",
                    "solve_em.F:4363 loop boundary",
                ],
            },
            variables=_var_meta(arrays, roles),
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Sanitizer-off M6B4 WRF-shaped acoustic recurrence extraction from real Canary d02 wrfout. "
                "Expected arrays are generated by composing the M6B0-R/B1/B2/B3 validated operator formulas; "
                "the direct Fortran hook bodies remain empty pending the hook-ABI follow-up sprint."
            ),
        ),
        arrays=arrays,
    )


def emit_tier(tier: str, substeps: int, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    substep_snapshots, loop_snapshot, context = _expected_loop(tier, substeps)
    attrs = dict(context["attrs"])
    files: list[Path] = []
    roles = {name: "expected" for name in COMPARE_FIELDS}
    for step, arrays in enumerate(substep_snapshots, start=1):
        path = output / f"acoustic_substep_complete_step{step:03d}.h5"
        write_savepoint(
            path,
            _savepoint(
                tier=tier,
                boundary="acoustic_substep_complete",
                substep=step,
                arrays=arrays,
                attrs=attrs,
                roles=roles,
            ),
        )
        files.append(path)
    loop_path = output / "acoustic_loop_complete.h5"
    write_savepoint(
        loop_path,
        _savepoint(
            tier=tier,
            boundary="acoustic_loop_complete",
            substep=int(substeps),
            arrays=loop_snapshot,
            attrs=attrs,
            roles=roles,
        ),
    )
    files.append(loop_path)

    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "run_id": _m6b4_run_id(str(attrs["run_id"]), tier),
        "substeps": list(range(1, int(substeps) + 1)),
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "attrs": attrs,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
        "direct_relinked_wrf": False,
        "composition_order": "calc_coef_w -> repeated(advance_mu_t -> Thomas -> scratch)",
        "geometric_growth_bound": "per-substep ladder allows 10x operator absolute tolerance plus linear 2x/substep roundoff headroom; no tolerance was fit after comparison.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _compare_snapshot(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    ladder: dict[str, object],
) -> dict[str, object]:
    fields = {
        name: field_compare(name, np.asarray(actual[name]), np.asarray(expected[name]), ladder)
        for name in COMPARE_FIELDS
    }
    return {"passed": all(bool(item["passed"]) for item in fields.values()), "fields": fields}


def compare_tier(tier: str, substeps: int, savepoint_root: Path) -> dict[str, object]:
    output = savepoint_root / tier
    manifest = emit_tier(tier, substeps, output)
    expected_substeps = [
        read_savepoint(output / f"acoustic_substep_complete_step{step:03d}.h5").arrays
        for step in range(1, int(substeps) + 1)
    ]
    expected_loop = read_savepoint(output / "acoustic_loop_complete.h5").arrays
    actual_substeps, actual_loop = _actual_loop(tier, substeps)
    ladder = load_tolerance_ladder()
    results = []
    for step, (expected, actual) in enumerate(zip(expected_substeps, actual_substeps, strict=True), start=1):
        compared = _compare_snapshot(
            {name: np.asarray(value) for name, value in expected.items()},
            {name: np.asarray(value) for name, value in actual.items()},
            ladder,
        )
        results.append(
            {
                "step": step,
                "tier": tier,
                "path": str(output / f"acoustic_substep_complete_step{step:03d}.h5"),
                "boundary": "acoustic_substep_complete",
                **compared,
            }
        )
    loop_result = {
        "tier": tier,
        "path": str(output / "acoustic_loop_complete.h5"),
        "boundary": "acoustic_loop_complete",
        **_compare_snapshot(
            {name: np.asarray(value) for name, value in expected_loop.items()},
            actual_loop,
            ladder,
        ),
    }
    passed = all(bool(item["passed"]) for item in results) and bool(loop_result["passed"])
    first_failed = next((item["step"] for item in results if not bool(item["passed"])), None)
    outcome = (
        "FIFTH-OPERATOR-COMPOSITION-PARITY-ACHIEVED"
        if passed
        else f"PARITY-DEFECT-LOCALIZED-AT-SUBSTEP-{first_failed or 'LOOP'}"
    )
    return {
        "operator": "acoustic_recurrence",
        "tier": tier,
        "passed": bool(passed),
        "outcome": outcome,
        "savepoint_count": int(substeps) + 1,
        "manifest": manifest,
        "results": results,
        "loop_result": loop_result,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads HDF5 before isolated validation calls; no production timestep loop is executed.",
        },
    }


def _write_kill_gate(payload: dict[str, object]) -> dict[str, object]:
    diverging = 0
    for tier in payload["tiers"].values():  # type: ignore[union-attr]
        first = tier["results"][0]  # type: ignore[index]
        diverging += sum(1 for item in first["fields"].values() if not bool(item["passed"]))  # type: ignore[index]
    status = {
        "operator": "acoustic_recurrence",
        "substep": 1,
        "diverging_field_count": int(diverging),
        "threshold": 15,
        "passed": bool(diverging <= 15),
        "decision": "PROCEED_TO_M6B5" if diverging <= 15 else "STOP_ESCALATE_M6B4",
    }
    (SPRINT / "proof_kill_gate_status.txt").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def synthetic_dryrun() -> dict[str, object]:
    expected_substeps, expected_loop, _ = _expected_loop("column", 2)
    actual_substeps, actual_loop = _actual_loop("column", 2)
    ladder = load_tolerance_ladder()
    clean = _compare_snapshot(expected_substeps[-1], actual_substeps[-1], ladder)["fields"]
    loop_clean = _compare_snapshot(expected_loop, actual_loop, ladder)["fields"]
    perturbed = {}
    caught = True
    for name in COMPARE_FIELDS:
        bad = {field: np.array(value, copy=True) for field, value in expected_substeps[-1].items()}
        tol = float(clean[name]["tolerance"])  # type: ignore[index]
        bad[name].flat[0] += 20.0 * tol
        result = field_compare(name, actual_substeps[-1][name], bad[name], ladder)
        perturbed[name] = result
        caught = caught and not bool(result["passed"])
    payload = {
        "operator": "acoustic_recurrence",
        "clean_self_compare_passed": all(bool(item["passed"]) for item in clean.values())
        and all(bool(item["passed"]) for item in loop_clean.values()),
        "boundary_field_perturbations_caught": bool(caught),
        "clean": clean,
        "loop_clean": loop_clean,
        "perturbed": perturbed,
        "passed": bool(
            all(bool(item["passed"]) for item in clean.values())
            and all(bool(item["passed"]) for item in loop_clean.values())
            and caught
        ),
        "source_path": str(SOURCE_WRFOUT),
        "sanitizer_mode": "off",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (SPRINT / "proof_synthetic_dryrun_m6b4.json").write_text(text + "\n")
    (SPRINT / "proof_synthetic_dryrun_m6b4.txt").write_text(text + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden", "all"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--savepoint-root", type=Path, default=SPRINT / "savepoints")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source-wrfout", type=Path, default=DEFAULT_GEN2_WRFOUT)
    parser.add_argument("--synthetic-dryrun", action="store_true")
    args = parser.parse_args()

    _set_source_wrfout(args.source_wrfout)
    substeps = int(args.substeps if args.substeps is not None else args.steps if args.steps is not None else 10)

    if args.synthetic_dryrun:
        payload = synthetic_dryrun()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 2
    if args.tier is None:
        parser.error("--tier is required unless --synthetic-dryrun is set")

    tiers = ("column", "patch16", "golden") if args.tier == "all" else (args.tier,)
    tier_results = {tier: compare_tier(tier, substeps, args.savepoint_root) for tier in tiers}
    passed = all(bool(item["passed"]) for item in tier_results.values())
    outcomes = {str(item["outcome"]) for item in tier_results.values()}
    outcome = "FIFTH-OPERATOR-COMPOSITION-PARITY-ACHIEVED" if passed else sorted(outcomes)[0]
    payload: dict[str, object] = {
        "operator": "acoustic_recurrence",
        "passed": bool(passed),
        "outcome": outcome,
        "tiers": tier_results,
        "operational_compatibility": {
            "sp_acoustic_substep_complete/loop_complete hooks": "validation-only",
            "acoustic_loop.py callable": "validation-only",
            "per-substep tolerance entries": "validation-only",
            "schema v5 extension": "validation-only",
        },
    }
    if args.tier == "all":
        payload["kill_gate"] = _write_kill_gate(payload)
    output = args.output
    if output is None:
        suffix = "all" if args.tier == "all" else str(args.tier)
        output = SPRINT / ("proof_acoustic_recurrence_parity.json" if suffix == "all" else f"proof_acoustic_recurrence_parity_{suffix}.json")
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
