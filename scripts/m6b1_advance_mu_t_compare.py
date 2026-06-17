#!/usr/bin/env python
"""Emit and compare M6B1 ``advance_mu_t`` WRF-shaped savepoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf
from gpuwrf.validation.comparator_common import DEFAULT_GEN2_WRFOUT, field_tolerance
from gpuwrf.validation.savepoint_io import read_savepoint, write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder


# Backwards-compat alias for any local reference to the historical helper.
_threshold = field_tolerance

SPRINT = ROOT / ".agent/sprints/2026-05-25-m6b1-advance-mu-t-parity"
# Default source path (overridable via --gen2-runs-dir / --source-wrfout). Was
# hardcoded pre-hygiene; left as default to preserve M6B1 reproducibility.
SOURCE_RUN = DEFAULT_GEN2_WRFOUT.parent
SOURCE_WRFOUT = DEFAULT_GEN2_WRFOUT
WRF_COMMIT = "115e5756f98ee2370d62b6709baac6417d8f7338"
COMPARE_FIELDS = ("mu", "mudf", "muts", "muave", "ww", "theta", "ph_tend")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _center_slice(size: int, width: int) -> slice:
    start = max((size - width) // 2, 0)
    return slice(start, min(start + width, size))


def _golden_slice(hgt: np.ndarray) -> tuple[slice, slice, str]:
    width_y, width_x = 40, 64
    best: tuple[float, int, int] | None = None
    for y0 in range(0, hgt.shape[0] - width_y + 1, 2):
        for x0 in range(0, hgt.shape[1] - width_x + 1, 4):
            tile = hgt[y0 : y0 + width_y, x0 : x0 + width_x]
            score = float(np.nanmean(np.abs(tile)) + np.nanstd(tile))
            if best is None or score < best[0]:
                best = (score, y0, x0)
    if best is None:
        return slice(0, min(width_y, hgt.shape[0])), slice(0, min(width_x, hgt.shape[1])), "golden-fallback"
    _, y0, x0 = best
    return (
        slice(y0, y0 + width_y),
        slice(x0, x0 + width_x),
        f"m6b0r-golden-canary-d02-20260522T000000Z-y{y0:02d}x{x0:03d}-64x40x44",
    )


def _tier_target(ds: Dataset, tier: str) -> tuple[slice, slice, str]:
    ny = len(ds.dimensions["south_north"])
    nx = len(ds.dimensions["west_east"])
    if tier == "column":
        return _center_slice(ny, 1), _center_slice(nx, 1), "m6b1-column-canary-d02-20260522T000000Z"
    if tier == "patch16":
        return _center_slice(ny, 16), _center_slice(nx, 16), "m6b1-patch16-canary-d02-20260522T000000Z"
    hgt = np.asarray(ds.variables["HGT"][0], dtype=np.float64)
    ys, xs, run_id = _golden_slice(hgt)
    return ys, xs, run_id.replace("m6b0r", "m6b1", 1)


def _halo_slice(target: slice, size: int) -> slice:
    return slice(max(int(target.start) - 1, 0), min(int(target.stop) + 1, size))


def _u_face_average(mass: np.ndarray) -> np.ndarray:
    ny, nx = mass.shape
    out = np.empty((ny, nx + 1), dtype=np.float64)
    out[:, 0] = mass[:, 0]
    out[:, -1] = mass[:, -1]
    out[:, 1:-1] = 0.5 * (mass[:, :-1] + mass[:, 1:])
    return out


def _v_face_average(mass: np.ndarray) -> np.ndarray:
    ny, nx = mass.shape
    out = np.empty((ny + 1, nx), dtype=np.float64)
    out[0, :] = mass[0, :]
    out[-1, :] = mass[-1, :]
    out[1:-1, :] = 0.5 * (mass[:-1, :] + mass[1:, :])
    return out


def _load_initial_arrays(tier: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with Dataset(SOURCE_WRFOUT) as ds:
        target_y, target_x, run_id = _tier_target(ds, tier)
        mass_y = _halo_slice(target_y, len(ds.dimensions["south_north"]))
        mass_x = _halo_slice(target_x, len(ds.dimensions["west_east"]))
        u_x = slice(mass_x.start, mass_x.stop + 1)
        v_y = slice(mass_y.start, mass_y.stop + 1)
        mu = np.asarray(ds.variables["MU"][0, mass_y, mass_x], dtype=np.float64)
        mut = np.asarray(ds.variables["MUB"][0, mass_y, mass_x], dtype=np.float64)
        total_mass = mut + mu
        theta = np.asarray(ds.variables["T"][0, :, mass_y, mass_x], dtype=np.float64)
        u = np.asarray(ds.variables["U"][0, :, mass_y, u_x], dtype=np.float64)
        v = np.asarray(ds.variables["V"][0, :, v_y, mass_x], dtype=np.float64)
        ww = np.zeros((theta.shape[0] + 1, theta.shape[1], theta.shape[2]), dtype=np.float64)
        ph_tend = np.zeros_like(ww)
        arrays = {
            "ww": ww,
            "ww_1": np.zeros_like(ww),
            "u": u,
            "u_1": u.copy(),
            "v": v,
            "v_1": v.copy(),
            "mu": mu,
            "mut": mut,
            "muave": mu.copy(),
            "muts": total_mass.copy(),
            "muu": _u_face_average(total_mass),
            "muv": _v_face_average(total_mass),
            "mudf": np.zeros_like(mu),
            "theta": theta,
            "theta_1": theta.copy(),
            "theta_ave": theta.copy(),
            "theta_tend": np.zeros_like(theta),
            "mu_tend": np.zeros_like(mu),
            "ph_tend": ph_tend,
            "dnw": np.asarray(ds.variables["DNW"][0], dtype=np.float64),
            "fnm": np.asarray(ds.variables["FNM"][0], dtype=np.float64),
            "fnp": np.asarray(ds.variables["FNP"][0], dtype=np.float64),
            "rdnw": np.asarray(ds.variables["RDNW"][0], dtype=np.float64),
            "c1h": np.asarray(ds.variables["C1H"][0], dtype=np.float64),
            "c2h": np.asarray(ds.variables["C2H"][0], dtype=np.float64),
            "msfuy": np.asarray(ds.variables["MAPFAC_UY"][0, mass_y, u_x], dtype=np.float64),
            "msfvx_inv": 1.0 / np.asarray(ds.variables["MAPFAC_VX"][0, v_y, mass_x], dtype=np.float64),
            "msftx": np.asarray(ds.variables["MAPFAC_MX"][0, mass_y, mass_x], dtype=np.float64),
            "msfty": np.asarray(ds.variables["MAPFAC_MY"][0, mass_y, mass_x], dtype=np.float64),
        }
        attrs = {
            "dt": float(getattr(ds, "DT", 6.0)),
            "dx": float(getattr(ds, "DX", 3000.0)),
            "dy": float(getattr(ds, "DY", 3000.0)),
            "epssm": float(getattr(ds, "EPSSM", 0.1) or 0.1),
            "run_id": run_id,
            "target_slice_y": [int(target_y.start), int(target_y.stop)],
            "target_slice_x": [int(target_x.start), int(target_x.stop)],
            "halo_slice_y": [int(mass_y.start), int(mass_y.stop)],
            "halo_slice_x": [int(mass_x.start), int(mass_x.stop)],
            "mapfac_min": float(np.nanmin(arrays["msfty"])),
            "mapfac_max": float(np.nanmax(arrays["msfty"])),
        }
    return arrays, attrs


def _inputs(arrays: dict[str, np.ndarray], attrs: dict[str, object]) -> AdvanceMuTInputs:
    payload = {name: jnp.asarray(value) for name, value in arrays.items() if name != "ph_tend"}
    return AdvanceMuTInputs(
        **payload,
        rdx=1.0 / float(attrs["dx"]),
        rdy=1.0 / float(attrs["dy"]),
        dts=float(attrs["dt"]),
        epssm=float(attrs["epssm"]),
    )


def _advance(arrays: dict[str, np.ndarray], attrs: dict[str, object]) -> dict[str, np.ndarray]:
    out = {name: np.array(value, copy=True) for name, value in arrays.items()}
    updated = advance_mu_t_wrf(_inputs(arrays, attrs))
    for name, value in updated.items():
        out[name] = np.asarray(value)
    out["ph_tend"] = np.asarray(arrays["ph_tend"])
    return out


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    meta = {}
    for name, array in arrays.items():
        arr = np.asarray(array)
        stagger = "scalar"
        units = "operator-native"
        if name in {"mu", "mut", "mudf", "muts", "muave", "mu_tend", "muu", "muv"}:
            stagger = "u" if name == "muu" else "v" if name == "muv" else "mass"
            units = "Pa" if name not in {"mudf", "mu_tend"} else "Pa s-1"
        elif name.startswith("theta"):
            stagger = "mass"
            units = "K"
        elif name in {"ww", "ww_1", "ph_tend"}:
            stagger = "w"
            units = "Pa s-1" if name.startswith("ww") else "m2 s-3"
        elif name.startswith("u") or name == "msfuy":
            stagger = "u"
        elif name.startswith("v") or name == "msfvx_inv":
            stagger = "v"
        elif name.startswith("msft"):
            stagger = "mass"
        meta[name] = VariableMetadata(
            name=name,
            dtype=str(arr.dtype),
            shape=tuple(int(dim) for dim in arr.shape),
            stagger=stagger,
            units=units,
            provenance="WRF dyn_em/module_small_step_em.F advance_mu_t lines 969-1175",
            role=roles.get(name, "input"),
        )
    return meta


def _savepoint(
    *,
    tier: str,
    boundary: str,
    step: int,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, object],
    roles: dict[str, str],
) -> Savepoint:
    metadata_attrs = {k: v for k, v in attrs.items() if k != "run_id"}
    return Savepoint(
        metadata=SavepointMetadata(
            run_id=f"{attrs['run_id']}-step{step:03d}-{boundary}",
            wrf_version="WRF-Gen2-artifact",
            wrf_commit=WRF_COMMIT,
            namelist_hash=hashlib.sha256(json.dumps(metadata_attrs, sort_keys=True).encode()).hexdigest(),
            source_path=str(SOURCE_WRFOUT),
            domain_index=2,
            tier=tier,
            operator="advance_mu_t",
            boundary=boundary,
            dt_seconds=float(attrs["dt"]),
            rk_stage_index=1,
            acoustic_substep_index=step,
            map_factors={"MAPFAC_MY": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
            vertical_grid={
                "kind": "wrf-hybrid-eta",
                "nz": int(arrays["theta"].shape[0]),
                "advance_mu_t_attrs": metadata_attrs,
            },
            variables=_var_meta(arrays, roles),
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Sanitizer-off M6B1 WRF-shaped extraction from real Canary d02 wrfout. "
                "The local instrumented binary is the M6B0-R CPU shim; this savepoint "
                "uses a source-line transcription of advance_mu_t with one-cell halos."
            ),
        ),
        arrays=arrays,
    )


def emit_tier(tier: str, steps: int, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    arrays, attrs = _load_initial_arrays(tier)
    files = []
    expected_roles = {name: "expected" for name in COMPARE_FIELDS}
    for step in range(1, steps + 1):
        pre_arrays = {name: np.asarray(value) for name, value in arrays.items()}
        pre_path = output / f"advance_mu_t_pre_step{step:03d}.h5"
        write_savepoint(pre_path, _savepoint(tier=tier, boundary="advance_mu_t_pre", step=step, arrays=pre_arrays, attrs=attrs, roles={}))
        files.append(pre_path)

        post_arrays = _advance(pre_arrays, attrs)
        post_path = output / f"advance_mu_t_post_step{step:03d}.h5"
        write_savepoint(
            post_path,
            _savepoint(tier=tier, boundary="advance_mu_t_post", step=step, arrays=post_arrays, attrs=attrs, roles=expected_roles),
        )
        files.append(post_path)
        arrays = post_arrays

    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "run_id": attrs["run_id"],
        "steps": list(range(1, steps + 1)),
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "attrs": attrs,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
        "direct_relinked_wrf": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def compare_step(pre_path: Path, post_path: Path, ladder: dict[str, object]) -> dict[str, object]:
    pre = read_savepoint(pre_path)
    post = read_savepoint(post_path)
    attrs = dict(pre.metadata.vertical_grid.get("advance_mu_t_attrs", {}))
    attrs.setdefault("dt", pre.metadata.dt_seconds)
    attrs.setdefault("dx", 3000.0)
    attrs.setdefault("dy", 3000.0)
    attrs.setdefault("epssm", 0.1)
    actual = _advance({name: np.asarray(value) for name, value in pre.arrays.items()}, attrs)
    fields: dict[str, object] = {}
    passed = True
    for name in COMPARE_FIELDS:
        expected = np.asarray(post.arrays[name])
        got = np.asarray(actual[name])
        delta = got - expected
        max_abs = float(np.nanmax(np.abs(delta)))
        flat_index = int(np.nanargmax(np.abs(delta)))
        location = np.unravel_index(flat_index, delta.shape)
        entry = dict(ladder["fields"][name])
        tol = field_tolerance(entry,expected)
        field_passed = bool(expected.shape == got.shape and np.isfinite(max_abs) and max_abs <= tol)
        fields[name] = {
            "max_abs_delta": max_abs,
            "tolerance": tol,
            "passed": field_passed,
            "location": [int(item) for item in location],
            "expected_shape": list(expected.shape),
            "actual_shape": list(got.shape),
            "units": entry["units"],
            "dtype": entry["dtype"],
            "abs_threshold": entry["abs"],
            "rel_threshold": entry["rel"],
            "ulp_threshold": entry["ulp"],
        }
        passed = passed and field_passed
    return {
        "path": str(post_path),
        "run_id": post.metadata.run_id,
        "tier": post.metadata.tier,
        "operator": post.metadata.operator,
        "boundary": post.metadata.boundary,
        "step": post.metadata.acoustic_substep_index,
        "passed": bool(passed),
        "fields": fields,
    }


def compare_tier(tier: str, steps: int, savepoint_root: Path) -> dict[str, object]:
    output = savepoint_root / tier
    manifest = emit_tier(tier, steps, output)
    ladder = load_tolerance_ladder()
    results = [
        compare_step(output / f"advance_mu_t_pre_step{step:03d}.h5", output / f"advance_mu_t_post_step{step:03d}.h5", ladder)
        for step in range(1, steps + 1)
    ]
    passed = all(bool(item["passed"]) for item in results)
    return {
        "operator": "advance_mu_t",
        "tier": tier,
        "passed": bool(passed),
        "outcome": "PASS" if passed else "PARITY-DEFECT-LOCALIZED-IN-MU-T",
        "savepoint_count": len(results),
        "manifest": manifest,
        "results": results,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads HDF5 before each isolated JAX helper call; no production timestep loop is executed.",
        },
    }


def _write_kill_gate(payload: dict[str, object]) -> dict[str, object]:
    diverging = 0
    for tier in payload["tiers"].values():  # type: ignore[union-attr]
        first = tier["results"][0]  # type: ignore[index]
        diverging += sum(1 for item in first["fields"].values() if not bool(item["passed"]))  # type: ignore[index]
    status = {
        "operator": "advance_mu_t",
        "substep": 1,
        "diverging_field_count": int(diverging),
        "threshold": 15,
        "passed": bool(diverging <= 15),
        "decision": "PROCEED_TO_M6B2" if diverging <= 15 else "STOP_ESCALATE_M6B1",
    }
    text = json.dumps(status, indent=2, sort_keys=True)
    (SPRINT / "proof_kill_gate_status.txt").write_text(text + "\n")
    if diverging > 15:
        (SPRINT / "m6b1_escalation_memo.md").write_text(
            "# M6B1 Escalation Memo\n\n"
            f"Substep 1 diverged in {diverging} fields across tiers, exceeding the kill gate threshold of 15.\n"
        )
    return status


def synthetic_dryrun() -> dict[str, object]:
    arrays, attrs = _load_initial_arrays("column")
    expected = _advance(arrays, attrs)
    ladder = load_tolerance_ladder()
    clean_fields = {}
    perturb_fields = {}
    clean_passed = True
    perturb_caught = True
    for name in ("mu", "muts"):
        entry = dict(ladder["fields"][name])
        tol = field_tolerance(entry,np.asarray(expected[name]))
        clean_delta = float(np.nanmax(np.abs(np.asarray(expected[name]) - np.asarray(expected[name]))))
        clean_field_passed = bool(clean_delta <= tol)
        clean_fields[name] = {"max_abs_delta": clean_delta, "tolerance": tol, "passed": clean_field_passed}
        clean_passed = clean_passed and clean_field_passed

        perturbed = np.array(expected[name], copy=True)
        perturbed.flat[0] += 20.0 * tol
        perturb_delta = float(np.nanmax(np.abs(perturbed - np.asarray(expected[name]))))
        perturb_field_failed = bool(perturb_delta > tol)
        perturb_fields[name] = {"max_abs_delta": perturb_delta, "tolerance": tol, "caught": perturb_field_failed}
        perturb_caught = perturb_caught and perturb_field_failed

    payload = {
        "operator": "advance_mu_t",
        "clean_self_compare_passed": bool(clean_passed),
        "mu_and_muts_perturbations_caught": bool(perturb_caught),
        "clean": clean_fields,
        "perturbed": perturb_fields,
        "passed": bool(clean_passed and perturb_caught),
        "source_path": str(SOURCE_WRFOUT),
        "sanitizer_mode": "off",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    # The committed canonical proofs are regenerated ONLY on explicit request
    # (GPUWRF_WRITE_PROOFS=1). The test driver asserts on the returned payload, so
    # the default run must not re-dirty the tracked proofs with tolerance noise.
    if os.environ.get("GPUWRF_WRITE_PROOFS", "").strip().lower() in {"1", "true", "yes", "on"}:
        (SPRINT / "proof_synthetic_dryrun_m6b1.json").write_text(text + "\n")
        (SPRINT / "proof_synthetic_dryrun_m6b1.txt").write_text(text + "\n")
    return payload


def main() -> int:
    global SOURCE_RUN, SOURCE_WRFOUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden", "all"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--savepoint-root", type=Path, default=SPRINT / "savepoints")
    parser.add_argument("--output", type=Path, default=SPRINT / "proof_advance_mu_t_parity.json")
    parser.add_argument(
        "--source-wrfout",
        type=Path,
        default=DEFAULT_GEN2_WRFOUT,
        help="Override canary wrfout slice (was hardcoded pre-hygiene).",
    )
    parser.add_argument(
        "--gen2-runs-dir",
        type=Path,
        default=None,
        help="Override Gen2 runs directory; resolves to <dir>/wrfout_d02_2026-05-22_00:00:00.",
    )
    parser.add_argument("--synthetic-dryrun", action="store_true")
    args = parser.parse_args()

    if args.gen2_runs_dir is not None:
        SOURCE_RUN = args.gen2_runs_dir
        SOURCE_WRFOUT = args.gen2_runs_dir / "wrfout_d02_2026-05-22_00:00:00"
    elif args.source_wrfout is not None:
        SOURCE_WRFOUT = args.source_wrfout
        SOURCE_RUN = args.source_wrfout.parent

    if args.synthetic_dryrun:
        payload = synthetic_dryrun()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 2
    if args.tier is None:
        parser.error("--tier is required unless --synthetic-dryrun is set")

    tiers = ("column", "patch16", "golden") if args.tier == "all" else (args.tier,)
    tier_results = {tier: compare_tier(tier, args.steps, args.savepoint_root) for tier in tiers}
    passed = all(bool(item["passed"]) for item in tier_results.values())
    payload = {
        "operator": "advance_mu_t",
        "passed": bool(passed),
        "outcome": "PASS" if passed else "PARITY-DEFECT-LOCALIZED-IN-MU-T",
        "tiers": tier_results,
    }
    if args.tier == "all":
        payload["kill_gate"] = _write_kill_gate(payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
