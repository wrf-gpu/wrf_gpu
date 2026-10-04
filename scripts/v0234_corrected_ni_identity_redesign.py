#!/usr/bin/env python3
"""Offline, non-observing causal redesign for corrected-d03 step 9314.

This program deliberately imports neither JAX nor gpuwrf.  It authenticates the
retained ordinary carries and observer-failure proofs, reads only the ordinary
carry arrays, checks the exact nonfinite geometry, and evaluates small NumPy
algebraic oracles against source text from the launch commit and pristine WRF.

The phase-tap summaries are fail-closed: only the ``DISCARDED_UNREAD`` marker and
complete-carry identity metadata are inspected.  No model executable is built or
called by this program.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-13-v0234-corrected-ni-phase-tap"
RCA_SPRINT = ROOT / ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max"
BISECTION_SPRINT = ROOT / ".agent/sprints/2026-07-13-v0234-corrected-ni-ordinary-bisection"

LAUNCH_COMMIT = "818975804c34edb5989969513ee07f7cc7fc392e"
LAUNCH_PARENT = "5baf399708fa318ca66f0a758302957e87007573"
LAUNCH_TREE = "a2cd2d26370e9b4a2fa2c7ac150da77312c2b563"

RECORDER_PROOF = RCA_SPRINT / "recorder-proof.json"
RECORDER_PROOF_SHA256 = "cd0ca04df4f96e1f896c97fde2a68dd5d7b97dcaa84c07e2c5baa4209046cbc5"
ORDINARY_PROOF = BISECTION_SPRINT / "ordinary-bisection-proof.json"
ORDINARY_PROOF_SHA256 = "237e6e59e3f080a28434b72e8e149eb85f2d032139c7f8d4524caee04ff5087b"
PHASE_PROOF = SPRINT / "phase-tap-identity-failure-proof.json"
PHASE_PROOF_SHA256 = "5ea4ed1faf0f795ab72c0396a12bcf85bc37ee60f1b23af57f204fa2f18faa95"

WORK_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
CHECKPOINT_DIR = WORK_ROOT / "ordinary_bisection_effe1f43/checkpoints"
STEP_9313 = CHECKPOINT_DIR / "last-finite-input-to-first-bad-d03-step-9313.pkl"
STEP_9314 = CHECKPOINT_DIR / "first-bad-output-d03-step-9314.pkl"
STEP_9313_SHA256 = "f82a25c35e6bd738cd248d759c291d825ca0cfc4e08753dc5082741a78b23fc7"
STEP_9314_SHA256 = "633d9b09b8d19084ab69b3d260abab7f5bffed1a22fb5fba4f51da583e47298f"
STEP_9313_MANIFEST = "bef61cfa3e91b9e1a4c50b21975f57e1cf08bf6678738a5bdd48397e5f1d24e8"
STEP_9314_MANIFEST = "60bba36797c09100bd86d09f885930cdb0f72d5be587d05692a998ef9080bfc7"
TAPPED_MANIFEST = "13afeb21ce36b41129984e4d79016dec4ab956be7cea474cd82cfe17fc536a5b"
TERMINAL_MANIFEST = "2dcfc195baaf3e59ba539f6701fdb82b9a134a0461cfc68dec3a429e38433565"

PHASE_CACHE = (
    WORK_ROOT
    / "cache/bb2ffe33dbff-8ec38f8e1a90/jit/"
    "jit_advance_one_step_with_corrected_ni_phase_tap-"
    "ba31f251e42d3c0ca3e315fac43bc8990c3abde0312b965ab60d209bbfa696cf-cache"
)
PHASE_CACHE_SHA256 = "0bfc7fb46294eefdb80149d89bf2d35b5ec1a8eadfeb369e3e12cda809b90f3d"

WRF_REPO = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
WRF_SOURCES = {
    "dyn_em/solve_em.F": "3322d34e0ad070cf4d891aa6dbc93be70ef8240b84eb5f366a149487550aef89",
    "dyn_em/module_em.F": "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa",
    "dyn_em/module_small_step_em.F": "cabf1a177d50fb0096db79644af20cfe6d75217dbe63ab406a7e29bb54c17634",
    "dyn_em/module_bc_em.F": "6cfb52b849e3dd0b24769709cf502faa75886da454d9b668af725cccdc56d47f",
    "share/module_bc.F": "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
}

SOURCE_PATHS = {
    "operational": "src/gpuwrf/runtime/operational_mode.py",
    "state": "src/gpuwrf/contracts/state.py",
    "carry": "src/gpuwrf/runtime/operational_state.py",
    "mu": "src/gpuwrf/dynamics/mu_t_advance.py",
    "acoustic": "src/gpuwrf/dynamics/core/acoustic.py",
    "prep": "src/gpuwrf/dynamics/core/small_step_prep.py",
    "boundary": "src/gpuwrf/coupling/boundary_apply.py",
    "domain_tree": "src/gpuwrf/runtime/domain_tree.py",
    "nested_pipeline": "src/gpuwrf/integration/nested_pipeline.py",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return sha256_bytes(encoded)


def verify_embedded_digest(payload: dict[str, Any]) -> bool:
    expected = payload["proof_sha256"]
    unsigned = dict(payload)
    del unsigned["proof_sha256"]
    return canonical_digest(unsigned) == expected


def git_bytes(repo: Path, rev: str, path: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(repo), "show", f"{rev}:{path}"))


def git_text(repo: Path, rev: str, path: str) -> str:
    return git_bytes(repo, rev, path).decode()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def source_anchor(text: str, needle: str) -> dict[str, Any]:
    lines = text.splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if needle in line]
    if not matches:
        raise AssertionError(f"source anchor absent: {needle!r}")
    line = matches[0]
    return {"needle": needle, "line": line, "text": lines[line - 1].strip()}


def ast_state_slots(source: str) -> list[str]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "State":
            for child in node.body:
                if (
                    isinstance(child, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "__slots__" for target in child.targets)
                ):
                    value = ast.literal_eval(child.value)
                    return list(value)
    raise AssertionError("State.__slots__ not found")


def ast_dataclass_fields(source: str, class_name: str) -> list[str]:
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
            ]
    raise AssertionError(f"{class_name} not found")


class _OpaqueGpuwrfObject:
    """Inert pickle target; it cannot import or execute gpuwrf code."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "_OpaqueGpuwrfObject":
        del args, kwargs
        return object.__new__(cls)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._opaque_state = state


class _CarryUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("gpuwrf"):
            return _OpaqueGpuwrfObject
        if module.startswith("numpy") or module in {
            "builtins",
            "copyreg",
            "collections",
        }:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked class {module}.{name}")


def load_carry(path: Path) -> _OpaqueGpuwrfObject:
    with path.open("rb") as handle:
        value = _CarryUnpickler(handle).load()
    if not isinstance(value, _OpaqueGpuwrfObject):
        raise AssertionError("checkpoint root is not an inert carry object")
    return value


def state_arrays(carry: _OpaqueGpuwrfObject) -> dict[str, np.ndarray]:
    state = carry.state
    raw = getattr(state, "_opaque_state", None)
    if isinstance(raw, tuple) and raw and isinstance(raw[-1], dict):
        values = raw[-1]
    elif isinstance(getattr(state, "__dict__", None), dict):
        values = state.__dict__
    else:
        raise AssertionError("cannot decode inert State fields")
    return {name: value for name, value in values.items() if isinstance(value, np.ndarray)}


def finite_stat(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    nonfinite = int(array.size - int(finite.sum())) if np.issubdtype(array.dtype, np.floating) else 0
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nonfinite_count": nonfinite,
    }
    if array.size and (finite.any() if np.issubdtype(array.dtype, np.floating) else True):
        if np.issubdtype(array.dtype, np.floating):
            magnitudes = np.where(finite, np.abs(array), -np.inf)
        else:
            magnitudes = np.abs(array)
        flat = int(np.argmax(magnitudes))
        location = tuple(int(index) for index in np.unravel_index(flat, array.shape))
        result["max_abs_finite"] = float(magnitudes[location])
        result["max_abs_finite_index"] = list(location)
    else:
        result["max_abs_finite"] = None
        result["max_abs_finite_index"] = None
    return result


def transition_stat(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    if before.shape != after.shape:
        raise AssertionError("transition arrays have different shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.asarray(after) - np.asarray(before)
    return {
        "input": finite_stat(before),
        "output": finite_stat(after),
        "delta": finite_stat(delta),
    }


def projected_mask(value: np.ndarray) -> np.ndarray:
    mask = ~np.isfinite(np.asarray(value))
    while mask.ndim > 2:
        mask = np.any(mask, axis=0)
    return mask


def directed_cones(ny: int, nx: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    southeast = {
        (y, x)
        for y in range(ny)
        for x in range(nx)
        if y >= 1 and x <= nx - 2 and (y - 1) + ((nx - 2) - x) <= 9
    }
    northwest = {
        (y, x)
        for y in range(ny)
        for x in range(nx)
        if y <= ny - 2 and x >= 1 and ((ny - 2) - y) + (x - 1) <= 9
    }
    return southeast, northwest


def mask_coordinates(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(y), int(x)) for y, x in np.argwhere(mask)}


def vertical_nonfinite_counts(value: np.ndarray) -> list[dict[str, int]]:
    array = np.asarray(value)
    if array.ndim != 3:
        return []
    counts = np.count_nonzero(~np.isfinite(array), axis=(1, 2))
    return [
        {"level": int(level), "count": int(count)}
        for level, count in enumerate(counts)
        if count
    ]


def projected_threshold(value: np.ndarray, threshold: float) -> dict[str, Any]:
    array = np.abs(np.asarray(value))
    projection = np.max(array, axis=0) if array.ndim == 3 else array
    ys, xs = np.where(projection > threshold)
    if not len(ys):
        return {"threshold": threshold, "count": 0, "bbox": None, "corner_counts": [0, 0, 0, 0]}
    radius = 12
    corner_counts = [
        int(np.count_nonzero((ys < radius) & (xs < radius))),
        int(np.count_nonzero((ys < radius) & (xs >= projection.shape[1] - radius))),
        int(np.count_nonzero((ys >= projection.shape[0] - radius) & (xs < radius))),
        int(np.count_nonzero((ys >= projection.shape[0] - radius) & (xs >= projection.shape[1] - radius))),
    ]
    return {
        "threshold": threshold,
        "count": int(len(ys)),
        "bbox": {"y": [int(ys.min()), int(ys.max())], "x": [int(xs.min()), int(xs.max())]},
        "corner_counts_sw_se_nw_ne": corner_counts,
    }


def named_tap_differences(
    phase: dict[str, Any], state_slots: list[str], carry_fields: list[str]
) -> list[dict[str, Any]]:
    gate = phase["complete_carry_identity_gate"]
    ordinary = {leaf["path"]: leaf for leaf in gate["ordinary_reference_manifest"]["leaves"]}
    tapped = {leaf["path"]: leaf for leaf in gate["tapped_output_manifest"]["leaves"]}
    if set(ordinary) != set(tapped):
        raise AssertionError("tap changed complete-carry tree paths")

    def name(path: str) -> str:
        state_match = re.fullmatch(r"\[<flat index 0>\]\[<flat index (\d+)>\]", path)
        if state_match:
            return f"state.{state_slots[int(state_match.group(1))]}"
        carry_match = re.fullmatch(r"\[<flat index (\d+)>\]", path)
        if carry_match:
            return carry_fields[int(carry_match.group(1))]
        return path

    rows = []
    for path in sorted(ordinary):
        left = ordinary[path]
        right = tapped[path]
        if left["sha256"] != right["sha256"]:
            rows.append(
                {
                    "name": name(path),
                    "path": path,
                    "ordinary_sha256": left["sha256"],
                    "tapped_sha256": right["sha256"],
                    "shape": left["shape"],
                    "ordinary_nonfinite_count": left["nonfinite_count"],
                    "tapped_nonfinite_count": right["nonfinite_count"],
                }
            )
    return rows


def boundary_response_oracle() -> dict[str, Any]:
    dt = 6.0
    dts = 0.6
    substeps = 10
    linear = 1.0
    wrf = substeps * dts * (0.1 / dt) * linear
    weight_20 = 20.0 * (dts / dt) * 0.1 * linear
    released_20 = 1.0 - (1.0 - weight_20) ** substeps
    weight_1 = 1.0 * (dts / dt) * 0.1 * linear
    released_1 = 1.0 - (1.0 - weight_1) ** substeps
    assert abs(wrf - 0.1) < 1.0e-15
    assert abs(released_20 - 0.8926258176) < 1.0e-15
    assert abs(released_1 - 0.09561792499119559) < 1.0e-15
    return {
        "fixture": {
            "dt_s": dt,
            "dts_s": dts,
            "substeps": substeps,
            "uniform_residual": 1.0,
            "spec_adjacent_relax_linear_weight": linear,
        },
        "pristine_wrf_frozen_tendency_response": wrf,
        "released_moving_residual_strength20": {
            "per_substep_weight": weight_20,
            "response": released_20,
            "ratio_to_wrf": released_20 / wrf,
        },
        "candidate_moving_residual_strength1": {
            "per_substep_weight": weight_1,
            "response": released_1,
            "ratio_to_wrf": released_1 / wrf,
        },
        "verdict": (
            "PROVED_SOURCE_DISCREPANCY_ONLY: released nested normal-momentum gain/cadence "
            "does not equal pristine WRF; this algebra does not prove incident causality"
        ),
    }


def continuity_oracles() -> dict[str, Any]:
    dnw = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    dvdxi = np.asarray([2.0, -1.0, 4.0, -3.0], dtype=np.float64)
    mu_tend = -0.75
    dmdt = float(np.sum(dnw * dvdxi))
    mu_tendency = dmdt + mu_tend
    mut = 90_000.0
    mu_work_old = -25.0
    dts = 0.6
    epssm = 0.5
    mu_work_new = mu_work_old + dts * mu_tendency
    mudf = mu_tendency
    muts = mut + mu_work_new
    muave = 0.5 * ((1.0 + epssm) * mu_work_new + (1.0 - epssm) * mu_work_old)
    assert np.isclose(mudf, dmdt + mu_tend, rtol=0.0, atol=1.0e-15)
    assert np.isclose(muts - mut, mu_work_new, rtol=0.0, atol=1.0e-11)
    assert np.isclose(
        muave,
        0.5 * ((1.0 + epssm) * (muts - mut) + (1.0 - epssm) * mu_work_old),
        rtol=0.0,
        atol=1.0e-11,
    )

    # A null-space perturbation proves that terminal dmdt/mudf cannot identify
    # the contributing vertical divergence profile.
    alternate = dvdxi.copy()
    delta = 7.0
    alternate[0] += delta
    alternate[1] -= delta * dnw[0] / dnw[1]
    alternate_dmdt = float(np.sum(dnw * alternate))
    assert np.isclose(alternate_dmdt, dmdt, rtol=0.0, atol=1.0e-15)
    assert not np.array_equal(alternate, dvdxi)

    with np.errstate(invalid="ignore"):
        inf_times_zero = float(np.float64(np.inf) * np.float64(0.0))
        nan_times_zero = float(np.float64(np.nan) * np.float64(0.0))
    assert np.isnan(inf_times_zero)
    assert np.isnan(nan_times_zero)
    return {
        "finite_fixture": {
            "dnw": dnw.tolist(),
            "dvdxi": dvdxi.tolist(),
            "mu_tend": mu_tend,
            "dmdt": dmdt,
            "mu_tendency_and_mudf": mudf,
            "muts": muts,
            "muave": muave,
        },
        "many_to_one_nullspace": {
            "alternate_dvdxi": alternate.tolist(),
            "alternate_dmdt": alternate_dmdt,
            "same_weighted_divergence": True,
            "different_vertical_profile": True,
        },
        "ieee_nonfinite_scale_zero": {
            "inf_times_zero_is_nan": bool(np.isnan(inf_times_zero)),
            "nan_times_zero_is_nan": bool(np.isnan(nan_times_zero)),
            "consequence": (
                "the current mu_scale=0 branch does not zero a nonfinite raw tendency "
                "when the next expression multiplies it by zero"
            ),
        },
    }


def authenticate_json(path: Path, expected_file_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_file_sha:
        raise AssertionError(f"authority hash changed: {path}: {actual}")
    payload = json.loads(path.read_text())
    if not verify_embedded_digest(payload):
        raise AssertionError(f"embedded proof digest invalid: {path}")
    return payload, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": actual,
        "payload_sha256": payload["proof_sha256"],
        "payload_digest_valid": True,
    }


def build_proof() -> dict[str, Any]:
    recorder, recorder_auth = authenticate_json(RECORDER_PROOF, RECORDER_PROOF_SHA256)
    ordinary, ordinary_auth = authenticate_json(ORDINARY_PROOF, ORDINARY_PROOF_SHA256)
    phase, phase_auth = authenticate_json(PHASE_PROOF, PHASE_PROOF_SHA256)

    if phase["tap_science"] != {
        "reason": (
            "complete tapped output carry differs from retained ordinary output; "
            "the device summary was not transferred or decoded"
        ),
        "status": "DISCARDED_UNREAD",
    }:
        raise AssertionError("phase tap science marker changed")
    if phase["complete_carry_identity_gate"]["passed"] is not False:
        raise AssertionError("phase tap unexpectedly passed identity")

    launch_commit = git_value(ROOT, "rev-parse", LAUNCH_COMMIT)
    launch_tree = git_value(ROOT, "show", "-s", "--format=%T", LAUNCH_COMMIT)
    launch_parents = git_value(ROOT, "show", "-s", "--format=%P", LAUNCH_COMMIT).split()
    if (launch_commit, launch_tree, launch_parents) != (LAUNCH_COMMIT, LAUNCH_TREE, [LAUNCH_PARENT]):
        raise AssertionError("launch commit object changed")

    launch_sources = {
        key: git_text(ROOT, LAUNCH_COMMIT, path) for key, path in SOURCE_PATHS.items()
    }
    source_auth = {
        key: {
            "path": path,
            "launch_blob_sha256": sha256_bytes(launch_sources[key].encode()),
        }
        for key, path in SOURCE_PATHS.items()
    }
    state_slots = ast_state_slots(launch_sources["state"])
    carry_fields = ast_dataclass_fields(launch_sources["carry"], "OperationalCarry")

    wrf_sources: dict[str, str] = {}
    wrf_auth: dict[str, Any] = {}
    if git_value(WRF_REPO, "rev-parse", WRF_COMMIT) != WRF_COMMIT:
        raise AssertionError("pristine WRF commit unavailable")
    for path, expected in WRF_SOURCES.items():
        raw = git_bytes(WRF_REPO, WRF_COMMIT, path)
        actual = sha256_bytes(raw)
        if actual != expected:
            raise AssertionError(f"pristine WRF blob changed: {path}")
        wrf_sources[path] = raw.decode()
        wrf_auth[path] = {"sha256": actual, "bytes": len(raw)}

    for path, expected in ((STEP_9313, STEP_9313_SHA256), (STEP_9314, STEP_9314_SHA256)):
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(f"retained checkpoint changed: {path}")
    input_carry = load_carry(STEP_9313)
    output_carry = load_carry(STEP_9314)
    state_in = state_arrays(input_carry)
    state_out = state_arrays(output_carry)

    key_state_fields = (
        "u",
        "v",
        "w",
        "p_total",
        "p_perturbation",
        "ph_total",
        "ph_perturbation",
        "mu_total",
        "mu_perturbation",
        "theta",
        "Ni",
    )
    state_transitions = {
        name: transition_stat(state_in[name], state_out[name]) for name in key_state_fields
    }
    if any(row["input"]["nonfinite_count"] for row in state_transitions.values()):
        raise AssertionError("step-9313 key State is no longer finite")
    if any(row["output"]["nonfinite_count"] for row in state_transitions.values()):
        raise AssertionError("completed step-9314 guarded key State is no longer finite")
    if state_transitions["p_total"]["output"]["max_abs_finite"] < 1.0e300:
        raise AssertionError("step-9314 catastrophic finite pressure signature changed")

    save_names = ("u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save", "ww_save")
    stage_entry = {}
    for name in save_names:
        array = np.asarray(getattr(output_carry, name))
        stage_entry[name] = {
            **finite_stat(array),
            "projected_gt_1e6": projected_threshold(array, 1.0e6),
        }
        if stage_entry[name]["nonfinite_count"]:
            raise AssertionError(f"RK3-entry save unexpectedly nonfinite: {name}")
        if not all(stage_entry[name]["projected_gt_1e6"]["corner_counts_sw_se_nw_ne"]):
            raise AssertionError(f"four-corner RK3-entry signature changed: {name}")
    if stage_entry["u_save"]["max_abs_finite"] < 5.0e8:
        raise AssertionError("RK3-entry u runaway signature changed")

    scratch_names = ("t_2ave", "ww", "mudf", "muave", "muts")
    scratch = {}
    shared_mask: set[tuple[int, int]] | None = None
    expected_counts = {"t_2ave": 4840, "ww": 4730, "mudf": 110, "muave": 110, "muts": 110}
    for name in scratch_names:
        array = np.asarray(getattr(output_carry, name))
        row = finite_stat(array)
        row["vertical_nonfinite_counts"] = vertical_nonfinite_counts(array)
        coords = mask_coordinates(projected_mask(array))
        row["projected_nonfinite_cells"] = len(coords)
        if row["nonfinite_count"] != expected_counts[name] or len(coords) != 110:
            raise AssertionError(f"first-bad scratch inventory changed: {name}")
        if shared_mask is None:
            shared_mask = coords
        elif coords != shared_mask:
            raise AssertionError(f"scratch masks differ: {name}")
        scratch[name] = row
    assert shared_mask is not None
    cone_a, cone_b = directed_cones(93, 111)
    cone_union = cone_a | cone_b
    if len(cone_a) != 55 or len(cone_b) != 55 or shared_mask != cone_union:
        raise AssertionError("directed two-cone geometry changed")

    tap_diffs = named_tap_differences(phase, state_slots, carry_fields)
    expected_tap_names = {
        "state.u",
        "state.v",
        "state.w",
        "state.p_total",
        "state.p_perturbation",
        "state.ph_total",
        "state.ph_perturbation",
        "state.mu_total",
        "state.mu_perturbation",
        "state.qc",
        "state.qi",
        "state.qg",
        "state.Ni",
        "state.Ng",
        "state.qke",
        "t_2ave",
        "ww",
        "mudf",
        "muave",
        "muts",
        "ph_tend",
        "ww_save",
    }
    if len(tap_diffs) != 22 or {row["name"] for row in tap_diffs} != expected_tap_names:
        raise AssertionError("tap differing-leaf map changed")
    if any(row["ordinary_nonfinite_count"] != row["tapped_nonfinite_count"] for row in tap_diffs):
        raise AssertionError("tap altered a differing leaf's nonfinite count")

    ordinary_program = ordinary["ordinary_program"]
    tap_program = phase["program_audit"]
    if ordinary_program["stablehlo_sha256"] != "24035858ab6c6f555d670a09491ac2737367741916f9edc51a00e5ec427dbfe3":
        raise AssertionError("ordinary HLO identity changed")
    if tap_program["stablehlo_sha256"] != "2dda84e00ebcdf6a589d5d8488e56a1f12fe6677eea8f80afd7fc279a41dae31":
        raise AssertionError("tap HLO identity changed")

    phase_cache_actual = sha256_file(PHASE_CACHE)
    if phase_cache_actual != PHASE_CACHE_SHA256:
        raise AssertionError("phase-tap compiled cache artifact changed")

    operational = launch_sources["operational"]
    mu_source = launch_sources["mu"]
    prep_source = launch_sources["prep"]
    boundary_source = launch_sources["boundary"]
    domain_source = launch_sources["domain_tree"]
    nested_source = launch_sources["nested_pipeline"]
    small_step_wrf = wrf_sources["dyn_em/module_small_step_em.F"]
    solve_wrf = wrf_sources["dyn_em/solve_em.F"]
    bc_wrf = wrf_sources["share/module_bc.F"]

    ordinary_hlo_bytes = int(ordinary_program["stablehlo_bytes"])
    tap_hlo_bytes = int(tap_program["stablehlo_bytes"])
    geometry_rows = sorted(shared_mask)
    ni_cell = (48, 78)
    nearest_ni_grid_distance = min(abs(ni_cell[0] - y) + abs(ni_cell[1] - x) for y, x in shared_mask)

    proof: dict[str, Any] = {
        "schema": "gpuwrf.v0234.corrected-ni-identity-redesign.v1",
        "status": "COMPLETE",
        "verdict": "NO_FIX_LOCALIZED",
        "scope": {
            "gpu_calls": 0,
            "model_calls": 0,
            "jax_imported": False,
            "tap_science": "DISCARDED_UNREAD",
            "model_or_numerical_edits": 0,
            "full_18h_run": False,
            "agents_launched": 0,
        },
        "authority": {
            "launch": {
                "commit": launch_commit,
                "parent": launch_parents[0],
                "tree": launch_tree,
                "commit_object_present": True,
            },
            "observer_failures": {
                "broad_recorder": recorder_auth,
                "phase_tap_exact_preserved_copy": phase_auth,
            },
            "ordinary_bisection": ordinary_auth,
            "retained_carries": {
                "step_9313": {
                    "path": str(STEP_9313),
                    "file_sha256": STEP_9313_SHA256,
                    "manifest_sha256": STEP_9313_MANIFEST,
                },
                "step_9314": {
                    "path": str(STEP_9314),
                    "file_sha256": STEP_9314_SHA256,
                    "manifest_sha256": STEP_9314_MANIFEST,
                },
            },
            "terminal_manifest": {
                "required": TERMINAL_MANIFEST,
                "actual": ordinary["terminal_identity"]["actual_manifest_sha256"],
                "ordinary_proof_authenticated": True,
                "unchanged": ordinary["terminal_identity"]["all_leaf_bytes_equal_to_frozen_ordinary"],
            },
            "compiler": {
                "ordinary": {
                    "callable": ordinary_program["callable"],
                    "stablehlo_bytes": ordinary_hlo_bytes,
                    "stablehlo_sha256": ordinary_program["stablehlo_sha256"],
                    "input_output_structure_and_avals_identical": ordinary_program[
                        "input_output_structure_and_avals_identical"
                    ],
                    "leaf_count": ordinary_program["output_leaf_count"],
                },
                "phase_tap": {
                    "callable": tap_program["callable"],
                    "stablehlo_bytes": tap_hlo_bytes,
                    "stablehlo_sha256": tap_program["stablehlo_sha256"],
                    "stablehlo_byte_delta_vs_ordinary": tap_hlo_bytes - ordinary_hlo_bytes,
                    "cache_artifact": {
                        "path": str(PHASE_CACHE),
                        "bytes": PHASE_CACHE.stat().st_size,
                        "sha256": phase_cache_actual,
                    },
                },
            },
            "launch_source": source_auth,
            "pristine_wrf": {
                "repo": str(WRF_REPO),
                "commit": WRF_COMMIT,
                "working_tree_used": False,
                "git_object_blobs": wrf_auth,
            },
        },
        "observer_failure_explanation": {
            "broad_recorder": {
                "ordinary_off_manifest": recorder["recorder_off"]["carry_manifest"]["manifest_sha256"],
                "recorder_on_manifest": recorder["recorder_on"]["carry_manifest"]["manifest_sha256"],
                "different_leaf_count": recorder["recorder_on"]["comparison_to_canonical"]["different_leaf_count"],
                "device_to_host_bytes": recorder["recorder_on"]["transfer_audit"]["device_to_host_bytes"],
                "host_to_device_bytes": recorder["recorder_on"]["transfer_audit"]["host_to_device_bytes"],
                "causal_use": "INADMISSIBLE_DEBUG_ONLY",
            },
            "phase_tap": {
                "required_manifest": STEP_9314_MANIFEST,
                "actual_manifest": TAPPED_MANIFEST,
                "different_leaf_count": 22,
                "named_differing_leaves": tap_diffs,
                "all_differing_leaf_nonfinite_counts_unchanged": True,
                "tap_science": "DISCARDED_UNREAD",
            },
            "source_structural_difference": {
                "ordinary_single_scan": source_anchor(
                    operational, "length=int(stage.number_of_small_timesteps),"
                ),
                "tap_first_substep_outside_scan": source_anchor(
                    operational, "acoustic, phase_tap_summary = acoustic_substep_core("
                ),
                "tap_remaining_scan": source_anchor(
                    operational, "remaining_substeps = int(stage.number_of_small_timesteps) - 1"
                ),
                "known_fusion_context_bit_identity_warning": source_anchor(
                    operational, "XLA FMA contraction is fusion-context-dependent"
                ),
            },
            "causal_statement": {
                "proved": (
                    "the tap split one ordinary scan into an out-of-scan first substep plus a shorter scan, "
                    "kept summary reductions live, and produced a different authenticated StableHLO program"
                ),
                "inference_not_proved": (
                    "the exact compiler scheduling/fusion/FMA/liveness change that altered each of the 22 leaf byte streams"
                ),
                "conclusion": (
                    "the 22 leaves are observer perturbation, not phase science; their unchanged nonfinite counts "
                    "do not rescue causal admissibility"
                ),
            },
        },
        "ordinary_localization": {
            "step_9313_key_state": {
                name: row["input"] for name, row in state_transitions.items()
            },
            "rk3_stage_entry_saved_by_completed_step_9314": stage_entry,
            "rk3_entry_semantics_source": {
                "prep_saves_stage_entry": source_anchor(prep_source, "u_save=jnp.asarray(state.u)"),
                "completed_carry_returns_prep_save": source_anchor(
                    operational, "u_save=prep.u_save"
                ),
                "stage_sequence": source_anchor(
                    operational, "carry = advance_stage(carry, stages[1])"
                ),
            },
            "completed_step_9314_guarded_state": {
                name: row["output"] for name, row in state_transitions.items()
            },
            "state_finite_is_not_health": {
                "p_total_max_abs": state_transitions["p_total"]["output"]["max_abs_finite"],
                "w_max_abs": state_transitions["w"]["output"]["max_abs_finite"],
                "u_max_abs": state_transitions["u"]["output"]["max_abs_finite"],
                "verdict": (
                    "completed State is finite only under the byte-health predicate; it is catastrophically unphysical"
                ),
            },
            "guard_and_overwrite_sources": {
                "mass_origin_fallback": source_anchor(
                    operational, "mu_total = jnp.where(valid_mu, candidate_mu_total, origin.mu_total)"
                ),
                "boundary_finite_origin": source_anchor(
                    operational, "u=_finite_or_origin(bounded.u, physical_origin.u)"
                ),
                "scratch_survives_state_replacement": source_anchor(
                    operational, "final_carry = _maybe_exchange_sharded_carry_halos(carry.replace(state=next_state))"
                ),
            },
            "first_bad_scratch": scratch,
            "two_cone_geometry": {
                "shape_yx": [93, 111],
                "southeast_apex_yx": [1, 109],
                "northwest_apex_yx": [91, 1],
                "radius_inclusive": 9,
                "cells_per_cone": 55,
                "union_cells": len(geometry_rows),
                "exactly_matches_all_five_projected_nonfinite_masks": True,
                "cells": [list(cell) for cell in geometry_rows],
                "boundary_or_edge": False,
                "interpretation": (
                    "a ten-layer directed stencil cone is proved geometrically; neither its temporal seed nor operator is identified"
                ),
            },
            "strongest_temporal_interval": {
                "lower_bound": "ordinary completed step 9313 key dry state and scratch are finite/normal-scale",
                "upper_bound": (
                    "by RK3 entry of step 9314 (after ordinary RK1+RK2), all four inner corners already carry 1e8-1e13 dry-mode amplitudes"
                ),
                "final_rk3": (
                    "the ten-substep final stage amplifies the established runaway to 1e206-1e302 and writes the five NaN scratch families"
                ),
                "tap_consequence": (
                    "a first-substep RK3 post-mu/post-w tap was downstream of the earliest retained causal interval"
                ),
            },
        },
        "equation_closure": {
            "released_reverse_slice": {
                "active_bounds": source_anchor(mu_source, "y0, y1 = (1, ny - 1)"),
                "dvdxi": source_anchor(mu_source, "dvdxi = inputs.msftx[ys, xs]"),
                "dmdt": source_anchor(mu_source, "dmdt_active = jnp.sum"),
                "raw_mu_tendency": source_anchor(mu_source, "mu_tendency = dmdt_active + inputs.mu_tend"),
                "nonfinite_scale_multiplication": source_anchor(mu_source, "mu_tendency = mu_tendency * mu_scale"),
                "mudf_final_overwrite": source_anchor(mu_source, "mudf_i = mu_tendency"),
                "muts_final_overwrite": source_anchor(mu_source, "muts_i = inputs.mut[ys, xs] + mu_work_new"),
                "muave_final_overwrite": source_anchor(mu_source, "muave_i = 0.5 *"),
                "ww_recurrence": source_anchor(mu_source, "ww_rows.append(ww_rows[-1] - increment)"),
            },
            "pristine_wrf": {
                "solve_sequence": [
                    source_anchor(solve_wrf, "CALL advance_uv"),
                    source_anchor(solve_wrf, "CALL advance_mu_t"),
                    source_anchor(solve_wrf, "CALL advance_w"),
                    source_anchor(solve_wrf, "CALL calc_p_rho"),
                ],
                "nested_or_specified_bounds": [
                    source_anchor(small_step_wrf, "i_start = max(its,ids+1)"),
                    source_anchor(small_step_wrf, "j_start = max(jts,jds+1)"),
                ],
                "continuity": [
                    source_anchor(small_step_wrf, "DMDT(i)    = DMDT(i) + dnw(k)*dvdxi(i,k)"),
                    source_anchor(small_step_wrf, "MUDF(i,j) = (DMDT(i)+MU_TEND(i,j))"),
                    source_anchor(small_step_wrf, "MUTS(i,j) = MUT(i,j)+MU(i,j)"),
                    source_anchor(small_step_wrf, "MUAVE(i,j) =.5*((1.+epssm)*MU(i,j)"),
                ],
                "relax_tendency_equation": source_anchor(
                    bc_wrf, "+ fcx(b_dist+1)*fls0"
                ),
            },
            "independent_numpy_oracles": {
                "continuity_and_identifiability": continuity_oracles(),
                "normal_boundary_response": boundary_response_oracle(),
            },
            "secondary_guard_defect": {
                "source_claim": source_anchor(mu_source, "A non-finite tendency is zeroed (scale 0)"),
                "actual_ieee_result": "Inf*0 and NaN*0 are NaN",
                "causal_status": (
                    "PROVED secondary propagation defect, NOT an authorized fix and NOT the upstream runaway cause"
                ),
            },
        },
        "candidate_cause_ranking": {
            "nested_normal_momentum_boundary": {
                "rank": 1,
                "evidence_for": [
                    "RK3-entry runaway exists at all four inner corners before final-stage scratch NaNs",
                    "the exact NaN masks are inward directed corner cones",
                    "live child resolves normal_bdy_relax_strength=None to the released global strength 20",
                    "pristine WRF uses a frozen RK1 relaxation tendency with nominal strength 1",
                    "the independent uniform-residual oracle proves 0.8926258176 released response versus 0.1 WRF response",
                ],
                "released_source": {
                    "live_child_no_override": source_anchor(domain_source, "force_geopotential=False"),
                    "global_strength_20": source_anchor(
                        boundary_source, 'NORMAL_BDY_RELAX_STRENGTH = float(os.environ.get("GPUWRF_NORMAL_BDY_RELAX_STRENGTH", "20.0"))'
                    ),
                    "moving_residual": source_anchor(
                        boundary_source, "u = u_work + wu * (u_target - u_work)"
                    ),
                    "root_only_strength_1": source_anchor(
                        nested_source, "normal_bdy_relax_strength=1.0"
                    ),
                },
                "incident_causality": "PLAUSIBLE_NOT_PROVED",
            },
            "remaining_nonunique_upstream_terms": [
                "advance_uv pressure-gradient/acoustic update",
                "large-step momentum and mu tendencies",
                "moving normal boundary work target/corner ownership",
                "advance_mu_t horizontal divergence and positivity-guard NaN propagation",
                "advance_w/geopotential/pressure feedback already active during RK1/RK2",
            ],
            "fix_authority": "NONE: retained outputs do not uniquely select one upstream operator",
        },
        "identifiability_no_go": {
            "lost_intermediate": (
                "ordinary pre/post arrays for RK1, RK2, and each acoustic substep: u_work/v_work, dvdxi, dmdt, "
                "mu_tend, raw/limited mu_tendency, w/ph work, and pressure coefficients"
            ),
            "why_terminal_carry_is_insufficient": [
                "u_save/v_save/w_save/t_save/ph_save/mu_save expose only RK3 stage entry, not RK1/RK2 operator outputs",
                "mudf/muts/muave share one aggregate final-overwrite tendency and cannot separate dmdt from mu_tend",
                "the NumPy null-space construction gives different vertical dvdxi profiles with identical dmdt/mudf",
                "ww and t_2ave are NaN at every affected vertical lane, erasing inversion information",
                "post-RK guards overwrite nonfinite State values from the physical origin while leaving scratch untouched",
                "both attempted output-returning observers changed the executable/output bytes",
            ],
            "unique_operator_identified": False,
            "general_fix_authorized": False,
        },
        "shortest_decisive_experiment": {
            "name": "NESTED_NORMAL_GAIN_1_NONOBSERVING_SAME_CARRY_AB",
            "authorization_required": True,
            "rationale": (
                "the ranked boundary discrepancy is the only candidate with independent geometry, pristine-source, "
                "and algebra support; its existing static strength hook permits a one-variable A/B without a tap"
            ),
            "A": {
                "input": STEP_9313_SHA256,
                "output_reference_manifest": STEP_9314_MANIFEST,
                "normal_strength": 20.0,
                "execution": "retained authenticated ordinary output; do not rerun",
            },
            "B": {
                "input": STEP_9313_SHA256,
                "only_static_difference": "GPUWRF_NORMAL_BDY_RELAX_STRENGTH=1.0 before import",
                "callable": ordinary_program["callable"],
                "output": "unchanged 106-leaf OperationalCarry only; no summaries, recorder, callbacks, or new leaves",
                "dispatches": (
                    "compile once; run completed step 9314; only if finite/scale-healthy, run completed step 9315 to cover Ni onset"
                ),
                "health": "materialize only after each completed dispatch, outside production HLO",
            },
            "pre_dispatch_gates": [
                "rehash launch, input carry, ordinary A proof, cache/source authority, and frozen terminal manifest",
                "prove B input/output treedef, 106 avals, callable, and callback-free surface equal ordinary",
                "prove environment/source audit differs only in the static normal gain; no numerical guard/tolerance/sanitizer change",
                "pass pure-NumPy boundary response and pristine-WRF source oracle before GPU authorization",
            ],
            "decision": {
                "if_step_9314_scratch_nonfinite": (
                    "falsify gain-only candidate and stop; do not launch a second blind candidate"
                ),
                "if_step_9314_finite_but_unphysical_scale": (
                    "inconclusive stabilization; stop without fix"
                ),
                "if_steps_9314_9315_finite_and_scale_healthy": (
                    "localize the incident trigger to strength-20 nested normal forcing; this authorizes a separate "
                    "contract to replace the moving residual with the full pristine-WRF frozen relax-bundle semantics, "
                    "not a strength-only production fix"
                ),
            },
            "identity_semantics": (
                "B must preserve the ordinary carry/program interface and contain no observer. Its output is expected "
                "to differ because the experiment intentionally changes one equation coefficient; requiring B bytes "
                "to equal the bad A output would make causal discrimination impossible."
            ),
            "maximum_gpu_scope": "one compile and one or two same-carry d03 dispatches; no prefix and no full forecast",
        },
        "mechanism_separation": {
            "canonical_ni_cell": {
                "jax_yx": [48, 78],
                "lat": 28.297913,
                "lon": -16.303406,
                "landmask": 0,
                "hgt_m": 0.0,
                "classification": "interior offshore E/NE; not boundary or edge",
                "minimum_manhattan_cells_to_step9314_scratch_wedges": nearest_ni_grid_distance,
            },
            "ni_link_to_corner_runaway": (
                "temporal adjacency exists (scratch first bad at 9314, Ni nonfinite after 9315), but an operator-causal "
                "link to the spatially remote Ni cell is not proved"
            ),
            "v10_last_matched_1500": {
                "rmse": 2.1128268857679338,
                "land_rmse": 2.2264724301076377,
                "sea_rmse": 2.0839931788025168,
                "max_abs": 11.358115434646606,
                "max_yx": [39, 19],
                "max_lat": 28.216018676757812,
                "max_lon": -16.90625,
                "max_landmask": 0,
                "ni_cell_delta": -0.7763886451721191,
            },
            "common_ni_v10_root_proved": False,
            "v10_verdict": "SEPARATE_OPEN TRACK; no V10 fix or gate authority from this RCA",
        },
    }
    if proof["authority"]["terminal_manifest"]["actual"] != TERMINAL_MANIFEST:
        raise AssertionError("terminal manifest authority changed")
    if proof["authority"]["compiler"]["ordinary"]["leaf_count"] != 106:
        raise AssertionError("ordinary carry leaf count changed")
    proof["proof_sha256"] = canonical_digest(proof)
    return proof


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=SPRINT / "identity-redesign-proof.json",
    )
    args = parser.parse_args(argv)
    proof = build_proof()
    atomic_write_json(args.output, proof)
    print(
        f"IDENTITY_REDESIGN_COMPLETE verdict={proof['verdict']} "
        f"proof_sha256={proof['proof_sha256']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
