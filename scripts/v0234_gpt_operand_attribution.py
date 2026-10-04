#!/usr/bin/env python3
"""Partition the v0234 single-authority CPU SP2 momentum error.

This is a backend-dark, NumPy-only comparator.  It validates every array in
the one permitted production-adapter capture, independently reads the pristine
WRF dump, reconstructs the complete operational U/V systems, and uses exact
Shapley decompositions over captured coefficient operands.  It does not run
JAX, a GPU, WRF, MPI, or another production adapter invocation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
CAPTURE_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_gpt_single_authority_attribution/"
    "cpu-production-adapter-authority-v1"
)
CAPTURE_ARCHIVE = CAPTURE_ROOT / "single-authority-capture.npz"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
CAPTURE_PROOF = SPRINT / "single-authority-capture-proof.json"
AUDIT_SCRIPT = REPO / "scripts/v0234_gpt_adversarial_audit.py"
WRF_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/"
    "evidence-dumps-fresh-d03-runtime"
)
WRF_SOURCE = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2/"
    "source/instrumented/phys/MYNN-EDMF/module_bl_mynnedmf.F90"
)
WRF_DRIVER_SOURCE = WRF_SOURCE.parent / "WRF/module_bl_mynnedmf_driver.F90"
PORT_SOURCE = REPO / "src/gpuwrf/physics/mynn_pbl.py"
COUPLER_SOURCE = REPO / "src/gpuwrf/coupling/physics_couplers.py"
RUNTIME_SOURCE = REPO / "src/gpuwrf/runtime/operational_mode.py"

EXPECTED_CAPTURE_ARCHIVE = (
    "bc576333ba54d26db7272c8692b327b56f477e6183a6c02d596bf0e6685cd2fe"
)
EXPECTED_CAPTURE_MANIFEST = (
    "7835e8fecce89fb84d1dafaf316cb8ea0f448acb241bae789095886406b23e7a"
)
EXPECTED_CAPTURE_PROOF = (
    "3d0dbf3561f1f9d75cc3d68d872d301014287b489065b68037f21c6bcdd5c00b"
)
EXPECTED_WRF_TREE = (
    "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
)
DT = 6.0
NZ = 44
NY = 93
NX = 111
CATEGORIES = ("entry_state", "metric", "surface_drag", "mixing", "mass_flux")
TERMINALS = {
    "SINGLE_AUTHORITY_SOURCE_LOCALIZED_FIX_PROVEN",
    "SINGLE_AUTHORITY_SOURCE_LOCALIZED_NO_FIX",
    "SINGLE_AUTHORITY_ATTRIBUTION_INCONCLUSIVE",
}


class AttributionFailure(RuntimeError):
    """Fail-closed evidence or arithmetic failure."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items()
               if key != "canonical_payload_sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AttributionFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise AttributionFailure(f"TEMP_NOT_FRESH:{temporary}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode())
        stream.flush()
    os.link(temporary, path)
    temporary.unlink()


def finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise AttributionFailure(f"NONFINITE_OR_OBJECT:{name}")
    return array


def rms(value: np.ndarray) -> float:
    array = finite("rms", np.asarray(value, dtype=np.float64))
    return float(np.sqrt(np.mean(array * array)))


def sse(value: np.ndarray) -> float:
    array = finite("sse", np.asarray(value, dtype=np.float64))
    return float(np.sum(array * array))


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = finite("metric_left", np.asarray(left))
    rhs = finite("metric_right", np.asarray(right))
    if lhs.shape != rhs.shape:
        raise AttributionFailure(f"METRIC_SHAPE:{lhs.shape}:{rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    denominator = rms(rhs.astype(np.float64))
    return {
        "rms": rms(delta),
        "max_abs": float(np.max(np.abs(delta))),
        "sse": sse(delta),
        "exact_fraction": float(np.mean(lhs == rhs)),
        "reference_rms": denominator,
        "relative_rms": rms(delta) / denominator if denominator else None,
    }


def projection(contribution: np.ndarray, error: np.ndarray) -> dict[str, Any]:
    value = finite("contribution", np.asarray(contribution, dtype=np.float64))
    target = finite("error", np.asarray(error, dtype=np.float64))
    if value.shape != target.shape:
        raise AttributionFailure("PROJECTION_SHAPE")
    dot = float(np.sum(value * target))
    target_sse = sse(target)
    value_sse = sse(value)
    correlation = (
        dot / math.sqrt(target_sse * value_sse)
        if target_sse and value_sse else None
    )
    return {
        "rms": rms(value),
        "max_abs": float(np.max(np.abs(value))),
        "sse": value_sse,
        "signed_projection_fraction_of_authentic_error_sse": (
            dot / target_sse if target_sse else None
        ),
        "cosine_with_authentic_error": correlation,
        "rms_over_authentic_error_rms": (
            rms(value) / rms(target) if rms(target) else None
        ),
    }


def vertical_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    result = metrics(left, right)
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if delta.ndim != 3:
        return result
    level_rms = np.sqrt(np.mean(delta * delta, axis=(1, 2)))
    ranked = np.argsort(level_rms)[::-1]
    result["level_rms"] = level_rms.astype(float).tolist()
    result["top_levels_by_rms"] = [
        {"zero_based_level": int(level), "rms": float(level_rms[level])}
        for level in ranked[:8]
    ]
    return result


def import_dump_reader() -> tuple[Any, dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("v0234_phase_a_reader", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise AttributionFailure("AUDIT_IMPORT_SPEC")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.WRF_DUMP_ROOT != WRF_ROOT:
        raise AttributionFailure("WRF_ROOT_DRIFT")
    return module.IndependentWrfDump(WRF_ROOT), {
        "path": str(AUDIT_SCRIPT), "sha256": sha256_file(AUDIT_SCRIPT),
    }


def load_outer2(reader: Any, tag: str) -> np.ndarray:
    result = np.full((NY, NX), np.nan, dtype=np.float32)
    occupied = np.zeros(result.shape, dtype=bool)
    for rank, meta in enumerate(reader.meta):
        ims, ime, jms, jme, _kms, _kme = meta[
            "ims_ime_jms_jme_kms_kme"
        ]
        ips, ipe, jps, jpe, _kps, _kpe = meta[
            "ips_ipe_jps_jpe_kps_kpe"
        ]
        path = (
            WRF_ROOT / "mynnsp2" / f"rank{rank:04d}"
            / f"step000001_outer__{tag}.bin"
        )
        raw = np.fromfile(path, dtype=">f4")
        shape = (ime - ims + 1, jme - jms + 1)
        if raw.size != math.prod(shape):
            raise AttributionFailure(f"OUTER2_PAYLOAD:{tag}:{rank}")
        memory = raw.reshape(shape, order="F")
        for j in range(max(jps, reader.j.start), min(jpe, reader.j.stop - 1) + 1):
            for i in range(max(ips, reader.i.start), min(ipe, reader.i.stop - 1) + 1):
                target = (j - reader.j.start, i - reader.i.start)
                if occupied[target]:
                    raise AttributionFailure(f"OUTER2_DUPLICATE:{tag}:{target}")
                result[target] = memory[i - ims, j - jms]
                occupied[target] = True
    if not occupied.all():
        raise AttributionFailure(f"OUTER2_COVERAGE:{tag}")
    return finite(tag, result)


def validate_inputs() -> tuple[dict[str, np.ndarray], dict[str, Any], Any, dict[str, Any]]:
    git_head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if git_status:
        raise AttributionFailure(f"WORKTREE_NOT_CLEAN:{git_status}")
    expected = {
        CAPTURE_ARCHIVE: EXPECTED_CAPTURE_ARCHIVE,
        CAPTURE_MANIFEST: EXPECTED_CAPTURE_MANIFEST,
        CAPTURE_PROOF: EXPECTED_CAPTURE_PROOF,
    }
    records: dict[str, Any] = {}
    for path, digest in expected.items():
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise AttributionFailure(f"INPUT_NOT_REGULAR:{path}")
        actual = sha256_file(path)
        if actual != digest:
            raise AttributionFailure(f"INPUT_HASH:{path}:{actual}")
        records[str(path)] = {"sha256": actual, "size": path.stat().st_size}

    manifest = json.loads(CAPTURE_MANIFEST.read_text(encoding="utf-8"))
    proof = json.loads(CAPTURE_PROOF.read_text(encoding="utf-8"))
    if canonical_without_self(manifest) != manifest.get("canonical_payload_sha256"):
        raise AttributionFailure("CAPTURE_MANIFEST_CANONICAL")
    if canonical_without_self(proof) != proof.get("canonical_payload_sha256"):
        raise AttributionFailure("CAPTURE_PROOF_CANONICAL")
    if not (
        manifest.get("passed") is True
        and manifest.get("single_adapter_invocation") is True
        and manifest.get("production_operands_and_tendencies_same_invocation") is True
        and manifest.get("authority", {}).get("adapter_invocations") == 1
        and manifest.get("authority", {}).get("gpu_actions") == 0
        and manifest.get("authority", {}).get("wrf_or_mpi_executions") == 0
        and proof.get("passed") is True
    ):
        raise AttributionFailure("CAPTURE_AUTHORITY")
    declared = manifest.get("archive", {}).get("arrays")
    order = manifest.get("archive", {}).get("array_order")
    if not isinstance(declared, dict) or not isinstance(order, list):
        raise AttributionFailure("CAPTURE_ARRAY_MANIFEST")

    needed = {
        "entry_state_qke", "initialized_state_qke",
        "surface_terms_state_qv", "surface_terms_state_rho",
        "surface_terms_state_dz", "surface_terms_flux_ustar",
        "surface_terms_flux_theta_flux", "surface_terms_flux_qv_flux",
        "surface_terms_wind", "surface_terms_rhosfc",
        "mass_flux_s_aw", "mass_flux_s_awu", "mass_flux_s_awv",
        "turbulence_qke_input", "turbulence_el", "turbulence_dfm",
        "qke_predict_qke_after", "mean_tendencies_state_u",
        "mean_tendencies_state_v", "mean_tendencies_rho",
        "mean_tendencies_dz", "mean_tendencies_kmdz_raw",
        "mean_tendencies_kmdz_floored", "mean_tendencies_rhoz",
        "mean_tendencies_dtz", "mean_tendencies_rhoinv",
        "mean_tendencies_ustar", "mean_tendencies_wind",
        "mean_tendencies_rhosfc", "mean_tendencies_drag",
        "mean_tendencies_s_aw", "mean_tendencies_s_awu",
        "mean_tendencies_s_awv", "mean_tendencies_output_u",
        "mean_tendencies_output_v", "adapter_rublten", "adapter_rvblten",
        *(f"solve_{component}_{term}" for component in ("u", "v")
          for term in ("a", "b", "c", "d", "x")),
    }
    arrays: dict[str, np.ndarray] = {}
    with np.load(CAPTURE_ARCHIVE, allow_pickle=False) as archive:
        if archive.files != order or set(archive.files) != set(declared):
            raise AttributionFailure("CAPTURE_ARCHIVE_INVENTORY")
        for name in archive.files:
            value = finite(name, archive[name])
            observed = {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "nbytes": value.nbytes,
                "logical_c_bitpayload_sha256": array_sha(value),
            }
            if observed != declared[name]:
                raise AttributionFailure(f"CAPTURE_ARRAY_DRIFT:{name}")
            if name in needed:
                arrays[name] = np.ascontiguousarray(value)
    if set(arrays) != needed:
        raise AttributionFailure(f"CAPTURE_NEEDED:{sorted(needed-set(arrays))}")

    tree: dict[str, str] = {}
    for path in sorted(WRF_ROOT.rglob("*")):
        if path.is_symlink():
            raise AttributionFailure(f"WRF_TREE_SYMLINK:{path}")
        if path.is_file():
            tree[path.relative_to(WRF_ROOT).as_posix()] = sha256_file(path)
    tree_sha = hashlib.sha256(json.dumps(
        tree, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if len(tree) != 462 or tree_sha != EXPECTED_WRF_TREE:
        raise AttributionFailure(f"WRF_TREE_DRIFT:{len(tree)}:{tree_sha}")
    reader, reader_record = import_dump_reader()
    evidence = {
        "comparator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "git_head": git_head,
        },
        "files": records,
        "capture_manifest_canonical_sha256": canonical_without_self(manifest),
        "capture_array_count_validated": len(declared),
        "capture_array_count_loaded": len(arrays),
        "wrf_dump_tree": {
            "file_count": len(tree), "sha256": tree_sha,
        },
        "reader": reader_record,
    }
    return arrays, manifest, reader, evidence


def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray,
           d: np.ndarray) -> np.ndarray:
    if not (a.shape == b.shape == c.shape == d.shape):
        raise AttributionFailure("THOMAS_SHAPE")
    cp = np.empty_like(b, dtype=np.float64)
    dp = np.empty_like(b, dtype=np.float64)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for level in range(1, a.shape[0]):
        denominator = b[level] - cp[level - 1] * a[level]
        if not np.isfinite(denominator).all() or np.any(denominator == 0):
            raise AttributionFailure(f"THOMAS_SINGULAR:{level}")
        cp[level] = c[level] / denominator
        dp[level] = (d[level] - dp[level - 1] * a[level]) / denominator
    result = np.empty_like(d, dtype=np.float64)
    result[-1] = dp[-1]
    for level in range(a.shape[0] - 2, -1, -1):
        result[level] = dp[level] - cp[level] * result[level + 1]
    return finite("thomas", result)


def assemble(operands: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    state = np.asarray(operands["state"], dtype=np.float64)
    dtz = np.asarray(operands["dtz"], dtype=np.float64)
    rhoinv = np.asarray(operands["rhoinv"], dtype=np.float64)
    kmdz = np.asarray(operands["kmdz"], dtype=np.float64)
    drag = np.asarray(operands["drag"], dtype=np.float64)
    saw = np.asarray(operands["s_aw"], dtype=np.float64)
    sawx = np.asarray(operands["s_awx"], dtype=np.float64)
    if not (
        state.shape == dtz.shape == rhoinv.shape == (NZ, NY, NX)
        and kmdz.shape == saw.shape == sawx.shape == (NZ + 1, NY, NX)
        and drag.shape == (NY, NX)
    ):
        raise AttributionFailure("OPERAND_SHAPE")
    a = np.empty_like(state)
    b = np.empty_like(state)
    c = np.empty_like(state)
    d = np.empty_like(state)
    half0 = 0.5 * dtz[0] * rhoinv[0]
    a[0] = -dtz[0] * kmdz[0] * rhoinv[0]
    b[0] = (
        1.0 + dtz[0] * (kmdz[1] + kmdz[0] + drag) * rhoinv[0]
        - half0 * saw[1]
    )
    c[0] = -dtz[0] * kmdz[1] * rhoinv[0] - half0 * saw[1]
    d[0] = state[0] - dtz[0] * rhoinv[0] * sawx[1]
    interior = slice(1, NZ - 1)
    half = 0.5 * dtz[interior] * rhoinv[interior]
    a[interior] = (
        -dtz[interior] * kmdz[1:NZ - 1] * rhoinv[interior]
        + half * saw[1:NZ - 1]
    )
    b[interior] = (
        1.0 + dtz[interior]
        * (kmdz[1:NZ - 1] + kmdz[2:NZ]) * rhoinv[interior]
        + half * (saw[1:NZ - 1] - saw[2:NZ])
    )
    c[interior] = (
        -dtz[interior] * kmdz[2:NZ] * rhoinv[interior]
        - half * saw[2:NZ]
    )
    d[interior] = (
        state[interior] + dtz[interior] * rhoinv[interior]
        * (sawx[1:NZ - 1] - sawx[2:NZ])
    )
    a[-1] = 0.0
    b[-1] = 1.0
    c[-1] = 0.0
    d[-1] = state[-1]
    return {"a": a, "b": b, "c": c, "d": d}


def evaluate(operands: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    system = assemble(operands)
    x = thomas(system["a"], system["b"], system["c"], system["d"])
    return {
        **system,
        "x": x,
        "tendency": (x - np.asarray(operands["state"], dtype=np.float64)) / DT,
    }


def make_operands(capture: Mapping[str, np.ndarray], wrf: Mapping[str, np.ndarray],
                  component: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    lower = wrf["bc_lower_operands"]
    wrf_rho = wrf["bc_rho"]
    wrf_rhoinv = (
        np.float32(1.0) / np.maximum(wrf_rho, np.float32(1.0e-4))
    )
    wrf_drag = (
        lower[10] * lower[6] ** np.float32(2.0) / lower[7]
    ).astype(np.float32)
    port = {
        "state": capture[f"mean_tendencies_state_{component}"],
        "dtz": capture["mean_tendencies_dtz"],
        "rhoinv": capture["mean_tendencies_rhoinv"],
        "kmdz": capture["mean_tendencies_kmdz_floored"],
        "drag": capture["mean_tendencies_drag"],
        "s_aw": capture["mean_tendencies_s_aw"],
        "s_awx": capture[f"mean_tendencies_s_aw{component}"],
    }
    pristine = {
        "state": wrf[f"bc_{component}"],
        "dtz": wrf["bc_dtz"],
        "rhoinv": wrf_rhoinv,
        "kmdz": wrf["bc_kmdz"],
        "drag": wrf_drag,
        "s_aw": wrf["bc_s_aw"],
        "s_awx": wrf[f"bc_s_aw{component}"],
    }
    return (
        {key: finite(f"port_{key}", np.asarray(value)) for key, value in port.items()},
        {key: finite(f"wrf_{key}", np.asarray(value)) for key, value in pristine.items()},
    )


def select_operands(port: Mapping[str, np.ndarray], wrf: Mapping[str, np.ndarray],
                    port_categories: set[str]) -> dict[str, np.ndarray]:
    category_keys = {
        "entry_state": ("state",),
        "metric": ("dtz", "rhoinv"),
        "surface_drag": ("drag",),
        "mixing": ("kmdz",),
        "mass_flux": ("s_aw", "s_awx"),
    }
    selected: dict[str, np.ndarray] = {}
    for category, keys in category_keys.items():
        source = port if category in port_categories else wrf
        for key in keys:
            selected[key] = source[key]
    return selected


def mask_set(mask: int, names: tuple[str, ...]) -> set[str]:
    return {name for index, name in enumerate(names) if mask & (1 << index)}


def shapley_vectors(cache: Mapping[int, np.ndarray], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    count = len(names)
    denominator = math.factorial(count)
    result = {name: np.zeros_like(cache[0], dtype=np.float64) for name in names}
    for index, name in enumerate(names):
        bit = 1 << index
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size) * math.factorial(count - size - 1)
                / denominator
            )
            result[name] += weight * (cache[mask | bit] - cache[mask])
    return result


def coefficient_partition(port: Mapping[str, np.ndarray], wrf: Mapping[str, np.ndarray],
                          authentic_error: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache: dict[int, np.ndarray] = {}
    for mask in range(1 << len(CATEGORIES)):
        selected = select_operands(port, wrf, mask_set(mask, CATEGORIES))
        cache[mask] = evaluate(selected)["tendency"]
    vectors = shapley_vectors(cache, CATEGORIES)
    all_mask = (1 << len(CATEGORIES)) - 1
    formula_delta = cache[all_mask] - cache[0]
    vector_sum = sum(vectors.values(), np.zeros_like(formula_delta))
    records: dict[str, Any] = {}
    for index, name in enumerate(CATEGORIES):
        bit = 1 << index
        records[name] = {
            "shapley": projection(vectors[name], authentic_error),
            "standalone_port_in_all_wrf_context": projection(
                cache[bit] - cache[0], authentic_error,
            ),
            "leave_one_port_out_from_all_port_context": projection(
                cache[all_mask] - cache[all_mask ^ bit], authentic_error,
            ),
        }
    return {
        "method": (
            "Exact five-factor vector Shapley decomposition. A subset means "
            "that category uses same-authority port operands; all other "
            "categories use pristine WRF operands."
        ),
        "categories": list(CATEGORIES),
        "subset_count": len(cache),
        "all_port_minus_all_wrf_formula": projection(formula_delta, authentic_error),
        "shapley_additivity_closure": metrics(vector_sum, formula_delta),
        "contributions": records,
    }, vectors


def custom_two_factor(base: Mapping[str, np.ndarray], authentic_error: np.ndarray,
                      apply_first: Any, apply_second: Any,
                      names: tuple[str, str]) -> dict[str, Any]:
    cache: dict[int, np.ndarray] = {}
    for mask in range(4):
        operands = {key: np.array(value, copy=True) for key, value in base.items()}
        if mask & 1:
            apply_first(operands)
        if mask & 2:
            apply_second(operands)
        cache[mask] = evaluate(operands)["tendency"]
    vectors = shapley_vectors(cache, names)
    return {
        "factors": list(names),
        "all_port_minus_all_wrf": projection(cache[3] - cache[0], authentic_error),
        "additivity_closure": metrics(
            vectors[names[0]] + vectors[names[1]], cache[3] - cache[0]
        ),
        "contributions": {
            name: projection(vectors[name], authentic_error) for name in names
        },
    }


def mixing_detail(port: Mapping[str, np.ndarray], wrf: Mapping[str, np.ndarray],
                  authentic_error: np.ndarray) -> dict[str, Any]:
    base = select_operands(port, wrf, set())

    def bottom(operands: dict[str, np.ndarray]) -> None:
        operands["kmdz"][1] = port["kmdz"][1]

    def remaining(operands: dict[str, np.ndarray]) -> None:
        operands["kmdz"][:1] = port["kmdz"][:1]
        operands["kmdz"][2:] = port["kmdz"][2:]

    split = custom_two_factor(
        base, authentic_error, bottom, remaining,
        ("first_active_bottom_interface_kmdz_index_1", "all_other_kmdz_interfaces"),
    )
    per_interface = []
    baseline = evaluate(base)["tendency"]
    for level in range(NZ + 1):
        operands = {key: np.array(value, copy=True) for key, value in base.items()}
        operands["kmdz"][level] = port["kmdz"][level]
        effect = evaluate(operands)["tendency"] - baseline
        item = projection(effect, authentic_error)
        item["zero_based_interface"] = level
        item["operand_rms_delta"] = rms(
            port["kmdz"][level] - wrf["kmdz"][level]
        )
        per_interface.append(item)
    ranking = sorted(
        per_interface,
        key=lambda item: abs(item["signed_projection_fraction_of_authentic_error_sse"]),
        reverse=True,
    )
    return {
        "structural_bottom_interface_index_0": {
            "port_max_abs": float(np.max(np.abs(port["kmdz"][0]))),
            "wrf_max_abs": float(np.max(np.abs(wrf["kmdz"][0]))),
            "operand_comparison": metrics(port["kmdz"][0], wrf["kmdz"][0]),
            "note": "Index 0 is structurally present in the port but zero on both paths.",
        },
        "first_active_interface_index_1_operand": metrics(
            port["kmdz"][1], wrf["kmdz"][1]
        ),
        "two_factor_shapley_in_all_wrf_context": split,
        "top_interfaces_by_absolute_projection": ranking[:12],
        "all_interface_records": per_interface,
    }


def mass_flux_detail(port: Mapping[str, np.ndarray], wrf: Mapping[str, np.ndarray],
                     authentic_error: np.ndarray) -> dict[str, Any]:
    base = select_operands(port, wrf, set())

    def implicit(operands: dict[str, np.ndarray]) -> None:
        operands["s_aw"] = np.array(port["s_aw"], copy=True)

    def explicit(operands: dict[str, np.ndarray]) -> None:
        operands["s_awx"] = np.array(port["s_awx"], copy=True)

    return custom_two_factor(
        base, authentic_error, implicit, explicit,
        ("implicit_s_aw", "explicit_component_mass_flux"),
    )


def drag_detail(capture: Mapping[str, np.ndarray], wrf: Mapping[str, np.ndarray],
                port: Mapping[str, np.ndarray], pristine: Mapping[str, np.ndarray],
                authentic_error: np.ndarray) -> dict[str, Any]:
    lower = wrf["bc_lower_operands"]
    operands = ("rhosfc", "ustar", "wind")
    raw_port = {
        "rhosfc": capture["mean_tendencies_rhosfc"],
        "ustar": capture["mean_tendencies_ustar"],
        "wind": capture["mean_tendencies_wind"],
    }
    raw_wrf = {"rhosfc": lower[10], "ustar": lower[6], "wind": lower[7]}
    base = select_operands(port, pristine, set())
    cache: dict[int, np.ndarray] = {}
    drag_values: dict[int, np.ndarray] = {}
    for mask in range(8):
        selected = {
            name: raw_port[name] if mask & (1 << index) else raw_wrf[name]
            for index, name in enumerate(operands)
        }
        drag = selected["rhosfc"] * selected["ustar"] * selected["ustar"] / selected["wind"]
        current = {key: np.array(value, copy=True) for key, value in base.items()}
        current["drag"] = drag
        drag_values[mask] = drag
        cache[mask] = evaluate(current)["tendency"]
    vectors = shapley_vectors(cache, operands)
    drag_vectors = shapley_vectors(drag_values, operands)
    return {
        "raw_operands": {name: metrics(raw_port[name], raw_wrf[name]) for name in operands},
        "port_drag_reconstruction_vs_captured": metrics(drag_values[7], port["drag"]),
        "wrf_drag_reconstruction_vs_frozen_operand": metrics(drag_values[0], pristine["drag"]),
        "three_factor_shapley_in_all_wrf_context": {
            "all_port_minus_all_wrf": projection(cache[7] - cache[0], authentic_error),
            "tendency_contributions": {
                name: projection(vectors[name], authentic_error) for name in operands
            },
            "drag_operand_contributions": {
                name: {
                    "rms": rms(drag_vectors[name]),
                    "max_abs": float(np.max(np.abs(drag_vectors[name]))),
                } for name in operands
            },
            "additivity_closure": metrics(
                sum(vectors.values(), np.zeros_like(cache[0])), cache[7] - cache[0]
            ),
        },
    }


def source_evidence(capture: Mapping[str, np.ndarray], wrf: Mapping[str, np.ndarray]) -> dict[str, Any]:
    sources = {
        "wrf_mynn": WRF_SOURCE,
        "wrf_driver": WRF_DRIVER_SOURCE,
        "port_mynn": PORT_SOURCE,
        "port_coupler": COUPLER_SOURCE,
        "port_runtime": RUNTIME_SOURCE,
    }
    records = {}
    texts = {}
    for name, path in sources.items():
        if path.is_symlink() or not path.is_file():
            raise AttributionFailure(f"SOURCE_NOT_REGULAR:{path}")
        records[name] = {"path": str(path), "sha256": sha256_file(path)}
        texts[name] = path.read_text(encoding="utf-8")
    required = {
        "wrf_mynn": (
            "IF (initflag > 0 .and. .not.restart) THEN",
            "ELSE ! not cycling or restarting:",
            "INITIALIZE_QKE = .TRUE.",
            "qke1        =zero",
            "CALL mym_initialize",
        ),
        "port_coupler": (
            "if not first_timestep:",
            "restart: bool = False",
            "if restart:",
            "qke_seed = seed(None)",
        ),
        "port_runtime": ("first_timestep=jnp.equal(step_index, 1)",),
    }
    for name, needles in required.items():
        for needle in needles:
            if needle not in texts[name]:
                raise AttributionFailure(f"SOURCE_SYMBOL_DRIFT:{name}:{needle}")
    port_entry = capture["entry_state_qke"]
    port_initialized = capture["initialized_state_qke"]
    wrf_entry = wrf["driver_qke_entry"]
    wrf_initialized = wrf["mix_qke_initialized"]
    return {
        "bindings": records,
        "active_source_order": [
            "WRF driver records qke/rho/dz entry",
            "WRF initflag>0 and non-restart block clears qke",
            "WRF INITIALIZE_QKE true branch seeds and calls mym_initialize",
            "surface/EDMF/turbulence construct mass flux and dfm",
            "mynn_tendencies constructs and solves momentum systems",
        ],
        "first_semantic_divergence": {
            "symbol": "_mynn_state_with_first_call_qke",
            "wrf_active_behavior": (
                "At step-1 initflag>0, non-restart/non-cycling WRF unconditionally "
                "clears and cold-start initializes QKE, regardless of incoming max."
            ),
            "port_captured_behavior": (
                "The sealed pre-fix call had first_timestep=True but kept incoming "
                "QKE bit-exactly because the old code applied the cycling threshold."
            ),
            "current_source_fix": (
                "The current committed default is restart=False and unconditionally "
                "cold-starts QKE on the first call; explicit restart=True skips it."
            ),
            "port_entry_max": float(np.max(port_entry)),
            "wrf_entry_max": float(np.max(wrf_entry)),
            "threshold": 2.0e-4,
            "port_entry_vs_initialized": metrics(port_entry, port_initialized),
            "wrf_entry_vs_initialized": metrics(wrf_entry, wrf_initialized),
            "port_initialized_vs_wrf_initialized": vertical_metrics(
                port_initialized, wrf_initialized
            ),
            "source_authorized": True,
        },
        "earlier_input_ambiguity": {
            "rho_port_vs_wrf_driver": vertical_metrics(
                capture["mean_tendencies_rho"], wrf["driver_rho"]
            ),
            "meaning": (
                "The PBL-entry rho field already differs and is consumed without "
                "reinitialization. Its upstream producer is outside this sprint."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal", choices=sorted(TERMINALS), required=True)
    parser.add_argument("--focused-fix-proof", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        print(f"REFUSE:OUTPUT_NOT_FRESH:{output}", file=sys.stderr)
        return 74
    try:
        capture, manifest, reader, input_evidence = validate_inputs()
        focused_fix: dict[str, Any] | None = None
        if args.focused_fix_proof is not None:
            focused_path = args.focused_fix_proof.resolve()
            if focused_path.is_symlink() or not focused_path.is_file():
                raise AttributionFailure(f"FOCUSED_FIX_NOT_REGULAR:{focused_path}")
            focused_fix = json.loads(focused_path.read_text(encoding="utf-8"))
            if canonical_without_self(focused_fix) != focused_fix.get(
                "canonical_payload_sha256"
            ):
                raise AttributionFailure("FOCUSED_FIX_CANONICAL")
            if not (
                focused_fix.get("schema")
                == "wrfgpu2-v0234-gpt-qke-lifecycle-fix-proof-v1"
                and focused_fix.get("status")
                == "SOURCE_AUTHORIZED_FRESH_QKE_INIT_FIX_PROVEN"
                and focused_fix.get("terminal_eligible") is True
                and focused_fix.get("passed") is True
                and focused_fix.get("production_adapter_invocations") == 0
            ):
                raise AttributionFailure("FOCUSED_FIX_STATUS")
            current_coupler_sha = sha256_file(COUPLER_SOURCE)
            proven_coupler_sha = focused_fix.get("input_evidence", {}).get(
                "sources", {}
            ).get("port_coupler", {}).get("sha256")
            if proven_coupler_sha != current_coupler_sha:
                raise AttributionFailure("FOCUSED_FIX_SOURCE_DRIFT")
            input_evidence["focused_fix_proof"] = {
                "path": str(focused_path),
                "sha256": sha256_file(focused_path),
                "canonical_payload_sha256": focused_fix[
                    "canonical_payload_sha256"
                ],
                "status": focused_fix["status"],
            }
        if (
            args.terminal == "SINGLE_AUTHORITY_SOURCE_LOCALIZED_FIX_PROVEN"
            and focused_fix is None
        ):
            raise AttributionFailure("FIX_TERMINAL_WITHOUT_FOCUSED_PROOF")
        column_tags = (
            "bc_lower_operands", "bc_rho", "bc_dfm", "bc_u", "bc_v",
            "bc_dtz", "bc_rhoz", "bc_kmdz", "bc_s_aw", "bc_sd_aw",
            "bc_s_awu", "bc_sd_awu", "bc_s_awv", "bc_sd_awv",
            "bc_sub_u", "bc_det_u", "bc_sub_v", "bc_det_v",
            "driver_qke_entry", "driver_rho", "driver_dz",
            "mix_qke_initialized", "mix_qke_turbulence_input",
            "mix_qke_after_predict", "mix_el_for_dfm", "mix_dfm",
            *(f"solve_{component}_{term}" for component in ("u", "v")
              for term in ("a", "b", "c", "d", "x")),
        )
        wrf = {tag: reader.columns(tag) for tag in column_tags}
        for tag in ("ust", "hfx", "qfx", "tsk", "ch", "wspd"):
            wrf[f"surface_{tag}"] = load_outer2(reader, tag)
        wrf["rublten"] = reader.outer3("rublten_exit")
        wrf["rvblten"] = reader.outer3("rvblten_exit")

        thresholds = {
            "capture_formula_tendency_rms_max": 5.0e-13,
            "wrf_formula_tendency_rms_max": 2.0e-7,
            "wrf_formula_tendency_rms_over_authentic_error_max": 1.0e-3,
            "shapley_additivity_rms_max": 5.0e-15,
            "authentic_error_rms_min": 1.0e-5,
            "inactive_wrf_operand_max_abs": 0.0,
        }
        inactive = {
            tag: float(np.max(np.abs(wrf[tag])))
            for tag in (
                "bc_sd_aw", "bc_sd_awu", "bc_sd_awv", "bc_sub_u",
                "bc_det_u", "bc_sub_v", "bc_det_v",
            )
        }
        lower = wrf["bc_lower_operands"]
        inactive.update({
            "uoce": float(np.max(np.abs(lower[8]))),
            "voce": float(np.max(np.abs(lower[9]))),
        })
        if any(value > thresholds["inactive_wrf_operand_max_abs"]
               for value in inactive.values()):
            raise AttributionFailure(f"INACTIVE_OPERAND_NONZERO:{inactive}")
        if not np.array_equal(np.unique(lower[17]), np.array([1.0], dtype=np.float32)):
            raise AttributionFailure("MOMENTUM_MASS_FLUX_SWITCH")

        proof: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-gpt-single-authority-attribution-v1",
            "generated_utc": now(),
            "backend": "numpy-cpu",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_adapter_invocations_in_this_comparator": 0,
            "single_authority_capture_adapter_invocations": manifest["authority"]["adapter_invocations"],
            "thresholds": thresholds,
            "input_evidence": input_evidence,
            "operational_switches": {
                "dt_seconds": DT,
                "edmf": manifest["authority"]["edmf"],
                "sgs_cloud": manifest["authority"]["sgs_cloud"],
                "bl_mynn_edmf_mom_onoff_unique": [1.0],
                "inactive_wrf_operands_max_abs": inactive,
            },
            "components": {},
        }
        combined_error = []
        combined_vectors = {name: [] for name in CATEGORIES}
        all_pass = True
        for component, field in (("u", "rublten"), ("v", "rvblten")):
            port, pristine = make_operands(capture, wrf, component)
            port_eval = evaluate(port)
            wrf_eval = evaluate(pristine)
            authentic = capture[f"adapter_r{component}blten"] - wrf[field]
            capture_closure = {
                "coefficients": {
                    term: metrics(port_eval[term], capture[f"solve_{component}_{term}"])
                    for term in ("a", "b", "c", "d", "x")
                },
                "tendency": metrics(
                    port_eval["tendency"], capture[f"adapter_r{component}blten"]
                ),
            }
            wrf_closure = {
                "coefficients": {
                    term: metrics(wrf_eval[term], wrf[f"solve_{component}_{term}"])
                    for term in ("a", "b", "c", "d", "x")
                },
                "tendency": metrics(wrf_eval["tendency"], wrf[field]),
            }
            wrf_closure["tendency_rms_over_authentic_error_rms"] = (
                wrf_closure["tendency"]["rms"] / rms(authentic)
            )
            partition, vectors = coefficient_partition(port, pristine, authentic)
            component_pass = (
                capture_closure["tendency"]["rms"]
                <= thresholds["capture_formula_tendency_rms_max"]
                and wrf_closure["tendency"]["rms"]
                <= thresholds["wrf_formula_tendency_rms_max"]
                and wrf_closure["tendency_rms_over_authentic_error_rms"]
                <= thresholds[
                    "wrf_formula_tendency_rms_over_authentic_error_max"
                ]
                and partition["shapley_additivity_closure"]["rms"]
                <= thresholds["shapley_additivity_rms_max"]
                and rms(authentic) >= thresholds["authentic_error_rms_min"]
            )
            all_pass = all_pass and component_pass
            proof["components"][component] = {
                "authentic_same_authority_adapter_vs_pristine_wrf": vertical_metrics(
                    capture[f"adapter_r{component}blten"], wrf[field]
                ),
                "formula_closure": {
                    "all_port_vs_captured": capture_closure,
                    "all_wrf_port_formula_vs_pristine_wrf": wrf_closure,
                },
                "five_category_partition": partition,
                "bottom_interface_partition": mixing_detail(
                    port, pristine, authentic
                ),
                "surface_drag_partition": drag_detail(
                    capture, wrf, port, pristine, authentic
                ),
                "mass_flux_partition": mass_flux_detail(
                    port, pristine, authentic
                ),
                "passes": component_pass,
            }
            combined_error.append(authentic)
            for name in CATEGORIES:
                combined_vectors[name].append(vectors[name])

        error_both = np.stack(combined_error, axis=0)
        proof["combined_vector_shapley"] = {
            name: projection(np.stack(combined_vectors[name], axis=0), error_both)
            for name in CATEGORIES
        }
        proof["combined_vector_shapley_ranking_by_absolute_projection"] = sorted(
            CATEGORIES,
            key=lambda name: abs(proof["combined_vector_shapley"][name][
                "signed_projection_fraction_of_authentic_error_sse"
            ]),
            reverse=True,
        )

        qv0 = capture["surface_terms_state_qv"][0]
        cpm = 1004.5 * (1.0 + 0.84 * np.maximum(qv0, 0.0))
        port_hfx = (
            capture["surface_terms_flux_theta_flux"]
            * capture["surface_terms_rhosfc"] * cpm
        )
        port_qfx = (
            capture["surface_terms_flux_qv_flux"]
            * capture["surface_terms_rhosfc"]
        )
        proof["stage_differences"] = {
            "pbl_entry": {
                "rho": vertical_metrics(capture["mean_tendencies_rho"], wrf["driver_rho"]),
                "dz": vertical_metrics(capture["mean_tendencies_dz"], wrf["driver_dz"]),
                "u": vertical_metrics(capture["mean_tendencies_state_u"], wrf["bc_u"]),
                "v": vertical_metrics(capture["mean_tendencies_state_v"], wrf["bc_v"]),
                "qke": vertical_metrics(capture["entry_state_qke"], wrf["driver_qke_entry"]),
            },
            "qke_initialization_and_prediction": {
                "initialized_qke": vertical_metrics(
                    capture["initialized_state_qke"], wrf["mix_qke_initialized"]
                ),
                "turbulence_input_qke": vertical_metrics(
                    capture["turbulence_qke_input"], wrf["mix_qke_turbulence_input"]
                ),
                "predicted_qke": vertical_metrics(
                    capture["qke_predict_qke_after"], wrf["mix_qke_after_predict"]
                ),
            },
            "surface_span_raw_outputs": {
                "ustar": metrics(capture["surface_terms_flux_ustar"], wrf["surface_ust"]),
                "wind": metrics(capture["surface_terms_wind"], wrf["surface_wspd"]),
                "hfx_derived_w_m2": metrics(port_hfx, wrf["surface_hfx"]),
                "qfx_derived": metrics(port_qfx, wrf["surface_qfx"]),
                "rhosfc": metrics(capture["surface_terms_rhosfc"], lower[10]),
                "identifiability": (
                    "Direct momentum surface drag is separately partitioned. "
                    "HFX/QFX effects propagate into both mixing and mass flux; "
                    "one observational run cannot assign those mediated effects "
                    "again to surface without double counting."
                ),
            },
            "mixing_span": {
                "el": vertical_metrics(capture["turbulence_el"], wrf["mix_el_for_dfm"]),
                "dfm": vertical_metrics(capture["turbulence_dfm"], wrf["mix_dfm"]),
                "rhoz": vertical_metrics(capture["mean_tendencies_rhoz"], wrf["bc_rhoz"]),
                "kmdz_raw_vs_wrf_floored": vertical_metrics(
                    capture["mean_tendencies_kmdz_raw"], wrf["bc_kmdz"]
                ),
                "kmdz_operational_floored": vertical_metrics(
                    capture["mean_tendencies_kmdz_floored"], wrf["bc_kmdz"]
                ),
            },
            "mass_flux_span": {
                "s_aw": vertical_metrics(capture["mass_flux_s_aw"], wrf["bc_s_aw"]),
                "s_awu": vertical_metrics(capture["mass_flux_s_awu"], wrf["bc_s_awu"]),
                "s_awv": vertical_metrics(capture["mass_flux_s_awv"], wrf["bc_s_awv"]),
            },
        }
        proof["source_attribution"] = source_evidence(capture, wrf)
        proof["fix_authorization"] = {
            "candidate": "first-timestep non-restart QKE initialization branch",
            "same_authority_divergence_demonstrated": True,
            "pristine_wrf_target_present": True,
            "wrf_source_uniquely_specifies_active_branch": True,
            "sealed_namelist_restart_false": (
                focused_fix is not None
                and focused_fix.get("input_evidence", {}).get(
                    "namelist_receipt", {}
                ).get("restart_values") == [".false."]
            ),
            "upstream_ambiguity_material_to_full_sp2_closure": True,
            "upstream_ambiguity_material_to_qke_lifecycle_fix": False,
            "focused_fix_proof_present_in_this_object": focused_fix is not None,
            "authorized": focused_fix is not None,
            "meaning": (
                "The focused proof binds restart=.false., the pristine WRF branch, "
                "the identical sealed PBL-entry state, a before/after initialized-"
                "QKE comparison, and green lifecycle tests. It proves only the "
                "narrow QKE lifecycle fix; downstream full-SP2 closure remains open."
                if focused_fix is not None else
                "No focused before/after proof was supplied, so a fix terminal is "
                "not authorized."
            ),
        }
        proof["passes"] = bool(all_pass)
        proof["terminal_verdict"] = args.terminal
        if (
            args.terminal == "SINGLE_AUTHORITY_SOURCE_LOCALIZED_FIX_PROVEN"
            and not proof["fix_authorization"]["authorized"]
        ):
            raise AttributionFailure("FIX_TERMINAL_NOT_AUTHORIZED")
        if not all_pass:
            raise AttributionFailure("ATTRIBUTION_GATE_FAILED")
        atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "output": str(output),
            "terminal_verdict": args.terminal,
            "authentic_rms": {
                component: proof["components"][component][
                    "authentic_same_authority_adapter_vs_pristine_wrf"
                ]["rms"] for component in ("u", "v")
            },
            "combined_ranking": proof[
                "combined_vector_shapley_ranking_by_absolute_projection"
            ],
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True))
        return 0
    except (AttributionFailure, OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
