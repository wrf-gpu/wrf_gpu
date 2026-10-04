#!/usr/bin/env python3
"""Prove the v0234 d03 land-TSK humidity provenance correction.

The proof starts from the authenticated post-microphysics/pre-PBL seam and the
authenticated step-0 Noah-MP carry.  It audits the persisted land state against
the pristine WRF t=0 history, reproduces WRF's QVAPOR mixing-ratio to Noah-MP
specific-humidity expression, and runs a paired CPU Noah-MP replay in which the
only changed forcing leaf is QAIR.  The emitted NPZ contains the exact paired
surface-state deltas used to remeasure the frozen MYNN adapter without importing
the unrelated CPU-vs-GPU Noah-MP arithmetic offset.

No GPU, WRF, MPI, empirical tuning, mask-selected correction, or synthetic WRF
reference is used.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Any, Mapping

from netCDF4 import Dataset
import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
SEAM = STAGED / "seam-capture"
INPUTS = STAGED / "inputs"
WRF_ROOT = STAGED / "wrf-dumps"
WRF_SOURCE = STAGED / "wrf-source/pristine"
WRFINPUT = INPUTS / "wrfinput_d03"
WRFOUT0 = INPUTS / "wrfout_d03_2025-03-01_00:00:00"
CARRY = INPUTS / "last-healthy-d03-step-0.pkl"
WRF_DRIVER = WRF_SOURCE / "phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F"
WRF_NOAHMP = WRF_SOURCE / "phys/noahmp/src/module_sf_noahmplsm.F"
WRF_TABLES = WRF_SOURCE / "phys/noahmp/parameters"
OLD_ENERGY_PROOF = REPO / "proofs/v014/noahmp_land_tile_energy_closure.json"
RRTMG_PROOF = REPO / "proofs/v014/rrtmg_step1_forcing_parity.json"
PRIOR_SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
PRIOR_CAPTURE_PROOF = PRIOR_SPRINT / "tskin-capture-proof.json"
PRIOR_CAPTURE = STAGED / "surface-fixed-cpu-adapter/tskin-v1/single-authority-capture.npz"
AUDIT_READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"
OUTER_READER = REPO / "scripts/v0234_gpt_operand_attribution.py"

EXPECTED = {
    "seam_manifest": "ae584d25d0e8fc56afedc4a1745f9109d52fecf76635007324180f40bb709873",
    "seam_canonical": "f6d14eee33d1f43364238c9a42849cb0993aaa71fa9b660c397d0c3ee7096d6a",
    "carry": "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
    "wrfout0": "658ff1dc8d1ee0d0ef2ed98860d50dd81f68eb256ad8a6778f9a032c6210a114",
    "wrf_tree": "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b",
    "wrf_driver": "29f16622740d7388a17ff9ba6c2f4640487e44f3f26b8c4bdd47aad42e0a4572",
    "wrf_noahmp": "74b475600ad1c999fe43299bcaf4d3e5eaf59b9f861c47fdbff4d69739e3a45b",
    "MPTABLE.TBL": "7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d",
    "SOILPARM.TBL": "1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f",
    "GENPARM.TBL": "9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2",
    "old_energy_proof": "8d30ea2aa37f34d010fa9acbb791e6f24acc2b467cabcb92b64ab86927fae465",
    "rrtmg_proof": "20b414f7b8a3ac3e258555763e516d121c64f0f83f70afee6ae96a4b6dfcd613",
    "prior_capture_proof": "f1e6a568b6636461d7906936951f136c389327047ccbd04b32626d72d2c250f7",
    "prior_capture": "880ac4d84258d3f8fccc7af3aab86ede89511a05c4be4d8aa9549958fc6fd94c",
}

SOURCE_FILES = (
    "src/gpuwrf/physics/noahmp_coupler.py",
    "src/gpuwrf/physics/noahmp/types.py",
    "src/gpuwrf/coupling/noahmp_surface_hook.py",
    "src/gpuwrf/io/land_state.py",
    "src/gpuwrf/io/noahmp_land_init.py",
    "src/gpuwrf/physics/noahmp/noahmp_driver.py",
    "scripts/v0234_gpt_adversarial_audit.py",
    "scripts/v0234_gpt_operand_attribution.py",
    "scripts/v0234_gpt_land_tsk_provenance.py",
)
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class ProvenanceFailure(RuntimeError):
    """Fail-closed provenance or scientific-gate failure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def canonical_without_self(value: Mapping[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ProvenanceFailure(f"OUTPUT_NOT_FRESH:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise ProvenanceFailure(f"DELTA_NOT_FRESH:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        f"{name}_delta": np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        for name, value in arrays.items()
    }
    if set(clean) != {f"{name}_delta" for name in DELTA_FIELDS}:
        raise ProvenanceFailure(f"DELTA_SCHEMA:{sorted(clean)}")
    for name, value in clean.items():
        if value.shape != (93, 111) or not np.isfinite(value).all():
            raise ProvenanceFailure(f"DELTA_ARRAY:{name}:{value.shape}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez(temporary, **clean)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "logical_c_bitpayload_sha256": array_sha(value),
            }
            for name, value in clean.items()
        },
    }


def git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], check=False,
        text=not binary, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        stderr = result.stderr.decode() if binary else result.stderr
        raise ProvenanceFailure(f"GIT:{' '.join(args)}:{stderr.strip()}")
    return result.stdout if binary else result.stdout.strip()


def tracked_record(relative: str) -> dict[str, Any]:
    path = REPO / relative
    if path.is_symlink() or not path.is_file():
        raise ProvenanceFailure(f"SOURCE_NOT_REGULAR:{relative}")
    disk = path.read_bytes()
    head = git("show", f"HEAD:{relative}", binary=True)
    if disk != head:
        raise ProvenanceFailure(f"SOURCE_NOT_HEAD:{relative}")
    return {
        "path": str(path),
        "sha256": sha256_bytes(disk),
        "git_blob": git("rev-parse", f"HEAD:{relative}"),
        "size": len(disk),
    }


def resource_gate() -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    environment = {key: os.environ.get(key) for key in THREAD_ENV}
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if affinity != ALLOWED_CPUS:
        raise ProvenanceFailure(f"CPUSET:{sorted(affinity)}")
    if environment != THREAD_ENV:
        raise ProvenanceFailure(f"THREAD_ENV:{environment}")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ProvenanceFailure("JAX_PLATFORM")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProvenanceFailure("CUDA_VISIBILITY")
    if os.environ.get("JAX_ENABLE_X64", "").lower() not in {"1", "true"}:
        raise ProvenanceFailure("JAX_X64")
    if os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "").lower() != "false":
        raise ProvenanceFailure("JAX_COMPILATION_CACHE")
    if nice < 15 or ioprio < 0 or int(ioprio) >> 13 != 3:
        raise ProvenanceFailure(f"PRIORITY:{nice}:{ioprio}")
    if Path("/tmp/PREEMPT_CPU").exists():
        raise ProvenanceFailure("PREEMPT_CPU")
    return {
        "logical_cpu_ids": sorted(affinity),
        "thread_environment": environment,
        "jax_platforms": "cpu",
        "cuda_visible_devices": "",
        "jax_enable_x64": os.environ["JAX_ENABLE_X64"],
        "jax_enable_compilation_cache": False,
        "nice": nice,
        "ionice_class": int(ioprio) >> 13,
        "preempt_cpu_absent": True,
    }


def _selected(value: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    array = np.asarray(value)
    if mask is None:
        return array.reshape(-1)
    if array.shape[-2:] != mask.shape:
        raise ProvenanceFailure(f"MASK_SHAPE:{array.shape}:{mask.shape}")
    return array[..., mask].reshape(-1)


def metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    left = _selected(np.asarray(candidate), mask)
    right = _selected(np.asarray(reference), mask)
    if (
        left.shape != right.shape
        or left.dtype.hasobject
        or right.dtype.hasobject
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ProvenanceFailure(f"METRIC_INPUT:{left.shape}:{right.shape}")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "count": int(left.size),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "signed_bias": float(np.mean(delta)),
        "bitwise_mismatch_count": int(np.sum(left != right)),
        "exact_fraction": float(np.mean(left == right)),
        "candidate_dtype": left.dtype.str,
        "reference_dtype": right.dtype.str,
    }


def split_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    land: np.ndarray,
) -> dict[str, Any]:
    return {
        "all": metrics(candidate, reference),
        "land": metrics(candidate, reference, land),
        "water": metrics(candidate, reference, ~land),
    }


def float32_ulp(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    left = _selected(np.asarray(candidate, dtype=np.float32), mask)
    right = _selected(np.asarray(reference, dtype=np.float32), mask)
    # TSK is strictly positive, so IEEE-754 uint ordering is monotonic.
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ProvenanceFailure("ULP_NONPOSITIVE")
    distance = np.abs(
        left.view(np.uint32).astype(np.int64)
        - right.view(np.uint32).astype(np.int64)
    )
    edges = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    histogram: dict[str, int] = {}
    for index, lower in enumerate(edges):
        upper = edges[index + 1] if index + 1 < len(edges) else None
        key = f"{lower}" if upper == lower + 1 else (
            f"{lower}-{upper - 1}" if upper is not None else f">={lower}"
        )
        selected = distance == lower if upper == lower + 1 else (
            distance >= lower if upper is None else (distance >= lower) & (distance < upper)
        )
        histogram[key] = int(np.sum(selected))
    return {
        "storage_dtype": "<f4",
        "count": int(distance.size),
        "max_ulp": int(np.max(distance)),
        "median_ulp": float(np.median(distance)),
        "histogram": histogram,
    }


def validate_static_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ProvenanceFailure("TRACKED_WORKTREE_NOT_CLEAN")
    sources = {relative: tracked_record(relative) for relative in SOURCE_FILES}
    fixed_files = {
        "carry": CARRY,
        "wrfinput": WRFINPUT,
        "wrfout0": WRFOUT0,
        "wrf_driver": WRF_DRIVER,
        "wrf_noahmp": WRF_NOAHMP,
        "old_energy_proof": OLD_ENERGY_PROOF,
        "rrtmg_proof": RRTMG_PROOF,
        "prior_capture_proof": PRIOR_CAPTURE_PROOF,
        "prior_capture": PRIOR_CAPTURE,
    }
    for label, path in fixed_files.items():
        if path.is_symlink() or not path.is_file():
            raise ProvenanceFailure(f"INPUT_NOT_REGULAR:{label}:{path}")
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise ProvenanceFailure(f"INPUT_HASH:{label}:{actual}")
    tables: dict[str, str] = {}
    for name in ("MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"):
        path = WRF_TABLES / name
        actual = sha256_file(path)
        if actual != EXPECTED[name]:
            raise ProvenanceFailure(f"TABLE_HASH:{name}:{actual}")
        tables[name] = actual
    tree: dict[str, str] = {}
    for path in sorted(WRF_ROOT.rglob("*")):
        if path.is_symlink():
            raise ProvenanceFailure(f"WRF_TREE_SYMLINK:{path}")
        if path.is_file():
            tree[path.relative_to(WRF_ROOT).as_posix()] = sha256_file(path)
    tree_sha = sha256_bytes(json.dumps(
        tree, sort_keys=True, separators=(",", ":"),
    ).encode())
    if len(tree) != 462 or tree_sha != EXPECTED["wrf_tree"]:
        raise ProvenanceFailure(f"WRF_TREE:{len(tree)}:{tree_sha}")
    manifest_path = SEAM / "manifest.json"
    if sha256_file(manifest_path) != EXPECTED["seam_manifest"]:
        raise ProvenanceFailure("SEAM_MANIFEST_HASH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        canonical_without_self(manifest) == EXPECTED["seam_canonical"]
        and manifest.get("canonical_payload_sha256") == EXPECTED["seam_canonical"]
        and manifest.get("status") == "CAPTURE_COMPLETE_HOLD_5_OF_28"
        and manifest.get("seam", {}).get("trace_invocation_count") == 1
        and manifest.get("seam", {}).get("pbl_guard_invocation_count") == 0
        and manifest.get("provenance", {}).get("synthetic_reconstruction") is False
    ):
        raise ProvenanceFailure("SEAM_AUTHORITY")
    prior = json.loads(PRIOR_CAPTURE_PROOF.read_text(encoding="utf-8"))
    if not (
        prior.get("passed") is True
        and prior.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and prior.get("namespace_consumed_once") is True
        and prior.get("authority", {}).get("adapter_invocations") == 1
        and prior.get("authority", {}).get("wrf_or_mpi_executions") == 0
        and prior.get("authority", {}).get("gpu_actions") == 0
        and prior.get("authority", {}).get("backend") == "cpu"
    ):
        raise ProvenanceFailure("PRIOR_CAPTURE_AUTHORITY")
    return manifest, {
        "sources": sources,
        "authority": {
            "seam_manifest_sha256": EXPECTED["seam_manifest"],
            "seam_manifest_canonical_sha256": EXPECTED["seam_canonical"],
            "carry_sha256": EXPECTED["carry"],
            "wrfinput_d03_sha256": EXPECTED["wrfinput"],
            "wrfout_t0_sha256": EXPECTED["wrfout0"],
            "wrf_dump_tree": {"file_count": len(tree), "sha256": tree_sha},
            "pristine_wrf_driver_sha256": EXPECTED["wrf_driver"],
            "pristine_wrf_noahmp_sha256": EXPECTED["wrf_noahmp"],
            "pristine_table_sha256": tables,
            "prior_capture_sha256": EXPECTED["prior_capture"],
            "prior_capture_proof_sha256": EXPECTED["prior_capture_proof"],
        },
    }


def load_seam_state(manifest: Mapping[str, Any], State: Any, jax: Any) -> Any:
    values: list[Any] = []
    for expected_index, item in enumerate(manifest.get("state_leaves", [])):
        if item.get("slot_index") != expected_index:
            raise ProvenanceFailure(f"SEAM_SLOT_ORDER:{expected_index}")
        if item.get("kind") == "none":
            values.append(None)
            continue
        path = SEAM / item["file"]
        if path.is_symlink() or sha256_file(path) != item["file_sha256"]:
            raise ProvenanceFailure(f"SEAM_LEAF_FILE:{item['slot_name']}")
        value = np.load(path, allow_pickle=False)
        if not (
            list(value.shape) == item["shape"]
            and value.dtype.str == item["dtype"]
            and array_sha(value) == item["logical_c_bitpayload_sha256"]
        ):
            raise ProvenanceFailure(f"SEAM_LEAF_RECORD:{item['slot_name']}")
        values.append(jax.device_put(value))
    if len(values) != len(State.__slots__):
        raise ProvenanceFailure(f"SEAM_STATE_SCHEMA:{len(values)}")
    return State.tree_unflatten(None, tuple(values))


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProvenanceFailure(f"IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_wrf_tsk() -> np.ndarray:
    audit = import_file("v0234_land_tsk_dump_reader", AUDIT_READER)
    outer = import_file("v0234_land_tsk_outer_reader", OUTER_READER)
    outer.WRF_ROOT = WRF_ROOT
    reader = audit.IndependentWrfDump(WRF_ROOT)
    value = np.asarray(outer.load_outer2(reader, "tsk"))
    if value.shape != (93, 111) or value.dtype != np.dtype(np.float32):
        raise ProvenanceFailure(f"WRF_TSK_SCHEMA:{value.shape}:{value.dtype}")
    return value


def initial_land_audit(land_state: Any, land: np.ndarray) -> dict[str, Any]:
    mapping = {
        "t_skin": "TSK",
        "tslb": "TSLB",
        "smois": "SMOIS",
        "sh2o": "SH2O",
        "isnow": "ISNOW",
        "tsno": "TSNO",
        "snice": "SNICE",
        "snliq": "SNLIQ",
        "snowh": "SNOWH",
        "sneqv": "SNOW",
        "sneqvo": "SNEQVO",
        "tauss": "TAUSS",
        "tv": "TV",
        "tg": "TG",
        "tah": "TAH",
        "eah": "EAH",
        "canliq": "CANLIQ",
        "canice": "CANICE",
        "fwet": "FWET",
        "zsnso": "ZSNSO",
        "albold": "ALBOLD",
    }
    result: dict[str, Any] = {}
    with Dataset(WRFOUT0) as dataset:
        for field, variable in mapping.items():
            reference = np.asarray(dataset[variable][0])
            candidate = np.asarray(getattr(land_state, field)).astype(reference.dtype)
            result[field] = metrics(candidate, reference, land)
    material = tuple(mapping)
    exact = all(result[name]["bitwise_mismatch_count"] == 0 for name in material)
    return {
        "comparison_storage_dtype": "cast port carry to each WRF history variable dtype",
        "material_energy_fields": result,
        "all_material_energy_fields_bitwise_exact_over_land": exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--delta-output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    delta_output = args.delta_output.resolve()
    try:
        resources = resource_gate()
        manifest, static_evidence = validate_static_inputs()

        sys.path.insert(0, str(REPO / "src"))
        import jax
        import jax.numpy as jnp
        from gpuwrf.contracts.state import State
        from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
        from gpuwrf.dynamics.metrics import load_wrfinput_metrics
        from gpuwrf.io.gen2_accessor import Gen2Run
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
        from gpuwrf.physics.noah_mp import (
            mavail_from_prescribed_fields,
            roughness_from_prescribed_fields,
        )
        from gpuwrf.physics.noahmp_coupler import (
            assemble_noahmp_forcing,
            noahmp_surface_adapter,
        )
        from gpuwrf.runtime.operational_mode import _NoahMPClock, _NoahMPRadiation

        if jax.default_backend() != "cpu" or any(
            device.platform != "cpu" for device in jax.devices()
        ):
            raise ProvenanceFailure("NON_CPU_BACKEND")
        if not bool(jax.config.jax_enable_x64):
            raise ProvenanceFailure("JAX_X64_DISABLED")

        seam_state = load_seam_state(manifest, State, jax)
        with CARRY.open("rb") as stream:
            carry = pickle.load(stream)
        if type(carry).__name__ != "OperationalCarry" or not hasattr(carry, "noahmp_land"):
            raise ProvenanceFailure(f"CARRY_SCHEMA:{type(carry)}")

        run = Gen2Run(INPUTS)
        grid = dataclass_replace(
            run.grid("d03").as_grid_spec(),
            metrics=load_wrfinput_metrics(WRFINPUT),
        )
        with Dataset(WRFINPUT) as dataset:
            landmask = np.asarray(dataset["LANDMASK"][0], dtype=np.float64)
            lu_index = np.asarray(dataset["LU_INDEX"][0], dtype=np.int32)
            wrf_xland = np.asarray(dataset["XLAND"][0], dtype=np.float64)
        land = wrf_xland < 1.5
        if (
            int(np.sum(land)) != 2034
            or metrics(np.asarray(seam_state.xland), wrf_xland)["max_abs"] != 0.0
        ):
            raise ProvenanceFailure("LAND_AUTHORITY")

        # The frozen seam predates the already accepted winter landuse fix. Build
        # the current source-authorized step-0 surface fields before the paired
        # replay; both A/B arms receive these identical values.
        winter_z0 = np.asarray(roughness_from_prescribed_fields(
            carry.state.xland, landmask, lu_index=lu_index, season=2,
        ))
        winter_mavail = np.asarray(mavail_from_prescribed_fields(
            carry.state.xland, landmask, carry.state.soil_moisture,
            lu_index=lu_index, season=2,
        ))
        replay_updates = {
            name: getattr(carry.state, name)
            for name in ("t_skin", "soil_moisture", "xland", "lakemask", "ustar")
        }
        replay_updates.update(
            roughness_m=jnp.asarray(winter_z0),
            mavail=jnp.asarray(winter_mavail),
        )
        replay_state = seam_state.replace(**replay_updates)

        _built_land, noahmp_static, init_meta = build_noahmp_land_state(
            INPUTS, "d03", table_dir=WRF_TABLES,
        )
        energy_params, rad_params, nroot = build_noahmp_params(noahmp_static)
        init_audit = initial_land_audit(carry.noahmp_land, land)
        if not init_audit["all_material_energy_fields_bitwise_exact_over_land"]:
            raise ProvenanceFailure("INITIAL_LAND_MATERIAL_PARITY")

        view = _build_column_view(replay_state, grid)
        radiation = _NoahMPRadiation(*carry.noahmp_rad)
        clock = _NoahMPClock(julian=60.0, yearlen=365.0)
        corrected_forcing = assemble_noahmp_forcing(
            view, noahmp_static, radiation, clock, 6.0,
        )
        qv_mixing = np.asarray(view.qv, dtype=np.float64)[..., 0]
        wrf_specific = qv_mixing / (1.0 + qv_mixing)
        corrected_qair = np.asarray(corrected_forcing.qair, dtype=np.float64)
        if not np.array_equal(corrected_qair, wrf_specific):
            raise ProvenanceFailure("WRF_QML_EXPRESSION")
        old_forcing = corrected_forcing._replace(qair=jnp.asarray(qv_mixing))

        forcing_leaves = corrected_forcing._fields
        unchanged_forcing: dict[str, Any] = {}
        for name in forcing_leaves:
            old_value = getattr(old_forcing, name)
            new_value = getattr(corrected_forcing, name)
            if old_value is None or new_value is None:
                if old_value is not new_value:
                    raise ProvenanceFailure(f"FORCING_NONE:{name}")
                continue
            record = metrics(np.asarray(new_value), np.asarray(old_value))
            if name != "qair" and record["bitwise_mismatch_count"] != 0:
                raise ProvenanceFailure(f"FORCING_AB_NOT_SINGLE_LEAF:{name}")
            unchanged_forcing[name] = record

        def replay(forcing: Any) -> dict[str, np.ndarray]:
            view_out, _land_out, blended = noahmp_surface_adapter(
                view, carry.noahmp_land, noahmp_static,
                radiation=radiation, clock=clock, dt=6.0, forcing=forcing,
                energy_params=energy_params, rad_params=rad_params,
                first_timestep=True,
            )
            return {
                "theta_flux": np.asarray(blended.theta_flux),
                "qv_flux": np.asarray(blended.qv_flux),
                "fltv": np.asarray(blended.fltv),
                "t_skin": np.asarray(view_out.t_skin),
                "ustar": np.asarray(blended.ustar),
                "tau_u": np.asarray(blended.tau_u),
                "tau_v": np.asarray(blended.tau_v),
                "rhosfc": np.asarray(blended.rhosfc),
                "roughness_m": np.asarray(view_out.roughness_m),
            }

        old = replay(old_forcing)
        corrected = replay(corrected_forcing)
        deltas = {
            name: np.asarray(corrected[name], dtype=np.float64)
            - np.asarray(old[name], dtype=np.float64)
            for name in DELTA_FIELDS
        }
        invariance: dict[str, Any] = {}
        for name in corrected:
            record = split_metrics(corrected[name], old[name], land)
            invariance[name] = record
            if record["water"]["bitwise_mismatch_count"] != 0:
                raise ProvenanceFailure(f"WATER_CHANGED:{name}")
            if name not in DELTA_FIELDS and record["all"]["bitwise_mismatch_count"] != 0:
                raise ProvenanceFailure(f"NON_QAIR_SURFACE_CHANGED:{name}")

        wrf_tsk = load_wrf_tsk()
        frozen_tsk = np.asarray(seam_state.t_skin, dtype=np.float64)
        counterfactual = {
            name: np.asarray(getattr(seam_state, name), dtype=np.float64) + deltas[name]
            for name in DELTA_FIELDS
        }
        frozen_tsk_metrics = split_metrics(frozen_tsk, wrf_tsk, land)
        old_replay_metrics = split_metrics(old["t_skin"], wrf_tsk, land)
        new_replay_metrics = split_metrics(corrected["t_skin"], wrf_tsk, land)
        counterfactual_tsk_metrics = split_metrics(counterfactual["t_skin"], wrf_tsk, land)
        backend_offset = split_metrics(old["t_skin"], frozen_tsk, land)
        paired_shift = split_metrics(corrected["t_skin"], old["t_skin"], land)
        if not (
            frozen_tsk_metrics["land"]["rms"] == 0.02123018786362896
            and frozen_tsk_metrics["land"]["max_abs"] == 0.05415733031958325
            and frozen_tsk_metrics["water"]["bitwise_mismatch_count"] == 0
            and new_replay_metrics["land"]["rms"] < old_replay_metrics["land"]["rms"]
            and new_replay_metrics["land"]["max_abs"] < old_replay_metrics["land"]["max_abs"]
            and counterfactual_tsk_metrics["land"]["rms"] < frozen_tsk_metrics["land"]["rms"]
            and counterfactual_tsk_metrics["land"]["max_abs"] < frozen_tsk_metrics["land"]["max_abs"]
            and counterfactual_tsk_metrics["water"]["bitwise_mismatch_count"] == 0
        ):
            raise ProvenanceFailure("TSK_STRICT_IMPROVEMENT")

        # Bind the correction target to the exact entry handles used by the
        # immediately preceding authenticated adapter capture.
        prior_entry: dict[str, Any] = {}
        capture_names = {
            "theta_flux": "entry_state_theta_flux",
            "qv_flux": "entry_state_qv_flux",
            "fltv": "entry_state_fltv",
            "t_skin": "surface_terms_flux_t_skin",
        }
        with np.load(PRIOR_CAPTURE, allow_pickle=False) as archive:
            for name, archive_name in capture_names.items():
                if archive_name not in archive.files:
                    raise ProvenanceFailure(f"PRIOR_CAPTURE_FIELD:{archive_name}")
                prior_entry[name] = metrics(
                    np.asarray(archive[archive_name]), np.asarray(getattr(seam_state, name)),
                )
                if prior_entry[name]["bitwise_mismatch_count"] != 0:
                    raise ProvenanceFailure(f"PRIOR_ENTRY_NOT_SEAM:{name}")

        driver_text = WRF_DRIVER.read_text(encoding="utf-8")
        noahmp_text = WRF_NOAHMP.read_text(encoding="utf-8")
        qml_pattern = r"Q_ML\s*=\s*QV3D\(I,1,J\)/\(1\.0\+QV3D\(I,1,J\)\)"
        if re.search(qml_pattern, driver_text) is None:
            raise ProvenanceFailure("WRF_DRIVER_QML_SOURCE")
        if "QAIR   !specific humidity" not in noahmp_text:
            raise ProvenanceFailure("WRF_NOAHMP_QAIR_SEMANTICS")
        old_energy = json.loads(OLD_ENERGY_PROOF.read_text(encoding="utf-8"))
        old_hook_qair = old_energy["jax_forcing_vs_wrf_nmpin"]["qair"]
        if not (
            old_energy.get("verdict")
            == "NOAHMP_LAND_TILE_ENERGY_CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING"
            and old_hook_qair["rmse"] > 0.0
        ):
            raise ProvenanceFailure("PINNED_NMPIN_QAIR_AUTHORITY")
        rrtmg = json.loads(RRTMG_PROOF.read_text(encoding="utf-8"))

        artifact = atomic_npz(delta_output, deltas)
        proof = {
            "schema": "wrfgpu2-v0234-gpt-land-tsk-provenance-v1",
            "verdict": "WRF_QML_SPECIFIC_HUMIDITY_TSK_CANDIDATE_PROVEN",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "backend": "jax-cpu-and-numpy-cpu",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "production_pbl_adapter_invocations": 0,
            "git": {
                "head": git("rev-parse", "HEAD"),
                "tree": git("rev-parse", "HEAD^{tree}"),
                "branch": git("branch", "--show-current"),
                "tracked_worktree_clean": True,
            },
            "resource_gate": resources,
            **static_evidence,
            "lifecycle": {
                "initialization": {
                    "source": str(WRFOUT0),
                    "audit": init_audit,
                    "noahmp_init_meta": init_meta,
                    "nroot": int(nroot),
                    "land_cells": int(np.sum(land)),
                    "water_cells": int(np.sum(~land)),
                    "active_landuse_categories": sorted(
                        int(value) for value in np.unique(lu_index[land])
                    ),
                    "season": 2,
                    "julian_day": 60,
                },
                "port_update_order": {
                    "source": "src/gpuwrf/runtime/operational_mode.py:_physics_step_forcing",
                    "order": ["microphysics", "held radiation refresh", "Noah-MP surface", "MYNN PBL"],
                    "authenticated_seam": "after microphysics_and_surface_land; before mynn_adapter_with_source_leaves",
                },
                "wrf_update_order": {
                    "source": "dyn_em/module_first_rk_step_part1.F",
                    "order": ["phy_prep/radiation", "surface_driver/Noah-MP", "pbl_driver/MYNN"],
                },
            },
            "first_source_authorized_divergence": {
                "wrf_driver_location": "phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F:756",
                "wrf_expression": "Q_ML = QV3D/(1.0+QV3D)",
                "wrf_noahmp_semantics_location": "phys/noahmp/src/module_sf_noahmplsm.F:659,1111,1147",
                "wrf_noahmp_semantics": "QAIR is specific humidity; driver already converts Q2",
                "old_port_expression": "qair = QVAPOR mixing ratio",
                "corrected_port_expression": "qair = qv_mixing_ratio/(1+qv_mixing_ratio)",
                "pinned_d02_wrf_nmpin_old_port_qair_residual": old_hook_qair,
                "d03_expression_replay": {
                    "old_mixing_ratio_vs_wrf_specific": split_metrics(qv_mixing, wrf_specific, land),
                    "corrected_qair_vs_wrf_specific": split_metrics(corrected_qair, wrf_specific, land),
                },
                "mynn_density_separation": {
                    "rule": "MYNN density retains atmospheric QVAPOR mixing ratio",
                    "location": "phys/MYNN-EDMF/module_bl_mynnedmf.F90:3960",
                    "unchanged_rhosfc": invariance["rhosfc"],
                },
            },
            "paired_noahmp_replay": {
                "common_inputs": {
                    "state": "authenticated seam atmosphere plus source-proven winter step-0 surface fields",
                    "land_carry": "authenticated last-healthy d03 step-0 OperationalCarry.noahmp_land",
                    "radiation": {
                        "source": "authenticated OperationalCarry.noahmp_rad",
                        "soldn_sha256": array_sha(np.asarray(carry.noahmp_rad[0])),
                        "lwdn_sha256": array_sha(np.asarray(carry.noahmp_rad[1])),
                        "cosz_sha256": array_sha(np.asarray(carry.noahmp_rad[2])),
                    },
                    "dt_seconds": 6.0,
                    "julian": 60.0,
                    "yearlen": 365.0,
                },
                "only_changed_forcing_leaf": "qair",
                "forcing_leaf_ab_metrics": unchanged_forcing,
                "surface_ab_metrics": invariance,
                "old_replay_vs_frozen_gpu_seam": backend_offset,
                "paired_tsk_shift": paired_shift,
            },
            "tsk_parity": {
                "wrf_storage_dtype": "float32 (sealed dump meta real_bytes=4)",
                "frozen_seam_vs_pristine_wrf": frozen_tsk_metrics,
                "old_common_cpu_replay_vs_pristine_wrf": old_replay_metrics,
                "corrected_common_cpu_replay_vs_pristine_wrf": new_replay_metrics,
                "counterfactual_authenticated_seam_plus_paired_delta_vs_pristine_wrf": counterfactual_tsk_metrics,
                "frozen_land_float32_ulp": float32_ulp(frozen_tsk, wrf_tsk, land),
                "corrected_land_float32_ulp": float32_ulp(counterfactual["t_skin"], wrf_tsk, land),
                "land_rms_improvement_factor": (
                    frozen_tsk_metrics["land"]["rms"]
                    / counterfactual_tsk_metrics["land"]["rms"]
                ),
                "closed_bitwise": (
                    counterfactual_tsk_metrics["land"]["bitwise_mismatch_count"] == 0
                ),
            },
            "remaining_localization": {
                "current_d03_status": "TSK_NOT_CLOSED_AFTER_QML_FIX",
                "cross_fixture_solver_boundary": {
                    "proof_sha256": EXPECTED["old_energy_proof"],
                    "verdict": old_energy["verdict"],
                    "exact_wrf_radiation_swap_tsk": (
                        old_energy["flux_closure"]
                        ["post_fix_plus_wrf_exact_radiation"]["tsk_vs_trad"]
                    ),
                },
                "cross_fixture_rrtmg_boundary": {
                    "proof_sha256": EXPECTED["rrtmg_proof"],
                    "verdict": rrtmg["verdict"],
                    "summary_numbers": rrtmg["summary_numbers"],
                },
                "claim_limit": (
                    "These pinned d02 proofs bound the remaining source family but do not "
                    "supply a current-d03 GLW value, so they do not authorize a second fix."
                ),
            },
            "delta_artifact": artifact,
            "delta_contract": {
                "fields": list(DELTA_FIELDS),
                "application": "authenticated seam field + (corrected common replay - old common replay)",
                "prior_capture_entry_exactly_equals_seam": prior_entry,
                "paired_ab_same_inputs_except_qair": True,
                "water_delta_bitwise_zero": True,
                "empirical_tuning": False,
                "tolerance_selected_after_observation": False,
                "synthetic_wrf_reference": False,
                "mask_selected_correction": False,
            },
            "gates": {
                "initial_material_land_state_bitwise_exact_at_wrf_storage_dtype": True,
                "corrected_qair_bitwise_equals_wrf_source_expression": True,
                "paired_ab_changes_only_qair": True,
                "paired_noahmp_tsk_land_rms_and_max_strictly_improve": True,
                "counterfactual_frozen_tsk_land_rms_and_max_strictly_improve": True,
                "all_water_surface_outputs_bitwise_invariant": True,
                "land_tsk_bitwise_closed": False,
            },
            "passed": True,
        }
        atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "output": str(output),
            "delta": artifact,
            "frozen_land_tsk_rms": frozen_tsk_metrics["land"]["rms"],
            "corrected_counterfactual_land_tsk_rms": counterfactual_tsk_metrics["land"]["rms"],
            "old_common_replay_land_tsk_rms": old_replay_metrics["land"]["rms"],
            "new_common_replay_land_tsk_rms": new_replay_metrics["land"]["rms"],
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
