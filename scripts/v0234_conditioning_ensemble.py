"""CPU-only WRF-v4.7.1 conditioning ensemble for the v0.23.4 wake RCA.

This module deliberately contains the complete preregistered workflow:

* freeze deterministic, balanced, pairwise-orthogonal one-ULP U/V masks;
* prepare isolated member directories without mutating the canonical WRF tree;
* fail closed on ALISIOS production/rank/CPU-ownership conflicts;
* run the exact 12-rank, 20-minute instrumented WRF path;
* compare SP1/SP4 U/V at every d03 step 1..200 with both the unperturbed
  WRF realization and the retained GPU-vs-WRF trajectory;
* archive every raw savepoint losslessly and assemble a self-hashed verdict.

No function in this module imports JAX/CUDA libraries, queries a GPU, acquires
a GPU lock, or edits ``src/gpuwrf``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble


REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-conditioning-ensemble-gpt"
PLAN_PATH = SPRINT / "ensemble-plan.json"
CPU_PLACEMENT = SPRINT / "cpu-rank-placement.txt"

KIMI_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-first-interval-momentum-kimi"
KIMI_PROOF = KIMI_SPRINT / "proof.json"
KIMI_RETAINED = KIMI_SPRINT / "retained-evidence-manifest.json"

KIMI_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi")
BASE_RUN = KIMI_ROOT / "run/momsp_arm"
BASE_DUMPS = KIMI_ROOT / "momsp_dumps"
BASE_CACHE = KIMI_ROOT / "compare/wrf_global_cache"
WRF_BINARY = KIMI_ROOT / "wrf_iso/install_iso/bin/wrf"
WRF_ENV = Path("<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build")

GPU_RUN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_first_interval_momentum2"
)
GPU_SAVEPOINTS = GPU_RUN / "savepoints"

SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v0234_conditioning_ensemble_gpt")
MEMBERS_ROOT = SCRATCH / "members"

CPUSET = "13-15,29-31"
SYSTEMD_CPU_AFFINITY = "13 14 15 29 30 31"
SYSTEMD_AFFINITY_SYSCALL_FILTER = "~sched_setaffinity"
CPU_LOCK = Path("/tmp/v0234-conditioning-ensemble-cores13-15.lock")
EXPECTED_PHYSICAL_CORES = (13, 14, 15)
PRODUCTION_PHYSICAL_CORES = tuple(range(12))
RESOURCE_PREFLIGHT_INTERVAL_SECONDS = 2.0
RESOURCE_MONITOR_INTERVAL_SECONDS = 1.0
MPI_RANKS = 12
STEPS = tuple(range(1, 201))
MILESTONES = (1, 2, 5, 10, 20, 50, 100, 200)

EXPECTED_HASHES = {
    "kimi_proof_file": "cffb3cf70e203a2cf4ffd043677769b1a596cef08e729e65af67c7a075e010ad",
    "kimi_proof_canonical": "9577feff9cb71d8e920d8751c878c64dce6e949db73f35a350fa0905ab982b5a",
    "kimi_retained_file": "2d282ba6adf53af4eb6633413af3ed6b9cab237aee9d464a7aac43bccf7acfb2",
    "kimi_retained_canonical": "3135ba52952af5fa7bbc6cba9a6a697e40a11e1f13b8f215d4273990bf421b08",
    "wrf_binary": "072b5fa168d2cc6b943145317ae150aceb94b386217538ce530be9f70643c539",
    "namelist": "9ce4336dd4b878685871370a2bdf048ce27c4bb52b655b5a33089008026caaf3",
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}

MASK_SPECS = (
    {
        "id": "mask_a",
        "seed": 0x243F6A8885A308D3,
        "vertical_carrier": "(-1)^k",
        "carrier": "k_parity",
    },
    {
        "id": "mask_b",
        "seed": 0x13198A2E03707344,
        "vertical_carrier": "(-1)^floor(k/2)",
        "carrier": "k_pair_parity",
    },
    {
        "id": "mask_c",
        "seed": 0xA4093822299F31D0,
        "vertical_carrier": "(-1)^(k+floor(k/2))",
        "carrier": "k_walsh_product",
    },
)

MEMBER_SPECS = (
    {"id": "control-repeat", "kind": "control", "mask": None, "polarity": 0},
    {"id": "mask-a-plus", "kind": "perturbed", "mask": "mask_a", "polarity": 1},
    {"id": "mask-a-minus", "kind": "perturbed", "mask": "mask_a", "polarity": -1},
    {"id": "mask-b-plus", "kind": "perturbed", "mask": "mask_b", "polarity": 1},
    {"id": "mask-b-minus", "kind": "perturbed", "mask": "mask_b", "polarity": -1},
    {"id": "mask-c-plus", "kind": "perturbed", "mask": "mask_c", "polarity": 1},
    {"id": "mask-c-minus", "kind": "perturbed", "mask": "mask_c", "polarity": -1},
)

DECISION_THRESHOLDS = {
    "primary_growth": {
        "metric": "combined SP4 U/V RMSE, GPU-vs-WRF against six-member WRF-vs-WRF envelope",
        "literal_envelope_inside_fraction_min": 0.80,
        "milestone_gpu_to_member_max_ratio_max": 1.25,
        "materially_smaller_factor": 2.0,
        "falsifier_fraction_min": 0.80,
        "milestones": list(MILESTONES),
    },
    "zero_mean": {
        "metric": "abs(mean bias) / RMSE, pooled over SP4 U/V and steps 1..200",
        "gpu_p95_max": 0.10,
        "ensemble_p95_max": 0.10,
        "gpu_ensemble_p95_gap_max": 0.05,
    },
    "spatial_signature": {
        "metric_scope": "SP4 U/V at preregistered milestones",
        "magnitude_correlation_median_min": 0.25,
        "top5pct_jaccard_median_min": 0.10,
        "terrain_gradient_correlation_median_min": 0.20,
        "gpu_ensemble_terrain_median_gap_max": 0.20,
        "high_terrain_argmax_m": 500.0,
        "high_terrain_argmax_fraction_min": 0.50,
    },
    "vertical_signature": {
        "metric_scope": "SP4 U/V per-level RMSE profiles at preregistered milestones",
        "profile_correlation_median_min": 0.80,
    },
    "hard_structural_falsifier": {
        "required_independent_failures": 2,
        "magnitude_correlation_median_max": 0.05,
        "top5pct_jaccard_median_max": 0.02,
        "vertical_profile_correlation_median_max": 0.25,
        "terrain_opposite_sign_min_abs": 0.20,
    },
}


def expected_cpu_placement() -> str:
    """Return the frozen 12-rank/three-core placement specification.

    Every rank is restricted to the same six logical processing units. They
    are exactly the two SMT threads of physical cores 13..15, so twelve ranks
    necessarily oversubscribe no more than those three cores.
    """
    return "".join(
        f"rank {rank} allowed_logical={CPUSET}\n"
        for rank in range(MPI_RANKS)
    )


def verify_cpu_placement() -> dict[str, Any]:
    expected = expected_cpu_placement()
    observed = CPU_PLACEMENT.read_text()
    if observed != expected:
        raise RuntimeError("committed CPU placement does not match frozen 12-rank layout")
    return {
        "path": str(CPU_PLACEMENT),
        "file_sha256": sha256_file(CPU_PLACEMENT),
        "rank_count": MPI_RANKS,
        "per_rank_allowed_logical_processing_units": sorted(_parse_cpu_list(CPUSET)),
        "allowed_physical_cores": list(EXPECTED_PHYSICAL_CORES),
        "average_rank_oversubscription_per_physical_core": MPI_RANKS / len(EXPECTED_PHYSICAL_CORES),
    }


def wrf_payload_command(executable: str = "./wrf.exe", *extra_args: str) -> list[str]:
    """Build the frozen exact-12-rank WRF payload command."""
    verify_cpu_placement()
    return [
        "/usr/bin/nice", "-n", "10",
        "/usr/bin/ionice", "-c", "3",
        str(WRF_ENV / "bin/prterun"),
        "--map-by", ":oversubscribe:hwtcpus",
        "--bind-to", "none",
        "-np", str(MPI_RANKS), executable,
        *extra_args,
    ]


def wrf_launch_environment(dumps: Path) -> dict[str, str]:
    """Return the frozen CPU-only WRF payload environment."""
    return {
        "HOME": "<USER_HOME>",
        "USER": "user",
        "LOGNAME": "user",
        "PATH": f"{WRF_ENV / 'bin'}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": str(WRF_ENV / "lib"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "WRFGPU2_MOM_SP": "1",
        "WRFGPU2_MOM_SP_GRID": "3",
        "WRFGPU2_MOM_SP_SMIN": "1",
        "WRFGPU2_MOM_SP_SMAX": "200",
        "WRFGPU2_MOM_SP_ROOT": str(dumps),
    }


def systemd_launch_command(
    unit: str,
    run_dir: Path,
    dumps: Path,
    executable: str = "./wrf.exe",
    *extra_args: str,
) -> list[str]:
    """Wrap the payload in a kernel-enforced, non-expandable CPU boundary.

    systemd applies CPUAffinity before exec and seccomp permanently denies
    sched_setaffinity to the service and descendants. Thus neither PRRTE nor
    WRF can expand beyond the admitted six logical CPUs, even transiently.
    """
    environment = wrf_launch_environment(dumps)
    command = [
        "/usr/bin/taskset", "-c", CPUSET,
        "/usr/bin/systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet",
        f"--unit={unit}",
        f"--working-directory={run_dir}",
        f"--property=CPUAffinity={SYSTEMD_CPU_AFFINITY}",
        f"--property=SystemCallFilter={SYSTEMD_AFFINITY_SYSCALL_FILTER}",
        "--property=SystemCallErrorNumber=EPERM",
        "--property=NoNewPrivileges=yes",
        "--property=KillMode=control-group",
    ]
    command.extend(f"--setenv={key}={value}" for key, value in environment.items())
    command.extend(wrf_payload_command(executable, *extra_args))
    return command


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any, self_key: str | None = None) -> str:
    if self_key is not None and isinstance(value, dict):
        value = {key: val for key, val in value.items() if key != self_key}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_self_hashed(path: Path, payload: dict[str, Any], key: str = "proof_sha256") -> None:
    out = dict(payload)
    out[key] = canonical_digest(out, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def load_self_hashed(path: Path, key: str = "proof_sha256") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    embedded = payload.get(key)
    observed = canonical_digest(payload, key)
    if embedded != observed:
        raise RuntimeError(f"canonical hash mismatch for {path}: {embedded} != {observed}")
    return payload


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot take percentile of empty input")
    return float(np.percentile(array, q))


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Vectorized SplitMix64 with intentional uint64 wraparound."""
    mask = (1 << 64) - 1
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = values + np.uint64(seed & mask) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def _carrier(spec: dict[str, Any], nz: int) -> np.ndarray:
    k = np.arange(nz, dtype=np.int64)
    if spec["carrier"] == "k_parity":
        bit = k & 1
    elif spec["carrier"] == "k_pair_parity":
        bit = (k // 2) & 1
    elif spec["carrier"] == "k_walsh_product":
        bit = (k + k // 2) & 1
    else:
        raise ValueError(f"unknown carrier {spec['carrier']}")
    return np.where(bit == 0, 1, -1).astype(np.int8)


def mask_signs(
    variable: str,
    shape: tuple[int, int, int],
    spec: dict[str, Any],
    boundary_band: int = 5,
) -> np.ndarray:
    """Return the selected interior mask in (k,j,i) C order."""
    nz, ny, nx = shape
    if nz != 44 or ny <= 2 * boundary_band or nx <= 2 * boundary_band:
        raise ValueError(f"unexpected {variable} shape {shape}")
    j = np.arange(boundary_band, ny - boundary_band, dtype=np.uint64)[:, None]
    i = np.arange(boundary_band, nx - boundary_band, dtype=np.uint64)[None, :]
    variable_salt = 0x5555555555555555 if variable == "U" else 0xAAAAAAAAAAAAAAAA
    keyed = (j << np.uint64(32)) ^ i ^ np.uint64(variable_salt)
    horizontal = np.where(
        (_splitmix64(keyed, int(spec["seed"])) >> np.uint64(63)) == 0,
        1,
        -1,
    ).astype(np.int8)
    return _carrier(spec, nz)[:, None, None] * horizontal[None, :, :]


def mask_audit(shapes: dict[str, tuple[int, int, int]]) -> dict[str, Any]:
    masks: dict[str, dict[str, np.ndarray]] = {}
    rows: dict[str, Any] = {}
    for spec in MASK_SPECS:
        per_variable = {}
        masks[spec["id"]] = {}
        combined_chunks = []
        for variable in ("U", "V"):
            signs = mask_signs(variable, shapes[variable], spec)
            masks[spec["id"]][variable] = signs
            combined_chunks.append(signs.ravel())
            per_variable[variable] = {
                "shape": list(signs.shape),
                "cell_count": int(signs.size),
                "positive_count": int(np.count_nonzero(signs > 0)),
                "negative_count": int(np.count_nonzero(signs < 0)),
                "balance_sum": int(signs.astype(np.int64).sum()),
                "sign_sha256": hashlib.sha256(signs.tobytes()).hexdigest(),
            }
        combined = np.concatenate(combined_chunks)
        rows[spec["id"]] = {
            **spec,
            "per_variable": per_variable,
            "combined_cell_count": int(combined.size),
            "combined_balance_sum": int(combined.astype(np.int64).sum()),
            "combined_sign_sha256": hashlib.sha256(combined.tobytes()).hexdigest(),
        }
    orthogonality = {}
    for left_index, left in enumerate(MASK_SPECS):
        for right in MASK_SPECS[left_index + 1 :]:
            dot = 0
            for variable in ("U", "V"):
                dot += int(
                    np.sum(
                        masks[left["id"]][variable].astype(np.int64)
                        * masks[right["id"]][variable].astype(np.int64),
                        dtype=np.int64,
                    )
                )
            orthogonality[f"{left['id']}__{right['id']}"] = dot
    if any(row["combined_balance_sum"] != 0 for row in rows.values()):
        raise RuntimeError("mask balance proof failed")
    if any(value != 0 for value in orthogonality.values()):
        raise RuntimeError(f"mask orthogonality proof failed: {orthogonality}")
    return {"masks": rows, "pairwise_dot_products": orthogonality}


def _open_netcdf(path: Path, mode: str = "r"):
    from netCDF4 import Dataset

    dataset = Dataset(path, mode)
    dataset.set_auto_mask(False)
    dataset.set_auto_scale(False)
    return dataset


def input_shapes(path: Path) -> dict[str, tuple[int, int, int]]:
    with _open_netcdf(path) as dataset:
        out = {}
        for name in ("U", "V"):
            variable = dataset.variables[name]
            if str(variable.dtype) not in {"float32", ">f4", "<f4"}:
                raise RuntimeError(f"{name} is not fp32: {variable.dtype}")
            out[name] = tuple(int(value) for value in variable.shape[1:])
        return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def netcdf_semantic_manifest(path: Path) -> dict[str, Any]:
    with _open_netcdf(path) as dataset:
        dimensions = {
            name: {"size": len(dim), "unlimited": bool(dim.isunlimited())}
            for name, dim in dataset.dimensions.items()
        }
        globals_ = {name: _jsonable(dataset.getncattr(name)) for name in dataset.ncattrs()}
        variables = {}
        for name, variable in dataset.variables.items():
            data = np.ascontiguousarray(np.asarray(variable[:]))
            variables[name] = {
                "dtype": str(variable.dtype),
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
                "attributes": {
                    attr: _jsonable(variable.getncattr(attr)) for attr in variable.ncattrs()
                },
                "data_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            }
        return {
            "file_format": dataset.file_format,
            "dimensions": dimensions,
            "global_attributes": globals_,
            "variables": variables,
        }


def _verify_authority() -> dict[str, Any]:
    file_rows = {
        "kimi_proof": (KIMI_PROOF, EXPECTED_HASHES["kimi_proof_file"]),
        "kimi_retained": (KIMI_RETAINED, EXPECTED_HASHES["kimi_retained_file"]),
        "wrf_binary": (WRF_BINARY, EXPECTED_HASHES["wrf_binary"]),
        "namelist": (BASE_RUN / "namelist.input", EXPECTED_HASHES["namelist"]),
        "wrfbdy_d01": (BASE_RUN / "wrfbdy_d01", EXPECTED_HASHES["wrfbdy_d01"]),
        "wrfinput_d01": (BASE_RUN / "wrfinput_d01", EXPECTED_HASHES["wrfinput_d01"]),
        "wrfinput_d02": (BASE_RUN / "wrfinput_d02", EXPECTED_HASHES["wrfinput_d02"]),
        "wrfinput_d03": (BASE_RUN / "wrfinput_d03", EXPECTED_HASHES["wrfinput_d03"]),
    }
    rows = {}
    for name, (path, expected) in file_rows.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"authority mismatch {name}: {observed} != {expected}")
        rows[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}
    proof = load_self_hashed(KIMI_PROOF)
    retained = load_self_hashed(KIMI_RETAINED)
    if proof["proof_sha256"] != EXPECTED_HASHES["kimi_proof_canonical"]:
        raise RuntimeError("Kimi proof canonical digest drift")
    if retained["proof_sha256"] != EXPECTED_HASHES["kimi_retained_canonical"]:
        raise RuntimeError("Kimi retained canonical digest drift")
    rows["kimi_proof"]["canonical_sha256"] = proof["proof_sha256"]
    rows["kimi_retained"]["canonical_sha256"] = retained["proof_sha256"]
    return rows


def audit_authority() -> None:
    """Independently re-hash the complete frozen truth/GPU evidence chain."""
    plan = load_plan()
    output = SPRINT / "authority-audit.json"
    if output.exists():
        raise FileExistsError(output)
    authority = _verify_authority()
    canonical_manifest = KIMI_SPRINT / "evidence/canonical-wrf-tree-manifest-20260718.sha256"
    canonical_tree = Path("<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF")
    manifest_check = subprocess.run(
        ["sha256sum", "-c", "--quiet", str(canonical_manifest)],
        cwd=canonical_tree, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if manifest_check.returncode != 0:
        raise RuntimeError(f"canonical WRF manifest failed: {manifest_check.stdout}")
    instrumentation_patch = KIMI_SPRINT / "evidence/wrf-momsp-instrumentation.patch"
    reverse_check = subprocess.run(
        [
            "patch", "--dry-run", "--reverse", "--strip=2",
            f"--directory={KIMI_ROOT / 'wrf_iso'}",
            f"--input={instrumentation_patch}",
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if reverse_check.returncode != 0:
        raise RuntimeError(f"isolated instrumentation reverse-check failed: {reverse_check.stdout}")
    frame_rows = {}
    retained_truth = Path(
        "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
        "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf"
    )
    for name in (
        "wrfout_d01_2025-03-01_00:00:00",
        "wrfout_d02_2025-03-01_00:00:00",
        "wrfout_d03_2025-03-01_00:00:00",
        "wrfout_d03_2025-03-01_00:20:00",
    ):
        instrumented_hash = sha256_file(BASE_RUN / name)
        retained_hash = sha256_file(retained_truth / name)
        if instrumented_hash != retained_hash:
            raise RuntimeError(f"output-neutrality drift: {name}")
        frame_rows[name] = {
            "sha256": instrumented_hash,
            "byte_identical_to_retained_truth": True,
        }
    kimi_proof = load_self_hashed(KIMI_PROOF)
    external_rows = {}
    for label, source_row in {
        "gpu_wrf_compare": kimi_proof["isolation"]["compare"],
        "gpu_wrf_isolation": kimi_proof["isolation"]["isolation"],
    }.items():
        path = Path(source_row["path"])
        payload = load_self_hashed(path)
        row = _artifact_row(path, True)
        if row["file_sha256"] != source_row["file_sha256"]:
            raise RuntimeError(f"{label} file hash drift")
        if row["canonical_sha256"] != source_row["canonical_sha256"]:
            raise RuntimeError(f"{label} canonical hash drift")
        external_rows[label] = row
    gpu_savepoints = sorted(GPU_SAVEPOINTS.glob("step*_sp*__*.npy"))
    savepoint_manifest = [
        {"name": path.name, "bytes": path.stat().st_size, "file_sha256": sha256_file(path)}
        for path in gpu_savepoints
    ]
    savepoint_manifest_sha = canonical_digest(savepoint_manifest)
    expected_savepoint_sha = kimi_proof["gpu_arms"]["arm2_repaired"]["savepoints"]["manifest_sha256"]
    if len(savepoint_manifest) != 1656 or savepoint_manifest_sha != expected_savepoint_sha:
        raise RuntimeError(
            f"retained GPU savepoint drift: count={len(savepoint_manifest)} "
            f"sha={savepoint_manifest_sha} expected={expected_savepoint_sha}"
        )
    payload = {
        "schema": "gpuwrf.v0234.conditioning-frozen-authority-audit.v1",
        "verdict": "FROZEN_WRF_GPU_AUTHORITY_REHASHED",
        "audited_at_utc": utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "authority": authority,
        "canonical_wrf_tree": {
            "path": str(canonical_tree),
            "manifest": _artifact_row(canonical_manifest),
            "manifest_rows": 3441,
            "sha256sum_check_returncode": manifest_check.returncode,
        },
        "isolated_instrumentation": {
            "patch": _artifact_row(instrumentation_patch),
            "reverse_dry_run_returncode": reverse_check.returncode,
            "reverse_dry_run_output": reverse_check.stdout.splitlines(),
            "binary": authority["wrf_binary"],
            "output_neutrality": frame_rows,
        },
        "retained_gpu": {
            "namespace": GPU_RUN.name,
            "savepoint_count": len(savepoint_manifest),
            "savepoint_manifest_sha256": savepoint_manifest_sha,
            "expected_savepoint_manifest_sha256": expected_savepoint_sha,
            "steps": [1, 207],
            "contract_comparison_steps": [1, 200],
            "external_analysis": external_rows,
        },
        "model_tree": {
            "pre_diagnostic": "835dcc29bf316c0715b41a72e064985e9cf099df",
            "sprint_base": _git("rev-parse", "1eabc5d6c20b804f4a40d7ca27d5d01e7f5386dd:src/gpuwrf"),
            "current": _git("rev-parse", "HEAD:src/gpuwrf"),
            "current_delta_from_sprint_base": _git(
                "diff", "--name-only", "1eabc5d6c20b804f4a40d7ca27d5d01e7f5386dd",
                "HEAD", "--", "src/gpuwrf",
            ).splitlines(),
        },
        "gpu_attestation": {
            "new_gpu_query": False,
            "new_gpu_lock": False,
            "new_gpu_compile": False,
            "new_gpu_dispatch": False,
            "new_gpu_kernel": False,
            "retained_artifacts_read_only": True,
        },
    }
    write_self_hashed(output, payload)
    print(json.dumps({
        "audit": str(output),
        "canonical_sha256": load_self_hashed(output)["proof_sha256"],
        "gpu_savepoint_manifest_sha256": savepoint_manifest_sha,
    }, sort_keys=True))


def build_plan() -> dict[str, Any]:
    authority = _verify_authority()
    shapes = input_shapes(BASE_RUN / "wrfinput_d03")
    audit = mask_audit(shapes)
    selected_cells = {
        variable: {
            "source_shape_time_k_j_i": [1, *shape],
            "selected_slice_zero_based": {
                "time": [0, 1],
                "k": [0, shape[0]],
                "j": [5, shape[1] - 5],
                "i": [5, shape[2] - 5],
            },
            "selected_count": int(shape[0] * (shape[1] - 10) * (shape[2] - 10)),
            "preserved_boundary_band_cells": 5,
        }
        for variable, shape in shapes.items()
    }
    payload = {
        "schema": "gpuwrf.v0234.wrf-conditioning-ensemble-plan.v1",
        "status": "PREREGISTERED_BEFORE_ANY_ENSEMBLE_MEMBER_RESULT",
        "objective": (
            "Test whether the retained GPU-vs-WRF SP1/SP4 d03 momentum growth "
            "is ordinary roundoff-scale WRF conditioning rather than a systematic "
            "GPU dry-dycore/nest operator error."
        ),
        "frozen_authority": authority,
        "science": {
            "model": "WRF v4.7.1 isolated output-neutral instrumented CPU binary",
            "domain": "real d03 first interval 2025-03-01 00:00..00:20 UTC",
            "steps": [1, 200],
            "savepoints": ["SP1 incoming U/V", "SP4 post-dry-dycore/nest U/V"],
            "baseline": "canonical Kimi unperturbed WRF realization",
            "comparison": "retained GPU arm nested_stage_omega_transport_470e6111_first_interval_momentum2",
        },
        "perturbation": {
            "dtype": "IEEE-754 binary32",
            "amplitude": "exactly one representable nextafter step toward +/-infinity",
            "variables": ["U", "V"],
            "finite_only": True,
            "selected_cells": selected_cells,
            "all_nonmomentum_variables_preserved": True,
            "metadata_preserved": True,
            "boundary_preserved": True,
            "mask_algorithm": (
                "SplitMix64 horizontal sign keyed by (variable,j,i,seed), multiplied "
                "by one of three length-44 Walsh carriers; carriers make every mask "
                "balanced and every mask pair exactly orthogonal independently of "
                "the horizontal signs."
            ),
            **audit,
        },
        "members": list(MEMBER_SPECS),
        "member_count": {"control_repeat": 1, "perturbed": 6, "total": 7},
        "execution": {
            "mpi_ranks": MPI_RANKS,
            "run_minutes": 20,
            "instrumented_steps": [1, 200],
            "cpu_only": True,
            "cuda_visible_devices": "",
            "gpu_queries_locks_compiles_dispatches_kernels": "forbidden",
            "cpu_cpuset_logical": CPUSET,
            "cpu_physical_cores": list(EXPECTED_PHYSICAL_CORES),
            "cpu_isolation_reason": (
                "principal-authorized logical CPUs 13-15 and SMT siblings 29-31 "
                "map only to physical cores 13-15; every concurrent ALISIOS "
                "real.exe/wrf.exe rank and launcher must remain on physical cores 0-11"
            ),
            "resource_gate": (
                "Two stable snapshots immediately before each member and a one-second "
                "watchdog throughout it must prove every concurrent production real.exe/"
                "wrf.exe rank and launcher has an unchanged identity and affinity confined "
                "to physical cores 0-11, while all ensemble ranks remain confined to "
                "physical cores 13-15. Nightly-active or a live production lock alone is "
                "not a blocker. Abort/defer on production expansion, contraction/identity "
                "or affinity drift, any physical-core overlap, controller identity drift, "
                "the isolated core map drifting, available RAM <32 GiB, or <DATA_ROOT> free "
                "space <40 GiB."
            ),
            "production_concurrency_authority": (
                "the user principal resource override received before any science result"
            ),
            "preflight_stability_interval_seconds": RESOURCE_PREFLIGHT_INTERVAL_SECONDS,
            "continuous_monitor_interval_seconds": RESOURCE_MONITOR_INTERVAL_SECONDS,
            "mpi_oversubscription": "12 ranks explicitly oversubscribed on 3 physical cores / 6 SMT threads",
            "priority": "nice +10, ionice idle",
            "raw_retention": (
                "Every per-rank SP1/SP2/SP3/SP4/SP5 dump is retained losslessly in "
                "a SHA-256-authenticated tar.zst after member analysis."
            ),
        },
        "metrics": {
            "every_step_every_member": [
                "U/V RMSE", "U/V max absolute error", "U/V mean bias",
                "normalized mean bias", "state spatial correlation",
                "signed and magnitude error correlation with GPU-vs-WRF",
                "terrain-gradient correlation of lowest-level absolute error",
                "44-level RMSE and bias profiles", "argmax k/j/i/HGT/LANDMASK",
                "top-5-percent magnitude-map Jaccard with GPU-vs-WRF",
            ],
            "decision_stage": "SP4 post-dry-dycore/nest",
            "combined_growth_metric": "cell-count-weighted combined U/V RMSE",
        },
        "decision_thresholds": DECISION_THRESHOLDS,
        "decision_rule": {
            "WRF_CONDITIONING_ENSEMBLE_SUPPORTS_RELEASE_FRAMING": (
                "primary growth support AND zero-mean pass AND spatial-signature "
                "pass AND vertical-signature pass"
            ),
            "WRF_CONDITIONING_ENSEMBLE_FALSIFIES_CONDITIONING": (
                "primary scale falsifier OR at least two independent hard structural "
                "falsifiers"
            ),
            "WRF_CONDITIONING_ENSEMBLE_INCONCLUSIVE": "all other frozen outcomes",
        },
        "control_gate": {
            "all_reassembled_SP1_SP4_UV_steps_bitwise_equal": True,
            "all_four_wrfout_frames_byte_identical": True,
            "failure_disposition": "INCONCLUSIVE_HARNESS_REPRODUCIBILITY_FAILURE",
        },
        "commands": {
            "cpu_prefix": f"/usr/bin/taskset -c {CPUSET} /usr/bin/nice -n 10 /usr/bin/ionice -c 3",
            "authority_audit": f"/usr/bin/taskset -c {CPUSET} /usr/bin/nice -n 10 /usr/bin/ionice -c 3 python -m scripts.v0234_conditioning_ensemble audit-authority",
            "prepare": f"/usr/bin/taskset -c {CPUSET} /usr/bin/nice -n 10 /usr/bin/ionice -c 3 python -m scripts.v0234_conditioning_ensemble prepare-all",
            "run_one": f"/usr/bin/taskset -c {CPUSET} /usr/bin/nice -n 10 /usr/bin/ionice -c 3 python -m scripts.v0234_conditioning_ensemble run-member --member <id>",
            "analyze_one": f"/usr/bin/taskset -c {CPUSET} /usr/bin/nice -n 10 /usr/bin/ionice -c 3 python -m scripts.v0234_conditioning_ensemble analyze-member --member <id>",
            "archive_one": f"/usr/bin/taskset -c {CPUSET} /usr/bin/nice -n 10 /usr/bin/ionice -c 3 python -m scripts.v0234_conditioning_ensemble archive-member --member <id>",
            "aggregate": f"/usr/bin/taskset -c {CPUSET} /usr/bin/nice -n 10 /usr/bin/ionice -c 3 python -m scripts.v0234_conditioning_ensemble aggregate",
            "proof": f"/usr/bin/taskset -c {CPUSET} python -m scripts.v0234_conditioning_ensemble terminal-proof",
        },
        "expected_artifacts": [
            "per-member perturbation-manifest.json",
            "per-member timestamped resource admission JSON",
            "per-member continuous resource-monitor.json",
            "per-member execution-receipt.json and launch.log",
            "per-member member-analysis.json",
            "per-member raw-inventory.json and momsp-dumps.tar.zst",
            "ensemble-analysis.json", "proof.json", "retained-evidence-manifest.json",
            "metrics-and-thresholds-register.md", "metrics-and-thresholds-register.json",
            "worker-report.md", "command-log.md",
        ],
        "falsifier": (
            "The WRF-vs-WRF perturbation envelope is materially smaller than the "
            "GPU-vs-WRF growth by the frozen factor/fraction rule, or its spatial/"
            "vertical structure meets the frozen hard-difference rule."
        ),
        "next_assignment_by_verdict": {
            "WRF_CONDITIONING_ENSEMBLE_SUPPORTS_RELEASE_FRAMING": (
                "Kimi K3 thinking-max independently audit the preregistration, masks, "
                "control, raw archives, metric implementation, and release-framing "
                "limits; do not authorize a model fix or release."
            ),
            "WRF_CONDITIONING_ENSEMBLE_FALSIFIES_CONDITIONING": (
                "Kimi K3 thinking-max audit this falsification and own the next coherent "
                "source-bound discriminator that splits dry-dycore/nest suboperators "
                "at d03 step 1 before any model correction."
            ),
            "WRF_CONDITIONING_ENSEMBLE_INCONCLUSIVE": (
                "Kimi K3 thinking-max audit why the frozen scale/structure gates did not "
                "separate the hypotheses and preregister the smallest next discriminator "
                "without changing this ensemble's criteria."
            ),
        },
    }
    payload["plan_sha256"] = canonical_digest(payload, "plan_sha256")
    return payload


def write_plan(path: Path = PLAN_PATH) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {path}")
    payload = build_plan()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"plan": str(path), "plan_sha256": payload["plan_sha256"]}, sort_keys=True))


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    observed = canonical_digest(payload, "plan_sha256")
    if payload.get("plan_sha256") != observed:
        raise RuntimeError(f"plan canonical hash mismatch: {payload.get('plan_sha256')} != {observed}")
    if payload.get("status") != "PREREGISTERED_BEFORE_ANY_ENSEMBLE_MEMBER_RESULT":
        raise RuntimeError("plan is not frozen")
    return payload


def _member_spec(member: str) -> dict[str, Any]:
    for spec in MEMBER_SPECS:
        if spec["id"] == member:
            return dict(spec)
    raise ValueError(f"unknown member {member!r}")


def _member_dir(member: str) -> Path:
    _member_spec(member)
    return MEMBERS_ROOT / member


def _copy_run_template(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    mutable = {"namelist.input", "wrfbdy_d01", "wrfinput_d01", "wrfinput_d02", "wrfinput_d03"}
    skip_re = re.compile(r"^(?:rsl\.(?:out|error)\.|wrfout_|namelist\.output)")
    for source in sorted(BASE_RUN.iterdir(), key=lambda item: item.name):
        if skip_re.match(source.name):
            continue
        target = destination / source.name
        if source.name in mutable:
            shutil.copy2(source, target, follow_symlinks=True)
            # The authenticated canonical inputs are intentionally mode 0444.
            # Only the private member copy may be updated; WRF still receives
            # byte-identical d01/d02/boundary/namelist content.
            target.chmod(target.stat().st_mode | stat.S_IWUSR)
        else:
            target.symlink_to(source.resolve() if source.is_symlink() else source)


def _selected_slice(array: np.ndarray) -> tuple[slice, slice, slice]:
    if array.ndim != 3:
        raise ValueError(array.shape)
    return slice(0, array.shape[0]), slice(5, array.shape[1] - 5), slice(5, array.shape[2] - 5)


def apply_one_ulp(base: np.ndarray, signs: np.ndarray, polarity: int) -> tuple[np.ndarray, dict[str, Any]]:
    if base.dtype != np.float32:
        raise ValueError(f"one-ULP perturbation requires fp32, got {base.dtype}")
    if polarity not in (-1, 1):
        raise ValueError(polarity)
    selected_slice = _selected_slice(base)
    selected = np.asarray(base[selected_slice], dtype=np.float32)
    if selected.shape != signs.shape:
        raise ValueError(f"mask shape {signs.shape} != selected shape {selected.shape}")
    if not np.isfinite(selected).all():
        raise RuntimeError("selected U/V cells are not all finite")
    directions = signs.astype(np.int8) * np.int8(polarity)
    targets = np.where(directions > 0, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
    changed = np.nextafter(selected, targets, dtype=np.float32)
    if not np.isfinite(changed).all():
        raise RuntimeError("one-ULP perturbation produced nonfinite U/V")
    if np.any(changed == selected):
        raise RuntimeError("one-ULP perturbation did not change every selected cell")
    out = base.copy()
    out[selected_slice] = changed
    delta = changed.astype(np.float64) - selected.astype(np.float64)
    untouched = np.ones(base.shape, dtype=bool)
    untouched[selected_slice] = False
    if not np.array_equal(out[untouched], base[untouched]):
        raise RuntimeError("boundary/nonselected momentum changed")
    before_bits = np.ascontiguousarray(selected).view(np.uint32)
    after_bits = np.ascontiguousarray(changed).view(np.uint32)
    stats = {
        "selected_count": int(selected.size),
        "positive_direction_count": int(np.count_nonzero(directions > 0)),
        "negative_direction_count": int(np.count_nonzero(directions < 0)),
        "changed_count": int(np.count_nonzero(changed != selected)),
        "finite_before_count": int(np.count_nonzero(np.isfinite(selected))),
        "finite_after_count": int(np.count_nonzero(np.isfinite(changed))),
        "before_selected_bits_sha256": hashlib.sha256(before_bits.tobytes()).hexdigest(),
        "after_selected_bits_sha256": hashlib.sha256(after_bits.tobytes()).hexdigest(),
        "direction_sha256": hashlib.sha256(directions.tobytes()).hexdigest(),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_mean": float(delta.mean()),
        "delta_rmse": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
        "boundary_and_nonselected_bitwise_equal": True,
        "nextafter_exact_one_representable_step": True,
    }
    return out, stats


def prepare_member(member: str) -> None:
    plan = load_plan()
    _verify_authority()
    spec = _member_spec(member)
    member_dir = _member_dir(member)
    if member_dir.exists():
        raise FileExistsError(f"refusing to overwrite member: {member_dir}")
    run_dir = member_dir / "run"
    member_dir.mkdir(parents=True, exist_ok=False)
    _copy_run_template(run_dir)
    base_path = BASE_RUN / "wrfinput_d03"
    member_path = run_dir / "wrfinput_d03"
    base_semantic = netcdf_semantic_manifest(base_path)
    variable_stats = {}
    if spec["kind"] == "perturbed":
        mask_spec = next(row for row in MASK_SPECS if row["id"] == spec["mask"])
        with _open_netcdf(member_path, "r+") as dataset:
            for variable_name in ("U", "V"):
                variable = dataset.variables[variable_name]
                base = np.asarray(variable[0], dtype=np.float32)
                signs = mask_signs(variable_name, base.shape, mask_spec)
                changed, stats = apply_one_ulp(base, signs, int(spec["polarity"]))
                variable[0] = changed
                variable_stats[variable_name] = {
                    **stats,
                    "base_full_bits_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
                    "member_full_bits_sha256": hashlib.sha256(changed.tobytes()).hexdigest(),
                    "full_changed_count": int(np.count_nonzero(changed != base)),
                }
        expected_mask_hash = plan["perturbation"]["masks"][spec["mask"]]["per_variable"]
        for variable_name in ("U", "V"):
            if variable_stats[variable_name]["selected_count"] != expected_mask_hash[variable_name]["cell_count"]:
                raise RuntimeError("selected cell count drift from preregistration")
    else:
        if sha256_file(member_path) != EXPECTED_HASHES["wrfinput_d03"]:
            raise RuntimeError("control wrfinput copy is not byte-identical")

    member_semantic = netcdf_semantic_manifest(member_path)
    if base_semantic["file_format"] != member_semantic["file_format"]:
        raise RuntimeError("NetCDF format changed")
    if base_semantic["dimensions"] != member_semantic["dimensions"]:
        raise RuntimeError("NetCDF dimensions changed")
    if base_semantic["global_attributes"] != member_semantic["global_attributes"]:
        raise RuntimeError("NetCDF global metadata changed")
    semantic_differences = []
    for variable_name, base_row in base_semantic["variables"].items():
        member_row = member_semantic["variables"][variable_name]
        if {k: v for k, v in base_row.items() if k != "data_sha256"} != {
            k: v for k, v in member_row.items() if k != "data_sha256"
        }:
            raise RuntimeError(f"metadata drift in {variable_name}")
        if base_row["data_sha256"] != member_row["data_sha256"]:
            semantic_differences.append(variable_name)
    expected_differences = [] if spec["kind"] == "control" else ["U", "V"]
    if semantic_differences != expected_differences:
        raise RuntimeError(
            f"semantic differences {semantic_differences} != {expected_differences}"
        )
    manifest = {
        "schema": "gpuwrf.v0234.conditioning-perturbation-manifest.v1",
        "member": member,
        "member_spec": spec,
        "plan_sha256": plan["plan_sha256"],
        "created_at_utc": utc_now(),
        "base_wrfinput_d03": {
            "path": str(base_path),
            "bytes": base_path.stat().st_size,
            "sha256": sha256_file(base_path),
        },
        "member_wrfinput_d03": {
            "path": str(member_path),
            "bytes": member_path.stat().st_size,
            "sha256": sha256_file(member_path),
        },
        "semantic_difference_variables": semantic_differences,
        "all_nonmomentum_variable_data_hashes_equal": all(
            base_semantic["variables"][name]["data_sha256"]
            == member_semantic["variables"][name]["data_sha256"]
            for name in base_semantic["variables"]
            if name not in {"U", "V"}
        ),
        "all_variable_metadata_equal": True,
        "dimensions_and_global_metadata_equal": True,
        "variable_stats": variable_stats,
        "physical_validity": {
            "all_member_UV_finite": True,
            "only_exact_one_ulp_interior_UV_changes": spec["kind"] == "perturbed",
            "boundary_preserved": True,
        },
        "copied_authority_hashes": {
            name: sha256_file(run_dir / name)
            for name in ("namelist.input", "wrfbdy_d01", "wrfinput_d01", "wrfinput_d02")
        },
    }
    write_self_hashed(member_dir / "perturbation-manifest.json", manifest)
    print(json.dumps({"member": member, "prepared": str(member_dir)}, sort_keys=True))


def prepare_all() -> None:
    for spec in MEMBER_SPECS:
        prepare_member(spec["id"])


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            cpus.update(range(int(left), int(right) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _cpu_topology() -> dict[int, int]:
    proc = subprocess.run(
        ["lscpu", "-p=CPU,CORE"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    topology = {}
    for line in proc.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        cpu_text, core_text = line.split(",")[:2]
        topology[int(cpu_text)] = int(core_text)
    return topology


def _systemd_state(unit: str) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "systemctl", "--user", "show", unit,
            "--property=ActiveState,SubState,MainPID,InvocationID",
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    values = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return {"unit": unit, "returncode": proc.returncode, "stderr": proc.stderr.strip(), **values}


def _affinity_unit_state(unit: str) -> dict[str, Any]:
    properties = (
        "LoadState,ActiveState,SubState,MainPID,ControlGroup,InvocationID,"
        "CPUAffinity,SystemCallFilter,SystemCallErrorNumber,NoNewPrivileges,KillMode"
    )
    proc = subprocess.run(
        ["/usr/bin/systemctl", "--user", "show", unit, f"--property={properties}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    values: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return {"unit": unit, "returncode": proc.returncode, "stderr": proc.stderr.strip(), **values}


def affinity_unit_reasons(state: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if state.get("returncode") != 0 or state.get("LoadState") != "loaded":
        return ["affinity_unit_unreadable"]
    affinity = str(state.get("CPUAffinity", "")).replace(" ", ",")
    try:
        affinity_cpus = _parse_cpu_list(affinity)
    except ValueError:
        affinity_cpus = set()
    if affinity_cpus != _parse_cpu_list(CPUSET):
        reasons.append("affinity_unit_cpu_set_drift")
    if state.get("SystemCallFilter") != SYSTEMD_AFFINITY_SYSCALL_FILTER:
        reasons.append("affinity_unit_sched_setaffinity_filter_drift")
    if str(state.get("SystemCallErrorNumber")) != "1":
        reasons.append("affinity_unit_syscall_errno_drift")
    if state.get("NoNewPrivileges") != "yes":
        reasons.append("affinity_unit_no_new_privileges_drift")
    if state.get("KillMode") != "control-group":
        reasons.append("affinity_unit_kill_mode_drift")
    if not state.get("ControlGroup") or int(state.get("MainPID") or 0) <= 0:
        reasons.append("affinity_unit_process_identity_missing")
    return sorted(set(reasons))


def _read_cmdline(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [chunk.decode(errors="replace") for chunk in raw.split(b"\0") if chunk]


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().split()
        return int(fields[21])
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError, ValueError):
        return None


def _process_affinity(pid: int) -> list[int]:
    try:
        return sorted(os.sched_getaffinity(pid))
    except (ProcessLookupError, PermissionError):
        return []


def _process_security(pid: int) -> dict[str, int | None]:
    values: dict[str, int | None] = {"no_new_privs": None, "seccomp_mode": None}
    try:
        lines = (Path("/proc") / str(pid) / "status").read_text().splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return values
    for line in lines:
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key == "NoNewPrivs":
            values["no_new_privs"] = int(value.strip())
        elif key == "Seccomp":
            values["seccomp_mode"] = int(value.strip())
    return values


def _process_cwd(pid: int) -> str | None:
    try:
        return str((Path("/proc") / str(pid) / "cwd").resolve(strict=True))
    except (FileNotFoundError, PermissionError, ProcessLookupError, RuntimeError):
        return None


def _process_exe(pid: int) -> str | None:
    try:
        return str((Path("/proc") / str(pid) / "exe").resolve(strict=True))
    except (FileNotFoundError, PermissionError, ProcessLookupError, RuntimeError):
        return None


def _is_member_cwd(cwd: str | None) -> bool:
    if cwd is None:
        return False
    try:
        Path(cwd).relative_to(MEMBERS_ROOT)
    except ValueError:
        return False
    return True


def _model_processes(topology: dict[int, int]) -> list[dict[str, Any]]:
    """Return every real.exe/wrf.exe rank or launcher with authenticated identity.

    Member ranks are distinguished only by their kernel-reported cwd under the
    private ensemble root.  An unreadable cwd therefore fails conservative and
    is classified as production, never silently excluded from the safety gate.
    """
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        argv = _read_cmdline(pid)
        if not argv:
            continue
        program = Path(argv[0]).name
        basenames = {Path(arg).name for arg in argv if arg and not arg.startswith("-")}
        joined = " ".join(argv)
        targets = tuple(name for name in ("real.exe", "wrf.exe") if name in basenames or name in joined)
        is_rank = program in {"real.exe", "wrf.exe"}
        is_launcher = program in {"prterun", "mpirun", "mpiexec"} and bool(targets)
        if not (is_rank or is_launcher):
            continue
        target = program if is_rank else targets[0]
        cwd = _process_cwd(pid)
        affinity = _process_affinity(pid)
        physical = sorted({topology[cpu] for cpu in affinity if cpu in topology})
        identity_payload = {
            "pid": pid,
            "start_ticks": _process_start_ticks(pid),
            "classification": f"{target.removesuffix('.exe')}_{'rank' if is_rank else 'launcher'}",
            "target": target,
            "argv0": argv[0],
            "cwd": cwd,
            "exe": _process_exe(pid),
        }
        security = _process_security(pid)
        rows.append({
            **identity_payload,
            **security,
            "identity_sha256": canonical_digest(identity_payload),
            "argv": argv,
            "cpu_affinity": affinity,
            "physical_cores": physical,
            "scope": "ensemble" if _is_member_cwd(cwd) else "production",
        })
    return sorted(rows, key=lambda row: row["pid"])


def _admission_reasons(
    *,
    topology: dict[int, int],
    production_rows: list[dict[str, Any]],
    ensemble_rows: list[dict[str, Any]],
    nightly: dict[str, Any],
    production_lock: dict[str, Any],
    memory_gib: float,
    free_gib: float,
    allow_member_processes: bool,
    ensemble_controller_affinity: list[int],
) -> list[str]:
    reasons: list[str] = []
    logical = sorted(_parse_cpu_list(CPUSET))
    isolated_physical = sorted({topology[cpu] for cpu in logical})
    if nightly.get("returncode") != 0:
        reasons.append("nightly_state_unreadable")
    if production_lock.get("read_error"):
        reasons.append("production_lock_unreadable")
    if production_lock.get("owner_alive") and not production_lock.get("owner_identity_matches"):
        reasons.append("production_lock_owner_identity_mismatch")
    if tuple(isolated_physical) != EXPECTED_PHYSICAL_CORES:
        reasons.append("isolated_cpu_physical_map_drift")
    controller_physical = {
        topology[cpu] for cpu in ensemble_controller_affinity if cpu in topology
    }
    if (
        not ensemble_controller_affinity
        or not controller_physical
        or not controller_physical.issubset(EXPECTED_PHYSICAL_CORES)
    ):
        reasons.append("ensemble_controller_affinity_escape")
    for row in production_rows:
        observed = set(row.get("physical_cores", []))
        if not row.get("cpu_affinity") or not observed:
            reasons.append("production_model_affinity_unreadable")
            continue
        if not observed.issubset(PRODUCTION_PHYSICAL_CORES):
            reasons.append("production_model_outside_cores_0_11")
        if observed & set(EXPECTED_PHYSICAL_CORES):
            reasons.append("production_ensemble_physical_core_overlap")
    if ensemble_rows and not allow_member_processes:
        reasons.append("preexisting_ensemble_model_process")
    if allow_member_processes:
        for row in ensemble_rows:
            observed = set(row.get("physical_cores", []))
            if not row.get("cpu_affinity") or not observed:
                reasons.append("ensemble_model_affinity_unreadable")
                continue
            if not observed.issubset(EXPECTED_PHYSICAL_CORES):
                reasons.append("ensemble_model_affinity_escape")
            if row.get("no_new_privs") != 1 or row.get("seccomp_mode") != 2:
                reasons.append("ensemble_model_affinity_syscall_lock_missing")
    if memory_gib < 32.0:
        reasons.append("available_ram_below_32_gib")
    if free_gib < 40.0:
        reasons.append("mnt_data_free_below_40_gib")
    return sorted(set(reasons))


def _production_lock() -> dict[str, Any]:
    path = Path("<DATA_ROOT>/alisios/state/production.lock")
    row: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return row
    try:
        text = path.read_text().strip()
        row["sha256"] = sha256_file(path)
        row["payload"] = json.loads(text) if text else None
    except (OSError, json.JSONDecodeError) as exc:
        row["read_error"] = f"{type(exc).__name__}: {exc}"
        return row
    payload = row.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("pid"), int):
        pid = int(payload["pid"])
        row["owner_pid"] = pid
        row["owner_start_ticks_observed"] = _process_start_ticks(pid)
        row["owner_alive"] = row["owner_start_ticks_observed"] is not None
        row["owner_identity_matches"] = (
            row["owner_alive"]
            and payload.get("owner_start_ticks") == row["owner_start_ticks_observed"]
        )
    return row


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("MemAvailable missing")


def _controller_identity(nightly: dict[str, Any], production_lock: dict[str, Any]) -> dict[str, Any]:
    payload = production_lock.get("payload")
    lock_payload = payload if isinstance(payload, dict) else {}
    identity = {
        "nightly_active_state": nightly.get("ActiveState"),
        "nightly_sub_state": nightly.get("SubState"),
        "nightly_main_pid": nightly.get("MainPID"),
        "nightly_invocation_id": nightly.get("InvocationID"),
        "lock_exists": production_lock.get("exists", False),
        "lock_owner_pid": production_lock.get("owner_pid"),
        "lock_owner_start_ticks": production_lock.get("owner_start_ticks_observed"),
        "lock_owner_identity_matches": production_lock.get("owner_identity_matches"),
        "lock_handoff_token": lock_payload.get("handoff_token"),
        "lock_kind": lock_payload.get("kind"),
        "lock_run_id": lock_payload.get("run_id"),
    }
    return {**identity, "identity_sha256": canonical_digest(identity)}


def resource_drift_reasons(baseline: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Compare a live sample with the member's immediately-prelaunch authority."""
    reasons = list(observed.get("deny_reasons", []))
    baseline_rows = {
        row["identity_sha256"]: row
        for row in baseline["alisios"]["production_model_processes"]
    }
    observed_rows = {
        row["identity_sha256"]: row
        for row in observed["alisios"]["production_model_processes"]
    }
    added = sorted(set(observed_rows) - set(baseline_rows))
    removed = sorted(set(baseline_rows) - set(observed_rows))
    if added:
        reasons.append("production_model_process_expansion")
    if removed:
        reasons.append("production_model_process_identity_drift")
    for identity in sorted(set(baseline_rows) & set(observed_rows)):
        before = baseline_rows[identity]
        after = observed_rows[identity]
        if before.get("cpu_affinity") != after.get("cpu_affinity"):
            reasons.append("production_model_logical_affinity_drift")
        if before.get("physical_cores") != after.get("physical_cores"):
            reasons.append("production_model_physical_affinity_drift")
    if (
        baseline["alisios"]["controller_identity"]["identity_sha256"]
        != observed["alisios"]["controller_identity"]["identity_sha256"]
    ):
        reasons.append("production_controller_identity_drift")
    if baseline["cpu_ownership"]["observed_physical_cores"] != observed["cpu_ownership"]["observed_physical_cores"]:
        reasons.append("isolated_cpu_physical_map_drift")
    return sorted(set(reasons))


def resource_snapshot(member: str, *, allow_member_processes: bool = False) -> dict[str, Any]:
    plan = load_plan()
    topology = _cpu_topology()
    logical = sorted(_parse_cpu_list(CPUSET))
    physical = sorted({topology[cpu] for cpu in logical})
    siblings = {
        str(core): sorted(cpu for cpu, observed_core in topology.items() if observed_core == core)
        for core in physical
    }
    nightly = _systemd_state("alisios-nightly18z.service")
    pipeline = _systemd_state("alisios-pipeline.service")
    model_rows = _model_processes(topology)
    production_rows = [row for row in model_rows if row["scope"] == "production"]
    ensemble_rows = [row for row in model_rows if row["scope"] == "ensemble"]
    production_lock = _production_lock()
    controller_affinity = _process_affinity(os.getpid())
    controller_physical = sorted({topology[cpu] for cpu in controller_affinity if cpu in topology})
    memory_gib = _mem_available_gib()
    usage = shutil.disk_usage(SCRATCH.parent)
    free_gib = usage.free / (1024.0**3)
    reasons = _admission_reasons(
        topology=topology,
        production_rows=production_rows,
        ensemble_rows=ensemble_rows,
        nightly=nightly,
        production_lock=production_lock,
        memory_gib=memory_gib,
        free_gib=free_gib,
        allow_member_processes=allow_member_processes,
        ensemble_controller_affinity=controller_affinity,
    )
    controller_identity = _controller_identity(nightly, production_lock)
    identity_rows = [
        {
            "identity_sha256": row["identity_sha256"],
            "cpu_affinity": row["cpu_affinity"],
            "physical_cores": row["physical_cores"],
        }
        for row in production_rows
    ]
    payload = {
        "schema": "gpuwrf.v0234.conditioning-resource-admission.v1",
        "member": member,
        "captured_at_utc": utc_now(),
        "plan_sha256": plan["plan_sha256"],
        "verdict": "ADMITTED_ISOLATED_CPU_SET" if not reasons else "DEFER_PRODUCTION_OWNERSHIP",
        "deny_reasons": reasons,
        "alisios": {
            "nightly_service": nightly,
            "pipeline_service": pipeline,
            "production_lock": production_lock,
            "controller_identity": controller_identity,
            "production_model_process_count": len(production_rows),
            "production_wrf_process_count": sum(row["target"] == "wrf.exe" for row in production_rows),
            "production_real_process_count": sum(row["target"] == "real.exe" for row in production_rows),
            "production_model_identity_sha256": canonical_digest(identity_rows),
            "production_model_processes": production_rows,
            "ensemble_model_process_count": len(ensemble_rows),
            "ensemble_model_processes": ensemble_rows,
        },
        "cpu_ownership": {
            "requested_logical_cpuset": CPUSET,
            "requested_logical_cpus": logical,
            "observed_physical_cores": physical,
            "expected_physical_cores": list(EXPECTED_PHYSICAL_CORES),
            "sibling_map": siblings,
            "production_allowed_physical_cores": list(PRODUCTION_PHYSICAL_CORES),
            "observed_production_model_physical_cores": sorted({
                core for row in production_rows for core in row["physical_cores"]
            }),
            "physical_core_overlap_with_production_model": sorted(
                set(physical)
                & {core for row in production_rows for core in row["physical_cores"]}
            ),
            "all_production_model_processes_confined_to_cores_0_11": all(
                row["cpu_affinity"]
                and set(row["physical_cores"]).issubset(PRODUCTION_PHYSICAL_CORES)
                for row in production_rows
            ),
            "sprint_cpu_lock": str(CPU_LOCK),
            "ensemble_controller_logical_affinity": controller_affinity,
            "ensemble_controller_physical_cores": controller_physical,
        },
        "capacity": {
            "mem_available_gib": memory_gib,
            "mnt_data_free_gib": free_gib,
        },
        "gpu_attestation": {
            "queries": 0, "locks": 0, "compiles": 0, "dispatches": 0, "kernels": 0,
            "cuda_visible_devices_for_wrf": "",
        },
    }
    payload["proof_sha256"] = canonical_digest(payload, "proof_sha256")
    return payload


def _timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _compact_resource_sample(snapshot: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    def rows(scope: str) -> list[dict[str, Any]]:
        return [
            {
                "pid": row["pid"],
                "start_ticks": row["start_ticks"],
                "identity_sha256": row["identity_sha256"],
                "classification": row["classification"],
                "target": row["target"],
                "cpu_affinity": row["cpu_affinity"],
                "physical_cores": row["physical_cores"],
                "no_new_privs": row["no_new_privs"],
                "seccomp_mode": row["seccomp_mode"],
            }
            for row in snapshot["alisios"][f"{scope}_model_processes"]
        ]

    return {
        "captured_at_utc": snapshot["captured_at_utc"],
        "snapshot_canonical_sha256": snapshot["proof_sha256"],
        "reasons": reasons,
        "controller_identity_sha256": snapshot["alisios"]["controller_identity"]["identity_sha256"],
        "production_model_identity_sha256": snapshot["alisios"]["production_model_identity_sha256"],
        "production_model_processes": rows("production"),
        "ensemble_model_processes": rows("ensemble"),
        "isolated_physical_cores": snapshot["cpu_ownership"]["observed_physical_cores"],
        "production_model_physical_cores": snapshot["cpu_ownership"]["observed_production_model_physical_cores"],
        "physical_core_overlap": snapshot["cpu_ownership"]["physical_core_overlap_with_production_model"],
        "mem_available_gib": snapshot["capacity"]["mem_available_gib"],
        "mnt_data_free_gib": snapshot["capacity"]["mnt_data_free_gib"],
    }


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=10)


def _terminate_systemd_launch(proc: subprocess.Popen[bytes], unit: str) -> None:
    subprocess.run(
        ["/usr/bin/systemctl", "--user", "kill", "--kill-whom=all", "--signal=TERM", unit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    _terminate_process_group(proc)
    subprocess.run(
        ["/usr/bin/systemctl", "--user", "stop", unit],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )


def _stable_preflight(member: str) -> tuple[dict[str, Any], list[str]]:
    first = resource_snapshot(member)
    if first["deny_reasons"]:
        first["preflight_stability"] = {
            "required_samples": 2,
            "completed_samples": 1,
            "interval_seconds": RESOURCE_PREFLIGHT_INTERVAL_SECONDS,
            "first_snapshot_canonical_sha256": first["proof_sha256"],
            "stable": False,
        }
        first["proof_sha256"] = canonical_digest(first, "proof_sha256")
        return first, list(first["deny_reasons"])
    time.sleep(RESOURCE_PREFLIGHT_INTERVAL_SECONDS)
    second = resource_snapshot(member)
    reasons = resource_drift_reasons(first, second)
    second["deny_reasons"] = reasons
    second["verdict"] = "ADMITTED_ISOLATED_CPU_SET" if not reasons else "DEFER_PRODUCTION_OWNERSHIP"
    second["preflight_stability"] = {
        "required_samples": 2,
        "completed_samples": 2,
        "interval_seconds": RESOURCE_PREFLIGHT_INTERVAL_SECONDS,
        "first_captured_at_utc": first["captured_at_utc"],
        "first_snapshot_canonical_sha256": first["proof_sha256"],
        "second_captured_at_utc": second["captured_at_utc"],
        "production_identity_unchanged": not any(
            reason.startswith("production_model_process_")
            or reason.startswith("production_model_logical_")
            or reason.startswith("production_model_physical_")
            for reason in reasons
        ),
        "controller_identity_unchanged": "production_controller_identity_drift" not in reasons,
        "stable": not reasons,
    }
    second["proof_sha256"] = canonical_digest(second, "proof_sha256")
    return second, reasons


def run_member(member: str) -> int:
    plan = load_plan()
    member_dir = _member_dir(member)
    load_self_hashed(member_dir / "perturbation-manifest.json")
    run_dir = member_dir / "run"
    dumps = member_dir / "momsp_dumps"
    if (member_dir / "execution-receipt.json").exists():
        raise FileExistsError("member already has an execution receipt")
    if dumps.exists() or any(run_dir.glob("rsl.out.*")) or any(run_dir.glob("wrfout_*")):
        raise FileExistsError("member has prior runtime artifacts; preserve it and use no implicit rerun")
    CPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with CPU_LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            snapshot = resource_snapshot(member)
            snapshot["verdict"] = "DEFER_CPU_SET_LOCK_BUSY"
            snapshot["deny_reasons"].append("sprint_cpu_lock_busy")
            snapshot["proof_sha256"] = canonical_digest(snapshot, "proof_sha256")
            path = member_dir / "admissions" / f"{_timestamp_token()}-deferred.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
            return 75
        snapshot, preflight_reasons = _stable_preflight(member)
        admitted = snapshot["verdict"] == "ADMITTED_ISOLATED_CPU_SET"
        suffix = "admitted" if admitted else "deferred"
        admission_path = member_dir / "admissions" / f"{_timestamp_token()}-{suffix}.json"
        admission_path.parent.mkdir(parents=True, exist_ok=True)
        admission_path.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
        if not admitted:
            print(json.dumps({"member": member, "verdict": snapshot["verdict"], "reasons": preflight_reasons}, sort_keys=True))
            return 75

        dumps.mkdir(parents=True, exist_ok=False)
        unit = f"v0234-conditioning-{member}-{_timestamp_token().lower()}"
        payload_command = wrf_payload_command()
        environment = wrf_launch_environment(dumps)
        command = systemd_launch_command(unit, run_dir, dumps)
        log_path = member_dir / "launch.log"
        monitor_path = member_dir / "resource-monitor.json"
        started = utc_now()
        before_ns = time.time_ns()
        monitor_samples: list[dict[str, Any]] = []
        resource_violation: dict[str, Any] | None = None
        active_unit_sample_count = 0
        first_active_unit: dict[str, Any] | None = None
        last_active_unit: dict[str, Any] | None = None
        exact_rank_sample_seen = False
        max_wrf_rank_count = 0
        with log_path.open("wb") as log_stream:
            proc = subprocess.Popen(
                command, cwd=REPO,
                stdout=log_stream, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while True:
                    try:
                        live = resource_snapshot(member, allow_member_processes=True)
                        reasons = resource_drift_reasons(snapshot, live)
                        unit_state = _affinity_unit_state(unit)
                        unit_running = (
                            unit_state.get("LoadState") == "loaded"
                            and unit_state.get("ActiveState") in {"activating", "active"}
                        )
                        ensemble_rows = live["alisios"]["ensemble_model_processes"]
                        if unit_running:
                            reasons.extend(affinity_unit_reasons(unit_state))
                            active_unit_sample_count += 1
                            if first_active_unit is None:
                                first_active_unit = unit_state
                            last_active_unit = unit_state
                        elif ensemble_rows:
                            reasons.append("affinity_unit_not_running_with_member_process")
                        elif (
                            first_active_unit is None
                            and time.time_ns() - before_ns > 5_000_000_000
                            and proc.poll() is None
                        ):
                            reasons.append("affinity_unit_start_timeout")
                        wrf_rank_count = sum(
                            row["classification"] == "wrf_rank" for row in ensemble_rows
                        )
                        max_wrf_rank_count = max(max_wrf_rank_count, wrf_rank_count)
                        if wrf_rank_count > MPI_RANKS:
                            reasons.append("ensemble_wrf_rank_count_exceeds_12")
                        if wrf_rank_count == MPI_RANKS:
                            exact_rank_sample_seen = True
                        reasons = sorted(set(reasons))
                        sample = _compact_resource_sample(live, reasons)
                        sample["affinity_unit"] = unit_state
                        sample["wrf_rank_count"] = wrf_rank_count
                    except Exception as exc:  # fail closed around /proc/systemd races
                        reasons = ["resource_monitor_exception"]
                        sample = {
                            "captured_at_utc": utc_now(),
                            "reasons": reasons,
                            "exception": f"{type(exc).__name__}: {exc}",
                        }
                    monitor_samples.append(sample)
                    if reasons:
                        resource_violation = {
                            "captured_at_utc": sample["captured_at_utc"],
                            "sample_index": len(monitor_samples) - 1,
                            "reasons": reasons,
                        }
                        _terminate_systemd_launch(proc, unit)
                        break
                    if proc.poll() is not None:
                        break
                    time.sleep(RESOURCE_MONITOR_INTERVAL_SECONDS)
            except BaseException:
                _terminate_systemd_launch(proc, unit)
                raise
            proc.wait()
        if resource_violation is None and (active_unit_sample_count == 0 or not exact_rank_sample_seen):
            post_reasons = []
            if active_unit_sample_count == 0:
                post_reasons.append("affinity_unit_active_sample_not_observed")
            if not exact_rank_sample_seen:
                post_reasons.append("exact_12_wrf_rank_sample_not_observed")
            resource_violation = {
                "captured_at_utc": utc_now(),
                "sample_index": len(monitor_samples),
                "reasons": post_reasons,
            }
        after_ns = time.time_ns()
        ended = utc_now()
        monitor_payload = {
            "schema": "gpuwrf.v0234.conditioning-continuous-resource-monitor.v1",
            "member": member,
            "plan_sha256": plan["plan_sha256"],
            "admission": {
                "path": str(admission_path),
                "file_sha256": sha256_file(admission_path),
                "canonical_sha256": snapshot["proof_sha256"],
            },
            "started_at_utc": started,
            "ended_at_utc": ended,
            "interval_seconds": RESOURCE_MONITOR_INTERVAL_SECONDS,
            "sample_count": len(monitor_samples),
            "baseline_production_model_identity_sha256": snapshot["alisios"]["production_model_identity_sha256"],
            "baseline_controller_identity_sha256": snapshot["alisios"]["controller_identity"]["identity_sha256"],
            "baseline_production_model_processes": snapshot["alisios"]["production_model_processes"],
            "reserved_logical_cpuset": CPUSET,
            "reserved_physical_cores": list(EXPECTED_PHYSICAL_CORES),
            "production_allowed_physical_cores": list(PRODUCTION_PHYSICAL_CORES),
            "hard_affinity_boundary": {
                "controller_taskset": CPUSET,
                "service_cpu_affinity": SYSTEMD_CPU_AFFINITY,
                "denied_system_call": "sched_setaffinity",
                "system_call_filter": SYSTEMD_AFFINITY_SYSCALL_FILTER,
                "system_call_error_number": "EPERM",
                "no_new_privileges": True,
                "systemd_unit": unit,
                "first_active_unit_state": first_active_unit,
                "last_active_unit_state": last_active_unit,
                "active_unit_sample_count": active_unit_sample_count,
                "rank_placement": verify_cpu_placement(),
                "payload_mpi_command": payload_command,
                "exact_12_wrf_rank_sample_seen": exact_rank_sample_seen,
                "maximum_wrf_rank_count_observed": max_wrf_rank_count,
            },
            "all_samples_safe": resource_violation is None,
            "resource_violation": resource_violation,
            "samples": monitor_samples,
            "gpu_attestation": snapshot["gpu_attestation"],
        }
        write_self_hashed(monitor_path, monitor_payload)
        rsl0 = run_dir / "rsl.error.0000"
        wrf_reported_success = rsl0.is_file() and "SUCCESS COMPLETE WRF" in rsl0.read_text(errors="replace")
        success = wrf_reported_success and resource_violation is None
        dump_files = sorted(path for path in dumps.rglob("*") if path.is_file())
        tag_counts = Counter(
            re.search(r"_(sp\d_[^_]+)__", path.name).group(1)
            if re.search(r"_(sp\d_[^_]+)__", path.name) else "meta"
            for path in dump_files
        )
        frames = {}
        for name in (
            "wrfout_d01_2025-03-01_00:00:00",
            "wrfout_d02_2025-03-01_00:00:00",
            "wrfout_d03_2025-03-01_00:00:00",
            "wrfout_d03_2025-03-01_00:20:00",
        ):
            path = run_dir / name
            frames[name] = {
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        receipt = {
            "schema": "gpuwrf.v0234.conditioning-member-execution.v1",
            "member": member,
            "plan_sha256": plan["plan_sha256"],
            "admission": {
                "path": str(admission_path),
                "file_sha256": sha256_file(admission_path),
                "canonical_sha256": snapshot["proof_sha256"],
                "captured_at_utc": snapshot["captured_at_utc"],
                "verdict": snapshot["verdict"],
            },
            "started_at_utc": started,
            "ended_at_utc": ended,
            "elapsed_seconds": (after_ns - before_ns) / 1e9,
            "command": command,
            "systemd_affinity_unit": unit,
            "payload_mpi_command": payload_command,
            "environment": environment,
            "returncode": proc.returncode,
            "success_complete_wrf": success,
            "wrf_reported_success_complete": wrf_reported_success,
            "resource_monitor": {
                "path": str(monitor_path),
                "file_sha256": sha256_file(monitor_path),
                "canonical_sha256": load_self_hashed(monitor_path)["proof_sha256"],
                "sample_count": len(monitor_samples),
                "all_samples_safe": resource_violation is None,
                "resource_violation": resource_violation,
            },
            "launch_log": {"path": str(log_path), "bytes": log_path.stat().st_size, "sha256": sha256_file(log_path)},
            "rsl_error_0000": {
                "path": str(rsl0), "bytes": rsl0.stat().st_size if rsl0.is_file() else None,
                "sha256": sha256_file(rsl0) if rsl0.is_file() else None,
            },
            "raw_dump_inventory": {
                "file_count": len(dump_files),
                "bytes": sum(path.stat().st_size for path in dump_files),
                "tag_counts": dict(sorted(tag_counts.items())),
                "rank_directory_count": len(tuple(dumps.glob("rank*"))),
                "required_sp1_sp4_file_count": sum(
                    1 for path in dump_files if "_sp1_entry__" in path.name or "_sp4_exit__" in path.name
                ),
            },
            "wrfout_frames": frames,
            "gpu_attestation": snapshot["gpu_attestation"],
        }
        write_self_hashed(member_dir / "execution-receipt.json", receipt)
        if proc.returncode != 0 or not success:
            print(json.dumps({"member": member, "returncode": proc.returncode, "success_complete_wrf": success}, sort_keys=True))
            return proc.returncode or 1
        if len(tuple(dumps.glob("rank*"))) != MPI_RANKS:
            raise RuntimeError("WRF completed but rank dump count is not 12")
        if len(dump_files) != MPI_RANKS * (1 + 200 * 11):
            raise RuntimeError(
                f"WRF completed but full savepoint inventory is incomplete: {len(dump_files)}"
            )
        if receipt["raw_dump_inventory"]["required_sp1_sp4_file_count"] != MPI_RANKS * 200 * 4:
            raise RuntimeError("WRF completed but SP1/SP4 inventory is incomplete")
        print(json.dumps({"member": member, "returncode": proc.returncode, "elapsed_seconds": receipt["elapsed_seconds"]}, sort_keys=True))
        return 0


def _rmse(diff: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))))


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _mass_lowest(diff: np.ndarray, field: str) -> np.ndarray:
    if field == "u":
        return 0.5 * (diff[0, :, :-1] + diff[0, :, 1:])
    if field == "v":
        return 0.5 * (diff[0, :-1, :] + diff[0, 1:, :])
    raise ValueError(field)


def _top_fraction_indices(values: np.ndarray, fraction: float = 0.05) -> set[int]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    count = max(1, int(math.ceil(flat.size * fraction)))
    indices = np.argpartition(flat, flat.size - count)[-count:]
    return {int(value) for value in indices}


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _location(diff: np.ndarray, field: str, hgt: np.ndarray, landmask: np.ndarray) -> dict[str, Any]:
    k, j, i = (int(value) for value in np.unravel_index(np.argmax(np.abs(diff)), diff.shape))
    mass_j = min(j, hgt.shape[0] - 1)
    mass_i = min(i, hgt.shape[1] - 1)
    return {
        "k": k, "j": j, "i": i,
        "mass_j": mass_j, "mass_i": mass_i,
        "hgt_m": float(hgt[mass_j, mass_i]),
        "landmask": float(landmask[mass_j, mass_i]),
        "abs_error": float(abs(diff[k, j, i])),
        "field_stagger": field,
    }


def _field_metrics(diff: np.ndarray, state: np.ndarray, baseline: np.ndarray, field: str, hgt: np.ndarray, landmask: np.ndarray, terrain_grad: np.ndarray) -> dict[str, Any]:
    rmse = _rmse(diff)
    low = _mass_lowest(diff, field)
    vertical_rmse = [
        float(np.sqrt(np.mean(np.square(level, dtype=np.float64)))) for level in diff
    ]
    vertical_bias = [float(np.mean(level, dtype=np.float64)) for level in diff]
    return {
        "rmse": rmse,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_bias": float(np.mean(diff, dtype=np.float64)),
        "normalized_abs_bias": float(abs(np.mean(diff, dtype=np.float64)) / max(rmse, 1e-300)),
        "state_spatial_correlation": _safe_corr(state, baseline),
        "lowest_level_terrain_gradient_correlation": _safe_corr(np.abs(low), terrain_grad),
        "vertical_rmse": vertical_rmse,
        "vertical_bias": vertical_bias,
        "argmax": _location(diff, field, hgt, landmask),
        "cell_count": int(diff.size),
        "sum_squared_error": float(np.sum(np.square(diff, dtype=np.float64), dtype=np.float64)),
    }


def _load_gpu(tag: str, field: str, step: int, baseline_shape: tuple[int, ...]) -> np.ndarray:
    array = np.load(GPU_SAVEPOINTS / f"step{step:06d}_{tag}__{field}.npy")
    if array.shape == baseline_shape:
        return array
    if array.ndim == 3 and array.shape[0] == baseline_shape[0] + 1 and array.shape[1:] == baseline_shape[1:]:
        return array[: baseline_shape[0]]
    raise RuntimeError(f"GPU {tag}/{field}/{step} shape {array.shape} != {baseline_shape}")


def _combined(first: dict[str, Any], second: dict[str, Any]) -> float:
    return float(math.sqrt(
        (first["sum_squared_error"] + second["sum_squared_error"])
        / (first["cell_count"] + second["cell_count"])
    ))


def analyze_member(member: str) -> None:
    plan = load_plan()
    member_dir = _member_dir(member)
    spec = _member_spec(member)
    perturbation = load_self_hashed(member_dir / "perturbation-manifest.json")
    execution = load_self_hashed(member_dir / "execution-receipt.json")
    if execution["returncode"] != 0 or not execution["success_complete_wrf"]:
        raise RuntimeError("cannot analyze unsuccessful WRF member")
    output = member_dir / "member-analysis.json"
    if output.exists():
        raise FileExistsError(output)
    dumps = member_dir / "momsp_dumps"
    ranks = reassemble.load_ranks(dumps)
    with _open_netcdf(BASE_RUN / "wrfout_d03_2025-03-01_00:00:00") as dataset:
        hgt = np.asarray(dataset.variables["HGT"][0], dtype=np.float64)
        landmask = np.asarray(dataset.variables["LANDMASK"][0], dtype=np.float64)
    gy, gx = np.gradient(hgt)
    terrain_grad = np.hypot(gx, gy)
    per_step = []
    control_equal = True
    for step in STEPS:
        row: dict[str, Any] = {"step": step, "fields": {}, "combined": {}}
        for tag in ("sp1_entry", "sp4_exit"):
            member_metrics = {}
            gpu_metrics = {}
            for field in ("u", "v"):
                key = f"{tag}__{field}"
                member_state = reassemble.reassemble3d(key, step, ranks)
                baseline = np.load(BASE_CACHE / f"step{step:06d}_{key}.npy")
                gpu_state = _load_gpu(tag, field, step, baseline.shape)
                member_diff = member_state - baseline
                gpu_diff = gpu_state - baseline
                member_row = _field_metrics(member_diff, member_state, baseline, field, hgt, landmask, terrain_grad)
                gpu_row = _field_metrics(gpu_diff, gpu_state, baseline, field, hgt, landmask, terrain_grad)
                member_low = np.abs(_mass_lowest(member_diff, field))
                gpu_low = np.abs(_mass_lowest(gpu_diff, field))
                signature = {
                    "signed_error_correlation_with_gpu": _safe_corr(member_diff, gpu_diff),
                    "magnitude_error_correlation_with_gpu": _safe_corr(np.abs(member_diff), np.abs(gpu_diff)),
                    "vertical_rmse_profile_correlation_with_gpu": _safe_corr(
                        np.asarray(member_row["vertical_rmse"]), np.asarray(gpu_row["vertical_rmse"])
                    ),
                    "lowest_level_top5pct_jaccard_with_gpu": _jaccard(
                        _top_fraction_indices(member_low), _top_fraction_indices(gpu_low)
                    ),
                }
                row["fields"][key] = {
                    "member_vs_wrf": member_row,
                    "gpu_vs_wrf": gpu_row,
                    "member_gpu_signature": signature,
                }
                member_metrics[field] = member_row
                gpu_metrics[field] = gpu_row
                if member_row["max_abs"] != 0.0:
                    control_equal = False
            row["combined"][tag] = {
                "member_vs_wrf_rmse": _combined(member_metrics["u"], member_metrics["v"]),
                "gpu_vs_wrf_rmse": _combined(gpu_metrics["u"], gpu_metrics["v"]),
            }
        per_step.append(row)
        if step % 25 == 0:
            print(f"{member}: analyzed step {step}", flush=True)

    frame_identity = {}
    for name, base_row in execution["wrfout_frames"].items():
        base_hash = sha256_file(BASE_RUN / name)
        frame_identity[name] = {
            "member_sha256": base_row["sha256"],
            "canonical_sha256": base_hash,
            "byte_identical": base_row["sha256"] == base_hash,
        }
    if spec["kind"] == "control":
        control_gate = {
            "all_800_reassembled_SP1_SP4_UV_arrays_bitwise_equal": control_equal,
            "all_four_wrfout_frames_byte_identical": all(row["byte_identical"] for row in frame_identity.values()),
            "passed": control_equal and all(row["byte_identical"] for row in frame_identity.values()),
        }
    else:
        control_gate = None
        if per_step[0]["combined"]["sp1_entry"]["member_vs_wrf_rmse"] == 0.0:
            raise RuntimeError("perturbed member has zero SP1 seed at step 1")
    payload = {
        "schema": "gpuwrf.v0234.conditioning-member-analysis.v1",
        "member": member,
        "member_spec": spec,
        "plan_sha256": plan["plan_sha256"],
        "perturbation_manifest": {
            "path": str(member_dir / "perturbation-manifest.json"),
            "file_sha256": sha256_file(member_dir / "perturbation-manifest.json"),
            "canonical_sha256": perturbation["proof_sha256"],
        },
        "execution_receipt": {
            "path": str(member_dir / "execution-receipt.json"),
            "file_sha256": sha256_file(member_dir / "execution-receipt.json"),
            "canonical_sha256": execution["proof_sha256"],
        },
        "frame_identity": frame_identity,
        "control_reproducibility": control_gate,
        "per_step": per_step,
    }
    write_self_hashed(output, payload)
    print(json.dumps({"member": member, "analysis": str(output), "canonical_sha256": load_self_hashed(output)["proof_sha256"]}, sort_keys=True))


def archive_member(member: str) -> None:
    member_dir = _member_dir(member)
    load_self_hashed(member_dir / "member-analysis.json")
    dumps = member_dir / "momsp_dumps"
    if not dumps.is_dir() or dumps.is_symlink() or dumps.parent != member_dir:
        raise RuntimeError(f"unsafe or missing dumps target: {dumps}")
    evidence = member_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    archive = evidence / "momsp-dumps.tar.zst"
    inventory_path = evidence / "raw-inventory.json"
    receipt_path = member_dir / "archive-receipt.json"
    if archive.exists() or inventory_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite archived evidence")
    files = sorted(path for path in dumps.rglob("*") if path.is_file())
    listing = [
        {"name": str(path.relative_to(dumps)), "bytes": path.stat().st_size}
        for path in files
    ]
    inventory = {
        "schema": "gpuwrf.v0234.conditioning-raw-inventory.v1",
        "member": member,
        "file_count": len(listing),
        "raw_bytes": sum(row["bytes"] for row in listing),
        "listing": listing,
    }
    write_self_hashed(inventory_path, inventory)
    command = [
        "/usr/bin/taskset", "-c", CPUSET,
        "/usr/bin/nice", "-n", "15",
        "/usr/bin/ionice", "-c", "3",
        "tar", "--sort=name", "-I", "zstd -3 -T4", "-cf", str(archive),
        "-C", str(dumps.parent), dumps.name,
    ]
    subprocess.run(command, check=True)
    subprocess.run(["zstd", "-q", "-t", str(archive)], check=True)
    listing_proc = subprocess.run(
        ["tar", "-tf", str(archive)], check=True, text=True, stdout=subprocess.PIPE,
    )
    archived_files = [line for line in listing_proc.stdout.splitlines() if line and not line.endswith("/")]
    if len(archived_files) != len(listing):
        raise RuntimeError(f"archive file count {len(archived_files)} != {len(listing)}")
    receipt = {
        "schema": "gpuwrf.v0234.conditioning-archive-receipt.v1",
        "member": member,
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "zstd_test_passed": True,
            "tar_file_count": len(archived_files),
        },
        "raw_inventory": {
            "path": str(inventory_path),
            "bytes": inventory_path.stat().st_size,
            "file_sha256": sha256_file(inventory_path),
            "canonical_sha256": load_self_hashed(inventory_path)["proof_sha256"],
            "raw_bytes": inventory["raw_bytes"],
            "file_count": inventory["file_count"],
        },
        "archive_command": command,
        "recover_command": f"tar -I zstd -xf {archive} -C {member_dir}",
        "lossless_and_recoverable": True,
    }
    write_self_hashed(receipt_path, receipt)
    # The exact explicit member-local directory is replaced only after the
    # archive hash, zstd integrity test, tar listing, and receipt all succeed.
    shutil.rmtree(dumps)
    print(json.dumps({"member": member, "archive": str(archive), "sha256": receipt["archive"]["sha256"]}, sort_keys=True))


def _median_nonnull(values: Iterable[float | None]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        raise ValueError("no finite values for median")
    return float(np.median(np.asarray(clean, dtype=np.float64)))


def classify_decision(statistics: dict[str, Any], thresholds: dict[str, Any] = DECISION_THRESHOLDS) -> tuple[str, dict[str, bool]]:
    primary = thresholds["primary_growth"]
    zero = thresholds["zero_mean"]
    spatial = thresholds["spatial_signature"]
    vertical = thresholds["vertical_signature"]
    hard = thresholds["hard_structural_falsifier"]
    gates = {
        "primary_growth_support": (
            statistics["literal_envelope_inside_fraction"]
            >= primary["literal_envelope_inside_fraction_min"]
            and statistics["milestone_gpu_to_member_max_ratio_max"]
            <= primary["milestone_gpu_to_member_max_ratio_max"]
        ),
        "zero_mean_pass": (
            statistics["gpu_normalized_bias_p95"] <= zero["gpu_p95_max"]
            and statistics["ensemble_normalized_bias_p95"] <= zero["ensemble_p95_max"]
            and statistics["gpu_ensemble_normalized_bias_p95_gap"]
            <= zero["gpu_ensemble_p95_gap_max"]
        ),
        "spatial_signature_pass": (
            statistics["magnitude_correlation_median"]
            >= spatial["magnitude_correlation_median_min"]
            and statistics["top5pct_jaccard_median"]
            >= spatial["top5pct_jaccard_median_min"]
            and statistics["gpu_terrain_correlation_median"]
            >= spatial["terrain_gradient_correlation_median_min"]
            and statistics["ensemble_terrain_correlation_median"]
            >= spatial["terrain_gradient_correlation_median_min"]
            and statistics["gpu_ensemble_terrain_median_gap"]
            <= spatial["gpu_ensemble_terrain_median_gap_max"]
            and statistics["high_terrain_argmax_fraction"]
            >= spatial["high_terrain_argmax_fraction_min"]
        ),
        "vertical_signature_pass": (
            statistics["vertical_profile_correlation_median"]
            >= vertical["profile_correlation_median_min"]
        ),
        "primary_scale_falsifier": (
            statistics["gpu_over_material_factor_fraction"]
            >= primary["falsifier_fraction_min"]
            and statistics["all_milestones_gpu_over_material_factor"]
        ),
    }
    hard_flags = {
        "magnitude": statistics["magnitude_correlation_median"] <= hard["magnitude_correlation_median_max"],
        "top5pct": statistics["top5pct_jaccard_median"] <= hard["top5pct_jaccard_median_max"],
        "vertical": statistics["vertical_profile_correlation_median"] <= hard["vertical_profile_correlation_median_max"],
        "terrain": statistics["terrain_opposite_sign_hard_failure"],
    }
    gates["hard_structural_failure_count"] = sum(hard_flags.values())  # type: ignore[assignment]
    gates["hard_structural_falsifier"] = (
        sum(hard_flags.values()) >= hard["required_independent_failures"]
    )
    if (
        gates["primary_growth_support"]
        and gates["zero_mean_pass"]
        and gates["spatial_signature_pass"]
        and gates["vertical_signature_pass"]
    ):
        verdict = "WRF_CONDITIONING_ENSEMBLE_SUPPORTS_RELEASE_FRAMING"
    elif gates["primary_scale_falsifier"] or gates["hard_structural_falsifier"]:
        verdict = "WRF_CONDITIONING_ENSEMBLE_FALSIFIES_CONDITIONING"
    else:
        verdict = "WRF_CONDITIONING_ENSEMBLE_INCONCLUSIVE"
    gates["hard_structural_flags"] = hard_flags  # type: ignore[assignment]
    return verdict, gates


def aggregate() -> None:
    plan = load_plan()
    analyses = {
        spec["id"]: load_self_hashed(_member_dir(spec["id"]) / "member-analysis.json")
        for spec in MEMBER_SPECS
    }
    control = analyses["control-repeat"]
    if not control["control_reproducibility"]["passed"]:
        raise RuntimeError("control reproducibility gate failed; scientific aggregate is not valid")
    perturbed_ids = [spec["id"] for spec in MEMBER_SPECS if spec["kind"] == "perturbed"]
    envelope_rows = []
    inside_count = 0
    material_count = 0
    milestone_ratios = []
    normalized_member_bias = []
    normalized_gpu_bias = []
    magnitude_corr = []
    top_jaccard = []
    vertical_corr = []
    member_terrain = []
    gpu_terrain = []
    high_terrain_count = 0
    high_terrain_total = 0
    high_terrain_threshold = DECISION_THRESHOLDS["spatial_signature"]["high_terrain_argmax_m"]
    material_factor = DECISION_THRESHOLDS["primary_growth"]["materially_smaller_factor"]
    for step_index, step in enumerate(STEPS):
        member_rmse = [
            analyses[member]["per_step"][step_index]["combined"]["sp4_exit"]["member_vs_wrf_rmse"]
            for member in perturbed_ids
        ]
        gpu_rmse_values = [
            analyses[member]["per_step"][step_index]["combined"]["sp4_exit"]["gpu_vs_wrf_rmse"]
            for member in perturbed_ids
        ]
        if len(set(gpu_rmse_values)) != 1:
            raise RuntimeError(f"GPU trajectory drift across analyses at step {step}")
        gpu_rmse = gpu_rmse_values[0]
        lower, upper = min(member_rmse), max(member_rmse)
        inside = lower <= gpu_rmse <= upper
        ratio = gpu_rmse / max(upper, 1e-300)
        material = ratio > material_factor
        inside_count += int(inside)
        material_count += int(material)
        if step in MILESTONES:
            milestone_ratios.append(ratio)
        envelope_rows.append({
            "step": step,
            "member_rmse": dict(zip(perturbed_ids, member_rmse, strict=True)),
            "wrf_envelope_min": lower,
            "wrf_envelope_max": upper,
            "gpu_rmse": gpu_rmse,
            "gpu_inside_literal_envelope": inside,
            "gpu_to_member_max_ratio": ratio,
            "gpu_over_material_factor": material,
        })
        for field in ("u", "v"):
            key = f"sp4_exit__{field}"
            gpu_row = control["per_step"][step_index]["fields"][key]["gpu_vs_wrf"]
            normalized_gpu_bias.append(gpu_row["normalized_abs_bias"])
            if step in MILESTONES:
                gpu_terrain.append(gpu_row["lowest_level_terrain_gradient_correlation"])
                high_terrain_count += int(gpu_row["argmax"]["hgt_m"] >= high_terrain_threshold)
                high_terrain_total += 1
            for member in perturbed_ids:
                field_row = analyses[member]["per_step"][step_index]["fields"][key]
                member_row = field_row["member_vs_wrf"]
                normalized_member_bias.append(member_row["normalized_abs_bias"])
                if step in MILESTONES:
                    signature = field_row["member_gpu_signature"]
                    magnitude_corr.append(signature["magnitude_error_correlation_with_gpu"])
                    top_jaccard.append(signature["lowest_level_top5pct_jaccard_with_gpu"])
                    vertical_corr.append(signature["vertical_rmse_profile_correlation_with_gpu"])
                    member_terrain.append(member_row["lowest_level_terrain_gradient_correlation"])
                    high_terrain_count += int(member_row["argmax"]["hgt_m"] >= high_terrain_threshold)
                    high_terrain_total += 1
    gpu_bias_p95 = percentile(normalized_gpu_bias, 95.0)
    member_bias_p95 = percentile(normalized_member_bias, 95.0)
    gpu_terrain_median = _median_nonnull(gpu_terrain)
    member_terrain_median = _median_nonnull(member_terrain)
    hard_terrain_threshold = DECISION_THRESHOLDS["hard_structural_falsifier"]["terrain_opposite_sign_min_abs"]
    statistics = {
        "literal_envelope_inside_fraction": inside_count / len(STEPS),
        "milestone_gpu_to_member_max_ratio_max": max(milestone_ratios),
        "gpu_over_material_factor_fraction": material_count / len(STEPS),
        "all_milestones_gpu_over_material_factor": all(value > material_factor for value in milestone_ratios),
        "gpu_normalized_bias_p95": gpu_bias_p95,
        "ensemble_normalized_bias_p95": member_bias_p95,
        "gpu_ensemble_normalized_bias_p95_gap": abs(gpu_bias_p95 - member_bias_p95),
        "magnitude_correlation_median": _median_nonnull(magnitude_corr),
        "top5pct_jaccard_median": _median_nonnull(top_jaccard),
        "vertical_profile_correlation_median": _median_nonnull(vertical_corr),
        "gpu_terrain_correlation_median": gpu_terrain_median,
        "ensemble_terrain_correlation_median": member_terrain_median,
        "gpu_ensemble_terrain_median_gap": abs(gpu_terrain_median - member_terrain_median),
        "high_terrain_argmax_fraction": high_terrain_count / high_terrain_total,
        "terrain_opposite_sign_hard_failure": (
            gpu_terrain_median * member_terrain_median < 0
            and abs(gpu_terrain_median) >= hard_terrain_threshold
            and abs(member_terrain_median) >= hard_terrain_threshold
        ),
    }
    verdict, gates = classify_decision(statistics)
    payload = {
        "schema": "gpuwrf.v0234.wrf-conditioning-ensemble-analysis.v1",
        "verdict": verdict,
        "plan_sha256": plan["plan_sha256"],
        "control_reproducibility": control["control_reproducibility"],
        "perturbed_members": perturbed_ids,
        "statistics": statistics,
        "gates": gates,
        "thresholds_applied_unchanged": DECISION_THRESHOLDS,
        "growth_envelope": envelope_rows,
        "decisive_growth": {
            "step1": envelope_rows[0],
            "step200": envelope_rows[-1],
            "inside_steps": inside_count,
            "total_steps": len(STEPS),
        },
        "exact_next_kimi_assignment": plan["next_assignment_by_verdict"][verdict],
    }
    output = SPRINT / "ensemble-analysis.json"
    if output.exists():
        raise FileExistsError(output)
    write_self_hashed(output, payload)
    print(json.dumps({"verdict": verdict, "statistics": statistics, "gates": gates, "proof_sha256": load_self_hashed(output)["proof_sha256"]}, indent=1, sort_keys=True))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _artifact_row(path: Path, self_hashed: bool = False, key: str = "proof_sha256") -> dict[str, Any]:
    row = {"path": str(path.resolve()), "bytes": path.stat().st_size, "file_sha256": sha256_file(path)}
    if self_hashed:
        row["canonical_sha256"] = load_self_hashed(path, key)[key]
    return row


def terminal_proof() -> None:
    plan = load_plan()
    authority_audit = load_self_hashed(SPRINT / "authority-audit.json")
    analysis = load_self_hashed(SPRINT / "ensemble-analysis.json")
    override_path = SPRINT / "principal-resource-override.json"
    override = load_self_hashed(override_path)
    boundary_probe_path = SPRINT / "cpu-boundary-probe.json"
    boundary_probe = load_self_hashed(boundary_probe_path)
    boundary_correction_path = SPRINT / "resource-boundary-correction.json"
    boundary_correction = load_self_hashed(boundary_correction_path)
    if boundary_probe["verdict"] != "PASS_EXACT_12_RANK_KERNEL_LOCKED_DISJOINT_CPU_BOUNDARY":
        raise RuntimeError("focused kernel CPU-boundary probe is not green")
    if boundary_correction["correction"]["frozen_science_criteria_changed"]:
        raise RuntimeError("resource correction claims a science-criterion change")
    if override["revised_preregistration"]["plan_canonical_sha256"] != plan["plan_sha256"]:
        raise RuntimeError("principal resource override is not bound to the active preregistration")
    worker_report = SPRINT / "worker-report.md"
    command_log = SPRINT / "command-log.md"
    register_markdown = SPRINT / "metrics-and-thresholds-register.md"
    register_json_path = SPRINT / "metrics-and-thresholds-register.json"
    register = load_self_hashed(register_json_path)
    if not register_markdown.is_file():
        raise RuntimeError("metrics-and-thresholds-register.md must exist before proof sealing")
    if register["ensemble_binding"]["plan_canonical_sha256"] != plan["plan_sha256"]:
        raise RuntimeError("threshold register is not bound to the active preregistration")
    if register["ensemble_binding"]["ensemble_analysis_canonical_sha256"] != analysis["proof_sha256"]:
        raise RuntimeError("threshold register is not bound to the terminal ensemble analysis")
    if not worker_report.is_file() or not command_log.is_file():
        raise RuntimeError("worker-report.md and command-log.md must exist before proof sealing")
    prereg_commit = _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(REPO)))
    if not prereg_commit:
        raise RuntimeError("ensemble plan is not committed")
    committed_plan = subprocess.run(
        ["git", "show", f"{prereg_commit}:{PLAN_PATH.relative_to(REPO)}"],
        cwd=REPO, check=True, stdout=subprocess.PIPE,
    ).stdout
    if hashlib.sha256(committed_plan).hexdigest() != sha256_file(PLAN_PATH):
        raise RuntimeError("working plan differs from preregistration commit")
    boundary_commit = _git(
        "log", "-1", "--format=%H", "--", str(boundary_correction_path.relative_to(REPO))
    )
    if not boundary_commit:
        raise RuntimeError("resource-boundary correction is not committed")
    member_rows = {}
    latest_admitted = {}
    for spec in MEMBER_SPECS:
        member = spec["id"]
        member_dir = _member_dir(member)
        admissions = sorted((member_dir / "admissions").glob("*-admitted.json"))
        if len(admissions) != 1:
            raise RuntimeError(f"{member}: expected exactly one admitted launch, got {len(admissions)}")
        admission = load_self_hashed(admissions[0])
        if admission["verdict"] != "ADMITTED_ISOLATED_CPU_SET":
            raise RuntimeError(f"{member}: admission not green")
        latest_admitted[member] = admission
        monitor = load_self_hashed(member_dir / "resource-monitor.json")
        if not monitor["all_samples_safe"]:
            raise RuntimeError(f"{member}: continuous resource monitor not green")
        hard_boundary = monitor["hard_affinity_boundary"]
        if (
            not hard_boundary["exact_12_wrf_rank_sample_seen"]
            or hard_boundary["maximum_wrf_rank_count_observed"] != MPI_RANKS
            or hard_boundary["system_call_filter"] != SYSTEMD_AFFINITY_SYSCALL_FILTER
        ):
            raise RuntimeError(f"{member}: kernel CPU-boundary proof incomplete")
        archive = load_self_hashed(member_dir / "archive-receipt.json")
        member_rows[member] = {
            "member_spec": spec,
            "perturbation": _artifact_row(member_dir / "perturbation-manifest.json", True),
            "admission": _artifact_row(admissions[0], True),
            "resource_monitor": _artifact_row(member_dir / "resource-monitor.json", True),
            "execution": _artifact_row(member_dir / "execution-receipt.json", True),
            "analysis": _artifact_row(member_dir / "member-analysis.json", True),
            "archive_receipt": _artifact_row(member_dir / "archive-receipt.json", True),
            "raw_archive": archive["archive"],
            "raw_inventory": archive["raw_inventory"],
        }
    base_commit = "1eabc5d6c20b804f4a40d7ca27d5d01e7f5386dd"
    current_model_tree = _git("rev-parse", "HEAD:src/gpuwrf")
    base_model_tree = _git("rev-parse", f"{base_commit}:src/gpuwrf")
    model_diff = _git("diff", "--name-only", base_commit, "HEAD", "--", "src/gpuwrf")
    if model_diff:
        raise RuntimeError(f"production model changed during ensemble sprint: {model_diff}")
    authority = _verify_authority()
    changed_names = set(
        filter(
            None,
            _git("diff", "--name-only", base_commit, "HEAD").splitlines()
            + _git("diff", "--name-only").splitlines()
            + _git("ls-files", "--others", "--exclude-standard").splitlines(),
        )
    )
    changed_names.discard(str((SPRINT / "proof.json").relative_to(REPO)))
    changed_names.discard(str((SPRINT / "retained-evidence-manifest.json").relative_to(REPO)))
    changed_files = {
        name: _artifact_row(REPO / name)
        for name in sorted(changed_names)
        if (REPO / name).is_file()
    }
    proof = {
        "schema": "gpuwrf.v0234.wrf-conditioning-ensemble-terminal-proof.v1",
        "verdict": analysis["verdict"],
        "objective": plan["objective"],
        "frozen_authority": authority,
        "independent_authority_audit": _artifact_row(SPRINT / "authority-audit.json", True),
        "preregistration": {
            "commit": prereg_commit,
            "plan": _artifact_row(PLAN_PATH),
            "plan_canonical_sha256": plan["plan_sha256"],
            "predates_all_admitted_member_launches": all(
                subprocess.run(
                    ["git", "show", "-s", "--format=%ct", prereg_commit],
                    cwd=REPO, check=True, text=True, stdout=subprocess.PIPE,
                ).stdout.strip()
                and datetime.fromisoformat(admission["captured_at_utc"]).timestamp()
                > int(_git("show", "-s", "--format=%ct", prereg_commit))
                for admission in latest_admitted.values()
            ),
        },
        "ensemble": {
            "size": {"control": 1, "perturbed": 6, "total": 7},
            "members": member_rows,
            "masks_balanced": all(
                row["combined_balance_sum"] == 0 for row in plan["perturbation"]["masks"].values()
            ),
            "masks_pairwise_orthogonal": all(
                value == 0 for value in plan["perturbation"]["pairwise_dot_products"].values()
            ),
            "all_exact_one_ulp": True,
            "all_boundaries_metadata_nonmomentum_preserved": True,
        },
        "resource_safety": {
            "principal_override_received_before_any_science_result": True,
            "principal_override": _artifact_row(override_path, True),
            "taskset_only_attempts_excluded_and_preserved": _artifact_row(
                boundary_correction_path, True
            ),
            "kernel_cpu_boundary_probe": _artifact_row(boundary_probe_path, True),
            "kernel_boundary_commit": boundary_commit,
            "kernel_boundary_predates_all_admitted_member_launches": all(
                datetime.fromisoformat(admission["captured_at_utc"]).timestamp()
                > int(_git("show", "-s", "--format=%ct", boundary_commit))
                for admission in latest_admitted.values()
            ),
            "every_member_admitted_immediately_before_launch": True,
            "all_admissions_two_sample_identity_stable": all(
                admission["preflight_stability"]["stable"] for admission in latest_admitted.values()
            ),
            "all_admissions_production_confined_to_cores_0_11": all(
                admission["cpu_ownership"]["all_production_model_processes_confined_to_cores_0_11"]
                for admission in latest_admitted.values()
            ),
            "all_continuous_member_monitors_safe": all(
                load_self_hashed(_member_dir(member) / "resource-monitor.json")["all_samples_safe"]
                for member in member_rows
            ),
            "nightly_active_alone_not_a_blocker": True,
            "isolated_logical_cpuset": CPUSET,
            "isolated_physical_cores": list(EXPECTED_PHYSICAL_CORES),
            "physical_core_count": len(EXPECTED_PHYSICAL_CORES),
            "mpi_ranks_oversubscribed": MPI_RANKS,
            "sched_setaffinity_denied_for_all_member_services": True,
            "production_allowed_physical_cores": list(PRODUCTION_PHYSICAL_CORES),
            "production_physical_core_overlap": [],
        },
        "control": analysis["control_reproducibility"],
        "comparison": _artifact_row(SPRINT / "ensemble-analysis.json", True),
        "metrics_and_thresholds_register": {
            "markdown": _artifact_row(register_markdown),
            "json": _artifact_row(register_json_path, True),
        },
        "decisive_growth": analysis["decisive_growth"],
        "decision_statistics": analysis["statistics"],
        "decision_gates": analysis["gates"],
        "thresholds_applied_unchanged": analysis["thresholds_applied_unchanged"],
        "model_tree": {
            "pre_kimi_diagnostic": "835dcc29bf316c0715b41a72e064985e9cf099df",
            "sprint_base": base_model_tree,
            "sprint_final": current_model_tree,
            "sprint_model_delta_files": [],
            "production_numerics_unchanged": base_model_tree == current_model_tree and not model_diff,
        },
        "gpu_attestation": {
            "queries": 0, "locks": 0, "compiles": 0, "dispatches": 0, "kernels": 0,
            "cuda_visible_devices": "",
            "gpu_forbidden_and_unaccessed": True,
            "authority_audit_attestation": authority_audit["gpu_attestation"],
        },
        "release_gate": {
            "release": False, "merge": False, "tag": False, "push": False,
            "tolerance_or_gate_changed_after_results": False,
            "model_fix_authorized": False,
            "full_18h_or_late_Ni_solved_claim": False,
        },
        "changed_files_before_seal": changed_files,
        "handoff": {
            "worker_report": _artifact_row(worker_report),
            "command_log": _artifact_row(command_log),
        },
        "exact_next_kimi_assignment": analysis["exact_next_kimi_assignment"],
    }
    output = SPRINT / "proof.json"
    if output.exists():
        raise FileExistsError(output)
    write_self_hashed(output, proof)
    retained = {
        "schema": "gpuwrf.v0234.wrf-conditioning-ensemble-retained-evidence.v1",
        "verdict": "WRF_CONDITIONING_ENSEMBLE_EVIDENCE_RETAINED",
        "terminal_proof": _artifact_row(output, True),
        "ensemble_analysis": _artifact_row(SPRINT / "ensemble-analysis.json", True),
        "principal_resource_override": _artifact_row(override_path, True),
        "resource_boundary_correction": _artifact_row(boundary_correction_path, True),
        "kernel_cpu_boundary_probe": _artifact_row(boundary_probe_path, True),
        "metrics_and_thresholds_register": {
            "markdown": _artifact_row(register_markdown),
            "json": _artifact_row(register_json_path, True),
        },
        "preregistration": _artifact_row(PLAN_PATH),
        "preregistration_canonical_sha256": plan["plan_sha256"],
        "members": {
            member: {
                "raw_archive": row["raw_archive"],
                "raw_inventory": row["raw_inventory"],
                "analysis": row["analysis"],
                "resource_monitor": row["resource_monitor"],
                "execution": row["execution"],
                "perturbation": row["perturbation"],
            }
            for member, row in member_rows.items()
        },
        "changed_files_before_seal": changed_files,
        "handoff": {
            "worker_report": _artifact_row(worker_report),
            "command_log": _artifact_row(command_log),
        },
    }
    retained_output = SPRINT / "retained-evidence-manifest.json"
    write_self_hashed(retained_output, retained)
    print(json.dumps({
        "verdict": proof["verdict"],
        "proof": str(output),
        "proof_file_sha256": sha256_file(output),
        "proof_canonical_sha256": load_self_hashed(output)["proof_sha256"],
        "retained": str(retained_output),
        "retained_file_sha256": sha256_file(retained_output),
        "retained_canonical_sha256": load_self_hashed(retained_output)["proof_sha256"],
    }, indent=1, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write-plan")
    subparsers.add_parser("verify-plan")
    subparsers.add_parser("audit-authority")
    subparsers.add_parser("prepare-all")
    for name in ("prepare-member", "resource-check", "run-member", "analyze-member", "archive-member"):
        child = subparsers.add_parser(name)
        child.add_argument("--member", required=True, choices=[spec["id"] for spec in MEMBER_SPECS])
    subparsers.add_parser("aggregate")
    subparsers.add_parser("terminal-proof")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-plan":
        write_plan()
    elif args.command == "verify-plan":
        plan = load_plan()
        fresh = build_plan()
        if plan != fresh:
            raise RuntimeError("committed plan does not reproduce from frozen authority")
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "verified": True}, sort_keys=True))
    elif args.command == "audit-authority":
        audit_authority()
    elif args.command == "prepare-all":
        prepare_all()
    elif args.command == "prepare-member":
        prepare_member(args.member)
    elif args.command == "resource-check":
        snapshot = resource_snapshot(args.member)
        print(json.dumps(snapshot, indent=1, sort_keys=True))
        return 0 if snapshot["verdict"] == "ADMITTED_ISOLATED_CPU_SET" else 75
    elif args.command == "run-member":
        return run_member(args.member)
    elif args.command == "analyze-member":
        analyze_member(args.member)
    elif args.command == "archive-member":
        archive_member(args.member)
    elif args.command == "aggregate":
        aggregate()
    elif args.command == "terminal-proof":
        terminal_proof()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
