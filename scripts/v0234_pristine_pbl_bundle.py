#!/usr/bin/env python3
"""Construct and independently audit the sealed V0234 pristine authority bundle.

This tool is deliberately CPU-only.  It imports no JAX/gpuwrf code and never
launches WRF or MPI.  Source bytes come only from pinned Git objects; runtime
files are regular, content-addressed copies with explicit origin receipts.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-pristine-pbl-entry-closure-gpt"
CONTRACT = SPRINT / "CONTRACT.md"
CONTRACT_COMMIT = "5bbfe0416979eb41dd72bf80a8539191740f8eea"
CONTRACT_SHA256 = "db445ed5ca18250c932651c4bf506799778895baf9390d692f65c9dfa21b6f35"
COORDINATION_COMMIT = "8231225573683dc56828caba2be3326508bdfd73"
OUTPUT_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2")
NONCE = "0b18530a1dc9cac2587e3f8319db70c0a450b6ffa642990504ae46b69c1a14cd"

WRF_OBJECT_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
WRF_TREE = "6b658fbc98077fe0648cba724921679126464181"
SUBMODULES = {
    ".ci/hpc-workflows": "dfc8e6d823b80497ea41bab94e1fdf3f4594ad18",
    "phys/MYNN-EDMF": "90f36c25259ec1960b24325f5b29ac7c5adeac73",
    "phys/noahmp": "e5c0859874407859936739e8be8741f9aed369ee0",
}

INPUT_ORIGINS = {
    "namelist.input": (
        Path("<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/namelist.input"),
        "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    ),
    "wrfinput_d03": (
        Path("<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/wrfinput_d03"),
        "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
    ),
    "last-healthy-d03-step-0.pkl": (
        Path("<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/corrected_ni_rca_max_22c2bd7a/v0234_1500_science_60659a2e_terminal2/science-runtime/failure/last-healthy-d03-step-0.pkl"),
        "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d",
    ),
}

CAPTURE_INPUT_ORIGINS = {
    "wrfinput_d01": (Path("<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/wrfinput_d01"), "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756"),
    "wrfinput_d02": (Path("<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/wrfinput_d02"), "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964"),
    "wrfbdy_d01": (Path("<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/wrfbdy_d01"), "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec"),
}

WRF_RUNTIME_FILES = (
    "GENPARM.TBL", "SOILPARM.TBL", "VEGPARM.TBL",
    "RRTM_DATA", "RRTM_DATA_DBL", "RRTMG_LW_DATA", "RRTMG_LW_DATA_DBL",
    "RRTMG_SW_DATA", "RRTMG_SW_DATA_DBL", "CAM_ABS_DATA", "CAM_AEROPT_DATA",
    "ETAMPNEW_DATA", "ETAMPNEW_DATA.expanded_rain",
    "grib2map.tbl", "LANDUSE.TBL", "URBPARM.TBL", "URBPARM_UZE.TBL",
    "aerosol.formatted", "aerosol_lat.formatted", "aerosol_lon.formatted",
    "aerosol_plev.formatted", "ozone.formatted", "ozone_lat.formatted",
    "ozone_plev.formatted",
)

PROJECT_PAYLOADS = {
    "thompson_tables": "data/fixtures/thompson-tables-v1.npz",
    "thompson_cold_collection": "data/fixtures/thompson-cold-collection-v1.npz",
    "rrtmg_tables": "data/fixtures/rrtmg-tables-v1.npz",
}

TOOLCHAIN_ROOT = Path("<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build")
TOOLCHAIN_PROGRAMS = ("cmake", "gcc", "gfortran", "mpicc", "mpif90", "mpirun", "nc-config", "nf-config")

WRF_DYNAMIC_SOURCES = (
    "phys/module_ra_rrtmg_lw.F",
    "phys/module_ra_rrtmg_sw.F",
    "phys/module_mp_thompson.F",
)

ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


class Refusal(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def run(*argv: str, cwd: Path | None = None, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        list(argv), cwd=cwd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=not binary,
    )
    if result.returncode:
        stderr = result.stderr.decode(errors="replace") if binary else result.stderr
        raise Refusal(f"command failed ({result.returncode}): {' '.join(argv)}: {stderr.strip()}")
    return result.stdout


def git(root: Path, *args: str) -> str:
    return str(run("git", "-C", str(root), *args)).strip()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    value = dict(value)
    value["canonical_payload_sha256"] = canonical(value)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_canonical_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Refusal(f"canonical JSON is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Refusal(f"invalid canonical JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Refusal(f"canonical JSON is not an object: {path}")
    claimed = value.pop("canonical_payload_sha256", None)
    if not isinstance(claimed, str) or canonical(value) != claimed:
        raise Refusal(f"canonical JSON drift: {path}")
    return value


def assert_regular(path: Path, expected: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Refusal(f"required regular file absent: {path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise Refusal(f"identity drift: {path}: {actual}")
    info = path.stat()
    return {"path": str(path), "sha256": actual, "size": info.st_size, "mode": stat.S_IMODE(info.st_mode)}


def resource_gate(root: Path) -> dict[str, Any]:
    affinity = set(os.sched_getaffinity(0))
    if affinity != ALLOWED_CPUS:
        raise Refusal(f"CPUSET_ESCAPE:{sorted(affinity)}")
    observed = {key: os.environ.get(key) for key in THREAD_ENV}
    if observed != THREAD_ENV:
        raise Refusal(f"THREAD_ENV_DRIFT:{observed}")
    niceness = os.getpriority(os.PRIO_PROCESS, 0)
    if niceness < 15:
        raise Refusal(f"NICENESS:{niceness}")
    ioprio = ctypes.CDLL(None, use_errno=True).syscall(252, 1, 0)
    if ioprio < 0 or int(ioprio) >> 13 != 3:
        raise Refusal(f"IOPRIO:{ioprio}")
    if Path("/tmp/PREEMPT_CPU").exists() or Path("/tmp/PREEMPT_CPU").is_symlink():
        raise Refusal("PREEMPT_CPU")
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0]) * 1024
    if mem.get("MemAvailable", 0) < 16 * 1024**3:
        raise Refusal(f"MEMORY_PRESSURE:{mem.get('MemAvailable')}")
    statvfs = os.statvfs(root.parent)
    available = statvfs.f_bavail * statvfs.f_frsize
    if available < 120 * 1024**3:
        raise Refusal(f"STORAGE_GUARD:{available}")
    return {
        "logical_cpu_ids": sorted(affinity), "thread_environment": observed,
        "nice": niceness, "ionice_class": int(ioprio) >> 13,
        "memory_available_bytes": mem["MemAvailable"], "storage_available_bytes": available,
        "preempt_cpu_absent": True,
    }


def safe_extract(data: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = archive.getmembers()
        link_paths: list[Path] = []
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or member.isdev() or member.isfifo():
                raise Refusal(f"unsafe Git archive member: {member.name}")
            if member.islnk():
                raise Refusal(f"hardlink forbidden in Git archive: {member.name}")
            if member.issym():
                link_paths.append(name)
        for member in members:
            name = Path(member.name)
            if any(link != name and link in name.parents for link in link_paths):
                raise Refusal(f"archive member nested below symlink: {member.name}")
        # Git stores symlinks as blobs and WRF v4.7.1 intentionally contains an
        # absolute var/run/crtm_coeffs link.  It is preserved only in source;
        # the runtime admission below independently forbids every symlink.
        archive.extractall(destination, filter="fully_trusted")


def archive_git_tree(provider: Path, commit: str, destination: Path) -> None:
    if git(provider, "cat-file", "-t", commit) != "commit":
        raise Refusal(f"missing commit object: {provider}:{commit}")
    data = run("git", "-C", str(provider), "archive", "--format=tar", commit, binary=True)
    assert isinstance(data, bytes)
    safe_extract(data, destination)


def git_tree_records(provider: Path, commit: str, prefix: str = "") -> list[dict[str, Any]]:
    output = run("git", "-C", str(provider), "ls-tree", "-rz", "-r", commit, binary=True)
    assert isinstance(output, bytes)
    records: list[dict[str, Any]] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        header, name = raw.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        path = (Path(prefix) / name.decode()).as_posix()
        records.append({"path": path, "git_mode": mode, "git_type": kind, "git_object": blob})
    return records


def verify_materialized_tree(root: Path, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in records:
        path = root / item["path"]
        mode = item["git_mode"]
        if mode == "160000":
            continue
        if not path.exists() and not path.is_symlink():
            raise Refusal(f"materialized tracked path absent: {item['path']}")
        if mode == "120000":
            if not path.is_symlink():
                raise Refusal(f"tracked symlink drift: {item['path']}")
            payload = os.readlink(path).encode()
        else:
            if path.is_symlink() or not path.is_file():
                raise Refusal(f"tracked file-type drift: {item['path']}")
            payload = path.read_bytes()
        framed = b"blob " + str(len(payload)).encode() + b"\0" + payload
        actual_git_blob = hashlib.sha1(framed).hexdigest()
        if actual_git_blob != item["git_object"]:
            raise Refusal(f"materialized blob drift: {item['path']}:{actual_git_blob}")
        result.append({**item, "sha256": sha256_bytes(payload), "git_object_rechecked": actual_git_blob, "git_blob_sha256_frame": sha256_bytes(framed), "size": len(payload)})
    return result


def copy_regular(source: Path, destination: Path, expected: str | None = None) -> dict[str, Any]:
    origin = assert_regular(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Refusal(f"duplicate destination: {destination}")
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    copied = assert_regular(destination, origin["sha256"])
    return {"origin": origin, "copy": copied}


def git_origin(provider: Path, commit: str, relative: str) -> dict[str, Any]:
    line = git(provider, "ls-tree", commit, "--", relative)
    fields = line.split(None, 3)
    if len(fields) != 4 or fields[1] != "blob":
        raise Refusal(f"not a tracked blob: {provider}:{commit}:{relative}")
    return {"provider": str(provider), "commit": commit, "path": relative, "git_mode": fields[0], "git_blob": fields[2]}


def npz_schema(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: {"shape": list(bundle[name].shape), "dtype": bundle[name].dtype.str, "nbytes": int(bundle[name].nbytes)} for name in sorted(bundle.files)}


def wrfinput_schema(path: Path) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise Refusal("netCDF4 required for wrfinput schema closure") from exc
    with Dataset(path, "r") as dataset:
        dimensions = {name: len(value) for name, value in dataset.dimensions.items()}
        required_dimensions = {"west_east": 111, "south_north": 93, "bottom_top": 44, "bottom_top_stag": 45}
        if any(dimensions.get(name) != value for name, value in required_dimensions.items()):
            raise Refusal(f"wrfinput dimension drift: {dimensions}")
        required_attributes = {"GRID_ID": 3, "DX": 1000.0, "DY": 1000.0}
        observed_attributes = {name: float(getattr(dataset, name)) for name in required_attributes}
        if observed_attributes != {name: float(value) for name, value in required_attributes.items()}:
            raise Refusal(f"wrfinput attribute drift: {observed_attributes}")
        return {
            "dimensions": dimensions,
            "required_attributes": observed_attributes,
            "variables": {name: {"dimensions": list(value.dimensions), "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in dataset.variables.items()},
        }


def netcdf_schema(path: Path) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise Refusal("netCDF4 required for input schema closure") from exc
    with Dataset(path, "r") as dataset:
        return {
            "dimensions": {name: len(value) for name, value in dataset.dimensions.items()},
            "variables": {name: {"dimensions": list(value.dimensions), "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in dataset.variables.items()},
            "global_attributes": {name: str(getattr(dataset, name)) for name in sorted(dataset.ncattrs())},
        }


def namelist_route(path: Path) -> dict[str, list[int]]:
    import re
    text = path.read_text(encoding="utf-8")
    expected = {
        "mp_physics": 8, "ra_lw_physics": 4, "ra_sw_physics": 4,
        "sf_sfclay_physics": 5, "sf_surface_physics": 4, "bl_pbl_physics": 5,
    }
    observed: dict[str, list[int]] = {}
    for name, required in expected.items():
        matches = re.findall(rf"(?mi)^\s*{re.escape(name)}\s*=\s*([^!\n/]+)", text)
        if len(matches) != 1:
            raise Refusal(f"namelist route key not unique: {name}:{len(matches)}")
        try:
            values = [int(token.strip()) for token in matches[0].split(",") if token.strip()]
        except ValueError as exc:
            raise Refusal(f"namelist route parse failure: {name}") from exc
        if values != [required, required, required]:
            raise Refusal(f"namelist route drift: {name}:{values}")
        observed[name] = values
    return observed


def toolchain_manifest() -> dict[str, Any]:
    programs: dict[str, Any] = {}
    for name in TOOLCHAIN_PROGRAMS:
        logical = TOOLCHAIN_ROOT / "bin" / name
        if not logical.exists():
            raise Refusal(f"toolchain program absent: {logical}")
        resolved = logical.resolve()
        receipt = assert_regular(resolved)
        receipt["logical_path"] = str(logical)
        receipt["logical_is_symlink"] = logical.is_symlink()
        programs[name] = receipt
    return {
        "schema": "wrfgpu2-v0234-toolchain-v1", "environment_root": str(TOOLCHAIN_ROOT),
        "programs": programs, "cmake_generator": "Unix Makefiles", "build_parallelism": 6,
        "use_mpi": True, "use_openmp": False, "build_externals": False,
    }


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def source_provider_for(path: str) -> tuple[Path, str, str]:
    for subpath, commit in sorted(SUBMODULES.items(), key=lambda value: -len(value[0])):
        if path == subpath or path.startswith(subpath + "/"):
            relative = path[len(subpath):].lstrip("/")
            return WRF_OBJECT_ROOT / subpath, commit, relative
    return WRF_OBJECT_ROOT, WRF_COMMIT, path


def relocate_staging_paths(value: Any, staging: Path, root: Path) -> Any:
    """Make manifests name the final namespace before the atomic rename."""
    if isinstance(value, dict):
        return {key: relocate_staging_paths(item, staging, root) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_staging_paths(item, staging, root) for item in value]
    if isinstance(value, str) and (value == str(staging) or value.startswith(str(staging) + os.sep)):
        return str(root) + value[len(str(staging)):]
    return value


def construct(root: Path) -> dict[str, Any]:
    if root != OUTPUT_ROOT:
        raise Refusal(f"output authority mismatch: {root}")
    if root.exists() or root.is_symlink():
        raise Refusal(f"output is not fresh: {root}")
    gate = resource_gate(root)
    if git(REPO, "merge-base", "--is-ancestor", CONTRACT_COMMIT, "HEAD") != "":
        pass
    if git(REPO, "rev-parse", CONTRACT_COMMIT) != CONTRACT_COMMIT:
        raise Refusal("contract commit absent")
    if git(REPO, "status", "--porcelain"):
        raise Refusal("construction requires clean approved worktree")
    contract = assert_regular(CONTRACT, CONTRACT_SHA256)
    if git(REPO, "rev-parse", f"{COORDINATION_COMMIT}^{{commit}}") != COORDINATION_COMMIT:
        raise Refusal("coordination commit absent")
    if git(WRF_OBJECT_ROOT, "rev-parse", f"{WRF_COMMIT}^{{tree}}") != WRF_TREE:
        raise Refusal("WRF commit/tree drift")
    if git(WRF_OBJECT_ROOT, "rev-parse", "v4.7.1^{}") != WRF_COMMIT:
        raise Refusal("WRF tag target drift")

    staging = root.parent / f".{root.name}.construct-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise Refusal(f"staging collision: {staging}")
    staging.mkdir(mode=0o700)
    try:
        for name in ("control", "source", "runtime", "inputs", "build", "capture", "cpu-reference", "comparator", "candidate-proof", "terminal-proof"):
            (staging / name).mkdir()
        pristine = staging / "source/pristine"
        archive_git_tree(WRF_OBJECT_ROOT, WRF_COMMIT, pristine)
        tree_records = git_tree_records(WRF_OBJECT_ROOT, WRF_COMMIT)
        submodule_receipts: dict[str, Any] = {}
        for subpath, commit in SUBMODULES.items():
            provider = WRF_OBJECT_ROOT / subpath
            target = pristine / subpath
            target.mkdir(parents=True, exist_ok=True)
            archive_git_tree(provider, commit, target)
            records = git_tree_records(provider, commit, subpath)
            tree_records.extend(records)
            submodule_receipts[subpath] = {"commit": commit, "tree": git(provider, "rev-parse", f"{commit}^{{tree}}"), "provider": str(provider)}
        verified = verify_materialized_tree(pristine, tree_records)

        runtime_run = staging / "runtime/wrf/run"
        runtime_phys = staging / "runtime/wrf/phys"
        runtime_receipts: dict[str, Any] = {}
        for name in WRF_RUNTIME_FILES:
            source = pristine / "run" / name
            if not source.is_file():
                raise Refusal(f"declared WRF runtime payload absent: run/{name}")
            receipt = copy_regular(source, runtime_run / name)
            receipt["logical_role"] = "WRF runtime table; route-audited, may be NOT_REACHED before PBL"
            receipt["origin"] |= git_origin(WRF_OBJECT_ROOT, WRF_COMMIT, f"run/{name}")
            runtime_receipts[f"run/{name}"] = receipt
        mp_relative = "parameters/MPTABLE.TBL"
        mp_provider = WRF_OBJECT_ROOT / "phys/noahmp"
        receipt = copy_regular(pristine / "phys/noahmp" / mp_relative, runtime_run / "MPTABLE.TBL")
        receipt["logical_role"] = "Noah-MP parameter table reached by sf_surface_physics=4"
        receipt["origin"] |= git_origin(mp_provider, SUBMODULES["phys/noahmp"], mp_relative)
        runtime_receipts["run/MPTABLE.TBL"] = receipt
        for relative in WRF_DYNAMIC_SOURCES:
            receipt = copy_regular(pristine / relative, runtime_phys / Path(relative).name)
            receipt["logical_role"] = "dynamic source parser authority"
            provider, commit, local = source_provider_for(relative)
            receipt["origin"] |= git_origin(provider, commit, local)
            runtime_receipts[relative] = receipt

        project_receipts: dict[str, Any] = {}
        for role, relative in PROJECT_PAYLOADS.items():
            source = REPO / relative
            target = staging / "runtime/project-fixtures" / Path(relative).name
            receipt = copy_regular(source, target)
            receipt["logical_role"] = role
            receipt["source_commit"] = git(REPO, "rev-parse", "HEAD")
            receipt["source_relative"] = relative
            receipt["schema"] = npz_schema(target)
            project_receipts[role] = receipt

        input_receipts: dict[str, Any] = {}
        for name, (source, digest) in INPUT_ORIGINS.items():
            target = staging / "inputs" / name
            receipt = copy_regular(source, target, digest)
            receipt["logical_role"] = name
            if name == "wrfinput_d03":
                receipt["schema"] = wrfinput_schema(target)
            input_receipts[name] = receipt
        sealed_route = namelist_route(staging / "inputs/namelist.input")

        model_files = {}
        for relative in (
            "src/gpuwrf/runtime/operational_mode.py", "src/gpuwrf/contracts/state.py",
            "src/gpuwrf/coupling/physics_couplers.py", "src/gpuwrf/physics/mynn_pbl.py",
            "src/gpuwrf/physics/thompson_tables.py", "src/gpuwrf/physics/rrtmg_tables.py",
            "src/gpuwrf/physics/rrtmg_lw.py", "src/gpuwrf/physics/rrtmg_sw.py",
            "src/gpuwrf/config/paths.py", "src/gpuwrf/integration/nested_pipeline.py",
        ):
            source = REPO / relative
            target = staging / "runtime/model-source" / relative
            model_files[relative] = copy_regular(source, target)

        manifests = {
            "source-manifest.json": {
                "schema": "wrfgpu2-v0234-pristine-source-v1", "wrf_commit": WRF_COMMIT,
                "wrf_tree": WRF_TREE, "tag": "v4.7.1", "tag_cryptographically_signed": False,
                "object_provider_dirty_but_not_copied": bool(git(WRF_OBJECT_ROOT, "status", "--porcelain")),
                "submodules": submodule_receipts, "tracked_records": verified,
            },
            "runtime-manifest.json": {
                "schema": "wrfgpu2-v0234-runtime-closure-v1", "route": {
                    "mp_physics": 8, "ra_lw_physics": 4, "ra_sw_physics": 4,
                    "sf_sfclay_physics": 5, "sf_surface_physics": 4, "bl_pbl_physics": 5,
                }, "runtime_files": runtime_receipts, "project_payloads": project_receipts,
                "model_source": model_files,
                "reachability": {
                    "MPTABLE.TBL": "REACHED by Noah-MP domain load",
                    "SOILPARM.TBL": "REACHED by Noah-MP domain load",
                    "GENPARM.TBL": "REACHED by Noah-MP domain load",
                    "Thompson fixtures": "REACHED by mp_physics=8 import/prefix",
                    "RRTMG fixtures and module_ra_rrtmg_[lw,sw].F": "REACHED by route initialization/radiation schedule",
                    "classic RRTM_DATA[_DBL]": "NOT_REACHED: ra_lw_physics=4, retained only as sealed negative-control runtime closure",
                    "CAM aerosol payloads": "NOT_REACHED: RRTMG route uses no CAM radiation scheme",
                    "CAMtr_volume_mixing_ratio": "NOT_REACHED and not admitted: upstream generic path is a dangling absolute tracked symlink; route-selected CAM radiation is disabled",
                    "AOT_deserialize_executable": "NOT_REACHED: direct _physics_step_forcing capture has GPUWRF_NESTED_AOT=0 and invokes no nested scan executable loader",
                },
                "symlinks_forbidden": True,
            },
            "input-manifest.json": {
                "schema": "wrfgpu2-v0234-input-authority-v1", "inputs": input_receipts,
                "namelist_route": sealed_route,
                "scientific_tuple": {"cycle": "20250228_18z", "domain": "d03", "outer_step": 1,
                    "dt_seconds": 6.0, "first_timestep": True, "mass_grid": [93, 111, 44],
                    "staggered_vertical": 45, "dx_m": 1000.0, "dy_m": 1000.0, "use_theta_m": 1},
            },
            "environment-manifest.json": {
                "schema": "wrfgpu2-v0234-environment-v1", "resource_gate": gate,
                "python": assert_regular(Path(sys.executable).resolve()), "numpy_version": np.__version__,
                "numpy_source": assert_regular(Path(np.__file__).resolve()), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_actions": 0, "wrf_or_mpi_executions": 0,
            },
            "toolchain-manifest.json": toolchain_manifest(),
            "authority-manifest.json": {
                "schema": "wrfgpu2-v0234-authority-v1", "contract_commit": CONTRACT_COMMIT,
                "contract": contract, "coordination_commit": COORDINATION_COMMIT, "nonce": NONCE,
                "nonce_state": "RESERVED_NOT_CONSUMED", "output_root": str(root),
                "model_commit": git(REPO, "rev-parse", "HEAD"), "created_utc": now(),
            },
        }
        control = staging / "control"
        for name, value in manifests.items():
            atomic_json(control / name, relocate_staging_paths(value, staging, root))
        aggregate = {path.name: sha256_file(path) for path in sorted(control.glob("*.json"))}
        atomic_json(control / "receipt-graph.json", {"schema": "wrfgpu2-v0234-receipt-graph-v1", "manifests": aggregate, "complete": True})

        # Runtime and admitted inputs must be immutable and symlink-free.  Source
        # preserves upstream tracked symlinks but contains no checkout detritus.
        for path in (staging / "runtime").rglob("*"):
            if path.is_symlink():
                raise Refusal(f"runtime symlink leaked: {path}")
        make_read_only(staging / "source/pristine")
        make_read_only(staging / "runtime")
        make_read_only(staging / "inputs")
        os.rename(staging, root)
        return {"verdict": "PRISTINE_BUNDLE_CONSTRUCTED", "root": str(root), "receipt_graph_sha256": sha256_file(root / "control/receipt-graph.json")}
    except BaseException:
        if staging.exists() and staging.parent == root.parent and staging.name.startswith(f".{root.name}.construct-"):
            shutil.rmtree(staging)
        raise


def audit(root: Path) -> dict[str, Any]:
    if root != OUTPUT_ROOT or root.is_symlink() or not root.is_dir():
        raise Refusal("invalid output authority root")
    gate = resource_gate(root)
    graph_path = root / "control/receipt-graph.json"
    graph = load_canonical_json(graph_path)
    if not graph.get("complete"):
        raise Refusal("receipt graph canonical drift")
    for name, digest in graph["manifests"].items():
        path = root / "control" / name
        if sha256_file(path) != digest:
            raise Refusal(f"manifest drift: {name}")
        load_canonical_json(path)
    runtime_manifest = json.loads((root / "control/runtime-manifest.json").read_text())
    for group in ("runtime_files", "project_payloads"):
        for item in runtime_manifest[group].values():
            assert_regular(Path(item["copy"]["path"]), item["copy"]["sha256"])
    for path in (root / "runtime").rglob("*"):
        if path.is_symlink():
            raise Refusal(f"runtime symlink drift: {path}")
    input_manifest = json.loads((root / "control/input-manifest.json").read_text())
    for item in input_manifest["inputs"].values():
        assert_regular(Path(item["copy"]["path"]), item["copy"]["sha256"])
    capture_manifest_path = root / "control/capture-input-manifest.json"
    if capture_manifest_path.exists():
        capture_manifest = load_canonical_json(capture_manifest_path)
        for item in capture_manifest["inputs"].values():
            assert_regular(Path(item["copy"]["path"]), item["copy"]["sha256"])
    return {"verdict": "PRISTINE_BUNDLE_AUDIT_PASS", "root": str(root), "resource_gate": gate, "receipt_graph_sha256": sha256_file(graph_path)}


def augment_capture_inputs(root: Path) -> dict[str, Any]:
    audit(root)
    destination = root / "capture-inputs"
    manifest_path = root / "control/capture-input-manifest.json"
    if destination.exists() or destination.is_symlink() or manifest_path.exists():
        raise Refusal("capture input augmentation is not fresh")
    destination.mkdir()
    receipts: dict[str, Any] = {}
    sources = dict(CAPTURE_INPUT_ORIGINS)
    sources["namelist.input"] = (root / "inputs/namelist.input", INPUT_ORIGINS["namelist.input"][1])
    sources["wrfinput_d03"] = (root / "inputs/wrfinput_d03", INPUT_ORIGINS["wrfinput_d03"][1])
    try:
        for name, (source, digest) in sources.items():
            receipt = copy_regular(source, destination / name, digest)
            receipt["logical_role"] = "authentic nested d01-d03 domain loading for d03 pre-PBL capture"
            if name != "namelist.input":
                receipt["schema"] = netcdf_schema(destination / name)
            receipts[name] = receipt
        manifest = {
            "schema": "wrfgpu2-v0234-capture-input-closure-v1", "inputs": receipts,
            "loader_chain": "NestedPipelineConfig -> _load_domains -> build_replay_case/build_noahmp_land_state",
            "scope": "domain loading only; no WRF/MPI runtime and no wrfout replay input",
            "required_files": ["namelist.input", "wrfinput_d01", "wrfinput_d02", "wrfinput_d03", "wrfbdy_d01"],
            "complete": True,
        }
        atomic_json(manifest_path, manifest)
        graph_path = root / "control/receipt-graph.json"
        graph = load_canonical_json(graph_path)
        graph["manifests"][manifest_path.name] = sha256_file(manifest_path)
        atomic_json(graph_path, graph)
        make_read_only(destination)
        return {"verdict": "CAPTURE_INPUT_CLOSURE_COMPLETE", "manifest_sha256": sha256_file(manifest_path), "receipt_graph_sha256": sha256_file(graph_path)}
    except BaseException:
        # Do not hide an admitted augmentation failure; leave its partial tree as evidence.
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("construct", "audit", "augment-capture-inputs"))
    value.add_argument("--output-root", default=str(OUTPUT_ROOT))
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "construct":
            result = construct(Path(args.output_root))
        elif args.command == "augment-capture-inputs":
            result = augment_capture_inputs(Path(args.output_root))
        else:
            result = audit(Path(args.output_root))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (Refusal, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
