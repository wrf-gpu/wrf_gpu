#!/usr/bin/env python3
"""Build the sealed, instrumented WRF-v4.7.1 CPU oracle executable.

The command is one-shot with respect to the external source/build paths.  It
never starts WRF or MPI; MPI support is compiled into the executable only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parent.parent
BUNDLE_SCRIPT = REPO / "scripts/v0234_pristine_pbl_bundle.py"
SPEC = importlib.util.spec_from_file_location("v0234_pristine_pbl_bundle", BUNDLE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load bundle authority")
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)

ROOT = bundle.OUTPUT_ROOT
PRISTINE = ROOT / "source/pristine"
EXTERNAL = ROOT / "source/external-mmm-physics-pristine"
INSTRUMENTED = ROOT / "source/instrumented"
BUILD = ROOT / "build/wrf"
INSTALL = BUILD / "install"
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-pristine-pbl-entry-closure-gpt"
PATCH = SPRINT / "wrf-mynn-sp2-instrumentation.patch"
PATCH_SHA256 = "a0c2c988db002e822ea5c97c2107da3a5cadfa6b349eefa2cd9dc2131299ce87"
ROUTE_PATCH = SPRINT / "wrf-cmake-disable-unselected-clm.patch"
ROUTE_PATCH_SHA256 = "b77c2d9e829d56bede58591c074d90bbfe06c732d7b8bba6627a8775e81fe082"
REGISTRY_PATCH = SPRINT / "wrf-cmake-registry-build-root.patch"
REGISTRY_PATCH_SHA256 = "4203212b538cdd8eeeb6e4e72ab959ea6e2dd159d8ab8572b692b1c3e4efcfcb"
TOOLCHAIN = SPRINT / "wrf_config.cmake"
TOOLCHAIN_ROOT = bundle.TOOLCHAIN_ROOT
STOP_HOUR = 23
STOP_MINUTE = 15
MMM_PROVIDER = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/physics_mmm")
MMM_COMMIT = "0ea59b1cd673006ee7a9a9958c533a6a0e354243"
MMM_TREE = "165b532cb1e619ae308ca1e7c9d8a79edd8dad06"
MMM_TAG = "20240626-MPASv8.2"


def check_preempt_and_pressure() -> None:
    bundle.resource_gate(ROOT)
    local = datetime.now(ZoneInfo("Atlantic/Canary"))
    if (local.hour, local.minute) >= (STOP_HOUR, STOP_MINUTE):
        raise bundle.Refusal(f"CLEAN_STOP_DEADLINE:{local.isoformat()}")


def parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.strip().split(","):
        if "-" in part:
            lower, upper = (int(item) for item in part.split("-", 1))
            result.update(range(lower, upper + 1))
        elif part:
            result.add(int(part))
    return result


def assert_no_production_collision() -> None:
    runtime_names = {"wrf.exe", "real.exe", "mpirun", "prterun"}
    collisions: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            if comm not in runtime_names:
                continue
            status = (entry / "status").read_text().splitlines()
            allowed_text = next(line.split(":", 1)[1].strip() for line in status if line.startswith("Cpus_allowed_list:"))
            overlap = parse_cpu_list(allowed_text) & bundle.ALLOWED_CPUS
            if overlap:
                collisions.append(f"{entry.name}:{comm}:{sorted(overlap)}")
        except (FileNotFoundError, ProcessLookupError, StopIteration, PermissionError):
            continue
    if collisions:
        raise bundle.Refusal(f"PRODUCTION_COLLISION:{collisions}")


def chmod_owner_writable(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode | stat.S_IWUSR)
    root.chmod(stat.S_IMODE(root.stat().st_mode) | stat.S_IWUSR)


def seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def tree_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if path.is_symlink():
            payload = os.readlink(path).encode()
            kind = "symlink"
        elif path.is_file():
            payload = path.read_bytes()
            kind = "file"
        elif path.is_dir():
            continue
        else:
            raise bundle.Refusal(f"unsupported source entry: {path}")
        records.append({"path": relative, "kind": kind, "mode": stat.S_IMODE(info.st_mode), "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return bundle.canonical(records), records


def run_logged(argv: list[str], log: Path, env: dict[str, str]) -> dict[str, Any]:
    check_preempt_and_pressure()
    assert_no_production_collision()
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(
            argv, stdout=stream, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                try:
                    check_preempt_and_pressure()
                    assert_no_production_collision()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    raise
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
    if process.returncode != 0:
        raise bundle.Refusal(f"build command failed ({process.returncode}); log={log}")
    return {"argv": argv, "log": bundle.assert_regular(log), "elapsed_seconds": time.monotonic() - started, "exit_code": process.returncode}


def add_manifest_to_graph(path: Path) -> None:
    graph_path = ROOT / "control/receipt-graph.json"
    graph = bundle.load_canonical_json(graph_path)
    graph["manifests"][path.name] = bundle.sha256_file(path)
    bundle.atomic_json(graph_path, graph)


def ensure_mmm_external() -> dict[str, Any]:
    manifest_path = ROOT / "control/external-source-manifest.json"
    if EXTERNAL.exists() or manifest_path.exists():
        manifest = bundle.load_canonical_json(manifest_path)
        if manifest.get("commit") != MMM_COMMIT or manifest.get("tree") != MMM_TREE:
            raise bundle.Refusal("MMM external authority drift")
        return manifest
    if bundle.git(MMM_PROVIDER, "rev-parse", f"{MMM_COMMIT}^{{tree}}") != MMM_TREE:
        raise bundle.Refusal("MMM external commit/tree drift")
    if bundle.git(MMM_PROVIDER, "rev-parse", f"{MMM_TAG}^{{}}") != MMM_COMMIT:
        raise bundle.Refusal("MMM external tag target drift")
    bundle.archive_git_tree(MMM_PROVIDER, MMM_COMMIT, EXTERNAL)
    records = bundle.git_tree_records(MMM_PROVIDER, MMM_COMMIT)
    verified = bundle.verify_materialized_tree(EXTERNAL, records)
    manifest = {
        "schema": "wrfgpu2-v0234-external-source-v1", "logical_role": "required WRF CMake MMM-physics external",
        "origin_config": "WRF v4.7.1 arch/Externals.cfg", "repo_url": "https://github.com/NCAR/MMM-physics.git",
        "tag": MMM_TAG, "commit": MMM_COMMIT, "tree": MMM_TREE, "provider": str(MMM_PROVIDER),
        "provider_dirty_but_not_copied": bool(bundle.git(MMM_PROVIDER, "status", "--porcelain")),
        "tracked_records": verified,
    }
    bundle.atomic_json(manifest_path, manifest)
    add_manifest_to_graph(manifest_path)
    seal_tree(EXTERNAL)
    return manifest


def preserve_failed_attempt() -> dict[str, Any] | None:
    """Move a recognized no-compilation harness failure aside without deletion."""
    if not (INSTRUMENTED.exists() or BUILD.exists()):
        return None
    build_log = ROOT / "build/build.log"
    if (ROOT / "control/build-manifest.json").exists():
        raise bundle.Refusal("existing completed build manifest forbids another build")
    if build_log.exists():
        if not (INSTRUMENTED.is_dir() and BUILD.is_dir()):
            raise bundle.Refusal("compile failure lacks source/build tree")
        text = build_log.read_text(encoding="utf-8", errors="replace")
        if all(marker in text for marker in ("Symbol 'chem_dname_table'", "shr_megan_mod.mod", "module_sf_clm.f90.o", "Error 2")):
            failure_index = 3
            cause = "pristine v4.7.1 CMake defined unselected WRF_USE_CLM against a no-chem registry, unlike legacy build semantics"
        elif all(marker in text for marker in ("'sst'", "domain' structure", "module_cpl.F.o", "Error 2")):
            failure_index = 4
            cause = "CMake generated io_boilerplate_temporary.inc in build/Registry but resolved Registry.EM includes from source/Registry"
        else:
            raise bundle.Refusal("existing compile failure identity is not recognized")
        if any(path.is_file() for path in (INSTALL / "bin/wrf", INSTALL / "run/wrf.exe", INSTALL / "main/wrf.exe")):
            raise bundle.Refusal("compile-failure namespace unexpectedly contains WRF binary")
        configure_log = ROOT / "build/configure.log"
        patch_log = ROOT / "control/wrf-instrumentation-patch.log"
        destinations = {
            INSTRUMENTED: ROOT / f"source/instrumented-compile-failure-{failure_index}",
            BUILD: ROOT / f"build/wrf-compile-failure-{failure_index}",
            build_log: ROOT / f"build/build-failure-{failure_index}.log",
            configure_log: ROOT / f"build/configure-success-before-failure-{failure_index}.log",
            patch_log: ROOT / f"control/wrf-instrumentation-patch-failure-{failure_index}.log",
        }
        route_patch_log = ROOT / "control/wrf-cmake-route-patch.log"
        if route_patch_log.exists():
            destinations[route_patch_log] = ROOT / f"control/wrf-cmake-route-patch-failure-{failure_index}.log"
        if any(not source.exists() for source in destinations):
            raise bundle.Refusal("compile-failure evidence path absent")
        if any(destination.exists() or destination.is_symlink() for destination in destinations.values()):
            raise bundle.Refusal("compile-failure preservation collision")
        for source, destination in destinations.items():
            os.rename(source, destination)
        proof_path = ROOT / f"control/compile-failure-{failure_index}.json"
        proof = {
            "schema": "wrfgpu2-v0234-compile-failure-v1", "classification": "HARNESS_ONLY_NO_WRF_OR_MPI_EXECUTION",
            "cause": cause,
            "build_log": bundle.assert_regular(destinations[build_log]),
            "configure_log": bundle.assert_regular(destinations[configure_log]),
            "preserved_paths": [str(value) for value in destinations.values()], "gpu_actions": 0,
            "wrf_or_mpi_executions": 0, "scientific_alternation_advanced": False,
        }
        bundle.atomic_json(proof_path, proof)
        add_manifest_to_graph(proof_path)
        return proof
    configure_log = ROOT / "build/configure.log"
    if INSTRUMENTED.exists() and not BUILD.exists() and not configure_log.exists():
        physics_mmm = INSTRUMENTED / "phys/physics_mmm"
        patch_log = ROOT / "control/wrf-instrumentation-patch.log"
        if physics_mmm.exists() or physics_mmm.is_symlink() or patch_log.exists():
            raise bundle.Refusal("unrecognized partial source assembly")
        destination = ROOT / "source/instrumented-assembly-failure-2"
        if destination.exists() or destination.is_symlink():
            raise bundle.Refusal("assembly-failure preservation collision")
        os.rename(INSTRUMENTED, destination)
        proof_path = ROOT / "control/source-assembly-failure-2.json"
        proof = {
            "schema": "wrfgpu2-v0234-source-assembly-failure-v1", "classification": "HARNESS_ONLY_NO_CONFIGURE_OR_COMPILATION",
            "cause": "sealed pristine parent was not made owner-writable before inserting pinned MMM-physics external",
            "preserved_path": str(destination), "gpu_actions": 0, "wrf_or_mpi_executions": 0,
            "scientific_alternation_advanced": False,
        }
        bundle.atomic_json(proof_path, proof)
        add_manifest_to_graph(proof_path)
        return proof
    if not configure_log.is_file():
        raise bundle.Refusal("existing incomplete build lacks configure log")
    text = configure_log.read_text(encoding="utf-8", errors="replace")
    marker = "Cannot find source file:"
    missing = str(INSTRUMENTED / "phys/physics_mmm/bl_gwdo.F90")
    if marker not in text or missing not in text or "CMake Generate step failed" not in text:
        raise bundle.Refusal("existing configure failure identity is not recognized")
    destinations = {
        INSTRUMENTED: ROOT / "source/instrumented-configure-failure-1",
        BUILD: ROOT / "build/wrf-configure-failure-1",
        configure_log: ROOT / "build/configure-failure-1.log",
    }
    patch_log = ROOT / "control/wrf-instrumentation-patch.log"
    if patch_log.exists():
        destinations[patch_log] = ROOT / "control/wrf-instrumentation-patch-failure-1.log"
    if any(destination.exists() or destination.is_symlink() for destination in destinations.values()):
        raise bundle.Refusal("failed-configure preservation destination collision")
    for source, destination in destinations.items():
        os.rename(source, destination)
    proof_path = ROOT / "control/configure-failure-1.json"
    proof = {
        "schema": "wrfgpu2-v0234-configure-failure-v1", "classification": "HARNESS_ONLY_NO_COMPILATION",
        "cause": "required arch/Externals.cfg MMM-physics source was not materialized from its pinned Git object",
        "configure_log": bundle.assert_regular(destinations[configure_log]),
        "preserved_paths": [str(value) for value in destinations.values()], "gpu_actions": 0,
        "wrf_or_mpi_executions": 0, "scientific_alternation_advanced": False,
    }
    bundle.atomic_json(proof_path, proof)
    add_manifest_to_graph(proof_path)
    return proof


def construct_build() -> dict[str, Any]:
    if git_head_dirty := bundle.git(REPO, "status", "--porcelain"):
        raise bundle.Refusal(f"build requires clean approved worktree:{git_head_dirty}")
    audit = bundle.audit(ROOT)
    assert_no_production_collision()
    failed_attempt = preserve_failed_attempt()
    external_manifest = ensure_mmm_external()
    if INSTRUMENTED.exists() or INSTRUMENTED.is_symlink() or BUILD.exists() or BUILD.is_symlink():
        raise bundle.Refusal("instrumented source/build namespace is not fresh after preservation")
    bundle.assert_regular(PATCH, PATCH_SHA256)
    bundle.assert_regular(ROUTE_PATCH, ROUTE_PATCH_SHA256)
    bundle.assert_regular(REGISTRY_PATCH, REGISTRY_PATCH_SHA256)
    toolchain_record = bundle.assert_regular(TOOLCHAIN)
    pristine_manifest_before = bundle.sha256_file(ROOT / "control/source-manifest.json")

    shutil.copytree(PRISTINE, INSTRUMENTED, symlinks=True)
    chmod_owner_writable(INSTRUMENTED)
    shutil.copytree(EXTERNAL, INSTRUMENTED / "phys/physics_mmm", symlinks=True)
    chmod_owner_writable(INSTRUMENTED / "phys/physics_mmm")
    patch_log = ROOT / "control/wrf-instrumentation-patch.log"
    with PATCH.open("rb") as patch_stream, patch_log.open("xb") as log:
        result = subprocess.run(["patch", "-p1", "--batch", "--forward"], cwd=INSTRUMENTED, stdin=patch_stream, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise bundle.Refusal(f"instrumentation patch failed: {patch_log}")
    route_patch_log = ROOT / "control/wrf-cmake-route-patch.log"
    with ROUTE_PATCH.open("rb") as patch_stream, route_patch_log.open("xb") as log:
        result = subprocess.run(["patch", "-p1", "--batch", "--forward"], cwd=INSTRUMENTED, stdin=patch_stream, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise bundle.Refusal(f"WRF CMake route patch failed: {route_patch_log}")
    registry_patch_log = ROOT / "control/wrf-cmake-registry-patch.log"
    with REGISTRY_PATCH.open("rb") as patch_stream, registry_patch_log.open("xb") as log:
        result = subprocess.run(["patch", "-p1", "--batch", "--forward"], cwd=INSTRUMENTED, stdin=patch_stream, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise bundle.Refusal(f"WRF CMake registry patch failed: {registry_patch_log}")
    if bundle.sha256_file(ROOT / "control/source-manifest.json") != pristine_manifest_before:
        raise bundle.Refusal("pristine source manifest changed during overlay")

    BUILD.mkdir(parents=True)
    shutil.copy2(TOOLCHAIN, BUILD / "wrf_config.cmake")
    env = dict(os.environ)
    env["PATH"] = str(TOOLCHAIN_ROOT / "bin") + os.pathsep + env.get("PATH", "")
    env.update({"CC": "gcc", "FC": "gfortran", "F77": "gfortran", "F90": "gfortran", "CUDA_VISIBLE_DEVICES": ""})
    cmake = TOOLCHAIN_ROOT / "bin/cmake"
    configure_argv = [
        str(cmake), "--fresh", "-S", str(INSTRUMENTED), "-B", str(BUILD), "-G", "Unix Makefiles",
        f"-DCMAKE_INSTALL_PREFIX={INSTALL}", f"-DCMAKE_TOOLCHAIN_FILE={BUILD / 'wrf_config.cmake'}",
        "-DWRF_CORE=ARW", "-DWRF_CASE=EM_REAL", "-DWRF_NESTING=BASIC",
        f"-DMPI_ROOT={TOOLCHAIN_ROOT}", f"-DnetCDF_ROOT={TOOLCHAIN_ROOT}",
        f"-DnetCDF-Fortran_ROOT={TOOLCHAIN_ROOT}", f"-DHDF5_ROOT={TOOLCHAIN_ROOT}",
        f"-DJasper_ROOT={TOOLCHAIN_ROOT}", f"-DZLIB_ROOT={TOOLCHAIN_ROOT}",
        "-DUSE_MPI=ON", "-DUSE_OPENMP=OFF", "-DBUILD_EXTERNALS=OFF",
    ]
    configure = run_logged(configure_argv, ROOT / "build/configure.log", env)
    build_argv = [str(cmake), "--build", str(BUILD), "--target", "install", "--parallel", "6"]
    built = run_logged(build_argv, ROOT / "build/build.log", env)
    binary_candidates = [INSTALL / "bin/wrf", INSTALL / "run/wrf.exe", INSTALL / "main/wrf.exe"]
    binaries = [path for path in binary_candidates if path.is_file() and not path.is_symlink()]
    if len(binaries) != 1:
        raise bundle.Refusal(f"WRF binary authority ambiguous: {binaries}")
    binary = binaries[0]
    source_tree_sha, source_records = tree_manifest(INSTRUMENTED)
    manifest = {
        "schema": "wrfgpu2-v0234-pristine-pbl-build-v1", "contract_commit": bundle.CONTRACT_COMMIT,
        "approved_head": bundle.git(REPO, "rev-parse", "HEAD"), "wrf_commit": bundle.WRF_COMMIT,
        "wrf_tree": bundle.WRF_TREE, "patch": bundle.assert_regular(PATCH, PATCH_SHA256),
        "patch_log": bundle.assert_regular(patch_log), "toolchain_file": toolchain_record,
        "route_patch": bundle.assert_regular(ROUTE_PATCH, ROUTE_PATCH_SHA256),
        "route_patch_log": bundle.assert_regular(route_patch_log),
        "route_patch_scope": "disable unselected CLM macro; sf_surface_physics=4 Noah-MP route unchanged",
        "registry_patch": bundle.assert_regular(REGISTRY_PATCH, REGISTRY_PATCH_SHA256),
        "registry_patch_log": bundle.assert_regular(registry_patch_log),
        "registry_patch_scope": "resolve exact generated Registry include closure inside fresh build root",
        "copied_toolchain_file": bundle.assert_regular(BUILD / "wrf_config.cmake", toolchain_record["sha256"]),
        "configure": configure, "build": built, "wrf_binary": bundle.assert_regular(binary),
        "instrumented_source_root": str(INSTRUMENTED), "instrumented_source_tree_sha256": source_tree_sha,
        "instrumented_source_records": source_records, "pristine_source_manifest_sha256": pristine_manifest_before,
        "mmm_external": external_manifest, "prior_configure_failure": failed_attempt,
        "resource_envelope": audit["resource_gate"], "build_parallelism": 6, "use_mpi": True,
        "use_openmp": False, "wrf_or_mpi_executions": 0, "gpu_actions": 0,
        "completed_utc": bundle.now(), "verdict": "FRESH_PINNED_WRF_BUILD_COMPLETE",
    }
    build_manifest = ROOT / "control/build-manifest.json"
    bundle.atomic_json(build_manifest, manifest)
    add_manifest_to_graph(build_manifest)
    seal_tree(INSTALL)
    return {"verdict": manifest["verdict"], "binary": str(binary), "binary_sha256": bundle.sha256_file(binary), "build_manifest_sha256": bundle.sha256_file(build_manifest), "receipt_graph_sha256": bundle.sha256_file(ROOT / "control/receipt-graph.json")}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--execute", action="store_true", help="perform the one-shot fresh build")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"verdict": "STATIC_ONLY", "root": str(ROOT)}, sort_keys=True))
        return 0
    try:
        print(json.dumps(construct_build(), sort_keys=True))
        return 0
    except (bundle.Refusal, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
