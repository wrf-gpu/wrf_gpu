#!/usr/bin/env python3
"""Emit immutable WRF-v4.7.1 source and forcedown-generator authority.

Every authoritative byte is read from a git object at the pinned commit.  The
generated forcedown include is reproduced in a temporary directory from an
archive containing only tracked ``Registry`` and ``tools`` objects; the dirty
WRF worktree and its untracked ``inc`` directory are never inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


WRF_GIT = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
WRF_TREE = "6b658fbc98077fe0648cba724921679126464181"
EXPECTED_FORCEDOWN_SHA256 = (
    "ff3e2ef7c1032bd54941252618377a261c91a5f32c84b356151f5615a6b807b4"
)

CORE_SOURCE_PATHS = (
    "share/sint.F",
    "share/interp_fcn.F",
    "share/mediation_force_domain.F",
    "share/module_bc.F",
    "dyn_em/couple_or_uncouple_em.F",
    "dyn_em/solve_em.F",
    "dyn_em/module_em.F",
    "dyn_em/module_bc_em.F",
    "external/RSL_LITE/force_domain_em_part2.F",
)

GENERATOR_BUILD_PATHS = (
    "tools/Makefile",
    "tools/registry.c",
    "tools/registry.h",
    "tools/protos.h",
    "tools/data.c",
    "tools/data.h",
    "tools/type.c",
    "tools/misc.c",
    "tools/my_strtok.c",
    "tools/reg_parse.c",
    "tools/gen_defs.c",
    "tools/gen_allocs.c",
    "tools/gen_mod_state_descr.c",
    "tools/gen_scalar_indices.c",
    "tools/gen_args.c",
    "tools/gen_config.c",
    "tools/sym.c",
    "tools/sym.h",
    "tools/symtab_gen.c",
    "tools/gen_irr_diag.c",
    "tools/gen_model_data_ord.c",
    "tools/gen_interp.c",
    "tools/gen_comms.stub",
    "tools/gen_scalar_derefs.c",
    "tools/set_dim_strs.c",
    "tools/gen_wrf_io.c",
    "tools/gen_streams.c",
    "tools/standard.c",
    "tools/CMakeLists.txt",
    "inc/streams.h",
    "frame/Makefile",
    "arch/postamble",
)

REGISTRY_COMMAND = (
    "tools/registry",
    "-DEM_CORE=1",
    "-DDA_CORE=0",
    "-DWRF_EM_CORE=1",
    "-DNEW_BDYS",
    "Registry/Registry.EM",
)


def _run(args: tuple[str, ...] | list[str], *, cwd: Path | None = None) -> bytes:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[-4000:]
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(args)}\n{stderr}"
        )
    return result.stdout


def _git(*args: str) -> bytes:
    return _run(("git", "-C", str(WRF_GIT), *args))


def _object_bytes(path: str) -> bytes:
    return _git("show", f"{WRF_COMMIT}:{path}")


def _object_record(path: str) -> dict[str, object]:
    value = _object_bytes(path)
    blob = _git("rev-parse", f"{WRF_COMMIT}:{path}").decode().strip()
    return {
        "path": path,
        "blob": blob,
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def _tracked_paths(prefix: str) -> tuple[str, ...]:
    raw = _git("ls-tree", "-r", "--name-only", WRF_COMMIT, "--", prefix)
    return tuple(line for line in raw.decode().splitlines() if line)


def _reproduce_forcedown() -> dict[str, object]:
    # Archive only the tracked inputs required by Registry generation.  Using
    # ``git archive`` is the fail-closed boundary that excludes every dirty or
    # untracked worktree byte.
    archive = _git(
        "archive",
        "--format=tar",
        WRF_COMMIT,
        "Registry",
        "tools",
        "inc/streams.h",
    )
    with tempfile.TemporaryDirectory(prefix="v0234-wrf-registry-") as raw_tmp:
        root = Path(raw_tmp)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(root, filter="data")
        (root / "inc").mkdir(exist_ok=True)
        (root / "frame").mkdir()
        _run(("make", "-C", "tools", "registry", "CC_TOOLS=cc"), cwd=root)
        registry_log = _run(REGISTRY_COMMAND, cwd=root)
        generated_path = root / "inc/nest_forcedown_interp.inc"
        generated = generated_path.read_bytes()

    digest = hashlib.sha256(generated).hexdigest()
    if digest != EXPECTED_FORCEDOWN_SHA256:
        raise RuntimeError(
            "tracked Registry reproduction changed: "
            f"expected {EXPECTED_FORCEDOWN_SHA256}, got {digest}"
        )
    text = generated.decode()
    required = (
        "grid%u_2",
        "grid%v_2",
        "grid%w_2",
        "grid%ph_2",
        "grid%t_2",
        "grid%mu_2",
        "DO itrace = PARAM_FIRST_SCALAR, num_moist",
        "DO itrace = PARAM_FIRST_SCALAR, num_scalar",
        "moist_btxs",
        "scalar_btxs",
    )
    missing = tuple(fragment for fragment in required if fragment not in text)
    if missing:
        raise RuntimeError(f"generated forcedown include lacks {missing}")
    return {
        "output": "inc/nest_forcedown_interp.inc",
        "sha256": digest,
        "bytes": len(generated),
        "lines": len(text.splitlines()),
        "required_fragments": list(required),
        "registry_stdout_sha256": hashlib.sha256(registry_log).hexdigest(),
        "build_command": "make -C tools registry CC_TOOLS=cc",
        "generation_command": " ".join(REGISTRY_COMMAND),
        "archive_inputs": ["Registry", "tools", "inc/streams.h"],
        "dirty_worktree_input": False,
    }


def build_proof() -> dict[str, object]:
    commit = _git("rev-parse", f"{WRF_COMMIT}^{{commit}}").decode().strip()
    tree = _git("rev-parse", f"{WRF_COMMIT}^{{tree}}").decode().strip()
    if commit != WRF_COMMIT or tree != WRF_TREE:
        raise RuntimeError(f"WRF authority mismatch: commit={commit} tree={tree}")

    registry_paths = _tracked_paths("Registry")
    if "Registry/Registry.EM" not in registry_paths or "Registry/Registry.EM_COMMON" not in registry_paths:
        raise RuntimeError("tracked ARW Registry roots are missing")
    bound_paths = tuple(
        dict.fromkeys((*CORE_SOURCE_PATHS, *GENERATOR_BUILD_PATHS, *registry_paths))
    )
    records = tuple(_object_record(path) for path in bound_paths)
    proof: dict[str, object] = {
        "schema": "v0234-wrf-source-authority-v2",
        "verdict": "TRACKED_WRF_SOURCE_AND_GENERATOR_REPRODUCED",
        "wrf_git": str(WRF_GIT),
        "commit": commit,
        "tree": tree,
        "object_read_method": "git show <commit>:<path> plus blob id",
        "dirty_worktree_rejected": True,
        "source_objects": list(records),
        "registry_generator": _reproduce_forcedown(),
    }
    canonical = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()
    proof["proof_sha256"] = hashlib.sha256(canonical).hexdigest()
    return proof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    proof = build_proof()
    rendered = json.dumps(proof, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
