#!/usr/bin/env python
"""Run the relinked WRF binary path and emit the M6B0-R relinked oracle bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.validation.savepoint_io import read_savepoint
from m6b0r_wrf_savepoint_extract import emit_tier


SPRINT = ROOT / ".agent/sprints/2026-05-24-m6b0r-relink-completion"
PATCH_ROOT = ROOT / "external/wrf_savepoint_patch"
SOURCE_COPY = PATCH_ROOT / "source_copy"
RELINKED_WRF = SOURCE_COPY / "main/wrf.exe"
RUN_ROOT = PATCH_ROOT / "run_relinked"
SAVEPOINT_ROOT = PATCH_ROOT / "savepoints" / "relinked"
GOLDEN_MANIFEST = (
    ROOT
    / ".agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints/golden/manifest.json"
)
SOURCE_RUN = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")
ENV_SCRIPT = Path("<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_preflight_only() -> dict[str, object]:
    """Exercise the relinked executable enough to prove it is runnable.

    The inherited WRF patch currently exposes zero-argument hooks and does not
    carry field arrays into Fortran-side HDF5 writes. Until the patch interface
    is widened, the oracle bundle below is generated from the same Canary d02
    WRF source state as M6B0-R, but namespaced as the relinked proof lane.
    """

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not RELINKED_WRF.exists():
        raise FileNotFoundError(f"missing relinked WRF executable: {RELINKED_WRF}")
    copied = RUN_ROOT / "wrf.exe"
    shutil.copy2(RELINKED_WRF, copied)
    result = subprocess.run(
        ["bash", "-lc", f"source {ENV_SCRIPT} && mpirun -np 1 ./wrf.exe"],
        cwd=RUN_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )
    (SPRINT / "proof_relinked_run_binary_probe.txt").write_text(result.stdout)
    return {
        "binary": str(copied),
        "sha256": _sha256(copied),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
    }


def _inspect_files(files: list[str]) -> list[dict[str, object]]:
    inspected: list[dict[str, object]] = []
    for raw in files:
        path = Path(raw)
        savepoint = read_savepoint(path)
        inspected.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "boundary": savepoint.metadata.boundary,
                "operator": savepoint.metadata.operator,
                "tier": savepoint.metadata.tier,
                "rk_stage_index": savepoint.metadata.rk_stage_index,
                "acoustic_substep_index": savepoint.metadata.acoustic_substep_index,
                "fields": {
                    name: {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "units": savepoint.metadata.variables[name].units,
                        "stagger": savepoint.metadata.variables[name].stagger,
                    }
                    for name, array in sorted(savepoint.arrays.items())
                },
            }
        )
    return inspected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=SAVEPOINT_ROOT)
    parser.add_argument("--skip-binary-probe", action="store_true")
    args = parser.parse_args()

    SPRINT.mkdir(parents=True, exist_ok=True)
    binary_probe = None if args.skip_binary_probe else _run_preflight_only()
    output = args.output_root / args.tier
    manifest = emit_tier(args.tier, args.steps, output)
    inspected = _inspect_files([str(path) for path in manifest["files"]])
    golden = json.loads(GOLDEN_MANIFEST.read_text())
    proof = {
        "status": "PARTIAL_RELINKED_BINARY_PROBED_SAVEPOINTS_FROM_EXISTING_WRF_SOURCE_REPRODUCTION",
        "limitation": (
            "The applied M6B0-R solve_em.F.patch uses zero-argument hooks, so the "
            "relinked WRF binary cannot emit field-bearing calc_coef_w savepoints "
            "from inside the timestep loop without changing the patch interface."
        ),
        "tier": args.tier,
        "steps": args.steps,
        "binary_probe": binary_probe,
        "source_run": str(SOURCE_RUN),
        "golden_manifest_run_id": golden.get("run_id"),
        "manifest": manifest,
        "inspected_files": inspected,
    }
    listing_path = SPRINT / "proof_relinked_savepoints_listing.txt"
    listing_path.write_text(
        "\n".join(
            f"{item['sha256']} {item['path']} {item['operator']} {item['boundary']} "
            f"rk={item['rk_stage_index']} acoustic={item['acoustic_substep_index']}"
            for item in inspected
        )
        + "\n"
    )
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
