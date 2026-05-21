#!/usr/bin/env python
"""Validate M6-S3 surface-layer proof artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.proof_schemas import SurfaceLayerArtifact, validate_artifact


DEFAULT_ARTIFACTS = (
    ROOT / "artifacts" / "m6" / "radiation_conditioning_feasibility.json",
    ROOT / "artifacts" / "m6" / "land_state_manifest.json",
    ROOT / "artifacts" / "m6" / "surface_operational_delta.json",
)


def run(paths: tuple[Path, ...]) -> dict[str, object]:
    results = {}
    for path in paths:
        data = validate_artifact(path)
        SurfaceLayerArtifact.validate_dict(data)
        results[str(path.relative_to(ROOT))] = data["status"]
    delta_path = ROOT / "artifacts" / "m6" / "surface_operational_delta.json"
    delta = json.loads(delta_path.read_text(encoding="utf-8"))
    variables = delta.get("variables", {})
    for name in ("U10", "V10", "T2", "Q2"):
        if name not in variables:
            raise ValueError(f"surface operational delta missing {name}")
        if "0h" not in variables[name]:
            raise ValueError(f"surface operational delta missing {name}.0h")
    return {"status": "PASS", "validated": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="*", default=[str(path) for path in DEFAULT_ARTIFACTS])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = tuple(Path(item) if Path(item).is_absolute() else ROOT / item for item in args.artifacts)
    print(json.dumps(run(paths), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
