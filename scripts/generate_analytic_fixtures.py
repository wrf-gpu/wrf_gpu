#!/usr/bin/env python3
"""Generate the M1 analytic fixture manifests and committed sample slices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.fixtures.analytic import generate_all  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic analytic M1 fixtures.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic generator seed.")
    parser.add_argument("--out", default="fixtures/samples/", help="Output directory for sample .npz files.")
    parser.add_argument(
        "--manifest-out",
        default="fixtures/manifests/",
        help="Output directory for fixture manifest YAML files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_dir = Path(args.out)
    manifest_dir = Path(args.manifest_out)
    generated = generate_all(args.seed, sample_dir, manifest_dir)
    for path in generated:
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
