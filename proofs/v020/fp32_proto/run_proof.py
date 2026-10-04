"""Generate CPU-only proof artifacts for the v0.20 fp32 prototype."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .acceptance_bands import export_spec
    from .fp32_column_proto import run_all_cases
except ImportError:  # pragma: no cover - direct script execution fallback
    from acceptance_bands import export_spec
    from fp32_column_proto import run_all_cases


OUT_DIR = Path(__file__).resolve().parent


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsteps", type=int, default=20_000)
    args = parser.parse_args()

    results = run_all_cases(nsteps=args.nsteps)
    results["generated_utc"] = datetime.now(timezone.utc).isoformat()
    results["command_context"] = {
        "cwd": str(Path.cwd()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "python_module": __file__,
        "nsteps": args.nsteps,
        "note": "Pure NumPy CPU-only prototype; no JAX/CUDA imports.",
    }

    _write_json(OUT_DIR / "proof_results.json", results)
    _write_json(OUT_DIR / "acceptance_bands_snapshot.json", export_spec())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

