#!/usr/bin/env python3
"""Run the ADR-023 1h Gen2 d02 boundary replay and write a JSON proof."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("GPUWRF_D02_REPLAY_DEBUG", "1")

_START = time.perf_counter()


def _log(message: str) -> None:
    if os.environ.get("GPUWRF_D02_REPLAY_DEBUG", "1").lower() not in {"0", "false", "no", "off"}:
        print(f"[d02-replay-cli +{time.perf_counter() - _START:8.3f}s] {message}", flush=True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_log("importing jax.config")
from jax import config

_log("importing replay dependencies")
from gpuwrf.coupling.boundary_apply import BoundaryConfig
from gpuwrf.integration.d02_replay import DEFAULT_OUTPUT_FIELD_PATH, DEFAULT_REPLAY_RUN_DIR, ReplayConfig, run_replay_proof


config.update("jax_enable_x64", True)
_log("imports complete")

DEFAULT_PROOF = ROOT / ".agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_d02_replay.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROOF)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_REPLAY_RUN_DIR)
    parser.add_argument("--output-fields", type=Path, default=DEFAULT_OUTPUT_FIELD_PATH)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--n-acoustic", type=int, default=4)
    parser.add_argument("--radiation-cadence-steps", type=int, default=60)
    parser.add_argument("--spec-bdy-width", type=int, default=5)
    parser.add_argument("--spec-zone", type=int, default=1)
    parser.add_argument("--relax-zone", type=int, default=4)
    parser.add_argument("--spec-exp", type=float, default=0.0)
    parser.add_argument("--skip-final-radiation", dest="final_radiation", action="store_false")
    parser.add_argument("--skip-trace-audit", dest="include_trace_audit", action="store_false")
    parser.add_argument("--skip-static-audit", dest="include_static_audit", action="store_false")
    parser.set_defaults(final_radiation=True, include_trace_audit=True, include_static_audit=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    effective_dt_s = float(args.dt_s)
    if 0.0 < float(args.duration_s) < effective_dt_s:
        effective_dt_s = float(args.duration_s)
    _log(
        "parsed args "
        f"duration_s={args.duration_s:g} dt_s={args.dt_s:g} effective_dt_s={effective_dt_s:g} "
        f"n_acoustic={args.n_acoustic} "
        f"run_dir={args.run_dir} output={args.output}"
    )
    boundary = BoundaryConfig(
        spec_bdy_width=args.spec_bdy_width,
        spec_zone=args.spec_zone,
        relax_zone=args.relax_zone,
        update_cadence_s=3600.0,
        spec_exp=args.spec_exp,
    )
    replay_config = ReplayConfig(
        dt_s=effective_dt_s,
        duration_s=args.duration_s,
        n_acoustic=args.n_acoustic,
        radiation_cadence_steps=args.radiation_cadence_steps,
        final_radiation=args.final_radiation,
        boundary_config=boundary,
    )
    _log("calling run_replay_proof")
    payload = run_replay_proof(
        run_dir=args.run_dir,
        output_fields_path=args.output_fields,
        replay_config=replay_config,
        trace_dir=args.trace_dir,
        include_trace_audit=args.include_trace_audit,
        include_static_audit=args.include_static_audit,
    )
    _log(f"run_replay_proof returned status={payload.get('status')}")
    target = args.output
    if not target.is_absolute():
        target = ROOT / target
    _log(f"writing proof JSON to {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    _log("done")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
