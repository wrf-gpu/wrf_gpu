"""``gpuwrf`` command-line interface.

This is the single public, README-driven entrypoint for running a forecast
through the JAX GPU port. It is a *thin* wrapper over the existing operational
pipeline (:func:`gpuwrf.integration.daily_pipeline.execute_daily_pipeline`); it
does not implement or reimplement any physics or dynamics.

Usage::

    gpuwrf run \\
        --namelist  <input-dir>/namelist.input \\
        --input-dir <CPU-WRF/Gen2 run dir> \\
        --output-dir runs/my_forecast \\
        --domain d02 \\
        --hours 1 \\
        --compare-cpu-dir <input-dir>

The ``run`` subcommand:

1. Validates the namelist *fail-closed* (before any expensive JAX import/compile)
   via :func:`gpuwrf.io.namelist_check.validate_supported_namelist`.
2. Loads the CPU-WRF/Gen2 case from ``--input-dir``, advances ``--hours`` hours
   through the GPU port, and writes ``wrfout`` history files + proof JSON.
3. Optionally compares generated ``wrfout`` *dimensions* against a CPU-WRF
   reference directory (``--compare-cpu-dir``) and writes
   ``<proof-dir>/dimension_compare.json``.
4. Prints one JSON payload to stdout and exits non-zero on any
   blocked/partial/dimension-fail result.

Heavy imports (JAX, the daily pipeline, netCDF4) are deferred into
:func:`_cmd_run` so that ``gpuwrf --help`` / ``gpuwrf run --help`` and basic
argument validation stay instant and do not require a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

__all__ = ["main", "build_parser", "compare_wrfout_dimensions"]


# --------------------------------------------------------------------------- #
# Dimension compare (P0 binding-gate helper; value/RMSE compare is a P1 follow) #
# --------------------------------------------------------------------------- #
def compare_wrfout_dimensions(
    generated_paths: Sequence[str | Path],
    compare_dir: str | Path,
) -> dict[str, Any]:
    """Compare NetCDF *dimensions* of each generated wrfout against a CPU reference.

    For each generated file, the CPU reference is the file of the same basename
    under ``compare_dir``. Every dimension name and length must match exactly.

    Returns a JSON-serializable payload with an overall ``status`` of
    ``"PASS"``/``"FAIL"`` (or ``"NO_OUTPUT"`` when nothing was generated).
    """
    from netCDF4 import Dataset  # deferred: only needed when comparing

    compare_dir = Path(compare_dir)
    files: list[dict[str, Any]] = []
    overall_pass = True

    if not generated_paths:
        return {
            "schema": "GpuwrfDimensionCompare",
            "schema_version": 1,
            "status": "NO_OUTPUT",
            "reason": "no generated wrfout files to compare",
            "compare_dir": str(compare_dir),
            "files": [],
        }

    for gen in generated_paths:
        gen_path = Path(gen)
        ref_path = compare_dir / gen_path.name
        entry: dict[str, Any] = {
            "generated": str(gen_path),
            "reference": str(ref_path),
        }
        if not gen_path.is_file():
            entry.update(status="MISSING_GENERATED", pass_=False)
            entry["pass"] = False
            overall_pass = False
            files.append(entry)
            continue
        if not ref_path.is_file():
            entry["status"] = "MISSING_REFERENCE"
            entry["pass"] = False
            overall_pass = False
            files.append(entry)
            continue

        with Dataset(gen_path) as gen_ds, Dataset(ref_path) as ref_ds:
            gen_dims = {name: len(dim) for name, dim in gen_ds.dimensions.items()}
            ref_dims = {name: len(dim) for name, dim in ref_ds.dimensions.items()}

        mismatches: list[dict[str, Any]] = []
        for name in sorted(set(gen_dims) | set(ref_dims)):
            g = gen_dims.get(name)
            r = ref_dims.get(name)
            if g != r:
                mismatches.append({"dim": name, "generated": g, "reference": r})

        file_pass = not mismatches
        entry.update(
            status="PASS" if file_pass else "FAIL",
            generated_dims=gen_dims,
            reference_dims=ref_dims,
            mismatches=mismatches,
            pass_=file_pass,
        )
        entry["pass"] = file_pass
        entry.pop("pass_", None)
        overall_pass = overall_pass and file_pass
        files.append(entry)

    return {
        "schema": "GpuwrfDimensionCompare",
        "schema_version": 1,
        "status": "PASS" if overall_pass else "FAIL",
        "compare_dir": str(compare_dir),
        "file_count": len(files),
        "files": files,
    }


# --------------------------------------------------------------------------- #
# Argument parser                                                              #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuwrf",
        description=(
            "GPU-native WRF-compatible regional NWP. Run a forecast through the "
            "JAX GPU port over a CPU-WRF/Gen2 backfill case."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    run = subparsers.add_parser(
        "run",
        help="run a forecast through the GPU port and (optionally) dimension-compare vs CPU-WRF",
        description=(
            "Run a forecast through the JAX GPU port. Reads a CPU-WRF/Gen2 run "
            "directory (--input-dir), advances --hours hours, and writes wrfout "
            "history files plus proof JSON under --output-dir. Optionally compares "
            "the generated wrfout dimensions against a CPU-WRF reference directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run.add_argument(
        "--namelist",
        required=True,
        type=Path,
        help="WRF namelist.input to validate fail-closed before the run "
        "(must be the <input-dir>/namelist.input).",
    )
    run.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="CPU-WRF/Gen2 run directory: source of IC, boundaries, land/SST, "
        "and (by default) the CPU reference wrfouts.",
    )
    run.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated wrfout files and the run payload.",
    )
    run.add_argument(
        "--domain",
        default="d02",
        help="Domain id to run (e.g. d02).",
    )
    run.add_argument(
        "--hours",
        type=int,
        default=1,
        help="Number of forecast hours to advance.",
    )
    run.add_argument(
        "--proof-dir",
        type=Path,
        default=None,
        help="Directory for proof JSON. Defaults to <output-dir>/proofs.",
    )
    run.add_argument(
        "--compare-cpu-dir",
        type=Path,
        default=None,
        help="CPU-WRF reference directory for the dimension compare. If omitted, "
        "no dimension compare is run; pass --input-dir to compare against the "
        "case's own CPU wrfouts.",
    )
    run.add_argument(
        "--score",
        action="store_true",
        help="Also score against AEMET station observations (needs GPUWRF_AEMET_ROOT; "
        "not part of the README runnability gate).",
    )
    run.set_defaults(func=_cmd_run)

    return parser


def _fail(message: str, *, code: int = 2) -> int:
    """Print a clean error to stderr and return an exit code (no traceback)."""
    print(f"gpuwrf: error: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------- #
# run subcommand                                                              #
# --------------------------------------------------------------------------- #
def _cmd_run(args: argparse.Namespace) -> int:
    namelist: Path = args.namelist
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    # --- Cheap fail-closed validation BEFORE any heavy import/compile. --------
    if not input_dir.is_dir():
        return _fail(f"--input-dir does not exist or is not a directory: {input_dir}")
    if not namelist.is_file():
        return _fail(f"--namelist file not found: {namelist}")
    # P0 low-churn rule: the namelist must be the case's own namelist.input.
    expected = input_dir / "namelist.input"
    try:
        same = namelist.resolve() == expected.resolve()
    except OSError:
        same = False
    if not same:
        return _fail(
            "--namelist must be <input-dir>/namelist.input "
            f"(expected {expected}, got {namelist})"
        )
    if args.hours <= 0:
        return _fail(f"--hours must be a positive integer, got {args.hours}")

    compare_dir: Path | None = args.compare_cpu_dir
    if compare_dir is not None and not compare_dir.is_dir():
        return _fail(f"--compare-cpu-dir does not exist or is not a directory: {compare_dir}")

    proof_dir: Path = args.proof_dir if args.proof_dir is not None else (output_dir / "proofs")

    # --- Namelist registry check (fail-closed, still pre-JAX). ----------------
    try:
        from gpuwrf.io.namelist_check import (
            UnsupportedNamelistOption,
            validate_supported_namelist,
        )

        validate_supported_namelist(namelist)
    except UnsupportedNamelistOption as exc:
        return _fail(str(exc))
    except Exception as exc:  # parsing / IO problems should also fail cleanly
        return _fail(f"could not validate namelist {namelist}: {type(exc).__name__}: {exc}")

    # --- Heavy path: import the pipeline and run. -----------------------------
    try:
        from gpuwrf.integration.daily_pipeline import (
            DailyPipelineConfig,
            execute_daily_pipeline,
        )
    except Exception as exc:  # pragma: no cover - environment/dependency issue
        return _fail(
            f"failed to import the forecast pipeline ({type(exc).__name__}: {exc}). "
            "Is the package installed (pip install -e .) and is JAX available?"
        )

    config = DailyPipelineConfig(
        run_id=str(input_dir.resolve()),
        run_root=input_dir.parent,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=proof_dir,
        domain=args.domain,
        score=bool(args.score),
        restart_at_hour=None,
        repeat=False,
    )

    payload = execute_daily_pipeline(config)

    verdict = str(payload.get("verdict", "UNKNOWN"))
    exit_code = 0 if verdict == "PIPELINE_GREEN" else 1

    # --- Optional dimension compare for the binding gate. ---------------------
    if compare_dir is not None:
        generated = payload.get("wrfout_files", []) or []
        dim_result = compare_wrfout_dimensions(generated, compare_dir)
        proof_dir.mkdir(parents=True, exist_ok=True)
        dim_path = proof_dir / "dimension_compare.json"
        dim_path.write_text(json.dumps(dim_result, indent=2, sort_keys=True))
        payload["dimension_compare_status"] = dim_result["status"]
        payload["dimension_compare_path"] = str(dim_path)
        if dim_result["status"] != "PASS":
            exit_code = 1

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help(sys.stderr)
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
