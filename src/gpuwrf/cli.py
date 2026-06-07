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
   via :func:`gpuwrf.io.namelist_check.validate_operational_namelist` -- which
   additionally refuses parity-proven-but-not-operationally-wired schemes so the
   operational run never silently substitutes a different scheme.
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
        required=False,
        default=None,
        type=Path,
        help="WRF namelist.input to validate fail-closed before the run. "
        "Defaults to <input-dir>/namelist.input (the case's own namelist).",
    )
    run.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Run directory holding the case inputs. STANDALONE native-init when it "
        "has wrfinput_<domain> + wrfbdy_d01 but no CPU wrfout history; CPU-WRF REPLAY "
        "when it has >=2 wrfout_<domain> history files (auto-detected).",
    )
    run.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated wrfout files and the run payload.",
    )
    run.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="Disk-backed scratch directory for transient NetCDF/diagnostics "
        "(NEVER /tmp tmpfs). Env GPUWRF_SCRATCH overrides; default is "
        "<output-dir>/.scratch. Cleaned up on exit.",
    )
    run.add_argument(
        "--domain",
        default="d02",
        help="Domain id to run for a SINGLE-domain run (e.g. d02). Ignored when "
        "the run is nested (max_dom > 1): all domains d01..dN are run together.",
    )
    run.add_argument(
        "--max-dom",
        type=int,
        default=None,
        help="Number of nested domains to run (d01..dN). Defaults to 1 "
        "(SINGLE domain = --domain). Set >1 to run the STANDALONE LIVE-NESTED driver "
        "(parent feeds each child's lateral boundary live; no CPU-WRF wrfout). "
        "1 runs the single-domain path on --domain.",
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
    run.add_argument(
        "--feedback",
        action="store_true",
        help="Enable TWO-WAY nesting (nested runs only): after each child finishes "
        "its subcycle, feed its interior back onto the overlapping parent cells "
        "(WRF copy_fcn area-average + sm121 feedback-zone smoother). Default off "
        "(one-way nesting, the v0.11.0/v0.12.0-validated wiring).",
    )
    run.set_defaults(func=_cmd_run)

    return parser


def _fail(message: str, *, code: int = 2) -> int:
    """Print a clean error to stderr and return an exit code (no traceback)."""
    print(f"gpuwrf: error: {message}", file=sys.stderr)
    return code


def _namelist_max_dom(namelist: Path) -> int:
    """Read ``&domains max_dom`` from a WRF namelist (cheap; pre-JAX). Defaults to 1."""
    from gpuwrf.io.gen2_accessor import parse_namelist

    parsed = parse_namelist(namelist)
    raw = parsed.get("domains", {}).get("max_dom", 1)
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else 1
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


# --------------------------------------------------------------------------- #
# run subcommand                                                              #
# --------------------------------------------------------------------------- #
def _resolve_scratch_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    """Resolve a DISK-backed scratch directory (never /tmp tmpfs).

    Precedence: ``--scratch-dir`` > ``$GPUWRF_SCRATCH`` > ``<output-dir>/.scratch``.
    The output dir is user-chosen and disk-backed, so ``.scratch`` under it inherits
    a real filesystem. ``$HOME/.gpuwrf_scratch`` is the fallback if the output dir is
    itself on tmpfs.
    """
    import os

    if args.scratch_dir is not None:
        return Path(args.scratch_dir)
    env = os.environ.get("GPUWRF_SCRATCH", "").strip()
    if env:
        return Path(env).expanduser()
    return output_dir / ".scratch"


def _is_tmpfs(path: Path) -> bool:
    """Best-effort check that ``path`` (or its nearest existing parent) is NOT tmpfs."""
    import shutil

    try:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        # A RAM-backed tmpfs typically reports a tiny total; the real risk the task
        # flags is the default /tmp tmpfs. Treat an explicit /tmp prefix as tmpfs.
        if str(path).startswith("/tmp/") or str(path) == "/tmp":
            return True
        del shutil  # filesystem-type probe is platform-specific; prefix check suffices
        return False
    except OSError:
        return False


def _maybe_reexec_for_nested_allocator(args: argparse.Namespace) -> None:
    """NESTED-OOM FIX: re-exec the process with the platform GPU allocator set.

    The live-nested path allocates a recurring ~8-9 GiB RRTMG g-point radiation
    transient every radiation step.  Under the default XLA BFC arena (especially
    with ``XLA_PYTHON_CLIENT_PREALLOCATE=false``) a 24 h run fragments the pool so
    that transient can no longer find a contiguous block -- the production
    "allocate 9.24 GiB" OOM -- even though peak in-use stays ~9 GiB.  The
    synchronous *platform* (cudaMalloc/cudaFree) allocator has no arena and so
    cannot fragment; it also keeps the resident set ~2x smaller (measured ~15 GiB
    vs ~29 GiB on this case), giving large headroom on a 32 GiB card.

    JAX reads ``XLA_PYTHON_CLIENT_ALLOCATOR`` from the OS environment when the GPU
    backend first initializes, and importing ``gpuwrf`` already imports ``jax``;
    setting the variable from Python at this point is not reliably honoured.  The
    robust, version-independent fix is to set it in the *environment* and re-exec
    the same interpreter command (``sys.orig_argv``) ONCE so the fresh process
    initializes the backend with the platform allocator from the start.  This is
    gated on the nested opt-in (``--max-dom > 1``) so the single-domain
    operational path keeps the faster default BFC arena, and on a one-shot guard
    env so we never loop.  An explicit operator ``XLA_PYTHON_CLIENT_ALLOCATOR``
    is always honoured (no re-exec).
    """
    import os
    import sys

    if not (args.max_dom is not None and int(args.max_dom) > 1):
        return
    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"):
        return  # operator already chose an allocator -- honour it, do not re-exec.
    if os.environ.get("_GPUWRF_NESTED_ALLOC_REEXEC") == "1":
        return  # already re-exec'd once; avoid an exec loop.
    orig = list(getattr(sys, "orig_argv", []) or [])
    if not orig:
        # No faithful argv to re-exec; fall back to a best-effort in-process set.
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        return
    new_env = dict(os.environ)
    new_env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    new_env["_GPUWRF_NESTED_ALLOC_REEXEC"] = "1"
    print(
        "gpuwrf: nested run -- re-exec with XLA_PYTHON_CLIENT_ALLOCATOR=platform "
        "(no-fragment cudaMalloc allocator; nested-OOM fix)",
        file=sys.stderr,
    )
    sys.stderr.flush()
    os.execvpe(orig[0], orig, new_env)


def _cmd_run(args: argparse.Namespace) -> int:
    # NESTED-OOM FIX: ensure the platform GPU allocator for live-nested runs by
    # re-exec'ing with it set in the environment (see the helper docstring). MUST
    # run before any jax device op; the nested pipeline also setdefaults it.
    _maybe_reexec_for_nested_allocator(args)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    # Default the namelist to the case's own namelist.input (naive-user friendly).
    namelist: Path = args.namelist if args.namelist is not None else (input_dir / "namelist.input")

    # --- Cheap fail-closed validation BEFORE any heavy import/compile. --------
    if not input_dir.is_dir():
        return _fail(f"--input-dir does not exist or is not a directory: {input_dir}")
    if not namelist.is_file():
        return _fail(
            f"--namelist file not found: {namelist} "
            "(pass --namelist or place namelist.input in --input-dir)"
        )
    if args.hours <= 0:
        return _fail(f"--hours must be a positive integer, got {args.hours}")

    compare_dir: Path | None = args.compare_cpu_dir
    if compare_dir is not None and not compare_dir.is_dir():
        return _fail(f"--compare-cpu-dir does not exist or is not a directory: {compare_dir}")

    proof_dir: Path = args.proof_dir if args.proof_dir is not None else (output_dir / "proofs")

    # --- Scratch directory: disk-backed, off /tmp tmpfs, cleaned on exit. ------
    scratch_dir = _resolve_scratch_dir(args, output_dir)
    if _is_tmpfs(scratch_dir):
        fallback = Path.home() / ".gpuwrf_scratch"
        print(
            f"gpuwrf: scratch dir {scratch_dir} is on /tmp tmpfs; using disk-backed "
            f"{fallback} instead (override with --scratch-dir / GPUWRF_SCRATCH).",
            file=sys.stderr,
        )
        scratch_dir = fallback

    # --- Namelist registry check (fail-closed, still pre-JAX). ----------------
    # The OPERATIONAL run path uses the *strict* operational validator: in
    # addition to the full validate_namelist support/out-of-scope checks, it also
    # refuses parity-proven-but-not-operationally-wired (REFERENCE_ONLY) schemes
    # -- classic RRTM/Dudhia radiation, MYJ/Janjic, New-Tiedtke -- because the
    # operational GPU scan cannot select them and would otherwise SILENTLY run a
    # different scheme (e.g. RRTMG for a requested RRTM/Dudhia). validate_namelist
    # alone accepts those for reference comparisons; the operational forecast must
    # not (v0.12.0 "no silent wrong path" contract).
    try:
        from gpuwrf.io.namelist_check import (
            UnsupportedSchemeError,
            validate_operational_namelist,
        )

        validate_operational_namelist(namelist)
    except UnsupportedSchemeError as exc:
        return _fail(str(exc))
    except Exception as exc:  # parsing / IO problems should also fail cleanly
        return _fail(f"could not validate namelist {namelist}: {type(exc).__name__}: {exc}")

    # --- Resolve the domain count. Nested is OPT-IN via --max-dom > 1; the
    # default is SINGLE-domain (--domain). This keeps CPU-wrfout REPLAY and
    # single-domain standalone runs on a multi-domain (max_dom>1) namelist from
    # being silently turned into a live-nested run. ----------------------------
    try:
        namelist_max_dom = _namelist_max_dom(namelist)
    except Exception:  # noqa: BLE001 - default to single-domain if max_dom unreadable
        namelist_max_dom = 1
    max_dom = int(args.max_dom) if args.max_dom is not None else 1
    if max_dom < 1:
        return _fail(f"--max-dom must be >= 1, got {max_dom}")
    if args.max_dom is None and namelist_max_dom > 1:
        print(
            f"gpuwrf: note: namelist max_dom={namelist_max_dom} but running a SINGLE "
            f"domain ({args.domain}) by default; pass --max-dom {namelist_max_dom} to "
            f"run the standalone live-nested forecast (d01..d{namelist_max_dom:02d}).",
            file=sys.stderr,
        )

    # --- Nested (max_dom > 1): STANDALONE LIVE-NESTED driver. -----------------
    # The parent advances, builds each child's lateral boundary LIVE, and recurses
    # to the child; the child IC comes from wrfinput_d0N and only wrfbdy_d01 forces
    # the root -- NO CPU-WRF wrfout dependency. max_dom == 1 keeps the single-domain
    # standalone/replay path below.
    if max_dom > 1:
        import os
        import shutil

        scratch_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("GPUWRF_SCRATCH", str(scratch_dir))
        os.environ["GPUWRF_TMPDIR"] = str(scratch_dir)
        # NESTED-OOM FIX: use the synchronous platform (cudaMalloc) allocator for
        # the live-nested path so the recurring ~9 GiB RRTMG radiation transient
        # cannot fragment a BFC arena over a 24 h run (the production
        # "allocate 9.24 GiB" OOM). Set BEFORE the JAX backend initializes; the
        # nested pipeline also setdefaults it. An explicit operator value wins.
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        cleanup_scratch = args.scratch_dir is None and not os.environ.get("GPUWRF_KEEP_SCRATCH")
        print(
            f"gpuwrf: init mode = standalone_native_init_nested -- STANDALONE "
            f"LIVE-NESTED (d01..d{max_dom:02d}; parent feeds each child LBC live; "
            f"no CPU-WRF wrfout); hours={args.hours}; scratch={scratch_dir}",
            file=sys.stderr,
        )
        try:
            from gpuwrf.integration.nested_pipeline import (
                NestedPipelineConfig,
                execute_nested_pipeline,
            )
        except Exception as exc:  # pragma: no cover - environment/dependency issue
            return _fail(
                f"failed to import the nested forecast driver "
                f"({type(exc).__name__}: {exc}). Is the package installed and is JAX available?"
            )

        nested_config = NestedPipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            proof_dir=proof_dir,
            hours=int(args.hours),
            max_dom=int(max_dom),
            scratch_dir=scratch_dir,
            feedback=bool(getattr(args, "feedback", False)),
        )
        if nested_config.feedback:
            print(
                "gpuwrf: TWO-WAY nesting ENABLED (child->parent copy_fcn + sm121 "
                "feedback-zone smoother).",
                file=sys.stderr,
            )
        try:
            payload = execute_nested_pipeline(nested_config)
        except Exception as exc:  # noqa: BLE001 - report cleanly, no traceback
            if cleanup_scratch:
                shutil.rmtree(scratch_dir, ignore_errors=True)
            return _fail(
                f"nested forecast failed ({type(exc).__name__}: {exc})", code=1
            )
        finally:
            if cleanup_scratch:
                shutil.rmtree(scratch_dir, ignore_errors=True)

        payload["scratch_dir"] = str(scratch_dir)
        # Persist the run payload alongside the single-domain pipeline artifact name.
        proof_dir.mkdir(parents=True, exist_ok=True)
        (proof_dir / "nested_pipeline_run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str)
        )
        verdict = str(payload.get("verdict", "UNKNOWN"))
        exit_code = 0 if verdict == "PIPELINE_GREEN" else 1
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return exit_code

    # --- Heavy path: import the pipeline and run. -----------------------------
    try:
        from gpuwrf.integration.daily_pipeline import (
            DailyPipelineConfig,
            detect_init_mode,
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

    # Auto-detect and announce the init mode so the user sees which path ran.
    init_mode = detect_init_mode(config)
    mode_label = (
        "STANDALONE native-init (IC from wrfinput, LBC from wrfbdy; no CPU-WRF wrfout)"
        if init_mode == "standalone_native_init"
        else "CPU-WRF REPLAY (IC + LBC from existing wrfout history)"
    )
    print(
        f"gpuwrf: init mode = {init_mode} -- {mode_label}; domain={args.domain} "
        f"hours={args.hours}; scratch={scratch_dir}",
        file=sys.stderr,
    )

    # Set scratch for any library code that honours it; create + clean it up.
    import os
    import shutil

    scratch_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GPUWRF_SCRATCH", str(scratch_dir))
    os.environ["GPUWRF_TMPDIR"] = str(scratch_dir)  # d02_replay trace root etc.
    cleanup_scratch = args.scratch_dir is None and not os.environ.get("GPUWRF_KEEP_SCRATCH")

    try:
        payload = execute_daily_pipeline(config)
    finally:
        if cleanup_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    payload["init_mode"] = init_mode
    payload["scratch_dir"] = str(scratch_dir)

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
