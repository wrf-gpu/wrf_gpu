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
        --domain d01 \\
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
        default=None,
        help="Domain id to run for a SINGLE-domain run (e.g. d01). Ignored when "
        "the run is nested (max_dom > 1): all domains d01..dN are run together. "
        "When omitted, defaults to d01 (WRF's root domain).",
    )
    run.add_argument(
        "--max-dom",
        type=int,
        default=None,
        help="Number of nested domains to run (d01..dN). Defaults to 1 "
        "(SINGLE domain = --domain). Set >1 to run the STANDALONE LIVE-NESTED driver "
        "(parent feeds each child's lateral boundary live; no CPU-WRF wrfout). "
        "1 runs the single-domain path on --domain. Use --domains-from-namelist to "
        "take this count from the namelist &domains max_dom instead.",
    )
    run.add_argument(
        "--domains-from-namelist",
        action="store_true",
        help="WRF-parity: resolve --max-dom from the namelist &domains max_dom so "
        "every domain the namelist declares (d01..dN) is run. Mutually exclusive "
        "with --max-dom.",
    )
    run.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Number of forecast hours to advance. When omitted, the forecast "
        "length is read from the namelist &time_control "
        "(run_days*24 + run_hours + run_minutes/60 + run_seconds/3600), rounded "
        "DOWN to whole hours but never below 1; if the namelist declares no "
        "positive duration it falls back to 1. An explicit --hours always wins.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the namelist, detect the input mode, resolve "
        "hours/domain/max_dom + scratch, print the effective run plan as JSON, and "
        "exit 0 WITHOUT running the forecast (no heavy JAX/GPU pipeline import).",
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
    run.add_argument(
        "--force-gpu-run",
        action="store_true",
        help="Nested runs only: bypass the GPU lock/free-VRAM preflight. Prefer "
        "GPUWRF_MIN_FREE_VRAM_FRACTION for card-relative tuning or "
        "GPUWRF_MIN_FREE_VRAM_GIB for an explicit threshold override; this force flag is for "
        "deliberate operator overrides and is also available as "
        "GPUWRF_FORCE_GPU_RUN=1.",
    )
    run.set_defaults(func=_cmd_run)

    # --- namelist-support: offline scheme-support registry (no JAX / no GPU). ---
    support = subparsers.add_parser(
        "namelist-support",
        help="print which physics schemes are operational / reference-only / "
        "fail-closed / out-of-scope (offline; no JAX, no GPU)",
        description=(
            "Print the operational scheme-support registry for the gated physics "
            "namelist keys (mp_physics, cu_physics, bl_pbl_physics, "
            "sf_sfclay_physics, sf_surface_physics, ra_lw_physics, ra_sw_physics). "
            "Reads the physics registry only; imports no JAX and allocates no GPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    support.add_argument(
        "--json",
        action="store_true",
        help="Emit the registry as JSON instead of a readable table.",
    )
    support.set_defaults(func=_cmd_namelist_support)

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


def _namelist_forecast_hours(namelist: Path) -> int | None:
    """Total forecast hours from ``&time_control`` run_days/hours/minutes/seconds.

    Cheap, pre-JAX. Computes ``run_days*24 + run_hours + run_minutes/60 +
    run_seconds/3600`` and rounds DOWN to whole hours (``int()`` floor of a
    positive value) but never below 1. Returns ``None`` when the namelist
    declares none of those keys or they sum to a non-positive duration, so the
    caller falls back to the historical default of 1 hour.
    """
    from gpuwrf.io.gen2_accessor import parse_namelist

    parsed = parse_namelist(namelist)
    time_control = parsed.get("time_control", {})

    def _first_number(key: str) -> float:
        value = time_control.get(key)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    total_hours = (
        _first_number("run_days") * 24.0
        + _first_number("run_hours")
        + _first_number("run_minutes") / 60.0
        + _first_number("run_seconds") / 3600.0
    )
    if total_hours <= 0:
        return None
    # Round DOWN to whole hours, but a positive sub-hour duration still runs >=1h.
    return max(1, int(total_hours))


def _effective_max_dom(args: argparse.Namespace) -> int:
    """Resolve the domain count import-light for the allocator/preflight gates.

    Mirrors the authoritative resolution in :func:`_cmd_run` (``--max-dom`` >
    ``--domains-from-namelist`` > default 1) but is defensive: it never raises,
    so it is safe to call before ``--input-dir``/``--namelist`` are validated.
    """
    if bool(getattr(args, "domains_from_namelist", False)):
        namelist = (
            args.namelist
            if args.namelist is not None
            else (args.input_dir / "namelist.input")
        )
        try:
            if Path(namelist).is_file():
                return _namelist_max_dom(Path(namelist))
        except Exception:  # noqa: BLE001 - defensive; real errors surface in _cmd_run
            return 1
        return 1
    if args.max_dom is not None:
        try:
            return int(args.max_dom)
        except (TypeError, ValueError):
            return 1
    return 1


def _detect_init_mode_light(input_dir: Path, domain: str, max_dom: int) -> str:
    """Import-light replica of ``daily_pipeline.detect_init_mode`` for --dry-run.

    A nested run (max_dom > 1) is always the standalone live-nested driver; a
    single-domain run dir with >=2 ``wrfout_<domain>`` history files is CPU-WRF
    replay, otherwise standalone native-init. Kept in lockstep with
    ``daily_pipeline.detect_init_mode`` (same glob) but imports no JAX pipeline.
    """
    if max_dom > 1:
        return "standalone_native_init_nested"
    run_dir = Path(input_dir).expanduser()
    wrfout_count = len(sorted(run_dir.glob(f"wrfout_{domain}_*")))
    return "cpu_wrf_replay" if wrfout_count >= 2 else "standalone_native_init"


# WRF runtime tables that gpuwrf reads from ``$GPUWRF_WRF_ROOT/run`` at startup.
# Only these two selected schemes actually load from the pristine WRF tree in
# this port (RRTMG/Thompson tables ship as bundled fixture assets); the preflight
# note therefore keys on exactly them to avoid false alarms.
_WRF_ROOT_TABLE_SCHEMES: tuple[tuple[str, int, str, tuple[str, ...], bool], ...] = (
    # (key, code, scheme label, required run/ files, require_all)
    (
        "sf_surface_physics",
        4,
        "Noah-MP land surface (sf_surface_physics=4)",
        ("MPTABLE.TBL", "SOILPARM.TBL", "GENPARM.TBL"),
        True,
    ),
    (
        "ra_lw_physics",
        1,
        "classic RRTM longwave (ra_lw_physics=1)",
        ("RRTM_DATA_DBL", "RRTM_DATA"),
        False,
    ),
)


def _namelist_scheme_codes(physics: dict[str, Any], key: str) -> set[int]:
    """Integer codes selected for ``key`` (handles scalars, per-domain lists and
    Fortran ``N*M`` repeat tokens)."""
    raw = physics.get(key)
    if raw is None:
        return set()
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    codes: set[int] = set()
    for item in items:
        try:
            codes.add(int(item))
            continue
        except (TypeError, ValueError):
            pass
        text = str(item)
        if "*" in text:  # Fortran repeat count, e.g. "3*8" -> value 8
            _, _, value = text.partition("*")
            try:
                codes.add(int(value.strip()))
            except ValueError:
                pass
    return codes


def _wrf_root_preflight_note(namelist: Path) -> str | None:
    """Pre-JAX UX note when a $GPUWRF_WRF_ROOT table-loading scheme is selected
    but the pristine WRF runtime tables are missing.

    Non-fatal and advisory: the physics table loader still fail-closes later.
    Returns ``None`` when no such scheme is selected or the tables are present.
    """
    import os

    try:
        from gpuwrf.io.gen2_accessor import parse_namelist

        parsed = parse_namelist(namelist)
    except Exception:  # noqa: BLE001 - advisory only
        return None
    physics = parsed.get("physics", {})

    from gpuwrf.config.paths import wrf_root, wrf_run_dir

    run_dir = wrf_run_dir()
    missing: list[tuple[str, tuple[str, ...]]] = []
    for key, code, label, files, require_all in _WRF_ROOT_TABLE_SCHEMES:
        if code not in _namelist_scheme_codes(physics, key):
            continue
        present = (
            all((run_dir / f).is_file() for f in files)
            if require_all
            else any((run_dir / f).is_file() for f in files)
        )
        if not present:
            missing.append((label, files))
    if not missing:
        return None

    root = wrf_root()
    env_set = bool(os.environ.get("GPUWRF_WRF_ROOT", "").strip())
    lines = [
        "gpuwrf: note: a namelist scheme needs the unmodified WRF runtime tables, "
        "but they were not found where gpuwrf searched:",
    ]
    for label, files in missing:
        lines.append(f"  - {label} reads {', '.join(files)} from {run_dir}")
    lines.append(
        "  Why: gpuwrf parses these UNMODIFIED WRF tables at startup; without a "
        "valid pristine WRF v4 tree the run will fail-closed when the scheme "
        "initializes."
    )
    if env_set:
        lines.append(
            f"  Fix: GPUWRF_WRF_ROOT={root} is set but its run/ dir lacks the tables "
            "above -- point it at a real pristine WRF v4 checkout: "
            "export GPUWRF_WRF_ROOT=/path/to/WRF"
        )
    else:
        lines.append(
            f"  Fix: GPUWRF_WRF_ROOT is unset, so the default {root} was searched. "
            "Set it to your pristine WRF v4 checkout: "
            "export GPUWRF_WRF_ROOT=/path/to/WRF"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# namelist-support subcommand (offline; no JAX, no GPU)                        #
# --------------------------------------------------------------------------- #
# Ordered physics keys shown in the support registry (the operational gate).
_NAMELIST_SUPPORT_KEYS: tuple[str, ...] = (
    "mp_physics",
    "cu_physics",
    "bl_pbl_physics",
    "sf_sfclay_physics",
    "sf_surface_physics",
    "ra_lw_physics",
    "ra_sw_physics",
)

# SupportStatus.value -> user-facing bucket used by the task ("operational /
# reference-only / fail-closed / disabled/out-of-scope").
_SUPPORT_STATUS_LABEL: dict[str, str] = {
    "implemented": "operational",
    "reference_only": "reference-only",
    "recognized_approximated": "approximated",
    "recognized_fail_closed": "fail-closed",
    "out_of_scope": "out-of-scope",
}


def _namelist_support_registry() -> dict[str, Any]:
    """Build the offline scheme-support registry (no JAX / no GPU import)."""
    from gpuwrf.io.scheme_catalog import classify_scheme
    from gpuwrf.io.wrf_scheme_catalog import WRF_PARAM_LABEL, WRF_SCHEME_CATALOG

    keys: dict[str, Any] = {}
    for key in _NAMELIST_SUPPORT_KEYS:
        options = []
        for code in sorted(WRF_SCHEME_CATALOG.get(key, ())):
            support = classify_scheme(key, code)
            options.append(
                {
                    "code": code,
                    "status": support.status.value,
                    "status_label": _SUPPORT_STATUS_LABEL.get(
                        support.status.value, support.status.value
                    ),
                    "wrf_name": support.wrf_name,
                    "reason": support.reason,
                    "alternative": support.alternative,
                }
            )
        keys[key] = {"label": WRF_PARAM_LABEL.get(key, key), "options": options}
    return {
        "schema": "GpuwrfNamelistSupport",
        "schema_version": 1,
        "keys": keys,
    }


def _cmd_namelist_support(args: argparse.Namespace) -> int:
    registry = _namelist_support_registry()
    if bool(getattr(args, "json", False)):
        print(json.dumps(registry, indent=2, sort_keys=True, default=str))
        return 0

    lines = [
        "GPU-WRF operational scheme support",
        "==================================",
        "operational   = GPU scan-wired (runs normally)",
        "reference-only = oracle-backed, NOT operational (fail-closed in a real run)",
        "fail-closed    = recognized WRF option, refused with a named reason",
        "out-of-scope   = a deliberate design decision not to port this option",
    ]
    for key, info in registry["keys"].items():
        lines.append("")
        lines.append(f"{key} ({info['label']})")
        lines.append(f"  {'code':>4}  {'status':<14}  scheme")
        for opt in info["options"]:
            name = opt["wrf_name"] or ""
            lines.append(f"  {opt['code']:>4}  {opt['status_label']:<14}  {name}")
    print("\n".join(lines))
    return 0


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


# --------------------------------------------------------------------------- #
# Nested GPU allocator selection (v0.20.0 speed lever G_allocator_env)         #
# --------------------------------------------------------------------------- #
# DEFAULT for the live-nested path is now ``cuda_async`` -- the CUDA
# stream-ordered pooling allocator (``cudaMallocAsync``).  Historically this
# path forced the synchronous ``platform`` allocator (raw cudaMalloc/cudaFree
# per buffer, no pool) to dodge the production 1km-nest "allocate 9.24 GiB" OOM:
# the default XLA *BFC* arena fragments over a 24 h run so the recurring ~9 GiB
# RRTMG radiation transient can no longer find a contiguous block.  ``platform``
# has no arena and so cannot fragment, but it pays a full driver malloc/free per
# device buffer on every op, which shows up as host-side dispatch overhead.
#
# ``cuda_async`` is a *different* mechanism from BFC: it is a stream-ordered
# memory pool managed by the CUDA driver.  It POOLS (so per-op malloc/free churn
# is amortised, unlike ``platform``) but it does NOT use the BFC best-fit arena
# whose fragmentation caused the original OOM -- the driver's async pool releases
# memory back at stream-ordered sync points and is designed to stay
# VRAM-bounded.  So it should recover most of ``platform``'s per-op overhead
# while keeping the nest VRAM-bounded.  This is a NUMERICS-FREE change: the
# allocator only governs *where* device buffers live, never the math.
#
# Selection precedence (highest first):
#   1. explicit operator ``XLA_PYTHON_CLIENT_ALLOCATOR`` -- always authoritative.
#   2. ``GPUWRF_ALLOCATOR`` -- documented gpuwrf-level knob; accepts
#      ``cuda_async`` (default), ``platform`` (no-fragment fallback), ``bfc``
#      (default XLA arena), or ``default``.
#   3. built-in default ``cuda_async``.
#
# The platform fallback remains one env var away (``GPUWRF_ALLOCATOR=platform``)
# for any case where ``cuda_async`` is unavailable or regresses.
_DEFAULT_NESTED_ALLOCATOR = "cuda_async"
_VALID_ALLOCATORS = ("cuda_async", "platform", "bfc", "default")


def _resolve_nested_allocator() -> str | None:
    """Return the XLA allocator string to use for the live-nested path.

    Returns ``None`` when an explicit operator ``XLA_PYTHON_CLIENT_ALLOCATOR`` is
    already set (caller must NOT override it).  Otherwise maps the documented
    ``GPUWRF_ALLOCATOR`` knob to an XLA allocator string, defaulting to
    ``cuda_async``.  An unrecognised ``GPUWRF_ALLOCATOR`` value is passed through
    verbatim (so a future XLA allocator name still works) but a warning is left
    to the caller; here we just normalise the well-known aliases.
    """
    import os

    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"):
        return None  # operator chose explicitly -- honour it, do not override.
    requested = os.environ.get("GPUWRF_ALLOCATOR", "").strip().lower()
    if not requested:
        return _DEFAULT_NESTED_ALLOCATOR
    # ``bfc`` is the XLA default arena; XLA spells that allocator "default".
    if requested == "bfc":
        return "default"
    return requested


def _maybe_reexec_for_nested_allocator(args: argparse.Namespace) -> None:
    """NESTED allocator selection: re-exec the process with the chosen GPU allocator.

    The default is now ``cuda_async`` (stream-ordered pool); see the module
    comment above for why this replaced the old ``platform`` default.  The
    selection is robust and version-independent: JAX reads
    ``XLA_PYTHON_CLIENT_ALLOCATOR`` from the OS environment when the GPU backend
    first initializes, and importing ``gpuwrf`` already imports ``jax``, so
    setting the variable from Python at this point is not reliably honoured.  We
    therefore set it in the *environment* and re-exec the same interpreter
    command (``sys.orig_argv``) ONCE so the fresh process initializes the backend
    with the chosen allocator from the start.

    Gated on the nested opt-in (``--max-dom > 1``) so the single-domain
    operational path keeps the faster default BFC arena, and on a one-shot guard
    env so we never loop.  An explicit operator ``XLA_PYTHON_CLIENT_ALLOCATOR``
    is always honoured (no re-exec); ``GPUWRF_ALLOCATOR`` selects among
    ``cuda_async`` (default), ``platform`` (no-fragment fallback), ``bfc``.
    """
    import os
    import sys

    if bool(getattr(args, "dry_run", False)):
        return  # --dry-run never touches the GPU allocator.
    if _effective_max_dom(args) <= 1:
        return
    allocator = _resolve_nested_allocator()
    if allocator is None:
        return  # operator already chose an allocator -- honour it, do not re-exec.
    if os.environ.get("_GPUWRF_NESTED_ALLOC_REEXEC") == "1":
        return  # already re-exec'd once; avoid an exec loop.
    orig = list(getattr(sys, "orig_argv", []) or [])
    if not orig:
        # No faithful argv to re-exec; fall back to a best-effort in-process set.
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", allocator)
        return
    new_env = dict(os.environ)
    new_env["XLA_PYTHON_CLIENT_ALLOCATOR"] = allocator
    new_env["_GPUWRF_NESTED_ALLOC_REEXEC"] = "1"
    print(
        f"gpuwrf: nested run -- re-exec with XLA_PYTHON_CLIENT_ALLOCATOR={allocator} "
        f"(GPUWRF_ALLOCATOR={os.environ.get('GPUWRF_ALLOCATOR') or 'cuda_async (default)'}; "
        "stream-ordered pool by default, GPUWRF_ALLOCATOR=platform for the "
        "no-fragment cudaMalloc fallback)",
        file=sys.stderr,
    )
    sys.stderr.flush()
    os.execvpe(orig[0], orig, new_env)


def _cmd_run(args: argparse.Namespace) -> int:
    dry_run = bool(getattr(args, "dry_run", False))

    # Domain-count selectors are mutually exclusive (cheap check, pre-anything).
    if bool(getattr(args, "domains_from_namelist", False)) and args.max_dom is not None:
        return _fail(
            "--max-dom and --domains-from-namelist are mutually exclusive: pass "
            "--max-dom N to set the domain count explicitly, or "
            "--domains-from-namelist to take it from the namelist &domains max_dom, "
            "not both."
        )

    # NESTED-OOM FIX: ensure the cuda_async GPU allocator for live-nested runs by
    # re-exec'ing with it set in the environment (see the helper docstring). MUST
    # run before any jax device op; the nested pipeline also setdefaults it.
    # (Skipped for --dry-run, which never touches the GPU.)
    _maybe_reexec_for_nested_allocator(args)
    gpu_preflight: dict[str, Any] | None = None

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
    # An EXPLICIT --hours must be positive; an omitted --hours (None) is resolved
    # from the namelist &time_control below.
    if args.hours is not None and args.hours <= 0:
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
            collect_namelist_warnings,
            validate_operational_namelist,
        )

        validate_operational_namelist(namelist)
    except UnsupportedSchemeError as exc:
        return _fail(str(exc))
    except Exception as exc:  # parsing / IO problems should also fail cleanly
        return _fail(f"could not validate namelist {namelist}: {type(exc).__name__}: {exc}")

    # Non-fatal approximation warnings (the run PROCEEDS). The cumulus/PBL
    # cadence keys (cudt/bldt > 0) are not honored verbatim -- the GPU port runs
    # those physics every dynamics step, a conservative approximation -- so a real
    # WRF namelist (e.g. cudt=5) is accepted with a named warning rather than
    # rejected, mirroring what the operational pipeline already does.
    for _warning in collect_namelist_warnings(namelist):
        print(f"gpuwrf: warning: {_warning}", file=sys.stderr)

    # --- Resolve the domain count. Nested is OPT-IN via --max-dom > 1 (or the
    # WRF-parity --domains-from-namelist); the default is SINGLE-domain
    # (--domain). This keeps CPU-wrfout REPLAY and single-domain standalone runs
    # on a multi-domain (max_dom>1) namelist from being silently turned into a
    # live-nested run. ---------------------------------------------------------
    try:
        namelist_max_dom = _namelist_max_dom(namelist)
    except Exception:  # noqa: BLE001 - default to single-domain if max_dom unreadable
        namelist_max_dom = 1
    if bool(getattr(args, "domains_from_namelist", False)):
        max_dom = namelist_max_dom
        maxdom_source = "namelist"
    elif args.max_dom is not None:
        max_dom = int(args.max_dom)
        maxdom_source = "flag"
    else:
        max_dom = 1
        maxdom_source = "default"
    if max_dom < 1:
        return _fail(f"--max-dom must be >= 1, got {max_dom}")

    # --- Resolve the single-domain id. WRF's root domain d01 is the default
    # (the historical d02 default surprised WRF users). An explicit --domain wins;
    # it is ignored for nested runs (all domains d01..dN run together). ---------
    if args.domain is not None:
        effective_domain = str(args.domain)
        domain_source = "flag"
    else:
        effective_domain = "d01"
        domain_source = "default"

    # --- Resolve the forecast length. An explicit --hours wins; otherwise it is
    # computed from the namelist &time_control, falling back to 1 hour. ---------
    if args.hours is not None:
        effective_hours = int(args.hours)
        hours_source = "flag"
        hours_source_phrase = "--hours override"
    else:
        namelist_hours = _namelist_forecast_hours(namelist)
        if namelist_hours is not None:
            effective_hours = namelist_hours
            hours_source = "namelist"
            hours_source_phrase = "namelist &time_control"
        else:
            effective_hours = 1
            hours_source = "default"
            hours_source_phrase = "default"

    # Announce the resolved forecast length + domain(s) so the user always sees
    # what will run and where each value came from.
    print(
        f"gpuwrf: forecast length = {effective_hours} h (source: {hours_source_phrase})",
        file=sys.stderr,
    )
    if max_dom > 1:
        print(
            f"gpuwrf: domains = d01..d{max_dom:02d} (source: {maxdom_source}); "
            "--domain is ignored for nested runs",
            file=sys.stderr,
        )
    else:
        print(
            f"gpuwrf: domain = {effective_domain} (source: {domain_source})",
            file=sys.stderr,
        )
    if maxdom_source == "default" and namelist_max_dom > 1:
        print(
            f"gpuwrf: note: namelist max_dom={namelist_max_dom} but running a SINGLE "
            f"domain ({effective_domain}) by default; pass --max-dom {namelist_max_dom} "
            f"(or --domains-from-namelist) to run the standalone live-nested forecast "
            f"(d01..d{namelist_max_dom:02d}).",
            file=sys.stderr,
        )

    # --- WRF_ROOT table preflight (pre-JAX UX aid; non-fatal). ----------------
    wrf_root_note = _wrf_root_preflight_note(namelist)
    if wrf_root_note:
        print(wrf_root_note, file=sys.stderr)

    # --- Effective-values metadata carried into the run payload / dry-run plan.
    run_metadata: dict[str, Any] = {
        "namelist_path": str(namelist),
        "namelist_max_dom": namelist_max_dom,
        "effective_max_dom": max_dom,
        "effective_domain": effective_domain,
        "effective_hours": effective_hours,
        "override_sources": {
            "hours": hours_source,
            "domain": domain_source,
            "max_dom": maxdom_source,
        },
    }

    # --- DRY RUN: print the effective plan as JSON and exit WITHOUT importing
    # the heavy JAX/GPU forecast pipeline or allocating a GPU. -----------------
    if dry_run:
        plan = {
            "schema": "GpuwrfRunPlan",
            "schema_version": 1,
            "dry_run": True,
            "run_type": "nested_live" if max_dom > 1 else "single_domain",
            "init_mode": _detect_init_mode_light(input_dir, effective_domain, max_dom),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "proof_dir": str(proof_dir),
            "scratch_dir": str(scratch_dir),
            "compare_cpu_dir": str(compare_dir) if compare_dir is not None else None,
            "feedback": bool(getattr(args, "feedback", False)),
            "score": bool(getattr(args, "score", False)),
            "wrf_root_preflight": wrf_root_note,
            **run_metadata,
        }
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
        return 0

    # --- Nested GPU preflight (nested runs only; skipped above for --dry-run).
    if max_dom > 1:
        try:
            from gpuwrf.runtime.gpu_preflight import (
                GpuPreflightError,
                run_nested_gpu_preflight,
            )

            gpu_preflight = run_nested_gpu_preflight(
                force=bool(getattr(args, "force_gpu_run", False))
            )
        except GpuPreflightError as exc:
            return _fail(str(exc), code=75)

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
        # NESTED allocator (v0.20.0 speed lever): default to ``cuda_async`` (the
        # stream-ordered CUDA pool) so per-op malloc/free churn is amortised while
        # staying VRAM-bounded; ``GPUWRF_ALLOCATOR=platform`` restores the old
        # no-fragment synchronous cudaMalloc fallback that dodged the 1km-nest
        # "allocate 9.24 GiB" OOM.  Set BEFORE the JAX backend initializes (the
        # re-exec above normally already did this); the nested pipeline also
        # setdefaults it. An explicit operator XLA_PYTHON_CLIENT_ALLOCATOR wins.
        _alloc = _resolve_nested_allocator()
        if _alloc is not None:
            os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", _alloc)
        cleanup_scratch = args.scratch_dir is None and not os.environ.get("GPUWRF_KEEP_SCRATCH")
        print(
            f"gpuwrf: init mode = standalone_native_init_nested -- STANDALONE "
            f"LIVE-NESTED (d01..d{max_dom:02d}; parent feeds each child LBC live; "
            f"no CPU-WRF wrfout); hours={effective_hours}; scratch={scratch_dir}",
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
            hours=int(effective_hours),
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
        payload.update(run_metadata)
        if gpu_preflight is not None:
            payload["gpu_preflight"] = gpu_preflight
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
        hours=int(effective_hours),
        output_dir=output_dir,
        proof_dir=proof_dir,
        domain=effective_domain,
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
        f"gpuwrf: init mode = {init_mode} -- {mode_label}; domain={effective_domain} "
        f"hours={effective_hours}; scratch={scratch_dir}",
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
    payload.update(run_metadata)

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
