#!/usr/bin/env python3
"""GPU acceptance-gate runner for v0.20.1 #114 (cross-date NEST warm cache) + #122
(compact training-ready wrfout subset).

WHAT THIS PROVES
----------------
#114: the fused 9-nest megakernel must NOT recompile when only the forecast DATE
changes. We run THREE dates -- DATE_A, DATE_B, leap -- in ONE Python process via
the in-process run API ``execute_nested_pipeline`` (NOT a subprocess per date), so
the in-memory ``@jax.jit`` trace cache is shared across the three runs (that
sharing is the whole point of the in-memory-hit check). Authoritative signal =
the PERSISTENT compile cache (``<DATA_ROOT>/gpuwrf_jax_cache``) gains N entries on the
cold DATE_A and ZERO new entries on DATE_B + leap (``persistent_new_zero``). A
second, corroborating signal = jax compile-logging + first-chunk wall time:
DATE_A logs many "Compiling ..." events and is slow; DATE_B/leap log ~0 and are
fast. (jit-lifecycle nuance: if the pipeline builds a fresh ``jax.jit`` object per
process-level call the in-memory cache may not be shared, so DATE_B then hits the
PERSISTENT cache instead -- still a fast no-recompile, still a pass. We capture
BOTH signals and report which path it took; persistent ``new==0`` is authoritative.)

#122: with ``GPUWRF_TRAINING_OUTPUT_SUBSET`` set in the env (0:2 sets it), each
date's per-domain wrfout is the compact 36-var lossless-compressed subset. We just
WRITE them to a known per-date out dir; 0:2's ``scripts/b200_training_smoketest.py``
on the 5090 is the LATER joint acceptance gate (this runner does not validate the
subset contents beyond emitting them).

Bit-identity (check c): for DATE_A=20260512 there is a v0.20.0 wrfout baseline
(``runs/20260512/gpu``). This runner just writes the DATE_A wrfout to a stated
path; 0:2 compares it to the baseline with ``compare_wrfout_grid.py`` (this runner
does NOT do the compare).

USAGE (0:2 runs this on the 5090, ~19:00, from the v0201-nest-prep worktree)
----------------------------------------------------------------------------
    GPUWRF_TRAINING_OUTPUT_SUBSET=1 \
    taskset -c 4-31 python scripts/gate_n114_n122.py \
        --template      <DATA_ROOT>/wrf_downscale/runs/20260512/gpu/namelist.input \
        --reference-dir <DATA_ROOT>/wrf_downscale/runs/20260512/gpu \
        --wrfinput-a    <DATA_ROOT>/wrf_downscale/runs/20260512/gpu \
        --wrfinput-b    <DATA_ROOT>/wrf_downscale/staging/20260228 \
        --wrfinput-leap <DATA_ROOT>/wrf_downscale/staging/20241124 \
        --tables-src    <DATA_ROOT>/wrf_downscale/canary_all7/run_cadvariant \
        --wrf-phys      <USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/phys \
        --cache-dir     <DATA_ROOT>/gpuwrf_jax_cache \
        --out-dir       <DATA_ROOT>/n114_n122_gate_$(date +%Y%m%dT%H%M%S)

All of the above paths have sensible defaults baked in (see ``build_parser``), so
the minimal invocation is just ``--out-dir <fresh>`` with ``GPUWRF_TRAINING_OUTPUT_SUBSET=1``.

SAFETY
------
``runs/20260512/gpu`` is the LIVE v0.20.0 validation run AND the bit-identity
baseline; it is STILL RUNNING. This runner NEVER writes into or overwrites it: it
stages all three dates into FRESH gate run-dirs under ``--out-dir`` and writes all
wrfouts there. For DATE_A it sources the wrfinput/wrfbdy from ``runs/20260512/gpu``
via READ-ONLY symlink into the fresh DATE_A gate dir.

Outputs ``<out-dir>/gate_n114_n122_report.json`` (+ readable stdout). Exit code 0
iff persistent_new_zero is True for BOTH later dates (the authoritative #114 pass).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# --- Baked-in default paths (coordinator-confirmed 0:2 staging layout). -------
DEF_REFERENCE_DIR = "<DATA_ROOT>/wrf_downscale/runs/20260512/gpu"
DEF_TEMPLATE = "<DATA_ROOT>/wrf_downscale/runs/20260512/gpu/namelist.input"
DEF_WRFINPUT_A = "<DATA_ROOT>/wrf_downscale/runs/20260512/gpu"
DEF_WRFINPUT_B = "<DATA_ROOT>/wrf_downscale/staging/20260228"
DEF_WRFINPUT_LEAP = "<DATA_ROOT>/wrf_downscale/staging/20241124"
DEF_TABLES_SRC = "<DATA_ROOT>/wrf_downscale/canary_all7/run_cadvariant"
DEF_WRF_PHYS = "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/phys"
DEF_CACHE_DIR = "<DATA_ROOT>/gpuwrf_jax_cache"

# (label, YYYYMMDD, start_hour) -- start_hour is the canary init hour (18Z).
DATES = (
    ("DATE_A", "20260512", 18),  # has a v0.20.0 wrfout baseline -> bit-identity check
    ("DATE_B", "20260228", 18),  # different month -> cross-date treedef axis
    ("leap", "20241124", 18),    # leap year (yearlen=366) -> the noahmp_yearlen axis
)

# Entries in the reference run-dir that are date/run-specific and must NOT be
# copied wholesale into a fresh gate dir (we re-create or re-point them per date).
_PER_RUN_NAMES = re.compile(
    r"^(wrfinput_d\d+|wrfbdy_d\d+|wrfout_.*|namelist\.input|gpuwrf_wrf_root|proofs|scratch|run_manifest\.json|.*\.pid)$"
)


# ---------------------------------------------------------------------------
# Namelist date templating
# ---------------------------------------------------------------------------
def _swap_namelist_dates(template_text: str, start: datetime, run_hours: int, max_dom: int) -> str:
    """Return ``template_text`` with the &time_control date block swapped to ``start``.

    Swaps ONLY the date-derived fields the coordinator confirmed: run_hours +
    start_/end_ year/month/day/hour/minute/second (identical across all domains).
    There is NO julyr/julday in the namelist -- gpuwrf computes those at runtime
    (#91) -- so nothing else is touched. The wrfinput ``Times[0]`` (which actually
    drives ``run_start``) already encodes the date for the wrfinput we symlink; we
    keep the namelist consistent so the namelist registry/parse and the wrfinput
    agree.
    """

    end = start + timedelta(hours=int(run_hours))

    def _vec(value: int) -> str:
        return ", ".join([str(int(value))] * int(max_dom)) + ","

    repl = {
        "run_hours": f"{int(run_hours)},",
        "run_days": "0,",
        "run_minutes": "0,",
        "run_seconds": "0,",
        "start_year": _vec(start.year),
        "start_month": _vec(start.month),
        "start_day": _vec(start.day),
        "start_hour": _vec(start.hour),
        "start_minute": _vec(start.minute),
        "start_second": _vec(start.second),
        "end_year": _vec(end.year),
        "end_month": _vec(end.month),
        "end_day": _vec(end.day),
        "end_hour": _vec(end.hour),
        "end_minute": _vec(end.minute),
        "end_second": _vec(end.second),
    }

    out_lines: list[str] = []
    seen: set[str] = set()
    # Match a leading-whitespace key, '=' and the rest of the line, preserving the
    # original column alignment of the value (WRF namelists are whitespace-tolerant
    # but we keep them tidy).
    key_re = re.compile(r"^(\s*)([A-Za-z_]\w*)(\s*)=\s*(.*)$")
    for line in template_text.splitlines():
        m = key_re.match(line)
        if m and m.group(2) in repl:
            indent, key, pad, _rest = m.groups()
            out_lines.append(f"{indent}{key}{pad}= {repl[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    missing = set(repl) - seen
    if missing:
        # Fail closed: a template that lacks a date field we expected to swap would
        # silently run the wrong date.
        raise ValueError(
            f"template {sorted(missing)} date field(s) not found in namelist -- "
            "cannot guarantee the date was swapped"
        )
    return "\n".join(out_lines) + "\n"


def _namelist_max_dom(text: str) -> int:
    m = re.search(r"^\s*max_dom\s*=\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else 9


# ---------------------------------------------------------------------------
# Per-date run-dir staging
# ---------------------------------------------------------------------------
def stage_run_dir(
    *,
    label: str,
    date_str: str,
    start_hour: int,
    run_hours: int,
    reference_dir: Path,
    template: Path,
    wrfinput_dir: Path,
    tables_src: Path,
    wrf_phys: Path,
    out_dir: Path,
    max_dom: int,
) -> Path:
    """Assemble a FRESH gate run-dir for one date and return its path.

    Replicates the working canary 9-nest layout:
      * ~80-100 physics-table symlinks copied verbatim from ``reference_dir``
        (they all point at ``tables_src``); never copies the heavy files.
      * wrfinput_d01..d0N + wrfbdy_d01 READ-ONLY symlinks into ``wrfinput_dir``.
      * a ``gpuwrf_wrf_root/`` dir with ``phys -> wrf_phys`` and ``run -> <this run-dir>``.
      * a date-swapped ``namelist.input`` written from ``template``.
    NEVER touches ``reference_dir`` / the live baseline run.
    """

    run_dir = out_dir / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # 1. Copy the reference dir's NON-per-run entries verbatim, preserving
    #    symlinks (copy the link, not its target -- the heavy table data stays at
    #    tables_src). Real small files (e.g. README copies) are copied as-is.
    for entry in sorted(reference_dir.iterdir()):
        if _PER_RUN_NAMES.match(entry.name):
            continue
        dst = run_dir / entry.name
        if entry.is_symlink():
            os.symlink(os.readlink(entry), dst)
        elif entry.is_dir():
            shutil.copytree(entry, dst, symlinks=True)
        else:
            shutil.copy2(entry, dst)

    # 2. Read-only wrfinput/wrfbdy symlinks pointing at this date's source.
    for dom in range(1, int(max_dom) + 1):
        src = wrfinput_dir / f"wrfinput_d{dom:02d}"
        if not src.exists():
            raise FileNotFoundError(f"{label}: missing {src}")
        os.symlink(src, run_dir / f"wrfinput_d{dom:02d}")
    wrfbdy = wrfinput_dir / "wrfbdy_d01"
    if wrfbdy.exists():
        os.symlink(wrfbdy, run_dir / "wrfbdy_d01")

    # 3. gpuwrf_wrf_root: phys -> the WRF phys tree, run -> THIS fresh run-dir.
    gpuwrf_root = run_dir / "gpuwrf_wrf_root"
    gpuwrf_root.mkdir()
    os.symlink(wrf_phys, gpuwrf_root / "phys")
    os.symlink(run_dir, gpuwrf_root / "run")

    # 4. Date-swapped namelist.input.
    start = datetime(int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]), int(start_hour))
    text = template.read_text()
    swapped = _swap_namelist_dates(text, start, run_hours, max_dom)
    (run_dir / "namelist.input").write_text(swapped)

    return run_dir


# ---------------------------------------------------------------------------
# jax compile-event capture (corroborating in-memory-hit signal)
# ---------------------------------------------------------------------------
class _CompileCounter(logging.Handler):
    """Counts jax "Compiling ..." / "Finished XLA compilation" log records.

    jax emits these at INFO on the ``jax`` logger when ``jax_log_compiles`` is on.
    A cold date logs many; a warm (no-recompile) date logs ~0. We snapshot the
    count delta per date.
    """

    _PAT = re.compile(r"compil", re.IGNORECASE)

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            if self._PAT.search(record.getMessage()):
                self.count += 1
        except Exception:  # pragma: no cover - never let logging crash the run
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="v0.20.1 #114+#122 GPU acceptance-gate runner (3 dates, one process).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, required=True,
                   help="FRESH output root for the gate run-dirs + wrfouts + report (never the baseline).")
    p.add_argument("--reference-dir", type=Path, default=Path(DEF_REFERENCE_DIR),
                   help="A complete working canary 9-nest run-dir whose table/symlink set is replicated.")
    p.add_argument("--template", type=Path, default=Path(DEF_TEMPLATE),
                   help="namelist.input template (the date fields are swapped per date).")
    p.add_argument("--wrfinput-a", type=Path, default=Path(DEF_WRFINPUT_A),
                   help="DATE_A (20260512) wrfinput_d01-d09 + wrfbdy source dir (READ-ONLY).")
    p.add_argument("--wrfinput-b", type=Path, default=Path(DEF_WRFINPUT_B),
                   help="DATE_B (20260228) wrfinput source dir.")
    p.add_argument("--wrfinput-leap", type=Path, default=Path(DEF_WRFINPUT_LEAP),
                   help="leap (20241124) wrfinput source dir.")
    p.add_argument("--tables-src", type=Path, default=Path(DEF_TABLES_SRC),
                   help="Canonical physics-table dir the reference symlinks point at (sanity-checked).")
    p.add_argument("--wrf-phys", type=Path, default=Path(DEF_WRF_PHYS),
                   help="WRF phys tree for gpuwrf_wrf_root/phys.")
    p.add_argument("--cache-dir", type=Path, default=Path(DEF_CACHE_DIR),
                   help="Persistent JAX compile cache dir (the #114 authoritative signal source).")
    p.add_argument("--hours", type=int, default=1, help="Forecast hours per date.")
    p.add_argument("--max-dom", type=int, default=None,
                   help="Domain count; default = the template's max_dom (9).")
    p.add_argument("--dry-check", action="store_true",
                   help="Validate inputs + stage all three run-dirs, then STOP (no GPU, no run).")
    return p


def _wrfinput_for(label: str, args: argparse.Namespace) -> Path:
    return {
        "DATE_A": args.wrfinput_a,
        "DATE_B": args.wrfinput_b,
        "leap": args.wrfinput_leap,
    }[label]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # --- Pin the persistent cache dir BEFORE importing gpuwrf/jax (the package
    #     import hook configures the cache from this env var). ------------------
    os.environ.setdefault("GPUWRF_JAX_CACHE_DIR", str(args.cache_dir))

    # Cheap fail-closed input validation (pre-import, pre-GPU).
    for name, path, must_be_dir in (
        ("--reference-dir", args.reference_dir, True),
        ("--template", args.template, False),
        ("--tables-src", args.tables_src, True),
        ("--wrf-phys", args.wrf_phys, True),
        ("--wrfinput-a", args.wrfinput_a, True),
        ("--wrfinput-b", args.wrfinput_b, True),
        ("--wrfinput-leap", args.wrfinput_leap, True),
    ):
        ok = path.is_dir() if must_be_dir else path.is_file()
        if not ok:
            print(f"gate: {name} does not exist or is wrong type: {path}", file=sys.stderr)
            return 2

    out_dir: Path = args.out_dir
    if out_dir.resolve() == args.reference_dir.resolve():
        print("gate: --out-dir must NOT be the reference/baseline run-dir", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    template_text = args.template.read_text()
    max_dom = int(args.max_dom) if args.max_dom is not None else _namelist_max_dom(template_text)

    subset_env = os.environ.get("GPUWRF_TRAINING_OUTPUT_SUBSET")
    print(
        f"gate: out_dir={out_dir} max_dom={max_dom} hours={args.hours} "
        f"cache_dir={args.cache_dir} GPUWRF_TRAINING_OUTPUT_SUBSET={subset_env!r}",
        file=sys.stderr,
    )

    # --- Stage all three run-dirs (no GPU yet). -------------------------------
    staged: dict[str, Path] = {}
    for label, date_str, start_hour in DATES:
        run_dir = stage_run_dir(
            label=label,
            date_str=date_str,
            start_hour=start_hour,
            run_hours=int(args.hours),
            reference_dir=args.reference_dir,
            template=args.template,
            wrfinput_dir=_wrfinput_for(label, args),
            tables_src=args.tables_src,
            wrf_phys=args.wrf_phys,
            out_dir=out_dir,
            max_dom=max_dom,
        )
        staged[label] = run_dir
        print(f"gate: staged {label} ({date_str}) -> {run_dir}", file=sys.stderr)

    if args.dry_check:
        print("gate: --dry-check complete; staged all three run-dirs, no run performed.", file=sys.stderr)
        print(json.dumps({"dry_check": True, "staged": {k: str(v) for k, v in staged.items()}}, indent=2))
        return 0

    # --- Import gpuwrf (configures the persistent cache to --cache-dir) + jax. -
    import jax  # noqa: F401  (import triggers backend init under the cache hook)
    from gpuwrf.integration.nested_pipeline import (  # noqa: E402
        NestedPipelineConfig,
        execute_nested_pipeline,
    )
    from gpuwrf.runtime.compile_cache import cache_entry_count, cache_report  # noqa: E402

    # Turn on jax compile logging for the corroborating in-memory-hit signal.
    try:
        jax.config.update("jax_log_compiles", True)
    except Exception:  # pragma: no cover - knob name drift; persistent count still authoritative
        pass
    compile_counter = _CompileCounter()
    jax_logger = logging.getLogger("jax")
    prev_level = jax_logger.level
    jax_logger.setLevel(logging.INFO)
    jax_logger.addHandler(compile_counter)

    cache_dir = str(args.cache_dir)
    report: dict[str, object] = {
        "cache_dir": cache_dir,
        "cache_report_at_start": cache_report(),
        "training_subset_env": subset_env,
        "max_dom": max_dom,
        "hours": int(args.hours),
        "dates": {},
    }

    prev_total = cache_entry_count(cache_dir)
    report["cache_entries_before_any_run"] = prev_total

    overall_ok = True
    for idx, (label, date_str, _start_hour) in enumerate(DATES):
        run_dir = staged[label]
        date_out = out_dir / f"{label}_out"
        date_out.mkdir(parents=True, exist_ok=True)
        proof_dir = date_out / "proofs"

        # Point GPUWRF_WRF_ROOT at THIS date's staged gpuwrf_wrf_root (phys tree +
        # run/ -> this run-dir's table symlinks), exactly as the reference canary
        # run did (run_manifest: GPUWRF_WRF_ROOT=<run-dir>/gpuwrf_wrf_root).
        # wrf_root() reads this env at call time, so per-date assignment is correct.
        os.environ["GPUWRF_WRF_ROOT"] = str(run_dir / "gpuwrf_wrf_root")

        compiles_before = compile_counter.count
        t0 = time.perf_counter()
        config = NestedPipelineConfig(
            input_dir=run_dir,
            output_dir=date_out,
            proof_dir=proof_dir,
            hours=int(args.hours),
            max_dom=max_dom,
            scratch_dir=date_out / "scratch",
        )
        run_error = None
        try:
            payload = execute_nested_pipeline(config)
        except Exception as exc:  # noqa: BLE001 - record cleanly, keep the report
            run_error = f"{type(exc).__name__}: {exc}"
            payload = {"verdict": "RUN_FAILED", "error": run_error}
        wall = time.perf_counter() - t0
        compiles_detected = compile_counter.count - compiles_before

        after_total = cache_entry_count(cache_dir)
        cache_delta = after_total - prev_total

        wrfouts = sorted(str(p) for p in date_out.glob("wrfout_*"))
        date_record = {
            "date": date_str,
            "run_dir": str(run_dir),
            "out_dir": str(date_out),
            "wall_seconds": round(wall, 2),
            "first_call_wall_s": round(wall, 2),  # one execute call/date => same
            "compiles_detected": int(compiles_detected),
            "cache_entries_before": prev_total,
            "cache_entries_after": after_total,
            "cache_delta": int(cache_delta),
            "persistent_new_zero": bool(cache_delta == 0),
            "wrfout_count": len(wrfouts),
            "wrfout_paths": wrfouts,
            "verdict": str(payload.get("verdict", "UNKNOWN")),
            "error": run_error,
        }
        # DATE_A wrfout path for 0:2's bit-identity compare (check c).
        if label == "DATE_A":
            date_record["bit_identity_compare_target"] = (
                wrfouts[0] if wrfouts else None
            )
            report["date_a_wrfout_dir"] = str(date_out)

        report["dates"][label] = date_record  # type: ignore[index]
        prev_total = after_total

        if run_error is not None:
            overall_ok = False
        print(
            f"gate: {label} ({date_str}) wall={wall:.1f}s "
            f"compiles={compiles_detected} cache_delta={cache_delta} "
            f"wrfouts={len(wrfouts)} verdict={date_record['verdict']}",
            file=sys.stderr,
        )

    jax_logger.removeHandler(compile_counter)
    jax_logger.setLevel(prev_level)

    # --- Verdicts. ------------------------------------------------------------
    # #114 AUTHORITATIVE: the two later dates add ZERO persistent cache entries.
    later = [report["dates"][k] for k in ("DATE_B", "leap")]  # type: ignore[index]
    persistent_new_zero = all(bool(d["persistent_new_zero"]) for d in later)
    # Corroborating: the later dates trigger ~0 compiles OR were fast (warm).
    in_memory_or_persistent_hit = all(int(d["compiles_detected"]) == 0 for d in later)
    date_a = report["dates"]["DATE_A"]  # type: ignore[index]
    date_a_cold = int(date_a["cache_delta"]) > 0 or int(date_a["compiles_detected"]) > 0

    report["verdict"] = {
        "persistent_new_zero": bool(persistent_new_zero),  # authoritative #114 pass
        "in_memory_or_persistent_hit": bool(in_memory_or_persistent_hit),
        "date_a_was_cold": bool(date_a_cold),
        "all_dates_ran": bool(overall_ok),
        "n114_pass": bool(persistent_new_zero and overall_ok),
    }
    report["cache_report_at_end"] = cache_report()

    report_path = out_dir / "gate_n114_n122_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    print("\n===== #114+#122 GATE REPORT =====")
    print(json.dumps(report["verdict"], indent=2, sort_keys=True))
    print(f"per-date: cache_delta / compiles / wall_s / wrfouts")
    for label, _ds, _sh in DATES:
        d = report["dates"][label]  # type: ignore[index]
        print(
            f"  {label:7s} {d['date']}: delta={d['cache_delta']:>4} "
            f"compiles={d['compiles_detected']:>4} wall={d['wall_seconds']:>7}s "
            f"wrfouts={d['wrfout_count']}  verdict={d['verdict']}"
        )
    print(f"report: {report_path}")
    print(f"DATE_A wrfout dir (0:2 bit-identity compare vs the v0.20.0 baseline): "
          f"{report.get('date_a_wrfout_dir')}")

    # Exit 0 iff the authoritative #114 signal passed AND all dates ran.
    return 0 if (persistent_new_zero and overall_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
