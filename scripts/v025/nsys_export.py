#!/usr/bin/env python3
"""The ONE real `nsys stats` export path (CPU-only; reads an existing .nsys-rep).

Manager review on main `c227cfc7`: only `step1_pushpop.csv` was ever exported
from the real report. `nvtx_gpu_proj_sum.csv`, `cuda_gpu_mem_time_sum.csv` and
`cuda_gpu_mem_usage.csv` were written **only by the stub**, so the live analyser
silently saw missing files and the dry run proved nothing about the exporter.

Everything the analysis consumes now comes through here, and each output records
the exact command and a SHA-256 of its bytes. Two properties matter more than the
plumbing:

* **an empty real report is `BLOCKED`, not a successful empty measurement.**
  A report with no rows is equally consistent with "the run did nothing" and
  "the export silently failed", and only one of those is safe to score.
* **`nsys stats` on an existing `.nsys-rep` is CPU-only file processing.** It
  reads a file and writes CSV; it does not open the device. That is what lets
  this run outside the coordinated window, after the GPU is released.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

#: report name -> output filename. The analysis reads nothing else.
REPORTS: dict[str, str] = {
    "nvtx_pushpop_trace": "step1_pushpop.csv",
    "cuda_gpu_kern_sum": "cuda_gpu_kern_sum.csv",
    # Unlike cuda_gpu_mem_time_sum, this is a timestamped event table.  The
    # timestep-loop transfer audit is not allowed to consume an aggregate.
    "cuda_gpu_trace": "cuda_gpu_trace.csv",
    "cuda_gpu_mem_time_sum": "cuda_gpu_mem_time_sum.csv",
    "cuda_gpu_mem_size_sum": "cuda_gpu_mem_size_sum.csv",
    "cuda_gpu_sum": "cuda_gpu_sum.csv",
    "cuda_api_trace": "cuda_api_trace.csv",
    "cuda_api_sum": "cuda_api_sum.csv",
    "nvtx_gpu_proj_sum": "nvtx_gpu_proj_sum.csv",
}

#: Reports the census cannot be taken without. A soft failure here is a BLOCKED
#: census, never a partial one.
REQUIRED = (
    "nvtx_pushpop_trace",
    "cuda_gpu_kern_sum",
    "cuda_gpu_trace",
)


#: The exact notice nsys 2025.5 prints when it reuses a sibling `.sqlite`
#: instead of re-exporting the report (verified against the installed binary).
#: `parse_profiler.STALE_EXPORT_MARKER` carries the same literal for the reader
#: side; this module stays import-free of the parsers on purpose.
STALE_EXPORT_MARKER = "Existing SQLite export found"


class ExportFailed(RuntimeError):
    """The exporter could not produce a usable table."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_rows(text: str) -> int:
    """CSV rows after the header, ignoring nsys's preamble and blank lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    header = next((i for i, line in enumerate(lines) if "," in line and any(
        token in line for token in (
            "Name", "Range", "Operation", "Duration", "Time", "API"
        ))), None)
    return 0 if header is None else max(0, len(lines) - header - 1)


def _atomic_write_text(
    path: Path,
    text: str,
    *,
    allow_replace: bool = False,
) -> None:
    """Publish one complete export or no export at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = os.path.lexists(path)
    if existed and not allow_replace:
        raise ExportFailed(f"refusing to replace profiler export: {path}")
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if allow_replace:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_report(rep: Path, report: str, out_path: Path, *,
                  runner: Callable[[Sequence[str]], subprocess.CompletedProcess] | None = None,
                  run_id: str | None = None,
                  allow_replace: bool = False,
                  sqlite_path: Path | None = None,
                  create_sqlite: bool | None = None,
                  ) -> dict[str, Any]:
    """Run one ``nsys stats`` report and atomically persist its stdout.

    Destinations are unique/no-replace.  A pre-existing file is rejected before
    ``nsys`` runs so no stale cross-run artifact can be mistaken for this export.

    ``nsys stats`` silently reuses a ``.sqlite`` sitting next to the report if it
    finds one, printing only a ``NOTICE:`` line.  In a coordinated session that
    would reduce **a different run's** rows under this run's hashes, so the
    export is forced (``--force-export=true``) into a private per-analysis
    ``--sqlite`` path instead of the report's sibling default.
    """
    runner = runner or (lambda cmd: subprocess.run(list(cmd), capture_output=True, text=True,
                                                   check=False, timeout=1800))
    existed = os.path.lexists(out_path)
    if existed and not allow_replace:
        raise ExportFailed(f"profiler export target already exists: {out_path}")
    if sqlite_path is None:
        sqlite_path = out_path.parent / f".{out_path.stem}.private.sqlite"
        create_sqlite = True
    elif create_sqlite is None:
        create_sqlite = not Path(sqlite_path).exists()
    if Path(sqlite_path).suffix != ".sqlite":
        raise ExportFailed(
            f"private nsys export path must end in .sqlite: {sqlite_path}"
        )
    command = ["nsys", "stats", "--report", report, "--format", "csv"]
    if create_sqlite:
        # Exactly one call in export_all takes this branch.  It exports the
        # hashed .nsys-rep into a private SQLite database.
        command.extend(
            ["--force-export=true", "--sqlite", str(sqlite_path)]
        )
        input_path = rep
    else:
        if not Path(sqlite_path).is_file():
            raise ExportFailed(
                f"private SQLite input is missing: {sqlite_path}"
            )
        # Every later report reads the already-created private SQLite directly.
        # Supplying --sqlite or --force-export here would re-enter export logic.
        input_path = Path(sqlite_path)
    command += ["--output", "-", str(input_path)]
    proc = runner(command)
    record: dict[str, Any] = {
        "report": report,
        "command": " ".join(command),
        "argv": command,
        "command_sha256": hashlib.sha256(
            "\0".join(command).encode("utf-8")
        ).hexdigest(),
        "returncode": proc.returncode,
        "path": str(out_path),
        "run_id": run_id,
        "atomic_publish": True,
        "replacement": bool(existed and allow_replace),
        "sqlite_mode": "CREATE_ONCE" if create_sqlite else "READ_EXISTING",
        "sqlite_path": str(sqlite_path),
    }
    if proc.returncode != 0:
        record.update({"status": "FAILED", "stderr_tail": (proc.stderr or "")[-400:]})
        return record
    combined = (proc.stdout or "") + (proc.stderr or "")
    if STALE_EXPORT_MARKER in combined:
        # Defence in depth behind --force-export=true: if the tool still says it
        # reused an export, the rows may belong to another run and are refused.
        record.update({
            "status": "BLOCKED_STALE_EXPORT",
            "reason": (
                "nsys reported an existing SQLite export despite "
                "the private one-export path; these rows cannot be proven to come from "
                "the hashed report"
            ),
            "stderr_tail": (proc.stderr or "")[-400:],
        })
        return record

    _atomic_write_text(
        out_path,
        proc.stdout or "",
        allow_replace=allow_replace,
    )
    rows = _data_rows(proc.stdout or "")
    record.update({
        "status": "OK" if rows > 0 else "EMPTY",
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "data_rows": rows,
    })
    if rows == 0:
        record["reason"] = ("the report produced no data rows. That is NOT a measurement of "
                            "zero; it is an absent measurement.")
    return record


def export_all(rep: Path, out_root: Path, *, runner=None,
               reports: dict[str, str] | None = None,
               run_id: str | None = None,
               expected_source_sha256: str | None = None,
               allow_replace: bool = False,
               legacy_dry_path: bool = False) -> dict[str, Any]:
    """Export every report. Fails closed if a REQUIRED one is absent or empty.

    ``allow_replace`` exists only for the legacy canned/dry `step1_driver` path,
    which re-runs into a directory it already populated.  The authoritative
    post-lock path must never replace an export: replacement is exactly how two
    runs get mixed under one set of hashes.  Requiring the caller to also assert
    ``legacy_dry_path`` keeps that permission from spreading by default argument.
    """
    if allow_replace and not legacy_dry_path:
        raise ExportFailed(
            "export replacement is reserved for the legacy dry path; the "
            "authoritative post-lock export is no-replace so two runs can "
            "never be mixed under one manifest"
        )
    if not rep.is_file():
        raise ExportFailed(f"no nsys report at {rep}")
    out_root = Path(out_root)
    if os.path.lexists(out_root):
        if not out_root.is_dir() or (
            not allow_replace and any(out_root.iterdir())
        ):
            raise ExportFailed(
                f"profiler export root must be absent or empty: {out_root}"
            )
    else:
        out_root.mkdir(parents=True, exist_ok=False)
    reports = reports or REPORTS
    source_sha256 = sha256_file(rep)
    source_bytes = rep.stat().st_size
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ExportFailed(
            "nsys report hash changed between capture and analysis: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )

    effective_runner = runner or (
        lambda cmd: subprocess.run(
            list(cmd), capture_output=True, text=True, check=False, timeout=1800
        )
    )
    version_command = ["nsys", "--version"]
    version_proc = effective_runner(version_command)
    tool_version = {
        "argv": version_command,
        "command_sha256": hashlib.sha256(
            "\0".join(version_command).encode("utf-8")
        ).hexdigest(),
        "returncode": version_proc.returncode,
        "stdout": (version_proc.stdout or "").strip(),
        "stderr": (version_proc.stderr or "").strip(),
    }
    if version_proc.returncode != 0:
        raise ExportFailed("nsys --version failed before report export")
    # One private SQLite export per analysis, inside the (empty) export root.
    # Without this, `nsys stats` reuses whatever `<report>.sqlite` happens to sit
    # beside the .nsys-rep -- possibly a previous run's -- and only says so in a
    # NOTICE line that the CSV reduction would have to notice for itself.
    private_sqlite = out_root / "private_export.sqlite"
    if os.path.lexists(private_sqlite) and not allow_replace:
        raise ExportFailed(
            f"private nsys SQLite export already exists: {private_sqlite}"
        )
    exports: dict[str, dict[str, Any]] = {}
    for index, (name, filename) in enumerate(reports.items()):
        exports[name] = export_report(
            rep,
            name,
            out_root / filename,
            runner=effective_runner,
            run_id=run_id,
            allow_replace=allow_replace,
            sqlite_path=private_sqlite,
            create_sqlite=index == 0,
        )
        if index == 0 and (
            private_sqlite.is_symlink()
            or not private_sqlite.is_file()
            or private_sqlite.stat().st_size <= 0
        ):
            raise ExportFailed(
                "the one forced nsys export did not create a nonempty private SQLite"
            )
    private_sqlite_hash = sha256_file(private_sqlite)
    private_sqlite_bytes = private_sqlite.stat().st_size
    for record in exports.values():
        record["source_rep_sha256"] = source_sha256
        record["source_rep_bytes"] = source_bytes
        record["private_sqlite_sha256"] = private_sqlite_hash
    unusable = [name for name in REQUIRED
                if exports.get(name, {}).get("status") != "OK"]
    payload = {
        "schema": "wrf_gpu2.v025.m0.nsys_export.v2",
        "source_rep": str(rep),
        "source_rep_sha256": source_sha256,
        "source_rep_bytes": source_bytes,
        "run_id": run_id,
        "tool_version": tool_version,
        "output_root": str(out_root),
        "private_sqlite_export": str(private_sqlite),
        "private_sqlite_bytes": private_sqlite_bytes,
        "private_sqlite_sha256": private_sqlite_hash,
        "forced_sqlite_export_count": sum(
            "--force-export=true" in (record.get("argv") or [])
            for record in exports.values()
        ),
        "sqlite_report_read_count": sum(
            record.get("sqlite_mode") == "READ_EXISTING"
            for record in exports.values()
        ),
        "stale_sibling_sqlite_reuse_possible": False,
        "exports": exports,
        "required": list(REQUIRED),
        "required_unusable": unusable,
        "status": "OK" if not unusable else "BLOCKED",
        "note": ("`nsys stats` on an existing .nsys-rep is CPU-only file processing; it opens no "
                 "device, which is why this runs after the GPU is released."),
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "source_rep_sha256": source_sha256,
                "run_id": run_id,
                "tool_version": tool_version,
                "exports": {
                    name: {
                        key: record.get(key)
                        for key in (
                            "report", "argv", "returncode", "status", "path",
                            "bytes", "sha256", "data_rows",
                            "source_rep_sha256",
                            "private_sqlite_sha256",
                            "sqlite_mode",
                        )
                    }
                    for name, record in sorted(exports.items())
                },
                "private_sqlite": {
                    "path": str(private_sqlite),
                    "bytes": private_sqlite_bytes,
                    "sha256": private_sqlite_hash,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload
