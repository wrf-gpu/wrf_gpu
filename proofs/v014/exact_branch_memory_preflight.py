#!/usr/bin/env python3
"""v0.14 exact-branch memory preflight.

This is a short gate before any long validation campaign. It verifies that the
current branch contains the known memory controls and, when explicitly asked,
runs a bounded representative nested GPU smoke through scripts/run_gpu_lowprio.sh.

Recommended GPU invocation:

  scripts/run_gpu_lowprio.sh --cores 0-23 -- \
    python proofs/v014/exact_branch_memory_preflight.py --run-gpu

The default without --run-gpu is audit-only and writes a no-run proof.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs" / "v014" / "exact_branch_memory_preflight.json"
OUT_MD = ROOT / "proofs" / "v014" / "exact_branch_memory_preflight.md"

DEFAULT_NESTED_INPUT = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z"
)
DEFAULT_RUN_ROOT_PARENT = Path("/mnt/data/wrf_gpu_validation")
GPU_LOCK = Path("/tmp/wrf_gpu_validation_gpu.lock")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def run_cmd(cmd: list[str], *, timeout_s: float | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "timeout_s": timeout_s,
        }


def ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_check(path: Path, checks: dict[str, str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    rows: dict[str, Any] = {}
    for name, pattern in checks.items():
        match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
        rows[name] = {
            "ok": bool(match),
            "pattern": pattern,
            "line": text[: match.start()].count("\n") + 1 if match else None,
        }
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "checks": rows,
        "ok": all(row["ok"] for row in rows.values()),
    }


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.relative_to(ROOT)), "ok": False, "reason": "missing"}
    try:
        return {
            "path": str(path.relative_to(ROOT)),
            "ok": True,
            "sha256": sha256_file(path),
            "payload": json.loads(path.read_text(encoding="utf-8")),
        }
    except Exception as exc:  # noqa: BLE001 - proof should record parse failures.
        return {
            "path": str(path.relative_to(ROOT)),
            "ok": False,
            "sha256": sha256_file(path),
            "reason": f"{type(exc).__name__}: {exc}",
        }


def git_snapshot() -> dict[str, Any]:
    branch = run_cmd(["git", "branch", "--show-current"])
    head = run_cmd(["git", "rev-parse", "HEAD"])
    status = run_cmd(["git", "status", "--short", "--branch"])
    return {
        "branch": branch["stdout"].strip(),
        "head": head["stdout"].strip(),
        "status_short": status["stdout"].splitlines(),
        "dirty": any(
            line and not line.startswith("##") for line in status["stdout"].splitlines()
        ),
    }


def nvidia_smi_query() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"ok": False, "reason": "nvidia-smi not found"}
    gpu = run_cmd(
        [
            exe,
            "--query-gpu=name,index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = run_cmd(
        [
            exe,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    payload: dict[str, Any] = {
        "ok": gpu["returncode"] == 0 and bool(gpu["stdout"].strip()),
        "gpu_query": {
            "returncode": gpu["returncode"],
            "stdout": gpu["stdout"].strip(),
            "stderr": gpu["stderr"].strip(),
        },
        "compute_apps_query": {
            "returncode": apps["returncode"],
            "stdout": apps["stdout"].strip(),
            "stderr": apps["stderr"].strip(),
        },
    }
    if not payload["ok"]:
        return payload
    first = gpu["stdout"].strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) >= 5:
        name, index, util, used, total = parts[:5]
        payload.update(
            {
                "name": name,
                "index": int(float(index)),
                "utilization_gpu_pct": int(float(util)),
                "memory_used_mib": int(float(used)),
                "memory_total_mib": int(float(total)),
            }
        )
    apps_rows = []
    for line in apps["stdout"].strip().splitlines():
        if not line.strip():
            continue
        app_parts = [part.strip() for part in line.split(",")]
        if len(app_parts) >= 3:
            apps_rows.append(
                {
                    "pid": int(float(app_parts[0])),
                    "process_name": app_parts[1],
                    "used_memory_mib": int(float(app_parts[2])),
                }
            )
    payload["compute_apps"] = apps_rows
    return payload


def gpu_idle_ok(sample: dict[str, Any]) -> tuple[bool, str]:
    if not sample.get("ok"):
        return False, str(sample.get("reason") or "nvidia-smi query failed")
    util = int(sample.get("utilization_gpu_pct", 999))
    used = int(sample.get("memory_used_mib", 999999))
    apps = sample.get("compute_apps") or []
    desktop_markers = (
        "/usr/bin/plasmashell",
        "/usr/bin/krunner",
        "/usr/bin/kwin",
        "Xorg",
        "wayland",
        # Resident principal-side bridge process on this workstation; steady
        # ~0.5 GiB, captured by the baseline VRAM sample like the desktop apps.
        "/.hermes/hermes-agent/",
    )
    non_desktop_apps = [
        app
        for app in apps
        if not any(marker in str(app.get("process_name", "")) for marker in desktop_markers)
    ]
    if non_desktop_apps:
        return False, f"non-desktop compute apps present: {non_desktop_apps}"
    if util > 10:
        return False, f"GPU utilization {util}% > 10%"
    if used > 4096:
        return False, f"GPU memory used {used} MiB > 4096 MiB idle threshold"
    return True, "idle"


def lock_held_by_wrapper_or_other() -> dict[str, Any]:
    # If this script was launched by scripts/run_gpu_lowprio.sh, that process has
    # exec'd into Python while preserving fd 9 and the flock. A separate flock
    # probe must fail. If it succeeds, this script is not protected by the repo
    # GPU mutex and should not start GPU work.
    proc = run_cmd(["flock", "-n", str(GPU_LOCK), "true"])
    return {
        "lock_path": str(GPU_LOCK),
        "probe_returncode": proc["returncode"],
        "held": proc["returncode"] != 0,
        "required_for_gpu_run": True,
    }


def branch_control_audit() -> dict[str, Any]:
    sw = text_check(
        ROOT / "src" / "gpuwrf" / "physics" / "rrtmg_sw.py",
        {
            "column_tiling_default_true": r"_SW_COLUMN_TILING\s*=\s*_env_bool\([^,]+,\s*True\)",
            "column_tile_default_16384": r"_SW_COLUMN_TILE_COLS\s*=\s*max\(0,\s*_env_int\([^,]+,\s*16384\)\)",
            "tiled_impl": r"def _shortwave_column_tiled_impl\(",
            "scan_tiles": r"lax\.scan\(body, init, jnp\.arange\(n_tiles",
            "solver_uses_tiled_impl": r"if not _SW_COLUMN_TILING.*?return _shortwave_impl.*?return _shortwave_column_tiled_impl",
        },
    )
    lw = text_check(
        ROOT / "src" / "gpuwrf" / "physics" / "rrtmg_lw.py",
        {
            "column_tiling_default_true": r"_LW_COLUMN_TILING\s*=\s*_env_bool\([^,]+,\s*True\)",
            "column_tile_default_16384": r"_LW_COLUMN_TILE_COLS\s*=\s*max\(0,\s*_env_int\([^,]+,\s*16384\)\)",
            "tiled_impl": r"def _longwave_column_tiled_impl\(",
            "scan_tiles": r"lax\.scan\(body, init, jnp\.arange\(n_tiles",
            "solver_uses_tiled_impl": r"if not _LW_COLUMN_TILING.*?return _longwave_impl.*?return _longwave_column_tiled_impl",
        },
    )
    cli = text_check(
        ROOT / "src" / "gpuwrf" / "cli.py",
        {
            "nested_allocator_reexec_helper": r"def _maybe_reexec_for_nested_allocator\(",
            "nested_only_gate": r"args\.max_dom is not None and int\(args\.max_dom\) > 1",
            "honors_operator_allocator": r"if os\.environ\.get\(\"XLA_PYTHON_CLIENT_ALLOCATOR\"\):\s*return",
            "platform_allocator_env": r"XLA_PYTHON_CLIENT_ALLOCATOR\"\]\s*=\s*\"platform\"",
            "reexec_guard": r"_GPUWRF_NESTED_ALLOC_REEXEC",
        },
    )
    nested = text_check(
        ROOT / "src" / "gpuwrf" / "integration" / "nested_pipeline.py",
        {
            "platform_allocator_setdefault": r"os\.environ\.setdefault\(\"XLA_PYTHON_CLIENT_ALLOCATOR\",\s*\"platform\"\)",
            "gwd_nested_default_on_note": r"GWD operational coupling is ON BY DEFAULT",
            "output_interval_segment_size": r"root_seg_steps\s*=\s*int\(output_cadence\[root\]\)",
            "segmented_loop": r"while start < root_steps:",
            "segment_block_until_ready": r"jax\.block_until_ready\(tuple\(state\.theta for state in result\.states\.values\(\)\)\)",
            "carry_own_steps_across_segments": r"carries\s*=\s*result\.carries.*?own_steps\s*=\s*dict\(result\.own_steps\)",
        },
    )
    domain_tree = text_check(
        ROOT / "src" / "gpuwrf" / "runtime" / "domain_tree.py",
        {
            "initial_own_steps_argument": r"initial_own_steps: dict\[str, int\] \| None = None",
            "global_clock_doc": r"GLOBAL step clock",
            "resumable_carries_argument": r"carries: dict\[str, Any\] \| None = None",
            "passes_initial_own_steps": r"initial_own_steps=initial_own_steps",
        },
    )
    audits = {
        "rrtmg_sw": sw,
        "rrtmg_lw": lw,
        "cli_nested_allocator": cli,
        "nested_pipeline_segmentation": nested,
        "domain_tree_resume": domain_tree,
    }
    return {
        "items": audits,
        "rrtmg_column_tiling_present": bool(sw["ok"] and lw["ok"]),
        "nested_allocator_controls_present": bool(cli["ok"] and nested["ok"] and domain_tree["ok"]),
        "ok": all(item["ok"] for item in audits.values()),
    }


def prior_proof_summary() -> dict[str, Any]:
    rrtmg_vram = read_json_if_exists(ROOT / "proofs" / "v013" / "rrtmg_column_tile_vram_suite.json")
    rrtmg_exact = read_json_if_exists(ROOT / "proofs" / "v013" / "rrtmg_column_tile.json")
    nested_oom = read_json_if_exists(ROOT / "proofs" / "v0120" / "nested_oom_fix.json")
    gwd = read_json_if_exists(ROOT / "proofs" / "v013" / "gwd_nested_24h_gate.json")
    twoway = read_json_if_exists(ROOT / "proofs" / "v013" / "twoway_vram.json")

    summary: dict[str, Any] = {
        "rrtmg_column_tile_vram_suite": rrtmg_vram,
        "rrtmg_column_tile_exact": rrtmg_exact,
        "nested_oom_fix": nested_oom,
        "gwd_nested_24h_gate": gwd,
        "twoway_vram": twoway,
    }
    if rrtmg_vram.get("ok"):
        rows = rrtmg_vram["payload"].get("rows", [])
        summary["rrtmg_vram_highlights"] = [
            {
                "kind": row.get("kind"),
                "column_mode": row.get("column_mode"),
                "result": row.get("result", "OK"),
                "peak_mib": row.get("peak_mib"),
                "error_tail": row.get("error_tail"),
            }
            for row in rows
        ]
    if nested_oom.get("ok"):
        peak = nested_oom["payload"].get("peak_vram_bounded_proof", {})
        summary["nested_allocator_highlights"] = {
            "after_fix_platform_allocator_reexec": peak.get("after_fix_platform_allocator_reexec"),
            "correctness_identical": peak.get("correctness_identical"),
        }
    return summary


class NvidiaSmiSampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = float(interval_s)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "NvidiaSmiSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.sample()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_s)

    def sample(self) -> None:
        payload = nvidia_smi_query()
        payload["timestamp_utc"] = dt.datetime.now(dt.UTC).isoformat()
        self.samples.append(payload)

    def summary(self, baseline_mib: int | None = None) -> dict[str, Any]:
        ok = [s for s in self.samples if s.get("ok")]
        used = [int(s.get("memory_used_mib", 0)) for s in ok]
        util = [int(s.get("utilization_gpu_pct", 0)) for s in ok]
        app_used = []
        for sample in ok:
            apps = sample.get("compute_apps") or []
            app_used.append(sum(int(app.get("used_memory_mib", 0)) for app in apps))
        peak_index = max(range(len(ok)), key=lambda i: used[i]) if ok else None
        peak_total = max(used) if used else None
        peak_compute = max(app_used) if app_used else None
        return {
            "sample_count": len(self.samples),
            "ok_sample_count": len(ok),
            "peak_total_vram_mib": peak_total,
            "peak_compute_apps_mib": peak_compute,
            "baseline_total_vram_mib": baseline_mib,
            "peak_increment_over_baseline_mib": (
                peak_total - baseline_mib
                if peak_total is not None and baseline_mib is not None
                else None
            ),
            "peak_utilization_gpu_pct": max(util) if util else None,
            "peak_sample": ok[peak_index] if peak_index is not None else None,
            "first_samples": self.samples[:5],
            "last_samples": self.samples[-5:],
        }


def required_nested_inputs(input_dir: Path, max_dom: int) -> dict[str, Any]:
    required = ["namelist.input", "wrfbdy_d01"] + [
        f"wrfinput_d{i:02d}" for i in range(1, int(max_dom) + 1)
    ]
    rows = []
    for name in required:
        path = input_dir / name
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return {
        "input_dir": str(input_dir),
        "required": rows,
        "ok": all(row["exists"] for row in rows),
    }


def parse_nested_payload(proof_dir: Path) -> dict[str, Any]:
    path = proof_dir / "nested_pipeline_run.json"
    if not path.is_file():
        return {"ok": False, "path": str(path), "reason": "missing nested_pipeline_run.json"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": str(path), "reason": f"{type(exc).__name__}: {exc}"}
    domains = payload.get("domains") or payload.get("per_domain") or {}
    return {
        "ok": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "verdict": payload.get("verdict"),
        "init_mode": payload.get("init_mode"),
        "all_finite": payload.get("all_domains_finite"),
        "all_outputs_present": payload.get("all_outputs_present"),
        "wrfout_files": payload.get("wrfout_files"),
        "domains": domains,
        "raw_keys": sorted(payload.keys()),
    }


def run_nested_gpu_smoke(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root or (
        DEFAULT_RUN_ROOT_PARENT / f"v014_exact_branch_memory_preflight_{timestamp}"
    )
    output_dir = run_root / "nested_1h_out"
    scratch_dir = run_root / "scratch"
    proof_dir = run_root / "proofs"
    log_stdout = run_root / "gpu_command.stdout.log"
    log_stderr = run_root / "gpu_command.stderr.log"
    run_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["JAX_ENABLE_X64"] = "true"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    env["GPUWRF_RRTMG_SW_COLUMN_TILING"] = "true"
    env["GPUWRF_RRTMG_LW_COLUMN_TILING"] = "true"
    env["GPUWRF_RRTMG_SW_COLUMN_TILE_COLS"] = str(args.tile_cols)
    env["GPUWRF_RRTMG_LW_COLUMN_TILE_COLS"] = str(args.tile_cols)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("OMP_NUM_THREADS", "4")

    cmd = [
        sys.executable,
        "-m",
        "gpuwrf.cli",
        "run",
        "--input-dir",
        str(args.nested_input),
        "--output-dir",
        str(output_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--proof-dir",
        str(proof_dir),
        "--max-dom",
        str(args.max_dom),
        "--hours",
        str(args.hours),
    ]
    if args.feedback:
        cmd.append("--feedback")

    baseline = nvidia_smi_query()
    baseline_mib = baseline.get("memory_used_mib") if baseline.get("ok") else None
    started = time.perf_counter()
    timed_out = False
    proc_rc: int | None = None
    stdout = ""
    stderr = ""
    with NvidiaSmiSampler(interval_s=args.sample_interval_s) as sampler:
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=float(args.timeout_s),
                check=False,
            )
            proc_rc = int(proc.returncode)
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = ensure_text(exc.stdout)
            stderr = ensure_text(exc.stderr)
    wall_s = time.perf_counter() - started

    log_stdout.write_text(stdout, encoding="utf-8")
    log_stderr.write_text(stderr, encoding="utf-8")
    payload = parse_nested_payload(proof_dir)
    output_files = sorted(str(p) for p in output_dir.glob("wrfout_d??_*"))
    oom_markers = [
        line
        for line in (stdout + "\n" + stderr).splitlines()
        if "RESOURCE_EXHAUSTED" in line
        or "Out of memory" in line
        or "CUDA_ERROR_OUT_OF_MEMORY" in line
    ]
    reexec_seen = "nested run -- re-exec with XLA_PYTHON_CLIENT_ALLOCATOR=platform" in stderr
    result_ok = (
        proc_rc == 0
        and not timed_out
        and payload.get("verdict") == "PIPELINE_GREEN"
        and payload.get("all_finite") is True
        and payload.get("all_outputs_present") is True
        and not oom_markers
    )
    return {
        "attempted": True,
        "case": "nested_1h",
        "run_root": str(run_root),
        "command": cmd,
        "outer_wrapper_required": "scripts/run_gpu_lowprio.sh --cores 0-23 --",
        "duration_s": round(wall_s, 3),
        "timeout_s": float(args.timeout_s),
        "timed_out": timed_out,
        "returncode": proc_rc,
        "output_dir": str(output_dir),
        "scratch_dir": str(scratch_dir),
        "proof_dir": str(proof_dir),
        "stdout_log": str(log_stdout),
        "stderr_log": str(log_stderr),
        "stderr_tail": stderr.splitlines()[-30:],
        "stdout_tail": stdout.splitlines()[-30:],
        "environment": {
            "PYTHONPATH": env.get("PYTHONPATH"),
            "JAX_ENABLE_X64": env.get("JAX_ENABLE_X64"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": env.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_ALLOCATOR_initial": env.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
            "GPUWRF_RRTMG_SW_COLUMN_TILING": env.get("GPUWRF_RRTMG_SW_COLUMN_TILING"),
            "GPUWRF_RRTMG_LW_COLUMN_TILING": env.get("GPUWRF_RRTMG_LW_COLUMN_TILING"),
            "GPUWRF_RRTMG_SW_COLUMN_TILE_COLS": env.get("GPUWRF_RRTMG_SW_COLUMN_TILE_COLS"),
            "GPUWRF_RRTMG_LW_COLUMN_TILE_COLS": env.get("GPUWRF_RRTMG_LW_COLUMN_TILE_COLS"),
            "GPUWRF_GWD_NESTED": env.get("GPUWRF_GWD_NESTED", "unset-default-on"),
        },
        "settings": {
            "max_dom": int(args.max_dom),
            "hours": int(args.hours),
            "feedback": bool(args.feedback),
            "gwd_nested": env.get("GPUWRF_GWD_NESTED", "unset-default-on"),
            "tile_cols": int(args.tile_cols),
        },
        "nvidia_smi_baseline": baseline,
        "nvidia_smi_peak": sampler.summary(
            int(baseline_mib) if baseline_mib is not None else None
        ),
        "allocator_evidence": {
            "reexec_line_seen": bool(reexec_seen),
            "platform_allocator_expected_after_reexec": True,
            "memory_stats_caveat": (
                "Nested path uses XLA platform allocator, so nvidia-smi total/process "
                "sampling is the allocator-agnostic peak source."
            ),
        },
        "nested_payload": payload,
        "output_count": len(output_files),
        "output_files": output_files,
        "oom_markers": oom_markers[-10:],
        "ok": bool(result_ok),
    }


def no_run_record(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attempted": False,
        "ok": False,
        "reason": reason,
        "planned_command": [
            "scripts/run_gpu_lowprio.sh",
            "--cores",
            "0-23",
            "--",
            "python",
            "proofs/v014/exact_branch_memory_preflight.py",
            "--run-gpu",
            "--nested-input",
            str(args.nested_input),
            "--max-dom",
            str(args.max_dom),
            "--hours",
            str(args.hours),
            "--timeout-s",
            str(args.timeout_s),
        ],
    }


def observed_timeout_attempt(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.observed_timeout_run_root is None:
        return None
    run_root = args.observed_timeout_run_root
    output_dir = run_root / "nested_1h_out"
    proof_dir = run_root / "proofs"
    output_files = sorted(str(p) for p in output_dir.glob("wrfout_d??_*"))
    files = sorted(str(p) for p in run_root.rglob("*") if p.is_file())
    return {
        "attempted": True,
        "source": "operator polling from previous bounded run in this turn",
        "run_root": str(run_root),
        "command": [
            "scripts/run_gpu_lowprio.sh",
            "--cores",
            "0-23",
            "--",
            "python",
            "proofs/v014/exact_branch_memory_preflight.py",
            "--run-gpu",
            "--timeout-s",
            str(args.observed_timeout_s),
        ],
        "duration_class": "timed_out_or_reached_cap",
        "duration_s": float(args.observed_timeout_s) if args.observed_timeout_s else None,
        "peak_total_vram_mib_observed": (
            float(args.observed_timeout_peak_total_vram_mib)
            if args.observed_timeout_peak_total_vram_mib is not None
            else None
        ),
        "baseline_total_vram_mib_observed": (
            float(args.observed_timeout_baseline_total_vram_mib)
            if args.observed_timeout_baseline_total_vram_mib is not None
            else None
        ),
        "peak_increment_over_baseline_mib_observed": (
            float(args.observed_timeout_peak_total_vram_mib)
            - float(args.observed_timeout_baseline_total_vram_mib)
            if args.observed_timeout_peak_total_vram_mib is not None
            and args.observed_timeout_baseline_total_vram_mib is not None
            else None
        ),
        "output_dir": str(output_dir),
        "proof_dir": str(proof_dir),
        "output_count": len(output_files),
        "output_files": output_files,
        "files_after_attempt": files,
        "no_oom_observed": True,
        "completed": False,
        "note": (
            "The first 600 s nested run stayed under the observed peak VRAM but "
            "did not complete and exposed a timeout-handler bug fixed in this script. "
            "Because no nested_pipeline_run.json or wrfout was produced, this is "
            "not accepted as a completed memory-fit proof."
        ),
    }


def build_markdown(record: dict[str, Any]) -> str:
    verdict = record.get("verdict", "UNKNOWN")
    gpu = record.get("gpu_run", {})
    git = record.get("git", {})
    peak = (gpu.get("nvidia_smi_peak") or {}) if isinstance(gpu, dict) else {}
    payload = (gpu.get("nested_payload") or {}) if isinstance(gpu, dict) else {}
    controls = record.get("branch_controls", {})
    lines = [
        "# v0.14 Exact-Branch Memory Preflight",
        "",
        f"- Verdict: `{verdict}`",
        f"- Branch: `{git.get('branch')}`",
        f"- HEAD: `{git.get('head')}`",
        f"- Dirty worktree: `{git.get('dirty')}`",
        "",
        "## Static Controls",
        "",
        f"- RRTMG column tiling present: `{controls.get('rrtmg_column_tiling_present')}`",
        f"- Nested allocator/segmentation controls present: `{controls.get('nested_allocator_controls_present')}`",
        "",
        "## GPU Run",
        "",
    ]
    if gpu.get("attempted"):
        lines.extend(
            [
                f"- Command: `{' '.join(gpu.get('command', []))}`",
                f"- Output path: `{gpu.get('output_dir')}`",
                f"- Duration: `{gpu.get('duration_s')}` s",
                f"- Return code: `{gpu.get('returncode')}`",
                f"- Nested payload verdict: `{payload.get('verdict')}`",
                f"- All finite: `{payload.get('all_finite')}`",
                f"- All outputs present: `{payload.get('all_outputs_present')}`",
                f"- Output count: `{gpu.get('output_count')}`",
                f"- Peak total VRAM: `{peak.get('peak_total_vram_mib')}` MiB",
                f"- Peak compute-app VRAM: `{peak.get('peak_compute_apps_mib')}` MiB",
                f"- Peak increment over baseline: `{peak.get('peak_increment_over_baseline_mib')}` MiB",
                f"- Allocator re-exec line seen: `{(gpu.get('allocator_evidence') or {}).get('reexec_line_seen')}`",
                f"- OOM markers: `{len(gpu.get('oom_markers') or [])}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Run attempted: `False`",
                f"- Reason: `{gpu.get('reason')}`",
                f"- Planned command: `{' '.join(gpu.get('planned_command', []))}`",
            ]
        )
    observed = record.get("observed_timeout_attempt")
    if observed:
        lines.extend(
            [
                "",
                "## Observed Timed-Out Attempt",
                "",
                f"- Command: `{' '.join(observed.get('command', []))}`",
                f"- Run root: `{observed.get('run_root')}`",
                f"- Duration class: `{observed.get('duration_class')}`",
                f"- Duration cap: `{observed.get('duration_s')}` s",
                f"- Peak total VRAM observed: `{observed.get('peak_total_vram_mib_observed')}` MiB",
                f"- Baseline total VRAM observed: `{observed.get('baseline_total_vram_mib_observed')}` MiB",
                f"- Peak increment observed: `{observed.get('peak_increment_over_baseline_mib_observed')}` MiB",
                f"- Output count: `{observed.get('output_count')}`",
                f"- Completed: `{observed.get('completed')}`",
                f"- No OOM observed: `{observed.get('no_oom_observed')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is a memory-fit preflight, not TOST, not a long validation, and not a skill/equivalence claim.",
            "- The nested run intentionally uses the platform allocator path; peak VRAM is from nvidia-smi sampling, not JAX memory_stats.",
            "- This is not a full transfer audit. Hourly wrfout preparation necessarily moves output payloads to host; no claim is made that every loop is transfer-free.",
            "- Feedback is recorded as a setting. If the next long validation enables feedback, rerun this preflight with `--feedback`.",
            "",
            "## Next",
            "",
            "Run V014-MEM-1: empirical memory map on the same exact branch, measuring MYNN BouLac, non-radiation column physics, post-physics merge, and moisture limiter liveness before any new memory rewrite.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-gpu", action="store_true", help="Run the bounded nested GPU smoke.")
    ap.add_argument("--nested-input", type=Path, default=DEFAULT_NESTED_INPUT)
    ap.add_argument("--max-dom", type=int, default=3)
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--tile-cols", type=int, default=16384)
    # The nested 1h memory smoke can spend >15 minutes in cold JIT/forecast on
    # this workstation; 600s caused false-red timeouts despite low VRAM.
    ap.add_argument("--timeout-s", type=float, default=2400.0)
    ap.add_argument("--sample-interval-s", type=float, default=0.5)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--observed-timeout-run-root", type=Path, default=None)
    ap.add_argument("--observed-timeout-s", type=float, default=None)
    ap.add_argument("--observed-timeout-peak-total-vram-mib", type=float, default=None)
    ap.add_argument("--observed-timeout-baseline-total-vram-mib", type=float, default=None)
    args = ap.parse_args()

    record: dict[str, Any] = {
        "proof": "v0.14 exact-branch memory preflight",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "intent": (
            "Confirm exact branch contains RRTMG leading-column tiling and nested "
            "allocator/segmentation controls, then run a short representative "
            "nested memory exercise only when GPU is idle and the wrapper lock is held."
        ),
        "git": git_snapshot(),
        "branch_controls": branch_control_audit(),
        "prior_proofs": prior_proof_summary(),
        "inputs": {
            "nested": required_nested_inputs(args.nested_input, args.max_dom),
        },
        "gpu_precheck": {
            "nvidia_smi": nvidia_smi_query(),
            "lock": lock_held_by_wrapper_or_other(),
        },
        "write_scope": [
            "proofs/v014/exact_branch_memory_preflight.py",
            "proofs/v014/exact_branch_memory_preflight.json",
            "proofs/v014/exact_branch_memory_preflight.md",
            ".agent/reviews/2026-06-08-v014-exact-branch-memory-preflight.md",
        ],
    }

    controls_ok = bool(record["branch_controls"].get("ok"))
    inputs_ok = bool(record["inputs"]["nested"].get("ok"))
    idle_ok, idle_reason = gpu_idle_ok(record["gpu_precheck"]["nvidia_smi"])
    lock_ok = bool(record["gpu_precheck"]["lock"].get("held"))
    if args.run_gpu:
        if not controls_ok:
            record["gpu_run"] = no_run_record("branch memory controls audit failed", args)
        elif not inputs_ok:
            record["gpu_run"] = no_run_record("required nested input files are missing", args)
        elif not lock_ok:
            record["gpu_run"] = no_run_record(
                "GPU mutex is not held; rerun through scripts/run_gpu_lowprio.sh",
                args,
            )
        elif not idle_ok:
            record["gpu_run"] = no_run_record(f"GPU not idle: {idle_reason}", args)
        else:
            record["gpu_run"] = run_nested_gpu_smoke(args)
    else:
        record["gpu_run"] = no_run_record(
            "audit-only mode; pass --run-gpu through scripts/run_gpu_lowprio.sh",
            args,
        )
    record["observed_timeout_attempt"] = observed_timeout_attempt(args)

    gpu = record["gpu_run"]
    if controls_ok and gpu.get("ok"):
        verdict = "PASS_SHORT_GPU_PREFLIGHT"
    elif controls_ok and not gpu.get("attempted"):
        verdict = "NO_RUN_PLAN"
    else:
        verdict = "FAIL_OR_INCONCLUSIVE"
    record["verdict"] = verdict
    record["caveats"] = [
        "This is not TOST or long validation.",
        "This is not a full transfer audit; expected hourly wrfout output preparation moves payloads to host.",
        "Nested platform allocator disables useful JAX memory_stats, so peak is nvidia-smi sampled.",
        "A later branch or changed feedback setting invalidates this exact-branch preflight.",
    ]
    record["next_memory_measurement_sprint"] = {
        "name": "V014-MEM-1 empirical memory map",
        "scope": [
            "MYNN BouLac liveness",
            "non-radiation column physics peak memory",
            "post-physics merge transients",
            "moisture limiter/advection scratch",
        ],
        "constraints": [
            "No semantic fixes in MEM-1",
            "Use exact branch after grid-parity decision",
            "Capture peak VRAM and transfer audit before performance claims",
        ],
    }

    write_json(args.out_json, record)
    args.out_md.write_text(build_markdown(record), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(args.out_json), "md": str(args.out_md)}, indent=2))
    return 0 if verdict in {"PASS_SHORT_GPU_PREFLIGHT", "NO_RUN_PLAN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
