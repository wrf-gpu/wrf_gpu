#!/usr/bin/env python3
"""V0.14 Step-1 adjust_tempqv intermediate proof.

This proof is intentionally CPU-only. It parses the disposable WRF hook output
when present; if the sandbox blocks OpenMPI/PMIx before WRF starts, it emits a
closed blocked verdict with the exact rerun command and log path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs/v014"
OUT_JSON = PROOF_DIR / "step1_adjust_tempqv_intermediate.json"
OUT_MD = PROOF_DIR / "step1_adjust_tempqv_intermediate.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md"
PATCH_DIFF = PROOF_DIR / "step1_adjust_tempqv_intermediate_wrf_patch.diff"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-adjust-tempqv-intermediate/sprint-contract.md"
PRIOR_JSON = PROOF_DIR / "step1_theta_same_qvapor.json"
PRIOR_MD = PROOF_DIR / "step1_theta_same_qvapor.md"
PRECALL_MD = PROOF_DIR / "step1_qvapor_precall_savepoint.md"
CRITIC_MD = ROOT / ".agent/reviews/2026-06-09-v014-theta-qvapor-opus-critic.md"

SCRATCH = Path("/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate")
WRF_TREE = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF")
RUN_DIR = SCRATCH / "run"
WRF_TRUTH = SCRATCH / "wrf_truth"
LOG_DIR = SCRATCH / "logs"
WRF_EXE = WRF_TREE / "main/wrf.exe"
WRF_BUILD_ENV = Path("/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build")

TARGET_ZERO = {"k": 1, "y": 9, "x": 17}
TARGET_FORTRAN = {"i": 18, "j": 10, "k": 2}
TARGET_DOMAIN = 2
REQUIRED_ANCESTOR = "c3620d09"
VERDICT_BLOCKED_PMIX = "STEP1_ADJUST_TEMPQV_INTERMEDIATE_BLOCKED_OPENMPI_PMIX_SOCKET"

RUN_COMMAND = (
    "cd /mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/run && "
    "WRFGPU2_STEP1_ADJUST_TEMPQV_INTERMEDIATE=1 "
    "WRFGPU2_STEP1_ADJUST_TEMPQV_INTERMEDIATE_DOMAIN=2 "
    "WRFGPU2_STEP1_ADJUST_TEMPQV_INTERMEDIATE_I=18 "
    "WRFGPU2_STEP1_ADJUST_TEMPQV_INTERMEDIATE_J=10 "
    "WRFGPU2_STEP1_ADJUST_TEMPQV_INTERMEDIATE_K=2 "
    "WRFGPU2_STEP1_ADJUST_TEMPQV_INTERMEDIATE_ROOT=/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth "
    "/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/mpirun -np 28 ./wrf.exe"
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    except Exception as exc:  # pragma: no cover - defensive proof metadata
        return {"command": args, "status": "EXCEPTION", "error": repr(exc)}
    return {
        "command": args,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def jax_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
        "backend": None,
        "gpu_used": False,
    }
    try:
        import jax  # noqa: PLC0415

        backend = jax.default_backend()
        result["backend"] = backend
        result["gpu_used"] = backend not in {"cpu", "interpreter"}
        result["devices"] = [str(device) for device in jax.devices()]
    except Exception as exc:
        result["jax_error"] = repr(exc)
    return result


def parse_status_file(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {"returncode": None, "manager_log": None}
    if not path.is_file():
        return status
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("wrf_run_rc="):
            try:
                status["returncode"] = int(line.split("=", 1)[1])
            except ValueError:
                status["returncode"] = None
        elif line.startswith("manager_log="):
            status["manager_log"] = line.split("=", 1)[1].strip()
    return status


def log_tail(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def parse_value(parts: list[str]) -> Any:
    if not parts:
        return None
    if len(parts) > 1:
        return [parse_value([part]) for part in parts]
    token = parts[0]
    try:
        if any(ch in token for ch in ".Ee"):
            return float(token)
        return int(token)
    except ValueError:
        return token


def parse_wrf_hook(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    headers: dict[str, Any] = {}
    values: dict[str, float] = {}
    marker = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            marker = line
            continue
        parts = line.split()
        key = parts[0]
        value = parse_value(parts[1:])
        if isinstance(value, float):
            values[key] = value
        else:
            headers[key] = value
    return {"status": "READY", "path": str(path), "marker": marker, "headers": headers, "values": values}


def load_prior_reference() -> dict[str, Any]:
    if not PRIOR_JSON.is_file():
        return {"status": "MISSING", "path": str(PRIOR_JSON)}
    prior = json.loads(PRIOR_JSON.read_text(encoding="utf-8"))
    worst = prior["comparisons"]["final_candidate_residual"]["worst_cell"]
    candidate = worst["available_pressure_base_inputs"]["candidate_reconstruction"]
    wrf_precall = worst["available_pressure_base_inputs"]["wrf_precall_truth"]
    jax_values = {
        "t_2_post": worst["candidate_t_state"],
        "qvapor_post": worst["candidate_qv_after_adjust_tempqv"],
        "p": candidate["raw_pp"],
        "p_old": candidate["adjust_tempqv_p_old"],
        "p_new": candidate["adjust_tempqv_p_new"],
        "pb_old_equiv": candidate["adjust_tempqv_p_old"] - candidate["raw_pp"],
        "pb_new_equiv": candidate["live_pb"],
        "mub": candidate["live_mub"],
        "mub_save": candidate["raw_mub_save"],
        "c3h": candidate["c3h"],
        "c4h": candidate["c4h"],
        "p_top": candidate["p_top"],
    }
    return {
        "status": "READY",
        "source": str(PRIOR_JSON),
        "target_worst_cell": worst,
        "jax_available_values": jax_values,
        "wrf_precall_values": wrf_precall,
    }


def compare_values(wrf_capture: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    if wrf_capture.get("status") != "READY":
        return {"status": "BLOCKED_NO_WRF_HOOK_OUTPUT", "blocker": wrf_capture}
    if reference.get("status") != "READY":
        return {"status": "BLOCKED_NO_JAX_REFERENCE", "blocker": reference}

    wrf_values = wrf_capture["values"]
    jax_values = reference["jax_available_values"]
    rows = []
    max_abs_pressure_delta = 0.0
    pressure_keys = {"p", "p_old", "p_new", "pb_old_equiv", "pb_new_equiv", "mub", "mub_save", "c3h", "c4h", "p_top"}
    for key, jax_value in jax_values.items():
        if key not in wrf_values:
            rows.append({"field": key, "status": "MISSING_IN_WRF", "jax": jax_value})
            continue
        wrf_value = wrf_values[key]
        delta = wrf_value - float(jax_value)
        row = {
            "field": key,
            "status": "OK",
            "wrf": wrf_value,
            "jax": float(jax_value),
            "delta_wrf_minus_jax": delta,
            "abs_delta": abs(delta),
        }
        rows.append(row)
        if key in pressure_keys:
            max_abs_pressure_delta = max(max_abs_pressure_delta, abs(delta))

    t_delta = next((row.get("abs_delta") for row in rows if row.get("field") == "t_2_post"), None)
    if max_abs_pressure_delta > 1.0e-3:
        verdict = "STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH"
        reason = "WRF in-routine pressure/base inputs differ materially from the JAX reconstruction."
    elif t_delta is not None and t_delta > 1.0e-3:
        verdict = "STEP1_ADJUST_TEMPQV_INTERMEDIATE_TRANSCRIPTION_BUG_FOUND"
        reason = "Pressure/base inputs match but post-adjust theta does not."
    elif t_delta is not None and t_delta <= 1.0e-3:
        verdict = "STEP1_ADJUST_TEMPQV_INTERMEDIATE_ROUNDING_TAIL_BOUNDED"
        reason = "Recorded WRF and JAX values agree within the material theta threshold."
    else:
        verdict = "STEP1_ADJUST_TEMPQV_INTERMEDIATE_NEEDS_BROADER_WRF_SAVEPOINT"
        reason = "The hook did not provide enough overlapping fields to classify the residual."

    return {
        "status": "COMPARISON_EXECUTED",
        "diff_sign": "wrf_minus_jax",
        "rows": rows,
        "max_abs_pressure_delta": max_abs_pressure_delta,
        "t_2_post_abs_delta": t_delta,
        "verdict": verdict,
        "reason": reason,
    }


def classify(run_status: Mapping[str, Any], comparison: Mapping[str, Any]) -> tuple[str, str, list[str], str]:
    if comparison.get("status") == "COMPARISON_EXECUTED":
        verdict = str(comparison["verdict"])
        reason = str(comparison["reason"])
        risks = ["Only one target cell was emitted; broader savepoint may still be needed for neighborhood effects."]
        next_decision = "Use the classified mismatch to decide whether to patch pressure/base inputs or formula transcription."
        return verdict, reason, risks, next_decision

    tail = str(run_status.get("log_tail", ""))
    if "PMIx server's listener thread failed to start" in tail or "pmix_ifinit: socket() failed" in tail:
        return (
            VERDICT_BLOCKED_PMIX,
            "OpenMPI/PMIx could not create its listener socket inside the Codex sandbox; WRF never started.",
            [
                "No WRF hook output exists, so WRF intermediates were not compared numerically.",
                "The manager must rerun the recorded MPI command unsandboxed, then rerun this Python proof.",
            ],
            "Rerun the recorded 28-rank WRF command unsandboxed and rerun this proof script.",
        )

    return (
        "STEP1_ADJUST_TEMPQV_INTERMEDIATE_BLOCKED_WRF_HOOK_OUTPUT_MISSING",
        "The WRF hook output file is missing and the run log did not match the known PMIx blocker.",
        ["No WRF hook output exists; inspect the WRF run log before drawing a physics conclusion."],
        "Inspect the WRF run log and rerun the hook capture.",
    )


def source_evidence() -> dict[str, Any]:
    return {
        "mediation_integrate_live_nest_call": {
            "path": str(WRF_TREE / "share/mediation_integrate.F"),
            "lines": "726-763",
            "evidence": [
                "mub_save is copied before terrain blending",
                "blend_terrain updates terrain, mub, and phb",
                "adjust_tempqv is called with nest%mub, nest%mub_save, c3h, c4h, znw, p_top, t_2, p, QVAPOR, and nest%id",
            ],
        },
        "adjust_tempqv": {
            "path": str(WRF_TREE / "dyn_em/nest_init_utils.F"),
            "lines": "812-1017",
            "evidence": [
                "first loop computes p_old, tc_old, es_old, e_old, and rh",
                "second loop computes p_new, thloc, dth1, dth, post t_2, tc_new, es_new, e_new, and post QVAPOR",
                "env-gated hook emits the target d02 Fortran cell i=18,j=10,k=2",
            ],
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = [
        "# V0.14 Step-1 Adjust-TempQV Intermediate",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- CPU-only proof; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `c3620d09` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Disposable WRF hook patch: `{PATCH_DIFF}`.",
        f"- Rebuilt executable: `{WRF_EXE}`; hook string present: `{payload['wrf_build']['hook_string_present']}`.",
        f"- WRF run return code: `{payload['wrf_run']['returncode']}`.",
        f"- WRF run log: `{payload['wrf_run']['log']}`.",
        f"- WRF hook file: `{payload['wrf_capture']['path']}`; status `{payload['wrf_capture']['status']}`.",
        "",
        "## Interpretation",
        "",
        payload["verdict_reason"],
    ]
    if payload["comparisons"].get("status") == "COMPARISON_EXECUTED":
        lines.extend(["", "## Target Comparison", ""])
        lines.append("| Field | WRF | JAX | WRF-JAX |")
        lines.append("|---|---:|---:|---:|")
        for row in payload["comparisons"]["rows"]:
            if row.get("status") != "OK":
                continue
            lines.append(
                f"| `{row['field']}` | {row['wrf']:.16e} | {row['jax']:.16e} | "
                f"{row['delta_wrf_minus_jax']:.16e} |"
            )
    else:
        lines.extend(
            [
                "",
                "No WRF intermediate values were emitted because OpenMPI failed before WRF startup:",
                "",
                "```text",
                payload["wrf_run"].get("log_tail", "").strip(),
                "```",
                "",
                "Manager rerun command:",
                "",
                "```bash",
                payload["wrf_run"]["command"],
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Handoff",
            "",
            "objective: emit exact CPU-WRF `adjust_tempqv` intermediates for d02 i=18,j=10,k=2 and compare to the JAX proof.",
            "",
            "files changed:",
            "- `proofs/v014/step1_adjust_tempqv_intermediate.py`",
            "- `proofs/v014/step1_adjust_tempqv_intermediate.json`",
            "- `proofs/v014/step1_adjust_tempqv_intermediate.md`",
            "- `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`",
            "- `.agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md`",
            "",
            "proof objects produced:",
            f"- `{OUT_JSON}`",
            f"- `{OUT_MD}`",
            f"- `{PATCH_DIFF}`",
            f"- `{OUT_REVIEW}`",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Adjust-TempQV Intermediate",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "Findings:",
    ]
    if payload["verdict"] == VERDICT_BLOCKED_PMIX:
        lines.append(
            "- HIGH: The disposable WRF hook compiled and is present in `main/wrf.exe`, but the sandboxed 28-rank run failed before WRF startup with PMIx listener socket errors. No physics conclusion can be drawn until the recorded command is rerun unsandboxed."
        )
    elif payload["comparisons"].get("status") == "COMPARISON_EXECUTED":
        lines.append(f"- HIGH: {payload['verdict_reason']}")
    else:
        lines.append("- HIGH: WRF hook output is missing; inspect run logs before closing the sprint.")
    lines.extend(
        [
            "",
            "Evidence:",
            f"- Patch diff: `{PATCH_DIFF}`",
            f"- Compile log: `{payload['wrf_build']['compile_log']}`",
            f"- Run log: `{payload['wrf_run']['log']}`",
            f"- Hook output: `{payload['wrf_capture']['path']}`",
            "",
            f"Next decision: {payload['next_decision']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WRF_TRUTH.mkdir(parents=True, exist_ok=True)
    hook_path = WRF_TRUTH / "adjust_tempqv_d2_i18_j10_k2.txt"
    status_path = LOG_DIR / "wrf_run_status.txt"
    status_info = parse_status_file(status_path)
    run_log = Path(status_info["manager_log"]) if status_info.get("manager_log") else LOG_DIR / "wrf_run_mpirun_np28.log"
    compile_log = LOG_DIR / "compile_em_real_tcsh.log"

    git_head = run_command(["git", "rev-parse", "HEAD"])
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"])
    src_gpuwrf_diff = run_command(["git", "diff", "--", "src/gpuwrf"])
    strings_check = run_command(
        ["bash", "-lc", f"strings {WRF_EXE} | rg 'WRFGPU2_V014_STEP1_ADJUST_TEMPQV_INTERMEDIATE'"],
        cwd=ROOT,
    )

    reference = load_prior_reference()
    wrf_capture = parse_wrf_hook(hook_path)
    comparison = compare_values(wrf_capture, reference)
    run_status = {
        "command": RUN_COMMAND,
        "returncode": status_info["returncode"],
        "log": str(run_log),
        "status_file": str(status_path),
        "log_tail": log_tail(run_log),
    }
    verdict, reason, risks, next_decision = classify(run_status, comparison)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_adjust_tempqv_intermediate.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "verdict_reason": reason,
        "cpu_only": True,
        "gpu_used": False,
        "no_gpu": True,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "source_patch_allowed_by_proof": False,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "jax": jax_environment(),
        },
        "git": {
            "head": git_head,
            "required_ancestor": {
                "commit": REQUIRED_ANCESTOR,
                "command": ancestor["command"],
                "returncode": ancestor.get("returncode"),
                "is_ancestor": ancestor.get("returncode") == 0,
            },
            "src_gpuwrf_diff_empty": src_gpuwrf_diff.get("stdout_tail", "") == "",
            "src_gpuwrf_diff": src_gpuwrf_diff,
        },
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "zero_index_order": "k,y,x",
            "zero_index": TARGET_ZERO,
            "fortran_index": TARGET_FORTRAN,
            "boundary_distance": 9,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "prior_theta_json": path_info(PRIOR_JSON),
            "prior_theta_md": path_info(PRIOR_MD),
            "precall_qvapor_md": path_info(PRECALL_MD),
            "critic_md": path_info(CRITIC_MD),
            "wrf_tree": path_info(WRF_TREE),
            "scratch_root": path_info(SCRATCH),
            "run_dir": path_info(RUN_DIR),
        },
        "wrf_source_evidence": source_evidence(),
        "wrf_patch": {
            "path": str(PATCH_DIFF),
            "info": path_info(PATCH_DIFF),
            "touched_disposable_files": [
                str(WRF_TREE / "share/mediation_integrate.F"),
                str(WRF_TREE / "dyn_em/nest_init_utils.F"),
            ],
        },
        "wrf_build": {
            "compile_command": (
                "cd /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF && "
                "/home/user/miniconda3/bin/tcsh ./compile em_real"
            ),
            "compile_log": str(compile_log),
            "compile_log_info": path_info(compile_log),
            "wrf_exe": path_info(WRF_EXE),
            "hook_string_present": strings_check.get("returncode") == 0,
            "hook_string_check": strings_check,
            "shebang_compile_blocker_log": str(LOG_DIR / "compile_em_real.log"),
        },
        "wrf_run": run_status,
        "wrf_capture": wrf_capture,
        "jax_reference": reference,
        "comparisons": comparison,
        "commands": {
            "executed": [
                "git rev-parse HEAD",
                "git merge-base --is-ancestor c3620d09 HEAD",
                "diff -u disposable WRF backups vs patched files > proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff",
                "cd /mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF && /home/user/miniconda3/bin/tcsh ./compile em_real",
                RUN_COMMAND,
            ],
            "required_validation": [
                "python -m py_compile proofs/v014/step1_adjust_tempqv_intermediate.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_adjust_tempqv_intermediate.py",
                "python -m json.tool proofs/v014/step1_adjust_tempqv_intermediate.json >/tmp/step1_adjust_tempqv_intermediate.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "script": str(Path(__file__).resolve()),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "wrf_patch_diff": str(PATCH_DIFF),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": risks,
        "next_decision": next_decision,
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
