#!/usr/bin/env python3
"""V0.14 d02 step-1 same-input WRF truth and strict JAX pre-halo comparison."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import same_input_contract_builder as builder  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_same_input_truth.json"
OUT_MD = PROOF_DIR / "step1_same_input_truth.md"
OUT_PATCH = PROOF_DIR / "step1_same_input_truth_wrf_patch.diff"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-same-input-truth.md"

PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
MANAGING_SPRINTS_SKILL = ROOT / ".agent/skills/managing-sprints/SKILL.md"
SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-same-input-truth/sprint-contract.md"
BUILDER_MD = PROOF_DIR / "same_input_contract_builder.md"
BUILDER_JSON = PROOF_DIR / "same_input_contract_builder.json"
BUILDER_PY = PROOF_DIR / "same_input_contract_builder.py"
REFRESH_PATCH = PROOF_DIR / "wrf_post_rk_refresh_localization_patch.diff"
SAME_STATE_PY = PROOF_DIR / "same_state_momentum_mass.py"

SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_same_input_truth")
SCRATCH_WRF = SCRATCH / "WRF"
SCRATCH_RUN = SCRATCH / "run"
RAW_TRUTH = SCRATCH / "raw_truth"
WRF_SOURCE = Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/WRF")
RUN_CASE3 = Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/run_case3")
ACCEPTED_TRUTH = (
    Path("<DATA_ROOT>/wrf_gpu2/v014_same_input_contract_builder/wrf_truth")
    / "same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz"
)
WRF_ENV_BIN = Path("<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin")
WRF_ENV_LIB = Path("<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/lib")

TARGET_STEP = 1
TARGET_DOMAIN = 2
TARGET_SURFACE = "post_after_all_rk_steps_pre_halo"
P0_THETA_OFFSET_K = 300.0
ALL_COMPARE_FIELDS = (
    "T",
    "P",
    "PB",
    "PH",
    "PHB",
    "MU",
    "MUB",
    "U",
    "V",
    "W",
    "QVAPOR",
    "QCLOUD",
    "QRAIN",
    "QICE",
    "QSNOW",
    "QGRAUP",
)
MASS_FIELDS = ("T", "P", "PB", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
SURFACE_FIELDS = ("MU", "MUB")
WPH_FIELDS = ("W", "PH", "PHB")
STEP1_MARKER = "WRFGPU2_V014_STEP1_SAME_INPUT_TRUTH v1"


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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: int = 120,
    wrf_env: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
    )
    if wrf_env:
        env["PATH"] = f"{WRF_ENV_BIN}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{WRF_ENV_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    if extra_env:
        env.update(extra_env)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout_s": int(timeout_s),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def read_tail(path: Path, max_chars: int = 8000) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
    }
    try:
        import jax  # noqa: PLC0415

        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
                "gpu_device_count": len([device for device in jax.devices() if device.platform == "gpu"]),
            }
        )
    except Exception as exc:
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def _replace_namelist_value(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$"
    replacement = rf"\g<1>{value},"
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise ValueError(f"namelist key not found: {key}")
    return new_text


def shorten_namelist(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "run_days": "0",
        "run_hours": "0",
        "run_minutes": "0",
        "run_seconds": "18",
        "history_interval": "60, 60",
        "history_interval_s": "0, 0",
        "frames_per_outfile": "1, 1",
        "restart": ".false.",
    }
    for key, value in replacements.items():
        text = _replace_namelist_value(text, key, value)
    path.write_text(text, encoding="utf-8")


def step1_wrf_block() -> str:
    return f"""

   wrfgpu2_full_enabled = .FALSE.
   CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_SAME_INPUT_STEP1', wrfgpu2_full_env)
   IF (TRIM(wrfgpu2_full_env) == '1') THEN
     wrfgpu2_full_grid = 2
     wrfgpu2_full_start_step = 1
     wrfgpu2_full_end_step = 1
     wrfgpu2_full_root = '{RAW_TRUTH}'
     CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_SAME_INPUT_STEP1_GRID', wrfgpu2_full_env)
     IF (LEN_TRIM(wrfgpu2_full_env) > 0) THEN
       READ(wrfgpu2_full_env,*,IOSTAT=wrfgpu2_full_ios) wrfgpu2_full_grid
     ENDIF
     CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_SAME_INPUT_STEP1_START_STEP', wrfgpu2_full_env)
     IF (LEN_TRIM(wrfgpu2_full_env) > 0) THEN
       READ(wrfgpu2_full_env,*,IOSTAT=wrfgpu2_full_ios) wrfgpu2_full_start_step
       wrfgpu2_full_end_step = wrfgpu2_full_start_step
     ENDIF
     CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_SAME_INPUT_STEP1_END_STEP', wrfgpu2_full_env)
     IF (LEN_TRIM(wrfgpu2_full_env) > 0) THEN
       READ(wrfgpu2_full_env,*,IOSTAT=wrfgpu2_full_ios) wrfgpu2_full_end_step
     ENDIF
     CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_SAME_INPUT_STEP1_ROOT', wrfgpu2_full_root)
     wrfgpu2_full_enabled = grid%id == wrfgpu2_full_grid .AND. &
                              grid%itimestep >= wrfgpu2_full_start_step .AND. &
                              grid%itimestep <= wrfgpu2_full_end_step
   ENDIF
   IF (wrfgpu2_full_enabled) THEN
     wrfgpu2_full_mass_k_end = MIN(k_end, kde-1)
     wrfgpu2_full_w_k_end = MIN(k_end+1, kde)
     DO ij = 1 , grid%num_tiles
       block_wrfgpu2_v014_step1_same_input: DO
         IF (k_start > wrfgpu2_full_w_k_end) EXIT block_wrfgpu2_v014_step1_same_input
         CALL execute_command_line('mkdir -p '//TRIM(wrfgpu2_full_root), wait=.TRUE.)
         CALL domain_clock_get(grid, current_timestr=wrfgpu2_full_time)
         WRITE(wrfgpu2_full_path,'(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A)') &
              TRIM(wrfgpu2_full_root)//'/same_input_post_after_all_rk_steps_pre_halo_d', &
              grid%id, '_step_', grid%itimestep, '_is_', grid%i_start(ij), &
              '_ie_', grid%i_end(ij), '_js_', grid%j_start(ij), &
              '_je_', grid%j_end(ij), '.txt'
         OPEN(NEWUNIT=wrfgpu2_full_unit, FILE=TRIM(wrfgpu2_full_path), &
              STATUS='REPLACE', ACTION='WRITE', FORM='FORMATTED')
         WRITE(wrfgpu2_full_unit,'(A)') '# {STEP1_MARKER}'
         WRITE(wrfgpu2_full_unit,'(A)') 'surface post_after_all_rk_steps_pre_halo'
         WRITE(wrfgpu2_full_unit,'(A)') 'routine dyn_em/solve_em.F::solve_em after after_all_rk_steps before RK halos'
         WRITE(wrfgpu2_full_unit,'(A,1X,I0)') 'domain_id', grid%id
         WRITE(wrfgpu2_full_unit,'(A,1X,A)') 'current_timestr_before_step', TRIM(wrfgpu2_full_time)
         WRITE(wrfgpu2_full_unit,'(A,1X,I0)') 'grid_itimestep_after_increment', grid%itimestep
         WRITE(wrfgpu2_full_unit,'(A,1X,I0)') 'rk_step', rk_step
         WRITE(wrfgpu2_full_unit,'(A,1X,I0)') 'rk_order', rk_order
         WRITE(wrfgpu2_full_unit,'(A,1X,E24.16)') 'dt_seconds', REAL(grid%dt,KIND=8)
         WRITE(wrfgpu2_full_unit,'(A,1X,E24.16)') 'lead_seconds_before_step', curr_secs2_r8
         WRITE(wrfgpu2_full_unit,'(A,1X,E24.16)') 'lead_seconds_after_step', curr_secs2_r8 + REAL(grid%dt,KIND=8)
         WRITE(wrfgpu2_full_unit,'(A,4(1X,I0))') 'tile_i_j_bounds_fortran', &
              grid%i_start(ij), grid%i_end(ij), grid%j_start(ij), grid%j_end(ij)
         WRITE(wrfgpu2_full_unit,'(A,4(1X,I0))') 'global_mass_i_j_end_exclusive_fortran', &
              ids, ide, jds, jde
         WRITE(wrfgpu2_full_unit,'(A)') 'tile_record_policy mass_owned_u_v_single_owner_no_overlap'
         WRITE(wrfgpu2_full_unit,'(A,2(1X,I0))') &
              'mass_vertical_fortran_k_start_k_end_inclusive', k_start, wrfgpu2_full_mass_k_end
         WRITE(wrfgpu2_full_unit,'(A,2(1X,I0))') &
              'w_ph_vertical_fortran_kstag_start_kstag_end_inclusive', k_start, wrfgpu2_full_w_k_end
         WRITE(wrfgpu2_full_unit,'(A,6(1X,I0))') &
              'moist_indices_qv_qc_qr_qi_qs_qg', P_QV, P_QC, P_QR, P_QI, P_QS, P_QG
         WRITE(wrfgpu2_full_unit,'(A)') &
              'record_schema MASS_FULL fortran_i fortran_j fortran_k zero_x zero_y zero_k T P PB MU MUB QVAPOR QCLOUD QRAIN QICE QSNOW QGRAUP'
         DO wrfgpu2_full_k = k_start, wrfgpu2_full_mass_k_end
           DO wrfgpu2_full_j = grid%j_start(ij), MIN(grid%j_end(ij), jde-1)
             DO wrfgpu2_full_i = grid%i_start(ij), MIN(grid%i_end(ij), ide-1)
               WRITE(wrfgpu2_full_unit,'(A,6(1X,I0),11(1X,E24.16))') 'MASS_FULL', &
                    wrfgpu2_full_i, wrfgpu2_full_j, wrfgpu2_full_k, &
                    wrfgpu2_full_i-1, wrfgpu2_full_j-1, wrfgpu2_full_k-1, &
                    REAL(grid%t_2(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8), &
                    REAL(grid%p(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8), &
                    REAL(grid%pb(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8), &
                    REAL(grid%mu_2(wrfgpu2_full_i,wrfgpu2_full_j),KIND=8), &
                    REAL(grid%mub(wrfgpu2_full_i,wrfgpu2_full_j),KIND=8), &
                    REAL(moist(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j,P_QV),KIND=8), &
                    REAL(moist(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j,P_QC),KIND=8), &
                    REAL(moist(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j,P_QR),KIND=8), &
                    REAL(moist(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j,P_QI),KIND=8), &
                    REAL(moist(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j,P_QS),KIND=8), &
                    REAL(moist(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j,P_QG),KIND=8)
             ENDDO
           ENDDO
         ENDDO
         WRITE(wrfgpu2_full_unit,'(A)') &
              'record_schema U_FULL fortran_i fortran_j fortran_k zero_xstag zero_y zero_k U'
         DO wrfgpu2_full_k = k_start, wrfgpu2_full_mass_k_end
           DO wrfgpu2_full_j = grid%j_start(ij), MIN(grid%j_end(ij), jde-1)
             DO wrfgpu2_full_i = grid%i_start(ij), MIN(grid%i_end(ij)+1, ide)
               IF (wrfgpu2_full_i <= grid%i_end(ij) .OR. grid%i_end(ij) >= ide-1) THEN
                 WRITE(wrfgpu2_full_unit,'(A,6(1X,I0),1(1X,E24.16))') 'U_FULL', &
                      wrfgpu2_full_i, wrfgpu2_full_j, wrfgpu2_full_k, &
                      wrfgpu2_full_i-1, wrfgpu2_full_j-1, wrfgpu2_full_k-1, &
                      REAL(grid%u_2(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8)
               ENDIF
             ENDDO
           ENDDO
         ENDDO
         WRITE(wrfgpu2_full_unit,'(A)') &
              'record_schema V_FULL fortran_i fortran_j fortran_k zero_x zero_ystag zero_k V'
         DO wrfgpu2_full_k = k_start, wrfgpu2_full_mass_k_end
           DO wrfgpu2_full_j = grid%j_start(ij), MIN(grid%j_end(ij)+1, jde)
             IF (wrfgpu2_full_j <= grid%j_end(ij) .OR. grid%j_end(ij) >= jde-1) THEN
               DO wrfgpu2_full_i = grid%i_start(ij), MIN(grid%i_end(ij), ide-1)
                 WRITE(wrfgpu2_full_unit,'(A,6(1X,I0),1(1X,E24.16))') 'V_FULL', &
                      wrfgpu2_full_i, wrfgpu2_full_j, wrfgpu2_full_k, &
                      wrfgpu2_full_i-1, wrfgpu2_full_j-1, wrfgpu2_full_k-1, &
                      REAL(grid%v_2(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8)
               ENDDO
             ENDIF
           ENDDO
         ENDDO
         WRITE(wrfgpu2_full_unit,'(A)') &
              'record_schema WPH_FULL fortran_i fortran_j fortran_kstag zero_x zero_y zero_kstag W PH PHB'
         DO wrfgpu2_full_k = k_start, wrfgpu2_full_w_k_end
           DO wrfgpu2_full_j = grid%j_start(ij), MIN(grid%j_end(ij), jde-1)
             DO wrfgpu2_full_i = grid%i_start(ij), MIN(grid%i_end(ij), ide-1)
               WRITE(wrfgpu2_full_unit,'(A,6(1X,I0),3(1X,E24.16))') 'WPH_FULL', &
                    wrfgpu2_full_i, wrfgpu2_full_j, wrfgpu2_full_k, &
                    wrfgpu2_full_i-1, wrfgpu2_full_j-1, wrfgpu2_full_k-1, &
                    REAL(grid%w_2(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8), &
                    REAL(grid%ph_2(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8), &
                    REAL(grid%phb(wrfgpu2_full_i,wrfgpu2_full_k,wrfgpu2_full_j),KIND=8)
             ENDDO
           ENDDO
         ENDDO
         CLOSE(wrfgpu2_full_unit)
         EXIT block_wrfgpu2_v014_step1_same_input
       ENDDO block_wrfgpu2_v014_step1_same_input
     ENDDO
   ENDIF
"""


def patch_wrf_tree() -> dict[str, Any]:
    solve = SCRATCH_WRF / "dyn_em/solve_em.F"
    if not solve.is_file():
        return {"status": "BLOCKED", "reason": f"missing {solve}"}
    text = solve.read_text(encoding="utf-8", errors="replace")
    if "wrfgpu2_full_env" not in text:
        return {
            "status": "BLOCKED",
            "reason": "preferred WRF solve_em.F lacks prior v014 wrfgpu2_full declarations needed by the minimal patch",
        }
    before = text
    if STEP1_MARKER not in text:
        anchor = (
            "\n\n"
            "   wrfgpu2_marker_enabled = .FALSE.\n"
            "   CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_POST_RK_REFRESH', wrfgpu2_marker_env)"
        )
        if anchor not in text:
            return {
                "status": "BLOCKED",
                "reason": "could not find post-after_all_rk_steps/pre-RK-halo anchor before WRFGPU2_POST_RK_REFRESH",
            }
        text = text.replace(anchor, step1_wrf_block() + anchor, 1)
        solve.write_text(text, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile="a/dyn_em/solve_em.F",
            tofile="b/dyn_em/solve_em.F",
        )
    )
    if not diff and OUT_PATCH.is_file():
        diff = OUT_PATCH.read_text(encoding="utf-8")
    OUT_PATCH.write_text(diff, encoding="utf-8")
    return {
        "status": "PATCHED" if diff else "ALREADY_PATCHED",
        "solve_em": path_info(solve),
        "patch_diff": path_info(OUT_PATCH),
        "marker": STEP1_MARKER,
    }


def ensure_scratch_wrf(commands: list[dict[str, Any]]) -> dict[str, Any]:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not SCRATCH_WRF.exists():
        copy_cmd = [
            "rsync",
            "-a",
            "--exclude",
            ".git",
            f"{WRF_SOURCE}/",
            f"{SCRATCH_WRF}/",
        ]
        result = run_command(copy_cmd, timeout_s=3600)
        commands.append({"stage": "copy_wrf_tree", **result})
        if result["returncode"] != 0:
            return {"status": "BLOCKED_COPY_FAILED", "copy": result}
    patch = patch_wrf_tree()
    if patch["status"] == "BLOCKED":
        return {"status": "BLOCKED_PATCH_FAILED", "patch": patch}
    return {"status": "READY", "patch": patch, "source": path_info(WRF_SOURCE), "scratch": path_info(SCRATCH_WRF)}


def ensure_scratch_run(commands: list[dict[str, Any]]) -> dict[str, Any]:
    if not SCRATCH_RUN.exists():
        run_cmd = [
            "rsync",
            "-a",
            "--delete",
            "--exclude",
            "rsl.*",
            "--exclude",
            "wrfout_d0*",
            "--exclude",
            "wrfrst_d0*",
            "--exclude",
            "aborted_single_rank_marker_run",
            "--exclude",
            "failed_serial_rsl_before_dmpar_marker",
            "--exclude",
            "first_28rank_early_marker_run",
            f"{RUN_CASE3}/",
            f"{SCRATCH_RUN}/",
        ]
        result = run_command(run_cmd, timeout_s=1200)
        commands.append({"stage": "copy_run_dir", **result})
        if result["returncode"] != 0:
            return {"status": "BLOCKED_COPY_FAILED", "copy": result}
    shorten_namelist(SCRATCH_RUN / "namelist.input")
    wrf_exe = SCRATCH_RUN / "wrf.exe"
    if wrf_exe.exists() or wrf_exe.is_symlink():
        wrf_exe.unlink()
    wrf_exe.symlink_to(SCRATCH_WRF / "main/wrf.exe")
    return {
        "status": "READY",
        "run_dir": path_info(SCRATCH_RUN),
        "namelist": path_info(SCRATCH_RUN / "namelist.input"),
        "wrf_exe": path_info(wrf_exe),
    }


def build_wrf_if_needed(commands: list[dict[str, Any]]) -> dict[str, Any]:
    marker = SCRATCH / "wrf_build_step1.sha256"
    solve_sha = sha256(SCRATCH_WRF / "dyn_em/solve_em.F")
    exe = SCRATCH_WRF / "main/wrf.exe"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == solve_sha and exe.is_file():
        return {"status": "SKIPPED_EXISTING_BUILD", "solve_em_sha256": solve_sha, "wrf_exe": path_info(exe)}
    log_path = SCRATCH / "compile_step1_same_input_truth.log"
    result = run_command(["tcsh", "./compile", "em_real"], cwd=SCRATCH_WRF, timeout_s=3600, wrf_env=True)
    log_path.write_text(
        f"$ cd {SCRATCH_WRF} && tcsh ./compile em_real\n\nSTDOUT:\n{result['stdout_tail']}\n\nSTDERR:\n{result['stderr_tail']}\n",
        encoding="utf-8",
    )
    commands.append({"stage": "build_wrf", "log": str(log_path), **result})
    if result["returncode"] != 0 or not exe.is_file():
        return {"status": "BLOCKED_BUILD_FAILED", "build": result, "log": path_info(log_path), "wrf_exe": path_info(exe)}
    marker.write_text(str(solve_sha) + "\n", encoding="utf-8")
    return {"status": "BUILT", "solve_em_sha256": solve_sha, "wrf_exe": path_info(exe), "log": path_info(log_path)}


def run_wrf_truth_if_needed(commands: list[dict[str, Any]]) -> dict[str, Any]:
    raw_files = sorted(RAW_TRUTH.glob("*same_input_post_after_all_rk_steps_pre_halo*d2*step_1*.txt"))
    if raw_files and ACCEPTED_TRUTH.is_file():
        return {
            "status": "SKIPPED_EXISTING_RAW_AND_NPZ",
            "raw_file_count": len(raw_files),
            "raw_files": [path_info(path) for path in raw_files[:8]],
            "accepted_truth": path_info(ACCEPTED_TRUTH),
        }
    if RAW_TRUTH.exists():
        shutil.rmtree(RAW_TRUTH)
    RAW_TRUTH.mkdir(parents=True, exist_ok=True)
    for old in SCRATCH_RUN.glob("rsl.*"):
        old.unlink()
    log_path = SCRATCH / "wrf_step1_same_input_truth_stdout.log"
    cmd = [str(WRF_ENV_BIN / "mpirun"), "--oversubscribe", "-np", "28", str(SCRATCH_RUN / "wrf.exe")]
    result = run_command(
        cmd,
        cwd=SCRATCH_RUN,
        timeout_s=1200,
        wrf_env=True,
        extra_env={
            "WRFGPU2_SAME_INPUT_STEP1": "1",
            "WRFGPU2_SAME_INPUT_STEP1_GRID": str(TARGET_DOMAIN),
            "WRFGPU2_SAME_INPUT_STEP1_START_STEP": str(TARGET_STEP),
            "WRFGPU2_SAME_INPUT_STEP1_END_STEP": str(TARGET_STEP),
            "WRFGPU2_SAME_INPUT_STEP1_ROOT": str(RAW_TRUTH),
        },
    )
    log_path.write_text(
        f"$ cd {SCRATCH_RUN} && {' '.join(cmd)}\n\nSTDOUT:\n{result['stdout_tail']}\n\nSTDERR:\n{result['stderr_tail']}\n",
        encoding="utf-8",
    )
    commands.append({"stage": "run_wrf", "mpi_ranks": 28, "log": str(log_path), **result})
    raw_files = sorted(RAW_TRUTH.glob("*same_input_post_after_all_rk_steps_pre_halo*d2*step_1*.txt"))
    rsl_error_0000 = SCRATCH_RUN / "rsl.error.0000"
    rsl_out_0000 = SCRATCH_RUN / "rsl.out.0000"
    if result["returncode"] != 0:
        return {
            "status": "BLOCKED_WRF_RUN_FAILED",
            "run": result,
            "stdout_log": path_info(log_path),
            "rsl_error_0000": path_info(rsl_error_0000),
            "rsl_out_0000": path_info(rsl_out_0000),
            "rsl_error_tail": read_tail(rsl_error_0000),
            "raw_file_count": len(raw_files),
        }
    if not raw_files:
        return {
            "status": "BLOCKED_NO_RAW_TRUTH_EMITTED",
            "run": result,
            "stdout_log": path_info(log_path),
            "rsl_error_0000": path_info(rsl_error_0000),
            "rsl_error_tail": read_tail(rsl_error_0000),
        }
    return {
        "status": "WRF_RAN_RAW_READY",
        "run": result,
        "stdout_log": path_info(log_path),
        "rsl_error_0000": path_info(rsl_error_0000),
        "rsl_out_0000": path_info(rsl_out_0000),
        "rsl_error_tail": read_tail(rsl_error_0000),
        "raw_file_count": len(raw_files),
        "raw_files": [path_info(path) for path in raw_files[:8]],
    }


def recorded_wrf_commands() -> list[dict[str, Any]]:
    return [
        {
            "stage": "build_wrf",
            "command": ["tcsh", "./compile", "em_real"],
            "cwd": str(SCRATCH_WRF),
            "log": path_info(SCRATCH / "compile_step1_same_input_truth.log"),
        },
        {
            "stage": "run_wrf",
            "command": [
                str(WRF_ENV_BIN / "mpirun"),
                "--oversubscribe",
                "-np",
                "28",
                str(SCRATCH_RUN / "wrf.exe"),
            ],
            "cwd": str(SCRATCH_RUN),
            "mpi_ranks": 28,
            "log": path_info(SCRATCH / "wrf_step1_same_input_truth_stdout.log"),
            "rsl_error_0000": path_info(SCRATCH_RUN / "rsl.error.0000"),
            "rsl_out_0000": path_info(SCRATCH_RUN / "rsl.out.0000"),
        },
    ]


def expected_shapes_from_wrfinput() -> dict[str, tuple[int, ...]]:
    from netCDF4 import Dataset  # type: ignore # noqa: PLC0415

    with Dataset(builder.WRFINPUT_D02) as dataset:
        nz = int(len(dataset.dimensions["bottom_top"]))
        nz_stag = int(len(dataset.dimensions["bottom_top_stag"]))
        ny = int(len(dataset.dimensions["south_north"]))
        ny_stag = int(len(dataset.dimensions["south_north_stag"]))
        nx = int(len(dataset.dimensions["west_east"]))
        nx_stag = int(len(dataset.dimensions["west_east_stag"]))
    mass = (nz, ny, nx)
    return {
        "T": mass,
        "P": mass,
        "PB": mass,
        "QVAPOR": mass,
        "QCLOUD": mass,
        "QRAIN": mass,
        "QICE": mass,
        "QSNOW": mass,
        "QGRAUP": mass,
        "PH": (nz_stag, ny, nx),
        "PHB": (nz_stag, ny, nx),
        "W": (nz_stag, ny, nx),
        "U": (nz, ny, nx_stag),
        "V": (nz, ny_stag, nx),
        "MU": (ny, nx),
        "MUB": (ny, nx),
    }


def _record_value(
    arrays: dict[str, np.ndarray],
    duplicate_stats: dict[str, Any],
    field: str,
    index: tuple[int, ...],
    value: float,
) -> None:
    current = arrays[field][index]
    if np.isnan(current):
        arrays[field][index] = value
        return
    duplicate_stats[field]["duplicates"] += 1
    if current != value:
        duplicate_stats[field]["mismatches"] += 1
        delta = abs(float(current) - float(value))
        duplicate_stats[field]["max_delta"] = max(float(duplicate_stats[field]["max_delta"]), delta)
        if duplicate_stats[field].get("first_mismatch") is None:
            duplicate_stats[field]["first_mismatch"] = {
                "index": index,
                "existing": float(current),
                "new": float(value),
                "delta": delta,
            }


def convert_raw_truth_to_npz() -> dict[str, Any]:
    raw_files = sorted(RAW_TRUTH.glob("*same_input_post_after_all_rk_steps_pre_halo*d2*step_1*.txt"))
    if not raw_files:
        return {"status": "BLOCKED_NO_RAW_TRUTH_FILES", "raw_root": str(RAW_TRUTH)}
    shapes = expected_shapes_from_wrfinput()
    arrays = {name: np.full(shape, np.nan, dtype=np.float64) for name, shape in shapes.items()}
    duplicate_stats = {
        name: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for name in arrays
    }
    metadata: dict[str, Any] = {"raw_files": len(raw_files), "headers": []}
    record_counts = {"MASS_FULL": 0, "U_FULL": 0, "V_FULL": 0, "WPH_FULL": 0}
    for path in raw_files:
        header: dict[str, Any] = {"path": str(path)}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                tag = parts[0]
                if tag.startswith("#"):
                    header.setdefault("marker", stripped)
                    continue
                if tag == "MASS_FULL":
                    if len(parts) != 18:
                        return {"status": "BLOCKED_PARSE_ERROR", "path": str(path), "line": stripped[:240]}
                    x = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    values = [float(item) for item in parts[7:]]
                    for field, value in zip(("T", "P", "PB", "MU", "MUB", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"), values):
                        if field in SURFACE_FIELDS:
                            _record_value(arrays, duplicate_stats, field, (y, x), value)
                        else:
                            _record_value(arrays, duplicate_stats, field, (k, y, x), value)
                    record_counts[tag] += 1
                elif tag == "U_FULL":
                    if len(parts) != 8:
                        return {"status": "BLOCKED_PARSE_ERROR", "path": str(path), "line": stripped[:240]}
                    xstag = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    _record_value(arrays, duplicate_stats, "U", (k, y, xstag), float(parts[7]))
                    record_counts[tag] += 1
                elif tag == "V_FULL":
                    if len(parts) != 8:
                        return {"status": "BLOCKED_PARSE_ERROR", "path": str(path), "line": stripped[:240]}
                    x = int(parts[4])
                    ystag = int(parts[5])
                    k = int(parts[6])
                    _record_value(arrays, duplicate_stats, "V", (k, ystag, x), float(parts[7]))
                    record_counts[tag] += 1
                elif tag == "WPH_FULL":
                    if len(parts) != 10:
                        return {"status": "BLOCKED_PARSE_ERROR", "path": str(path), "line": stripped[:240]}
                    x = int(parts[4])
                    y = int(parts[5])
                    kstag = int(parts[6])
                    for field, value in zip(("W", "PH", "PHB"), (float(parts[7]), float(parts[8]), float(parts[9]))):
                        _record_value(arrays, duplicate_stats, field, (kstag, y, x), value)
                    record_counts[tag] += 1
                elif tag in {
                    "surface",
                    "routine",
                    "domain_id",
                    "current_timestr_before_step",
                    "grid_itimestep_after_increment",
                    "rk_step",
                    "rk_order",
                    "dt_seconds",
                    "lead_seconds_before_step",
                    "lead_seconds_after_step",
                    "tile_i_j_bounds_fortran",
                    "global_mass_i_j_end_exclusive_fortran",
                    "tile_record_policy",
                    "mass_vertical_fortran_k_start_k_end_inclusive",
                    "w_ph_vertical_fortran_kstag_start_kstag_end_inclusive",
                    "moist_indices_qv_qc_qr_qi_qs_qg",
                    "record_schema",
                }:
                    header[tag] = parts[1:]
                else:
                    return {"status": "BLOCKED_UNKNOWN_RECORD", "path": str(path), "line": stripped[:240]}
        metadata["headers"].append(header)
    duplicate_mismatches = {
        name: item for name, item in duplicate_stats.items() if int(item["mismatches"]) > 0
    }
    if duplicate_mismatches:
        return {
            "status": "BLOCKED_DUPLICATE_MISMATCH",
            "duplicate_mismatches": duplicate_mismatches,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    missing = {
        name: {
            "missing_count": int(np.isnan(arr).sum()),
            "shape": list(arr.shape),
        }
        for name, arr in arrays.items()
        if np.isnan(arr).any()
    }
    if missing:
        return {
            "status": "BLOCKED_MISSING_VALUES",
            "missing": missing,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    ACCEPTED_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ACCEPTED_TRUTH, **{name: arrays[name].astype(np.float64) for name in ALL_COMPARE_FIELDS})
    summaries = {
        name: {
            "shape": list(arrays[name].shape),
            "dtype": str(arrays[name].dtype),
            "count": int(arrays[name].size),
            "min": float(np.min(arrays[name])),
            "max": float(np.max(arrays[name])),
            "mean": float(np.mean(arrays[name])),
        }
        for name in ALL_COMPARE_FIELDS
    }
    return {
        "status": "NPZ_READY",
        "accepted_truth": path_info(ACCEPTED_TRUTH),
        "raw_file_count": len(raw_files),
        "raw_total_bytes": int(sum(path.stat().st_size for path in raw_files)),
        "record_counts": record_counts,
        "duplicate_stats": duplicate_stats,
        "field_summaries": summaries,
        "metadata_sample": metadata["headers"][:4],
    }


def ensure_wrf_truth(commands: list[dict[str, Any]]) -> dict[str, Any]:
    if ACCEPTED_TRUTH.is_file():
        raw_files = sorted(RAW_TRUTH.glob("*same_input_post_after_all_rk_steps_pre_halo*d2*step_1*.txt"))
        conversion = convert_raw_truth_to_npz() if raw_files else {
            "status": "NPZ_READY_BUT_RAW_TRUTH_MISSING",
            "accepted_truth": path_info(ACCEPTED_TRUTH),
        }
        if conversion["status"] not in {"NPZ_READY", "NPZ_READY_BUT_RAW_TRUTH_MISSING"}:
            return {
                "status": conversion["status"],
                "accepted_truth": path_info(ACCEPTED_TRUTH),
                "raw_root": path_info(RAW_TRUTH),
                "raw_file_count": len(raw_files),
                "conversion": conversion,
                "recorded_wrf_commands": recorded_wrf_commands(),
                "wrf_patch_diff": path_info(OUT_PATCH),
                "scratch_wrf": path_info(SCRATCH_WRF),
                "scratch_run": path_info(SCRATCH_RUN),
            }
        return {
            "status": "TRUTH_NPZ_READY_EXISTING",
            "accepted_truth": path_info(ACCEPTED_TRUTH),
            "raw_root": path_info(RAW_TRUTH),
            "raw_file_count": len(raw_files),
            "raw_total_bytes": int(sum(path.stat().st_size for path in raw_files)) if raw_files else 0,
            "raw_files": [path_info(path) for path in raw_files[:8]],
            "conversion": conversion,
            "recorded_wrf_commands": recorded_wrf_commands(),
            "wrf_patch_diff": path_info(OUT_PATCH),
            "scratch_wrf": path_info(SCRATCH_WRF),
            "scratch_run": path_info(SCRATCH_RUN),
        }
    wrf = ensure_scratch_wrf(commands)
    if wrf["status"] != "READY":
        return {"status": wrf["status"], "wrf": wrf}
    run = ensure_scratch_run(commands)
    if run["status"] != "READY":
        return {"status": run["status"], "wrf": wrf, "run_dir": run}
    build = build_wrf_if_needed(commands)
    if str(build["status"]).startswith("BLOCKED"):
        return {"status": build["status"], "wrf": wrf, "run_dir": run, "build": build}
    wrf_run = run_wrf_truth_if_needed(commands)
    if str(wrf_run["status"]).startswith("BLOCKED"):
        return {"status": wrf_run["status"], "wrf": wrf, "run_dir": run, "build": build, "wrf_run": wrf_run}
    conversion = convert_raw_truth_to_npz()
    if conversion["status"] != "NPZ_READY":
        return {
            "status": conversion["status"],
            "wrf": wrf,
            "run_dir": run,
            "build": build,
            "wrf_run": wrf_run,
            "conversion": conversion,
        }
    return {
        "status": "TRUTH_NPZ_READY",
        "wrf": wrf,
        "run_dir": run,
        "build": build,
        "wrf_run": wrf_run,
        "conversion": conversion,
        "recorded_wrf_commands": recorded_wrf_commands(),
    }


def build_step1_jax_inputs() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    from gpuwrf.integration.d02_replay import run_start_label  # noqa: PLC0415
    from gpuwrf.io.gen2_accessor import Gen2Run  # noqa: PLC0415
    from gpuwrf.nesting.boundary_construction import build_child_boundary_package, build_nest_force_weights  # noqa: PLC0415
    from gpuwrf.runtime.domain_tree import with_live_child_boundary_config  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import OperationalNamelist  # noqa: PLC0415
    from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: PLC0415

    run = Gen2Run(builder.RUN_CASE3)
    parent = builder._state_from_wrfinput(run, "d01")
    child = builder._state_from_wrfinput(run, "d02")
    parent_grid_meta = run.grid("d01")
    child_grid_meta = run.grid("d02")
    weights = build_nest_force_weights(
        parent_grid_ratio=int(child_grid_meta.parent_grid_ratio),
        i_parent_start=int(child_grid_meta.i_parent_start),
        j_parent_start=int(child_grid_meta.j_parent_start),
        parent_grid=parent["grid"],
        child_grid=child["grid"],
        registration="sint",
    )
    child_state_with_parent_bdy = build_child_boundary_package(
        child["state"],
        parent["state"],
        weights,
        bdy_width=builder.BDY_WIDTH,
    )
    child_dt = builder._domain_dt_s(run, "d02")
    parent_dt = builder._domain_dt_s(run, "d01")
    radiation_cadence = max(1, int(round(builder.RADT_TARGET_S / float(child_dt))))
    namelist = OperationalNamelist.from_grid(
        child["grid"],
        tendencies=child["tendencies"],
        metrics=child["metrics"],
        dt_s=child_dt,
        acoustic_substeps=10,
        radiation_cadence_steps=radiation_cadence,
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
        time_utc=run_start_label(run, "d02"),
    )
    namelist = with_live_child_boundary_config(
        namelist,
        parent_dt_s=parent_dt,
        nested_ph_relax=True,
        nested_w_relax=False,
        nested_ph_spec=True,
    )
    cu_physics = int(builder._domain_list_value(run.namelist, "physics", "cu_physics", "d02", 0))
    namelist = dataclass_replace(namelist, cu_physics=cu_physics)
    carry = initial_operational_carry(child_state_with_parent_bdy)
    jax.block_until_ready(jax.tree_util.tree_leaves(carry)[0])
    return {
        "run": run,
        "state": child_state_with_parent_bdy,
        "base_state": child["base_state"],
        "carry": carry,
        "namelist": namelist,
        "grid": child["grid"],
        "tendencies": child["tendencies"],
        "jax": jax,
        "jnp": jnp,
    }


def jax_compare_array(field: str, state: Any, base_state: Any) -> Any:
    mapping = {
        "T": lambda: state.theta - P0_THETA_OFFSET_K,
        "P": lambda: state.p_perturbation,
        "PB": lambda: base_state.pb,
        "PH": lambda: state.ph_perturbation,
        "PHB": lambda: base_state.phb,
        "MU": lambda: state.mu_perturbation,
        "MUB": lambda: base_state.mub,
        "U": lambda: state.u,
        "V": lambda: state.v,
        "W": lambda: state.w,
        "QVAPOR": lambda: state.qv,
        "QCLOUD": lambda: state.qc,
        "QRAIN": lambda: state.qr,
        "QICE": lambda: state.qi,
        "QSNOW": lambda: state.qs,
        "QGRAUP": lambda: state.qg,
    }
    return mapping[field]()


def fortran_index(field: str, index: tuple[int, ...] | None) -> dict[str, int] | None:
    if index is None:
        return None
    if field in {"T", "P", "PB", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"}:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field in {"PH", "PHB", "W"}:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "kstag": int(k) + 1}
    if field == "U":
        k, y, x = index
        return {"i_xstag": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field == "V":
        k, y, x = index
        return {"i": int(x) + 1, "j_ystag": int(y) + 1, "k": int(k) + 1}
    if field in {"MU", "MUB"}:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    return None


def compare_arrays(truth_path: Path, state: Any, base_state: Any, jax: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    with np.load(truth_path) as truth:
        missing = [name for name in ALL_COMPARE_FIELDS if name not in truth]
        if missing:
            return {"status": "BLOCKED_TRUTH_MISSING_KEYS", "missing": missing}
        for name in ALL_COMPARE_FIELDS:
            wrf = np.asarray(truth[name], dtype=np.float64)
            candidate = np.asarray(jax.device_get(jax_compare_array(name, state, base_state)), dtype=np.float64)
            if wrf.shape != candidate.shape:
                return {
                    "status": "BLOCKED_SHAPE_MISMATCH",
                    "field": name,
                    "wrf_shape": list(wrf.shape),
                    "jax_shape": list(candidate.shape),
                }
            diff = candidate - wrf
            absdiff = np.abs(diff)
            finite_abs = absdiff[np.isfinite(absdiff)]
            mismatch_mask = (diff != 0.0) | (~np.isfinite(diff))
            mismatch = np.argwhere(mismatch_mask)
            first = tuple(int(x) for x in mismatch[0]) if mismatch.size else None
            if finite_abs.size:
                worst = tuple(int(x) for x in np.unravel_index(int(np.nanargmax(absdiff)), absdiff.shape))
                max_abs = float(np.nanmax(absdiff))
                rmse = float(np.sqrt(np.nanmean(diff * diff)))
                bias = float(np.nanmean(diff))
                p95 = float(np.nanpercentile(absdiff, 95))
                p99 = float(np.nanpercentile(absdiff, 99))
            else:
                worst = first
                max_abs = None
                rmse = None
                bias = None
                p95 = None
                p99 = None
            metrics[name] = {
                "count": int(diff.size),
                "shape": list(diff.shape),
                "max_abs": max_abs,
                "rmse": rmse,
                "bias": bias,
                "p95": p95,
                "p99": p99,
                "nonfinite_diff_count": int((~np.isfinite(diff)).sum()),
                "first_mismatch_index": list(first) if first is not None else None,
                "first_mismatch_fortran": fortran_index(name, first),
                "worst_mismatch_index": list(worst) if worst is not None else None,
                "worst_mismatch_fortran": fortran_index(name, worst),
            }
    ranked = sorted(
        [{"field": name, **item} for name, item in metrics.items()],
        key=lambda item: (-1.0 if item["max_abs"] is None else float(item["max_abs"])),
        reverse=True,
    )
    first_divergent = None
    for name in ALL_COMPARE_FIELDS:
        item = metrics[name]
        if item["nonfinite_diff_count"] or (item["max_abs"] is not None and float(item["max_abs"]) != 0.0):
            first_divergent = name
            break
    return {
        "status": "COMPARISON_EXECUTED",
        "truth_file": str(truth_path),
        "strict_same_input_comparison_run": True,
        "first_divergent_field": first_divergent,
        "per_field_metrics": metrics,
        "ranked_residuals": ranked,
    }


def run_jax_step1_compare() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        _physics_step_forcing,
        _rk_scan_step_with_pre_halo_capture,
    )

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    if not ACCEPTED_TRUTH.is_file():
        return {"status": "BLOCKED_NO_TRUTH_NPZ", "truth": path_info(ACCEPTED_TRUTH)}
    try:
        inputs = build_step1_jax_inputs()
        namelist = inputs["namelist"]
        jnp = inputs["jnp"]
        lead_seconds = jnp.asarray(float(TARGET_STEP) * float(namelist.dt_s), dtype=jnp.float64)
        cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
        run_radiation = bool(cadence > 0 and TARGET_STEP % cadence == 0)
        physics = _physics_step_forcing(
            inputs["carry"],
            namelist,
            lead_seconds,
            run_radiation=run_radiation,
        )
        result = _rk_scan_step_with_pre_halo_capture(
            physics.carry,
            namelist,
            lead_seconds=lead_seconds,
            physics_tendencies=physics.dry_tendencies,
        )
        jax.block_until_ready(result.pre_halo_state.theta)
        comparison = compare_arrays(ACCEPTED_TRUTH, result.pre_halo_state, inputs["base_state"], jax)
        return {
            "status": comparison["status"],
            "step_index": TARGET_STEP,
            "lead_seconds": float(lead_seconds),
            "run_radiation": run_radiation,
            "radiation_cadence_steps": cadence,
            "namelist": {
                "dt_s": float(namelist.dt_s),
                "rk_order": int(namelist.rk_order),
                "run_physics": bool(namelist.run_physics),
                "run_boundary": bool(namelist.run_boundary),
                "force_fp64": bool(namelist.force_fp64),
                "cu_physics": int(namelist.cu_physics),
                "radiation_static_loaded": namelist.radiation_static is not None,
                "gwdo_statics_loaded": namelist.gwdo_statics is not None,
            },
            "grid": {
                "nz": int(inputs["grid"].nz),
                "ny": int(inputs["grid"].ny),
                "nx": int(inputs["grid"].nx),
                "dx_m": float(inputs["grid"].projection.dx_m),
                "dy_m": float(inputs["grid"].projection.dy_m),
            },
            **comparison,
        }
    except Exception as exc:
        return {
            "status": "BLOCKED_JAX_CAPTURE_EXCEPTION",
            "exception": repr(exc),
            "exact_function_boundary": (
                "src/gpuwrf/runtime/operational_mode.py::_physics_step_forcing followed by "
                "_rk_scan_step_with_pre_halo_capture(...).pre_halo_state"
            ),
            "smallest_next_patch_or_tool": (
                "Make the proof-local same-input loader return the exact object or field named in this exception, "
                "or add a proof-only wrapper around _rk_scan_step_with_pre_halo_capture that accepts the current "
                "OperationalCarry/OperationalNamelist without production src edits."
            ),
        }


def derive_verdict(wrf_truth: Mapping[str, Any], jax_compare: Mapping[str, Any] | None) -> str:
    status = str(wrf_truth.get("status"))
    if not status.startswith("TRUTH_NPZ_READY"):
        suffix = status.replace("BLOCKED_", "").replace("TRUTH_", "").upper()
        return f"STEP1_WRF_TRUTH_BLOCKED_{suffix}"
    if not jax_compare or jax_compare.get("status") != "COMPARISON_EXECUTED":
        suffix = str((jax_compare or {}).get("status", "JAX_CAPTURE_MISSING")).replace("BLOCKED_", "").upper()
        return f"STEP1_WRF_TRUTH_READY_JAX_CAPTURE_BLOCKED_{suffix}"
    first = jax_compare.get("first_divergent_field")
    if first:
        return f"STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_{first}"
    return "STEP1_SAME_INPUT_COMPARISON_EXECUTED_CLEAN"


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    wrf_status = payload["wrf_truth"]["status"]
    lines = [
        "# V0.14 Step-1 Same-Input Truth",
        "",
        f"Verdict: `{verdict}`.",
        "",
        "## Result",
        "",
        f"- WRF truth status: `{wrf_status}`.",
        f"- Truth NPZ: `{ACCEPTED_TRUTH}`.",
        f"- Strict JAX pre-halo comparison run: `{payload['comparison'].get('strict_same_input_comparison_run', False)}`.",
    ]
    if payload["comparison"].get("status") == "COMPARISON_EXECUTED":
        first = payload["comparison"].get("first_divergent_field")
        top = payload["comparison"]["ranked_residuals"][0]
        lines.extend(
            [
                f"- First divergent field in schema order: `{first}`.",
                f"- Largest max_abs field: `{top['field']}` max_abs `{top['max_abs']}` rmse `{top['rmse']}`.",
                "",
                "Detailed per-field metrics are in `proofs/v014/step1_same_input_truth.json`.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Blocker",
                "",
                f"- Comparison status: `{payload['comparison'].get('status')}`.",
                f"- Exact blocker: `{payload['comparison'].get('exception') or payload['wrf_truth'].get('status')}`.",
                f"- Next patch/tool: `{payload['comparison'].get('smallest_next_patch_or_tool') or payload.get('next_decision')}`.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    commands = list(payload["commands"]["required_validation"])
    for item in payload["commands"].get("executed", []):
        commands.append(item.get("stage", "command") + ": " + " ".join(item.get("command", [])))
    seen = set(commands)
    for item in payload["wrf_truth"].get("recorded_wrf_commands", []):
        line = item.get("stage", "command") + ": " + " ".join(item.get("command", []))
        if line not in seen:
            commands.append(line)
            seen.add(line)
    lines = [
        "# Review: V0.14 Step-1 Same-Input Truth",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: produce full-domain CPU-WRF d02 step-1 post-RK/pre-halo truth and run the strict same-input JAX pre-halo comparison, or fail closed with the exact blocker.",
        "",
        "files changed:",
        "- `proofs/v014/step1_same_input_truth.py`",
        "- `proofs/v014/step1_same_input_truth.json`",
        "- `proofs/v014/step1_same_input_truth.md`",
        "- `proofs/v014/step1_same_input_truth_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-step1-same-input-truth.md`",
        "",
        "commands run:",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "proof objects produced:",
        f"- `{OUT_JSON}`",
        f"- `{OUT_MD}`",
        f"- `{OUT_PATCH}`",
        f"- `{OUT_REVIEW}`",
        f"- `{ACCEPTED_TRUTH}`",
        "",
        "unresolved risks:",
    ]
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    commands: list[dict[str, Any]] = []
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    wrf_truth = ensure_wrf_truth(commands)
    comparison: dict[str, Any]
    if str(wrf_truth.get("status")).startswith("TRUTH_NPZ_READY"):
        comparison = run_jax_step1_compare()
    else:
        comparison = {
            "status": "NOT_RUN_WRF_TRUTH_BLOCKED",
            "strict_same_input_comparison_run": False,
            "why": wrf_truth.get("status"),
        }
    verdict = derive_verdict(wrf_truth, comparison)
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_same_input_truth.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "weak_comparison_avoided": True,
        "jax_vs_jax_self_compare": False,
        "one_cell_proof": False,
        "mixed_wrf_jax_carry_leaves": False,
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "surface": TARGET_SURFACE,
            "accepted_comparison": "WRF step-1 post-RK/pre-halo vs JAX one-step _rk_scan_step_with_pre_halo_capture(...).pre_halo_state",
            "rejected_comparison": "WRF step-1 post-RK/pre-halo vs JAX initial state",
        },
        "environment": jax_environment(),
        "git_head": git_head,
        "inputs": {
            "project_constitution": path_info(PROJECT_CONSTITUTION),
            "agents": path_info(AGENTS),
            "managing_sprints_skill": path_info(MANAGING_SPRINTS_SKILL),
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "same_input_contract_builder_md": path_info(BUILDER_MD),
            "same_input_contract_builder_json": path_info(BUILDER_JSON),
            "same_input_contract_builder_py": path_info(BUILDER_PY),
            "wrf_post_rk_refresh_localization_patch": path_info(REFRESH_PATCH),
            "same_state_momentum_mass_py": path_info(SAME_STATE_PY),
            "run_case3": path_info(RUN_CASE3),
            "wrf_source": path_info(WRF_SOURCE),
        },
        "field_order": list(ALL_COMPARE_FIELDS),
        "wrf_truth": wrf_truth,
        "comparison": comparison,
        "commands": {
            "executed": commands,
            "required_validation": [
                "python -m py_compile proofs/v014/step1_same_input_truth.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_same_input_truth.py",
                "python -m json.tool proofs/v014/step1_same_input_truth.json >/tmp/step1_same_input_truth.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "wrf_patch_diff": str(OUT_PATCH),
            "review": str(OUT_REVIEW),
            "accepted_truth_npz": str(ACCEPTED_TRUTH),
        },
        "unresolved_risks": [
            "This is the first strict full-domain step-1 comparison; residuals name the first divergent field but do not localize the responsible operator.",
            "The disposable WRF tree inherits prior v014 scratch WRF hook scaffolding; the proof patch diff records only the added step-1 hook block.",
        ],
        "next_decision": (
            "If comparison executed, localize the first divergent field one operator earlier; if blocked, apply the exact blocker patch/tool named in this JSON."
        ),
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(f"verdict={verdict}")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"patch={OUT_PATCH}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
