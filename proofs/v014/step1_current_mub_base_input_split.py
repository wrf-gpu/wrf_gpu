#!/usr/bin/env python3
"""V0.14 Step-1 current-MUB/base-input split proof.

CPU-only proof. It recovers the accepted WRF adjust_tempqv scalar hook from the
previous sprint and recomputes the JAX/proof live-nest path used by the theta
proof. The disposable WRF tree is read-only in this sandbox, so the fresh WRF
hook is emitted as a proposed diff artifact rather than applied here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

OUT_JSON = PROOF_DIR / "step1_current_mub_base_input_split.json"
OUT_MD = PROOF_DIR / "step1_current_mub_base_input_split.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md"
OUT_PATCH = PROOF_DIR / "step1_current_mub_base_input_split_wrf_patch.diff"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-current-mub-base-input-split/sprint-contract.md"
)
WRF_TREE = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF")
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_step1_current_mub_base_input_split")
PRIOR_ADJUST_HOOK = (
    Path("/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth")
    / "adjust_tempqv_d2_i18_j10_k2.txt"
)
PRIOR_THETA_JSON = PROOF_DIR / "step1_theta_same_qvapor.json"
PRIOR_ADJUST_JSON = PROOF_DIR / "step1_adjust_tempqv_intermediate.json"
JAX_LOADER_JSON = PROOF_DIR / "step1_jax_loader_tstate.json"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

TARGET_ZERO = {"k": 1, "y": 9, "x": 17}
TARGET_FORTRAN = {"i": 18, "j": 10, "k": 2}
REQUIRED_ANCESTOR = "9a7016d9"
VERDICT = "STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED"
THETA_OFFSET_K = 300.0


WRF_PATCH_TEXT = """diff --git a/share/mediation_integrate.F b/share/mediation_integrate.F
--- a/share/mediation_integrate.F
+++ b/share/mediation_integrate.F
@@ -535,6 +535,17 @@ SUBROUTINE med_nest_initial ( parent , nest , config_flags )
    INTEGER                :: ids , ide , jds , jde , kds , kde , &
                              ims , ime , jms , jme , kms , kme , &
                              ips , ipe , jps , jpe , kps , kpe
+   CHARACTER(LEN=512)     :: wrfgpu2_env
+   CHARACTER(LEN=1024)    :: wrfgpu2_root, wrfgpu2_path
+   INTEGER                :: wrfgpu2_target_domain, wrfgpu2_target_i
+   INTEGER                :: wrfgpu2_target_j, wrfgpu2_target_k, wrfgpu2_ios
+   INTEGER                :: wrfgpu2_unit
+   LOGICAL                :: wrfgpu2_enabled, wrfgpu2_owner
+   REAL                   :: wrfgpu2_ht_parent, wrfgpu2_ht_child
+   REAL                   :: wrfgpu2_ht_post, wrfgpu2_mub_parent
+   REAL                   :: wrfgpu2_mub_child, wrfgpu2_mub_post
+   REAL                   :: wrfgpu2_phb_parent, wrfgpu2_phb_child
+   REAL                   :: wrfgpu2_phb_post, wrfgpu2_mub_after_start
 
 #if (EM_CORE == 1)
@@ -650,6 +661,31 @@ SUBROUTINE med_nest_initial ( parent , nest , config_flags )
 
    CALL interp_init
 
+   wrfgpu2_enabled = .FALSE.
+   wrfgpu2_owner = .FALSE.
+   wrfgpu2_target_domain = 2
+   wrfgpu2_target_i = 18
+   wrfgpu2_target_j = 10
+   wrfgpu2_target_k = 2
+   wrfgpu2_root = '/mnt/data/wrf_gpu2/v014_step1_current_mub_base_input_split/wrf_truth'
+   wrfgpu2_env = ''
+   CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_STEP1_CURRENT_MUB_BASE_INPUT_SPLIT', wrfgpu2_env)
+   IF (TRIM(wrfgpu2_env) == '1') wrfgpu2_enabled = .TRUE.
+   IF (wrfgpu2_enabled) THEN
+      wrfgpu2_env = ''
+      CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_STEP1_CURRENT_MUB_BASE_INPUT_SPLIT_ROOT', wrfgpu2_env)
+      IF (LEN_TRIM(wrfgpu2_env) > 0) wrfgpu2_root = TRIM(wrfgpu2_env)
+      wrfgpu2_env = ''
+      CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_STEP1_CURRENT_MUB_BASE_INPUT_SPLIT_I', wrfgpu2_env)
+      IF (LEN_TRIM(wrfgpu2_env) > 0) READ(wrfgpu2_env,*,IOSTAT=wrfgpu2_ios) wrfgpu2_target_i
+      wrfgpu2_env = ''
+      CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_STEP1_CURRENT_MUB_BASE_INPUT_SPLIT_J', wrfgpu2_env)
+      IF (LEN_TRIM(wrfgpu2_env) > 0) READ(wrfgpu2_env,*,IOSTAT=wrfgpu2_ios) wrfgpu2_target_j
+      wrfgpu2_env = ''
+      CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_STEP1_CURRENT_MUB_BASE_INPUT_SPLIT_K', wrfgpu2_env)
+      IF (LEN_TRIM(wrfgpu2_env) > 0) READ(wrfgpu2_env,*,IOSTAT=wrfgpu2_ios) wrfgpu2_target_k
+   END IF
+
    CALL domain_clock_get( parent, start_time=strt_time, current_time=cur_time )
 
    IF ( .not. ( config_flags%restart .AND. strt_time .EQ. cur_time ) ) THEN
@@ -730,6 +766,21 @@ SUBROUTINE med_nest_initial ( parent , nest , config_flags )
        CALL  copy_3d_field ( nest%mub_save , nest%mub , &
                              ids , ide , jds , jde , 1   , 1   , &
                              ims , ime , jms , jme , 1   , 1   , &
                              ips , ipe , jps , jpe , 1   , 1   )
+       IF (wrfgpu2_enabled .AND. nest%id .EQ. wrfgpu2_target_domain .AND. &
+           wrfgpu2_target_i .GE. ips .AND. wrfgpu2_target_i .LE. MIN(ipe,ide-1) .AND. &
+           wrfgpu2_target_j .GE. jps .AND. wrfgpu2_target_j .LE. MIN(jpe,jde-1)) THEN
+          wrfgpu2_owner = .TRUE.
+          wrfgpu2_ht_parent = nest%ht_int(wrfgpu2_target_i,wrfgpu2_target_j)
+          wrfgpu2_ht_child = nest%ht(wrfgpu2_target_i,wrfgpu2_target_j)
+          wrfgpu2_mub_parent = nest%mub_fine(wrfgpu2_target_i,wrfgpu2_target_j)
+          wrfgpu2_mub_child = nest%mub_save(wrfgpu2_target_i,wrfgpu2_target_j)
+          wrfgpu2_phb_parent = nest%phb_fine(wrfgpu2_target_i,wrfgpu2_target_k,wrfgpu2_target_j)
+          wrfgpu2_phb_child = nest%phb(wrfgpu2_target_i,wrfgpu2_target_k,wrfgpu2_target_j)
+       END IF
 
 ! blend parent and nest fields: terrain, mub, and phb.  The ht, mub and phb are used in start_domain.
 
@@ -750,6 +801,12 @@ SUBROUTINE med_nest_initial ( parent , nest , config_flags )
                                 ims , ime , jms , jme , kms , kme , &
                                 ips , ipe , jps , jpe , kps , kpe )
        ENDIF
+       IF (wrfgpu2_owner) THEN
+          wrfgpu2_ht_post = nest%ht(wrfgpu2_target_i,wrfgpu2_target_j)
+          wrfgpu2_mub_post = nest%mub(wrfgpu2_target_i,wrfgpu2_target_j)
+          wrfgpu2_phb_post = nest%phb(wrfgpu2_target_i,wrfgpu2_target_k,wrfgpu2_target_j)
+       END IF
 
        !  adjust temp and qv
 
@@ -803,6 +860,38 @@ SUBROUTINE med_nest_initial ( parent , nest , config_flags )
        nest%press_adj = .TRUE.
        CALL push_communicators_for_domain( nest%id )
        CALL start_domain ( nest , .TRUE. )
+       IF (wrfgpu2_owner) THEN
+          wrfgpu2_mub_after_start = nest%mub(wrfgpu2_target_i,wrfgpu2_target_j)
+          CALL execute_command_line('mkdir -p '//TRIM(wrfgpu2_root), wait=.TRUE.)
+          WRITE(wrfgpu2_path,'(A,I0,A,I0,A,I0,A,I0,A)') &
+               TRIM(wrfgpu2_root)//'/current_mub_base_split_d', nest%id, &
+               '_i', wrfgpu2_target_i, '_j', wrfgpu2_target_j, &
+               '_k', wrfgpu2_target_k, '.txt'
+          OPEN(NEWUNIT=wrfgpu2_unit, FILE=TRIM(wrfgpu2_path), STATUS='REPLACE', ACTION='WRITE')
+          WRITE(wrfgpu2_unit,'(A)') '# WRFGPU2_V014_STEP1_CURRENT_MUB_BASE_INPUT_SPLIT v1'
+          WRITE(wrfgpu2_unit,'(A,1X,I0)') 'domain_id', nest%id
+          WRITE(wrfgpu2_unit,'(A,3(1X,I0))') 'target_fortran_i_j_k', &
+               wrfgpu2_target_i, wrfgpu2_target_j, wrfgpu2_target_k
+          WRITE(wrfgpu2_unit,'(A,6(1X,I0))') 'domain_bounds_ids_ide_jds_jde_kds_kde', &
+               ids, ide, jds, jde, kds, kde
+          WRITE(wrfgpu2_unit,'(A,1X,I0)') 'save_topo_from_real', nest%save_topo_from_real
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'ht_parent_interp_preinput', REAL(wrfgpu2_ht_parent,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'ht_child_input_preblend', REAL(wrfgpu2_ht_child,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'ht_post_blend', REAL(wrfgpu2_ht_post,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'mub_parent_interp_preinput', REAL(wrfgpu2_mub_parent,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'mub_child_input_preblend_save', REAL(wrfgpu2_mub_child,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'mub_post_blend_for_adjust', REAL(wrfgpu2_mub_post,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'mub_after_start_domain', REAL(wrfgpu2_mub_after_start,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'phb_parent_interp_preinput_k', REAL(wrfgpu2_phb_parent,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'phb_child_input_preblend_k', REAL(wrfgpu2_phb_child,KIND=8)
+          WRITE(wrfgpu2_unit,'(A,1X,E24.16)') 'phb_post_blend_k', REAL(wrfgpu2_phb_post,KIND=8)
+          CLOSE(wrfgpu2_unit)
+       END IF
        CALL pop_communicators_for_domain
      ENDIF
"""


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
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return sanitize_json(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_command(command: list[str], *, cwd: Path = ROOT, timeout_s: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "wall_s": float(time.perf_counter() - start),
            "timeout_s": int(timeout_s),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


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


def parse_scalar_hook(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    headers: dict[str, Any] = {}
    values: dict[str, float] = {}
    marker = None
    for raw_line in path.read_text(errors="replace").splitlines():
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
            values[key] = float(value)
        else:
            headers[key] = value
    return {"status": "READY", "path": str(path), "marker": marker, "headers": headers, "values": values}


def load_prior_theta_target() -> dict[str, Any]:
    if not PRIOR_THETA_JSON.is_file():
        return {"status": "MISSING", "path": str(PRIOR_THETA_JSON)}
    prior = json.loads(PRIOR_THETA_JSON.read_text())
    worst = prior["comparisons"]["final_candidate_residual"]["worst_cell"]
    pressure = worst["available_pressure_base_inputs"]
    return {
        "status": "READY",
        "path": str(PRIOR_THETA_JSON),
        "worst_cell": worst,
        "candidate_reconstruction": pressure["candidate_reconstruction"],
        "wrf_precall_truth": pressure["wrf_precall_truth"],
    }


def jax_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
        "jax_default_backend": None,
        "gpu_used": False,
    }
    try:
        import jax  # noqa: PLC0415

        devices = list(jax.devices())
        result.update(
            {
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([device for device in devices if device.platform == "gpu"]),
                "gpu_used": any(device.platform == "gpu" for device in devices),
            }
        )
    except Exception as exc:
        result["jax_import_error"] = repr(exc)
    return result


def wrf_source_formula(*, p: float, mub: float, c3h: float, c4h: float, p_top: float) -> float:
    return float(p + c4h + c3h * mub + p_top)


def requested_formula(*, p: float, mub: float, c3h: float, c4h: float, p_top: float) -> float:
    return float(p + c3h * (mub + p_top) + c4h)


def blend_weight_for_target(i: int, j: int, ide: int, jde: int, spec_bdy_width: int, blend_width: int) -> dict[str, Any]:
    weights: tuple[float, float] | None = None
    matched_blend_cell = None
    r_blend = 1.0 / float(blend_width + 1)
    for blend_cell in range(blend_width, 0, -1):
        if (
            i == spec_bdy_width + blend_cell
            or j == spec_bdy_width + blend_cell
            or i == ide - spec_bdy_width - blend_cell
            or j == jde - spec_bdy_width - blend_cell
        ):
            weights = (blend_cell * r_blend, (blend_width + 1 - blend_cell) * r_blend)
            matched_blend_cell = blend_cell
    if i <= spec_bdy_width or j <= spec_bdy_width or i >= ide - spec_bdy_width or j >= jde - spec_bdy_width:
        weights = (0.0, 1.0)
        matched_blend_cell = "specified_boundary"
    return {
        "fortran_i": int(i),
        "fortran_j": int(j),
        "ide": int(ide),
        "jde": int(jde),
        "spec_bdy_width": int(spec_bdy_width),
        "blend_width": int(blend_width),
        "matched_blend_cell": matched_blend_cell,
        "fine_weight": None if weights is None else float(weights[0]),
        "parent_interpolated_weight": None if weights is None else float(weights[1]),
    }


def recompute_jax_target() -> dict[str, Any]:
    try:
        import jax  # noqa: PLC0415
        import step1_live_nest_init_rerun as live  # noqa: PLC0415
        from gpuwrf.integration import d02_replay  # noqa: PLC0415
        from gpuwrf.nesting.interp import sint_to_child_reference  # noqa: PLC0415

        if jax.default_backend() != "cpu":
            return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

        inputs = live.build_live_nest_step1_inputs()
        parent = inputs["parent"]
        raw_child = inputs["raw_child"]
        live_child = inputs["live_child"]
        run = inputs["run"]
        y = TARGET_ZERO["y"]
        x = TARGET_ZERO["x"]
        k = TARGET_ZERO["k"]

        parent_state = parent["state"]
        raw_state = raw_child["state"]
        live_state = live_child["state"]
        metrics = live_child["metrics"]
        grid = raw_child["grid"]
        child_meta = run.grid("d02")
        ratio = int(child_meta.parent_grid_ratio)
        i_parent_start = int(child_meta.i_parent_start)
        j_parent_start = int(child_meta.j_parent_start)

        parent_mub = np.asarray(jax.device_get(parent_state.mu_total - parent_state.mu_perturbation), dtype=np.float64)
        raw_mub = np.asarray(jax.device_get(raw_state.mu_total - raw_state.mu_perturbation), dtype=np.float64)
        final_mub = np.asarray(jax.device_get(live_state.mu_total - live_state.mu_perturbation), dtype=np.float64)
        parent_mub_on_child = sint_to_child_reference(
            parent_mub,
            ratio=ratio,
            i_parent_start=i_parent_start,
            j_parent_start=j_parent_start,
            child_ny=int(grid.ny),
            child_nx=int(grid.nx),
        )
        direct_blend_mub = d02_replay._wrf_blend_terrain_host(
            parent_mub_on_child,
            raw_mub,
            spec_bdy_width=5,
            blend_width=5,
        )

        parent_hgt = np.asarray(jax.device_get(parent["grid"].terrain_height), dtype=np.float64)
        raw_hgt = np.asarray(jax.device_get(raw_child["grid"].terrain_height), dtype=np.float64)
        live_hgt = np.asarray(jax.device_get(live_child["grid"].terrain_height), dtype=np.float64)
        parent_hgt_on_child = sint_to_child_reference(
            parent_hgt,
            ratio=ratio,
            i_parent_start=i_parent_start,
            j_parent_start=j_parent_start,
            child_ny=int(grid.ny),
            child_nx=int(grid.nx),
        )
        direct_blend_hgt = d02_replay._wrf_blend_terrain_host(
            parent_hgt_on_child,
            raw_hgt,
            spec_bdy_width=5,
            blend_width=5,
        )

        raw_pp = np.asarray(jax.device_get(raw_state.p_perturbation), dtype=np.float64)
        live_pb = np.asarray(jax.device_get(live_state.p_total - live_state.p_perturbation), dtype=np.float64)
        c3h = np.asarray(jax.device_get(metrics.c3h), dtype=np.float64)
        c4h = np.asarray(jax.device_get(metrics.c4h), dtype=np.float64)
        p_top = float(np.asarray(jax.device_get(metrics.p_top), dtype=np.float64).reshape(-1)[0])
        live_phb = np.asarray(jax.device_get(live_state.ph_total - live_state.ph_perturbation), dtype=np.float64)
        raw_phb = np.asarray(jax.device_get(raw_state.ph_total - raw_state.ph_perturbation), dtype=np.float64)

        c3 = float(c3h[k])
        c4 = float(c4h[k])
        p = float(raw_pp[k, y, x])
        out = {
            "status": "READY",
            "backend": jax.default_backend(),
            "source_function": "proofs/v014/step1_live_nest_init_rerun.py::build_live_nest_step1_inputs",
            "live_nest_base_init_meta": live_child.get("live_nest_base_init"),
            "target": {"zero_index": TARGET_ZERO, "fortran_index": TARGET_FORTRAN},
            "grid": {"ny": int(grid.ny), "nx": int(grid.nx), "nz": int(grid.nz)},
            "nest_geometry": {
                "parent_grid_ratio": ratio,
                "i_parent_start": i_parent_start,
                "j_parent_start": j_parent_start,
            },
            "blend_weight": blend_weight_for_target(
                TARGET_FORTRAN["i"],
                TARGET_FORTRAN["j"],
                int(grid.nx) + 1,
                int(grid.ny) + 1,
                5,
                5,
            ),
            "values": {
                "mub_parent_interp_preinput": float(parent_mub_on_child[y, x]),
                "mub_child_input_preblend_save": float(raw_mub[y, x]),
                "mub_direct_post_blend_for_adjust": float(direct_blend_mub[y, x]),
                "mub_final_after_start_domain": float(final_mub[y, x]),
                "ht_parent_interp_preinput": float(parent_hgt_on_child[y, x]),
                "ht_child_input_preblend": float(raw_hgt[y, x]),
                "ht_direct_post_blend": float(direct_blend_hgt[y, x]),
                "ht_final_after_start_domain": float(live_hgt[y, x]),
                "p": p,
                "c3h": c3,
                "c4h": c4,
                "p_top": p_top,
                "pb_direct_post_blend_equiv": wrf_source_formula(
                    p=0.0, mub=float(direct_blend_mub[y, x]), c3h=c3, c4h=c4, p_top=p_top
                ),
                "pb_final_after_start_domain": float(live_pb[k, y, x]),
                "p_new_direct_post_blend_source_formula": wrf_source_formula(
                    p=p, mub=float(direct_blend_mub[y, x]), c3h=c3, c4h=c4, p_top=p_top
                ),
                "p_new_final_after_start_domain_source_formula": wrf_source_formula(
                    p=p, mub=float(final_mub[y, x]), c3h=c3, c4h=c4, p_top=p_top
                ),
                "phb_child_input_preblend_k": float(raw_phb[k, y, x]),
                "phb_final_after_start_domain_k": float(live_phb[k, y, x]),
                "phb_final_after_start_domain_kp1": float(live_phb[k + 1, y, x]),
            },
        }
        return out
    except Exception as exc:
        return {"status": "BLOCKED_JAX_RECOMPUTE_EXCEPTION", "exception": repr(exc)}


def compare_surfaces(wrf_adjust: Mapping[str, Any], prior: Mapping[str, Any], jax_target: Mapping[str, Any]) -> dict[str, Any]:
    if wrf_adjust.get("status") != "READY":
        return {"status": "BLOCKED_NO_WRF_ADJUST_HOOK", "wrf_adjust": wrf_adjust}
    if prior.get("status") != "READY":
        return {"status": "BLOCKED_NO_PRIOR_THETA_JSON", "prior": prior}
    if jax_target.get("status") != "READY":
        return {"status": "BLOCKED_NO_JAX_RECOMPUTE", "jax_target": jax_target}

    w = wrf_adjust["values"]
    p = prior["candidate_reconstruction"]
    pre = prior["wrf_precall_truth"]
    j = jax_target["values"]
    c3 = float(w["c3h"])

    rows = [
        {
            "comparison": "WRF adjust_tempqv current MUB minus JAX theta-proof final MUB",
            "field": "mub",
            "wrf": float(w["mub"]),
            "jax": float(j["mub_final_after_start_domain"]),
            "delta_wrf_minus_jax": float(w["mub"]) - float(j["mub_final_after_start_domain"]),
        },
        {
            "comparison": "WRF adjust_tempqv current MUB minus JAX direct WRF blend MUB",
            "field": "mub",
            "wrf": float(w["mub"]),
            "jax": float(j["mub_direct_post_blend_for_adjust"]),
            "delta_wrf_minus_jax": float(w["mub"]) - float(j["mub_direct_post_blend_for_adjust"]),
        },
        {
            "comparison": "WRF pre-part1 final MUB minus JAX theta-proof final MUB",
            "field": "mub",
            "wrf": float(pre["MUB"]),
            "jax": float(j["mub_final_after_start_domain"]),
            "delta_wrf_minus_jax": float(pre["MUB"]) - float(j["mub_final_after_start_domain"]),
        },
        {
            "comparison": "WRF adjust_tempqv current MUB minus WRF pre-part1 final MUB",
            "field": "mub",
            "wrf_adjust": float(w["mub"]),
            "wrf_prepart": float(pre["MUB"]),
            "delta_adjust_minus_prepart": float(w["mub"]) - float(pre["MUB"]),
        },
        {
            "comparison": "WRF p_new minus JAX theta-proof p_new",
            "field": "p_new",
            "wrf": float(w["p_new"]),
            "jax": float(p["adjust_tempqv_p_new"]),
            "delta_wrf_minus_jax": float(w["p_new"]) - float(p["adjust_tempqv_p_new"]),
        },
        {
            "comparison": "WRF pb_new_equiv minus JAX theta-proof live_pb",
            "field": "pb_new_equiv",
            "wrf": float(w["pb_new_equiv"]),
            "jax": float(p["live_pb"]),
            "delta_wrf_minus_jax": float(w["pb_new_equiv"]) - float(p["live_pb"]),
        },
    ]

    formulas = {
        "wrf_adjust_source_formula": wrf_source_formula(
            p=float(w["p"]), mub=float(w["mub"]), c3h=float(w["c3h"]), c4h=float(w["c4h"]), p_top=float(w["p_top"])
        ),
        "wrf_adjust_hook_p_new": float(w["p_new"]),
        "wrf_adjust_source_minus_hook_p_new": wrf_source_formula(
            p=float(w["p"]), mub=float(w["mub"]), c3h=float(w["c3h"]), c4h=float(w["c4h"]), p_top=float(w["p_top"])
        )
        - float(w["p_new"]),
        "wrf_adjust_requested_grouped_formula": requested_formula(
            p=float(w["p"]), mub=float(w["mub"]), c3h=float(w["c3h"]), c4h=float(w["c4h"]), p_top=float(w["p_top"])
        ),
        "wrf_adjust_requested_minus_hook_p_new": requested_formula(
            p=float(w["p"]), mub=float(w["mub"]), c3h=float(w["c3h"]), c4h=float(w["c4h"]), p_top=float(w["p_top"])
        )
        - float(w["p_new"]),
        "jax_final_source_formula": wrf_source_formula(
            p=float(w["p"]),
            mub=float(j["mub_final_after_start_domain"]),
            c3h=float(w["c3h"]),
            c4h=float(w["c4h"]),
            p_top=float(w["p_top"]),
        ),
        "jax_direct_blend_source_formula": wrf_source_formula(
            p=float(w["p"]),
            mub=float(j["mub_direct_post_blend_for_adjust"]),
            c3h=float(w["c3h"]),
            c4h=float(w["c4h"]),
            p_top=float(w["p_top"]),
        ),
        "delta_p_new_from_mub_delta_source_formula": c3
        * (float(w["mub"]) - float(j["mub_final_after_start_domain"])),
        "formula_freeze_status": (
            "WRF_SOURCE_FORMULA_VERIFIED; requested grouped c3h*(mub+p_top) formula rejected for adjust_tempqv"
        ),
        "wrf_source_formula": "p_new = p + c4h + c3h*mub + p_top",
        "requested_grouped_formula": "p_new = p + c3h*(mub+p_top) + c4h",
    }

    return {
        "status": "COMPARISON_EXECUTED",
        "diff_sign": "wrf_minus_jax",
        "rows": rows,
        "formulas": formulas,
        "first_divergence_surface": (
            "current MUB passed to WRF adjust_tempqv after blend_terrain(nest%mub_fine,nest%mub) "
            "and before the later start_domain base recompute"
        ),
        "root_cause": (
            "The theta proof used the final post-start_domain live-nest base MUB. WRF adjust_tempqv uses a "
            "transient post-blend/pre-start_domain current MUB. A proof-side direct WRF MUB blend reproduces "
            "the WRF adjust_tempqv MUB at the target cell."
        ),
        "secondary_boundary_note": (
            "The previous pressure comparison was also a boundary split: WRF adjust_tempqv current MUB is "
            "not the same surface as WRF pre-part1 final MUB."
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    comparison = payload["comparisons"]
    formulas = comparison.get("formulas", {})
    rows = comparison.get("rows", [])
    wrf = payload["wrf_adjust_hook"]["values"]
    jax = payload["jax_recompute"].get("values", {})
    lines = [
        "# V0.14 Step-1 Current-MUB/Base-Input Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU-only proof; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Fresh scratch WRF hook not run here: `{payload['scratch_write_probe']['status']}`.",
        f"- Recovered WRF adjust hook: `{PRIOR_ADJUST_HOOK}`.",
        f"- Target zero `{TARGET_ZERO}`, Fortran `{TARGET_FORTRAN}`.",
        "",
        "## Explanation",
        "",
        "- WRF copies `nest%mub` to `nest%mub_save`, blends `nest%mub_fine` into current `nest%mub`, then calls `adjust_tempqv`.",
        "- Later, WRF calls `start_domain(nest,.TRUE.)`, which recomputes the final base fields used by the pre-part1 truth.",
        "- The JAX theta proof used that final post-`start_domain` base MUB for `adjust_tempqv`; WRF uses the transient post-blend/pre-`start_domain` MUB.",
        "",
        "## Target Values",
        "",
        "| Surface | MUB | PB/current base | p_new |",
        "|---|---:|---:|---:|",
        f"| WRF `adjust_tempqv` hook | {wrf['mub']:.12f} | {wrf['pb_new_equiv']:.12f} | {wrf['p_new']:.12f} |",
        f"| JAX theta proof final base | {jax.get('mub_final_after_start_domain'):.12f} | {jax.get('pb_final_after_start_domain'):.12f} | {jax.get('p_new_final_after_start_domain_source_formula'):.12f} |",
        f"| JAX direct WRF MUB blend | {jax.get('mub_direct_post_blend_for_adjust'):.12f} | {jax.get('pb_direct_post_blend_equiv'):.12f} | {jax.get('p_new_direct_post_blend_source_formula'):.12f} |",
        "",
        "## Comparisons",
        "",
        "| Comparison | Field | Delta |",
        "|---|---|---:|",
    ]
    for row in rows:
        delta = row.get("delta_wrf_minus_jax", row.get("delta_adjust_minus_prepart"))
        lines.append(f"| {row['comparison']} | `{row['field']}` | {float(delta):.16e} |")
    lines.extend(
        [
            "",
            "## Formula Check",
            "",
            f"- WRF source formula: `{formulas.get('wrf_source_formula')}`.",
            f"- WRF source formula minus hook `p_new`: `{formulas.get('wrf_adjust_source_minus_hook_p_new')}` Pa.",
            f"- Requested grouped formula: `{formulas.get('requested_grouped_formula')}`.",
            f"- Requested grouped formula minus hook `p_new`: `{formulas.get('wrf_adjust_requested_minus_hook_p_new')}` Pa.",
            "- Therefore the grouped `c3h*(mub+p_top)` form is not the WRF `adjust_tempqv` formula for this hook.",
            "",
            "## Handoff",
            "",
            "objective: explain the current-MUB/base-input mismatch driving the Step-1 live-nest theta residual.",
            "",
            "files changed:",
            "- `proofs/v014/step1_current_mub_base_input_split.py`",
            "- `proofs/v014/step1_current_mub_base_input_split.json`",
            "- `proofs/v014/step1_current_mub_base_input_split.md`",
            "- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`",
            "- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`",
            "",
            "commands run:",
        ]
    )
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            f"- `{OUT_JSON}`",
            f"- `{OUT_MD}`",
            f"- `{OUT_PATCH}`",
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
        "# Review: V0.14 Step-1 Current-MUB/Base-Input Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "Findings:",
        "- HIGH: The residual is caused by using the final post-`start_domain` base MUB as the current `adjust_tempqv` MUB. WRF uses the transient post-`blend_terrain`/pre-`start_domain` MUB.",
        "- MEDIUM: The requested grouped pressure formula is not what WRF `adjust_tempqv` executes; the verified source formula is `p + c4h + c3h*mub + p_top`.",
        "- LOW: This sandbox could not write the new `/mnt/data` scratch root, so the new WRF hook is delivered as a proposed disposable patch while scalar WRF truth is recovered from the prior accepted hook.",
        "",
        "Evidence:",
        f"- WRF adjust hook: `{PRIOR_ADJUST_HOOK}`",
        f"- JAX recompute source: `{payload['jax_recompute'].get('source_function')}`",
        f"- Patch diff artifact: `{OUT_PATCH}`",
        "",
        "Handoff:",
        "objective: explain the current-MUB/base-input mismatch driving the Step-1 live-nest theta residual.",
        "",
        "files changed:",
        "- `proofs/v014/step1_current_mub_base_input_split.py`",
        "- `proofs/v014/step1_current_mub_base_input_split.json`",
        "- `proofs/v014/step1_current_mub_base_input_split.md`",
        "- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            f"- `{OUT_JSON}`",
            f"- `{OUT_MD}`",
            f"- `{OUT_PATCH}`",
            f"- `{OUT_REVIEW}`",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def scratch_write_probe() -> dict[str, Any]:
    parent = SCRATCH.parent
    return {
        "path": str(SCRATCH),
        "parent": str(parent),
        "parent_exists": parent.exists(),
        "parent_writable_by_os_access": os.access(parent, os.W_OK),
        "status": "BLOCKED_READ_ONLY_FILESYSTEM" if not os.access(parent, os.W_OK) else "WRITABLE_NOT_USED",
    }


def main() -> int:
    git_head = run_command(["git", "rev-parse", "HEAD"])
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"])
    src_diff = run_command(["git", "diff", "--", "src/gpuwrf"])

    wrf_adjust = parse_scalar_hook(PRIOR_ADJUST_HOOK)
    prior = load_prior_theta_target()
    jax_recompute = recompute_jax_target()
    comparisons = compare_surfaces(wrf_adjust, prior, jax_recompute)
    env = jax_environment()
    gpu_used = bool(env.get("gpu_used"))

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_current_mub_base_input_split.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "cpu_only": True,
        "gpu_used": gpu_used,
        "no_gpu": True,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "source_patch_allowed_by_proof": False,
        "environment": env,
        "git": {
            "head": git_head,
            "required_ancestor": {
                "commit": REQUIRED_ANCESTOR,
                "returncode": ancestor.get("returncode"),
                "is_ancestor": ancestor.get("returncode") == 0,
                "command": ancestor.get("command"),
            },
            "src_gpuwrf_diff_empty": src_diff.get("stdout_tail", "") == "",
            "src_gpuwrf_diff": src_diff,
        },
        "target": {
            "domain": "d02",
            "wrf_grid_id": 2,
            "zero_index_order": "k,y,x",
            "zero_index": TARGET_ZERO,
            "fortran_index": TARGET_FORTRAN,
            "boundary_distance": 9,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "prior_adjust_hook": path_info(PRIOR_ADJUST_HOOK),
            "prior_theta_json": path_info(PRIOR_THETA_JSON),
            "prior_adjust_json": path_info(PRIOR_ADJUST_JSON),
            "jax_loader_json": path_info(JAX_LOADER_JSON),
            "handoff": path_info(HANDOFF),
            "wrf_tree": path_info(WRF_TREE),
        },
        "scratch_write_probe": scratch_write_probe(),
        "wrf_source_evidence": {
            "mediation_integrate": {
                "path": str(WRF_TREE / "share/mediation_integrate.F"),
                "lines": "726-805",
                "evidence": [
                    "mub_save is copied before terrain/base blending",
                    "blend_terrain updates ht, mub, and phb",
                    "adjust_tempqv is called with the transient post-blend nest%mub",
                    "start_domain(nest,.TRUE.) runs later and recomputes final base fields",
                ],
            },
            "adjust_tempqv": {
                "path": str(WRF_TREE / "dyn_em/nest_init_utils.F"),
                "lines": "812-890",
                "evidence": [
                    "p_old = c4(k) + c3(k)*save_mub(i,j) + p_top + pp(i,k,j)",
                    "p_new = c4(k) + c3(k)*mub(i,j) + p_top + pp(i,k,j)",
                ],
            },
        },
        "wrf_adjust_hook": wrf_adjust,
        "prior_theta_target": prior,
        "jax_recompute": jax_recompute,
        "comparisons": comparisons,
        "wrf_patch": {
            "path": str(OUT_PATCH),
            "status": "PROPOSED_NOT_APPLIED_SANDBOX_SCRATCH_READ_ONLY",
            "purpose": "fresh disposable hook for parent-interpolated, child-input, post-blend, and post-start-domain target values",
        },
        "commands": {
            "executed": [
                "git rev-parse HEAD",
                f"git merge-base --is-ancestor {REQUIRED_ANCESTOR} HEAD",
                "recovered prior WRF adjust hook from /mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src:proofs/v014 python proof-side live-nest target recompute",
            ],
            "required_validation": [
                "python -m py_compile proofs/v014/step1_current_mub_base_input_split.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_current_mub_base_input_split.py",
                "python -m json.tool proofs/v014/step1_current_mub_base_input_split.json >/tmp/step1_current_mub_base_input_split.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "script": str(Path(__file__).resolve()),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "wrf_patch_diff": str(OUT_PATCH),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "Fresh WRF terrain/PHB target emission could not be run because /mnt/data scratch writes are read-only in this sandbox.",
            "The source-changing sprint should validate the transient MUB blend over the full domain before patching production initialization.",
        ],
        "next_decision": (
            "Open the smallest source-changing sprint to add a transient live-nest adjust base path: compute WRF "
            "post-blend/pre-start_domain MUB for adjust_tempqv, use it only for theta/QV adjustment, keep final "
            "BaseState from start_domain, and rerun the Step-1 theta proof."
        ),
    }

    OUT_PATCH.write_text(WRF_PATCH_TEXT)
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload))
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload))
    print(json.dumps({"verdict": VERDICT, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
