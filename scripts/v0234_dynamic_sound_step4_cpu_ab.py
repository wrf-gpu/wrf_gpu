"""Bounded retained-A10 versus candidate-B4 d01 CPU discriminator.

The retained ordinary N=10 arm is authenticated from the completed root-MUDF
audit and is not rerun.  This script changes only the nested-loader's explicit
``GPUWRF_ACOUSTIC_SUBSTEPS`` input before import, compiles one ordinary B4
callable, and materializes progress only between completed dispatches.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "dynamic-sound-step4-complete-cpu-ab-proof.json"
BLOCKER = SPRINT / "dynamic-sound-step4-complete-cpu-ab-blocker.json"
RETAINED_A = SPRINT / "root-rk1-mudf-complete-cpu-ab-proof.json"
RETAINED_A_FILE_SHA256 = (
    "18072df793bd33806aec31db60041a89b053aaf3a86102e5364e38fb6315f65b"
)
RETAINED_A_CANONICAL_SHA256 = (
    "236701fb33d1123389bb021bc83fdd792839fa92cba28de9d52a4895263941b4"
)
NAMELIST_INPUT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "run/wrf/namelist.input"
)
NAMELIST_INPUT_SHA256 = (
    "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
)
NAMELIST_OUTPUT = NAMELIST_INPUT.with_name("namelist.output")
NAMELIST_OUTPUT_SHA256 = (
    "e35f1a0c4f0484fd93749485b7290a66476c46b7fec539697babf0090d4a76d0"
)
WRF_REGISTRY = Path("<USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON")
WRF_REGISTRY_SHA256 = (
    "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a"
)
WRF_SOLVE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/solve_em.F")
WRF_SOLVE_SHA256 = (
    "de58116cf306e40e60f9c9bb1aa74ed5090fbe8631e122ec73e77487ef55e271"
)
MODEL_PARENT = "395fb800df0bb619462db8e312f734ca0c538161"
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
    "GPUWRF_ACOUSTIC_SUBSTEPS": "4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", "-C", str(ROOT), *args), text=True).strip()


def _wrf_sound_steps(dt_s: float, spacing_m: float) -> int:
    """Literal non-adaptive ``solve_em.F`` time_step_sound=0 expression."""

    return max(2 * (int(300.0 * float(dt_s) / float(spacing_m) - 0.01) + 1), 4)


def _source_oracle(tree, dt_by_domain: dict[str, float]) -> dict[str, Any]:
    input_text = NAMELIST_INPUT.read_text()
    output_text = NAMELIST_OUTPUT.read_text()
    registry_text = WRF_REGISTRY.read_text()
    solve_text = WRF_SOLVE.read_text()

    if re.search(r"\btime_step_sound\b", input_text, flags=re.IGNORECASE):
        raise RuntimeError("canonical namelist.input unexpectedly sets time_step_sound")
    if re.search(r"\buse_adaptive_time_step\b", input_text, flags=re.IGNORECASE):
        raise RuntimeError(
            "canonical namelist.input unexpectedly sets use_adaptive_time_step"
        )
    if not re.search(
        r"USE_ADAPTIVE_TIME_STEP\s*=\s*F", output_text, flags=re.IGNORECASE
    ):
        raise RuntimeError("namelist.output does not authenticate non-adaptive mode")
    if not re.search(
        r"TIME_STEP_SOUND\s*=\s*11\*0", output_text, flags=re.IGNORECASE
    ):
        raise RuntimeError("namelist.output does not authenticate time_step_sound=0")
    if not re.search(
        r"integer\s+time_step_sound\s+namelist,dynamics\s+max_domains\s+0",
        registry_text,
        flags=re.IGNORECASE,
    ):
        raise RuntimeError("WRF Registry time_step_sound default changed")
    for needle in (
        "IF ( grid%time_step_sound == 0 ) THEN",
        "spacing = min(grid%dx, grid%dy)",
        "num_sound_steps = max ( 2 * ( INT (300. * grid%dt /  spacing",
        "number_of_small_timesteps = num_sound_steps/2",
        "number_of_small_timesteps = num_sound_steps",
    ):
        if needle not in solve_text:
            raise RuntimeError(f"WRF sound-step source changed: {needle!r}")

    domains: dict[str, Any] = {}
    for name in ("d01", "d02", "d03"):
        namelist = tree.domains[name].namelist
        dx = float(namelist.grid.projection.dx_m)
        dy = float(namelist.grid.projection.dy_m)
        spacing = min(dx, dy)
        dt = float(dt_by_domain[name])
        count = _wrf_sound_steps(dt, spacing)
        domains[name] = {
            "dt_s": dt,
            "dx_m": dx,
            "dy_m": dy,
            "spacing_m": spacing,
            "formula_argument_before_int": 300.0 * dt / spacing - 0.01,
            "wrf_sound_steps": count,
            "loaded_candidate_sound_steps": int(namelist.acoustic_substeps),
            "wrf_dts_s": dt / float(count),
            "wrf_schedule": [1, count // 2, count],
            "retained_schedule": [1, 5, 10],
            "wrf_stage_integrated_seconds": [dt / 3.0, dt / 2.0, dt],
            "candidate_stage_integrated_seconds": [
                dt / 3.0,
                (count // 2) * (dt / count),
                count * (dt / count),
            ],
            "retained_stage_integrated_seconds": [
                dt / 3.0,
                5 * (dt / 10.0),
                10 * (dt / 10.0),
            ],
        }
    return {
        "authority": {
            "namelist_input": {
                "path": str(NAMELIST_INPUT),
                "sha256": _sha256(NAMELIST_INPUT),
            },
            "namelist_output": {
                "path": str(NAMELIST_OUTPUT),
                "sha256": _sha256(NAMELIST_OUTPUT),
            },
            "registry": {"path": str(WRF_REGISTRY), "sha256": _sha256(WRF_REGISTRY)},
            "solve_em": {"path": str(WRF_SOLVE), "sha256": _sha256(WRF_SOLVE)},
        },
        "adaptive_timestep": False,
        "time_step_sound": 0,
        "formula": "max(2*(INT(300*dt/min(dx,dy)-0.01)+1),4)",
        "domains": domains,
    }


def _dispatch_chunked(
    executable,
    jax,
    jnp,
    carry,
    namelist,
    clock,
    *,
    start_step: int,
    terminal_step: int,
):
    from scripts import v0234_root_rk1_mudf_cpu_ab as common

    live = carry
    step = int(start_step)
    rows = []
    total_seconds = 0.0
    while step <= int(terminal_step):
        count = min(2, int(terminal_step) - step + 1)
        live, elapsed = common._dispatch(
            executable,
            jax,
            jnp,
            live,
            namelist,
            clock,
            count,
            start_step=step,
        )
        completed = step + count - 1
        total_seconds += elapsed
        rows.append(
            {
                "start_step": step,
                "n_steps": count,
                "completed_step": completed,
                "dispatch_and_materialize_seconds": elapsed,
            }
        )
        print(
            f"SOUND_STEP4_CPU_AB progress completed_d01_step={completed}/67",
            flush=True,
        )
        step = completed + 1
    return live, total_seconds, rows


def main() -> int:
    stage = "startup"
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        expected_hashes = {
            RETAINED_A: RETAINED_A_FILE_SHA256,
            NAMELIST_INPUT: NAMELIST_INPUT_SHA256,
            NAMELIST_OUTPUT: NAMELIST_OUTPUT_SHA256,
            WRF_REGISTRY: WRF_REGISTRY_SHA256,
            WRF_SOLVE: WRF_SOLVE_SHA256,
        }
        for path, expected in expected_hashes.items():
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(f"authority hash mismatch {path}: {actual}")
        if _git("diff", "--name-only", MODEL_PARENT, "HEAD", "--", "src/gpuwrf"):
            raise RuntimeError("model bytes changed after retained 395fb800 candidate")

        retained = json.loads(RETAINED_A.read_text())
        if retained.get("proof_sha256") != RETAINED_A_CANONICAL_SHA256:
            raise RuntimeError("retained A declared canonical hash changed")
        if _canonical(retained) != RETAINED_A_CANONICAL_SHA256:
            raise RuntimeError("retained A canonical payload changed")
        if retained.get("verdict") != "ROOT_RK1_MUDF_CPU_AB_RED":
            raise RuntimeError("retained A container verdict changed")
        retained_a = retained["retained_A"]
        retained_metrics = retained["step67_metrics"]["retained_A"]

        import jax
        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]

        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
        from scripts import v0234_root_rk1_mudf_cpu_ab as common

        scratch = Path(tempfile.mkdtemp(prefix="v0234-sound-step4-cpu-ab-"))
        stage = "canonical_load"
        tree, names, initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
            scratch
        )
        if names != ("d01", "d02", "d03"):
            raise RuntimeError(f"domain order changed: {names!r}")
        if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
            raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
        oracle = _source_oracle(tree, dt_by_domain)
        carry = initial["d01"]
        namelist = tree.domains["d01"].namelist
        clock = runtime.build_clock_base(namelist)
        initial_manifest = common._manifest(jax, np, carry)
        if initial_manifest["sha256"] != retained["input"]["manifest"]["sha256"]:
            raise RuntimeError("candidate B initial carry differs from retained A")

        stage = "candidate_B_lower_compile"
        print("SOUND_STEP4_CPU_AB candidate B lower/compile", flush=True)
        executable, audit = common._lower_compile_arm(
            runtime, jax, jnp, carry, namelist, clock
        )
        stage = "candidate_B_step1"
        print("SOUND_STEP4_CPU_AB candidate B dispatch step1", flush=True)
        step1, audit["step1_dispatch_seconds"] = common._dispatch(
            executable, jax, jnp, carry, namelist, clock, 1
        )
        stage = "candidate_B_chunked_steps_2_67"
        print("SOUND_STEP4_CPU_AB candidate B chunked dispatch through step67", flush=True)
        full, audit["step67_dispatch_seconds"], audit["progress"] = _dispatch_chunked(
            executable,
            jax,
            jnp,
            step1,
            namelist,
            clock,
            start_step=2,
            terminal_step=67,
        )
        del executable

        step1_manifest = common._manifest(jax, np, step1)
        full_manifest = common._manifest(jax, np, full)
        frames, frame_authority = common._load_frames(np)
        metrics = common._frame_metrics(np, full, frames)
        directional: dict[str, Any] = {}
        for anchor in ("cpu_wrf", "retry20"):
            directional[anchor] = {}
            for field in ("THM", "T", "U", "V"):
                a = float(retained_metrics[anchor][field]["ring1"])
                b = float(metrics[anchor][field]["ring1"])
                directional[anchor][field] = {
                    "retained_A10_ring1_rmse": a,
                    "candidate_B4_ring1_rmse": b,
                    "delta_B_minus_A": b - a,
                    "improves": b < a,
                    "not_worse": b <= a,
                }

        retained_full = retained_a["step67_manifest"]
        checks = {
            "authority_hashes_exact": all(
                _sha256(path) == expected for path, expected in expected_hashes.items()
            ),
            "retained_A_canonical_authenticated": _canonical(retained)
            == RETAINED_A_CANONICAL_SHA256,
            "all_domains_literal_wrf_count_4": all(
                row["wrf_sound_steps"] == 4
                for row in oracle["domains"].values()
            ),
            "all_domains_candidate_loaded_count_4": all(
                row["loaded_candidate_sound_steps"] == 4
                for row in oracle["domains"].values()
            ),
            "all_stage_integrated_durations_identical": all(
                row["wrf_stage_integrated_seconds"]
                == row["candidate_stage_integrated_seconds"]
                == row["retained_stage_integrated_seconds"]
                for row in oracle["domains"].values()
            ),
            "same_initial_carry_as_retained_A": initial_manifest["sha256"]
            == retained["input"]["manifest"]["sha256"],
            "candidate_interface_identity": audit["interface_identity"],
            "candidate_callback_transfer_unknown_free": not audit["forbidden_tokens"]
            and not audit["unknown_custom_call_targets"],
            "candidate_step1_active": step1_manifest["sha256"]
            != retained_a["step1_manifest"]["sha256"],
            "candidate_step67_active": full_manifest["sha256"]
            != retained_full["sha256"],
            "complete_leaf_structure_identity": all(
                left["path"] == right["path"]
                and left["shape"] == right["shape"]
                and left["dtype"] == right["dtype"]
                for left, right in zip(
                    retained_full["leaves"], full_manifest["leaves"], strict=True
                )
            ),
            "candidate_step67_all_finite": all(
                row["finite"] for row in full_manifest["leaves"]
            ),
            "THM_T_improve_both_anchors": all(
                directional[anchor][field]["improves"]
                for anchor in ("cpu_wrf", "retry20")
                for field in ("THM", "T")
            ),
            "U_V_not_worse_both_anchors": all(
                directional[anchor][field]["not_worse"]
                for anchor in ("cpu_wrf", "retry20")
                for field in ("U", "V")
            ),
        }
        verdict = (
            "DYNAMIC_SOUND_STEP4_CPU_AB_GREEN"
            if all(checks.values())
            else "DYNAMIC_SOUND_STEP4_CPU_AB_RED"
        )
        proof = {
            "schema": "gpuwrf.v0234.dynamic-sound-step4-complete-cpu-ab.v1",
            "environment": actual_env,
            "model_parent": MODEL_PARENT,
            "proof_tooling_commit": _git("rev-parse", "HEAD"),
            "retained_A10": {
                "path": str(RETAINED_A),
                "file_sha256": RETAINED_A_FILE_SHA256,
                "canonical_payload_sha256": RETAINED_A_CANONICAL_SHA256,
                "step1_manifest": retained_a["step1_manifest"],
                "step67_manifest": retained_full,
                "step67_metrics": retained_metrics,
            },
            "source_oracle": oracle,
            "load_authority": load_authority,
            "candidate_B4": {
                **audit,
                "domain": "d01",
                "acoustic_substeps": int(namelist.acoustic_substeps),
                "start": "2025-03-01T00:00:00+00:00",
                "terminal_step": 67,
                "terminal_time": "2025-03-01T01:00:18+00:00",
                "initial_manifest": initial_manifest,
                "step1_manifest": step1_manifest,
                "step67_manifest": full_manifest,
                "step67_metrics": metrics,
            },
            "frame_authority": frame_authority,
            "directional_gate": directional,
            "checks": checks,
            "verdict": verdict,
        }
        proof["proof_sha256"] = _canonical(proof)
        _atomic_json(OUT, proof)
        print(f"verdict={verdict}", flush=True)
        print(f"proof_sha256={proof['proof_sha256']}", flush=True)
        return 0 if verdict == "DYNAMIC_SOUND_STEP4_CPU_AB_GREEN" else 3
    except BaseException as exc:
        blocker = {
            "schema": "gpuwrf.v0234.dynamic-sound-step4-complete-cpu-ab-blocker.v1",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "verdict": "DYNAMIC_SOUND_STEP4_CPU_AB_BLOCKED",
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(BLOCKER, blocker)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
