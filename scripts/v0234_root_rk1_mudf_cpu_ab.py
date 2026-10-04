"""Authenticated CPU A/B for pristine-WRF RK1 ``MUDF=0`` ownership.

The experiment is deliberately parent-first.  It runs the ordinary d01
callable from the canonical corrected hierarchy, proves the first step is
identical, checks the exact stale-MUDF impulse on the real step-1 carry, and
then advances both arms to d01 step 67 (01:00:18) for directional comparison
against CPU WRF and Retry20.  It does not import, query, compile, or run CUDA.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "root-rk1-mudf-complete-cpu-ab-proof.json"
BLOCKER = SPRINT / "root-rk1-mudf-complete-cpu-ab-blocker.json"
CASE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
FRAMES = {
    "cpu_wrf": (
        CASE / "run/wrf/wrfout_d01_2025-03-01_01:00:18",
        "7ac0b67385ce6a060042128d34261f2880b82d25fe631b7481759ea374d17ea2",
    ),
    "retry20": (
        CASE
        / "gpu_validation_retry20_relative_rmse_3ee02c19/gpu-output/"
        "wrfout_d01_2025-03-01_01:00:18",
        "efe8e0df86242de5e1f82f2028735ae10abeb36e3cb4fa5948f13a041088f474",
    ),
    "immutable_448": (
        CASE
        / "corrected_ni_rca_max_22c2bd7a/"
        "nested_boundary_final_4484be85_full18h_owner_override1/output/"
        "wrfout_d01_2025-03-01_01:00:18",
        "b71d992e273baa103b380c34fa2769d1cd288e1d76fbac2403ed8bec8680dcea",
    ),
}
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}
ALLOWED_CPU_CUSTOM_TARGETS = {"lapack_dgtsv_ffi"}
RVOVRD = 461.6 / 287.0


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


def _path_name(path: tuple[Any, ...]) -> str:
    parts = []
    for key in path:
        for attr in ("name", "key", "idx"):
            if hasattr(key, attr):
                parts.append(str(getattr(key, attr)))
                break
        else:
            parts.append(str(key))
    return ".".join(parts)


def _manifest(jax, np, value: Any) -> dict[str, Any]:
    rows = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        array = np.asarray(leaf)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
        rows.append(
            {
                "path": _path_name(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": digest.hexdigest(),
                "finite": bool(
                    not np.issubdtype(array.dtype, np.floating)
                    or np.all(np.isfinite(array))
                ),
            }
        )
    return {
        "leaf_count": len(rows),
        "leaves": rows,
        "sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _rms(np, value) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _ring(np, ny: int, nx: int, distance: int) -> Any:
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) == distance


def _lower_compile_arm(runtime, jax, jnp, carry, namelist, clock) -> tuple[Any, dict[str, Any]]:
    runtime._advance_chunk_fori.clear_cache()
    jax.clear_caches()
    start_step = jnp.asarray(1, dtype=jnp.int32)
    started = time.perf_counter()
    lowered = runtime._advance_chunk_fori.lower(
        carry,
        namelist,
        start_step,
        clock,
        n_steps=1,
        cadence=int(namelist.radiation_cadence_steps),
    )
    lower_seconds = time.perf_counter() - started
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    stablehlo_lower = stablehlo.lower()
    custom_targets = sorted(
        set(re.findall(r'call_target_name\s*=\s*"([^"]+)"', stablehlo))
    )
    unknown_targets = sorted(set(custom_targets) - ALLOWED_CPU_CUSTOM_TARGETS)
    forbidden_tokens = sorted(
        token
        for token in (
            "xla_python_cpu_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "outside_compilation",
            "send_to_host",
            "recv_from_host",
        )
        if token in stablehlo_lower
    )
    input_tree = jax.tree_util.tree_structure(carry)
    output_tree = jax.tree_util.tree_structure(lowered.out_info)
    input_avals = [
        (tuple(value.shape), str(value.dtype))
        for value in jax.tree_util.tree_leaves(carry)
    ]
    output_avals = [
        (tuple(value.shape), str(value.dtype))
        for value in jax.tree_util.tree_leaves(lowered.out_info)
    ]
    started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    return executable, {
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
        "stablehlo_bytes": len(stablehlo.encode()),
        "custom_call_targets": custom_targets,
        "unknown_custom_call_targets": unknown_targets,
        "forbidden_tokens": forbidden_tokens,
        "interface_identity": bool(
            input_tree == output_tree and input_avals == output_avals
        ),
        "input_leaf_count": len(input_avals),
        "output_leaf_count": len(output_avals),
    }


def _dispatch(
    executable,
    jax,
    jnp,
    carry,
    namelist,
    clock,
    n_steps: int,
    *,
    start_step: int = 1,
):
    started = time.perf_counter()
    result = executable(
        carry,
        namelist,
        jnp.asarray(int(start_step), dtype=jnp.int32),
        clock,
        n_steps=int(n_steps),
        cadence=int(namelist.radiation_cadence_steps),
    )
    result = jax.device_get(result)
    return result, time.perf_counter() - started


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
    max_chunk: int = 2,
):
    """Advance with the ordinary callable and synchronize bounded progress.

    This is CPU-audit orchestration outside production HLO.  Each dispatch still
    runs the unchanged compiled fori-loop callable; materialization occurs only
    between completed dispatches to avoid the retained single-67-step CPU
    ``device_get`` stall.  No production carry, callback, or loop transfer is
    added.
    """

    live = carry
    step = int(start_step)
    rows = []
    total_seconds = 0.0
    while step <= int(terminal_step):
        count = min(int(max_chunk), int(terminal_step) - step + 1)
        live, elapsed = _dispatch(
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
            f"ROOT_MUDF_CPU_AB progress completed_d01_step={completed}/67",
            flush=True,
        )
        step = completed + 1
    return live, total_seconds, rows


def _load_frames(np) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from netCDF4 import Dataset

    loaded = {}
    authority = {}
    for name, (path, expected) in FRAMES.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frame hash mismatch {name}: {actual}")
        with Dataset(path) as dataset:
            loaded[name] = {
                field: np.asarray(dataset.variables[field][0], dtype=np.float64)
                for field in ("T", "THM", "QVAPOR", "U", "V")
            }
        authority[name] = {"path": str(path), "sha256": actual}
    return loaded, authority


def _frame_metrics(np, carry, frames) -> dict[str, Any]:
    thm = np.asarray(carry.state.theta, dtype=np.float64) - 300.0
    qv = np.asarray(carry.state.qv, dtype=np.float64)
    diagnostic_t = (thm + 300.0) / (1.0 + RVOVRD * np.maximum(qv, 0.0)) - 300.0
    values = {
        "THM": thm,
        "T": diagnostic_t,
        "QVAPOR": qv,
        "U": np.asarray(carry.state.u, dtype=np.float64),
        "V": np.asarray(carry.state.v, dtype=np.float64),
    }
    result: dict[str, Any] = {}
    for anchor in ("cpu_wrf", "retry20"):
        result[anchor] = {}
        for field, value in values.items():
            ny, nx = value.shape[-2:]
            bands = {
                "ring0": _ring(np, ny, nx, 0),
                "ring1": _ring(np, ny, nx, 1),
                "ring2": _ring(np, ny, nx, 2),
            }
            result[anchor][field] = {
                band: _rms(np, value[..., mask] - frames[anchor][field][..., mask])
                for band, mask in bands.items()
            }
    return result


def _literal_mudf_oracle(acoustic, runtime, jax, jnp, np, carry, namelist):
    base = runtime._acoustic_core_state(carry, namelist)
    base = jax.tree_util.tree_map(
        lambda value: (
            jnp.asarray(value)
            if value is not None and hasattr(value, "shape")
            else value
        ),
        base,
        is_leaf=lambda value: value is None,
    )
    stale = base.replace(mudf=jnp.asarray(carry.mudf, dtype=jnp.float64))
    zero = base.replace(mudf=jnp.zeros_like(stale.mudf))
    kwargs = {
        "dts_rk": float(namelist.dt_s) / 3.0,
        "dx": float(namelist.grid.projection.dx_m),
        "dy": float(namelist.grid.projection.dy_m),
        "top_lid": bool(namelist.top_lid),
        "emdiv": 0.01,
        "dt_full": float(namelist.dt_s),
    }
    out_stale = acoustic.advance_uv_wrf(stale, **kwargs)
    out_zero = acoustic.advance_uv_wrf(zero, **kwargs)
    mudf = jnp.asarray(carry.mudf, dtype=jnp.float64)
    mudf_l_x, mudf_r_x = acoustic._x_face_pair_2d(mudf)
    expected_u = stale.c1h[:, None, None] * (
        -0.01
        * float(namelist.grid.projection.dx_m)
        * (mudf_r_x - mudf_l_x)[None, :, :]
        / stale.msfuy[None, :, :]
    )
    mudf_s_y, mudf_n_y = acoustic._y_face_pair_2d(mudf)
    expected_v = stale.c1h[:, None, None] * (
        -0.01
        * float(namelist.grid.projection.dy_m)
        * (mudf_n_y - mudf_s_y)[None, :, :]
        * stale.msfvx_inv[None, :, :]
    )
    actual_u = out_stale.u - out_zero.u
    actual_v = out_stale.v - out_zero.v
    return {
        "input_mudf_rms": _rms(np, mudf),
        "input_mudf_max_abs": float(np.max(np.abs(np.asarray(mudf)))),
        "u_impulse_rms": _rms(np, actual_u),
        "v_impulse_rms": _rms(np, actual_v),
        "u_literal_max_abs_error": float(
            np.max(np.abs(np.asarray(actual_u - expected_u)))
        ),
        "v_literal_max_abs_error": float(
            np.max(np.abs(np.asarray(actual_v - expected_v)))
        ),
    }


def main() -> int:
    stage = "startup"
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")

        import jax
        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]

        import gpuwrf.dynamics.core.acoustic as acoustic
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

        scratch = Path(tempfile.mkdtemp(prefix="v0234-root-mudf-cpu-ab-"))
        load_dir = scratch / "load"
        load_dir.mkdir()
        tree, names, initial, dt_by_domain, load_authority = ordinary.load_corrected_tree(
            load_dir
        )
        if names != ("d01", "d02", "d03"):
            raise RuntimeError(f"domain order changed: {names!r}")
        if dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
            raise RuntimeError(f"timestep hierarchy changed: {dt_by_domain!r}")
        carry = initial["d01"]
        namelist = tree.domains["d01"].namelist
        clock = runtime.build_clock_base(namelist)
        initial_manifest = _manifest(jax, np, carry)
        initial_mudf_zero = bool(np.count_nonzero(np.asarray(carry.mudf)) == 0)
        if not initial_mudf_zero:
            raise RuntimeError("canonical d01 initial MUDF is not exact zero")

        original_selector = runtime._stage_entry_mudf
        stage = "retained_A_lower_compile"
        print("ROOT_MUDF_CPU_AB retained A lower/compile", flush=True)
        executable_a, audit_a = _lower_compile_arm(
            runtime, jax, jnp, carry, namelist, clock
        )
        stage = "retained_A_step1"
        print("ROOT_MUDF_CPU_AB retained A dispatch step1", flush=True)
        step1_a, audit_a["step1_dispatch_seconds"] = _dispatch(
            executable_a, jax, jnp, carry, namelist, clock, 1
        )
        stage = "retained_A_chunked_steps_2_67"
        print("ROOT_MUDF_CPU_AB retained A chunked dispatch through step67", flush=True)
        full_a, audit_a["step67_dispatch_seconds"], audit_a["progress"] = _dispatch_chunked(
            executable_a,
            jax,
            jnp,
            step1_a,
            namelist,
            clock,
            start_step=2,
            terminal_step=67,
        )
        del executable_a

        def wrf_selector(carried_mudf, *, rk_step: int, nested_frozen_bundle: bool):
            del nested_frozen_bundle
            if int(rk_step) == 1:
                return jnp.zeros_like(carried_mudf, dtype=jnp.float64)
            return carried_mudf.astype(jnp.float64)

        runtime._stage_entry_mudf = wrf_selector
        try:
            stage = "candidate_B_lower_compile"
            print("ROOT_MUDF_CPU_AB candidate B lower/compile", flush=True)
            executable_b, audit_b = _lower_compile_arm(
                runtime, jax, jnp, carry, namelist, clock
            )
            stage = "candidate_B_step1"
            print("ROOT_MUDF_CPU_AB candidate B dispatch step1", flush=True)
            step1_b, audit_b["step1_dispatch_seconds"] = _dispatch(
                executable_b, jax, jnp, carry, namelist, clock, 1
            )
            stage = "candidate_B_chunked_steps_2_67"
            print("ROOT_MUDF_CPU_AB candidate B chunked dispatch through step67", flush=True)
            full_b, audit_b["step67_dispatch_seconds"], audit_b["progress"] = _dispatch_chunked(
                executable_b,
                jax,
                jnp,
                step1_b,
                namelist,
                clock,
                start_step=2,
                terminal_step=67,
            )
            del executable_b
        finally:
            runtime._stage_entry_mudf = original_selector

        step1_a_manifest = _manifest(jax, np, step1_a)
        step1_b_manifest = _manifest(jax, np, step1_b)
        full_a_manifest = _manifest(jax, np, full_a)
        full_b_manifest = _manifest(jax, np, full_b)
        oracle = _literal_mudf_oracle(
            acoustic, runtime, jax, jnp, np, step1_a, namelist
        )
        frames, frame_authority = _load_frames(np)
        metrics_a = _frame_metrics(np, full_a, frames)
        metrics_b = _frame_metrics(np, full_b, frames)

        directional = {}
        for anchor in ("cpu_wrf", "retry20"):
            directional[anchor] = {
                field: {
                    "retained_A_ring1_rmse": metrics_a[anchor][field]["ring1"],
                    "candidate_B_ring1_rmse": metrics_b[anchor][field]["ring1"],
                    "improves": metrics_b[anchor][field]["ring1"]
                    < metrics_a[anchor][field]["ring1"],
                }
                for field in ("THM", "T", "U", "V")
            }

        checks = {
            "canonical_initial_mudf_exact_zero": initial_mudf_zero,
            "step1_complete_carry_bit_identical": step1_a_manifest["sha256"]
            == step1_b_manifest["sha256"],
            "step1_nonzero_mudf_for_next_step": oracle["input_mudf_rms"] > 0.0,
            "literal_u_impulse_matches_fp64": oracle["u_literal_max_abs_error"]
            <= 2.0e-13,
            "literal_v_impulse_matches_fp64": oracle["v_literal_max_abs_error"]
            <= 2.0e-13,
            "retained_interface_identity": audit_a["interface_identity"],
            "candidate_interface_identity": audit_b["interface_identity"],
            "complete_leaf_structure_identity": all(
                a["path"] == b["path"]
                and a["shape"] == b["shape"]
                and a["dtype"] == b["dtype"]
                for a, b in zip(
                    full_a_manifest["leaves"],
                    full_b_manifest["leaves"],
                    strict=True,
                )
            ),
            "both_step67_all_finite": all(
                row["finite"] for row in full_a_manifest["leaves"]
            )
            and all(row["finite"] for row in full_b_manifest["leaves"]),
            "both_callback_transfer_unknown_free": not audit_a["forbidden_tokens"]
            and not audit_b["forbidden_tokens"]
            and not audit_a["unknown_custom_call_targets"]
            and not audit_b["unknown_custom_call_targets"],
            "candidate_active_by_step67": full_a_manifest["sha256"]
            != full_b_manifest["sha256"],
            "THM_T_U_improve_both_anchors": all(
                directional[anchor][field]["improves"]
                for anchor in ("cpu_wrf", "retry20")
                for field in ("THM", "T", "U")
            ),
            "V_not_worse_both_anchors": all(
                metrics_b[anchor]["V"]["ring1"]
                <= metrics_a[anchor]["V"]["ring1"]
                for anchor in ("cpu_wrf", "retry20")
            ),
        }
        verdict = (
            "ROOT_RK1_MUDF_CPU_AB_GREEN"
            if all(checks.values())
            else "ROOT_RK1_MUDF_CPU_AB_RED"
        )
        proof = {
            "schema": "gpuwrf.v0234.root-rk1-mudf-complete-cpu-ab.v1",
            "environment": actual_env,
            "load_authority": load_authority,
            "input": {
                "domain": "d01",
                "dt_s": float(namelist.dt_s),
                "start": "2025-03-01T00:00:00+00:00",
                "terminal_step": 67,
                "terminal_time": "2025-03-01T01:00:18+00:00",
                "manifest": initial_manifest,
            },
            "source_mechanism": {
                "wrf": "module_small_step_em.F RK1 sets MUDF=0 before advance_uv",
                "retained": "root d01 imports prior-step carry.mudf at RK1",
                "candidate": "all domains zero RK1; RK2/RK3 retain within-step MUDF",
            },
            "literal_real_step2_oracle": oracle,
            "retained_A": {**audit_a, "step1_manifest": step1_a_manifest, "step67_manifest": full_a_manifest},
            "candidate_B": {**audit_b, "step1_manifest": step1_b_manifest, "step67_manifest": full_b_manifest},
            "frame_authority": frame_authority,
            "step67_metrics": {"retained_A": metrics_a, "candidate_B": metrics_b},
            "directional_gate": directional,
            "checks": checks,
            "verdict": verdict,
        }
        proof["proof_sha256"] = _canonical(proof)
        _atomic_json(OUT, proof)
        print(f"verdict={verdict}", flush=True)
        print(f"proof_sha256={proof['proof_sha256']}", flush=True)
        return 0 if verdict == "ROOT_RK1_MUDF_CPU_AB_GREEN" else 3
    except BaseException as exc:
        blocker = {
            "schema": "gpuwrf.v0234.root-rk1-mudf-complete-cpu-ab-blocker.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stage": stage,
            "traceback": traceback.format_exc(),
            "verdict": "ROOT_RK1_MUDF_CPU_AB_BLOCKED",
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(BLOCKER, blocker)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
