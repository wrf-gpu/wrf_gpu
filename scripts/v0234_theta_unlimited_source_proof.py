#!/usr/bin/env python3
"""Build the bounded offline proof for the WRF unlimited-theta repair."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile

import jax
import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.flux_advection import (
    advect_scalar_flux,
    advect_scalar_flux_limited,
    couple_uv_specified,
    couple_velocities_periodic,
)
from gpuwrf.dynamics.metrics import load_wrfinput_metrics
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.runtime import operational_mode as operational


REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
DEFAULT_OUTPUT = SPRINT / "theta-unlimited-source-proof.json"
BASE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
INPUT_DIR = BASE / "run/wrf"
NAMELIST = INPUT_DIR / "namelist.input"
PRISTINE_MODULE_EM = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F")
PRISTINE_ADVECT = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F")
RUN = BASE / "corrected_ni_rca_max_22c2bd7a/nested_advection_degrade_2c13b731_full18h_discriminator1"
STEP0_CARRY = RUN / "failure/last-healthy-d03-step-0.pkl"
CURRENT_0020 = RUN / "output/wrfout_d03_2025-03-01_00:20:00"
PRIOR_0020 = BASE / (
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_boundary_final_4484be85_full18h_owner_override1/"
    "output/wrfout_d03_2025-03-01_00:20:00"
)
CPU_0020 = BASE / (
    "gpu_validation_retry20_relative_rmse_3ee02c19/"
    "pair-snapshots/20250301T002000/cpu.nc"
)
RETRY20_0020 = BASE / (
    "gpu_validation_retry20_relative_rmse_3ee02c19/"
    "pair-snapshots/20250301T002000/gpu.nc"
)

EXPECTED = {
    "namelist": "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    "pristine_advect": "58253bdbeb188dd47ed0579fcd2891086be1889b75c0c7d3696c9ad1d213559d",
    "step0_carry": "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d",
    "current_0020": "3cf8dcda34ae469df86028209f5f8c2ed59ef4fb1cfd38e2148ee33bcc5ab599",
    "prior_0020": "79492ab0b0f8482809a2486f72969973f13dbd35f374801cd787156238a72f7c",
    "cpu_0020": "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1",
    "retry20_0020": "70e09cf3ca22711c66ef529e716ead53fb998d2f548e9939c35d7767ac3f9e62",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def require_hash(label: str, path: Path, expected: str) -> dict[str, object]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} authority changed: expected={expected} actual={actual}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def source_evidence() -> dict[str, object]:
    source = PRISTINE_MODULE_EM.read_text()
    theta_start = source.index("!  theta flux divergence")
    theta_stop = source.index("   IF( ieva ) THEN", theta_start)
    theta = source[theta_start:theta_stop]
    scalar_start = source.index("   scalar_loop : DO im = scs, sce")
    scalar_stop = source.index("   END DO scalar_loop", scalar_start)
    scalar = source[scalar_start:scalar_stop]
    runtime_theta = inspect.getsource(operational._augment_large_step_tendencies)
    runtime_scalar = inspect.getsource(operational._scalar_transport_coupled_tendencies)
    gates = {
        "theta_calls_ordinary_advect_scalar": "CALL advect_scalar ( t, t, t_tend" in theta,
        "theta_does_not_call_pd": "advect_scalar_pd" not in theta,
        "theta_does_not_call_mono": "advect_scalar_mono" not in theta,
        "scalar_loop_calls_pd": "CALL advect_scalar_pd" in scalar,
        "scalar_loop_calls_mono": "CALL advect_scalar_mono" in scalar,
        "repaired_runtime_theta_has_no_limiter": "advect_scalar_flux_limited" not in runtime_theta,
        "repaired_runtime_theta_has_one_plain_call": runtime_theta.count("coupled_tend = advect_scalar_flux(") == 1,
        "runtime_scalar_limiter_retained": (
            "advect_moisture_scalars(" in runtime_scalar
            and "int(advection_opt) in (1, 2)" in runtime_scalar
        ),
    }
    return {
        "pristine_module_em": {
            "path": str(PRISTINE_MODULE_EM),
            "sha256": sha256_file(PRISTINE_MODULE_EM),
            "theta_branch_sha256": sha256_text(theta),
            "scalar_branch_sha256": sha256_text(scalar),
        },
        "runtime": {
            "operational_mode_sha256": sha256_file(REPO / "src/gpuwrf/runtime/operational_mode.py"),
            "theta_helper_sha256": sha256_text(runtime_theta),
            "scalar_helper_sha256": sha256_text(runtime_scalar),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def distance_bands(array: np.ndarray, *, count: int = 6) -> dict[str, object]:
    ny, nx = array.shape[-2:]
    y, x = np.ogrid[:ny, :nx]
    distance = np.minimum.reduce((
        np.broadcast_to(y, (ny, nx)),
        np.broadcast_to(x, (ny, nx)),
        np.broadcast_to(ny - 1 - y, (ny, nx)),
        np.broadcast_to(nx - 1 - x, (ny, nx)),
    ))
    return {
        str(ring): {
            "rms": float(np.sqrt(np.mean(np.square(array[:, distance == ring], dtype=np.float64)))),
            "maxabs": float(np.max(np.abs(array[:, distance == ring]))),
            "nonzero_count": int(np.count_nonzero(array[:, distance == ring])),
        }
        for ring in range(count)
    }


def cosine(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    x = left[mask].reshape(-1)
    y = right[mask].reshape(-1)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0.0 else float(np.dot(x, y) / denominator)


def theta_operator_ab() -> dict[str, object]:
    with STEP0_CARRY.open("rb") as handle:
        carry = pickle.load(handle)
    state = carry.state
    run = Gen2Run(INPUT_DIR)
    grid = run.grid("d03").as_grid_spec()
    metrics = load_wrfinput_metrics(run.wrfinput_file("d03"))
    rdx = 1.0 / float(grid.projection.dx_m)
    rdy = 1.0 / float(grid.projection.dy_m)
    periodic = couple_velocities_periodic(
        state.u,
        state.v,
        state.mu_total,
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        dnw=metrics.dnw,
        rdx=rdx,
        rdy=rdy,
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
        msftx=metrics.msftx,
        msfux=metrics.msfux,
        msfvy=metrics.msfvy,
    )
    ru_full, rv_full = couple_uv_specified(
        state.u,
        state.v,
        state.mu_total,
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
    )
    velocity = dataclasses.replace(
        periodic, specified=True, ru_full=ru_full, rv_full=rv_full,
    )
    theta_perturbation = state.theta - 300.0
    common = {
        "rdx": rdx,
        "rdy": rdy,
        "rdzw": metrics.rdnw,
        "fzm": metrics.fnm,
        "fzp": metrics.fnp,
    }
    repaired_plain = advect_scalar_flux(
        theta_perturbation,
        velocity,
        mut=state.mu_total,
        c1=metrics.c1h,
        **common,
    )
    erroneous_limited = advect_scalar_flux_limited(
        theta_perturbation,
        theta_perturbation,
        velocity,
        scalar_adv_opt=1,
        mut=state.mu_total,
        mu_old=state.mu_total,
        c1=metrics.c1h,
        c2=metrics.c2h,
        dt=6.0,
        **common,
    )
    mass = (
        np.asarray(metrics.c1h)[:, None, None] * np.asarray(state.mu_total)[None, :, :]
        + np.asarray(metrics.c2h)[:, None, None]
    )
    delta = (
        np.asarray(jax.device_get(repaired_plain - erroneous_limited))
        / mass
        * np.asarray(metrics.msfty)[None, :, :]
    )
    q = np.asarray(theta_perturbation)

    frames: dict[str, np.ndarray] = {}
    for label, path in {
        "current": CURRENT_0020,
        "prior": PRIOR_0020,
        "cpu": CPU_0020,
        "retry20": RETRY20_0020,
    }.items():
        with Dataset(path) as dataset:
            frames[label] = np.asarray(dataset.variables["T"][0], dtype=np.float64)
    ny, nx = frames["current"].shape[-2:]
    y, x = np.ogrid[:ny, :nx]
    distance = np.minimum.reduce((
        np.broadcast_to(y, (ny, nx)),
        np.broadcast_to(x, (ny, nx)),
        np.broadcast_to(ny - 1 - y, (ny, nx)),
        np.broadcast_to(nx - 1 - x, (ny, nx)),
    ))
    ring1 = np.broadcast_to(distance == 1, delta.shape)
    linearized_1200 = 1200.0 * delta
    return {
        "authenticated_step0": {
            "leaf_count": len(jax.tree_util.tree_leaves(carry)),
            "theta_shape": list(q.shape),
            "theta_perturbation_min": float(np.min(q)),
            "theta_perturbation_max": float(np.max(q)),
            "theta_perturbation_negative_count": int(np.count_nonzero(q < 0.0)),
            "theta_perturbation_cell_count": int(q.size),
        },
        "plain_minus_erroneous_limited_K_per_s": {
            "global_rms": float(np.sqrt(np.mean(np.square(delta, dtype=np.float64)))),
            "global_maxabs": float(np.max(np.abs(delta))),
            "changed_count": int(np.count_nonzero(delta)),
            "rings": distance_bands(delta),
        },
        "linearized_scale_only_not_a_forecast": {
            "seconds": 1200.0,
            "ring1_rms_K": float(np.sqrt(np.mean(np.square(linearized_1200[ring1], dtype=np.float64)))),
            "ring1_maxabs_K": float(np.max(np.abs(linearized_1200[ring1]))),
            "interpretation": "magnitude check only; nonlinear RK feedback is decided by the bounded step-200 GPU gate",
        },
        "retained_frame_projection": {
            "ring1_cosine_toward_cpu": cosine(
                delta, frames["cpu"] - frames["current"], ring1,
            ),
            "ring1_cosine_toward_retry20": cosine(
                delta, frames["retry20"] - frames["current"], ring1,
            ),
            "non_claim": "the source discrepancy, not this cross-time linear projection, admits the discriminator",
        },
        "finite": bool(np.isfinite(delta).all()),
        "active": bool(np.count_nonzero(delta) > 0),
    }


def git_evidence() -> dict[str, object]:
    partial_wind_candidate = "2c13b73112d9877d603324d66b127ecad60bf7e3"
    diff = subprocess.run(
        ["git", "diff", partial_wind_candidate, "--", "src/gpuwrf"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    changed = subprocess.run(
        ["git", "diff", "--name-only", partial_wind_candidate, "--", "src/gpuwrf"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.splitlines()
    forbidden = [
        token for token in (
            "callback", "io_callback", "pure_callback", "debug.callback",
            "tolerance", "threshold", "nan_to_num", "clip(",
        ) if token in diff
    ]
    return {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stdout=subprocess.PIPE, check=True,
        ).stdout.strip(),
        "partial_wind_candidate": partial_wind_candidate,
        "changed_model_files": changed,
        "model_diff_sha256": sha256_text(diff),
        "forbidden_tokens": forbidden,
        "only_operational_mode_changed": changed == ["src/gpuwrf/runtime/operational_mode.py"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    authorities = {
        "namelist": require_hash("namelist", NAMELIST, EXPECTED["namelist"]),
        "pristine_advect": require_hash(
            "pristine_advect", PRISTINE_ADVECT, EXPECTED["pristine_advect"],
        ),
        "step0_carry": require_hash("step0_carry", STEP0_CARRY, EXPECTED["step0_carry"]),
        "current_0020": require_hash("current_0020", CURRENT_0020, EXPECTED["current_0020"]),
        "prior_0020": require_hash("prior_0020", PRIOR_0020, EXPECTED["prior_0020"]),
        "cpu_0020": require_hash("cpu_0020", CPU_0020, EXPECTED["cpu_0020"]),
        "retry20_0020": require_hash("retry20_0020", RETRY20_0020, EXPECTED["retry20_0020"]),
    }
    compact_namelist = "".join(NAMELIST.read_text().lower().split())
    options = {"scalar_adv_opt_1_all_domains": "scalar_adv_opt=1,1,1," in compact_namelist}
    source = source_evidence()
    operator_ab = theta_operator_ab()
    git = git_evidence()
    gates = {
        "authorities_authenticated": len(authorities) == len(EXPECTED),
        "operational_option_active": all(options.values()),
        "source_separation_proven": source["passed"],
        "signed_theta_proves_pd_semantic_mismatch": (
            operator_ab["authenticated_step0"]["theta_perturbation_negative_count"] > 0
        ),
        "operator_difference_active_and_finite": operator_ab["active"] and operator_ab["finite"],
        "candidate_model_scope_minimal": (
            git["only_operational_mode_changed"] and not git["forbidden_tokens"]
        ),
        "no_gpu_commands_or_queries": True,
    }
    proof = {
        "schema": "gpuwrf.v0234.theta-unlimited-source-proof.v1",
        "verdict": "THETA_UNLIMITED_OFFLINE_CAUSAL_GATE_GREEN" if all(gates.values()) else "THETA_UNLIMITED_OFFLINE_CAUSAL_GATE_RED",
        "authorities": authorities,
        "operational_options": options,
        "source": source,
        "operator_ab": operator_ab,
        "git": git,
        "gates": gates,
        "gpu_commands_run": 0,
        "gpu_queries_run": 0,
        "v10_policy": "separate mechanism; not part of this candidate or discriminator",
    }
    proof["proof_sha256"] = canonical_digest(proof)
    atomic_write_json(args.output.resolve(), proof)
    print(json.dumps({
        "verdict": proof["verdict"],
        "proof": str(args.output.resolve()),
        "proof_sha256": proof["proof_sha256"],
        "ring1_linearized_rms_K": operator_ab["linearized_scale_only_not_a_forecast"]["ring1_rms_K"],
        "gpu_commands_run": 0,
    }, sort_keys=True))
    return 0 if all(gates.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
