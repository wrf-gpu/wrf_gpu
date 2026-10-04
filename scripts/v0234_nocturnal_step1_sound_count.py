"""Current-tree CPU-only d03 step-one discriminator for WRF sound-step count.

The Tenerife namelist leaves ``time_step_sound=0``.  Pristine WRF therefore
derives four sound steps on every domain, while the nested runtime selects ten.
This script advances the authenticated d03 Step0 carry once with each count and
compares end-of-step U/V directly with the output-neutral instrumented-WRF SP4
savepoint.  It is a directional localization proof, not a model-fix gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "NOCTURNAL_STEP1_SOUND_COUNT.json"
BLOCKER = SPRINT / "NOCTURNAL_STEP1_SOUND_COUNT_BLOCKER.json"
PRIOR = (
    ROOT
    / ".agent/sprints/2026-07-14-v0234-final-ni-fable5/"
    "dynamic-sound-step4-complete-cpu-ab-proof.json"
)
PRIOR_FILE_SHA256 = (
    "9e1894058a51456af1c2f3fe5fdc1ea37ac383abfce212cc02173648485fb553"
)
PRIOR_PROOF_SHA256 = (
    "0d03b8b6bcf692ee3934f8e37d315de6aa6b0036eeff88f7f57e039cef596daa"
)
WRF_NAMELIST = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/namelist.input"
)
WRF_NAMELIST_SHA256 = (
    "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
)
WRF_SOLVE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/solve_em.F")
WRF_SOLVE_SHA256 = (
    "de58116cf306e40e60f9c9bb1aa74ed5090fbe8631e122ec73e77487ef55e271"
)
REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_ENABLE_COMPILATION_CACHE": "false",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(np, candidate, truth) -> dict[str, Any]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    ny, nx = delta.shape[-2:]
    yy, xx = np.indices((ny, nx))
    distance = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))

    def rms(value) -> float:
        array = np.asarray(value, dtype=np.float64)
        return float(np.sqrt(np.mean(array * array)))

    return {
        "rmse": rms(delta),
        "bias": float(np.mean(delta)),
        "max_abs": float(np.max(np.abs(delta))),
        "lowest_level_rmse": rms(delta[0]),
        "lowest_three_level_rmse": rms(delta[:3]),
        "interior5_rmse": rms(delta[:, distance >= 5]),
        "outer5_rmse": rms(delta[:, distance < 5]),
        "vertical_rmse": [rms(delta[level]) for level in range(delta.shape[0])],
    }


def _wrf_receipt(reassemble, fields: tuple[str, ...]) -> dict[str, Any]:
    rows = []
    for rank in reassemble.load_ranks():
        rank_dir = Path(rank["dir"])
        for path in [rank_dir / "meta.txt"] + [
            rank_dir / f"step000001_{field}.f64" for field in fields
        ]:
            rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    rows.sort(key=lambda row: row["path"])
    return {
        "files": rows,
        "rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> int:
    stage = "startup"
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if os.environ.get("GPUWRF_ACOUSTIC_SUBSTEPS") is not None:
            raise RuntimeError("GPUWRF_ACOUSTIC_SUBSTEPS must be absent")
        expected = {
            common.STEP0: common.STEP0_SHA256,
            PRIOR: PRIOR_FILE_SHA256,
            WRF_NAMELIST: WRF_NAMELIST_SHA256,
            WRF_SOLVE: WRF_SOLVE_SHA256,
        }
        for path, digest in expected.items():
            if _sha256(path) != digest:
                raise RuntimeError(f"authority hash changed: {path}")
        prior = json.loads(PRIOR.read_text())
        if prior.get("proof_sha256") != PRIOR_PROOF_SHA256:
            raise RuntimeError("prior sound-count proof canonical hash changed")
        solve_text = WRF_SOLVE.read_text()
        for needle in (
            "IF ( grid%time_step_sound == 0 ) THEN",
            "spacing = min(grid%dx, grid%dy)",
            "num_sound_steps = max ( 2 * ( INT (300. * grid%dt /  spacing",
        ):
            if needle not in solve_text:
                raise RuntimeError(f"WRF source oracle changed: {needle!r}")

        stage = "imports"
        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
        if sorted(os.sched_getaffinity(0)) != [13, 14, 15, 29, 30, 31]:
            raise RuntimeError("CPU affinity changed")
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
        from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble

        stage = "load"
        scratch = Path(tempfile.mkdtemp(prefix="v0234-nocturnal-sound-step1-"))
        try:
            load_dir = scratch / "load"
            load_dir.mkdir()
            tree, names, _initial, dt_by_domain, load_authority = (
                ordinary.load_corrected_tree(load_dir)
            )
            if names != ("d01", "d02", "d03") or dt_by_domain != {
                "d01": 54.0,
                "d02": 18.0,
                "d03": 6.0,
            }:
                raise RuntimeError("canonical hierarchy changed")
            loaded = tree.domains["d03"].namelist
            if (
                int(loaded.moist_adv_opt),
                int(loaded.scalar_adv_opt),
                int(loaded.acoustic_substeps),
            ) != (1, 1, 10):
                raise RuntimeError("current loader controls changed")
            wrf_count = max(
                2
                * (
                    int(
                        300.0
                        * float(loaded.dt_s)
                        / min(
                            float(loaded.grid.projection.dx_m),
                            float(loaded.grid.projection.dy_m),
                        )
                        - 0.01
                    )
                    + 1
                ),
                4,
            )
            if wrf_count != 4:
                raise RuntimeError(f"literal WRF sound count changed: {wrf_count}")
            with common.STEP0.open("rb") as stream:
                carry = pickle.load(stream)

            truth = {
                "U": reassemble.reassemble3d("sp4_exit__u", 1),
                "V": reassemble.reassemble3d("sp4_exit__v", 1),
            }
            receipt = _wrf_receipt(
                reassemble, ("sp4_exit__u", "sp4_exit__v")
            )
            arms: dict[str, Any] = {}
            outputs: dict[str, Any] = {}
            for label, count in (("runtime_N10", 10), ("wrf_N4", 4)):
                stage = f"run_{label}"
                namelist = dataclasses.replace(loaded, acoustic_substeps=count)
                clock = runtime.build_clock_base(namelist)
                result, audit = common._run_arm(
                    runtime, jax, jnp, carry, namelist, clock
                )
                manifest = common._manifest(jax, np, result)
                fields = {
                    "U": _metric(np, result.state.u, truth["U"]),
                    "V": _metric(np, result.state.v, truth["V"]),
                }
                arms[label] = {
                    "acoustic_substeps": count,
                    "dt_sound_s": float(namelist.dt_s) / count,
                    "audit": audit,
                    "output_manifest_sha256": manifest["sha256"],
                    "output_leaf_count": manifest["leaf_count"],
                    "output_all_finite": all(row["finite"] for row in manifest["leaves"]),
                    "fields": fields,
                }
                outputs[label] = result

            stage = "decision"
            dimensions = (
                "rmse",
                "lowest_level_rmse",
                "lowest_three_level_rmse",
                "interior5_rmse",
            )
            directional = {}
            for field in ("U", "V"):
                directional[field] = {}
                for dimension in dimensions:
                    retained = arms["runtime_N10"]["fields"][field][dimension]
                    candidate = arms["wrf_N4"]["fields"][field][dimension]
                    directional[field][dimension] = {
                        "runtime_N10": retained,
                        "wrf_N4": candidate,
                        "delta_N4_minus_N10": candidate - retained,
                        "ratio_N4_over_N10": candidate / retained,
                        "improves": candidate < retained,
                    }
            directional_green = all(
                row[dimension]["improves"]
                for row in directional.values()
                for dimension in dimensions
            )
            output_delta = common._state_delta(
                np, outputs["runtime_N10"], outputs["wrf_N4"]
            )
            checks = {
                "authority_hashes_exact": True,
                "current_loader_opts11_N10": True,
                "literal_wrf_count_is_4": True,
                "both_arms_interface_106": all(
                    arm["audit"]["interface_identity"]
                    and arm["output_leaf_count"] == 106
                    for arm in arms.values()
                ),
                "both_arms_finite": all(
                    arm["output_all_finite"] for arm in arms.values()
                ),
                "both_arms_callback_transfer_unknown_free": all(
                    not arm["audit"]["forbidden_tokens"]
                    and not arm["audit"]["unknown_custom_call_targets"]
                    for arm in arms.values()
                ),
                "N4_changes_the_program": bool(output_delta["changed_fields"]),
                "N4_improves_U_V_full_low3_interior": directional_green,
            }
            verdict = (
                "CURRENT_STEP1_WRF_N4_DIRECTIONAL_GREEN__C4_REOPENED"
                if all(checks.values())
                else "CURRENT_STEP1_WRF_N4_NOT_DIRECTIONALLY_GREEN__PRIOR_RED_STANDS"
            )
            proof = {
                "schema": "gpuwrf.v0234.nocturnal-step1-sound-count.v1",
                "verdict": verdict,
                "scope": (
                    "CPU-only current-tree d03 Step0-to-step1 directional A/B; "
                    "not a correction, causal trajectory, or release gate"
                ),
                "source_oracle": {
                    "time_step_sound": 0,
                    "formula": "max(2*(INT(300*dt/min(dx,dy)-0.01)+1),4)",
                    "d03_dt_s": float(loaded.dt_s),
                    "d03_spacing_m": min(
                        float(loaded.grid.projection.dx_m),
                        float(loaded.grid.projection.dy_m),
                    ),
                    "wrf_sound_steps": wrf_count,
                    "runtime_sound_steps": int(loaded.acoustic_substeps),
                    "namelist": {"path": str(WRF_NAMELIST), "sha256": _sha256(WRF_NAMELIST)},
                    "solve_em": {"path": str(WRF_SOLVE), "sha256": _sha256(WRF_SOLVE)},
                },
                "prior_dual_anchor_result": {
                    "path": str(PRIOR),
                    "file_sha256": _sha256(PRIOR),
                    "proof_sha256": prior["proof_sha256"],
                    "verdict": prior["verdict"],
                    "cpu_wrf_all_T_THM_U_V_improved": all(
                        row["improves"]
                        for row in prior["directional_gate"]["cpu_wrf"].values()
                    ),
                    "retry20_all_T_THM_U_V_improved": all(
                        row["improves"]
                        for row in prior["directional_gate"]["retry20"].values()
                    ),
                },
                "authority": {
                    "step0": {"path": str(common.STEP0), "sha256": _sha256(common.STEP0)},
                    "wrf_sp4_receipt": receipt,
                    "load_authority_sha256": common._canonical(load_authority),
                },
                "source_sha256": {
                    "nested_pipeline.py": _sha256(ROOT / "src/gpuwrf/integration/nested_pipeline.py"),
                    "operational_mode.py": _sha256(ROOT / "src/gpuwrf/runtime/operational_mode.py"),
                },
                "arms": arms,
                "directional_comparison": directional,
                "N4_minus_N10_state_delta": output_delta,
                "checks": checks,
                "gpu_commands": 0,
                "gpu_queries": 0,
                "wrf_or_mpi_executions": 0,
            }
            proof["proof_sha256"] = common._canonical(proof)
            common._atomic_json(OUT, proof)
            print(
                json.dumps(
                    {
                        "verdict": verdict,
                        "proof_sha256": proof["proof_sha256"],
                        "directional_comparison": directional,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0 if all(checks.values()) else 3
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except Exception as exc:
        blocker = {
            "schema": "gpuwrf.v0234.nocturnal-step1-sound-count-blocker.v1",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_commands": 0,
            "gpu_queries": 0,
            "wrf_or_mpi_executions": 0,
        }
        blocker["proof_sha256"] = common._canonical(blocker)
        common._atomic_json(BLOCKER, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
