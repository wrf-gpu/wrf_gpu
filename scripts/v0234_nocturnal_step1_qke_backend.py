"""CPU-only Step1 discriminator for the cold-start QKE backend split.

The repaired H5 GPU arm failed closed because the current GPU cold-load QKE
does not hash byte-identically to the independently reconstructed JAX-CPU
field.  Historical GPU Step0 authority shows that the split is concentrated
in the lowest four levels over sea.  This proof holds every retained Step0
leaf fixed, replaces only QKE with the authenticated current-tree CPU cold
load, advances one current operational step, and compares with the already
sealed retained-GPU-QKE Step1 arm and the output-neutral WRF SP4 savepoint.

No WRF/MPI executable, GPU command/query, or trajectory is run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from scripts import v0234_nocturnal_step1_pbl_ablation as pbl
from scripts import v0234_nocturnal_step1_sound_count as sound
from scripts import v0234_t2save_ownership_cpu_ab as common
from scripts import v0234_v10_rootcause_step200 as step200


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "NOCTURNAL_STEP1_QKE_BACKEND.json"
BLOCKER = SPRINT / "NOCTURNAL_STEP1_QKE_BACKEND_BLOCKER.json"
PRIOR = SPRINT / "NOCTURNAL_STEP1_SOUND_COUNT.json"
PRIOR_FILE_SHA256 = (
    "82482bc63f4db3e31e2dcb38c8ad086bc460d769736d82f10e9ca8a369f0130d"
)
PRIOR_PROOF_SHA256 = (
    "7aa5d21eef1335eb38a7c0a285dde72869160b56bf4a69d931456cdfa4eade29"
)
COLD_PROOF = SPRINT / "COLD_STEP0_ANCHOR_REPAIR.json"
COLD_PROOF_FILE_SHA256 = (
    "e98c689ad9aab02a59e4eb4896e92bb4b34a7cd85371d766c332e0df6a8beeef"
)
COLD_PROOF_SHA256 = (
    "eada51b566dfe9497261226fd6e29b43278a8a9bf0f4e4d77e46325ca2d3aaad"
)
CURRENT_CPU_QKE_SHA256 = (
    "07ea2a840d9a6793eb2822640eb75dd5c7d7adacf7bf11ef96020ab7f0f5ac95"
)
RETAINED_GPU_QKE_SHA256 = (
    "cce1430d05888b6c0a617d4ee264b08b4b6a1cd85d735b4eae6061a2a0376546"
)
EXPECTED_COLD_RETAINED_MISMATCHES = ("mavail", "qke", "roughness_m")
SOURCE_SHA256 = dict(pbl.SOURCE_SHA256)
REQUIRED_ENV = dict(sound.REQUIRED_ENV)
PRIMARY_DIMENSIONS = pbl.PRIMARY_DIMENSIONS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rms(np: Any, value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def _qke_delta(np: Any, cpu_qke: Any, retained_qke: Any, xland: Any) -> dict[str, Any]:
    cpu = np.asarray(cpu_qke, dtype=np.float64)
    retained = np.asarray(retained_qke, dtype=np.float64)
    delta = cpu - retained
    # WRF XLAND convention: 1=land, 2=water.
    land = np.asarray(xland) < 1.5
    rows = []
    for level in range(delta.shape[0]):
        slab = delta[level]
        rows.append(
            {
                "level": level,
                "rms": _rms(np, slab),
                "max_abs": float(np.max(np.abs(slab))),
                "land_rms": _rms(np, slab[land]),
                "sea_rms": _rms(np, slab[~land]),
            }
        )
    return {
        "shape": list(delta.shape),
        "unequal_cells": int(np.count_nonzero(delta)),
        "total_cells": int(delta.size),
        "rms": _rms(np, delta),
        "max_abs": float(np.max(np.abs(delta))),
        "mean": float(np.mean(delta, dtype=np.float64)),
        "abs_quantiles": {
            str(q): float(np.quantile(np.abs(delta), q))
            for q in (0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)
        },
        "vertical": rows,
    }


def _field_comparison(
    retained: dict[str, Any], cpu_seed: dict[str, Any]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for field in ("U", "V"):
        rows[field] = {}
        for dimension in PRIMARY_DIMENSIONS:
            old = float(retained[field][dimension])
            new = float(cpu_seed[field][dimension])
            rows[field][dimension] = {
                "retained_GPU_qke_seed": old,
                "current_CPU_qke_seed": new,
                "CPU_seed_minus_GPU_seed": new - old,
                "absolute_change": abs(new - old),
            }
    return rows


def main() -> int:
    stage = "startup"
    scratch: Path | None = None
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if sorted(os.sched_getaffinity(0)) != [13, 14, 15, 29, 30, 31]:
            raise RuntimeError("CPU affinity changed")
        if _sha256(PRIOR) != PRIOR_FILE_SHA256:
            raise RuntimeError("sealed retained-QKE Step1 proof changed")
        prior = json.loads(PRIOR.read_text())
        if (
            prior.get("proof_sha256") != PRIOR_PROOF_SHA256
            or prior.get("verdict")
            != "CURRENT_STEP1_WRF_N4_DIRECTIONAL_GREEN__C4_REOPENED"
        ):
            raise RuntimeError("sealed retained-QKE Step1 semantics changed")
        if _sha256(COLD_PROOF) != COLD_PROOF_FILE_SHA256:
            raise RuntimeError("cold Step0 repair proof file changed")
        cold_proof = json.loads(COLD_PROOF.read_text())
        if (
            cold_proof.get("proof_sha256") != COLD_PROOF_SHA256
            or cold_proof.get("verdict")
            != (
                "PINNED_BASE_AND_REPAIRED_CANDIDATE_COLD_STEP0_BYTE_EXACT__"
                "HISTORICAL_THREE_LEAF_DRIFT_AUTHORIZED"
            )
        ):
            raise RuntimeError("cold Step0 repair semantics changed")
        if _sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("retained GPU Step0 changed")
        for relative, expected in SOURCE_SHA256.items():
            actual = _sha256(ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"current model source changed: {relative}: {actual}")

        stage = "imports"
        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)
        import jax.numpy as jnp
        import numpy as np

        if jax.default_backend() != "cpu":
            raise RuntimeError(f"unexpected backend: {jax.default_backend()}")
        import gpuwrf.contracts.state as state_contract

        state_contract._gpu_device = lambda: jax.devices("cpu")[0]
        import gpuwrf.runtime.operational_mode as runtime
        from scripts import v0234_corrected_ni_ordinary_bisection as ordinary
        from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble

        stage = "cold_load"
        scratch = Path(tempfile.mkdtemp(prefix="v0234-nocturnal-qke-backend-"))
        tree, names, cold_carries, dt_by_domain, load_authority = (
            ordinary.load_corrected_tree(scratch / "load")
        )
        if names != ("d01", "d02", "d03") or dt_by_domain != {
            "d01": 54.0,
            "d02": 18.0,
            "d03": 6.0,
        }:
            raise RuntimeError("canonical hierarchy changed")
        loaded = tree.domains["d03"].namelist
        controls = {
            "bl_pbl_physics": int(loaded.bl_pbl_physics),
            "sf_sfclay_physics": int(loaded.sf_sfclay_physics),
            "acoustic_substeps": int(loaded.acoustic_substeps),
            "moist_adv_opt": int(loaded.moist_adv_opt),
            "scalar_adv_opt": int(loaded.scalar_adv_opt),
        }
        if controls != {
            "bl_pbl_physics": 5,
            "sf_sfclay_physics": 5,
            "acoustic_substeps": 10,
            "moist_adv_opt": 1,
            "scalar_adv_opt": 1,
        }:
            raise RuntimeError(f"current controls changed: {controls}")
        if prior["authority"]["load_authority_sha256"] != common._canonical(
            load_authority
        ):
            raise RuntimeError("prior and current load authority differ")

        cold_arrays = step200._state_arrays(
            np, jax.device_get(cold_carries["d03"].state)
        )
        expected_arrays = cold_proof["repaired_candidate"]["cold_state"]["arrays"]
        cold_drift = [
            name
            for name in sorted(cold_arrays)
            if step200._array_sha256(np, cold_arrays[name])
            != expected_arrays[name]["sha256"]
        ]
        if cold_drift:
            raise RuntimeError(f"current CPU cold load changed: {cold_drift}")
        cpu_qke = cold_arrays["qke"]
        if step200._array_sha256(np, cpu_qke) != CURRENT_CPU_QKE_SHA256:
            raise RuntimeError("current CPU QKE authority changed")

        with common.STEP0.open("rb") as stream:
            retained = pickle.load(stream)
        retained_arrays = step200._state_arrays(np, retained.state)
        if step200._array_sha256(np, retained_arrays["qke"]) != RETAINED_GPU_QKE_SHA256:
            raise RuntimeError("retained GPU QKE authority changed")
        retained_mismatches = tuple(
            name
            for name in sorted(cold_arrays)
            if not np.array_equal(cold_arrays[name], retained_arrays[name])
        )
        if retained_mismatches != EXPECTED_COLD_RETAINED_MISMATCHES:
            raise RuntimeError(
                f"cold/retained mismatch set changed: {retained_mismatches}"
            )
        qke_delta = _qke_delta(
            np, cpu_qke, retained_arrays["qke"], retained_arrays["xland"]
        )
        cpu_seed_carry = retained.replace(
            state=retained.state.replace(qke=jnp.asarray(cpu_qke))
        )

        stage = "wrf_receipt"
        receipt = sound._wrf_receipt(reassemble, ("sp4_exit__u", "sp4_exit__v"))
        if receipt != prior["authority"]["wrf_sp4_receipt"]:
            raise RuntimeError("WRF SP4 receipt differs from prior proof")
        truth = {
            "U": reassemble.reassemble3d("sp4_exit__u", 1),
            "V": reassemble.reassemble3d("sp4_exit__v", 1),
        }

        stage = "run_cpu_qke_seed"
        clock = runtime.build_clock_base(loaded)
        cpu_result, audit = common._run_arm(
            runtime, jax, jnp, cpu_seed_carry, loaded, clock
        )
        manifest = common._manifest(jax, np, cpu_result)
        fields = {
            field: sound._metric(
                np, getattr(cpu_result.state, field.lower()), value
            )
            for field, value in truth.items()
        }

        stage = "decision"
        retained_arm = prior["arms"]["runtime_N10"]
        comparison = _field_comparison(retained_arm["fields"], fields)
        max_primary_change = max(
            row["absolute_change"]
            for field in comparison.values()
            for row in field.values()
        )
        output_exact = manifest["sha256"] == retained_arm["output_manifest_sha256"]
        primary_exact = max_primary_change == 0.0
        checks = {
            "prior_retained_GPU_qke_arm_authenticated": True,
            "current_CPU_cold_load_all_60_arrays_authenticated": True,
            "cold_vs_retained_mismatch_set_exact": True,
            "only_qke_replaced_in_retained_step0": True,
            "current_CPU_qke_finite": bool(np.all(np.isfinite(cpu_qke))),
            "current_CPU_qke_physical_bounds": bool(
                np.min(cpu_qke) >= 0.0 and np.max(cpu_qke) <= 25.0
            ),
            "CPU_qke_seed_step1_interface_106": bool(
                audit["interface_identity"] and manifest["leaf_count"] == 106
            ),
            "CPU_qke_seed_step1_all_finite": all(
                row["finite"] for row in manifest["leaves"]
            ),
            "CPU_qke_seed_step1_callback_transfer_unknown_free": bool(
                not audit["forbidden_tokens"]
                and not audit["unknown_custom_call_targets"]
            ),
        }
        if not all(checks.values()):
            verdict = "QKE_BACKEND_DISCRIMINATOR_HARNESS_RED"
        elif output_exact:
            verdict = "QKE_BACKEND_SEED_STEP1_BIT_INERT__HARNESS_GATE_REPAIRABLE"
        elif max_primary_change <= 1.0e-6:
            verdict = "QKE_BACKEND_SEED_STEP1_UV_NEGLIGIBLE__HARNESS_GATE_REPAIRABLE"
        else:
            verdict = "QKE_BACKEND_SEED_STEP1_UV_ACTIVE__PROMOTE_H4"
        proof = {
            "schema": "gpuwrf.v0234.nocturnal-step1-qke-backend.v1",
            "verdict": verdict,
            "scope": (
                "CPU-only current-tree d03 Step0-to-step1 single-leaf QKE seed "
                "discriminator; prior retained-GPU-QKE arm reused; no WRF/MPI, "
                "GPU command/query, correction, or trajectory claim"
            ),
            "authority": {
                "prior_retained_GPU_qke_step1": {
                    "path": str(PRIOR),
                    "file_sha256": PRIOR_FILE_SHA256,
                    "proof_sha256": PRIOR_PROOF_SHA256,
                    "output_manifest_sha256": retained_arm[
                        "output_manifest_sha256"
                    ],
                },
                "cold_step0_repair": {
                    "path": str(COLD_PROOF),
                    "file_sha256": COLD_PROOF_FILE_SHA256,
                    "proof_sha256": COLD_PROOF_SHA256,
                },
                "retained_GPU_step0": {
                    "path": str(common.STEP0),
                    "file_sha256": common.STEP0_SHA256,
                },
                "current_CPU_qke_sha256": CURRENT_CPU_QKE_SHA256,
                "retained_GPU_qke_sha256": RETAINED_GPU_QKE_SHA256,
                "wrf_sp4_receipt": receipt,
                "load_authority_sha256": common._canonical(load_authority),
                "source_sha256": SOURCE_SHA256,
            },
            "controls": controls,
            "input_qke_backend_delta": qke_delta,
            "current_CPU_qke_seed_step1": {
                "audit": audit,
                "output_manifest_sha256": manifest["sha256"],
                "output_leaf_count": manifest["leaf_count"],
                "fields": fields,
            },
            "retained_GPU_qke_seed_step1_fields": retained_arm["fields"],
            "directional_comparison": comparison,
            "max_primary_UV_RMSE_change": max_primary_change,
            "primary_UV_metrics_exact": primary_exact,
            "complete_output_manifest_exact": output_exact,
            "checks": checks,
            "interpretation_limits": {
                "one_step_cannot_close_long_trajectory_PBL_error": True,
                "retained_GPU_qke_is_historical_same-initializer_backend_authority": True,
                "CPU_WRF_pre_step_qke_is_zero_and_first_MYNN_call_initializes_it": True,
                "backend_delta_is_lowest_four_levels_and_sea_concentrated": True,
            },
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
                    "qke_input_rms": qke_delta["rms"],
                    "qke_input_max_abs": qke_delta["max_abs"],
                    "max_primary_UV_RMSE_change": max_primary_change,
                    "complete_output_manifest_exact": output_exact,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if all(checks.values()) else 3
    except Exception as exc:
        blocker = {
            "schema": "gpuwrf.v0234.nocturnal-step1-qke-backend-blocker.v1",
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
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
