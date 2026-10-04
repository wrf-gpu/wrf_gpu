"""CPU-only Step0-to-step1 MYNN PBL directional ablation.

The authenticated current-tree N=10/full-MYNN arm already exists in
``NOCTURNAL_STEP1_SOUND_COUNT.json``.  This discriminator compiles only the
otherwise-identical ``bl_pbl_physics=0`` arm, scores it against the same
output-neutral WRF SP4 savepoint, and compares it with the retained full-MYNN
metrics.  It ranks the PBL family; it is not a correction or trajectory gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from scripts import v0234_nocturnal_step1_sound_count as sound
from scripts import v0234_t2save_ownership_cpu_ab as common


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "NOCTURNAL_STEP1_PBL_ABLATION.json"
BLOCKER = SPRINT / "NOCTURNAL_STEP1_PBL_ABLATION_BLOCKER.json"
PRIOR = SPRINT / "NOCTURNAL_STEP1_SOUND_COUNT.json"
PRIOR_FILE_SHA256 = (
    "82482bc63f4db3e31e2dcb38c8ad086bc460d769736d82f10e9ca8a369f0130d"
)
PRIOR_PROOF_SHA256 = (
    "7aa5d21eef1335eb38a7c0a285dde72869160b56bf4a69d931456cdfa4eade29"
)
SOURCE_SHA256 = {
    "src/gpuwrf/integration/nested_pipeline.py": (
        "77b6a690647ec4b0b615f425338ca41f3e594c29566eeafc0f9e884e1f0e98c4"
    ),
    "src/gpuwrf/runtime/operational_mode.py": (
        "d5ce22236188c4264ab13e3108f163830d9c5bce1560b0090db1c927e8e1a19d"
    ),
}
REQUIRED_ENV = dict(sound.REQUIRED_ENV)
PRIMARY_DIMENSIONS = (
    "lowest_level_rmse",
    "lowest_three_level_rmse",
    "interior5_rmse",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison(
    full: dict[str, Any], ablated: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("U", "V"):
        result[field] = {}
        for dimension in PRIMARY_DIMENSIONS:
            full_value = float(full[field][dimension])
            ablated_value = float(ablated[field][dimension])
            result[field][dimension] = {
                "full_MYNN5": full_value,
                "ablated_PBL0": ablated_value,
                "full_minus_ablated": full_value - ablated_value,
                "full_over_ablated": full_value / ablated_value,
                "full_MYNN5_closer_to_WRF": full_value < ablated_value,
            }
    return result


def main() -> int:
    stage = "startup"
    scratch: Path | None = None
    try:
        actual_env = {key: os.environ.get(key) for key in REQUIRED_ENV}
        if actual_env != REQUIRED_ENV:
            raise RuntimeError(f"environment mismatch: {actual_env!r}")
        if os.environ.get("GPUWRF_ACOUSTIC_SUBSTEPS") is not None:
            raise RuntimeError("GPUWRF_ACOUSTIC_SUBSTEPS must be absent")
        if _sha256(PRIOR) != PRIOR_FILE_SHA256:
            raise RuntimeError("prior N10/N4 proof file changed")
        prior = json.loads(PRIOR.read_text())
        if (
            prior.get("proof_sha256") != PRIOR_PROOF_SHA256
            or prior.get("verdict")
            != "CURRENT_STEP1_WRF_N4_DIRECTIONAL_GREEN__C4_REOPENED"
        ):
            raise RuntimeError("prior N10/N4 proof semantics changed")
        if _sha256(common.STEP0) != common.STEP0_SHA256:
            raise RuntimeError("authenticated Step0 changed")
        for relative, expected in SOURCE_SHA256.items():
            actual = _sha256(ROOT / relative)
            if actual != expected:
                raise RuntimeError(f"current model source changed: {relative}: {actual}")
            if prior["source_sha256"][Path(relative).name] != expected:
                raise RuntimeError(f"prior model source differs: {relative}")

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
        scratch = Path(tempfile.mkdtemp(prefix="v0234-nocturnal-pbl-step1-"))
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
        if prior["arms"]["runtime_N10"]["acoustic_substeps"] != 10:
            raise RuntimeError("prior full-MYNN arm is not the current N10 arm")
        if prior["authority"]["load_authority_sha256"] != common._canonical(
            load_authority
        ):
            raise RuntimeError("prior and current load authority differ")
        receipt = sound._wrf_receipt(
            reassemble, ("sp4_exit__u", "sp4_exit__v")
        )
        if receipt != prior["authority"]["wrf_sp4_receipt"]:
            raise RuntimeError("WRF SP4 receipt differs from prior proof")
        truth = {
            "U": reassemble.reassemble3d("sp4_exit__u", 1),
            "V": reassemble.reassemble3d("sp4_exit__v", 1),
        }
        with common.STEP0.open("rb") as stream:
            carry = pickle.load(stream)

        stage = "run_PBL0"
        ablated_namelist = dataclasses.replace(loaded, bl_pbl_physics=0)
        ablated_clock = runtime.build_clock_base(ablated_namelist)
        ablated_result, audit = common._run_arm(
            runtime,
            jax,
            jnp,
            carry,
            ablated_namelist,
            ablated_clock,
        )
        manifest = common._manifest(jax, np, ablated_result)
        ablated_fields = {
            field: sound._metric(np, getattr(ablated_result.state, field.lower()), value)
            for field, value in truth.items()
        }

        stage = "decision"
        full_arm = prior["arms"]["runtime_N10"]
        comparison = _comparison(full_arm["fields"], ablated_fields)
        directional_rows = [
            comparison[field][dimension]["full_MYNN5_closer_to_WRF"]
            for field in ("U", "V")
            for dimension in PRIMARY_DIMENSIONS
        ]
        full_beneficial = all(directional_rows)
        ablated_beneficial = not any(directional_rows)
        checks = {
            "prior_full_arm_and_sources_authenticated": True,
            "current_loader_controls_exact": True,
            "same_step0_load_and_wrf_receipt": True,
            "PBL0_interface_106": bool(
                audit["interface_identity"] and manifest["leaf_count"] == 106
            ),
            "PBL0_all_finite": all(row["finite"] for row in manifest["leaves"]),
            "PBL0_callback_transfer_unknown_free": bool(
                not audit["forbidden_tokens"]
                and not audit["unknown_custom_call_targets"]
            ),
            "PBL0_changes_output_vs_prior_full_MYNN5": (
                manifest["sha256"] != full_arm["output_manifest_sha256"]
            ),
        }
        if not all(checks.values()):
            verdict = "PBL_ABLATION_HARNESS_RED"
        elif full_beneficial:
            verdict = "PBL5_DIRECTIONALLY_BENEFICIAL_STEP1__H4_DEMOTED_NOT_CLOSED"
        elif ablated_beneficial:
            verdict = "PBL0_DIRECTIONALLY_CLOSER_STEP1__H4_PROMOTED"
        else:
            verdict = "PBL_ABLATION_MIXED__H4_REMAINS"
        proof = {
            "schema": "gpuwrf.v0234.nocturnal-step1-pbl-ablation.v1",
            "verdict": verdict,
            "scope": (
                "CPU-only current-tree d03 Step0-to-step1 PBL-slot directional "
                "ablation; prior full-MYNN N10 arm reused; not a correction, "
                "trajectory claim, or release gate"
            ),
            "authority": {
                "prior_full_MYNN5": {
                    "path": str(PRIOR),
                    "file_sha256": PRIOR_FILE_SHA256,
                    "proof_sha256": PRIOR_PROOF_SHA256,
                    "output_manifest_sha256": full_arm["output_manifest_sha256"],
                },
                "step0": {
                    "path": str(common.STEP0),
                    "sha256": common.STEP0_SHA256,
                },
                "wrf_sp4_receipt": receipt,
                "load_authority_sha256": common._canonical(load_authority),
                "source_sha256": SOURCE_SHA256,
            },
            "controls": {
                "full_MYNN5": controls,
                "ablated_PBL0": {**controls, "bl_pbl_physics": 0},
                "only_control_changed": "bl_pbl_physics: 5 -> 0",
            },
            "prior_full_MYNN5_fields": full_arm["fields"],
            "ablated_PBL0": {
                "audit": audit,
                "output_manifest_sha256": manifest["sha256"],
                "output_leaf_count": manifest["leaf_count"],
                "fields": ablated_fields,
            },
            "directional_comparison": comparison,
            "full_MYNN5_beneficial_all_primary_U_V_dimensions": full_beneficial,
            "ablated_PBL0_beneficial_all_primary_U_V_dimensions": ablated_beneficial,
            "checks": checks,
            "interpretation_limits": {
                "outer_boundary_stage_asymmetry_excluded_from_primary": True,
                "one_step_cannot_close_long_trajectory_PBL_error": True,
                "surface_layer_and_land_model_remain_active_in_PBL0_arm": True,
                "C4_sound_count_unchanged_at_N10": True,
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
                    "directional_comparison": comparison,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if all(checks.values()) else 3
    except Exception as exc:
        blocker = {
            "schema": "gpuwrf.v0234.nocturnal-step1-pbl-ablation-blocker.v1",
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
