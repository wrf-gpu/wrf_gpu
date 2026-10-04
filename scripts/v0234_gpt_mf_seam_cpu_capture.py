#!/usr/bin/env python3
"""Fresh combined-radiation endpoint capture for the MYNN seam sprint.

This is a narrow profile over the sealed QML + static-top capture chain.  It
replaces the earlier O3-only paired delta with the current-production paired
delta (O3RAD + hydrostatic P3D/P8W/T8W + dated CLWRF gases), then binds a fresh
namespace and the current sprint proof directory.  The one-shot production
adapter engine is unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-mf-seam-closure"
PROFILE = REPO / "scripts/v0234_gpt_land_tsk_o3_cpu_capture.py"
NAMESPACE_PARENT = Path("/tmp/v0234_gpt_mf_seam_capture")
COMBINED_PROOF = SPRINT / "combined-radiation-tsk-proof.json"
COMBINED_DELTA = Path("/tmp/v0234_gpt_mf_seam_combined_radiation_delta.npz")
COMBINED_PROOF_SHA256 = "6cbc55d2c0f9018f90110abb963e3fe75e7d307b80f873cba967ddc12d5236a2"
COMBINED_PROOF_CANONICAL = "fd620f4d5b9eda43fd86869d83f3ab898f4921400e962de1e76c7fa1a651111a"
COMBINED_DELTA_SHA256 = "1c2a5beb17960cc3c5a07bff9b0b71b6f21a233175dad0be9e14b5e8d0e356e8"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")


def _load_profile():
    spec = importlib.util.spec_from_file_location("v0234_mf_seam_o3_profile", PROFILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"capture profile unavailable: {PROFILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_chain(profile):
    if _sha256(COMBINED_PROOF) != COMBINED_PROOF_SHA256:
        raise RuntimeError("combined radiation proof hash drift")
    proof = json.loads(COMBINED_PROOF.read_text(encoding="utf-8"))
    if not (
        profile._canonical(proof) == COMBINED_PROOF_CANONICAL
        and proof.get("canonical_payload_sha256") == COMBINED_PROOF_CANONICAL
        and proof.get("schema")
        == "v0234-rrtmg-wrf-clwrf-production-composition-tsk-v1"
        and proof.get("verdict") == "WRF_PRODUCTION_STRICTLY_IMPROVES_D03_TSK"
        and proof.get("passed") is True
        and proof.get("candidate_authorizes_fix_request") is True
        and proof.get("tsk_parity", {}).get("land_rms_and_max_strictly_improve")
        is True
        and proof.get("tsk_parity", {}).get("water_bitwise_invariant") is True
        and set(
            proof.get("paired_radiation", {}).get(
                "only_changed_kernel_inputs", []
            )
        )
        == {
            "CO2 VMR",
            "N2O VMR",
            "CH4 VMR",
            "CFC11 VMR",
            "CFC12 VMR",
            "hydrostatic P3D",
            "hydrostatic P8W",
            "phy_prep T8W",
            "O3RAD",
        }
        and proof.get("delta_artifact", {}).get("sha256")
        == COMBINED_DELTA_SHA256
    ):
        raise RuntimeError("combined radiation proof authority drift")
    if _sha256(COMBINED_DELTA) != COMBINED_DELTA_SHA256:
        raise RuntimeError("combined radiation delta hash drift")
    arrays: dict[str, np.ndarray] = {}
    with np.load(COMBINED_DELTA, allow_pickle=False) as archive:
        if set(archive.files) != {f"{name}_delta" for name in DELTA_FIELDS}:
            raise RuntimeError("combined radiation delta schema drift")
        for name in DELTA_FIELDS:
            value = np.asarray(archive[f"{name}_delta"])
            declared = proof["delta_artifact"]["arrays"][f"{name}_delta"]
            if not (
                value.shape == (93, 111)
                and value.dtype.str == declared["dtype"]
                and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                == declared["logical_c_bitpayload_sha256"]
                and np.isfinite(value).all()
            ):
                raise RuntimeError(f"combined radiation delta array drift: {name}")
            arrays[name] = np.array(value, copy=True)
    return proof, arrays


def main() -> int:
    profile = _load_profile()
    combined_proof, combined_arrays = _combined_chain(profile)
    original_configure = profile.provenance.configure
    original_install = profile._install_o3_preflight

    def configure_for_sprint():
        manifest = original_configure()
        profile.base.SPRINT = SPRINT
        return manifest

    profile.provenance.configure = configure_for_sprint
    profile.NAMESPACE_PARENT = NAMESPACE_PARENT
    profile.O3_TSK = combined_proof
    profile.O3_ARRAYS = combined_arrays
    profile.O3_TSK_PROOF = COMBINED_PROOF
    profile.O3_TSK_PROOF_SHA256 = COMBINED_PROOF_SHA256
    profile.O3_TSK_PROOF_CANONICAL = COMBINED_PROOF_CANONICAL
    profile.O3_DELTA = COMBINED_DELTA
    profile.O3_DELTA_SHA256 = COMBINED_DELTA_SHA256

    def install_combined_preflight(*args, **kwargs):
        original_install(*args, **kwargs)
        previous = profile.base.static_preflight

        def combined(namespace: Path, proof_output: Path):
            result = previous(namespace, proof_output)
            result["combined_radiation_correction"] = {
                **result["wrf_o3rad_correction"],
                "composition": [
                    "O3RAD",
                    "hydrostatic P3D/P8W/T8W",
                    "dated CLWRF gases",
                ],
                "combined_proof_sha256": COMBINED_PROOF_SHA256,
                "combined_proof_canonical_payload_sha256": (
                    COMBINED_PROOF_CANONICAL
                ),
                "combined_delta_sha256": COMBINED_DELTA_SHA256,
                "land_tsk_rms_k": combined_proof["tsk_parity"][
                    "candidate_plus_wrf_cam_ghg"
                ]["land"]["rms"],
                "water_tsk_bitwise_invariant": True,
            }
            result["correction_composition"] = {
                "order": [
                    "authenticated seam",
                    "seasonal absolute surface handles",
                    "QML paired delta",
                    "RRTMG static-top paired delta",
                    "combined production radiation paired delta",
                ],
                "combined_radiation_fields": list(DELTA_FIELDS),
                "backend_offset_imported": False,
            }
            return result

        profile.base.static_preflight = combined

    profile._install_o3_preflight = install_combined_preflight
    wrapper = Path(__file__).resolve().relative_to(REPO).as_posix()
    proof_driver = "scripts/v0234_gpt_land_tsk_rrtmg_co2_year.py"
    profile.MODEL_SOURCES = (*profile.MODEL_SOURCES, proof_driver, wrapper)
    return int(profile.main())


if __name__ == "__main__":
    raise SystemExit(main())
