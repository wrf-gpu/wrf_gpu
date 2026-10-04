#!/usr/bin/env python3
"""Build the CPU-only proof for the v0.23.4 critic-repaired boundary bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from v0234_wrf_sint_source_oracle import sint_full


REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-13-v0234-nested-boundary-science-repair"
ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
CPU_DIR = ROOT / "run/wrf"
FAILURE_DIR = (
    ROOT
    / "corrected_ni_rca_max_22c2bd7a"
    / "nested_frozen_bundle_e0d0b05a"
    / "failure"
)
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
RETRY20 = ROOT / "gpu_validation_retry20_relative_rmse_3ee02c19"
SEALED_NI_V10_AUTHORITY = (
    REPO
    / ".agent/sprints/2026-07-13-v0234-corrected-ni-rca-max/authority-proof.json"
)
WRF_SOURCE_AUTHORITY = SPRINT / "wrf-source-authority.json"
CANDIDATE_OFF_AUTHORITY = SPRINT / "candidate-off-e0d0-proof.json"
FINAL_P1_REVIEW_DIR = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v0234-corrected-validation-review/"
    ".agent/sprints/2026-07-14-v0234-nested-boundary-science-repair-rereview"
)
FINAL_P1_REVIEW = {
    "independent_rereview": FINAL_P1_REVIEW_DIR / "independent-rereview.md",
    "findings": FINAL_P1_REVIEW_DIR / "findings.json",
    "adversarial_probes": FINAL_P1_REVIEW_DIR / "adversarial-probes.json",
}
FINAL_P1_REVIEW_SHA256 = {
    "independent_rereview": "59b4d8d1d1cbc9e66bde140865ea20bddc95543b29075dbb9441ae8c6f7d111d",
    "findings": "1ebcdbd2c87842abd57bacffb0485ce985c884c17218430ecc91ef31eb823287",
    "adversarial_probes": "e927476b7d8dc260b869ea278e51565b9846201818fbcc8c265f8604b2335b82",
}

CORRECTED_INPUTS = {
    "namelist.input": CPU_DIR / "namelist.input",
    "wrfbdy_d01": CPU_DIR / "wrfbdy_d01",
    "wrfinput_d01": CPU_DIR / "wrfinput_d01",
    "wrfinput_d02": CPU_DIR / "wrfinput_d02",
    "wrfinput_d03": CPU_DIR / "wrfinput_d03",
}

RETRY20_AUTHORITY = {
    "cache_authority": RETRY20 / "retry20-cache-source-authority.json",
    "final_accept": RETRY20 / "retry20-final-accept.json",
    "incremental_pairs": RETRY20 / "incremental-pairs.json",
    "runtime_attestation": RETRY20 / "gpu-proof/retry20-runtime-source-attestation.json",
    "terminal_contract": ROOT / "terminal_cpu_authority_contract_v1.json",
}

EXPECTED_SHA256 = {
    "failure_proof": "10336c9ed866aa88d3e8df3f27e43da4166d5f769d8e25225f1deefa6dd377e4",
    "step8800": "ccaf8a462506755c3d557f09c304a77e7a2722170c22fa466349cff643bca9aa",
    "step9000": "4d57a964e802407b584b24a54833816b69a304e165d99f6ffdeb9b8931805d69",
    "cpu_d01": "dfebd308c81c29655715f410c102dee7860a02816ece1579870478e085912f98",
    "cpu_d02": "db4c417a0e9718f19f3407c2db21f46c869e69df4c846ea0618b8f276296811f",
    "cpu_d03": "1ea00bd68bfa4098abd14d49e031389bae53bd0c921c80b2af264cd9dcbe2f0c",
    "namelist.input": "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838",
    "wrfbdy_d01": "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec",
    "wrfinput_d01": "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756",
    "wrfinput_d02": "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964",
    "wrfinput_d03": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
    "cache_authority": "35b03ad00f6b652c60fbc2c72a9e1228c8823c95f87bb4c13299ba5817ecdb5c",
    "final_accept": "72e9e6df6e23dc3889586afdc27b782de19aba54a8c7f62060baf3d2087648c8",
    "incremental_pairs": "9436da45de29518abc25c7fbe7f458f0323057f5768dc2b38b6e96e4713b4ffd",
    "runtime_attestation": "fec1ea5e1be9ca6f272dbbf955a7cb96ade1710becedf87c6cc5b194a45e8a26",
    "terminal_contract": "26b16e21782400308ec4d58175907796bf8aaddb01062bb06be8b87a2a88848a",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated_file(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def canonical_sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(ds: Dataset, name: str) -> np.ndarray:
    value = np.asarray(ds.variables[name][:])
    return np.asarray(value[0] if value.ndim and value.shape[0] == 1 else value, dtype=np.float64)


def load_domain(path: Path) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        return {
            name: _read(ds, name)
            for name in (
                "U",
                "V",
                "W",
                "PH",
                "T",
                "QVAPOR",
                "MU",
                "MUB",
                "MAPFAC_MY",
                "MAPFAC_UY",
                "MAPFAC_VX",
                "C1H",
                "C2H",
                "C1F",
                "C2F",
            )
        }


def interp_sint(
    field: np.ndarray,
    *,
    ratio: int,
    i_start: int,
    j_start: int,
    shape,
    xstag: bool = False,
    ystag: bool = False,
):
    """Independent source-literal full donor/TR4/limiter SINT."""

    return sint_full(
        np.asarray(field, dtype=np.float64),
        ratio=int(ratio),
        i_parent_start=int(i_start),
        j_parent_start=int(j_start),
        child_ny=int(shape[0]),
        child_nx=int(shape[1]),
        xstag=bool(xstag),
        ystag=bool(ystag),
    )


def coupled(domain: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mu = domain["MUB"] + domain["MU"]
    c1h, c2h = domain["C1H"][:, None, None], domain["C2H"][:, None, None]
    c1f, c2f = domain["C1F"][:, None, None], domain["C2F"][:, None, None]
    mass_h = c1h * mu[None] + c2h
    mass_f = c1f * mu[None] + c2f
    muu = 0.5 * (
        np.concatenate((mu[:, :1], mu), axis=1)
        + np.concatenate((mu, mu[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu[:1, :], mu), axis=0)
        + np.concatenate((mu, mu[-1:, :]), axis=0)
    )
    return {
        "U": domain["U"] * (c1h * muu[None] + c2h) / domain["MAPFAC_UY"][None],
        "V": domain["V"] * (c1h * muv[None] + c2h) / domain["MAPFAC_VX"][None],
        "W": domain["W"] * mass_f / domain["MAPFAC_MY"][None],
        "PH": domain["PH"] * mass_f,
        "T": domain["T"] * mass_h,
        "QVAPOR": domain["QVAPOR"] * mass_h,
        "MU": domain["MU"],
    }


def late_coupled(
    parent: dict[str, np.ndarray],
    child: dict[str, np.ndarray],
    *,
    ratio: int,
    i_start: int,
    j_start: int,
) -> dict[str, np.ndarray]:
    ny, nx = child["MU"].shape
    interp = lambda value, shape, xstag=False, ystag=False: interp_sint(
        value,
        ratio=ratio,
        i_start=i_start,
        j_start=j_start,
        shape=shape,
        xstag=xstag,
        ystag=ystag,
    )
    mu_p = interp(parent["MU"], (ny, nx))
    # The released consumer couples a decoupled target with the live child's
    # step-start mass/map factors.  This is stronger and more literal than a
    # hypothetical interpolate-mass-then-couple comparison.
    mu = child["MUB"] + child["MU"]
    c1h, c2h = child["C1H"][:, None, None], child["C2H"][:, None, None]
    c1f, c2f = child["C1F"][:, None, None], child["C2F"][:, None, None]
    mass_h = c1h * mu[None] + c2h
    mass_f = c1f * mu[None] + c2f
    muu = 0.5 * (
        np.concatenate((mu[:, :1], mu), axis=1)
        + np.concatenate((mu, mu[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu[:1, :], mu), axis=0)
        + np.concatenate((mu, mu[-1:, :]), axis=0)
    )
    return {
        "U": interp(parent["U"], (ny, nx + 1), xstag=True)
        * (c1h * muu[None] + c2h)
        / child["MAPFAC_UY"][None],
        "V": interp(parent["V"], (ny + 1, nx), ystag=True)
        * (c1h * muv[None] + c2h)
        / child["MAPFAC_VX"][None],
        "W": interp(parent["W"], (ny, nx))
        * mass_f
        / child["MAPFAC_MY"][None],
        "PH": interp(parent["PH"], (ny, nx)) * mass_f,
        "T": interp(parent["T"], (ny, nx)) * mass_h,
        "QVAPOR": interp(parent["QVAPOR"], (ny, nx)) * mass_h,
        "MU": mu_p,
    }


def exact_coupled(
    parent: dict[str, np.ndarray], child: dict[str, np.ndarray], *, ratio: int, i_start: int, j_start: int
) -> dict[str, np.ndarray]:
    ny, nx = child["MU"].shape
    source = coupled(parent)
    shapes = {
        "U": ((ny, nx + 1), True, False),
        "V": ((ny + 1, nx), False, True),
        "W": ((ny, nx), False, False),
        "PH": ((ny, nx), False, False),
        "T": ((ny, nx), False, False),
        "QVAPOR": ((ny, nx), False, False),
        "MU": ((ny, nx), False, False),
    }
    return {
        name: interp_sint(
            value,
            ratio=ratio,
            i_start=i_start,
            j_start=j_start,
            shape=shapes[name][0],
            xstag=shapes[name][1],
            ystag=shapes[name][2],
        )
        for name, value in source.items()
    }


def decouple_for_child(
    values: dict[str, np.ndarray], child: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Convert coupled boundary targets to physical units with child metrics."""

    mu = child["MUB"] + child["MU"]
    c1h, c2h = child["C1H"][:, None, None], child["C2H"][:, None, None]
    c1f, c2f = child["C1F"][:, None, None], child["C2F"][:, None, None]
    mass_h = c1h * mu[None] + c2h
    mass_f = c1f * mu[None] + c2f
    muu = 0.5 * (
        np.concatenate((mu[:, :1], mu), axis=1)
        + np.concatenate((mu, mu[:, -1:]), axis=1)
    )
    muv = 0.5 * (
        np.concatenate((mu[:1, :], mu), axis=0)
        + np.concatenate((mu, mu[-1:, :]), axis=0)
    )
    return {
        "U": values["U"] * child["MAPFAC_UY"][None] / (c1h * muu[None] + c2h),
        "V": values["V"] * child["MAPFAC_VX"][None] / (c1h * muv[None] + c2h),
        "W": values["W"] * child["MAPFAC_MY"][None] / mass_f,
        "PH": values["PH"] / mass_f,
        "T": values["T"] / mass_h,
        "QVAPOR": values["QVAPOR"] / mass_h,
        "MU": values["MU"],
    }


def ring_mask(shape, width: int = 5) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    return np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx)) < width


def stats(exact: np.ndarray, late: np.ndarray) -> dict[str, float | int | dict]:
    mask2 = ring_mask(exact.shape[-2:])
    mask = np.broadcast_to(mask2, exact.shape)
    a, b = exact[mask], late[mask]
    delta = b - a
    rms_exact = float(np.sqrt(np.mean(a * a)))
    result = {
        "count": int(delta.size),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_bias": float(np.mean(delta)),
        "relative_rms": float(np.sqrt(np.mean(delta * delta)) / max(rms_exact, 1.0e-300)),
        "exact_ring_sum": float(np.sum(a)),
        "late_ring_sum": float(np.sum(b)),
        "ring_sum_delta": float(np.sum(delta)),
    }
    side_masks = {
        "W": np.indices(mask2.shape)[1] < 5,
        "E": np.indices(mask2.shape)[1] >= mask2.shape[1] - 5,
        "S": np.indices(mask2.shape)[0] < 5,
        "N": np.indices(mask2.shape)[0] >= mask2.shape[0] - 5,
    }
    result["side_rmse"] = {}
    for side, side2 in side_masks.items():
        smask = np.broadcast_to(side2, exact.shape)
        sdelta = (late - exact)[smask]
        result["side_rmse"][side] = float(np.sqrt(np.mean(sdelta * sdelta)))
    return result


def real_edge_oracle(parent_path: Path, child_path: Path, *, i_start: int, j_start: int):
    parent, child = load_domain(parent_path), load_domain(child_path)
    exact = exact_coupled(parent, child, ratio=3, i_start=i_start, j_start=j_start)
    late = late_coupled(parent, child, ratio=3, i_start=i_start, j_start=j_start)
    fields = {name: stats(exact[name], late[name]) for name in exact}
    exact_physical = decouple_for_child(exact, child)
    late_physical = decouple_for_child(late, child)
    physical_fields = {
        name: stats(exact_physical[name], late_physical[name]) for name in exact
    }
    nonzero = all(fields[name]["max_abs"] > 0.0 for name in ("U", "V", "W", "PH", "T", "QVAPOR"))
    return {
        "parent": str(parent_path),
        "child": str(child_path),
        "ratio": 3,
        "i_parent_start": i_start,
        "j_parent_start": j_start,
        "ring_width": 5,
        "operator": "full donor/TR4/monotonic-limiter SINT",
        "fields": fields,
        "physical_unit_fields": physical_fields,
        "late_coupling_commutator_nonzero_for_all_coupled_fields": nonzero,
        "mu_control_exact": fields["MU"]["max_abs"] == 0.0,
    }


def retained_package_semantics(path: Path) -> dict:
    # Force CPU before unpickling imports the JAX-backed State class.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    with path.open("rb") as handle:
        carry = pickle.load(handle)
    state = carry.state
    rows = {}
    for leaf_name, physical_name in (
        ("u_bdy", "u"),
        ("v_bdy", "v"),
        ("w_bdy", "w"),
        ("theta_bdy", "theta"),
        ("qv_bdy", "qv"),
        ("ph_bdy", "ph_perturbation"),
        ("mu_bdy", "mu_perturbation"),
    ):
        leaf = np.asarray(getattr(state, leaf_name))
        physical = np.asarray(getattr(state, physical_name))
        rows[leaf_name] = {
            "leaf_max_abs": float(np.max(np.abs(leaf))),
            "physical_state_max_abs": float(np.max(np.abs(physical))),
            "leaf_to_physical_scale": float(
                np.max(np.abs(leaf)) / max(float(np.max(np.abs(physical))), 1.0e-300)
            ),
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": rows,
        "classification": "decoupled_physical_records",
        "reason": "U/V/W/T/QV/PH record scales match physical state, not mass-coupled WRF magnitudes",
    }


def focused_tests() -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_v0234_nested_boundary_critic_repair.py",
        "tests/test_v0234_nested_boundary_science_repair.py",
        "tests/test_v0234_nested_frozen_wrf_boundary_bundle.py",
        "tests/test_v0234_nested_boundary_spatial_causal.py",
    ]
    env = dict(os.environ)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_COMPILATION_CACHE": "0",
            "PYTHONPATH": f"{REPO}:{REPO / 'src'}",
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    # Pytest appends wall-clock duration to an otherwise deterministic summary.
    # Normalize only that volatile suffix so repeated proof generation over the
    # same code and authorities produces an identical payload and file.
    stable_stdout = re.sub(
        r"\s+in\s+[0-9]+(?:\.[0-9]+)?s(?:\s+\([^\n)]*\))?(?=\n?$)",
        " in <elapsed>s",
        completed.stdout,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(stable_stdout.encode()).hexdigest(),
        "stdout_tail": stable_stdout[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SPRINT / "candidate-cpu-proof.json")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    wrfouts = {
        name: CPU_DIR / f"wrfout_{name}_2025-03-01_15:00:00"
        for name in ("d01", "d02", "d03")
    }
    retained = {
        "step8800": FAILURE_DIR / "last-healthy-d03-step-8800.pkl",
        "step9000": FAILURE_DIR / "first-failed-d03-step-9000.pkl",
        "failure_proof": FAILURE_DIR / "failure-proof.json",
    }
    tests = {"returncode": 0, "skipped": True} if args.skip_tests else focused_tests()
    edges = {
        "d01_to_d02": real_edge_oracle(wrfouts["d01"], wrfouts["d02"], i_start=16, j_start=16),
        "d02_to_d03": real_edge_oracle(wrfouts["d02"], wrfouts["d03"], i_start=92, j_start=36),
    }
    source_authority = json.loads(WRF_SOURCE_AUTHORITY.read_text())
    source_payload = dict(source_authority)
    source_digest = source_payload.pop("proof_sha256")
    if canonical_sha(source_payload) != source_digest:
        raise RuntimeError("WRF source-authority canonical digest mismatch")
    interface_authority = json.loads(CANDIDATE_OFF_AUTHORITY.read_text())
    interface_payload = dict(interface_authority)
    interface_digest = interface_payload.pop("proof_sha256")
    if canonical_sha(interface_payload) != interface_digest:
        raise RuntimeError("candidate interface-authority canonical digest mismatch")
    payload = {
        "schema": "gpuwrf.v0234.nested-boundary-final-p1-repair.cpu-proof.v3",
        "generated_utc": "2026-07-13T00:00:00Z",
        "candidate": "exact_stagger_full_sint_moving_fill_and_map_scaled_frozen_scalar_rk_bundle",
        "authority": {
            "repair_parent": "5dbde92fbd5d9d6c44b52729581ead50d4bd542b",
            "retained_rereview": {
                "commit": "47db5952d9060abdc4bf211cd8b0f611b431033b",
                "tree": "6cc2d313b2609d2a050e71795478f918800a802a",
                "artifacts": {
                    name: authenticated_file(path)
                    for name, path in FINAL_P1_REVIEW.items()
                },
                "mandatory_findings": [
                    "P1-FULL-SINT-STAGGER-PLAN",
                    "P1-MOVING-FULL-SINT-FALLBACK",
                    "P1-SCALAR-RK-MAP-FACTOR",
                ],
                "p2_source_authority_status": "CLOSED",
            },
            "baseline_commit": {
                "commit": "e0d0b05a20b83502bbfe41e7808e9387168d7976",
                "object_type": subprocess.check_output(
                    ["git", "cat-file", "-t", "e0d0b05a20b83502bbfe41e7808e9387168d7976"],
                    cwd=REPO,
                    text=True,
                ).strip(),
            },
            "failure_artifacts": {
                name: authenticated_file(path)
                for name, path in retained.items()
            },
            "cpu_wrfouts": {
                name: authenticated_file(path)
                for name, path in wrfouts.items()
            },
            "corrected_inputs": {
                name: authenticated_file(path)
                for name, path in CORRECTED_INPUTS.items()
            },
            "corrected_input_authority_sha256": (
                "d8254c97795b527ef9e1354180f4eac4974919e59026e93f7aa849dbd3e73c2b"
            ),
            "retry20_authority": {
                name: authenticated_file(path)
                for name, path in RETRY20_AUTHORITY.items()
            },
            "sealed_ni_v10_authority": authenticated_file(SEALED_NI_V10_AUTHORITY),
            "pristine_wrf_objects": {
                "commit": source_authority["commit"],
                "tree": source_authority["tree"],
                "verdict": source_authority["verdict"],
                "canonical_proof_sha256": source_digest,
                "proof_file": authenticated_file(WRF_SOURCE_AUTHORITY),
                "source_object_count": len(source_authority["source_objects"]),
                "forcedown_generated_sha256": source_authority[
                    "registry_generator"
                ]["sha256"],
                "dirty_worktree_rejected": source_authority[
                    "dirty_worktree_rejected"
                ],
            },
            "candidate_off_and_interface": {
                "verdict": interface_authority["verdict"],
                "canonical_proof_sha256": interface_digest,
                "proof_file": authenticated_file(CANDIDATE_OFF_AUTHORITY),
                "candidate_off_e0d0_exact": interface_authority["candidate_off"][
                    "exact"
                ],
                "production_leaf_count": interface_authority[
                    "production_interface"
                ]["leaf_count"],
            },
        },
        "retained_step8800_package": retained_package_semantics(retained["step8800"]),
        "real_cpu_wrf_1500_commutator": edges,
        "source_ordering": {
            "couple_parent_before_interp": "mediation_force_domain.F:111-130",
            "couple_child_before_interp": "mediation_force_domain.F:131-145",
            "forcedown_operands": (
                "clean Registry.EM -> tools/registry -> generated "
                "nest_forcedown_interp.inc sha256="
                + source_authority["registry_generator"]["sha256"]
            ),
            "full_sint": "share/sint.F SINTB donor/TR4/OV-UN limiter, x then y",
            "stagger_destination_shift": (
                "interp_fcn.F bdy_interp1: fine destination + "
                "ioff/joff=max((ratio-1)//2,1) before coarse center/subcell"
            ),
            "stagger_even_ratio_subcell": (
                "share/sint.F keeps distinct rioff/rjoff=1 only for even ratios"
            ),
            "records_remain_coupled": "mediation_force_domain.F:176-199 uncouples live states only",
            "u_v_w_ph_t_scalar_equations": "couple_or_uncouple_em.F:121-181,270-345",
            "bdy_start_target": "interp_fcn.F:2578-2617",
            "relax_consumes_coupled_records": "module_bc_em.F:161-346",
            "scalar_rk1_freeze": "solve_em.F moist/scalar loops call relax/spec only at rk_step==1",
            "scalar_stage_update": "module_em.F rk_update_scalar uses scalar_1, mu_old/mu_base, mu_new/mu_base and dt_rk",
            "scalar_map_factor": (
                "module_em.F rk_update_scalar merges advect_tend*msfty + sc_tend; "
                "specified zone excludes advection and sc_tend remains unscaled"
            ),
            "moving_exposed_fill": (
                "candidate-on shift_domain_em exposed state and registry scratch use "
                "the same corrected full SINT; categorical masks remain nearest"
            ),
        },
        "feature_map": {
            "F1_child_clock": {
                "verdict": "RETAIN_SOURCE_FAITHFUL",
                "repair_effect": "none",
                "reason": "integer child endpoint cadence is independent of record representation",
            },
            "F2_spec_ring": {
                "verdict": "RETAIN_OPERATOR_REPAIR_OPERAND_REPRESENTATION",
                "repair_effect": "consume coupled U/V/W/T/PH records without second mass coupling",
                "reason": "spec operators were source-shaped but received decoupled forcedown records",
            },
            "F3_frozen_RK1_relax": {
                "verdict": "RETAIN_CADENCE_REPAIR_OPERAND_REPRESENTATION",
                "repair_effect": "compare live coupled fields directly to coupled records",
                "reason": "RK1 freezing is source-faithful; target units were not",
            },
            "F4_retired_moving_gain": {
                "verdict": "RETAIN_SOURCE_FAITHFUL",
                "repair_effect": "none",
                "reason": "pristine WRF has no independent gain-20 moving-residual lane",
            },
            "F5_end_step_ownership": {
                "verdict": "RETAIN_DRY_OWNERSHIP_REMOVE_SCALAR_END_PASS",
                "repair_effect": (
                    "no end-step scalar nudge; freeze moist/QNI/QNR boundary "
                    "tendencies at RK1 and consume through every rk_update_scalar stage"
                ),
                "reason": (
                    "pristine WRF integrates coupled scalar tendencies with "
                    "start scalar/mass, post-acoustic mass, evolving advection and dt_rk"
                ),
            },
            "F6_mudf_cadence": {
                "verdict": "RETAIN_SOURCE_FAITHFUL",
                "repair_effect": "none",
                "reason": "RK1 zeroing is independent of boundary record construction",
            },
            "cross_cutting_root": {
                "verdict": "SUPPORTED_GENERAL_SOURCE_DISCREPANCY",
                "mechanism": (
                    "couple parent and child before destination-exact full nonlinear SINT, "
                    "use that operator for moving exposed state/scratch, leave records "
                    "coupled, and consume scalar records at frozen RK cadence with "
                    "raw advection multiplied by msfty"
                ),
                "spatial_consistency": "boundary operator change propagated broadly into the interior by 15:00",
            },
        },
        "analytic_falsifiers": {
            "operator": "full source-literal nonlinear donor/TR4/limiter SINT",
            "variable_mass_and_field_covariance": "SINT_full(M*f) != M_child*SINT_full(f); generally also != SINT_full(M)*SINT_full(f)",
            "uniform_mass": "commutator exactly zero",
            "constant_field_with_consistent_mass": "commutator exactly zero",
            "mu_control": "uncoupled MU is identical in exact and late paths",
        },
        "incident_track_separation": {
            "ni_first_bad_cell": {
                "python_zyx": [1, 48, 78],
                "lat": 28.297913,
                "lon": -16.303406,
                "landmask": 0,
                "hgt_m": 0,
                "classification": "interior offshore E/NE; not boundary or edge",
            },
            "retry20_v10_at_1500": {
                "rmse": 2.1128268857679338,
                "land_rmse": 2.2264724301076377,
                "sea_rmse": 2.0839931788025168,
                "max_abs": 11.358115434646606,
                "max_python_yx": [39, 19],
                "max_distance_from_ni_km": 59.7430710652915,
                "ni_cell_delta": -0.7763886451721191,
            },
            "verdict": "NI_AND_V10_REMAIN_SEPARATE_WITH_DISTINCT_GPU_GATES",
        },
        "candidate_properties": {
            "single_existing_default_off_flag": "nested_frozen_wrf_boundary_bundle",
            "state_or_carry_leaf_added": False,
            "boundary_shape_changed": False,
            "host_device_transfer_added": False,
            "observer_or_callback_added": False,
            "dry_end_step_overwrite": False,
            "coupled_moisture_records": True,
            "full_gpu_native_sint": True,
            "full_sint_stagger_destination_shift_exact": True,
            "full_sint_even_ratio_rioff_rjoff_preserved": True,
            "full_sint_host_oracle_is_independent": True,
            "candidate_moving_noncategorical_full_sint": True,
            "candidate_moving_categorical_nearest": True,
            "scalar_end_step_nudge": False,
            "scalar_boundary_tendency_frozen_at_rk1": True,
            "scalar_rk_families": list(
                ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr")
            ),
            "qni_qnr_use_scalar_advection_option": True,
            "scalar_raw_advection_multiplied_by_msfty": True,
            "scalar_boundary_tendency_map_scaled": False,
            "candidate_off_e0d0_exact": interface_authority["candidate_off"][
                "exact"
            ],
            "production_carry_leaf_count": interface_authority[
                "production_interface"
            ]["leaf_count"],
            "v10_linked_to_ni": False,
        },
        "retained_carry_admissibility": {
            "old_step8800_is_authenticated_evidence": True,
            "old_step8800_is_valid_corrected_candidate_input": False,
            "missing_exact_operand": (
                "the d01/d02 live parent state (equivalently each SINT(parent_mass*field) "
                "covariance); the d03-only carry stores decoupled low-order SINT(field) and "
                "SINT(MU), from which full-SINT(parent_mass*field) and its nonlinear limiter "
                "history cannot be reconstructed"
            ),
            "falsifier": "real d01->d02 and d02->d03 coupled-before-SINT commutators are nonzero",
            "shortest_gpu_design": (
                "one fresh corrected same-process prefix, authenticate and retain its exact d03 "
                "step8800 checkpoint, continue that same carry to step9000, then only on the "
                "predeclared no-worse gate continue same-process through step9405"
            ),
            "forbidden_shortcut": "reinterpret or approximately recouple the old decoupled step8800 package",
        },
        "focused_tests": tests,
        "gpu_commands_run": 0,
        "gpu_queries_run": 0,
    }
    payload["gates"] = {
        "all_frozen_inputs_authenticate": all(
            payload["authority"]["failure_artifacts"][name]["sha256"]
            == EXPECTED_SHA256[name]
            for name in retained
        )
        and all(
            payload["authority"]["cpu_wrfouts"][name]["sha256"]
            == EXPECTED_SHA256[f"cpu_{name}"]
            for name in wrfouts
        )
        and all(
            payload["authority"]["corrected_inputs"][name]["sha256"]
            == EXPECTED_SHA256[name]
            for name in CORRECTED_INPUTS
        )
        and all(
            payload["authority"]["retry20_authority"][name]["sha256"]
            == EXPECTED_SHA256[name]
            for name in RETRY20_AUTHORITY
        )
        and source_authority["commit"]
        == "f52c197ed39d12e087d02c50f412d90d418f6186"
        and source_authority["tree"]
        == "6b658fbc98077fe0648cba724921679126464181",
        "wrf_git_object_and_generator_authority": (
            source_authority["verdict"]
            == "TRACKED_WRF_SOURCE_AND_GENERATOR_REPRODUCED"
            and source_authority["dirty_worktree_rejected"]
            and source_authority["registry_generator"]["sha256"]
            == "ff3e2ef7c1032bd54941252618377a261c91a5f32c84b356151f5615a6b807b4"
        ),
        "candidate_off_e0d0_and_interface_clean": (
            interface_authority["verdict"]
            == "CANDIDATE_OFF_E0D0_EXACT_CANDIDATE_ON_INTERFACE_CLEAN"
            and interface_authority["candidate_off"]["exact"]
            and interface_authority["production_interface"]["leaf_count"] == 106
            and not any(
                interface_authority["candidate_on"]["forbidden_hlo_tokens"].values()
            )
        ),
        "sealed_ni_v10_authority_authenticates": payload["authority"]
        ["sealed_ni_v10_authority"]["sha256"]
        == "4132b6e467b6313d4ae80171b1817119478fc3d0a8cfebc0244d0b893f407497",
        "retained_rereview_authenticates": all(
            payload["authority"]["retained_rereview"]["artifacts"][name]["sha256"]
            == FINAL_P1_REVIEW_SHA256[name]
            for name in FINAL_P1_REVIEW
        ),
        "tests_passed": tests["returncode"] == 0,
        "both_real_edges_have_nonzero_coupled_commutator": all(
            row["late_coupling_commutator_nonzero_for_all_coupled_fields"]
            for row in edges.values()
        ),
        "both_mu_controls_exact": all(row["mu_control_exact"] for row in edges.values()),
        "retained_package_is_decoupled": payload["retained_step8800_package"]["classification"]
        == "decoupled_physical_records",
        "no_gpu": True,
    }
    payload["verdict"] = (
        "NESTED_BOUNDARY_FINAL_P1_REPAIR"
        if all(payload["gates"].values())
        else "REPAIR_BLOCKED"
    )
    payload["proof_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"verdict": payload["verdict"], "proof": str(args.output), "proof_sha256": payload["proof_sha256"]}, sort_keys=True))
    return (
        0
        if payload["verdict"]
        == "NESTED_BOUNDARY_FINAL_P1_REPAIR"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
