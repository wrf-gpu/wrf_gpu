#!/usr/bin/env python3
"""Independent adversarial audit of the v0234 Fable PBL discriminator.

This script intentionally does not import any Fable script.  It reconstructs
the WRF dump arrays, coefficient systems, Thomas solves, mixed-authority
tendency identities, substitution floor, and lower-span state identity from
the immutable artifacts named by the sprint contract.

CPU/numpy only.  No JAX, GPU, WRF, or MPI execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SEALED_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084"
)
WRF_DUMP_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/"
    "evidence-dumps-fresh-d03-runtime"
)
FABLE_OPERANDS = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_fable_pbl_solve_discriminator/gpu-operands.npz"
)
REFERENCE_ARCHIVE = SEALED_ROOT / "cpu-reference/pbl-sp2-reference-28.npz"
REFERENCE_MANIFEST = SEALED_ROOT / "cpu-reference/pbl-sp2-reference-manifest.json"
COMPARISON = SEALED_ROOT / "comparator/scientific-comparison.json"
CAPTURE_MANIFEST = (
    SEALED_ROOT
    / "capture/authentic-ac6712170cbe5084/manifest.json"
)
PRIOR_REFERENCE = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v0234-mynn-sp2-input-provenance/"
    ".agent/sprints/2026-07-19-v0234-offline-comparator-preparation-gpt/"
    "retained-carry-reference-bundle.json"
)
WRF_SOURCE = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2/"
    "source/instrumented/phys/MYNN-EDMF/module_bl_mynnedmf.F90"
)
PORT_SOURCE = REPO / "src/gpuwrf/physics/mynn_pbl.py"

EXPECTED = {
    REFERENCE_ARCHIVE:
        "a6416b7245d26f23f0df398dd6a3a926a1749cba2069dc3d0ea39c98bc3d2566",
    FABLE_OPERANDS:
        "90d0b6fbfe4d57a5b2fbf29dd7fcd9c88939667c1bbc892b0a9884a5d4b64922",
    COMPARISON:
        "b582d39042eacb8e3a96b78e5cc91814bbd99f76a5e802aa03eb4d5094cbc0ba",
}
EXPECTED_DUMP_TREE = (
    "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
)
DT = 6.0
RANK_COUNT = 6


class AuditFailure(RuntimeError):
    """Fail-closed evidence or arithmetic failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items()
               if key != "canonical_payload_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise AuditFailure(f"NONFINITE_OR_OBJECT:{name}")
    return array


def rms(value: np.ndarray) -> float:
    array = require_finite("rms", np.asarray(value, np.float64))
    return float(np.sqrt(np.mean(array * array)))


def sse(value: np.ndarray) -> float:
    array = require_finite("sse", np.asarray(value, np.float64))
    return float(np.sum(array * array))


def delta_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_array = require_finite("left", np.asarray(left))
    right_array = require_finite("right", np.asarray(right))
    if left_array.shape != right_array.shape:
        raise AuditFailure(
            f"SHAPE_MISMATCH:{left_array.shape}:{right_array.shape}"
        )
    delta = left_array.astype(np.float64) - right_array.astype(np.float64)
    return {
        "rms": rms(delta),
        "max_abs": float(np.max(np.abs(delta))),
        "sse": sse(delta),
        "exact_fraction": float(np.mean(left_array == right_array)),
    }


def max_ulp_float32(left: np.ndarray, right: np.ndarray) -> int:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        raise AuditFailure("ULP_SHAPE")
    left_bits = left_array.view(np.uint32)
    right_bits = right_array.view(np.uint32)
    sign_mismatch = ((left_bits ^ right_bits) & np.uint32(0x80000000)) != 0
    both_zero = (left_array == 0.0) & (right_array == 0.0)
    if np.any(sign_mismatch & ~both_zero):
        raise AuditFailure("ULP_SIGN_CROSSING")
    distance = np.abs(
        left_bits.astype(np.int64) - right_bits.astype(np.int64)
    )
    return int(np.max(distance))


def parse_meta(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    integer_keys = {
        "rank", "real_bytes", "ids_ide_jds_jde_kds_kde",
        "ims_ime_jms_jme_kms_kme", "ips_ipe_jps_jpe_kps_kpe",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        result[parts[0]] = (
            [int(item) for item in parts[1:]]
            if parts[0] in integer_keys else " ".join(parts[1:])
        )
    if result.get("schema") != "wrfgpu2-mynn-sp2-v1":
        raise AuditFailure(f"WRF_META_SCHEMA:{path}")
    if result.get("real_bytes") != [4]:
        raise AuditFailure(f"WRF_META_PRECISION:{path}")
    return result


class IndependentWrfDump:
    """Separate big-endian Fortran-dump reader with overlap rejection."""

    def __init__(self, root: Path):
        self.root = root
        self.meta = [
            parse_meta(root / "mynnsp2" / f"rank{rank:04d}" / "meta.txt")
            for rank in range(RANK_COUNT)
        ]
        ids, ide, jds, jde, kds, kde = self.meta[0][
            "ids_ide_jds_jde_kds_kde"
        ]
        self.i = range(ids, ide)
        self.j = range(jds, jde)
        self.k = range(kds, kde)

    def columns(self, tag: str) -> np.ndarray:
        records: dict[tuple[int, int], tuple[int, int, np.ndarray]] = {}
        vertical_bounds: tuple[int, int] | None = None
        for rank in range(RANK_COUNT):
            path = (
                self.root / "mynnsp2" / f"rank{rank:04d}"
                / f"step000001_columns__{tag}.bin"
            )
            with path.open("rb") as stream:
                while True:
                    header = np.fromfile(stream, dtype=">i4", count=5)
                    if header.size == 0:
                        break
                    if header.size != 5:
                        raise AuditFailure(f"COLUMN_HEADER:{tag}:{rank}")
                    i, j, lower, upper, count = map(int, header)
                    values = np.fromfile(stream, dtype=">f4", count=count)
                    if values.size != count or count != upper - lower + 1:
                        raise AuditFailure(f"COLUMN_PAYLOAD:{tag}:{rank}:{i}:{j}")
                    if (i, j) in records:
                        raise AuditFailure(f"COLUMN_DUPLICATE:{tag}:{i}:{j}")
                    if not (i in self.i and j in self.j):
                        raise AuditFailure(f"COLUMN_OUTSIDE_DOMAIN:{tag}:{i}:{j}")
                    current = (lower, upper)
                    vertical_bounds = current if vertical_bounds is None else vertical_bounds
                    if current != vertical_bounds:
                        raise AuditFailure(f"COLUMN_VERTICAL_DRIFT:{tag}")
                    records[(i, j)] = (
                        lower, upper, values.astype(np.float32, copy=False)
                    )
        if vertical_bounds is None or len(records) != len(self.i) * len(self.j):
            raise AuditFailure(f"COLUMN_COVERAGE:{tag}:{len(records)}")
        nz = vertical_bounds[1] - vertical_bounds[0] + 1
        result = np.empty((nz, len(self.j), len(self.i)), dtype=np.float32)
        for (i, j), (_lower, _upper, values) in records.items():
            result[:, j - self.j.start, i - self.i.start] = values
        return require_finite(tag, result)

    def outer3(self, tag: str) -> np.ndarray:
        result = np.full(
            (len(self.k), len(self.j), len(self.i)), np.nan, dtype=np.float32
        )
        occupied = np.zeros(result.shape, dtype=bool)
        for rank, meta in enumerate(self.meta):
            ims, ime, jms, jme, kms, kme = meta[
                "ims_ime_jms_jme_kms_kme"
            ]
            ips, ipe, jps, jpe, kps, kpe = meta[
                "ips_ipe_jps_jpe_kps_kpe"
            ]
            path = (
                self.root / "mynnsp2" / f"rank{rank:04d}"
                / f"step000001_outer__{tag}.bin"
            )
            raw = np.fromfile(path, dtype=">f4")
            shape = (ime - ims + 1, kme - kms + 1, jme - jms + 1)
            if raw.size != math.prod(shape):
                raise AuditFailure(f"OUTER_PAYLOAD:{tag}:{rank}")
            memory = raw.reshape(shape, order="F")
            for j in range(max(jps, self.j.start), min(jpe, self.j.stop - 1) + 1):
                for k in range(max(kps, self.k.start), min(kpe, self.k.stop - 1) + 1):
                    for i in range(max(ips, self.i.start), min(ipe, self.i.stop - 1) + 1):
                        target = (
                            k - self.k.start, j - self.j.start, i - self.i.start
                        )
                        if occupied[target]:
                            raise AuditFailure(f"OUTER_DUPLICATE:{tag}:{target}")
                        result[target] = memory[i - ims, k - kms, j - jms]
                        occupied[target] = True
        if not occupied.all():
            raise AuditFailure(f"OUTER_COVERAGE:{tag}")
        return require_finite(tag, result)


def thomas_reference(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                     d: np.ndarray) -> np.ndarray:
    """Independent WRF tridiag2 recurrence in the input dtype."""

    if not (a.shape == b.shape == c.shape == d.shape):
        raise AuditFailure("THOMAS_SHAPE")
    cp = np.empty_like(b)
    dp = np.empty_like(b)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for level in range(1, a.shape[0]):
        denominator = b[level] - cp[level - 1] * a[level]
        if not np.isfinite(denominator).all() or np.any(denominator == 0):
            raise AuditFailure(f"THOMAS_SINGULAR:{level}")
        cp[level] = c[level] / denominator
        dp[level] = (
            d[level] - dp[level - 1] * a[level]
        ) / denominator
    result = np.empty_like(d)
    result[-1] = dp[-1]
    for level in range(a.shape[0] - 2, -1, -1):
        result[level] = dp[level] - cp[level] * result[level + 1]
    return require_finite("thomas_result", result)


def reconstruct_wrf_coefficients(wrf: dict[str, np.ndarray],
                                 component: str) -> tuple[dict[str, np.ndarray],
                                                          dict[str, Any]]:
    """Re-evaluate all active and inactive WRF U/V coefficient terms."""

    f32 = np.float32
    state = wrf[f"bc_{component}"]
    rho = wrf["bc_rho"]
    dtz = wrf["bc_dtz"]
    kmdz = wrf["bc_kmdz"]
    saw = wrf["bc_s_aw"]
    sdaw = wrf["bc_sd_aw"]
    sawx = wrf[f"bc_s_aw{component}"]
    sdawx = wrf[f"bc_sd_aw{component}"]
    subs = wrf[f"bc_sub_{component}"]
    detr = wrf[f"bc_det_{component}"]
    lower = wrf["bc_lower_operands"]
    delt = lower[5]
    ust = lower[6]
    wind = lower[7]
    ocean = lower[8] if component == "u" else lower[9]
    rhosfc = lower[10]
    onoff = lower[17]
    rhoinv = np.empty_like(rho)
    rhoinv[0] = f32(1.0) / rho[0]
    rhoinv[1:] = f32(1.0) / np.maximum(rho[1:], f32(1.0e-4))
    p = dtz * rhoinv
    nz = state.shape[0]
    a = np.empty_like(state)
    b = np.empty_like(state)
    c = np.empty_like(state)
    d = np.empty_like(state)

    a[0] = -dtz[0] * kmdz[0] * rhoinv[0]
    b[0] = (
        f32(1.0)
        + dtz[0] * (kmdz[1] + rhosfc * ust ** f32(2.0) / wind) * rhoinv[0]
        - f32(0.5) * dtz[0] * rhoinv[0] * saw[1] * onoff
        - f32(0.5) * dtz[0] * rhoinv[0] * sdaw[1] * onoff
    )
    c[0] = (
        -dtz[0] * kmdz[1] * rhoinv[0]
        - f32(0.5) * dtz[0] * rhoinv[0] * saw[1] * onoff
        - f32(0.5) * dtz[0] * rhoinv[0] * sdaw[1] * onoff
    )
    d[0] = (
        state[0]
        + dtz[0] * ocean * ust ** f32(2.0) / wind
        - dtz[0] * rhoinv[0] * sawx[1] * onoff
        + dtz[0] * rhoinv[0] * sdawx[1] * onoff
        + subs[0] * delt
        + detr[0] * delt
    )
    interior = slice(1, nz - 1)
    a[interior] = (
        -dtz[interior] * kmdz[1:nz - 1] * rhoinv[interior]
        + f32(0.5) * dtz[interior] * rhoinv[interior]
        * saw[1:nz - 1] * onoff
        + f32(0.5) * dtz[interior] * rhoinv[interior]
        * sdaw[1:nz - 1] * onoff
    )
    b[interior] = (
        f32(1.0)
        + dtz[interior] * (kmdz[1:nz - 1] + kmdz[2:nz])
        * rhoinv[interior]
        + f32(0.5) * dtz[interior] * rhoinv[interior]
        * (saw[1:nz - 1] - saw[2:nz]) * onoff
        + f32(0.5) * dtz[interior] * rhoinv[interior]
        * (sdaw[1:nz - 1] - sdaw[2:nz]) * onoff
    )
    c[interior] = (
        -dtz[interior] * kmdz[2:nz] * rhoinv[interior]
        - f32(0.5) * dtz[interior] * rhoinv[interior]
        * saw[2:nz] * onoff
        - f32(0.5) * dtz[interior] * rhoinv[interior]
        * sdaw[2:nz] * onoff
    )
    d[interior] = (
        state[interior]
        + dtz[interior] * rhoinv[interior]
        * (sawx[1:nz - 1] - sawx[2:nz]) * onoff
        - dtz[interior] * rhoinv[interior]
        * (sdawx[1:nz - 1] - sdawx[2:nz]) * onoff
        + subs[interior] * delt
        + detr[interior] * delt
    )
    a[-1] = f32(0.0)
    b[-1] = f32(1.0)
    c[-1] = f32(0.0)
    d[-1] = state[-1]

    active_terms = {
        "component": component,
        "dt_unique": np.unique(delt).astype(float).tolist(),
        "onoff_unique": np.unique(onoff).astype(float).tolist(),
        "minimum_rho": float(np.min(rho)),
        "kmdz_bottom_max_abs": float(np.max(np.abs(kmdz[0]))),
        "surface_diffusion_term_port_extra_max_abs": float(
            np.max(np.abs(p[0] * kmdz[0]))
        ),
        "drag_term_rms": rms(p[0] * rhosfc * ust * ust / wind),
        "mass_flux_implicit_nonzero": bool(np.any(saw != 0)),
        "mass_flux_rhs_nonzero": bool(np.any(sawx != 0)),
        "downdraft_implicit_max_abs": float(np.max(np.abs(sdaw))),
        "downdraft_rhs_max_abs": float(np.max(np.abs(sdawx))),
        "subsidence_max_abs": float(np.max(np.abs(subs))),
        "detrainment_max_abs": float(np.max(np.abs(detr))),
        "ocean_velocity_max_abs": float(np.max(np.abs(ocean))),
    }
    return {"a": a, "b": b, "c": c, "d": d}, active_terms


def port_semantics_on_wrf_operands(wrf: dict[str, np.ndarray],
                                   component: str) -> dict[str, np.ndarray]:
    """Evaluate the port's operational momentum formula on WRF operands.

    Inactive WRF-only terms are deliberately absent.  The port's structural
    bottom kmdz[0] term remains present so the numerical zero is tested rather
    than assumed.
    """

    state = wrf[f"bc_{component}"]
    rho = wrf["bc_rho"]
    dtz = wrf["bc_dtz"]
    kmdz = wrf["bc_kmdz"]
    saw = wrf["bc_s_aw"]
    sawx = wrf[f"bc_s_aw{component}"]
    lower = wrf["bc_lower_operands"]
    ust, wind, rhosfc = lower[6], lower[7], lower[10]
    rhoinv = np.float32(1.0) / np.maximum(rho, np.float32(1.0e-4))
    drag = rhosfc * ust * ust / wind
    nz = state.shape[0]
    a = np.empty_like(state)
    b = np.empty_like(state)
    c = np.empty_like(state)
    d = np.empty_like(state)
    half0 = np.float32(0.5) * dtz[0] * rhoinv[0]
    a[0] = -dtz[0] * kmdz[0] * rhoinv[0]
    b[0] = (
        np.float32(1.0)
        + dtz[0] * (kmdz[1] + kmdz[0] + drag) * rhoinv[0]
        - half0 * saw[1]
    )
    c[0] = -dtz[0] * kmdz[1] * rhoinv[0] - half0 * saw[1]
    d[0] = state[0] - dtz[0] * rhoinv[0] * sawx[1]
    interior = slice(1, nz - 1)
    half_i = np.float32(0.5) * dtz[interior] * rhoinv[interior]
    a[interior] = (
        -dtz[interior] * kmdz[1:nz - 1] * rhoinv[interior]
        + half_i * saw[1:nz - 1]
    )
    b[interior] = (
        np.float32(1.0)
        + dtz[interior] * (kmdz[1:nz - 1] + kmdz[2:nz])
        * rhoinv[interior]
        + half_i * (saw[1:nz - 1] - saw[2:nz])
    )
    c[interior] = (
        -dtz[interior] * kmdz[2:nz] * rhoinv[interior]
        - half_i * saw[2:nz]
    )
    d[interior] = (
        state[interior]
        + dtz[interior] * rhoinv[interior]
        * (sawx[1:nz - 1] - sawx[2:nz])
    )
    a[-1] = np.float32(0.0)
    b[-1] = np.float32(1.0)
    c[-1] = np.float32(0.0)
    d[-1] = state[-1]
    return {"a": a, "b": b, "c": c, "d": d}


def audit_hashes() -> dict[str, Any]:
    records: dict[str, Any] = {}
    all_match = True
    for path, expected in EXPECTED.items():
        if not path.is_file() or path.is_symlink():
            raise AuditFailure(f"INPUT_NOT_REGULAR:{path}")
        actual = sha256_file(path)
        match = actual == expected
        records[str(path)] = {
            "expected": expected, "actual": actual, "match": match,
        }
        all_match = all_match and match
    tree: dict[str, str] = {}
    for path in sorted(WRF_DUMP_ROOT.rglob("*")):
        if path.is_symlink():
            raise AuditFailure(f"DUMP_SYMLINK:{path}")
        if path.is_file():
            tree[path.relative_to(WRF_DUMP_ROOT).as_posix()] = sha256_file(path)
    tree_digest = hashlib.sha256(
        json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tree_match = len(tree) == 462 and tree_digest == EXPECTED_DUMP_TREE
    all_match = all_match and tree_match
    if not all_match:
        raise AuditFailure("SEALED_INPUT_HASH_DRIFT")
    return {
        "files": records,
        "dump_tree": {
            "expected": EXPECTED_DUMP_TREE,
            "actual": tree_digest,
            "file_count": len(tree),
            "match": tree_match,
        },
        "all_match": all_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        print(f"REFUSE:OUTPUT_NOT_FRESH:{output}", file=sys.stderr)
        return 74

    try:
        proof: dict[str, Any] = {
            "schema": "wrfgpu2-v0234-gpt-adversarial-audit-v1",
            "backend": "numpy-cpu",
            "gpu_actions": 0,
            "wrf_or_mpi_executions": 0,
            "predeclared_thresholds": {
                "authority_mismatch_rms_ratio_min_each_component": 3.0,
                "floor_span_sse_ratio_relative_distance_max": 0.10,
                "substitution_effect_rms_over_floor_max": 0.25,
                "claimed_one_order_effect_rms_over_floor_max": 0.10,
                "tautology_absolute_rms_max": 5.0e-12,
                "port_thomas_max_abs_max": 5.0e-15,
                "port_formula_on_wrf_operands_max_float32_ulp": 1,
                "port_formula_on_wrf_operands_exact_fraction_min": 0.99999,
            },
        }
        proof["input_audit"] = audit_hashes()
        manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
        comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
        capture_manifest = json.loads(CAPTURE_MANIFEST.read_text(encoding="utf-8"))
        prior = json.loads(PRIOR_REFERENCE.read_text(encoding="utf-8"))
        if payload_sha(manifest) != manifest.get("canonical_payload_sha256"):
            raise AuditFailure("REFERENCE_MANIFEST_CANONICAL_DRIFT")
        if payload_sha(comparison) != comparison.get("canonical_payload_sha256"):
            raise AuditFailure("COMPARISON_CANONICAL_DRIFT")
        if payload_sha(capture_manifest) != capture_manifest.get("canonical_payload_sha256"):
            raise AuditFailure("CAPTURE_MANIFEST_CANONICAL_DRIFT")
        if payload_sha(prior) != prior.get("canonical_payload_sha256"):
            raise AuditFailure("PRIOR_REFERENCE_CANONICAL_DRIFT")

        dump = IndependentWrfDump(WRF_DUMP_ROOT)
        wrf_tags = (
            "bc_lower_operands", "bc_rho", "bc_u", "bc_v", "bc_dtz",
            "bc_kmdz", "bc_s_aw", "bc_sd_aw", "bc_s_awu", "bc_sd_awu",
            "bc_s_awv", "bc_sd_awv", "bc_sub_u", "bc_det_u",
            "bc_sub_v", "bc_det_v", "solve_u_a", "solve_u_b", "solve_u_c",
            "solve_u_d", "solve_u_x", "solve_v_a", "solve_v_b", "solve_v_c",
            "solve_v_d", "solve_v_x",
        )
        wrf = {name: dump.columns(name) for name in wrf_tags}
        wrf_tendency = {
            "u": dump.outer3("rublten_exit"),
            "v": dump.outer3("rvblten_exit"),
        }
        with np.load(REFERENCE_ARCHIVE, allow_pickle=False) as archive_file:
            archive = {name: require_finite(name, archive_file[name])
                       for name in archive_file.files}
        with np.load(FABLE_OPERANDS, allow_pickle=False) as operands_file:
            operands = {name: require_finite(name, operands_file[name])
                        for name in operands_file.files}

        # A1: mixed producer lineage plus independently reconstructed tendency.
        matrix = {
            item["name"]: item
            for item in prior.get("availability_matrix", [])
            if item.get("available")
        }
        lineage: dict[str, Any] = {
            "reference_manifest_approved_head": manifest.get("approved_head"),
            "capture_nonce": manifest.get("capture_nonce"),
            "capture_manifest_file_sha256": sha256_file(CAPTURE_MANIFEST),
            "sp2_manifest_source_u": manifest["provenance"]["sp2_rublten"]["source"],
            "solve_manifest_source_u": manifest["provenance"]["solve_u_x"]["source"],
            "sp2_manifest_source_v": manifest["provenance"]["sp2_rvblten"]["source"],
            "solve_manifest_source_v": manifest["provenance"]["solve_v_x"]["source"],
            "mixed_producer_classes": (
                manifest["provenance"]["sp2_rublten"]["source"]
                != manifest["provenance"]["solve_u_x"]["source"]
            ),
            "prior_sp2_sources": {},
        }
        for component, field in (("u", "sp2_rublten"), ("v", "sp2_rvblten")):
            source_path = Path(matrix[field]["source_path"])
            source_array = np.load(source_path, allow_pickle=False)
            source_match = bool(np.array_equal(source_array, archive[field]))
            lineage["prior_sp2_sources"][component] = {
                "path": str(source_path),
                "file_sha256": sha256_file(source_path),
                "bitpayload_sha256": array_sha(source_array),
                "bundle_bitpayload_sha256": array_sha(archive[field]),
                "bit_exact_to_bundle": source_match,
            }
            if not source_match:
                raise AuditFailure(f"PRIOR_SP2_NOT_BUNDLE:{component}")

        authority: dict[str, Any] = {"lineage": lineage, "components": {}}
        authority_pass = lineage["mixed_producer_classes"]
        for component in ("u", "v"):
            field = f"sp2_r{component}blten"
            implied = (archive[f"solve_{component}_x"] - operands[component]) / DT
            delta = implied - archive[field]
            baseline = archive[field] - wrf_tendency[component]
            item = {
                "solve_implied_vs_bundle_sp2": delta_metrics(implied, archive[field]),
                "bundle_sp2_vs_wrf": delta_metrics(archive[field], wrf_tendency[component]),
                "mismatch_rms_ratio_to_bundle_sp2_vs_wrf": rms(delta) / rms(baseline),
                "implied_tendency_bitpayload_sha256": array_sha(implied),
            }
            item["passes_ratio_gate"] = (
                item["mismatch_rms_ratio_to_bundle_sp2_vs_wrf"]
                >= proof["predeclared_thresholds"][
                    "authority_mismatch_rms_ratio_min_each_component"
                ]
            )
            authority["components"][component] = item
            authority_pass = authority_pass and item["passes_ratio_gate"]
        authority["passes"] = bool(authority_pass)
        proof["A1_authority_mismatch"] = authority

        # A2: compare substitution residuals to the replay floor and expose
        # Fable's over-strong one-to-two-order wording separately.
        floor_proof: dict[str, Any] = {"components": {}}
        floor_pass = True
        one_order_all_channels = True
        for component in ("u", "v"):
            field = f"r{component}blten"
            baseline = archive[f"sp2_{field}"] - wrf_tendency[component]
            implied = (archive[f"solve_{component}_x"] - operands[component]) / DT
            floor = implied - wrf_tendency[component]
            component_record: dict[str, Any] = {
                "baseline_rms": rms(baseline),
                "replay_floor_rms": rms(floor),
                "replay_floor_sse_ratio_vs_mixed_baseline": sse(floor) / sse(baseline),
                "spans": {},
            }
            for span in ("surface", "mixing"):
                residual = archive[f"residual_after_{span}_{field}"]
                effect = residual - floor
                ratio = sse(residual) / sse(baseline)
                floor_ratio = sse(floor) / sse(baseline)
                ratio_distance = abs(ratio - floor_ratio) / floor_ratio
                effect_over_floor = rms(effect) / rms(floor)
                core_pass = (
                    ratio_distance
                    <= proof["predeclared_thresholds"][
                        "floor_span_sse_ratio_relative_distance_max"
                    ]
                    and effect_over_floor
                    <= proof["predeclared_thresholds"][
                        "substitution_effect_rms_over_floor_max"
                    ]
                )
                one_order = (
                    effect_over_floor
                    <= proof["predeclared_thresholds"][
                        "claimed_one_order_effect_rms_over_floor_max"
                    ]
                )
                component_record["spans"][span] = {
                    "sealed_sse_ratio_vs_mixed_baseline": ratio,
                    "replay_floor_sse_ratio_vs_mixed_baseline": floor_ratio,
                    "relative_sse_ratio_distance_from_floor": ratio_distance,
                    "substitution_effect_rms": rms(effect),
                    "substitution_effect_rms_over_floor": effect_over_floor,
                    "substitution_effect_sse_over_floor": sse(effect) / sse(floor),
                    "residual_sse_over_replay_floor_sse": sse(residual) / sse(floor),
                    "floor_dominance_core_pass": core_pass,
                    "fable_claimed_at_least_one_order_pass": one_order,
                }
                floor_pass = floor_pass and core_pass
                one_order_all_channels = one_order_all_channels and one_order
            floor_proof["components"][component] = component_record
        floor_proof["floor_dominance_passes"] = bool(floor_pass)
        floor_proof["fable_one_to_two_orders_wording_passes_all_channels"] = bool(
            one_order_all_channels
        )
        floor_proof["adversarial_verdict"] = (
            "FLOOR_DOMINATED_BUT_ONE_TO_TWO_ORDERS_OVERSTATED"
            if floor_pass and not one_order_all_channels
            else "FLOOR_DOMINANCE_CONFIRMED"
        )
        proof["A2_floor_artifact"] = floor_proof

        # A3: lower residual is algebraically the pre-solve state delta / dt.
        tautology: dict[str, Any] = {"components": {}}
        tautology_pass = True
        for component in ("u", "v"):
            field = f"r{component}blten"
            residual = archive[f"residual_after_lower_bc_{field}"]
            state_identity = (wrf[f"bc_{component}"] - operands[component]) / DT
            closure = delta_metrics(residual, state_identity)
            item = {
                "lower_residual": delta_metrics(residual, np.zeros_like(residual)),
                "state_delta_over_dt": delta_metrics(
                    state_identity, np.zeros_like(state_identity)
                ),
                "identity_closure": closure,
                "port_vs_wrf_presolve_state": delta_metrics(
                    operands[component], wrf[f"bc_{component}"]
                ),
                "passes": closure["rms"]
                <= proof["predeclared_thresholds"]["tautology_absolute_rms_max"],
            }
            tautology["components"][component] = item
            tautology_pass = tautology_pass and item["passes"]
        tautology["passes"] = bool(tautology_pass)
        tautology["meaning"] = (
            "The lower residual is the pre-solve WRF-versus-port state delta "
            "divided by dt, not evidence that lower coefficients caused the "
            "mixed-authority production tendency error."
        )
        proof["A3_lower_span_state_identity"] = tautology

        # A4: all operational WRF coefficient terms and both solvers.
        coefficient_proof: dict[str, Any] = {
            "source_bindings": {
                "wrf_source": {
                    "path": str(WRF_SOURCE), "sha256": sha256_file(WRF_SOURCE),
                    "momentum_lines": "3980-4199", "solver_lines": "5426-5455",
                },
                "port_source": {
                    "path": str(PORT_SOURCE), "sha256": sha256_file(PORT_SOURCE),
                    "coefficient_symbol": "_diffusion_solve_with_mf",
                    "caller_symbol": "_apply_mean_tendencies",
                },
            },
            "components": {},
            "solver": {},
        }
        coefficients_pass = True
        for component in ("u", "v"):
            rebuilt, coverage = reconstruct_wrf_coefficients(wrf, component)
            port_on_wrf = port_semantics_on_wrf_operands(wrf, component)
            comparisons: dict[str, Any] = {}
            port_comparisons: dict[str, Any] = {}
            for name in ("a", "b", "c", "d"):
                dumped = wrf[f"solve_{component}_{name}"]
                rebuilt_bit_exact = bool(np.array_equal(
                    rebuilt[name].view(np.uint32), dumped.view(np.uint32)
                ))
                port_bit_exact = bool(np.array_equal(
                    port_on_wrf[name].view(np.uint32), dumped.view(np.uint32)
                ))
                port_metrics = delta_metrics(port_on_wrf[name], dumped)
                port_max_ulp = max_ulp_float32(port_on_wrf[name], dumped)
                port_near_exact = (
                    port_max_ulp
                    <= proof["predeclared_thresholds"][
                        "port_formula_on_wrf_operands_max_float32_ulp"
                    ]
                    and port_metrics["exact_fraction"]
                    >= proof["predeclared_thresholds"][
                        "port_formula_on_wrf_operands_exact_fraction_min"
                    ]
                )
                comparisons[name] = {
                    **delta_metrics(rebuilt[name], dumped),
                    "bit_exact_uint32": rebuilt_bit_exact,
                }
                port_comparisons[name] = {
                    **port_metrics,
                    "bit_exact_uint32": port_bit_exact,
                    "max_float32_ulp": port_max_ulp,
                    "near_exact_gate": port_near_exact,
                }
                coefficients_pass = (
                    coefficients_pass and rebuilt_bit_exact and port_near_exact
                )
            inactive_ok = (
                coverage["dt_unique"] == [DT]
                and coverage["onoff_unique"] == [1.0]
                and coverage["minimum_rho"] > 1.0e-4
                and coverage["kmdz_bottom_max_abs"] == 0.0
                and coverage["downdraft_implicit_max_abs"] == 0.0
                and coverage["downdraft_rhs_max_abs"] == 0.0
                and coverage["subsidence_max_abs"] == 0.0
                and coverage["detrainment_max_abs"] == 0.0
                and coverage["ocean_velocity_max_abs"] == 0.0
                and coverage["mass_flux_implicit_nonzero"]
                and coverage["mass_flux_rhs_nonzero"]
            )
            coefficients_pass = coefficients_pass and inactive_ok
            coefficient_proof["components"][component] = {
                "active_and_inactive_term_coverage": coverage,
                "operational_coverage_passes": inactive_ok,
                "wrf_reconstruction_vs_dump": comparisons,
                "port_formula_on_wrf_operands_vs_dump": port_comparisons,
            }

            wrf_solution = thomas_reference(
                wrf[f"solve_{component}_a"], wrf[f"solve_{component}_b"],
                wrf[f"solve_{component}_c"], wrf[f"solve_{component}_d"],
            )
            port_solution = thomas_reference(
                archive[f"solve_{component}_a"].astype(np.float64),
                archive[f"solve_{component}_b"].astype(np.float64),
                archive[f"solve_{component}_c"].astype(np.float64),
                archive[f"solve_{component}_d"].astype(np.float64),
            )
            wrf_solver_metrics = delta_metrics(
                wrf_solution, wrf[f"solve_{component}_x"]
            )
            port_solver_metrics = delta_metrics(
                port_solution, archive[f"solve_{component}_x"]
            )
            wrf_bit_exact = bool(np.array_equal(
                wrf_solution.view(np.uint32),
                wrf[f"solve_{component}_x"].view(np.uint32),
            ))
            port_within = (
                port_solver_metrics["max_abs"]
                <= proof["predeclared_thresholds"]["port_thomas_max_abs_max"]
            )
            coefficient_proof["solver"][component] = {
                "wrf_float32_thomas_vs_dump": {
                    **wrf_solver_metrics, "bit_exact_uint32": wrf_bit_exact,
                },
                "port_float64_thomas_vs_sealed_x": {
                    **port_solver_metrics, "within_gate": port_within,
                },
            }
            coefficients_pass = coefficients_pass and wrf_bit_exact and port_within
        coefficient_proof["passes"] = bool(coefficients_pass)
        coefficient_proof["literal_port_on_wrf_bit_exact_all"] = bool(
            all(
                item["bit_exact_uint32"]
                for component in coefficient_proof["components"].values()
                for item in component[
                    "port_formula_on_wrf_operands_vs_dump"
                ].values()
            )
        )
        coefficient_proof["adversarial_note"] = (
            "WRF reconstruction is bit-exact. Port semantics are within one "
            "float32 ULP but not literally bit-exact because the bottom drag "
            "product is associated as rhosfc*ust*ust rather than "
            "rhosfc*(ust**2); this is recorded, not treated as a material "
            "coefficient mechanism."
        )
        proof["A4_coefficient_and_solver"] = coefficient_proof

        core_invalidation_stands = bool(
            proof["A1_authority_mismatch"]["passes"]
            and proof["A2_floor_artifact"]["floor_dominance_passes"]
            and proof["A3_lower_span_state_identity"]["passes"]
            and proof["A4_coefficient_and_solver"]["passes"]
        )
        proof["terminal"] = {
            "fable_core_invalidation_stands": core_invalidation_stands,
            "fable_quantifier_correction_required": not one_order_all_channels,
            "fable_literal_coefficient_identity_correction_required": not (
                coefficient_proof["literal_port_on_wrf_bit_exact_all"]
            ),
            "sealed_source_localized_lower_bc_valid_as_production_attribution": False,
            "phase_b_authorized": core_invalidation_stands,
            "verdict": (
                "FABLE_INVALIDATION_STANDS_WITH_A2_MAGNITUDE_CORRECTION"
                if core_invalidation_stands
                else "FABLE_INVALIDATION_NOT_ESTABLISHED"
            ),
        }
        if not core_invalidation_stands:
            raise AuditFailure(
                "CORE_INVALIDATION_GATE_FAILED:"
                f"A1={proof['A1_authority_mismatch']['passes']}:"
                f"A2={proof['A2_floor_artifact']['floor_dominance_passes']}:"
                f"A3={proof['A3_lower_span_state_identity']['passes']}:"
                f"A4={proof['A4_coefficient_and_solver']['passes']}"
            )

        proof["canonical_payload_sha256"] = payload_sha(proof)
        output.write_text(
            json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "verdict": proof["terminal"]["verdict"],
            "authority_ratio_u": authority["components"]["u"][
                "mismatch_rms_ratio_to_bundle_sp2_vs_wrf"
            ],
            "authority_ratio_v": authority["components"]["v"][
                "mismatch_rms_ratio_to_bundle_sp2_vs_wrf"
            ],
            "one_order_wording_all_channels": one_order_all_channels,
            "coefficient_solver_pass": coefficients_pass,
            "output": str(output),
        }, sort_keys=True))
        return 0
    except (
        AuditFailure, OSError, ValueError, TypeError, KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
