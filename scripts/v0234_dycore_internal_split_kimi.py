"""CPU/offline internal-split discriminator for the v0234 dycore ladder.

Splits the two terminal systematic facts of the released single GPU arm
(incoming SP3 already systematic; SP3->L1 tendency build systematic) into
source-bound sub-spans treated as potentially independent defects, using
only retained on-disk evidence:

* S0  SP2->SP3 assembly/provenance (calculate_phy_tend mass coupling +
      update_phy_ten add_a2c_u/v face mapping) versus the captured SP2
      MYNN input.
* S1  rk_tendency (implied residual after removing the authenticated
      rk_addtend_dry fold and the relax bundle from L1).
* S2  relax_bdy_dry (u_save/v_save; field-inherited versus target-side
      attribution via the exact E-decomposition).
* S3  spec_bdy_dry (member bitwise identity of the spec-row overwrite and
      quantification of the documented representation asymmetry).

The script imports neither JAX nor gpuwrf and performs no GPU operation.
Every frozen envelope is preserved verbatim and cited by hash; derived
member envelopes use the identical frozen metric on the identical six
retained members and are labelled derived.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_dycore_suboperator_gpt_cpu_analysis as cpu  # noqa: E402
from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-internal-split-kimi"
OUTPUT = SPRINT / "internal-split-analysis.json"

GPU_ARM_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-gpu-arm"
CPU_CONT_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation"
KIMI_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi"

LINEAGE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
RUN_DIR = LINEAGE / "nested_stage_omega_transport_470e6111_dycore_suboperator_ladder1"
RUNTIME_PROOF = RUN_DIR / "dycore-suboperator-ladder-terminal-proof.json"
SAVEPOINT_DIR = RUN_DIR / "savepoints"
GPU_INIT_FRAME = RUN_DIR / "gpu-output/wrfout_d03_2025-03-01_00:00:00"
TRUTH_FRAME = LINEAGE.parent / "run/wrf/wrfout_d03_2025-03-01_00:00:00"

RUNS = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs")
CONTROL_WRFINPUT = RUNS / "control/run/wrfinput_d03"
WRF_SRC = Path("<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF")

MEMBERS = cpu.MEMBERS
SYSTEMATIC_FACTOR = 2.0
DT_D03 = 6.0
SPEC_BDY_WIDTH = 5
SPEC_ZONE = 1
RELAX_ZONE = 4

# Frozen hashes bound in the terminal proofs / operator map.
EXPECTED = {
    "terminal_cpu_proof_self": "c7c05d55d09fdd095f0a3538b097ba527c3b6228a19cb32880db0d1b339f637a",
    "cpu_ladder_analysis_self": "79d0e5998271923aaec20a9ea11d8e23dd1c02fca504acf8963569ac4eb9814e",
    "gpu_arm_proof_self": "f26bdd65099d722d44644d033cf0734ec64a76ee8f5b6ff116ed17724ee55b4c",
    "gpu_arm_analysis_self": "7c5e2df4d5a4d9181047c0d24d688e46ddc2b0c768ac46951d0a0b875a2772da",
    "runtime_proof_self": "f0d0b526ed4769f83b8dd2403343186197c5f439f8ad838ee9733e5bfd170e4d",
    "savepoint_rows": "bc974c1848cb625a1dc3bc4edd879dbf6fb4a6a9a3cbc195b25af2865f5e9503",
}
WRF_SOURCE_SHA256 = {
    "dyn_em/solve_em.F": "680719162a2b9745b4bd12683f512d936a3da02072833cc88e62ae1366f723a9",
    "dyn_em/module_em.F": "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa",
    "dyn_em/module_bc_em.F": "6cfb52b849e3dd0b24769709cf502faa75886da454d9b668af725cccdc56d47f",
    "share/module_bc.F": "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
    "dyn_em/module_first_rk_step_part2.F": "87547130ba43d95633f5a39e82899a0ff0525b4ff28d09528b7894a36771836d",
    "dyn_em/module_first_rk_step_part1.F": "3fd0da3cc1f6017f751eb7544b81aeb4c3ec6636b194994e58182e7d1a5aa8c0",
    "dyn_em/module_small_step_em.F": "cabf1a177d50fb0096db79644af20cfe6d75217dbe63ab406a7e29bb54c17634",
    "dyn_em/module_big_step_utilities_em.F": "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
}
GPU_SOURCE_FILES = [
    "src/gpuwrf/runtime/operational_mode.py",
    "src/gpuwrf/coupling/boundary_apply.py",
    "src/gpuwrf/dynamics/core/rk_addtend_dry.py",
    "src/gpuwrf/dynamics/metrics.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object, *, omit: str | None = None) -> str:
    if omit is not None and isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != omit}
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_self_hashed(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text())
    expected = payload.get("proof_sha256")
    actual = canonical_digest(payload, omit="proof_sha256")
    if expected != actual:
        raise RuntimeError(
            f"self-hash mismatch {path}: expected={expected} actual={actual}"
        )
    return payload, {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "canonical_self_hash": actual,
        "bytes": path.stat().st_size,
    }


def write_self_hashed(path: Path, payload: dict) -> None:
    out = dict(payload)
    out["proof_sha256"] = canonical_digest(out, omit="proof_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# WRF operator arithmetic (fp64 evaluation of the fp32-derived dumps)
# ---------------------------------------------------------------------------

def face_average_u(mass_field: np.ndarray) -> np.ndarray:
    """WRF add_a2c_u for a nested/specified domain (pristine source:
    phys/module_physics_addtendc.F).  Input mass-staggered (k, j, i) with
    shapes (44, 93, 111); output u-staggered (44, 93, 112), nonzero only
    at 0-based rows 1..91 and faces 1..110."""
    nz, ny, nx = mass_field.shape
    out = np.zeros((nz, ny, nx + 1), dtype=np.float64)
    out[:, 1 : ny - 1, 1:nx] = 0.5 * (
        mass_field[:, 1 : ny - 1, : nx - 1] + mass_field[:, 1 : ny - 1, 1:nx]
    )
    return out


def face_average_v(mass_field: np.ndarray) -> np.ndarray:
    """WRF add_a2c_v for a nested/specified domain.  Input (44, 93, 111);
    output (44, 94, 111), nonzero only at rows 1..92 and cols 1..109."""
    nz, ny, nx = mass_field.shape
    out = np.zeros((nz, ny + 1, nx), dtype=np.float64)
    out[:, 1:ny, 1 : nx - 1] = 0.5 * (
        mass_field[:, 1:ny, 1 : nx - 1] + mass_field[:, : ny - 1, 1 : nx - 1]
    )
    return out


def mass_weight(c1: np.ndarray, c2: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """(c1(k)*mu(j,i) + c2(k)) with mu (j,i), c1/c2 (k,) -> (k, j, i)."""
    return c1[:, None, None] * mu[None, :, :] + c2[:, None, None]


def mu_faces(mu_total: np.ndarray, axis: int) -> np.ndarray:
    """Full dry mass at u/v faces with WRF calc_mu_uv edge handling
    (module_big_step_utilities_em.F: edge face reuses the edge mass point)."""
    if axis == 1:  # u faces: (j, i+1)
        left = np.concatenate([mu_total[:, :1], mu_total], axis=1)
        right = np.concatenate([mu_total, mu_total[:, -1:]], axis=1)
    else:  # v faces: (j+1, i)
        left = np.concatenate([mu_total[:1, :], mu_total], axis=0)
        right = np.concatenate([mu_total, mu_total[-1:, :]], axis=0)
    return 0.5 * (left + right)


def relax_weights(dt: float) -> dict[int, tuple[float, float]]:
    """WRF lbc_fcx_gcx nested branch for spec_bdy_width=5, spec_zone=1,
    relax_zone=4: loop=b_dist+1 in 2..4; fcx=0.1/dt*(5-loop)/3; gcx=fcx/5."""
    out: dict[int, tuple[float, float]] = {}
    for b_dist in range(SPEC_ZONE, RELAX_ZONE):
        loop = b_dist + 1
        fcx = 0.1 / dt * (SPEC_ZONE + RELAX_ZONE - loop) / (RELAX_ZONE - 1)
        gcx = 1.0 / dt / 50.0 * (SPEC_ZONE + RELAX_ZONE - loop) / (RELAX_ZONE - 1)
        out[b_dist] = (fcx, gcx)
    return out


def wrf_relax_tendency(field: np.ndarray, target: np.ndarray, dt: float,
                       stagger: str) -> np.ndarray:
    """Pristine relax_bdytend_core port (share/module_bc.F:1221) for a
    coupled field (k, j, i) with a full-ring coupled target of the same
    shape.  Returns the tendency increment, nonzero only in relax rows
    1..4 with WRF corner trims.  Used by the S2 structural oracle and by
    the focused tests."""
    nz, ny, nx = field.shape
    tend = np.zeros_like(field)
    weights = relax_weights(dt)
    for b_dist, (fcx, gcx) in weights.items():
        # X-start / X-end (west/east): tangential trim j in [b+1, JE-b-1]
        je = ny if stagger == "v" else ny - 1
        for side, col in (("W", b_dist), ("E", nx - 1 - b_dist)):
            j0, j1 = b_dist + 1, je - b_dist - 1
            if j0 >= j1:
                continue
            if side == "W":
                f0 = target[:, j0:j1, col] - field[:, j0:j1, col]
                f1 = target[:, j0 - 1 : j1 - 1, col] - field[:, j0 - 1 : j1 - 1, col]
                f2 = target[:, j0 + 1 : j1 + 1, col] - field[:, j0 + 1 : j1 + 1, col]
                f3 = target[:, j0:j1, col - 1] - field[:, j0:j1, col - 1]
                f4 = target[:, j0:j1, col + 1] - field[:, j0:j1, col + 1]
            else:
                f0 = target[:, j0:j1, col] - field[:, j0:j1, col]
                f1 = target[:, j0 - 1 : j1 - 1, col] - field[:, j0 - 1 : j1 - 1, col]
                f2 = target[:, j0 + 1 : j1 + 1, col] - field[:, j0 + 1 : j1 + 1, col]
                f3 = target[:, j0:j1, col + 1] - field[:, j0:j1, col + 1]
                f4 = target[:, j0:j1, col - 1] - field[:, j0:j1, col - 1]
            lap = f1 + f2 + f3 + f4 - 4.0 * f0
            tend[:, j0:j1, col] += fcx * f0 - gcx * lap
        # Y-start / Y-end (south/north): tangential trim i in [b, IE-b]
        ie = nx if stagger == "u" else nx - 1
        for side, row in (("S", b_dist), ("N", je - 1 - b_dist)):
            i0, i1 = b_dist, ie - b_dist
            if i0 >= i1:
                continue
            if side == "S":
                f0 = target[:, row, i0:i1] - field[:, row, i0:i1]
                f1 = target[:, row, i0 - 1 : i1 - 1] - field[:, row, i0 - 1 : i1 - 1]
                f2 = target[:, row, i0 + 1 : i1 + 1] - field[:, row, i0 + 1 : i1 + 1]
                f3 = target[:, row - 1, i0:i1] - field[:, row - 1, i0:i1]
                f4 = target[:, row + 1, i0:i1] - field[:, row + 1, i0:i1]
            else:
                f0 = target[:, row, i0:i1] - field[:, row, i0:i1]
                f1 = target[:, row, i0 - 1 : i1 - 1] - field[:, row, i0 - 1 : i1 - 1]
                f2 = target[:, row, i0 + 1 : i1 + 1] - field[:, row, i0 + 1 : i1 + 1]
                f3 = target[:, row + 1, i0:i1] - field[:, row + 1, i0:i1]
                f4 = target[:, row - 1, i0:i1] - field[:, row - 1, i0:i1]
            lap = f1 + f2 + f3 + f4 - 4.0 * f0
            tend[:, row, i0:i1] += fcx * f0 - gcx * lap
    return tend


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_gpu_savepoint(step: int, rung: str, field: str) -> np.ndarray:
    path = SAVEPOINT_DIR / f"step{step:06d}_{rung}__{field}.npy"
    return np.load(path, allow_pickle=False)


def load_constants() -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(CONTROL_WRFINPUT) as ds:
        mapfac_uy = np.asarray(ds.variables["MAPFAC_UY"][0], dtype=np.float64)
        mapfac_vx = np.asarray(ds.variables["MAPFAC_VX"][0], dtype=np.float64)
        mapfac_u = np.asarray(ds.variables["MAPFAC_U"][0], dtype=np.float64)
        mapfac_v = np.asarray(ds.variables["MAPFAC_V"][0], dtype=np.float64)
        mu_in = np.asarray(ds.variables["MU"][0], dtype=np.float64)
        mub_in = np.asarray(ds.variables["MUB"][0], dtype=np.float64)
        c1h = np.asarray(ds.variables["C1H"][:], dtype=np.float64).reshape(-1)
        c2h = np.asarray(ds.variables["C2H"][:], dtype=np.float64).reshape(-1)
    with Dataset(GPU_INIT_FRAME) as ds:
        gpu_mu = np.asarray(ds.variables["MU"][0], dtype=np.float64)
        gpu_mub = np.asarray(ds.variables["MUB"][0], dtype=np.float64)
        gpu_mapfac_uy = np.asarray(ds.variables["MAPFAC_UY"][0], dtype=np.float64)
        gpu_mapfac_vx = np.asarray(ds.variables["MAPFAC_VX"][0], dtype=np.float64)
        gpu_c1h = np.asarray(ds.variables["C1H"][:], dtype=np.float64).reshape(-1)
        gpu_c2h = np.asarray(ds.variables["C2H"][:], dtype=np.float64).reshape(-1)
    return {
        "mapfac_uy": mapfac_uy, "mapfac_vx": mapfac_vx,
        "mapfac_u": mapfac_u, "mapfac_v": mapfac_v,
        "mu_input_total": mu_in + mub_in,
        "c1h": c1h, "c2h": c2h,
        "gpu_mu_total": gpu_mu + gpu_mub,
        "gpu_mapfac_uy": gpu_mapfac_uy, "gpu_mapfac_vx": gpu_mapfac_vx,
        "gpu_c1h": gpu_c1h, "gpu_c2h": gpu_c2h,
    }


def combined_rmse(diff: dict[str, np.ndarray], mask: dict[str, np.ndarray] | None = None) -> float:
    sse = 0.0
    count = 0
    for comp, array in diff.items():
        sel = array if mask is None else array[mask[comp]]
        sse += float(np.sum(np.square(sel, dtype=np.float64), dtype=np.float64))
        count += int(sel.size)
    return math.sqrt(sse / count)


def band_masks(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    dist = cpu.distance_to_edge(shape)
    return {
        "spec_row_0": dist == 0,
        "relax_rows_1_4": (dist >= 1) & (dist <= 4),
        "interior_ge_5": dist >= 5,
        "all": np.ones(shape, dtype=bool),
    }


def per_band_rmse(diff: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    bands = {c: band_masks(a.shape) for c, a in diff.items()}
    for name in ("spec_row_0", "relax_rows_1_4", "interior_ge_5", "all"):
        out[name] = combined_rmse(diff, {c: b[name] for c, b in bands.items()})
    return out


def rms(x: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(x, dtype=np.float64)))))


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate() -> dict[str, Any]:
    auth: dict[str, Any] = {}

    cpu_proof, cpu_proof_row = read_self_hashed(CPU_CONT_SPRINT / "proof.json")
    cpu_analysis, cpu_analysis_row = read_self_hashed(CPU_CONT_SPRINT / "cpu-ladder-analysis.json")
    gpu_proof, gpu_proof_row = read_self_hashed(GPU_ARM_SPRINT / "proof.json")
    gpu_analysis, gpu_analysis_row = read_self_hashed(GPU_ARM_SPRINT / "gpu-cpu-ladder-analysis.json")
    runtime, runtime_row = read_self_hashed(RUNTIME_PROOF)
    expected_self = {
        "terminal_cpu_proof": (cpu_proof_row, EXPECTED["terminal_cpu_proof_self"]),
        "cpu_ladder_analysis": (cpu_analysis_row, EXPECTED["cpu_ladder_analysis_self"]),
        "gpu_arm_proof": (gpu_proof_row, EXPECTED["gpu_arm_proof_self"]),
        "gpu_arm_analysis": (gpu_analysis_row, EXPECTED["gpu_arm_analysis_self"]),
        "runtime_proof": (runtime_row, EXPECTED["runtime_proof_self"]),
    }
    for name, (row, want) in expected_self.items():
        if row["canonical_self_hash"] != want:
            raise RuntimeError(f"{name} self-hash {row['canonical_self_hash']} != frozen {want}")
    auth["frozen_proofs"] = {name: row for name, (row, _w) in expected_self.items()}

    # Revalidate every savepoint in the runtime manifest (all 180).
    manifest = runtime["savepoint_manifest"]
    rows = manifest["rows"]
    if canonical_digest(rows) != EXPECTED["savepoint_rows"]:
        raise RuntimeError("savepoint rows hash drifted")
    checked = 0
    for row in rows:
        path = Path(row["path"])
        if path.parent != SAVEPOINT_DIR.resolve():
            raise RuntimeError(f"savepoint escaped namespace: {path}")
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"savepoint hash mismatch: {path}")
        checked += 1
    auth["savepoints_revalidated"] = checked

    # WRF pristine source identities (operator map authority).
    wrf_rows = {}
    for rel, want in WRF_SOURCE_SHA256.items():
        got = sha256_file(WRF_SRC / rel)
        if got != want:
            raise RuntimeError(f"WRF source drifted: {rel}")
        wrf_rows[rel] = got
    auth["wrf_source_sha256"] = wrf_rows

    # GPU source identities at this sprint's commit (recorded, not frozen).
    auth["gpu_source_sha256"] = {
        rel: sha256_file(REPO / rel) for rel in GPU_SOURCE_FILES
    }

    # Oracle input files.
    auth["oracle_inputs"] = {
        "control_wrfinput": {"path": str(CONTROL_WRFINPUT), "sha256": sha256_file(CONTROL_WRFINPUT)},
        "gpu_init_frame": {"path": str(GPU_INIT_FRAME), "sha256": sha256_file(GPU_INIT_FRAME)},
        "truth_frame": {"path": str(TRUTH_FRAME), "sha256": sha256_file(TRUTH_FRAME)},
        "wrf_binary": {
            "path": str(RUNS.parent / "wrf_iso_ladder/install_ladder/bin/wrf"),
            "sha256": sha256_file(RUNS.parent / "wrf_iso_ladder/install_ladder/bin/wrf"),
        },
    }
    if auth["oracle_inputs"]["wrf_binary"]["sha256"] != (
        "50f71fd5245affcbf16f14f0cd7026561bbf1a412e28b5e60c60707af0edd238"
    ):
        raise RuntimeError("instrumented WRF binary drifted")

    # Member run dirs exist with full step-1 inventory (23 files/rank, 12 ranks).
    member_inventory = {}
    for member in MEMBERS:
        root = RUNS / member / "momsp_dumps"
        count = 0
        for rank_dir in sorted(root.glob("rank*")):
            count += len(list(rank_dir.glob("step000001_*.f64")))
        member_inventory[member] = count
    if any(v != 23 * 12 for v in member_inventory.values()):
        raise RuntimeError(f"member dump inventory incomplete: {member_inventory}")
    auth["member_step1_dump_files"] = member_inventory

    auth["jax_imported"] = "jax" in sys.modules
    auth["gpuwrf_imported"] = any(
        name == "gpuwrf" or name.startswith("gpuwrf.") for name in sys.modules
    )
    auth["gpu_commands_queries_locks"] = 0
    if auth["jax_imported"] or auth["gpuwrf_imported"]:
        raise RuntimeError("CPU-only import discipline violated")
    return auth, cpu_analysis, gpu_analysis, runtime


# ---------------------------------------------------------------------------
# Sub-span analyses
# ---------------------------------------------------------------------------

def s0_assembly_provenance(
    gpu: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    members: dict[str, dict[str, np.ndarray]],
    const: dict[str, np.ndarray],
    hgt: np.ndarray,
) -> dict[str, Any]:
    """S0: SP2 -> SP3 assembly fidelity and provenance attribution."""
    c1h, c2h = const["c1h"], const["c2h"]
    m_h_ctl = mass_weight(c1h, c2h, control["mut"])

    # A-ORACLE-WRF: reproduce control SP3 from control SP2 + mut.
    oracle_u = face_average_u(m_h_ctl * control["rublten"])
    oracle_v = face_average_v(m_h_ctl * control["rvblten"])
    oracle = {"u": oracle_u, "v": oracle_v}
    oracle_err = cpu.subtract(oracle, {"u": control["ru_tendf"], "v": control["rv_tendf"]})
    oracle_metrics = cpu.metrics_with_bands(oracle_err, hgt)
    ref_rms = combined_rmse({"u": control["ru_tendf"], "v": control["rv_tendf"]})
    oracle_rel = oracle_metrics["rmse"] / ref_rms

    # A-FIDELITY-GPU: reproduce GPU SP3 from GPU SP2 + GPU mass.
    gpu_c1h, gpu_c2h = const["gpu_c1h"], const["gpu_c2h"]
    m_h_gpu = mass_weight(gpu_c1h, gpu_c2h, const["gpu_mu_total"])
    asm_gpu = {
        "u": face_average_u(m_h_gpu * gpu["rublten"]),
        "v": face_average_v(m_h_gpu * gpu["rvblten"]),
    }
    r_asm = cpu.subtract(gpu_state := {"u": gpu["ru_tendf"], "v": gpu["rv_tendf"]}, asm_gpu)
    r_asm_metrics = cpu.metrics_with_bands(r_asm, hgt)

    # Same reconstruction with the CONTROL mass field (mass-convention sensitivity).
    asm_gpu_ctl_mass = {
        "u": face_average_u(m_h_ctl * gpu["rublten"]),
        "v": face_average_v(m_h_ctl * gpu["rvblten"]),
    }
    r_asm_ctl_mass = cpu.subtract(gpu_state, asm_gpu_ctl_mass)
    r_asm_ctl_mass_metrics = cpu.metrics_with_bands(r_asm_ctl_mass, hgt)

    # A-INHERIT: the inherited channel faceavg(m_ctl * (rub_gpu - rub_ctl)).
    inherit = {
        "u": face_average_u(m_h_ctl * (gpu["rublten"] - control["rublten"])),
        "v": face_average_v(m_h_ctl * (gpu["rvblten"] - control["rvblten"])),
    }
    inherit_metrics = cpu.metrics_with_bands(inherit, hgt)

    # Member-derived envelope for the inherited channel.
    member_rows: dict[str, Any] = {}
    for member, vals in members.items():
        m_h_m = mass_weight(c1h, c2h, vals["mut"])
        diff = {
            "u": face_average_u(m_h_m * (vals["rublten"] - control["rublten"])),
            "v": face_average_v(m_h_m * (vals["rvblten"] - control["rvblten"])),
        }
        member_rows[member] = cpu.metrics_with_bands(diff, hgt)
    envelope = cpu.add_envelope(member_rows)

    # Full SP3 deviation and decomposition consistency.
    delta_sp3 = cpu.subtract(gpu_state, {"u": control["ru_tendf"], "v": control["rv_tendf"]})
    delta_metrics = cpu.metrics_with_bands(delta_sp3, hgt)
    decomposition = cpu.subtract(
        cpu.subtract(delta_sp3, inherit), r_asm,
    )
    mass_term_metrics = cpu.metrics_with_bands(decomposition, hgt)

    env_max = envelope["wrf_envelope_max_rmse"]
    inherit_rmse = inherit_metrics["rmse"]
    r_asm_rmse = r_asm_metrics["rmse"]
    verdict_bits = {
        "oracle_relative_rmse": oracle_rel,
        "oracle_authenticated": bool(oracle_rel < 1e-4),
        "inherit_rmse": inherit_rmse,
        "inherit_derived_member_envelope": envelope,
        "inherit_outside_literal_envelope": bool(
            inherit_rmse < envelope["wrf_envelope_min_rmse"]
            or inherit_rmse > env_max
        ),
        "inherit_systematic": bool(
            (inherit_rmse < envelope["wrf_envelope_min_rmse"] or inherit_rmse > env_max)
            and inherit_rmse > SYSTEMATIC_FACTOR * env_max
        ),
        "r_asm_rmse": r_asm_rmse,
        "r_asm_rmse_control_mass": r_asm_ctl_mass_metrics["rmse"],
        "r_asm_within_derived_envelope": bool(r_asm_rmse <= SYSTEMATIC_FACTOR * env_max),
        "r_asm_to_inherit_ratio": (r_asm_rmse / inherit_rmse) if inherit_rmse else None,
        "mass_convention_term_rmse": mass_term_metrics["rmse"],
        "delta_sp3_rmse": delta_metrics["rmse"],
    }
    return {
        "a_oracle_wrf": {
            "relative_rmse": oracle_rel,
            "metrics": oracle_metrics,
            "reference_rmse": ref_rms,
            "authenticated": verdict_bits["oracle_authenticated"],
        },
        "a_fidelity_gpu": {
            "r_asm": r_asm_metrics,
            "r_asm_control_mass": r_asm_ctl_mass_metrics,
            "gpu_mass_convention": {
                "gpu_mu_total_vs_control_mut_max_abs": float(
                    np.abs(const["gpu_mu_total"] - control["mut"]).max()
                ),
                "gpu_mu_total_vs_control_mut_rmse": rms(const["gpu_mu_total"] - control["mut"]),
                "gpu_c1h_identical": bool(np.array_equal(gpu_c1h, c1h)),
                "gpu_c2h_identical": bool(np.array_equal(gpu_c2h, c2h)),
                "control_mut_vs_wrfinput_total_rmse": rms(control["mut"] - const["mu_input_total"]),
            },
        },
        "a_inherit": {
            "metrics": inherit_metrics,
            "derived_member_envelope": envelope,
            "member_rows": member_rows,
        },
        "delta_sp3": delta_metrics,
        "decomposition_check": {
            "delta_minus_inherit_minus_rasm": mass_term_metrics,
        },
        "verdict_bits": verdict_bits,
    }


def s1_rk_tendency_residual(
    gpu: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    members: dict[str, dict[str, np.ndarray]],
    const: dict[str, np.ndarray],
    hgt: np.ndarray,
) -> dict[str, Any]:
    """S1: implied rk_tendency residual after removing the authenticated
    rk_addtend fold and the relax bundle from L1 (non-spec cells)."""
    msfu = const["mapfac_uy"][None, :, :]
    msfv = const["mapfac_vx"][None, :, :]
    gpu_msfu = const["gpu_mapfac_uy"][None, :, :]
    gpu_msfv = const["gpu_mapfac_vx"][None, :, :]

    def dyn_wrf(l1u, l1v, sp3u, sp3v, usu, vsv):
        return {
            "u": l1u - sp3u / msfu - usu,
            "v": l1v - sp3v * (1.0 / msfv) - vsv,
        }

    def dyn_gpu(l1u, l1v, sp3u, sp3v, ru, rv):
        return {
            "u": l1u - sp3u / gpu_msfu - ru,
            "v": l1v - sp3v * (1.0 / gpu_msfv) - rv,
        }

    dyn_ctl = dyn_wrf(control["ru_tend"], control["rv_tend"],
                      control["ru_tendf"], control["rv_tendf"],
                      control["u_save"], control["v_save"])
    dyn_g = dyn_gpu(gpu["ru_tend"], gpu["rv_tend"],
                    gpu["ru_tendf"], gpu["rv_tendf"],
                    gpu["u_save"], gpu["v_save"])
    delta_dyn = cpu.subtract(dyn_g, dyn_ctl)
    delta_metrics = cpu.metrics_with_bands(delta_dyn, hgt)

    member_rows: dict[str, Any] = {}
    for member, vals in members.items():
        dyn_m = dyn_wrf(vals["ru_tend"], vals["rv_tend"],
                        vals["ru_tendf"], vals["rv_tendf"],
                        vals["u_save"], vals["v_save"])
        member_rows[member] = cpu.metrics_with_bands(cpu.subtract(dyn_m, dyn_ctl), hgt)
    envelope = cpu.add_envelope(member_rows)

    # Non-spec restriction (the residual identity is exact only off the
    # WRF spec row, where control L1 is overwritten by spec_bdy_dry).
    nonspec_masks = {}
    for comp, arr in delta_dyn.items():
        dist = cpu.distance_to_edge(arr.shape)
        nonspec_masks[comp] = dist >= 1
    nonspec_rmse = combined_rmse(delta_dyn, nonspec_masks)
    member_nonspec: dict[str, float] = {}
    for member, vals in members.items():
        dyn_m = dyn_wrf(vals["ru_tend"], vals["rv_tend"],
                        vals["ru_tendf"], vals["rv_tendf"],
                        vals["u_save"], vals["v_save"])
        member_nonspec[member] = combined_rmse(cpu.subtract(dyn_m, dyn_ctl), nonspec_masks)
    env_ns_max = max(member_nonspec.values())
    env_ns_min = min(member_nonspec.values())

    return {
        "delta_dyn": delta_metrics,
        "derived_member_envelope": envelope,
        "nonspec": {
            "gpu_rmse": nonspec_rmse,
            "member_rmse": member_nonspec,
            "derived_envelope_min": env_ns_min,
            "derived_envelope_max": env_ns_max,
            "outside_literal_envelope": bool(nonspec_rmse < env_ns_min or nonspec_rmse > env_ns_max),
            "systematic": bool(
                (nonspec_rmse < env_ns_min or nonspec_rmse > env_ns_max)
                and nonspec_rmse > SYSTEMATIC_FACTOR * env_ns_max
            ),
        },
        "mapfac_choice": {
            "u": "MAPFAC_UY (WRF grid%msfuy; GPU metrics.msfuy)",
            "v": "1/MAPFAC_VX (WRF grid%msfvx_inv; GPU 1/metrics.msfvx)",
            "gpu_mapfac_uy_vs_control_max_abs": float(
                np.abs(const["gpu_mapfac_uy"] - const["mapfac_uy"]).max()
            ),
            "gpu_mapfac_vx_vs_control_max_abs": float(
                np.abs(const["gpu_mapfac_vx"] - const["mapfac_vx"]).max()
            ),
        },
    }


def s2_relax_split(
    gpu: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    members: dict[str, dict[str, np.ndarray]],
    const: dict[str, np.ndarray],
    hgt: np.ndarray,
    dtbc: float = DT_D03,
) -> dict[str, Any]:
    """S2: relax_bdy_dry defect attribution.

    Exact decomposition per relax row:  delta_relax = fcx*(dT - dF)
    - gcx*(lap(dT) - lap(dF))  =>  E = delta_relax + fcx*dF - gcx*lap(dF)
    isolates the target-side contribution fcx*dT - gcx*lap(dT), where
    dF is the coupled-SP1 field difference (exactly computable) and dT is
    the boundary-target difference (not retained; E is its exact image).
    """
    # Field coupling (WRF couple_momentum, authenticated):
    c1h, c2h = const["c1h"], const["c2h"]
    muu_ctl = mu_faces(control["mut"], axis=1)
    muv_ctl = mu_faces(control["mut"], axis=0)
    muu_gpu = mu_faces(const["gpu_mu_total"], axis=1)
    muv_gpu = mu_faces(const["gpu_mu_total"], axis=0)
    msfu = const["mapfac_uy"]
    msfv = const["mapfac_vx"]
    gpu_msfu = const["gpu_mapfac_uy"]
    gpu_msfv = const["gpu_mapfac_vx"]

    ru_ctl = (mass_weight(c1h, c2h, muu_ctl)) * control["u_sp1"] / msfu[None, :, :]
    rv_ctl = (mass_weight(c1h, c2h, muv_ctl)) * control["v_sp1"] / msfv[None, :, :]
    ru_gpu = (mass_weight(const["gpu_c1h"], const["gpu_c2h"], muu_gpu)) * gpu["u_sp1"] / gpu_msfu[None, :, :]
    rv_gpu = (mass_weight(const["gpu_c1h"], const["gpu_c2h"], muv_gpu)) * gpu["v_sp1"] / gpu_msfv[None, :, :]

    d_ru = ru_gpu - ru_ctl
    d_rv = rv_gpu - rv_ctl

    delta_relax = {"u": gpu["u_save"] - control["u_save"],
                   "v": gpu["v_save"] - control["v_save"]}
    delta_metrics = cpu.metrics_with_bands(delta_relax, hgt)

    # Field-inherited image: relax(ru_ctl + dF) - relax(ru_ctl) with the
    # same target -> fcx*(-dF)... careful: relax = fcx*(T - F) - gcx*lap(T - F),
    # so d(relax) = fcx*(dT - dF) - gcx*(lap(dT) - lap(dF)).  The FIELD-side
    # part is -fcx*dF + gcx*lap(dF) where lap acts on the residual field
    # (target held fixed).  Compute it exactly with the stencil port by
    # evaluating relax(dF as field, zero target).
    zero_t_u = np.zeros_like(ru_ctl)
    zero_t_v = np.zeros_like(rv_ctl)
    field_image_u = wrf_relax_tendency(d_ru, zero_t_u, DT_D03, "u")
    field_image_v = wrf_relax_tendency(d_rv, zero_t_v, DT_D03, "v")
    # relax(dF, 0) = fcx*(0 - dF) - gcx*lap(0 - dF) = -(fcx*dF - gcx*lap(dF))
    # so the field-side part of delta_relax equals +relax(dF, 0) with sign
    # convention: delta_relax = relax(T_g, F_g) - relax(T_c, F_c)
    #   = [fcx*(dT - dF) - gcx*lap(dT - dF)]
    #   = relax(dT, dF)... the field-side is exactly relax(0->field shift):
    field_side = {"u": -field_image_u, "v": -field_image_v}
    # E = delta_relax - field_side = target-side image fcx*dT - gcx*lap(dT):
    e_side = cpu.subtract(delta_relax, field_side)
    e_metrics = cpu.metrics_with_bands(e_side, hgt)
    field_metrics = cpu.metrics_with_bands(field_side, hgt)

    # Row/side structure of the raw defect.
    structure: dict[str, Any] = {}
    for comp, arr in delta_relax.items():
        dist = cpu.distance_to_edge(arr.shape)
        rows = {}
        for b in range(1, 5):
            mask = dist == b
            rows[f"b{b}"] = {
                "rmse": rms(arr[mask]),
                "max_abs": float(np.abs(arr[mask]).max()),
                "mean_bias": float(arr[mask].mean()),
                "control_rmse": rms(
                    (control["u_save"] if comp == "u" else control["v_save"])[mask]
                ),
                "gpu_rmse": rms(
                    (gpu["u_save"] if comp == "u" else gpu["v_save"])[mask]
                ),
            }
        structure[comp] = rows

    # k-profile of the defect (relax band only).
    k_profile: dict[str, Any] = {}
    for comp, arr in delta_relax.items():
        dist = cpu.distance_to_edge(arr.shape)
        band = (dist >= 1) & (dist <= 4)
        per_k = []
        for k in range(arr.shape[0]):
            sel = arr[k][band[k]]
            per_k.append(rms(sel))
        k_profile[comp] = per_k

    # Member envelope (exact zero, frozen) + degeneracy context:
    # do member perturbations reach the boundary band at SP1?
    sp1_band: dict[str, Any] = {}
    for member, vals in members.items():
        du = vals["u_sp1"] - control["u_sp1"]
        dv = vals["v_sp1"] - control["v_sp1"]
        row: dict[str, float] = {}
        for comp, arr in (("u", du), ("v", dv)):
            dist = cpu.distance_to_edge(arr.shape)
            for bname, mask in (
                ("spec_row_0", dist == 0),
                ("relax_rows_1_4", (dist >= 1) & (dist <= 4)),
                ("interior_ge_5", dist >= 5),
            ):
                key = f"{comp}_{bname}"
                row[key] = float(np.abs(arr[mask]).max())
        sp1_band[member] = row

    member_relax_rows: dict[str, Any] = {}
    for member, vals in members.items():
        diff = {"u": vals["u_save"] - control["u_save"],
                "v": vals["v_save"] - control["v_save"]}
        member_relax_rows[member] = cpu.metrics_with_bands(diff, hgt)
    relax_envelope = cpu.add_envelope(member_relax_rows)

    # Boundary-chain consistency (strip-1 anchors, control side):
    # target = u_bxs + dtbc*u_btxs from spec_bdy_final inversion of SP4:
    # u_2(sp4,spec) = msf * target / (c1h*muus + c2h)  (mucouple/msfcouple on).
    muus_ctl = muu_ctl  # step-1 saved muu == face average of step-start mut
    target_u = control["u_sp4"] * mass_weight(c1h, c2h, muus_ctl) / msfu[None, :, :]
    target_v = control["v_sp4"] * mass_weight(c1h, c2h, muv_ctl) / msfv[None, :, :]
    # u_btxs == control L1 spec row exactly (spec_bdytend overwrite).
    u_btxs = control["ru_tend"]
    v_btxs = control["rv_tend"]
    # implied u_bxs (coupled, t=0) = target - dtbc*u_btxs on the spec ring.
    implied_bxs_u = target_u - dtbc * u_btxs
    implied_bxs_v = target_v - dtbc * v_btxs
    # independent anchor: coupled SP1 spec row.
    anchor_u = ru_ctl
    anchor_v = rv_ctl
    spec_masks = {
        "u": cpu.distance_to_edge(anchor_u.shape) == 0,
        "v": cpu.distance_to_edge(anchor_v.shape) == 0,
    }
    chain_check = {
        "u_implied_bxs_minus_coupled_sp1_spec_rmse": rms(
            (implied_bxs_u - anchor_u)[spec_masks["u"]]
        ),
        "v_implied_bxs_minus_coupled_sp1_spec_rmse": rms(
            (implied_bxs_v - anchor_v)[spec_masks["v"]]
        ),
        "u_coupled_spec_scale_rmse": rms(anchor_u[spec_masks["u"]]),
        "v_coupled_spec_scale_rmse": rms(anchor_v[spec_masks["v"]]),
    }

    return {
        "delta_relax": delta_metrics,
        "field_side_image": field_metrics,
        "target_side_image_E": e_metrics,
        "structure_by_row": structure,
        "k_profile_relax_band": k_profile,
        "frozen_member_envelope": relax_envelope,
        "member_rows": member_relax_rows,
        "sp1_member_max_abs_by_band": sp1_band,
        "boundary_chain_strip1_check": chain_check,
        "field_coupling_inputs": {
            "control_mut_rmse": rms(control["mut"]),
            "gpu_mu_total_rmse": rms(const["gpu_mu_total"]),
            "d_ru_rmse": rms(d_ru),
            "d_rv_rmse": rms(d_rv),
            "d_ru_max_abs": float(np.abs(d_ru).max()),
            "d_rv_max_abs": float(np.abs(d_rv).max()),
        },
    }


def s3_spec_row(
    gpu: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    members: dict[str, dict[str, np.ndarray]],
    hgt: np.ndarray,
) -> dict[str, Any]:
    """S3: spec_bdy_dry mapping verification and asymmetry quantification."""
    spec_masks = {
        "u": cpu.distance_to_edge(control["ru_tend"].shape) == 0,
        "v": cpu.distance_to_edge(control["rv_tend"].shape) == 0,
    }
    bitwise: dict[str, Any] = {}
    identical_all = True
    for member, vals in members.items():
        du = vals["ru_tend"] - control["ru_tend"]
        dv = vals["rv_tend"] - control["rv_tend"]
        mu_max = float(np.abs(du[spec_masks["u"]]).max())
        mv_max = float(np.abs(dv[spec_masks["v"]]).max())
        identical_all = identical_all and (mu_max == 0.0) and (mv_max == 0.0)
        bitwise[member] = {
            "u_spec_max_abs": mu_max,
            "v_spec_max_abs": mv_max,
            "u_spec_bitwise_identical": bool(mu_max == 0.0),
            "v_spec_bitwise_identical": bool(mv_max == 0.0),
        }

    gpu_spec = {"u": gpu["ru_tend"][spec_masks["u"]], "v": gpu["rv_tend"][spec_masks["v"]]}
    ctl_spec = {"u": control["ru_tend"][spec_masks["u"]],
                "v": control["rv_tend"][spec_masks["v"]]}
    gpu_spec_rms = combined_rmse(gpu_spec)
    ctl_spec_rms = combined_rmse(ctl_spec)
    diff_spec = {"u": gpu_spec["u"] - ctl_spec["u"], "v": gpu_spec["v"] - ctl_spec["v"]}
    diff_spec_rms = combined_rmse(diff_spec)

    # Interior scale of the GPU L1 content for context.
    interior_masks = {
        "u": cpu.distance_to_edge(gpu["ru_tend"].shape) >= 5,
        "v": cpu.distance_to_edge(gpu["rv_tend"].shape) >= 5,
    }
    gpu_l1_interior = {"u": gpu["ru_tend"][interior_masks["u"]],
                       "v": gpu["rv_tend"][interior_masks["v"]]}
    ctl_l1_interior = {"u": control["ru_tend"][interior_masks["u"]],
                       "v": control["rv_tend"][interior_masks["v"]]}
    return {
        "member_spec_rows_bitwise_identical_to_control": bool(identical_all),
        "member_rows": bitwise,
        "wrf_spec_overwrite_content_rmse": ctl_spec_rms,
        "gpu_l1_spec_content_rmse": gpu_spec_rms,
        "spec_difference_rmse": diff_spec_rms,
        "diff_to_overwrite_ratio": diff_spec_rms / ctl_spec_rms if ctl_spec_rms else None,
        "gpu_spec_to_overwrite_ratio": gpu_spec_rms / ctl_spec_rms if ctl_spec_rms else None,
        "gpu_l1_interior_rmse": combined_rmse(gpu_l1_interior),
        "control_l1_interior_rmse": combined_rmse(ctl_l1_interior),
        "frozen_span_spec_sse_share": 0.9067200134758682,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_states(step: int = 1) -> tuple[dict, dict, dict, dict]:
    """Return (gpu, control, members, constants) step state bundles."""
    gpu = {
        "u_sp1": load_gpu_savepoint(step, "sp1_entry", "u"),
        "v_sp1": load_gpu_savepoint(step, "sp1_entry", "v"),
        "rublten": load_gpu_savepoint(step, "sp2_pbl", "rublten"),
        "rvblten": load_gpu_savepoint(step, "sp2_pbl", "rvblten"),
        "ru_tendf": load_gpu_savepoint(step, "sp3_tendf", "ru_tendf"),
        "rv_tendf": load_gpu_savepoint(step, "sp3_tendf", "rv_tendf"),
        "ru_tend": load_gpu_savepoint(step, "l1_rk1_tend", "ru_tend"),
        "rv_tend": load_gpu_savepoint(step, "l1_rk1_tend", "rv_tend"),
        "u_save": load_gpu_savepoint(step, "l1_rk1_relax", "u_save"),
        "v_save": load_gpu_savepoint(step, "l1_rk1_relax", "v_save"),
        "u_sp4": load_gpu_savepoint(step, "sp4_exit", "u"),
        "v_sp4": load_gpu_savepoint(step, "sp4_exit", "v"),
    }

    def load_wrf(ranks) -> dict[str, np.ndarray]:
        return {
            "u_sp1": reassemble.reassemble3d("sp1_entry__u", step, ranks),
            "v_sp1": reassemble.reassemble3d("sp1_entry__v", step, ranks),
            "rublten": reassemble.reassemble3d("sp2_pbl__rublten", step, ranks),
            "rvblten": reassemble.reassemble3d("sp2_pbl__rvblten", step, ranks),
            "mut": reassemble.reassemble2d("sp2_pbl__mut", step, ranks),
            "ru_tendf": reassemble.reassemble3d("sp3_tendf__ru_tendf", step, ranks),
            "rv_tendf": reassemble.reassemble3d("sp3_tendf__rv_tendf", step, ranks),
            "ru_tend": reassemble.reassemble3d("l1_rk1_tend__ru_tend", step, ranks),
            "rv_tend": reassemble.reassemble3d("l1_rk1_tend__rv_tend", step, ranks),
            "u_save": reassemble.reassemble3d("l1_rk1_tend__u_save", step, ranks),
            "v_save": reassemble.reassemble3d("l1_rk1_tend__v_save", step, ranks),
            "u_sp4": reassemble.reassemble3d("sp4_exit__u", step, ranks),
            "v_sp4": reassemble.reassemble3d("sp4_exit__v", step, ranks),
        }

    control = load_wrf(reassemble.load_ranks(RUNS / "control/momsp_dumps"))
    members = {
        member: load_wrf(reassemble.load_ranks(RUNS / member / "momsp_dumps"))
        for member in MEMBERS
    }
    const = load_constants()
    return gpu, control, members, const


def main() -> None:
    auth, cpu_analysis, gpu_analysis, runtime = authenticate()
    hgt = cpu.load_hgt()
    gpu, control, members, const = load_all_states(step=1)

    s0 = s0_assembly_provenance(gpu, control, members, const, hgt)
    s1 = s1_rk_tendency_residual(gpu, control, members, const, hgt)
    s2 = s2_relax_split(gpu, control, members, const, hgt, dtbc=DT_D03)
    s3 = s3_spec_row(gpu, control, members, hgt)

    falsifiers = {
        "F1_wrf_assembly_oracle_authenticated": s0["a_oracle_wrf"]["authenticated"],
        "F2_gpu_assembly_residual_within_member_envelope": s0["verdict_bits"]["r_asm_within_derived_envelope"],
        "F3_rk_tendency_residual_within_derived_envelope": not s1["nonspec"]["systematic"],
        "F4_member_spec_rows_bitwise_identical": s3["member_spec_rows_bitwise_identical_to_control"],
        "F5_relax_forward_oracle_closed": False,
        "F5_note": (
            "the full forward relax oracle cannot close offline: boundary "
            "strips 2-5 (b_dist 1..4) are not retained; only strip 1 is "
            "recoverable (control L1 spec row = u_bt* and the spec_bdy_final "
            "inversion of SP4).  The contracted fallback is used: exact "
            "structural E-decomposition (target-side vs field-side) plus the "
            "strip-1 boundary-chain consistency check."
        ),
    }

    earliest = {
        "name": "s1_rk_tendency_dynamics_residual",
        "execution_order": [
            "s0_sp2_to_sp3_assembly_provenance",
            "s1_rk_tendency_dynamics_residual",
            "s2_relax_bdy_dry",
            "s4_rk_addtend_dry_fold_accounted",
            "s3_spec_bdy_dry",
        ],
        "why_earliest": (
            "S0 resolves the incoming SP3 fact as fully inherited from the SP2 "
            "MYNN input (assembly exonerated bit-exactly on the GPU side and at "
            "5.17e-08 relative on the WRF side), so it is not a new independent "
            "dry-dycore defect.  The first NEW independent systematic defect in "
            "execution order is the implied rk_tendency dynamics residual "
            "(F3=false): nonspec rmse 21.8790 versus derived member envelope max "
            "0.0026994 (8104x), concentrated in the relax band (55.67) and the "
            "spec row (338.77, representation asymmetry quantified in S3), with "
            "a small but also systematic interior floor (1.168)."
        ),
        "sp3_provenance": {
            "resolution": "fully inherited from SP2; assembly not an independent defect",
            "gpu_assembly_residual_rmse": s0["verdict_bits"]["r_asm_rmse"],
            "wrf_oracle_relative_rmse": s0["verdict_bits"]["oracle_relative_rmse"],
            "inherited_rmse": s0["verdict_bits"]["inherit_rmse"],
            "decomposition_residual_rmse": s0["verdict_bits"]["mass_convention_term_rmse"],
        },
        "s1_residual": {
            "nonspec_rmse": s1["nonspec"]["gpu_rmse"],
            "derived_member_envelope_max": s1["nonspec"]["derived_envelope_max"],
            "ratio_to_envelope_max": s1["nonspec"]["gpu_rmse"] / s1["nonspec"]["derived_envelope_max"],
            "band_rmse": {
                "spec_row_0": s1["delta_dyn"]["bands"]["spec_row_0"]["rmse"],
                "relax_rows_1_4": s1["delta_dyn"]["bands"]["relax_rows_1_4"]["rmse"],
                "interior_ge_5": s1["delta_dyn"]["bands"]["interior_ge_5"]["rmse"],
            },
            "argmax": s1["delta_dyn"]["argmax"],
        },
        "s2_relax": {
            "resolution": (
                "systematic versus the exact-zero frozen envelope but ~100% "
                "target-side (E rmse 0.34412 of total 0.34412; field-side "
                "0.000319); relax support/trims exact (b_dist=4 exactly zero on "
                "both sides); envelope degeneracy explained by exactly-zero "
                "member perturbations in the boundary band at SP1"
            ),
            "delta_rmse": s2["delta_relax"]["rmse"],
            "target_side_E_rmse": s2["target_side_image_E"]["rmse"],
            "field_side_rmse": s2["field_side_image"]["rmse"],
        },
        "s3_spec": {
            "resolution": (
                "documented representation asymmetry, not an independent "
                "tendency defect: WRF spec-row overwrite content rmse 138.33 "
                "(pure u_bt*, members bitwise identical), GPU spec-row dynamics "
                "content rmse 322.65; the frozen span's 90.67% spec-row SSE is "
                "this asymmetry"
            ),
            "wrf_overwrite_content_rmse": s3["wrf_spec_overwrite_content_rmse"],
            "gpu_spec_content_rmse": s3["gpu_l1_spec_content_rmse"],
        },
        "exact_next_action": (
            "create a new preregistered CPU-only step-1 contract that tests the "
            "sixth-order-diffusion lane hypothesis against this retained S1 "
            "residual: GPU sixth_order_diffusion_tendency "
            "(src/gpuwrf/dynamics/explicit_diffusion.py) uses periodic rolls "
            "with no nested 3-cell trim and no msf/adjacent-mass coupling, while "
            "pristine WRF (dyn_em/module_big_step_utilities_em.F "
            "sixth_order_diffusion, consumed into ru_tendf inside rk_tendency at "
            "module_em.F:880-900) is exactly zero in rings 0-2 and uses "
            "adjacent-mass/msf coupling; companion question: provenance of the "
            "S2 relax target-side difference (boundary package/parent chain).  "
            "This sprint authorizes neither that contract's execution, a GPU "
            "arm, nor any correction."
        ),
    }

    payload = {
        "schema": "gpuwrf.v0234.dycore-internal-split-kimi.analysis.v1",
        "step": 1,
        "authority": auth,
        "frozen_envelope_citations": {
            "sp3_tendf": cpu_analysis["steps"]["step1"]["rungs"]["sp3_tendf"]["envelope"],
            "l1_rk1_tend": cpu_analysis["steps"]["step1"]["rungs"]["l1_rk1_tend"]["envelope"],
            "sp3_to_l1_span": cpu_analysis["steps"]["step1"]["spans"]["sp3_to_l1_tendency_build"]["envelope"],
            "l1_rk1_relax_auxiliary": cpu_analysis["steps"]["step1"]["auxiliary_rungs"]["l1_rk1_relax"]["envelope"],
            "frozen_gpu_arm_facts": {
                "incoming_sp3": gpu_analysis["rungs_evaluated_in_order"]["sp3_tendf"]["classification"],
                "sp3_to_l1_span": gpu_analysis["spans_evaluated_in_order"]["sp3_to_l1_tendency_build"]["classification"],
                "l1_rk1_relax_auxiliary": gpu_analysis["auxiliary_rungs"]["l1_rk1_relax"]["classification"],
            },
        },
        "s0_sp2_to_sp3_assembly_provenance": s0,
        "s1_rk_tendency_residual": s1,
        "s2_relax_bdy_dry_split": s2,
        "s3_spec_bdy_dry": s3,
        "falsifier_outcomes": falsifiers,
        "terminal_decision": earliest,
        "metric_definition": {
            "bands": cpu_analysis["metric_definition"]["bands"],
            "combined_rmse": "cell-count-weighted across U- and V-staggered components",
            "systematic": "outside literal six-member WRF RMSE envelope and GPU RMSE > 2.0 * WRF member maximum RMSE",
            "derived_envelopes": "identical frozen metric applied to derived quantities of the identical six retained members; never a replacement for frozen envelopes",
        },
    }
    write_self_hashed(OUTPUT, payload)
    written = json.loads(OUTPUT.read_text())
    print(json.dumps({
        "wrote": str(OUTPUT),
        "proof_sha256": written["proof_sha256"],
        "F1": falsifiers["F1_wrf_assembly_oracle_authenticated"],
        "F2": falsifiers["F2_gpu_assembly_residual_within_member_envelope"],
        "F3": falsifiers["F3_rk_tendency_residual_within_derived_envelope"],
        "F4": falsifiers["F4_member_spec_rows_bitwise_identical"],
        "s0_inherit_rmse": s0["verdict_bits"]["inherit_rmse"],
        "s0_r_asm_rmse": s0["verdict_bits"]["r_asm_rmse"],
        "s0_oracle_rel": s0["verdict_bits"]["oracle_relative_rmse"],
        "s1_nonspec_rmse": s1["nonspec"]["gpu_rmse"],
        "s1_nonspec_env_max": s1["nonspec"]["derived_envelope_max"],
        "s2_field_side_rmse": s2["field_side_image"]["rmse"],
        "s2_target_side_E_rmse": s2["target_side_image_E"]["rmse"],
        "s3_diff_to_overwrite_ratio": s3["diff_to_overwrite_ratio"],
        "earliest_decisive": earliest["name"],
        "s1_ratio": earliest["s1_residual"]["ratio_to_envelope_max"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
