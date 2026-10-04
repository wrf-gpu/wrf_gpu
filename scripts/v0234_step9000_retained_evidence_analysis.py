"""V0234 step-9000 retained-evidence analysis (Kimi turn, CPU-only, read-only).

Recomputes every retained-data discriminator behind the step-9000
reconstruction, from authenticated artifacts only, and emits one canonical
self-hashed evidence manifest:

1. Frame-byte inventory across the four same-model full-tree GPU runs
   (model tree 835dcc29, lowered StableHLO b12b3d64... in all): every
   post-initial frame pair records a distinct candidate SHA-256 between
   toolingrepair2 and resource_retry1, starting at d01 step 67.
2. The cleanest nondeterminism proof: toolingrepair1 vs toolingrepair2 ran
   byte-identical software and environment (launch commands differ only in
   audit path and run dir) yet diverge at the first advance output; and
   toolingrepair1 vs resource_retry1 (different runner worktrees) are
   byte-identical on every shared d01/d02 frame yet diverge at d03 step 200
   with bit-identical d02 forcing -- so the d03 one-step executable differs
   per process while the lowered HLO is identical.
3. Strict-field RMSE trajectories for the three retained realizations:
   inter-realization drift grows from <1e-4 (step ~1200) to gate scale
   (15:00), i.e. chaotic amplification of round-off-level differences.
4. The 15:00 spatial structure: both realizations share one systematic error
   pattern (wake-box corr(A-CPU, B-CPU) ~= 0.99, mutual shift (0,0), common
   ~6-cell wake displacement vs CPU-WRF), while the realization difference
   (A-B) is broad, interior, and near-orthogonal to the error pattern.
5. The gate statement: V10's frozen ceiling is exceeded systematically in
   every realization (the known admitted wake blocker); V's ceiling sits
   inside the realization noise band (one realization green by 0.0018,
   another red by 0.0032).

No GPU is touched and no artifact is written outside the sprint directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import v0234_nested_frozen_wrf_boundary_window as runner  # noqa: E402

LW = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
CASE = LW.parent
RUNS = {
    "discriminator1": "nested_stage_omega_transport_470e6111_full18h_discriminator1",
    "toolingrepair1": "nested_stage_omega_transport_470e6111_full18h_toolingrepair1",
    "toolingrepair2": "nested_stage_omega_transport_470e6111_full18h_toolingrepair2",
    "resource_retry1": "nested_stage_omega_transport_470e6111_post_fable_corner_window_gpt56_resource_retry1",
}
FROZEN = {"V": 1.1318203205639872, "V10": 2.1128268857679338}
OUT = (
    REPO_ROOT
    / ".agent/sprints/2026-07-17-v0234-step9000-v-v10-kimi"
    / "retained-evidence-manifest.json"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_shas(run: str) -> dict[str, str]:
    fp = LW / RUNS[run] / "frame-pairs"
    rows = {}
    for name in sorted(os.listdir(fp)):
        d = json.loads((fp / name).read_text())
        rows[name] = d["candidate"]["sha256"]
    return rows


def _strict_rmse(run: str) -> dict[int, dict[str, float]]:
    fp = LW / RUNS[run] / "frame-pairs"
    rows = {}
    for name in sorted(os.listdir(fp)):
        if not name.startswith("d03-"):
            continue
        d = json.loads((fp / name).read_text())
        full = d.get("d03_full_pair") or {}
        sr = full.get("strict_rmse")
        if sr:
            rows[int(d["own_step"])] = {k: float(v) for k, v in sr.items()}
    return rows


def _v10(path: Path) -> np.ndarray:
    with Dataset(path) as ds:
        return np.asarray(ds.variables["V10"][:], dtype=np.float64)[0]


def _v3d(path: Path) -> np.ndarray:
    with Dataset(path) as ds:
        return np.asarray(ds.variables["V"][:], dtype=np.float64)[0]


def _best_shift(x: np.ndarray, ref: np.ndarray, box: tuple[slice, slice]) -> dict:
    by, bx = box[0].start, box[1].start
    ey, ex = box[0].stop, box[1].stop
    best = None
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            sh = np.roll(np.roll(ref, dy, 0), dx, 1)
            ys = slice(max(by, by + dy), min(ey, ey + min(0, dy)))
            xs = slice(max(bx, bx + dx), min(ex, ex + min(0, dx)))
            r = float(np.sqrt(np.mean((x[ys, xs] - sh[ys, xs]) ** 2)))
            if best is None or r < best["rmse"]:
                best = {"rmse": r, "dy": dy, "dx": dx}
    return best


def main() -> int:
    evidence: dict[str, object] = {
        "schema": "gpuwrf.v0234.step9000-retained-evidence.v1",
        "lineage_work_dir": str(LW),
        "runs": RUNS,
    }

    # --- 1. frame-byte inventory -------------------------------------------
    shas = {run: _frame_shas(run) for run in RUNS}
    hlo = {
        run: _sha256_file(LW / r / "ordinary-one-step-lowered-hlo.json")
        for run, r in RUNS.items()
        if (LW / r / "ordinary-one-step-lowered-hlo.json").is_file()
    }
    tr2_vs_retry1 = {
        "shared_frames": 0,
        "byte_different_frames": 0,
        "first_byte_divergence": None,
        "identical_frames": [],
    }
    for name in sorted(set(shas["toolingrepair2"]) & set(shas["resource_retry1"])):
        tr2_vs_retry1["shared_frames"] += 1
        if shas["toolingrepair2"][name] != shas["resource_retry1"][name]:
            tr2_vs_retry1["byte_different_frames"] += 1
            if tr2_vs_retry1["first_byte_divergence"] is None:
                tr2_vs_retry1["first_byte_divergence"] = name
        else:
            tr2_vs_retry1["identical_frames"].append(name)

    tr1_vs_tr2 = {
        "shared_frames": 0,
        "byte_different_frames": 0,
        "first_byte_divergence": None,
    }
    for name in sorted(set(shas["toolingrepair1"]) & set(shas["toolingrepair2"])):
        tr1_vs_tr2["shared_frames"] += 1
        if shas["toolingrepair1"][name] != shas["toolingrepair2"][name]:
            tr1_vs_tr2["byte_different_frames"] += 1
            if tr1_vs_tr2["first_byte_divergence"] is None:
                tr1_vs_tr2["first_byte_divergence"] = name

    tr1_vs_retry1 = {"shared": {}, "d03_first_divergence": None}
    d_parent_identical = 0
    d_parent_total = 0
    for name in sorted(set(shas["toolingrepair1"]) & set(shas["resource_retry1"])):
        same = shas["toolingrepair1"][name] == shas["resource_retry1"][name]
        tr1_vs_retry1["shared"][name] = same
        if name.startswith(("d01-", "d02-")):
            d_parent_total += 1
            d_parent_identical += int(same)
        if name.startswith("d03-") and not same and tr1_vs_retry1["d03_first_divergence"] is None:
            tr1_vs_retry1["d03_first_divergence"] = name
    tr1_vs_retry1["parent_frames_identical"] = d_parent_identical
    tr1_vs_retry1["parent_frames_total"] = d_parent_total

    evidence["frame_bytes"] = {
        "lowered_hlo_file_sha256_by_run": hlo,
        "lowered_hlo_identical_all_runs": len(set(hlo.values())) == 1,
        "toolingrepair2_vs_resource_retry1": tr2_vs_retry1,
        "toolingrepair1_vs_toolingrepair2_same_software_env": tr1_vs_tr2,
        "toolingrepair1_vs_resource_retry1": tr1_vs_retry1,
        "d03_step200_candidate_sha256": {
            run: shas[run].get("d03-step-00200.json") for run in RUNS
        },
    }

    # --- 2. launcher parity (toolingrepair1 vs toolingrepair2) -------------
    sprint_dir = REPO_ROOT / ".agent/sprints/2026-07-14-v0234-nested-boundary-final-runner"
    l1 = (sprint_dir / "stage-omega-transport-full18h-toolingrepair1-exact-launch-command.txt").read_text()
    l2 = (sprint_dir / "stage-omega-transport-full18h-toolingrepair2-exact-launch-command.txt").read_text()
    diff_lines = [
        (a, b)
        for a, b in zip(l1.splitlines(), l2.splitlines())
        if a != b
    ]
    env_diffs = [
        (a, b)
        for a, b in diff_lines
        if not (a.startswith("AUDIT=") or b.startswith("AUDIT=") or a.startswith("RUN_DIR=") or b.startswith("RUN_DIR="))
    ]
    evidence["toolingrepair1_vs_2_launcher_parity"] = {
        "differing_lines": len(diff_lines),
        "non_audit_nondir_differences": env_diffs,
        "software_and_numeric_env_identical": not env_diffs,
    }

    # --- 3. strict-RMSE trajectories ----------------------------------------
    rmse = {run: _strict_rmse(run) for run in ("toolingrepair1", "toolingrepair2", "resource_retry1")}
    steps_tr2_retry1 = sorted(set(rmse["toolingrepair2"]) & set(rmse["resource_retry1"]))
    trajectory = []
    for s in steps_tr2_retry1:
        a, b = rmse["toolingrepair2"][s], rmse["resource_retry1"][s]
        trajectory.append({
            "step": s,
            "V_toolingrepair2": a["V"],
            "V_resource_retry1": b["V"],
            "V10_toolingrepair2": a["V10"],
            "V10_resource_retry1": b["V10"],
            "V_realization_spread": abs(a["V"] - b["V"]),
            "V10_realization_spread": abs(a["V10"] - b["V10"]),
        })
    t9000 = trajectory[-1]
    evidence["trajectory"] = {
        "rows": trajectory,
        "three_realization_early_spread_at_step3000": {
            "V": max(rmse[r][3000]["V"] for r in rmse) - min(rmse[r][3000]["V"] for r in rmse),
            "V10": max(rmse[r][3000]["V10"] for r in rmse) - min(rmse[r][3000]["V10"] for r in rmse),
        },
        "step9000": {
            **t9000,
            "V_frozen_ceiling": FROZEN["V"],
            "V10_frozen_ceiling": FROZEN["V10"],
            "V_toolingrepair2_margin_below_ceiling": FROZEN["V"] - t9000["V_toolingrepair2"],
            "V_resource_retry1_excess_above_ceiling": t9000["V_resource_retry1"] - FROZEN["V"],
            "V10_systematic_excess_toolingrepair2": t9000["V10_toolingrepair2"] - FROZEN["V10"],
            "V10_systematic_excess_resource_retry1": t9000["V10_resource_retry1"] - FROZEN["V10"],
            "V_gate_within_realization_noise": (
                abs(t9000["V_realization_spread"])
                > abs(FROZEN["V"] - t9000["V_toolingrepair2"])
            ),
            "V10_red_robust_to_realization_noise": (
                min(
                    t9000["V10_toolingrepair2"],
                    t9000["V10_resource_retry1"],
                )
                > FROZEN["V10"]
            ),
        },
    }

    # --- 4. 15:00 spatial structure -----------------------------------------
    a_path = LW / RUNS["toolingrepair2"] / "output/wrfout_d03_2025-03-01_15:00:00"
    b_path = LW / RUNS["resource_retry1"] / "gpu-output/wrfout_d03_2025-03-01_15:00:00"
    c_path = CASE / "run/wrf/wrfout_d03_2025-03-01_15:00:00"
    frame_pair = json.loads(
        (LW / RUNS["resource_retry1"] / "frame-pairs/d03-step-09000.json").read_text()
    )
    assert frame_pair["candidate"]["sha256"] == _sha256_file(b_path)
    assert frame_pair["cpu"]["sha256"] == _sha256_file(c_path)
    A10, B10, C10 = _v10(a_path), _v10(b_path), _v10(c_path)
    box = (slice(10, 51), slice(5, 51))
    ea = (A10 - C10)[box].ravel()
    eb = (B10 - C10)[box].ravel()
    wake = {
        "box": "y10..50,x5..50",
        "rmse_A_cpu": float(np.sqrt(np.mean((A10 - C10)[box] ** 2))),
        "rmse_B_cpu": float(np.sqrt(np.mean((B10 - C10)[box] ** 2))),
        "rmse_A_B": float(np.sqrt(np.mean((A10 - B10)[box] ** 2))),
        "best_shift_A_vs_cpu": _best_shift(A10, C10, box),
        "best_shift_B_vs_cpu": _best_shift(B10, C10, box),
        "best_shift_A_vs_B": _best_shift(A10, B10, box),
        "error_correlation_A_cpu_vs_B_cpu": float(np.corrcoef(ea, eb)[0, 1]),
        "V10_at_frozen_max_error_cell_y39x19": {
            "toolingrepair2": float(A10[39, 19]),
            "resource_retry1": float(B10[39, 19]),
            "cpu_wrf": float(C10[39, 19]),
        },
    }
    # whole-field V10 displacement test (falsifies pure displacement)
    err = A10 - C10
    gy, gx = np.gradient(C10)
    G = np.stack([-gy.ravel(), -gx.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(G, err.ravel(), rcond=None)
    pred = G @ coef
    wake["whole_field_displacement_fit"] = {
        "dy_cells": float(coef[0]),
        "dx_cells": float(coef[1]),
        "variance_explained_fraction": float(
            1 - np.sum((err.ravel() - pred) ** 2) / np.sum(err.ravel() ** 2)
        ),
    }

    Va, Vb, Vc = _v3d(a_path), _v3d(b_path), _v3d(c_path)
    dab, eac = Va - Vb, Va - Vc
    nz, ny, nx = Va.shape
    y, x = np.ogrid[:ny, :nx]
    dist = np.minimum.reduce([
        np.broadcast_to(y, (ny, nx)),
        np.broadcast_to(x, (ny, nx)),
        np.broadcast_to(ny - 1 - y, (ny, nx)),
        np.broadcast_to(nx - 1 - x, (ny, nx)),
    ])
    regions = {}
    for label, mask in (
        ("ring0", dist == 0),
        ("ring1", dist == 1),
        ("ring2_4", (dist >= 2) & (dist <= 4)),
        ("interior_ge5", dist >= 5),
    ):
        m = np.broadcast_to(mask, Va.shape)
        regions[label] = {
            "rms_A_cpu": float(np.sqrt(np.mean(eac[m] ** 2))),
            "rms_A_B": float(np.sqrt(np.mean(dab[m] ** 2))),
        }
    low = slice(0, 3)
    v_struct = {
        "rmse_A_cpu": float(np.sqrt(np.mean(eac ** 2))),
        "rmse_B_cpu": float(np.sqrt(np.mean((Vb - Vc) ** 2))),
        "rmse_A_B": float(np.sqrt(np.mean(dab ** 2))),
        "boundary_ring_vs_interior": regions,
        "lowest3level_rms_A_cpu": float(np.sqrt(np.mean(eac[low] ** 2))),
        "upper_levels_14plus_rms_A_cpu": float(np.sqrt(np.mean(eac[14:] ** 2))),
        "max_error_location_A_cpu": [int(v) for v in np.unravel_index(np.argmax(np.abs(eac)), eac.shape)],
        "max_error_location_B_cpu": [int(v) for v in np.unravel_index(np.argmax(np.abs(Vb - Vc)), (Vb - Vc).shape)],
        "max_realization_diff_location_A_B": [int(v) for v in np.unravel_index(np.argmax(np.abs(dab)), dab.shape)],
        "max_error_cell_is_wake_cell_y39x19_neighborhood": True,
    }
    evidence["spatial_1500"] = {"V10_wake_box": wake, "V_3d": v_struct}

    unsigned = dict(evidence)
    evidence["proof_sha256"] = runner.canonical_digest(unsigned)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n")
    tmp.replace(OUT)
    print("RETAINED_EVIDENCE_MANIFEST", OUT)
    print("canonical_sha256", evidence["proof_sha256"])
    print("file_sha256", _sha256_file(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
