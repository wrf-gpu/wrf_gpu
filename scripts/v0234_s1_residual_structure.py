#!/usr/bin/env python3
"""Reconstruct and stratify the authentic fixed-run step-1 S1 residual.

CPU/offline only.  This script imports neither JAX nor gpuwrf and does not
touch the GPU lock.  The residual is the exact Kimi/Fable algebra:

  DYN_WRF_u = ru_tend - ru_tendf / msfuy - u_save
  DYN_GPU_u = ru_tend - ru_tendf / gpu_msfuy - u_save
  DYN_WRF_v = rv_tend - rv_tendf / msfvx - v_save
  DYN_GPU_v = rv_tend - rv_tendf / gpu_msfvx - v_save
  R = DYN_GPU - DYN_WRF

The input GPU arrays are the chief-authorized post-diff6 savepoints.  The WRF
arrays are reassembled from the output-neutral pristine-WRF-v4.7.1 rank dumps.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_dycore_internal_split_kimi as split  # noqa: E402
from scripts import v0234_dycore_suboperator_gpt_cpu_analysis as cpu  # noqa: E402
from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble  # noqa: E402
from scripts import v0234_s1_dyn_attribution_fable5 as s1  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt"
OUTPUT = SPRINT / "residual-field-analysis.json"
RUN = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
)
TERMINAL = RUN / "s1-diff6-validation-terminal-proof.json"
WRFINPUT = split.CONTROL_WRFINPUT
CONTROL_DUMPS = split.RUNS / "control/momsp_dumps"
FROZEN_CLOSEOUT = (
    REPO / ".agent/sprints/2026-07-18-v0234-s1-diff6-gpt-gpu-arm/proof.json"
)

EXPECTED_CLOSEOUT = "bd27f51711427e35fb966ac20a8ad18dd9a1ff840cd95aabeecf563d1f443d33"
EXPECTED_TERMINAL = "6e8ed620932459a6d663feafb8a64cdd3bf5cbd3aaa65a6e878dcdb1e70c677e"
EXPECTED_FRESH_ROWS = "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a"
EXPECTED_METRICS = {
    "nonspec": 3.22385592467145,
    "relax_rows_1_4": 7.765706866136884,
    "interior_ge_5": 1.1401580540399108,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def authenticate_json(path: Path, expected: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    observed = canonical({key: value for key, value in payload.items() if key != "proof_sha256"})
    if observed != payload.get("proof_sha256") or observed != expected:
        raise RuntimeError(f"self-hash mismatch: {path}")
    return payload


def metric(values: Iterable[np.ndarray]) -> dict[str, Any]:
    arrays = [np.asarray(value, dtype=np.float64).ravel() for value in values]
    count = sum(array.size for array in arrays)
    if count == 0:
        return {"cells": 0, "sse": 0.0, "rmse": None, "mean_bias": None, "max_abs": None}
    finite = sum(int(np.count_nonzero(np.isfinite(array))) for array in arrays)
    if finite != count:
        raise RuntimeError(f"nonfinite residual selection: {finite}/{count}")
    sse = sum(float(np.sum(np.square(array, dtype=np.float64), dtype=np.float64)) for array in arrays)
    total = sum(float(np.sum(array, dtype=np.float64)) for array in arrays)
    return {
        "cells": count,
        "sse": sse,
        "rmse": math.sqrt(sse / count),
        "mean_bias": total / count,
        "max_abs": max(float(np.max(np.abs(array))) for array in arrays),
    }


def corr(left: Iterable[np.ndarray], right: Iterable[np.ndarray]) -> float | None:
    x = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in left])
    y = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in right])
    if x.size == 0 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def stagger_x(mass: np.ndarray) -> np.ndarray:
    return np.concatenate((mass[:, :1], 0.5 * (mass[:, :-1] + mass[:, 1:]), mass[:, -1:]), axis=1)


def stagger_y(mass: np.ndarray) -> np.ndarray:
    return np.concatenate((mass[:1, :], 0.5 * (mass[:-1, :] + mass[1:, :]), mass[-1:, :]), axis=0)


def zone_mask(array: np.ndarray, name: str) -> np.ndarray:
    distance = cpu.distance_to_edge(array.shape)
    if name == "all":
        return np.ones(array.shape, dtype=bool)
    if name == "nonspec":
        return distance >= 1
    if name == "relax_rows_1_4":
        return (distance >= 1) & (distance <= 4)
    if name == "interior_ge_5":
        return distance >= 5
    raise KeyError(name)


def side_labels(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    _, ny, nx = shape
    west = np.broadcast_to(np.arange(nx)[None, :], (ny, nx))
    east = np.broadcast_to(np.arange(nx)[::-1][None, :], (ny, nx))
    south = np.broadcast_to(np.arange(ny)[:, None], (ny, nx))
    north = np.broadcast_to(np.arange(ny)[::-1][:, None], (ny, nx))
    distances = np.stack((west, east, south, north), axis=0)
    minimum = distances.min(axis=0)
    unique = np.sum(distances == minimum, axis=0) == 1
    labels = np.argmin(distances, axis=0)
    return np.broadcast_to(labels, shape), np.broadcast_to(unique, shape)


def rows_for_masks(residual: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
    return metric(residual[name][masks[name]] for name in ("u", "v"))


def main() -> None:
    closeout = json.loads(FROZEN_CLOSEOUT.read_text())
    closeout_body = {
        key: value for key, value in closeout.items()
        if key not in {"canonical_payload_sha256", "normalized_whole_file_self_sha256"}
    }
    if canonical(closeout_body) != closeout.get("canonical_payload_sha256") or closeout.get(
        "canonical_payload_sha256"
    ) != EXPECTED_CLOSEOUT:
        raise RuntimeError("terminal closeout canonical hash mismatch")
    terminal = authenticate_json(TERMINAL, EXPECTED_TERMINAL)
    rows = terminal["savepoint_manifest"]["rows"]
    if canonical(rows) != EXPECTED_FRESH_ROWS or len(rows) != 180:
        raise RuntimeError("fresh savepoint manifest mismatch")
    step_rows = {(row["tag"], row["field"]): row for row in rows if row["step"] == 1}
    if len(step_rows) != 20:
        raise RuntimeError("step-1 savepoint inventory mismatch")
    for row in step_rows.values():
        path = Path(row["path"])
        if path.parent != (RUN / "savepoints").resolve() or sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"fresh savepoint drift: {path}")

    def load(tag: str, field: str) -> np.ndarray:
        return np.load(step_rows[(tag, field)]["path"], allow_pickle=False)

    gpu = {
        "ru_tend": load("l1_rk1_tend", "ru_tend"),
        "rv_tend": load("l1_rk1_tend", "rv_tend"),
        "ru_tendf": load("sp3_tendf", "ru_tendf"),
        "rv_tendf": load("sp3_tendf", "rv_tendf"),
        "u_save": load("l1_rk1_relax", "u_save"),
        "v_save": load("l1_rk1_relax", "v_save"),
    }
    ranks = reassemble.load_ranks(CONTROL_DUMPS)
    control = {
        "ru_tend": reassemble.reassemble3d("l1_rk1_tend__ru_tend", 1, ranks),
        "rv_tend": reassemble.reassemble3d("l1_rk1_tend__rv_tend", 1, ranks),
        "ru_tendf": reassemble.reassemble3d("sp3_tendf__ru_tendf", 1, ranks),
        "rv_tendf": reassemble.reassemble3d("sp3_tendf__rv_tendf", 1, ranks),
        "u_save": reassemble.reassemble3d("l1_rk1_tend__u_save", 1, ranks),
        "v_save": reassemble.reassemble3d("l1_rk1_tend__v_save", 1, ranks),
    }
    split.GPU_INIT_FRAME = RUN / "gpu-output/wrfout_d03_2025-03-01_00:00:00"
    const = split.load_constants()
    residual = s1.s1_residual(gpu, control, const)
    bands = split.per_band_rmse(residual)
    reproduced = {
        "nonspec": split.combined_rmse(residual, s1.nonspec_masks(residual)),
        "relax_rows_1_4": bands["relax_rows_1_4"],
        "interior_ge_5": bands["interior_ge_5"],
    }
    if reproduced != EXPECTED_METRICS:
        raise RuntimeError(f"frozen metrics changed: {reproduced}")

    with Dataset(WRFINPUT) as dataset:
        hgt = np.asarray(dataset.variables["HGT"][0], dtype=np.float64)
        land = np.asarray(dataset.variables["LANDMASK"][0], dtype=np.float64)
    geometry = {
        "u": {"hgt": stagger_x(hgt), "land": stagger_x(land)},
        "v": {"hgt": stagger_y(hgt), "land": stagger_y(land)},
    }
    dhdy, dhdx = np.gradient(hgt)
    geometry["u"].update({"dhdx": stagger_x(dhdx), "dhdy": stagger_x(dhdy)})
    geometry["v"].update({"dhdx": stagger_y(dhdx), "dhdy": stagger_y(dhdy)})
    for comp in geometry:
        geometry[comp]["grad"] = np.hypot(geometry[comp]["dhdx"], geometry[comp]["dhdy"])

    zones = ("all", "nonspec", "relax_rows_1_4", "interior_ge_5")
    component_by_zone: dict[str, Any] = {}
    combined_by_zone: dict[str, Any] = {}
    for zone in zones:
        masks = {comp: zone_mask(array, zone) for comp, array in residual.items()}
        combined_by_zone[zone] = rows_for_masks(residual, masks)
        component_by_zone[zone] = {
            comp: metric((array[masks[comp]],)) for comp, array in residual.items()
        }

    vertical_by_zone: dict[str, Any] = {}
    vertical_groups_by_zone: dict[str, Any] = {}
    for zone in zones:
        masks = {comp: zone_mask(array, zone) for comp, array in residual.items()}
        profile = []
        for k in range(residual["u"].shape[0]):
            profile.append({
                "zero_based_k": k,
                "combined": metric((
                    residual["u"][k][masks["u"][k]],
                    residual["v"][k][masks["v"][k]],
                )),
                "u": metric((residual["u"][k][masks["u"][k]],)),
                "v": metric((residual["v"][k][masks["v"][k]],)),
            })
        vertical_by_zone[zone] = {
            "profile": profile,
            "top10_by_combined_rmse": sorted(
                ({"zero_based_k": row["zero_based_k"], "rmse": row["combined"]["rmse"]}
                 for row in profile),
                key=lambda row: -float(row["rmse"]),
            )[:10],
        }
        groups = {}
        for name, k0, k1 in (("k0_4", 0, 5), ("k5_9", 5, 10), ("k10_19", 10, 20),
                             ("k20_33", 20, 34), ("k34_43", 34, 44)):
            groups[name] = metric((
                residual["u"][k0:k1][masks["u"][k0:k1]],
                residual["v"][k0:k1][masks["v"][k0:k1]],
            ))
        vertical_groups_by_zone[zone] = groups

    boundary_distance = {}
    for distance in range(13):
        masks = {comp: cpu.distance_to_edge(array.shape) == distance for comp, array in residual.items()}
        boundary_distance[str(distance)] = rows_for_masks(residual, masks)
    masks = {comp: cpu.distance_to_edge(array.shape) >= 13 for comp, array in residual.items()}
    boundary_distance["13_plus"] = rows_for_masks(residual, masks)

    land_sea = {}
    terrain_gradient = {}
    terrain_elevation = {}
    grad_edges = (("0_10", 0.0, 10.0), ("10_50", 10.0, 50.0),
                  ("50_150", 50.0, 150.0), ("150_300", 150.0, 300.0),
                  ("300_plus", 300.0, float("inf")))
    elevation_edges = (("sea_or_zero", -float("inf"), 0.0), ("0_500", 0.0, 500.0),
                       ("500_1500", 500.0, 1500.0), ("1500_plus", 1500.0, float("inf")))
    for zone in ("relax_rows_1_4", "interior_ge_5", "nonspec"):
        zmask = {comp: zone_mask(array, zone) for comp, array in residual.items()}
        land_sea[zone] = {}
        for name, predicate in (
            ("sea", lambda value: value <= 0.25),
            ("coast_mixed", lambda value: (value > 0.25) & (value < 0.75)),
            ("land", lambda value: value >= 0.75),
        ):
            masks = {
                comp: zmask[comp] & np.broadcast_to(predicate(geometry[comp]["land"]), array.shape)
                for comp, array in residual.items()
            }
            land_sea[zone][name] = rows_for_masks(residual, masks)
        terrain_gradient[zone] = {}
        for name, low, high in grad_edges:
            masks = {
                comp: zmask[comp] & np.broadcast_to(
                    (geometry[comp]["grad"] >= low) & (geometry[comp]["grad"] < high), array.shape,
                ) for comp, array in residual.items()
            }
            terrain_gradient[zone][name] = rows_for_masks(residual, masks)
        terrain_elevation[zone] = {}
        for name, low, high in elevation_edges:
            masks = {
                comp: zmask[comp] & np.broadcast_to(
                    (geometry[comp]["hgt"] >= low) & (geometry[comp]["hgt"] < high), array.shape,
                ) for comp, array in residual.items()
            }
            terrain_elevation[zone][name] = rows_for_masks(residual, masks)

    correlations = {}
    for zone in ("relax_rows_1_4", "interior_ge_5", "nonspec"):
        masks = {comp: zone_mask(array, zone) for comp, array in residual.items()}
        correlations[zone] = {
            "abs_residual_vs_terrain_gradient": corr(
                (np.abs(residual[comp][masks[comp]]) for comp in ("u", "v")),
                (np.broadcast_to(geometry[comp]["grad"], residual[comp].shape)[masks[comp]] for comp in ("u", "v")),
            ),
            "u_signed_vs_dHdx": corr(
                (residual["u"][masks["u"]],),
                (np.broadcast_to(geometry["u"]["dhdx"], residual["u"].shape)[masks["u"]],),
            ),
            "v_signed_vs_dHdy": corr(
                (residual["v"][masks["v"]],),
                (np.broadcast_to(geometry["v"]["dhdy"], residual["v"].shape)[masks["v"]],),
            ),
        }

    directional = {}
    side_names = ("west", "east", "south", "north")
    labels = {comp: side_labels(array.shape) for comp, array in residual.items()}
    for zone in ("relax_rows_1_4", "interior_ge_5", "nonspec"):
        directional[zone] = {}
        for side_index, side in enumerate(side_names):
            masks = {
                comp: zone_mask(array, zone) & (labels[comp][0] == side_index) & labels[comp][1]
                for comp, array in residual.items()
            }
            row = rows_for_masks(residual, masks)
            row["u"] = metric((residual["u"][masks["u"]],))
            row["v"] = metric((residual["v"][masks["v"]],))
            if side in {"west", "east"}:
                row["normal_component"] = "u"
                row["tangential_component"] = "v"
            else:
                row["normal_component"] = "v"
                row["tangential_component"] = "u"
            directional[zone][side] = row
        tie_masks = {
            comp: zone_mask(array, zone) & ~labels[comp][1] for comp, array in residual.items()
        }
        directional[zone]["equal_distance_corner_ties"] = rows_for_masks(residual, tie_masks)

    dump_rows = []
    selected_tags = (
        "l1_rk1_tend__ru_tend", "l1_rk1_tend__rv_tend",
        "sp3_tendf__ru_tendf", "sp3_tendf__rv_tendf",
        "l1_rk1_tend__u_save", "l1_rk1_tend__v_save",
    )
    for rank in ranks:
        for tag in selected_tags:
            path = rank["dir"] / f"step000001_{tag}.f64"
            dump_rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.s1-residual-field-structure.v1",
        "step": 1,
        "domain": "authentic Tenerife d03 111x93 mass grid, 44 mass levels",
        "quantity": {
            "name": "post-diff6 S1 implied rk_tendency momentum residual",
            "formula": {
                "u": "(GPU ru_tend - GPU ru_tendf/GPU msfuy - GPU u_save) - (WRF ru_tend - WRF ru_tendf/WRF msfuy - WRF u_save)",
                "v": "(GPU rv_tend - GPU rv_tendf/GPU msfvx - GPU v_save) - (WRF rv_tend - WRF rv_tendf/WRF msfvx - WRF v_save)",
            },
            "units": "native coupled dry-momentum tendency units (ru_tend/rv_tend; approximately Pa m s^-2)",
            "normalization": "cell-count-weighted SSE/RMSE across native U- and V-staggered cells; no area or component reweighting",
            "precision": "GPU retained arrays float64; WRF native values were emitted as big-endian f64 by an output-neutral dumper and reassembled/evaluated in float64",
            "ownership": "full native U (44,93,112) and V (44,94,111); nonspec is distance-to-own-array-edge >=1; relax=1..4; interior>=5",
        },
        "authority": {
            "closeout_canonical_sha256": EXPECTED_CLOSEOUT,
            "terminal_self_sha256": EXPECTED_TERMINAL,
            "fresh_savepoint_rows_sha256": EXPECTED_FRESH_ROWS,
            "fresh_step1_rows": sorted(step_rows.values(), key=lambda row: (row["tag"], row["field"])),
            "wrf_dump_rows": dump_rows,
            "wrf_dump_rows_canonical_sha256": canonical(dump_rows),
            "wrfinput": {"path": str(WRFINPUT), "bytes": WRFINPUT.stat().st_size, "sha256": sha256_file(WRFINPUT)},
            "gpu_initial_frame": {
                "path": str(split.GPU_INIT_FRAME), "bytes": split.GPU_INIT_FRAME.stat().st_size,
                "sha256": sha256_file(split.GPU_INIT_FRAME),
            },
        },
        "reproduction": {"expected": EXPECTED_METRICS, "observed": reproduced, "exact": True},
        "residual_identity": {
            comp: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256_native_c_order": sha256_array(array)}
            for comp, array in residual.items()
        },
        "combined_by_zone": combined_by_zone,
        "component_by_zone": component_by_zone,
        "vertical_by_zone": vertical_by_zone,
        "vertical_groups_by_zone": vertical_groups_by_zone,
        "boundary_distance": boundary_distance,
        "land_sea": land_sea,
        "terrain_gradient_m_per_gridcell": terrain_gradient,
        "terrain_elevation_m": terrain_elevation,
        "correlations": correlations,
        "directional_unique_nearest_side": directional,
        "geometry_mapping": {
            "u_faces": "arithmetic mean of adjacent mass geometry, exterior face copies nearest mass cell",
            "v_faces": "arithmetic mean of adjacent mass geometry, exterior face copies nearest mass cell",
            "land_classes": "sea <=0.25, coast-mixed (0.25,0.75), land >=0.75 on staggered face average",
            "terrain_gradient": "numpy.gradient of mass HGT in metres per grid cell, then face averaged",
            "directional_sides": "exclusive unique-nearest side; equal-distance corner ties reported separately",
        },
        "attestation": {
            "gpu_commands_queries_locks_compiles_dispatches": 0,
            "jax_imported": "jax" in sys.modules,
            "gpuwrf_imported": any(name == "gpuwrf" or name.startswith("gpuwrf.") for name in sys.modules),
        },
    }
    if payload["attestation"]["jax_imported"] or payload["attestation"]["gpuwrf_imported"]:
        raise RuntimeError("CPU-only import discipline violated")
    payload["proof_sha256"] = canonical(payload)
    OUTPUT.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(OUTPUT), "proof_sha256": payload["proof_sha256"],
        "file_sha256": sha256_file(OUTPUT), "metrics": reproduced,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
