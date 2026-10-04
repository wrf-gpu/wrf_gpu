"""Detailed CPU closeout analysis for the v0234 dycore-suboperator ladder.

This continuation-owned, CPU-only analysis consumes the retained control and
six preregistered pristine-WRF-v4.7.1 members.  It does not import JAX, inspect
the GPU, or change the frozen discriminator.  It materializes the WRF rung and
span envelopes needed before the one frozen GPU comparison can be admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble  # noqa: E402

SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-gpt-continuation"
RUNS = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs")
OUTPUT = SPRINT / "cpu-ladder-analysis.json"
HGT_SOURCE = RUNS / "control/run/wrfinput_d03"
MEMBERS = (
    "mask-a-minus",
    "mask-a-plus",
    "mask-b-minus",
    "mask-b-plus",
    "mask-c-minus",
    "mask-c-plus",
)
STEPS = (1, 2)

# Canonical component names keep U/V correspondence explicit even when WRF's
# source field names change between tendency and state representations.
RUNG_CHAIN = (
    {
        "name": "sp3_tendf",
        "fields": {"u": "ru_tendf", "v": "rv_tendf"},
        "representation": "assembled_mass_coupled_momentum_tendency",
        "source_span_entry": "module_first_rk_step_part2.F after update_phy_ten",
    },
    {
        "name": "l1_rk1_tend",
        "fields": {"u": "ru_tend", "v": "rv_tend"},
        "representation": "rk1_merged_mass_coupled_momentum_tendency",
        "source_span_entry": "solve_em.F after relax_bdy_dry/rk_addtend_dry/spec_bdy_dry",
    },
    {
        "name": "l2_rk1_fin",
        "fields": {"u": "u", "v": "v"},
        "representation": "finished_momentum_state",
        "source_span_entry": "solve_em.F after rk1 small_step_finish",
    },
    {
        "name": "l3_rk2_fin",
        "fields": {"u": "u", "v": "v"},
        "representation": "finished_momentum_state",
        "source_span_entry": "solve_em.F after rk2 small_step_finish",
    },
    {
        "name": "l4_rk3_fin",
        "fields": {"u": "u", "v": "v"},
        "representation": "finished_momentum_state",
        "source_span_entry": "solve_em.F after rk3 small_step_finish",
    },
    {
        "name": "l5_prebdry",
        "fields": {"u": "u", "v": "v"},
        "representation": "pre_spec_bdy_final_momentum_state",
        "source_span_entry": "solve_em.F immediately before spec_bdy_final",
    },
    {
        "name": "sp4_exit",
        "fields": {"u": "u", "v": "v"},
        "representation": "end_of_step_momentum_state",
        "source_span_entry": "solve_em.F before RETURN",
    },
)

AUXILIARY_RUNGS = (
    {
        "name": "l1_rk1_relax",
        "dump_tag": "l1_rk1_tend",
        "fields": {"u": "u_save", "v": "v_save"},
        "representation": "rk1_boundary_relax_tendency_bundle",
        "source_span_entry": "solve_em.F after rk1 relax_bdy_dry",
    },
)

SPAN_NAMES = (
    "sp3_to_l1_tendency_build",
    "l1_to_l2_rk1_integration_finish",
    "l2_to_l3_rk2",
    "l3_to_l4_rk3",
    "l4_to_l5_post_loop_pre_final",
    "l5_to_sp4_end_of_step_boundary",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical_digest(value: object, *, omit: str | None = None) -> str:
    if omit is not None and isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != omit}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_self_hashed(path: Path, payload: dict) -> None:
    out = dict(payload)
    out["proof_sha256"] = canonical_digest(out, omit="proof_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def distance_to_edge(shape: tuple[int, int, int]) -> np.ndarray:
    _, ny, nx = shape
    jj = np.minimum(np.arange(ny), np.arange(ny)[::-1])[:, None]
    ii = np.minimum(np.arange(nx), np.arange(nx)[::-1])[None, :]
    return np.broadcast_to(np.minimum(jj, ii), shape)


def _hgt_at(hgt: np.ndarray, component: str, index: tuple[int, int, int]) -> dict:
    k, j, i = (int(value) for value in index)
    mass_j = min(j, hgt.shape[0] - 1)
    mass_i = min(i, hgt.shape[1] - 1)
    return {
        "component": component,
        "zero_based_k_j_i": [k, j, i],
        "wrf_one_based_k_j_i": [k + 1, j + 1, i + 1],
        "nearest_mass_zero_based_j_i": [mass_j, mass_i],
        "hgt_m": float(hgt[mass_j, mass_i]),
    }


def state_identity(arrays: dict[str, np.ndarray]) -> dict:
    return {
        component: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256_native_c_order": sha256_array(array),
            "finite_fraction": float(np.count_nonzero(np.isfinite(array)) / array.size),
        }
        for component, array in arrays.items()
    }


def combined_metrics(
    arrays: dict[str, np.ndarray],
    hgt: np.ndarray,
    *,
    band: str | None = None,
) -> dict:
    selected: list[tuple[str, np.ndarray, np.ndarray]] = []
    for component, array in arrays.items():
        dist = distance_to_edge(array.shape)
        if band == "spec_row_0":
            mask = dist == 0
        elif band == "relax_rows_1_4":
            mask = (dist >= 1) & (dist <= 4)
        elif band == "interior_ge_5":
            mask = dist >= 5
        else:
            mask = np.ones(array.shape, dtype=bool)
        selected.append((component, array, mask))

    count = sum(int(np.count_nonzero(mask)) for _, _, mask in selected)
    finite_count = sum(
        int(np.count_nonzero(np.isfinite(array[mask])))
        for _, array, mask in selected
    )
    if count == 0:
        raise RuntimeError(f"empty metric selection for band={band}")
    if finite_count != count:
        raise RuntimeError(f"non-finite ladder data for band={band}: {finite_count}/{count}")
    sse = sum(
        float(np.sum(np.square(array[mask], dtype=np.float64), dtype=np.float64))
        for _, array, mask in selected
    )
    total = sum(
        float(np.sum(array[mask], dtype=np.float64))
        for _, array, mask in selected
    )
    max_component = ""
    max_index = (0, 0, 0)
    max_abs = -1.0
    signed_at_max = 0.0
    for component, array, mask in selected:
        candidate = np.where(mask, np.abs(array), -np.inf)
        flat = int(np.argmax(candidate))
        index = tuple(int(value) for value in np.unravel_index(flat, array.shape))
        value = float(abs(array[index]))
        if value > max_abs:
            max_abs = value
            max_component = component
            max_index = index
            signed_at_max = float(array[index])
    return {
        "cells": count,
        "finite_fraction": float(finite_count / count),
        "sse": sse,
        "rmse": float(math.sqrt(sse / count)),
        "mean_bias": float(total / count),
        "max_abs": max_abs,
        "signed_error_at_max_abs": signed_at_max,
        "argmax": _hgt_at(hgt, max_component, max_index),
    }


def metrics_with_bands(arrays: dict[str, np.ndarray], hgt: np.ndarray) -> dict:
    overall = combined_metrics(arrays, hgt)
    bands = {
        name: combined_metrics(arrays, hgt, band=name)
        for name in ("spec_row_0", "relax_rows_1_4", "interior_ge_5")
    }
    for row in bands.values():
        row["sse_share"] = float(row["sse"] / overall["sse"]) if overall["sse"] else 0.0
    overall["bands"] = bands
    overall["difference_identity"] = state_identity(arrays)
    return overall


def load_rung(
    ranks: list[dict],
    step: int,
    rung: dict,
) -> dict[str, np.ndarray]:
    return {
        component: reassemble.reassemble3d(
            f"{rung.get('dump_tag', rung['name'])}__{field}", step, ranks,
        )
        for component, field in rung["fields"].items()
    }


def subtract(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(left) != set(right):
        raise RuntimeError(f"component mismatch: {sorted(left)} != {sorted(right)}")
    out = {}
    for component in left:
        if left[component].shape != right[component].shape:
            raise RuntimeError(
                f"shape mismatch {component}: {left[component].shape} != {right[component].shape}"
            )
        out[component] = left[component] - right[component]
    return out


def add_envelope(rows: dict[str, dict]) -> dict:
    rmse = {member: float(row["rmse"]) for member, row in rows.items()}
    max_abs = {member: float(row["max_abs"]) for member, row in rows.items()}
    return {
        "member_rmse": rmse,
        "wrf_envelope_min_rmse": min(rmse.values()),
        "wrf_envelope_max_rmse": max(rmse.values()),
        "wrf_envelope_max_member": max(rmse, key=rmse.get),
        "member_max_abs": max_abs,
        "wrf_maximum_member_max_abs": max(max_abs.values()),
    }


def load_hgt() -> np.ndarray:
    from netCDF4 import Dataset

    with Dataset(HGT_SOURCE) as dataset:
        hgt = np.asarray(dataset.variables["HGT"][0], dtype=np.float64)
    if hgt.shape != (reassemble.JDE - 1, reassemble.IDE - 1):
        raise RuntimeError(f"unexpected HGT shape {hgt.shape}")
    if not np.isfinite(hgt).all():
        raise RuntimeError("non-finite HGT")
    return hgt


def main() -> None:
    hgt = load_hgt()
    control_ranks = reassemble.load_ranks(RUNS / "control/momsp_dumps")
    member_ranks = {
        member: reassemble.load_ranks(RUNS / member / "momsp_dumps")
        for member in MEMBERS
    }
    steps: dict[str, dict] = {}
    for step in STEPS:
        control = {
            rung["name"]: load_rung(control_ranks, step, rung)
            for rung in RUNG_CHAIN
        }
        members = {
            member: {
                rung["name"]: load_rung(ranks, step, rung)
                for rung in RUNG_CHAIN
            }
            for member, ranks in member_ranks.items()
        }
        control_auxiliary = {
            rung["name"]: load_rung(control_ranks, step, rung)
            for rung in AUXILIARY_RUNGS
        }
        member_auxiliary = {
            member: {
                rung["name"]: load_rung(ranks, step, rung)
                for rung in AUXILIARY_RUNGS
            }
            for member, ranks in member_ranks.items()
        }
        rung_rows = {}
        for rung in RUNG_CHAIN:
            name = rung["name"]
            comparisons = {
                member: metrics_with_bands(
                    subtract(member_values[name], control[name]), hgt,
                )
                for member, member_values in members.items()
            }
            rung_rows[name] = {
                "representation": rung["representation"],
                "source_boundary": rung["source_span_entry"],
                "fields": rung["fields"],
                "control_state": state_identity(control[name]),
                "member_state": {
                    member: state_identity(member_values[name])
                    for member, member_values in members.items()
                },
                "member_vs_control": comparisons,
                "envelope": add_envelope(comparisons),
            }

        span_rows = {}
        for index, span_name in enumerate(SPAN_NAMES):
            before_spec = RUNG_CHAIN[index]
            after_spec = RUNG_CHAIN[index + 1]
            before = before_spec["name"]
            after = after_spec["name"]
            control_update = subtract(control[after], control[before])
            member_rows = {}
            for member, member_values in members.items():
                member_update = subtract(member_values[after], member_values[before])
                response_delta = subtract(member_update, control_update)
                member_rows[member] = {
                    "before_state": state_identity(member_values[before]),
                    "after_state": state_identity(member_values[after]),
                    "member_update": state_identity(member_update),
                    "control_update": state_identity(control_update),
                    "update_difference": metrics_with_bands(response_delta, hgt),
                }
            # Every frozen span keeps like physical dimensions except L1->L2,
            # which deliberately crosses from a merged tendency to a finished
            # velocity state.  Retain its preregistered arithmetic but flag it.
            compatible = span_name != "l1_to_l2_rk1_integration_finish"
            comparison_rows = {
                member: row["update_difference"] for member, row in member_rows.items()
            }
            span_rows[span_name] = {
                "before_rung": before,
                "after_rung": after,
                "before_representation": before_spec["representation"],
                "after_representation": after_spec["representation"],
                "dimensionally_compatible_update_subtraction": compatible,
                "interpretation": (
                    "frozen span update-difference metric"
                    if compatible
                    else "frozen arithmetic retained, but cross-representation value is not a physical tendency"
                ),
                "control_before_state": state_identity(control[before]),
                "control_after_state": state_identity(control[after]),
                "control_update": state_identity(control_update),
                "members": member_rows,
                "envelope": add_envelope(comparison_rows),
            }
        auxiliary_rows = {}
        for rung in AUXILIARY_RUNGS:
            name = rung["name"]
            comparisons = {
                member: metrics_with_bands(
                    subtract(member_values[name], control_auxiliary[name]), hgt,
                )
                for member, member_values in member_auxiliary.items()
            }
            auxiliary_rows[name] = {
                "representation": rung["representation"],
                "source_boundary": rung["source_span_entry"],
                "wrf_dump_tag": rung["dump_tag"],
                "fields": rung["fields"],
                "control_state": state_identity(control_auxiliary[name]),
                "member_state": {
                    member: state_identity(member_values[name])
                    for member, member_values in member_auxiliary.items()
                },
                "member_vs_control": comparisons,
                "envelope": add_envelope(comparisons),
                "included_in_primary_chain": False,
            }
        steps[f"step{step}"] = {
            "rungs": rung_rows,
            "auxiliary_rungs": auxiliary_rows,
            "spans": span_rows,
        }

    payload = {
        "schema": "gpuwrf.v0234.dycore-suboperator-gpt-continuation.cpu-ladder-analysis.v1",
        "verdict": "CPU_LADDER_ENVELOPES_COMPLETE_GPU_VALUES_REQUIRED",
        "steps": steps,
        "member_order": list(MEMBERS),
        "rung_order": [rung["name"] for rung in RUNG_CHAIN],
        "span_order": list(SPAN_NAMES),
        "auxiliary_rung_order": [rung["name"] for rung in AUXILIARY_RUNGS],
        "metric_definition": {
            "rung": "member_rung_state - control_rung_state",
            "span": "(member_after - member_before) - (control_after - control_before)",
            "combined_rmse": "cell-count-weighted across the U- and V-staggered components",
            "arithmetic_dtype": "float64 from WRF f64 diagnostic dumps",
            "bands": {
                "spec_row_0": "minimum staggered-grid distance to edge == 0",
                "relax_rows_1_4": "minimum distance 1..4",
                "interior_ge_5": "minimum distance >= 5",
            },
            "argmax_hgt_mapping": "nearest lower/clamped mass-grid HGT cell",
            "frozen_systematic_factor": 2.0,
        },
        "authority": {
            "wrf_binary": str(
                Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/")
                / "wrf_iso_ladder/install_ladder/bin/wrf"
            ),
            "wrf_binary_sha256": sha256_file(
                Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/")
                / "wrf_iso_ladder/install_ladder/bin/wrf"
            ),
            "hgt_source": str(HGT_SOURCE),
            "hgt_source_sha256": sha256_file(HGT_SOURCE),
            "hgt_shape": list(hgt.shape),
            "hgt_sha256_native_c_order": sha256_array(hgt),
            "runs_root": str(RUNS),
            "gpu_values_present": False,
        },
        "terminal_cpu_conclusion": {
            "cpu_envelopes_ready": True,
            "earliest_gpu_systematic_rung_resolvable_without_gpu_values": False,
            "reason": "the retained GPU evidence has SP3/SP4 but no L1-L5 values",
            "exact_next_discriminator": (
                "run the single preregistered proof-only d03 ladder arm and merge its "
                "L1-L5 values with these frozen WRF envelopes"
            ),
        },
    }
    write_self_hashed(OUTPUT, payload)
    print(json.dumps({
        "wrote": str(OUTPUT),
        "verdict": payload["verdict"],
        "proof_sha256": json.loads(OUTPUT.read_text())["proof_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
