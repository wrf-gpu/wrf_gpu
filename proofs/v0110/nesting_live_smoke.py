#!/usr/bin/env python3
"""Short real-data live nested forecast smoke for v0.11.0 nesting.

This is intentionally a smoke gate, not a 24 h equivalence claim.  It loads the
Gen2 Canary domains, builds a live ``DomainTree``, advances the root for a small
number of native d01 steps, and records finite/stability and multi-domain output
events.  Any OOM, unsupported option, or non-finite state is reported honestly in
the JSON proof.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.integration.d02_replay import DEFAULT_REPLAY_RUN_DIR, ReplayCase, build_replay_case
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    run_operational_domain_tree,
    with_live_child_boundary_config,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist


DT_BY_DOMAIN = {
    "d01": 18.0,
    "d02": 6.0,
    "d03": 2.0,
    "d04": 2.0,
    "d05": 2.0,
}


def _jsonable_error(exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _parent_for(domain: str) -> str | None:
    if domain == "d01":
        return None
    if domain == "d02":
        return "d01"
    return "d02"


def _nest_from_child(case: ReplayCase, parent: str) -> DomainNest:
    nesting = case.metadata.get("boundary", {}).get("nesting", {})
    return DomainNest(
        parent,
        str(case.metadata["domain"]),
        int(nesting["parent_grid_ratio"]),
        int(nesting["i_parent_start"]),
        int(nesting["j_parent_start"]),
        feedback=False,
    )


def _make_namelist(case: ReplayCase, *, domain: str, parent_dt_s: float | None, ratio: int | None) -> OperationalNamelist:
    state_dt = DT_BY_DOMAIN[domain]
    if parent_dt_s is not None and ratio is not None:
        expected = float(parent_dt_s) / float(ratio)
        if abs(state_dt - expected) > 1.0e-9:
            raise ValueError(f"{domain}: dt_s={state_dt} does not match parent_dt/ratio={expected}")
    radiation_cadence = max(1, int(round(1800.0 / state_dt)))
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=state_dt,
        acoustic_substeps=10,
        radiation_cadence_steps=radiation_cadence,
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
    )
    if parent_dt_s is not None:
        namelist = with_live_child_boundary_config(
            namelist,
            parent_dt_s=float(parent_dt_s),
            nested_ph_relax=True,
            # Keep WRF's w leaf in the forced package, but leave in-loop w relax
            # off for this short smoke until the longer feedback/radiation gate
            # revalidates it.  This is recorded as a carry-over in status.
            nested_w_relax=False,
            nested_ph_spec=True,
        )
    if domain == "d01":
        # Canary d01 uses cumulus.  KF is scan-wired in this v0.10.0 base; if it
        # regresses, _resolve_operational_suite will fail closed in the runtime.
        namelist = dataclass_replace(namelist, cu_physics=1)
    return namelist


def _load_domains(run_dir: Path, max_dom: int) -> tuple[DomainHierarchy, dict[str, DomainBundle], dict[str, Any]]:
    names = tuple(f"d{i:02d}" for i in range(1, int(max_dom) + 1))
    cases: dict[str, ReplayCase] = {}
    edges: list[DomainNest] = []
    for name in names:
        parent = _parent_for(name)
        cases[name] = build_replay_case(run_dir, domain=name, boundary_domain=parent)
        if parent is not None:
            edges.append(_nest_from_child(cases[name], parent))

    hierarchy = DomainHierarchy.from_edges(names, tuple(edges), max_dom=5)
    bundles: dict[str, DomainBundle] = {}
    meta: dict[str, Any] = {"domains": {}, "edges": [edge.__dict__ for edge in edges]}
    ratio_by_child = {edge.child: edge.parent_grid_ratio for edge in edges}
    for name, case in cases.items():
        parent = hierarchy.parent(name)
        parent_dt = DT_BY_DOMAIN[parent] if parent is not None else None
        state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
        namelist = _make_namelist(
            case,
            domain=name,
            parent_dt_s=parent_dt,
            ratio=ratio_by_child.get(name),
        )
        bundles[name] = DomainBundle(name=name, state=state, namelist=namelist, grid=case.grid, metrics=case.metrics)
        meta["domains"][name] = {
            "grid": case.metadata.get("grid", {}),
            "boundary_source": case.metadata.get("boundary", {}).get("source"),
            "qke_coldstart": case.metadata.get("qke_coldstart", {}),
            "namelist": {
                "dt_s": float(namelist.dt_s),
                "acoustic_substeps": int(namelist.acoustic_substeps),
                "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
                "boundary_update_cadence_s": float(namelist.boundary_config.update_cadence_s),
                "force_geopotential": bool(namelist.boundary_config.force_geopotential),
                "nested_ph_relax": bool(namelist.boundary_config.nested_ph_relax),
                "nested_w_relax": bool(namelist.boundary_config.nested_w_relax),
                "nested_ph_spec": bool(namelist.boundary_config.nested_ph_spec),
                "cu_physics": int(namelist.cu_physics),
            },
        }
    return hierarchy, bundles, meta


def _finite_stats(state) -> dict[str, Any]:
    fields = (
        "theta",
        "qv",
        "u",
        "v",
        "w",
        "p_perturbation",
        "ph_perturbation",
        "mu_perturbation",
        "qke",
    )
    stats: dict[str, Any] = {}
    all_finite = True
    for name in fields:
        value = getattr(state, name)
        arr = np.asarray(value)
        finite = np.isfinite(arr)
        all_finite = all_finite and bool(np.all(finite))
        stats[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite": bool(np.all(finite)),
            "finite_fraction": float(np.mean(finite)),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }
    stats["all_finite"] = bool(all_finite)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_REPLAY_RUN_DIR)
    parser.add_argument("--max-dom", type=int, default=3, choices=(2, 3, 4, 5))
    parser.add_argument("--root-steps", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("proofs/v0110/nesting_live_smoke.json"))
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "proof": "v0110 live nested real-data smoke",
        "run_dir": str(args.run_dir),
        "max_dom": int(args.max_dom),
        "root_steps": int(args.root_steps),
        "feedback": "off",
        "status": "UNKNOWN",
    }
    try:
        hierarchy, bundles, meta = _load_domains(args.run_dir, int(args.max_dom))
        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
        output_cadence = hierarchy.expected_step_counts(root_steps=int(args.root_steps))
        result = run_operational_domain_tree(
            tree,
            root_steps=int(args.root_steps),
            feedback_enabled=False,
            output=lambda name, step, state: {
                "domain": name,
                "own_step": int(step),
                "theta_shape": list(state.theta.shape),
                "boundary_time_levels": int(state.theta_bdy.shape[0]),
            },
            output_cadence_steps=output_cadence,
            block_between=True,
        )
        finite = {name: _finite_stats(state) for name, state in result.states.items()}
        payload.update(
            {
                "status": "PASS" if all(item["all_finite"] for item in finite.values()) else "FAIL",
                "metadata": meta,
                "hierarchy": {
                    "order": list(hierarchy.order),
                    "edges": [edge.__dict__ for edge in hierarchy.nests],
                    "observed_own_steps": result.own_steps,
                    "expected_own_steps": output_cadence,
                    "persistent_state_bytes": tree.persistent_state_bytes(),
                },
                "outputs": list(result.outputs),
                "event_count": len(result.events),
                "finite": finite,
                "carry_overs": [
                    "This is a short smoke, not 24 h equivalence.",
                    "Two-way feedback is implemented behind a gate but not enabled here.",
                    "In-loop nested w relaxation is off for this smoke pending the longer stability gate.",
                    "No CPU-WRF/RMSE scoring is claimed from this sub-minute window.",
                ],
            }
        )
    except BaseException as exc:  # noqa: BLE001 - proof must report any blocker honestly
        payload.update({"status": "BLOCKED", "error": _jsonable_error(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "path": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
