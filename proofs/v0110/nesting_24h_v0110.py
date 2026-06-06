#!/usr/bin/env python3
"""v0.11.0 live nested 24 h validation proof.

Runs the real Canary d01 -> d02 -> d03 hierarchy with live parent-produced
child boundary packages and child subcycling.  Hourly output callbacks score
T2/U10/V10 against the paired CPU-WRF wrfout files.  This is a stability and
measurement proof, not a two-way-feedback or TOST equivalence proof.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace as dataclass_replace
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.coupling.physics_couplers import surface_layer_diagnostics
from gpuwrf.integration.d02_replay import DEFAULT_REPLAY_RUN_DIR, ReplayCase, build_replay_case
from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
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
SCORE_FIELDS = ("T2", "U10", "V10")
FINITE_FIELDS = (
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


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _jsonable_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _git_head() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        subject = subprocess.check_output(["git", "log", "-1", "--pretty=%s"], cwd=root, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        return {"sha": sha, "branch": branch, "subject": subject}
    except Exception:
        return {"sha": None, "branch": None, "subject": None}


def _coerce_run_start(value: str) -> datetime:
    text = str(value).strip().replace("Z", "")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def _make_namelist(
    case: ReplayCase,
    *,
    domain: str,
    parent_dt_s: float | None,
    ratio: int | None,
    run_start: datetime,
) -> OperationalNamelist:
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
        time_utc=run_start,
    )
    if parent_dt_s is not None:
        namelist = with_live_child_boundary_config(
            namelist,
            parent_dt_s=float(parent_dt_s),
            nested_ph_relax=True,
            # Match the v0.11.0 live smoke wiring: w is in the package, but
            # in-loop w relaxation stays deferred to a longer stability gate.
            nested_w_relax=False,
            nested_ph_spec=True,
        )
    if domain == "d01":
        namelist = dataclass_replace(namelist, cu_physics=1)
    return namelist


def _load_domains(
    run_dir: Path,
    max_dom: int,
) -> tuple[DomainHierarchy, dict[str, DomainBundle], dict[str, ReplayCase], dict[str, Any], datetime]:
    names = tuple(f"d{i:02d}" for i in range(1, int(max_dom) + 1))
    cases: dict[str, ReplayCase] = {}
    edges: list[DomainNest] = []
    for name in names:
        parent = _parent_for(name)
        cases[name] = build_replay_case(run_dir, domain=name, boundary_domain=parent)
        if parent is not None:
            edges.append(_nest_from_child(cases[name], parent))

    run_start = _coerce_run_start(str(cases[names[0]].metadata["run_start_label"]))
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
            run_start=run_start,
        )
        bundles[name] = DomainBundle(name=name, state=state, namelist=namelist, grid=case.grid, metrics=case.metrics)
        meta["domains"][name] = {
            "grid": case.metadata.get("grid", {}),
            "boundary_source": case.metadata.get("boundary", {}).get("source"),
            "qke_coldstart": case.metadata.get("qke_coldstart", {}),
            "run_start_label": case.metadata.get("run_start_label"),
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
                "time_utc": str(namelist.time_utc),
            },
        }
    return hierarchy, bundles, cases, meta, run_start


def _field_scalar_stats(value: Any) -> dict[str, Any]:
    arr = jnp.asarray(value)
    finite = jnp.isfinite(arr)
    finite_count, min_value, max_value, mean_value = jax.device_get(
        (
            jnp.sum(finite),
            jnp.nanmin(jnp.where(finite, arr, jnp.nan)),
            jnp.nanmax(jnp.where(finite, arr, jnp.nan)),
            jnp.nanmean(jnp.where(finite, arr, jnp.nan)),
        )
    )
    size = int(arr.size)
    count = int(finite_count)
    all_finite = count == size
    return {
        "shape": [int(item) for item in arr.shape],
        "dtype": str(arr.dtype),
        "finite": bool(all_finite),
        "finite_fraction": float(count / size) if size else 1.0,
        "nonfinite_count": int(size - count),
        "min": float(min_value) if math.isfinite(float(min_value)) else None,
        "max": float(max_value) if math.isfinite(float(max_value)) else None,
        "mean": float(mean_value) if math.isfinite(float(mean_value)) else None,
    }


def _finite_stats(state: Any) -> dict[str, Any]:
    fields = {name: _field_scalar_stats(getattr(state, name)) for name in FINITE_FIELDS}
    return {
        "all_finite": bool(all(item["finite"] for item in fields.values())),
        "fields": fields,
    }


def _score_pair(gpu: np.ndarray, wrf: np.ndarray) -> dict[str, Any]:
    gpu = np.asarray(gpu, dtype=np.float64)
    wrf = np.asarray(wrf, dtype=np.float64)
    if gpu.shape != wrf.shape:
        raise ValueError(f"shape mismatch gpu={gpu.shape} wrf={wrf.shape}")
    finite = np.isfinite(gpu) & np.isfinite(wrf)
    n = int(np.count_nonzero(finite))
    total = int(gpu.size)
    if n == 0:
        return {
            "rmse": None,
            "bias": None,
            "mae": None,
            "gpu_mean": None,
            "wrf_mean": None,
            "gpu_finite": bool(np.isfinite(gpu).all()),
            "wrf_finite": bool(np.isfinite(wrf).all()),
            "finite_pair_fraction": 0.0,
            "n_points": total,
            "n_finite_pairs": 0,
        }
    diff = gpu[finite] - wrf[finite]
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "gpu_mean": float(np.mean(gpu[finite])),
        "wrf_mean": float(np.mean(wrf[finite])),
        "gpu_finite": bool(np.isfinite(gpu).all()),
        "wrf_finite": bool(np.isfinite(wrf).all()),
        "finite_pair_fraction": float(n / total) if total else 1.0,
        "n_points": total,
        "n_finite_pairs": n,
    }


def _surface_fields(state: Any, namelist: OperationalNamelist) -> dict[str, np.ndarray]:
    surf = surface_layer_diagnostics(state, namelist.grid)
    return {
        "T2": np.asarray(jax.device_get(surf.t2), dtype=np.float64),
        "U10": np.asarray(jax.device_get(surf.u10), dtype=np.float64),
        "V10": np.asarray(jax.device_get(surf.v10), dtype=np.float64),
    }


def _reference_fields(run_dir: Path, domain: str, valid_time: datetime) -> tuple[dict[str, np.ndarray], Path]:
    path = run_dir / f"wrfout_{domain}_{valid_time:%Y-%m-%d_%H:%M:%S}"
    payload = read_wrfout_file(path, fields=SCORE_FIELDS)
    fields = {name: np.asarray(payload["fields"][name], dtype=np.float64) for name in SCORE_FIELDS}
    return fields, path


class HourlyScorer:
    def __init__(
        self,
        *,
        output_path: Path,
        payload: dict[str, Any],
        run_dir: Path,
        run_start: datetime,
        bundles: dict[str, DomainBundle],
        output_cadence_steps: dict[str, int],
        write_every: int = 3,
    ) -> None:
        self.output_path = output_path
        self.payload = payload
        self.run_dir = run_dir
        self.run_start = run_start
        self.bundles = bundles
        self.output_cadence_steps = output_cadence_steps
        self.write_every = max(1, int(write_every))
        self.records_seen = 0

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )

    def __call__(self, name: str, own_step: int, state: Any) -> dict[str, Any]:
        cadence = int(self.output_cadence_steps[name])
        lead_h = int(round(int(own_step) / cadence))
        valid_time = self.run_start + timedelta(hours=lead_h)
        t0 = time.perf_counter()
        gpu = _surface_fields(state, self.bundles[name].namelist)
        wrf, wrfout = _reference_fields(self.run_dir, name, valid_time)
        scores = {field: _score_pair(gpu[field], wrf[field]) for field in SCORE_FIELDS}
        finite = _finite_stats(state)
        record = {
            "domain": name,
            "lead_h": int(lead_h),
            "own_step": int(own_step),
            "valid_time_utc": valid_time.isoformat(),
            "cpu_wrfout": str(wrfout),
            "scores": scores,
            "state_finite": finite,
            "boundary_time_levels": {
                "theta_bdy": int(state.theta_bdy.shape[0]),
                "u_bdy": int(state.u_bdy.shape[0]),
                "w_bdy": int(state.w_bdy.shape[0]),
            },
            "score_wall_s": float(time.perf_counter() - t0),
        }
        self.payload["hourly_records"].append(record)
        self.payload["progress"] = {
            "last_domain": name,
            "last_lead_h": int(lead_h),
            "records_written": int(len(self.payload["hourly_records"])),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.records_seen += 1
        if self.records_seen % self.write_every == 0:
            self.write()
        return {
            "domain": name,
            "lead_h": int(lead_h),
            "own_step": int(own_step),
            "all_finite": bool(finite["all_finite"]),
            "rmse": {field: scores[field]["rmse"] for field in SCORE_FIELDS},
        }


def _summarize(records: list[dict[str, Any]], *, domains: tuple[str, ...], hours: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for domain in domains:
        domain_records = [record for record in records if record["domain"] == domain]
        by_field: dict[str, Any] = {}
        for field in SCORE_FIELDS:
            values = [
                float(record["scores"][field]["rmse"])
                for record in domain_records
                if record["scores"][field]["rmse"] is not None
            ]
            final_record = next((record for record in domain_records if int(record["lead_h"]) == int(hours)), None)
            by_field[field] = {
                "lead_count": int(len(values)),
                "mean_rmse": float(np.mean(values)) if values else None,
                "max_rmse": float(np.max(values)) if values else None,
                "final_rmse_24h": (
                    final_record["scores"][field]["rmse"] if final_record is not None else None
                ),
                "final_bias_24h": (
                    final_record["scores"][field]["bias"] if final_record is not None else None
                ),
                "all_gpu_finite": bool(all(record["scores"][field]["gpu_finite"] for record in domain_records)),
            }
        expected_leads = set(range(1, int(hours) + 1))
        observed_leads = {int(record["lead_h"]) for record in domain_records}
        summary[domain] = {
            "record_count": int(len(domain_records)),
            "expected_record_count": int(hours),
            "missing_leads_h": sorted(expected_leads - observed_leads),
            "all_hourly_state_finite": bool(
                domain_records and all(record["state_finite"]["all_finite"] for record in domain_records)
            ),
            "fields": by_field,
        }
    return summary


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("rmse_summary", {})
    lines = [
        "# v0.11.0 live nested 24 h validation",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Verdict: `{payload.get('verdict')}`",
        f"- Run: `{payload.get('run_dir')}`",
        f"- Git: `{payload.get('git', {}).get('sha')}`",
        f"- Live nesting: one-way d01->d02->d03, parent boundary packages generated from live parent state, child subcycling enabled.",
        f"- Feedback: `{payload.get('feedback')}`.",
        "",
        "## 24 h RMSE vs CPU-WRF",
        "",
        "| domain | T2 RMSE K | U10 RMSE m/s | V10 RMSE m/s | hourly finite | missing leads |",
        "|---|---:|---:|---:|---|---|",
    ]
    for domain in payload.get("domains", []):
        item = summary.get(domain, {})
        fields = item.get("fields", {})

        def val(field: str) -> str:
            value = fields.get(field, {}).get("final_rmse_24h")
            return "n/a" if value is None else f"{float(value):.6g}"

        missing = item.get("missing_leads_h", [])
        lines.append(
            f"| {domain} | {val('T2')} | {val('U10')} | {val('V10')} | "
            f"{item.get('all_hourly_state_finite')} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Proof Boundary",
            "",
            "- Proven: a 24 h live one-way nested d01->d02->d03 run completed, emitted paired hourly scores, and recorded finite state checks at the output cadence.",
            "- Not proven: two-way feedback, TOST/ensemble equivalence, profiler/transfer claims, longer horizons, or the separate KI-1 d03 1 km gate.",
            "- Replay-nest baseline was not rerun in this proof; CPU-WRF wrfout files are the paired truth source.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_REPLAY_RUN_DIR)
    parser.add_argument("--max-dom", type=int, default=3, choices=(2, 3))
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("proofs/v0110/nesting_24h_v0110.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("proofs/v0110/val_nest24h.md"))
    parser.add_argument("--block-between", action="store_true")
    args = parser.parse_args()

    if int(args.hours) <= 0:
        raise ValueError("--hours must be positive")
    root_steps_raw = float(args.hours) * 3600.0 / DT_BY_DOMAIN["d01"]
    root_steps = int(round(root_steps_raw))
    if abs(root_steps_raw - root_steps) > 1.0e-9:
        raise ValueError(f"hours={args.hours} does not align with d01 dt={DT_BY_DOMAIN['d01']}")

    domains = tuple(f"d{i:02d}" for i in range(1, int(args.max_dom) + 1))
    payload: dict[str, Any] = {
        "proof": "v0110 live nested 24h validation",
        "schema_version": 1,
        "status": "RUNNING",
        "verdict": "RUNNING",
        "git": _git_head(),
        "run_dir": str(args.run_dir),
        "domains": list(domains),
        "hours": int(args.hours),
        "root_steps": int(root_steps),
        "feedback": "off",
        "scored_fields": list(SCORE_FIELDS),
        "hourly_records": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "carry_overs": [
            "Two-way feedback is not enabled in this proof.",
            "In-loop nested w relaxation remains off, matching the v0.11.0 smoke gate.",
            "No profiler or host/device transfer audit is claimed here.",
            "No TOST/ensemble equivalence gate is claimed from a single 24 h case.",
            "The separate KI-1 d03 1 km validation gate is not closed by this proof.",
        ],
    }

    t0 = time.perf_counter()
    try:
        hierarchy, bundles, _cases, meta, run_start = _load_domains(args.run_dir, int(args.max_dom))
        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
        output_cadence = {
            name: int(round(3600.0 / DT_BY_DOMAIN[name]))
            for name in domains
        }
        expected_counts = hierarchy.expected_step_counts(root_steps=root_steps)
        payload.update(
            {
                "run_start_utc": run_start.isoformat(),
                "metadata": meta,
                "hierarchy": {
                    "order": list(hierarchy.order),
                    "edges": [edge.__dict__ for edge in hierarchy.nests],
                    "expected_own_steps": expected_counts,
                    "output_cadence_steps": output_cadence,
                    "persistent_state_bytes": tree.persistent_state_bytes(),
                },
            }
        )
        scorer = HourlyScorer(
            output_path=args.output,
            payload=payload,
            run_dir=args.run_dir,
            run_start=run_start,
            bundles=bundles,
            output_cadence_steps=output_cadence,
            write_every=len(domains),
        )
        scorer.write()
        result = run_operational_domain_tree(
            tree,
            root_steps=root_steps,
            feedback_enabled=False,
            output=scorer,
            output_cadence_steps=output_cadence,
            block_between=bool(args.block_between),
        )
        jax.block_until_ready(tuple(state.theta for state in result.states.values()))
        event_counts = Counter(event[0] for event in result.events)
        force_counts = Counter(
            f"{event[1]}->{event[2]}"
            for event in result.events
            if event and event[0] == "force"
        )
        final_finite = {name: _finite_stats(state) for name, state in result.states.items()}
        rmse_summary = _summarize(payload["hourly_records"], domains=domains, hours=int(args.hours))
        expected_records = int(args.hours) * len(domains)
        all_records_present = len(payload["hourly_records"]) == expected_records and all(
            not rmse_summary[name]["missing_leads_h"] for name in domains
        )
        all_finite = bool(
            all(item["all_finite"] for item in final_finite.values())
            and all(item["all_hourly_state_finite"] for item in rmse_summary.values())
        )
        status = "PASS" if all_records_present and all_finite else "FAIL"
        verdict = (
            "LIVE_NESTED_24H_FINITE_RMSE_RECORDED"
            if status == "PASS"
            else "LIVE_NESTED_24H_VALIDATION_ISSUE"
        )
        payload.update(
            {
                "status": status,
                "verdict": verdict,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "wall_s": float(time.perf_counter() - t0),
                "hierarchy": {
                    **payload["hierarchy"],
                    "observed_own_steps": result.own_steps,
                    "event_counts": dict(event_counts),
                    "force_counts": dict(force_counts),
                    "output_count": int(len(result.outputs)),
                    "first_outputs": list(result.outputs[:6]),
                    "last_outputs": list(result.outputs[-6:]),
                },
                "final_finite": final_finite,
                "rmse_summary": rmse_summary,
                "acceptance_checks": {
                    "expected_hourly_records": expected_records,
                    "observed_hourly_records": int(len(payload["hourly_records"])),
                    "all_records_present": bool(all_records_present),
                    "final_and_hourly_states_finite": bool(all_finite),
                    "paired_cpu_wrfout_truth": True,
                },
            }
        )
        scorer.write()
        _write_markdown(args.markdown_output, payload)
        print(json.dumps({"status": status, "verdict": verdict, "path": str(args.output)}))
        return 0 if status == "PASS" else 1
    except BaseException as exc:  # noqa: BLE001 - proof must report blockers honestly
        payload.update(
            {
                "status": "BLOCKED",
                "verdict": "BLOCKED",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "wall_s": float(time.perf_counter() - t0),
                "error": _jsonable_error(exc),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        _write_markdown(args.markdown_output, payload)
        print(json.dumps({"status": "BLOCKED", "path": str(args.output), "error": payload["error"]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
