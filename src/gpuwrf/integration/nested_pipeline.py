"""v0.12.0 standalone LIVE-NESTED forecast driver.

This is the multi-domain analogue of :mod:`gpuwrf.integration.daily_pipeline`:
it runs a ``d01 -> d02 -> ... -> dN`` nest OUT-OF-THE-BOX from ``real.exe``
outputs (``wrfinput_d0N`` + ``wrfbdy_d01``) with **no CPU-WRF wrfout
dependency**.  It is a *thin* composition layer -- it does not implement physics,
dynamics, nesting interpolation, or the boundary construction; those live in the
already-validated runtime:

  * the per-domain initial states load through
    :func:`gpuwrf.integration.d02_replay.build_replay_case` (the same loader the
    single-domain standalone CLI uses);
  * the root domain (``d01``) takes its lateral boundary forcing from
    ``wrfbdy_d01`` (decoded directly, no wrfout history);
  * each child loads its IC from ``wrfinput_d0N`` with ``*_bdy`` leaves left at
    their ``State.zeros`` shapes -- the LIVE parent constructs the child boundary
    package every parent timestep
    (:func:`gpuwrf.nesting.boundary_construction.build_child_boundary_package`);
  * the device runtime is the VALIDATED
    :func:`gpuwrf.runtime.domain_tree.run_operational_domain_tree` that drove the
    v0.11.0 24 h ``d01 -> d02 -> d03`` nesting proof.

The driver writes one ``wrfout_<domain>_<valid_time>`` per domain at the hourly
output cadence and returns a JSON-serializable payload mirroring the daily
pipeline's ``M7DailyPipelineRun`` shape (so the CLI can print/branch on it
uniformly).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.io.radiation_static import load_radiation_static
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    run_operational_domain_tree,
    with_live_child_boundary_config,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist


__all__ = [
    "NestedPipelineConfig",
    "execute_nested_pipeline",
    "domain_names_for",
]

# Half-hour radiation update target (radt = dt_s * radiation_cadence_steps == 1800 s),
# matching the daily pipeline / v0.11.0 nesting proof radiation cadence selection.
_RADT_TARGET_S = 1800.0


@dataclass(frozen=True)
class NestedPipelineConfig:
    """Inputs for one standalone live-nested forecast."""

    input_dir: Path
    output_dir: Path
    proof_dir: Path
    hours: int
    max_dom: int
    scratch_dir: Path | None = None


def domain_names_for(max_dom: int) -> tuple[str, ...]:
    """``("d01", "d02", ...)`` for ``max_dom`` domains."""

    if int(max_dom) < 1:
        raise ValueError(f"max_dom must be >= 1, got {max_dom}")
    return tuple(f"d{i:02d}" for i in range(1, int(max_dom) + 1))


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


def _dt_by_domain(run, names: tuple[str, ...]) -> dict[str, float]:
    """Per-domain model timestep from the namelist (root time_step / ratio chain).

    WRF sets the child timestep from ``parent_time_step / parent_grid_ratio``; the
    Canary nests use a fixed integer ratio so the result is exact.
    """

    nml = run.namelist
    root_dt = nml.get("domains", {}).get("time_step")
    if root_dt is None:
        root_dt = nml.get("time_control", {}).get("time_step")
    if root_dt is None:
        raise ValueError("namelist has no domains/time_control time_step for the root domain")
    dt: dict[str, float] = {names[0]: float(root_dt)}
    for name in names[1:]:
        grid = run.grid(name)
        parent = f"d{int(grid.parent_id):02d}"
        ratio = int(grid.parent_grid_ratio)
        if ratio <= 1:
            raise ValueError(f"{name}: parent_grid_ratio must be > 1 for a child, got {ratio}")
        if parent not in dt:
            raise ValueError(f"{name}: parent {parent} not loaded before child (bad domain order)")
        dt[name] = dt[parent] / float(ratio)
    return dt


def _radiation_cadence_steps(dt_s: float) -> int:
    return max(1, int(round(_RADT_TARGET_S / float(dt_s))))


def _make_namelist(
    *,
    grid,
    tendencies,
    metrics,
    dt_s: float,
    parent_dt_s: float | None,
    run_start: datetime,
    radiation_static: Any | None,
    cu_physics: int,
) -> OperationalNamelist:
    """Per-domain operational namelist (mirrors the v0.11.0 nesting proof config).

    The dynamics knobs (flux advection, fp64 acoustic solve, 6th-order filter,
    Rayleigh + w damping, rigid lid) are the F7-closed operational settings the
    real-case path uses (see ``daily_pipeline._build_real_case``).  Children get
    the WRF live-nest boundary cadence (``update_cadence_s == parent_dt``) so the
    parent-built two-time package interpolates exactly across the subcycle.
    """

    namelist = OperationalNamelist.from_grid(
        grid,
        tendencies=tendencies,
        metrics=metrics,
        dt_s=float(dt_s),
        acoustic_substeps=10,
        radiation_cadence_steps=_radiation_cadence_steps(dt_s),
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
        radiation_static=radiation_static,
        time_utc=run_start,
    )
    if parent_dt_s is not None:
        namelist = with_live_child_boundary_config(
            namelist,
            parent_dt_s=float(parent_dt_s),
            nested_ph_relax=True,
            # Match the v0.11.0 validated wiring: w is in the package, but in-loop
            # w relaxation stays deferred to a longer stability gate.
            nested_w_relax=False,
            nested_ph_spec=True,
        )
    namelist = dataclass_replace(namelist, cu_physics=int(cu_physics))
    return namelist


def _domain_cu_physics(run, domain: str) -> int:
    """Per-domain ``cu_physics`` from the namelist (cumulus normally off on fine nests)."""

    raw = run.namelist.get("physics", {}).get("cu_physics", 0)
    if isinstance(raw, (list, tuple)):
        index = max(int(domain[1:]) - 1, 0)
        if index < len(raw):
            return int(raw[index])
        return int(raw[-1]) if raw else 0
    return int(raw)


def _nest_edge(run, child: str, parent: str) -> DomainNest:
    grid = run.grid(child)
    return DomainNest(
        parent=parent,
        child=child,
        parent_grid_ratio=int(grid.parent_grid_ratio),
        i_parent_start=int(grid.i_parent_start),
        j_parent_start=int(grid.j_parent_start),
        feedback=False,
    )


def _load_domains(
    config: NestedPipelineConfig,
    names: tuple[str, ...],
) -> tuple[DomainHierarchy, dict[str, DomainBundle], dict[str, Any], datetime, dict[str, float]]:
    """Load every domain standalone: d01 LBC from wrfbdy, children IC-only (live LBC)."""

    run_dir = Path(config.input_dir)
    # Build the root case first so we share its Gen2Run for namelist/grid metadata.
    root_case = build_replay_case(run_dir, domain=names[0], standalone=True)
    run = root_case.run
    dt_by_domain = _dt_by_domain(run, names)

    run_start = _coerce_run_start(str(root_case.metadata["run_start_label"]))
    edges: list[DomainNest] = []
    for name in names[1:]:
        grid = run.grid(name)
        parent = f"d{int(grid.parent_id):02d}"
        edges.append(_nest_edge(run, name, parent))
    hierarchy = DomainHierarchy.from_edges(names, tuple(edges), max_dom=max(5, len(names)))

    bundles: dict[str, DomainBundle] = {}
    meta: dict[str, Any] = {"domains": {}, "edges": [edge.__dict__ for edge in edges]}

    for name in names:
        if name == names[0]:
            case = root_case
            parent_dt = None
        else:
            # Standalone live-nested CHILD: IC from wrfinput_<child>, NO lateral
            # forcing read from disk (no wrfbdy_<child> / wrfout_<child>); the live
            # parent supplies the boundary package each parent step.
            case = build_replay_case(run_dir, domain=name, load_lateral_boundaries=False)
            parent = hierarchy.parent(name)
            parent_dt = dt_by_domain[parent]

        radiation_static = None
        try:
            radiation_static, _ = load_radiation_static(
                case.run, name, grid=case.grid, metrics=case.metrics
            )
        except Exception:  # noqa: BLE001 -- radiation static is best-effort; never block init.
            radiation_static = None

        # Seed the transitional legacy aliases (p/ph/mu) from the authoritative totals,
        # matching the single-domain operational path.
        state = case.state.replace(
            p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total
        )
        namelist = _make_namelist(
            grid=case.grid,
            tendencies=case.tendencies,
            metrics=case.metrics,
            dt_s=dt_by_domain[name],
            parent_dt_s=parent_dt,
            run_start=run_start,
            radiation_static=radiation_static,
            cu_physics=_domain_cu_physics(run, name),
        )
        bundles[name] = DomainBundle(
            name=name, state=state, namelist=namelist, grid=case.grid, metrics=case.metrics
        )
        meta["domains"][name] = {
            "ic_source": f"wrfinput_{name}",
            "standalone_native_init": bool(case.metadata.get("standalone_native_init", True)),
            "lbc_source": (
                case.metadata.get("boundary", {}).get("source")
                if name == names[0]
                else "live parent boundary package (build_child_boundary_package)"
            ),
            "wrfbdy_path": case.metadata.get("boundary", {}).get("wrfbdy_path"),
            "qke_coldstart": case.metadata.get("qke_coldstart", {}),
            "grid": case.metadata.get("grid", {}),
            "namelist": {
                "dt_s": float(namelist.dt_s),
                "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
                "boundary_update_cadence_s": float(namelist.boundary_config.update_cadence_s),
                "cu_physics": int(namelist.cu_physics),
                "radiation_static_loaded": radiation_static is not None,
            },
        }
    return hierarchy, bundles, meta, run_start, dt_by_domain


class _PerDomainWrfoutWriter:
    """Output callback: write one wrfout per domain at the hourly cadence."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_start: datetime,
        bundles: dict[str, DomainBundle],
        output_cadence_steps: dict[str, int],
    ) -> None:
        self.output_dir = output_dir
        self.run_start = run_start
        self.bundles = bundles
        self.output_cadence_steps = output_cadence_steps
        self.written: dict[str, list[str]] = {name: [] for name in bundles}
        # Lazy imports kept off the module top-level so importing this module stays
        # light for non-GPU callers (mirrors daily_pipeline).
        from gpuwrf.integration.daily_pipeline import (
            _surface_diagnostics_for_output,
            finite_summary,
        )

        self._surface_diagnostics_for_output = _surface_diagnostics_for_output
        self._finite_summary = finite_summary

    def __call__(self, name: str, own_step: int, state: Any) -> dict[str, Any]:
        cadence = int(self.output_cadence_steps[name])
        lead_h = int(round(int(own_step) / cadence))
        valid_time = self.run_start + timedelta(hours=lead_h)
        namelist = self.bundles[name].namelist
        grid = self.bundles[name].grid
        diagnostics = self._surface_diagnostics_for_output(
            state, namelist, self.run_start, lead_seconds=float(lead_h) * 3600.0
        )
        path = self.output_dir / f"wrfout_{name}_{valid_time:%Y-%m-%d_%H:%M:%S}"
        write_wrfout_netcdf(
            state,
            grid,
            namelist,
            path,
            valid_time=valid_time,
            lead_hours=float(lead_h),
            run_start=self.run_start,
            diagnostics=diagnostics,
        )
        self.written[name].append(str(path))
        summary = self._finite_summary(state)
        return {
            "domain": name,
            "lead_h": int(lead_h),
            "own_step": int(own_step),
            "all_finite": bool(summary["all_finite"]),
            "wrfout": str(path),
        }


def _finite_stats_host(state: Any) -> dict[str, Any]:
    from gpuwrf.integration.daily_pipeline import finite_summary

    return finite_summary(state)


def execute_nested_pipeline(config: NestedPipelineConfig) -> dict[str, Any]:
    """Run a standalone live-nested forecast and write per-domain wrfout.

    Returns an ``M7DailyPipelineRun``-shaped payload with ``init_mode``
    ``standalone_native_init_nested`` and a per-domain finite/output summary.
    """

    import jax  # local import keeps module import light for --help / arg parsing.

    from gpuwrf.profiling.transfer_audit import visible_gpu_name

    names = domain_names_for(config.max_dom)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.proof_dir.mkdir(parents=True, exist_ok=True)

    if int(config.hours) <= 0:
        raise ValueError("hours must be positive")

    overall_start = time.perf_counter()
    hierarchy, bundles, meta, run_start, dt_by_domain = _load_domains(config, names)

    root = names[0]
    root_dt = dt_by_domain[root]
    root_steps_raw = float(config.hours) * 3600.0 / root_dt
    root_steps = int(round(root_steps_raw))
    if abs(root_steps_raw - root_steps) > 1.0e-9:
        raise ValueError(
            f"hours={config.hours} does not align with root dt={root_dt}s "
            f"(would need {root_steps_raw} steps)"
        )
    output_cadence = {name: int(round(3600.0 / dt_by_domain[name])) for name in names}

    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    writer = _PerDomainWrfoutWriter(
        output_dir=config.output_dir,
        run_start=run_start,
        bundles=bundles,
        output_cadence_steps=output_cadence,
    )

    forecast_start = time.perf_counter()
    result = run_operational_domain_tree(
        tree,
        root_steps=root_steps,
        feedback_enabled=False,
        output=writer,
        output_cadence_steps=output_cadence,
        block_between=True,
    )
    jax.block_until_ready(tuple(state.theta for state in result.states.values()))
    forecast_wall_s = time.perf_counter() - forecast_start

    final_finite = {name: _finite_stats_host(state) for name, state in result.states.items()}
    event_counts = Counter(event[0] for event in result.events)
    force_counts = Counter(
        f"{event[1]}->{event[2]}" for event in result.events if event and event[0] == "force"
    )

    per_domain: dict[str, Any] = {}
    all_finite = True
    all_output_present = True
    expected_outputs_per_domain = int(config.hours)
    for name in names:
        outputs = writer.written.get(name, [])
        finite_ok = bool(final_finite[name]["all_finite"])
        output_ok = len(outputs) == expected_outputs_per_domain
        all_finite = all_finite and finite_ok
        all_output_present = all_output_present and output_ok
        per_domain[name] = {
            "final_state_finite": finite_ok,
            "wrfout_count": len(outputs),
            "expected_wrfout_count": expected_outputs_per_domain,
            "wrfout_files": outputs,
            "dt_s": float(dt_by_domain[name]),
            "own_steps": int(result.own_steps.get(name, 0)),
        }

    total_wall_s = time.perf_counter() - overall_start
    verdict = "PIPELINE_GREEN" if (all_finite and all_output_present) else "PIPELINE_PARTIAL"

    payload: dict[str, Any] = {
        "schema": "M7DailyPipelineRun",
        "schema_version": 1,
        "verdict": verdict,
        "init_mode": "standalone_native_init_nested",
        "run_id": str(Path(config.input_dir).resolve()),
        "input_dir": str(Path(config.input_dir).resolve()),
        "output_dir": str(config.output_dir),
        "max_dom": int(config.max_dom),
        "domains": list(names),
        "hours": int(config.hours),
        "root_steps": int(root_steps),
        "device": visible_gpu_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "run_start_utc": run_start.isoformat(),
        "wall_clock_total_s": float(total_wall_s),
        "wall_clock_forecast_only_s": float(forecast_wall_s),
        "wrfout_files": [path for name in names for path in writer.written.get(name, [])],
        "per_domain": per_domain,
        "all_domains_finite": bool(all_finite),
        "all_outputs_present": bool(all_output_present),
        "hierarchy": {
            "order": list(hierarchy.order),
            "edges": [edge.__dict__ for edge in hierarchy.nests],
            "observed_own_steps": dict(result.own_steps),
            "output_cadence_steps": output_cadence,
            "event_counts": dict(event_counts),
            "force_counts": dict(force_counts),
            "persistent_state_bytes": tree.persistent_state_bytes(),
        },
        "metadata": meta,
        "carry_overs": [
            "Two-way feedback is OFF (one-way nesting); matches the v0.11.0 validated wiring.",
            "In-loop nested w relaxation is OFF (deferred to a longer stability gate).",
            "No TOST/ensemble equivalence or CPU-speedup baseline is claimed by a standalone smoke.",
        ],
    }
    return payload
