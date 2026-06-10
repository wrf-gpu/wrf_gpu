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

import calendar
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
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.io.radiation_static import load_radiation_static
from gpuwrf.io.gwdo_static import load_gwdo_statics
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    DomainTreeResult,
    run_operational_domain_tree,
    with_live_child_boundary_config,
)
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _initial_carry_for_run,
    noahmp_initial_rad,
)


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
    # Two-way nesting: when True, after each child completes its parent_grid_ratio
    # subcycle its interior is fed back onto the overlapping parent cells (WRF
    # copy_fcn area-average) followed by the WRF sm121 feedback-zone smoother.
    # Defaults to False to preserve the v0.11.0/v0.12.0-validated one-way wiring;
    # opt in for the two-way path.
    feedback: bool = False


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
    gwd_opt: int = 0,
    gwdo_statics: Any | None = None,
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
        gwd_opt=int(gwd_opt),
        gwdo_statics=gwdo_statics,
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


def _root_boundary_cadence_override(
    namelist: OperationalNamelist, case_metadata: dict[str, Any]
) -> OperationalNamelist:
    """Match the root-domain LBC interpolation cadence to its wrfbdy interval.

    The standalone root's ``*_bdy`` leaves carry ONE time level per wrfbdy
    forcing interval (``interval_seconds``, e.g. 21600 s for 6-hourly AIFS/GFS),
    NOT one per hour. ``interpolate_boundary_leaf`` walks the leaf time axis at
    ``boundary_config.update_cadence_s``; leaving the hourly replay default
    (3600 s) makes the run consume the wrfbdy levels 6x too fast and then clamp
    frozen on the last level (proofs/v014/lbc_cadence_root_cause: the v0.14
    Canary 72h PSFC/MU/P/PH drift). WRF advances each interval linearly with the
    ``_BT*`` tendency over bdyfrq == interval_seconds; linear interpolation
    between consecutive level values at that same cadence is the identical
    forcing.
    """

    interval_s = (case_metadata.get("boundary") or {}).get("interval_seconds")
    if not interval_s:
        return namelist
    return dataclass_replace(
        namelist,
        boundary_config=dataclass_replace(
            namelist.boundary_config, update_cadence_s=float(interval_s)
        ),
    )


def _domain_int(run, group: str, key: str, domain: str, default: int = 0) -> int:
    """Per-domain integer namelist value from ``group`` (max-dom list or scalar)."""

    raw = run.namelist.get(group, {}).get(key, default)
    if isinstance(raw, (list, tuple)):
        index = max(int(domain[1:]) - 1, 0)
        if index < len(raw):
            return int(raw[index])
        return int(raw[-1]) if raw else int(default)
    return int(raw)


def _domain_physics_int(run, key: str, domain: str, default: int = 0) -> int:
    """Per-domain integer ``&physics`` namelist value (max-dom list or scalar)."""

    return _domain_int(run, "physics", key, domain, default)


def _domain_gwd_opt(run, domain: str) -> int:
    """Per-domain ``gwd_opt`` (WRF &dynamics control; &physics fallback)."""

    value = _domain_int(run, "dynamics", "gwd_opt", domain, 0)
    if value == 0:
        value = _domain_int(run, "physics", "gwd_opt", domain, 0)
    return value


def _domain_cu_physics(run, domain: str) -> int:
    """Per-domain ``cu_physics`` from the namelist (cumulus normally off on fine nests)."""

    return _domain_physics_int(run, "cu_physics", domain, 0)


# Land-surface options this standalone nested pipeline can wire: 4 = Noah-MP
# (the prognostic land path CPU truth runs) and 0 = no LSM selected (legacy
# prescribed bulk surface). Anything else fails closed -- silently falling back
# to the bulk path freezes land TSK for the whole run on every domain
# (proofs/v014/canary_h24_residual_adjudication.md, the v0.14 release blocker).
_SUPPORTED_NESTED_LAND_OPTIONS = (0, 4)


def _domain_sf_surface_physics(run, domain: str) -> int:
    """Per-domain ``sf_surface_physics``; fail closed on unsupported land options."""

    option = _domain_physics_int(run, "sf_surface_physics", domain, 0)
    if option not in _SUPPORTED_NESTED_LAND_OPTIONS:
        raise ValueError(
            f"{domain}: sf_surface_physics={option} is not wired in the standalone "
            "nested pipeline (supported: 0 = prescribed bulk surface, 4 = Noah-MP). "
            "Refusing the silent bulk-surface fallback: it leaves land TSK frozen "
            "for the whole run (proofs/v014/canary_h24_residual_adjudication.md)."
        )
    return option


def _wrf_julian_yearlen(run_start: datetime) -> tuple[float, float]:
    """WRF Noah-MP clock ``(julian, yearlen)`` at the run start.

    WRF ``grid%julian`` is the 0-based FRACTIONAL day-of-year: ESMF
    ``dayOfYear_r8 - 1.0`` (frame/module_domain.F:2165), NOT ``tm_yday``
    (proofs/v014/noahmp_step1_closure.md). ``yearlen`` honours leap years.
    """

    julian = float(run_start.timetuple().tm_yday - 1) + (
        run_start.hour * 3600.0 + run_start.minute * 60.0 + run_start.second
    ) / 86400.0
    yearlen = 366.0 if calendar.isleap(run_start.year) else 365.0
    return julian, yearlen


def _nest_edge(run, child: str, parent: str, *, feedback: bool = False) -> DomainNest:
    grid = run.grid(child)
    return DomainNest(
        parent=parent,
        child=child,
        parent_grid_ratio=int(grid.parent_grid_ratio),
        i_parent_start=int(grid.i_parent_start),
        j_parent_start=int(grid.j_parent_start),
        feedback=bool(feedback),
    )


def _load_domains(
    config: NestedPipelineConfig,
    names: tuple[str, ...],
) -> tuple[
    DomainHierarchy,
    dict[str, DomainBundle],
    dict[str, Any],
    datetime,
    dict[str, float],
    dict[str, Any],
]:
    """Load every domain standalone: d01 LBC from wrfbdy, children IC-only (live LBC).

    Also returns the per-domain INITIAL ``OperationalCarry`` dict.  The carries are
    built here (with the same ``_initial_carry_for_run`` the domain-tree cold start
    uses, so the non-Noah-MP path is bit-identical) because the Noah-MP land carry
    must be seeded BEFORE the first ``_advance_chunk`` scan: the carry pytree
    structure is frozen across scan iterations, so a ``None -> NoahMPLandState``
    promotion inside the run is impossible by construction.
    """

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
        edges.append(_nest_edge(run, name, parent, feedback=bool(config.feedback)))
    hierarchy = DomainHierarchy.from_edges(names, tuple(edges), max_dom=max(5, len(names)))

    bundles: dict[str, DomainBundle] = {}
    initial_carries: dict[str, Any] = {}
    loaded_cases: dict[str, Any] = {names[0]: root_case}
    meta: dict[str, Any] = {"domains": {}, "edges": [edge.__dict__ for edge in edges]}

    for name in names:
        if name == names[0]:
            case = root_case
            parent_dt = None
        else:
            # Standalone live-nested CHILD: IC from wrfinput_<child>, NO lateral
            # forcing read from disk (no wrfbdy_<child> / wrfout_<child>); the live
            # parent supplies the boundary package each parent step.  The parent
            # case is passed explicitly so build_replay_case can reproduce WRF's
            # live-nest terrain/base initialization before timestep ownership.
            parent = hierarchy.parent(name)
            if parent is None or parent not in loaded_cases:
                raise ValueError(f"{name}: parent case must be loaded before live-nest child init")
            case = build_replay_case(
                run_dir,
                domain=name,
                load_lateral_boundaries=False,
                live_nest_parent=loaded_cases[parent],
            )
            parent_dt = dt_by_domain[parent]
        loaded_cases[name] = case

        radiation_static = None
        try:
            radiation_static, _ = load_radiation_static(
                case.run, name, grid=case.grid, metrics=case.metrics
            )
        except Exception:  # noqa: BLE001 -- radiation static is best-effort; never block init.
            radiation_static = None

        # Orographic gravity-wave drag per nested domain: read this domain's
        # &physics gwd_opt and, when on, build its GWDOStatics from the geo_em
        # sub-grid orography.  Fails closed to gwd_opt=0 if the statics are
        # absent (no fabricated drag), mirroring the single-domain path.
        gwd_opt = _domain_gwd_opt(run, name)
        gwdo_statics = None
        # v0.13: GWD operational coupling is ON BY DEFAULT on the nested path. v0.12.0
        # gated it off (24h nested-1km + GWD OOM'd at ~sim-hr 7); v0.13's RRTMG g-point
        # + optics/taumol VRAM chunking (SW -88.6% / LW -43.6%) made the 24h nested-1km
        # + GWD run FIT and pass GREEN (proofs/v013/gwd_nested_24h_gate.json: 24/24
        # wrfout, all-finite). Kernel oracle-validated. Honour gwd_opt=1; set
        # GPUWRF_GWD_NESTED=0 to force it off for a memory-tighter config.
        if gwd_opt == 1 and os.environ.get("GPUWRF_GWD_NESTED", "1") == "0":
            gwd_opt = 0
        if gwd_opt == 1:
            try:
                gwdo_statics, _ = load_gwdo_statics(
                    case.run, name, grid=case.grid, metrics=case.metrics
                )
            except Exception:  # noqa: BLE001 -- GWD is opt-in; never block init.
                gwdo_statics = None
            if gwdo_statics is None:
                gwd_opt = 0

        # Land surface per nested domain (v0.14 release-blocker fix): read this
        # domain's &physics sf_surface_physics and, when 4, wire the SAME
        # prognostic Noah-MP coupler the single-domain/TOST drivers run
        # (proofs/noahmp/s6b_activate_validate.py, proofs/m20/tost_noahmp_runner.py).
        # Before this, the nested namelist never set use_noahmp, so the land tile
        # stayed on the prescribed bulk path and land TSK was FROZEN for the whole
        # run on every domain (proofs/v014/canary_h24_residual_adjudication.md).
        # Unsupported land options fail closed in _domain_sf_surface_physics.
        sf_surface_physics = _domain_sf_surface_physics(run, name)
        noahmp_land = None
        noahmp_init_meta = None
        if sf_surface_physics == 4:
            noahmp_land, noahmp_static, noahmp_init_meta = build_noahmp_land_state(
                run_dir, name
            )
            noahmp_energy_params, noahmp_rad_params, noahmp_nroot = build_noahmp_params(
                noahmp_static
            )
            noahmp_julian, noahmp_yearlen = _wrf_julian_yearlen(run_start)

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
            gwd_opt=gwd_opt,
            gwdo_statics=gwdo_statics,
        )
        if noahmp_land is not None:
            namelist = dataclass_replace(
                namelist,
                use_noahmp=True,
                sf_surface_physics=4,
                noahmp_static=noahmp_static,
                noahmp_energy_params=noahmp_energy_params,
                noahmp_rad_params=noahmp_rad_params,
                noahmp_nroot=noahmp_nroot,
                noahmp_julian=noahmp_julian,
                noahmp_yearlen=noahmp_yearlen,
            )
        if name == names[0]:
            namelist = _root_boundary_cadence_override(namelist, case.metadata)
        # Initial carry: identical to the domain-tree cold start for the bulk path
        # (same _initial_carry_for_run on the same state/namelist); under Noah-MP the
        # prognostic land carry plus the REAL t=0 held surface radiation are seeded
        # NOW so the scan carry pytree is structurally stable from step 1 (mirrors
        # the proven s6b/TOST carry seeding; nocturnal LWDN cold-start mitigation).
        carry = _initial_carry_for_run(state, namelist)
        if noahmp_land is not None:
            carry = carry.replace(
                noahmp_land=noahmp_land,
                noahmp_rad=noahmp_initial_rad(carry.state, namelist, land_state=noahmp_land),
            )
        initial_carries[name] = carry
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
            "live_nest_base_init": case.metadata.get("live_nest_base_init", {}),
            "grid": case.metadata.get("grid", {}),
            "namelist": {
                "dt_s": float(namelist.dt_s),
                "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
                "boundary_update_cadence_s": float(namelist.boundary_config.update_cadence_s),
                "cu_physics": int(namelist.cu_physics),
                "radiation_static_loaded": radiation_static is not None,
                "gwd_opt": int(namelist.gwd_opt),
                "gwdo_statics_loaded": namelist.gwdo_statics is not None,
            },
            "land_surface": {
                "sf_surface_physics": int(sf_surface_physics),
                "use_noahmp": bool(namelist.use_noahmp),
                "noahmp_static_loaded": namelist.noahmp_static is not None,
                "noahmp_energy_params_loaded": namelist.noahmp_energy_params is not None,
                "noahmp_rad_params_loaded": namelist.noahmp_rad_params is not None,
                "noahmp_land_seeded": noahmp_land is not None,
                "noahmp_n_land_cells": (
                    int(noahmp_init_meta["n_land_cells"]) if noahmp_init_meta else None
                ),
                "noahmp_julian": float(namelist.noahmp_julian),
                "noahmp_yearlen": float(namelist.noahmp_yearlen),
                "provenance": (
                    noahmp_init_meta.get("wrfinput_file") if noahmp_init_meta else None
                ),
            },
        }
    return hierarchy, bundles, meta, run_start, dt_by_domain, initial_carries


def _noahmp_surface_diagnostics_for_output(
    state: Any,
    namelist: OperationalNamelist,
    run_start: datetime,
    *,
    lead_seconds: float,
    noahmp_land: Any,
    noahmp_rad: Any,
) -> dict[str, np.ndarray] | None:
    """Writer surface map with the ACTIVE Noah-MP carry threaded into the overlay.

    Mirrors ``daily_pipeline._surface_diagnostics_for_output`` but passes the
    EVOLVED ``noahmp_land``/``noahmp_rad`` to ``compute_m9_diagnostics`` so the
    land HFX/LH/TSK and the LSM 2-m T2 come from the prognostic Noah-MP overlay
    and SWDOWN/GLW report the held WRF-cadence radiation (the L1 COSZEN-phase
    fix).  Deliberately NOT best-effort: a wiring error in the active Noah-MP
    output path must fail at the first hourly output, not silently degrade to
    the writer's raw lowest-level fallbacks (which would resurrect the frozen
    land-surface record this sprint removes).
    """

    import jax  # noqa: PLC0415 -- lazy: keeps module import light (mirrors writer).

    from gpuwrf.integration.daily_pipeline import _M9_OUTPUT_FIELDS  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        compute_m9_diagnostics,
        surface_layer_diagnostics,
    )

    clock_namelist = namelist
    if getattr(namelist, "time_utc", None) is None:
        clock_namelist = dataclass_replace(namelist, time_utc=run_start)
    m9 = compute_m9_diagnostics(
        state,
        clock_namelist,
        lead_seconds,
        noahmp_land=noahmp_land,
        noahmp_rad=noahmp_rad,
    )
    # Q2 stays the bulk surface-layer diagnostic (matches the single-domain path).
    try:
        q2 = getattr(surface_layer_diagnostics(state, clock_namelist.grid), "q2", None)
    except Exception:  # noqa: BLE001 -- Q2 is auxiliary; the writer keeps its default.
        q2 = None
    out: dict[str, np.ndarray] = {}
    for wrf_name, attr in _M9_OUTPUT_FIELDS:
        value = q2 if wrf_name == "Q2" else (getattr(m9, attr, None) if attr else None)
        if value is None:
            continue
        out[wrf_name] = np.asarray(jax.device_get(value))
    return out or None


class _PerDomainWrfoutWriter:
    """Output callback: write one wrfout per domain at the hourly cadence.

    Declares ``wants_carry`` so the domain-tree runner hands it the full
    ``OperationalCarry``: under Noah-MP the writer diagnostics must read the
    EVOLVED land carry (``carry.noahmp_land``) and the held surface radiation
    (``carry.noahmp_rad``), not just the post-step ``State``.
    """

    wants_carry = True

    def __init__(
        self,
        *,
        output_dir: Path,
        input_dir: Path,
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
            _load_static_latlon_writer_diagnostics,
            _merge_output_diagnostics,
            _surface_diagnostics_for_output,
            finite_summary,
        )
        from gpuwrf.io.gen2_accessor import Gen2Run

        self._surface_diagnostics_for_output = _surface_diagnostics_for_output
        self._merge_output_diagnostics = _merge_output_diagnostics
        self._finite_summary = finite_summary
        run = Gen2Run(input_dir)
        self.writer_diagnostics: dict[str, dict[str, Any]] = {}
        self.writer_static_latlon_metadata: dict[str, Any] = {}
        for domain, bundle in bundles.items():
            diagnostics, meta = _load_static_latlon_writer_diagnostics(
                run, domain, grid=bundle.grid
            )
            self.writer_static_latlon_metadata[domain] = meta
            if diagnostics:
                self.writer_diagnostics[domain] = diagnostics

    def __call__(self, name: str, own_step: int, carry: Any) -> dict[str, Any]:
        state = getattr(carry, "state", carry)
        cadence = int(self.output_cadence_steps[name])
        lead_h = int(round(int(own_step) / cadence))
        valid_time = self.run_start + timedelta(hours=lead_h)
        namelist = self.bundles[name].namelist
        grid = self.bundles[name].grid
        noahmp_land = getattr(carry, "noahmp_land", None)
        if bool(getattr(namelist, "use_noahmp", False)) and noahmp_land is not None:
            surface_diagnostics = _noahmp_surface_diagnostics_for_output(
                state,
                namelist,
                self.run_start,
                lead_seconds=float(lead_h) * 3600.0,
                noahmp_land=noahmp_land,
                noahmp_rad=getattr(carry, "noahmp_rad", None),
            )
        else:
            surface_diagnostics = self._surface_diagnostics_for_output(
                state, namelist, self.run_start, lead_seconds=float(lead_h) * 3600.0
            )
        diagnostics = self._merge_output_diagnostics(
            self.writer_diagnostics.get(name), surface_diagnostics
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

    # NESTED-OOM FIX (allocator).  The live nest allocates a recurring ~8-9 GiB
    # RRTMG g-point radiation transient every radiation step.  Under the default
    # XLA BFC arena (esp. with XLA_PYTHON_CLIENT_PREALLOCATE=false) a long 24 h
    # run fragments the pool so that single transient can no longer find a
    # contiguous block -- the production "allocate 9.24 GiB" OOM -- even though
    # peak in-use stays ~9 GiB.  The synchronous platform (cudaMalloc/cudaFree)
    # allocator has NO arena and so cannot fragment: every transient gets a fresh
    # contiguous device allocation and is returned to the driver immediately on
    # free.  Combined with the output-interval segmentation below this keeps peak
    # VRAM flat (one segment working set + ONE transient) across any forecast
    # length.  It MUST be set before JAX initializes its GPU backend; the nested
    # path's first device op is inside this function, and the only earlier jax
    # touch (CLI namelist parsing) does no device op, so setting it here is in
    # time.  ``setdefault`` keeps an explicit operator override authoritative.
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    import jax  # local import keeps module import light for --help / arg parsing.

    from gpuwrf.profiling.transfer_audit import visible_gpu_name

    names = domain_names_for(config.max_dom)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.proof_dir.mkdir(parents=True, exist_ok=True)

    if int(config.hours) <= 0:
        raise ValueError("hours must be positive")

    overall_start = time.perf_counter()
    hierarchy, bundles, meta, run_start, dt_by_domain, initial_carries = _load_domains(
        config, names
    )

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

    feedback_enabled = bool(config.feedback)
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=feedback_enabled)
    writer = _PerDomainWrfoutWriter(
        output_dir=config.output_dir,
        input_dir=Path(config.input_dir),
        run_start=run_start,
        bundles=bundles,
        output_cadence_steps=output_cadence,
    )
    for domain, latlon_meta in writer.writer_static_latlon_metadata.items():
        meta.setdefault("domains", {}).setdefault(domain, {})["writer_static_latlon"] = latlon_meta

    # MEMORY-BOUNDED segmented host loop (v0.12.0 nested-OOM fix).  The whole
    # forecast was previously ONE run_operational_domain_tree call: a single host
    # recursion over all root_steps.  The recurring RRTMG g-point radiation
    # transient (~8-9 GiB on the d02 grid) is allocated whenever radiation fires;
    # across a 24 h run the BFC allocator fragments and can no longer find a
    # contiguous block for it (the production "allocate 9.24 GiB" OOM), even
    # though peak in-use stays ~9 GiB.  We now drive the SAME validated recursion
    # one OUTPUT INTERVAL at a time, carrying the device carries + the global step
    # clock (own_steps) across segments and block_until_ready-ing + dropping the
    # prior segment's result between segments so each segment's scratch is freed
    # before the next allocates -- the nested analogue of
    # run_forecast_operational_segmented.  The recursion cadence + radiation
    # schedule are byte-identical to the single full-length call (the in-chunk
    # radiation gate keys off the threaded global step index); only the
    # memory/segmentation orchestration changes, NOT the physics/dynamics or the
    # live parent->child boundary coupling.
    forecast_start = time.perf_counter()
    root_seg_steps = int(output_cadence[root])  # one wrfout hour of root steps
    # Pre-seeded initial carries from _load_domains: bit-identical to the former
    # domain-tree cold start for the bulk path, and REQUIRED under Noah-MP so the
    # land carry is structurally present from the very first scan segment.
    carries: dict[str, Any] | None = initial_carries
    own_steps: dict[str, int] = {name: 0 for name in names}
    events: list[Any] = []
    final_states: dict[str, Any] = {}
    start = 0
    while start < root_steps:
        seg = min(root_seg_steps, root_steps - start)
        result = run_operational_domain_tree(
            tree,
            root_steps=seg,
            feedback_enabled=feedback_enabled,
            output=writer,
            output_cadence_steps=output_cadence,
            block_between=True,
            carries=carries,
            initial_own_steps=own_steps,
        )
        # Block so this segment's device scratch (incl. the RRTMG transient) is
        # freed before the next segment allocates -- bounds peak VRAM to one
        # segment's working set regardless of forecast length.
        jax.block_until_ready(tuple(state.theta for state in result.states.values()))
        carries = result.carries
        own_steps = dict(result.own_steps)
        events.extend(result.events)
        final_states = result.states
        start += seg
    jax.block_until_ready(tuple(state.theta for state in final_states.values()))
    forecast_wall_s = time.perf_counter() - forecast_start
    result = DomainTreeResult(
        carries=carries or {},
        states=final_states,
        own_steps=own_steps,
        events=tuple(events),
        outputs=(),
    )

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
        "feedback": bool(feedback_enabled),
        "nesting_mode": "two_way" if feedback_enabled else "one_way",
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
            (
                "Two-way feedback (child->parent copy_fcn area-average + WRF sm121 "
                "feedback-zone smoother) is ENABLED."
                if feedback_enabled
                else "Two-way feedback is OFF (one-way nesting); matches the "
                "v0.11.0 validated wiring."
            ),
            "In-loop nested w relaxation is OFF (deferred to a longer stability gate).",
            "No TOST/ensemble equivalence or CPU-speedup baseline is claimed by a standalone smoke.",
        ],
    }
    return payload
