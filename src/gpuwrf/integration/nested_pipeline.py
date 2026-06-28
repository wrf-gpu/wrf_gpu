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

The driver writes one ``wrfout_<domain>_<valid_time>`` per domain at the
namelist ``history_interval`` cadence and returns a JSON-serializable payload
mirroring the daily pipeline's ``M7DailyPipelineRun`` shape (so the CLI can
print/branch on it uniformly).
"""

from __future__ import annotations

import calendar
from collections import Counter, deque
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
from gpuwrf.io.data_inventory import wrfout_name
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.io.radiation_static import load_radiation_static
from gpuwrf.io.gwdo_static import load_gwdo_statics
from gpuwrf.io.wrfout_writer import (
    FULL_WRFOUT_VARIABLES,
    MINIMAL_TRAINING_SET,
    prepare_wrfout_payload,
    write_wrfout_netcdf,
)
from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    DomainTreeResult,
    maybe_prewarm_defused_nest,
    nested_precompile_report,
    run_operational_domain_tree,
    with_live_child_boundary_config,
)
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _commit_to_operational_device,
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


def _wrfout_path(output_dir: Path, domain: str, valid_time: datetime) -> Path:
    return output_dir / wrfout_name(domain, valid_time)


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
    # WRF namelists may pack several params per line, so the parser can return a
    # 1-element list for a scalar key (e.g. "time_step = 18, ..."); coerce to scalar.
    if isinstance(root_dt, (list, tuple)):
        root_dt = root_dt[0]
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


def _domain_list_value(value: Any, index: int, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[index] if index < len(value) else value[-1]
    return value


def _history_interval_minutes_by_domain(run, names: tuple[str, ...]) -> dict[str, float]:
    """Per-domain WRF history interval in minutes, defaulting to hourly output."""

    raw = run.namelist.get("time_control", {}).get("history_interval", 60)
    out: dict[str, float] = {}
    for idx, name in enumerate(names):
        minutes = float(_domain_list_value(raw, idx, 60))
        if minutes <= 0.0:
            raise ValueError(f"{name}: history_interval must be positive, got {minutes}")
        out[name] = minutes
    return out


def _output_cadence_steps_by_domain(
    run,
    names: tuple[str, ...],
    dt_by_domain: dict[str, float],
) -> tuple[dict[str, int], dict[str, float]]:
    """Return output cadence steps and interval minutes per domain."""

    interval_minutes = _history_interval_minutes_by_domain(run, names)
    cadence_steps: dict[str, int] = {}
    for name in names:
        dt_s = float(dt_by_domain[name])
        raw_steps = interval_minutes[name] * 60.0 / dt_s
        steps = int(math.ceil(raw_steps - 1.0e-12))
        if steps <= 0:
            raise ValueError(f"{name}: history_interval produced nonpositive cadence")
        cadence_steps[name] = steps
    return cadence_steps, interval_minutes


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
        acoustic_substeps=int(os.environ.get("GPUWRF_ACOUSTIC_SUBSTEPS", 10)),
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
        # WRF Registry default hypsometric_opt=2 (LOG form); see daily_pipeline.
        hypsometric_opt=2,
        radiation_static=radiation_static,
        time_utc=run_start,
        gwd_opt=int(gwd_opt),
        gwdo_statics=gwdo_statics,
        # v0.20 S4: production opt-in for perturbation-authoritative mixed fp32.
        # Unset remains fp64_default; invalid strings fail closed in
        # OperationalNamelist.__post_init__.
        acoustic_precision_mode=os.environ.get("GPUWRF_ACOUSTIC_PRECISION_MODE", "fp64_default"),
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
    """Enable WRF-native specified-boundary handling for standalone roots.

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

    Native wrfbdy roots also need the WRF specified-boundary timestep cadence:
    per-stage dry relax/spec pins and specified-domain advection degradation at
    the edge. Leaving those opt-in replay toggles off lets the root d01 boundary
    behave like a periodic/high-order edge between end-of-step nudges, which is
    dynamically fatal on the Mont-Blanc terrain fixture.
    """

    interval_s = (case_metadata.get("boundary") or {}).get("interval_seconds")
    if not interval_s:
        return namelist
    return dataclass_replace(
        namelist,
        specified_bdy_cadence=True,
        specified_adv_degrade=True,
        boundary_config=dataclass_replace(
            namelist.boundary_config,
            update_cadence_s=float(interval_s),
            normal_bdy_relax_strength=1.0,
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
        # v0.17 nested compile-CHURN fix.  `_advance_chunk` RETURNS device-committed
        # leaves; if the FIRST nested advance for a domain receives this HOST/
        # uncommitted seed while the SECOND receives the prior chunk's COMMITTED
        # output, JAX keys them as different shardings and recompiles an otherwise
        # identical executable -- TWICE per domain (~18-20 cold compiles for the
        # all-7, every ~4-5 min, GPU idle, 0 forecast output until they all finish).
        # Committing the seed ONCE here makes the first advance reuse the committed-
        # carry cache key -> ~9 compiles (one per domain).  This mirrors the
        # single-domain segmented/diagnostics entries (`run_forecast_operational_
        # segmented`/`..._with_m9_diagnostics`) which already seed via
        # `_committed_initial_carry_for_run`.  Pure device placement
        # (`jax.device_put`); leaf VALUES are bit-identical, so wrfout is unchanged.
        carry = _commit_to_operational_device(carry)
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
        build_clock_base,
        compute_m9_diagnostics,
        surface_layer_diagnostics,
    )

    clock_namelist = namelist
    if getattr(namelist, "time_utc", None) is None:
        clock_namelist = dataclass_replace(namelist, time_utc=run_start)
    # #91: traced per-run date scalars so the M9 diagnostic HLO is date-independent.
    m9 = compute_m9_diagnostics(
        state,
        clock_namelist,
        lead_seconds,
        noahmp_land=noahmp_land,
        noahmp_rad=noahmp_rad,
        clock_base=build_clock_base(clock_namelist),
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


def _resolve_training_output_subset() -> tuple[str, ...] | None:
    """Resolve the OPT-IN compact training-output variable subset (#122).

    Returns ``MINIMAL_TRAINING_SET`` when the ``GPUWRF_TRAINING_OUTPUT_SUBSET`` env
    flag is truthy (``1``/``true``/``yes``/``on``, case-insensitive), else ``None``.
    ``None`` keeps the per-domain nest output at the full, uncompressed,
    byte-identical default for every existing caller -- the feature is off unless
    explicitly enabled for a training run.
    """

    raw = os.environ.get("GPUWRF_TRAINING_OUTPUT_SUBSET")
    if raw is None:
        return None
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return MINIMAL_TRAINING_SET
    return None


def _resolve_full_wrfout_variables() -> bool:
    """Resolve the OPT-IN 375-variable WRF history stream for nested output."""

    for env_name in ("GPUWRF_FULL_WRFOUT_VARIABLES", "GPUWRF_FULL_WRFOUT"):
        raw = os.environ.get(env_name, "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
    return False


class _PerDomainWrfoutWriter:
    """Output callback: write one wrfout per domain at history cadence.

    Declares ``wants_carry`` so the domain-tree runner hands it the full
    ``OperationalCarry``: under Noah-MP the writer diagnostics must read the
    EVOLVED land carry (``carry.noahmp_land``) and the held surface radiation
    (``carry.noahmp_rad``), not just the post-step ``State``.

    When the ``GPUWRF_TRAINING_OUTPUT_SUBSET`` env flag is set, the per-domain
    wrfout is restricted to the compact, lossless-compressed
    ``MINIMAL_TRAINING_SET`` (#122 training output); otherwise the full
    uncompressed wrfout is written exactly as before (byte-identical default).
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
        dt_by_domain: dict[str, float],
        async_writer: AsyncWrfoutWriter | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.run_start = run_start
        self.bundles = bundles
        self.output_cadence_steps = output_cadence_steps
        self.dt_by_domain = dt_by_domain
        # When set, the device->host materialization stays on the step thread but
        # the NetCDF write is submitted to this background writer (the step loop
        # resumes GPU compute immediately). When None, the write is synchronous on
        # the step thread (legacy path). The written wrfout bytes are identical in
        # both cases; ``self.written`` records the deterministic output path at
        # submit time so the output-present check below stays valid either way.
        self._async_writer = async_writer
        # Opt-in compact training output (#122): None => full byte-identical output.
        self._variable_subset = _resolve_training_output_subset()
        self._full_variable_set = _resolve_full_wrfout_variables()
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
        lead_seconds = float(own_step) * float(self.dt_by_domain[name])
        lead_hours = lead_seconds / 3600.0
        assert_state_finite_at_boundary(
            state, domain=name, step=int(own_step), sim_time_s=lead_seconds
        )
        valid_time = self.run_start + timedelta(seconds=lead_seconds)
        namelist = self.bundles[name].namelist
        grid = self.bundles[name].grid
        noahmp_land = getattr(carry, "noahmp_land", None)
        if bool(getattr(namelist, "use_noahmp", False)) and noahmp_land is not None:
            surface_diagnostics = _noahmp_surface_diagnostics_for_output(
                state,
                namelist,
                self.run_start,
                lead_seconds=lead_seconds,
                noahmp_land=noahmp_land,
                noahmp_rad=getattr(carry, "noahmp_rad", None),
            )
        else:
            surface_diagnostics = self._surface_diagnostics_for_output(
                state, namelist, self.run_start, lead_seconds=lead_seconds
            )
        diagnostics = self._merge_output_diagnostics(
            self.writer_diagnostics.get(name), surface_diagnostics
        )
        path = _wrfout_path(self.output_dir, name, valid_time)
        if self._async_writer is not None:
            # Keep the device->host pull on the step thread (so no off-thread
            # touch of a device buffer the GPU may reuse), then submit the
            # host-only payload to the background writer and resume compute. The
            # deterministic output path is recorded NOW (at submit time) so the
            # output-present check stays valid; join() runs before that check.
            prepared = prepare_wrfout_payload(
                state,
                grid,
                namelist,
                path,
                valid_time=valid_time,
                lead_hours=float(lead_hours),
                run_start=self.run_start,
                diagnostics=diagnostics,
                full_variable_set=self._full_variable_set,
            )
            if self._variable_subset is None:
                self._async_writer.submit(prepared)
            else:
                # Compact training stream: same host payload, subset + mandatory
                # coords + lossless compression (self-contained, ~10 GB/day target).
                self._async_writer.submit_subset(
                    prepared,
                    variable_subset=self._variable_subset,
                    target=path,
                    include_mandatory_coords=True,
                    compress=True,
                )
        else:
            write_wrfout_netcdf(
                state,
                grid,
                namelist,
                path,
                valid_time=valid_time,
                lead_hours=float(lead_hours),
                run_start=self.run_start,
                diagnostics=diagnostics,
                variable_subset=self._variable_subset,
                include_mandatory_coords=self._variable_subset is not None,
                compress=self._variable_subset is not None,
                full_variable_set=self._full_variable_set,
            )
        self.written[name].append(str(path))
        summary = self._finite_summary(state)
        return {
            "domain": name,
            "lead_hours": float(lead_hours),
            "own_step": int(own_step),
            "all_finite": bool(summary["all_finite"]),
            "wrfout": str(path),
            "prepared_full_variable_set": bool(self._full_variable_set),
            "full_variable_count": int(len(FULL_WRFOUT_VARIABLES)) if self._full_variable_set else None,
        }


def _finite_stats_host(state: Any) -> dict[str, Any]:
    from gpuwrf.integration.daily_pipeline import finite_summary

    return finite_summary(state)


def _nested_async_output_from_env() -> bool:
    """Resolve whether the nested path writes wrfout on a background thread.

    The per-domain writer used to do the device->host pull AND the synchronous
    NetCDF write on the step thread, stalling GPU compute for the full write of
    every output group (~30 s on the all-7 nest -- the single biggest discrete
    host bubble).  When async output is on, the step thread keeps the
    device->host materialization (``prepare_wrfout_payload`` -> host-only
    ``PreparedWrfout``) and then hands the host payload to a single bounded-queue
    background writer thread (the already-proven :class:`AsyncWrfoutWriter`); the
    step loop resumes GPU compute immediately while the write overlaps.

    The written NetCDF bytes are byte-for-byte identical to the synchronous path
    (``write_prepared_wrfout`` is pure host work and the single writer thread
    serializes writes deterministically); only the wall-clock timing of the write
    changes.  A failed background write still fails the run (``join()`` re-raises).

    Default = ON (the lever's purpose).  ``GPUWRF_NESTED_ASYNC_OUTPUT=0`` (also
    ``false``/``off``/``no``) forces the legacy synchronous write -- used to
    reproduce the slow baseline for A/B measurement and for byte-identity proofs
    that want the write fully on the step thread.
    """

    raw = os.environ.get("GPUWRF_NESTED_ASYNC_OUTPUT", "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    return True


def _nested_sync_mode_from_env() -> tuple[bool, int | None]:
    """Resolve the nested host-sync granularity (``GPUWRF_NESTED_SYNC_MODE``).

    Returns ``(block_between, root_sync_cadence)`` for
    :func:`run_operational_domain_tree`.  The v0.17 GPU-idle fix makes the
    asynchronous per-root-step sync the DEFAULT: the legacy path drained the GPU
    queue after every single domain advance (~5,000 blocks/forecast-hour for the
    all-7 geometry), idling the GPU between host-built boundary packages.  Syncing
    once per root step instead keeps the queue full across each root-step cascade
    while bounding how far the host races ahead (peak VRAM).  ``block_until_ready``
    is purely a host wait -- it changes NO dispatched op -- so every mode produces
    byte-identical wrfout; only utilization / wallclock / peak-VRAM differ.

    Modes:
      * unset / ``root`` / ``root:K``  -> async, host sync every K root steps
        (K>=1, default 1).  The release default.
      * ``advance``                    -> legacy per-advance block (pre-v0.17).
        Used only to reproduce the slow baseline for A/B measurement.
      * ``segment``                    -> no intra-segment host sync; rely on the
        output/segment boundary block (maximum overlap, highest peak VRAM).
    """
    raw = os.environ.get("GPUWRF_NESTED_SYNC_MODE", "").strip().lower()
    if raw in ("", "root"):
        return False, 1
    if raw == "advance":
        return True, None
    if raw == "segment":
        return False, None
    if raw.startswith("root:"):
        try:
            cadence = max(1, int(raw.split(":", 1)[1]))
        except ValueError:
            cadence = 1
        return False, cadence
    # Unknown token -> safe async default rather than silently reverting to slow.
    return False, 1


def _nested_event_tail_cap_from_env(default: int = 4096) -> int:
    """Resolve the cross-segment event-tail cap (``GPUWRF_NESTED_EVENT_TAIL``).

    HOST-RAM GUARD (v0.20 / v0.19.2 item 7).  The segmented host loop folds each
    output-segment's :attr:`DomainTreeResult.events` into running ``event_counts``
    / ``force_counts`` Counters (the ONLY values any downstream consumer reads)
    and retains just the most-recent ``cap`` raw event tuples for diagnostics.
    The per-segment ``events`` list is itself bounded (one output interval), but
    the OLD code did ``events.extend(result.events)`` every segment, so the host
    list grew O(forecast_length) -- one batch of (str/int) tuples per segment --
    over a 24-120 h skill-gate run.  At ~5,000 events/forecast-hour x 120 h that
    is ~600k tuples (tens of MB of fragmented Python objects) accumulated purely
    for a summary; near the swap-thrash incident this is real host-RAM pressure.
    Folding to counts + a bounded tail makes host RAM O(1) in forecast length.

    ``cap <= 0`` keeps an UNBOUNDED tail (legacy behaviour; the summary counts are
    identical either way).  Default keeps the last 4096 events (a few root-step
    cascades) -- ample for a post-mortem, trivially small.
    """
    raw = os.environ.get("GPUWRF_NESTED_EVENT_TAIL", "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def execute_nested_pipeline(config: NestedPipelineConfig) -> dict[str, Any]:
    """Run a standalone live-nested forecast and write per-domain wrfout.

    Returns an ``M7DailyPipelineRun``-shaped payload with ``init_mode``
    ``standalone_native_init_nested`` and a per-domain finite/output summary.
    """

    # NESTED allocator (v0.20.0 speed lever G_allocator_env).  The live nest
    # allocates a recurring ~8-9 GiB RRTMG g-point radiation transient every
    # radiation step.  The DEFAULT is now ``cuda_async`` -- the stream-ordered
    # CUDA memory pool -- which amortises the per-op malloc/free churn that the
    # old synchronous ``platform`` allocator paid on every device buffer, while
    # (unlike the default XLA BFC arena) NOT using the best-fit arena whose
    # fragmentation caused the original "allocate 9.24 GiB" 1km-nest OOM.  The
    # ``platform`` (raw cudaMalloc/cudaFree, no arena, cannot fragment) path
    # remains one env var away: ``GPUWRF_ALLOCATOR=platform``.  This is a
    # NUMERICS-FREE knob (governs only where device buffers live, never the math)
    # and is coordinated with cli.py:_resolve_nested_allocator (same default and
    # precedence).  It MUST be set before JAX initializes its GPU backend; the
    # nested path's first device op is inside this function, and the only earlier
    # jax touch (CLI namelist parsing) does no device op, so setting it here is in
    # time.  ``setdefault`` keeps an explicit operator override authoritative.
    if not os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"):
        _requested = os.environ.get("GPUWRF_ALLOCATOR", "").strip().lower()
        if not _requested:
            _allocator = "cuda_async"  # v0.20.0 default (was "platform")
        elif _requested == "bfc":
            _allocator = "default"  # XLA spells its default BFC arena "default"
        else:
            _allocator = _requested
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = _allocator

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
    from gpuwrf.io.gen2_accessor import Gen2Run

    cadence_run = Gen2Run(Path(config.input_dir))
    output_cadence, history_interval_minutes = _output_cadence_steps_by_domain(
        cadence_run, names, dt_by_domain
    )

    feedback_enabled = bool(config.feedback)
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=feedback_enabled)

    # CROSS-DOMAIN PARALLEL PRE-COMPILE (vNext de-fuse cold-wall win). When the
    # de-fuse compile path is active (GPUWRF_NESTED_DEFUSE_COMPILE=1 /
    # GPUWRF_NESTED_FUSE=0 / GPUWRF_BITWISE) the N independent per-domain
    # _advance_chunk_fori modules would otherwise compile SEQUENTIALLY (~Sum(N)
    # ~50 min). This warms the shared version-keyed (locked) cache CONCURRENTLY in
    # spawned child processes BEFORE the integration loop, so the eager loop below
    # warm-hits all N (cold wall ~max(one body) + pool overhead). No-op for the
    # fused default and fully FAIL-OPEN (a failure just
    # cold-compiles as today). Numerically inert. The two stderr markers below
    # bracket the wall so a GPU A/B can time the cold-compile-wall precisely. The
    # gate self-gates (no-op for the fused default / GPUWRF_NESTED_PARALLEL_COMPILE=0)
    # so it is always safe to call here. De-fuse remains opt-in.
    _pc_t0 = time.perf_counter()
    sys.stderr.write("[parallel-compile] PREWARM_START de-fuse nest\n")
    sys.stderr.flush()
    _pc_status = maybe_prewarm_defused_nest(tree, carries=initial_carries)
    _pc_dt = time.perf_counter() - _pc_t0
    _pc_rep = _pc_status.get("report") or {}
    sys.stderr.write(
        "[parallel-compile] PREWARM_DONE active=%s source=%s workers=%s "
        "wall_s=%.1f warm_all=%s error=%s\n"
        % (
            _pc_status.get("active"),
            _pc_status.get("source"),
            _pc_status.get("workers"),
            _pc_dt,
            _pc_rep.get("warm_all") if isinstance(_pc_rep, dict) else None,
            _pc_status.get("error"),
        )
    )
    sys.stderr.flush()
    meta.setdefault("parallel_compile", {})["nested_precompile"] = nested_precompile_report()

    # Async history output (v0.20 host-bubble lever): when on, the per-domain
    # writer keeps the device->host materialization on the step thread but submits
    # the host payload to this single bounded-queue background writer thread, so
    # the step loop resumes GPU compute instead of stalling on the ~30 s/output-
    # group synchronous NetCDF write. Byte-identical output; default-on, disable
    # with GPUWRF_NESTED_ASYNC_OUTPUT=0. ``max_pending`` is kept small to bound the
    # queued 9-domain PreparedWrfout host RAM. join()ed below before the output-
    # present check / pipeline exit so a failed background write fails the run.
    async_output_enabled = _nested_async_output_from_env()
    async_writer = AsyncWrfoutWriter(max_pending=2) if async_output_enabled else None
    writer = _PerDomainWrfoutWriter(
        output_dir=config.output_dir,
        input_dir=Path(config.input_dir),
        run_start=run_start,
        bundles=bundles,
        output_cadence_steps=output_cadence,
        dt_by_domain=dt_by_domain,
        async_writer=async_writer,
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
    root_seg_steps = int(output_cadence[root])  # one root history-output segment
    # Pre-seeded initial carries from _load_domains: bit-identical to the former
    # domain-tree cold start for the bulk path, and REQUIRED under Noah-MP so the
    # land carry is structurally present from the very first scan segment.
    carries: dict[str, Any] | None = initial_carries
    own_steps: dict[str, int] = {name: 0 for name in names}
    # HOST-RAM GUARD (v0.20): fold each segment's events into running summary
    # Counters + a bounded tail instead of accumulating EVERY event tuple across
    # the whole forecast (the old ``events.extend`` grew O(forecast_length) on the
    # host -- a real concern for the 24-120 h skill-gate runs, implicated near the
    # swap-thrash incident).  ``event_counts`` / ``force_counts`` are the only
    # values any downstream consumer reads, and are IDENTICAL whether folded
    # incrementally or computed once over the full list (counting is associative).
    event_tail_cap = _nested_event_tail_cap_from_env()
    event_counts: Counter = Counter()
    force_counts: Counter = Counter()
    events_tail: deque = deque(maxlen=event_tail_cap if event_tail_cap > 0 else None)
    final_states: dict[str, Any] = {}
    # Host-sync granularity for the live nest (v0.17 GPU-idle fix).  Default =
    # async per-root-step sync (keeps the GPU queue full across nested cascades);
    # GPUWRF_NESTED_SYNC_MODE=advance reproduces the legacy per-advance baseline.
    # The per-segment block below is ALWAYS kept (peak-VRAM bound between hours).
    nested_block_between, nested_root_sync_cadence = _nested_sync_mode_from_env()
    start = 0
    try:
        while start < root_steps:
            seg = min(root_seg_steps, root_steps - start)
            result = run_operational_domain_tree(
                tree,
                root_steps=seg,
                feedback_enabled=feedback_enabled,
                output=writer,
                output_cadence_steps=output_cadence,
                block_between=nested_block_between,
                root_sync_cadence=nested_root_sync_cadence,
                carries=carries,
                initial_own_steps=own_steps,
            )
            # Block so this segment's device scratch (incl. the RRTMG transient) is
            # freed before the next segment allocates -- bounds peak VRAM to one
            # segment's working set regardless of forecast length.
            jax.block_until_ready(tuple(state.theta for state in result.states.values()))
            for domain_name, state in result.states.items():
                domain_step = int(result.own_steps.get(domain_name, own_steps.get(domain_name, 0)))
                assert_state_finite_at_boundary(
                    state,
                    domain=domain_name,
                    step=domain_step,
                    sim_time_s=float(domain_step) * float(dt_by_domain[domain_name]),
                )
            carries = result.carries
            own_steps = dict(result.own_steps)
            # Fold this segment's events into the running summary + bounded tail, then
            # DROP the segment's tuple (host RAM stays O(1) in forecast length).
            for event in result.events:
                if not event:
                    continue
                event_counts[event[0]] += 1
                if event[0] == "force":
                    force_counts[f"{event[1]}->{event[2]}"] += 1
                events_tail.append(event)
            final_states = result.states
            start += seg
        jax.block_until_ready(tuple(state.theta for state in final_states.values()))
        forecast_wall_s = time.perf_counter() - forecast_start
        # Drain the background wrfout writer: all submitted output groups must be
        # on disk (and any writer error surfaced -- join() re-raises, failing the
        # run) before the output-present check below reads writer.written. The
        # writes overlapped GPU compute, so this join only waits on the last
        # in-flight write and is outside the forecast_wall_s timing. Idempotent /
        # no-op when async output is disabled (async_writer is None).
        if async_writer is not None:
            async_writer.join()
    finally:
        # Fail-closed: if the forecast loop above raised (NaN/OOM/etc.), still
        # drain the background writer so the daemon thread cannot outlive the run
        # and any in-flight write is flushed -- but do NOT mask the body's
        # exception with a secondary writer error. join() is idempotent, so on the
        # success path this is a no-op.
        if async_writer is not None:
            try:
                async_writer.join()
            except BaseException:
                if sys.exc_info()[0] is None:
                    raise
    result = DomainTreeResult(
        carries=carries or {},
        states=final_states,
        own_steps=own_steps,
        # ``events`` now carries only the bounded recent-event tail (last N); the
        # authoritative aggregate lives in event_counts/force_counts below.
        events=tuple(events_tail),
        outputs=(),
    )

    final_finite = {name: _finite_stats_host(state) for name, state in result.states.items()}

    per_domain: dict[str, Any] = {}
    all_finite = True
    all_output_present = True
    for name in names:
        outputs = writer.written.get(name, [])
        finite_ok = bool(final_finite[name]["all_finite"])
        total_steps = int(math.floor(float(config.hours) * 3600.0 / float(dt_by_domain[name])))
        expected_outputs = int(total_steps // int(output_cadence[name]))
        output_ok = len(outputs) == expected_outputs
        all_finite = all_finite and finite_ok
        all_output_present = all_output_present and output_ok
        per_domain[name] = {
            "final_state_finite": finite_ok,
            "wrfout_count": len(outputs),
            "expected_wrfout_count": expected_outputs,
            "wrfout_files": outputs,
            "dt_s": float(dt_by_domain[name]),
            "history_interval_min": float(history_interval_minutes[name]),
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
