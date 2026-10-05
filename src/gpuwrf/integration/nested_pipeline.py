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
from fractions import Fraction
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.integration.d02_replay import build_replay_case, wrf_nest_opens_with_parent_start
from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
from gpuwrf.io.data_inventory import wrfout_name
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
from gpuwrf.io.radiation_static import load_radiation_static
from gpuwrf.io.gwdo_static import load_gwdo_statics
from gpuwrf.io.wrfout_writer import (
    FULL_WRFOUT_VARIABLES,
    MINIMAL_TRAINING_SET,
    bind_wrfout_domain_authority,
    prepare_wrfout_payload,
    write_prepared_wrfout,
)
from gpuwrf.nesting.boundary_construction import (
    build_child_boundary_package,
    initialize_child_scalar_boundaries,
)
from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    DomainTreeResult,
    _coupled_forcedown_enabled,
    _prepare_operational_domain_tree_runtime,
    maybe_prewarm_defused_nest,
    nested_aot_report,
    nested_precompile_report,
    run_domain_tree_callbacks,
    run_operational_domain_tree,
    with_live_child_boundary_config,
)
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _advance_chunk,
    _commit_to_operational_device,
    _initial_carry_for_run,
    _resolve_operational_suite,
    build_clock_base,
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
_FALSEY_BATCH_ENV = {"0", "false", "off", "no", ""}
def _batch_ensemble_size_from_env() -> int:
    raw = os.environ.get("GPUWRF_BATCH_ENSEMBLE")
    if raw is None or raw.strip().lower() in _FALSEY_BATCH_ENV:
        return 1
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(
            "GPUWRF_BATCH_ENSEMBLE must be a positive integer; got "
            f"{raw!r}"
        ) from exc
    if size < 1:
        raise ValueError(
            "GPUWRF_BATCH_ENSEMBLE must be a positive integer; "
            f"got {size}"
        )
    return size


def batch_ensemble_size_from_env() -> int:
    return _batch_ensemble_size_from_env()


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
    # WRF history includes the initialized state at lead zero.  Keep this explicit
    # and default-off so historical callers retain their exact post-step-only
    # output count/bytes; corrected identity runs opt in through the CLI.
    emit_initial_history: bool = False
    # Optional F1 homogeneous ensemble inputs.  Used only when
    # GPUWRF_BATCH_ENSEMBLE is a fixed supported B; unset repeats ``input_dir``
    # for all lanes (useful for bit-identity/perturbation gates).
    batch_input_dirs: tuple[Path, ...] | None = None
    checkpoint_dir: Path | None = None
    checkpoint_interval_steps: int = 0
    checkpoint_max_bytes: int = 8 * 1024**3
    checkpoint_max_generations: int = 2
    checkpoint_reserve_bytes: int = 10 * 1024**3
    resume_checkpoint: Path | None = None
    # Speculative cached loads; actual runtime keys still gate executable reuse.
    # Default-off until concurrent GPU loading and wall-time gates pass.
    aot_prefetch: bool = False


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
    root_dt = _domain_list_value(nml.get("domains", {}).get("time_step"), 0, None)
    if root_dt is None:
        root_dt = _domain_list_value(nml.get("time_control", {}).get("time_step"), 0, None)
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


def _domain_list_value(value: Any, index: int, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[index] if index < len(value) else value[-1]
    return default if value is None else value


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


def _positive_fraction(value: float, *, label: str) -> Fraction:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return Fraction(str(numeric))


def _history_alarm_steps(
    *,
    total_steps: int,
    dt_s: float,
    interval_minutes: float,
) -> tuple[tuple[int, ...], bool]:
    """Return exact absolute alarm steps and whether cadence is integral.

    Alarm ``i`` fires on ``ceil(i * interval / dt)``.  Exact decimal-rational
    arithmetic prevents floating rounding from moving a rollover by one step.
    Multiple sub-step alarms that land on the same model step are coalesced,
    matching the scheduler's one-history-callback-per-completed-step boundary.
    """

    steps = int(total_steps)
    if steps < 0 or steps != total_steps:
        raise ValueError(f"total_steps must be a nonnegative integer, got {total_steps!r}")
    dt = _positive_fraction(dt_s, label="dt_s")
    interval = _positive_fraction(
        interval_minutes, label="history_interval_minutes"
    ) * 60
    ratio = interval / dt
    alarm_count = int((steps * dt) // interval)
    alarms: list[int] = []
    for alarm_index in range(1, alarm_count + 1):
        alarm_ratio = alarm_index * ratio
        alarm_step = -(-alarm_ratio.numerator // alarm_ratio.denominator)
        if alarm_step > steps:
            raise AssertionError("derived history alarm exceeds forecast step count")
        if not alarms or alarms[-1] != alarm_step:
            alarms.append(alarm_step)
    return tuple(alarms), ratio.denominator == 1


def _output_alarm_steps_by_domain(
    names: tuple[str, ...],
    dt_by_domain: dict[str, float],
    interval_minutes_by_domain: dict[str, float],
    total_steps_by_domain: dict[str, int],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Return all exact schedules and scheduler overrides for non-integral ones."""

    schedules: dict[str, tuple[int, ...]] = {}
    nonintegral: dict[str, tuple[int, ...]] = {}
    for name in names:
        alarms, integral = _history_alarm_steps(
            total_steps=int(total_steps_by_domain[name]),
            dt_s=float(dt_by_domain[name]),
            interval_minutes=float(interval_minutes_by_domain[name]),
        )
        schedules[name] = alarms
        if not integral:
            nonintegral[name] = alarms
    return schedules, nonintegral


def _radiation_cadence_steps(dt_s: float, radt_minutes: float = 30.0) -> int:
    # WRF module_physics_init.F uses NINT, whose positive half ties round up.
    return max(1, int(float(radt_minutes) * 60.0 / float(dt_s) + 0.5))


def _wrf_sound_steps(grid, dt_s: float, configured: int = 0) -> int:
    """WRF solve_em.F:447-464, fixed timestep (RK3's sound-step count)."""
    if configured:
        return int(configured)
    spacing = min(float(grid.projection.dx_m), float(grid.projection.dy_m))
    return max(2 * (int(300.0 * float(dt_s) / spacing - 0.01) + 1), 4)


def _cumulus_cadence_metadata(namelist):
    """Report the applied operands, including domains with cumulus disabled."""
    return {
        "cudt_minutes": float(namelist.cudt_minutes),
        "cumulus_cadence_steps": int(namelist.cumulus_cadence_steps),
    }


def _resolved_namelist_options(namelist, grid):
    """Every scalar control of the resolved OperationalNamelist this domain runs with (physics/dynamics/diffusion/damping
    options, cadences), plus the mass-grid shape: run-proof attestation of the EXECUTED options (metadata only; host
    values read from the frozen dataclass, no device access, nothing traced)."""
    out = {}
    for field in getattr(namelist, "__dataclass_fields__", {}):
        value = getattr(namelist, field, None)
        if value is None or isinstance(value, (bool, str)):
            out[field] = value
        elif isinstance(value, int):
            out[field] = int(value)
        elif isinstance(value, float):
            out[field] = float(value)
    out["mass_shape"] = [int(grid.nz), int(grid.ny), int(grid.nx)]
    return out


def _aot_prefetch_artifact_candidates(names, *, directory=None):
    """Select unambiguous artifact identities without guessing runtime keys.

    These are speculative load hints only. A caller must still require the
    exact runtime key before using a loaded executable; the loader owns all
    target, integrity, ABI and quarantine guards. Several complete keys use the
    runtime's last-used key; otherwise ambiguous/cold caches skip.
    """
    if directory is None:
        from gpuwrf.runtime.aot_precompile import aot_dir
        directory = aot_dir()
    if directory is None:
        return {}
    selected = {}
    for name in names:
        candidates = []
        for meta in (Path(directory) / name).glob("k_*.meta"):
            key = meta.stem[2:]
            if len(key) == 64 and all(c in "0123456789abcdef" for c in key):
                blob = meta.with_suffix(".xlaexec")
                if blob.is_file() and blob.stat().st_size:
                    candidates.append(key)
        from gpuwrf.runtime.aot_precompile import preferred_prefetch_key

        # Several complete keys: hint only the runtime's last-used one.
        key = preferred_prefetch_key(directory, name, candidates)
        if key is not None:
            selected[name] = key
    return selected


def _carry_donation_enabled() -> bool:
    """True when any ``GPUWRF_*DONATE*`` flag is on (e.g. GPUWRF_CARRY_DONATE: the advance
    deletes the carry it receives). Holders of a carry across a later device call must
    then keep their own device copy."""
    return any(
        name.startswith("GPUWRF_") and "DONATE" in name and value.strip().lower() in {"1", "true", "on", "yes"}
        for name, value in os.environ.items()
    )


def _device_copy_tree(tree: Any) -> Any:
    """Distinct device buffers for every ``jax.Array`` leaf; other leaves pass through."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    return jax.tree.map(lambda leaf: jnp.copy(leaf) if isinstance(leaf, jax.Array) else leaf, tree)


def _start_forcedown_warmup(tree: Any, initial_carries: dict[str, Any]) -> dict[str, Any] | None:
    """Load each coupled force-down executable on a background thread during init.

    Calls the runtime's own ``_operational_force`` once per coupled edge on the
    committed initial carries (the same signature as every real call) and drops
    the result, so the first real force-down hits the in-memory jit cache instead
    of tracing and loading inside the first forecast segment. Pure: no carry or
    output is touched. ``GPUWRF_FORCEDOWN_WARMUP=0`` disables it.
    """
    if os.environ.get("GPUWRF_FORCEDOWN_WARMUP", "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    try:  # fail-open: any unexpected tree/carry shape only skips the warm-up
        import threading  # noqa: PLC0415

        import jax  # noqa: PLC0415

        from gpuwrf.runtime.domain_tree import _operational_force  # noqa: PLC0415

        jobs = [
            (edge, initial_carries[parent], initial_carries[edge.child])
            for parent, edges in dict(getattr(tree, "edges", None) or {}).items()
            for edge in edges
            if bool(getattr(edge, "coupled_forcedown", False))
            and parent in initial_carries and edge.child in initial_carries
        ]
        if not jobs:
            return None
        report: dict[str, Any] = {"edges": [f"{e.parent}->{e.child}" for e, _p, _c in jobs], "error": None}
        donating = _carry_donation_enabled()

        def work() -> None:
            try:
                for edge, parent, child in jobs:
                    if donating:  # a donating force-down must not delete the live init carries
                        parent, child = _device_copy_tree(parent), _device_copy_tree(child)
                    jax.block_until_ready(_operational_force(edge, parent, child).state.u_bdy)
            except Exception as exc:  # noqa: BLE001 -- fail-open: the real call compiles as before
                report["error"] = f"{type(exc).__name__}: {exc}"

        report["thread"] = threading.Thread(target=work, name="forcedown-warmup", daemon=True)
        report["thread"].start()
    except Exception:  # noqa: BLE001
        return None
    return report


def _join_forcedown_warmup(report: dict[str, Any], timeout_s: float = 600.0) -> dict[str, Any]:
    """Wait for the force-down warm-up, bounded: a stuck warm-up never stalls the run.

    On timeout the daemon thread is left behind (fail-open: the real force-down call
    traces/loads as it would without the warm-up); ``joined`` records which happened.
    """
    wait_start = time.perf_counter()
    report["thread"].join(timeout=timeout_s)
    return {
        "edges": report["edges"],
        "remaining_wait_s": time.perf_counter() - wait_start,
        "error": report["error"],
        "joined": not report["thread"].is_alive(),
    }


def _two_time_live_nest_bdy(state: Any) -> Any:
    """Give a live-nest child's 1-level ``*_bdy`` placeholders the force-down's 2-level shape.

    The first force-down replaces every ``*_bdy`` package with ``[old_ring, new_target]``
    built from the child's current fields; ``two_time`` reads only the placeholder's
    last-level shape/dtype, never its values. Matching that shape at init lets the
    first force-down share the steady jit signature (one load instead of two).
    """
    import jax.numpy as jnp  # noqa: PLC0415

    updates = {}
    for name in getattr(type(state), "__slots__", ()):
        leaf = getattr(state, name, None) if name.endswith("_bdy") else None
        if leaf is not None and getattr(leaf, "ndim", 0) >= 1 and leaf.shape[0] == 1:
            updates[name] = jnp.concatenate([leaf, leaf], axis=0)
    return state.replace(**updates) if updates else state


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
    diff_opt: int = 0,
    km_opt: int = 0,
    h_sca_adv_order: int = 5,
    moist_adv_opt: int = 0,
    scalar_adv_opt: int = 0,
    topo_shading: int = 0,
    slope_rad: int = 0,
    time_step_sound: int = 0,
    radt_minutes: float = 30.0,
    cudt_minutes: float = 0.0,
    top_lid: bool = False,
) -> OperationalNamelist:
    """Per-domain operational namelist (mirrors the v0.11.0 nesting proof config).

    The dynamics knobs (flux advection, fp64 acoustic solve, 6th-order filter,
    Rayleigh + w damping) are the F7-closed operational settings the
    real-case path uses (see ``daily_pipeline._build_real_case``).  Children get
    the WRF live-nest boundary cadence (``update_cadence_s == parent_dt``) so the
    parent-built two-time package interpolates exactly across the subcycle.
    """

    namelist = OperationalNamelist.from_grid(
        grid,
        tendencies=tendencies,
        metrics=metrics,
        dt_s=float(dt_s),
        acoustic_substeps=int(os.environ.get(
            "GPUWRF_ACOUSTIC_SUBSTEPS", _wrf_sound_steps(grid, dt_s, time_step_sound)
        )),
        radiation_cadence_steps=_radiation_cadence_steps(dt_s, radt_minutes),
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
        # Registry.EM_COMMON: top_lid defaults to .false. (constant-pressure top).
        # A rigid lid changes the column-mass response to lateral pressure forcing.
        top_lid=bool(top_lid),
        # WRF Registry default hypsometric_opt=2 (LOG form); see daily_pipeline.
        hypsometric_opt=2,
        # WRF Registry.EM_COMMON defaults h_sca_adv_order to 5.  The standalone
        # daily path has always threaded that case value, but live nesting used
        # OperationalNamelist's idealized compatibility default (2).  That sent
        # rhs_ph through its periodic second-order branch on every real parent
        # and child even though WRF selects the map-factored specified/nested
        # order<=6 branch.  Bind the per-domain value here; explicit namelist
        # overrides remain authoritative and omitted values resolve to WRF's 5.
        h_sca_adv_order=int(h_sca_adv_order),
        # These are per-domain dynamics controls, not compatibility defaults.
        # WRF's option 1 selects ordinary scalar advection on RK1/RK2 and the
        # positive-definite branch on final RK3.  The Tenerife authority sets
        # both values explicitly on every domain.
        moist_adv_opt=int(moist_adv_opt),
        scalar_adv_opt=int(scalar_adv_opt),
        # Keep WRF's per-domain SW terrain controls through the live-nest seam.
        topo_shading=int(topo_shading),
        slope_rad=int(slope_rad),
        radiation_static=radiation_static,
        time_utc=run_start,
        gwd_opt=int(gwd_opt),
        gwdo_statics=gwdo_statics,
        # WRF folds RTHRATEN/RTHBLTEN and the QV contribution to moist theta
        # into the RK1-frozen ``t_tendf`` lane.  The single-domain real-case
        # loader has selected this source-leaf path by default since v0.14;
        # live nesting must not silently fall back to the legacy step-entry
        # Euler increment.  Keep the same explicit rollback used there.
        rad_rk_tendf=(
            0 if os.environ.get("GPUWRF_PHYS_RK_TENDF", "1") == "0" else 1
        ),
        # v0.20 S4: production opt-in for perturbation-authoritative mixed fp32.
        # Unset remains fp64_default; invalid strings fail closed in
        # OperationalNamelist.__post_init__.
        acoustic_precision_mode=os.environ.get("GPUWRF_ACOUSTIC_PRECISION_MODE", "fp64_default"),
        # The standalone daily path already binds these per-domain WRF
        # diffusion controls.  The nested path previously fell through to the
        # OperationalNamelist 0/0 defaults even when namelist.input selected
        # diff_opt=1/km_opt=4 on every domain, silently removing WRF's RK1
        # forward horizontal-diffusion bundle from production nests.
        diff_opt=int(diff_opt),
        km_opt=int(km_opt),
    )
    if parent_dt_s is not None:
        # This coherent WRF live-child boundary path completed its correctness
        # gates and is the production configuration.  It was originally wired
        # default-off while still a candidate, which left clean CLI/validation
        # launches on the obsolete boundary path unless every supervisor
        # remembered an environment opt-in.  Keep an exact explicit rollback,
        # but make a fresh production-facing invocation select the accepted
        # configuration by default.
        nested_frozen_bundle = (
            os.environ.get("GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE", "1") == "1"
        )
        namelist = with_live_child_boundary_config(
            namelist,
            parent_dt_s=float(parent_dt_s),
            nested_ph_relax=True,
            # Match the v0.11.0 validated wiring: w is in the package, but in-loop
            # w relaxation stays deferred to a longer stability gate.
            nested_w_relax=False,
            nested_ph_spec=True,
            nested_frozen_wrf_boundary_bundle=nested_frozen_bundle,
        )
    namelist = dataclass_replace(
        namelist, cu_physics=int(cu_physics),
        radiation_interval_s=float(radt_minutes) * 60.0,
        cumulus_cadence_steps=_radiation_cadence_steps(dt_s, cudt_minutes),
        cudt_minutes=float(cudt_minutes),
    )
    return namelist


def _root_boundary_cadence_override(
    namelist: OperationalNamelist, case_metadata: dict[str, Any], wrf_namelist: dict | None = None
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
            # WRF &bdy_control (Registry default .false.): without them the
            # specified root relaxes only QV; other moist/scalars flow_dep_bdy.
            have_bcs_moist=_root_bdy_flag(wrf_namelist, "have_bcs_moist"),
            have_bcs_scalar=_root_bdy_flag(wrf_namelist, "have_bcs_scalar"),
        ),
    )


def _root_bdy_flag(wrf_namelist: dict | None, key: str) -> bool:
    """Root (first-domain) value of a WRF &bdy_control logical; Registry default False."""

    raw = ((wrf_namelist or {}).get("bdy_control") or {}).get(key, False)
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else False
    return bool(raw)


def _domain_float(run, group: str, key: str, domain: str, default: float = 0.0) -> float:
    """Read WRF's scalar or per-domain REAL namelist control."""
    raw = run.namelist.get(group, {}).get(key, default)
    index = max(int(domain[1:]) - 1, 0) if isinstance(raw, (list, tuple)) else 0
    return float(_domain_list_value(raw, index, default))


def _domain_int(run, group: str, key: str, domain: str, default: int = 0) -> int:
    """Per-domain integer namelist value from ``group`` (max-dom list or scalar)."""

    raw = run.namelist.get(group, {}).get(key, default)
    index = max(int(domain[1:]) - 1, 0) if isinstance(raw, (list, tuple)) else 0
    return int(_domain_list_value(raw, index, default))


def _domain_top_lid(run, domain: str) -> bool:
    """Resolve WRF's per-domain logical, retaining defaults for omitted slots."""

    raw = run.namelist.get("dynamics", {}).get("top_lid", False)
    values = raw if isinstance(raw, (list, tuple)) else (raw,)
    index = int(domain[1:]) - 1
    value = values[index] if index < len(values) else None
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{domain}: top_lid must be a Fortran logical, got {value!r}")
    return value


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
    *,
    _init_kernel_call=None,
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

    if _init_kernel_call is None:
        from gpuwrf.integration.init_kernels import InitKernelCache
        _init_kernel_call = InitKernelCache()
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
                live_nest_child_opens_at_start=any(
                    edge.child in names and wrf_nest_opens_with_parent_start(run, name, edge.child)
                    for edge in hierarchy.children(name)
                ),
            )
            parent_dt = dt_by_domain[parent]
        loaded_cases[name] = case

        slope_rad = _domain_physics_int(run, "slope_rad", name, 0)
        radiation_static = None
        try:
            radiation_static, _ = load_radiation_static(
                case.run, name, grid=case.grid, metrics=case.metrics
            )
        except Exception as exc:  # noqa: BLE001 -- required when slope radiation is on.
            if slope_rad == 1:
                raise RuntimeError(f"{name}: slope_rad=1 requires radiation static") from exc
            radiation_static = None
        if slope_rad == 1 and radiation_static is None:
            raise RuntimeError(f"{name}: slope_rad=1 requires radiation static")

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
            if _resolve_full_wrfout_variables():
                # These options require bucket counters / diagnostic time-window
                # resets. Never silently publish unbucketed or unreset sums.
                for group, option, default in (("physics", "bucket_j", -1), ("physics", "bucket_mm", -1),
                                                ("noah_mp", "noahmp_acc_dt", 0)):
                    value = _domain_float(run, group, option, name, default)
                    if value > 0:
                        raise ValueError(f"full land history requires inactive {option}; got {value}")
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
        if name != names[0]:
            state = _two_time_live_nest_bdy(state)
        namelist = _make_namelist(
            grid=case.grid,
            tendencies=case.tendencies,
            metrics=case.metrics,
            dt_s=dt_by_domain[name],
            parent_dt_s=parent_dt,
            run_start=run_start,
            radiation_static=radiation_static,
            cu_physics=_domain_cu_physics(run, name),
            time_step_sound=_domain_int(run, "dynamics", "time_step_sound", name, 0),
            top_lid=_domain_top_lid(run, name),
            radt_minutes=_domain_float(run, "physics", "radt", name, 30.0),
            cudt_minutes=_domain_float(run, "physics", "cudt", name, 0.0),
            gwd_opt=gwd_opt,
            gwdo_statics=gwdo_statics,
            diff_opt=_domain_int(run, "dynamics", "diff_opt", name, 0),
            km_opt=_domain_int(run, "dynamics", "km_opt", name, 0),
            h_sca_adv_order=_domain_int(
                run, "dynamics", "h_sca_adv_order", name, 5
            ),
            moist_adv_opt=_domain_int(
                run, "dynamics", "moist_adv_opt", name, 0
            ),
            scalar_adv_opt=_domain_int(
                run, "dynamics", "scalar_adv_opt", name, 0
            ),
            topo_shading=_domain_physics_int(run, "topo_shading", name, 0),
            slope_rad=slope_rad,
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
            namelist = _root_boundary_cadence_override(namelist, case.metadata, run.namelist)
        from gpuwrf.io.lower_boundary import load_lower_boundary
        lower_boundary = load_lower_boundary(
            run_dir, run.namelist, name,
            run_start=run_start, dt_s=dt_by_domain[name], shape=state.t_skin.shape,
        )
        if lower_boundary is not None:
            namelist = dataclass_replace(namelist, lower_boundary=lower_boundary)
        if parent_dt is not None and _coupled_forcedown_enabled(namelist):
            # Registry moist/scalar force-down records must exist before the
            # child carry is frozen for compilation or restart.  The first
            # parent force-down replaces these unread two-time templates; the
            # predicate is the edge's own coupled_forcedown switch, so legacy
            # (uncoupled) packages never see zero records.
            state = initialize_child_scalar_boundaries(state)
        # Initial carry: identical to the domain-tree cold start for the bulk path
        # (same _initial_carry_for_run on the same state/namelist); under Noah-MP the
        # prognostic land carry plus the REAL t=0 held surface radiation are seeded
        # NOW so the scan carry pytree is structurally stable from step 1 (mirrors
        # the proven s6b/TOST carry seeding; nocturnal LWDN cold-start mitigation).
        carry = _initial_carry_for_run(state, namelist)
        if noahmp_land is not None:
            initial_surface_rad, initial_full_rad = noahmp_initial_rad(
                carry.state, namelist, land_state=noahmp_land, _with_diagnostics=True,
                _kernel_call=_init_kernel_call,
            )
            carry = carry.replace(
                noahmp_land=noahmp_land, noahmp_rad=initial_surface_rad,
                radiation_diagnostics=(initial_full_rad if carry.radiation_diagnostics is not None else None),
            )
            from gpuwrf.diagnostics.census import count_work
            carry = carry.replace(census=count_work(carry.census, "radiation_init_calls"))
            from gpuwrf.runtime.operational_state import (  # noqa: PLC0415
                history_instep_diag_enabled,
                seed_history_diagnostics,
            )
            if history_instep_diag_enabled():
                carry = carry.replace(history_diagnostics=seed_history_diagnostics(carry.state))
        from gpuwrf.kernels.dyn_carry_fp32 import real_carry
        carry = real_carry(carry)  # GPUWRF_CARRY_REAL_ALL: post-seed held radiation as REAL
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
                "top_lid": bool(namelist.top_lid),
                "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
                "boundary_update_cadence_s": float(namelist.boundary_config.update_cadence_s),
                "nested_frozen_wrf_boundary_bundle": bool(
                    namelist.boundary_config.nested_frozen_wrf_boundary_bundle
                ),
                "cu_physics": int(namelist.cu_physics),
                **_cumulus_cadence_metadata(namelist),
                "radiation_static_loaded": radiation_static is not None,
                "gwd_opt": int(namelist.gwd_opt),
                "moist_adv_opt": int(namelist.moist_adv_opt),
                "scalar_adv_opt": int(namelist.scalar_adv_opt),
                "gwdo_statics_loaded": namelist.gwdo_statics is not None,
                # Effective terrain-radiation controls of this domain (twin-delivery evidence; metadata only).
                "topo_shading": int(namelist.topo_shading),
                "slope_rad": int(namelist.slope_rad),
            },
            "namelist_resolved": _resolved_namelist_options(namelist, case.grid),
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


def _resolve_batch_input_dirs(config: NestedPipelineConfig, batch_size: int) -> tuple[Path, ...]:
    if int(batch_size) <= 1:
        return (Path(config.input_dir),)
    raw = os.environ.get("GPUWRF_BATCH_INPUT_DIRS", "").strip()
    if raw:
        # The B distinct-init dirs may be separated by a comma, a newline, or the OS
        # path separator (":"). Operators reach for commas first, so accept all three
        # (this stays backward-compatible with the original ":"-only contract).
        import re  # noqa: PLC0415

        sep = r"[,\n" + re.escape(os.pathsep) + r"]"
        items = [item.strip() for item in re.split(sep, raw) if item.strip()]
        paths = tuple(Path(item) for item in items)
    elif config.batch_input_dirs is not None:
        paths = tuple(Path(path) for path in config.batch_input_dirs)
    else:
        paths = tuple(Path(config.input_dir) for _ in range(int(batch_size)))
    if len(paths) != int(batch_size):
        raise ValueError(
            f"GPUWRF_BATCH_ENSEMBLE={int(batch_size)} requires exactly {int(batch_size)} "
            f"input dirs in GPUWRF_BATCH_INPUT_DIRS (separate them with ',' or ':'), "
            f"but parsed {len(paths)}: {[str(p) for p in paths]}. Example: "
            f"GPUWRF_BATCH_INPUT_DIRS='/case/day1,/case/day2,...' — one dir per lane, "
            f"all the same grid/physics (only the initial/boundary day differs)."
        )
    return paths


def _leaf_signature(tree: Any) -> tuple[tuple[tuple[int, ...], str], ...]:
    import jax  # noqa: PLC0415

    sig = []
    for leaf in jax.tree_util.tree_leaves(tree):
        shape = tuple(int(v) for v in getattr(leaf, "shape", ()))
        dtype = str(getattr(leaf, "dtype", type(leaf).__name__))
        sig.append((shape, dtype))
    return tuple(sig)


def _grid_shape_signature(grid: Any) -> tuple[Any, ...]:
    return (
        int(getattr(grid, "nx", 0)),
        int(getattr(grid, "ny", 0)),
        int(getattr(grid, "nz", 0)),
        int(getattr(grid, "halo_width", 0)),
        getattr(getattr(grid, "projection", None), "kind", None),
        float(getattr(getattr(grid, "projection", None), "dx_m", 0.0)),
        float(getattr(getattr(grid, "projection", None), "dy_m", 0.0)),
    )


def _namelist_homogeneous_signature(namelist: Any) -> tuple[Any, ...]:
    fields = (
        "dt_s",
        "acoustic_substeps",
        "rk_order",
        "radiation_cadence_steps",
        "mp_physics",
        "bl_pbl_physics",
        "sf_sfclay_physics",
        "cu_physics",
        "sf_surface_physics",
        "ra_sw_physics",
        "ra_lw_physics",
        "use_noahmp",
        "gwd_opt",
        "rad_rk_tendf",
        "acoustic_precision_mode",
    )
    return tuple(getattr(namelist, field, None) for field in fields)


_BATCH_CANONICAL_NAMELIST_FIELDS = (
    "grid",
    "tendencies",
    "metrics",
    "radiation_static",
    "gwdo_statics",
    "data_assimilation",
    "noahmp_static",
    "noahmp_energy_params",
    "noahmp_rad_params",
    "noahmp_land",
    "noahclassic_static",
    "noahclassic_land",
    "noahclassic_rad",
    "slab_static",
    "slab_land",
    "slab_rad",
    "px_static",
    "px_land",
    "px_rad",
)


def _same_static_value(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return left is None and right is None
    try:
        from gpuwrf.runtime.aot_cheap_key import canonical_digest

        return canonical_digest(left) == canonical_digest(right)
    except Exception:  # noqa: BLE001 - leave uncanonicalized so the assertion fails closed.
        return False


def _canonicalize_batch_namelist_static(
    reference: OperationalNamelist,
    candidate: OperationalNamelist,
) -> OperationalNamelist:
    """Share reference static objects when a lane's values are byte-identical.

    JAX treedef equality keys static aux by Python equality.  Some production
    static bundles deliberately use identity equality inside ``_StaticHolder`` to
    keep normal jit cache keys conservative, so two separately loaded identical
    lanes otherwise fail the raw treedef check.  Canonicalize only after a
    deterministic by-value digest match; genuine static differences stay distinct
    and the downstream treedef assertion remains fail-closed.
    """

    updates: dict[str, Any] = {}
    for field in _BATCH_CANONICAL_NAMELIST_FIELDS:
        ref_value = getattr(reference, field, None)
        cand_value = getattr(candidate, field, None)
        if ref_value is cand_value:
            continue
        if _same_static_value(ref_value, cand_value):
            updates[field] = ref_value
    if not updates:
        return candidate
    return dataclass_replace(candidate, **updates)


def _canonicalize_batch_bundles(
    *,
    names: tuple[str, ...],
    reference: dict[str, DomainBundle],
    candidate: dict[str, DomainBundle],
) -> dict[str, DomainBundle]:
    updated = dict(candidate)
    for name in names:
        bundle = candidate[name]
        namelist = _canonicalize_lane_value_tolerant(
            reference[name].namelist,
            bundle.namelist,
        )
        namelist = _canonicalize_batch_namelist_static(
            reference[name].namelist,
            namelist,
        )
        if namelist is not bundle.namelist:
            updated[name] = dataclass_replace(bundle, namelist=namelist)
    return updated


def _assert_homogeneous_batch(
    *,
    names: tuple[str, ...],
    reference: tuple[DomainHierarchy, dict[str, DomainBundle], dict[str, float], dict[str, Any]],
    candidate: tuple[DomainHierarchy, dict[str, DomainBundle], dict[str, float], dict[str, Any]],
    lane: int,
) -> None:
    import jax  # noqa: PLC0415

    ref_hierarchy, ref_bundles, ref_dt, ref_carries = reference
    hierarchy, bundles, dt_by_domain, carries = candidate
    if hierarchy != ref_hierarchy:
        raise ValueError(f"batch lane {lane}: domain hierarchy/nesting differs")
    if set(dt_by_domain) != set(ref_dt):
        raise ValueError(f"batch lane {lane}: dt domain set differs")
    for name in names:
        if float(dt_by_domain[name]) != float(ref_dt[name]):
            raise ValueError(f"batch lane {lane} {name}: dt differs")
        if _grid_shape_signature(bundles[name].grid) != _grid_shape_signature(ref_bundles[name].grid):
            raise ValueError(f"batch lane {lane} {name}: grid shape/projection differs")
        if jax.tree_util.tree_structure(bundles[name].namelist) != jax.tree_util.tree_structure(
            ref_bundles[name].namelist
        ):
            raise ValueError(f"batch lane {lane} {name}: namelist/static treedef differs")
        if _namelist_homogeneous_signature(bundles[name].namelist) != _namelist_homogeneous_signature(
            ref_bundles[name].namelist
        ):
            raise ValueError(f"batch lane {lane} {name}: physics suite/cadence differs")
        if jax.tree_util.tree_structure(carries[name]) != jax.tree_util.tree_structure(ref_carries[name]):
            raise ValueError(f"batch lane {lane} {name}: carry treedef differs")
        if _leaf_signature(carries[name]) != _leaf_signature(ref_carries[name]):
            raise ValueError(f"batch lane {lane} {name}: carry leaf shapes/dtypes differ")


def _stack_batched_tree(*items: Any) -> Any:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *items)


def _slice_batched_tree(tree: Any, lane: int) -> Any:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    def take(leaf: Any) -> Any:
        if hasattr(leaf, "shape") and len(getattr(leaf, "shape", ())) > 0:
            return jnp.asarray(leaf)[int(lane)]
        return leaf

    return jax.tree_util.tree_map(take, tree)


# Tenerife B=4 batch extension: NoahMPStatic slots that legitimately differ
# ACROSS lanes (they are per-case wrfinput data: TMN->tbot deep-soil BC,
# VEGFRA->shdfac/shdmax green-vegetation fraction) but must NOT discriminate the
# batch namelist treedef. They are extracted per lane BEFORE canonicalization and
# threaded as TRACED per-lane clock-base leaves (runtime.operational_mode).
_NOAHMP_LANE_OVERRIDE_SLOTS = ("tbot", "shdfac", "shdmax")


def _noahmp_lane_arrays(bundle: Any) -> tuple[Any, Any, Any] | None:
    """Extract the ``(tbot, shdfac, shdmax)`` arrays from one lane's bundle.

    Returns ``None`` when the domain runs without Noah-MP or the static is
    absent/all-None, so B-runs without Noah-MP keep the previous clock shape.
    """

    namelist = bundle.namelist
    if not bool(getattr(namelist, "use_noahmp", False)):
        return None
    static = getattr(namelist, "noahmp_static", None)
    if static is None:
        return None
    import jax.numpy as jnp  # noqa: PLC0415

    arrays = []
    for slot in _NOAHMP_LANE_OVERRIDE_SLOTS:
        value = getattr(static, slot, None)
        arrays.append(None if value is None else jnp.asarray(value))
    if all(arr is None for arr in arrays):
        return None
    return tuple(arrays)


def _terrain_provenance_equivalent(left: Any, right: Any) -> bool:
    """True when two TerrainProvenance records match ignoring PATH metadata.

    ``source_path`` embeds the per-lane input directory (lane00/lane01/...) and
    ``sha256`` labels the manifest it came from; neither is read by any kernel.
    Every other provenance field (shape, units, transform, max elevation,
    coastline check) and the terrain DATA itself (compared by the grid digest)
    must match.
    """

    if left is right:
        return True
    if left is None or right is None:
        return left is None and right is None
    for field in (
        "shape",
        "units",
        "projection_transform",
        "max_elevation_m",
        "coastline_sanity_check_passed",
    ):
        if getattr(left, field, None) != getattr(right, field, None):
            return False
    return True


def _grid_metrics_provenance_blind(metrics: Any) -> Any:
    """Return ``metrics`` with its provenance STRING neutralized for digesting.

    ``DycoreMetrics.provenance`` embeds the input FILE PATH (lane-specific);
    every metric ARRAY is grid data. Kernels never read the string.
    """

    if metrics is None:
        return None
    try:
        import dataclasses

        return dataclasses.replace(metrics, provenance="")
    except Exception:  # noqa: BLE001 -- fail closed: keep the raw object.
        return metrics


def _grids_equal_modulo_provenance(left: Any, right: Any) -> bool:
    """True when two GridSpecs differ ONLY in provenance PATH metadata.

    Tolerated: ``terrain`` path fields and ``metrics.provenance`` (both embed the
    per-lane input path; neither is read by any kernel). Field comparison is
    digest-based (GridSpec carries array fields where a bare ``!=`` returns an
    elementwise array, not a bool).
    """

    if left is right:
        return True
    try:
        from dataclasses import fields as dc_fields

        from gpuwrf.runtime.aot_cheap_key import canonical_digest

        if type(left) is not type(right):
            return False
        for field in dc_fields(type(left)):
            if field.name == "terrain":
                continue
            lv = getattr(left, field.name, None)
            rv = getattr(right, field.name, None)
            if field.name == "metrics":
                lv = _grid_metrics_provenance_blind(lv)
                rv = _grid_metrics_provenance_blind(rv)
            if canonical_digest(lv) != canonical_digest(rv):
                return False
        return _terrain_provenance_equivalent(left.terrain, right.terrain)
    except Exception:  # noqa: BLE001 -- fail closed: leave uncanonicalized.
        return False


def _noahmp_statics_equal_modulo_lane_slots(left: Any, right: Any) -> bool:
    """True when two NoahMPStatics differ ONLY in the per-lane override slots.

    Every non-override slot must digest-match (categories, soil geometry,
    parameter tables...); override slots only need matching dtype/shape (their
    values are the per-lane physics data that ride the clock base instead).
    """

    if left is right:
        return True
    if left is None or right is None:
        return left is None and right is None
    if type(left) is not type(right):
        return False
    try:
        from gpuwrf.runtime.aot_cheap_key import canonical_digest

        for slot in type(left).__slots__:
            lv = getattr(left, slot, None)
            rv = getattr(right, slot, None)
            if slot in _NOAHMP_LANE_OVERRIDE_SLOTS:
                if (lv is None) != (rv is None):
                    return False
                if lv is None:
                    continue
                import numpy as np  # noqa: PLC0415

                if (
                    np.asarray(lv).shape != np.asarray(rv).shape
                    or np.asarray(lv).dtype != np.asarray(rv).dtype
                ):
                    return False
                continue
            if canonical_digest(lv) != canonical_digest(rv):
                return False
        return True
    except Exception:  # noqa: BLE001 -- fail closed: leave uncanonicalized.
        return False


def _canonicalize_lane_value_tolerant(reference: Any, candidate: Any) -> Any:
    """Neutralize lane-varying METADATA in a candidate namelist's statics.

    Two differences are tolerated between batch lanes (Tenerife B=4, 2026-09-18):
      1. ``GridSpec`` terrain provenance PATH metadata (lane00/ vs lane01/ ...);
      2. ``NoahMPStatic`` per-lane arrays {tbot, shdfac, shdmax} (extracted into
         the per-lane clock base by the caller BEFORE this canonicalization, so
         replacing the static with the reference's loses nothing).
    Everything else must already match -- genuine physics/static differences keep
    the treedef distinct and the downstream treedef assertion still fails closed.
    """

    import dataclasses

    updates: dict[str, Any] = {}
    ref_grid = getattr(reference, "grid", None)
    cand_grid = getattr(candidate, "grid", None)
    if cand_grid is not None and ref_grid is not None and cand_grid != ref_grid:
        if _grids_equal_modulo_provenance(cand_grid, ref_grid):
            try:
                cand_metrics = cand_grid.metrics
                ref_metrics = ref_grid.metrics
                if (
                    cand_metrics is not None
                    and ref_metrics is not None
                    and getattr(cand_metrics, "provenance", None)
                    != getattr(ref_metrics, "provenance", None)
                ):
                    cand_metrics = dataclasses.replace(
                        cand_metrics, provenance=ref_metrics.provenance
                    )
                updates["grid"] = dataclasses.replace(
                    cand_grid,
                    terrain=dataclasses.replace(
                        cand_grid.terrain,
                        source_path=ref_grid.terrain.source_path,
                        sha256=ref_grid.terrain.sha256,
                    ),
                    metrics=cand_metrics,
                )
            except Exception:  # noqa: BLE001 -- fail closed, keep the raw grid.
                pass
    ref_static = getattr(reference, "noahmp_static", None)
    cand_static = getattr(candidate, "noahmp_static", None)
    if cand_static is not None and ref_static is not None and cand_static is not ref_static:
        if _noahmp_statics_equal_modulo_lane_slots(ref_static, cand_static):
            updates["noahmp_static"] = ref_static
    # ``metrics`` is a CHILD pytree of the namelist whose DycoreMetrics aux is the
    # provenance STRING (embeds the per-lane input path); arrays are grid data.
    ref_metrics = getattr(reference, "metrics", None)
    cand_metrics = getattr(candidate, "metrics", None)
    if cand_metrics is not None and ref_metrics is not None and cand_metrics is not ref_metrics:
        ref_blind = _grid_metrics_provenance_blind(ref_metrics)
        cand_blind = _grid_metrics_provenance_blind(cand_metrics)
        try:
            from gpuwrf.runtime.aot_cheap_key import canonical_digest

            if canonical_digest(ref_blind) == canonical_digest(cand_blind):
                updates["metrics"] = ref_metrics
        except Exception:  # noqa: BLE001 -- fail closed: leave unshared.
            pass
    if not updates:
        return candidate
    return dataclasses.replace(candidate, **updates)


def _load_batched_domains(
    config: NestedPipelineConfig,
    names: tuple[str, ...],
    *,
    batch_size: int,
) -> tuple[
    DomainHierarchy,
    dict[str, DomainBundle],
    dict[str, Any],
    datetime,
    dict[str, float],
    dict[str, Any],
    dict[str, tuple[OperationalNamelist, ...]],
    tuple[Path, ...],
    tuple[datetime, ...],
    dict[str, tuple[tuple[Any, Any, Any] | None, ...]],
]:
    input_dirs = _resolve_batch_input_dirs(config, batch_size)
    loaded = []
    loaded_by_path = {}
    for path in input_dirs:
        resolved_path = Path(path).resolve()
        if resolved_path not in loaded_by_path:
            lane_config = dataclass_replace(config, input_dir=Path(path), batch_input_dirs=None)
            loaded_by_path[resolved_path] = _load_domains(lane_config, names)
        loaded.append(loaded_by_path[resolved_path])

    ref_hierarchy, ref_bundles, ref_meta, ref_run_start, ref_dt, ref_carries = loaded[0]
    # Tenerife B=4: extract the genuinely per-lane Noah-MP arrays (tbot/shdfac/
    # shdmax) BEFORE canonicalization shares the reference statics -- the lane
    # values must survive to feed the per-lane clock bases.
    batch_noahmp_arrays: dict[str, tuple[tuple[Any, Any, Any] | None, ...]] = {}
    for name in names:
        batch_noahmp_arrays[name] = tuple(
            _noahmp_lane_arrays(lane[1][name]) for lane in loaded
        )
    canonical_loaded = [loaded[0]]
    for lane, (hierarchy, bundles, meta, run_start, dt_by_domain, carries) in enumerate(
        loaded[1:], start=1
    ):
        bundles = _canonicalize_batch_bundles(
            names=names,
            reference=ref_bundles,
            candidate=bundles,
        )
        _assert_homogeneous_batch(
            names=names,
            reference=(ref_hierarchy, ref_bundles, ref_dt, ref_carries),
            candidate=(hierarchy, bundles, dt_by_domain, carries),
            lane=lane,
        )
        canonical_loaded.append((hierarchy, bundles, meta, run_start, dt_by_domain, carries))
    loaded = canonical_loaded

    batched_carries = {
        name: _stack_batched_tree(*(lane[5][name] for lane in loaded))
        for name in names
    }
    batch_namelists = {
        name: tuple(lane[1][name].namelist for lane in loaded)
        for name in names
    }
    meta = dict(ref_meta)
    meta["batch_ensemble"] = {
        "enabled": True,
        "batch_size": int(batch_size),
        "input_dirs": [str(path) for path in input_dirs],
        "run_starts": [lane[3].isoformat() for lane in loaded],
        "mode": "homogeneous_outer_vmap",
    }
    return (
        ref_hierarchy,
        ref_bundles,
        meta,
        ref_run_start,
        ref_dt,
        batched_carries,
        batch_namelists,
        input_dirs,
        tuple(lane[3] for lane in loaded),
        batch_noahmp_arrays,
    )


def _clock_bases_for_batch_domain(
    *,
    tree: DomainTree,
    name: str,
    batch_namelists: dict[str, tuple[OperationalNamelist, ...]],
    batch_size: int,
    noahmp_lane_arrays: tuple[tuple[Any, Any, Any] | None, ...] | None = None,
) -> Any:
    if int(batch_size) <= 1:
        return build_clock_base(tree.domains[name].namelist)
    namelists = batch_namelists.get(name)
    if namelists is None:
        raise ValueError(f"{name}: missing batch namelists for B={int(batch_size)}")
    if len(namelists) != int(batch_size):
        raise ValueError(
            f"{name}: expected {int(batch_size)} batched namelists, got {len(namelists)}"
        )
    clock_bases = []
    for lane, namelist in enumerate(namelists):
        arrays = None if noahmp_lane_arrays is None else noahmp_lane_arrays[lane]
        clock_bases.append(build_clock_base(namelist, noahmp_static_arrays=arrays))
    return _stack_batched_tree(*clock_bases)


def _batch_subchunk_size(batch_size: int) -> int:
    """Sub-batch width for the outer vmap (Tenerife B=4 VRAM fix, 2026-09-18).

    The batched fused executable's device TEMP scales ~linearly with B (measured:
    B=1 fits, B=4 requests a single ~28.8 GiB temp on the 32 GB RTX 5090 and OOMs
    at execution). ``GPUWRF_BATCH_SUBCHUNK=G`` runs the SAME batched vmap in
    ceil(B/G) device groups of G lanes: per-lane numerics are unchanged (vmap is
    lane-independent, so results are bitwise-equal to the single-group run); only
    peak VRAM and launch granularity change. Default: full batch in one group.
    """

    raw = os.environ.get("GPUWRF_BATCH_SUBCHUNK", "").strip()
    if not raw:
        return int(batch_size)
    try:
        group = int(raw)
    except ValueError as exc:
        raise ValueError(f"GPUWRF_BATCH_SUBCHUNK must be a positive integer; got {raw!r}") from exc
    if group < 1:
        raise ValueError(f"GPUWRF_BATCH_SUBCHUNK must be a positive integer; got {raw!r}")
    return min(group, int(batch_size))


def _slice_leaves(tree: Any, start: int, stop: int) -> Any:
    import jax  # noqa: PLC0415

    return jax.tree_util.tree_map(
        lambda leaf: leaf[start:stop] if hasattr(leaf, "shape") and len(leaf.shape) > 0 else leaf,
        tree,
    )


def _concat_leaves(*trees: Any) -> Any:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *trees)


def _batched_advance_factory(
    *,
    tree: DomainTree,
    batch_namelists: dict[str, tuple[OperationalNamelist, ...]],
    batch_size: int,
    batch_noahmp_arrays: dict[str, tuple[tuple[Any, Any, Any] | None, ...]] | None = None,
):
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    clock_bases = {
        name: _clock_bases_for_batch_domain(
            tree=tree,
            name=name,
            batch_namelists=batch_namelists,
            batch_size=int(batch_size),
            noahmp_lane_arrays=None if batch_noahmp_arrays is None else batch_noahmp_arrays.get(name),
        )
        for name in tree.domains
    }

    def advance(name: str, carry: Any, start_step: int, n_steps: int) -> Any:
        namelist = tree.domains[name].namelist
        cadence = int(namelist.radiation_cadence_steps)
        start = jnp.asarray(int(start_step), dtype=jnp.int32)
        if int(batch_size) <= 1:
            return _advance_chunk(
                carry,
                namelist,
                start,
                clock_bases[name],
                n_steps=int(n_steps),
                cadence=cadence,
            )
        group = _batch_subchunk_size(int(batch_size))

        def run_group(group_carry: Any, group_clock: Any) -> Any:
            return jax.vmap(
                lambda lane_carry, lane_clock_base: _advance_chunk(
                    lane_carry,
                    namelist,
                    start,
                    lane_clock_base,
                    n_steps=int(n_steps),
                    cadence=cadence,
                ),
                in_axes=(0, 0),
                out_axes=0,
            )(group_carry, group_clock)

        if group >= int(batch_size):
            return run_group(carry, clock_bases[name])
        results = []
        for lo in range(0, int(batch_size), group):
            hi = min(lo + group, int(batch_size))
            results.append(run_group(_slice_leaves(carry, lo, hi), _slice_leaves(clock_bases[name], lo, hi)))
        return _concat_leaves(*results)

    return advance


def _batched_force(edge, parent: Any, child: Any, *, batch_size: int) -> Any:
    import jax  # noqa: PLC0415

    child_state = child.state
    # Mirror _operational_force's edge dispatch EXACTLY (Tenerife B=4 fix,
    # 2026-09-18): when the edge is coupled_forcedown (live-nest child with
    # nested_frozen_wrf_boundary_bundle + force_geopotential=False -- the
    # production default), the boundary package MUST be built with the coupled
    # signature (parent/child metrics + parent_grid_ratio). The batched path
    # previously ALWAYS used the legacy package, silently diverging the child
    # LBC from the B=1 runtime (~50x worse GPU-vs-CPU deltas, growing with
    # lead). The extra arguments are lane-INVARIANT edge statics, so the vmap
    # broadcasts them unchanged.
    coupled = bool(getattr(edge, "coupled_forcedown", False))
    if int(batch_size) <= 1:
        bdy_width = int(child_state.u_bdy.shape[2])
        forced_state = build_child_boundary_package(
            child_state,
            parent.state,
            edge.weights,
            bdy_width=bdy_width,
            **(
                {
                    "parent_metrics": edge.parent_metrics,
                    "child_metrics": edge.child_metrics,
                    "coupled_forcedown": True,
                    "parent_grid_ratio": int(edge.parent_grid_ratio),
                }
                if coupled
                else {}
            ),
        )
    else:
        bdy_width = int(child_state.u_bdy.shape[3])
        forced_state = jax.vmap(
            lambda lane_child_state, lane_parent_state: build_child_boundary_package(
                lane_child_state,
                lane_parent_state,
                edge.weights,
                bdy_width=bdy_width,
                **(
                    {
                        "parent_metrics": edge.parent_metrics,
                        "child_metrics": edge.child_metrics,
                        "coupled_forcedown": True,
                        "parent_grid_ratio": int(edge.parent_grid_ratio),
                    }
                    if coupled
                    else {}
                ),
            ),
            in_axes=(0, 0),
            out_axes=0,
        )(child_state, parent.state)
    return child.replace(state=forced_state)


def run_batched_operational_domain_tree(
    tree: DomainTree,
    *,
    batch_namelists: dict[str, tuple[OperationalNamelist, ...]],
    batch_size: int,
    root_steps: int,
    root: str | None = None,
    feedback_enabled: bool | None = None,
    adaptive_dt: Any | None = None,
    move: Any | None = None,
    output: Any | None = None,
    output_cadence_steps: dict[str, int] | None = None,
    output_alarm_steps: dict[str, tuple[int, ...]] | None = None,
    block_between: bool = True,
    root_sync_cadence: int | None = None,
    carries: dict[str, Any] | None = None,
    initial_own_steps: dict[str, int] | None = None,
    max_event_tail: int | None = None,
    batch_noahmp_arrays: dict[str, tuple[tuple[Any, Any, Any] | None, ...]] | None = None,
) -> DomainTreeResult:
    """Run the opt-in F1 B>1 vmap path without changing the B=1 runtime."""

    if int(batch_size) <= 1:
        return run_operational_domain_tree(
            tree,
            root_steps=root_steps,
            root=root,
            feedback_enabled=feedback_enabled,
            adaptive_dt=adaptive_dt,
            move=move,
            output=output,
            output_cadence_steps=output_cadence_steps,
            output_alarm_steps=output_alarm_steps,
            block_between=block_between,
            root_sync_cadence=root_sync_cadence,
            carries=carries,
            initial_own_steps=initial_own_steps,
            max_event_tail=max_event_tail,
        )
    if carries is None:
        raise ValueError("B>1 run requires pre-stacked batched carries from _load_batched_domains")

    for name, bundle in tree.domains.items():
        _resolve_operational_suite(bundle.namelist)
        if int(bundle.namelist.rk_order) != 3:
            raise ValueError(f"{name}: operational nesting currently supports RK3 only")

    effective_feedback = tree.feedback_enabled if feedback_enabled is None else bool(feedback_enabled)
    if effective_feedback:
        raise ValueError("GPUWRF_BATCH_ENSEMBLE>1 currently supports one-way nesting only")

    edge_by_pair = {
        (edge.parent, edge.child): edge
        for edges in tree.edges.values()
        for edge in edges
    }

    def lookup(spec):
        return edge_by_pair[(spec.parent, spec.child)]

    return run_domain_tree_callbacks(
        tree.hierarchy,
        dict(carries),
        root_steps=int(root_steps),
        advance=_batched_advance_factory(
            tree=tree,
            batch_namelists=batch_namelists,
            batch_size=int(batch_size),
            batch_noahmp_arrays=batch_noahmp_arrays,
        ),
        force=lambda edge, parent, child: _batched_force(
            edge,
            parent,
            child,
            batch_size=int(batch_size),
        ),
        feedback=None,
        root=root,
        feedback_enabled=False,
        adaptive_dt=adaptive_dt,
        move=move,
        output=output,
        output_cadence_steps=output_cadence_steps,
        output_alarm_steps=output_alarm_steps,
        block_between=block_between,
        root_sync_cadence=root_sync_cadence,
        edge_lookup=lookup,
        fused_cascade=None,
        initial_own_steps=initial_own_steps,
        max_event_tail=max_event_tail,
    )


def _nested_m9_radiation_from_carry_from_env() -> bool:
    """Opt in to the nested Noah-MP carry-backed M9 output shortcut.

    Default is OFF because the carry is not a byte-identical replacement for the
    full output-time RRTMG diagnostic solve: it holds only SOLDN/LWDN/COSZ at the
    WRF radiation-held time, not the instantaneous upwelling / TOA flux slices.
    """

    raw = os.environ.get("GPUWRF_NESTED_M9_RADIATION_FROM_CARRY")
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}


def _noahmp_surface_diagnostics_from_held_radiation(
    state: Any,
    namelist: OperationalNamelist,
    run_start: datetime,
    *,
    lead_seconds: float,
    noahmp_land: Any,
    noahmp_rad: Any,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
) -> dict[str, np.ndarray] | None:
    """Build nested output diagnostics from the resident Noah-MP radiation carry.

    This is intentionally opt-in and not a full ``M9Diagnostics`` replacement.
    ``carry.noahmp_rad`` contains only held WRF-cadence ``(SOLDN, LWDN, COSZ)``.
    It can back SWDOWN/GLW, Noah-MP land HFX/LH/TSK/T2, and the directly
    equivalent bottom-down diagnostics, but it cannot provide SWUPB/LWUPB or any
    TOA flux. Those unavailable fields are omitted so the writer skips them
    instead of fabricating values.
    """

    if noahmp_rad is None:
        return None

    import jax  # noqa: PLC0415 -- lazy: keeps module import light.
    import jax.numpy as jnp  # noqa: PLC0415

    from gpuwrf.integration.daily_pipeline import (  # noqa: PLC0415
        _M9_OUTPUT_FIELDS,
        _requested_m9_output_names,
    )
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        _NoahMPClock,
        _NoahMPRadiation,
        _noahmp_params,
        _psfc_from_state,
        build_clock_base,
        overlay_noahmp_land_diagnostics,
        surface_layer_diagnostics,
    )

    clock_namelist = namelist
    if getattr(namelist, "time_utc", None) is None:
        clock_namelist = dataclass_replace(namelist, time_utc=run_start)
    clock_base = build_clock_base(clock_namelist)
    surf = surface_layer_diagnostics(state, clock_namelist.grid)

    sw_enabled = int(clock_namelist.ra_sw_physics) != 0
    lw_enabled = int(clock_namelist.ra_lw_physics) != 0
    soldn = jnp.asarray(noahmp_rad[0], dtype=jnp.float64)
    lwdn = jnp.asarray(noahmp_rad[1], dtype=jnp.float64)
    cosz = jnp.asarray(noahmp_rad[2], dtype=jnp.float64)
    if not sw_enabled:
        soldn = jnp.zeros_like(soldn)
    if not lw_enabled:
        lwdn = jnp.zeros_like(lwdn)

    clock = _NoahMPClock(julian=clock_base.noahmp_julian, yearlen=clock_base.noahmp_yearlen)
    ep, rp = _noahmp_params(clock_namelist)
    hfx, lh, tsk, t2 = overlay_noahmp_land_diagnostics(
        state,
        noahmp_land,
        clock_namelist.noahmp_static,
        surf.hfx,
        surf.lh,
        state.t_skin,
        float(clock_namelist.dt_s),
        bulk_t2=surf.t2,
        radiation=_NoahMPRadiation(soldn, lwdn, cosz),
        clock=clock,
        energy_params=ep,
        rad_params=rp,
    )

    values: dict[str, Any] = {
        "T2": t2,
        "U10": surf.u10,
        "V10": surf.v10,
        "Q2": getattr(surf, "q2", None),
        "PSFC": _psfc_from_state(state, clock_namelist.metrics),
        "SWDOWN": soldn,
        "GLW": lwdn,
        "PBLH": surf.pblh,
        "TSK": tsk,
        "HFX": hfx,
        "LH": lh,
        "LWDNB": lwdn,
        "SWNORM": soldn,
    }
    if int(clock_namelist.slope_rad) != 1:
        values["SWDNB"] = soldn

    out: dict[str, Any] = {
        # _M9_OUTPUT_FIELDS historically omits HFX/LH, but the carry-backed path
        # has the Noah-MP overlay result in hand. Returning them here lets the
        # writer override its raw fallback in the opt-in stream only.
        "HFX": hfx,
        "LH": lh,
    }
    for wrf_name, _attr in _M9_OUTPUT_FIELDS:
        value = values.get(wrf_name)
        if value is None:
            continue
        out[wrf_name] = value
    requested = _requested_m9_output_names(variable_subset)
    if requested is not None:
        requested_with_fluxes = set(requested)
        requested_with_fluxes.update({"HFX", "LH"} & set(variable_subset or ()))
        out = {name: value for name, value in out.items() if name in requested_with_fluxes}
    host_out = jax.device_get(out)
    return {name: np.asarray(value) for name, value in host_out.items()} or None


def _noahmp_surface_diagnostics_for_output(
    state: Any,
    namelist: OperationalNamelist,
    run_start: datetime,
    *,
    lead_seconds: float,
    noahmp_land: Any,
    noahmp_rad: Any,
    radiation_diagnostics: Any = None,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
    output_radiation_work: list[int] | None = None,
    history_diagnostics: Any = None,
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

    if radiation_diagnostics is None and _nested_m9_radiation_from_carry_from_env() and noahmp_rad is not None:
        return _noahmp_surface_diagnostics_from_held_radiation(
            state,
            namelist,
            run_start,
            lead_seconds=lead_seconds,
            noahmp_land=noahmp_land,
            noahmp_rad=noahmp_rad,
            variable_subset=variable_subset,
        )
    if history_diagnostics is not None:
        return _history_surface_diagnostics_for_output(
            state,
            namelist,
            run_start,
            lead_seconds=lead_seconds,
            noahmp_land=noahmp_land,
            noahmp_rad=noahmp_rad,
            radiation_diagnostics=radiation_diagnostics,
            history=history_diagnostics,
            variable_subset=variable_subset,
            output_radiation_work=output_radiation_work,
        )

    import jax  # noqa: PLC0415 -- lazy: keeps module import light (mirrors writer).

    from gpuwrf.integration.daily_pipeline import (  # noqa: PLC0415
        _M9_OUTPUT_FIELDS,
        _byte_identical_selected_m9_attrs,
        _m9_attrs_for_requested_names,
        _requested_m9_output_names,
    )
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        build_clock_base,
        compute_m9_diagnostics,
        compute_m9_selected_diagnostics,
        surface_layer_diagnostics,
    )

    clock_namelist = namelist
    if getattr(namelist, "time_utc", None) is None:
        clock_namelist = dataclass_replace(namelist, time_utc=run_start)
    requested_names = _requested_m9_output_names(variable_subset)
    selected_attrs = _byte_identical_selected_m9_attrs(requested_names)
    attrs = selected_attrs or _m9_attrs_for_requested_names(requested_names)
    m9_by_attr: dict[str, Any] = {}
    surface = None
    if attrs:
        # #91: traced per-run date scalars so the M9 diagnostic HLO is date-independent.
        clock_base = build_clock_base(clock_namelist)
        if selected_attrs is not None:
            values = compute_m9_selected_diagnostics(
                state,
                clock_namelist,
                lead_seconds,
                clock_base,
                selected_attrs,
                noahmp_land=noahmp_land,
                noahmp_rad=noahmp_rad,
                radiation_diagnostics=radiation_diagnostics,
                **({"_with_radiation_count": True} if output_radiation_work is not None else {}),
            )
            if output_radiation_work is not None:
                values, calls = values
                output_radiation_work.append(int(jax.device_get(calls)))
            m9_by_attr = dict(zip(selected_attrs, values, strict=True))
        else:
            surface = surface_layer_diagnostics(state, clock_namelist.grid)
            m9 = compute_m9_diagnostics(
                state,
                clock_namelist,
                lead_seconds,
                noahmp_land=noahmp_land,
                noahmp_rad=noahmp_rad,
                radiation_diagnostics=radiation_diagnostics,
                surface_diagnostics=surface,
                **({"_with_radiation_count": True} if output_radiation_work is not None else {}),
                clock_base=clock_base,
            )
            if output_radiation_work is not None:
                m9, calls = m9
                output_radiation_work.append(int(jax.device_get(calls)))
            m9_by_attr = {attr: getattr(m9, attr, None) for attr in attrs}
    # Q2 stays the bulk surface-layer diagnostic (matches the single-domain path).
    q2 = None
    if requested_names is None or "Q2" in requested_names:
        try:
            if surface is None:
                surface = surface_layer_diagnostics(state, clock_namelist.grid)
            q2 = getattr(surface, "q2", None)
        except Exception:  # noqa: BLE001 -- Q2 is auxiliary; the writer keeps its default.
            q2 = None
    out: dict[str, np.ndarray] = {}
    for wrf_name, attr in _M9_OUTPUT_FIELDS:
        if requested_names is not None and wrf_name not in requested_names:
            continue
        if wrf_name == "Q2":
            value = q2
        elif attr is not None:
            value = m9_by_attr.get(attr)
        else:
            value = None
        if value is None:
            continue
        out[wrf_name] = np.asarray(jax.device_get(value))
    return out or None


def _history_surface_diagnostics_for_output(
    state: Any,
    namelist: OperationalNamelist,
    run_start: datetime,
    *,
    lead_seconds: float,
    noahmp_land: Any,
    noahmp_rad: Any,
    radiation_diagnostics: Any,
    history: Any,
    variable_subset: tuple[str, ...] | frozenset[str] | None,
    output_radiation_work: list[int] | None,
) -> dict[str, np.ndarray] | None:
    """Writer surface map from the last step's in-step physics (WRF history semantics).

    WRF writes history before ``solve`` (module_integrate.F:375 vs :393), so
    T2/Q2/U10/V10/HFX/LH/PSFC are the previous step's surface_driver values held in
    ``carry.history_diagnostics``; radiation fields keep the held carry path. No
    surface-layer or Noah-MP re-solve runs here; HFX/LH replace the writer fallback.
    """

    import jax  # noqa: PLC0415 -- lazy: keeps module import light (mirrors writer).

    from gpuwrf.integration.daily_pipeline import (  # noqa: PLC0415
        _M9_OUTPUT_FIELDS,
        _requested_m9_output_names,
    )
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        build_clock_base,
        compute_m9_diagnostics,
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
        radiation_diagnostics=radiation_diagnostics,
        history_diagnostics=history,
        clock_base=build_clock_base(clock_namelist),
        **({"_with_radiation_count": True} if output_radiation_work is not None else {}),
    )
    if output_radiation_work is not None:
        m9, calls = m9
        output_radiation_work.append(int(jax.device_get(calls)))
    values = {
        wrf_name: history.q2 if attr is None else getattr(m9, attr)
        for wrf_name, attr in _M9_OUTPUT_FIELDS
    }
    values.update(HFX=m9.hfx, LH=m9.lh)
    requested = _requested_m9_output_names(variable_subset)
    out = {
        name: value
        for name, value in values.items()
        if value is not None and (requested is None or name in requested)
    }
    host_out = jax.device_get(out)
    return {name: np.asarray(value) for name, value in host_out.items()} or None


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
    """Default to full WRF history; explicit =0 selects the reduced stream."""
    from gpuwrf.config.history_output import full_wrfout_variables_enabled

    return full_wrfout_variables_enabled()


def _nested_perf_timers_from_env() -> bool:
    raw = os.environ.get("GPUWRF_NEST_PERF_TIMERS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _prepared_runtime_reuse_from_env() -> bool:
    """Resolve the fail-closed A/B control for B=1 prepared-runtime reuse.

    The production/default path is prepared reuse.  An explicit false value is
    retained solely so the controlled v0.23.4 A/B can reproduce the released
    per-segment runtime lifetime from the *same* source tree.  Unknown values
    fail before the forecast instead of silently selecting either arm.
    """

    raw = os.environ.get("GPUWRF_PREPARED_RUNTIME_REUSE")
    if raw is None:
        return True
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "GPUWRF_PREPARED_RUNTIME_REUSE must be one of "
        "1/true/yes/on or 0/false/no/off; got "
        f"{raw!r}"
    )


def _nested_event_aware_fusion_k_from_env() -> int:
    """Resolve the fail-closed, default-off B2 event-aware fusion depth."""

    raw = os.environ.get("GPUWRF_NESTED_EVENT_AWARE_FUSION_K")
    if raw is None:
        return 0
    value = raw.strip()
    if value == "0":
        return 0
    if value == "1":
        return 1
    raise ValueError(
        "GPUWRF_NESTED_EVENT_AWARE_FUSION_K must be 0 or 1; got "
        f"{raw!r}"
    )


def _aggregate_nested_aot_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-factory legacy telemetry into one process-lifetime report."""

    if not reports:
        return nested_aot_report()
    aggregate = dict(reports[-1])
    for field in ("load_count", "load_wall_seconds", "cache_hit_count"):
        aggregate[field] = sum(report.get(field, 0) for report in reports)
    domains: dict[str, dict[str, Any]] = {}
    count_fields = (
        "load_count",
        "load_wall_seconds",
        "load_attempt_count",
        "load_attempt_wall_seconds",
        "cache_hit_count",
    )
    for report in reports:
        for name, source in (report.get("domains") or {}).items():
            target = domains.setdefault(str(name), {})
            for key, value in source.items():
                if key == "keys":
                    target_keys = target.setdefault("keys", {})
                    for key_name, key_source in (value or {}).items():
                        key_target = target_keys.setdefault(str(key_name), {})
                        for metric, metric_value in key_source.items():
                            if metric in count_fields:
                                key_target[metric] = key_target.get(metric, 0) + metric_value
                            else:
                                key_target[metric] = metric_value
                elif key in count_fields:
                    target[key] = target.get(key, 0) + value
                else:
                    target[key] = value
    aggregate["domains"] = domains
    aggregate["factory_report_count"] = len(reports)
    aggregate["telemetry_scope"] = "process_lifetime_sum_of_per_factory_reports"
    return aggregate


def _nested_output_pipeline_from_env() -> bool:
    """Resolve whether the structural async output pipeline (Phase 2 / S1) is on.

    DEFAULT = OFF.  When ``GPUWRF_NEST_OUTPUT_PIPELINE`` is truthy
    (``1``/``true``/``yes``/``on``), the per-domain output callback no longer runs
    the finite guard, the M9 surface re-solve, and the ~70-leaf device->host pull
    INLINE on the step thread between two GPU compute bursts.  Instead the callback
    captures the post-step device state (by immutable reference -- the nested
    advance/cascade path does NOT donate the carry, so the captured leaves stay
    byte-identical and VRAM-resident until drained) and enqueues a small
    :class:`OutputSnapshot`; a single background MATERIALIZE thread runs the finite
    check + M9 + ``prepare_wrfout_payload`` (the D2H + host field build) and hands
    the host payload to the existing #101 :class:`AsyncWrfoutWriter` thread.  The
    GPU's next segment is dispatched immediately, so the host materialization
    overlaps the next compute instead of idling the GPU at every output boundary.

    The written NetCDF bytes, the finite-guard verdict, and the per-domain write
    order are byte-for-byte identical to the OFF (legacy) path -- the pipeline moves
    only WHICH THREAD and WHEN the same pure operations run, never the operations or
    their inputs (see ``PHASE2_ASYNC_PIPELINE_DESIGN.md`` Sec.3).  A finite error or
    a build/write error is surfaced fail-closed: it is re-raised on the next output
    boundary's step-thread call (before the next frame is committed) and again at
    ``join()`` (segment/run end), so a NaN can never escape into a green run or land
    a corrupt frame on disk (Invariant F).  The per-segment device finite guard
    (kept unconditionally below) is the synchronous backstop.

    OFF by default until the GPU A/B + VRAM gate (0:2's run) confirms the
    idle-reduction win and the VRAM bound on live geometry.
    """

    raw = os.environ.get("GPUWRF_NEST_OUTPUT_PIPELINE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _nested_output_pipeline_depth_from_env(default: int = 1) -> int:
    """Resolve the snapshot-queue depth (``GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH``).

    Bounds how many output frames' device snapshots may be in flight (pinning the
    captured carry leaves + the overlapped M9 radiation transient in VRAM) before
    the step thread blocks on ``snapshot_queue.put`` -- the device-memory analogue
    of #101's host-RAM backpressure.  Default ``1`` (one output frame in flight)
    keeps peak VRAM to baseline + ~one radiation transient; raise only after the
    VRAM A/B gate confirms headroom.  Values < 1 are clamped to 1.
    """

    raw = os.environ.get("GPUWRF_NEST_OUTPUT_PIPELINE_DEPTH", "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


class _NestedOutputPerfTimers:
    """Env-gated wall timers for the nested output boundary."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}
        self._counts: Counter = Counter()
        self._max_s: dict[str, float] = {}
        self._records: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> "_NestedOutputPerfTimers":
        return cls(enabled=_nested_perf_timers_from_env())

    @classmethod
    def disabled(cls) -> "_NestedOutputPerfTimers":
        return cls(enabled=False)

    def start(self) -> float:
        return time.perf_counter() if self.enabled else 0.0

    def stop(self, phase: str, started: float, **context: Any) -> None:
        if not self.enabled:
            return
        self.add(phase, time.perf_counter() - float(started), **context)

    def add(self, phase: str, seconds: float, **context: Any) -> None:
        if not self.enabled:
            return
        record: dict[str, Any] = {
            "phase": str(phase),
            "seconds": float(seconds),
        }
        for key, value in context.items():
            if value is None:
                continue
            record[key] = str(value) if isinstance(value, Path) else value
        with self._lock:
            name = str(phase)
            value = float(seconds)
            self._totals[name] = self._totals.get(name, 0.0) + value
            self._counts[name] += 1
            self._max_s[name] = max(self._max_s.get(name, 0.0), value)
            self._records.append(record)

    def record_writer_write(self, path: Path, seconds: float) -> None:
        self.add("writer_write", seconds, path=path)

    def summary(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            totals = dict(self._totals)
            counts = dict(self._counts)
            max_s = dict(self._max_s)
            records = list(self._records)
        return {
            "schema": "NestedOutputBoundaryPerfTimers",
            "schema_version": 1,
            "enabled_env": "GPUWRF_NEST_PERF_TIMERS",
            "totals_s": totals,
            "counts": counts,
            "mean_s": {
                name: totals[name] / counts[name]
                for name in totals
                if counts.get(name, 0) > 0
            },
            "max_s": max_s,
            "records": records,
        }


@dataclass(frozen=True)
class OutputSnapshot:
    """Step-thread -> materialize-thread handle for one output frame (S1).

    Captures the post-step ``carry`` BY IMMUTABLE REFERENCE (not by copy). A
    ``jax.Array`` is an immutable value: ``integrate`` REBINDS ``out[name]`` to a
    fresh array each advance, it never mutates the old one, and the nested
    advance/cascade path does NOT donate the threaded carry (audited: no
    ``donate_argnums`` on ``_advance_chunk`` / ``_advance_chunk_fori`` / the fused
    cascade). So holding this reference keeps the captured leaves byte-identical and
    VRAM-resident until the materialize thread drains them -- the immutable handle
    IS the snapshot, no device double-buffer needed (design Sec.2.2). Under a
    ``GPUWRF_*DONATE*`` flag the advance DOES delete its input carry, so the writer
    hands this snapshot a device copy instead (``_device_copy_tree``). Everything
    else here is host metadata (datetimes / floats / a deterministic Path), safe to
    read on the materialize thread.
    """

    writer: "_PerDomainWrfoutWriter"
    name: str
    own_step: int
    carry: Any
    valid_time: datetime
    lead_seconds: float
    lead_hours: float
    path: Path


class OutputPipeline:
    """Three-stage async output pipeline (Phase 2 / S1).

    STEP thread (the per-domain callback) only captures the post-step device state
    into an :class:`OutputSnapshot` and ``submit``s it -- no finite guard, no M9
    re-solve, no device->host pull on the step thread, so the GPU's next segment is
    dispatched immediately. A single MATERIALIZE thread drains the bounded snapshot
    queue and, per frame in order, runs the finite guard + M9 + ``prepare_wrfout_
    payload`` (the D2H + host field build) and hands the host payload to the existing
    #101 :class:`AsyncWrfoutWriter` (the WRITE stage). The materialization overlaps
    the next GPU segment's compute.

    Stage separation (snapshot queue vs the writer's own queue) decouples VRAM
    lifetime (the snapshot pins device leaves until the D2H lands -- early in
    materialize) from disk latency (the writer holds only a host ``PreparedWrfout``).
    A SINGLE materialize thread serializes the per-frame host work so device
    snapshots are released in output order -> deterministic VRAM high-water mark.

    Fail-closed (Invariant F): a finite/build error in MATERIALIZE (or a write error)
    is stored in ``_error``; the stage stops materializing and DRAINS its input queue
    (releasing any step-thread producer blocked on ``put``). ``submit`` re-raises the
    stored error on the NEXT output boundary's step-thread call (before that frame is
    committed) and ``join`` re-raises it at segment/run end -- so a NaN never escapes
    into a green run and never lands a later committed frame.
    """

    def __init__(
        self,
        async_writer: AsyncWrfoutWriter,
        *,
        max_snapshots: int = 1,
        perf_timers: _NestedOutputPerfTimers | None = None,
    ) -> None:
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")
        self._async_writer = async_writer
        self._perf_timers = perf_timers or _NestedOutputPerfTimers.disabled()
        self._queue: "queue.Queue[OutputSnapshot | None]" = queue.Queue(
            maxsize=max_snapshots
        )
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker, name="wrfout-materialize", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        while True:
            snapshot = self._queue.get()
            try:
                if snapshot is None:  # sentinel: shut down
                    return
                # Skip new materializations once an earlier frame failed; still drain
                # the queue so a step-thread producer blocked on put() is released.
                with self._error_lock:
                    failed = self._error is not None
                if not failed:
                    try:
                        snapshot.writer._materialize_and_submit(
                            name=snapshot.name,
                            own_step=int(snapshot.own_step),
                            carry=snapshot.carry,
                            valid_time=snapshot.valid_time,
                            lead_seconds=float(snapshot.lead_seconds),
                            lead_hours=float(snapshot.lead_hours),
                            path=snapshot.path,
                        )
                    except BaseException as exc:  # noqa: BLE001 - record & keep draining
                        with self._error_lock:
                            if self._error is None:
                                self._error = exc
            finally:
                self._queue.task_done()

    def submit(self, snapshot: OutputSnapshot) -> None:
        """Enqueue a snapshot for background materialization + write.

        Blocks (device-memory backpressure) if ``max_snapshots`` frames are already
        in flight. Re-raises a prior materialize/finite/write error PROMPTLY so the
        run fails closed before this frame is committed (Invariant F).
        """

        if self._closed:
            raise RuntimeError("OutputPipeline is closed")
        self._raise_if_failed()
        t0 = self._perf_timers.start()
        try:
            self._queue.put(snapshot)
        finally:
            self._perf_timers.stop(
                "queue_put",
                t0,
                domain=snapshot.name,
                own_step=int(snapshot.own_step),
                path=snapshot.path,
            )

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            err = self._error
        if err is not None:
            raise err

    def join(self) -> None:
        """Drain materialize -> write in order; re-raise the first error.

        Idempotent. Drains the snapshot queue (finishing all in-flight
        materializations), then the writer queue (flushing all writes), then
        re-raises the FIRST error from either stage. Must be called before reading
        the produced wrfouts / the output-present check.
        """

        if not self._closed:
            self._queue.put(None)  # sentinel
            self._closed = True
        self._thread.join()
        # Flush the downstream writer (all materialized payloads on disk) and
        # surface its first error.
        writer_error: BaseException | None = None
        try:
            self._async_writer.join()
        except BaseException as exc:  # noqa: BLE001 - record, prefer materialize error
            writer_error = exc
        # Prefer the earliest fault: a materialize/finite error (upstream) wins over
        # a writer error so the true root cause is surfaced.
        self._raise_if_failed()
        if writer_error is not None:
            raise writer_error


class _PerDomainWrfoutWriter:
    """Output callback: write one wrfout per domain at history cadence.

    Declares ``wants_carry`` so the domain-tree runner hands it the full
    ``OperationalCarry``: under Noah-MP the writer diagnostics must read the
    EVOLVED land carry (``carry.noahmp_land``) and the held surface radiation
    (``carry.noahmp_rad``), not just the post-step ``State``.

    When the ``GPUWRF_TRAINING_OUTPUT_SUBSET`` env flag is set, the per-domain
    wrfout is restricted to the compact, lossless-compressed
    ``MINIMAL_TRAINING_SET`` (#122 training output); otherwise the full
    uncompressed numerical/variable payload is written exactly as before; the
    common writer additionally requires authenticated global GRID_ID metadata.
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
        perf_timers: _NestedOutputPerfTimers | None = None,
        output_pipeline: "OutputPipeline | None" = None,
        initial_lands: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.input_dir = Path(input_dir)
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
        # Phase 2 (S1): when set, the STEP THREAD only captures the post-step device
        # state (by immutable reference -- the nested advance/cascade path does NOT
        # donate the carry, so the leaves stay byte-identical + resident) and enqueues
        # a snapshot; this :class:`OutputPipeline`'s single MATERIALIZE thread runs the
        # finite guard + M9 + device->host pull + field build (overlapping the next
        # GPU segment) and hands the host payload to ``async_writer``.  When None the
        # legacy synchronous step-thread materialization runs (default).
        self._output_pipeline = output_pipeline
        self._perf_timers = perf_timers or _NestedOutputPerfTimers.disabled()
        # Opt-in compact training output (#122): None => full byte-identical output.
        self._variable_subset = _resolve_training_output_subset()
        self._full_variable_set = _resolve_full_wrfout_variables()
        self.written: dict[str, list[str]] = {name: [] for name in bundles}
        from gpuwrf.diagnostics.census import WriterIoLedger, enabled as census_enabled
        self.census_io_ledger = WriterIoLedger() if census_enabled() else None
        self._census_persist_info = {}
        if self.census_io_ledger is not None and async_writer is not None:
            previous_callback = async_writer._write_timing_callback
            def counted_persist(path, seconds):
                self._record_census_persist(path)
                if previous_callback is not None:
                    previous_callback(path, seconds)
            async_writer._write_timing_callback = counted_persist
        # Lazy imports kept off the module top-level so importing this module stays
        # light for non-GPU callers (mirrors daily_pipeline).
        from gpuwrf.integration.daily_pipeline import (
            _load_static_latlon_writer_diagnostics,
            _merge_output_diagnostics,
            _surface_diagnostics_for_output,
        )
        from gpuwrf.io.gen2_accessor import Gen2Run

        self._surface_diagnostics_for_output = _surface_diagnostics_for_output
        self._merge_output_diagnostics = _merge_output_diagnostics
        run = Gen2Run(input_dir)
        self.writer_diagnostics: dict[str, dict[str, Any]] = {}
        self.writer_static_latlon_metadata: dict[str, Any] = {}
        self.domain_authorities = {}
        self._land_history_inputs = {}
        self._land_history_initial = dict(initial_lands or {})
        self.writer_global_attrs = {}
        from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes
        for domain, bundle in bundles.items():
            self.writer_global_attrs[domain] = wrf_history_global_attributes(
                self.input_dir / f"wrfinput_{domain}", run.namelist, domain, self.dt_by_domain[domain]
            )
            self.domain_authorities[domain] = bind_wrfout_domain_authority(
                domain, run.grid(domain), bundle.grid
            )
            diagnostics, meta = _load_static_latlon_writer_diagnostics(
                run, domain, grid=bundle.grid
            )
            self.writer_static_latlon_metadata[domain] = meta
            if diagnostics:
                self.writer_diagnostics[domain] = diagnostics
            if bool(getattr(bundle.namelist, "use_noahmp", False)):
                from gpuwrf.config.paths import wrf_run_dir
                from gpuwrf.io.land_history import load_land_history_inputs
                self._land_history_inputs[domain] = load_land_history_inputs(
                    self.input_dir / f"wrfinput_{domain}", table_dir=wrf_run_dir(),
                    parameters=bundle.namelist.noahmp_static.parameters,
                )

    def _initial_surface_fields(self, name: str) -> dict[str, np.ndarray]:
        """WRF writes initialized diagnostics before the first surface solve.

        Lead-zero history precedes ``solve`` (module_integrate.F:375 vs :393): the six
        surface fields are the wrfinput values, and every other surface/radiation
        diagnostic the writer would re-solve (M9 radiation/PBL fields, HFX/LH) holds
        its initialized value -- the wrfinput field when present, else WRF's zero
        (== CPU-WRF t=0 history on 9/9 WN3 gate domain-cases, SI38).
        """
        from gpuwrf.io.netcdf_lock import Dataset

        from gpuwrf.integration.daily_pipeline import _M9_OUTPUT_FIELDS

        grid = self.bundles[name].grid
        fields = {}
        path = self.input_dir / f"wrfinput_{name}"
        with Dataset(path, "r") as initial:
            for field in ("PSFC", "T2", "TSK", "U10", "V10", "Q2"):
                variable = initial.variables[field]
                value = variable[0] if variable.dimensions[0] == "Time" else variable[:]
                if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
                    raise ValueError(f"{path}:{field} has uninitialized values")
                value = np.asarray(value)
                if value.shape != (grid.ny, grid.nx) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{path}:{field} is not a finite initialized surface field")
                fields[field] = value
            for field in [wrf_name for wrf_name, _attr in _M9_OUTPUT_FIELDS] + [
                "HFX", "LH", "SWDNBC", "SWUPBC", "SWDNTC", "SWUPTC",
                "LWDNBC", "LWUPBC", "LWDNTC", "LWUPTC",
                "DUSFCG", "DVSFCG",
            ]:
                if field in fields:
                    continue
                if field not in initial.variables:
                    fields[field] = np.zeros((grid.ny, grid.nx), dtype=np.float32)
                    continue
                variable = initial.variables[field]
                value = np.ma.filled(variable[0] if variable.dimensions[0] == "Time" else variable[:], np.nan)
                value = np.asarray(value)
                if value.shape != (grid.ny, grid.nx) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{path}:{field} is not a finite initialized surface field")
                fields[field] = value
            for field in ("QKE", "CLDFRA", "QC_BL", "CLDFRA_BL", "DTAUX3D", "DTAUY3D"):
                nz = getattr(grid, "nz", None)
                if nz is None:
                    nz = len(initial.dimensions["bottom_top"])
                shape = (int(nz), grid.ny, grid.nx)
                if field in initial.variables:
                    variable = initial.variables[field]
                    value = np.ma.filled(variable[0] if variable.dimensions[0] == "Time" else variable[:], np.nan)
                    value = np.asarray(value)
                    if value.shape != shape or not np.all(np.isfinite(value)):
                        raise ValueError(f"{path}:{field} is not a finite initialized volume field")
                    fields[field] = value
                else:
                    fields[field] = np.zeros(shape, dtype=np.float32)
            # WRF history "T" is th_phy_m_t0 (Registry irhd): read from wrfinput and only re-derived
            # from t_2/qv by the first phy_prep, so nest adjust_tempqv never reaches the lead-zero T
            # (CPU-WRF WN3 nest T == wrfinput T bitwise; THM/QVAPOR carry the adjustment; fid-q2 R01).
            if "T" in initial.variables:
                value = np.asarray(np.ma.filled(initial.variables["T"][0], np.nan))
                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"{path}:T is not a finite initialized volume field")
                fields["T"] = value
        return fields

    def _record_census_persist(self, path):
        if self.census_io_ledger is None:
            return
        info = self._census_persist_info.pop(str(path), None)
        if info is not None:
            domain, radiation_calls = info
            self.census_io_ledger.record_persist(domain, radiation_output_calls=radiation_calls)

    def __call__(self, name: str, own_step: int, carry: Any) -> dict[str, Any]:
        """Output-boundary callback (``name, own_step, carry) -> result dict``.

        Step-thread entry. Computes the deterministic output ``path`` (host-only,
        cheap) and records it into ``self.written`` so the output-present check
        stays valid + order-deterministic. Then EITHER:

        * **pipeline path** (S1, ``self._output_pipeline`` set): capture the
          post-step ``carry`` by immutable reference into an :class:`OutputSnapshot`
          and enqueue it -- NO finite guard / M9 / D2H on the step thread. A prior
          frame's finite/build/write error is re-raised here (fail-closed, before
          this frame is committed). Returns immediately so the GPU resumes compute.
        * **legacy path** (default): run the finite guard + M9 + device->host pull
          + write/submit synchronously on the step thread (unchanged behaviour).

        The returned dict + the written bytes are identical in both paths.
        """

        lead_seconds = float(own_step) * float(self.dt_by_domain[name])
        lead_hours = lead_seconds / 3600.0
        valid_time = self.run_start + timedelta(seconds=lead_seconds)
        path = _wrfout_path(self.output_dir, name, valid_time)
        # Record the deterministic path NOW (step thread, before any background
        # work) so ``self.written`` is order-deterministic and the output-present
        # count is valid regardless of when the materialize/write lands.
        self.written[name].append(str(path))

        if self._output_pipeline is not None:
            # S1: hand the post-step carry (device leaves by reference) + host
            # metadata to the materialize stage and return. ``submit`` re-raises a
            # prior-frame error fail-closed before enqueuing -> Invariant F. Under carry
            # donation the next advance deletes this carry -> the snapshot owns a copy.
            self._output_pipeline.submit(
                OutputSnapshot(
                    writer=self,
                    name=name,
                    own_step=int(own_step),
                    carry=_device_copy_tree(carry) if _carry_donation_enabled() else carry,
                    valid_time=valid_time,
                    lead_seconds=float(lead_seconds),
                    lead_hours=float(lead_hours),
                    path=path,
                )
            )
        else:
            self._materialize_and_submit(
                name=name,
                own_step=int(own_step),
                carry=carry,
                valid_time=valid_time,
                lead_seconds=float(lead_seconds),
                lead_hours=float(lead_hours),
                path=path,
            )
        return {
            "domain": name,
            "lead_hours": float(lead_hours),
            "own_step": int(own_step),
            "all_finite": True,
            "wrfout": str(path),
            "prepared_full_variable_set": bool(self._full_variable_set),
            "full_variable_count": int(len(FULL_WRFOUT_VARIABLES)) if self._full_variable_set else None,
        }

    def _materialize_and_submit(
        self,
        *,
        name: str,
        own_step: int,
        carry: Any,
        valid_time: datetime,
        lead_seconds: float,
        lead_hours: float,
        path: Path,
    ) -> None:
        """Device->host materialization + finite guard + write/submit.

        Runs on the STEP THREAD in the legacy path, or on the single background
        MATERIALIZE thread in the S1 pipeline path. Pure-function of its inputs +
        the captured immutable device state, so the produced bytes are identical
        regardless of which thread runs it. Raises :class:`NonFiniteStateError`
        (caught by the guard) if the boundary-K state is non-finite -- in the
        pipeline path that error is recorded into the pipeline's fail-closed slot
        and surfaced before the next commit (Invariant F).
        """

        perf_timers = getattr(self, "_perf_timers", _NestedOutputPerfTimers.disabled())
        state = getattr(carry, "state", carry)
        t0 = perf_timers.start()
        try:
            assert_state_finite_at_boundary(
                state, domain=name, step=int(own_step), sim_time_s=lead_seconds
            )
        finally:
            perf_timers.stop(
                "finite_guard", t0, domain=name, own_step=int(own_step)
            )
        namelist = self.bundles[name].namelist
        grid = self.bundles[name].grid
        noahmp_land = getattr(carry, "noahmp_land", None)
        radiation_work = [] if self.census_io_ledger is not None else None
        t0 = perf_timers.start()
        try:
            initial = self._initial_surface_fields(name) if int(own_step) == 0 else None
            if initial is not None:
                # WRF lead-zero history precedes the first solve: no surface-layer,
                # Noah-MP or radiation re-solve at step 0 (SI38).
                surface_diagnostics = dict(initial)
            elif bool(getattr(namelist, "use_noahmp", False)) and noahmp_land is not None:
                surface_diagnostics = _noahmp_surface_diagnostics_for_output(
                    state,
                    namelist,
                    self.run_start,
                    lead_seconds=lead_seconds,
                    noahmp_land=noahmp_land,
                    noahmp_rad=getattr(carry, "noahmp_rad", None),
                    radiation_diagnostics=getattr(carry, "radiation_diagnostics", None),
                    variable_subset=self._variable_subset,
                    **({"output_radiation_work": radiation_work} if radiation_work is not None else {}),
                    history_diagnostics=getattr(carry, "history_diagnostics", None),
                )
            else:
                surface_diagnostics = self._surface_diagnostics_for_output(
                    state,
                    namelist,
                    self.run_start,
                    lead_seconds=lead_seconds,
                    variable_subset=self._variable_subset,
                    **({"output_radiation_work": radiation_work} if radiation_work is not None else {}),
                )
            diagnostics = self._merge_output_diagnostics(
                self.writer_diagnostics.get(name), surface_diagnostics
            )
            if initial is None:
                diagnostics = dict(diagnostics or {})
                rad = getattr(carry, "radiation_diagnostics", None)
                for wrf_name, attr in (
                    ("CLDFRA", "cloud_fraction"),
                    ("SWDNBC", "sw_clear_sfc_down"), ("SWUPBC", "sw_clear_sfc_up"),
                    ("SWDNTC", "sw_clear_toa_down"), ("SWUPTC", "sw_clear_toa_up"),
                    ("LWDNBC", "lw_clear_sfc_down"), ("LWUPBC", "lw_clear_sfc_up"),
                    ("LWDNTC", "lw_clear_toa_down"), ("LWUPTC", "lw_clear_toa_up"),
                ):
                    value = getattr(rad, attr, None)
                    if value is not None:
                        diagnostics[wrf_name] = value
                for wrf_name, attr in (("QC_BL", "qc_bl"), ("CLDFRA_BL", "cldfra_bl"),
                                       ("DTAUX3D", "dtaux3d"), ("DTAUY3D", "dtauy3d"),
                                       ("DUSFCG", "dusfcg"), ("DVSFCG", "dvsfcg")):
                    value = getattr(state, attr, None)
                    if value is not None:
                        diagnostics[wrf_name] = value
            if os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1":
                base = getattr(carry, "base_state", None)
                if base is None:
                    raise ValueError("native RK output requires its initial WRF base state")
                diagnostics = dict(diagnostics or {})
                diagnostics.update(PB=base.pb, PHB=base.phb, MUB=base.mub)
            if initial is not None:
                diagnostics = dict(diagnostics or {})
                diagnostics.update(initial)
            if noahmp_land is not None and name in getattr(self, "_land_history_inputs", {}):
                from gpuwrf.io.land_history import land_history_diagnostics
                initial_land = self._land_history_initial.get(name)
                if initial_land is None:
                    if int(own_step) != 0:
                        raise ValueError("land history requires the initialized water-tile carry")
                    initial_land = jax.device_get(noahmp_land)
                    self._land_history_initial[name] = initial_land
                land_fields, noahmp_land = land_history_diagnostics(
                    noahmp_land, initial_land, self._land_history_inputs[name],
                    own_step=int(own_step), history=getattr(carry, "land_history", None),
                    water_sst=(state.t_skin if getattr(namelist, "lower_boundary", None) is not None else None),
                )
                diagnostics = dict(diagnostics or {})
                diagnostics.update(land_fields)
                diagnostics.update(getattr(carry, "energy_accumulators", None) or {})
            if getattr(namelist, "lower_boundary", None) is not None:
                from gpuwrf.io.lower_boundary import lower_boundary_history
                diagnostics.update(lower_boundary_history(namelist.lower_boundary, own_step))
        finally:
            perf_timers.stop(
                "M9-diag", t0, domain=name, own_step=int(own_step)
            )
        t0 = perf_timers.start()
        try:
            prepared = prepare_wrfout_payload(
                state,
                grid,
                namelist,
                path,
                domain=name,
                domain_authority=self.domain_authorities[name],
                valid_time=valid_time,
                lead_hours=float(lead_hours),
                run_start=self.run_start,
                diagnostics=diagnostics,
                land_state=noahmp_land,
                variable_subset=self._variable_subset,
                include_mandatory_coords=self._variable_subset is not None,
                full_variable_set=self._full_variable_set,
                **({"source_global_attrs": self.writer_global_attrs[name]}
                   if getattr(self, "writer_global_attrs", None) is not None else {}),
            )
        finally:
            perf_timers.stop(
                "prepare_payload", t0, domain=name, own_step=int(own_step), path=path
            )
        if getattr(self, "history_journal", None) is not None:
            prepared = dataclass_replace(prepared, restart_publication=self.history_journal.publish)
        if self.census_io_ledger is not None:
            self._census_persist_info[str(path)] = (name, sum(radiation_work))
        if self._async_writer is not None:
            # Keep the device->host pull on the step thread (so no off-thread
            # touch of a device buffer the GPU may reuse), then submit the
            # host-only payload to the background writer and resume compute. The
            # deterministic output path is recorded NOW (at submit time) so the
            # output-present check stays valid; join() runs before that check.
            t0 = perf_timers.start()
            if self._variable_subset is None:
                try:
                    self._async_writer.submit(
                        prepared,
                        expected_domain=name,
                        expected_domain_authority=self.domain_authorities[name],
                    )
                finally:
                    perf_timers.stop(
                        "queue_put",
                        t0,
                        domain=name,
                        own_step=int(own_step),
                        path=path,
                    )
            else:
                # Compact training stream: same host payload, subset + mandatory
                # coords + lossless compression (self-contained, ~10 GB/day target).
                try:
                    self._async_writer.submit_subset(
                        prepared,
                        expected_domain=name,
                        expected_domain_authority=self.domain_authorities[name],
                        variable_subset=self._variable_subset,
                        target=path,
                        include_mandatory_coords=True,
                        compress=True,
                    )
                finally:
                    perf_timers.stop(
                        "queue_put",
                        t0,
                        domain=name,
                        own_step=int(own_step),
                        path=path,
                    )
        else:
            t0 = perf_timers.start()
            try:
                write_prepared_wrfout(
                    prepared,
                    expected_domain=name,
                    expected_domain_authority=self.domain_authorities[name],
                    variable_subset=self._variable_subset,
                    include_mandatory_coords=self._variable_subset is not None,
                    compress=self._variable_subset is not None,
                )
                self._record_census_persist(path)
            finally:
                perf_timers.stop(
                    "writer_write",
                    t0,
                    domain=name,
                    own_step=int(own_step),
                    path=path,
                    async_writer=False,
                )


class _BatchedPerDomainWrfoutWriter:
    """Debatch an F1 ensemble carry and feed the existing writer per lane."""

    wants_carry = True

    def __init__(self, writers: tuple[_PerDomainWrfoutWriter, ...]) -> None:
        if not writers:
            raise ValueError("batched writer requires at least one lane writer")
        self.writers = tuple(writers)
        self.batch_size = len(self.writers)
        domains = self.writers[0].written.keys()
        self.written: dict[str, list[str]] = {name: [] for name in domains}
        self.writer_static_latlon_metadata = self.writers[0].writer_static_latlon_metadata

    def __call__(self, name: str, own_step: int, carry: Any) -> dict[str, Any]:
        lane_results = []
        for lane, writer in enumerate(self.writers):
            lane_carry = _slice_batched_tree(carry, lane)
            result = writer(name, int(own_step), lane_carry)
            lane_results.append(result)
            wrfout = result.get("wrfout") if isinstance(result, dict) else None
            if wrfout is not None:
                self.written.setdefault(name, []).append(str(wrfout))
        return {
            "domain": name,
            "own_step": int(own_step),
            "batch_size": int(self.batch_size),
            "lanes": tuple(lane_results),
        }


def _emit_initial_history_frames(
    writer: Any,
    names: tuple[str, ...],
    initial_carries: dict[str, Any],
    *,
    enabled: bool,
) -> tuple[Any, ...]:
    """Route each exact initialized carry through the normal writer at lead zero.

    This helper deliberately does not participate in the domain-tree scheduler:
    no step clock, carry, or numerical state is replaced.  The existing writer
    derives the exact run-start valid time, path, GRID_ID authority, and schema
    from ``own_step=0`` just as it does for every later history boundary.
    """

    if not bool(enabled):
        return ()
    missing = [name for name in names if name not in initial_carries]
    if missing:
        raise ValueError(f"initial history carries missing domains: {missing}")
    return tuple(writer(name, 0, initial_carries[name]) for name in names)


def _finite_stats_host(state: Any) -> dict[str, Any]:
    from gpuwrf.integration.daily_pipeline import finite_guard_summary

    # The final verdict consumes only all_finite; failures retain full host stats.
    return finite_guard_summary(state)


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
    """Run the nested pipeline and join all owned AOT loader threads on exit."""
    prefetch_handles = []
    try:
        return _execute_nested_pipeline(config, _prefetch_handles=prefetch_handles)
    finally:
        for handle in prefetch_handles:
            handle.close()



def _require_native_base_state(carries, source) -> None:
    """Refuse a legacy checkpoint under native RK before any step runs (v0.3 fast defaults)."""

    if os.environ.get("GPUWRF_DYN_RK_FP32", "0") != "1":
        return
    missing = sorted(name for name, carry in carries.items() if getattr(carry, "base_state", None) is None)
    if missing:
        raise ValueError(
            f"checkpoint {source} has no native-RK base_state for {missing}: it was written by a "
            "legacy build (pre-v0.3 or GPUWRF_FAST_DEFAULTS=0). Resume it with GPUWRF_FAST_DEFAULTS=0, "
            "or restart from a checkpoint written by the fast-default build."
        )

def _execute_nested_pipeline(
    config: NestedPipelineConfig, *, _prefetch_handles: list,
) -> dict[str, Any]:
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
    batch_size = _batch_ensemble_size_from_env()
    restart_enabled = config.checkpoint_dir is not None or config.resume_checkpoint is not None
    if restart_enabled and batch_size != 1:
        raise ValueError("operational checkpoint/resume currently requires B=1")
    if config.checkpoint_dir is not None and config.checkpoint_interval_steps < 1:
        raise ValueError("checkpoint cadence must be positive in root-domain steps")
    prefetch_report = {"enabled": bool(config.aot_prefetch), "domains": {}}
    prefetch_started = None
    if config.aot_prefetch and batch_size == 1:
        # Allocator/flags are settled above; initialize the client on this thread
        # before the loader's workers run alongside input reading/native init.
        device = jax.devices()[0]
        if device.platform == "gpu":
            from gpuwrf.runtime import aot_precompile
            try:
                candidates = _aot_prefetch_artifact_candidates(names)
                if candidates:
                    prefetch_started = time.perf_counter()
                    handle = aot_precompile.prefetch_domain_blobs(
                        candidates, dev=device, max_workers=2,
                    )
                    _prefetch_handles.append(handle)
            except Exception as exc:
                prefetch_report["error"] = f"{type(exc).__name__}: {exc}"
        else:
            prefetch_report["skip"] = "non-gpu-backend"
    batch_namelists: dict[str, tuple[OperationalNamelist, ...]] = {}
    batch_noahmp_arrays: dict[str, tuple[tuple[Any, Any, Any] | None, ...]] = {}
    batch_input_dirs: tuple[Path, ...] = (Path(config.input_dir),)
    batch_run_starts: tuple[datetime, ...] = ()
    if batch_size > 1:
        (
            hierarchy,
            bundles,
            meta,
            run_start,
            dt_by_domain,
            initial_carries,
            batch_namelists,
            batch_input_dirs,
            batch_run_starts,
            batch_noahmp_arrays,
        ) = _load_batched_domains(config, names, batch_size=batch_size)
    else:
        hierarchy, bundles, meta, run_start, dt_by_domain, initial_carries = _load_domains(
            config, names
        )
        batch_run_starts = (run_start,)

    # Preserve initialized water-tile values before restart replaces the carries.
    # Noah-MP advances land only in WRF; its vectorized dummy water values must
    # never become history, including after a resumed forecast.
    writer_initial_lands = {
        name: jax.device_get(carry.noahmp_land)
        for name, carry in initial_carries.items() if carry.noahmp_land is not None
    }

    root = names[0]
    root_dt = dt_by_domain[root]
    restart_store = None
    restart_snapshot = None
    restart_report = {"enabled": restart_enabled, "checkpoints": []}
    if restart_enabled:
        from gpuwrf.runtime.restart_store import RestartStore, nested_identity
        restart_store = RestartStore(
            config.checkpoint_dir or Path(config.resume_checkpoint).parent,
            config.checkpoint_max_bytes, config.checkpoint_max_generations,
            config.checkpoint_reserve_bytes,
        )
        restart_identity = nested_identity(config, bundles, run_start)
        if config.resume_checkpoint is not None:
            resume_t0 = time.perf_counter()
            restart_snapshot, receipt = restart_store.read(
                config.resume_checkpoint, expected_identity=restart_identity,
            )
            if tuple(receipt["domains"]) != names or receipt["dt_s"] != dt_by_domain:
                raise ValueError("restart domain topology/timesteps differ from current run")
            initial_carries = {name: _commit_to_operational_device(carry) for name, carry in restart_snapshot["carries"].items()}
            _require_native_base_state(initial_carries, config.resume_checkpoint)
            restart_report.update(resumed_from=str(config.resume_checkpoint), read_wall_s=time.perf_counter() - resume_t0)
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
    total_steps_by_domain: dict[str, int] = {}
    for name in names:
        raw_total = float(config.hours) * 3600.0 / float(dt_by_domain[name])
        rounded_total = int(round(raw_total))
        if abs(raw_total - rounded_total) > 1.0e-9:
            raise ValueError(
                f"hours={config.hours} does not align with {name} dt="
                f"{dt_by_domain[name]}s (would need {raw_total} steps)"
            )
        total_steps_by_domain[name] = rounded_total
    output_alarm_schedule, nonintegral_output_alarms = (
        _output_alarm_steps_by_domain(
            names,
            dt_by_domain,
            history_interval_minutes,
            total_steps_by_domain,
        )
    )

    feedback_enabled = bool(config.feedback)
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=feedback_enabled)
    forcedown_warmup = (
        _start_forcedown_warmup(tree, initial_carries) if int(batch_size) == 1 else None
    )

    # Cold de-fused domains lower on this thread and compile/export on at most
    # two worker threads before stepping. A shared-cache lease prevents parallel
    # cases from duplicating the cold work. Warm artifacts skip this phase;
    # failures leave the ordinary serial compile/load path available. No forecast
    # is executed here. The markers expose the cold wall in the run receipt.
    _pc_t0 = time.perf_counter()
    sys.stderr.write("[parallel-compile] PREWARM_START de-fuse nest\n")
    sys.stderr.flush()
    if batch_size > 1:
        _pc_status = {
            "active": False,
            "source": "skip:batch-ensemble-vmap",
            "workers": 0,
            "report": {},
            "error": None,
        }
    else:
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
    perf_timers = _NestedOutputPerfTimers.from_env()
    async_writer = (
        AsyncWrfoutWriter(
            max_pending=2, write_timing_callback=perf_timers.record_writer_write
        )
        if async_output_enabled
        else None
    )
    # Phase 2 (S1) structural async output pipeline. OFF by default
    # (GPUWRF_NEST_OUTPUT_PIPELINE=1 enables it); requires the async writer (it
    # wraps it as the WRITE stage). When on, the per-domain callback only captures
    # the post-step device state into an OutputSnapshot and enqueues it; a single
    # background MATERIALIZE thread runs the finite guard + M9 + device->host pull +
    # field build and submits to async_writer, overlapping the next GPU segment.
    # Default-output bytes + the finite verdict + the write order are byte-identical
    # to the OFF path (only the thread + timing of the same pure ops change). join()ed
    # below (drains snapshot -> materialize -> write) before the output-present check.
    output_pipeline = (
        OutputPipeline(
            async_writer,
            max_snapshots=_nested_output_pipeline_depth_from_env(),
            perf_timers=perf_timers,
        )
        if (async_writer is not None and _nested_output_pipeline_from_env())
        else None
    )
    if batch_size > 1:
        lane_writers = []
        for lane, input_dir in enumerate(batch_input_dirs):
            lane_output_dir = config.output_dir / f"case{lane:03d}"
            lane_output_dir.mkdir(parents=True, exist_ok=True)
            lane_writers.append(
                _PerDomainWrfoutWriter(
                    output_dir=lane_output_dir,
                    input_dir=Path(input_dir),
                    run_start=batch_run_starts[lane],
                    bundles=bundles,
                    output_cadence_steps=output_cadence,
                    dt_by_domain=dt_by_domain,
                    async_writer=async_writer,
                    perf_timers=perf_timers,
                    output_pipeline=output_pipeline,
                    initial_lands={name: jax.tree.map(lambda x: x[lane], land)
                                   for name, land in writer_initial_lands.items()},
                )
            )
        writer = _BatchedPerDomainWrfoutWriter(tuple(lane_writers))
    else:
        writer = _PerDomainWrfoutWriter(
            output_dir=config.output_dir,
            input_dir=Path(config.input_dir),
            run_start=run_start,
            bundles=bundles,
            output_cadence_steps=output_cadence,
            dt_by_domain=dt_by_domain,
            async_writer=async_writer,
            perf_timers=perf_timers,
            output_pipeline=output_pipeline,
            initial_lands=writer_initial_lands,
        )
    for domain, latlon_meta in writer.writer_static_latlon_metadata.items():
        meta.setdefault("domains", {}).setdefault(domain, {})["writer_static_latlon"] = latlon_meta
    if restart_snapshot is not None:
        from gpuwrf.runtime.restart_store import restore_output_receipts, restore_writer_census
        restore_output_receipts(writer, restart_snapshot["driver_state"]["outputs"])
        restore_writer_census(writer, restart_snapshot["driver_state"]["writer_census"])
    if restart_enabled:
        from gpuwrf.runtime.restart_store import HistoryJournal
        stream = restart_snapshot["driver_state"].get("history_stream") if restart_snapshot is not None else None
        writer.history_journal = HistoryJournal(restart_store, restart_identity, stream)
        if restart_snapshot is not None:
            restart_report["replayed_history"] = writer.history_journal.recover(restart_snapshot["driver_state"]["outputs"])

    # WRF-compatible lead-zero history is an explicit corrected-validation opt-in.
    # It uses the same authenticated per-domain writer before the first numerical
    # advance; the domain-tree scheduler and every later callback remain untouched.
    initial_history_results = _emit_initial_history_frames(
        writer,
        names,
        initial_carries,
        enabled=bool(config.emit_initial_history) and restart_snapshot is None,
    )

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
    # Complete loads before the first segment; keep their futures registered
    # until the real runtime-key lookup consumes them. Wait time stays in the
    # process/pre-step wall, rather than being hidden from whole-run S1.
    if _prefetch_handles:
        wait_start = time.perf_counter()
        future = None
        for name, future in _prefetch_handles[0].futures.items():
            try:
                status = future.result()[1]
                prefetch_report["domains"][name] = status
            except Exception as exc:
                prefetch_report["domains"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        del future  # A loop local must not retain an unused executable.
        prefetch_report["remaining_wait_s"] = time.perf_counter() - wait_start
        prefetch_report["elapsed_s"] = time.perf_counter() - prefetch_started
    meta["aot_prefetch"] = prefetch_report
    if forcedown_warmup is not None:
        meta["forcedown_warmup"] = _join_forcedown_warmup(forcedown_warmup)

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
    cascade_counts: Counter = Counter()
    events_tail: deque = deque(maxlen=event_tail_cap if event_tail_cap > 0 else None)
    final_states: dict[str, Any] = {}
    # Host-sync granularity for the live nest (v0.17 GPU-idle fix).  Default =
    # async per-root-step sync (keeps the GPU queue full across nested cascades);
    # GPUWRF_NESTED_SYNC_MODE=advance reproduces the legacy per-advance baseline.
    # The per-segment block below is ALWAYS kept (peak-VRAM bound between hours).
    nested_block_between, nested_root_sync_cadence = _nested_sync_mode_from_env()
    prepared_runtime_reuse = (
        _prepared_runtime_reuse_from_env() if int(batch_size) == 1 else False
    )
    event_aware_fusion_k = (
        _nested_event_aware_fusion_k_from_env() if int(batch_size) == 1 else 0
    )
    prepared_runtime = (
        _prepare_operational_domain_tree_runtime(
            tree, feedback_enabled=feedback_enabled
        )
        if prepared_runtime_reuse
        else None
    )
    legacy_aot_reports: list[dict[str, Any]] = []
    start = 0
    if restart_snapshot is not None:
        own_steps = dict(restart_snapshot["own_steps"])
        start = own_steps[root]
        if start > root_steps:
            raise ValueError("restart is beyond the requested forecast endpoint")
        saved_driver = restart_snapshot["driver_state"]
        event_counts.update(saved_driver["event_counts"])
        force_counts.update(saved_driver["force_counts"])
        cascade_counts.update(saved_driver["cascade_counts"])
        events_tail.extend(saved_driver["events_tail"])
        final_states = {name: carry.state for name, carry in carries.items()}
    first_resumed_segment = restart_snapshot is not None
    async_writer_joined = False
    try:
        while start < root_steps:
            seg = min(root_seg_steps, root_steps - start)
            if config.checkpoint_dir is not None:
                seg = min(seg, config.checkpoint_interval_steps - start % config.checkpoint_interval_steps)
            run_tree = (
                run_batched_operational_domain_tree
                if int(batch_size) > 1
                else run_operational_domain_tree
            )
            run_kwargs = {}
            if int(batch_size) > 1:
                run_kwargs["batch_namelists"] = batch_namelists
                run_kwargs["batch_size"] = int(batch_size)
                run_kwargs["batch_noahmp_arrays"] = batch_noahmp_arrays
            else:
                run_kwargs["prepared_runtime"] = prepared_runtime
                run_kwargs["event_aware_fusion_k"] = event_aware_fusion_k
            result = run_tree(
                tree,
                root_steps=seg,
                feedback_enabled=feedback_enabled,
                output=writer,
                output_cadence_steps=output_cadence,
                output_alarm_steps=nonintegral_output_alarms,
                block_between=nested_block_between,
                root_sync_cadence=nested_root_sync_cadence,
                carries=carries,
                initial_own_steps=own_steps,
                **run_kwargs,
            )
            if int(batch_size) == 1 and not prepared_runtime_reuse:
                # The released lifetime reconstructs the factory each segment,
                # and each factory resets its local telemetry. Capture the report
                # before the next segment resets it, then fold all segments below.
                legacy_aot_reports.append(nested_aot_report())
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
            from gpuwrf.diagnostics.census import write_segment
            write_segment(config.output_dir, carries, bundles, own_steps, writer, output_cadence)
            # Fold this segment's events into the running summary + bounded tail, then
            # DROP the segment's tuple (host RAM stays O(1) in forecast length).
            for event in result.events:
                if not event:
                    continue
                event_counts[event[0]] += 1
                if event[0] == "force":
                    force_counts[f"{event[1]}->{event[2]}"] += 1
                events_tail.append(event)
            cascade_counts.update(result.cascade_counts)
            final_states = result.states
            start += seg
            if first_resumed_segment:
                restart_report["resume_to_first_segment_ready_s"] = time.perf_counter() - overall_start
                first_resumed_segment = False
            if config.checkpoint_dir is not None and (start % config.checkpoint_interval_steps == 0 or start == root_steps):
                from gpuwrf.runtime.restart_store import drain_checkpoint_output, output_receipts, writer_census_snapshot
                checkpoint_t0 = time.perf_counter()
                drain_checkpoint_output(output_pipeline, async_writer)
                generation, storage_receipt = restart_store.save(
                    carries, own_steps, dt_by_domain, restart_identity,
                    driver_state={
                        "history_stream": writer.history_journal.stream,
                        "outputs": output_receipts(writer.written),
                        "writer_census": writer_census_snapshot(writer),
                        "prepared_runtime_reuse": prepared_runtime_reuse,
                        "aot": nested_aot_report(),
                        "event_counts": dict(event_counts), "force_counts": dict(force_counts),
                        "cascade_counts": dict(cascade_counts), "events_tail": tuple(events_tail),
                    },
                )
                restart_report["checkpoints"].append({"path": str(generation), "root_step": start, "wall_s": time.perf_counter() - checkpoint_t0, **storage_receipt})
                sys.stderr.write(f"[checkpoint] VERIFIED {generation} root_step={start}\n")
                sys.stderr.flush()
        jax.block_until_ready(tuple(state.theta for state in final_states.values()))
        forecast_wall_s = time.perf_counter() - forecast_start
        # Drain the background output stages: every submitted output frame must be
        # materialized + on disk (and the first stage error surfaced -- join()
        # re-raises, failing the run) before the output-present check below reads
        # writer.written. With the S1 pipeline ``output_pipeline.join()`` drains
        # snapshot -> materialize -> write IN ORDER (re-raising the FIRST error from
        # either stage); without it ``async_writer.join()`` drains the single writer
        # stage. Both are idempotent / no-op when async output is disabled. The
        # background work overlapped GPU compute, so this join only waits on the last
        # in-flight frame and is outside ``forecast_wall_s``.
        if output_pipeline is not None:
            t0 = perf_timers.start()
            try:
                output_pipeline.join()
            finally:
                perf_timers.stop(
                    "join",
                    t0,
                    context="success",
                    submitted_paths=sum(len(paths) for paths in writer.written.values()),
                )
            async_writer_joined = True
        elif async_writer is not None:
            t0 = perf_timers.start()
            try:
                async_writer.join()
            finally:
                perf_timers.stop(
                    "join",
                    t0,
                    context="success",
                    submitted_paths=sum(len(paths) for paths in writer.written.values()),
                )
            async_writer_joined = True
        from gpuwrf.diagnostics.census import write_segment
        write_segment(config.output_dir, carries, bundles, own_steps, writer, output_cadence)
    finally:
        # Fail-closed: if the forecast loop above raised (NaN/OOM/etc.), still drain
        # the background stages so no daemon thread outlives the run and any in-flight
        # frame is flushed -- but do NOT mask the body's exception with a secondary
        # stage error. join() is idempotent, so on the success path this is a no-op.
        if not async_writer_joined and (
            output_pipeline is not None or async_writer is not None
        ):
            try:
                t0 = perf_timers.start()
                try:
                    if output_pipeline is not None:
                        output_pipeline.join()
                    else:
                        async_writer.join()
                finally:
                    perf_timers.stop("join", t0, context="finally")
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
        cascade_counts=dict(cascade_counts),
    )

    final_finite = {name: _finite_stats_host(state) for name, state in result.states.items()}

    per_domain: dict[str, Any] = {}
    all_finite = True
    all_output_present = True
    for name in names:
        outputs = writer.written.get(name, [])
        finite_ok = bool(final_finite[name]["all_finite"])
        expected_outputs = len(output_alarm_schedule[name]) + int(
            bool(config.emit_initial_history)
        )
        expected_output_files = expected_outputs * int(batch_size)
        output_ok = len(outputs) == expected_output_files
        all_finite = all_finite and finite_ok
        all_output_present = all_output_present and output_ok
        per_domain[name] = {
            "final_state_finite": finite_ok,
            "wrfout_count": len(outputs),
            "expected_wrfout_count": expected_output_files,
            "wrfout_files": outputs,
            "dt_s": float(dt_by_domain[name]),
            "history_interval_min": float(history_interval_minutes[name]),
            "own_steps": int(result.own_steps.get(name, 0)),
        }
        if batch_size > 1:
            per_domain[name]["batch_size"] = int(batch_size)
            per_domain[name]["expected_wrfout_count_per_lane"] = int(expected_outputs)

    total_wall_s = time.perf_counter() - overall_start
    perf_timer_summary = perf_timers.summary()
    if perf_timer_summary is not None:
        meta["nested_output_perf_timers"] = perf_timer_summary
    if int(batch_size) == 1:
        meta["nested_runtime"] = {
            "prepared_runtime_reuse": bool(prepared_runtime_reuse),
            "selection_env": "GPUWRF_PREPARED_RUNTIME_REUSE",
        }
        meta["nested_aot"] = (
            nested_aot_report()
            if prepared_runtime_reuse
            else _aggregate_nested_aot_reports(legacy_aot_reports)
        )
        if event_aware_fusion_k:
            meta["nested_event_aware_fusion"] = {
                "k": int(event_aware_fusion_k),
                "selection_env": "GPUWRF_NESTED_EVENT_AWARE_FUSION_K",
                "cascade_counts": dict(cascade_counts),
            }
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
        "batch_ensemble": {
            "enabled": bool(batch_size > 1),
            "batch_size": int(batch_size),
            "input_dirs": [str(path.resolve()) for path in batch_input_dirs],
            "run_starts_utc": [dt.isoformat() for dt in batch_run_starts],
        },
        "wall_clock_total_s": float(total_wall_s),
        "wall_clock_forecast_only_s": float(forecast_wall_s),
        "restart": restart_report,
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
    if config.emit_initial_history:
        payload["initial_history_output"] = {
            "enabled": True,
            "own_step": 0,
            "domain_count": len(initial_history_results),
            "uses_existing_authenticated_writer": True,
        }
    return payload
