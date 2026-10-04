#!/usr/bin/env python3
"""Build ``proofs/v025/m0/cancellation_map.json`` (contract §10).

Phase A5. CPU-only: no GPU import, query, compile, or run.

What this driver does, in order:

1. Loads a REAL ``20260725_18z``-derived production state (``real_state.py``).
2. Derives the real stage intermediates the operators actually consume, using
   the production functions themselves (``small_step_prep_wrf``,
   ``diagnose_pressure_al_alt``, ``calc_coef_w_wrf_coefficients``, ...), so no
   operator is fed a hand-rolled approximation of its own input.
3. Runs every inventoried operator three times -- fp64 baseline, fp32-rounded
   inputs with fp64 arithmetic, aggressive fp32 -- and scores the frozen
   classification in ``cancellation.py``.
4. Proves activity/inactivity of every physics scheme by executing the
   production dispatch resolver rather than by reading the namelist by eye.
5. Cross-references the 24 live ``force_fp64_island`` call sites against the
   measurement, so the proposed island set is minimal and evidence-linked.

Nothing here writes to ``src/gpuwrf``; every production symbol is imported
read-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v025 import cancellation as canc  # noqa: E402
from scripts.v025.real_state import (  # noqa: E402
    assert_cpu_only,
    select_state_pair,
    default_run_dir,
    load_real_snapshot,
    stage_advanced_state,
)

SCHEMA = "wrf_gpu2.v025.m0.cancellation_map.v1"


# --------------------------------------------------------------------------- #
# fp64-island neutralisation                                                   #
# --------------------------------------------------------------------------- #
import contextlib  # noqa: E402


@contextlib.contextmanager
def islands_disabled():
    """Temporarily make ``force_fp64_island`` a pass-through.

    This is what turns "is this island earning its keep?" from an opinion into
    a measurement. With fp32 inputs the island is NOT inert -- it upcasts them
    -- so an fp32 arm run normally measures fp32 storage plus island-protected
    arithmetic, not aggressive fp32. Neutralising the island gives the true
    aggressive-fp32 arm, and the difference between the two is the island's
    measured benefit.

    The symbol is imported by value into each consuming module, so every
    already-imported ``gpuwrf.*`` module holding a reference is patched, not
    just the defining one. JAX caches are cleared on entry and exit because the
    island-on and island-off arms have identical input dtypes and would
    otherwise collide on the same jit cache key.
    """

    import jax

    import gpuwrf.contracts.precision as precision

    def passthrough(*arrays):
        return arrays[0] if len(arrays) == 1 else arrays

    patched: list[tuple[Any, Any]] = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith("gpuwrf"):
            continue
        current = getattr(module, "force_fp64_island", None)
        if current is precision.force_fp64_island:
            patched.append((module, current))
    jax.clear_caches()
    try:
        for module, _ in patched:
            setattr(module, "force_fp64_island", passthrough)
        yield len(patched)
    finally:
        for module, original in patched:
            setattr(module, "force_fp64_island", original)
        jax.clear_caches()


# --------------------------------------------------------------------------- #
# dtype casting                                                                #
# --------------------------------------------------------------------------- #
def _metric_names(metrics: Any) -> tuple[str, ...]:
    return metrics._array_names()


def float_dtypes_present(value: Any, *, _depth: int = 0) -> set[str]:
    """Every float dtype reachable in ``value``, through pytrees AND dataclasses.

    ``jax.tree_util.tree_map`` only descends registered pytrees. A plain frozen
    dataclass -- ``CoupledVelocities`` is one -- is a single opaque leaf to it,
    so a cast walks straight past the arrays inside and returns the object
    unchanged, silently. This walker exists to catch that: it is what the fp32
    arm's input check uses to prove the demotion actually reached everything.
    """

    import dataclasses as dc

    import jax

    if value is None or _depth > 6:
        return set()
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        name = str(dtype)
        return {name} if name.startswith(("float", "bfloat")) else set()
    if dc.is_dataclass(value) and not isinstance(value, type):
        found: set[str] = set()
        for field in dc.fields(value):
            found |= float_dtypes_present(getattr(value, field.name, None), _depth=_depth + 1)
        return found
    try:
        leaves, _ = jax.tree_util.tree_flatten(value)
    except Exception:  # noqa: BLE001
        return set()
    found = set()
    for leaf in leaves:
        if leaf is value:
            continue
        found |= float_dtypes_present(leaf, _depth=_depth + 1)
    return found


def make_cast(mode: str, *, seed: int | None = None, epsilon: float | None = None) -> Callable[[Any], Any]:
    """Return a pytree-aware caster for one arm of the comparison.

    ``fp64``    identity (the baseline).
    ``repr32``  round every float leaf through fp32 and widen back to fp64:
                the storage penalty alone, with arithmetic still exact.
    ``fp32``    store and compute in fp32: what an aggressive rewrite pays.
    ``probe``   apply a seeded relative perturbation of size ``epsilon`` in
                fp64: the condition-number probe.

    ``DycoreMetrics`` refuses non-fp64 arrays in ``__post_init__``. That is a
    production invariant, not something this audit may quietly break, so the
    caster clones the object and installs the cast arrays through
    ``object.__setattr__`` -- the same mechanism ``__post_init__`` itself uses.
    For the ``fp32`` arm the metric arrays are stored as fp32 so JAX type
    promotion carries fp32 through the operator, which is the honest
    representation of an fp32 rewrite that also demotes its static metrics.
    """

    import copy

    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed) if seed is not None else None

    def cast_leaf(value: Any) -> Any:
        dtype = getattr(value, "dtype", None)
        if dtype is None or not jnp.issubdtype(dtype, jnp.floating):
            return value
        if mode == "fp64":
            return value
        if mode == "repr32":
            return value.astype(jnp.float32).astype(jnp.float64)
        if mode == "fp32":
            return value.astype(jnp.float32)
        if mode == "probe":
            assert rng is not None and epsilon is not None
            array = np.asarray(value, dtype=np.float64)
            noise = rng.uniform(-1.0, 1.0, size=array.shape)
            return jnp.asarray(array * (1.0 + epsilon * noise))
        raise ValueError(f"unknown cast mode {mode!r}")

    import dataclasses as dc

    def cast(value: Any) -> Any:
        if value is None:
            return None
        if mode == "fp64":
            return value
        if isinstance(value, (int, float, bool, str)):
            return value
        if getattr(value, "dtype", None) is not None:
            return cast_leaf(value)
        # DycoreMetrics: clone + setattr, bypassing the fp64 type guard in
        # __post_init__. That guard is a production invariant this audit must
        # not break; the shim is the same mechanism __post_init__ itself uses.
        if hasattr(value, "_array_names") and hasattr(value, "provenance"):
            clone = copy.copy(value)
            for name in _metric_names(value):
                object.__setattr__(clone, name, cast_leaf(getattr(value, name)))
            return clone
        # A dataclass that is NOT a registered pytree is opaque to tree_map --
        # it would be returned unchanged and the arms would silently differ only
        # in name. CoupledVelocities is exactly this case, and it feeds every
        # flux-advection operator. Cast it field by field instead.
        if dc.is_dataclass(value) and not hasattr(type(value), "tree_flatten"):
            clone = copy.copy(value)
            for field in dc.fields(value):
                object.__setattr__(clone, field.name, cast(getattr(value, field.name)))
            return clone
        try:
            return jax.tree_util.tree_map(cast_leaf, value)
        except Exception:  # noqa: BLE001 - opaque objects pass through
            return value

    return cast


# --------------------------------------------------------------------------- #
# operator specification                                                       #
# --------------------------------------------------------------------------- #
class Operator:
    """One inventory entry: how to call it, and how to read its outputs."""

    def __init__(
        self,
        name: str,
        *,
        family: str,
        module: str,
        entrypoint: str,
        wrf_source: str,
        role: str,
        call: Callable[[dict[str, Any], Callable[[Any], Any]], Any],
        outputs: Callable[[Any], dict[str, Any]] | None = None,
        conserved: Callable[[Any], float] | None = None,
        downstream: tuple[str, ...] = (),
        activity: str = "namelist-selected",
    ) -> None:
        self.name = name
        self.family = family
        self.module = module
        self.entrypoint = entrypoint
        self.wrf_source = wrf_source
        self.role = role
        self.call = call
        self.outputs = outputs or (lambda result: _default_outputs(result))
        self.conserved = conserved
        self.downstream = downstream
        self.activity = activity


def _default_outputs(result: Any) -> dict[str, Any]:
    """Flatten whatever an operator returned into named arrays."""

    import jax

    if hasattr(result, "dtype"):
        return {"out": result}
    if isinstance(result, dict):
        return {key: value for key, value in result.items() if hasattr(value, "dtype")}
    if isinstance(result, tuple) and hasattr(result, "_fields"):
        return {name: value for name, value in zip(result._fields, result) if hasattr(value, "dtype")}
    if isinstance(result, (tuple, list)):
        return {f"out{index}": value for index, value in enumerate(result) if hasattr(value, "dtype")}
    leaves, _ = jax.tree_util.tree_flatten(result)
    return {f"leaf{index}": leaf for index, leaf in enumerate(leaves) if hasattr(leaf, "dtype")}


# --------------------------------------------------------------------------- #
# real context                                                                 #
# --------------------------------------------------------------------------- #
def build_context(snapshot: Any, previous: Any, domain: str = "d01") -> dict[str, Any]:
    """Derive the real stage intermediates every acoustic operator consumes."""

    import jax.numpy as jnp

    from gpuwrf.contracts.state import Tendencies
    from gpuwrf.dynamics.acoustic_wrf import (
        calc_coef_w_wrf_coefficients,
        diagnose_pressure_al_alt,
        moisture_coupling_factors,
    )
    from gpuwrf.dynamics.core.advance_w import dry_cqw
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
    from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
    from gpuwrf.dynamics.flux_advection import couple_velocities_periodic

    state = snapshot.state
    base_state = snapshot.base_state
    metrics = snapshot.metrics
    config = snapshot.config
    nz, ny, nx = snapshot.shape
    dt = float(config["dt"])
    # WRF RK3: the third stage uses the full dt; the acoustic substep is
    # dt_rk / n_substeps. Stage 2 is used for the tap because stage 1 zeroes
    # the work primes by construction (mu_work = 0), which would hide exactly
    # the cancellation the map is looking for.
    dt_rk = dt
    n_acoustic = 4
    dts = dt_rk / float(n_acoustic)

    advanced, stage_pair_meta = stage_advanced_state(previous, snapshot, dt_rk)
    prep = small_step_prep_wrf(
        advanced,
        2,
        dt_rk,
        metrics=metrics,
        reference_state=state,
        base_state=base_state,
    )
    # step=0 is the stage seed; the per-substep form is calc_p_rho_step and is
    # inventoried separately with the seed's p' as its pm1 previous-substep input.
    prho = calc_p_rho_wrf(prep, step=0, non_hydrostatic=bool(config["non_hydrostatic"]))
    pressure, al, alt = diagnose_pressure_al_alt(state, base_state, metrics, hypsometric_opt=2)
    cqu, cqv = moisture_coupling_factors(state)
    cqw = dry_cqw(nz, ny, nx, dtype=jnp.float64)
    a_coef, alpha_coef, gamma_coef = calc_coef_w_wrf_coefficients(
        prep.mut,
        metrics,
        dt=dts,
        epssm=float(config["epssm"]),
        top_lid=False,
        cqw=cqw,
        c2a=prep.c2a,
    )

    tendencies = Tendencies(
        u=jnp.zeros((nz, ny, nx + 1)),
        v=jnp.zeros((nz, ny + 1, nx)),
        w=jnp.zeros((nz + 1, ny, nx)),
        theta=jnp.zeros((nz, ny, nx)),
        qv=jnp.zeros((nz, ny, nx)),
        p=jnp.zeros((nz, ny, nx)),
        ph=jnp.zeros((nz + 1, ny, nx)),
        mu=jnp.zeros((ny, nx)),
    )

    # Mass-coupled face velocities for the flux-form advection operators, built
    # by the production coupler from the real wind and mass fields.
    vel = couple_velocities_periodic(
        state.u, state.v, state.mu_total,
        c1h=metrics.c1h, c2h=metrics.c2h, dnw=metrics.dnw,
        rdx=1.0 / float(config["dx"]), rdy=1.0 / float(config["dy"]),
        msfuy=metrics.msfuy, msfvx=metrics.msfvx, msftx=metrics.msftx,
        msfux=metrics.msfux, msfvy=metrics.msfvy,
    )
    ww = vel.rom

    # Physics needs a GridSpec, real GWD statics, and the Noah-MP land carry;
    # all three come from the case's own files through production loaders.
    from gpuwrf.io.gen2_accessor import Gen2Run

    run_path = Path(snapshot.provenance["run_dir"])
    run = Gen2Run(run_path)
    grid = run.grid(domain).as_grid_spec()
    grid = dataclasses.replace(grid, metrics=metrics) if dataclasses.is_dataclass(grid) else grid

    raw = snapshot.raw
    gwdo_statics = None
    try:
        from gpuwrf.coupling.physics_couplers import build_gwdo_statics_from_wrf_fields

        gwdo_statics = build_gwdo_statics_from_wrf_fields(
            raw["VAR"], raw["CON"],
            raw["OA1"], raw["OA2"], raw["OA3"], raw["OA4"],
            raw["OL1"], raw["OL2"], raw["OL3"], raw["OL4"],
            dx_m=float(config["dx"]),
            sina=raw.get("SINALPHA"), cosa=raw.get("COSALPHA"),
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        gwdo_statics = None
        print(f"  [gwdo statics unavailable: {type(exc).__name__}: {exc}]", flush=True)

    noahmp_land = noahmp_static = None
    try:
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state

        noahmp_land, noahmp_static, _meta = build_noahmp_land_state(
            run_path, domain=domain, table_dir=run_path,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        print(f"  [noahmp land state unavailable: {type(exc).__name__}: {exc}]", flush=True)

    valid = snapshot.provenance["valid_times"][0]
    time_utc = datetime.strptime(valid, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)

    return {
        "snapshot": snapshot,
        "grid": grid,
        "gwdo_statics": gwdo_statics,
        "noahmp_land": noahmp_land,
        "noahmp_static": noahmp_static,
        "time_utc": time_utc,
        "advanced": advanced,
        "stage_pair": stage_pair_meta,
        "vel": vel,
        "state": state,
        "base_state": base_state,
        "metrics": metrics,
        "config": config,
        "prep": prep,
        "prho": prho,
        "pressure": pressure,
        "al": al,
        "alt": alt,
        "cqu": cqu,
        "cqv": cqv,
        "cqw": cqw,
        "a": a_coef,
        "alpha": alpha_coef,
        "gamma": gamma_coef,
        "tendencies": tendencies,
        "ww": ww,
        "dt": dt,
        "dt_rk": dt_rk,
        "dts": dts,
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "rdx": 1.0 / float(config["dx"]),
        "rdy": 1.0 / float(config["dy"]),
    }


# --------------------------------------------------------------------------- #
# the inventory                                                                #
# --------------------------------------------------------------------------- #
def build_inventory() -> list[Operator]:
    """Every dycore operator on the FAST-v025 active path, plus active physics."""

    import jax.numpy as jnp

    from gpuwrf.dynamics import acoustic_wrf as aw
    from gpuwrf.dynamics import damping, explicit_diffusion, flux_advection, hybrid_eta, limiters
    from gpuwrf.dynamics import tridiag_solve as tri
    from gpuwrf.dynamics.core import advance_w as aww
    from gpuwrf.dynamics.core import calc_p_rho as cpr
    from gpuwrf.dynamics.core import rhs_ph as rph
    from gpuwrf.dynamics.core import rk_addtend_dry as rad
    from gpuwrf.dynamics.core import small_step_prep as ssp

    ops: list[Operator] = []
    add = ops.append

    # ---- small-step preparation / mass ------------------------------------
    add(Operator(
        "small_step_prep",
        family="dycore.small_step",
        module="gpuwrf.dynamics.core.small_step_prep",
        entrypoint="small_step_prep_wrf",
        wrf_source="dyn_em/module_small_step_em.F:125-285",
        role="builds the RK-stage work primes (reference - current) and the mass family",
        call=lambda ctx, cast: ssp.small_step_prep_wrf(
            cast(ctx["state"]), 2, ctx["dt_rk"],
            metrics=cast(ctx["metrics"]), base_state=cast(ctx["base_state"]),
        ),
        outputs=lambda prep: {
            "u_work": prep.u_work, "v_work": prep.v_work, "w_work": prep.w_work,
            "theta_work": prep.theta_work, "ph_work": prep.ph_work, "mu_work": prep.mu_work,
            "c2a": prep.c2a, "al": prep.al, "alt": prep.alt, "php": prep.php,
        },
        downstream=("calc_p_rho", "advance_w", "rhs_ph", "acoustic substep"),
    ))

    # ---- equation of state / pressure -------------------------------------
    add(Operator(
        "calc_p_rho",
        family="dycore.eos",
        module="gpuwrf.dynamics.core.calc_p_rho",
        entrypoint="calc_p_rho_wrf",
        wrf_source="dyn_em/module_small_step_em.F:1849-1949 (calc_p_rho_phi)",
        role="diagnoses p' and inverse density al from the perturbation EOS",
        call=lambda ctx, cast: cpr.calc_p_rho_wrf(
            cast(ctx["prep"]), step=0, non_hydrostatic=True,
        ),
        outputs=lambda out: {"p": out.p, "al": out.al},
        downstream=("advance_uv PGF", "advance_w buoyancy"),
    ))
    add(Operator(
        "calc_p_rho_step",
        family="dycore.eos",
        module="gpuwrf.dynamics.core.calc_p_rho",
        entrypoint="calc_p_rho_step",
        wrf_source="dyn_em/module_small_step_em.F:1849-1949 (per-substep form with smdiv)",
        role="per-acoustic-substep EOS refresh including the divergence filter",
        call=lambda ctx, cast: cpr.calc_p_rho_step(
            mu_work=cast(ctx["prep"].mu_work), muts_total=cast(ctx["prep"].muts),
            ph_work=cast(ctx["prep"].ph_work), theta_work=cast(ctx["prep"].theta_work),
            theta_1=cast(ctx["prep"].theta_1), c2a=cast(ctx["prep"].c2a),
            alt=cast(ctx["prep"].alt), c1h=cast(ctx["metrics"].c1h),
            c2h=cast(ctx["metrics"].c2h), rdnw=cast(ctx["metrics"].rdnw),
            pm1=cast(ctx["prho"].p),
        ),
        outputs=lambda out: {"p": out.p, "al": out.al},
        downstream=("advance_uv PGF", "advance_w buoyancy"),
    ))
    add(Operator(
        "calc_al_p_kernel",
        family="dycore.eos",
        module="gpuwrf.dynamics.core.calc_p_rho",
        entrypoint="_calc_al_p",
        wrf_source="dyn_em/module_small_step_em.F:1888-1932",
        role="the raw al/p bracket without the smdiv filter",
        call=lambda ctx, cast: cpr._calc_al_p(
            mu_work=cast(ctx["prep"].mu_work), muts_total=cast(ctx["prep"].muts),
            ph_work=cast(ctx["prep"].ph_work), theta_work=cast(ctx["prep"].theta_work),
            theta_1=cast(ctx["prep"].theta_1), c2a=cast(ctx["prep"].c2a),
            alt=cast(ctx["prep"].alt), c1h=cast(ctx["metrics"].c1h),
            c2h=cast(ctx["metrics"].c2h), rdnw=cast(ctx["metrics"].rdnw), t0=300.0,
        ),
    ))
    add(Operator(
        "diagnose_pressure_al_alt",
        family="dycore.eos",
        module="gpuwrf.dynamics.acoustic_wrf",
        entrypoint="diagnose_pressure_al_alt",
        wrf_source="dyn_em/module_big_step_utilities_em.F (calc_p_rho_phi diagnostic form)",
        role="full-state pressure / al / alt diagnosis used by the large-step PGF",
        call=lambda ctx, cast: aw.diagnose_pressure_al_alt(
            cast(ctx["state"]), cast(ctx["base_state"]), cast(ctx["metrics"]), hypsometric_opt=2,
        ),
        outputs=lambda out: {"pressure": out[0], "al": out[1], "alt": out[2]},
    ))
    add(Operator(
        "moisture_coupling_factors",
        family="dycore.eos",
        module="gpuwrf.dynamics.acoustic_wrf",
        entrypoint="moisture_coupling_factors",
        wrf_source="dyn_em/module_big_step_utilities_em.F (cqu/cqv construction)",
        role="1/(1+qtot) momentum moisture coupling on u/v faces",
        call=lambda ctx, cast: aw.moisture_coupling_factors(cast(ctx["state"])),
        outputs=lambda out: {"cqu": out[0], "cqv": out[1]},
    ))

    # ---- horizontal pressure gradient / momentum --------------------------
    add(Operator(
        "horizontal_pressure_gradient",
        family="dycore.pgf",
        module="gpuwrf.dynamics.acoustic_wrf",
        entrypoint="horizontal_pressure_gradient",
        wrf_source="dyn_em/module_small_step_em.F:828-935 (advance_uv PGF terms)",
        role="the four-term non-hydrostatic horizontal PGF",
        call=lambda ctx, cast: aw.horizontal_pressure_gradient(
            cast(ctx["state"]), cast(ctx["base_state"]), cast(ctx["metrics"]),
            cast(ctx["pressure"]), cast(ctx["al"]), cast(ctx["alt"]),
            cast(ctx["cqu"]), cast(ctx["cqv"]),
            dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"],
            non_hydrostatic=True, top_lid=False,
        ),
        outputs=lambda out: {"du_dt": out[0], "dv_dt": out[1]},
        downstream=("u", "v"),
    ))
    add(Operator(
        "large_step_horizontal_pgf",
        family="dycore.pgf",
        module="gpuwrf.dynamics.core.rk_addtend_dry",
        entrypoint="large_step_horizontal_pgf",
        wrf_source="dyn_em/module_big_step_utilities_em.F (pg_ru/pg_rv)",
        role="RK large-step horizontal PGF",
        call=lambda ctx, cast: rad.large_step_horizontal_pgf(
            cast(ctx["state"]), cast(ctx["metrics"]),
            dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"],
            non_hydrostatic=True, hypsometric_opt=2, base_state=cast(ctx["base_state"]),
        ),
        outputs=lambda out: {"ru_tend": out[0], "rv_tend": out[1]},
    ))
    add(Operator(
        "large_step_coriolis",
        family="dycore.momentum",
        module="gpuwrf.dynamics.core.rk_addtend_dry",
        entrypoint="large_step_coriolis",
        wrf_source="dyn_em/module_big_step_utilities_em.F (coriolis)",
        role="Coriolis tendency on u/v faces",
        call=lambda ctx, cast: rad.large_step_coriolis(
            cast(ctx["state"]), cast(ctx["metrics"]), specified=True,
        ),
        outputs=lambda out: {"ru_tend": out[0], "rv_tend": out[1]},
    ))
    add(Operator(
        "large_step_horizontal_curvature",
        family="dycore.momentum",
        module="gpuwrf.dynamics.core.rk_addtend_dry",
        entrypoint="large_step_horizontal_curvature",
        wrf_source="dyn_em/module_big_step_utilities_em.F (curvature)",
        role="map-factor curvature tendency on u/v faces",
        call=lambda ctx, cast: rad.large_step_horizontal_curvature(
            cast(ctx["state"]), cast(ctx["metrics"]),
            dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"], specified=True,
        ),
        outputs=lambda out: {"ru_tend": out[0], "rv_tend": out[1]},
    ))
    add(Operator(
        "x_face_pressure_dpn",
        family="dycore.pgf",
        module="gpuwrf.dynamics.acoustic_wrf",
        entrypoint="x_face_pressure_dpn",
        wrf_source="dyn_em/module_small_step_em.F (dpn face construction)",
        role="x-face vertical pressure difference used by the PGF",
        call=lambda ctx, cast: aw.x_face_pressure_dpn(
            cast(ctx["pressure"]), cast(ctx["metrics"]), False,
        ),
    ))
    add(Operator(
        "y_face_pressure_dpn",
        family="dycore.pgf",
        module="gpuwrf.dynamics.acoustic_wrf",
        entrypoint="y_face_pressure_dpn",
        wrf_source="dyn_em/module_small_step_em.F (dpn face construction)",
        role="y-face vertical pressure difference used by the PGF",
        call=lambda ctx, cast: aw.y_face_pressure_dpn(
            cast(ctx["pressure"]), cast(ctx["metrics"]), False,
        ),
    ))

    # ---- mass continuity ---------------------------------------------------
    add(Operator(
        "mu_continuity_tendency",
        family="dycore.mass",
        module="gpuwrf.dynamics.acoustic_wrf",
        entrypoint="mu_continuity_tendency",
        wrf_source="dyn_em/module_small_step_em.F:1066-1175 (advance_mu_t)",
        role="column dry-mass continuity: the conserved-mass divergence",
        call=lambda ctx, cast: aw.mu_continuity_tendency(
            cast(ctx["state"]), cast(ctx["base_state"]), cast(ctx["metrics"]),
            dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"],
        ),
        conserved=lambda out: float(np.sum(np.asarray(out, dtype=np.float64))),
        downstream=("mu", "every mass-coupled scalar"),
    ))

    # ---- vertical implicit solve ------------------------------------------
    add(Operator(
        "calc_coef_w",
        family="dycore.vertical_implicit",
        module="gpuwrf.dynamics.acoustic_wrf",
        entrypoint="calc_coef_w_wrf_coefficients",
        wrf_source="dyn_em/module_small_step_em.F:1400-1500 (calc_coef_w)",
        role="tridiagonal coefficients for the implicit w/phi solve",
        call=lambda ctx, cast: aw.calc_coef_w_wrf_coefficients(
            cast(ctx["prep"].mut), cast(ctx["metrics"]),
            dt=ctx["dts"], epssm=float(ctx["config"]["epssm"]), top_lid=False,
            cqw=cast(ctx["cqw"]), c2a=cast(ctx["prep"].c2a),
        ),
        outputs=lambda out: {"a": out[0], "alpha": out[1], "gamma": out[2]},
    ))
    add(Operator(
        "thomas_solve_scan",
        family="dycore.vertical_implicit",
        module="gpuwrf.dynamics.tridiag_solve",
        entrypoint="thomas_solve_scan",
        wrf_source="dyn_em/module_small_step_em.F:1533-1550 (advance_w back-substitution)",
        role="the vertical implicit Thomas solve; the §11 Pallas operator",
        call=lambda ctx, cast: tri.thomas_solve_scan(
            cast(ctx["a"]), cast(ctx["alpha"]), cast(ctx["gamma"]),
            cast(ctx["prep"].w_work),
        ),
        outputs=lambda out: {"fwd": out[0], "solved": out[1]},
    ))
    add(Operator(
        "advance_w",
        family="dycore.vertical_implicit",
        module="gpuwrf.dynamics.core.advance_w",
        entrypoint="advance_w_wrf",
        wrf_source="dyn_em/module_small_step_em.F:1300-1600 (advance_w)",
        role="full implicit vertical momentum + geopotential advance",
        call=lambda ctx, cast: _call_advance_w(ctx, cast),
        outputs=lambda out: {"w": out[0], "ph": out[1], "ph_tend": out[2]},
        downstream=("w", "ph", "buoyancy"),
    ))
    add(Operator(
        "pg_buoy_w_dry",
        family="dycore.vertical_implicit",
        module="gpuwrf.dynamics.core.advance_w",
        entrypoint="pg_buoy_w_dry",
        wrf_source="dyn_em/module_small_step_em.F (advance_w term A/B, dry)",
        role="dry vertical pressure-gradient + buoyancy bracket",
        call=lambda ctx, cast: aww.pg_buoy_w_dry(
            cast(ctx["prho"].p), cast(ctx["prep"].mu_work),
            c1f=cast(ctx["metrics"].c1f), rdnw=cast(ctx["metrics"].rdnw),
            rdn=cast(ctx["metrics"].rdn), msfty=cast(ctx["metrics"].msfty),
        ),
    ))
    add(Operator(
        "pg_buoy_w_moist",
        family="dycore.vertical_implicit",
        module="gpuwrf.dynamics.core.advance_w",
        entrypoint="pg_buoy_w_moist",
        wrf_source="dyn_em/module_small_step_em.F (advance_w term A/B, moist)",
        role="moisture-loaded vertical PGF + buoyancy bracket",
        call=lambda ctx, cast: aww.pg_buoy_w_moist(
            cast(ctx["prho"].p), cast(ctx["prep"].mu_work), cast(ctx["prep"].mub),
            cast(ctx["cqw"]),
            c1f=cast(ctx["metrics"].c1f), c2f=cast(ctx["metrics"].c2f),
            rdnw=cast(ctx["metrics"].rdnw), rdn=cast(ctx["metrics"].rdn),
            msfty=cast(ctx["metrics"].msfty),
        ),
        outputs=lambda out: {"pg": out[0], "buoy": out[1]},
    ))
    add(Operator(
        "moist_cqw_calc_face",
        family="dycore.vertical_implicit",
        module="gpuwrf.dynamics.core.advance_w",
        entrypoint="moist_cqw_calc_face",
        wrf_source="dyn_em/module_small_step_em.F (cqw face construction)",
        role="face-interpolated moisture loading for the w solve",
        call=lambda ctx, cast: aww.moist_cqw_calc_face(
            cast(_qtot_mass(ctx)),
        ),
    ))
    add(Operator(
        "w_damp_vertical_cfl",
        family="dycore.damping",
        module="gpuwrf.dynamics.core.advance_w",
        entrypoint="w_damp_vertical_cfl",
        wrf_source="dyn_em/module_small_step_em.F (w_damping CFL filter)",
        role="vertical-CFL w damping (namelist w_damping=1)",
        call=lambda ctx, cast: aww.w_damp_vertical_cfl(
            cast(ctx["prep"].w_work), ww=cast(ctx["ww"]), w=cast(ctx["state"].w),
            mut=cast(ctx["prep"].mut), c1f=cast(ctx["metrics"].c1f),
            c2f=cast(ctx["metrics"].c2f), rdnw=cast(ctx["metrics"].rdnw),
            dt=ctx["dts"], w_damp_on=1.0,
        ),
    ))

    # ---- geopotential ------------------------------------------------------
    add(Operator(
        "rhs_ph",
        family="dycore.geopotential",
        module="gpuwrf.dynamics.core.rhs_ph",
        entrypoint="rhs_ph_wrf",
        wrf_source="dyn_em/module_big_step_utilities_em.F (rhs_ph)",
        role="geopotential RHS: horizontal + vertical phi advection and w source",
        call=lambda ctx, cast: rph.rhs_ph_wrf(
            u=cast(ctx["state"].u), v=cast(ctx["state"].v), ww=cast(ctx["ww"]),
            ph=cast(ctx["state"].ph_perturbation), phb=cast(ctx["base_state"].phb),
            w=cast(ctx["state"].w), mut=cast(ctx["prep"].mut),
            muu=cast(ctx["prep"].muu), muv=cast(ctx["prep"].muv),
            c1f=cast(ctx["metrics"].c1f), c2f=cast(ctx["metrics"].c2f),
            fnm=cast(ctx["metrics"].fnm), fnp=cast(ctx["metrics"].fnp),
            rdnw=cast(ctx["metrics"].rdnw), rdx=ctx["rdx"], rdy=ctx["rdy"],
            msfty=cast(ctx["metrics"].msfty), non_hydrostatic=True,
            specified=True, advective_order=5,
            msfux=cast(ctx["metrics"].msfux), msfvy=cast(ctx["metrics"].msfvy),
            top_lid=False,
        ),
        downstream=("ph", "w"),
    ))

    # ---- hybrid vertical coordinate ---------------------------------------
    for label, fn, source in (
        ("hybrid_mass_level_pressure", hybrid_eta.mass_level_pressure,
         "dyn_em hybrid_opt=2 mass-level pressure c3h*mu + c4h"),
        ("hybrid_face_level_pressure", hybrid_eta.face_level_pressure,
         "dyn_em hybrid_opt=2 face-level pressure c3f*mu + c4f"),
        ("hybrid_mass_weight", hybrid_eta.mass_weight,
         "dyn_em hybrid_opt=2 mass weight c1h*mu + c2h"),
        ("hybrid_face_weight", hybrid_eta.face_weight,
         "dyn_em hybrid_opt=2 face weight c1f*mu + c2f"),
        ("hybrid_pressure_thickness", hybrid_eta.pressure_thickness,
         "dyn_em hybrid_opt=2 layer pressure thickness"),
    ):
        add(Operator(
            label,
            family="dycore.vertical_coordinate",
            module="gpuwrf.dynamics.hybrid_eta",
            entrypoint=fn.__name__,
            wrf_source=source,
            role="hybrid-eta coefficient application (namelist hybrid_opt=2)",
            call=(lambda f: lambda ctx, cast: f(
                cast(ctx["state"].mu_total), cast(ctx["metrics"]),
            ))(fn),
        ))

    # ---- advection ---------------------------------------------------------
    _adv_kwargs = lambda ctx, cast: dict(  # noqa: E731
        rdx=ctx["rdx"], rdy=ctx["rdy"],
        rdzw=cast(ctx["metrics"].rdnw),
        fzm=cast(ctx["metrics"].fnm), fzp=cast(ctx["metrics"].fnp),
    )
    add(Operator(
        "advect_scalar_flux",
        family="dycore.advection",
        module="gpuwrf.dynamics.flux_advection",
        entrypoint="advect_scalar_flux",
        wrf_source="dyn_em/module_advect_em.F (advect_scalar)",
        role="flux-form 5th/3rd-order scalar advection of theta",
        call=lambda ctx, cast: flux_advection.advect_scalar_flux(
            cast(ctx["state"].theta), cast(ctx["vel"]),
            mut=cast(ctx["prep"].mut), c1=cast(ctx["metrics"].c1h),
            **_adv_kwargs(ctx, cast),
        ),
        conserved=lambda out: float(np.sum(np.asarray(out, dtype=np.float64))),
        downstream=("theta",),
    ))
    add(Operator(
        "advect_u_flux",
        family="dycore.advection",
        module="gpuwrf.dynamics.flux_advection",
        entrypoint="advect_u_flux",
        wrf_source="dyn_em/module_advect_em.F (advect_u)",
        role="flux-form momentum advection of u",
        call=lambda ctx, cast: flux_advection.advect_u_flux(
            cast(ctx["state"].u), cast(ctx["vel"]), **_adv_kwargs(ctx, cast),
        ),
        downstream=("u",),
    ))
    add(Operator(
        "advect_v_flux",
        family="dycore.advection",
        module="gpuwrf.dynamics.flux_advection",
        entrypoint="advect_v_flux",
        wrf_source="dyn_em/module_advect_em.F (advect_v)",
        role="flux-form momentum advection of v",
        call=lambda ctx, cast: flux_advection.advect_v_flux(
            cast(ctx["state"].v), cast(ctx["vel"]), **_adv_kwargs(ctx, cast),
        ),
        downstream=("v",),
    ))
    add(Operator(
        "advect_w_flux",
        family="dycore.advection",
        module="gpuwrf.dynamics.flux_advection",
        entrypoint="advect_w_flux",
        wrf_source="dyn_em/module_advect_em.F (advect_w)",
        role="flux-form momentum advection of w",
        call=lambda ctx, cast: flux_advection.advect_w_flux(
            cast(ctx["state"].w), cast(ctx["vel"]),
            rdx=ctx["rdx"], rdy=ctx["rdy"], rdn=cast(ctx["metrics"].rdn),
            fzm=cast(ctx["metrics"].fnm), fzp=cast(ctx["metrics"].fnp),
        ),
        downstream=("w",),
    ))
    add(Operator(
        "advect_moisture_scalars",
        family="dycore.advection",
        module="gpuwrf.dynamics.flux_advection",
        entrypoint="advect_moisture_scalars",
        wrf_source="dyn_em/module_advect_em.F:6069 (advect_scalar_pd, moist_adv_opt=1)",
        role="positive-definite moisture advection of the six species",
        call=lambda ctx, cast: flux_advection.advect_moisture_scalars(
            tuple(cast(getattr(ctx["state"], name))
                  for name in ("qv", "qc", "qr", "qi", "qs", "qg")),
            tuple(cast(getattr(ctx["state"], name))
                  for name in ("qv", "qc", "qr", "qi", "qs", "qg")),
            cast(ctx["vel"]),
            moist_adv_opt=int(ctx["config"]["moist_adv_opt"]),
            is_final_rk_stage=True,
            mut=cast(ctx["prep"].mut), mu_old=cast(ctx["prep"].mut),
            c1=cast(ctx["metrics"].c1h), c2=cast(ctx["metrics"].c2h),
            dt=ctx["dt_rk"], **_adv_kwargs(ctx, cast),
        ),
        downstream=("qv", "qc", "qr", "qi", "qs", "qg"),
    ))
    add(Operator(
        "flux5_face_periodic",
        family="dycore.advection",
        module="gpuwrf.dynamics.flux_advection",
        entrypoint="flux5_face_periodic",
        wrf_source="dyn_em/module_advect_em.F (flux5)",
        role="the 5th-order upwind face flux stencil",
        call=lambda ctx, cast: flux_advection.flux5_face_periodic(
            cast(ctx["state"].theta), cast(_u_mass(ctx)), 2,
        ),
    ))
    add(Operator(
        "positive_definite_limiter",
        family="dycore.advection",
        module="gpuwrf.dynamics.limiters",
        entrypoint="positive_definite_limiter",
        wrf_source="dyn_em/module_advect_em.F:6069 (advect_scalar_pd)",
        role="positive-definite moisture limiter (namelist moist_adv_opt=1)",
        call=lambda ctx, cast: limiters.positive_definite_limiter(
            cast(ctx["state"].qv), cast(_mass_column(ctx)),
            limiters.LimiterConfig(),
        ),
        conserved=lambda out: float(np.sum(np.asarray(out, dtype=np.float64))),
    ))

    # ---- diffusion / damping ----------------------------------------------
    add(Operator(
        "horizontal_deformation_2d",
        family="dycore.diffusion",
        module="gpuwrf.dynamics.explicit_diffusion",
        entrypoint="horizontal_deformation_2d",
        wrf_source="dyn_em/module_diffusion_em.F (cal_deform_and_div 2-D part)",
        role="horizontal deformation tensor feeding the Smagorinsky K",
        call=lambda ctx, cast: explicit_diffusion.horizontal_deformation_2d(
            cast(ctx["state"].u), cast(ctx["state"].v),
            dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"],
        ),
        outputs=lambda out: {f"d{i}": v for i, v in enumerate(out)}
        if isinstance(out, (tuple, list)) else {"out": out},
    ))
    add(Operator(
        "smag2d_horizontal_km",
        family="dycore.diffusion",
        module="gpuwrf.dynamics.explicit_diffusion",
        entrypoint="smag2d_horizontal_km",
        wrf_source="dyn_em/module_diffusion_em.F (smag2d_km)",
        role="2-D Smagorinsky eddy viscosity (namelist km_opt=4)",
        call=lambda ctx, cast: _call_smag2d(ctx, cast, explicit_diffusion),
        outputs=lambda out: {"xkmh": out[0], "xkhh": out[1]},
    ))
    add(Operator(
        "horizontal_diffusion_coord_scalar_tendency",
        family="dycore.diffusion",
        module="gpuwrf.dynamics.explicit_diffusion",
        entrypoint="horizontal_diffusion_coord_scalar_tendency",
        wrf_source="dyn_em/module_diffusion_em.F (horizontal_diffusion, diff_opt=1)",
        role="coordinate-surface horizontal diffusion of theta (namelist diff_opt=1)",
        call=lambda ctx, cast: explicit_diffusion.horizontal_diffusion_coord_scalar_tendency(
            cast(ctx["state"].theta), cast(_xkhh(ctx, cast, explicit_diffusion)),
            cast(_mass_column(ctx)),
            dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"],
            msftx=cast(ctx["metrics"].msftx), msfty=cast(ctx["metrics"].msfty),
            msfux=cast(ctx["metrics"].msfux), msfuy=cast(ctx["metrics"].msfuy),
            msfvx=cast(ctx["metrics"].msfvx), msfvy=cast(ctx["metrics"].msfvy),
        ),
    ))
    add(Operator(
        "wrf_sixth_order_scalar_tendf",
        family="dycore.diffusion",
        module="gpuwrf.dynamics.explicit_diffusion",
        entrypoint="wrf_sixth_order_scalar_tendf",
        wrf_source="dyn_em/module_diffusion_em.F (sixth_order_diffusion, diff_6th_opt=2)",
        role="WRF 6th-order monotonic scalar filter (namelist diff_6th_opt=2)",
        call=lambda ctx, cast: explicit_diffusion.wrf_sixth_order_scalar_tendf(
            cast(ctx["state"].theta), cast(ctx["state"].mu_total),
            c1=cast(ctx["metrics"].c1h), c2=cast(ctx["metrics"].c2h),
            msftx=cast(ctx["metrics"].msftx), msfty=cast(ctx["metrics"].msfty),
            dt=ctx["dt"], diff_6th_factor=float(ctx["config"]["diff_6th_factor"]),
            monotonic=True, specified_or_nested=True,
        ),
    ))
    add(Operator(
        "sixth_order_diffusion_tendency",
        family="dycore.diffusion",
        module="gpuwrf.dynamics.explicit_diffusion",
        entrypoint="sixth_order_diffusion_tendency",
        wrf_source="dyn_em/module_diffusion_em.F (sixth_order_diffusion kernel)",
        role="the bare 6th-order filter stencil",
        call=lambda ctx, cast: explicit_diffusion.sixth_order_diffusion_tendency(
            cast(ctx["state"].theta), dt=ctx["dt"],
            diff_6th_factor=float(ctx["config"]["diff_6th_factor"]),
        ),
    ))
    add(Operator(
        "dry_brunt_vaisala_squared",
        family="dycore.diffusion",
        module="gpuwrf.dynamics.explicit_diffusion",
        entrypoint="dry_brunt_vaisala_squared",
        wrf_source="dyn_em/module_diffusion_em.F (bvf calculation)",
        role="dry N^2 from the vertical theta gradient: a difference of near-equal values",
        call=lambda ctx, cast: explicit_diffusion.dry_brunt_vaisala_squared(
            cast(ctx["state"].theta), dz_m=cast(_dz_mean_2d(ctx)),
        ),
    ))
    add(Operator(
        "apply_rayleigh_w",
        family="dycore.damping",
        module="gpuwrf.dynamics.damping",
        entrypoint="apply_rayleigh_w",
        wrf_source="dyn_em/module_small_step_em.F:1559-1569 (w Rayleigh damping, damp_opt=3)",
        role="upper-level Rayleigh damping on w (namelist damp_opt=3)",
        call=lambda ctx, cast: damping.apply_rayleigh_w(
            cast(ctx["state"].w),
            damping.RayleighConfig(
                enabled=True,
                coefficient=float(ctx["config"]["dampcoef"]),
            ),
        ),
    ))
    add(Operator(
        "apply_smdiv_pressure",
        family="dycore.damping",
        module="gpuwrf.dynamics.damping",
        entrypoint="apply_smdiv_pressure",
        wrf_source="dyn_em/module_small_step_em.F (smdiv divergence damping)",
        role="acoustic divergence damping on p'",
        call=lambda ctx, cast: damping.apply_smdiv_pressure(
            cast(ctx["prho"].p), cast(ctx["state"].p_perturbation),
            damping.SmdivConfig(),
        ),
    ))

    ops.extend(build_physics_inventory())
    return ops


def build_physics_inventory() -> list[Operator]:
    """Every ACTIVE d01 physics package, resolved by the production dispatcher.

    The five suite members come from ``resolve_physics_suite`` rather than from
    reading the namelist by eye, so the inventory cannot silently drift from
    what the model would actually run. Radiation and gravity-wave drag are not
    part of ``PhysicsSuite`` (they are selected by ``ra_*_physics`` / ``gwd_opt``
    and dispatched separately) and are added explicitly.
    """

    from gpuwrf.coupling import physics_couplers as pc

    ops: list[Operator] = []
    add = ops.append

    add(Operator(
        "thompson_microphysics",
        family="physics.microphysics",
        module="gpuwrf.coupling.physics_couplers",
        entrypoint="thompson_adapter",
        wrf_source="phys/module_mp_thompson.F (mp_physics=8)",
        role="Thompson bulk microphysics; saturation adjustment is the classic fp32 risk",
        activity="resolve_physics_suite -> microphysics option 8",
        call=lambda ctx, cast: pc.thompson_adapter(
            cast(ctx["state"]), ctx["dt"], ctx["grid"],
        ),
        outputs=lambda out: _state_delta_outputs(out, ("qv", "qc", "qr", "qi", "qs", "qg", "theta", "Ni", "Nr")),
        downstream=("qv", "qc", "qr", "qi", "qs", "qg", "theta", "rain_acc"),
    ))
    add(Operator(
        "mynn_pbl",
        family="physics.pbl",
        module="gpuwrf.coupling.physics_couplers",
        entrypoint="mynn_adapter",
        wrf_source="phys/module_bl_mynn.F (bl_pbl_physics=5)",
        role="MYNN 2.5 PBL; the TKE budget is the known fp32 non-finite risk (qke is FP64-locked)",
        activity="resolve_physics_suite -> pbl option 5",
        call=lambda ctx, cast: pc.mynn_adapter(
            cast(ctx["state"]), ctx["dt"], ctx["grid"],
        ),
        outputs=lambda out: _state_delta_outputs(out, ("u", "v", "theta", "qv", "qke", "qc")),
        downstream=("u", "v", "theta", "qv", "qke"),
    ))
    add(Operator(
        "mynn_surface_layer",
        family="physics.surface_layer",
        module="gpuwrf.coupling.physics_couplers",
        entrypoint="surface_adapter",
        wrf_source="phys/module_sf_mynn.F (sf_sfclay_physics=5)",
        role="MYNN surface layer; Monin-Obukhov iteration on near-equal stability ratios",
        activity="resolve_physics_suite -> surface_layer option 5",
        call=lambda ctx, cast: pc.surface_adapter(
            cast(ctx["state"]), ctx["dt"], ctx["grid"],
        ),
        outputs=lambda out: _state_delta_outputs(
            out, ("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        ),
        downstream=("ustar", "theta_flux", "qv_flux", "PBL forcing"),
    ))
    add(Operator(
        "rrtmg_lw",
        family="physics.radiation",
        module="gpuwrf.coupling.physics_couplers",
        entrypoint="rrtmg_lw_theta_tendency",
        wrf_source="phys/module_ra_rrtmg_lw.F (ra_lw_physics=4)",
        role="RRTMG longwave; a cadence-event operator (radt=30 min)",
        activity="namelist ra_lw_physics=4; dispatched outside PhysicsSuite",
        call=lambda ctx, cast: pc.rrtmg_lw_theta_tendency(
            cast(ctx["state"]), ctx["grid"], time_utc=ctx["time_utc"],
        ),
        downstream=("theta",),
    ))
    add(Operator(
        "rrtmg_sw",
        family="physics.radiation",
        module="gpuwrf.coupling.physics_couplers",
        entrypoint="rrtmg_sw_theta_tendency",
        wrf_source="phys/module_ra_rrtmg_sw.F (ra_sw_physics=4)",
        role="RRTMG shortwave with topo_shading=1 / slope_rad=1; cadence-event operator",
        activity="namelist ra_sw_physics=4; dispatched outside PhysicsSuite",
        call=lambda ctx, cast: pc.rrtmg_sw_theta_tendency(
            cast(ctx["state"]), ctx["grid"], time_utc=ctx["time_utc"],
            topo_shading=1, slope_rad=1,
        ),
        downstream=("theta",),
    ))
    add(Operator(
        "gwdo_gravity_wave_drag",
        family="physics.gwd",
        module="gpuwrf.coupling.physics_couplers",
        entrypoint="gwdo_adapter",
        wrf_source="phys/module_bl_gwdo.F (gwd_opt=1)",
        role="sub-grid orographic gravity-wave drag (namelist gwd_opt=1 on d01)",
        activity="namelist gwd_opt=1",
        call=lambda ctx, cast: pc.gwdo_adapter(
            cast(ctx["state"]), ctx["dt"], ctx["gwdo_statics"], ctx["grid"],
        ),
        outputs=lambda out: _state_delta_outputs(out, ("u", "v")),
        downstream=("u", "v"),
    ))
    add(Operator(
        "kain_fritsch_cumulus",
        family="physics.cumulus",
        module="gpuwrf.physics.cumulus_kf",
        entrypoint="step_kf_column",
        wrf_source="phys/module_cu_kfeta.F (cu_physics=1)",
        role="Kain-Fritsch cumulus on d01; column CAPE closure over near-cancelling buoyancy",
        activity="resolve_physics_suite -> cumulus option 1",
        call=lambda ctx, cast: _call_kf(ctx, cast),
        downstream=("theta", "qv", "rainc_acc"),
    ))
    add(Operator(
        "noahmp_land_surface",
        family="physics.land_surface",
        module="gpuwrf.coupling.noahmp_surface_hook",
        entrypoint="noahmp_surface_step",
        wrf_source="phys/module_sf_noahmpdrv.F (sf_surface_physics=4)",
        role="Noah-MP land surface; the energy-balance closure is a difference of large fluxes",
        activity="resolve_physics_suite -> land_surface option 4",
        call=lambda ctx, cast: _call_noahmp(ctx, cast),
        outputs=lambda out: _state_delta_outputs(out[0], ("t_skin", "theta_flux", "qv_flux", "ustar")),
        downstream=("t_skin", "surface fluxes"),
    ))
    return ops


def _state_delta_outputs(new_state: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """Read named leaves off a returned State (or pass an array through)."""

    outputs: dict[str, Any] = {}
    for name in fields:
        value = getattr(new_state, name, None)
        if value is not None and hasattr(value, "dtype"):
            outputs[name] = value
    return outputs or _default_outputs(new_state)


KF_COLUMN_SAMPLE = 128


def _call_kf(ctx: dict[str, Any], cast):
    """Kain-Fritsch on real columns built from the real state.

    ``step_kf_column`` is a single-column program, so it is vmapped over a
    deterministic stride-sampled subset of the real domain rather than all
    8400 columns: the full sweep costs minutes per arm in the CPU interpreter
    and adds nothing to a conditioning measurement. The sample is fixed and
    recorded, so the three arms see exactly the same columns.
    """
    import jax
    import jax.numpy as jnp

    from gpuwrf.physics.cumulus_kf import step_kf_column

    state = cast(ctx["state"])
    p = state.p_total
    temperature = state.theta * (p / 1.0e5) ** (287.0 / 1004.5)
    dz = cast(_dz_column(ctx))
    rho = p / (287.0 * jnp.maximum(temperature, 1.0))
    u_mass = 0.5 * (state.u[:, :, :-1] + state.u[:, :, 1:])
    v_mass = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
    w_mass = 0.5 * (state.w[:-1] + state.w[1:])

    def columns(field):
        flat = field.reshape(field.shape[0], -1)
        stride = max(1, flat.shape[1] // KF_COLUMN_SAMPLE)
        return jnp.transpose(flat[:, ::stride][:, :KF_COLUMN_SAMPLE], (1, 0))

    stepcu = max(1, int(round(ctx["config"]["cudt_min"] * 60.0 / ctx["dt"])))

    def one(t, qv, pr, dzq, rhoe, w0, u0, v0):
        # Only the state tendencies are returned: the full PhysicsStepResult
        # carries non-array leaves that vmap cannot batch, and the tendencies
        # are the quantity whose conditioning this map is about.
        out = step_kf_column(
            t, qv, pr, dzq, rhoe, w0, u0, v0,
            ctx["dt"], ctx["config"]["dx"],
            stepcu=stepcu, cudt=float(ctx["config"]["cudt_min"]),
        )
        return {
            name: jnp.asarray(value)
            for name, value in out.tendency.state_tendencies.items()
        }

    packed = [columns(f) for f in
              (temperature, state.qv, p, dz, rho, w_mass, u_mass, v_mass)]
    return jax.vmap(one)(*packed)


def _call_noahmp(ctx: dict[str, Any], cast):
    """Noah-MP with the real land carry cold-started from the case's wrfinput."""
    from gpuwrf.coupling.noahmp_surface_hook import noahmp_surface_step

    return noahmp_surface_step(
        cast(ctx["state"]),
        cast(ctx["noahmp_land"]),
        ctx["noahmp_static"],
        ctx["dt"],
        grid=ctx["grid"],
    )


# --- helpers the specs above use ------------------------------------------- #
def _qtot_mass(ctx: dict[str, Any]):
    import jax.numpy as jnp

    state = ctx["state"]
    total = jnp.zeros_like(state.qv)
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        field = getattr(state, name, None)
        if field is not None:
            total = total + jnp.asarray(field)
    return total


def _u_mass(ctx: dict[str, Any]):
    """u interpolated to mass points, for the bare stencil probe."""
    state = ctx["state"]
    return 0.5 * (state.u[:, :, :-1] + state.u[:, :, 1:])


def _mass_column(ctx: dict[str, Any]):
    import jax.numpy as jnp

    metrics = ctx["metrics"]
    mu = ctx["state"].mu_total
    return metrics.c1h[:, None, None] * mu[None, :, :] + metrics.c2h[:, None, None]


def _dz_column(ctx: dict[str, Any]):
    """Real layer thickness from the geopotential, in metres."""
    import jax.numpy as jnp

    ph = ctx["state"].ph_total
    return jnp.maximum((ph[1:] - ph[:-1]) / 9.81, 1.0)


def _dz_mean_2d(ctx: dict[str, Any]):
    """Column-mean real layer thickness, (ny, nx).

    ``dry_brunt_vaisala_squared`` takes a single ``dz_m`` broadcast against a
    2-D level slice, so it cannot accept the full 3-D thickness. The
    column-mean of the REAL thickness is passed rather than a nominal constant.
    """
    import jax.numpy as jnp

    return jnp.mean(_dz_column(ctx), axis=0)


def _call_advance_w(ctx: dict[str, Any], cast):
    from gpuwrf.dynamics.core.advance_w import advance_w_wrf

    prep = cast(ctx["prep"])
    metrics = cast(ctx["metrics"])
    config = ctx["config"]
    return advance_w_wrf(
        w=prep.w_work, rw_tend=cast(ctx["tendencies"].w), ww=cast(ctx["ww"]),
        u=prep.u_work, v=prep.v_work, mu_work=prep.mu_work, mut=prep.mut,
        muave=cast(ctx["state"].mu_perturbation), muts=prep.muts,
        t_2ave=prep.theta_work, t_2=prep.theta_work, t_1=prep.theta_1,
        ph=prep.ph_work, ph_1=prep.ph_1, phb=prep.phb,
        ph_tend=cast(ctx["tendencies"].ph),
        ht=cast(ctx["base_state"].phb[0] / 9.81),
        c2a=prep.c2a, cqw=cast(ctx["cqw"]), alt=prep.alt,
        a=cast(ctx["a"]), alpha=cast(ctx["alpha"]), gamma=cast(ctx["gamma"]),
        c1h=metrics.c1h, c2h=metrics.c2h, c1f=metrics.c1f, c2f=metrics.c2f,
        rdnw=metrics.rdnw, rdn=metrics.rdn, fnm=metrics.fnm, fnp=metrics.fnp,
        cf1=metrics.cf1, cf2=metrics.cf2, cf3=metrics.cf3,
        msftx=metrics.msftx, msfty=metrics.msfty,
        rdx=ctx["rdx"], rdy=ctx["rdy"], dts=ctx["dts"],
        epssm=float(config["epssm"]), top_lid=False,
        w_save=prep.w_save, damp_opt=int(config["damp_opt"]),
        dampcoef=float(config["dampcoef"]), zdamp=float(config["zdamp"]),
        w_damping=int(config["w_damping"]),
    )


def _call_smag2d(ctx: dict[str, Any], cast, explicit_diffusion):
    """2-D Smagorinsky K from the real deformation tensor."""
    deformation = explicit_diffusion.horizontal_deformation_2d(
        cast(ctx["state"].u), cast(ctx["state"].v),
        dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"],
    )
    d11, d22, d12 = deformation[0], deformation[1], deformation[2]
    return explicit_diffusion.smag2d_horizontal_km(
        d11, d22, d12,
        dx_m=ctx["config"]["dx"], dy_m=ctx["config"]["dy"],
        msftx=cast(ctx["metrics"].msftx), msfty=cast(ctx["metrics"].msfty),
    )


def _xkhh(ctx: dict[str, Any], cast, explicit_diffusion):
    """The real scalar eddy diffusivity produced by the Smagorinsky closure."""
    return _call_smag2d(ctx, cast, explicit_diffusion)[1]


# --------------------------------------------------------------------------- #
# per-operator evaluation                                                      #
# --------------------------------------------------------------------------- #
def evaluate(op: Operator, ctx: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen three-arm comparison for one operator."""

    import jax

    record: dict[str, Any] = {
        "name": op.name,
        "family": op.family,
        "module": op.module,
        "entrypoint": op.entrypoint,
        "wrf_source": op.wrf_source,
        "role": op.role,
        "activity": op.activity,
        "active": True,
        "downstream_sensitive_fields": list(op.downstream),
    }

    # Static algebra scan of the operator's own source.
    try:
        module = __import__(op.module, fromlist=["*"])
        target = getattr(module, op.entrypoint)
        source = inspect.getsource(target)
        scan = canc.scan_algebra(source)
        record["source_algebra"] = {
            "total_minus_perturbation_occurrences": list(scan.total_minus_perturbation),
            "subtraction_count": scan.subtraction_count,
            "force_fp64_island_calls": scan.force_fp64_island_calls,
            "has_explicit_fp64_island": scan.has_explicit_island,
        }
    except Exception as exc:  # noqa: BLE001
        record["source_algebra"] = {"error": f"{type(exc).__name__}: {exc}"}

    identity = make_cast("fp64")
    started = time.perf_counter()
    try:
        base_result = jax.block_until_ready(op.call(ctx, identity))
    except Exception as exc:  # noqa: BLE001
        record["status"] = "NOT_EVALUATED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["classification"] = None
        return record

    base_outputs = op.outputs(base_result)
    base_vector = canc.combine_outputs(base_outputs)
    record["output_names"] = sorted(base_outputs)
    record["input_output_scale"] = {
        name: canc.field_scale(value) for name, value in base_outputs.items()
    }
    record["output_dtypes_fp64_arm"] = {
        name: str(getattr(value, "dtype", "?")) for name, value in base_outputs.items()
    }

    # Prove the fp32 demotion actually reached every input this operator reads.
    # A cast that walks past an opaque container leaves the arm running in fp64
    # while reporting a tiny error -- which reads as fp32 safety and is wrong.
    probe_cast = make_cast("fp32")
    residual_fp64_inputs: set[str] = set()

    def _recording_cast(value: Any) -> Any:
        result = probe_cast(value)
        residual_fp64_inputs.update(
            dtype for dtype in float_dtypes_present(result) if dtype != "float32"
        )
        return result

    try:
        op.call(ctx, _recording_cast)
    except Exception:  # noqa: BLE001 - the arm below reports the real failure
        pass
    record["fp32_arm_inputs_fully_demoted"] = not residual_fp64_inputs
    record["fp32_arm_residual_input_dtypes"] = sorted(residual_fp64_inputs)

    arms: dict[str, Any] = {}
    for mode in ("repr32", "fp32"):
        cast = make_cast(mode)
        try:
            result = jax.block_until_ready(op.call(ctx, cast))
            outputs = op.outputs(result)
            vector = canc.combine_outputs(outputs)
            arms[mode] = canc.error_metrics(vector, base_vector)
            arms[mode]["output_dtypes"] = {
                name: str(getattr(value, "dtype", "?")) for name, value in outputs.items()
            }
            if op.conserved is not None:
                base_integral = op.conserved(base_result)
                arm_integral = op.conserved(result)
                scale = abs(base_integral) if base_integral != 0.0 else 1.0
                arms[mode]["conservation"] = {
                    "fp64_integral": base_integral,
                    "arm_integral": float(arm_integral),
                    "relative_delta": float(abs(arm_integral - base_integral) / scale),
                }
        except Exception as exc:  # noqa: BLE001
            arms[mode] = {"error": f"{type(exc).__name__}: {exc}"}

    # Fourth arm: fp32 with every fp64 island neutralised. With fp32 inputs the
    # island is NOT inert, so the plain fp32 arm above measures island-PROTECTED
    # fp32. This arm is the true aggressive-fp32 number and, by difference, the
    # measured benefit of the islands this operator actually reaches.
    #
    # It is run for EVERY operator, not only those whose own body calls
    # force_fp64_island. The lexical scan is body-local and cannot see an island
    # inside a helper the operator calls -- diagnose_pressure_al_alt reaches one
    # that way. Running the arm unconditionally turns "does this operator have a
    # live island?" from a source guess into a measurement: if neutralising the
    # symbol changes nothing, no island was reached.
    island_off: dict[str, Any] | None = None
    island_off_vector: np.ndarray | None = None
    cast = make_cast("fp32")
    try:
        with islands_disabled() as patched_modules:
            result = jax.block_until_ready(op.call(ctx, cast))
            island_off_vector = canc.combine_outputs(op.outputs(result))
            island_off = canc.error_metrics(island_off_vector, base_vector)
            island_off["patched_modules"] = patched_modules
    except Exception as exc:  # noqa: BLE001
        island_off = {"error": f"{type(exc).__name__}: {exc}"}

    protected = arms.get("fp32", {})
    p99_protected = float(protected.get("p99_rel", float("nan")))
    p99_unprotected = float((island_off or {}).get("p99_rel", float("nan")))
    has_island = bool(
        island_off is not None
        and "error" not in island_off
        and np.isfinite(p99_protected)
        and np.isfinite(p99_unprotected)
        and not np.isclose(p99_protected, p99_unprotected, rtol=1e-12, atol=0.0)
    )
    record["reaches_live_fp64_island"] = has_island
    record["reaches_live_fp64_island_basis"] = (
        "measured: neutralising force_fp64_island changed this operator's fp32 output"
        if has_island
        else "measured: neutralising force_fp64_island left this operator's fp32 output unchanged"
    )

    record["arms"] = {
        "fp64_baseline": {"note": "reference; all errors are measured against this arm"},
        "fp32_representation_only": arms.get("repr32"),
        "fp32_islands_active": arms.get("fp32") if has_island else None,
        "fp32_aggressive": island_off if has_island else arms.get("fp32"),
        "note": (
            "for an operator containing force_fp64_island, 'fp32_aggressive' is the "
            "island-NEUTRALISED arm; 'fp32_islands_active' is what today's code does "
            "under fp32 storage. For an operator with no island the two are the same run."
        ),
    }

    # Condition probe: response to a relative fp32-sized input perturbation.
    def probe(epsilon: float, seed: int) -> np.ndarray:
        cast = make_cast("probe", seed=seed, epsilon=epsilon)
        result = op.call(ctx, cast)
        return canc.combine_outputs(op.outputs(result))

    try:
        record["condition"] = canc.condition_number(
            probe, base_vector, samples=canc.FROZEN["condition_probe_samples"],
        )
    except Exception as exc:  # noqa: BLE001
        record["condition"] = {"error": f"{type(exc).__name__}: {exc}"}

    aggressive = (island_off if (island_off and "error" not in island_off)
                  else arms.get("fp32", {}))
    representation = arms.get("repr32", {})
    p99_total = float(aggressive.get("p99_rel", float("nan")))
    p99_repr = float(representation.get("p99_rel", float("nan")))
    finite = bool(aggressive.get("all_finite", False))
    classification, reason = canc.classify(p99_total, p99_repr, finite=finite)

    record["island_gain_bound"] = canc.island_gain(p99_total, p99_repr)
    record["island_gain_bound_note"] = (
        "ceiling for ANY in-operator fp64 island, from the representation floor"
    )
    if has_island:
        record["existing_island_measured_benefit"] = canc.island_gain(p99_total, p99_protected)
        record["existing_island_verdict"] = (
            "SUPPORTED_BY_MEASUREMENT"
            if record["existing_island_measured_benefit"] >= canc.FROZEN["island_gain_threshold"]
            else "NOT_SUPPORTED_BY_MEASUREMENT"
        )
    else:
        record["existing_island_measured_benefit"] = None
        record["existing_island_verdict"] = "NO_LIVE_ISLAND_REACHED"

    record["fp32_amplification_vs_unit_roundoff"] = (
        p99_total / canc.EPS32 if np.isfinite(p99_total) else None
    )

    # --- honesty guards: three ways a zero error means nothing ---------------
    baseline_finite = base_vector[np.isfinite(base_vector)]
    not_exercised = bool(baseline_finite.size == 0 or np.all(baseline_finite == 0.0))
    fp32_dtypes = list((aggressive.get("output_dtypes") or {}).values())
    dtype_normalised = bool(
        fp32_dtypes and any(dtype == "float64" for dtype in fp32_dtypes)
    )
    arm_failed = "error" in (aggressive or {})

    if arm_failed:
        classification = "FP32_TRACE_FAILS"
        reason = (
            "the operator does not trace under fp32 inputs at all: "
            f"{aggressive['error']}. That is a harder result than a large error and is "
            "reported as its own status rather than hidden behind one."
        )
    elif not_exercised:
        classification = "NOT_EXERCISED_ON_THIS_STATE"
        reason = (
            "the fp64 baseline output is identically zero on this state, so no arithmetic "
            "was exercised and no fp32 verdict is earned here"
        )
    elif dtype_normalised:
        classification = "INCONCLUSIVE_DTYPE_NORMALISED"
        reason = (
            "the operator re-normalises its inputs to fp64 internally, so the fp32 arm never "
            "ran in fp32; the measured delta describes the adapter boundary, not the scheme"
        )
    elif not record["fp32_arm_inputs_fully_demoted"]:
        classification = "INCONCLUSIVE_DTYPE_NORMALISED"
        reason = (
            "the fp32 arm still reached inputs at "
            f"{record['fp32_arm_residual_input_dtypes']}: the demotion did not cover every "
            "container this operator reads, so the measured error understates aggressive fp32"
        )

    record["classification"] = classification
    record["classification_reason"] = reason
    record["not_exercised_on_this_state"] = not_exercised
    record["dtype_normalising_operator"] = dtype_normalised
    record["status"] = "EVALUATED"
    record["seconds"] = time.perf_counter() - started

    # --- aliases consumed by scripts/v025/validate_cancellation_map.py -------
    record["operator"] = op.name
    record["input_dtype"] = "float64 (production storage); arms recast to float32"
    record["output_dtype"] = record.get("output_dtypes_fp64_arm")
    record["fp64_result"] = record.get("input_output_scale")
    record["fp32_result"] = {
        "scale_source": "fp32_aggressive arm",
        "p99_rel": p99_total,
        "all_finite": finite,
        "output_dtypes": aggressive.get("output_dtypes"),
    }
    record["max_rel_error"] = aggressive.get("max_rel")
    record["median_rel_error"] = aggressive.get("median_rel")
    record["p99_rel_error"] = aggressive.get("p99_rel")
    record["ulps"] = {
        "max_fp32_ulp": aggressive.get("max_ulp_fp32"),
        "p99_fp32_ulp": aggressive.get("p99_ulp_fp32"),
    }
    record["source_algebra_identity"] = record.get("source_algebra")
    record["conservation_contribution"] = (aggressive or {}).get("conservation")
    return record


# --------------------------------------------------------------------------- #
# island census + physics activity proof                                       #
# --------------------------------------------------------------------------- #
def island_call_sites() -> list[dict[str, Any]]:
    """The live ``force_fp64_island`` call sites, straight from the source tree."""

    proc = subprocess.run(
        ["grep", "-rn", "force_fp64_island(", str(REPO / "src" / "gpuwrf")],
        capture_output=True, text=True, check=False,
    )
    sites: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        path, _, rest = line.partition(":")
        lineno, _, text = rest.partition(":")
        stripped = text.strip()
        if stripped.startswith("#") or "def force_fp64_island" in stripped:
            continue
        sites.append({
            "file": str(Path(path).relative_to(REPO)),
            "line": int(lineno),
            "code": stripped,
        })
    return sites


def physics_activity(config: dict[str, Any]) -> dict[str, Any]:
    """Executable proof of which physics schemes are active for this case.

    §10 allows an inventory entry to be omitted only with "an explicit
    executable proof that an inventory entry is inactive". This resolves the
    production dispatch table for the real namelist options, so the active set
    is produced by the model's own resolver rather than asserted here.
    """

    from gpuwrf.coupling.physics_dispatch import dispatch_matrix, resolve_physics_suite

    namelist_keys = (
        "mp_physics", "bl_pbl_physics", "sf_sfclay_physics",
        "cu_physics", "sf_surface_physics",
    )
    suite = resolve_physics_suite(
        {"physics": {key: int(config[key]) for key in namelist_keys}}
    )
    summary = suite.summary()
    selected = {
        row["namelist_key"]: int(row["option"])
        for row in [
            {"namelist_key": key, "option": value["option"]}
            for key, value in summary["schemes"].items()
        ]
    }
    matrix = dispatch_matrix()
    inactive = [
        {
            "namelist_key": row["namelist_key"],
            "option": row["option"],
            "name": row["name"],
            "selected_option": selected.get(row["namelist_key"]),
            "proof": (
                f"resolve_physics_suite dispatches {row['namelist_key']}="
                f"{selected.get(row['namelist_key'])} for this case; option {row['option']} "
                f"({row['name']}) is never routed, so its operators are not on the active path"
            ),
        }
        for row in matrix["rows"]
        if selected.get(row["namelist_key"]) != row["option"]
    ]
    return {
        "resolver": "gpuwrf.coupling.physics_dispatch.resolve_physics_suite",
        "active_suite": summary,
        "radiation_and_gwd": {
            "ra_lw_physics": int(config["ra_lw_physics"]),
            "ra_sw_physics": int(config["ra_sw_physics"]),
            "gwd_opt": int(config["gwd_opt"]),
            "sf_urban_physics": int(config["sf_urban_physics"]),
            "note": (
                "radiation and GWD are selected outside PhysicsSuite; they are inventoried "
                "explicitly. sf_urban_physics=0 means no urban operator is on the path."
            ),
        },
        "inactive_count": len(inactive),
        "inactive_schemes": inactive,
        "namelist_options": {key: int(config[key]) for key in namelist_keys},
        "dynamics_options": {
            key: config[key] for key in (
                "diff_opt", "km_opt", "diff_6th_opt", "diff_6th_factor",
                "damp_opt", "w_damping", "zdamp", "dampcoef", "gwd_opt",
                "moist_adv_opt", "scalar_adv_opt", "hybrid_opt",
                "non_hydrostatic", "epssm", "specified",
            )
        },
    }


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--domain", default="d01",
                        help="real production domain to load (d01..d09)")
    parser.add_argument("--previous", default=None,
                        help="EXACT earlier wrfout filename for the stage pair")
    parser.add_argument("--snapshot", default=None,
                        help="EXACT later wrfout filename for the stage pair")
    parser.add_argument("--expect-sha256", action="append", default=[],
                        help="required sha256 of --previous then --snapshot; fail-closed")
    parser.add_argument("--out", type=Path,
                        default=REPO / "proofs/v025/m0/cancellation_map.json")
    parser.add_argument("--only", default=None, help="substring filter for a quick probe")
    args = parser.parse_args()

    environment = assert_cpu_only()
    run_dir = args.run_dir or default_run_dir()
    started = time.perf_counter()
    pair = select_state_pair(Path(run_dir), args.domain,
                            previous=args.previous, snapshot=args.snapshot,
                            expect_sha256=args.expect_sha256)
    snapshot = load_real_snapshot(run_dir, wrfout_name=pair["snapshot"].name, domain=args.domain)
    previous = load_real_snapshot(run_dir, wrfout_name=pair["previous"].name, domain=args.domain)
    ctx = build_context(snapshot, previous, domain=args.domain)

    inventory = build_inventory()
    if args.only:
        inventory = [op for op in inventory if args.only in op.name]

    records = []
    for op in inventory:
        record = evaluate(op, ctx)
        records.append(record)
        status = record.get("classification") or record.get("status")
        print(f"  {op.name:<34s} {status}", flush=True)

    evaluated = [r for r in records if r["status"] == "EVALUATED"]
    by_class: dict[str, list[str]] = {}
    for record in evaluated:
        by_class.setdefault(record["classification"], []).append(record["name"])

    sites = island_call_sites()
    proposed = sorted(
        record["name"] for record in evaluated
        if record["classification"] == "FP64_ISLAND_CANDIDATE"
    )

    obj = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_section": "§10 cancellation-sensitivity map",
        "phase": "A5 (CPU-only; no GPU import, query, compile, or run)",
        "frozen_thresholds": canc.FROZEN,
        "method": {
            "arms": {
                "fp64_baseline": "production dtype; the reference for every error",
                "fp32_representation_only": (
                    "every float input rounded through fp32 and widened back to fp64; "
                    "arithmetic still exact. Isolates the irreducible storage penalty "
                    "that an in-operator fp64 island CANNOT remove."
                ),
                "fp32_aggressive": (
                    "every float input stored and computed in fp32; what an aggressive "
                    "fp32 rewrite actually pays."
                ),
            },
            "island_gain": (
                "1 - p99_rel(representation_only) / p99_rel(aggressive). The fraction of the "
                "fp32 penalty that upcasting inputs at the operator boundary can remove."
            ),
            "why_this_answers_minimality": (
                "force_fp64_island upcasts inputs INSIDE the operator, so it can only remove "
                "arithmetic error. An operator whose error is dominated by input representation "
                "gains nothing from an island; keeping one there is cost without benefit, which "
                "is why 'keep all current islands for safety' is not an admissible answer."
            ),
        },
        # Short form consumed by validate_cancellation_map.py; the full evidence
        # is in state_source_detail so nothing is lost to the alias.
        "state_source": "20260725_18z_production_snapshot",
        "coverage_fraction": (len(evaluated) / len(records)) if records else 0.0,
        "rationale": (
            "islands proposed only where an in-operator upcast measurably removes at least "
            "half of the aggressive-fp32 error"
        ),
        "proposed_fp64_islands": [
            {
                "site": record["name"],
                "module": record["module"],
                "entrypoint": record["entrypoint"],
                "evidence": {
                    "fp32_amplification_vs_unit_roundoff":
                        record.get("fp32_amplification_vs_unit_roundoff"),
                    "island_gain_bound": record.get("island_gain_bound"),
                    "condition_kappa_median": (record.get("condition") or {}).get("kappa_median"),
                    "p99_rel_aggressive_fp32": record.get("p99_rel_error"),
                    "classification_reason": record.get("classification_reason"),
                },
            }
            for record in evaluated
            if record["classification"] == "FP64_ISLAND_CANDIDATE"
        ],
        "state_source_detail": {
            "state_source_is_real": True,
            "stage_pair": ctx["stage_pair"],
            **snapshot.provenance,
        },
        "existing_island_audit": {
            "operators_reaching_a_live_island": sorted(
                r["name"] for r in records if r.get("reaches_live_fp64_island")
            ),
            "operators_with_body_local_island_call": sorted(
                r["name"] for r in records
                if r.get("source_algebra", {}).get("has_explicit_fp64_island")
            ),
            "supported_by_measurement": sorted(
                r["name"] for r in records
                if r.get("existing_island_verdict") == "SUPPORTED_BY_MEASUREMENT"
            ),
            "not_supported_by_measurement": sorted(
                r["name"] for r in records
                if r.get("existing_island_verdict") == "NOT_SUPPORTED_BY_MEASUREMENT"
            ),
            "method": (
                "the island symbol is monkeypatched to a pass-through and the fp32 arm rerun "
                "with cleared JAX caches, for EVERY operator. Island presence is therefore "
                "measured (did the output change?) rather than inferred from a body-local "
                "source scan, which cannot see an island inside a called helper."
            ),
            "removal_candidates": (
                "operators listed under not_supported_by_measurement reach a live island whose "
                "measured benefit is below the frozen 0.50 threshold. §10 requires the proposed "
                "island set to be MINIMAL; these are where the current set is not."
            ),
        },
        "case_config": snapshot.config,
        "environment": {
            "host": platform.node(),
            "python": sys.version.split()[0],
            "jax_platforms": environment,
        },
        "physics_activity_proof": physics_activity(snapshot.config),
        "force_fp64_island_call_sites": {
            "count": len(sites),
            "sites": sites,
        },
        "inventory": {
            "total_entries": len(records),
            "evaluated": len(evaluated),
            "not_evaluated": [
                {"name": r["name"], "error": r.get("error")}
                for r in records if r["status"] != "EVALUATED"
            ],
            "coverage_fraction": (len(evaluated) / len(records)) if records else 0.0,
        },
        "classification_summary": {
            key: {"count": len(value), "operators": sorted(value)}
            for key, value in sorted(by_class.items())
        },
        "proposed_fp64_island_set": {
            "operators": proposed,
            "count": len(proposed),
            "rule": (
                "an operator is proposed for an fp64 island only when its measured "
                "island_gain >= 0.50 outside the safe band; every other operator is either "
                "safe in fp32 or must be reformulated"
            ),
            "policy_note": (
                "this is a MAP and a PROPOSAL. Contract §10: it is not authority to change "
                "precision policy. PRECISION_MATRIX and force_fp64_island are untouched."
            ),
        },
        "operators": records,
        "seconds_total": time.perf_counter() - started,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {args.out}")
    print(f"  entries {len(records)}  evaluated {len(evaluated)}  "
          f"islands proposed {len(proposed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
