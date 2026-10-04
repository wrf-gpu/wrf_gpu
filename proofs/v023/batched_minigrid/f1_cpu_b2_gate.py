#!/usr/bin/env python3
"""Real-dycore B=2 correctness gate for the F1 batched runtime path.

This proof drives the production two-domain ``DomainTree`` callbacks with real
``State`` / ``OperationalCarry`` objects.  By default it runs on the JAX CPU
backend; set ``F1_GATE_BACKEND=gpu`` for the GPU confirmation.  The batched case
uses the F1 outer ``jax.vmap`` around the real ``_advance_chunk`` and
``build_child_boundary_package`` path.

The original byte-identity expectation is intentionally relaxed here.  The real
gate diagnosis proved the non-zero lane-vs-standalone delta is present with a
singleton vmap and identical-IC lanes remain byte-identical, so the accepted CPU
property is:

* contamination-free vmap orchestration;
* per-lane real-dycore differences no larger than the measured CPU singleton
  vmap roundoff for this fixture.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

F1_GATE_BACKEND = os.environ.get("F1_GATE_BACKEND", "cpu").strip().lower()
if F1_GATE_BACKEND in ("", "cpu"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
elif F1_GATE_BACKEND not in ("gpu", "cuda"):
    raise ValueError(f"F1_GATE_BACKEND must be cpu or gpu, got {F1_GATE_BACKEND!r}")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "0")
os.environ.setdefault("GPUWRF_NESTED_AOT", "0")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)
if F1_GATE_BACKEND in ("gpu", "cuda") and jax.default_backend() == "cpu":
    raise RuntimeError("F1_GATE_BACKEND=gpu requested but JAX default backend is CPU")

from gpuwrf.contracts.grid import (  # noqa: E402
    BCMetadata,
    DomainHierarchy,
    DomainNest,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes  # noqa: E402
from gpuwrf.runtime.domain_tree import (  # noqa: E402
    DomainBundle,
    DomainTree,
    run_domain_tree_callbacks,
    run_operational_domain_tree,
    with_live_child_boundary_config,
)
from gpuwrf.nesting.boundary_construction import build_child_boundary_package  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    OperationalNamelist,
    _advance_chunk,
    _initial_carry_for_run,
    build_clock_base,
)
from gpuwrf.runtime.operational_state import OperationalCarry  # noqa: E402
from gpuwrf.integration.nested_pipeline import run_batched_operational_domain_tree  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
JSON_OUT = Path(
    os.environ.get(
        "F1_GATE_OUTPUT",
        str(OUT_DIR / "F1_CPU_B2_BIT_IDENTITY.json"),
    )
)

RUN_STARTS = (
    datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    datetime(2026, 1, 2, 6, 0, tzinfo=timezone.utc),
)
DOMAIN_NAMES = ("d01", "d02")


def batch_ensemble_size_from_env() -> int:
    raw = os.environ.get("GPUWRF_BATCH_ENSEMBLE", "1").strip()
    return int(raw or "1")
ROOT_STEPS = 2
NEST_RATIO = 3
OUTPUT_CADENCE_STEPS = {"d01": 1, "d02": NEST_RATIO}
CPU_STANDALONE_VMAP_ABS_TOL = 1.0e-9
CPU_STANDALONE_VMAP_REL_TOL = 1.0e-12
GRAVITY = 9.81
R_D = 287.0
C_P = 1004.0
P0_PA = 100000.0
THETA0_K = 300.0
RUNTIME_ENV_KEYS = (
    "GPUWRF_BATCH_ENSEMBLE",
    "GPUWRF_NESTED_FUSE",
    "GPUWRF_BITWISE",
)


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _set_runtime_mode(*, batch: int, fused: bool) -> None:
    os.environ["GPUWRF_BATCH_ENSEMBLE"] = str(int(batch))
    if fused:
        os.environ["GPUWRF_NESTED_FUSE"] = "1"
        os.environ.pop("GPUWRF_BITWISE", None)
    else:
        os.environ["GPUWRF_NESTED_FUSE"] = "0"
        os.environ["GPUWRF_BITWISE"] = "1"


def _vertical_profile(nz: int, *, dz_m: float = 300.0) -> dict[str, np.ndarray | float]:
    z_face = np.arange(int(nz) + 1, dtype=np.float64) * float(dz_m)
    p_face = np.zeros(int(nz) + 1, dtype=np.float64)
    p_face[0] = P0_PA
    for k in range(int(nz)):
        dz = z_face[k + 1] - z_face[k]
        exner = (p_face[k] / P0_PA) ** (R_D / C_P)
        temperature = THETA0_K * exner
        rho = p_face[k] / (R_D * temperature)
        p_face[k + 1] = p_face[k] - GRAVITY * rho * dz
    p_top = float(p_face[-1])
    mu = float(p_face[0] - p_top)
    eta = (p_face - p_top) / mu
    p_mass = 0.5 * (p_face[:-1] + p_face[1:])
    ph_face = GRAVITY * z_face
    return {"eta": eta, "p_mass": p_mass, "p_top": p_top, "mu": mu, "ph_face": ph_face}


def _grid(domain: str, *, nz: int = 6, ny: int = 6, nx: int = 6) -> GridSpec:
    profile = _vertical_profile(nz)
    eta = jnp.asarray(profile["eta"], dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain = TerrainProvenance(
        source_path=f"f1-realgate-{domain}",
        sha256=f"f1-realgate-{domain}",
        shape=(ny, nx),
        units="m",
        projection_transform="native-wrf-lambert",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny,
        nx=nx,
        nz=nz,
        eta_levels=eta,
        top_pressure_pa=float(profile["p_top"]),
        provenance=f"f1-realgate-flat-{domain}",
    )
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    metrics = DycoreMetrics(
        msftx=metrics.msftx,
        msfty=metrics.msfty,
        msfux=metrics.msfux,
        msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,
        msfvy=metrics.msfvy,
        c1h=one_h,
        c2h=zero_h,
        c3h=one_h,
        c4h=zero_h,
        c1f=one_f,
        c2f=zero_f,
        c3f=one_f,
        c4f=zero_f,
        dn=metrics.dn,
        dnw=metrics.dnw,
        rdn=metrics.rdn,
        rdnw=metrics.rdnw,
        cf1=metrics.cf1,
        cf2=metrics.cf2,
        cf3=metrics.cf3,
        fnm=metrics.fnm,
        fnp=metrics.fnp,
        dzdx=metrics.dzdx,
        dzdy=metrics.dzdy,
        dzdx_u=metrics.dzdx_u,
        dzdy_v=metrics.dzdy_v,
        f=metrics.f,
        e=metrics.e,
        sina=metrics.sina,
        cosa=metrics.cosa,
        p_top=metrics.p_top,
        provenance=f"f1-realgate-flat-{domain}-pure-sigma",
    )
    return GridSpec(
        projection=projection,
        terrain=terrain,
        vertical=vertical,
        bc=bc,
        eta_levels=eta,
        terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
        metrics=metrics,
    )


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    zero = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    return Tendencies(
        zero((nz, ny, nx + 1)),
        zero((nz, ny + 1, nx)),
        zero((nz + 1, ny, nx)),
        zero((nz, ny, nx)),
        zero((nz, ny, nx)),
        zero((nz, ny, nx)),
        zero((nz + 1, ny, nx)),
        zero((ny, nx)),
    )


def _base_state(grid: GridSpec, *, seed: int, domain_bias: float) -> State:
    """Deterministic, non-uniform CPU state with real WRF-shaped leaves."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    rng = np.random.default_rng(seed)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in _state_field_shapes(grid).items()}

    profile = _vertical_profile(nz)
    z_face = np.arange(nz + 1, dtype=np.float64) * 300.0
    z_mid = 0.5 * (z_face[:-1] + z_face[1:])
    theta_col = np.full((nz,), THETA0_K, dtype=np.float64)
    p_col = np.asarray(profile["p_mass"], dtype=np.float64)

    def mass3(base: np.ndarray | float, noise: float) -> jax.Array:
        arr = np.asarray(base, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full((nz,), float(arr), dtype=np.float64)
        out = arr[:, None, None] + noise * rng.standard_normal((nz, ny, nx))
        return jnp.asarray(out, dtype=jnp.float64)

    fields["theta"] = mass3(theta_col, 0.0)
    fields["p_total"] = mass3(p_col, 0.0)
    fields["p_perturbation"] = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    fields["qv"] = jnp.clip(mass3(0.010 * np.exp(-z_mid / 3000.0), 1.0e-5), 0.0, None)
    fields["qc"] = jnp.clip(mass3(np.where(z_mid < 1200.0, 2.0e-5, 0.0), 2.0e-6), 0.0, None)
    fields["qr"] = jnp.clip(mass3(np.where((z_mid > 300.0) & (z_mid < 1500.0), 1.0e-5, 0.0), 1.0e-6), 0.0, None)
    fields["qi"] = jnp.clip(mass3(np.zeros(nz), 1.0e-8), 0.0, None)
    fields["qs"] = jnp.clip(mass3(np.zeros(nz), 1.0e-8), 0.0, None)
    fields["Ni"] = jnp.full((nz, ny, nx), 1.0e5 + seed, dtype=jnp.float64)
    fields["Nr"] = jnp.full((nz, ny, nx), 1.0e5 + 2 * seed, dtype=jnp.float64)
    fields["Ns"] = jnp.full((nz, ny, nx), 5.0e4 + seed, dtype=jnp.float64)
    fields["Ng"] = jnp.full((nz, ny, nx), 1.0e4 + seed, dtype=jnp.float64)
    fields["Nc"] = jnp.full((nz, ny, nx), 1.0e8 + 16 * seed, dtype=jnp.float64)
    fields["Nn"] = jnp.full((nz, ny, nx), 1.0e8 + 8 * seed, dtype=jnp.float64)
    fields["u"] = jnp.asarray(1.0 + 0.1 * domain_bias + 0.001 * rng.standard_normal((nz, ny, nx + 1)), dtype=jnp.float64)
    fields["v"] = jnp.asarray(-0.5 + 0.05 * domain_bias + 0.001 * rng.standard_normal((nz, ny + 1, nx)), dtype=jnp.float64)
    fields["w"] = jnp.asarray(1.0e-5 * rng.standard_normal((nz + 1, ny, nx)), dtype=jnp.float64)
    ph_total = jnp.asarray(
        np.broadcast_to(np.asarray(profile["ph_face"])[:, None, None], (nz + 1, ny, nx)),
        dtype=jnp.float64,
    )
    fields["ph_total"] = ph_total
    fields["ph_perturbation"] = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    fields["mu_total"] = jnp.full((ny, nx), float(profile["mu"]), dtype=jnp.float64)
    fields["mu_perturbation"] = jnp.zeros((ny, nx), dtype=jnp.float64)
    fields["qke"] = jnp.full((nz, ny, nx), 0.3 + 0.001 * seed, dtype=jnp.float64)
    fields["xland"] = jnp.ones((ny, nx), dtype=jnp.float64)
    fields["t_skin"] = jnp.full((ny, nx), 300.0 + domain_bias, dtype=jnp.float64)
    fields["soil_moisture"] = jnp.full((ny, nx), 0.25 + 1.0e-4 * seed, dtype=jnp.float64)
    fields["mavail"] = jnp.full((ny, nx), 0.6, dtype=jnp.float64)
    fields["roughness_m"] = jnp.full((ny, nx), 0.1, dtype=jnp.float64)
    fields["ustar"] = jnp.full((ny, nx), 0.25, dtype=jnp.float64)
    fields["rhosfc"] = jnp.full((ny, nx), 1.15, dtype=jnp.float64)
    fields["lu_index"] = jnp.zeros((ny, nx), dtype=jnp.int32)
    return State(**fields)


def _namelist(
    grid: GridSpec,
    *,
    time_utc: datetime,
    dt_s: float,
    parent_dt_s: float | None,
    run_boundary: bool,
) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(
        grid,
        tendencies=_cpu_tendencies(grid),
        metrics=grid.metrics,
        dt_s=dt_s,
        acoustic_substeps=2,
        radiation_cadence_steps=999999,
        force_fp64=True,
        disable_guards=True,
        use_vertical_solver=True,
    )
    base = dataclasses.replace(
        base,
        time_utc=time_utc,
        run_physics=False,
        run_boundary=bool(run_boundary),
        noahmp_julian=float(time_utc.timetuple().tm_yday - 1) + time_utc.hour / 24.0,
        noahmp_yearlen=366.0 if time_utc.year % 4 == 0 else 365.0,
        mp_physics=0,
        bl_pbl_physics=0,
        sf_sfclay_physics=0,
        cu_physics=0,
        ra_sw_physics=0,
        ra_lw_physics=0,
    )
    if parent_dt_s is not None:
        base = with_live_child_boundary_config(base, parent_dt_s=parent_dt_s)
    return base


def _lane_tree(lane: int) -> DomainTree:
    parent_grid = _grid("d01")
    child_grid = _grid("d02")
    start = RUN_STARTS[int(lane)]
    parent_dt = 0.5
    child_dt = parent_dt / float(NEST_RATIO)
    domains = {
        "d01": DomainBundle(
            "d01",
            _base_state(parent_grid, seed=101 + 17 * lane, domain_bias=0.0),
            _namelist(
                parent_grid,
                time_utc=start,
                dt_s=parent_dt,
                parent_dt_s=None,
                run_boundary=False,
            ),
            grid=parent_grid,
            metrics=parent_grid.metrics,
        ),
        "d02": DomainBundle(
            "d02",
            _base_state(child_grid, seed=401 + 19 * lane, domain_bias=1.5),
            _namelist(
                child_grid,
                time_utc=start,
                dt_s=child_dt,
                parent_dt_s=parent_dt,
                run_boundary=True,
            ),
            grid=child_grid,
            metrics=child_grid.metrics,
        ),
    }
    hierarchy = DomainHierarchy.from_edges(
        DOMAIN_NAMES,
        (DomainNest("d01", "d02", NEST_RATIO, 2, 2),),
        max_dom=2,
    )
    return DomainTree.from_domains(hierarchy, domains, registration="sint", feedback_enabled=False)


def _stack_pytrees(*items: Any) -> Any:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *items)


def _slice_batched_tree(tree: Any, lane: int) -> Any:
    return jax.tree_util.tree_map(lambda x: x[int(lane)], tree)


def _batch_namelists(lanes: tuple[DomainTree, DomainTree]) -> dict[str, tuple[OperationalNamelist, ...]]:
    return {
        name: tuple(lane.domains[name].namelist for lane in lanes)
        for name in DOMAIN_NAMES
    }


def _initial_carries(tree: DomainTree) -> dict[str, OperationalCarry]:
    return {
        name: _initial_carry_for_run(bundle.state, bundle.namelist)
        for name, bundle in tree.domains.items()
    }


def _batched_initial_carries(lanes: tuple[DomainTree, DomainTree]) -> dict[str, OperationalCarry]:
    lane_carries = tuple(_initial_carries(tree) for tree in lanes)
    return {
        name: _stack_pytrees(*(carries[name] for carries in lane_carries))
        for name in DOMAIN_NAMES
    }


class _OutputRecorder:
    wants_carry = True

    def __init__(self, *, batch_size: int) -> None:
        self.batch_size = int(batch_size)

    def __call__(self, name: str, own_step: int, carry: Any) -> dict[str, Any]:
        if self.batch_size == 1:
            lanes = [_output_digest(carry)]
        else:
            lanes = [
                _output_digest(_slice_batched_tree(carry, lane))
                for lane in range(self.batch_size)
            ]
        return {"domain": name, "own_step": int(own_step), "lanes": lanes}


def _output_digest(carry: OperationalCarry) -> dict[str, str]:
    return {
        "theta_sha256": _array_sha256(carry.state.theta),
        "u_sha256": _array_sha256(carry.state.u),
        "v_sha256": _array_sha256(carry.state.v),
        "qv_sha256": _array_sha256(carry.state.qv),
        "mu_total_sha256": _array_sha256(carry.state.mu_total),
    }


def _run_tree(
    tree: DomainTree,
    *,
    carries: dict[str, Any] | None,
    batch_size: int,
    batch_namelists: dict[str, tuple[OperationalNamelist, ...]] | None = None,
):
    output = _OutputRecorder(batch_size=batch_size)
    run_tree = run_batched_operational_domain_tree if int(batch_size) > 1 else run_operational_domain_tree
    kwargs: dict[str, Any] = {}
    if int(batch_size) > 1:
        if batch_namelists is None:
            raise ValueError("B>1 proof run requires explicit batch_namelists")
        kwargs["batch_namelists"] = batch_namelists
        kwargs["batch_size"] = int(batch_size)
    result = run_tree(
        tree,
        root_steps=ROOT_STEPS,
        carries=carries,
        output=output,
        output_cadence_steps=OUTPUT_CADENCE_STEPS,
        block_between=False,
        **kwargs,
    )
    jax.block_until_ready([result.states[name].theta for name in DOMAIN_NAMES])
    return result


class _SingletonVmapOutputRecorder:
    wants_carry = True

    def __call__(self, name: str, own_step: int, carry: Any) -> dict[str, Any]:
        return {
            "domain": name,
            "own_step": int(own_step),
            "lanes": [_output_digest(_slice_batched_tree(carry, 0))],
        }


def _run_singleton_vmap_tree(tree: DomainTree):
    """Run the real two-domain fixture through a length-1 manual vmap shell."""

    base_carries = _initial_carries(tree)
    carries = {
        name: _stack_pytrees(base_carries[name])
        for name in DOMAIN_NAMES
    }
    clock_bases = {
        name: _stack_pytrees(build_clock_base(tree.domains[name].namelist))
        for name in DOMAIN_NAMES
    }
    edge_by_pair = {
        (edge.parent, edge.child): edge
        for edges in tree.edges.values()
        for edge in edges
    }

    def lookup(spec):
        return edge_by_pair[(spec.parent, spec.child)]

    def advance(name: str, carry: OperationalCarry, start_step: int, n_steps: int) -> OperationalCarry:
        namelist = tree.domains[name].namelist
        cadence = int(namelist.radiation_cadence_steps)
        start = jnp.asarray(int(start_step), dtype=jnp.int32)
        clock_base = clock_bases[name]
        return jax.vmap(
            lambda lane_carry, lane_clock_base: _advance_chunk(
                lane_carry,
                namelist,
                start,
                lane_clock_base,
                n_steps=int(n_steps),
                cadence=int(cadence),
            ),
            in_axes=(0, 0),
            out_axes=0,
        )(carry, clock_base)

    def force(edge, parent: OperationalCarry, child: OperationalCarry) -> OperationalCarry:
        bdy_width = int(child.state.u_bdy.shape[3])
        forced_state = jax.vmap(
            lambda lane_child_state, lane_parent_state: build_child_boundary_package(
                lane_child_state,
                lane_parent_state,
                edge.weights,
                bdy_width=bdy_width,
            ),
            in_axes=(0, 0),
            out_axes=0,
        )(child.state, parent.state)
        return child.replace(state=forced_state)

    result = run_domain_tree_callbacks(
        tree.hierarchy,
        carries,
        root_steps=ROOT_STEPS,
        advance=advance,
        force=force,
        output=_SingletonVmapOutputRecorder(),
        output_cadence_steps=OUTPUT_CADENCE_STEPS,
        block_between=False,
        edge_lookup=lookup,
    )
    sliced_carries = {
        name: _slice_batched_tree(result.carries[name], 0)
        for name in DOMAIN_NAMES
    }
    sliced_states = {
        name: _slice_batched_tree(result.states[name], 0)
        for name in DOMAIN_NAMES
    }
    jax.block_until_ready([sliced_states[name].theta for name in DOMAIN_NAMES])
    return dataclasses.replace(result, carries=sliced_carries, states=sliced_states)


def _array_sha256(value: Any) -> str:
    import hashlib

    arr = np.asarray(jax.device_get(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _named_arrays(obj: Any, prefix: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def add(name: str, value: Any) -> None:
        if value is None:
            return
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            arrays[name] = np.asarray(jax.device_get(value))
            return
        arrays.update(_named_arrays(value, name))

    if isinstance(obj, OperationalCarry):
        for name in obj.__dataclass_fields__:
            add(f"{prefix}.{name}", getattr(obj, name))
        return arrays
    if isinstance(obj, State):
        for name in obj.active_field_names():
            add(f"{prefix}.{name}", getattr(obj, name))
        add(f"{prefix}.p", obj.p)
        add(f"{prefix}.ph", obj.ph)
        add(f"{prefix}.mu", obj.mu)
        return arrays
    if isinstance(obj, Tendencies):
        for name in obj.__slots__:
            add(f"{prefix}.{name}", getattr(obj, name))
        return arrays
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            add(f"{prefix}.{field.name}", getattr(obj, field.name))
        return arrays
    if isinstance(obj, tuple):
        for idx, value in enumerate(obj):
            add(f"{prefix}[{idx}]", value)
        return arrays
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            add(f"{prefix}[{idx}]", value)
        return arrays
    if isinstance(obj, dict):
        for key, value in obj.items():
            add(f"{prefix}.{key}", value)
        return arrays
    return arrays


def _array_metric(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(
            f"shape/dtype differs: ref={reference.shape}/{reference.dtype} "
            f"candidate={candidate.shape}/{candidate.dtype}"
        )
    byte_identical = bool(np.array_equal(reference, candidate))
    if np.issubdtype(reference.dtype, np.floating):
        reference_finite = bool(np.all(np.isfinite(reference)))
        candidate_finite = bool(np.all(np.isfinite(candidate)))
        if not reference_finite or not candidate_finite:
            max_abs = float("nan")
            max_rel = float("nan")
            denom = float("nan")
        elif reference.size:
            abs_delta = np.abs(candidate - reference)
            max_abs = float(np.max(abs_delta))
            denom = float(np.max(np.abs(reference)))
            denom = max(denom, float(np.finfo(reference.dtype).tiny))
            max_rel = float(max_abs / denom)
        else:
            max_abs = 0.0
            max_rel = 0.0
            denom = 1.0
    elif reference.size:
        max_abs = float(np.max(np.abs(candidate.astype(np.int64) - reference.astype(np.int64))))
        denom = max(float(np.max(np.abs(reference.astype(np.int64)))), 1.0)
        max_rel = float(max_abs / denom)
        reference_finite = True
        candidate_finite = True
    else:
        max_abs = 0.0
        max_rel = 0.0
        denom = 1.0
        reference_finite = True
        candidate_finite = True
    return {
        "max_abs_diff": max_abs,
        "max_relative_diff": max_rel,
        "relative_denominator": denom,
        "byte_identical": byte_identical,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
    }


def _compare_arrays(reference: Any, candidate: Any) -> dict[str, dict[str, Any]]:
    ref_arrays = _named_arrays(reference, "carry")
    cand_arrays = _named_arrays(candidate, "carry")
    if set(ref_arrays) != set(cand_arrays):
        missing = sorted(set(ref_arrays) - set(cand_arrays))
        extra = sorted(set(cand_arrays) - set(ref_arrays))
        raise AssertionError(f"array field set differs; missing={missing} extra={extra}")

    metrics: dict[str, dict[str, Any]] = {}
    for name in sorted(ref_arrays):
        metrics[name] = _array_metric(ref_arrays[name], cand_arrays[name])
    return metrics


def _compare_standalone_to_batched(
    standalone_results: tuple[Any, Any],
    batched_result: Any,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    comparisons: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for lane, reference_result in enumerate(standalone_results):
        lane_key = f"lane{lane}"
        comparisons[lane_key] = {}
        for domain in DOMAIN_NAMES:
            candidate = _slice_batched_tree(batched_result.carries[domain], lane)
            reference = reference_result.carries[domain]
            comparisons[lane_key][domain] = _compare_arrays(reference, candidate)
    return comparisons


def _compare_batched_to_batched(reference_result: Any, candidate_result: Any) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    comparisons: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for lane in range(2):
        lane_key = f"lane{lane}"
        comparisons[lane_key] = {}
        for domain in DOMAIN_NAMES:
            reference = _slice_batched_tree(reference_result.carries[domain], lane)
            candidate = _slice_batched_tree(candidate_result.carries[domain], lane)
            comparisons[lane_key][domain] = _compare_arrays(reference, candidate)
    return comparisons


def _compare_singleton_to_standalone(
    standalone_result: Any,
    singleton_result: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        domain: _compare_arrays(standalone_result.carries[domain], singleton_result.carries[domain])
        for domain in DOMAIN_NAMES
    }


def _compare_singleton_to_batched_lane(
    singleton_result: Any,
    batched_result: Any,
    lane: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        domain: _compare_arrays(
            singleton_result.carries[domain],
            _slice_batched_tree(batched_result.carries[domain], lane),
        )
        for domain in DOMAIN_NAMES
    }


def _compare_batched_lanes(result: Any, lane_a: int, lane_b: int) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        domain: _compare_arrays(
            _slice_batched_tree(result.carries[domain], lane_a),
            _slice_batched_tree(result.carries[domain], lane_b),
        )
        for domain in DOMAIN_NAMES
    }


def _all_byte_identical(comparison: Any) -> bool:
    if isinstance(comparison, dict):
        if "byte_identical" in comparison:
            return bool(comparison["byte_identical"])
        return all(_all_byte_identical(value) for value in comparison.values())
    return True


def _output_schedule(outputs: tuple[Any, ...]) -> list[tuple[str, int]]:
    return [(str(item["domain"]), int(item["own_step"])) for item in outputs]


def _output_schedules_match(
    standalone_results: tuple[Any, Any],
    batched_result: Any,
) -> bool:
    batched_schedule = _output_schedule(batched_result.outputs)
    return all(_output_schedule(result.outputs) == batched_schedule for result in standalone_results)


def _identical_lane_outputs_equal(result: Any) -> bool:
    for item in result.outputs:
        lanes = item.get("lanes", ())
        if len(lanes) < 2 or lanes[0] != lanes[1]:
            return False
    return True


def _max_metric(comparison: Any, metric: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None

    def visit(node: Any, path: tuple[str, ...]) -> None:
        nonlocal best
        if isinstance(node, dict) and metric in node:
            value = float(node[metric])
            if best is None or value > float(best["value"]):
                best = {
                    "value": value,
                    "path": "/".join(path),
                    "field": path[-1] if path else "",
                    "metric": metric,
                }
            return
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, (*path, str(key)))

    visit(comparison, ())
    if best is None:
        return {"value": 0.0, "path": "", "field": "", "metric": metric}
    return best


def _legacy_max_abs_diff(comparison: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        lane: {
            domain: {
                field: float(metrics["max_abs_diff"])
                for field, metrics in fields.items()
            }
            for domain, fields in domains.items()
        }
        for lane, domains in comparison.items()
    }


def _nonfinite_failures(label: str, comparison: Any) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict) and "reference_finite" in node:
            if not bool(node["reference_finite"]) or not bool(node["candidate_finite"]):
                failures.append(
                    {
                        "comparison": label,
                        "path": "/".join(path),
                        "reason": "non_finite",
                        "reference_finite": bool(node["reference_finite"]),
                        "candidate_finite": bool(node["candidate_finite"]),
                    }
                )
            return
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, (*path, str(key)))

    visit(comparison, ())
    return failures


def _build_payload(
    *,
    verdict: str,
    standalone_results: tuple[Any, Any],
    batched_eager_result: Any,
    batched_fused_result: Any,
    singleton_vmap_result: Any,
    comparisons: dict[str, Any],
    global_metrics: dict[str, Any],
    checks: dict[str, bool],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    field_counts = {
        lane: {domain: len(fields) for domain, fields in domains.items()}
        for lane, domains in comparisons["f1_batched_eager_vs_standalone"].items()
    }
    return {
        "verdict": str(verdict),
        "gate": "F1_REAL_DYCORE_CPU_B2_CONTAMINATION_FREE_TOLERANCE",
        "root_cause_verdict": "B_XLA_ORDER_CONTAMINATION_FREE",
        "byte_identity_vs_standalone": "relaxed_unachievable_under_vmap_on_this_fixture",
        "backend": jax.default_backend(),
        "batch_ensemble_env": int(batch_ensemble_size_from_env()),
        "root_steps": ROOT_STEPS,
        "nest_ratio": NEST_RATIO,
        "output_cadence_steps": OUTPUT_CADENCE_STEPS,
        "relative_diff_definition": "max_abs_diff / max(max(abs(reference_field)), finfo(dtype).tiny)",
        "real_ops": {
            "advance": "gpuwrf.runtime.operational_mode._advance_chunk",
            "boundary": "gpuwrf.nesting.boundary_construction.build_child_boundary_package",
            "runner": "gpuwrf.runtime.domain_tree.run_operational_domain_tree",
            "batched_wrapper": "F1 outer jax.vmap in _operational_advance_factory/_build_fused_cascade_program/_operational_force",
        },
        "configuration": {
            "run_physics": False,
            "run_boundary": {"d01": False, "d02": True},
            "force_fp64": True,
            "acoustic_substeps": 2,
            "dt_s": {"d01": 0.5, "d02": 0.5 / 3.0},
            "domains": list(DOMAIN_NAMES),
        },
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "global_metrics": global_metrics,
        "lanes": [
            {"lane": idx, "run_start": RUN_STARTS[idx].isoformat()}
            for idx in range(2)
        ],
        "own_steps": {
            "standalone": [result.own_steps for result in standalone_results],
            "singleton_vmap": singleton_vmap_result.own_steps,
            "batched_eager": batched_eager_result.own_steps,
            "batched_fused": batched_fused_result.own_steps,
        },
        "output_counts": {
            "standalone": [len(result.outputs) for result in standalone_results],
            "singleton_vmap": len(singleton_vmap_result.outputs),
            "batched_eager": len(batched_eager_result.outputs),
            "batched_fused": len(batched_fused_result.outputs),
        },
        "field_counts": field_counts,
        "max_abs_diff": _legacy_max_abs_diff(comparisons["f1_batched_eager_vs_standalone"]),
        "comparisons": comparisons,
    }


def main() -> int:
    original_env = {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS}
    try:
        lanes = (_lane_tree(0), _lane_tree(1))

        _set_runtime_mode(batch=0, fused=False)
        standalone_results = tuple(
            _run_tree(tree, carries=None, batch_size=1)
            for tree in lanes
        )

        singleton_vmap_result = _run_singleton_vmap_tree(lanes[0])

        _set_runtime_mode(batch=2, fused=False)
        batched_eager_result = _run_tree(
            lanes[0],
            carries=_batched_initial_carries(lanes),
            batch_size=2,
            batch_namelists=_batch_namelists(lanes),
        )

        identical_lanes = (_lane_tree(0), _lane_tree(0))
        identical_batched_result = _run_tree(
            identical_lanes[0],
            carries=_batched_initial_carries(identical_lanes),
            batch_size=2,
            batch_namelists=_batch_namelists(identical_lanes),
        )

        _set_runtime_mode(batch=2, fused=True)
        batched_fused_result = _run_tree(
            lanes[0],
            carries=_batched_initial_carries(lanes),
            batch_size=2,
            batch_namelists=_batch_namelists(lanes),
        )

        comparisons = {
            "f1_batched_eager_vs_standalone": _compare_standalone_to_batched(
                standalone_results,
                batched_eager_result,
            ),
            "f1_default_fused_batched_vs_standalone": _compare_standalone_to_batched(
                standalone_results,
                batched_fused_result,
            ),
            "fused_vs_eager_band": _compare_batched_to_batched(
                batched_eager_result,
                batched_fused_result,
            ),
            "singleton_vmap_vs_standalone_lane0": _compare_singleton_to_standalone(
                standalone_results[0],
                singleton_vmap_result,
            ),
            "singleton_vmap_vs_batched_eager_lane0": _compare_singleton_to_batched_lane(
                singleton_vmap_result,
                batched_eager_result,
                0,
            ),
            "identical_ic_b2_lane0_vs_lane1": _compare_batched_lanes(
                identical_batched_result,
                0,
                1,
            ),
        }

        f1_abs = _max_metric(comparisons["f1_batched_eager_vs_standalone"], "max_abs_diff")
        f1_rel = _max_metric(comparisons["f1_batched_eager_vs_standalone"], "max_relative_diff")
        default_f1_abs = _max_metric(comparisons["f1_default_fused_batched_vs_standalone"], "max_abs_diff")
        default_f1_rel = _max_metric(comparisons["f1_default_fused_batched_vs_standalone"], "max_relative_diff")
        band_abs = _max_metric(comparisons["fused_vs_eager_band"], "max_abs_diff")
        band_rel = _max_metric(comparisons["fused_vs_eager_band"], "max_relative_diff")
        singleton_abs = _max_metric(comparisons["singleton_vmap_vs_standalone_lane0"], "max_abs_diff")
        singleton_rel = _max_metric(comparisons["singleton_vmap_vs_standalone_lane0"], "max_relative_diff")
        singleton_b2_abs = _max_metric(comparisons["singleton_vmap_vs_batched_eager_lane0"], "max_abs_diff")
        singleton_b2_rel = _max_metric(comparisons["singleton_vmap_vs_batched_eager_lane0"], "max_relative_diff")

        checks = {
            "output_schedules_match_batched_eager": _output_schedules_match(
                standalone_results,
                batched_eager_result,
            ),
            "output_schedules_match_batched_fused": _output_schedules_match(
                standalone_results,
                batched_fused_result,
            ),
            "identical_ic_b2_lanes_byte_identical": _all_byte_identical(
                comparisons["identical_ic_b2_lane0_vs_lane1"]
            ),
            "identical_ic_b2_output_lanes_byte_identical": _identical_lane_outputs_equal(
                identical_batched_result
            ),
            "singleton_vmap_standalone_abs_within_cpu_roundoff": (
                float(singleton_abs["value"]) <= CPU_STANDALONE_VMAP_ABS_TOL
            ),
            "singleton_vmap_standalone_rel_within_cpu_roundoff": (
                float(singleton_rel["value"]) <= CPU_STANDALONE_VMAP_REL_TOL
            ),
            "f1_abs_within_cpu_singleton_roundoff": float(f1_abs["value"]) <= CPU_STANDALONE_VMAP_ABS_TOL,
            "f1_rel_within_cpu_singleton_roundoff": float(f1_rel["value"]) <= CPU_STANDALONE_VMAP_REL_TOL,
            "default_fused_f1_abs_within_cpu_singleton_roundoff": (
                float(default_f1_abs["value"]) <= CPU_STANDALONE_VMAP_ABS_TOL
            ),
            "default_fused_f1_rel_within_cpu_singleton_roundoff": (
                float(default_f1_rel["value"]) <= CPU_STANDALONE_VMAP_REL_TOL
            ),
        }
        if jax.default_backend() == "cpu":
            checks["singleton_vmap_reproduces_b2_lane0_byte_identical"] = _all_byte_identical(
                comparisons["singleton_vmap_vs_batched_eager_lane0"]
            )
        else:
            checks["singleton_vmap_b2_lane0_abs_within_global_fused_band"] = (
                float(singleton_b2_abs["value"]) <= float(band_abs["value"])
            )
            checks["singleton_vmap_b2_lane0_rel_within_global_fused_band"] = (
                float(singleton_b2_rel["value"]) <= float(band_rel["value"])
            )

        global_metrics = {
            "f1_batched_eager_vs_standalone": {
                "max_abs": f1_abs,
                "max_relative": f1_rel,
            },
            "f1_default_fused_batched_vs_standalone": {
                "max_abs": default_f1_abs,
                "max_relative": default_f1_rel,
            },
            "fused_vs_eager_band": {
                "max_abs": band_abs,
                "max_relative": band_rel,
            },
            "singleton_vmap_vs_standalone_lane0": {
                "max_abs": singleton_abs,
                "max_relative": singleton_rel,
            },
            "singleton_vmap_vs_batched_eager_lane0": {
                "max_abs": singleton_b2_abs,
                "max_relative": singleton_b2_rel,
            },
        }

        failures: list[dict[str, Any]] = []
        for name, passed in checks.items():
            if not passed:
                failures.append({"check": name, "reason": "check_failed"})
        for label, comparison in comparisons.items():
            failures.extend(_nonfinite_failures(label, comparison))

        payload = _build_payload(
            verdict="FAIL" if failures else "PASS",
            standalone_results=standalone_results,
            singleton_vmap_result=singleton_vmap_result,
            batched_eager_result=batched_eager_result,
            batched_fused_result=batched_fused_result,
            comparisons=comparisons,
            global_metrics=global_metrics,
            checks=checks,
            failures=failures,
        )
        JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if failures:
            raise AssertionError(
                "F1 real B=2 contamination-free tolerance gate failed; "
                f"failures={failures}; proof={JSON_OUT}"
            )
        print(
            f"PASS F1 real {jax.default_backend()} B=2 contamination-free tolerance gate: "
            f"F1 max_abs={f1_abs['value']:.12g} <= {CPU_STANDALONE_VMAP_ABS_TOL:.1e}; "
            f"F1 max_rel={f1_rel['value']:.12g} <= {CPU_STANDALONE_VMAP_REL_TOL:.1e}; "
            f"proof={JSON_OUT}"
        )
        return 0
    finally:
        _restore_env(original_env)


if __name__ == "__main__":
    raise SystemExit(main())
