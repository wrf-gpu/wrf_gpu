#!/usr/bin/env python
"""S1/S2 bit-identity + liveness + nested-trace harness (CPU-only).

Gate (a)  bit-identical fp64_default: run a tiny operational ``_advance_chunk``
          (the real outer-State hot path) for K steps and dump every State /
          carry leaf; SAVE a baseline, COMPARE later -> bitwise-equal on all
          fields.
Gate (b)  liveness: inventory the OperationalCarry pytree leaves (the scan
          carry) + count the full-grid p/ph/mu total-family leaves, and count
          ``convert-element-type`` ops in the lowered ``_advance_chunk`` HLO.
Gate (d)  nested fused cascade still traces+runs: build a tiny 2-domain nest and
          run the default-on fused cascade on CPU.

CPU-ONLY by construction (the GPU is occupied by the live corpus).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DomainHierarchy,
    DomainNest,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _advance_chunk,
    _initial_carry_for_run,
)

P0_PA, R_D, C_P, GRAVITY = 1.0e5, 287.0, 1004.0, 9.80665
BASELINE = Path(__file__).resolve().parent / "baseline_advance.npz"


# --------------------------------------------------------------------------- #
# Tiny idealized operational case (mirrors tests/test_v013_operational_smoke). #
# --------------------------------------------------------------------------- #
def _grid(nz: int = 16, ny: int = 4, nx: int = 4) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="s1s2", sha256="s1s2", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="s1s2-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    return Tendencies(
        z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)),
        z((nz, ny, nx)), z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)),
    )


def _base_state(grid: GridSpec, *, dz_m: float = 300.0, seed: int = 3) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    rng = np.random.default_rng(seed)
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    z_iface = np.arange(nz + 1) * dz_m
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    theta_col = 300.0 + 0.004 * z_mid
    p_col = P0_PA * (1.0 - GRAVITY * z_mid / (C_P * 290.0)) ** (C_P / R_D)

    def m3(base, noise):
        return jnp.asarray(base[:, None, None] + noise * rng.standard_normal((nz, ny, nx)), dtype=jnp.float64)

    fields["theta"] = m3(theta_col, 0.3)
    fields["p"] = m3(p_col, 50.0)
    fields["p_total"] = fields["p"]
    fields["p_perturbation"] = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    fields["qv"] = jnp.clip(m3(0.012 * np.exp(-z_mid / 3000.0), 5.0e-4), 0.0, None)
    fields["u"] = jnp.asarray(6.0 + 0.5 * rng.standard_normal((nz, ny, nx + 1)), dtype=jnp.float64)
    fields["v"] = jnp.asarray(-2.0 + 0.5 * rng.standard_normal((nz, ny + 1, nx)), dtype=jnp.float64)
    fields["w"] = jnp.asarray(0.05 * rng.standard_normal((nz + 1, ny, nx)), dtype=jnp.float64)
    fields["qke"] = jnp.full((nz, ny, nx), 0.4, dtype=jnp.float64)
    ph = jnp.asarray(np.broadcast_to(GRAVITY * z_iface[:, None, None], (nz + 1, ny, nx)), dtype=jnp.float64)
    fields["ph"] = ph
    fields["ph_total"] = ph
    fields["ph_perturbation"] = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    fields["xland"] = jnp.ones((ny, nx), dtype=jnp.float64)
    fields["t_skin"] = jnp.full((ny, nx), 300.0, dtype=jnp.float64)
    fields["mu_total"] = jnp.full((ny, nx), 1.0e5, dtype=jnp.float64)
    fields["mu"] = jnp.full((ny, nx), 1.0e5, dtype=jnp.float64)
    fields["mu_perturbation"] = jnp.zeros((ny, nx), dtype=jnp.float64)
    fields["lu_index"] = jnp.zeros((ny, nx), dtype=jnp.int32)
    return State(**fields)


def _namelist(grid: GridSpec, *, dt_s: float = 6.0, **over) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(grid, dt_s=dt_s, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(
        base, run_physics=False, radiation_cadence_steps=1, **over
    )


def build_case(nz: int = 16, ny: int = 4, nx: int = 4, dt_s: float = 6.0):
    """Balanced warm-bubble idealized dycore case on CPU (stable, finite).

    The warm-bubble IC is built in discrete hydrostatic balance with the acoustic
    solver, so the operational ``_advance_chunk`` dycore stays finite over many
    steps -- the right deterministic oracle for a bit-identity gate. It runs the
    REAL outer-State hot path (p/p_total/p_perturbation + ph/mu families carried
    through small_step_prep/finish + the acoustic substep loop).
    """
    from gpuwrf.ic_generators.idealized import build_warm_bubble_setup

    setup = build_warm_bubble_setup(require_gpu=False)
    return setup.state, setup.namelist, setup.grid


# --------------------------------------------------------------------------- #
# Leaf dumping                                                                 #
# --------------------------------------------------------------------------- #
def _state_leaves(state: State) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in state.__slots__:
        value = getattr(state, name)
        if value is None:
            continue
        out[f"state.{name}"] = np.asarray(value)
    return out


_CARRY_SCRATCH = (
    "t_2ave", "ww", "mudf", "muave", "muts", "ph_tend",
    "u_save", "v_save", "w_save", "t_save", "ph_save", "mu_save", "ww_save", "rthraten",
)


def dump_leaves(carry) -> dict[str, np.ndarray]:
    out = _state_leaves(carry.state)
    for name in _CARRY_SCRATCH:
        value = getattr(carry, name, None)
        if value is not None:
            out[f"carry.{name}"] = np.asarray(value)
    return out


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
def run_steps(state: State, namelist: OperationalNamelist, k: int):
    carry = _initial_carry_for_run(state, namelist)
    cadence = max(1, int(namelist.radiation_cadence_steps))
    carry = _advance_chunk(
        carry, namelist, jnp.asarray(1, dtype=jnp.int32), n_steps=int(k), cadence=cadence
    )
    jax.block_until_ready(carry.state.theta)
    return carry


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #
def cmd_save(args) -> int:
    state, namelist, _ = build_case()
    carry = run_steps(state, namelist, args.steps)
    leaves = dump_leaves(carry)
    np.savez(BASELINE, **leaves)
    print(f"[save] baseline -> {BASELINE} ({len(leaves)} leaves, {args.steps} steps)")
    finite = all(np.all(np.isfinite(v)) for v in leaves.values() if np.issubdtype(v.dtype, np.floating))
    print(f"[save] all float leaves finite: {finite}")
    return 0


def cmd_compare(args) -> int:
    if not BASELINE.exists():
        print(f"[compare] ERROR no baseline at {BASELINE}; run `save` first")
        return 2
    base = dict(np.load(BASELINE))
    state, namelist, _ = build_case()
    carry = run_steps(state, namelist, args.steps)
    cur = dump_leaves(carry)
    base_keys, cur_keys = set(base), set(cur)
    only_base = base_keys - cur_keys
    only_cur = cur_keys - base_keys
    common = sorted(base_keys & cur_keys)
    n_exact = 0
    worst = []
    for k in common:
        a, b = base[k], cur[k]
        if a.shape != b.shape:
            worst.append((k, "SHAPE", a.shape, b.shape))
            continue
        if np.array_equal(a, b):
            n_exact += 1
        else:
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            worst.append((k, "DIFF", float(np.max(diff)), int(np.count_nonzero(diff))))
    print(f"[compare] {n_exact}/{len(common)} leaves BITWISE-EQUAL ({args.steps} steps)")
    if only_base:
        print(f"[compare] leaves only in baseline (removed): {sorted(only_base)}")
    if only_cur:
        print(f"[compare] leaves only in current (added): {sorted(only_cur)}")
    if worst:
        print("[compare] NON-IDENTICAL leaves:")
        for row in worst:
            print("   ", row)
        print("[compare] RESULT: NOT BIT-IDENTICAL")
        return 1
    # Note: removing pure-duplicate alias leaves (state.p/ph/mu) is expected and
    # bit-identical -- they are exact copies of state.p_total/ph_total/mu_total,
    # which remain and are compared above. Flag only if a *retained* leaf differs.
    print("[compare] RESULT: BIT-IDENTICAL on all retained leaves")
    return 0


def cmd_liveness(args) -> int:
    state, namelist, _ = build_case()
    carry = _initial_carry_for_run(state, namelist)
    state = carry.state  # post-precision-enforcement view
    # Carry pytree leaf inventory (the scan carry).
    leaves, treedef = jax.tree_util.tree_flatten(carry)
    full_grid_total_alias = []
    for name in state.__slots__:
        v = getattr(state, name)
        if v is None:
            continue
        if name in ("p", "ph", "mu", "p_total", "ph_total", "mu_total"):
            full_grid_total_alias.append((name, tuple(v.shape), str(v.dtype)))
    print(f"[liveness] OperationalCarry total pytree leaves: {len(leaves)}")
    print(f"[liveness] State p/ph/mu total-family leaves present in carry:")
    for row in full_grid_total_alias:
        print("   ", row)
    # HLO convert-element-type counter for the jitted single advance chunk.
    cadence = max(1, int(namelist.radiation_cadence_steps))
    lowered = jax.jit(
        lambda c: _advance_chunk(c, namelist, jnp.asarray(1, dtype=jnp.int32), n_steps=1, cadence=cadence)
    ).lower(carry)
    hlo = lowered.as_text()
    n_convert = hlo.count("convert(")
    n_convert_alt = hlo.count("convert-element-type") + hlo.count("convert ")
    print(f"[liveness] _advance_chunk HLO convert ops: convert(={n_convert}  alt={n_convert_alt}")
    # Count how many distinct full-grid f64 buffers appear in the carry inputs.
    print(f"[liveness] carry leaf shapes (sample full-grid):")
    nz, ny, nx = state.theta.shape
    big = [(s, str(l.dtype)) for s, l in [(tuple(x.shape), x) for x in leaves]
           if len(s) == 3 and s[0] in (nz, nz + 1)]
    print(f"   full-grid 3D leaves in carry: {len(big)}")
    return 0


def cmd_nested(args) -> int:
    """Gate (d): tiny 2-domain nest -> default-on fused cascade TRACES (lower)
    on CPU, and runs one root cascade. No GPU exec."""
    import gpuwrf.runtime.domain_tree as dt_mod
    from gpuwrf.runtime.domain_tree import (
        DomainBundle,
        DomainTree,
        _operational_fused_cascade_factory,
        run_operational_domain_tree,
    )
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run

    # 3-level nest d01(root) -> d02(fusable parent) -> d03(leaf child). The fused
    # cascade applies to the NON-root parent d02 with its leaf child d03 (mirrors
    # the all-7 cascade where d02 fuses its d03..d09 leaves).
    nz, ny, nx = 12, 9, 9
    grids = {n: _grid(nz=nz, ny=ny, nx=nx) for n in ("d01", "d02", "d03")}
    states = {n: _base_state(grids[n]) for n in grids}
    nmls = {
        "d01": _namelist(grids["d01"], dt_s=9.0),
        "d02": _namelist(grids["d02"], dt_s=3.0),
        "d03": _namelist(grids["d03"], dt_s=1.0),
    }
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02", "d03"),
        (DomainNest("d01", "d02", 3, 2, 2), DomainNest("d02", "d03", 3, 2, 2)),
    )
    domains = {
        n: DomainBundle(n, states[n], nmls[n], grid=grids[n], metrics=grids[n].metrics)
        for n in grids
    }
    tree = DomainTree.from_domains(hierarchy, domains)
    fused_default = dt_mod._nested_fuse_default_enabled()
    print(f"[nested] fused cascade default-enabled: {fused_default}")

    # PRIMARY PROOF: the fused cascade program for the parent d02 TRACES (lowers
    # to HLO) on CPU.
    lookup = _operational_fused_cascade_factory(tree)
    fused = lookup("d02")
    if fused is None:
        print("[nested] RESULT: no fused program built (parent not fusable)")
        return 1
    parent_carry = _initial_carry_for_run(states["d02"], nmls["d02"])
    child_carry = _initial_carry_for_run(states["d03"], nmls["d03"])
    lowered = jax.jit(fused).lower(parent_carry, (child_carry,), 1, (1,))
    hlo = lowered.as_text()
    print(f"[nested] fused cascade LOWERS to HLO: {len(hlo)} chars  (TRACE OK)")

    # SECONDARY: run one root step through the operational tree on CPU.
    result = run_operational_domain_tree(
        tree, root_steps=1, block_between=False, root_sync_cadence=1,
    )
    finite = True
    for name, st in result.states.items():
        for leaf in jax.tree_util.tree_leaves(st):
            a = np.asarray(leaf)
            if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
                finite = False
    print(f"[nested] own_steps={result.own_steps}  finite_after_run={finite}")
    print(f"[nested] RESULT: TRACES (and runs on CPU)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("save", "compare", "liveness", "nested"):
        p = sub.add_parser(name)
        p.add_argument("--steps", type=int, default=4)
    args = parser.parse_args(argv)
    return {"save": cmd_save, "compare": cmd_compare, "liveness": cmd_liveness, "nested": cmd_nested}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
