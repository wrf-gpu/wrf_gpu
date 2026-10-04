#!/usr/bin/env python3
"""CPU-only P3 proof for flat 2-domain root fusion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ["GPUWRF_NESTED_AOT"] = "0"
os.environ.pop("GPUWRF_BITWISE", None)
os.environ.pop("GPUWRF_NESTED_DEFUSE_COMPILE", None)

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest, GridSpec  # noqa: E402
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes  # noqa: E402
from gpuwrf.profiling.budget import (  # noqa: E402
    compiled_text,
    kernel_launches_per_step,
    write_hlo,
)
from gpuwrf.runtime.domain_tree import (  # noqa: E402
    DomainBundle,
    DomainTree,
    _FUSED_PROGRAM_CACHE,
    _operational_force,
    _operational_fused_cascade_factory,
    run_operational_domain_tree,
    with_live_child_boundary_config,
)
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    OperationalNamelist,
    _advance_chunk_fori,
    _initial_carry_for_run,
    build_clock_base,
)


OUT_DIR = Path(__file__).resolve().parent
HLO_DIR = OUT_DIR / "hlo"
PROOF_JSON = OUT_DIR / "P3_CPU_PROOF.json"


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _zero_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = jnp.zeros
    dtype = jnp.float64
    return Tendencies(
        u=z((nz, ny, nx + 1), dtype=dtype),
        v=z((nz, ny + 1, nx), dtype=dtype),
        w=z((nz + 1, ny, nx), dtype=dtype),
        theta=z((nz, ny, nx), dtype=dtype),
        qv=z((nz, ny, nx), dtype=dtype),
        p=z((nz, ny, nx), dtype=dtype),
        ph=z((nz + 1, ny, nx), dtype=dtype),
        mu=z((ny, nx), dtype=dtype),
    )


def _mass_profile(shape: tuple[int, ...], top: float, bottom: float) -> jax.Array:
    profile = jnp.linspace(top, bottom, int(shape[0]), dtype=jnp.float64)
    return profile.reshape((shape[0],) + (1,) * (len(shape) - 1)) * jnp.ones(shape, dtype=jnp.float64)


def _state(grid: GridSpec, *, seed: float) -> State:
    fields: dict[str, Any] = {}
    shapes = _state_field_shapes(grid, mp_physics=0)
    for name, shape in shapes.items():
        if name == "lu_index":
            fields[name] = jnp.ones(shape, dtype=jnp.int32)
        else:
            fields[name] = jnp.zeros(shape, dtype=jnp.float64)

    fields["theta"] = jnp.full(shapes["theta"], 300.0 + seed, dtype=jnp.float64)
    fields["qv"] = jnp.full(shapes["qv"], 1.0e-3, dtype=jnp.float64)
    fields["p_total"] = _mass_profile(shapes["p_total"], 95000.0 + seed, 50000.0 + seed)
    fields["p_perturbation"] = jnp.zeros(shapes["p_perturbation"], dtype=jnp.float64)
    fields["ph_total"] = _mass_profile(shapes["ph_total"], 0.0, 98100.0)
    fields["ph_perturbation"] = jnp.zeros(shapes["ph_perturbation"], dtype=jnp.float64)
    fields["mu_total"] = jnp.full(shapes["mu_total"], 90000.0 + seed, dtype=jnp.float64)
    fields["mu_perturbation"] = jnp.zeros(shapes["mu_perturbation"], dtype=jnp.float64)
    fields["xland"] = jnp.ones(shapes["xland"], dtype=jnp.float64)
    fields["mavail"] = jnp.ones(shapes["mavail"], dtype=jnp.float64)
    fields["rhosfc"] = jnp.full(shapes["rhosfc"], 1.1, dtype=jnp.float64)
    fields["t_skin"] = jnp.full(shapes["t_skin"], 290.0 + seed, dtype=jnp.float64)
    fields["roughness_m"] = jnp.full(shapes["roughness_m"], 0.1, dtype=jnp.float64)
    fields["Ni"] = jnp.full(shapes["Ni"], 1.0e6, dtype=jnp.float64)
    fields["Nr"] = jnp.full(shapes["Nr"], 1.0e6, dtype=jnp.float64)
    return State(**fields)


def _namelists(grid: GridSpec) -> tuple[OperationalNamelist, OperationalNamelist]:
    common = {
        "acoustic_substeps": 1,
        "radiation_cadence_steps": 999999,
        "force_fp64": True,
        "use_vertical_solver": False,
    }
    parent = OperationalNamelist.from_grid(
        grid,
        tendencies=_zero_tendencies(grid),
        dt_s=1.0,
        **common,
    )
    parent = replace(parent, run_physics=False, mp_physics=0, run_boundary=False)

    child = OperationalNamelist.from_grid(
        grid,
        tendencies=_zero_tendencies(grid),
        dt_s=1.0 / 3.0,
        **common,
    )
    child = replace(child, run_physics=False, mp_physics=0, run_boundary=True)
    child = with_live_child_boundary_config(
        child,
        parent_dt_s=1.0,
        nested_w_relax=False,
    )
    return parent, child


def _tree() -> DomainTree:
    grid = GridSpec.canary_3km_template()
    parent_namelist, child_namelist = _namelists(grid)
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", 3, 1, 1),),
        max_dom=2,
    )
    domains = {
        "d01": DomainBundle("d01", _state(grid, seed=0.0), parent_namelist, grid=grid),
        "d02": DomainBundle("d02", _state(grid, seed=10.0), child_namelist, grid=grid),
    }
    return DomainTree.from_domains(hierarchy, domains, feedback_enabled=False)


def _initial_carries(tree: DomainTree) -> dict[str, Any]:
    return {
        name: _initial_carry_for_run(bundle.state, bundle.namelist)
        for name, bundle in tree.domains.items()
    }


def _block(result: Any) -> Any:
    jax.block_until_ready(jax.tree_util.tree_leaves(result))
    return result


def _run(tree: DomainTree, carries: dict[str, Any], *, fused: bool):
    os.environ["GPUWRF_NESTED_FUSE"] = "1" if fused else "0"
    _FUSED_PROGRAM_CACHE.clear()
    result = run_operational_domain_tree(
        tree,
        root_steps=1,
        carries=carries,
        block_between=False,
    )
    return _block(result)


def _domain_digest(carry: Any) -> dict[str, Any]:
    leaves = jax.tree_util.tree_leaves(carry)
    digest = hashlib.sha256()
    for leaf in leaves:
        arr = np.asarray(leaf)
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes(order="C"))
    return {
        "leaf_count": len(leaves),
        "sha256": digest.hexdigest(),
    }


def _assert_byte_identical(eager: Any, fused: Any) -> dict[str, Any]:
    if eager.events != fused.events:
        raise AssertionError(f"event mismatch: {eager.events!r} != {fused.events!r}")
    if eager.own_steps != fused.own_steps:
        raise AssertionError(f"own_steps mismatch: {eager.own_steps!r} != {fused.own_steps!r}")

    domains: dict[str, Any] = {}
    for name in sorted(eager.carries):
        eager_leaves = jax.tree_util.tree_leaves(eager.carries[name])
        fused_leaves = jax.tree_util.tree_leaves(fused.carries[name])
        if len(eager_leaves) != len(fused_leaves):
            raise AssertionError(f"{name}: leaf count mismatch")
        for idx, (lhs, rhs) in enumerate(zip(eager_leaves, fused_leaves, strict=True)):
            lhs_arr = np.asarray(lhs)
            rhs_arr = np.asarray(rhs)
            if lhs_arr.shape != rhs_arr.shape or lhs_arr.dtype != rhs_arr.dtype:
                raise AssertionError(
                    f"{name}[{idx}]: shape/dtype mismatch "
                    f"{lhs_arr.shape}/{lhs_arr.dtype} != {rhs_arr.shape}/{rhs_arr.dtype}"
                )
            if lhs_arr.tobytes(order="C") != rhs_arr.tobytes(order="C"):
                raise AssertionError(f"{name}[{idx}]: byte mismatch")
        domains[name] = _domain_digest(fused.carries[name])
    return {
        "events_identical": True,
        "own_steps_identical": True,
        "events": eager.events,
        "own_steps": eager.own_steps,
        "domains": domains,
    }


def _compile_hlo(name: str, lowered: Any) -> dict[str, Any]:
    compiled = lowered.compile()
    text = compiled_text(compiled).rstrip() + "\n"
    path = HLO_DIR / f"{name}.txt"
    full_path = HLO_DIR / "full" / f"{name}.txt"
    write_hlo(path, text, full_path=full_path)
    return {
        "path": _rel(path),
        "full_path": _rel(full_path) if full_path.exists() else None,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "bytes": len(text.encode("utf-8")),
        "entry_count": len(re.findall(r"\bENTRY\b", text)),
        "launch_marker_count": int(kernel_launches_per_step(text)),
    }


def _hlo_proof(tree: DomainTree, initial: dict[str, Any], eager: Any) -> dict[str, Any]:
    edge = tree.children("d01")[0]
    parent_namelist = tree.domains["d01"].namelist
    child_namelist = tree.domains["d02"].namelist

    parent_lowered = _advance_chunk_fori.lower(
        initial["d01"],
        parent_namelist,
        jnp.asarray(1, dtype=jnp.int32),
        build_clock_base(parent_namelist),
        n_steps=1,
        cadence=int(parent_namelist.radiation_cadence_steps),
    )

    @jax.jit
    def boundary_force_program(parent_carry, child_carry):
        return _operational_force(edge, parent_carry, child_carry)

    boundary_lowered = boundary_force_program.lower(eager.carries["d01"], initial["d02"])
    forced_child = _operational_force(edge, eager.carries["d01"], initial["d02"])
    forced_child = _block(forced_child)

    child_lowered = _advance_chunk_fori.lower(
        forced_child,
        child_namelist,
        jnp.asarray(1, dtype=jnp.int32),
        build_clock_base(child_namelist),
        n_steps=int(edge.parent_grid_ratio),
        cadence=int(child_namelist.radiation_cadence_steps),
    )

    os.environ["GPUWRF_NESTED_FUSE"] = "1"
    _FUSED_PROGRAM_CACHE.clear()
    program = _operational_fused_cascade_factory(tree)("d01")
    if program is None:
        raise AssertionError("expected fused root cascade program for d01")

    @jax.jit
    def fused_root_program(parent_carry, child_carry):
        return program(parent_carry, (child_carry,), 1, (1,))

    fused_lowered = fused_root_program.lower(initial["d01"], initial["d02"])

    modules = {
        "unfused_parent_advance": _compile_hlo("p3_unfused_parent_advance", parent_lowered),
        "unfused_boundary_force": _compile_hlo("p3_unfused_boundary_force", boundary_lowered),
        "unfused_child_advance": _compile_hlo("p3_unfused_child_advance", child_lowered),
        "fused_root_cascade": _compile_hlo("p3_fused_root_cascade", fused_lowered),
    }
    unfused_names = (
        "unfused_parent_advance",
        "unfused_boundary_force",
        "unfused_child_advance",
    )
    return {
        "modules": modules,
        "top_level_programs_unfused": len(unfused_names),
        "top_level_programs_fused": 1,
        "top_level_program_delta": 1 - len(unfused_names),
        "launch_marker_count_unfused_sum": sum(modules[name]["launch_marker_count"] for name in unfused_names),
        "launch_marker_count_fused": modules["fused_root_cascade"]["launch_marker_count"],
        "entry_count_unfused_sum": sum(modules[name]["entry_count"] for name in unfused_names),
        "entry_count_fused": modules["fused_root_cascade"]["entry_count"],
    }


def main() -> None:
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"expected CPU backend, got {jax.default_backend()}")
    platforms = sorted({device.platform for device in jax.devices()})
    if platforms != ["cpu"]:
        raise RuntimeError(f"expected only CPU JAX devices, got {platforms!r}")

    tree = _tree()
    initial = _initial_carries(tree)
    eager = _run(tree, initial, fused=False)
    fused = _run(tree, initial, fused=True)
    identity = _assert_byte_identical(eager, fused)
    hlo = _hlo_proof(tree, initial, eager)

    payload = {
        "proof": "P3 flat 2-domain root fusion CPU proof",
        "cpu_only": True,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "env": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "GPUWRF_NESTED_AOT": os.environ.get("GPUWRF_NESTED_AOT"),
            "GPUWRF_NESTED_FUSE_FINAL": os.environ.get("GPUWRF_NESTED_FUSE"),
        },
        "geometry": {
            "domains": list(tree.hierarchy.order),
            "edge": "d01->d02",
            "parent_grid_ratio": int(tree.children("d01")[0].parent_grid_ratio),
            "root_steps": 1,
        },
        "bit_identity": identity,
        "hlo_program_proof": hlo,
        "gpu_perf": {
            "status": "deferred",
            "expected_win": "one root-cascade executable should reduce GPU top-level dispatch/host orchestration versus parent advance + boundary force + child advance",
            "required_measurement": "warm GPU launch-count and wall-time comparison in the validation lane",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROOF_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
