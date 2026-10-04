"""B37: the root RK step transports represented Ni/Nr (CPU, abstract trace of the real own-step).

Numerical WRF evidence (pristine advect_scalar_pd/advect_scalar -> rk_update_scalar on real
PROD d01 operands) lives in tests/v025/b_core/root_numbers_evidence.json; this test proves the
dispatch on the actual production own-step and rejects the pre-fix root branch.
"""
import __future__
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from prod_inputs import prod_domains

from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree


def _root_case():
    hierarchy, bundles, _, _, _, carries = prod_domains()
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    nml = tree.domains["d01"].namelist
    assert not op._nested_frozen_wrf_boundary_active(nml)
    return nml, carries["d01"]


def _trace_own_step(nml, carry):
    calls = {"numbers": 0, "updates": []}
    transport = op._scalar_transport_coupled_tendencies
    update = op._apply_moisture_large_step

    def spy_numbers(*args, **kwargs):
        # Ni/Nr reach the root either alone or merged with the moist species
        # (one call when moist_adv_opt == scalar_adv_opt).
        if {"Ni", "Nr"} <= set(kwargs.get("species", ())):
            calls["numbers"] += 1
        return transport(*args, **kwargs)

    def spy_update(*args, **kwargs):
        calls["updates"].append(tuple(kwargs.get("species", ())))
        return update(*args, **kwargs)

    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    op._scalar_transport_coupled_tendencies = spy_numbers
    op._apply_moisture_large_step = spy_update
    try:
        jax.clear_caches()
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1,
            cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        op._scalar_transport_coupled_tendencies = transport
        op._apply_moisture_large_step = update
        jax.clear_caches()
    return calls


def _pre_fix_root_branch():
    """Re-define _rk_scan_step with the released root branch (no number transport)."""
    source = Path(op.__file__)
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rk_scan_step")
    hits = 0
    for node in ast.walk(function):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "number_scalars_advected"
                and "haloed.Ni" in ast.unparse(node.value)):
            node.value = ast.Constant(False)
            hits += 1
    assert hits == 1, "root number-transport dispatch not found"
    return compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec",
                   flags=__future__.annotations.compiler_flag, dont_inherit=True)


def test_root_own_step_transports_ni_nr():
    nml, carry = _root_case()
    assert jax.devices()[0].platform == "cpu"
    assert carry.state.Ni is not None and carry.state.Nr is not None
    calls = _trace_own_step(nml, carry)
    assert calls["numbers"] == int(nml.rk_order)
    assert len(calls["updates"]) == int(nml.rk_order)
    assert all(("Ni" in s and "Nr" in s and "qv" in s) for s in calls["updates"])

    original = op._rk_scan_step
    try:
        exec(_pre_fix_root_branch(), op.__dict__)
        mutant = _trace_own_step(nml, carry)
    finally:
        op._rk_scan_step = original
        jax.clear_caches()
    assert mutant["numbers"] == 0
    assert all("Ni" not in s and "Nr" not in s for s in mutant["updates"])
