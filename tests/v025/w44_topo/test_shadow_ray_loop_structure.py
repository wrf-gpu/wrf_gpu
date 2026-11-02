"""v46-topo: the topographic ray search must stay a loop, never a Python unroll.

Pre-v0.4.4 production called physics_couplers._wrf_topographic_shadow_mask,
which unrolls 4 x int(shadlen/dx+1) ray steps in Python. On a 24x24 WN3 crop
XLA fused that into one kLoop with STACK 560 B (warmstart WS46, E41 FAIL).
Production now runs topographic_shadow (fori_loop) on CPU and the native
Pallas kernel (in-kernel while_loop) on CUDA. These tests pin both loop forms
and keep the legacy unrolled comparator out of every production module.
"""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gpuwrf
from gpuwrf.kernels.topographic_shadow import topographic_shadow_native
from gpuwrf.physics.topographic_shading import topographic_shadow

_TRIG = {"atan", "atan2"}


def _count(jaxpr, names):
    total, loops = 0, 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in names:
            total += 1
        if eqn.primitive.name == "while":
            loops += 1
        if eqn.primitive.name == "scan":  # static-bound fori_loop; a scan unroll is a Python unroll too
            assert eqn.params["unroll"] in (1, False), eqn.params["unroll"]
            loops += 1
        for value in eqn.params.values():
            for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    t, l = _count(inner, names)
                    total, loops = total + t, loops + l
    return total, loops


def _terrain(n=8):
    rng = np.random.default_rng(46)
    return jnp.asarray(rng.uniform(0.0, 2000.0, (n, n)), jnp.float32)


def _reference_jaxpr(shadow_length_m):
    terrain = _terrain()
    z = jnp.zeros_like(terrain)
    fn = lambda t: topographic_shadow(t, latitude_deg=z + 28.3, longitude_deg=z - 16.5, sina=z,
        cosa=z + 1, declination_rad=0.3, xtime_minutes=600., gmt_hours=0., radt_minutes=0.,
        dx_m=1000., dy_m=1000., shadow_length_m=shadow_length_m)
    return jax.make_jaxpr(fn)(terrain).jaxpr


def _native_jaxpr(shadow_length_m):
    terrain = _terrain()
    z = jnp.zeros_like(terrain)
    fn = lambda t: topographic_shadow_native(t, z, z + 0.3, z + 1.0, dx_m=1000., dy_m=1000.,
        shadow_length_m=shadow_length_m, nested=False, interpret=True)
    return jax.make_jaxpr(fn)(terrain).jaxpr


def test_reference_ray_scan_is_one_loop_independent_of_ray_length():
    short, long_ = _reference_jaxpr(3000.), _reference_jaxpr(25000.)
    assert _count(short, _TRIG) == _count(long_, _TRIG)
    trig, loops = _count(long_, _TRIG)
    assert loops >= 1 and 1 <= trig <= 2, (trig, loops)


def test_native_ray_scan_is_one_in_kernel_loop_independent_of_ray_length():
    short, long_ = _native_jaxpr(3000.), _native_jaxpr(25000.)
    assert _count(short, _TRIG) == _count(long_, _TRIG)
    calls = [e for e in long_.eqns if e.primitive.name == "pallas_call"]
    assert len(calls) == 1
    trig, loops = _count(calls[0].params["jaxpr"], _TRIG)
    assert loops >= 1 and 1 <= trig <= 2, (trig, loops)


def test_legacy_unrolled_mask_is_not_called_from_production_sources():
    root = Path(gpuwrf.__file__).parent
    callers = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name == "_wrf_topographic_shadow_mask":
                callers.append(f"{path.relative_to(root)}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and any(a.name == "_wrf_topographic_shadow_mask" for a in node.names):
                callers.append(f"{path.relative_to(root)}:{node.lineno}")
    assert callers == [], callers
