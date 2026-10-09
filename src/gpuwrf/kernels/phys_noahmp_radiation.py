"""Noah-MP's existing REAL RADIATION phase in one pointwise native call.

The literal reference equations run on tile values. Band and two-stream
intermediates have tile shape; only the existing eight radiation fields and
six extras are requested as outputs. Actual registers/spills require native
measurement. Inputs retain their original shapes without packing.
"""
from functools import partial
from math import prod
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

_LAND = ("tg", "tv", "fwet", "smois", "sneqv", "sneqvo", "snowh", "tauss", "albold")
_FORCE = ("cosz", "soldn", "prcpsnow")
_PHEN = ("elai", "esai", "fveg")
_PARAMS = ("rhol", "rhos", "taul", "taus", "xl", "albsat", "albdry", "omegas",
           "betads", "betais", "swemx", "mfsno", "scffac", "tau0", "grain_growth",
           "extra_growth", "dirt_soot")
_BANDS = frozenset(("rhol", "rhos", "taul", "taus", "albsat", "albdry", "omegas"))
_EXTRAS = ("fsun", "laisun", "laisha", "tauss", "albold", "vai")
_NINPUTS = len(_LAND) + len(_FORCE) + len(_PHEN) + len(_PARAMS) + 1


def _kernel(*refs, shape, block):
    from gpuwrf.physics.noahmp.energy_radiation import TwoStreamParams, _radiation_twostream_impl
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = flat < prod(shape)
    # Keep padded lanes outside the leading dimension: wrapped invalid indices
    # alias valid outputs in the interpreter's masked scatter.
    coords = tuple(flat // prod(shape[1:]) if axis == 0
                   else (flat // prod(shape[axis + 1:])) % size
                   for axis, size in enumerate(shape))

    def point(ref, prefix=()):
        dimensions = ref.shape[len(prefix):]
        if not dimensions:
            return jnp.broadcast_to(ref[prefix], (block,))
        indices = tuple(jnp.zeros_like(index) if size == 1 else index
                        for size, index in zip(dimensions, coords[-len(dimensions):]))
        return pt.load(ref.at[(*prefix, *indices)], mask=valid, other=0.)

    offset = 0
    def group(names, kind):
        nonlocal offset
        values = {}
        for name in names:
            ref = refs[offset]
            offset += 1
            if kind == "land" and name == "smois":
                values[name] = (point(ref, (0,)),)
            elif kind == "params" and name in _BANDS:
                values[name] = (point(ref, (0,)), point(ref, (1,)))
            else:
                values[name] = point(ref)
        return values

    land = SimpleNamespace(**group(_LAND, "land"))
    forcing = SimpleNamespace(**group(_FORCE, "forcing"))
    phen = SimpleNamespace(**group(_PHEN, "phen"))
    params = TwoStreamParams(**group(_PARAMS, "params"))
    # A real runtime scalar operand, not a captured Python float.
    dt = refs[offset][()]
    rad, extras = _radiation_twostream_impl(land, forcing, None, phen, params, dt, _tile_bands=True)
    outputs = (*rad, *(extras[name] for name in _EXTRAS))
    for ref, value in zip(refs[_NINPUTS:], outputs):
        pt.store(ref.at[coords], value, mask=valid)


def radiation_native(land_state, forcing, static, phen, params, dt, *, interpret=False, block=128):
    """Same public result; per-point equations with no new full-grid temporaries."""
    from gpuwrf.physics.noahmp.energy_radiation import (
        _OMEGAS_DEF, _BETADS_DEF, _BETAIS_DEF, _SWEMX_DEF, _MFSNO_DEF, _SCFFAC_DEF,
        _TAU0_DEF, _GRAIN_GROWTH_DEF, _EXTRA_GROWTH_DEF, _DIRT_SOOT_DEF,
    )
    from gpuwrf.physics.noahmp.types import NoahMPRadInputs
    defaults = dict(omegas=_OMEGAS_DEF, betads=_BETADS_DEF, betais=_BETAIS_DEF,
                    swemx=_SWEMX_DEF, mfsno=_MFSNO_DEF, scffac=_SCFFAC_DEF,
                    tau0=_TAU0_DEF, grain_growth=_GRAIN_GROWTH_DEF,
                    extra_growth=_EXTRA_GROWTH_DEF, dirt_soot=_DIRT_SOOT_DEF)
    shape = tuple(forcing.cosz.shape)
    values = []
    for obj, names in [(land_state, _LAND), (forcing, _FORCE), (phen, _PHEN), (params, _PARAMS)]:
        for name in names:
            value = getattr(obj, name)
            if value is None:
                value = defaults[name]
            values.append(jnp.asarray(value, jnp.float32))
    values.append(jnp.reshape(jnp.asarray(dt, jnp.float32), ()))
    result = pl.pallas_call(
        partial(_kernel, shape=shape, block=block),
        out_shape=tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for _ in range(14)),
        grid=((prod(shape) + block - 1) // block,), name="b_noahmp_radiation_real",
        interpret=interpret, compiler_params=pt.CompilerParams(num_warps=4),
    )(*values)
    return NoahMPRadInputs(*result[:8]), dict(zip(_EXTRAS, result[8:]))
