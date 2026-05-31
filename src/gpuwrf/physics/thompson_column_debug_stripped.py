"""Hand-stripped Thompson column sibling with all debug calls removed."""

from __future__ import annotations

from functools import partial

import jax
from jax import config

import jax.numpy as jnp

from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    _cast_state,
    _clip_species,
    _finish,
    _ice_sources_with_process_flags,
    _instant_melt_freeze,
    _rain_evaporation,
    _restore_state,
    _saturation_adjustment_with_condensation,
    _warm_rain_collection,
    _work_dtype,
)


config.update("jax_enable_x64", True)


def _step_thompson_column_stripped_impl(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Duplicates the source/sink production sequencing without debug hooks.

    Mirrors :func:`_step_thompson_column_impl` (no sedimentation), which is the
    HLO-identity sibling used by the M5 debuggability test.  It carries the same
    work-precision cast/restore wrapper as the production body so the HLO match
    holds in both fp64 (default) and fp32 work modes.
    """

    work = _work_dtype()
    storage = {name: jnp.asarray(getattr(state, name)).dtype for name in ThompsonColumnState.__slots__}
    state = _cast_state(state, work)
    state = _clip_species(state)
    state = _warm_rain_collection(state, dt)
    state, graupel_melt = _ice_sources_with_process_flags(state, dt)
    state, cloud_condensed = _saturation_adjustment_with_condensation(state, dt)
    state = _rain_evaporation(state, dt, skip_evaporation=cloud_condensed, graupel_melt=graupel_melt)
    state = _instant_melt_freeze(state, dt)
    state = _finish(state)
    return _restore_state(state, storage)


@partial(jax.jit, static_argnames=("dt",))
def step_thompson_column_debug_stripped(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Runs the source-level stripped Thompson step for HLO identity checks."""

    return _step_thompson_column_stripped_impl(state, dt)
