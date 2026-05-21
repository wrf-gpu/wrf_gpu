"""Hand-stripped Thompson column sibling with all debug calls removed."""

from __future__ import annotations

from functools import partial

import jax
from jax import config

from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    _clip_species,
    _finish,
    _ice_sources_with_process_flags,
    _instant_melt_freeze,
    _rain_evaporation,
    _saturation_adjustment_with_condensation,
    _warm_rain_collection,
)


config.update("jax_enable_x64", True)


def _step_thompson_column_stripped_impl(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Duplicates production sequencing while physically omitting debug hooks."""

    state = _clip_species(state)
    state = _warm_rain_collection(state, dt)
    state, graupel_melt = _ice_sources_with_process_flags(state, dt)
    state, cloud_condensed = _saturation_adjustment_with_condensation(state, dt)
    state = _rain_evaporation(state, dt, skip_evaporation=cloud_condensed, graupel_melt=graupel_melt)
    state = _instant_melt_freeze(state, dt)
    return _finish(state)


@partial(jax.jit, static_argnames=("dt",))
def step_thompson_column_debug_stripped(state: ThompsonColumnState, dt: float) -> ThompsonColumnState:
    """Runs the source-level stripped Thompson step for HLO identity checks."""

    return _step_thompson_column_stripped_impl(state, dt)
