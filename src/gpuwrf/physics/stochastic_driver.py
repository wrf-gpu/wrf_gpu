"""Device-resident original WRF spectral-pattern driver and restart carry."""
from datetime import datetime
from typing import Any, NamedTuple

from functools import partial

import jax
import jax.numpy as jnp

from gpuwrf.io.stochastic_config import VARIABLES
from .stochastic_rng import gaussian_innovations, seed_array
from .stochastic_setup import initialize_spectrum
from .stochastic import update_coefficients, spectral_field


class PatternCarry(NamedTuple):
    sine: Any
    cosine: Any
    amplitude: Any
    alpha: Any
    cutoff: Any
    field: Any


class StochasticCarry(NamedTuple):
    patterns: tuple
    seeds: tuple
    vertical: Any
    clip_counts: Any
    schema: Any


def grid_spacing(grid):
    """Retained GridSpec stores metric spacing on its Projection."""
    if hasattr(grid,"projection"):
        return grid.projection.dx_m,grid.projection.dy_m
    return grid.dx,grid.dy


@jax.jit
def _spin_random(pattern, seed):
    def body(_index, carry):
        p, seed = carry
        ns,nc,seed = gaussian_innovations(seed,p.sine.shape)
        sine,cosine = update_coefficients(p.sine,p.cosine,p.amplitude,p.alpha,ns,nc)
        return p._replace(sine=sine,cosine=cosine),seed
    return jax.lax.fori_loop(0,10,body,(pattern,seed))


def initial_stochastic(config, grid, dt, start, *, tiles_ready=False):
    """Original operational setup; optional already-tiled standalone caller.

    start_em calls INITIALIZE_STOCH at line330, BEFORE set_tiles at992. Its
    nominal ten R spin calls therefore execute zero UPDATE_STOCH tile bodies.
    The complete original CPU-WRF seed proof confirms the first step consumes
    exactly one R innovation plane. A standalone already-tiled caller differs,
    and so does every later start_domain(d01) call that med_nest_initial makes
    for a d01 child (``config.nest_init_rounds``; share/mediation_integrate.F:845).
    """
    nx, ny, nz = grid.nx+1, grid.ny+1, grid.nz+1
    dx,dy = grid_spacing(grid)
    patterns, vertical = [], None
    for index,control in enumerate(config.patterns):
        if control is None:
            patterns.append(None)
            continue
        shape = (nz,ny,nx) if control.vertical else (1,ny,nx)
        field = jnp.zeros(shape,jnp.float32)
        if control.variable == "W":
            field = (field,jnp.zeros(shape,jnp.float32))
        if not config.root or not config.produce[index]:
            # WRF initializes/evolves only d01. Children retain parent-forced
            # fields, not FFT plans or independently advancing random streams.
            empty = jnp.zeros((0,0),jnp.float32)
            patterns.append(PatternCarry(empty,empty,empty,jnp.float32(0),
                jnp.float32(control.stddev)*jnp.float32(control.cutoff),field))
            continue
        plan = initialize_spectrum(nx,ny,nz,dx=dx,dy=dy,dt=int(dt),**control.setup_kwargs())
        patterns.append(PatternCarry(jnp.zeros((ny,nx),jnp.float32),jnp.zeros((ny,nx),jnp.float32),
            jnp.asarray(plan["amplitude"]),jnp.asarray(plan["alpha"]),
            jnp.float32(control.stddev)*jnp.float32(control.cutoff),field))
        if control.vertical and vertical is None:
            vertical = tuple(jnp.asarray(plan[key]) for key in
                ("vertical_cosine","vertical_sine","vertical_amplitude"))
    date = None
    if config.root and any(config.produce):
        date = datetime.fromisoformat(str(start).replace("Z","+00:00")) if not isinstance(start,datetime) else start
    enabled_seeds = (patterns[0] is not None, *(p is not None for p in patterns[2:]))
    seed_produced = (config.produce[0],*config.produce[2:])
    seeds = tuple((jnp.asarray(seed_array(seed,config.nens,date)) if config.root and produced
                   else jnp.zeros(8,jnp.int32)) if enabled else None
                  for seed,enabled,produced in zip(config.seeds,enabled_seeds,seed_produced))
    result = StochasticCarry(tuple(patterns),seeds,vertical,jnp.zeros(7,jnp.int32),jnp.int32(1))
    rounds = int(tiles_ready) + int(getattr(config,"nest_init_rounds",0))
    if rounds and config.root and patterns[3] is not None and config.produce[3]:
        # INITIALIZE_STOCH spins RAND_PERT_UPDATE('R') TEN times. Those consume
        # AR updates but have no T/U/V spectral projection; the first RK call
        # projects it correctly as T. Do not substitute a first-step update.
        # Nested runs: med_nest_initial re-calls start_domain(d01) per d01
        # child after set_tiles. Each call re-seeds the stream (rand_seed) but
        # SETUP_RAND_PERTURB keeps SPFORCS/SPFORCC, so every round spins ten
        # more AR updates from the evolved coefficients with the SAME draws.
        initial_seed = seeds[2]
        for _round in range(rounds):
            patterns[3],next_seed = _spin_random(patterns[3],initial_seed)
        seeds = list(seeds)
        seeds[2] = next_seed
        result = result._replace(patterns=tuple(patterns),seeds=tuple(seeds))
    return result


def advance_stochastic(carry, config, *, dx, dy):
    """Root update: T then shared-streamfunction U/V, P/R/S/Q/O thereafter."""
    if not config.root:
        # The child repeats perturb_physics_tend's own SPPT cap after force-down.
        p = carry.patterns[2]
        if p is None:
            return carry
        patterns = list(carry.patterns)
        value = p.field
        selected = value if value.shape[0] == 1 else value[:-1]
        clips = jnp.sum(jnp.abs(selected)>p.cutoff,dtype=jnp.int32)
        capped = jnp.clip(selected,-p.cutoff,p.cutoff)
        patterns[2] = p._replace(field=capped if value.shape[0] == 1 else value.at[:-1].set(capped))
        return carry._replace(patterns=tuple(patterns),clip_counts=carry.clip_counts.at[2].add(clips))
    patterns, seeds = list(carry.patterns), list(carry.seeds)
    counts = carry.clip_counts
    for index in (1,0,2,3,4,5,6):
        p = patterns[index]
        if p is None or not config.produce[index]:
            continue
        seed_index = 0 if index < 2 else index-1
        ns,nc,seeds[seed_index] = gaussian_innovations(seeds[seed_index],p.sine.shape)
        sine,cosine = update_coefficients(p.sine,p.cosine,p.amplitude,p.alpha,ns,nc)
        kwargs = {} if not config.patterns[index].vertical else dict(zip(
            ("vertical_cosine","vertical_sine","vertical_amplitude"),carry.vertical))
        components = ("U","V") if index == 0 else ("T",)
        fields = []
        clips = jnp.int32(0)
        for component in components:
            field = spectral_field(sine,cosine,component=component,dx=dx,dy=dy,**kwargs)
            clips += jnp.sum(jnp.abs(field)>p.cutoff,dtype=jnp.int32)
            fields.append(jnp.clip(field,-p.cutoff,p.cutoff))
        patterns[index] = p._replace(sine=sine,cosine=cosine,field=tuple(fields) if index==0 else fields[0])
        counts = counts.at[index].add(clips)
    return carry._replace(patterns=tuple(patterns),seeds=tuple(seeds),clip_counts=counts)


@partial(jax.jit, static_argnums=(1, 2, 3))
def lead_zero_advance(carry, config, dx, dy):
    """One jitted (E41-scannable) advance for the child lead-zero history phase only.

    ``config`` is static by its layout hash/eq, which is all advance_stochastic reads (vertical/produce/root).
    """
    return advance_stochastic(carry, config, dx=dx, dy=dy)


def field(carry, variable, *, levels=None, component=0, mass_only=True):
    p = carry.patterns[VARIABLES.index(variable)]
    if p is None:
        return None
    value = p.field[component] if variable == "W" else p.field
    if mass_only:
        value = value[:,:-1,:-1]
    if levels is not None and value.shape[0] == 1:
        value = jnp.broadcast_to(value,(levels,*value.shape[1:]))
    elif levels is not None:
        value = value[:levels]
    return value
