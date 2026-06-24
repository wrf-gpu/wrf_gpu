import dataclasses
import datetime

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import (
    DEFAULT_ACOUSTIC_PRECISION_MODE,
    AcousticPrecisionMode,
    acoustic_precision_mode_label,
)
from gpuwrf.contracts.state import Tendencies
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_mode import _DateClockAux, _StaticHolder


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    def z(shape):
        return jnp.zeros(shape, dtype=jnp.float64)

    return Tendencies(
        z((nz, ny, nx + 1)),
        z((nz, ny + 1, nx)),
        z((nz + 1, ny, nx)),
        z((nz, ny, nx)),
        z((nz, ny, nx)),
        z((nz, ny, nx)),
        z((nz + 1, ny, nx)),
        z((ny, nx)),
    )


def test_static_holder_none_hash_is_stable_across_flatten_rebuilds():
    """Disabled static bundles must not fragment JIT cache keys."""

    first = _StaticHolder(None)
    second = _StaticHolder(None)

    assert first == second
    assert hash(first) == hash(second)


def test_static_holder_real_bundle_hashes_by_identity():
    bundle = object()
    same = _StaticHolder(bundle)
    again = _StaticHolder(bundle)
    different = _StaticHolder(object())

    assert same == again
    assert hash(same) == hash(again)
    assert same != different


def test_acoustic_precision_mode_default_label_roundtrips_as_static_aux():
    grid = GridSpec.canary_3km_template()
    nml = OperationalNamelist.from_grid(grid, tendencies=_cpu_tendencies(grid))

    assert nml.acoustic_precision_mode == DEFAULT_ACOUSTIC_PRECISION_MODE
    assert acoustic_precision_mode_label(None) == DEFAULT_ACOUSTIC_PRECISION_MODE

    leaves, treedef = jax.tree_util.tree_flatten(nml)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.acoustic_precision_mode == DEFAULT_ACOUSTIC_PRECISION_MODE


def test_mixed_perturb_fp32_mode_is_static_cache_key_only():
    grid = GridSpec.canary_3km_template()
    nml = OperationalNamelist.from_grid(grid, tendencies=_cpu_tendencies(grid))
    mixed = dataclasses.replace(
        nml,
        acoustic_precision_mode=AcousticPrecisionMode.MIXED_PERTURB_FP32,
    )

    default_leaves, default_treedef = jax.tree_util.tree_flatten(nml)
    mixed_leaves, mixed_treedef = jax.tree_util.tree_flatten(mixed)

    assert mixed.acoustic_precision_mode == "mixed_perturb_fp32"
    assert mixed_treedef != default_treedef
    assert len(mixed_leaves) == len(default_leaves)
    assert all(
        getattr(left, "shape", None) == getattr(right, "shape", None)
        and getattr(left, "dtype", None) == getattr(right, "dtype", None)
        for left, right in zip(default_leaves, mixed_leaves, strict=True)
    )


def test_acoustic_precision_mode_invalid_value_fails_closed():
    grid = GridSpec.canary_3km_template()
    nml = OperationalNamelist.from_grid(grid, tendencies=_cpu_tendencies(grid))

    with pytest.raises(ValueError, match="unsupported acoustic_precision_mode"):
        dataclasses.replace(nml, acoustic_precision_mode="global_fp32")


# --- #114: cross-date NEST warm-cache fix -----------------------------------
#
# The forecast init DATE feeds three scalars (time_utc, noahmp_julian,
# noahmp_yearlen). Before the fix these were discriminating entries of the
# namelist STATIC AUX, so a different date -> a different treedef -> an in-memory
# @jax.jit cache MISS + a full re-trace/re-lower of the fused nest, and a fresh
# persistent-cache key (the #114 per-date recompile). #91 already threads the
# values via the traced clock_base, so they are dead inside the compiled fn; the
# fix moves them into a date-BLIND holder so the treedef is date-invariant while
# the legacy clock_base=None path still reads the same values.


def _date_namelist(grid: GridSpec, dt: datetime.datetime) -> OperationalNamelist:
    """A Noah-MP namelist that differs from the default ONLY in the date axis.

    Sets the exact production date sources: ``time_utc`` AND the Noah-MP greenness
    clock ``noahmp_julian`` (0-based fractional day-of-year) / ``noahmp_yearlen``
    (366 for a leap year, else 365) -- mirroring
    ``nested_pipeline._wrf_julian_yearlen``. #91's probe varied only ``time_utc``,
    which is exactly why it missed #114.
    """

    yearlen = 366.0 if (dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0)) else 365.0
    julian = float(dt.timetuple().tm_yday - 1) + (
        dt.hour * 3600.0 + dt.minute * 60.0 + dt.second
    ) / 86400.0
    nml = OperationalNamelist.from_grid(
        grid, tendencies=_cpu_tendencies(grid), time_utc=dt
    )
    return dataclasses.replace(nml, noahmp_julian=julian, noahmp_yearlen=yearlen)


def test_date_clock_aux_is_date_blind_hash_and_eq():
    """The holder must hash to a constant and compare equal regardless of date,
    so two namelists differing only in date produce the SAME treedef."""

    a = _DateClockAux(datetime.datetime(2026, 2, 14, 0, 0), 44.0, 365.0)
    b = _DateClockAux(datetime.datetime(2026, 5, 12, 6, 0), 131.25, 365.0)
    leap = _DateClockAux(datetime.datetime(2024, 2, 29, 12, 0), 59.5, 366.0)

    assert a == b == leap
    assert hash(a) == hash(b) == hash(leap)
    # Still discriminates against unrelated aux types (no accidental cross-merge).
    assert a != _StaticHolder(None)
    assert a != object()


def test_namelist_treedef_is_date_invariant_including_leap_year():
    """Two namelists differing ONLY in the date (time_utc + noahmp_julian +
    noahmp_yearlen, incl. a leap-year date) must share an identical pytree
    treedef -- this IS the in-memory @jax.jit cache key. #91's HLO-only probe
    could not see this axis."""

    grid = GridSpec.canary_3km_template()
    nml_a = _date_namelist(grid, datetime.datetime(2026, 2, 14, 0, 0))  # non-leap
    nml_b = _date_namelist(grid, datetime.datetime(2026, 5, 12, 6, 0))  # non-leap, later
    nml_leap = _date_namelist(grid, datetime.datetime(2024, 2, 29, 12, 0))  # leap day

    leaves_a, treedef_a = jax.tree_util.tree_flatten(nml_a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(nml_b)
    leaves_leap, treedef_leap = jax.tree_util.tree_flatten(nml_leap)

    # The aux (treedef) is the jit cache key; it must be byte-identical across dates.
    assert treedef_a == treedef_b == treedef_leap
    assert hash(treedef_a) == hash(treedef_b) == hash(treedef_leap)
    # Leaf structure unchanged (the date scalars never were traced leaves anyway).
    assert len(leaves_a) == len(leaves_b) == len(leaves_leap)


def test_date_values_are_preserved_through_flatten_roundtrip():
    """tree_unflatten must restore the exact date scalars from the holder so the
    legacy clock_base=None path still reads correct julian/yearlen/time_utc."""

    grid = GridSpec.canary_3km_template()
    dt = datetime.datetime(2024, 2, 29, 12, 0)  # leap-day, fractional day
    nml = _date_namelist(grid, dt)

    leaves, treedef = jax.tree_util.tree_flatten(nml)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.time_utc == dt
    assert rebuilt.noahmp_julian == nml.noahmp_julian
    assert rebuilt.noahmp_yearlen == nml.noahmp_yearlen == 366.0
    # Exact value, not a default: the leap-day fractional julian must survive.
    assert rebuilt.noahmp_julian == pytest.approx(59.5)


def test_jit_fn_not_retraced_when_only_date_changes():
    """A function jit-traced once over a namelist must NOT re-trace when only the
    forecast date changes -- the in-memory trace-cache HIT that #91's persistent
    HLO probe could not assert. Re-tracing the fused 9-nest megakernel per date is
    the expensive #114 symptom; with a date-invariant treedef the trace is reused."""

    grid = GridSpec.canary_3km_template()
    trace_count = {"n": 0}

    @jax.jit
    def probe(namelist, x):
        # The namelist enters as a normal pytree arg, so its aux (treedef) is part
        # of the jit cache key -- exactly as _advance_chunk_fori receives it.
        trace_count["n"] += 1
        return x * float(namelist.dt_s)

    x = jnp.ones((4,), dtype=jnp.float64)
    probe(_date_namelist(grid, datetime.datetime(2026, 2, 14, 0, 0)), x)
    assert trace_count["n"] == 1

    # New DATE (and a leap-year date) must reuse the cached trace: no re-trace.
    probe(_date_namelist(grid, datetime.datetime(2026, 5, 12, 6, 0)), x)
    probe(_date_namelist(grid, datetime.datetime(2024, 2, 29, 12, 0)), x)
    assert trace_count["n"] == 1, "namelist re-traced when only the date changed (#114)"
