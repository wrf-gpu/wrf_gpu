"""B41: per-case Noah-MP inputs (TMN, VEGFRA) must not key the compiled program.

Two forecast cases on one grid differ in NoahMPStatic.tbot/shdfac (read from each
case's wrfinput) while all geography is identical. Those two fields ride the
namelist as TRACED leaves, so the treedef (in-memory jit key) and the AOT
static_config_hash are case-invariant, while geography still keys the program.
"""
import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.noahmp_state import NoahMPStatic
from gpuwrf.contracts.state import Tendencies
from gpuwrf.runtime.aot_cheap_key import static_config_hash
from gpuwrf.runtime.operational_mode import OperationalNamelist


def _tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    return Tendencies(z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)), z((nz, ny, nx)),
                      z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)))


def _static(grid: GridSpec, *, tbot: float, shdfac: float, veg: int = 7) -> NoahMPStatic:
    ny, nx = grid.ny, grid.nx
    f32 = lambda v: jnp.full((ny, nx), v, dtype=jnp.float32)  # noqa: E731
    i32 = lambda v: jnp.full((ny, nx), v, dtype=jnp.int32)  # noqa: E731
    return NoahMPStatic(
        ivgtyp=i32(veg), isltyp=i32(3), xland=f32(1.0), landmask=f32(1.0), lakemask=f32(0.0),
        lu_index=i32(veg), tbot=f32(tbot), dzs=jnp.asarray([0.1, 0.3, 0.6, 1.0], jnp.float32),
        zsoil=jnp.asarray([-0.1, -0.4, -1.0, -2.0], jnp.float32), lat=f32(28.3), dx_m=1000.0,
        parameters={"bexp": jnp.asarray([4.05, 4.26], jnp.float32)}, shdmax=f32(0.6), shdfac=f32(shdfac),
    )


def _namelist(grid: GridSpec, static: NoahMPStatic) -> OperationalNamelist:
    import dataclasses

    return dataclasses.replace(OperationalNamelist.from_grid(grid, tendencies=_tendencies(grid)),
                               noahmp_static=static)


def test_case_fields_do_not_key_treedef_or_static_hash():
    grid = GridSpec.canary_3km_template()
    case_a = _namelist(grid, _static(grid, tbot=285.0, shdfac=0.31))
    case_b = _namelist(grid, _static(grid, tbot=291.5, shdfac=0.44))

    leaves_a, treedef_a = jax.tree_util.tree_flatten(case_a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(case_b)
    assert treedef_a == treedef_b
    assert static_config_hash(case_a) == static_config_hash(case_b)
    # The case data are ordinary (traced) leaves now.
    assert len(leaves_a) == len(leaves_b)
    assert any(np.array_equal(np.asarray(leaf), np.asarray(case_a.noahmp_static.tbot)) for leaf in leaves_a)


def test_case_fields_round_trip_exactly():
    grid = GridSpec.canary_3km_template()
    static = _static(grid, tbot=288.25, shdfac=0.37)
    leaves, treedef = jax.tree_util.tree_flatten(_namelist(grid, static))
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves).noahmp_static

    for name in NoahMPStatic.__slots__:
        got, want = getattr(rebuilt, name), getattr(static, name)
        if name == "parameters":
            assert jax.tree_util.tree_all(jax.tree_util.tree_map(np.array_equal, got, want))
        elif hasattr(want, "shape"):
            assert np.array_equal(np.asarray(got), np.asarray(want)), name
        else:
            assert got == want, name


def test_geography_still_keys_the_program():
    grid = GridSpec.canary_3km_template()
    base = _namelist(grid, _static(grid, tbot=285.0, shdfac=0.31, veg=7))
    other_veg = _namelist(grid, _static(grid, tbot=285.0, shdfac=0.31, veg=12))
    assert static_config_hash(base) != static_config_hash(other_veg)
    assert jax.tree_util.tree_structure(base) != jax.tree_util.tree_structure(other_veg)


def test_jit_reuses_trace_across_cases_and_reads_the_new_values():
    grid = GridSpec.canary_3km_template()
    traces = {"n": 0}

    @jax.jit
    def probe(namelist):
        traces["n"] += 1
        static = namelist.noahmp_static
        return jnp.mean(static.tbot) + jnp.mean(static.shdfac)

    first = probe(_namelist(grid, _static(grid, tbot=285.0, shdfac=0.25)))
    second = probe(_namelist(grid, _static(grid, tbot=290.0, shdfac=0.5)))
    assert traces["n"] == 1, "namelist re-traced when only per-case TMN/VEGFRA changed (B41)"
    assert float(first) == 285.25
    assert float(second) == 290.5
