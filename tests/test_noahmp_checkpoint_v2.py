"""Sprint S6a — checkpoint FORMAT_VERSION 2 Noah-MP land carry round-trip (ADR §5).

Gates:
  * v2 write -> read round-trips the prognostic NoahMPLandState BIT-IDENTICALLY
    (every field, dtype + shape + value), the P0-5 restart land gate.
  * save -> load -> ONE noah_mp_step is bit-identical to the same step taken on the
    original (un-pickled) land carry (restart continuity).
  * a v1-shape checkpoint (land_state omitted) is still READABLE and yields
    land_state == None (cold-init path), and FORMAT_VERSION stayed backward-readable.
  * the field-order guard fails closed on schema drift.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
from gpuwrf.config.paths import wrf_run_dir
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.checkpoint import (
    FORMAT_VERSION,
    read_checkpoint_with_land_state,
    write_checkpoint,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist

TABLE_DIR = wrf_run_dir()
HAVE_TABLES = (TABLE_DIR / "MPTABLE.TBL").exists()


def _state(grid):
    shapes = _state_field_shapes(grid)
    fields = {
        f: jnp.asarray(np.arange(int(np.prod(s)), dtype=np.float64).reshape(s) + i,
                       dtype=DEFAULT_DTYPES.dtype_for(f))
        for i, (f, s) in enumerate(shapes.items(), start=1)
    }
    return State(**fields)


def _namelist(grid):
    shapes = _state_field_shapes(grid)
    shape_key = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tend = Tendencies(**{
        k: jnp.zeros(shapes[shape_key.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
        for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
    })
    return OperationalNamelist(grid=grid, tendencies=tend, metrics=grid.metrics,
                               dt_s=10.0, acoustic_substeps=10)


def _land(ny=2, nx=3, seed=1.0):
    sh = (ny, nx)
    rng = np.random.default_rng(int(seed))

    def s2(scale, base):
        return jnp.asarray(base + scale * rng.random(sh), dtype=jnp.float64)

    def soil(base):
        return jnp.asarray(base + 0.01 * rng.random((NSOIL, ny, nx)), dtype=jnp.float64)

    def snow(base):
        return jnp.asarray(base + 0.01 * rng.random((NSNOW, ny, nx)), dtype=jnp.float64)

    zsnso = jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, 0.0, -0.05, -0.25, -0.7, -1.5]).reshape(NSNOW + NSOIL, 1, 1),
        (NSNOW + NSOIL, ny, nx))
    return NoahMPLandState(
        tslb=soil(290.0), smois=soil(0.2), sh2o=soil(0.2), smcwtd=s2(0.0, 0.2),
        isnow=jnp.zeros(sh, dtype=jnp.int32),
        tsno=snow(272.0), snice=snow(0.0), snliq=snow(0.0), zsnso=zsnso,
        snowh=s2(0.0, 0.0), sneqv=s2(0.0, 0.0), sneqvo=s2(0.0, 0.0),
        tauss=s2(0.0, 0.0), albold=s2(0.0, 0.2),
        tv=s2(2.0, 295.0), tg=s2(2.0, 294.0), tah=s2(2.0, 294.0), eah=s2(10.0, 2000.0),
        canliq=s2(0.0, 0.0), canice=s2(0.0, 0.0), fwet=s2(0.0, 0.0),
        lai=s2(0.5, 2.0), sai=s2(0.1, 0.4), cm=s2(0.001, 0.01), ch=s2(0.001, 0.01),
        t_skin=s2(2.0, 294.0), qsfc=s2(0.001, 0.01), znt=s2(0.0, 0.05),
        emiss=s2(0.0, 0.97), albedo=s2(0.0, 0.2), sfcrunoff=s2(0.0, 0.0), udrunoff=s2(0.0, 0.0),
    )


def test_format_version_is_2():
    # The v0.6.0 integration bumped checkpoint FORMAT_VERSION 2->3 (additive
    # cumulus_carry / scheme-carry leaves); versions 1/2 stay backward-readable
    # (SUPPORTED_FORMAT_VERSIONS). The current authoritative version is 3.
    assert FORMAT_VERSION == 3


def test_v2_land_roundtrip_bit_identical(tmp_path: Path):
    grid = GridSpec.canary_3km_template()
    state, namelist = _state(grid), _namelist(grid)
    land = _land()

    path = tmp_path / "restart_v2.pkl"
    write_checkpoint(state, namelist, grid, 23, path, land_state=land)
    _, _, _, step, land_r = read_checkpoint_with_land_state(path)

    assert step == 23
    assert land_r is not None
    for f in NoahMPLandState.__slots__:
        a = np.asarray(getattr(land, f))
        b = np.asarray(getattr(land_r, f))
        assert b.dtype == a.dtype, f
        assert b.shape == a.shape, f
        assert np.array_equal(b, a), f


def test_v1_checkpoint_reads_land_none(tmp_path: Path):
    grid = GridSpec.canary_3km_template()
    state, namelist = _state(grid), _namelist(grid)
    # no land_state -> v1-equivalent payload (land keys None).
    path = tmp_path / "restart_v1.pkl"
    write_checkpoint(state, namelist, grid, 5, path)
    _, _, _, step, land_r = read_checkpoint_with_land_state(path)
    assert step == 5
    assert land_r is None  # cold-init path


def test_land_field_order_guard_fails_closed(tmp_path: Path):
    grid = GridSpec.canary_3km_template()
    state, namelist = _state(grid), _namelist(grid)
    land = _land()
    path = tmp_path / "restart_drift.pkl"
    write_checkpoint(state, namelist, grid, 1, path, land_state=land)

    import pickle
    payload = pickle.loads(path.read_bytes())
    # corrupt the recorded field order -> reader must reject.
    payload["noahmp_land_field_order"] = list(reversed(payload["noahmp_land_field_order"]))
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError):
        read_checkpoint_with_land_state(path)


@pytest.mark.skipif(not HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_restart_continuity_one_step_bit_identical(tmp_path: Path):
    """save -> load -> 1 noah_mp_step == same step on the un-pickled carry."""
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
    from gpuwrf.physics.noahmp.types import NoahMPForcing

    grid = GridSpec.canary_3km_template()
    state, namelist = _state(grid), _namelist(grid)
    ny, nx = 2, 3
    land = _land(ny, nx)

    z = jnp.zeros((ny, nx))
    veg = jnp.asarray([[5, 9, 10], [5, 9, 10]], dtype=jnp.int32)
    forcing = NoahMPForcing(
        sfctmp=jnp.full((ny, nx), 295.0), sfcprs=jnp.full((ny, nx), 95000.0),
        psfc=jnp.full((ny, nx), 95000.0), uu=jnp.full((ny, nx), 5.0), vv=jnp.full((ny, nx), 1.0),
        qair=jnp.full((ny, nx), 0.008), qc=z, soldn=jnp.full((ny, nx), 600.0),
        lwdn=jnp.full((ny, nx), 350.0), prcpconv=z, prcpnonc=z, prcpsnow=z, prcpgrpl=z,
        prcphail=z, cosz=jnp.full((ny, nx), 0.6), zlvl=jnp.full((ny, nx), 10.0),
        julian=jnp.asarray(142.0), yearlen=jnp.asarray(365.0),
    )
    static = NoahMPStatic(
        ivgtyp=veg, isltyp=jnp.full((ny, nx), 6, dtype=jnp.int32),
        xland=jnp.ones((ny, nx)), landmask=jnp.ones((ny, nx)), lakemask=z,
        lu_index=veg, tbot=jnp.full((ny, nx), 290.0),
        dzs=jnp.asarray([0.05, 0.20, 0.45, 0.80]), zsoil=jnp.asarray([-0.05, -0.25, -0.7, -1.5]),
        lat=jnp.full((ny, nx), 28.0), dx_m=1000.0,
        parameters=load_noahmp_parameters(TABLE_DIR),
        shdmax=jnp.full((ny, nx), 0.3), shdfac=jnp.full((ny, nx), 0.3),
    )

    path = tmp_path / "restart_cont.pkl"
    write_checkpoint(state, namelist, grid, 0, path, land_state=land)
    _, _, _, _, land_loaded = read_checkpoint_with_land_state(path)

    out_ref, fx_ref = noah_mp_step(land, forcing, static, 90.0)
    out_rst, fx_rst = noah_mp_step(land_loaded, forcing, static, 90.0)

    for f in NoahMPLandState.__slots__:
        a = np.asarray(getattr(out_ref, f))
        b = np.asarray(getattr(out_rst, f))
        assert np.array_equal(b, a), f
    for f in fx_ref._fields:
        assert np.array_equal(np.asarray(getattr(fx_rst, f)), np.asarray(getattr(fx_ref, f))), f
