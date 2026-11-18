"""EXPERIMENTAL ENERGY Z0 field: mask, snow blend, operands, files and refusal.

These are CPU regression/consumer tests, not GPU or full-forecast qualification.
"""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from w41_trackp_fixtures import SHAPE, _fixture, _namelist, pytestmark  # noqa: F401
from gpuwrf.physics.noahmp import energy
from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params, noah_mp_step
from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table
from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream
from gpuwrf.surface_params import controls as sc


def columns():
    land, forcing, static = _fixture()
    barren = int(static.parameters.isbarren)
    assert barren == 16  # MODIS table semantics, not an arbitrary integer mask
    veg = jnp.array([[barren, 7, barren], [2, int(static.parameters.isurban), 12]])
    static = static.replace(ivgtyp=veg, lu_index=veg)
    # Float32 operand as in the runtime. Preserve REAL semantics in the driver.
    ep, rp = build_energy_params(static, SHAPE)
    ep, rp = jax.tree.map(lambda x: x.astype(jnp.float32) if hasattr(x, 'dtype') and
                         jnp.issubdtype(x.dtype, jnp.floating) else x, (ep, rp))
    nl = SimpleNamespace(use_noahmp=True, noahmp_static=static, noahmp_energy_params=ep,
                         noahmp_rad_params=rp, noahmp_options=None)
    return land, forcing, static, nl


def test_z0bare_basis_round_once_and_positional_compatibility():
    _, _, _, nl = columns()
    p = nl.noahmp_energy_params
    assert p.z0bare is None
    assert energy.EnergyParams(*p[:-1]).z0bare is None
    base = sc.table_base('z0bare', nl)
    assert base.dtype == np.float64 and np.all(base == np.float64(.002))
    values = sc.expand_controls(nl, {'z0bare': {int(nl.noahmp_static.parameters.isbarren): 10}}, device=False)
    mask = np.asarray(nl.noahmp_static.ivgtyp) == nl.noahmp_static.parameters.isbarren
    assert values['z0bare'].dtype == np.float32
    assert values['z0bare'][mask].tobytes() == np.full(mask.sum(), .02, np.float32).tobytes()
    assert values['z0bare'][~mask].tobytes() == np.full((~mask).sum(), .002, np.float32).tobytes()


def energy_run(land, forcing, static, nl, field, fsno):
    """Real ENERGY, prescribed snow cover at its routine boundary (incl. urban)."""
    phen = noahmp_phenology_table(land, forcing, static)
    # Expose ground Z0MG as Z0WRF on each non-urban column, independently of LAI.
    phen = phen._replace(fveg=jnp.zeros(SHAPE), elai=jnp.zeros(SHAPE), esai=jnp.zeros(SHAPE))
    rad, extras = radiation_twostream(land, forcing, static, phen, nl.noahmp_rad_params, 60.)
    rad = rad._replace(fsno=jnp.full(SHAPE, fsno, jnp.float32))
    params = nl.noahmp_energy_params
    if field is not None:
        _, params, _ = sc.apply_surface_controls(static, params, nl.noahmp_rad_params, {'z0bare': field})
    return energy.noahmp_energy_canopy(land, forcing, static, rad, 60., phen=phen,
              params=params, rad_extras=extras, isurban=int(static.parameters.isurban))


@pytest.mark.parametrize('fsno', [0., .4, 1.])
def test_z0bare_mask_and_ground_only_snow_blend(fsno):
    land, forcing, static, nl = columns()
    field = sc.expand_controls(nl, {'z0bare': {int(static.parameters.isbarren): 10}})['z0bare']
    base = energy_run(land, forcing, static, nl, None, fsno)
    got = energy_run(land, forcing, static, nl, field, fsno)
    z0 = np.where(np.asarray(static.ivgtyp) == int(static.parameters.isbarren), .02, .002).astype(np.float32)
    expected = z0 * np.float32(1-fsno) + np.asarray(nl.noahmp_energy_params.z0sno) * np.float32(fsno)
    urban = np.asarray(static.ivgtyp) == int(static.parameters.isurban)
    expected = np.where(urban, np.asarray(nl.noahmp_energy_params.z0mvt), expected)
    np.testing.assert_allclose(got[1].z0wrf, expected, rtol=2e-7, atol=1e-9)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(got))
    mask = np.asarray(static.ivgtyp) == int(static.parameters.isbarren)
    # Every exported column is unchanged outside the requested vegetation mask.
    for a, b in zip(jax.tree.leaves(base), jax.tree.leaves(got)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape[-2:] == SHAPE:
            assert a[..., ~mask].tobytes() == b[..., ~mask].tobytes()
    if fsno < 1:
        assert np.all(np.asarray(got[1].z0wrf)[mask] > np.asarray(base[1].z0wrf)[mask])
    else:
        assert [np.asarray(x).tobytes() for x in jax.tree.leaves(base)] == [np.asarray(x).tobytes() for x in jax.tree.leaves(got)]


def test_z0bare_znt_ust_effect_and_jit_value_reuse():
    land, forcing, static, nl = columns()
    # Full Noah-MP step: real MODIS barren phenology (no synthetic mask in ENERGY).
    def step(field):
        s, e, r = sc.apply_surface_controls(static, nl.noahmp_energy_params, nl.noahmp_rad_params, {'z0bare': field})
        return noah_mp_step(land, forcing, s, 60., energy_params=e, rad_params=r)
    call = jax.jit(step)
    prior = sc.expand_controls(nl, {'z0bare': 1})['z0bare']
    changed = sc.expand_controls(nl, {'z0bare': {int(static.parameters.isbarren): 10}})['z0bare']
    base, got = call(prior), call(changed)
    assert call._cache_size() == 1
    mask = np.asarray(static.ivgtyp) == int(static.parameters.isbarren)
    assert np.all(np.asarray(got[1].znt)[mask] > np.asarray(base[1].znt)[mask])
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
    # ZNT feeds the following surface-layer call; ENERGY itself does not export UST.
    def ust(znt):
        lead = lambda v: jnp.asarray(v)[..., None]
        state = SimpleNamespace(u=lead(forcing.uu), v=lead(forcing.vv),
            theta=lead(forcing.sfctmp * (1e5/forcing.sfcprs)**(287./1004.)),
            qv=lead(forcing.qair / (1. - forcing.qair)), p=lead(forcing.sfcprs), dz=jnp.full(SHAPE, 60.),
            t_skin=forcing.sfctmp, xland=jnp.ones(SHAPE), lakemask=jnp.zeros(SHAPE),
            mavail=jnp.full(SHAPE, .6), roughness_m=znt, ustar=jnp.full(SHAPE, .2))
        return surface_layer_with_diagnostics(state).fluxes.ustar
    assert np.all(np.asarray(ust(got[1].znt))[mask] > np.asarray(ust(base[1].znt))[mask])
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(got))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, 0., .00009, .51])
def test_z0bare_refused_before_device_allocation(bad, monkeypatch):
    _, _, _, nl = columns()
    def allocation(*a, **kw):
        pytest.fail('device allocation before z0bare validation')
    monkeypatch.setattr(jnp, 'asarray', allocation)
    with pytest.raises(ValueError, match='non-finite|outside'):
        sc.expand_controls(nl, values={'z0bare': np.full(SHAPE, bad, np.float32)})


def test_z0bare_domains_restart_batch_and_keys(tmp_path):
    from gpuwrf.runtime.aot_cheap_key import canonical_digest, static_config_hash
    from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base
    from gpuwrf.surface_params import cli
    _, _, _, source = columns()
    nl = OperationalNamelist(grid=None, tendencies=None, metrics=None, use_noahmp=True,
           noahmp_static=source.noahmp_static, noahmp_energy_params=source.noahmp_energy_params,
           noahmp_rad_params=source.noahmp_rad_params)
    a = sc.expand_controls(nl, {'z0bare': 1}, device=False)
    b = sc.expand_controls(nl, {'z0bare': {int(source.noahmp_static.parameters.isbarren): 10}}, device=False)
    path = tmp_path/'bare.npz'
    sc.save_controls(path, {f'd{i:02}': b for i in range(3, 10)})
    assert sc.bind_domain_controls(nl, path, 'd01') is nl
    assert sc.bind_domain_controls(nl, path, 'd02') is nl
    for i in range(3, 10):
        bound = sc.bind_domain_controls(nl, path, f'd{i:02}', allow_host=False)
        assert sc.controls_digest(bound.surface_controls) == sc.controls_digest(b)
    na = replace(nl, surface_controls=a)
    nb = replace(nl, surface_controls=b)
    assert canonical_digest(sc.controls_aux(a)) == canonical_digest(sc.controls_aux(b))
    assert static_config_hash(na) == static_config_hash(nb)
    assert jax.tree.structure(na) == jax.tree.structure(nb)
    ca, cb = build_clock_base(na), build_clock_base(nb)
    assert jax.tree.structure(ca) == jax.tree.structure(cb)
    assert len(jax.tree.leaves(ca)) == len(jax.tree.leaves(build_clock_base(nl))) + 1
    # The same operand route used by vmap lanes preserves each member's values.
    stacked = jax.tree.map(lambda x, y: jnp.stack((x, y)), ca, cb)
    got = jax.vmap(lambda clock: sc.traced_controls(na, clock)['z0bare'])(stacked)
    np.testing.assert_array_equal(got, np.stack((a['z0bare'], b['z0bare'])))
    from dataclasses import make_dataclass
    from gpuwrf.integration.nested_pipeline import _bind_batch_surface_controls
    Bundle = make_dataclass('Bundle', ['namelist'], frozen=True)
    loaded = [(None, {'d03': Bundle(nl)}, {}, None, {}, {})] * 2
    rebound = _bind_batch_surface_controls(loaded, ('d03',), [{'d03': a}, {'d03': b}], 2)
    assert [sc.controls_digest(row[1]['d03'].namelist.surface_controls) for row in rebound] == [sc.controls_digest(a), sc.controls_digest(b)]
    with pytest.raises(ValueError, match='one signature'):
        _bind_batch_surface_controls(loaded, ('d03',), [{'d03': a}, None], 2)
    digest = sc.restart_controls_digest(na)
    sc.validate_controls_restart(digest, na)
    for requested in (nb, nl):
        with pytest.raises(ValueError, match='differ'):
            sc.validate_controls_restart(digest, requested)
    sc.refuse_unqualified_sources([path])  # older W2 refusal must not swallow this request
    with pytest.raises(ValueError, match='not available'):
        sc.refuse_unqualified_sources([{'d03': {'z0bare': b['z0bare'], 'z0sno': b['z0bare']}}])
    basis = tmp_path/'basis.npz'
    cli.write_basis({'d03': nl}, basis)
    assert cli.main_args(['expand', '--basis', str(basis), '--output', str(tmp_path/'cli.npz'),
                         '--class', 'd03:z0bare:16=10']) == 0
    loaded, _ = sc.load_controls(tmp_path/'cli.npz')
    assert sc.controls_digest(loaded['d03']) == sc.controls_digest(b)


def test_z0bare_program_digest_sees_large_constant_mutant():
    """E224: full proto sees the value which ordinary HLO text elides."""
    import hashlib
    from z0bare_proto import digest
    def lower(value):
        return jax.jit(lambda x: x + jnp.asarray(value)).lower(jnp.zeros(4096)).compiler_ir('hlo')
    values = np.random.default_rng(224).uniform(.001, .003, 4096).astype(np.float32)
    before, repeat = [lower(values) for _ in range(2)]
    text = before.as_hlo_text()
    raw = before.as_serialized_hlo_module_proto()
    full_digest, _ = digest(raw)
    repeated = repeat.as_serialized_hlo_module_proto()
    assert full_digest == digest(repeated)[0]
    full = before.as_hlo_text(print_large_constants=True)
    values[2048] = .02
    after = lower(values)
    assert text == after.as_hlo_text()
    assert full != after.as_hlo_text(print_large_constants=True)
    assert hashlib.sha256(raw).digest() != hashlib.sha256(after.as_serialized_hlo_module_proto()).digest()
    assert full_digest != digest(after.as_serialized_hlo_module_proto())[0]
