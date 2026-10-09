"""Original P0227 savepoints store ozone VMR despite their ppmv label.

Run with GPUWRF_FAST_DEFAULTS=1: the release hydrator requires native RK
base-state and parent-ozone carries, which the legacy CPU suite disables.
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from gpuwrf.contracts import state as state_module
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
from gpuwrf.validation import wrf_restart_hydration as hydration


@pytest.fixture(scope='module')
def original_restart(tmp_path_factory):
    if os.environ.get('GPUWRF_DYN_RK_FP32') != '1' or os.environ.get('GPUWRF_NEST_O3_FROM_PARENT') != '1':
        pytest.skip('requires native release carries; set GPUWRF_FAST_DEFAULTS=1')
    directory = Path(os.environ.get('GPUWRF_P0227_RESTART_DIR',
        '<USER_HOME>/wrf_gpu2_lanes/wn3/V33/p0227/run'))
    paths = {name: directory / f'wrfrst_{name}_2026-02-28_12:00:00'
             for name in ('d01', 'd02', 'd03')}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip('requires original P0227 CPU-WRF tau12 savepoints')
    assert jax.devices()[0].platform == 'cpu'
    snapshots = {name: hydration.read_wrfrst(path) for name, path in paths.items()}
    output = tmp_path_factory.mktemp('original_ozone')
    config = NestedPipelineConfig(input_dir=directory, output_dir=output/'history',
        proof_dir=output/'proof', hours=30, max_dom=3,
        feedback=False, emit_initial_history=False)
    original = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices('cpu')[0]
    try:
        _, bundles, _, _, _, templates = hydration.load_restart_templates(config, snapshots)
    finally:
        state_module._gpu_device = original
    return snapshots, bundles, templates


def assert_original_ozone(carry, snapshot):
    raw = snapshot.values['O3RAD']
    assert snapshot.units['O3RAD'] == 'ppmv'
    assert np.isfinite(raw).all()
    # These are the original stored VMR magnitudes, not interpreted units.
    assert 1e-8 < raw.min() < 1e-7
    assert 1e-6 < raw.max() < 1e-5
    actual = np.asarray(jax.device_get(carry.o3rad))
    expected = np.asarray(raw, dtype=actual.dtype)
    assert np.isfinite(actual).all()
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes(), 'hydrated ozone differs from original WRF VMR'


@pytest.mark.parametrize('name', ('d01', 'd02', 'd03'))
def test_original_ozone_survives_restart_import(original_restart, name):
    snapshots, bundles, templates = original_restart
    carry, _, receipt = hydration.hydrate_wrfrst(
        templates[name], snapshots[name], bundles[name].namelist)
    assert_original_ozone(carry, snapshots[name])
    assert receipt['source_sha256'] == snapshots[name].sha256


def test_original_file_gate_rejects_old_ppmv_conversion(original_restart):
    """Exercise the real importer with its historical erroneous scaling."""
    tree = ast.parse(inspect.getsource(hydration.hydrate_wrfrst))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == 'extra'
        and isinstance(node.targets[0].slice, ast.Constant)
        and node.targets[0].slice.value == 'o3rad']
    assert len(assignments) == 1
    node = assignments[0]
    node.value.args[1] = ast.BinOp(left=node.value.args[1], op=ast.Mult(),
                                  right=ast.Constant(value=1e-6))
    namespace = dict(vars(hydration))
    exec(compile(ast.fix_missing_locations(tree), '<old-ozone-conversion>', 'exec'), namespace)
    snapshots, bundles, templates = original_restart
    carry, _, _ = namespace['hydrate_wrfrst'](
        templates['d01'], snapshots['d01'], bundles['d01'].namelist)
    with pytest.raises(AssertionError, match='hydrated ozone differs'):
        assert_original_ozone(carry, snapshots['d01'])
