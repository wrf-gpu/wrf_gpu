"""Independent literal contract for the three reviewed v0.3.1 b-core selections (A1-SL + #14)."""
import os
import pytest
from gpuwrf import _fast_defaults as defaults

EXPECTED = {
    'GPUWRF_ACOUSTIC_W_RECUR_SL': '1',
    'GPUWRF_NOAHMP_LAYER_LISTS': '1',
    'GPUWRF_NOAHMP_COLUMN_KERNELS': '1',
}


@pytest.fixture(autouse=True)
def _restore_release_default_environment():
    # apply_fast_path_defaults writes every unset release key, beyond EXPECTED.
    # Preserve the caller's full flag set and status for the following tests.
    keys = tuple(defaults.FAST_PATH_DEFAULTS) + ('GPUWRF_FAST_DEFAULTS',)
    before = {key: os.environ.get(key) for key in keys}
    status = dict(defaults.FAST_DEFAULTS_STATUS)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        defaults.FAST_DEFAULTS_STATUS.clear()
        defaults.FAST_DEFAULTS_STATUS.update(status)


def test_v031_default_values_are_the_registered_selections():
    assert {key: defaults.FAST_PATH_DEFAULTS.get(key) for key in EXPECTED} == EXPECTED


@pytest.mark.parametrize('override', list(EXPECTED))
def test_v031_defaults_keep_each_explicit_optout(monkeypatch, override):
    for key in EXPECTED:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GPUWRF_FAST_DEFAULTS', '1')
    monkeypatch.setenv(override, '0')
    defaults.apply_fast_path_defaults()
    expected = dict(EXPECTED, **{override: '0'})
    assert {key: os.environ.get(key) for key in EXPECTED} == expected


def test_v031_master_optout_leaves_new_selections_unset(monkeypatch):
    for key in EXPECTED:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GPUWRF_FAST_DEFAULTS', '0')
    defaults.apply_fast_path_defaults()
    assert all(key not in os.environ for key in EXPECTED)
