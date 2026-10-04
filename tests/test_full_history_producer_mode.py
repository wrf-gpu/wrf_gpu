"""The writer and all resident producers must resolve the same output mode."""
import pytest


@pytest.mark.parametrize('canonical,alias,expected', [
    (None, None, True), ('0', None, False), (None, '0', False),
    ('0', '1', False), ('1', '0', True), (' OFF ', 'on', False),
    (None, '1', True),
])
def test_full_history_producers_match_writer(monkeypatch, canonical, alias, expected):
    for name, value in [('GPUWRF_FULL_WRFOUT_VARIABLES', canonical), ('GPUWRF_FULL_WRFOUT', alias)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    from gpuwrf.config.history_output import full_wrfout_variables_enabled
    from gpuwrf.integration.nested_pipeline import _resolve_full_wrfout_variables
    from gpuwrf.runtime.history_accumulators import full_history_enabled
    assert full_wrfout_variables_enabled() is expected
    assert _resolve_full_wrfout_variables() is expected
    assert full_history_enabled() is expected
