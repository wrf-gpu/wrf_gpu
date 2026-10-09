"""Stored metrics must pass E200 again before render or summary reductions."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hourly_deviation_bands as bands


@pytest.fixture
def payload():
    fields = {}
    for name, unit, scale in bands.FIELDS:
        rows = []
        for h in range(73):
            if name == 'RAINNC_hourly' and h == 0:
                rows.append({'lead_h': h, 'status': 'undefined: no preceding hour'})
            else:
                rows.append({'lead_h': h, 'n': 2, 'rmse': 2., 'bias': .5,
                             'p05_abs': .1, 'p25_abs': .2, 'p75_abs': 2., 'p95_abs': 3.,
                             'pair_rmse': 1., 'spread_ratio': 2.})
        fields[name] = {'unit': unit, 'scale': scale, 'limit': 4., 'by_lead': rows}
    return {'expected_domains': ['d01'], 'domains': {'d01': {'fields': fields}}}


@pytest.mark.parametrize('field', [f[0] for f in bands.FIELDS])
@pytest.mark.parametrize('metric', ['rmse', 'bias', 'p05_abs', 'p25_abs', 'p75_abs', 'p95_abs', 'pair_rmse', 'spread_ratio'])
@pytest.mark.parametrize('fault', [float('nan'), float('inf'), -float('inf')])
def test_every_rendered_field_rejects_nonfinite_before_plotting(payload, monkeypatch, tmp_path, field, metric, fault):
    payload['domains']['d01']['fields'][field]['by_lead'][37][metric] = fault
    def no_plot_work():
        raise AssertionError('render path reached plotting before finite validation')
    monkeypatch.setattr(bands, 'quiet', no_plot_work)
    with pytest.raises(ValueError, match='nonfinite'):
        bands.render(payload, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field', [f[0] for f in bands.FIELDS])
@pytest.mark.parametrize('key', ['scale', 'limit'])
@pytest.mark.parametrize('fault', [float('nan'), float('inf'), -float('inf')])
def test_field_metadata_nonfinite(payload, field, key, fault):
    payload['domains']['d01']['fields'][field][key] = fault
    with pytest.raises(ValueError, match='nonfinite'):
        bands.validate_metrics(payload)


def test_none_pairs_zero_pairs_and_initial_rain_are_legitimate(payload):
    row = payload['domains']['d01']['fields']['T2']['by_lead'][0]
    row.update(pair_rmse=None, spread_ratio=None)
    other = payload['domains']['d01']['fields']['T2']['by_lead'][1]
    other.update(pair_rmse=0., spread_ratio=None)
    assert bands.validate_metrics(payload) is payload


@pytest.mark.parametrize('defect', ['hour', 'all_tau72', 'duplicate', 'field', 'domain', 'statistic', 'rain0', 'ratio'])
def test_incomplete_or_invalid_payloads(payload, defect):
    fields = payload['domains']['d01']['fields']
    if defect == 'hour': fields['T2']['by_lead'].pop()
    elif defect == 'all_tau72':
        for field in fields.values(): field['by_lead'].pop()
    elif defect == 'duplicate': fields['T2']['by_lead'][2]['lead_h'] = 1
    elif defect == 'field': del fields['T2']
    elif defect == 'domain': payload['expected_domains'].append('d02')
    elif defect == 'statistic': del fields['T2']['by_lead'][4]['bias']
    elif defect == 'rain0': fields['RAINNC_hourly']['by_lead'][0]['rmse'] = float('nan')
    elif defect == 'ratio': fields['T2']['by_lead'][4]['spread_ratio'] = None
    with pytest.raises(ValueError):
        bands.validate_metrics(payload)
