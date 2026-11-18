"""Host-only opt-in ZNT history adapter; no traced-model or base-writer changes.

The CLI --history-znt selects WRFOUT_ZNT=1 (an output-only process setting).
The existing publication hook adds the carried roughness to the unpublished
NetCDF temporary, then delegates to the normal atomic publisher/journal.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
import json
from pathlib import Path

import numpy as np
from gpuwrf.io import wrfout_writer as base
from gpuwrf.io.netcdf_lock import Dataset

# Registry/Registry.EM_COMMON ZNT: real, ij, unstaggered, description and units.
ZNT_SPEC = base._spec('ZNT', base.XY, 'XY ', 'TIME-VARYING ROUGHNESS LENGTH', 'm',
                      coordinates='XLONG XLAT XTIME')


def enabled():
    return os.environ.get('WRFOUT_ZNT', '0').strip().lower() in {'1', 'true', 'yes', 'on'}


def require_serial(batch_size):
    if enabled() and int(batch_size) > 1:
        raise ValueError('--history-znt currently requires B=1; batched ZNT history is unverified')


def recorded_policy(identity):
    value = identity.get('output_policy', {}).get('history', {}).get('history_znt', False)
    if type(value) is not bool:
        raise ValueError('invalid recorded --history-znt output policy')
    return value


def require_same_policy(recorded, requested):
    if recorded_policy(recorded) != recorded_policy(requested):
        raise ValueError('--history-znt output policy differs from checkpoint; resume must preserve ON/OFF (legacy missing value is OFF)')


def require_resume_policy(generation):
    # Manifest-only admission before model initialization/advance or history recovery.
    identity = json.loads((Path(generation) / 'manifest.json').read_text())['identity']
    require_same_policy(identity, {'output_policy': {'history': {'history_znt': enabled()}}})


@dataclass(frozen=True)
class _ZntPublication:
    values: np.ndarray
    dimensions: dict
    publish: object = None

    def __call__(self, temporary, target):
        # No live device references or physics recomputation in the writer thread.
        with Dataset(temporary, 'a') as ds:
            base._write_float_variable(ds, ZNT_SPEC, self.values, self.dimensions, None)
        with Path(temporary).open('rb') as stream:
            os.fsync(stream.fileno())
        publisher = self.publish or base._publish_wrfout_noreplace
        publisher(temporary, target)


def attach_publication(prepared, publish):
    """Preserve the optional ZNT append when binding the restart journal."""
    current = prepared.restart_publication
    callback = replace(current, publish=publish) if isinstance(current, _ZntPublication) else publish
    return replace(prepared, restart_publication=callback)


def prepare_wrfout_payload(state, *args, history_znt=None, **kwargs):
    active = enabled() if history_znt is None else bool(history_znt)
    subset = kwargs.get('variable_subset')
    selected = subset is None or 'ZNT' in subset
    if subset is not None and 'ZNT' in subset:
        if not active:
            raise ValueError('ZNT subset requires --history-znt')
        kwargs = dict(kwargs, variable_subset=tuple(n for n in subset if n != 'ZNT'))
    prepared = base.prepare_wrfout_payload(state, *args, **kwargs)
    if not active or not selected:
        return prepared
    raw = base._lookup_raw(state, 'roughness_m', None)
    if raw is None:
        raise ValueError('ZNT history requires carried State.roughness_m')
    shape = (prepared.dimensions['south_north'], prepared.dimensions['west_east'])
    values = np.asarray(raw, dtype=np.float32)
    if values.size != shape[0] * shape[1]:
        raise ValueError('ZNT history roughness shape mismatch')
    values = values.reshape(shape).copy()
    if not np.isfinite(values).all():
        raise ValueError('nonfinite carried ZNT history')
    callback = _ZntPublication(values, dict(prepared.dimensions), prepared.restart_publication)
    return replace(prepared, restart_publication=callback)


def write_prepared_wrfout(prepared, **kwargs):
    # Shared full payloads may feed several subsets. Select per write, without
    # mutating the shared payload or discarding the underlying history journal.
    subset = kwargs.get('variable_subset')
    callback = prepared.restart_publication
    if isinstance(callback, _ZntPublication) and subset is not None and 'ZNT' not in subset:
        prepared = replace(prepared, restart_publication=callback.publish)
    return base.write_prepared_wrfout(prepared, **kwargs)


def write_wrfout_netcdf(state, *args, **kwargs):
    if not enabled():
        return base.write_wrfout_netcdf(state, *args, **kwargs)
    options = dict(kwargs)
    compress = options.pop('compress', False)
    prepared = prepare_wrfout_payload(state, *args, **options)
    return write_prepared_wrfout(prepared, expected_domain=kwargs['domain'],
        expected_domain_authority=kwargs['domain_authority'],
        variable_subset=kwargs.get('variable_subset'),
        include_mandatory_coords=kwargs.get('include_mandatory_coords', False), compress=compress)
