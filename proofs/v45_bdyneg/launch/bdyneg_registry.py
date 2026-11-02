#!/usr/bin/env python3
"""ALISIOS registry hooks; stdlib only, never import JAX or launch a model.

Use Registry.run for new invocations, update for an authorized metadata repair,
and backfill for a reviewed receipt with pinned metadata evidence. Smoke uses
the same register.main argv and disk append, but requires an explicit private
atlas (an empty atlas or a copy of the two production JSONL files).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
from unittest.mock import patch

REGISTER = Path('<USER_HOME>/src/alisios/ops/northstar/data_map/register.py')
PRODUCTION_ATLAS = Path('<DATA_ROOT>/alisios/state/manager/cross/ATLAS')
_LOCK = threading.RLock()  # register.main reads process-wide environment
_REGISTER = None


class RegistryError(RuntimeError):
    def __init__(self, message, rc=2):
        super().__init__(message)
        self.rc = rc


def load_register():
    global _REGISTER
    if _REGISTER is None:
        spec = importlib.util.spec_from_file_location('_alisios_register', REGISTER)
        module = importlib.util.module_from_spec(spec)
        # ALISIOS inserts its sibling directory into sys.path; do not retain it.
        previous = sys.path[:]
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = previous
        _REGISTER = module
    return _REGISTER


def argv_for(action, metadata):
    """Only WRF hooks, never DATA writes, generation, du or extract hashing."""
    allowed = {
        'start': {'run_id', 'by', 'producer', 'purpose', 'family', 'tags', 'forcing',
                  'engine', 'nests', 'lead_h', 'issue', 'run_dir', 'retention', 'started_utc'},
        'end': {'run_id', 'by', 'rc', 'status', 'corpus', 'raw_deleted', 'size',
                'extract', 'ended_utc', 'notes'},
        'update': {'run_id', 'by', 'rc', 'status', 'corpus', 'raw_deleted', 'size',
                   'extract', 'ended_utc', 'notes'},
    }
    if action not in allowed or set(metadata) - allowed[action]:
        raise RegistryError(f'Unsupported {action} metadata: {sorted(set(metadata) - allowed.get(action, set()))}')
    argv = ['--no-regen', 'wrf', action, '--no-hash']
    for key, value in metadata.items():
        if value is None:
            continue
        values = value if key == 'extract' else [value]
        if key == 'tags' and isinstance(value, (list, tuple)):
            values = [','.join(value)]
        if key == 'extract' and not isinstance(values, (list, tuple)):
            raise RegistryError('extract must be a list of PATH::SHA256 strings')
        for item in values:
            if key == 'raw_deleted' and isinstance(item, bool):
                item = 'yes' if item else 'no'
            argv += ['--' + key.replace('_', '-'), str(item)]
    return argv


@dataclass
class Outcome:
    # A body that forgets to set rc must never claim success.
    rc: int = 1


class Registry:
    def __init__(self, *, atlas_dir=None, smoke=False):
        self.reg = load_register()
        self.smoke = smoke or os.environ.get('WRF_GPU2_SEQ_SMOKE') == '1'
        if self.smoke and atlas_dir is None:
            raise RegistryError('SEQ_SMOKE requires an explicit private atlas_dir')
        self.atlas = Path(atlas_dir if atlas_dir is not None else self.reg.rl.atlas_dir()).resolve()
        production = PRODUCTION_ATLAS.resolve()
        if self.smoke and (self.atlas.is_relative_to(production) or production.is_relative_to(self.atlas)):
            raise RegistryError('Smoke atlas must be private, outside the production atlas')

    @contextmanager
    def _environment(self):
        with _LOCK, patch.dict(os.environ, {'ALISIOS_ATLAS_DIR': str(self.atlas),
                                          'ALISIOS_DATA_MAP_NO_REGEN': '1'}):
            if self.smoke:
                for name in (self.reg.rl.WRF_REGISTRY_NAME, self.reg.rl.WRF_REGISTRY_NAME + '.lock'):
                    private = self.atlas / name
                    real = PRODUCTION_ATLAS / name
                    if private.is_symlink() or (private.exists() and real.exists() and private.samefile(real)):
                        raise RegistryError('Smoke registry files must be private copies, not links')
            yield

    def _schema_check(self, argv):
        # Run the REAL parser/validator. Only final storage is intercepted;
        # state checks run on the real append immediately after this precheck.
        with patch.object(self.reg.rl, '_append', lambda path, event, check: None), redirect_stdout(io.StringIO()):
            return self.reg.main(argv)

    def _call(self, argv):
        return self.reg.main(argv)

    def _event(self, action, metadata):
        argv = argv_for(action, metadata)
        with self._environment():
            rc = self._schema_check(argv)
            if rc:
                raise RegistryError(f'{action} schema pre-check refused', rc)
            rc = self._call(argv)
            if rc:
                raise RegistryError(f'{action} registry write refused', rc)

    def start(self, metadata):
        self._event('start', metadata)

    def end(self, metadata):
        self._event('end', metadata)

    def update(self, metadata, *, authorization):
        if not isinstance(authorization, str) or not authorization.strip():
            raise RegistryError('Repair requires an explicit authorization reference')
        metadata = dict(metadata)
        if metadata.get('status') in self.reg.rl.RUN_OPEN_STATUS:
            raise RegistryError('Repair must not reopen a run; use a new ID for a new invocation')
        metadata['notes'] = (metadata.get('notes', '') + f' | Authorized repair: {authorization}').strip(' |')
        self._event('update', metadata)

    @contextmanager
    def run(self, start, *, end):
        """Register before yielding; always write one terminal event in finally.

        This cannot catch SIGKILL/power loss. Reconcile those from receipts with
        backfill; do not register a second start for the same invocation.
        Child argv/env must include ALISIOS_WRF_RUN_ID=start['run_id'].
        """
        outcome = Outcome()
        if set(end) & {'rc', 'status', 'run_id', 'by'}:
            raise RegistryError('run() determines terminal rc/status/identity from its outcome')
        terminal = dict(end, run_id=start.get('run_id'), by=start.get('by'))
        with self._environment():
            rc = self._schema_check(argv_for('end', dict(terminal, rc=1, status='FAILED')))
            if rc:
                raise RegistryError('Terminal metadata schema pre-check refused', rc)
        self.start(start)
        try:
            yield outcome
        except (KeyboardInterrupt, SystemExit) as exc:
            outcome.rc = 130 if isinstance(exc, KeyboardInterrupt) else (exc.code or 1)
            if not isinstance(outcome.rc, int):
                outcome.rc = 1
            terminal['status'] = 'ABORTED'
            raise
        except BaseException as exc:
            outcome.rc = getattr(exc, 'returncode', 1) or 1
            terminal['status'] = 'FAILED'
            raise
        finally:
            terminal['rc'] = outcome.rc
            terminal.setdefault('status', ('DONE_CLEAN' if terminal.get('raw_deleted') is True
                                          else 'DONE_HELD') if outcome.rc == 0 else 'FAILED')
            self.end(terminal)

    def _events(self):
        return self.reg.rl.read_events(self.atlas / self.reg.rl.WRF_REGISTRY_NAME)

    def backfill(self, receipt):
        """Replay reviewed {start, end, evidence: {absolute_metadata_path: SHA}}.

        The adapter producing the receipt owns its factual interpretation.
        Unknown starts/ends must not be invented. Explicit UTC timestamps and
        small pinned evidence files are required. Identical replay is a no-op;
        conflicting terminal state requires update with authorization instead.
        """
        start, end = dict(receipt['start']), dict(receipt['end'])
        if not receipt.get('evidence'):
            raise RegistryError('Backfill requires pinned receipt evidence')
        for name, expected in receipt['evidence'].items():
            path = Path(name)
            if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**2:
                raise RegistryError('Backfill evidence must be an absolute small metadata file')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RegistryError(f'Backfill evidence changed: {path}')
        if (not start.get('started_utc') or not end.get('ended_utc')
                or start.get('run_id') != end.get('run_id') or end['ended_utc'] < start['started_utc']):
            raise RegistryError('Backfill needs matching run IDs and ordered start/end timestamps')
        for stamp in (start['started_utc'], end['ended_utc']):
            datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ')
        with self._environment():
            # Precheck BOTH events before creating a possibly unfinishable start.
            for action, metadata in [('start', start), ('end', end)]:
                rc = self._schema_check(argv_for(action, metadata))
                if rc:
                    raise RegistryError(f'Backfill {action} schema refused', rc)
            events = [e for e in self._events() if e.get('run_id') == start['run_id']]
            if events:
                actual = self.reg.rl.fold_wrf(events)[start['run_id']]
                for key, value in start.items():
                    field = 'planned_retention' if key == 'retention' else key
                    if key == 'tags' and isinstance(value, str):
                        value = self.reg.tags(value)
                    if key != 'by' and actual.get(field) != value:
                        raise RegistryError(f'Backfill start conflicts: {field}')
                if any(e['event'] == 'end' for e in events):
                    # Compare event metadata, not later authorized updates.
                    actual_end = next(e for e in events if e['event'] == 'end')
                    for key, value in end.items():
                        field = 'size_bytes' if key == 'size' else key
                        if key == 'size':
                            value = self.reg.parse_size(value)
                        if key == 'raw_deleted' and isinstance(value, str):
                            value = self.reg.yesno(value)
                        if key not in ('by', 'extract') and actual_end.get(field) != value:
                            raise RegistryError(f'Backfill end conflicts: {key}')
                    wanted = self.reg.parse_extracts(end.get('extract'), hash_missing=False)
                    if actual_end.get('extracts') != wanted:
                        raise RegistryError('Backfill extracts conflict')
                    return 'already-recorded'
            else:
                self.start(start)
            self.end(end)
            return 'recorded'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'end', 'update', 'backfill'])
    parser.add_argument('metadata', type=Path, help='JSON metadata or reviewed backfill receipt')
    parser.add_argument('--atlas-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--authorization', default='')
    args = parser.parse_args(argv)
    try:
        registry = Registry(atlas_dir=args.atlas_dir, smoke=args.smoke)
        metadata = json.loads(args.metadata.read_text())
        if args.action == 'update':
            registry.update(metadata, authorization=args.authorization)
        elif args.action == 'backfill':
            registry.backfill(metadata)
        else:
            getattr(registry, args.action)(metadata)
        return 0
    except (RegistryError, OSError, ValueError, KeyError) as exc:
        print(f'Registry refused: {exc}', file=sys.stderr)
        return exc.rc if isinstance(exc, RegistryError) else 2


if __name__ == '__main__':
    sys.exit(main())
