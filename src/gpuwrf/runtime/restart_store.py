"""Durable, bounded generations for synchronized operational restart.

Only the manifest of an atomically published generation makes it eligible.
Temporary and corrupt generations are evidence and are never auto-deleted.
Pickles are local runtime artifacts, verified before loading; not an import
format for files from other producers.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import stat
import time
import uuid

import jax
import numpy as np

from gpuwrf.io.restart import _carry_to_payload, _restore_carry

FORMAT = "gpuwrf-synchronized-restart-v1"


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _storage_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _host(tree):
    return jax.tree.map(lambda x: np.asarray(x) if isinstance(x, (jax.Array, np.ndarray)) else x, tree)


def _diagnostic_restart_config(namelist):
    """Host-only selections/cadences, bound to the native per-domain identity."""
    from gpuwrf.diagnostics.online import configuration
    from gpuwrf.diagnostics.precipitation import configured_interval
    from gpuwrf.diagnostics.afwa import configuration as afwa_configuration
    config = {**configuration(namelist), "prec_acc_dt": configured_interval(namelist)}
    # Only an active selection enters the identity: OFF stores keep the released key set.
    return {**config, **{k: v for k, v in afwa_configuration(namelist).items() if v}}


def _grid_fdda_restart_config(namelist):
    from gpuwrf.assimilation.fdda_grid import GridFddaInputs
    from gpuwrf.assimilation.fdda_spectral import SpectralFddaInputs
    inputs = getattr(namelist, "data_assimilation", None)
    return isinstance(inputs, GridFddaInputs) and not isinstance(inputs, SpectralFddaInputs)


def _urban_bep_restart_config(namelist):
    """Effective BEP(2)/BEP+BEM(3) selection and domain bound to the native identity (E78)."""
    grid = namelist.grid
    config = {"sf_urban_physics": int(getattr(namelist, "sf_urban_physics", 0) or 0),
              "domain_shape": [int(grid.nz), int(grid.ny), int(grid.nx)]}
    bem = getattr(getattr(namelist, "urban_bundle", None), "bem", None)
    if config["sf_urban_physics"] == 3 and bem is not None:
        from gpuwrf.physics.urban_bem_model import extensions
        # Only active PV/green-roof selections are bound; PV/GR-off identities stay unchanged.
        if extensions(bem):
            config["bem_extensions"] = list(extensions(bem))
    return config


def _validate_urban_bep_snapshot(payload, configs, names, name, modern):
    """Same BEP/BEM admission (incl. the BEM QGR_URB3D leaf) on save, readback and public read."""
    group = payload["extra_fields"].get("urban_bep")
    if configs is None:
        if modern or group is not None:
            raise ValueError("restart BEP/BEM selection/domain binding missing (E78)")
        return
    if not isinstance(configs, Mapping) or set(configs) != set(names):
        raise ValueError("restart BEP/BEM configuration domains differ (E78)")
    config = configs[name]
    shape = config.get("domain_shape") if isinstance(config, Mapping) else None
    option = config.get("sf_urban_physics") if isinstance(config, Mapping) else None
    if (type(option) is not int or not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(n) is not int or n <= 0 for n in shape)
            or tuple(shape) != np.shape(payload["state_fields"]["theta"])):
        raise ValueError("restart BEP/BEM bound selection/domain shape differs (E78)")
    if set(config) - {"sf_urban_physics", "domain_shape", "bem_extensions"}:
        raise ValueError("restart BEP/BEM bound configuration has unknown keys (E78)")
    from gpuwrf.io.restart_schema import validate_urban_bep
    validate_urban_bep(group, {"sf_urban_physics": option}, grid_shape=tuple(shape[1:]), nz=shape[0],
                       extensions=config.get("bem_extensions", [] if option == 3 else None))


def _nested_output_policy(config, bundles):
    """Bind the writer's resolved policy independently of the AOT environment.

    IO-only flags are intentionally absent from the HLO key. They still belong
    to an authenticated output stream, including an explicitly forked stream.
    Use the writer's resolvers/constants rather than raw environment spellings.
    """
    from dataclasses import asdict
    from types import SimpleNamespace
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io import wrfout_writer as writer
    from gpuwrf.io.data_inventory import colonfree_output_enabled
    from gpuwrf.io.znt_history import enabled as znt_enabled
    from gpuwrf.runtime import operational_mode
    from gpuwrf.streams.config import FILENAME, read_streams

    names = tuple(bundles)
    subset = pipeline._resolve_training_output_subset()
    full = pipeline._resolve_full_wrfout_variables()
    namelist_path = Path(config.input_dir) / "namelist.input"
    if namelist_path.is_file():
        from gpuwrf.io.gen2_accessor import parse_namelist
        namelist = parse_namelist(namelist_path)
    else:
        # Structural callers can supply resolved bundles without a case tree.
        namelist = {"time_control": {"history_interval": [
            getattr(bundle.namelist, "diagnostic_history_interval", 60.)
            for bundle in bundles.values()]}}
    intervals = pipeline._history_interval_minutes_by_domain(SimpleNamespace(namelist=namelist), names)
    streams_path = Path(config.input_dir) / FILENAME
    regional = [asdict(spec) for spec in read_streams(streams_path)] if streams_path.is_file() else []
    asynchronous = pipeline._nested_async_output_from_env()
    materialization = asynchronous and pipeline._nested_output_pipeline_from_env()
    return {
        "schema_version": 1,
        "history": {
            "domains": list(names), "interval_minutes": intervals,
            "emit_initial": bool(config.emit_initial_history),
            "full_wrfout": full,
            # Missing is the canonical legacy OFF representation, so old OFF
            # policy dictionaries and journal owners remain byte-for-byte stable.
            **({"history_znt": True} if znt_enabled() else {}),
            "variable_order": list(writer.FULL_WRFOUT_VARIABLES if full else writer.OPERATIONAL_WRFOUT_VARIABLES),
            "variable_subset": None if subset is None else sorted(subset),
            "mandatory_coordinates": [] if subset is None else sorted(writer.MANDATORY_WRFOUT_COORDINATES),
            "compression": None if subset is None else dict(writer._SUBSET_STREAM_COMPRESSION),
        },
        "regional_streams": regional,
        # Opt-in extract: absent == OFF so OFF identities stay byte-for-byte stable.
        **({"extract": config.extract.policy()} if getattr(config, "extract", None) is not None else {}),
        "writer": {
            "format": "NETCDF4", "colonfree_filenames": colonfree_output_enabled(),
            "rvovrd_real": bool(writer._WRITER_RVOVRD_REAL),
            "native_base_state": os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1",
            "history_swdown_horizontal": bool(operational_mode._HISTORY_SWDOWN_HORIZONTAL),
            "m9_radiation_from_carry": pipeline._nested_m9_radiation_from_carry_from_env(),
            "asynchronous": asynchronous, "materialization_pipeline": materialization,
            "snapshot_queue_depth": pipeline._nested_output_pipeline_depth_from_env() if materialization else None,
        },
    }


def nested_identity(config, bundles, run_start):
    """Bind inputs, output stream, model source, configuration and AOT policy.

    The forecast end (``hours``) is deliberately NOT part of the identity: it is
    recorded in the verified driver state, so a resumed run can be extended under
    explicit control (nested_pipeline._restart_end_time) in the same stream.
    """
    from gpuwrf.runtime.aot_cheap_key import canonical_digest, global_trace_env_hash, version_fingerprint_hash
    from gpuwrf.runtime.restart_parameters import static_namelist
    from gpuwrf.runtime.runtime_data_identity import runtime_data_identity
    source = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_hash.update(str(path.relative_to(source)).encode())
        source_hash.update(path.read_bytes())
    inputs = {}
    for pattern in ("wrfinput*", "wrfbdy*", "namelist*", "gpuwrf.streams.json", "windturbines*.txt", "wind-turbine-*.tbl"):
        for path in sorted(Path(config.input_dir).glob(pattern)):
            if path.is_file():
                inputs[path.name] = _hash_file(path)
    from gpuwrf.physics.rrtmg_mp_re import mp_re_config
    from gpuwrf.io.topographic_shadow_schema import configuration as shadow_configuration
    from gpuwrf.diagnostics.solar_config import configuration as solar_configuration
    from gpuwrf.io.pbl_epsilon_schema import epsilon_pbl_config
    from gpuwrf.io.noahmp_options import restart_configuration
    from gpuwrf.io.noahmp_options import irrigation_configuration
    from gpuwrf.io.windfarm_schema import configuration as windfarm_configuration
    from gpuwrf.io.urban_schema import configuration as urban_configuration
    urban_sources={}
    for name,bundle in bundles.items():
        source=getattr(getattr(bundle.namelist,"urban_bundle",None),"_hourly_source",None)
        if source is not None:
            path,digest=source
            if _hash_file(Path(path))!=digest:
                raise ValueError("hourly urban AHE changed after initialization")
            urban_sources[name]={"path":path,"sha256":digest}
    from gpuwrf.tracers.restart import configuration as tracer_configuration
    from gpuwrf.io.noahmp_mmf import configuration as mmf_configuration
    from gpuwrf.assimilation.fdda_surface import SurfaceGridFddaInputs
    identity = {
        "source_sha256": source_hash.hexdigest(), "inputs": inputs,
        "input_dir": str(Path(config.input_dir).resolve()),
        "output_dir": str(Path(config.output_dir).resolve()),
        "run_start": run_start.isoformat(),
        "output_policy": _nested_output_policy(config, bundles),
        "runtime_data": runtime_data_identity(bundles),
        "feedback": config.feedback, "emit_initial_history": config.emit_initial_history,
        "namelists": {name: canonical_digest(static_namelist(bundle.namelist)) for name, bundle in bundles.items()},
        "online_diagnostics_config": {
            name: _diagnostic_restart_config(bundle.namelist) for name, bundle in bundles.items()
        },
        "solar_interpolation_config": {
            name: solar_configuration(bundle.namelist) for name, bundle in bundles.items()
        },
        "noahmp_mmf_config": {name: mmf_configuration(bundle.namelist) for name, bundle in bundles.items()},
        "sst_skin_config": {name: int(bundle.namelist.sst_skin) for name, bundle in bundles.items()},
        "noahmp_irrigation_config": {name:irrigation_configuration(bundle.namelist) for name,bundle in bundles.items()},
        "noahmp_options_config": {name:restart_configuration(bundle.namelist) for name,bundle in bundles.items()},
        "mp_re_config": {name: mp_re_config(bundle.namelist) for name, bundle in bundles.items()},
        "topographic_shadow_config": {
            name: shadow_configuration(bundle.namelist)
            for name, bundle in bundles.items()},
        "temf_config": {
            name: {
                "bl_pbl_physics": int(bundle.namelist.bl_pbl_physics),
                "sf_sfclay_physics": int(bundle.namelist.sf_sfclay_physics),
                "domain_shape": [int(bundle.namelist.grid.nz), int(bundle.namelist.grid.ny), int(bundle.namelist.grid.nx)],
            }
            for name, bundle in bundles.items()
        },
        "epsilon_pbl_config": {
            name: {
                **epsilon_pbl_config(bundle.namelist),
                "sf_sfclay_physics": int(bundle.namelist.sf_sfclay_physics),
                "domain_shape": [int(bundle.namelist.grid.nz), int(bundle.namelist.grid.ny), int(bundle.namelist.grid.nx)],
            }
            for name, bundle in bundles.items()
        },
        "aot_environment": version_fingerprint_hash(), "trace_environment": global_trace_env_hash(),
        "grid_fdda_config": {name: _grid_fdda_restart_config(bundle.namelist) for name, bundle in bundles.items()},
        "windfarm_config": {name: windfarm_configuration(bundle.namelist) for name,bundle in bundles.items()},
        "urban_inputs_config": {
            name: urban_configuration(bundle.namelist) for name,bundle in bundles.items()
        },
        "urban_bep_config": {name: _urban_bep_restart_config(bundle.namelist) for name, bundle in bundles.items()},
        **({"urban_ahe_sources":urban_sources} if urban_sources else {}),
        "tracer_config": {name: tracer_configuration(bundle.namelist) for name, bundle in bundles.items()},
        "ssib_active": {name: getattr(bundle.namelist,'sf_surface_physics',None)==8 for name,bundle in bundles.items()},
        "surface_fdda_config": {
            name: {'records':len(bundle.namelist.data_assimilation.surface_steps),
                   'shape':[bundle.namelist.grid.nz,bundle.namelist.grid.ny,bundle.namelist.grid.nx]}
            if isinstance(getattr(bundle.namelist,'data_assimilation',None),SurfaceGridFddaInputs) else None
            for name,bundle in bundles.items()
        },
    }
    if any(int(bundle.namelist.bl_pbl_physics) in (2,4) for bundle in bundles.values()):
        identity['qnse_config']={name:{key:int(getattr(bundle.namelist,key)) for key in
            ('bl_pbl_physics','sf_sfclay_physics','mfshconv')} for name,bundle in bundles.items()}
    identity["sas_config"] = {
        name: {"cu_physics": int(bundle.namelist.cu_physics),
               "osas_domain_id": int(bundle.namelist.osas_domain_id),
               "domain_shape": [int(bundle.namelist.grid.nz),int(bundle.namelist.grid.ny),int(bundle.namelist.grid.nx)]}
        for name,bundle in bundles.items()
    }
    identity["glacier_config"] = {
        name: {"required": bool(getattr(getattr(bundle.namelist,'noahmp_static',None),'glacier',None) is not None),
               "shape": [int(bundle.namelist.grid.ny),int(bundle.namelist.grid.nx)]}
        for name,bundle in bundles.items()}
    from gpuwrf.surface_params.controls import restart_controls_digest
    surface_controls = {name: restart_controls_digest(bundle.namelist)
                        for name, bundle in bundles.items()}
    if any(digest is not None for digest in surface_controls.values()):
        # Track P: the traced parameter VALUES are not in the (content-blind) namelist digest.
        identity["surface_controls"] = surface_controls
    from gpuwrf.runtime.restart_parameters import parameter_digests
    parameter_values = {name: parameter_digests(bundle.namelist) for name, bundle in bundles.items()}
    for family in sorted({family for values in parameter_values.values() for family in values}):
        identity[family] = {name: values.get(family) for name, values in parameter_values.items()}
    from gpuwrf.io.mynn_budget_options import configuration as mynn_configuration
    mynn_configs={name:mynn_configuration(bundle.namelist) for name,bundle in bundles.items()}
    if any(config!=mynn_configuration({}) for config in mynn_configs.values()):
        identity['mynn_budget_config']=mynn_configs
    # Manifests and history-journal owners use JSON. Canonicalize here so
    # resolved tuples (irrigation, BATS tables) have the same list form before
    # publication and after readback; keep all reader comparisons exact.
    if any(getattr(bundle.namelist, "stochastic_config", None) is not None for bundle in bundles.values()):
        from gpuwrf.io.stochastic_config import restart_controls
        identity["stochastic_config"] = {name: restart_controls(bundle.namelist.stochastic_config)
                                         for name, bundle in bundles.items()}
    if any(getattr(bundle.namelist, "rad_diag", 0) for bundle in bundles.values()):
        identity["rad_diag_config"] = {name: int(bundle.namelist.rad_diag) for name, bundle in bundles.items()}
    return json.loads(json.dumps(identity, allow_nan=False))


def _validate_qnse_restart_groups(carries,identity,*,serialized=False):
    from gpuwrf.io.restart_schema import QNSE_SCHEMA_VERSION,validate_qnse
    configurations=identity.get('qnse_config',{})
    for name,carry in carries.items():
        group=carry.get('extra_fields',{}).get('qnse') if serialized else getattr(carry,'qnse',None)
        version=carry.get('qnse_schema_version') if serialized else QNSE_SCHEMA_VERSION
        history=carry.get('extra_fields',{}).get('history_diagnostics') if serialized else getattr(carry,'history_diagnostics',None)
        shape=carry['state_fields']['theta'].shape if serialized else carry.state.theta.shape
        # TEMF's existing all-domain binding also authenticates MYJSFC; the
        # QNSE-specific configuration is only emitted when QNSE is selected.
        surface_config=configurations.get(name,identity.get('temf_config',{}).get(name,{}))
        validate_qnse(group,surface_config,version,history=history,expected_shape=shape)


def drain_checkpoint_output(output_pipeline, async_writer):
    """Drain existing bounded queues without closing their worker threads."""
    for worker in (output_pipeline, async_writer):
        if worker is not None:
            worker._queue.join()
            worker._raise_if_failed()


def output_receipts(written):
    receipts = {}
    parents = set()
    for name, paths in written.items():
        entries = []
        for path in paths:
            path = Path(path).resolve()
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            parents.add(path.parent)
            entries.append({"path": str(path), "sha256": _hash_file(path)})
        receipts[name] = entries
    for parent in parents:
        _sync_dir(parent)
    return receipts


def writer_census_snapshot(writer):
    ledger = getattr(writer, "census_io_ledger", None)
    return None if ledger is None else ledger.snapshot()


def restore_writer_census(writer, counts):
    ledger = getattr(writer, "census_io_ledger", None)
    if counts is not None:
        if ledger is None:
            raise ValueError("restart writer census ledger is unavailable")
        with ledger._lock:
            ledger._counts = counts


def restore_output_receipts(writer, receipts):
    for name, entries in receipts.items():
        if name not in writer.written:
            raise ValueError("restart output domain differs from writer")
        for entry in entries:
            path = Path(entry["path"])
            if not path.is_file() or _hash_file(path) != entry["sha256"]:
                raise ValueError("restart committed history frame is missing or corrupt")
        writer.written[name] = [entry["path"] for entry in entries]


class HistoryJournal:
    """Authenticate published frames independently of checkpoint publication.

    A durable intent records the completed temporary file's hash AND inode
    before the writer hard-links it. Thus a crash on either side of that link
    remains distinguishable from an unrelated file at the same target name.
    Replay removes only verified, uncommitted frames; checkpoint frames stay.
    """

    def __init__(self, store, identity, stream=None):
        self.store = store
        self.identity = identity
        self.output_dir = Path(identity["output_dir"]).resolve()
        self.stream = stream or uuid.uuid4().hex
        if len(self.stream) != 32 or any(c not in "0123456789abcdef" for c in self.stream):
            raise ValueError("invalid restart history stream")
        self.root = store.root / ("history-" + self.stream)
        owner = {"stream": self.stream, "identity": identity}
        if stream is None:
            self.root.mkdir(parents=True, exist_ok=False)
            self._write(self.root / "owner.json", owner)
            _sync_dir(store.root)
        else:
            owner_path = self.root / "owner.json"
            try:
                saved_owner = json.loads(owner_path.read_text())
            except FileNotFoundError as exc:
                raise ValueError(
                    f"missing restart history journal owner: {owner_path}. Ordinary resume continues "
                    "the parent journal and needs the parent's --checkpoint-dir, including "
                    f"history-{self.stream}/; use --branch-from for a new checkpoint directory."
                ) from exc
            if saved_owner != owner:
                raise ValueError("restart history stream identity mismatch")

    def _write(self, path, value):
        data = (json.dumps(value, sort_keys=True) + "\n").encode()
        if _storage_bytes(self.store.root) + len(data) > self.store.max_bytes:
            raise OSError("history journal exceeds checkpoint byte budget")
        if shutil.disk_usage(self.store.root).free < self.store.reserve_bytes + len(data):
            raise OSError("history journal would violate filesystem reserve")
        temporary = path.with_name(".intent-" + uuid.uuid4().hex)
        with temporary.open("xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)

    def _target(self, path):
        path = Path(path)
        if path.parent.resolve() != self.output_dir or not path.name.startswith("wrfout_"):
            raise ValueError("restart history target outside bound output stream")
        return self.output_dir / path.name

    def _signature(self, entry):
        return hmac.new(self.stream.encode(), json.dumps(entry, sort_keys=True).encode(), hashlib.sha256).hexdigest()

    def publish(self, temporary, target):
        from gpuwrf.io.wrfout_writer import _publish_wrfout_noreplace
        target = self._target(target)
        temporary = Path(temporary).parent.resolve() / Path(temporary).name
        if os.path.lexists(target):
            raise FileExistsError(f"WRFOUT_TARGET_EXISTS:{target}")
        info = Path(temporary).lstat()
        entry = {"path": str(target), "sha256": _hash_file(temporary),
                 "device": info.st_dev, "inode": info.st_ino}
        record = {"entry": entry, "signature": self._signature(entry)}
        self._write(self.root / (hashlib.sha256(str(target).encode()).hexdigest() + ".json"), record)
        _publish_wrfout_noreplace(temporary, target)

    def recover(self, committed):
        protected = {str(self._target(e["path"])) for entries in committed.values() for e in entries}
        pending = []
        # Validate EVERY later file before changing any directory entry.
        for receipt in sorted(self.root.glob("*.json")):
            if receipt.name == "owner.json":
                continue
            record = json.loads(receipt.read_text())
            entry = record["entry"]
            if not hmac.compare_digest(record["signature"], self._signature(entry)):
                raise ValueError("restart history receipt authentication failed")
            path = self._target(entry["path"])
            if str(path) in protected or not os.path.lexists(path):
                continue
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_dev != entry["device"]
                    or info.st_ino != entry["inode"] or _hash_file(path) != entry["sha256"]):
                raise ValueError("restart later history frame is foreign or corrupt")
            pending.append((path, info.st_dev, info.st_ino))
        removed = []
        for path, device, inode in pending:
            info = path.lstat()
            if (info.st_dev, info.st_ino) != (device, inode):
                raise ValueError("restart history changed during recovery")
            path.unlink()
            removed.append(str(path))
        if removed:
            _sync_dir(self.output_dir)
        return removed


def require_resume_history(store, generation):
    """Refuse generations copied without their journal before model initialization.

    The stream lives in the payload, not its manifest. Healthy stores need only
    a metadata lookup here; their authenticated payload read stays in the normal
    resume path. Only stores with no owner metadata read the verified host
    snapshot early to name the missing owner precisely, without device restore.
    HistoryJournal still checks the exact owner/identity at ordinary restoration.
    """
    if any(store.root.glob("history-*/owner.json")):
        return
    snapshot, manifest = store.read(generation, device=False)
    stream = snapshot["driver_state"].get("history_stream")
    if stream is not None:
        HistoryJournal(store, manifest["identity"], stream)


def _validate_snapshot(snapshot, manifest) -> None:
    names = tuple(manifest["domains"])
    if not names or len(names) != len(set(names)):
        raise ValueError("restart domain manifest is empty or duplicated")
    if set(snapshot["carries"]) != set(names) or set(snapshot["own_steps"]) != set(names):
        raise ValueError("restart contains missing or mixed domains")
    if snapshot["own_steps"] != manifest["own_steps"]:
        raise ValueError("restart timestep receipt does not match payload")
    elapsed = [snapshot["own_steps"][name] * manifest["dt_s"][name] for name in names]
    if any(step < 0 or int(step) != step for step in snapshot["own_steps"].values()):
        raise ValueError("restart has invalid own-step counters")
    if not np.allclose(elapsed, elapsed[0], rtol=0, atol=1e-8):
        raise ValueError("restart domains are not synchronized")
    from dataclasses import fields
    from gpuwrf.runtime.operational_state import OperationalCarry
    from gpuwrf.io.restart import (_CARRY_SCRATCH_FIELDS, _V1_UNSUPPORTED_CARRY_FIELDS,
                                  _validate_state_field_order, _validate_radiation_diagnostics_schema,
                                  _validate_gwdo_diagnostics_schema)
    from gpuwrf.io.restart_schema import validate_temf
    from types import SimpleNamespace
    temf_config = manifest["identity"].get("temf_config")
    if temf_config is not None or "namelists" in manifest["identity"]:
        if not isinstance(temf_config, dict) or set(temf_config) != set(names):
            raise ValueError("restart TEMF domain/selection binding missing (E78)")
    epsilon_config = manifest["identity"].get("epsilon_pbl_config")
    if epsilon_config is not None or "namelists" in manifest["identity"]:
        if not isinstance(epsilon_config, dict) or set(epsilon_config) != set(names):
            raise ValueError("restart EEPS/KEPS domain/selection binding missing (E78)")
    sas_config = manifest["identity"].get("sas_config")
    if sas_config is not None or "namelists" in manifest["identity"]:
        if not isinstance(sas_config, dict) or set(sas_config) != set(names):
            raise ValueError("restart SAS domain/selection binding missing (E78)")
    expected = tuple(f.name for f in fields(OperationalCarry))
    configs = manifest["identity"].get("online_diagnostics_config")
    solar_configs = manifest["identity"].get("solar_interpolation_config")
    surface_fdda_config = manifest['identity'].get('surface_fdda_config')
    if surface_fdda_config is None and 'namelists' in manifest['identity']:
        raise ValueError('restart lacks per-domain surface FDDA configuration (E78)')
    if surface_fdda_config is not None and (not isinstance(surface_fdda_config,dict)
            or set(surface_fdda_config)!=set(names)
            or any(v is not None and (not isinstance(v,dict) or set(v)!= {'records','shape'}
                or type(v['records']) is not int or v['records']<1) for v in surface_fdda_config.values())):
        raise ValueError('restart surface FDDA domain/selection binding invalid (E78)')
    grid_fdda_config = manifest["identity"].get("grid_fdda_config")
    if grid_fdda_config is not None and (not isinstance(grid_fdda_config, Mapping) or set(grid_fdda_config) != set(names)
                                        or any(type(v) is not bool for v in grid_fdda_config.values())):
        raise ValueError("restart grid FDDA selection binding differs (E78)")
    windfarm_configs = manifest['identity'].get('windfarm_config')
    if windfarm_configs is None and 'namelists' in manifest['identity']:
        raise ValueError('restart lacks per-domain Fitch configuration (E78)')
    if windfarm_configs is not None and (not isinstance(windfarm_configs,Mapping) or set(windfarm_configs)!=set(names)):
        raise ValueError('restart Fitch configuration domains differ (E78)')
    urban_configs = manifest["identity"].get("urban_inputs_config")
    if urban_configs is None and "namelists" in manifest["identity"]:
        raise ValueError("restart lacks per-domain urban input configuration (E78)")
    if urban_configs is not None and (not isinstance(urban_configs,Mapping) or set(urban_configs)!=set(names)):
        raise ValueError("restart urban input configuration domains differ (E78)")
    urban_bep_configs = manifest["identity"].get("urban_bep_config")
    tracer_configs = manifest["identity"].get("tracer_config")
    if tracer_configs is None and manifest.get("tracer_config_required"):
        raise ValueError("tracer restart lacks per-domain configuration")
    if tracer_configs is not None and (not isinstance(tracer_configs, Mapping) or set(tracer_configs) != set(names)):
        raise ValueError("tracer restart configuration domains differ")
    skin_configs = manifest["identity"].get("sst_skin_config")
    irrigation_configs=manifest["identity"].get("noahmp_irrigation_config")
    noahmp_configs=manifest["identity"].get("noahmp_options_config")
    if noahmp_configs is not None and (not isinstance(noahmp_configs,Mapping) or set(noahmp_configs)!=set(names)):
        raise ValueError("restart Noah-MP option configuration domains differ (E78)")
    # Digests cannot recover selectors. Native identities must carry the
    # explicit configuration even when an active group was deleted entirely.
    if configs is None and "namelists" in manifest["identity"]:
        raise ValueError("restart lacks per-domain online diagnostic configuration (E78)")
    if configs is not None and (not isinstance(configs, Mapping) or set(configs) != set(names)):
        raise ValueError("restart online diagnostic configuration domains differ (E78)")
    if solar_configs is None and "namelists" in manifest["identity"]:
        raise ValueError("restart lacks per-domain Solar configuration (E78)")
    if solar_configs is not None and (not isinstance(solar_configs, Mapping) or set(solar_configs) != set(names)):
        raise ValueError("restart Solar configuration domains differ (E78)")
    ssib_selection=manifest['identity'].get('ssib_active')
    if 'namelists' in manifest['identity'] and ssib_selection is None:
        raise ValueError('SSiB restart selection binding missing (E78)')
    if ssib_selection is not None and (not isinstance(ssib_selection,dict) or set(ssib_selection)!=set(names)
            or any(type(value) is not bool for value in ssib_selection.values())):
        raise ValueError('SSiB restart domain selection mismatch (E78)')
    for name, payload in snapshot["carries"].items():
        from gpuwrf.diagnostics.mynn_budget import validate_payload as validate_mynn_payload
        mynn_configs=manifest['identity'].get('mynn_budget_config')
        if mynn_configs is not None and (not isinstance(mynn_configs,dict) or set(mynn_configs)!=set(names)):
            raise ValueError('restart MYNN budget domain selection differs (E78)')
        requested={} if mynn_configs is None else mynn_configs[name]
        validate_mynn_payload(payload,requested)
        from gpuwrf.io.glacier_schema import validate as validate_glacier
        glacier_configs=manifest['identity'].get('glacier_config')
        glacier_config={} if glacier_configs is None else glacier_configs.get(name,{})
        if glacier_configs is not None and (set(glacier_configs)!=set(names) or set(glacier_config)!={'required','shape'}):
            raise ValueError('restart glacier domain/selection binding differs (E78)')
        validate_glacier(payload.get('noahmp_land_fields'),payload.get('noahmp_land_field_order') or (),
            payload.get('glacier_schema_version'),active=glacier_config.get('required',False),shape=glacier_config.get('shape'))
        from gpuwrf.io.feedback_restart import validate_payload
        feedback_configs=manifest['identity'].get('feedback_schema')
        if feedback_configs is not None and (
                not isinstance(feedback_configs,dict) or set(feedback_configs)!=set(names)):
            raise ValueError('two-way restart domain/selection schema binding differs (E78)')
        feedback_required=(feedback_configs is not None or
                           (len(names)>1 and manifest['identity'].get('feedback') is True))
        selected=manifest['identity'].get('feedback')
        if ((selected is False or (selected is True and len(names)==1))
                and (feedback_configs is not None or payload.get('feedback_schema') is not None)):
            raise ValueError('two-way restart active groups disagree with effective feedback selection (E78)')
        validate_payload(payload,required=feedback_required,
                         expected=None if feedback_configs is None else feedback_configs[name])
        from gpuwrf.io.windfarm_schema import validate_restart as validate_windfarm
        validate_windfarm(payload['extra_fields'].get('windfarm_power'),None if windfarm_configs is None else windfarm_configs[name],
                          payload['state_fields']['t_skin'].shape)
        from gpuwrf.io.urban_schema import validate_urban_inputs
        validate_urban_inputs(payload["extra_fields"].get("urban_slucm"),
            None if urban_configs is None else urban_configs[name],
            payload.get("urban_inputs_schema_version"),payload["state_fields"]["t_skin"].shape)
        _validate_urban_bep_snapshot(payload, urban_bep_configs, names, name, "namelists" in manifest["identity"])
        from gpuwrf.tracers.restart import validate_restart as validate_tracer_restart
        validate_tracer_restart(payload["state_fields"], payload["extra_fields"].get("tracer_tendencies"),
            None if tracer_configs is None else tracer_configs[name], payload.get("tracer_schema_version"),
            payload["state_fields"]["qv"].shape)
        from gpuwrf.assimilation.fdda_surface import validate_surface_restart,SurfaceGridFddaCarry
        surface_group=payload['extra_fields'].get('fdda')
        if isinstance(surface_group,SurfaceGridFddaCarry) and surface_fdda_config is None:
            raise ValueError('restart surface FDDA selection binding missing (E78)')
        validate_surface_restart(surface_group,None,
            required=surface_fdda_config is not None and surface_fdda_config[name] is not None)
        if isinstance(surface_group,SurfaceGridFddaCarry):
            config=surface_fdda_config[name]
            if config is None or tuple(config['shape'])!=np.shape(payload['state_fields']['theta']):
                raise ValueError('restart surface FDDA bound domain/selection differs (E78)')
            if np.shape(surface_group.t_old)!=tuple(config['shape']) or int(surface_group.surface_cursor)>=config['records']:
                raise ValueError('restart surface FDDA domain/cursor differs from binding (E78)')
        from gpuwrf.io.noahmp_carbon import validate_restart as validate_carbon
        from gpuwrf.io.noahmp_options import NoahMPOptions,validate_option_restart
        from types import SimpleNamespace
        cfg=None if noahmp_configs is None else noahmp_configs[name]
        validate_option_restart(cfg)
        saved_namelist=SimpleNamespace(noahmp_options=None if cfg is None else NoahMPOptions(**cfg))
        validate_carbon(payload["extra_fields"].get("noahmp_carbon"),saved_namelist,
                        payload.get("noahmp_carbon_schema_version"),payload["state_fields"]["t_skin"].shape)
        from gpuwrf.io.noahmp_groundwater import validate_restart as validate_groundwater
        validate_groundwater(payload["extra_fields"].get("noahmp_groundwater"),saved_namelist,
                            payload.get("noahmp_groundwater_schema_version"),payload["state_fields"]["t_skin"].shape)
        from gpuwrf.io.noahmp_hydrology import validate_tile_restart
        validate_tile_restart(payload['extra_fields'].get('noahmp_tile_drain'),saved_namelist,
            payload.get('noahmp_tile_drain_schema_version'),payload['state_fields']['t_skin'].shape)
        from gpuwrf.coupling.sst_skin_hook import validate_restart as validate_skin
        skin = payload["extra_fields"].get("sst_skin")
        if skin_configs is not None and (not isinstance(skin_configs, Mapping) or set(skin_configs) != set(names)):
            raise ValueError("restart SST skin configuration domains differ (E78)")
        validate_skin(skin, 0 if skin_configs is None else skin_configs[name], payload.get("sst_skin_schema_version"),
                      payload["state_fields"]["t_skin"].shape)
        from gpuwrf.io.noahmp_irrigation import validate_restart as validate_irrigation
        if irrigation_configs is not None and (not isinstance(irrigation_configs,Mapping) or set(irrigation_configs)!=set(names)):
            raise ValueError("restart irrigation configuration domains differ (E78)")
        validate_irrigation(payload["extra_fields"].get("noahmp_irrigation"),
            (0,0) if irrigation_configs is None else irrigation_configs[name],payload.get("noahmp_irrigation_schema_version"),
            payload["state_fields"]["t_skin"].shape)
        _validate_diagnostic_snapshot(payload, None if configs is None else configs[name])
        from gpuwrf.io.restart_schema import validate_ssib
        extra=payload['extra_fields']
        validate_ssib(extra.get('ssib_land'),extra.get('ssib_rad'),None,
            payload.get('ssib_schema_version'),payload['state_fields']['t_skin'].shape,
            active=False if ssib_selection is None else ssib_selection[name],
            driver_fields=dict(wspd=payload["state_fields"].get("sfc_wspd"),history=extra.get("history_diagnostics"),precipitation=extra.get("noahmp_precipitation")))
        from gpuwrf.diagnostics.solar_config import validate_restart as validate_solar_restart
        validate_solar_restart(payload["extra_fields"].get("solar_interpolation"),
            None if solar_configs is None else solar_configs[name],
            payload.get("solar_interpolation_schema_version"), payload["state_fields"]["t_skin"].shape)
        # Existing authenticated radius selection already carries both RA
        # selectors. It also detects loss of the entire Goddard group/marker.
        rad_configs = manifest['identity'].get('rad_diag_config')
        if rad_configs is not None and (not isinstance(rad_configs, Mapping) or set(rad_configs) != set(names)):
            raise ValueError('restart rad_diag configuration domains differ (E78)')
        saved_rad_config = manifest['identity'].get('mp_re_config', {}).get(name)
        # A genuinely absent legacy selector must stay None: Goddard/FLG/GFDL
        # infer their saved selection from their own authenticated schema.
        # Passing {} would instead invent the default RRTMG4/4 selection.
        rad_config = None if saved_rad_config is None else dict(saved_rad_config)
        if rad_configs is not None:
            if rad_config is None:
                rad_config = {}
            rad_config['rad_diag'] = rad_configs[name]
        elif getattr(payload['extra_fields'].get('radiation_diagnostics'),'rad_diag',None) is not None:
            raise ValueError('restart rad_diag configuration is missing (E78)')
        _validate_radiation_diagnostics_schema(payload, namelist=rad_config,
            topographic_shadow_required=manifest['identity'].get('topographic_shadow_config',{}).get(name))
        _validate_gwdo_diagnostics_schema(payload)
        from gpuwrf.io.noahmp_mmf import validate_restart as validate_mmf
        mmf_configs = manifest['identity'].get('noahmp_mmf_config')
        if mmf_configs is None and payload['extra_fields'].get('noahmp_mmf') is not None:
            raise ValueError('MMF restart lacks authenticated domain selection (E78)')
        if mmf_configs is not None and (not isinstance(mmf_configs, Mapping) or set(mmf_configs) != set(names)):
            raise ValueError('MMF restart domain selection differs (E78)')
        validate_mmf(payload['extra_fields'].get('noahmp_mmf'),
                     None if mmf_configs is None else mmf_configs[name],
                     payload.get('noahmp_mmf_schema_version'), payload['state_fields']['t_skin'].shape)
        from gpuwrf.io.noahmp_mmf import validate_land as validate_mmf_land
        validate_mmf_land(payload['extra_fields'].get('noahmp_mmf'),payload.get('noahmp_land_fields'),
                          payload['state_fields']['t_skin'].shape)
        recorded = tuple(payload.get('carry_field_order', ()))
        compatible = tuple(n for n in expected if n not in ('noahmp_mmf','mynn_budget','afwa_diagnostics') or n in recorded)
        if recorded != compatible:
            raise ValueError("restart carry schema differs from executable")
        order = tuple(payload["state_field_order"])
        _validate_state_field_order(order, namelist=manifest["identity"].get("mp_re_config", {}).get(name))
        if set(payload["state_fields"]) != set(order):
            raise ValueError("restart State schema differs from receipt")
        mp_config=manifest["identity"].get("mp_re_config", {}).get(name)
        from gpuwrf.io.restart_schema import physics_option,validate_held_state
        if physics_option(mp_config,'mp_physics',8) in (9,50,51,52,53,55):
            validate_held_state(payload['state_fields'],mp_config)
        if tuple(payload["scratch_field_order"]) != _CARRY_SCRATCH_FIELDS or set(payload["scratch_fields"]) != set(_CARRY_SCRATCH_FIELDS):
            raise ValueError("restart scratch schema differs from executable")
        expected_extra = set(_V1_UNSUPPORTED_CARRY_FIELDS)
        if 'noahmp_mmf' not in recorded:
            expected_extra.remove('noahmp_mmf')
        if 'mynn_budget' not in recorded:expected_extra.remove('mynn_budget')
        if 'afwa_diagnostics' not in recorded:expected_extra.remove('afwa_diagnostics')
        if set(payload["extra_fields"]) != expected_extra:
            raise ValueError("restart optional carry schema differs from executable")
        stochastic_ids = manifest["identity"].get("stochastic_config")
        stochastic_group = payload["extra_fields"].get("stochastic")
        if stochastic_ids is None:
            if stochastic_group is not None:
                raise ValueError("restart stochastic carry lacks authenticated controls (E78)")
        else:
            if not isinstance(stochastic_ids, Mapping) or set(stochastic_ids) != set(names):
                raise ValueError("restart stochastic control domains differ (E78)")
            if (stochastic_ids[name] is None) != (stochastic_group is None):
                raise ValueError(f"{name}: restart stochastic carry presence differs from its controls (E78)")
            if stochastic_group is not None:
                seeds = tuple(getattr(stochastic_group, "seeds", ()) or ())
                patterns = tuple(getattr(stochastic_group, "patterns", ()) or ())
                active = tuple(p is not None for p in stochastic_ids[name]["patterns"])
                if len(patterns) != 7 or len(seeds) != 6 or tuple(p is not None for p in patterns) != active:
                    raise ValueError(f"{name}: restart stochastic pattern layout differs from its controls (E78)")
                from gpuwrf.io.stochastic_schema import validate_stochastic
                recorded = stochastic_ids[name]
                try:  # the recorded per-domain controls decide root vs child coefficient layout and vertical
                    view = SimpleNamespace(stochastic_config=SimpleNamespace(
                        root=bool(recorded["root"]), produce=tuple(bool(x) for x in recorded["produce"]),
                        patterns=tuple(None if p is None else SimpleNamespace(vertical=int(p["vertical"]))
                                       for p in recorded["patterns"])))
                except (KeyError, TypeError, ValueError):
                    raise ValueError(f"{name}: restart stochastic controls malformed (E78)") from None
                if len(view.stochastic_config.produce) != 7:
                    raise ValueError(f"{name}: restart stochastic controls malformed (E78)")
                try:  # seeds per active stream, ranks, dtypes, finiteness, counters, layout, vertical plans
                    validate_stochastic(stochastic_group, view, required=True)
                except ValueError as exc:
                    raise ValueError(f"{name}: {exc}") from None
        group = payload["extra_fields"].get("temf")
        if temf_config is None:
            # Legacy unbound default stores remain readable. An active group
            # needs explicit authenticated selection/domain configuration.
            if group is not None:
                raise ValueError("restart TEMF domain/selection binding missing (E78)")
            temf_namelist = SimpleNamespace(bl_pbl_physics=0)
        else:
            config = temf_config[name]
            if (not isinstance(config, dict)
                    or set(config) != {"bl_pbl_physics", "sf_sfclay_physics", "domain_shape"}):
                raise ValueError("restart TEMF domain/selection binding invalid (E78)")
            shape = config["domain_shape"]
            if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or tuple(shape) != np.shape(payload["state_fields"]["theta"])):
                raise ValueError("restart TEMF bound domain shape differs (E78)")
            pbl, surface = config["bl_pbl_physics"], config["sf_sfclay_physics"]
            if (type(pbl) is not int or type(surface) is not int
                    or (pbl == 10) != (surface == 10) or (group is not None and pbl != 10)):
                raise ValueError("restart TEMF bound physics pair differs (E78)")
            nz, ny, nx = shape
            temf_namelist = SimpleNamespace(bl_pbl_physics=pbl, grid=SimpleNamespace(nz=nz, ny=ny, nx=nx))
        # _carry_to_payload already materialized these leaves on the host.
        # Use the same admission on save, verified readback, and public read,
        # before pickle publication or any _restore_carry device allocations.
        validate_temf(group, temf_namelist, payload.get("temf_schema_version"))
        _validate_epsilon_snapshot(payload, None if epsilon_config is None else epsilon_config[name])
        if sas_config is not None:
            from gpuwrf.io.restart_schema import validate_sas_held
            config=sas_config[name]
            if not isinstance(config,dict) or set(config)!={"cu_physics","osas_domain_id","domain_shape"}:
                raise ValueError("restart SAS selection binding invalid (E78)")
            shape=config['domain_shape']
            if tuple(shape)!=np.shape(payload['state_fields']['theta']):
                raise ValueError("restart SAS bound domain differs (E78)")
            nz,ny,nx=shape
            sas_namelist=SimpleNamespace(cu_physics=config['cu_physics'],osas_domain_id=config['osas_domain_id'],
                                        grid=SimpleNamespace(nz=nz,ny=ny,nx=nx))
            validate_sas_held(payload['extra_fields'].get('cumulus_tendencies'),sas_namelist,
                              payload['extra_fields'].get('osas_rng_key'), payload['extra_fields'].get('cumulus_carry'))
        from gpuwrf.io.cu_forcing_schema import validate as validate_cu_forcing
        # Legacy unbound stores stay readable; a populated group needs the bound selector (E78).
        validate_cu_forcing(payload['extra_fields'].get('cu_forcing'),
            {'cu_physics':0 if sas_config is None else sas_config[name]['cu_physics']},
            np.shape(payload['state_fields']['theta']),payload.get('cu_forcing_schema_version'))
        # TEMF finiteness was checked with its complete shape/dtype contract.
        extra_fields = {key: value for key, value in payload["extra_fields"].items() if key != "temf"}
        trees = (payload["state_fields"], payload["scratch_fields"], extra_fields,
                 payload.get("noahmp_land_fields"), payload.get("noahmp_rad"))
        for leaf in jax.tree.leaves(trees):
            value = np.asarray(leaf)
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError("restart contains a nonfinite carry leaf")
        from gpuwrf.io.restart_schema import validate_grid_fdda
        validate_grid_fdda(payload["extra_fields"].get("fdda"),
                           {"fdda": {"grid_fdda": int(grid_fdda_config[name]) if grid_fdda_config else 0}})


def _validate_epsilon_snapshot(payload, config):
    """Authenticate active EEPS/KEPS State leaves before device restoration."""
    from gpuwrf.io.pbl_epsilon_schema import validate_epsilon_state,validate_epsilon_history
    if config is None:
        # Legacy default checkpoints are readable; populated epsilon leaves
        # require the explicit selection, including when every leaf was lost.
        config = {"bl_pbl_physics": 0}
    else:
        keys = {"bl_pbl_physics", "sf_sfclay_physics", "domain_shape", "boundary_config"}
        if not isinstance(config, dict) or set(config) != keys:
            raise ValueError("restart EEPS/KEPS domain/selection binding invalid (E78)")
        shape = config["domain_shape"]
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or any(type(n) is not int or n <= 0 for n in shape)
                or tuple(shape) != np.shape(payload["state_fields"]["qv"])):
            raise ValueError("restart EEPS/KEPS bound domain shape differs (E78)")
        bl, sf = config["bl_pbl_physics"], config["sf_sfclay_physics"]
        if type(bl) is not int or type(sf) is not int or (bl in (16, 17) and sf != 1):
            raise ValueError("restart EEPS/KEPS bound physics pair differs (E78)")
        boundary = config["boundary_config"]
        if (not isinstance(boundary, dict)
                or set(boundary) != {"nested_frozen_wrf_boundary_bundle", "have_bcs_scalar"}
                or type(boundary["nested_frozen_wrf_boundary_bundle"]) is not bool
                or (boundary["have_bcs_scalar"] is not None and type(boundary["have_bcs_scalar"]) is not bool)):
            raise ValueError("restart EEPS/KEPS boundary binding invalid (E78)")
    validate_epsilon_state(payload["state_fields"], config, payload.get("epsilon_pbl_schema_version"))
    validate_epsilon_history(payload['extra_fields'].get('history_diagnostics'),config,
                             payload['state_fields']['t_skin'].shape)


def _validate_diagnostic_snapshot(payload, config):
    """E78 before publication or device restoration, including absent groups."""
    from gpuwrf.diagnostics.online import validate_restart
    from gpuwrf.diagnostics.precipitation import validate_restart_buckets
    extra = payload["extra_fields"]
    buckets = extra.get("precipitation_buckets")
    nwp, radar = extra.get("nwp_diagnostics"), extra.get("radar_reflectivity")
    if config is None:
        if any(value is not None for value in (buckets, nwp, radar)):
            raise ValueError("restart lacks per-domain online diagnostic configuration (E78)")
        config = {"prec_acc_dt": 0., "nwp_diagnostics": 0, "do_radar_ref": 0,
                  "history_interval": 60.}
    required = {"prec_acc_dt", "nwp_diagnostics", "do_radar_ref", "history_interval"}
    if not isinstance(config, dict) or set(config) - {"afwa_severe_opt"} != required:
        raise ValueError("restart online diagnostic configuration is incomplete (E78)")
    from gpuwrf.diagnostics.afwa import validate_restart_afwa
    afwa = config.get("afwa_severe_opt", 0)
    if afwa not in (0, 1) or ("afwa_severe_opt" in config and afwa != 1):
        raise ValueError("restart AFWA selection is invalid (E78)")
    validate_restart_afwa(payload["extra_fields"].get("afwa_diagnostics"), afwa,
                          payload.get("afwa_diagnostics_schema_version"),
                          payload["state_fields"]["t_skin"].shape,
                          history=payload["extra_fields"].get("history_diagnostics"), check_history=True)
    if (config["nwp_diagnostics"] not in (0, 1) or config["do_radar_ref"] not in (0, 1)
            or not math.isfinite(config["prec_acc_dt"]) or config["prec_acc_dt"] < 0
            or not math.isfinite(config["history_interval"]) or config["history_interval"] <= 0):
        raise ValueError("restart online diagnostic configuration is invalid (E78)")
    if (buckets is not None) != (config["prec_acc_dt"] > 0):
        raise ValueError("restart precipitation carry and selection disagree (E78)")
    state = payload["state_fields"]
    validate_restart_buckets(buckets, config["prec_acc_dt"],
                             payload.get("precipitation_buckets_schema_version"), state["t_skin"].shape)
    validate_restart(nwp, radar, config, payload.get("online_diagnostics_schema_version"), state["qv"].shape)


@dataclass(frozen=True)
class RestartStore:
    root: Path
    max_bytes: int
    max_generations: int
    reserve_bytes: int

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))
        if self.max_bytes <= 0 or self.max_generations < 1 or self.reserve_bytes < 0:
            raise ValueError("restart requires positive byte/count budgets and a nonnegative reserve")

    def read(self, generation: Path, *, expected_identity=None, device=True):
        generation = Path(generation)
        manifest = json.loads((generation / "manifest.json").read_text())
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported synchronized restart format")
        if expected_identity is not None:
            from gpuwrf.io.znt_history import require_same_policy
            require_same_policy(manifest['identity'], expected_identity)
        if expected_identity is not None and manifest["identity"] != expected_identity:
            raise ValueError("restart run/configuration/source identity mismatch")
        payload_path = generation / "snapshot.pkl"
        if payload_path.stat().st_size != manifest["payload_bytes"] or _hash_file(payload_path) != manifest["sha256"]:
            raise ValueError("restart payload hash/size verification failed")
        with payload_path.open("rb") as stream:
            snapshot = pickle.load(stream)
        _validate_snapshot(snapshot, manifest)
        _validate_qnse_restart_groups(snapshot['carries'],manifest['identity'],serialized=True)
        if device:
            snapshot["carries"] = {name: _restore_carry(value) for name, value in snapshot["carries"].items()}
        return snapshot, manifest

    def verified(self):
        result = []
        for path in self.root.glob("generation-*"):
            if not path.is_dir():
                continue
            try:
                _, manifest = self.read(path, device=False)
            except (OSError, ValueError, KeyError, EOFError, TypeError, AttributeError, pickle.UnpicklingError):
                continue
            result.append((manifest["created_ns"], path, manifest))
        return sorted(result)

    def latest(self, *, expected_identity=None):
        for _, path, manifest in reversed(self.verified()):
            if expected_identity is None or manifest["identity"] == expected_identity:
                return path
        raise ValueError("no verified restart generation for this run")

    def save(self, carries, own_steps, dt_s, identity, *, driver_state=None):
        _validate_qnse_restart_groups(carries,identity)
        self.root.mkdir(parents=True, exist_ok=True)
        if set(carries) != set(own_steps) or set(carries) != set(dt_s):
            raise ValueError("checkpoint domains/counters/timesteps do not match")
        snapshot = {
            "carries": {name: _carry_to_payload(carry,mynn_config=identity.get('mynn_budget_config',{}).get(name))
                        for name, carry in carries.items()},
            "own_steps": dict(own_steps),
            "driver_state": _host(driver_state or {}),
        }
        manifest = {
            "format": FORMAT, "identity": identity, "domains": list(carries),
            "branch_schema_version": 1,
            "own_steps": dict(own_steps), "dt_s": dict(dt_s),
            "created_ns": time.time_ns(),
            "tracer_config_required": "tracer_config" in identity,
        }
        _validate_snapshot(snapshot, manifest)
        # Serialize once: exact size is available before writing, including all
        # driver metadata. Limit peak host memory to one serialized checkpoint.
        data = pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.update(payload_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        receipt = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        required = len(data) + len(receipt)
        previous = self.verified()
        if any(m["identity"] != identity for _, _, m in previous):
            raise ValueError("checkpoint directory belongs to a different run/source identity")
        verified_paths = {p for _, p, _ in previous}
        evidence_bytes = sum(
            entry.stat().st_size if entry.is_file() else _storage_bytes(entry)
            for entry in self.root.iterdir() if entry not in verified_paths
        )
        evidence_count = sum(1 for p in self.root.glob("generation-*") if p not in verified_paths)
        # The newest verified generation is never removed before its successor
        # passes readback. Older generations already have that verified successor
        # and may be rotated first to keep publication itself within the budgets.
        retained = list(previous)
        def exceeds_budget():
            return (len(retained) + evidence_count + 1 > self.max_generations
                    or sum(_storage_bytes(p) for _, p, _ in retained) + evidence_bytes + required > self.max_bytes)
        while len(retained) > 1 and exceeds_budget():
            retained.pop(0)
        if exceeds_budget():
            raise OSError("checkpoint exceeds configured byte budget")
        keep = {p for _, p, _ in retained}
        reclaimable = sum(_storage_bytes(p) for _, p, _ in previous if p not in keep)
        if shutil.disk_usage(self.root).free + reclaimable < self.reserve_bytes + required:
            raise OSError("checkpoint would violate filesystem reserve")
        removed = []
        for _, path, _ in previous:
            if path not in keep:
                shutil.rmtree(path)  # exact verified child with an existing verified successor
                removed.append(str(path))
        _sync_dir(self.root)
        publication_peak_bytes = _storage_bytes(self.root) + required
        suffix = uuid.uuid4().hex
        temporary = self.root / (".partial-" + suffix)
        temporary.mkdir()
        for name, content in (("snapshot.pkl", data), ("manifest.json", receipt)):
            with (temporary / name).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        _sync_dir(temporary)
        self.read(temporary, expected_identity=identity, device=False)
        generation = self.root / ("generation-" + str(manifest["created_ns"]) + "-" + suffix)
        temporary.rename(generation)
        _sync_dir(self.root)
        # Verify the published name too; its newest predecessor is still retained.
        self.read(generation, expected_identity=identity, device=False)
        _sync_dir(self.root)
        return generation, {
            "bytes": required, "retained_count": len(retained) + evidence_count + 1,
            "retained_bytes": _storage_bytes(self.root), "evidence_bytes": evidence_bytes,
            "publication_peak_bytes": publication_peak_bytes,
            "removed": removed,
        }
