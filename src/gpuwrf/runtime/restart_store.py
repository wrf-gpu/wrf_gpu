"""Durable, bounded generations for synchronized operational restart.

Only the manifest of an atomically published generation makes it eligible.
Temporary and corrupt generations are evidence and are never auto-deleted.
Pickles are local runtime artifacts, verified before loading; not an import
format for files from other producers.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
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


def nested_identity(config, bundles, run_start):
    """Bind inputs, output stream, model source, configuration and AOT policy."""
    from gpuwrf.runtime.aot_cheap_key import canonical_digest, global_trace_env_hash, version_fingerprint_hash
    source = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        source_hash.update(str(path.relative_to(source)).encode())
        source_hash.update(path.read_bytes())
    inputs = {}
    for pattern in ("wrfinput*", "wrfbdy*", "namelist*"):
        for path in sorted(Path(config.input_dir).glob(pattern)):
            if path.is_file():
                inputs[path.name] = _hash_file(path)
    return {
        "source_sha256": source_hash.hexdigest(), "inputs": inputs,
        "input_dir": str(Path(config.input_dir).resolve()),
        "output_dir": str(Path(config.output_dir).resolve()),
        "run_start": run_start.isoformat(), "hours": config.hours,
        "feedback": config.feedback, "emit_initial_history": config.emit_initial_history,
        "namelists": {name: canonical_digest(bundle.namelist) for name, bundle in bundles.items()},
        "aot_environment": version_fingerprint_hash(), "trace_environment": global_trace_env_hash(),
    }


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
        elif json.loads((self.root / "owner.json").read_text()) != owner:
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
    expected = tuple(f.name for f in fields(OperationalCarry))
    for payload in snapshot["carries"].values():
        _validate_radiation_diagnostics_schema(payload)
        _validate_gwdo_diagnostics_schema(payload)
        if tuple(payload.get("carry_field_order", ())) != expected:
            raise ValueError("restart carry schema differs from executable")
        order = tuple(payload["state_field_order"])
        _validate_state_field_order(order)
        if set(payload["state_fields"]) != set(order):
            raise ValueError("restart State schema differs from receipt")
        if tuple(payload["scratch_field_order"]) != _CARRY_SCRATCH_FIELDS or set(payload["scratch_fields"]) != set(_CARRY_SCRATCH_FIELDS):
            raise ValueError("restart scratch schema differs from executable")
        if set(payload["extra_fields"]) != set(_V1_UNSUPPORTED_CARRY_FIELDS):
            raise ValueError("restart optional carry schema differs from executable")
        trees = (payload["state_fields"], payload["scratch_fields"], payload["extra_fields"],
                 payload.get("noahmp_land_fields"), payload.get("noahmp_rad"))
        for leaf in jax.tree.leaves(trees):
            value = np.asarray(leaf)
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError("restart contains a nonfinite carry leaf")


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
        if expected_identity is not None and manifest["identity"] != expected_identity:
            raise ValueError("restart run/configuration/source identity mismatch")
        payload_path = generation / "snapshot.pkl"
        if payload_path.stat().st_size != manifest["payload_bytes"] or _hash_file(payload_path) != manifest["sha256"]:
            raise ValueError("restart payload hash/size verification failed")
        with payload_path.open("rb") as stream:
            snapshot = pickle.load(stream)
        _validate_snapshot(snapshot, manifest)
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
        self.root.mkdir(parents=True, exist_ok=True)
        if set(carries) != set(own_steps) or set(carries) != set(dt_s):
            raise ValueError("checkpoint domains/counters/timesteps do not match")
        snapshot = {
            "carries": {name: _carry_to_payload(carry) for name, carry in carries.items()},
            "own_steps": dict(own_steps),
            "driver_state": _host(driver_state or {}),
        }
        manifest = {
            "format": FORMAT, "identity": identity, "domains": list(carries),
            "own_steps": dict(own_steps), "dt_s": dict(dt_s),
            "created_ns": time.time_ns(),
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
