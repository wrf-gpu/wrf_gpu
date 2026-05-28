"""HDF5 reader and writer for WRF small-step savepoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from gpuwrf.validation.savepoint_schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, Savepoint, SavepointMetadata


METADATA_ATTR = "metadata_json"
PAYLOAD_SHA256_ATTR = "payload_sha256"
FIELDS_GROUP = "fields"


def _canonical_metadata(metadata: SavepointMetadata) -> bytes:
    return json.dumps(metadata.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _payload_digest(metadata: SavepointMetadata, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_metadata(metadata))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _chunks_for(array: np.ndarray) -> bool | tuple[int, ...]:
    if array.ndim == 0 or array.size < 16:
        return False
    return tuple(max(1, min(dim, 16)) for dim in array.shape)


def write_savepoint(path: str | Path, savepoint: Savepoint) -> None:
    """Write one validated savepoint as HDF5 with compressed field datasets."""

    savepoint.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(array) for name, array in savepoint.arrays.items()}
    metadata_json = _canonical_metadata(savepoint.metadata).decode("utf-8")
    digest = _payload_digest(savepoint.metadata, arrays)
    with h5py.File(target, "w") as handle:
        handle.attrs[METADATA_ATTR] = metadata_json
        handle.attrs[PAYLOAD_SHA256_ATTR] = digest
        fields = handle.create_group(FIELDS_GROUP)
        for name, array in arrays.items():
            chunks = _chunks_for(array)
            kwargs: dict[str, Any] = {}
            if chunks:
                kwargs.update({"compression": "gzip", "compression_opts": 4, "shuffle": True, "chunks": chunks})
            fields.create_dataset(name, data=array, **kwargs)


def read_savepoint(
    path: str | Path,
    *,
    expected_schema_version: str | None = None,
    verify_tamper: bool = True,
) -> Savepoint:
    """Read and validate one HDF5 savepoint.

    ``expected_schema_version`` is intentionally explicit so dry-run tests can
    prove version mismatch failures without mutating global constants. When
    ``None`` (the default), any version in ``SUPPORTED_SCHEMA_VERSIONS`` is
    accepted (M6B-ladder-hygiene Stage 3: the schema is purely additive across
    v1→v4, so older savepoints remain readable). Pass an explicit string to
    force exact-version matching (used by the dry-run mismatch test).
    """

    source = Path(path)
    try:
        with h5py.File(source, "r") as handle:
            if METADATA_ATTR not in handle.attrs:
                raise ValueError(f"{source} is missing {METADATA_ATTR}")
            metadata_payload = json.loads(str(handle.attrs[METADATA_ATTR]))
            file_version = metadata_payload.get("schema_version")
            if expected_schema_version is None:
                if file_version not in SUPPORTED_SCHEMA_VERSIONS:
                    raise ValueError(f"unsupported savepoint schema: {file_version}")
            else:
                if file_version != expected_schema_version:
                    raise ValueError(f"unsupported savepoint schema: {file_version}")
            metadata = SavepointMetadata.from_json(metadata_payload)
            if FIELDS_GROUP not in handle:
                raise ValueError(f"{source} is missing /{FIELDS_GROUP}")
            arrays = {name: np.asarray(dataset) for name, dataset in handle[FIELDS_GROUP].items()}
            stored_digest = str(handle.attrs.get(PAYLOAD_SHA256_ATTR, ""))
    except OSError as exc:
        raise ValueError(f"{source} is not a readable HDF5 savepoint") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} has invalid savepoint metadata JSON") from exc
    savepoint = Savepoint(metadata=metadata, arrays=arrays)
    savepoint.validate()
    if verify_tamper:
        actual_digest = _payload_digest(savepoint.metadata, arrays)
        if not stored_digest or stored_digest != actual_digest:
            raise ValueError(f"{source} failed savepoint tamper detection")
    return savepoint
