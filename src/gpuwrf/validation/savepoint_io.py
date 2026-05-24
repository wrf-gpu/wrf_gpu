"""NPZ-bundle reader and writer for M6B0 savepoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gpuwrf.validation.savepoint_schema import SCHEMA_VERSION, Savepoint, SavepointMetadata


METADATA_KEY = "__metadata_json__"
FIELD_PREFIX = "field__"


def _field_key(name: str) -> str:
    return FIELD_PREFIX + name.replace("/", "__slash__")


def _field_name(key: str) -> str:
    return key[len(FIELD_PREFIX) :].replace("__slash__", "/")


def write_savepoint(path: str | Path, savepoint: Savepoint) -> None:
    """Writes one savepoint as a compressed NPZ bundle."""

    savepoint.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(savepoint.metadata.to_json(), sort_keys=True).encode("utf-8")
    payload: dict[str, np.ndarray] = {METADATA_KEY: np.frombuffer(metadata, dtype=np.uint8)}
    for name, array in savepoint.arrays.items():
        payload[_field_key(name)] = np.asarray(array)
    np.savez_compressed(target, **payload)


def read_savepoint(path: str | Path) -> Savepoint:
    """Reads and validates one NPZ-bundle savepoint."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as bundle:
            if METADATA_KEY not in bundle.files:
                raise ValueError(f"{source} is missing {METADATA_KEY}")
            metadata_json = bytes(bundle[METADATA_KEY].tolist()).decode("utf-8")
            metadata_payload = json.loads(metadata_json)
            if metadata_payload.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"unsupported savepoint schema: {metadata_payload.get('schema_version')}")
            metadata = SavepointMetadata.from_json(metadata_payload)
            arrays = {
                _field_name(key): np.asarray(bundle[key])
                for key in bundle.files
                if key.startswith(FIELD_PREFIX)
            }
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} has invalid savepoint metadata JSON") from exc
    except OSError as exc:
        raise ValueError(f"{source} is not a readable savepoint bundle") from exc
    savepoint = Savepoint(metadata=metadata, arrays=arrays)
    savepoint.validate()
    return savepoint
