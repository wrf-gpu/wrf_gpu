"""Lazy Gen2 d02 wrfout loader for validation consumers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover - non-JAX utility contexts.
    jnp = None

from gpuwrf.io.data_inventory import DEFAULT_DOMAIN, parse_wrfout_valid_time


DEFAULT_SURFACE_FIELDS = ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC")


def normalize_valid_time(value: str | datetime | np.datetime64) -> datetime:
    """Normalize common valid-time inputs to UTC datetimes."""

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype(int)
        result = datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        text = text.replace("_", "T")
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            result = datetime.strptime(str(value), "%Y-%m-%d_%H:%M:%S")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _decode_times_variable(dataset: Dataset) -> list[str]:
    if "Times" not in dataset.variables:
        return []
    raw = np.asarray(dataset.variables["Times"][:])
    decoded: list[str] = []
    for row in raw:
        if row.dtype.kind == "S":
            decoded.append(b"".join(row.tolist()).decode("ascii", errors="replace").strip())
        else:
            decoded.append("".join(row.astype(str).tolist()).strip())
    return decoded


def _valid_time_from_file(path: Path, dataset: Dataset | None = None) -> datetime:
    try:
        return parse_wrfout_valid_time(path)
    except ValueError:
        if dataset is None:
            with Dataset(path, "r") as opened:
                return _valid_time_from_file(path, opened)
        decoded = _decode_times_variable(dataset)
        if not decoded:
            raise
        return normalize_valid_time(decoded[0])


def _read_variable(dataset: Dataset, field: str, *, squeeze_time: bool = True) -> np.ndarray:
    if field not in dataset.variables:
        raise KeyError(f"{field!r} not present in {dataset.filepath()}")
    variable = dataset.variables[field]
    data = variable[:]
    if squeeze_time and variable.dimensions and variable.dimensions[0] == "Time":
        data = variable[0]
    return np.asarray(np.ma.filled(data, np.nan))


def read_wrfout_file(
    path: str | Path,
    *,
    fields: Iterable[str] = DEFAULT_SURFACE_FIELDS,
    as_jax: bool = False,
    squeeze_time: bool = True,
) -> dict[str, Any]:
    """Read one wrfout file into field arrays without retaining the open handle."""

    source = Path(path)
    with Dataset(source, "r") as dataset:
        arrays = {field: _read_variable(dataset, field, squeeze_time=squeeze_time) for field in fields}
        valid_time = _valid_time_from_file(source, dataset)
    if as_jax:
        if jnp is None:  # pragma: no cover - only when JAX is unavailable.
            raise RuntimeError("JAX is required when as_jax=True")
        arrays = {field: jnp.asarray(value) for field, value in arrays.items()}
    return {
        "fields": arrays,
        "valid_time": np.asarray([np.datetime64(valid_time.replace(tzinfo=None), "s")]),
        "valid_time_utc": valid_time.isoformat(),
        "source_file": str(source),
    }


class Gen2WrfoutLoader:
    """Open Gen2 WRF history files lazily, one valid time at a time."""

    def __init__(
        self,
        run_path: str | Path,
        valid_time: str | datetime | np.datetime64 | None = None,
        *,
        domain: str = DEFAULT_DOMAIN,
    ) -> None:
        self.run_path = Path(run_path).expanduser().resolve()
        if not self.run_path.is_dir():
            raise FileNotFoundError(f"Gen2 run directory not found: {self.run_path}")
        self.domain = domain
        self._files: list[Path] | None = None
        self._time_axis: list[datetime] | None = None
        self.valid_time = normalize_valid_time(valid_time) if valid_time is not None else None

    @property
    def files(self) -> list[Path]:
        if self._files is None:
            self._files = sorted(self.run_path.glob(f"wrfout_{self.domain}_*"), key=lambda path: path.name)
            if not self._files:
                raise FileNotFoundError(f"no wrfout_{self.domain}_* files in {self.run_path}")
        return list(self._files)

    @property
    def time_axis(self) -> list[datetime]:
        if self._time_axis is None:
            times: list[datetime] = []
            for path in self.files:
                with Dataset(path, "r") as dataset:
                    times.append(_valid_time_from_file(path, dataset))
            self._time_axis = times
        return list(self._time_axis)

    def file_for_valid_time(self, valid_time: str | datetime | np.datetime64 | None = None) -> Path:
        target = self.valid_time if valid_time is None else normalize_valid_time(valid_time)
        if target is None:
            return self.files[0]
        for path, time in zip(self.files, self.time_axis, strict=True):
            if time == target:
                return path
        raise FileNotFoundError(f"no {self.domain} wrfout file for valid time {target.isoformat()} in {self.run_path}")

    def load(
        self,
        fields: Iterable[str] = DEFAULT_SURFACE_FIELDS,
        *,
        valid_time: str | datetime | np.datetime64 | None = None,
        as_jax: bool = False,
        squeeze_time: bool = True,
    ) -> dict[str, Any]:
        """Load fields for one valid time.

        NetCDF files are opened and closed inside this method. The returned
        arrays are NumPy unless the caller explicitly asks for JAX arrays at
        the validation boundary.
        """

        path = self.file_for_valid_time(valid_time)
        return read_wrfout_file(path, fields=fields, as_jax=as_jax, squeeze_time=squeeze_time)

    def iter_chunks(
        self,
        fields: Iterable[str] = DEFAULT_SURFACE_FIELDS,
        *,
        as_jax: bool = False,
    ):
        """Yield one wrfout payload at a time without materializing the run."""

        for path in self.files:
            yield read_wrfout_file(path, fields=fields, as_jax=as_jax)


__all__ = [
    "DEFAULT_SURFACE_FIELDS",
    "Gen2WrfoutLoader",
    "normalize_valid_time",
    "read_wrfout_file",
]
