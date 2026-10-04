"""Run the frozen scripts/compare_wrfout_grid.py with one read-only handle per file.

Pure I/O change. The comparator opens both wrfout files again for every variable
and every lead; a netCDF-4 open of a 375-variable WRF history file costs ~57 ms
of header parsing (local or CIFS alike), i.e. ~18 min of a 25-lead domain score.
Here each path is opened once per process and the `with` blocks no longer close
it. The comparator source is imported unchanged; arguments and outputs are the
same as calling it directly.
"""
from __future__ import annotations

import atexit
import importlib.util
from pathlib import Path
import sys

import netCDF4

ROOT = Path(__file__).resolve().parents[1]
# Full-variable reads bypass the cache; keep 50 open files x 375 variables bounded.
netCDF4.set_chunk_cache(1 << 20, 521, 0.75)

spec = importlib.util.spec_from_file_location("compare_wrfout_grid", ROOT / "scripts/compare_wrfout_grid.py")
compare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare
spec.loader.exec_module(compare)

_handles: dict[str, netCDF4.Dataset] = {}


class _Held:
    def __init__(self, ds: netCDF4.Dataset):
        self.ds = ds

    def __enter__(self) -> netCDF4.Dataset:
        return self.ds

    def __exit__(self, *exc) -> bool:
        return False


def held_dataset(path, mode: str = "r", *args, **kwargs) -> _Held:
    if mode != "r" or args or kwargs:
        raise ValueError(f"read-only cached open only, got mode={mode!r} {args} {kwargs}")
    key = str(path)
    ds = _handles.get(key)
    if ds is None:
        ds = _handles[key] = netCDF4.Dataset(key, "r")
    return _Held(ds)


@atexit.register
def _close_all() -> None:
    for ds in _handles.values():
        if ds.isopen():
            ds.close()


compare.Dataset = held_dataset

if __name__ == "__main__":
    raise SystemExit(compare.main(sys.argv[1:]))
