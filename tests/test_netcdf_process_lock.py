"""Actual two-thread Dataset lifetimes; detect unsafe overlap before C can crash."""
from concurrent.futures import ThreadPoolExecutor
import ast
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from gpuwrf.io import netcdf_lock


def test_two_threads_hammer_actual_open_read_write_close(tmp_path, monkeypatch):
    original = netcdf_lock.netCDF4.Dataset
    gate = threading.Lock()
    owner = None
    live = 0
    opens = 0

    class Observed:
        def __init__(self, *args, **kwargs):
            nonlocal owner, live, opens
            ident = threading.get_ident()
            with gate:
                assert owner in (None, ident), "netCDF-C/HDF5 lifetime overlaps another thread"
                owner = ident
                live += 1
                opens += 1
            try:
                self.handle = original(*args, **kwargs)
            except BaseException:
                with gate:
                    live -= 1
                    if not live:
                        owner = None
                raise

        def __enter__(self):
            return self.handle

        def __exit__(self, *args):
            nonlocal owner, live
            try:
                # Both open and close execute under the same lifetime guard.
                time.sleep(.001)
                self.handle.close()
            finally:
                with gate:
                    live -= 1
                    if not live:
                        owner = None

    monkeypatch.setattr(netcdf_lock.netCDF4, "Dataset", Observed)
    barrier = threading.Barrier(2)

    def worker(index):
        path = tmp_path / f"thread{index}.nc"
        for iteration in range(20):
            barrier.wait(timeout=10)
            with netcdf_lock.Dataset(path, "w") as ds:
                ds.createDimension("x", 8)
                ds.createVariable("value", "i4", ("x",))[:] = index + iteration
                ds.setncattr("iteration", iteration)
                time.sleep(.001)
            with netcdf_lock.Dataset(path) as ds:
                np.testing.assert_array_equal(ds["value"][:], np.full(8, index + iteration))
                assert ds.iteration == iteration
        return True

    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(worker, index) for index in (1, 2)]
        assert all(f.result(timeout=30) for f in futures)
    assert opens == 80 and live == 0 and owner is None


def test_nested_datasets_and_exception_release_lock(tmp_path):
    for name in ("a", "b"):
        with netcdf_lock.Dataset(tmp_path / name, "w") as ds:
            ds.createDimension("x", 1)
    with pytest.raises(RuntimeError, match="body"):
        with netcdf_lock.Dataset(tmp_path / "a") as a, netcdf_lock.Dataset(tmp_path / "b") as b:
            assert len(a.dimensions["x"]) == len(b.dimensions["x"]) == 1
            raise RuntimeError("body")
    with pytest.raises(FileNotFoundError):
        with netcdf_lock.Dataset(tmp_path / "missing"):
            pass
    with ThreadPoolExecutor(1) as executor:
        def reopen():
            with netcdf_lock.Dataset(tmp_path / "a") as ds:
                return len(ds.dimensions["x"])
        assert executor.submit(reopen).result(timeout=5) == 1


def test_production_open_paths_share_the_process_lock():
    root = Path(__file__).resolve().parents[1] / "src/gpuwrf"
    for path in root.rglob("*.py"):
        if path.name == "netcdf_lock.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "netCDF4":
                assert not any(a.name == "Dataset" for a in node.names), str(path)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "Dataset" and isinstance(node.func.value, ast.Name):
                    assert node.func.value.id != "netCDF4", str(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            calls = [i.context_expr for i in node.items if isinstance(i.context_expr, ast.Call)]
            if any(isinstance(c.func, ast.Attribute) and c.func.attr in {"File", "open_dataset", "open_mfdataset"}
                   for c in calls):
                assert any(isinstance(i.context_expr, ast.Name) and i.context_expr.id == "NETCDF_LOCK"
                           for i in node.items), str(path)
