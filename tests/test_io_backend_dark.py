"""Header-only allocator setup must precede any JAX backend initialization."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


_PROBE = r'''
import json, sys
from pathlib import Path
from types import SimpleNamespace
from netCDF4 import Dataset
case = Path(sys.argv[1])
case.mkdir()
(case / "namelist.input").write_text("&domains\n max_dom=3,\n/\n")
for domain in ("d01", "d02", "d03"):
    with Dataset(case / ("wrfinput_" + domain), "w") as ds:
        ds.createDimension("west_east", 4)
        ds.createDimension("south_north", 3)
        ds.createDimension("bottom_top", 2)
        ds.DX = ds.DY = 3000.0
import jax._src.xla_bridge as xb
assert not xb._backends
attempts = []
def forbidden(*args, **kwargs):
    attempts.append(str(args))
    raise RuntimeError("BACKEND_DARK_VIOLATION")
xb._init_backend = forbidden
import gpuwrf.io
from gpuwrf.io.netcdf_lock import Dataset as locked_dataset
from gpuwrf.runtime import gpu_allocator as ga
def no_cuda():
    raise AssertionError("header-only sizing touched CUDA")
ga._pool_device_memory = no_cuda
args = SimpleNamespace(input_dir=case, namelist=None, max_dom=3,
                       feedback=False, emit_initial_history=True)
env = {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
       "GPUWRF_JAX_CACHE_DIR": str(case / "cache")}
ga.configure_cli_pool(args, env)
assert ga._SESSION is not None, "C-auto silently fell back instead of describing the case"
assert ga._SESSION["domains"] == 3
assert not xb._backends and not attempts
assert "gpuwrf.io.wrfout_writer" not in sys.modules
assert "gpuwrf.physics.thompson_tables" not in sys.modules
print(json.dumps({"backend_dark": True, "domains": ga._SESSION["domains"]}))
'''


def test_lock_import_and_actual_allocator_header_setup_are_backend_dark(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(source), GPUWRF_FAST_DEFAULTS="1")
    result = subprocess.run([sys.executable, "-c", _PROBE, str(tmp_path / "case")],
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"backend_dark": True, "domains": 3}


def test_lazy_public_exports_preserve_frozen_object_identities():
    import gpuwrf.io as io
    contract = json.loads((Path(__file__).parent / "fixtures/io_public_exports.json").read_text())
    assert io.__all__ == contract["all"]
    assert set(contract["all"]) <= set(dir(io))
    for name, (module, attribute) in contract["objects"].items():
        original = getattr(importlib.import_module(module), attribute)
        assert getattr(io, name) is original, name
        assert getattr(io, name) is original, "lazy export must cache the original singleton"
