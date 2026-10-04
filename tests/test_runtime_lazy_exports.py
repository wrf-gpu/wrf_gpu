"""Fresh-process checks for runtime's public API and configuration ordering."""
import os
from pathlib import Path
import subprocess
import sys


def test_runtime_exports_follow_package_configuration():
    source = r'''
import importlib.abc
import sys

seen = []
class PhysicsOrder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("gpuwrf.physics"):
            package = sys.modules["gpuwrf"]
            assert hasattr(package, "_COMMAND_BUFFER_STATUS"), fullname
            assert hasattr(package, "_JAX_CACHE_STATUS"), fullname
            seen.append(fullname)
        return None
sys.meta_path.insert(0, PhysicsOrder())
import gpuwrf
import gpuwrf.runtime as runtime
assert "gpuwrf.runtime.operational_mode" not in sys.modules
assert not seen
assert runtime.__all__ == ["OperationalNamelist", "read_checkpoint",
    "read_checkpoint_with_runtime_state", "run_forecast_operational", "write_checkpoint"]
from gpuwrf.runtime import operational_mode, checkpoint
for name in runtime.__all__:
    module = operational_mode if name in {"OperationalNamelist", "run_forecast_operational"} else checkpoint
    assert getattr(runtime, name) is getattr(module, name), name
    assert getattr(runtime, name) is runtime.__dict__[name], name
assert seen
try:
    runtime.unregistered_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown export accepted")
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", source], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
