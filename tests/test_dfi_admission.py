"""Admission controls only: no DFI implementation or numerical model execution."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from gpuwrf.io.namelist_check import (
    UnsupportedSchemeError,
    validate_namelist,
    validate_operational_namelist,
)


@pytest.mark.parametrize("value", [1, 2, 3, -1, "3", [0, 3]])
def test_nonzero_dfi_refused(value):
    with pytest.raises(UnsupportedSchemeError) as exc:
        validate_operational_namelist({"dfi_control": {"dfi_opt": value}})
    message = str(exc.value)
    assert "dfi_control.dfi_opt" in message
    assert "digital filter initialization" in message
    assert "dfi_opt=0" in message
    assert "CPU-WRF" in message


@pytest.mark.parametrize("config", [{}, {"dfi_opt": 0}, {"dfi_opt": "0"},
                                  {"dfi_control": {"dfi_opt": [0]}}])
def test_dfi_disabled_or_absent_passes(config):
    validate_operational_namelist(config)


def test_reference_validator_and_unknown_keys_unchanged():
    validate_namelist({"dfi_control": {"dfi_opt": 3}})
    validate_operational_namelist({"unregistered_control": 3})


def test_standard_wn3_dfi0_passes():
    text = (Path(__file__).parent / "fixtures/wn3_20260227_cadence.namelist").read_text()
    validate_operational_namelist(text)
    validate_operational_namelist(text + "\n&dfi_control\n dfi_opt=0,\n/\n")


def test_cli_refuses_dfi_before_backend_or_model_import(tmp_path):
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&dfi_control\n dfi_opt=3,\n/\n")
    code = '''
import importlib.abc
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts/v025"))
import cpu_guard

class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("jax_plugins", "jaxlib.cuda", "cupy", "triton")):
            raise ImportError("CPU admission test excludes GPU plugins: " + fullname)
        if fullname.startswith(("gpuwrf.integration", "gpuwrf.physics",
                                "gpuwrf.runtime.operational_mode")):
            raise AssertionError("heavy import before DFI refusal: " + fullname)
sys.meta_path.insert(0, NoHeavyImports())
from gpuwrf import cli
from jax._src import xla_bridge
def forbidden(*args, **kwargs):
    raise AssertionError("backend initialization before DFI refusal")
xla_bridge.get_backend = forbidden
xla_bridge.backends = forbidden
assert cli.main(["run", "--namelist", sys.argv[1], "--input-dir", sys.argv[2],
                 "--output-dir", sys.argv[3], "--hours", "3"]) == 2
assert not xla_bridge._backends
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-c", code, str(namelist), str(tmp_path),
         str(tmp_path / "output")], env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dfi_control.dfi_opt" in result.stderr
    assert "dfi_opt=0" in result.stderr
    assert not (tmp_path / "output").exists()
