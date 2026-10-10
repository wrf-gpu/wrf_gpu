"""The NSSL runtime asset must ship with the code: tracked by git (the repo-wide ``data/`` ignore rule hid it),
declared as setuptools package data, and loadable from a CLEAN checkout (git archive of HEAD), not only from a
worktree that happens to hold the untracked file."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ASSET = "src/gpuwrf/physics/nssl2mom/data/init_constants.json"
# every file the nssl2mom package reads at run time (constants.py _ASSET); keep in sync with the package
RUNTIME_ASSETS = (ASSET,)


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
def test_runtime_assets_are_tracked():
    for path in RUNTIME_ASSETS:
        r = _git("ls-files", "--error-unmatch", path)
        assert r.returncode == 0, f"{path} is not tracked by git (gitignored?): {r.stderr}"
        assert (REPO / path).stat().st_size < 1_000_000


def test_runtime_assets_are_package_data():
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    pkg_data = cfg["tool"]["setuptools"]["package-data"].get("gpuwrf.physics.nssl2mom", [])
    assert "data/init_constants.json" in pkg_data


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
def test_constants_load_from_clean_checkout(tmp_path):
    """git archive HEAD (committed src/data/scripts only) -> fresh interpreter imports the package, loads both builds."""
    tar = tmp_path / "src.tar"
    # a snapshot = committed files only; gpuwrf.physics imports repo-level tracked assets (data/fixtures Thompson
    # tables, scripts/ table extractors, E128), so archive those too -- untracked files cannot leak in.
    r = _git("archive", "--format=tar", "-o", str(tar), "HEAD", "src", "data", "scripts")
    assert r.returncode == 0, r.stderr
    with tarfile.open(tar) as t:
        t.extractall(tmp_path / "clean", filter="data")
    shutil.rmtree(tmp_path / "clean" / "src" / "gpuwrf" / "physics" / "nssl2mom" / "__pycache__", ignore_errors=True)
    code = (
        "import sys; from gpuwrf.physics.nssl2mom import constants as c; "
        "assert c.__file__.startswith(sys.argv[1]), c.__file__; "
        "f32 = c.get_constants('fp32'); f64 = c.get_constants('fp64'); "
        "assert f32.lhab == 8 and f32.na == 18 and f64.ipconc == 5; print('OK', c.__file__)"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(tmp_path / "clean" / "src"))
    clean = str(tmp_path / "clean" / "src")
    out = subprocess.run([sys.executable, "-c", code, clean], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path), timeout=600)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]
