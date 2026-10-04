"""Portable compiler prerequisite and frozen, repository-owned WRF oracles."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest


@pytest.fixture(scope="module")
def wrf_fortran_compiler():
    requested = os.environ.get("FC", "gfortran")
    compiler = shutil.which(requested)
    if compiler is None:
        pytest.skip(f"Frozen WRF oracle requires Fortran compiler {requested!r}; set FC or install gfortran")
    return compiler


@pytest.fixture(scope="module")
def frozen_wrf_oracles():
    sources = Path(__file__).with_name("oracles")
    for name, expected in json.loads((sources / "fixture_hashes.json").read_text()).items():
        assert hashlib.sha256((sources / name).read_bytes()).hexdigest() == expected, name
    return sources
