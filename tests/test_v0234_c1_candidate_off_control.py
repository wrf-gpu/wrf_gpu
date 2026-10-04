"""Binding C1 gate for the scalar candidate's released 0/0 control."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
# This 71-leaf identity gate was introduced with the v0.23.4 repair. Later
# accepted physics changes and State additions deliberately change these bytes.
# Replay its original repaired tree; preserve the released hash oracle below.
REPAIRED_SOURCE = "3b81fb5b093639e70c12cce87d602c45b326b18b"


def test_nested_scalar_options_zero_match_precandidate_full_carry_bytes(tmp_path) -> None:
    packed = subprocess.check_output([
        "git", "-C", str(ROOT), "archive", REPAIRED_SOURCE,
        "src", "data/fixtures", "data/manifests", "scripts/extract_rrtmg_tables.py",
    ])
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        archive.extractall(tmp_path, filter="data")
    (tmp_path / "data/wrf_pristine").symlink_to("<USER_HOME>/src/wrf_pristine")
    probe_path = ROOT / "scripts/v0234_c1_candidate_off_probe.py"
    code = (
        "import importlib.util, json; "
        f"spec = importlib.util.spec_from_file_location('probe', {str(probe_path)!r}); "
        "probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe); "
        "print(json.dumps(probe.run_probe()))"
    )
    environment = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0",
                       PYTHONPATH=str(tmp_path / "src"))
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=environment,
        capture_output=True, text=True, check=False, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["candidate_off_identity_pass"] is True
    assert result["input_leaf_count"] == result["output_leaf_count"] == 71
    assert result["options"] == {"moist_adv_opt": 0, "scalar_adv_opt": 0}
    assert result["nested_frozen_wrf_boundary_bundle"] is True
    assert result["broken_control_reproduced_as_nonidentity"] is True
    assert (
        result["verdict"]
        == "C1_REPAIRED__CANDIDATE_OFF_MATCHES_RELEASED_FULL_CARRY_BYTES"
    )
    assert len(result["canonical_payload_sha256"]) == 64
