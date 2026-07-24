from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
ACCEPTED_COMPONENT_SRC_TREE = "ede4f8a706f9140973746412c107cbadf3240f83"


def _load(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_before = set(sys.modules)
gpu = _load("v0234_gpt_postfix_sp2_capture_test", "scripts/v0234_gpt_postfix_sp2_capture.py")
cpu = _load("v0234_gpt_postfix_cpu_capture_test", "scripts/v0234_gpt_postfix_cpu_capture.py")
compare = _load("v0234_gpt_postfix_sp2_compare_test", "scripts/v0234_gpt_postfix_sp2_compare.py")
_imported = set(sys.modules) - _before


def test_profiles_import_backend_dark() -> None:
    assert not any(
        name == "jax" or name.startswith(("jax.", "gpuwrf."))
        for name in _imported
    )


def test_fresh_nonce_namespace_and_contract_patch_are_exact() -> None:
    assert gpu.NONCE == cpu.NONCE == compare.NONCE
    assert len(gpu.NONCE) == 64
    assert gpu.PREFIX == "a73aef522fbdca9c"
    assert gpu.AUTHORITY_ROOT == cpu.AUTHORITY_ROOT == compare.AUTHORITY_ROOT
    assert gpu.PREFIX not in gpu.CONSUMED_PREFIXES
    assert set(gpu.CONSUMED_PREFIXES) == {
        "0b18530a1dc9cac2", "ac6712170cbe5084", "2e1ad51a081af145",
    }
    assert gpu.AMENDMENT_COMMIT == "b7ef240c0c4f9916419f4dfa660c4a04fde51275"
    assert gpu.CONTRACT_COMMIT == "68d30d515d7cc4ce157de33528d97364d85dad55"
    assert gpu.base.sha256_file(gpu.AMENDMENT) == gpu.AMENDMENT_SHA256
    reservation = gpu.reservation_value(gpu.ACCEPTED_COMPONENT_HEAD)
    assert reservation["gpu_arm_limit"] == 1
    assert reservation["retry_permitted"] is False
    assert reservation["gpu_query_permitted"] is False


def test_lock_profile_is_exactly_one_canonical_preemptible_arm() -> None:
    command = gpu.build_lock_command("a" * 40)
    assert command.count(str(gpu.base.LOCK_WRAPPER)) == 1
    assert command[command.index("--timeout") + 1] == "0"
    assert command[command.index("--intent") + 1] == "production-preemptible"
    assert command.count("--execute") == 1
    assert command[-2:] == ["--approved-head", "a" * 40]
    freshness = inspect.signature(gpu.base.assert_fresh_output_and_nonce)
    assert freshness.parameters["output"].default is inspect.Parameter.empty
    assert freshness.parameters["receipt"].default is inspect.Parameter.empty
    validation = inspect.signature(gpu.base.validate_snapshot_manifest)
    assert validation.parameters["expected_nonce"].default is inspect.Parameter.empty
    assert gpu.base.OUTPUT_DIR == gpu.CAPTURE_ROOT
    assert gpu.base.EXECUTION_RECEIPT == gpu.EXECUTION_RECEIPT
    assert gpu.base.NONCE == gpu.NONCE
    source = Path(gpu.__file__).read_text(encoding="utf-8")
    assert "nvidia-smi" not in source
    assert "jax.devices" not in source
    assert "default_backend" not in source


def test_cpu_recorder_calls_explicit_fresh_lifecycle_once() -> None:
    source = cpu.BASE_SCRIPT.read_text(encoding="utf-8")
    assert "state, DT, grid, first_timestep=True, restart=False" in source
    assert "POSTFIX_CAPTURE_REQUIRES_FRESH_START" in source
    assert "mynn_adapter_with_source_leaves(state,6.0,grid," in source
    assert "first_timestep=True,restart=False)" in source
    assert cpu.base.sha256_file(cpu.BASE_SCRIPT) == cpu.BASE_SHA256


def test_reference_and_comparator_authorities_are_frozen() -> None:
    assert len(compare.REQUIRED) == 28
    assert tuple(compare.REQUIRED[-6:]) == tuple(compare.RESIDUALS)
    assert compare.sha256_file(compare.COMPARATOR) == compare.COMPARATOR_SHA256
    assert compare.sha256_file(compare.AUTHENTIC_28_ARCHIVE) == compare.AUTHENTIC_28_ARCHIVE_SHA256
    assert compare.sha256_file(compare.AUTHENTIC_28_MANIFEST) == compare.AUTHENTIC_28_MANIFEST_SHA256
    assert compare.PRE_FIX_RMS == {
        "rublten": 0.0002538443572720412,
        "rvblten": 0.00016647474452487508,
    }
    source = Path(compare.__file__).read_text(encoding="utf-8")
    assert '"mix_sm": diagnostic["mix_sm"]' in source
    assert '"mix_el": cpu_arrays["turbulence_el"]' in source
    assert '"mix_dfm": cpu_arrays["turbulence_dfm"]' in source
    assert "DIAGNOSTIC_REPLAY_NOT_BIT_EXACT" not in source
    assert "DIAGNOSTIC_REPLAY_LIFECYCLE_DRIFT" in source


def test_terminal_classification_is_strict_and_fail_closed() -> None:
    lower = {
        name: np.nextafter(value, 0.0)
        for name, value in compare.PRE_FIX_RMS.items()
    }
    assert compare.classify_postfix(lower) == "POSTFIX_SP2_VALIDATED_IMPROVED_BOTH"
    equal = dict(compare.PRE_FIX_RMS)
    assert compare.classify_postfix(equal) == "POSTFIX_SP2_VALIDATED_MIXED_OR_REGRESSED"
    mixed = dict(lower)
    mixed["rvblten"] = compare.PRE_FIX_RMS["rvblten"] * 1.01
    assert compare.classify_postfix(mixed) == "POSTFIX_SP2_VALIDATED_MIXED_OR_REGRESSED"
    with pytest.raises(compare.PostfixFailure, match="POSTFIX_RMS"):
        compare.classify_postfix({"rublten": np.nan, "rvblten": 0.0})


def test_postfix_bundle_validator_binds_order_payload_and_provenance(tmp_path: Path, monkeypatch) -> None:
    shapes = {name: (2, 3) for name in compare.REQUIRED}
    monkeypatch.setattr(compare, "EXPECTED_SHAPES", shapes)
    arrays = {
        name: np.full(shapes[name], index + 0.5, dtype="<f8")
        for index, name in enumerate(compare.REQUIRED)
    }
    archive = (tmp_path / "bundle.npz").resolve()
    np.savez(archive, **arrays)
    manifest = (tmp_path / "manifest.json").resolve()
    value = {
        "schema": "wrfgpu2-v0234-gpt-postfix-sp2-reference-v1",
        "reference_status": "AUTHENTIC_POSTFIX_28_OF_28",
        "array_order": list(compare.REQUIRED),
        "capture_nonce": compare.NONCE,
        "production_adapter_invocations": 1,
        "archive_sha256": compare.sha256_file(archive),
        "arrays": {
            name: compare.array_record(array) for name, array in arrays.items()
        },
    }
    compare.atomic_json(manifest, value)
    assert compare.validate_reference_bundle(manifest, archive)["valid"] is True
    arrays[compare.REQUIRED[0]][0, 0] += 1.0
    np.savez(archive, **arrays)
    with pytest.raises(compare.PostfixFailure, match="HASH"):
        compare.validate_reference_bundle(manifest, archive)


def test_accepted_component_source_tree_is_authenticated_and_ancestral() -> None:
    import subprocess

    accepted_tree = subprocess.run(
        [
            "git", "-C", str(REPO), "rev-parse",
            f"{gpu.ACCEPTED_COMPONENT_HEAD}:src/gpuwrf",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert accepted_tree == ACCEPTED_COMPONENT_SRC_TREE
    ancestry = subprocess.run(
        [
            "git", "-C", str(REPO), "merge-base", "--is-ancestor",
            gpu.ACCEPTED_COMPONENT_HEAD, "HEAD",
        ],
        check=False,
    )
    assert ancestry.returncode == 0
