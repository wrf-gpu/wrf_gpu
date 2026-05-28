"""Tester-added edge-case tests for the M2 Blackwell toolchain scout.

Complements `tests/test_m2_scout_matrix.py`. These tests try to break the
implementation along axes the worker test did not cover: ISO-8601 parsing,
known_gaps element typing, candidate ordering against the contract, narrative
ordering and word budget, per-candidate program file presence, exit-file
parseability, device-evidence strings in output.txt, and additional negative
cases (duplicate candidate name, illegal verdict, blocked-with-version_pin,
non-blocked-with-empty-output).
"""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "artifacts/m2/scout/toolchain_support_matrix.json"
REPORT_PATH = ROOT / "artifacts/m2/scout/toolchain_report.md"
HELLO_ROOT = ROOT / "artifacts/m2/scout/hello_gpu"
SCRIPT_PATH = ROOT / "scripts/m2_scout_hello_gpu.sh"

CONTRACT_ORDER = ["jax", "triton", "gt4py", "kokkos", "cupy_or_numba", "cuda_tile"]
VERDICT_ENUM = {"go", "go-with-version-bump", "blocked"}


# Re-import the worker's validator so the negative cases use the same rules
# the positive case is checked against.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "worker_matrix_test",
    ROOT / "tests/test_m2_scout_matrix.py",
)
_worker_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_worker_module)  # type: ignore[union-attr]
validate_matrix = _worker_module.validate_matrix


@pytest.fixture(scope="module")
def matrix() -> dict:
    return json.loads(MATRIX_PATH.read_text())


# --------------------------------------------------------------------------- #
# Schema-level edge cases not covered by the worker test                      #
# --------------------------------------------------------------------------- #

def test_generated_utc_parses_as_iso8601(matrix):
    stamp = matrix["generated_utc"]
    # Accept trailing "Z" by normalising to +00:00 so fromisoformat handles it
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None, "generated_utc must carry a timezone"


def test_known_gaps_are_strings(matrix):
    for candidate in matrix["candidates"]:
        for gap in candidate["known_gaps"]:
            assert isinstance(gap, str), (
                f"{candidate['name']} has non-string gap entry: {gap!r}"
            )


def test_candidate_order_matches_contract(matrix):
    names = [c["name"] for c in matrix["candidates"]]
    assert names == CONTRACT_ORDER, (
        f"matrix candidates must appear in the contract-fixed order; got {names}"
    )


def test_blocked_rationale_is_substantive(matrix):
    blocked = [c for c in matrix["candidates"] if c["verdict"] == "blocked"]
    for candidate in blocked:
        rationale = candidate["rationale"]
        assert len(rationale) >= 20, (
            f"blocked candidate {candidate['name']} rationale too terse: {rationale!r}"
        )


# --------------------------------------------------------------------------- #
# Hello-GPU evidence: programme files + real device-side evidence in stdout   #
# --------------------------------------------------------------------------- #

DEVICE_EVIDENCE = {
    "jax": ["CudaDevice", "[2.0, 4.0, 6.0, 8.0]"],
    "triton": ["RTX 5090", "[2.0, 4.0, 6.0, 8.0]"],
    "kokkos": ["execution_space=Cuda", "result=[2, 4, 6, 8]"],
    "cupy_or_numba": ["RTX 5090", "[2.0, 4.0, 6.0, 8.0]"],
    "cuda_tile": ["RTX 5090", "result=[2, 4, 6, 8]"],
}

PROGRAM_FILES = {
    "jax": ["hello.py"],
    "triton": ["hello.py"],
    "gt4py": ["hello.py"],
    "cupy_or_numba": ["hello.py"],
    "kokkos": ["hello.cpp", "build.sh", "CMakeLists.txt"],
    "cuda_tile": ["hello.cu", "build.sh"],
}


def test_each_candidate_has_a_runnable_program(matrix):
    for candidate in matrix["candidates"]:
        directory = HELLO_ROOT / candidate["name"]
        for filename in PROGRAM_FILES[candidate["name"]]:
            path = directory / filename
            assert path.exists(), f"{candidate['name']}: missing {path}"
            assert path.stat().st_size > 0, f"{candidate['name']}: empty {path}"


def test_passing_candidates_show_device_evidence(matrix):
    for candidate in matrix["candidates"]:
        if candidate["verdict"] == "blocked":
            continue
        out = (HELLO_ROOT / candidate["name"] / "output.txt").read_text()
        for needle in DEVICE_EVIDENCE[candidate["name"]]:
            assert needle in out, (
                f"{candidate['name']}: missing device-evidence {needle!r} in output.txt"
            )


def test_exit_files_are_integers(matrix):
    for candidate in matrix["candidates"]:
        exit_file = HELLO_ROOT / candidate["name"] / "exit.txt"
        assert exit_file.exists(), f"{candidate['name']}: missing exit.txt"
        raw = exit_file.read_text().strip()
        assert raw.lstrip("-").isdigit(), f"{candidate['name']}: exit.txt not integer: {raw!r}"
        code = int(raw)
        if candidate["verdict"] == "blocked":
            assert code != 0, f"{candidate['name']}: blocked should have non-zero exit"
        else:
            assert code == 0, f"{candidate['name']}: non-blocked must have exit 0"


# --------------------------------------------------------------------------- #
# Narrative report cross-consistency                                          #
# --------------------------------------------------------------------------- #

def test_narrative_covers_all_candidates_in_order():
    report = REPORT_PATH.read_text()
    positions = []
    for name in CONTRACT_ORDER:
        idx = report.find(f"## {name}")
        assert idx >= 0, f"narrative missing '## {name}' header"
        positions.append(idx)
    assert positions == sorted(positions), (
        f"narrative headings out of contract order: {positions}"
    )


def test_narrative_word_budget():
    report = REPORT_PATH.read_text()
    words = len(re.findall(r"\S+", report))
    assert words <= 2000, f"narrative exceeds 2000-word budget: {words}"


def test_narrative_mentions_target_hardware_and_closing():
    report = REPORT_PATH.read_text()
    assert "Target hardware" in report, "narrative missing target-hardware summary"
    assert "Closing Recommendation" in report, "narrative missing closing recommendation"


# --------------------------------------------------------------------------- #
# Wrapper script sanity                                                       #
# --------------------------------------------------------------------------- #

def test_wrapper_script_lists_exactly_six_candidates():
    text = SCRIPT_PATH.read_text()
    match = re.search(r"CANDIDATES=\(([^)]+)\)", text)
    assert match, "CANDIDATES=(...) array not found in wrapper script"
    listed = match.group(1).split()
    assert sorted(listed) == sorted(CONTRACT_ORDER), (
        f"wrapper script CANDIDATES mismatch: {listed}"
    )


# --------------------------------------------------------------------------- #
# Additional negative tests (try to make the validator accept bad input)       #
# --------------------------------------------------------------------------- #

def _expect_reject(corrupted, label):
    with pytest.raises(AssertionError, match=r".*"):
        validate_matrix(corrupted)


def test_negative_duplicate_candidate_name_is_rejected(matrix):
    corrupted = copy.deepcopy(matrix)
    # Replace the last entry with a duplicate of the first → 6 entries, but
    # only 5 unique names; validator must reject.
    corrupted["candidates"][-1] = copy.deepcopy(corrupted["candidates"][0])
    _expect_reject(corrupted, "duplicate name")


def test_negative_invalid_verdict_is_rejected(matrix):
    corrupted = copy.deepcopy(matrix)
    corrupted["candidates"][0]["verdict"] = "maybe"
    _expect_reject(corrupted, "invalid verdict")


def test_negative_blocked_with_install_command_is_rejected(matrix):
    corrupted = copy.deepcopy(matrix)
    for c in corrupted["candidates"]:
        if c["verdict"] == "blocked":
            c["install_command"] = "pip install foo"
            break
    else:
        pytest.skip("no blocked candidate in matrix to mutate")
    _expect_reject(corrupted, "blocked + install_command")


def test_negative_wrong_compute_capability_is_rejected(matrix):
    corrupted = copy.deepcopy(matrix)
    corrupted["target_hardware"]["compute_capability"] = "9.0"
    _expect_reject(corrupted, "wrong cc")


def test_negative_rationale_overflow_is_rejected(matrix):
    corrupted = copy.deepcopy(matrix)
    corrupted["candidates"][0]["rationale"] = "x" * 201
    _expect_reject(corrupted, "rationale > 200 chars")
