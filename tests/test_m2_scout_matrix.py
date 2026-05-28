import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "artifacts/m2/scout/toolchain_support_matrix.json"
EXPECTED = {"jax", "triton", "gt4py", "kokkos", "cupy_or_numba", "cuda_tile"}
VERDICTS = {"go", "go-with-version-bump", "blocked"}


def validate_matrix(matrix):
    assert isinstance(matrix["generated_utc"], str)
    target = matrix["target_hardware"]
    assert target["gpu_model"]
    assert target["compute_capability"] == "12.0"
    assert target["driver_version"]

    candidates = matrix["candidates"]
    assert len(candidates) == 6
    names = [candidate["name"] for candidate in candidates]
    assert set(names) == EXPECTED
    assert len(names) == len(set(names))

    for candidate in candidates:
        assert candidate["verdict"] in VERDICTS
        assert candidate["rationale"]
        assert len(candidate["rationale"]) <= 200
        assert isinstance(candidate["known_gaps"], list)
        artifact_dir = candidate["hello_gpu_artifact_dir"]
        assert artifact_dir == f"artifacts/m2/scout/hello_gpu/{candidate['name']}/"
        if candidate["verdict"] == "blocked":
            assert candidate["hello_gpu_passed"] is False
            assert candidate["version_pin"] is None
            assert candidate["install_command"] is None
        else:
            assert candidate["hello_gpu_passed"] is True
            assert candidate["version_pin"]
            assert candidate["install_command"]
            directory = ROOT / artifact_dir
            assert (directory / "exit.txt").read_text().strip() == "0"
            assert (directory / "output.txt").read_text().strip()


def test_matrix_schema_and_smoke_outputs():
    validate_matrix(json.loads(MATRIX.read_text()))


def test_corrupted_matrix_is_rejected():
    matrix = json.loads(MATRIX.read_text())
    corrupted = copy.deepcopy(matrix)
    corrupted["candidates"] = corrupted["candidates"][:-1]
    try:
        validate_matrix(corrupted)
    except AssertionError:
        return
    raise AssertionError("corrupted matrix should be rejected")
