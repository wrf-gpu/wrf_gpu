from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0234_gpt_single_authority_capture.py"
)
SPEC = importlib.util.spec_from_file_location(
    "v0234_gpt_single_authority_capture_metric_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
_BACKEND_MODULES_BEFORE = {
    name for name in sys.modules
    if name == "jax" or name.startswith(("jax.", "gpuwrf."))
}
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)
_BACKEND_MODULES_AFTER = {
    name for name in sys.modules
    if name == "jax" or name.startswith(("jax.", "gpuwrf."))
}


@dataclass(frozen=True)
class _Grid:
    metrics: object
    identity: str


class _GridBuilder:
    def as_grid_spec(self) -> _Grid:
        return _Grid(metrics="analytic-flat-fallback", identity="d03-grid")


class _Run:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def grid(self, domain: str) -> _GridBuilder:
        self.requested.append(domain)
        return _GridBuilder()


def test_real_case_capture_replaces_flat_metrics_with_exact_wrfinput_path(
    tmp_path: Path,
) -> None:
    run = _Run()
    loaded_paths: list[Path] = []
    authentic_metrics = object()

    def loader(path: Path) -> object:
        loaded_paths.append(path)
        return authentic_metrics

    result = capture.grid_with_wrfinput_metrics(run, tmp_path, loader)

    assert run.requested == ["d03"]
    assert loaded_paths == [tmp_path / "wrfinput_d03"]
    assert result.metrics is authentic_metrics
    assert result.identity == "d03-grid"


def test_capture_import_remains_backend_dark_and_binds_metrics_source() -> None:
    assert "src/gpuwrf/dynamics/metrics.py" in capture.STATIC_SOURCE_FILES
    assert _BACKEND_MODULES_AFTER == _BACKEND_MODULES_BEFORE
