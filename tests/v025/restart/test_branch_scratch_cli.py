"""Real `gpuwrf run --branch-from` path with the DEFAULT scratch (<output-dir>/.scratch).

The CLI creates its scratch directory before the driver re-checks that the branch output is empty. Before the
v50-branch fix that re-check refused every public branch run without --scratch-dir (WN3 CPU arm G2, 2026-11-10).
The driver is stopped by a sentinel right after the path check; nothing is loaded or compiled.
"""
import json
from pathlib import Path

import pytest

from gpuwrf.runtime.checkpoint_branch import validate_paths

SWISS = Path(__file__).resolve().parents[3] / "examples" / "switzerland_d01"


class _AfterAdmission(RuntimeError):
    pass


def _generation(tmp_path):
    generation = tmp_path / "ckpt" / "generation-1"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text(json.dumps({"identity": {"output_dir": str(tmp_path / "parent")}}))
    manifest = tmp_path / "empty.json"
    manifest.write_text('{"schema_version": 1}')
    return generation, manifest


def test_public_branch_run_passes_the_output_check_with_default_scratch(tmp_path, monkeypatch, capsys):
    from gpuwrf import cli
    from gpuwrf.integration import nested_pipeline
    generation, manifest = _generation(tmp_path)

    def stop(*_args, **_kwargs):
        raise _AfterAdmission("sentinel: driver reached case loading")

    monkeypatch.setattr(nested_pipeline, "_load_domains", stop)
    monkeypatch.delenv("GPUWRF_SCRATCH", raising=False)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    rc = cli.main(["run", "--input-dir", str(SWISS), "--output-dir", str(tmp_path / "branch"),
                   "--branch-from", str(generation), "--branch-manifest", str(manifest), "--minutes", "18"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "sentinel: driver reached case loading" in err, err[-2000:]
    assert "must be empty" not in err


def test_only_the_runs_own_scratch_is_tolerated(tmp_path):
    generation, _ = _generation(tmp_path)
    out = tmp_path / "branch"
    (out / ".scratch").mkdir(parents=True)
    (out / ".scratch" / "transient.nc").write_text("x")
    validate_paths(generation, out, None, out / ".scratch")
    with pytest.raises(ValueError, match="must be empty"):
        validate_paths(generation, out, None)  # no scratch declared: refused as before
    with pytest.raises(ValueError, match="must be empty"):
        validate_paths(generation, out, None, tmp_path / "elsewhere")
    (out / "wrfout_d01_2026-01-01_00:00:00").write_text("x")
    with pytest.raises(ValueError, match="must be empty"):
        validate_paths(generation, out, None, out / ".scratch")
    link = tmp_path / "other"
    link.mkdir()
    (out / "wrfout_d01_2026-01-01_00:00:00").unlink()
    (out / "alias").symlink_to(link)
    with pytest.raises(ValueError, match="must be empty"):
        validate_paths(generation, out, None, out / "alias")
