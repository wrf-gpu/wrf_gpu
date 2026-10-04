"""CPU-only product CLI selection; forecast execution is stubbed."""
import json
from pathlib import Path

import pytest
from gpuwrf import cli
from gpuwrf.integration import nested_pipeline as pipeline


@pytest.mark.parametrize("option, expected", [(None, True), ("--no-aot-prefetch", False),
                                             ("--aot-prefetch", True)])
def test_cli_passes_prefetch_default_and_opt_out(tmp_path, monkeypatch, capsys, option, expected):
    fixture = Path(__file__).parent / "fixtures/wn3_20260227_cadence.namelist"
    (tmp_path/"namelist.input").write_bytes(fixture.read_bytes())
    configs = []
    def execute(config):
        configs.append(config)
        return {"verdict": "PIPELINE_GREEN", "metadata": {}}
    monkeypatch.setattr(pipeline, "execute_nested_pipeline", execute)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    args = ["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path/"out"),
            "--domains-from-namelist", "--hours", "1", "--force-gpu-run"]
    if option:
        args.append(option)
    assert cli.main(args) == 0
    assert len(configs) == 1 and configs[0].aot_prefetch is expected
    capsys.readouterr()


@pytest.mark.parametrize("option, expected", [(None, True), ("--no-aot-prefetch", False)])
def test_nested_dry_run_reports_prefetch_choice(tmp_path, capsys, option, expected):
    fixture = Path(__file__).parent / "fixtures/wn3_20260227_cadence.namelist"
    (tmp_path/"namelist.input").write_bytes(fixture.read_bytes())
    args = ["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path/"out"),
            "--domains-from-namelist", "--hours", "1", "--dry-run"]
    if option:
        args.append(option)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["aot_prefetch"] is expected
