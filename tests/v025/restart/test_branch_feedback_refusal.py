"""Checkpoint branching is qualified for one-way nesting only: feedback-on parents/branches are refused (FIX1-6)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf.runtime.checkpoint_branch import SCHEMA_VERSION, branch_identity


def _parent(tmp_path, **identity):
    ident = {"source_sha256": "s", "output_dir": str(tmp_path / "parent"), "feedback": False, **identity}
    generation = tmp_path / "generation-1"
    generation.mkdir()
    manifest = {"branch_schema_version": SCHEMA_VERSION, "identity": ident, "sha256": "p", "domains": ["d01", "d02"],
                "own_steps": {"d01": 10, "d02": 30}}
    (generation / "manifest.json").write_text(json.dumps(manifest))
    return manifest, generation


@pytest.mark.parametrize("parent_extra, current_extra", [({"feedback": True}, {"feedback": True}),
                                                         ({"feedback_schema": {"d01": {}}}, {}),
                                                         ({}, {"feedback": True})])
def test_branch_identity_refuses_two_way_feedback(tmp_path, parent_extra, current_extra):
    manifest, generation = _parent(tmp_path, **parent_extra)
    current = {**manifest["identity"], **current_extra, "output_dir": str(tmp_path / "child")}
    with pytest.raises(ValueError, match="one-way nesting only"):
        branch_identity(manifest, current, {"schema_version": 1}, generation)


def test_one_way_identity_still_admits(tmp_path):
    manifest, generation = _parent(tmp_path)
    current = {**manifest["identity"], "output_dir": str(tmp_path / "child")}
    assert branch_identity(manifest, current, {"schema_version": 1}, generation)["output_dir"] == current["output_dir"]


def test_nested_driver_refuses_feedback_branch_before_any_work(tmp_path, monkeypatch):
    from gpuwrf.integration import nested_pipeline
    monkeypatch.setattr(nested_pipeline, "_admit_cammg_inputs", lambda config: None)
    config = SimpleNamespace(max_dom=2, branch_from=tmp_path / "g", branch_manifest=tmp_path / "m.json",
                             resume_checkpoint=None, feedback=True, output_dir=tmp_path / "out",
                             checkpoint_dir=None, scratch_dir=None, proof_dir=tmp_path / "out" / "proofs")
    with pytest.raises(ValueError, match="one-way nesting only"):
        nested_pipeline._execute_nested_pipeline(config, _prefetch_handles=[])
    assert not (tmp_path / "out").exists()
