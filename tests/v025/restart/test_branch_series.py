"""scripts/branch_series.py: declared manifests, argv and fail-closed series handling (CPU, structural)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.runtime.checkpoint_branch import branch_identity, load_manifest
from gpuwrf.surface_params.controls import controls_digest, save_controls

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("branch_series", ROOT / "scripts" / "branch_series.py")
series = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(series)


def _controls(tmp_path, name, factors):
    base = {"d01": np.full((3, 4), 0.1, np.float32), "d02": np.full((5, 6), 0.2, np.float32)}
    fields = {d: {"z0mvt": (a * np.float32(factors.get(d, 1.0))).astype(np.float32)} for d, a in base.items()}
    path = tmp_path / f"{name}.npz"
    save_controls(path, fields)
    return path, {d: controls_digest(v) for d, v in fields.items()}


@pytest.fixture
def case(tmp_path):
    neutral, neutral_digest = _controls(tmp_path, "neutral", {})
    variant, variant_digest = _controls(tmp_path, "variant", {"d02": 2.0})
    identity = {"source_sha256": "s", "inputs": {"wrfinput_d01": "i"}, "output_dir": str(tmp_path / "parent"),
                "namelists": {"d01": "n", "d02": "n"}, "surface_controls": neutral_digest}
    generation = tmp_path / "ckpt" / "generation-1"
    generation.mkdir(parents=True)
    manifest = {"branch_schema_version": 1, "identity": identity, "sha256": "payload",
                "domains": ["d01", "d02"], "own_steps": {"d01": 10, "d02": 30}}
    (generation / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "input").mkdir()
    spec = {"schema": series.SCHEMA, "input_dir": "input", "branch_from": "ckpt/generation-1", "out_root": "out",
            "end": {"minutes": 27}, "parent_surface_controls": "neutral.npz", "cli_args": ["--domains-from-namelist"],
            "variants": [{"name": "null"}, {"name": "z0", "surface_controls": "variant.npz"}]}
    path = tmp_path / "series.json"
    path.write_text(json.dumps(spec))
    return {"tmp": tmp_path, "spec": spec, "path": path, "parent": manifest, "generation": generation,
            "neutral_digest": neutral_digest, "variant_digest": variant_digest}


def test_declared_manifest_names_exactly_the_changed_domains_and_admits(case):
    spec = series.load_spec(case["path"])
    parent = case["parent"]
    assert series.declared_manifest(parent["identity"], None) == {"schema_version": 1}
    assert series.declared_manifest(parent["identity"], spec["parent_surface_controls"]) == {"schema_version": 1}
    declared = series.declared_manifest(parent["identity"], spec["variants"][1]["surface_controls"])
    assert declared == {"schema_version": 1, "parameters": {"surface_controls": {"d02": case["variant_digest"]["d02"]}}}
    # The written declaration is accepted by the real admission code, and only for the requested values.
    path = case["tmp"] / "m.json"
    path.write_text(json.dumps(declared))
    current = {**parent["identity"], "output_dir": str(case["tmp"] / "child"), "surface_controls": case["variant_digest"]}
    assert branch_identity(parent, current, load_manifest(path), case["generation"])["surface_controls"] == case["variant_digest"]
    undeclared = {**current, "surface_controls": {**case["variant_digest"], "d01": "f" * 64}}
    with pytest.raises(ValueError, match="undeclared"):
        branch_identity(parent, undeclared, load_manifest(path), case["generation"])
    with pytest.raises(ValueError, match="undeclared"):  # the null manifest does not admit changed values
        branch_identity(parent, current, {"schema_version": 1}, case["generation"])


def test_signature_and_stale_parent_guards(case, tmp_path):
    spec = series.load_spec(case["path"])
    parent = case["parent"]["identity"]
    neutral = spec["parent_surface_controls"]
    one_domain, _ = _controls(tmp_path, "d01only", {})
    from gpuwrf.surface_params.controls import load_controls, save_controls
    fields, _ = load_controls(one_domain)
    save_controls(one_domain, {"d01": fields["d01"]})  # the `expand --set z0mvt=1` trap: d01 only
    with pytest.raises(ValueError, match="does not reproduce"):  # stale parent file
        series.declared_manifest(parent, spec["variants"][1]["surface_controls"], one_domain)
    with pytest.raises(ValueError, match="does not reproduce"):
        series.declared_manifest(parent, neutral, one_domain)
    with pytest.raises(ValueError, match="signature"):
        series.declared_manifest(parent, one_domain, neutral)
    extra, _ = _controls(tmp_path, "extra", {})
    f2, _ = load_controls(extra)
    save_controls(extra, {d: {**v, "laim": np.ones((12, *v["z0mvt"].shape), np.float32)} for d, v in f2.items()})
    with pytest.raises(ValueError, match="signature"):
        series.declared_manifest(parent, extra, neutral)
    assert series.declared_manifest(parent, neutral, neutral) == {"schema_version": 1}


def test_parent_without_controls_and_foreign_domains_refuse(case):
    spec = series.load_spec(case["path"])
    no_family = {k: v for k, v in case["parent"]["identity"].items() if k != "surface_controls"}
    with pytest.raises(ValueError, match="not enabled"):
        series.declared_manifest(no_family, spec["variants"][1]["surface_controls"])
    one_domain = {**case["parent"]["identity"], "surface_controls": {"d01": case["neutral_digest"]["d01"]}}
    with pytest.raises(ValueError, match="outside the parent"):
        series.declared_manifest(one_domain, spec["variants"][1]["surface_controls"])


def test_argv_per_variant(case):
    spec = series.load_spec(case["path"])
    null, z0 = (series.variant_argv(spec, v, Path("/m.json")) for v in spec["variants"])
    tmp = case["tmp"].resolve()
    assert null == ["run", "--input-dir", str(tmp / "input"), "--output-dir", str(tmp / "out/null/wrfout"),
                    "--scratch-dir", str(tmp / "out/null/scratch"), "--branch-from", str(case["generation"].resolve()), "--branch-manifest", "/m.json",
                    "--minutes", "27", "--surface-controls", str(tmp / "neutral.npz"), "--domains-from-namelist"]
    assert z0[z0.index("--surface-controls") + 1] == str(tmp / "variant.npz")


@pytest.mark.parametrize("mutation, match", [
    (lambda s: s["cli_args"].append("--hours"), "cli_args"),
    (lambda s: s["cli_args"].append("--resume-checkpoint=x"), "may not set"),
    (lambda s: s["variants"].append({"name": "null"}), "unique"),
    (lambda s: s["variants"].append({"name": "../x"}), "unique"),
    (lambda s: s["variants"][1].update(manifest="m.json"), "not both"),
    (lambda s: s.update(end={"hours": 1, "minutes": 60}), "exactly one"),
    (lambda s: s.update(parent_surface_controls=None), "already be enabled"),
    (lambda s: s.update(branch_from="ckpt"), "not a checkpoint generation"),
    (lambda s: s.update(extra=1), "unknown series keys"),
])
def test_spec_refusals(case, mutation, match):
    spec = json.loads(case["path"].read_text())
    mutation(spec)
    (case["tmp"] / "m.json").write_text("{}")
    case["path"].write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        series.load_spec(case["path"])


def _fake_payload(argv):
    out = Path(argv[argv.index("--output-dir") + 1])
    (out / "proofs").mkdir(parents=True, exist_ok=True)
    (out / "proofs" / "nested_pipeline_run.json").write_text(json.dumps({"verdict": "PIPELINE_GREEN", "wall_s": 1.0}))


def test_process_mode_runs_every_variant_and_writes_receipt(case, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        _fake_payload(cmd)
        receipt = Path(cmd[cmd.index("--receipt") + 1])
        receipt.write_text(json.dumps({"platform": "gpu", "runs": [dict.fromkeys(series._Counters.KEYS, 0)]}))
        return type("P", (), {"returncode": 1 if "z0" in cmd[cmd.index("--output-dir") + 1] else 0})()

    monkeypatch.setattr(series.subprocess, "run", run)
    assert series.main([str(case["path"])]) == 1  # one variant failed: the series reports it, others still ran
    assert [c[0][1:3] for c in calls] == [["-B", str(series.ACCT)]] * 2  # each child under the counting wrapper
    receipt = json.loads((case["tmp"] / "out" / "series_receipt.json").read_text())
    assert [(v["name"], v["rc"], v["status"]) for v in receipt["variants"]] == [("null", 0, "OK"), ("z0", 1, "FAILED")]
    assert json.loads((case["tmp"] / "out/z0/branch_manifest.json").read_text())["parameters"]["surface_controls"] == {
        "d02": case["variant_digest"]["d02"]}
    assert json.loads((case["tmp"] / "out/null/branch_manifest.json").read_text()) == {"schema_version": 1}
    assert receipt["variants"][0]["run"]["verdict"] == "PIPELINE_GREEN"
    assert receipt["variants"][0]["platform"] == "gpu" and receipt["variants"][0]["jax_counters"]["fresh_compiles"] == 0
    assert series.main([str(case["path"])]) == 2  # existing variant directories are never reused


def test_inprocess_mode_fails_closed_and_counts(case, monkeypatch):
    import gpuwrf.cli as cli
    seen = []

    def fake_main(argv):
        seen.append(argv)
        _fake_payload(argv)
        return 1

    monkeypatch.setattr(cli, "main", fake_main)
    spec = json.loads(case["path"].read_text())
    spec["variants"].append({"name": "third"})
    case["path"].write_text(json.dumps(spec))
    assert series.main([str(case["path"]), "--mode", "inprocess"]) == 1
    receipt = json.loads((case["tmp"] / "out" / "series_receipt.json").read_text())
    assert len(seen) == 1
    assert [v["status"] for v in receipt["variants"]] == ["FAILED", "NOT_RUN", "NOT_RUN"]
    assert receipt["variants"][0]["jax_counters"]["cache_misses"] == 0
    assert "--force-gpu-run" not in seen[0]  # CPU: no preflight override


def test_dry_run_writes_nothing(case, capsys):
    assert series.main([str(case["path"]), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [p["name"] for p in plan] == ["null", "z0"]
    assert plan[1]["declared"]["parameters"]["surface_controls"] == {"d02": case["variant_digest"]["d02"]}
    assert not (case["tmp"] / "out").exists()


# ---- v50-branch FIX1 (critic 2026-11-11) ----------------------------------------------------------------------
def _two_variant_cuda(case, monkeypatch, seen):
    import gpuwrf.backends.platform as platform
    import gpuwrf.cli as cli

    def fake_main(argv):
        seen.append(list(argv))
        _fake_payload(argv)
        return 0

    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.setattr(platform, "target_backend", lambda: "cuda")


def test_inprocess_cuda_refuses_an_env_bit_without_the_real_lock(case, monkeypatch, capsys):
    seen = []
    _two_variant_cuda(case, monkeypatch, seen)
    monkeypatch.setenv("GPUWRF_GPU_LOCK_HELD", "1")  # the bit alone is no proof: no lock fd, no holder token
    monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", "987")
    monkeypatch.delenv("GPUWRF_FORCE_GPU_RUN", raising=False)
    assert series.main([str(case["path"]), "--mode", "inprocess"]) == 2
    assert seen == [] and "verified GPU lock" in capsys.readouterr().err


def test_inprocess_cuda_never_forces_the_preflight(case, monkeypatch):
    import gpuwrf.runtime.gpu_preflight as preflight
    seen = []
    _two_variant_cuda(case, monkeypatch, seen)
    monkeypatch.setattr(preflight, "gpu_lock_status", lambda *a, **k: {"ok": True, "reason": "", "fd": "9", "lock_file": "/x"})
    assert series.main([str(case["path"]), "--mode", "inprocess"]) == 0
    assert len(seen) == 2 and all("--force-gpu-run" not in argv for argv in seen)
    receipt = json.loads((case["tmp"] / "out" / "series_receipt.json").read_text())
    assert receipt["gpu_lock"]["ok"] is True and not any("forced_gpu_preflight" in v for v in receipt["variants"])


def test_force_env_is_refused(case, monkeypatch):
    monkeypatch.setenv("GPUWRF_FORCE_GPU_RUN", "1")
    assert series.main([str(case["path"])]) == 2


@pytest.mark.parametrize("args", [["--branch-fr", "x"], ["--branch-man", "x"], ["--output-d", "x"], ["--dry"],
                                  ["--dry-run"], ["--checkpoint-dir", "x"], ["--checkpoint-interval-steps", "5"],
                                  ["--feedback"], ["--force-gpu-run"], ["--proof-dir", "x"], ["--hours", "3"],
                                  ["--output-dir=x"], ["--scratch-dir", "x"], ["--extend-run"], ["--bogus"]])
def test_cli_args_go_through_the_real_parser_without_abbreviations(case, args):
    spec = json.loads(case["path"].read_text())
    spec["cli_args"] += args
    case["path"].write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="cli_args"):
        series.load_spec(case["path"])


def test_allowlisted_cli_args_pass(case):
    spec = json.loads(case["path"].read_text())
    spec["cli_args"] = ["--domains-from-namelist", "--emit-initial-history", "--max-dom", "3"]
    case["path"].write_text(json.dumps(spec))
    assert series.load_spec(case["path"])["cli_args"][-1] == "3"


def test_feedback_parent_is_refused(case):
    parent = dict(case["parent"])
    parent["identity"] = {**parent["identity"], "feedback": True}
    (case["generation"] / "manifest.json").write_text(json.dumps(parent))
    assert series.main([str(case["path"]), "--dry-run"]) == 2


def test_receipt_keeps_model_walls(case, tmp_path):
    out = tmp_path / "v"
    (out / "wrfout" / "proofs").mkdir(parents=True)
    (out / "wrfout" / "proofs" / "nested_pipeline_run.json").write_text(json.dumps(
        {"verdict": "PIPELINE_GREEN", "wall_clock_total_s": 12.5, "wall_clock_forecast_only_s": 10.0, "device": "cuda:0"}))
    run = series._payload_summary(out)["run"]
    assert run["wall_clock_total_s"] == 12.5 and run["wall_clock_forecast_only_s"] == 10.0 and run["device"] == "cuda:0"


def test_counters_report_cache_disabled_fresh_compiles():
    counters = series._Counters()
    counters.duration("/jax/core/compile/backend_compile_duration", 0.1)  # compiled, no cache request issued
    counters.event("/jax/compilation_cache/task_disabled_cache")
    counts = counters.take()
    assert counts["cache_misses"] == 0 and counts["fresh_compiles"] == 1 and counts["cache_disabled"] == 1


def test_process_mode_child_counters_reach_the_receipt(case, monkeypatch):
    def run(cmd, **kwargs):
        _fake_payload(cmd)
        Path(cmd[cmd.index("--receipt") + 1]).write_text(json.dumps({"platform": "gpu", "runs": [
            {**dict.fromkeys(series._Counters.KEYS, 0), "backend_compile_events": 2, "fresh_compiles": 2, "cache_disabled": 2}]}))
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(series.subprocess, "run", run)
    assert series.main([str(case["path"])]) == 0
    rows = json.loads((case["tmp"] / "out" / "series_receipt.json").read_text())["variants"]
    assert all(r["jax_counters"]["fresh_compiles"] == 2 and r["jax_counters"]["cache_disabled"] == 2 for r in rows)
