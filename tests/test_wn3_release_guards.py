"""WN3 release guards (review-writer false passes): WRF globals on EVERY paired frame + the full CPU-WRF header (scorer);
twin manifest attests the MEASURED source commit/tree and the actual CLI input files before writing anything."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = _load("wn3_score")
GLOBALS = {name: 1 for name in score.REQUIRED_GLOBAL_ATTRS}
RAMP = np.arange(8, dtype="f4")


def _frame(path: Path, attrs: dict, shift: float = 0.0) -> Path:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 8)
        for name, values in {"T2": RAMP + shift, "SMOIS": RAMP * 0.1, "W": RAMP - 3}.items():
            ds.createVariable(name, "f4", ("x",))[:] = values
        for key, value in attrs.items():
            ds.setncattr(key, value)
    return path


def _pair(tmp_path: Path, cpu_attrs: dict = GLOBALS, frames: int = 3):
    cpu = [_frame(tmp_path / f"cpu_{i}.nc", cpu_attrs, i) for i in range(frames)]
    gpu = [_frame(tmp_path / f"gpu_{i}.nc", cpu_attrs, i) for i in range(frames)]
    return cpu, gpu


def test_full_header_on_every_frame_passes(tmp_path):
    cpu, gpu = _pair(tmp_path, {**GLOBALS, "TITLE": "OUTPUT FROM WRF V4", "EXTRA_WRF_GLOBAL": "x"})
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] and result["global_attrs"]["by_frame"] == {} and result["global_attrs"]["frames_checked"] == 3


@pytest.mark.parametrize("frame", [0, 1, 2])
def test_required_global_missing_on_any_single_frame_fails(tmp_path, frame):
    cpu, gpu = _pair(tmp_path)
    with netCDF4.Dataset(gpu[frame], "r+") as ds:
        ds.delncattr("PARENT_ID")
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] is False
    assert result["global_attrs"]["by_frame"] == {gpu[frame].name: {"missing_vs_cpu": ["PARENT_ID"], "missing_required": ["PARENT_ID"]}}
    assert result["global_attrs"]["missing_required"] == ["PARENT_ID"]


def test_cpu_extra_global_missing_from_gpu_fails(tmp_path):
    cpu, gpu = _pair(tmp_path)
    for path in cpu:
        with netCDF4.Dataset(path, "r+") as ds:
            ds.EXTRA_WRF_GLOBAL = "frozen CPU header"
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] is False
    assert result["global_attrs"]["missing_vs_cpu"] == ["EXTRA_WRF_GLOBAL"] and not result["global_attrs"]["missing_required"]
    assert sorted(result["global_attrs"]["by_frame"]) == sorted(p.name for p in gpu)


def test_cpu_extra_global_missing_on_last_frame_only_fails(tmp_path):
    cpu, gpu = _pair(tmp_path, {**GLOBALS, "EXTRA_WRF_GLOBAL": "x"})
    with netCDF4.Dataset(gpu[-1], "r+") as ds:
        ds.delncattr("EXTRA_WRF_GLOBAL")
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] is False and list(result["global_attrs"]["by_frame"]) == [gpu[-1].name]


def test_documented_optional_global_may_be_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "OPTIONAL_CPU_GLOBALS", {"EXTRA_WRF_GLOBAL": "test-only optional"})
    cpu, gpu = _pair(tmp_path)
    for path in cpu:
        with netCDF4.Dataset(path, "r+") as ds:
            ds.EXTRA_WRF_GLOBAL = "x"
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] and result["global_attrs"]["optional_missing_allowed"] == {"EXTRA_WRF_GLOBAL": "test-only optional"}


def test_required_minimum_is_never_optional_via_cpu_absence(tmp_path):
    # CPU lacking a required name does not excuse the GPU: the alisios minimum is checked independently.
    cpu, gpu = _pair(tmp_path, {k: v for k, v in GLOBALS.items() if k != "DT"})
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] is False and result["global_attrs"]["missing_required"] == ["DT"]


@pytest.mark.parametrize("cpu_n,gpu_n", [(3, 2), (0, 0)])
def test_unequal_or_empty_pairing_fails(tmp_path, cpu_n, gpu_n):
    cpu, gpu = _pair(tmp_path)
    result = score.output_integrity(cpu[:cpu_n], gpu[:gpu_n])
    assert result["pass"] is False and result["pairing_ok"] is False


# ---- twin manifest: measured source + measured inputs (scripts/wn3_twin_manifest.py) ----
manifest_mod = _load("wn3_twin_manifest")
ISSUE = "20260227_18z"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def world(tmp_path, monkeypatch):
    snap = tmp_path / "snapshot"
    (snap / "src" / "gpuwrf").mkdir(parents=True)
    (snap / "scripts").mkdir()
    (snap / "src" / "gpuwrf" / "__init__.py").write_text("x = 1\n")
    (snap / "scripts" / "wn3_forecast.py").write_text("# harness\n")
    _git(snap, "init", "-q")
    _git(snap, "add", "-A")
    _git(snap, "commit", "-q", "-m", "snap")
    server = tmp_path / "server"
    case = server / f"wg_{ISSUE}_a1" / "run" / "run"
    case.mkdir(parents=True)
    for name in (*manifest_mod.INPUTS, "namelist.input"):
        (case / name).write_bytes(f"server {name}".encode())
    staged = tmp_path / "cases" / f"{ISSUE}_a1"  # what our arms pass: symlinks to the server files
    staged.mkdir(parents=True)
    for name in (*manifest_mod.INPUTS, "namelist.input"):
        (staged / name).symlink_to(case / name)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text("{}")
    monkeypatch.setattr(manifest_mod, "CASES", server)
    monkeypatch.setattr(manifest_mod, "THRESHOLDS", thresholds)
    return {"tmp": tmp_path, "snap": snap, "case": case, "staged": staged,
            "head": _git(snap, "rev-parse", "HEAD"), "tree": _git(snap, "rev-parse", "HEAD:src/gpuwrf")}


def _arm(w, launcher: bool = False, **over):
    arm = w["tmp"] / ("arm_l" if launcher else "arm_h")
    (arm / "wrfout" / "proofs").mkdir(parents=True, exist_ok=True)
    for nest in ("d01", "d02", "d03"):
        for h in (0, 1):
            (arm / "wrfout" / f"wrfout_{nest}_2026-02-28_0{h}:00:00").write_bytes(b"frame")
    argv = ["run", "--input-dir", str(over.pop("input_dir", w["staged"])), "--max-dom", "3", "--hours", "1"]
    if launcher:
        receipt = {"schema": "gpuwrf-parallel-case-v1", "rc": 0, "command": argv, "env": {}, "outputs": [],
                   "source": {"gpuwrf_path": str(w["snap"] / "src" / "gpuwrf"), "git_head": w["head"],
                              "src_tree": w["tree"], "dirty": False}}
        (arm / "wrfout" / "proofs" / "nested_pipeline_run.json").write_text(json.dumps(
            {"device": "cuda:0", "namelist_path": str(Path(argv[2]) / "namelist.input"), "metadata": {"domains": {}}}))
        if "source" in over:
            receipt["source"] = over.pop("source")
    else:
        receipt = {"rc": 0, "device": {"platform": "gpu", "kind": "test"}, "git_head": w["head"], "src_tree": w["tree"],
                   "argv": [str(w["snap"] / "scripts" / "wn3_forecast.py")], "cli_argv": argv, "runtime_namelist": {},
                   "derived": {"segments": [], "segment_rate_s_per_fc_h": []}}
        (arm / "arm_env.txt").write_text("JAX_PLATFORMS=cuda\n")
    receipt.update(over)
    (arm / "receipt.json").write_text(json.dumps(receipt))
    return arm


def _run(w, arm, monkeypatch):
    out = w["tmp"] / f"manifest_{arm.name}.json"
    monkeypatch.setattr(sys, "argv", ["m", ISSUE, str(arm), "1", str(w["snap"]), str(out), "--partial"])
    return out


def _refused(w, arm, monkeypatch, match):
    out = _run(w, arm, monkeypatch)
    with pytest.raises(SystemExit, match=match):
        manifest_mod.main()
    assert not out.exists() and not out.with_name(out.stem + "_namelist_diff.json").exists(), "wrote before attesting"


@pytest.mark.parametrize("launcher", [False, True])
def test_attested_run_writes_manifest(world, monkeypatch, launcher):
    out = _run(world, _arm(world, launcher), monkeypatch)
    manifest_mod.main()
    m = json.loads(out.read_text())
    assert m["gpu_commit"] == world["head"] and m["gpu_source_attestation"]["src_tree"] == world["tree"]
    files = m["gpu_input_attestation"]["files"]
    assert all(v["sha256"] == v["server_sha256"] for v in files.values()) and len(files) == 8
    assert m["input_sha256"] == {n: files[n]["sha256"] for n in manifest_mod.INPUTS}


def test_byte_identical_input_copies_are_accepted(world, monkeypatch):
    copies = world["tmp"] / "copies"
    copies.mkdir()
    for name in (*manifest_mod.INPUTS, "namelist.input"):
        (copies / name).write_bytes((world["case"] / name).read_bytes())
    _run(world, _arm(world, input_dir=copies), monkeypatch)
    manifest_mod.main()


@pytest.mark.parametrize("launcher", [False, True])
def test_refuses_measured_commit_different_from_snapshot(world, monkeypatch, launcher):
    zero = "0" * 40
    over = {"source": {"gpuwrf_path": str(world["snap"] / "src" / "gpuwrf"), "git_head": zero, "src_tree": world["tree"],
                       "dirty": False}} if launcher else {"git_head": zero}
    _refused(world, _arm(world, launcher, **over), monkeypatch, "source commit")


def test_refuses_measured_src_tree_different_from_snapshot(world, monkeypatch):
    _refused(world, _arm(world, src_tree="f" * 40), monkeypatch, "source commit")


def test_refuses_dirty_snapshot(world, monkeypatch):
    (world["snap"] / "src" / "gpuwrf" / "__init__.py").write_text("x = 2\n")
    _refused(world, _arm(world), monkeypatch, "uncommitted")


def test_refuses_harness_script_outside_snapshot(world, monkeypatch):
    _refused(world, _arm(world, argv=["/elsewhere/scripts/wn3_forecast.py"]), monkeypatch, "not inside the snapshot")


@pytest.mark.parametrize("source", [None, "elsewhere", "dirty"])
def test_refuses_launcher_without_matching_source_record(world, monkeypatch, source):
    rec = {"gpuwrf_path": str(world["snap"] / "src" / "gpuwrf"), "git_head": world["head"], "src_tree": world["tree"], "dirty": False}
    rec = None if source is None else {**rec, "gpuwrf_path": "/elsewhere/src/gpuwrf"} if source == "elsewhere" else {**rec, "dirty": True}
    _refused(world, _arm(world, True, source=rec), monkeypatch, "source|dirty|imported gpuwrf")


@pytest.mark.parametrize("launcher", [False, True])
def test_refuses_measured_input_dir_different_from_server_inputs(world, monkeypatch, launcher):
    wrong = world["tmp"] / "different-input"
    wrong.mkdir()
    _refused(world, _arm(world, launcher, input_dir=wrong), monkeypatch, "measured CLI input path")


def test_refuses_one_input_with_different_bytes(world, monkeypatch):
    bad = world["tmp"] / "bad"
    bad.mkdir()
    for name in (*manifest_mod.INPUTS, "namelist.input"):
        (bad / name).write_bytes((world["case"] / name).read_bytes() + (b"!" if name == "wrfbdy_d01" else b""))
    _refused(world, _arm(world, input_dir=bad), monkeypatch, "wrfbdy_d01")


def test_refuses_receipt_without_input_dir(world, monkeypatch):
    _refused(world, _arm(world, cli_argv=["run", "--max-dom", "3"]), monkeypatch, "no --input-dir")


def test_launcher_receipt_records_the_imported_source():
    spec = importlib.util.spec_from_file_location("pc_source", REPO / "scripts" / "parallel_cases.py")
    pc = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "src"))
    try:
        spec.loader.exec_module(pc)
        src = pc.source_ref()
    finally:
        sys.path.remove(str(REPO / "src"))
    assert Path(src["gpuwrf_path"]).resolve() == (REPO / "src" / "gpuwrf").resolve()
    if (REPO / ".git").exists():
        assert src["git_head"] == _git(REPO, "rev-parse", "HEAD") and src["src_tree"] == _git(REPO, "rev-parse", "HEAD:src/gpuwrf")
