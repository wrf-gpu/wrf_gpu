"""Release tooling in scripts/release (private-only; skipped in the curated public tree). CPU, no network."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "scripts" / "release"
if not (TOOLS / "curate.py").is_file():
    pytest.skip("release tooling is private-only (not in the curated public tree)", allow_module_level=True)


def _load(name):
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(f"_release_{name}", TOOLS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


curate = _load("curate")
verify = _load("verify")
readme_lint = _load("readme_lint")


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def test_no_release_is_hard_coded():
    pins = ('default="0.3.4"', "'0.3.4'", '"0.3.4"', "v034-release-prep", "wrf_gpu2_lanes/rel034", "taskset -c 24,25",
            "curated v0.3.4")
    for path in TOOLS.iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert not [pin for pin in pins if pin in text], path.name
    for args in ([sys.executable, str(TOOLS / "verify.py"), "--pub", ".", "--private", "HEAD"],
                 [sys.executable, str(TOOLS / "curate.py"), "--private", "HEAD", "--out", ".", "--work", "w"],
                 ["bash", str(TOOLS / "final.sh"), "--private", "HEAD"]):
        rc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        assert rc.returncode == 2 and "version" in rc.stderr.lower(), args


def test_tooling_and_build_hook_layout():
    assert "setup.py" in curate.ROOT_KEEP  # the wheel build hook must reach the public root
    assert curate.PRIVATE_ONLY == ("scripts/release/",)
    committed = json.loads(_git(ROOT, "show", f"HEAD:{curate.ASSETS_MANIFEST}"))["assets"]
    assert curate.remote_assets("HEAD") == {p: i for p, i in committed.items() if i["kind"] == "remote"}


@pytest.fixture
def toy_repo(tmp_path, monkeypatch):
    """A private repo with one remote table and one bundle file pinned in the assets manifest."""
    repo = tmp_path / "priv"
    (repo / "src/gpuwrf/physics/scheme").mkdir(parents=True)
    (repo / "scripts").mkdir()
    big = b"T" * 4096
    (repo / "src/gpuwrf/physics/scheme/big.npz").write_bytes(big)
    (repo / "scripts/small.py").write_text("x = 1\n")
    manifest = dict(schema="gpuwrf-runtime-assets-v1", url_template="u", max_bundle_bytes=10, assets={
        "src/gpuwrf/physics/scheme/big.npz": dict(kind="remote", asset="scheme-big.npz", bytes=len(big),
                                                 sha256=hashlib.sha256(big).hexdigest()),
        "scripts/small.py": dict(kind="bundle", asset="small.py", bytes=6,
                                 sha256=hashlib.sha256(b"x = 1\n").hexdigest())})
    (repo / curate.ASSETS_MANIFEST).write_text(json.dumps(manifest))
    (repo / "setup.py").write_text("")
    _git(tmp_path, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "toy")
    monkeypatch.setattr(curate, "PRIV_REPO", str(repo))
    return repo


def test_release_assets_are_exported_and_verified(toy_repo, tmp_path):
    remote = curate.remote_assets("HEAD")
    out = tmp_path / "assets"
    rows = curate.export_release_assets("HEAD", remote, out)
    assert set(rows) == {"scheme-big.npz"} and (out / "scheme-big.npz").read_bytes() == b"T" * 4096
    assert (out / "SHA256SUMS").read_text() == f"{rows['scheme-big.npz']['sha256']}  scheme-big.npz\n"
    pub = toy_repo  # a public tree carrying the same manifest
    good = verify.release_assets(pub, out)
    assert not good["missing"] and not good["bad"] and not good["unexpected"] and good["remote"] == 1
    (out / "scheme-big.npz").write_bytes(b"T" * 4095 + b"X")
    assert verify.release_assets(pub, out)["bad"] == ["scheme-big.npz"]
    (out / "scheme-big.npz").unlink()
    (out / "stray.bin").write_bytes(b"")
    bad = verify.release_assets(pub, out)
    assert bad["missing"] == ["scheme-big.npz"] and bad["unexpected"] == ["stray.bin"]
    assert verify.release_assets(pub, None)["missing"] == ["scheme-big.npz"]


def test_export_refuses_a_blob_that_differs_from_its_pin(toy_repo, tmp_path):
    remote = curate.remote_assets("HEAD")
    remote["src/gpuwrf/physics/scheme/big.npz"]["sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="differs from its manifest pin"):
        curate.export_release_assets("HEAD", remote, tmp_path / "assets")


def test_packaging_gate_needs_setup_py_and_pinned_bundles(toy_repo):
    assert verify.packaging(toy_repo) == dict(setup_py=True, manifest=True, bundle_bad=[])
    (toy_repo / "scripts/small.py").write_text("x = 2\n")
    assert verify.packaging(toy_repo)["bundle_bad"] == ["scripts/small.py"]
    (toy_repo / "setup.py").unlink()
    assert verify.packaging(toy_repo)["setup_py"] is False


def test_tree_parity_reports_remote_tables_as_release_assets(toy_repo, tmp_path, monkeypatch):
    pub = tmp_path / "pub"
    (pub / "src/gpuwrf").mkdir(parents=True)
    (pub / curate.ASSETS_MANIFEST).write_bytes((toy_repo / curate.ASSETS_MANIFEST).read_bytes())
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    rep = verify.tree_parity(pub, "HEAD", "src/gpuwrf", set(curate.remote_assets("HEAD")))
    assert rep["release_assets"] == ["src/gpuwrf/physics/scheme/big.npz"] and rep["missing"] == []
    rep = verify.tree_parity(pub, "HEAD", "src/gpuwrf", set())
    assert rep["missing"] == ["src/gpuwrf/physics/scheme/big.npz"]  # without the manifest it is a real gap


GATES = ("verify", "committed", "big", "uncommitted", "import", "collect", "help", "docsite", "wheel", "deptests")


@pytest.mark.parametrize("failing", GATES)
def test_final_gate_exits_nonzero_for_each_failed_gate(failing):
    gates = {name: 0 for name in GATES}
    ok = subprocess.run([sys.executable, str(TOOLS / "gate.py"), *(f"{k}={v}" for k, v in gates.items()),
                         "--info", "assets=20"], capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.startswith("GATE PASS")
    gates[failing] = 1
    bad = subprocess.run([sys.executable, str(TOOLS / "gate.py"), *(f"{k}={v}" for k, v in gates.items())],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and f"failed={failing}" in bad.stdout


def test_final_sh_exit_status_is_the_gate_and_forwards_base():
    text = (TOOLS / "final.sh").read_text()
    tail = text[text.index('python3 "$TOOLS/gate.py"'):]
    assert 'exit "${PIPESTATUS[0]}"' in tail and tail.strip().endswith('exit "${PIPESTATUS[0]}"')
    for gate in ("verify=$VG", "committed=$CS", "big=$BIG", "uncommitted=", "import=$CI", "collect=$CC", "help=$CL",
                 "docsite=$CD", "wheel=$CW", "deptests=$DT"):
        assert gate in tail, gate
    curate_call = text[text.index('"$TOOLS/curate.py"'):text.index("> \"$R/curate.out\"")]
    assert '--base "$BASE"' in curate_call
    assert subprocess.run([sys.executable, str(TOOLS / "gate.py"), "verify="], capture_output=True).returncode == 1


def test_published_release_records_keep_their_public_bytes(toy_repo, tmp_path, monkeypatch):
    (toy_repo / "docs/release").mkdir(parents=True)
    (toy_repo / "release_notes").mkdir()
    old = b"# v0.3.4 support matrix (published)\n"
    (toy_repo / "docs/release/SUPPORT_MATRIX_v0.3.4.md").write_bytes(old)
    (toy_repo / "release_notes/RELEASE_NOTES_v0.3.4.md").write_bytes(b"notes\n")
    (toy_repo / "docs/release/SUPPORT_MATRIX_v0.4.0.md").write_bytes(b"new\n")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    assert curate.historical_records(curate.ls_tree("HEAD"), "0.4.0") == [
        "docs/release/SUPPORT_MATRIX_v0.3.4.md", "release_notes/RELEASE_NOTES_v0.3.4.md"]
    # src/tests mirror the private commit (tree parity); a non-version token is not a release name
    assert curate.historical_records(["src/gpuwrf/x_v0.3.4.json", "tests/t_v0.3.4.py", "docs/x_v0.3.4x.md"], "0.4.0") == []
    pub = tmp_path / "pub"
    for rel in ("docs/release/SUPPORT_MATRIX_v0.3.4.md", "docs/release/SUPPORT_MATRIX_v0.4.0.md"):
        (pub / rel).parent.mkdir(parents=True, exist_ok=True)
    (pub / "docs/release/SUPPORT_MATRIX_v0.3.4.md").write_bytes(old + b"## appended v0.4 development section\n")
    (pub / "docs/release/SUPPORT_MATRIX_v0.4.0.md").write_bytes(b"new, edited\n")
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))  # verify.historical reads the base through verify.PRIV
    assert verify.historical(pub, "HEAD", "0.4.0") == [
        "docs/release/SUPPORT_MATRIX_v0.3.4.md", "release_notes/RELEASE_NOTES_v0.3.4.md"]
    assert curate.restore_historical(pub, "HEAD", "0.4.0") == [
        "docs/release/SUPPORT_MATRIX_v0.3.4.md", "release_notes/RELEASE_NOTES_v0.3.4.md"]
    assert (pub / "docs/release/SUPPORT_MATRIX_v0.3.4.md").read_bytes() == old
    assert (pub / "docs/release/SUPPORT_MATRIX_v0.4.0.md").read_bytes() == b"new, edited\n"  # current release
    assert verify.historical(pub, "HEAD", "0.4.0") == []


def test_no_remote_table_may_stay_in_the_public_tree(toy_repo, tmp_path):
    pub = tmp_path / "pub"
    (pub / "src/gpuwrf").mkdir(parents=True)
    (pub / curate.ASSETS_MANIFEST).write_bytes((toy_repo / curate.ASSETS_MANIFEST).read_bytes())
    assert verify.remote_in_tree(pub) == []
    (pub / "src/gpuwrf/physics/scheme").mkdir(parents=True)
    (pub / "src/gpuwrf/physics/scheme/big.npz").write_bytes(b"T")
    assert verify.remote_in_tree(pub) == ["src/gpuwrf/physics/scheme/big.npz"]
    curate_src = (TOOLS / "curate.py").read_text()
    assert "kept_identical_in_base" not in curate_src and "base_hash.get(repo_path)" not in curate_src


@dataclasses.dataclass(frozen=True)
class AotMeta:  # pickled under this module's name; compile_bundle reads it without importing gpuwrf
    hlo_sha256: str
    blob_sha256: str
    cheap_key: str
    fingerprint: dict
    in_tree: object = None


def _toy_cache(root, flags=""):
    cache = root / "jax"
    d01 = cache / "aot/0.4.0-cuda_sm120/d01"
    d01.mkdir(parents=True)
    blob = b"ELF-native-blob"
    (d01 / "k_a.xlaexec").write_bytes(blob)
    target = dict(platform="gpu", device_kind="NVIDIA GeForce RTX 5090", compute_capability="12.0", cuda_driver="13000",
                  jax_version="0.10.0", jaxlib_version="0.10.0", x64=True, xla_flags=flags)
    meta = AotMeta("h1", hashlib.sha256(blob).hexdigest(), "a", target, in_tree=range(3))
    (d01 / "k_a.meta").write_bytes(pickle.dumps(meta))
    (d01 / "last_cheap_key").write_text("a")
    (cache / "jit_f-00-cache").write_bytes(b"entry")
    (cache / "jit_f-00-atime").write_bytes(b"1")
    (cache / "cli_memory_plans").mkdir()
    (cache / "cli_memory_plans/k.json").write_text('{"live_peak_bytes": 1}')
    (cache / "xla_gpu_per_fusion_autotune_cache_dir").mkdir()
    (cache / "xla_gpu_per_fusion_autotune_cache_dir/t.textproto").write_text("tune")
    return cache, blob, meta


def test_compile_bundle_seal_is_deterministic_and_check_detects_compiles_outside(toy_repo, tmp_path, capsys):
    cb = _load("compile_bundle")
    cache, blob, meta = _toy_cache(tmp_path)
    seal = ["seal", "--cache", str(cache), "--commit", "HEAD", "--source", str(toy_repo), "--repo", str(toy_repo)]
    assert cb.main(seal + ["--out", str(tmp_path / "b1")]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line == (f"TUPLE commit={_git(toy_repo, 'rev-parse', 'HEAD').strip()} "
                    f"src_tree={_git(toy_repo, 'rev-parse', 'HEAD:src').strip()} "
                    f"gpuwrf_tree={_git(toy_repo, 'rev-parse', 'HEAD:src/gpuwrf').strip()} "
                    f"bundle_sha256={hashlib.sha256((tmp_path / 'b1/gpuwrf-bundle.tar.gz').read_bytes()).hexdigest()}")
    # a second process with another hash seed writes the same bytes (bundle SHA = id, never run-dependent)
    subprocess.run([sys.executable, str(TOOLS / "compile_bundle.py"), *seal, "--out", str(tmp_path / "b2")], check=True,
                   capture_output=True, env={**os.environ, "PYTHONHASHSEED": "12345"})
    assert (tmp_path / "b1/gpuwrf-bundle.tar.gz").read_bytes() == (tmp_path / "b2/gpuwrf-bundle.tar.gz").read_bytes()
    manifest = json.loads((tmp_path / "b1/BUNDLE_MANIFEST.json").read_text())
    assert manifest["aot_meta"]["aot/0.4.0-cuda_sm120/d01/k_a.meta"]["hlo_sha256"] == "h1"
    assert manifest["counts"] == dict(aot_blob=1, aot_meta=1, jax_entry=1, mutable=3, tune=1)
    with tarfile.open(tmp_path / "b1/gpuwrf-bundle.tar.gz") as tar:
        names = tar.getnames()
    assert "gpuwrf-bundle/cache/jit_f-00-cache" in names and not [n for n in names if n.endswith("-atime")]
    assert "gpuwrf-bundle/cache/cli_memory_plans/k.json" in names  # the C-auto plan ships as a sizing hint
    check = ["check", "--cache", str(cache), "--manifest", str(tmp_path / "b1/BUNDLE_MANIFEST.json")]
    assert cb.main(check) == 0
    (cache / "jit_f-00-atime").write_bytes(b"2")  # access stamps and the prefetch hint may change
    (cache / "aot/0.4.0-cuda_sm120/d01/last_cheap_key").write_text("b")
    # every successful GPU run rewrites its C-auto plan with the measured peak (gpu_allocator.finish_cli_pool)
    (cache / "cli_memory_plans/k.json").write_text('{"live_peak_bytes": 2}')
    alias = cache / "aot/0.4.0-cuda_sm120/d01/k_b.xlaexec"
    alias.write_bytes(blob)  # cheap-key alias of the sealed executable
    (alias.with_suffix(".meta")).write_bytes(pickle.dumps(dataclasses.replace(meta, cheap_key="b")))
    assert cb.main(check) == 0 and "aliases=2" in capsys.readouterr().out
    for orphan in ("k_c.meta", "k_d.xlaexec", "k_e"):  # half pairs and a foreign program never count as aliases
        d01 = cache / "aot/0.4.0-cuda_sm120/d01"
        if orphan == "k_c.meta":
            (d01 / orphan).write_bytes(pickle.dumps(dataclasses.replace(meta, cheap_key="c")))
        elif orphan == "k_d.xlaexec":
            (d01 / orphan).write_bytes(blob)
        else:
            (d01 / "k_e.xlaexec").write_bytes(blob)
            (d01 / "k_e.meta").write_bytes(pickle.dumps(dataclasses.replace(meta, cheap_key="e", hlo_sha256="other")))
        assert cb.main(check) == 1 and "BUNDLE_BROKEN" in capsys.readouterr().out, orphan
        for f in d01.glob(orphan if orphan != "k_e" else "k_e.*"):
            f.unlink()
    assert cb.main(check) == 0
    (cache / "jit_g-11-cache").write_bytes(b"new program")
    assert cb.main(check) == 1 and "new=1" in capsys.readouterr().out
    (cache / "jit_g-11-cache").unlink()
    (cache / "xla_gpu_per_fusion_autotune_cache_dir/t.textproto").write_text("retuned")
    assert cb.main(check) == 1 and "changed=1" in capsys.readouterr().out
    (cache / "xla_gpu_per_fusion_autotune_cache_dir/t.textproto").write_text("tune")
    assert cb.main(check) == 0
    capsys.readouterr()
    shutil.copytree(cache, tmp_path / "moved")  # identical bytes elsewhere: JAX/AOT keys pin the absolute path
    moved = ["check", "--cache", str(tmp_path / "moved"), "--manifest", str(tmp_path / "b1/BUNDLE_MANIFEST.json")]
    assert cb.main(moved) == 2  # bytes verified, but never rc 0: a relocated copy is not runnable at its own path
    out = capsys.readouterr().out
    assert out.startswith("BUNDLE_RELOCATED_OK missing=0 changed=0 new=0 ") and "moved=True" in out
    assert f"sealed_path={cache} sealed_path_hardlinked=False" in out


def _hardlink_copy(src, dst):
    """cp -al: the publication copy of release_bundles/v0.4.4 (same inodes as the sealed cache)."""
    for d, _dirs, files in os.walk(src):
        (dst / Path(d).relative_to(src)).mkdir(parents=True, exist_ok=True)
        for f in files:
            os.link(Path(d) / f, dst / Path(d).relative_to(src) / f)


@pytest.mark.parametrize("where", ["sealed", "relocated", "hardlinked"])
@pytest.mark.parametrize("mutant", [None, "changed_byte", "missing_file", "extra_file", "extra_alias_half",
                                    "file_symlink_same_bytes", "dir_symlink_extra", "fifo"])
def test_compile_bundle_relocated_copy_keeps_tamper_detection(toy_repo, tmp_path, capsys, where, mutant):
    """v0.4.4 incident: release_bundles/v0.4.4 (hardlink copy) reported moved=True, ALISIOS' check of the sealed path
    moved=False. A byte-intact relocated copy is BUNDLE_RELOCATED_OK (rc 2, never 0); every tamper is BUNDLE_BROKEN
    (rc 1) at the sealed path and at a copy alike."""
    cb = _load("compile_bundle")
    cache, blob, meta = _toy_cache(tmp_path / "sealed")
    seal = ["seal", "--cache", str(cache), "--commit", "HEAD", "--source", str(toy_repo), "--repo", str(toy_repo)]
    assert cb.main(seal + ["--out", str(tmp_path / "b")]) == 0
    capsys.readouterr()
    target = cache
    if where == "relocated":
        target = tmp_path / "copy/jax"
        shutil.copytree(cache, target)
    elif where == "hardlinked":
        target = tmp_path / "copy/jax"
        _hardlink_copy(cache, target)
    d01 = target / "aot/0.4.0-cuda_sm120/d01"
    if mutant == "changed_byte":  # replaced, not rewritten in place: a hardlink copy must not edit the sealed inode
        data = bytearray((target / "jit_f-00-cache").read_bytes())
        data[0] ^= 1
        (target / "jit_f-00-cache").unlink()
        (target / "jit_f-00-cache").write_bytes(bytes(data))
    elif mutant == "missing_file":
        (target / "xla_gpu_per_fusion_autotune_cache_dir/t.textproto").unlink()
    elif mutant == "extra_file":
        (target / "jit_g-11-cache").write_bytes(b"compiled outside")
    elif mutant == "extra_alias_half":
        (d01 / "k_z.xlaexec").write_bytes(blob)
    elif mutant == "file_symlink_same_bytes":  # identical bytes, but owned by another file: still a swap
        twin = tmp_path / "twin_entry"
        twin.write_bytes((target / "jit_f-00-cache").read_bytes())
        (target / "jit_f-00-cache").unlink()
        (target / "jit_f-00-cache").symlink_to(twin)
    elif mutant == "dir_symlink_extra":  # rglob never descends into it: invisible to the file scan alone
        (tmp_path / "hidden").mkdir()
        (tmp_path / "hidden/jit_h-00-cache").write_bytes(b"compiled outside")
        (target / "extra_dir").symlink_to(tmp_path / "hidden", target_is_directory=True)
    elif mutant == "fifo":
        os.mkfifo(target / "jit_p-00-cache")
    report = tmp_path / "rep.json"
    rc = cb.main(["check", "--cache", str(target), "--manifest", str(tmp_path / "b/BUNDLE_MANIFEST.json"),
                  "--out", str(report)])
    out, rep = capsys.readouterr().out, json.loads(report.read_text())
    if mutant is None:
        if where == "sealed":
            assert rc == 0 and out.startswith("BUNDLE_OK missing=0 changed=0 new=0 aliases=0 moved=False\n")
        else:
            assert rc == 2 and out.startswith("BUNDLE_RELOCATED_OK missing=0 changed=0 new=0 aliases=0 moved=True ")
            assert rep["relocated"] and not rep["ok"] and rep["sealed_path"] == str(cache)
            assert rep["sealed_path_exists"] and rep["sealed_path_hardlinked"] == (where == "hardlinked")
        return
    assert rc == 1 and out.startswith("BUNDLE_BROKEN ") and not rep["ok"] and not rep["relocated"], (mutant, out)
    key = {"changed_byte": "changed", "missing_file": "missing", "extra_file": "new", "extra_alias_half": "new",
           "file_symlink_same_bytes": "irregular", "dir_symlink_extra": "irregular", "fifo": "irregular"}[mutant]
    assert rep[key], (mutant, rep)
    assert (f"moved={where != 'sealed'}") in out
    if key == "irregular":
        assert f"irregular={len(rep['irregular'])}" in out


def test_compile_bundle_seal_refuses_links_in_the_cache(toy_repo, tmp_path, capsys):
    cb = _load("compile_bundle")
    cache, _blob, _meta = _toy_cache(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    (cache / "linked").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    seal = ["seal", "--cache", str(cache), "--commit", "HEAD", "--source", str(toy_repo), "--repo", str(toy_repo)]
    assert cb.main(seal + ["--out", str(tmp_path / "b")]) == 1
    assert "linked: symlink or special file in the cache" in capsys.readouterr().out


@pytest.mark.parametrize("defect", ["untracked_src", "foreign_source", "no_blob", "meta_mismatch", "outside_tunes",
                                    "unreadable_meta", "missing_meta", "legacy_digest", "missing_target"])
def test_compile_bundle_seal_refuses(toy_repo, tmp_path, capsys, defect):
    cb = _load("compile_bundle")
    outside = tmp_path / "autotune"
    cache, blob, meta = _toy_cache(tmp_path, flags=f"--xla_gpu_per_fusion_autotune_cache_dir={outside}")
    source = toy_repo
    seal = ["seal", "--cache", str(cache), "--commit", "HEAD", "--repo", str(toy_repo), "--out", str(tmp_path / "b")]
    assert cb.main(seal + ["--source", str(source)]) == 0  # an EMPTY outside flag dir is only recorded
    if defect == "untracked_src":
        (toy_repo / "src/gpuwrf/extra.py").write_text("")
    elif defect == "foreign_source":
        (toy_repo / "src/gpuwrf/__init__.py").write_text('__version__ = "9.9.9"\n')
        _git(toy_repo, "add", "-A")
        _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "other")
        seal[seal.index("HEAD")] = "HEAD~1"
    elif defect == "no_blob":
        for p in [*(cache / "aot").rglob("*.xlaexec"), *(cache / "aot").rglob("*.meta")]:
            p.unlink()
    elif defect == "meta_mismatch":
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.xlaexec").write_bytes(b"edited")
    elif defect == "unreadable_meta":
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").write_bytes(b"not a pickle")
    elif defect == "missing_meta":
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").unlink()
    elif defect == "legacy_digest":  # a digest-less record cannot attest which executable it describes
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").write_bytes(pickle.dumps(dataclasses.replace(meta, blob_sha256=None)))
    elif defect == "missing_target":
        fp = {k: v for k, v in meta.fingerprint.items() if k != "compute_capability"}
        (cache / "aot/0.4.0-cuda_sm120/d01/k_a.meta").write_bytes(pickle.dumps(dataclasses.replace(meta, fingerprint=fp)))
    else:
        outside.mkdir()
        (outside / "t.textproto").write_text("tune escaped the seal")
    assert cb.main(seal + ["--source", str(source)]) == 1
    assert "SEAL_REFUSED" in capsys.readouterr().out


def _tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _bwrap_overlay_ok(tmp_path):
    probe = tmp_path / "probe"
    for d in ("lo", "up", "wk"):
        (probe / d).mkdir(parents=True)
    return shutil.which("bwrap") is not None and subprocess.run(
        ["bwrap", "--dev-bind", "/", "/", "--overlay-src", str(probe / "lo"), "--overlay", str(probe / "up"),
         str(probe / "wk"), str(probe / "lo"), "true"], capture_output=True).returncode == 0


@pytest.fixture
def sealed_bundle(tmp_path):
    """A toy bundle with the manifest fields the real checker reads (files + decoded AOT metadata), built with the
    checker's own scan/decode_meta; sealing itself is tested above. No jax import, so the fixture stays cheap under
    host load."""
    if not _bwrap_overlay_ok(tmp_path):
        pytest.skip("no unprivileged bubblewrap overlay on this host")
    cb = _load("compile_bundle")
    bundle = tmp_path / "P"
    cache, blob, meta = _toy_cache(bundle)
    (bundle / "autotune").mkdir()
    (bundle / "sub").mkdir()
    (bundle / "sub/target").write_bytes(b"sealed target")
    (bundle / "sealed").mkdir()
    metas = {rel: cb.decode_meta(cache / rel) for rel, rec in cb.scan(cache).items() if rec["kind"] == "aot_meta"}
    (bundle / "sealed/BUNDLE_MANIFEST.json").write_text(json.dumps(dict(
        schema=cb.SCHEMA, cache=str(cache), files=cb.scan(cache), aot_meta=metas)))
    (bundle / "tools").mkdir()
    shutil.copy(TOOLS / "compile_bundle.py", bundle / "tools/compile_bundle.py")
    alias_meta = tmp_path / "alias.meta"  # a cheap-key alias pair of the sealed executable, written by the payload
    alias_meta.write_bytes(pickle.dumps(dataclasses.replace(meta, cheap_key="b")))
    yield bundle, alias_meta
    for d, _dirs, _files in os.walk(tmp_path):  # overlayfs leaves work/work at mode 000; let tmp cleanup remove it
        for sub in _dirs:
            path = Path(d) / sub
            if not path.is_symlink():
                path.chmod(0o755)


def _consume(bundle, layer, *cmd, extra=(), **kw):
    return subprocess.run(["bash", str(TOOLS / "bundle_consumer.sh"), *extra, "--bundle", str(bundle), "--layer",
                           str(layer), "--", *cmd], capture_output=True, text=True, timeout=900, **kw)


def _counts(stdout):
    last = stdout.strip().splitlines()[-1]
    assert last.startswith("BUNDLE_CONSUMER rc="), stdout
    return {k: v for k, v in (f.split("=", 1) for f in last.split()[1:])}


def test_bundle_consumer_counts_programs_by_content_with_the_checker_alias_rule(sealed_bundle, tmp_path):
    bundle, alias_meta = sealed_bundle
    cb, layer, j = _load("compile_bundle"), tmp_path / "layer", bundle / "jax"
    d01 = j / "aot/0.4.0-cuda_sm120/d01"
    sealed = _tree_bytes(bundle)
    payload = (f"cat {j}/jit_f-00-cache >/dev/null && echo new > {j}/jit_g-11-cache && echo 2 > {j}/jit_f-00-atime && "
               f"cp {d01}/k_a.xlaexec {d01}/k_b.xlaexec && cp {alias_meta} {d01}/k_b.meta && "
               f"mkdir {bundle}/meta && echo o > {bundle}/meta/o && echo \"env=${{JAX_PLATFORMS-unset}}\"")
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    first = _consume(bundle, layer, "bash", "-c", payload, env=env)
    assert first.returncode == 0, first.stderr
    assert "env=unset" in first.stdout  # the supervisor's checker import never leaks into the consumer's env
    c = _counts(first.stdout)
    assert (c["compiled_new"], c["aliases_new"], c["runtime_new"], c["other_new"], c["layer_compiled"]) == \
        ("1", "2", "1", "1", "1")
    assert "P check after: BUNDLE_OK missing=0 changed=0 new=0" in first.stdout
    assert _tree_bytes(bundle) == sealed
    merged = tmp_path / "merged"  # the real checker on the consumer's view gives the same verdicts
    shutil.copytree(j, merged / "jax")
    shutil.copytree(layer / "upper/jax", merged / "jax", dirs_exist_ok=True)
    rep = tmp_path / "rep.json"
    cb.main(["check", "--cache", str(merged / "jax"), "--manifest", str(bundle / "sealed/BUNDLE_MANIFEST.json"),
             "--out", str(rep)])
    rep = json.loads(rep.read_text())
    assert rep["new"] == ["jit_g-11-cache"] and len(rep["aliases"]) == 2 and not rep["changed"]
    # a rewrite that keeps size and mtime (new -> NEW, 4 bytes each) is still new content: counted
    rewrite = (f"python3 -c \"import os; p='{j}/jit_g-11-cache'; s=os.stat(p); open(p,'w').write('NEW\\n'); "
               f"os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))\"")
    again = _consume(bundle, layer, "bash", "-c", rewrite)
    c = _counts(again.stdout)
    assert again.returncode == 0 and c["compiled_new"] == "1" and c["layer_compiled"] == "1", again.stdout
    retune = f"echo retuned > {j}/xla_gpu_per_fusion_autotune_cache_dir/t.textproto; echo t > {bundle}/autotune/x"
    c = _counts(_consume(bundle, layer, "bash", "-c", retune).stdout)
    assert c["compiled_new"] == "2" and c["layer_compiled"] == "3"  # a rewritten sealed tune + a toolchain cache file
    c = _counts(_consume(bundle, layer, "true").stdout)
    assert c["compiled_new"] == "0" and c["aliases_new"] == "0" and c["layer_compiled"] == "3"
    assert _tree_bytes(bundle) == sealed


@pytest.mark.parametrize("escape", ["upper", "work", "runs", "lock", "layer", "inside", "contains", "hardlink",
                                    "work_after_upper"])
def test_bundle_consumer_refuses_layers_that_reach_the_bundle(sealed_bundle, tmp_path, escape):
    bundle, _ = sealed_bundle
    layer = tmp_path / "layer"
    sealed = _tree_bytes(bundle)
    if escape in ("upper", "work", "runs"):
        layer.mkdir()
        (layer / escape).symlink_to(bundle / "sub", target_is_directory=True)
    elif escape == "lock":
        layer.mkdir()
        (layer / ".consumer.lock").symlink_to(bundle / "sub/target")
    elif escape == "work_after_upper":  # a later component is bad, an earlier one would be created first
        layer.mkdir()
        (layer / "work").symlink_to(bundle / "sub", target_is_directory=True)
    elif escape == "layer":
        layer.symlink_to(bundle / "sub", target_is_directory=True)
    elif escape == "inside":
        layer = bundle / "sub/layer"
    elif escape == "contains":
        layer = tmp_path
    else:
        (layer / "upper/jax").mkdir(parents=True)
        os.link(bundle / "jax/jit_f-00-cache", layer / "upper/jax/jit_f-00-cache")
    entries = sorted(os.listdir(layer)) if layer.is_dir() and not layer.is_symlink() else None
    r = _consume(bundle, layer, "bash", "-c", f"echo escaped > {bundle}/sub/x; echo e > {bundle}/jax/jit_x-cache")
    assert r.returncode == 125 and "BUNDLE_CONSUMER_REFUSED" in r.stderr, (r.stdout, r.stderr)
    assert _tree_bytes(bundle) == sealed and not (bundle / "sub/layer").exists()
    if entries is not None:  # refused before creating anything in an existing layer
        assert sorted(os.listdir(layer)) == entries


def test_bundle_consumer_refuses_a_relocated_copy_before_running(sealed_bundle, tmp_path):
    """--bundle on a copy (release_bundles/v0.4.4 class): bundle.env and the AOT keys name the sealed path, so the
    command would write the sealed bundle outside the overlay -> refused (125) before the command starts."""
    bundle, _ = sealed_bundle
    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy, symlinks=True)
    marker = tmp_path / "ran"
    r = _consume(copy, tmp_path / "layer", "touch", str(marker))
    assert r.returncode == 125 and not marker.exists(), r.stdout + r.stderr
    assert f"is a relocated copy; its cache is sealed at {bundle / 'jax'}" in r.stderr
    assert not (tmp_path / "layer").exists()  # refused before any layer component is created
    ok = _consume(bundle, tmp_path / "layer", "touch", str(marker))
    assert ok.returncode == 0 and marker.exists(), ok.stdout + ok.stderr


@pytest.mark.parametrize("part", ["meta", "blob"])
def test_bundle_consumer_counts_a_warm_alias_with_its_unchanged_partner(sealed_bundle, tmp_path, part):
    """A reused layer: the run rewrites one half of an existing alias pair; the real checker still sees aliases=2."""
    bundle, alias_meta = sealed_bundle
    layer, d01 = tmp_path / "layer", bundle / "jax/aot/0.4.0-cuda_sm120/d01"
    first = _consume(bundle, layer, "bash", "-c", f"cp {d01}/k_a.xlaexec {d01}/k_b.xlaexec && cp {alias_meta} {d01}/k_b.meta")
    c = _counts(first.stdout)
    assert (c["compiled_new"], c["aliases_new"], c["layer_compiled"]) == ("0", "2", "0"), first.stdout
    if part == "meta":  # other retained metadata, same sealed blob and program
        meta = pickle.loads(alias_meta.read_bytes())
        edited = tmp_path / "edited.meta"
        edited.write_bytes(pickle.dumps(dataclasses.replace(meta, in_tree="different retained metadata")))
        step = f"cp {edited} {d01}/k_b.meta"
    else:  # foreign bytes first (a compile), then the sealed blob restored under the unchanged metadata
        c = _counts(_consume(bundle, layer, "bash", "-c", f"echo foreign > {d01}/k_b.xlaexec").stdout)
        assert (c["compiled_new"], c["layer_compiled"]) == ("1", "2")  # the blob, and its now-orphaned metadata
        step = f"cp {d01}/k_a.xlaexec {d01}/k_b.xlaexec"
    r = _consume(bundle, layer, "bash", "-c", step)
    assert "view check (P + layer): BUNDLE_OK missing=0 changed=0 new=0 aliases=2" in r.stdout, r.stdout
    c = _counts(r.stdout)
    assert (c["compiled_new"], c["aliases_new"], c["layer_compiled"]) == ("0", "1", "0"), r.stdout


@pytest.mark.parametrize("case", ["missing_layer_under_alias", "existing_layer_under_alias", "bound_child"])
def test_bundle_consumer_refuses_a_bind_mount_alias_of_the_bundle(sealed_bundle, tmp_path, case):
    """Real bind mounts (inside one bwrap): an alias of P above the layer is found by location, not by path."""
    bundle, _ = sealed_bundle
    (bundle / "sub/d").mkdir()
    alias = tmp_path / "alias"
    alias.mkdir()
    if case == "bound_child":
        layer = tmp_path / "layer"
        (layer / "upper").mkdir(parents=True)
        bind = [str(bundle / "sub"), str(layer / "upper")]
        expect = "inside the layer"  # the mount-point rule
    else:
        layer = alias / ("layer" if case == "missing_layer_under_alias" else "d")
        bind = [str(bundle / "sub"), str(alias)]
        expect = "lies inside or aliases the bundle"
    probe = (f"import sys; sys.path.insert(0, {str(TOOLS)!r}); import bundle_consumer as b\n"
             f"try:\n    b.open_layer({str(bundle)!r}, {str(layer)!r})\nexcept b.Refused as e:\n    print('REFUSED', e)")
    r = subprocess.run(["bwrap", "--dev-bind", "/", "/", "--bind", *bind, sys.executable, "-I", "-c", probe],
                       capture_output=True, text=True, timeout=60)
    assert "REFUSED" in r.stdout and expect in r.stdout, (r.stdout, r.stderr)
    assert sorted(os.listdir(bundle / "sub")) == ["d", "target"] and not os.listdir(bundle / "sub/d")  # nothing created
    b = _load("bundle_consumer")
    assert b.within(("8:1", "/a/b"), ("8:1", "/a")) and not b.within(("8:1", "/ab"), ("8:1", "/a"))
    assert not b.within(("8:2", "/a/b"), ("8:1", "/a"))


def _locked(path):
    import fcntl
    with open(path, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_bundle_consumer_signal_tears_down_the_tree_before_releasing_locks(sealed_bundle, tmp_path, sig):
    bundle, _ = sealed_bundle
    layer, gpu_lock, ready, sup = tmp_path / "layer", tmp_path / "gpu.lock", tmp_path / "ready", tmp_path / "sup"
    # The payload and a background child ignore TERM/INT (only the KILL after --grace ends them). The payload writes
    # a cache file the moment the wrapper process is gone: a tree that outlives the wrapper writes it at any load.
    payload = (f"trap '' TERM INT; (trap '' TERM INT; sleep 300) & echo $! > {ready}.bg; echo $$ > {ready}; "
               f"while [ ! -s {sup} ]; do sleep 0.05; done; s=$(cat {sup}); while [ -e /proc/$s ]; do sleep 0.05; done; "
               f"echo late > {bundle}/jax/jit_late-cache; sleep 300")
    proc = subprocess.Popen(["bash", "-c", 'exec 9>"$1"; flock -n 9; shift; exec "$@"', "outer", str(gpu_lock),
                             "bash", str(TOOLS / "bundle_consumer.sh"), "--grace", "0.5", "--bundle", str(bundle),
                             "--layer", str(layer), "--", "bash", "-c", payload],
                            stdout=open(tmp_path / "out", "w"), stderr=open(tmp_path / "err", "w"),
                            start_new_session=True)  # files, not pipes: a survivor must not stall the test
    sup.write_text(str(proc.pid))  # the wrapper (exec chain: bash -> bash -> python keeps this pid)
    try:
        end = time.monotonic() + 600  # generous: under host load every process start is slow
        complete = lambda p: p.exists() and p.read_text().endswith("\n")  # noqa: E731 - echo has finished writing
        while not (complete(ready) and complete(Path(f"{ready}.bg"))) and time.monotonic() < end:
            time.sleep(0.02)
        pids = [int(ready.read_text()), int(Path(f"{ready}.bg").read_text())]
        time.sleep(0.2)
        proc.send_signal(sig)
        try:
            proc.wait(timeout=600)  # the wrapper returns as soon as its whole tree is gone (no fixed teardown bound)
        except subprocess.TimeoutExpired:
            pass
        assert proc.returncode == 128 + sig, ((tmp_path / "out").read_text(), (tmp_path / "err").read_text())
        assert not [p for p in pids if Path(f"/proc/{p}").exists()]  # the whole tree is gone when the wrapper ends
        assert not _locked(gpu_lock) and not _locked(layer / ".consumer.lock")  # and only then are the locks free
        time.sleep(1.0)  # a surviving payload would write within 0.05 s of the wrapper's exit
        assert not (layer / "upper/jax/jit_late-cache").exists() and not (bundle / "jax/jit_late-cache").exists()
        receipt = json.loads(next((layer / "runs").glob("*.json")).read_text())
        assert receipt["signal"] == sig and receipt["teardown"]["survivors"] == [] and receipt["teardown"]["killed"]
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def test_bundle_consumer_teardown_waits_until_the_tree_is_gone(monkeypatch, capsys):
    """A descendant that needs seconds to die (uninterruptible I/O under load) must not outlive teardown."""
    b = _load("bundle_consumer")
    ghost = (1 << 22) + 4242  # above the kernel pid_max: never a real process, kill() raises ProcessLookupError
    gone_at = time.monotonic() + 3.0
    monkeypatch.setattr(b, "descendants", lambda root: [ghost] if time.monotonic() < gone_at else [])
    monkeypatch.setattr(b, "reap", lambda: None)
    calls = []
    monkeypatch.setattr(b.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    rec = b.teardown(grace=0.1, report_every=1.0)
    assert time.monotonic() >= gone_at and rec["survivors"] == [] and rec["seconds"] >= 2.9
    assert (ghost, signal.SIGTERM) in calls and (ghost, signal.SIGKILL) in calls and rec["killed"] > 1
    assert "teardown still waiting for 1 descendant(s)" in capsys.readouterr().err


def test_bundle_consumer_exit_paths_release_everything(sealed_bundle, tmp_path):
    bundle, _ = sealed_bundle
    layer = tmp_path / "layer"
    leftover = _consume(bundle, layer, "bash", "-c", f"sleep 30 >/dev/null 2>&1 & echo $! > {tmp_path}/bg; exit 42")
    assert leftover.returncode == 42 and _counts(leftover.stdout)["rc"] == "42"
    assert not Path(f"/proc/{(tmp_path / 'bg').read_text().strip()}").exists()  # an orphan is reaped at normal exit
    missing = _consume(bundle, layer, "/nonexistent/command")
    assert missing.returncode not in (0, 125) and _counts(missing.stdout)["rc"] == str(missing.returncode)
    assert not _locked(layer / ".consumer.lock")
    with open(layer / ".consumer.lock", "a") as held:
        import fcntl
        fcntl.flock(held, fcntl.LOCK_EX)
        busy = _consume(bundle, layer, "true")
    assert busy.returncode == 125 and "in use" in busy.stderr
    assert _consume(bundle, layer, "true").returncode == 0


def test_wheel_smoke_run_refuses_without_wrf_tables_before_any_work(tmp_path):
    work, wrf, venv = tmp_path / "work", tmp_path / "WRF", tmp_path / "venv"
    (wrf / "run").mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text("#!/bin/sh\nexit 1\n")  # a stub: nothing can ever be installed anywhere
    (venv / "bin/python").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "GPUWRF_WRF_ROOT"}
    for extra in ([], ["--wrf-root", str(wrf)]):  # unset, and a tree without run/MPTABLE.TBL
        rc = subprocess.run(["bash", str(TOOLS / "wheel_smoke.sh"), "--venv", str(venv), "--work", str(work), "--run",
                             *extra], capture_output=True, text=True, env=env)
        assert rc.returncode == 2 and "MPTABLE.TBL" in rc.stderr and not work.exists()


def test_final_label_requires_the_approved_readme_sha(toy_repo, tmp_path):
    # toy repo + tmp lane: even without the guard nothing outside tmp_path is touched
    rc = subprocess.run(["bash", str(TOOLS / "final.sh"), "--version", "9.9.9", "--private", "HEAD", "--label", "final",
                         "--repo", str(toy_repo), "--base", "HEAD", "--lane", str(tmp_path / "lane")],
                        cwd=toy_repo, capture_output=True, text=True)
    assert rc.returncode == 2 and "--readme-sha256" in rc.stderr and not (tmp_path / "lane").exists()


def test_proofs_scope_rules():
    ps = _load("proofs_scope")
    mib = 1 << 20
    new = {"proofs/a/fixtures/x.npz": 9 * mib, "proofs/a/fixtures/y.npz": 1000, "proofs/a/fixtures/run.log": 10,
           "proofs/a/fixtures/archive.tar.gz": 10, "proofs/a/helper.py": 100, "proofs/a/data/z.npz": 100,
           "proofs/b/VERDICT.md": 100, "proofs/b/big.json": 2 * mib, "proofs/b/raw.npz": 100,
           "proofs/b/trace.nsys-rep": 10, "proofs/b/wrfout_d01_x": 10, "proofs/b/__pycache__/m.cpython-313.pyc": 10,
           "proofs/s2_integrate/b1/QUALIFICATION.json": 100, "proofs/s2_integrate/b1/integrate_b1.py": 100,
           "proofs/s2_integrate/b1/sanity.log.gz": 100, "proofs/t_redsuite/FINAL.json": 2 * mib,
           "proofs/c/huge.npz": 11 * mib}
    readers = {"tests/test_a.py": "ROOT/'proofs/a/fixtures'\nsys.path.insert(0, str(ROOT/'proofs/a'))",
               "tests/test_c.py": "np.load(ROOT / 'proofs/c/huge.npz')",
               "proofs/a/helper.py": "np.load('proofs/a/data/z.npz')", "docs/x.md": "see proofs/b/big.json"}
    c = ps.classify(new, readers)
    d = {p: c[p]["decision"] for p in c}
    assert d["proofs/a/fixtures/x.npz"] == d["proofs/a/fixtures/y.npz"] == "fixture"
    assert c["proofs/a/fixtures/x.npz"]["consumers"] == ["tests/test_a.py"]
    assert d["proofs/a/fixtures/run.log"] == d["proofs/a/fixtures/archive.tar.gz"] == "exclude"  # per-run log, archive
    assert d["proofs/a/helper.py"] == "fixture" and d["proofs/a/data/z.npz"] == "fixture"  # transitive via the helper
    assert d["proofs/b/VERDICT.md"] == "receipt" and d["proofs/b/big.json"] == "exclude"  # receipts <= 1 MiB only
    assert c["proofs/b/big.json"]["docs"] == ["docs/x.md"]
    assert d["proofs/b/raw.npz"] == d["proofs/b/trace.nsys-rep"] == d["proofs/b/wrfout_d01_x"] == "exclude"
    assert d["proofs/b/__pycache__/m.cpython-313.pyc"] == "refuse" and d["proofs/c/huge.npz"] == "refuse"
    assert d["proofs/s2_integrate/b1/QUALIFICATION.json"] == "receipt"
    assert d["proofs/s2_integrate/b1/integrate_b1.py"] == d["proofs/s2_integrate/b1/sanity.log.gz"] == "exclude"
    assert d["proofs/t_redsuite/FINAL.json"] == "exclude"
    assert ps.dependent_tests(c, readers) == {"tests/test_a.py": ["proofs/a/fixtures/archive.tar.gz",
                                                                  "proofs/a/fixtures/run.log"]}
    idx = ps.retained_index(c, {p: "s" for p in c})
    assert set(idx) == {p for p, v in d.items() if v == "exclude"} and idx["proofs/b/big.json"]["sha256"] == "s"


def test_release_readme_is_the_approved_versioned_readme(toy_repo, monkeypatch):
    monkeypatch.setattr(curate, "PRIV_REPO", str(toy_repo))
    (toy_repo / "docs/release").mkdir(parents=True)
    (toy_repo / "docs/release/README_v0.3_draft.md").write_text("# stale draft v0.3.4\n")
    (toy_repo / "README.md").write_text("# wrf_gpu v0.4.0\nForecasts on one GPU.\n")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "readme")
    sha = hashlib.sha256((toy_repo / "README.md").read_bytes()).hexdigest()
    text, rec = curate.release_readme("HEAD", "0.4.0", sha, 4000)
    assert text.startswith("# wrf_gpu v0.4.0") and rec["source"].endswith(":README.md") and rec["source_sha256"] == sha
    with pytest.raises(AssertionError, match="not the approved"):
        curate.release_readme("HEAD", "0.4.0", "0" * 64, 4000)
    with pytest.raises(AssertionError, match="never names v0.4.1"):
        curate.release_readme("HEAD", "0.4.1", None, 4000)
    with pytest.raises(AssertionError, match="budget"):
        curate.release_readme("HEAD", "0.4.0", None, 3)
    with pytest.raises(AssertionError, match="bytecode"):
        curate.refuse_bytecode(["src/a.py", "proofs/x/__pycache__/m.cpython-313.pyc"])
    curate.refuse_bytecode(["src/a.py", "proofs/x/m.py"])


def test_committed_scan_sees_force_added_files(toy_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    base = _git(toy_repo, "rev-parse", "HEAD").strip()
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(dict(allowlist={"proofs/a/ok.json": {}}, exclusions={"proofs/a/run.log": {}})))
    (toy_repo / "proofs/a").mkdir(parents=True)
    (toy_repo / "proofs/a/ok.json").write_text("{}")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "clean")
    clean = verify.committed(toy_repo, base, _git(toy_repo, "rev-parse", "HEAD").strip(), scope)
    assert clean["fails"] == [] and clean["new"] == 1
    (toy_repo / "proofs/a/__pycache__").mkdir()
    (toy_repo / "proofs/a/__pycache__/m.cpython-313.pyc").write_bytes(b"\x00<USER_HOME>/x")  # ignored, force-added
    (toy_repo / "proofs/a/run.log").write_text("log")
    (toy_repo / "proofs/a/extra.json").write_text("{}")
    (toy_repo / "tests/fixtures").mkdir(parents=True)
    (toy_repo / "tests/fixtures/tiny.cubin").write_bytes(b"\x7fELF")  # a test fixture, not a dump
    (toy_repo / "proofs/a/k.cubin").write_bytes(b"\x7fELF")
    (toy_repo / "env.txt").write_text("PATH=/usr/bin\n")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "add", "-f", "proofs/a/__pycache__/m.cpython-313.pyc")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "dirty")
    dirty = verify.committed(toy_repo, base, _git(toy_repo, "rev-parse", "HEAD").strip(), scope)
    assert dirty["bytecode"] == ["proofs/a/__pycache__/m.cpython-313.pyc"]
    assert "proofs/a/__pycache__/m.cpython-313.pyc" in dirty["private"]
    assert dirty["proofs_excluded_committed"] == ["proofs/a/run.log"]
    assert sorted(dirty["dumps"]) == ["env.txt", "proofs/a/k.cubin"]
    assert "proofs/a/extra.json" in dirty["proofs_not_allowlisted"]
    assert {"bytecode", "private", "proofs_excluded_committed", "proofs_not_allowlisted"} <= set(dirty["fails"])
    assert "scope_missing" in verify.committed(toy_repo, base, "HEAD", None)["fails"]


OPS_TEXT = ("her" + "dr lane notes\n").encode()  # an ops-marker word, assembled so this public test never carries it


def test_proofs_scope_round_trip_and_test_refs(toy_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    monkeypatch.setattr(curate, "PRIV_REPO", str(toy_repo))
    base = _git(toy_repo, "rev-parse", "HEAD").strip()
    files = {"proofs/a/fixtures/f.npz": b"F", "proofs/a/fixtures/run.log": b"L", "proofs/a/VERDICT.md": b"# v\n",
             "proofs/a/raw.npz": b"R", "tests/test_a.py": b"DATA = ROOT/'proofs/a/fixtures'\n",
             "proofs/a/OPS.md": OPS_TEXT}
    for rel, data in files.items():
        (toy_repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (toy_repo / rel).write_bytes(data)
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "proofs")
    priv = _git(toy_repo, "rev-parse", "HEAD").strip()
    out, work = tmp_path / "out", tmp_path / "work"
    shutil.copytree(toy_repo, out, ignore=shutil.ignore_patterns(".git"))
    work.mkdir()
    base_set = set(_git(toy_repo, "ls-tree", "-r", "--name-only", base).split())
    (out / "proofs/a/OPS.md").unlink()  # publish_all never copies ops-marker files
    ops = {"proofs/a/OPS.md": (len(OPS_TEXT), hashlib.sha256(OPS_TEXT).hexdigest())}
    rep = curate.apply_proofs_scope(out, base_set, priv, "0.4.0", work, ops)
    assert json.loads((out / "proofs/RETAINED_PRIVATELY.json").read_text())["files"]["proofs/a/OPS.md"]["sha256"] == ops[
        "proofs/a/OPS.md"][1]
    assert rep["exclude_files"] == 2 and not (out / "proofs/a/raw.npz").exists() and (out / "proofs/a/fixtures/f.npz").exists()
    scope = json.loads((work / "proofs_scope.json").read_text())
    assert scope["dependent_tests"] == {"tests/test_a.py": ["proofs/a/fixtures/run.log"]}
    assert scope["allowlist"]["proofs/a/fixtures/f.npz"]["consumers"] == ["tests/test_a.py"]
    good = verify.proofs_scope(out, priv, base)
    assert good["index_exact"] and not good["missing_published"] and not good["excluded_present"]
    assert good["ops_private"] == 1
    assert verify.test_refs(out, priv)["missing"] == []
    (out / "proofs/a/raw.npz").write_bytes(b"R")  # an excluded file reappears
    assert verify.proofs_scope(out, priv, base)["excluded_present"] == ["proofs/a/raw.npz"]
    (out / "proofs/a/raw.npz").unlink()
    (out / "proofs/a/fixtures/f.npz").unlink()  # a consumed fixture goes missing
    assert verify.proofs_scope(out, priv, base)["missing_published"] == ["proofs/a/fixtures/f.npz"]
    assert verify.test_refs(out, priv)["missing"] == [("tests/test_a.py", "proofs/a/fixtures")]
    (out / "proofs/a/fixtures/f.npz").write_bytes(b"F")
    index = json.loads((out / "proofs/RETAINED_PRIVATELY.json").read_text())
    index["files"].pop("proofs/a/raw.npz")
    (out / "proofs/RETAINED_PRIVATELY.json").write_text(json.dumps(index))
    assert verify.proofs_scope(out, priv, base)["index_exact"] is False


def test_verify_binds_the_published_readme_bytes(toy_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "PRIV", str(toy_repo))
    readme = b"# wrf_gpu v0.4.0\nForecasts on one GPU.\n"
    (toy_repo / "README.md").write_bytes(readme)
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "readme")
    approved = hashlib.sha256(readme).hexdigest()
    pub = tmp_path / "pub"
    pub.mkdir()
    (pub / "README.md").write_bytes(readme)
    good = verify.readme_check(pub, "HEAD", "HEAD~1", "0.4.0", None, approved)
    assert good["approved_match"] is True and good["published_is_source"] and good["published_sha256"] == approved
    (pub / "README.md").write_bytes(readme + b"Also ships an unapproved claim.\n")  # public-only tamper, source unchanged
    bad = verify.readme_check(pub, "HEAD", "HEAD~1", "0.4.0", None, approved)
    assert bad["source_sha256"] == approved and bad["approved_match"] is False and not bad["published_is_source"]


def _attachments_tool():
    spec = importlib.util.spec_from_file_location("_release_attachments_t", ROOT / "tools/release_attachments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attach(repo, rel, data, tag):
    """Commit ``rel`` with ``data``, then replace it by its provenance index (the B17 move)."""
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_bytes(data)
    _git(repo, "add", rel)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", f"add {rel}")
    blob = _git(repo, "rev-parse", f"HEAD:{rel}").strip()
    _git(repo, "rm", "-q", rel)
    asset = "attachment-" + rel.replace("/", "-")
    info = dict(schema="gpuwrf-release-attachment-v1", path=rel, asset=asset, release_tag=tag, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), git_blob=blob, public_commits=[])
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)  # git rm dropped the emptied directory
    (repo / (rel + ".attachment.json")).write_text(json.dumps(info))
    _git(repo, "add", rel + ".attachment.json")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", f"attach {rel}")
    return repo / (rel + ".attachment.json"), info


def test_release_attachment_staging_sources_and_refusals(toy_repo, tmp_path, monkeypatch):
    ra = _attachments_tool()
    data = b"A" * 5000
    index, info = _attach(toy_repo, "proofs/a/big.json", data, "v9.9.9")
    monkeypatch.setenv("GPUWRF_TABLES_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GPUWRF_TABLES_OFFLINE", "1")
    staged = ra.stage(index, root=toy_repo)  # 1. from the checkout's git history (offline)
    assert staged == tmp_path / "cache" / info["sha256"] / info["asset"] and staged.read_bytes() == data
    assert stat.S_IMODE(staged.stat().st_mode) == 0o444 and not (toy_repo / "proofs/a/big.json").exists()
    nogit = tmp_path / "nogit"
    (nogit / "proofs/a").mkdir(parents=True)
    copy = nogit / "proofs/a" / index.name
    shutil.copy(index, copy)
    assert ra.stage(copy, root=nogit) == staged  # 2. a later stage is a verified cache hit
    staged.chmod(0o644)
    staged.write_bytes(b"B" * 5000)  # same size, other bytes
    with pytest.raises(ra.AttachmentUnavailable, match="does not match the pin"):
        ra.stage(copy, root=nogit)
    assert staged.read_bytes() == b"B" * 5000  # left for the operator, never silently replaced
    monkeypatch.setenv("GPUWRF_TABLES_DIR", str(tmp_path / "cache2"))
    with pytest.raises(ra.AttachmentUnavailable, match="GPUWRF_TABLES_OFFLINE is set") as exc:
        ra.stage(copy, root=nogit)
    assert info["sha256"] in str(exc.value) and "Fix:" in str(exc.value)
    assert not (tmp_path / "cache2").exists() or not any((tmp_path / "cache2").rglob("*"))  # nothing half-staged
    srv = tmp_path / "srv"
    srv.mkdir()
    monkeypatch.delenv("GPUWRF_TABLES_OFFLINE")
    monkeypatch.setenv("GPUWRF_ATTACHMENTS_URL", f"file://{srv}/{{tag}}/{{asset}}")
    (srv / "v9.9.9").mkdir()
    (srv / "v9.9.9" / info["asset"]).write_bytes(data[:-1] + b"X")  # 3. a download with wrong bytes is refused
    with pytest.raises(ra.AttachmentUnavailable, match="got 5000 B sha256"):
        ra.stage(copy, root=nogit)
    assert not any((tmp_path / "cache2").rglob("*"))
    (srv / "v9.9.9" / info["asset"]).write_bytes(data)
    assert ra.stage(copy, root=nogit).read_bytes() == data  # ... the right bytes are staged
    bad = dict(info, sha256="z" * 64)
    (nogit / "proofs/a/big.json.attachment.json").chmod(0o644)
    (nogit / "proofs/a/big.json.attachment.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="sha256"):
        ra.load(nogit / "proofs/a/big.json.attachment.json")
    with pytest.raises(ValueError, match="index must sit at"):
        ra.check(info, "proofs/b/big.json.attachment.json")


def test_curate_exports_release_attachments_and_verify_gates_them(toy_repo, tmp_path):
    data = b"P" * 7000
    index, info = _attach(toy_repo, "proofs/a/big.json", data, "v9.9.9")
    attached = curate.release_attachments("HEAD")
    assert list(attached) == ["proofs/a/big.json"] and attached["proofs/a/big.json"]["git_blob"] == info["git_blob"]
    out = tmp_path / "assets"
    rows = curate.export_release_assets("HEAD", curate.remote_assets("HEAD"), out, attached, "9.9.9")
    assert set(rows) == {"scheme-big.npz", info["asset"]} and (out / info["asset"]).read_bytes() == data
    assert f"{info['sha256']}  {info['asset']}\n" in (out / "SHA256SUMS").read_text()
    staging = tmp_path / "staging"  # an older public base still carries the file: curate drops it, keeps the index
    (staging / "proofs/a").mkdir(parents=True)
    (staging / "proofs/a/big.json").write_bytes(data)
    shutil.copy(index, staging / "proofs/a" / index.name)
    assert curate.drop_attached(staging, attached) == ["proofs/a/big.json"]
    assert sorted(p.name for p in (staging / "proofs/a").iterdir()) == [index.name]
    other = curate.export_release_assets("HEAD", curate.remote_assets("HEAD"), tmp_path / "a2", attached, "9.9.8")
    assert set(other) == {"scheme-big.npz"}  # attached to another release: not re-exported
    pub = tmp_path / "pub"
    (pub / "proofs/a").mkdir(parents=True)
    shutil.copy(index, pub / "proofs/a" / index.name)
    good = verify.attachments_check(pub, out, "9.9.9")
    assert good["indexes"] == 1 and good["this_release"] == [info["asset"]]
    assert not (good["invalid"] or good["in_tree"] or good["missing"] or good["bad"])
    shutil.copy(toy_repo / curate.ASSETS_MANIFEST, tmp_path / "manifest.json")
    (pub / "src/gpuwrf").mkdir(parents=True)
    shutil.copy(tmp_path / "manifest.json", pub / curate.ASSETS_MANIFEST)
    assert not verify.release_assets(pub, out, good["this_release"])["unexpected"]
    assert verify.release_assets(pub, out)["unexpected"] == [info["asset"]]
    (pub / "proofs/a/big.json").write_bytes(data)  # the attached file back in the tree
    assert verify.attachments_check(pub, out, "9.9.9")["in_tree"] == ["proofs/a/big.json"]
    (pub / "proofs/a/big.json").unlink()
    (out / info["asset"]).write_bytes(data[:-1] + b"Q")
    assert verify.attachments_check(pub, out, "9.9.9")["bad"] == [info["asset"]]
    (out / info["asset"]).unlink()
    assert verify.attachments_check(pub, out, "9.9.9")["missing"] == [info["asset"]]
    assert verify.attachments_check(pub, None, "9.9.9")["missing"] == [info["asset"]]
    (pub / "proofs/a" / index.name).write_text(json.dumps(dict(info, git_blob="nope")))
    assert verify.attachments_check(pub, out, "9.9.9")["invalid"]
    with pytest.raises(AssertionError, match="differs from its attachment pin"):
        curate.export_release_assets("HEAD", {}, tmp_path / "a3",
                                     {"proofs/a/big.json": dict(info, sha256="0" * 64)}, "9.9.9")


def test_no_file_above_10_mib_may_stay_in_the_public_tree(tmp_path):
    big, limit = tmp_path / "proofs/big.json", tmp_path / "tests/limit.bin"
    big.parent.mkdir(parents=True)
    limit.parent.mkdir(parents=True)
    with open(big, "wb") as fh:
        fh.truncate(verify.MAX_NEW_BLOB + 1)
    with open(limit, "wb") as fh:
        fh.truncate(verify.MAX_NEW_BLOB)
    assert verify.oversized_tree(tmp_path, [big, limit]) == [("proofs/big.json", verify.MAX_NEW_BLOB + 1)]


def test_repository_attachments_are_indexed_and_out_of_the_tree():
    ra = _attachments_tool()
    found = {ra.load(p)["path"]: ra.load(p) for p in ra.indexes(ROOT)}
    assert set(found) >= {"proofs/v015/finalgates/canary_l2_d02/atlas_grid_delta_summary.json",
                          "proofs/v015/finalgates/switzerland_d01/atlas_grid_delta_summary.json",
                          "tests/v025/restart/fixtures/shadow_dtype_native_cpu.pkl.gz"}
    tracked = set(_git(ROOT, "ls-files").split("\n"))
    for path, info in found.items():
        assert path not in tracked and path + ra.SUFFIX in tracked
        assert info["bytes"] > 10 * 1024 * 1024  # only > 10 MiB files are attachments
        have = subprocess.run(["git", "cat-file", "-e", info["git_blob"]], cwd=ROOT).returncode == 0
        if have:  # the pinned blob of the published history matches the pin
            blob = subprocess.run(["git", "cat-file", "blob", info["git_blob"]], cwd=ROOT, capture_output=True,
                                  check=True).stdout
            assert len(blob) == info["bytes"] and hashlib.sha256(blob).hexdigest() == info["sha256"]


# --- public README plain-language gate (lane w41-readme, Owner 2026-10-17) ---------------------------------------
_LINT_BODY = ("# wrf_gpu\n\n## Speed\n![s](docs/release/img/v032_speed.png)\n## What's new\n## How we check it\n"
              "## Install\n## Quickstart\n## Limitations\n## Citing\n## License\n")


def test_readme_lint_flags_internal_identifiers_but_not_hardware_or_schemes():
    clean = readme_lint.lint(_LINT_BODY + "Runs on H100, B200 and B300; P3 microphysics works.\n")
    assert clean["gate"] == "PASS", clean
    for leak, kind in [("E41 FAIL disclosed", "finding_or_gate_id"), ("C5\u2032 gate", "finding_or_gate_id"),
                       ("c06 rain", "composite_code"), ("frozen as 3356e249", "git_hash"), ("at 00:26Z", "utc_log_stamp"),
                       ("see the receipt", "process_word"), ("WN3 cases", "internal_name")]:
        rep = readme_lint.lint(_LINT_BODY + leak + "\n")
        assert rep["gate"] == "FAIL" and kind in rep["internal_identifiers"], (leak, rep["internal_identifiers"])
    assert readme_lint.lint(_LINT_BODY + "[ledger](docs/release/c06_E41_3356e249.md)\n")["gate"] == "PASS"  # targets
    assert "no_benchmark_image" in readme_lint.lint(_LINT_BODY.replace("docs/release/img/v032_speed.png", "x.png"))["fails"]
    assert readme_lint.lint(_LINT_BODY.replace("## Speed\n", "## Numbers\n"))["missing_core_sections"] == ["speed"]
    assert "word_budget" in readme_lint.lint(_LINT_BODY + "word " * 2600)["fails"]
    verify_src = (TOOLS / "verify.py").read_text()
    assert 'rep["readme_plain"] = readme_lint.lint(' in verify_src and 'fails.append("readme_plain")' in verify_src


def test_public_readme_passes_lint_and_matches_the_approved_sha():
    rep = readme_lint.lint((ROOT / "README.md").read_text(encoding="utf-8"))
    assert rep["gate"] == "PASS", rep
    approved = (TOOLS / "README_APPROVED.sha256").read_text().split()[0]
    assert hashlib.sha256((ROOT / "README.md").read_bytes()).hexdigest() == approved, (
        "README.md changed after approval: re-review it and update scripts/release/README_APPROVED.sha256")


def test_final_reads_the_approved_readme_sha_from_the_release_commit(toy_repo, tmp_path):
    assert 'show "$PRIV:scripts/release/README_APPROVED.sha256"' in (TOOLS / "final.sh").read_text()
    (toy_repo / "scripts/release").mkdir()
    (toy_repo / "scripts/release/README_APPROVED.sha256").write_text("not-a-sha  README.md\n")
    _git(toy_repo, "add", "-A")
    _git(toy_repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "malformed approval")
    rc = subprocess.run(["bash", str(TOOLS / "final.sh"), "--version", "9.9.9", "--private", "HEAD", "--label", "final",
                         "--repo", str(toy_repo), "--base", "HEAD", "--lane", str(tmp_path / "lane")],
                        cwd=toy_repo, capture_output=True, text=True)
    assert rc.returncode == 2 and "--readme-sha256" in rc.stderr and not (tmp_path / "lane").exists()
