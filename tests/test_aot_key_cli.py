"""`gpuwrf aot-key` plumbing (CPU, no case load): parser, namelist view, case comparison, no dispatch."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.cli import build_parser
from gpuwrf.contracts.grid import DomainHierarchy, GridSpec
from gpuwrf.runtime import aot_key_cli
from gpuwrf.runtime import domain_tree as dt
from gpuwrf.runtime.aot_key_probe import case_program_keys
from gpuwrf.runtime.operational_mode import OperationalNamelist


def test_aot_key_is_a_gpuwrf_command(tmp_path):
    args = build_parser().parse_args(["aot-key", "--input-dir", str(tmp_path), "--compare-input-dir", str(tmp_path)])
    assert args.func is aot_key_cli.command
    assert args.hours == 3.0 and args.namelist is None


def test_namelist_override_is_a_symlink_view(tmp_path):
    case, other = tmp_path / "case", tmp_path / "other.input"
    case.mkdir()
    (case / "namelist.input").write_text("&domains\n max_dom = 3\n/\n")
    (case / "wrfinput_d01").write_text("x")
    other.write_text("&domains\n max_dom = 2\n/\n")
    assert aot_key_cli._case_dir(case, None, tmp_path) == case
    view = aot_key_cli._case_dir(case, other, tmp_path)
    assert (view / "namelist.input").resolve() == other.resolve()
    assert (view / "wrfinput_d01").resolve() == (case / "wrfinput_d01").resolve()
    assert (case / "namelist.input").read_text().startswith("&domains\n max_dom = 3")


def _record(key, **components):
    base = {name: "same" for name in aot_key_cli._COMPONENTS}
    return {"cheap_key": key, "components": base | components}


def test_comparison_names_the_differing_component():
    a = {"d01": _record("k1"), "fused/d02": _record("k2", fn_identity="case-a")}
    b = {"d01": _record("k1"), "fused/d02": _record("k3", fn_identity="case-b"), "fused/d03": _record("k4")}
    rows = aot_key_cli._compare(a, b)
    assert rows["d01"] == {"shared": True, "differing_components": []}
    assert rows["fused/d02"] == {"shared": False, "differing_components": ["fn_identity"]}
    assert rows["fused/d03"]["shared"] is False


def test_missing_keys_are_never_declared_shared():
    rows = aot_key_cli._compare({"fused/d02": {"cheap_key": None}}, {"fused/d02": {"cheap_key": None}})
    assert rows["fused/d02"]["shared"] is False and rows["fused/d02"]["unavailable"] is True
    rows = aot_key_cli._compare({"d01": _record("k1")}, {"d01": {"cheap_key": None, "unavailable": "x"}})
    assert rows["d01"]["shared"] is False and "compare" in rows["d01"]["reason"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _Carry:
    state: object

    def replace(self, **kw):
        return replace(self, **kw)

    def tree_flatten(self):
        return (self.state,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@pytest.mark.parametrize("env", [{"GPUWRF_NESTED_AOT": "0"}, {"GPUWRF_ADVANCE_CHUNK_LOOP": "scan"}])
def test_probe_never_dispatches_a_model_step(monkeypatch, env):
    """cr-w41-bundlekey P2: AOT off or a non-fori loop mode must not let the dry-run step the model."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.0)
    tree = dt.DomainTree(hierarchy=DomainHierarchy.from_edges(("d01",), ()), domains={"d01": SimpleNamespace(namelist=nl)})
    calls = []
    monkeypatch.setattr(dt, "_resolve_operational_suite", lambda nl: None)

    def model_step(*args, **kwargs):
        calls.append("model-step")
        return args[0]

    monkeypatch.setattr(dt, "_advance_chunk", model_step)
    monkeypatch.setattr(dt, "_operational_fused_cascade_factory", lambda tree: lambda name: None)
    records = case_program_keys(tree, {"d01": _Carry(jnp.ones((1,), jnp.float32))}, components=True)
    assert not calls, "the dry-run stepped the model"
    record = records["d01"]
    if "GPUWRF_ADVANCE_CHUNK_LOOP" in env:  # no AOT for this loop mode: reported, never stepped
        assert record["cheap_key"] is None and record["unavailable"]
    else:  # the probe computes the key an AOT run would use, whatever the caller's env says
        assert record["cheap_key"] is not None


# --------------------------------------------------------------------------- #
# aot-key never writes a persistent cache (w42-aotkey; incident 2026-10-18: a CPU aot-key after
# `source <bundle>/bundle.env` wrote 1564 CPU cache entries into the sealed v0.4.1 bundle).
# --------------------------------------------------------------------------- #
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import site
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import gpuwrf
from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime import compile_cache

_REPO = Path(__file__).resolve().parents[1]
_SRC = Path(gpuwrf.__file__).resolve().parents[1]

# Stands in for case_keys inside a real `gpuwrf aot-key` process: the writes a case load makes,
# each through its product path (JAX persistent cache, init AOT store, init libm loop).
_RUNNER = r'''
import os, sys
from contextlib import contextmanager
from pathlib import Path

from gpuwrf import cli
from gpuwrf.runtime import aot_key_cli


def fake_case_keys(input_dir, *, hours, scratch):
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration import _init_libm_arrays
    from gpuwrf.integration.init_kernels import InitKernelCache
    from gpuwrf.runtime import aot_precompile

    jax.jit(lambda x: x * 3.0 + 1.0)(jnp.arange(5.0)).block_until_ready()

    @jax.jit
    def aotkey_probe_kernel(x):
        return jnp.sin(x) + 2.0

    InitKernelCache()(aotkey_probe_kernel, jnp.arange(4.0))
    aot_precompile.record_last_cheap_key("d01", "ef" * 32)  # prefetch hint, as a run's AOT load records it
    _init_libm_arrays.load()
    roots = {"jax": jax.config.jax_compilation_cache_dir, "aot": str(aot_precompile.aot_dir()),
             "libm": os.environ.get("GPUWRF_JAX_CACHE_DIR")}
    written = sorted({str(p) for root in roots.values() if root for p in Path(root).rglob("*") if p.is_file()})
    return {"d01": {"program": "advance", "cheap_key": "0" * 64, "roots": roots, "written": written}}


aot_key_cli.case_keys = fake_case_keys
if os.environ.get("AOTKEY_TEST_NO_REDIRECT") == "1":  # mutant: the redirect removed
    @contextmanager
    def _no_redirect(root):
        yield root

    aot_key_cli.private_compile_caches = _no_redirect
sys.exit(cli.main(sys.argv[1:]))
'''


def _compile_bundle():
    spec = importlib.util.spec_from_file_location("compile_bundle", _REPO / "scripts/release/compile_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path) -> dict[str, str]:
    """Every path under ``root`` with its SHA-256 (directories as 'dir'): file list + content."""
    return {p.relative_to(root).as_posix(): "dir" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))}


def _mini_bundle(root: Path) -> Path:
    """A sealed bundle in the release layout: caches, bundle.env, sealed/BUNDLE_MANIFEST.json."""
    jax_dir = root / "jax"
    (jax_dir / "aot" / "0.4.1-jax0.10.0-jaxlib0.10.0-cuda_sm120" / "d01").mkdir(parents=True)
    (jax_dir / ("jit__advance_chunk_fori-" + "ab" * 32 + "-cache")).write_bytes(b"sealed jit entry")
    (jax_dir / "aot" / "0.4.1-jax0.10.0-jaxlib0.10.0-cuda_sm120" / "d01" / ("k_" + "cd" * 32 + ".xlaexec")) \
        .write_bytes(b"sealed blob")
    for name in ("autotune", "triton", "cuda"):
        (root / name).mkdir()
        (root / name / "sealed.bin").write_bytes(name.encode())
    (root / "bundle.env").write_text(
        f"export PYTHONPATH={_SRC}\n"
        f"export GPUWRF_JAX_CACHE_DIR={jax_dir} JAX_COMPILATION_CACHE_DIR={jax_dir} "
        f"GPUWRF_XLA_AUTOTUNE_CACHE_DIR={root / 'autotune'}\n"
        f"export TRITON_CACHE_DIR={root / 'triton'} CUDA_CACHE_PATH={root / 'cuda'}\n")
    cb = _compile_bundle()
    (root / "sealed").mkdir()
    manifest = {"schema": cb.SCHEMA, "cache": str(jax_dir.resolve()), "files": cb.scan(jax_dir), "aot_meta": {}}
    (root / "sealed" / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
    return root


def _intrusions(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    """Every new, changed or missing path of ANY kind -- stricter than compile_bundle check, which never compares
    its MUTABLE files (-atime access stamps, lockfile, last_cheap_key prefetch hints)."""
    return {"new": sorted(set(after) - set(before)), "missing": sorted(set(before) - set(after)),
            "changed": sorted(p for p in set(before) & set(after) if before[p] != after[p])}


def _bundle_ok(bundle: Path) -> bool:
    cb = _compile_bundle()
    return cb.check(Namespace(cache=bundle / "jax", manifest=bundle / "sealed/BUNDLE_MANIFEST.json", out=None)) == 0


def _aot_key_in_bundle_env(tmp_path: Path, bundle: Path, *, no_redirect: bool = False):
    """`source <bundle>/bundle.env` + `gpuwrf aot-key` in a fresh process, as rd-s6 ran it."""
    runner, case, tmpdir = tmp_path / "runner.py", tmp_path / "case", tmp_path / "tmpdir"
    runner.write_text(_RUNNER)
    case.mkdir()
    tmpdir.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GPUWRF_JAX_CACHE", "JAX_COMPILATION_CACHE", "GPUWRF_CACHE", "PYTEST_"))}
    env.update(JAX_PLATFORMS="cpu", TMPDIR=str(tmpdir), PYTHONDONTWRITEBYTECODE="1",
               AOTKEY_TEST_NO_REDIRECT="1" if no_redirect else "0")
    report = tmp_path / "report.json"
    proc = subprocess.run(
        ["bash", "-c", f'source "{bundle}/bundle.env" && exec "{sys.executable}" "{runner}" aot-key '
                       f'--input-dir "{case}" --json "{report}"'],
        env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(report.read_text()), proc.stderr, tmpdir


def test_aot_key_leaves_a_sealed_bundle_byte_identical(tmp_path):
    bundle = _mini_bundle(tmp_path / "bundle")
    before = _tree(bundle)
    assert _bundle_ok(bundle)
    report, stderr, tmpdir = _aot_key_in_bundle_env(tmp_path, bundle)
    assert _intrusions(before, _tree(bundle)) == {"new": [], "missing": [], "changed": []}
    assert _bundle_ok(bundle)
    record = report["case"]["programs"]["d01"]
    # Not vacuous: the case load did compile/serialize/build, into the private cache only ...
    private = Path(record["roots"]["jax"])
    assert record["written"] and all(Path(p).is_relative_to(tmpdir) for p in record["written"])
    kinds = {Path(p).relative_to(private).parts[0] for p in record["written"]}
    assert "aot" in kinds and ("init-libm" in kinds or shutil.which("cc") is None), kinds
    names = [Path(p).name for p in record["written"]]
    assert any(n.endswith("-cache") for n in names) and any(n.endswith("-atime") for n in names), names  # JAX
    assert "last_cheap_key" in names, names
    # ... which is gone after the command, and the bundle setting was reported loudly.
    assert not any(tmpdir.iterdir())
    assert set(report["sealed_bundle_env"]) == {"JAX_COMPILATION_CACHE_DIR", "GPUWRF_JAX_CACHE_DIR", "TRITON_CACHE_DIR",
                                                "CUDA_CACHE_PATH", "GPUWRF_XLA_AUTOTUNE_CACHE_DIR",
                                                "jax_compilation_cache_dir"}
    assert f"WARNING: JAX_COMPILATION_CACHE_DIR points into the sealed compile bundle {bundle.resolve()}" in stderr


def test_without_the_redirect_aot_key_writes_the_bundle(tmp_path):
    """Mutant (redirect removed): the bundle check of the test above must fail, as in the incident."""
    bundle = _mini_bundle(tmp_path / "bundle")
    before = _tree(bundle)
    report, _stderr, _tmpdir = _aot_key_in_bundle_env(tmp_path, bundle, no_redirect=True)
    new = _intrusions(before, _tree(bundle))["new"]
    assert any(p.startswith("jax/aot/") and p.endswith(".xlaexec") for p in new), new
    assert any(p.startswith("jax/jit_") and p.endswith("-cache") for p in new), new
    # the files compile_bundle check never compares are written too (the 2026-10-18 cleanup left 1540 -atime)
    assert any(p.endswith("-atime") for p in new) and any(p.endswith("/last_cheap_key") for p in new), new
    assert not _bundle_ok(bundle)


def test_bundle_criterion_catches_what_compile_bundle_check_ignores(tmp_path):
    """Access stamps and prefetch hints alone keep BUNDLE_OK, but the aot-key bundle test fails on them."""
    bundle = _mini_bundle(tmp_path / "bundle")
    before = _tree(bundle)
    (bundle / "jax" / ("jit_add-" + "12" * 32 + "-atime")).write_bytes(b"0")
    hint = bundle / "jax" / "aot" / "0.4.1-jax0.10.0-jaxlib0.10.0-cpu" / "init_solve_rrtmg_lw_column" / "last_cheap_key"
    hint.parent.mkdir(parents=True)
    hint.write_text("34" * 32)
    assert _bundle_ok(bundle)
    intrusions = _intrusions(before, _tree(bundle))
    assert intrusions["new"] == ["jax/aot/0.4.1-jax0.10.0-jaxlib0.10.0-cpu",
                                 "jax/aot/0.4.1-jax0.10.0-jaxlib0.10.0-cpu/init_solve_rrtmg_lw_column",
                                 "jax/aot/0.4.1-jax0.10.0-jaxlib0.10.0-cpu/init_solve_rrtmg_lw_column/last_cheap_key",
                                 f"jax/jit_add-{'12' * 32}-atime"]


def test_private_caches_redirect_only_key_inert_settings():
    for name in aot_key_cli._PRIVATE_CACHE_ENV:
        assert name not in ck._EXEC_ENV_FLAGS
        assert not name.startswith("GPUWRF_") or ck.trace_env_is_inert(name), name
    # Keyed cache settings (E32: the autotune path is part of the cheap key) are never touched.
    assert not ck.trace_env_is_inert("GPUWRF_XLA_AUTOTUNE_CACHE_DIR")
    assert not {"GPUWRF_CACHE", "GPUWRF_XLA_AUTOTUNE_CACHE_DIR", "XLA_FLAGS"} & set(aot_key_cli._PRIVATE_CACHE_ENV)


def test_private_caches_keep_the_product_keys(monkeypatch, tmp_path):
    """The keys inside the redirect equal the keys a run in the caller's environment looks up."""
    bundle = tmp_path / "bundle"
    for name in ("JAX_COMPILATION_CACHE_DIR", "GPUWRF_JAX_CACHE_DIR"):
        monkeypatch.setenv(name, str(bundle / "jax"))
    monkeypatch.setenv("GPUWRF_XLA_AUTOTUNE_CACHE_DIR", str(bundle / "autotune"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(bundle / "triton"))
    monkeypatch.delenv("GPUWRF_JAX_CACHE", raising=False)
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.0)
    tree = dt.DomainTree(hierarchy=DomainHierarchy.from_edges(("d01",), ()), domains={"d01": SimpleNamespace(namelist=nl)})
    monkeypatch.setattr(dt, "_resolve_operational_suite", lambda nl: None)
    monkeypatch.setattr(dt, "_operational_fused_cascade_factory", lambda tree: lambda name: None)
    carries = {"d01": _Carry(jnp.ones((1,), jnp.float32))}
    outside = case_program_keys(tree, carries, components=True)
    saved_dir = jax.config.jax_compilation_cache_dir
    private = tmp_path / "private"
    with aot_key_cli.private_compile_caches(private) as jax_dir:
        assert jax_dir == private / "jax" and jax.config.jax_compilation_cache_dir == str(jax_dir)
        assert compile_cache.resolve_cache_dir() == jax_dir
        assert os.environ["GPUWRF_XLA_AUTOTUNE_CACHE_DIR"] == str(bundle / "autotune")
        inside = case_program_keys(tree, carries, components=True)
    assert outside["d01"]["cheap_key"] is not None
    assert inside == outside
    assert jax.config.jax_compilation_cache_dir == saved_dir
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == str(bundle / "jax")


def test_sealed_bundle_detection(tmp_path):
    bundle = tmp_path / "P"
    (bundle / "sealed").mkdir(parents=True)
    (bundle / "sealed" / "BUNDLE_MANIFEST.json").write_text("{}")
    assert aot_key_cli.sealed_bundle_of(bundle / "jax" / "aot") == bundle.resolve()
    assert aot_key_cli.sealed_bundle_of(bundle) == bundle.resolve()
    assert aot_key_cli.sealed_bundle_of(tmp_path / "elsewhere") is None


# cr-w42-aotkey lifecycle probe: the private cache is removed however the command ends.
_LIFECYCLE_RUNNER = r'''
import os, signal, sys
from pathlib import Path

signal.signal(signal.SIGTERM, signal.SIG_DFL)  # an interactive user's dispositions, whatever pytest inherited
signal.signal(signal.SIGINT, signal.default_int_handler)
from gpuwrf import cli
from gpuwrf.runtime import aot_key_cli


def case_keys(input_dir, *, hours, scratch):
    root = Path(os.environ["GPUWRF_JAX_CACHE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "probe-cache").write_text("private cache")
    Path(os.environ["MARKER"]).write_text(str(root))
    if os.environ["MODE"] != "exception":
        os.kill(os.getpid(), getattr(signal, os.environ["MODE"]))
    raise RuntimeError("intentional case-load failure")


aot_key_cli.case_keys = case_keys
sys.exit(cli.main(["aot-key", "--input-dir", os.environ["CASE"]]))
'''


@pytest.mark.parametrize("mode,returncode", [("exception", 1), ("SIGTERM", -15), ("SIGINT", -2)])
def test_aot_key_removes_its_private_cache_on_error_and_signal(tmp_path, mode, returncode):
    home, tmpdir, case, marker = tmp_path / "home", tmp_path / "tmpdir", tmp_path / "case", tmp_path / "marker"
    for path in (home, tmpdir, case):
        path.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPUWRF_", "JAX_", "XLA_", "CUDA_", "TRITON_"))}
    env.update(HOME=str(home), PYTHONUSERBASE=site.getuserbase(), TMPDIR=str(tmpdir), PYTHONPATH=str(_SRC),
               PYTHONDONTWRITEBYTECODE="1", JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", MARKER=str(marker),
               CASE=str(case), MODE=mode)
    proc = subprocess.run([sys.executable, "-c", _LIFECYCLE_RUNNER], env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == returncode, proc.stderr[-3000:]
    private = Path(marker.read_text())  # the case load ran and wrote into the private cache
    assert private.is_relative_to(tmpdir) and not private.exists()
    assert sorted(tmpdir.rglob("*")) == [] and sorted(home.rglob("*")) == []


def test_signal_handlers_are_restored_after_the_command(monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    seen = {}

    def case_keys(input_dir, *, hours, scratch):
        seen.update({s: signal.getsignal(s) for s in before})
        return {"d01": {"program": "advance", "cheap_key": "0" * 64}}

    monkeypatch.setattr(aot_key_cli, "case_keys", case_keys)
    args = build_parser().parse_args(["aot-key", "--input-dir", str(tmp_path)])
    assert aot_key_cli.command(args) == 0
    assert all(seen[s] is not before[s] for s in before if before[s] is not signal.SIG_IGN)  # armed during the load
    assert {s: signal.getsignal(s) for s in before} == before


def test_single_domain_case_is_refused_without_writes(tmp_path):
    """Contract amendment (manager 02:29Z): a single-domain (Swiss) case keeps the v0.4.1 refusal and rc and
    writes nothing, also with a sealed bundle's environment."""
    bundle = _mini_bundle(tmp_path / "bundle")
    before = _tree(bundle)
    case, tmpdir, home = tmp_path / "swiss", tmp_path / "tmpdir", tmp_path / "home"
    for path in (case, tmpdir, home):
        path.mkdir()
    (case / "namelist.input").write_text("&domains\n max_dom                             = 1,\n/\n")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GPUWRF_JAX_CACHE", "JAX_COMPILATION_CACHE", "GPUWRF_CACHE", "PYTEST_"))}
    env.update(JAX_PLATFORMS="cpu", TMPDIR=str(tmpdir), HOME=str(home), PYTHONUSERBASE=site.getuserbase(),
               PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        ["bash", "-c", f'source "{bundle}/bundle.env" && exec "{sys.executable}" -m gpuwrf.cli aot-key '
                       f'--input-dir "{case}"'], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, proc.stderr[-3000:]
    assert "ValueError: aot-key covers nested runs (max_dom >= 2)" in proc.stderr
    assert _intrusions(before, _tree(bundle)) == {"new": [], "missing": [], "changed": []}
    assert _bundle_ok(bundle)
    assert sorted(tmpdir.rglob("*")) == [] and sorted(home.rglob("*")) == []
    assert sorted(p.name for p in case.iterdir()) == ["namelist.input"]


def test_surface_control_value_sets_of_one_case_are_compared(monkeypatch, tmp_path, capsys):
    """Branch series pre-check: --compare-surface-controls alone compares two value sets of --input-dir."""
    seen = []

    def case_keys(input_dir, *, hours, scratch, surface_controls=None):
        seen.append((input_dir, None if surface_controls is None else surface_controls.name))
        key = "k-on" if surface_controls is not None else "k-off"
        return {"d01": {"cheap_key": key, "components": {name: key for name in aot_key_cli._COMPONENTS}}}

    monkeypatch.setattr(aot_key_cli, "case_keys", case_keys)
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text("&domains\n max_dom = 2\n/\n")
    a, b = tmp_path / "neutral.npz", tmp_path / "variant.npz"
    args = build_parser().parse_args(["aot-key", "--input-dir", str(case), "--surface-controls", str(a),
                                      "--compare-surface-controls", str(b), "--json", str(tmp_path / "r.json")])
    assert aot_key_cli.command(args) == 0
    assert seen == [(case, "neutral.npz"), (case, "variant.npz")]
    out = capsys.readouterr().out
    assert "d01          SHARED" in out and "NOT SHARED" not in out
    seen.clear()
    args = build_parser().parse_args(["aot-key", "--input-dir", str(case), "--compare-surface-controls", str(b)])
    assert aot_key_cli.command(args) == 0
    assert seen == [(case, None), (case, "variant.npz")]
    assert "NOT SHARED" in capsys.readouterr().out
