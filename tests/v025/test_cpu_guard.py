"""No v025 generator may reach the CUDA driver on import.

Written after the constraint was found to be enforced unevenly. Only
`real_state` pinned the platform, so any generator importing `gpuwrf` or `jax`
without importing `real_state` first probed the driver at import time.
`build_nvtx_compile_evidence` did exactly that on every run, logging
`cuda_executor.cc: Could not get kernel mode driver version` from a proof
generator whose entire subject is CPU-only evidence.

A one-off fix to three files would not hold: the next generator someone adds has
the same hazard and nothing would catch it. So the audit itself is the test.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "v025"

sys.path.insert(0, str(SCRIPTS))
import cpu_guard  # noqa: E402

# Modules that legitimately run on a GPU under dual-manager coordination and the
# canonical lock. They enforce §13 themselves rather than by pinning the platform.
COORDINATED = {
    "m0_exact_boundary_child.py",
    "m2_bakeoff_vertical_implicit.py",  # M2 device bake-off: runs only under
    #   scripts/with_gpu_lock.sh (coordinated GPU entry, ADR-038/M2 2026-09-18).
    "pallas_sm120_spike.py",
    "run_gpu_arm.py",
    "step1_driver.py",
}

PINNERS = {"cpu_guard", "real_state", "scripts.v025.real_state", "scripts.v025.cpu_guard"}
ACCELERATOR_ROOTS = {"jax", "jaxlib", "gpuwrf"}


def _module_imports(path: Path) -> list[tuple[int, str]]:
    """(line, root_module) for every top-level import in file order."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return sorted(found)


def _scripts_touching_accelerators() -> list[Path]:
    out = []
    for path in sorted(SCRIPTS.glob("*.py")):
        roots = {name.split(".")[0] for _, name in _module_imports(path)}
        if roots & ACCELERATOR_ROOTS:
            out.append(path)
    return out


def test_the_audit_actually_finds_scripts_to_check():
    """A guard that inspects nothing would pass forever."""
    assert len(_scripts_touching_accelerators()) >= 5


@pytest.mark.parametrize("path", _scripts_touching_accelerators(), ids=lambda p: p.name)
def test_a_pin_precedes_every_accelerator_import(path: Path):
    """The pin must come FIRST in file order; pinning after `import jax` is too late."""
    if path.name in COORDINATED:
        pytest.skip(f"{path.name} runs under coordination and enforces §13 itself")

    imports = _module_imports(path)
    first_accel = next((line for line, name in imports
                        if name.split(".")[0] in ACCELERATOR_ROOTS), None)
    first_pin = next((line for line, name in imports if name in PINNERS), None)

    assert first_pin is not None, (
        f"{path.name} imports an accelerator package but never imports a CPU pin "
        f"(cpu_guard or real_state)"
    )
    assert first_pin < first_accel, (
        f"{path.name} imports the pin at line {first_pin}, AFTER the accelerator import at "
        f"line {first_accel}. Environment set after `import jax` does not take effect."
    )


def test_cpu_guard_pins_the_platform_on_import():
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu_guard.platform_is_pinned_to_cpu() is True


def test_cpu_guard_forces_rather_than_defaults():
    """`setdefault` was FAIL-OPEN under an inherited GPU platform (main 965fe281)."""
    source = (SCRIPTS / "cpu_guard.py").read_text()
    assert 'os.environ["JAX_PLATFORMS"] = "cpu"' in source
    assert 'os.environ.setdefault("JAX_PLATFORMS"' not in source


@pytest.mark.parametrize("hostile", ["cuda", "gpu", "rocm", "cpu,cuda", "CUDA"])
def test_an_inherited_accelerator_platform_is_REFUSED(hostile):
    """A CPU-only entry point must refuse, not silently rewrite and continue."""
    probe = f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import cpu_guard"
    env = dict(os.environ, JAX_PLATFORMS=hostile)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode != 0, f"JAX_PLATFORMS={hostile} was accepted"
    assert "GpuPlatformRefused" in proc.stderr


def test_a_clean_environment_is_forced_to_cpu():
    probe = (f"import os, sys, json; sys.path.insert(0, {str(SCRIPTS)!r}); import cpu_guard; "
             "print(json.dumps({k: os.environ.get(k) for k in "
             "('JAX_PLATFORMS','CUDA_VISIBLE_DEVICES')}))")
    env = {k: v for k, v in os.environ.items()
           if k not in {"JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES"}}
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-400:]
    seen = json.loads(proc.stdout.strip().splitlines()[-1])
    assert seen == {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


def test_a_stale_visible_device_is_overwritten_not_kept():
    """CUDA_VISIBLE_DEVICES=0 with no JAX_PLATFORMS must still end up empty."""
    probe = (f"import os, sys; sys.path.insert(0, {str(SCRIPTS)!r}); import cpu_guard; "
             "print(repr(os.environ['CUDA_VISIBLE_DEVICES']))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    env.pop("JAX_PLATFORMS", None)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.strip().splitlines()[-1] == "''"


def test_importing_the_guard_after_jax_is_refused_rather_than_pretending():
    """Setting these variables after `import jax` is a no-op that looks like safety."""
    probe = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
             "sys.modules['jax'] = type(sys)('jax'); import cpu_guard")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          env=dict(os.environ, JAX_PLATFORMS="cpu"), timeout=300)
    assert proc.returncode != 0
    assert "AFTER jax" in proc.stderr


def test_coordinated_gpu_entry_points_do_not_import_the_cpu_guard():
    """They enforce §13 via receipt + lock; importing the guard muddies that."""
    # pallas_sm120_spike is deliberately dual-mode: --interpret imports the
    # guard after parsing, while --native consumes its handoff before JAX.
    for name in COORDINATED - {"pallas_sm120_spike.py"}:
        source = (SCRIPTS / name).read_text()
        assert "import cpu_guard" not in source, (
            f"{name} is a coordinated GPU entry point and must not import the CPU guard"
        )


# --------------------------------------------------------------------------- #
# hostile-env DYNAMIC audit, covering INDIRECT imports                         #
# --------------------------------------------------------------------------- #
def _cpu_only_scripts() -> list[Path]:
    return [p for p in sorted(SCRIPTS.glob("*.py"))
            if p.name not in COORDINATED and not p.name.startswith("_")]


@pytest.mark.parametrize("path", _cpu_only_scripts(), ids=lambda p: p.name)
def test_hostile_env_dynamic_import_audit(path: Path):
    """Actually import each CPU-only script with JAX_PLATFORMS=cuda exported.

    The static audit only sees an import written in the file itself. This one
    catches INDIRECT exposure -- module A importing module B which imports jax --
    because it runs the imports instead of reading them.

    Either outcome is safe; anything else is not:
      * the module refuses (GpuPlatformRefused), or
      * it ends up forced to CPU.
    A module that imports cleanly while still holding JAX_PLATFORMS=cuda has
    reached an accelerator under a CPU-only contract.
    """
    probe = (
        f"import sys, os, json; sys.path.insert(0, {str(SCRIPTS)!r}); "
        f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
        f"import importlib\n"
        f"try:\n"
        f"    importlib.import_module({path.stem!r})\n"
        f"except SystemExit:\n"
        f"    pass\n"
        f"print('AUDIT ' + json.dumps({{'plat': os.environ.get('JAX_PLATFORMS'), "
        f"'cvd': os.environ.get('CUDA_VISIBLE_DEVICES'), 'jax': 'jax' in sys.modules}}))"
    )
    env = dict(os.environ, JAX_PLATFORMS="cuda", CUDA_VISIBLE_DEVICES="0")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, cwd=REPO, env=env, timeout=900)

    if "GpuPlatformRefused" in proc.stderr:
        return  # refused: safe

    line = next((l for l in proc.stdout.splitlines() if l.startswith("AUDIT ")), None)
    if line is None:
        pytest.skip(f"{path.name} could not be imported standalone: {proc.stderr[-200:]}")

    seen = json.loads(line[len("AUDIT "):])
    if not seen["jax"]:
        return  # never reached an accelerator library at all

    assert seen["plat"] == "cpu" and seen["cvd"] == "", (
        f"{path.name} imported jax while JAX_PLATFORMS={seen['plat']!r} and "
        f"CUDA_VISIBLE_DEVICES={seen['cvd']!r} -- it reached an accelerator under a "
        f"CPU-only contract, most likely through an INDIRECT import"
    )


def test_the_guarded_generator_no_longer_probes_the_driver():
    """The specific regression: importing this module must not reach the driver."""
    probe = (
        "import sys; sys.path.insert(0, %r);"
        "import build_nvtx_compile_evidence" % str(SCRIPTS)
    )
    env = {k: v for k, v in os.environ.items()
           if k not in {"JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES"}}
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, cwd=REPO, env=env, timeout=600)
    combined = proc.stdout + proc.stderr
    assert "kernel mode driver" not in combined, (
        f"importing the generator still probes the CUDA driver:\n{combined[-600:]}"
    )
    assert "cuda_executor" not in combined


def test_the_dynamic_audit_is_not_vacuous():
    """At least the known accelerator-touching generators must actually REFUSE.

    Without this, the audit degrades silently: if every script started failing to
    import, or stopped importing jax, all the parametrised cases would pass while
    checking nothing. The six below are the ones the static audit flags as
    reaching jax/gpuwrf, directly or through `real_state`.
    """
    expected_to_refuse = {
        "build_cancellation_map", "build_hlo_census", "build_manifests",
        "build_nvtx_compile_evidence", "real_state", "cpu_guard",
    }
    env = dict(os.environ, JAX_PLATFORMS="cuda", CUDA_VISIBLE_DEVICES="0")
    refused = set()
    for stem in sorted(expected_to_refuse):
        probe = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); "
                 f"sys.path.insert(0, {str(REPO / 'src')!r}); "
                 f"import importlib; importlib.import_module({stem!r})")
        proc = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                              text=True, cwd=REPO, env=env, timeout=900)
        if "GpuPlatformRefused" in proc.stderr:
            refused.add(stem)
    assert refused == expected_to_refuse, (
        f"these did NOT refuse a hostile platform: {sorted(expected_to_refuse - refused)}"
    )


def test_no_test_name_is_defined_twice_in_a_module():
    """A duplicate def silently shadows the earlier one: a test that never runs.

    This actually happened while repairing the census: an updated test was added
    while its stale twin survived further down the file, so the stale one won and
    the repair looked broken for a reason unrelated to the repair.
    """
    import ast as _ast
    from collections import Counter as _Counter
    for path in sorted((REPO / "tests" / "v025").glob("test_*.py")):
        names = [n.name for n in _ast.walk(_ast.parse(path.read_text()))
                 if isinstance(n, _ast.FunctionDef) and n.name.startswith("test_")]
        duplicates = [name for name, count in _Counter(names).items() if count > 1]
        assert not duplicates, f"{path.name} defines these tests twice: {duplicates}"
