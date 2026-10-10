"""Deletion/mutant sensitivity for the v0.3.4 CPU tridiagonal fix (lane o1-tridiag).

Builds mutant copies of ``src/gpuwrf/physics/tridiagonal_solver.py`` (and the
vertical-implicit helper) under ``<out>/mutants/<name>/src`` and runs the
deletion-sensitive gate against each with ``pytest -o pythonpath=<mutant>/src``
(the pyproject ``pythonpath=["src"]`` otherwise shadows PYTHONPATH -- see the
o1-ruc lesson in INBOX 2026-10-10T13:57Z). A mutant is KILLED when the selected
test FAILS; every mutant must be killed or the gate is not deletion-sensitive.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
PYTEST_TARGETS = [
    "tests/test_v034_cpu_tridiagonal_solver.py::test_cpu_lowering_avoids_lapack_gtsv_ffi",
    "tests/test_v034_cpu_tridiagonal_solver.py::test_cpu_multi_rhs_matches_dense_reference",
    "tests/test_v034_cpu_tridiagonal_solver.py::test_cpu_thomas_matches_dense_reference",
]


def _mutations(text: str) -> dict[str, str]:
    out = {}
    # M1: the whole CPU routing removed (the pre-fix behaviour)
    m1 = text.replace(
        "    if _cpu_thomas_fallback():\n        return _solve_tridiagonal_cpu(a, b, c, d)\n", ""
    )
    assert m1 != text, "M1 anchor missing"
    out["M1_cpu_routing_removed"] = m1
    # M2: Thomas backward substitution broken (wrong recurrence)
    m2 = text.replace("        x_i = dp_i - cp_i * x_next\n", "        x_i = dp_i - 0.999999 * cp_i * x_next\n")
    assert m2 != text, "M2 anchor missing"
    out["M2_thomas_backward_wrong"] = m2
    # M3: multi-RHS handling removed
    m3 = text.replace(
        "    if d.ndim == b.ndim:\n        return solve_tridiagonal_thomas_reference(a, b, c, d)\n",
        "    return solve_tridiagonal_thomas_reference(a, b, c, d)\n",
    )
    assert m3 != text, "M3 anchor missing"
    out["M3_multi_rhs_removed"] = m3
    return out


def main() -> int:
    src = TREE / "src"
    target = src / "gpuwrf/physics/tridiagonal_solver.py"
    text = target.read_text()
    report = {"tree": str(TREE), "mutants": {}}
    all_killed = True
    for name, mutant_text in _mutations(text).items():
        root = OUT / "mutants" / name
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(src, root / "src")
        (root / "src/gpuwrf/physics/tridiagonal_solver.py").write_text(mutant_text)
        # The package resolves assets relative to the tree root; expose the real
        # data/ (fixtures + wrf_pristine symlink) to the mutant tree.
        data_link = root / "data"
        if not data_link.exists():
            data_link.symlink_to(TREE / "data")
        env = dict(os.environ)
        env.update(
            JAX_PLATFORMS="cpu",
            GPUWRF_JAX_CACHE="0",
            GPUWRF_FAST_DEFAULTS="0",
        )
        env.pop("PYTHONPATH", None)
        cmd = [
            sys.executable, "-m", "pytest", *PYTEST_TARGETS,
            "-q", "-p", "no:cacheprovider", "-o", f"pythonpath={root / 'src'}",
            "--basetemp", str(root / "pytest"),
        ]
        # Prove the mutant module is the one imported (pyproject pythonpath trap).
        locenv = dict(env)
        locenv["PYTHONPATH"] = str(root / "src")  # E152: child needs src on sys.path
        loc = subprocess.run(
            [sys.executable, "-c",
             "import gpuwrf.physics.tridiagonal_solver as m; print(m.__file__)"],
            cwd=TREE, env=locenv, capture_output=True, text=True, timeout=300,
        )
        imported = loc.stdout.strip().splitlines()[-1] if loc.stdout.strip() else ""
        assert str(root / "src") in imported, (imported, loc.stderr[-500:])
        proc = subprocess.run(
            cmd, cwd=TREE, env=env, capture_output=True, text=True, timeout=900,
            preexec_fn=lambda: os.nice(19),
        )
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
        killed = proc.returncode != 0
        report["mutants"][name] = {
            "returncode": proc.returncode, "killed": killed, "imported": imported, "tail": tail,
        }
        all_killed = all_killed and killed
        print(f"{name}: returncode={proc.returncode} killed={killed}")
        for line in tail:
            print("   ", line)
    report["all_killed"] = all_killed
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mutants.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all_killed else 1


if __name__ == "__main__":
    raise SystemExit(main())
