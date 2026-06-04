#!/usr/bin/env python3
"""Clean-clone DRY smoke for the v0.9.0 README runnability gate.

No GPU, no real forecast case. It validates the *install + entrypoint* surface a
naive agent hits when following README.md only:

  1. `pip install -e .` succeeds in a fresh venv.
  2. `python -c "import gpuwrf"` works.
  3. The `gpuwrf` console script exists and `gpuwrf --help` / `gpuwrf run --help`
     print usage.
  4. `gpuwrf run` fails CLEANLY (non-zero, helpful stderr, no traceback) for:
       - missing required args,
       - a non-existent --input-dir,
       - a --namelist that is not <input-dir>/namelist.input,
       - an unsupported namelist option (fail-closed registry check).

The venv is created with --system-site-packages so the heavy CPU deps already in
the environment (jax/netCDF4/numpy) are inherited; the smoke is about the gpuwrf
package install + console script, not about re-downloading GPU wheels.

Writes the result JSON to the path given as argv[1] (default: alongside this file).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(cmd, **kwargs):
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, **kwargs
    )
    return {
        "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO / "proofs" / "v090" / "readme_runnability_dry_smoke.json"
    )
    checks: list[dict] = []

    def record(name, *, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    with tempfile.TemporaryDirectory(prefix="gpuwrf_dry_smoke_") as tmp:
        venv = Path(tmp) / "venv"

        # 1. fresh venv (inherit CPU deps already present so we don't fetch GPU wheels)
        r = _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
        record("create_venv", ok=r["returncode"] == 0, detail=r)
        py = venv / "bin" / "python"
        gpuwrf_bin = venv / "bin" / "gpuwrf"

        # 2. pip install -e .
        r = _run([str(py), "-m", "pip", "install", "--no-build-isolation", "-e", str(REPO)])
        if r["returncode"] != 0:
            # fall back to default build isolation if --no-build-isolation failed
            r = _run([str(py), "-m", "pip", "install", "-e", str(REPO)])
        record("pip_install_editable", ok=r["returncode"] == 0, detail=r)

        # 3. import gpuwrf
        r = _run([str(py), "-c", "import gpuwrf; print('gpuwrf', gpuwrf.__file__)"])
        record("import_gpuwrf", ok=r["returncode"] == 0, detail=r)

        # 4. console script exists
        record("console_script_present", ok=gpuwrf_bin.is_file(), detail=str(gpuwrf_bin))

        # 5. gpuwrf --help
        r = _run([str(gpuwrf_bin), "--help"])
        record(
            "gpuwrf_help",
            ok=r["returncode"] == 0 and "run" in r["stdout_tail"],
            detail=r,
        )

        # 6. gpuwrf run --help
        r = _run([str(gpuwrf_bin), "run", "--help"])
        record(
            "gpuwrf_run_help",
            ok=r["returncode"] == 0 and "--compare-cpu-dir" in r["stdout_tail"],
            detail=r,
        )

        # 7. run with no args -> clean nonzero + usage on stderr
        r = _run([str(gpuwrf_bin), "run"])
        record(
            "run_missing_args_clean",
            ok=r["returncode"] != 0 and "required" in r["stderr_tail"].lower(),
            detail=r,
        )

        # 8. bad --input-dir -> clean nonzero, no traceback
        r = _run(
            [
                str(gpuwrf_bin), "run",
                "--namelist", "/no/such/nl",
                "--input-dir", "/no/such/dir",
                "--output-dir", str(Path(tmp) / "out"),
            ]
        )
        record(
            "run_bad_input_dir_clean",
            ok=(
                r["returncode"] != 0
                and "--input-dir does not exist" in r["stderr_tail"]
                and "Traceback" not in r["stderr_tail"]
            ),
            detail=r,
        )

        # 9. namelist not matching input-dir
        case = Path(tmp) / "case"
        case.mkdir()
        (case / "namelist.input").write_text("&physics\n mp_physics = 8,\n/\n")
        other = Path(tmp) / "other.input"
        other.write_text("&physics\n mp_physics = 8,\n/\n")
        r = _run(
            [
                str(gpuwrf_bin), "run",
                "--namelist", str(other),
                "--input-dir", str(case),
                "--output-dir", str(Path(tmp) / "out"),
            ]
        )
        record(
            "run_namelist_mismatch_clean",
            ok=(
                r["returncode"] != 0
                and "must be <input-dir>/namelist.input" in r["stderr_tail"]
                and "Traceback" not in r["stderr_tail"]
            ),
            detail=r,
        )

        # 10. unsupported namelist option -> fail-closed, no traceback, no JAX import
        (case / "namelist.input").write_text("&physics\n mp_physics = 99,\n/\n")
        r = _run(
            [
                str(gpuwrf_bin), "run",
                "--namelist", str(case / "namelist.input"),
                "--input-dir", str(case),
                "--output-dir", str(Path(tmp) / "out"),
            ]
        )
        record(
            "run_unsupported_namelist_failclosed",
            ok=(
                r["returncode"] != 0
                and "Unsupported namelist" in r["stderr_tail"]
                and "Traceback" not in r["stderr_tail"]
            ),
            detail=r,
        )

    overall = all(c["ok"] for c in checks)
    payload = {
        "schema": "GpuwrfReadmeRunnabilityDrySmoke",
        "schema_version": 1,
        "status": "PASS" if overall else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "python": sys.version.split()[0],
        "scope": "install + entrypoint surface only; no GPU, no real forecast case",
        "checks": checks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"status": payload["status"], "out": str(out_path),
                      "failed": [c["check"] for c in checks if not c["ok"]]}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
