from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from b200_io_lib import validate_b200_manifest  # noqa: E402


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_staged_case(tmp_path: Path, *, child_e_we: int = 898, child_e_sn: int = 898) -> tuple[Path, Path]:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "namelist.input").write_text(
        f"""&time_control
/
&domains
 max_dom = 2,
 e_we = 369, {child_e_we},
 e_sn = 369, {child_e_sn},
 parent_grid_ratio = 1, 3,
/
""",
        encoding="utf-8",
    )
    for name in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01"):
        (staged / name).write_bytes(f"{name}\n".encode("ascii"))
    manifest = tmp_path / "manifest.json"
    required = [
        {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(staged.iterdir())
        if path.is_file()
    ]
    manifest.write_text(
        json.dumps(
            {
                "schema": "test.b200",
                "required_inputs": required,
                "wrf_domains": [
                    {"id": "d01", "e_we": 369, "e_sn": 369, "parent_grid_ratio": 1},
                    {"id": "d02", "e_we": child_e_we, "e_sn": child_e_sn, "parent_grid_ratio": 3},
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, staged


def test_manifest_validator_accepts_898_child(tmp_path: Path) -> None:
    manifest, staged = _write_staged_case(tmp_path)
    report = validate_b200_manifest(manifest, staged)
    assert report["ok"], report["issues"]
    assert report["domains"][1]["e_we"] == 898


def test_manifest_validator_rejects_bad_checksum(tmp_path: Path) -> None:
    manifest, staged = _write_staged_case(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["required_inputs"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data), encoding="utf-8")
    report = validate_b200_manifest(manifest, staged)
    assert not report["ok"]
    assert any(issue["code"] == "required_input_sha256_mismatch" for issue in report["issues"])


def test_manifest_validator_rejects_897_child_dimension(tmp_path: Path) -> None:
    manifest, staged = _write_staged_case(tmp_path, child_e_we=897, child_e_sn=898)
    report = validate_b200_manifest(manifest, staged)
    assert not report["ok"]
    assert any(issue["code"] == "invalid_nested_e_we" for issue in report["issues"])


def test_manifest_validator_rejects_old_181_mini_nest_even_if_divisible(tmp_path: Path) -> None:
    manifest, staged = _write_staged_case(tmp_path, child_e_we=181, child_e_sn=181)
    report = validate_b200_manifest(manifest, staged)
    assert not report["ok"]
    assert any(issue["code"] == "old_mini_nest_rejected" for issue in report["issues"])


def test_manifest_validator_rejects_single_domain_181_grid(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "namelist.input").write_text(
        """&domains
 max_dom = 1,
 e_we = 181,
 e_sn = 181,
 parent_grid_ratio = 1,
/
""",
        encoding="utf-8",
    )
    (staged / "wrfinput_d01").write_bytes(b"wrfinput_d01\n")
    manifest = tmp_path / "manifest.json"
    required = [
        {"path": path.name, "sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(staged.iterdir())
        if path.is_file()
    ]
    manifest.write_text(
        json.dumps(
            {
                "schema": "test.b200",
                "required_inputs": required,
                "wrf_domains": [{"id": "d01", "e_we": 181, "e_sn": 181, "parent_grid_ratio": 1}],
            }
        ),
        encoding="utf-8",
    )

    report = validate_b200_manifest(manifest, staged)

    assert not report["ok"]
    assert any(issue["code"] == "old_mini_nest_rejected" for issue in report["issues"])


def test_manifest_validator_rejects_namelist_manifest_mismatch(tmp_path: Path) -> None:
    manifest, staged = _write_staged_case(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["wrf_domains"][1]["e_we"] = 901
    manifest.write_text(json.dumps(data), encoding="utf-8")
    report = validate_b200_manifest(manifest, staged)
    assert not report["ok"]
    assert any(issue["code"] == "namelist_e_we_mismatch" for issue in report["issues"])


def test_manifest_cli_writes_json_and_human_report(tmp_path: Path) -> None:
    manifest, staged = _write_staged_case(tmp_path)
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "scripts/b200_validate_manifest.py", str(manifest), str(staged), "--json-out", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "B200 manifest validation: PASS" in proc.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True
