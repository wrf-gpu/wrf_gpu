from __future__ import annotations

import importlib.util
from pathlib import Path
import stat


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v0234_pristine_pbl_build.py"
SPEC = importlib.util.spec_from_file_location("v0234_pristine_pbl_build", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_parse_cpu_list_handles_ranges_and_singletons() -> None:
    assert build.parse_cpu_list("0-2,7,13-15") == {0, 1, 2, 7, 13, 14, 15}


def test_tree_manifest_binds_regular_and_symlink_payloads(tmp_path: Path) -> None:
    (tmp_path / "regular").write_bytes(b"bytes")
    (tmp_path / "link").symlink_to("/upstream/absolute")
    canonical, records = build.tree_manifest(tmp_path)
    assert len(canonical) == 64
    assert {record["kind"] for record in records} == {"file", "symlink"}
    assert next(record for record in records if record["path"] == "link")["size"] == len("/upstream/absolute")


def test_seal_tree_removes_write_but_preserves_execute(tmp_path: Path) -> None:
    executable = tmp_path / "executable"
    executable.write_bytes(b"x")
    executable.chmod(0o755)
    ordinary = tmp_path / "ordinary"
    ordinary.write_bytes(b"x")
    ordinary.chmod(0o644)
    build.seal_tree(tmp_path)
    assert stat.S_IMODE(executable.stat().st_mode) == 0o555
    assert stat.S_IMODE(ordinary.stat().st_mode) == 0o444
    assert stat.S_IMODE(tmp_path.stat().st_mode) & 0o222 == 0


def test_route_patch_disables_only_unselected_clm() -> None:
    text = build.ROUTE_PATCH.read_text(encoding="utf-8")
    assert "elseif( ENABLE_CLM )" in text
    assert "WRF_USE_CLM" in text
    assert "WRF_USE_CTSM" in text
    assert "sf_surface_physics=4 is Noah-MP" in text


def test_registry_patch_keeps_generated_includes_in_build_root() -> None:
    text = build.REGISTRY_PATCH.read_text(encoding="utf-8")
    assert "copy_directory ${PROJECT_SOURCE_DIR}/Registry ${CMAKE_BINARY_DIR}/Registry" in text
    assert "${CMAKE_BINARY_DIR}/Registry/${REGISTRY_FILE_NAME}" in text
