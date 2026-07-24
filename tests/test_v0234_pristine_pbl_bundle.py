from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v0234_pristine_pbl_bundle.py"
SPEC = importlib.util.spec_from_file_location("v0234_pristine_pbl_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def test_wrong_output_namespace_refused(tmp_path: Path) -> None:
    with pytest.raises(bundle.Refusal, match="output authority mismatch"):
        bundle.construct(tmp_path / "wrong")


def test_copy_regular_rejects_missing_symlink_hash_and_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"authority")
    target = tmp_path / "target"
    with pytest.raises(bundle.Refusal, match="identity drift"):
        bundle.copy_regular(source, target, "0" * 64)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(source)
    with pytest.raises(bundle.Refusal, match="required regular"):
        bundle.copy_regular(symlink, target)
    receipt = bundle.copy_regular(source, target, hashlib.sha256(b"authority").hexdigest())
    assert receipt["origin"]["sha256"] == receipt["copy"]["sha256"]
    with pytest.raises(bundle.Refusal, match="duplicate destination"):
        bundle.copy_regular(source, target)


def test_safe_extract_rejects_path_escape(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        info = tarfile.TarInfo("../escape")
        payload = b"bad"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(bundle.Refusal, match="unsafe Git archive"):
        bundle.safe_extract(stream.getvalue(), tmp_path / "extract")


def test_safe_extract_preserves_source_symlink_but_rejects_nested_member(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        link = tarfile.TarInfo("tracked-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/pinned/upstream/target"
        archive.addfile(link)
    bundle.safe_extract(stream.getvalue(), tmp_path / "accepted")
    assert (tmp_path / "accepted/tracked-link").is_symlink()

    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        archive.addfile(link)
        nested = tarfile.TarInfo("tracked-link/escape")
        nested.size = 1
        archive.addfile(nested, io.BytesIO(b"x"))
    with pytest.raises(bundle.Refusal, match="nested below symlink"):
        bundle.safe_extract(stream.getvalue(), tmp_path / "refused")


def test_materialized_tree_rejects_blob_origin_drift(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    path.write_bytes(b"right")
    framed = b"blob 5\0wrong"
    records = [{"path": "payload", "git_mode": "100644", "git_type": "blob", "git_object": hashlib.sha1(framed).hexdigest()}]
    with pytest.raises(bundle.Refusal, match="materialized blob drift"):
        bundle.verify_materialized_tree(tmp_path, records)


def test_npz_schema_rejects_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, bad=np.asarray([object()], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        bundle.npz_schema(path)


def test_namelist_route_exact_and_drift(tmp_path: Path) -> None:
    values = {"mp_physics": 8, "ra_lw_physics": 4, "ra_sw_physics": 4,
              "sf_sfclay_physics": 5, "sf_surface_physics": 4, "bl_pbl_physics": 5}
    path = tmp_path / "namelist.input"
    path.write_text("\n".join(f" {name} = {value}, {value}, {value}," for name, value in values.items()))
    assert bundle.namelist_route(path)["bl_pbl_physics"] == [5, 5, 5]
    path.write_text(path.read_text().replace("mp_physics = 8, 8, 8", "mp_physics = 6, 6, 6"))
    with pytest.raises(bundle.Refusal, match="namelist route drift"):
        bundle.namelist_route(path)


def test_canonical_json_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    bundle.atomic_json(path, {"schema": "test-v1", "complete": True})
    assert bundle.load_canonical_json(path)["complete"] is True
    value = json.loads(path.read_text())
    value["complete"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(bundle.Refusal, match="canonical JSON drift"):
        bundle.load_canonical_json(path)


def test_relocate_staging_paths_is_exact(tmp_path: Path) -> None:
    staging = tmp_path / ".stage"
    root = tmp_path / "final"
    value = {"path": str(staging / "runtime/table"), "unrelated": str(tmp_path / "other")}
    relocated = bundle.relocate_staging_paths(value, staging, root)
    assert relocated["path"] == str(root / "runtime/table")
    assert relocated["unrelated"] == value["unrelated"]


def test_capture_input_authority_is_exact_and_complete() -> None:
    assert set(bundle.CAPTURE_INPUT_ORIGINS) == {"wrfinput_d01", "wrfinput_d02", "wrfbdy_d01"}
    for path, digest in bundle.CAPTURE_INPUT_ORIGINS.values():
        assert path.is_absolute()
        assert len(digest) == 64
    assert "augment-capture-inputs" in bundle.parser()._actions[1].choices


def test_netcdf_schema_requires_regular_netcdf_payload(tmp_path: Path) -> None:
    path = tmp_path / "not-netcdf"
    path.write_bytes(b"not a netcdf payload")
    with pytest.raises(Exception):
        bundle.netcdf_schema(path)
