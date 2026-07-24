from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/v0234_pristine_pbl_capture.py"
SPEC = importlib.util.spec_from_file_location("v0234_pristine_pbl_capture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def _valid_manifest(root: Path) -> Path:
    arrays = root / "arrays"
    arrays.mkdir()
    array = np.ascontiguousarray(np.arange(12, dtype="<f8").reshape(3, 4))
    path = arrays / "000-theta.npy"
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    payload_sha = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    leaves = [{
        "slot_name": "theta", "kind": "array", "slot_index": 0,
        "shape": [3, 4], "dtype": array.dtype.str,
        "byte_order": array.dtype.byteorder, "nbytes": array.nbytes,
        "logical_c_bitpayload_sha256": payload_sha,
        "file": "arrays/000-theta.npy", "file_sha256": capture.sha256_file(path),
        "file_bytes": path.stat().st_size,
    }, {"slot_name": "optional", "kind": "none", "slot_index": 1}]
    pytree = [{
        "slot_index": 0, "slot_name": "theta", "path": "[<flat index 0>]",
        "shape": [3, 4], "dtype": array.dtype.str, "nbytes": array.nbytes,
        "logical_c_bitpayload_sha256": payload_sha,
    }]
    auxiliary_dir = root / "auxiliary"
    auxiliary_dir.mkdir()
    auxiliary = []
    aux_values = {
        "sfclay_ch": np.full((2, 2), 1.0, dtype="<f8"),
        "noahmp_land_ch": np.full((2, 2), 2.0, dtype="<f8"),
        "is_land": np.asarray([[True, False], [False, True]], dtype=bool),
    }
    aux_values["surface_ch"] = np.where(
        aux_values["is_land"], aux_values["noahmp_land_ch"], aux_values["sfclay_ch"]
    )
    for name in ("sfclay_ch", "noahmp_land_ch", "is_land", "surface_ch"):
        aux = np.ascontiguousarray(aux_values[name])
        aux_path = auxiliary_dir / f"{name}.npy"
        with aux_path.open("xb") as stream:
            np.save(stream, aux, allow_pickle=False)
        auxiliary.append({
            "name": name, "shape": list(aux.shape), "dtype": aux.dtype.str,
            "byte_order": aux.dtype.byteorder, "nbytes": aux.nbytes,
            "logical_c_bitpayload_sha256": hashlib.sha256(aux.tobytes(order="C")).hexdigest(),
            "file": f"auxiliary/{name}.npy", "file_sha256": capture.sha256_file(aux_path),
            "file_bytes": aux_path.stat().st_size,
        })
    value = {
        "schema": "wrfgpu2-v0234-authentic-pbl-entry-snapshot-v1",
        "nonce": capture.NONCE,
        "state_leaves": leaves,
        "capture_auxiliary": auxiliary,
        "state_inventory": {
            "state_slot_count": 2, "array_slot_count": 1, "none_slot_count": 1,
            "slot_names": ["theta", "optional"],
            "slot_names_sha256": capture.canonical_sha256(["theta", "optional"]),
            "direct_tree_child_count": 2, "direct_tree_aux_is_none": True,
            "pytree_leaf_count": 1, "pytree_inventory_sha256": capture.canonical_sha256(pytree),
            "pytree_leaves": pytree,
            "leaf_inventory_sha256": capture.canonical_sha256(leaves),
        },
    }
    value["canonical_payload_sha256"] = capture.canonical_without_self(value)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest


def test_module_import_is_backend_dark() -> None:
    forbidden = [name for name in sys.modules if name == "jax" or name.startswith(("jax.", "gpuwrf."))]
    assert forbidden == []


def test_exact_source_seam_and_state_schema_are_hash_bound() -> None:
    seam = capture.seam_authority()
    assert seam["assignment"] == "pbl_entry_state = next_state"
    assert seam["stop_line"] == seam["assignment_line"] + 1
    state = capture.state_schema_authority()
    assert state["state_slot_count"] == len(state["slot_names"])
    assert state["state_slot_count"] > 60


def test_canonical_json_refuses_payload_drift(tmp_path: Path) -> None:
    path = (tmp_path / "authority.json").resolve()
    value = {"schema": "test-v1", "complete": True}
    value["canonical_payload_sha256"] = capture.canonical_without_self(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded, record = capture.load_canonical_json(path)
    assert loaded["complete"] is True
    assert record["sha256"] == capture.sha256_file(path)
    value["complete"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(capture.GateError, match="CANONICAL_JSON_DRIFT"):
        capture.load_canonical_json(path)


class _TreeUtil:
    @staticmethod
    def tree_flatten_with_path(state):
        return [((0,), state.theta)], object()

    @staticmethod
    def keystr(path):
        return repr(path)


class _Jax:
    tree_util = _TreeUtil

    @staticmethod
    def device_get(value):
        return value


class _State:
    __slots__ = ("theta", "optional")

    def __init__(self):
        self.theta = np.arange(6, dtype="<f8").reshape(2, 3)
        self.optional = None

    def tree_flatten(self):
        return (self.theta, self.optional), None


def test_complete_ordered_slot_materialization_includes_none(tmp_path: Path) -> None:
    leaves, inventory = capture.materialize_state(
        _State(), tmp_path, _Jax, expected_slot_names=("theta", "optional")
    )
    assert [leaf["kind"] for leaf in leaves] == ["array", "none"]
    assert inventory["array_slot_count"] == 1
    assert inventory["none_slot_count"] == 1


def test_manifest_validator_accepts_complete_and_refuses_drift(tmp_path: Path) -> None:
    manifest = _valid_manifest(tmp_path)
    assert capture.validate_snapshot_manifest(
        manifest.resolve(),
        expected_nonce=capture.NONCE,
        expected_slot_names=("theta", "optional"),
    )["valid"] is True
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["state_leaves"][0]["shape"] = [4, 3]
    value["state_inventory"]["leaf_inventory_sha256"] = capture.canonical_sha256(value["state_leaves"])
    value["canonical_payload_sha256"] = capture.canonical_without_self(value)
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(capture.GateError):
        capture.validate_snapshot_manifest(
            manifest.resolve(),
            expected_nonce=capture.NONCE,
            expected_slot_names=("theta", "optional"),
        )


def test_manifest_nonce_and_namespace_are_explicit_and_stale_nonce_refuses(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest(tmp_path)
    with pytest.raises(TypeError, match="expected_nonce"):
        capture.validate_snapshot_manifest(manifest.resolve())
    with pytest.raises(capture.GateError, match="MANIFEST_NONCE"):
        capture.validate_snapshot_manifest(
            manifest.resolve(),
            expected_nonce=(
                "2e1ad51a081af14542d1326672c4306e5fbdd849259e4bd01fcb9ae753532d8e"
            ),
        )
    with pytest.raises(TypeError, match="output"):
        capture.assert_fresh_output_and_nonce()


def test_trace_stops_before_pbl_and_guard_is_terminal() -> None:
    sentinel = object()

    def fake(next_state):
        bl_opt = 5
        source_leaf_mode = True
        pbl_entry_state = next_state
        marker = "pbl-would-start"
        return pbl_entry_state, marker, bl_opt, source_leaf_mode

    source, first = inspect.getsourcelines(fake)
    stop_line = first + next(index for index, line in enumerate(source) if "marker =" in line)
    tracer = capture.SeamTracer(fake.__code__, stop_line)
    sys.settrace(tracer)
    try:
        with pytest.raises(capture.CaptureStop):
            fake(sentinel)
    finally:
        sys.settrace(None)
    assert tracer.state is sentinel and tracer.count == 1
    guard = capture.PBLGuard()
    with pytest.raises(capture.GateError, match="PBL_EXECUTION_FORBIDDEN"):
        guard()
    assert guard.count == 1


def test_surface_ch_auxiliary_is_exact_live_blend(tmp_path: Path) -> None:
    tracer = capture.SeamTracer((lambda: None).__code__, 1)
    tracer.noahmp_count = 1
    tracer.surface_sfclay_ch = np.asarray([[1.0, 1.5]], dtype="<f8")
    tracer.surface_noahmp_land_ch = np.asarray([[2.0, 2.5]], dtype="<f8")
    tracer.surface_is_land = np.asarray([[True, False]])
    records = capture.materialize_surface_ch_auxiliary(tracer, tmp_path, _Jax)
    assert [record["name"] for record in records] == [
        "sfclay_ch", "noahmp_land_ch", "is_land", "surface_ch"
    ]
    assert np.array_equal(
        np.load(tmp_path / "auxiliary/surface_ch.npy", allow_pickle=False),
        np.asarray([[2.0, 1.5]], dtype="<f8"),
    )


def test_noahmp_trace_captures_live_ch_seed_without_diagnostic_ch() -> None:
    seed = np.asarray([[1.0, 1.5]], dtype="<f8")
    land_ch = np.asarray([[2.0, 2.5]], dtype="<f8")
    mask = np.asarray([[True, False]])

    class LandStateOut:
        ch = land_ch

    def fake_noahmp_adapter(ch_seed, is_land):
        diag = object()  # SurfaceLayerDiagnostics likewise has no ``ch`` field.
        land_state_out = LandStateOut()
        return diag, land_state_out, ch_seed, is_land

    tracer = capture.SeamTracer(
        (lambda: None).__code__, 1, noahmp_code=fake_noahmp_adapter.__code__
    )
    sys.settrace(tracer)
    try:
        fake_noahmp_adapter(seed, mask)
    finally:
        sys.settrace(None)
    assert tracer.noahmp_count == 1
    assert tracer.surface_sfclay_ch is seed
    assert tracer.surface_noahmp_land_ch is land_ch
    assert tracer.surface_is_land is mask


def test_lock_command_and_environment_are_exact() -> None:
    command = capture.build_lock_command("a" * 40)
    assert command.count(str(capture.LOCK_WRAPPER)) == 1
    assert command[command.index("--intent") + 1] == "production-preemptible"
    assert command.count("--execute") == 1
    assert "nvidia-smi" not in command
    environment = capture.minimal_launch_environment()
    assert {name: environment[name] for name in capture.THREAD_ENV} == capture.THREAD_ENV
    assert environment["GPUWRF_WRF_ROOT"] == str(capture.AUTHORITY_ROOT / "runtime/wrf")


def test_corrected_arm_binds_fresh_nonce_namespace_and_patch() -> None:
    assert capture.NONCE == "ac6712170cbe508475ea8572464c7742a6906d559b53ed9930292d6822229ed8"
    assert "ac6712170cbe5084" in str(capture.OUTPUT_DIR)
    assert "0b18530a1dc9cac2" not in str(capture.OUTPUT_DIR)
    assert capture.CONTRACT_PATCH_COMMIT == "aa16ef9024ad3fcbfc1d37c65e09d3540ffa390b"
    assert capture.LOCK_LABEL.endswith("ac6712170cbe5084")


def test_source_orders_live_lock_before_model_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    execute = source.index("def execute_payload")
    live_lock = source.index("lock = assert_live_lock()", execute)
    runtime = source.index("result = _normal_load_and_capture", live_lock)
    dynamic_import = source.index("    import jax", source.index("def _normal_load_and_capture"))
    assert dynamic_import < execute
    assert live_lock < runtime
    assert "jax.devices" not in source
    assert "default_backend" not in source
    assert "nvidia-smi" not in source
