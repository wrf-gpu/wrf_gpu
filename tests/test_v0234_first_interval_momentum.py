"""Focused tests for the v0234 first-interval momentum short GPU arm.

CPU-only: no JAX import, no GPU query, no GPU lock.  Profile application is
exercised in a subprocess so module mutation cannot leak between tests.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from scripts import v0234_first_interval_momentum_runner_profile as profile
from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble

REPO = Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-first-interval-momentum-kimi"
BUNDLE_SHA = "470e6111d516479bed4bc0c3b2be1007bb082afd"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_subprocess(code: str) -> dict:
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(REPO),
        "GPUWRF_FIRST_INTERVAL_MOMENTUM": "1",
        "GPUWRF_NESTED_BUNDLE_APPROVED_SHA": BUNDLE_SHA,
    })
    completed = subprocess.run(
        ["<USER_HOME>/miniconda3/bin/python", "-c", code],
        cwd=REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_contract_constants_are_arithmetically_consistent() -> None:
    assert profile.FIRST_INTERVAL_OWN_STEPS == {
        "d01": profile.FIRST_INTERVAL_ROOT_STEPS,
        "d02": profile.FIRST_INTERVAL_ROOT_STEPS * 3,
        "d03": profile.FIRST_INTERVAL_ROOT_STEPS * 9,
    }
    assert profile.FIRST_INTERVAL_D03_STEPS == tuple(
        range(1, profile.FIRST_INTERVAL_ROOT_STEPS * 9 + 1)
    )
    assert profile.FIRST_INTERVAL_OWN_STEPS["d03"] == 207
    assert profile.FIRST_INTERVAL_LAST_ANALYZED_D03_STEP == 200
    assert len(profile.RETAINED_PIN_SHA256) == 64
    assert len(profile.RETAINED_PRODUCTION_HLO_SHA256) == 64
    assert len(profile.FIRST_INTERVAL_MODEL_SHA256) == 64
    tags = [tag for tag, _ in profile.SAVEPOINT_TAGS]
    assert tags == ["sp1_entry", "sp2_pbl", "sp3_tendf", "sp4_exit"]


def test_profile_binds_short_arm_and_audits_pass() -> None:
    code = """
import json
from scripts import v0234_nested_frozen_wrf_boundary_window as r
print(json.dumps({
  'namespace': r.REPAIRED_PRODUCTION_NAMESPACE,
  'label': r.LOCK_LABEL,
  'known': r.REQUIRE_KNOWN_1500_V10_RECORD,
  'focused': r.CPU_FOCUSED_TEST_ARGS,
  'schedule': r.schedule_clock_oracle(),
  'static': r.static_source_audit(),
  'launch': r.audit_exact_launch_command(r.LAUNCH_COMMAND),
}))
"""
    payload = _profile_subprocess(code)
    assert payload["namespace"] == profile.NAMESPACE
    assert payload["label"] == profile.LOCK_LABEL
    assert payload["known"] is False
    assert list(payload["focused"]) == list(profile.CPU_FOCUSED_TESTS)
    assert payload["schedule"]["passed"] is True
    assert payload["schedule"]["output_counts"] == {"d01": 1, "d02": 1, "d03": 2}
    assert payload["schedule"]["d03_output_steps"] == [0, 200]
    assert payload["static"]["passed"] is True
    assert payload["static"]["first_interval_short_arm"]["passed"] is True
    assert payload["launch"]["passed"] is True
    assert payload["launch"]["deterministic_load_pin_dump_pair"] is True


def test_launcher_has_one_canonical_lock_and_no_cache_or_gpu_query() -> None:
    launcher = SPRINT / "first-interval-momentum-exact-launch-command.sh"
    text = launcher.read_text()
    assert text.count("/scripts/with_gpu_lock.sh") == 1
    assert "--intent production-preemptible" in text
    assert "--xla_gpu_load_autotune_results_from=" in text
    assert "--xla_gpu_dump_autotune_results_to=" in text
    assert "GPUWRF_FIRST_INTERVAL_MOMENTUM=1" in text
    assert profile.RETAINED_PIN_SHA256 in text
    assert "JAX_ENABLE_COMPILATION_CACHE=false" in text
    assert "nvidia-smi" not in text
    assert "rocm-smi" not in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_retained_pin_and_production_hlo_authenticate() -> None:
    assert _sha256(profile.RETAINED_PIN) == profile.RETAINED_PIN_SHA256
    assert (
        _sha256(profile.RETAINED_PRODUCTION_HLO)
        == profile.RETAINED_PRODUCTION_HLO_SHA256
    )
    payload = json.loads(profile.RETAINED_PRODUCTION_HLO.read_text())
    extraction = payload["extraction"]
    assert extraction["extraction_complete"] is True
    assert extraction["targets"] == ["cusparse_gtsv2_ffi"]


def test_model_capture_is_diagnostic_only_and_production_untouched() -> None:
    import ast

    source = (REPO / profile.MODEL_FILE).read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    def keyword_default(fn_name: str, keyword: str) -> object:
        node = functions[fn_name]
        keys = node.args.kwonlyargs
        defaults = node.args.kw_defaults
        for key, default in zip(keys, defaults):
            if key.arg == keyword:
                return ast.literal_eval(default)
        raise AssertionError(f"{fn_name} lacks kwonly {keyword}")

    assert keyword_default("_physics_step_forcing", "capture_first_interval") is False
    assert (
        keyword_default(
            "_physics_boundary_step_with_limiter_diagnostics", "capture_first_interval"
        )
        is False
    )
    assert "class FirstIntervalMomentumRecord(NamedTuple):" in source
    assert "class FirstIntervalStepResult(NamedTuple):" in source
    record_block = source.split("class FirstIntervalMomentumRecord(NamedTuple):", 1)[1]
    assert all(
        field in record_block
        for field in ("rublten", "rvblten", "ru_tendf", "rv_tendf")
    )
    # The dycore-suboperator sprint adds one separate proof-only ladder entry
    # beside the two original first-interval diagnostic entries.
    assert source.count("capture_first_interval=True") == 3
    assert "def _physics_boundary_step_with_first_interval_ladder" in source
    wrapper_node = functions["_physics_boundary_step"]
    wrapper_text = ast.get_source_segment(source, wrapper_node)
    assert "capture_first_interval" not in wrapper_text
    model_sha256 = _sha256(REPO / profile.MODEL_FILE)
    if "def _physics_boundary_step_with_first_interval_ladder" in source:
        from scripts import v0234_dycore_suboperator_kimi_gpu_ladder as ladder_profile

        assert model_sha256 == ladder_profile.LADDER_MODEL_SHA256
    else:
        assert model_sha256 == profile.FIRST_INTERVAL_MODEL_SHA256


def test_wrf_reassemble_stagger_map_and_meta_parse(tmp_path: Path) -> None:
    for key, stagger in (
        ("sp1_entry__u", "u"),
        ("sp1_entry__v", "v"),
        ("sp4_exit__u", "u"),
        ("sp4_exit__v", "v"),
        ("sp3_tendf__ru_tendf", "u"),
        ("sp3_tendf__rv_tendf", "v"),
        ("sp2_pbl__rublten", "mass"),
        ("sp2_pbl__rvblten", "mass"),
    ):
        assert reassemble.FIELD_STAGGER[key] == stagger
    rank_dir = tmp_path / "rank0000"
    rank_dir.mkdir()
    (rank_dir / "meta.txt").write_text(
        "# WRFGPU2_MOMSP per-rank metadata (v0234 first interval)\n"
        "rank 0\n"
        "ids_ide_jds_jde_kds_kde  1 112 1 94 1 45\n"
        "ims_ime_jms_jme_kms_kme  1 111 1 93 1 45\n"
        "ips_ipe_jps_jpe_kps_kpe  1 111 1 93 1 45\n"
    )
    meta = reassemble.read_meta(rank_dir)
    assert meta["rank"] == [0]
    assert meta["ims_ime_jms_jme_kms_kme"] == [1, 111, 1, 93, 1, 45]
    values = np.arange(45 * 93 * 111, dtype=">f8").reshape(93, 45, 111)
    values.astype(">f8").tofile(rank_dir / "step000001_sp2_pbl__rublten.f64")
    assembled = reassemble.reassemble3d("sp2_pbl__rublten", 1, [dict(meta, dir=rank_dir)])
    assert assembled.shape == (44, 93, 111)
    np.testing.assert_allclose(assembled[0, 0, :5], values[0, 0, :5])
    np.testing.assert_allclose(assembled[-1, -1, -5:], values[-1, 44 - 1, -5:])


# ---------------------------------------------------------------------------
# Lazy capture binder (repairs the time-dim harness failure): jax-free stubs
# ---------------------------------------------------------------------------


class _StubGateError(Exception):
    pass


class _StubRunner:
    RunnerGateError = _StubGateError

    def __init__(self, leaf_count: int = 106) -> None:
        self.leaf_count = leaf_count
        self.preempt_checks: list[str] = []

    def carry_interface_signature(self, carry, runtime):
        return {"leaf_count": self.leaf_count, "leaves": [str(leaf) for leaf in carry]}

    def atomically_retain_lowered_hlo_audit(self, path, stablehlo):
        return {
            "path": str(path),
            "extraction": {
                "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
                "stablehlo_bytes": len(stablehlo.encode()),
            },
        }

    def evaluate_stablehlo_policy(self, stablehlo):
        return {"passed": True}

    def assert_preemption_clear(self, label):
        self.preempt_checks.append(label)

    def sha256_file(self, path):
        return _sha256(Path(path))


class _StubJnp:
    int32 = "int32"

    @staticmethod
    def asarray(value, dtype=None):
        return ("asarray", int(value), dtype)


class _StubRuntime:
    jnp = _StubJnp()


class _StubLowered:
    def __init__(self) -> None:
        self.compile_calls = 0

    def compiler_ir(self, dialect: str = "") -> str:
        return "stablehlo-stub"

    def compile(self):
        self.compile_calls += 1

        def _executable(carry, namelist, step, clock, cadence):
            return types.SimpleNamespace(carry=carry, record="record")

        return _executable


class _StubEntry:
    def __init__(self) -> None:
        self.lowered = _StubLowered()
        self.lower_args = None

    def lower(self, carry, namelist, step, clock, cadence):
        self.lower_args = (carry, namelist, step, clock, cadence)
        return self.lowered


def _binder_with_stub_entry(tmp_path: Path, monkeypatch, leaf_count: int = 106):
    entry = _StubEntry()
    stub_module = types.ModuleType("gpuwrf.runtime.operational_mode")
    stub_module.advance_one_step_with_first_interval_capture = entry
    monkeypatch.setitem(
        sys.modules, "gpuwrf.runtime.operational_mode", stub_module,
    )
    monkeypatch.setitem(sys.modules, "gpuwrf", types.ModuleType("gpuwrf"))
    monkeypatch.setitem(sys.modules, "gpuwrf.runtime", types.ModuleType("gpuwrf.runtime"))
    runner = _StubRunner(leaf_count=leaf_count)
    binder = profile._LazyFirstIntervalCaptureExecutable(
        runner, _StubRuntime(), tmp_path / "capture-hlo.json",
    )
    return runner, binder, entry


def test_lazy_capture_binds_once_on_first_live_carry(tmp_path: Path, monkeypatch) -> None:
    runner, binder, entry = _binder_with_stub_entry(tmp_path, monkeypatch)
    carry = tuple(object() for _ in range(106))
    result = binder.one_step(carry, "namelist", 1, "clock", 300)
    assert binder.compile_calls == 1
    assert binder.dispatch_calls == 1
    assert result.record == "record"
    assert entry.lower_args[2] == ("asarray", 1, "int32")
    assert entry.lower_args[4] == 300
    audit = binder.audit()
    assert audit["lazy_compile_trigger"] == "first-live-d03-carry"
    assert audit["lower_calls"] == 1 and audit["compile_calls"] == 1
    assert runner.preempt_checks == ["pre-capture-compile"]

    again = binder.one_step(carry, "namelist", 2, "clock", 300)
    assert binder.compile_calls == 1
    assert binder.dispatch_calls == 2
    assert again.carry is carry


def test_lazy_capture_fails_closed_on_signature_drift(tmp_path: Path, monkeypatch) -> None:
    runner, binder, _ = _binder_with_stub_entry(tmp_path, monkeypatch)
    carry = tuple(object() for _ in range(106))
    binder.one_step(carry, "namelist", 1, "clock", 300)
    drifted = tuple(object() for _ in range(105)) + ("different",)
    try:
        binder.one_step(drifted, "namelist", 2, "clock", 300)
    except _StubGateError as exc:
        assert "FIRST_INTERVAL_LIVE_SIGNATURE_DRIFT" in str(exc)
    else:
        raise AssertionError("expected FIRST_INTERVAL_LIVE_SIGNATURE_DRIFT")
    assert binder.compile_calls == 1


def test_lazy_capture_fails_closed_on_leaf_count(tmp_path: Path, monkeypatch) -> None:
    runner, binder, _ = _binder_with_stub_entry(tmp_path, monkeypatch, leaf_count=105)
    try:
        binder.one_step(tuple(object() for _ in range(105)), "namelist", 1, "clock", 300)
    except _StubGateError as exc:
        assert "FIRST_INTERVAL_LIVE_LEAF_COUNT" in str(exc)
    else:
        raise AssertionError("expected FIRST_INTERVAL_LIVE_LEAF_COUNT")
    assert binder.compile_calls == 0
