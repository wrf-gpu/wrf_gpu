"""Native single-domain (max_dom=1) checkpoint/resume through the shared driver.

CPU structural evidence: actual CLI routing, the production segment/checkpoint
loop, RestartStore generations, HistoryJournal and the real NetCDF writer.
Stepping is a deterministic sentinel (no physics claim); the real-case CPU
continuation proof lives in the lane's Swiss experiment (O1R-*).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics
from gpuwrf.diagnostics.census import initial_census
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry

NAME = "d01"
DT = 54.0  # 3 h = 200 root steps; 60-min history is the non-integral 66.67-step alarm
RAD_CADENCE = 33  # held-tendency refresh cadence of the sentinel stepper
SWISS = Path(__file__).resolve().parents[3] / "examples/switzerland_d01"


@pytest.fixture
def seed():
    grid = GridSpec.canary_3km_template()
    state = State(**{
        name: jnp.asarray(np.arange(np.prod(shape)).reshape(shape) / 991 + 1, dtype=DEFAULT_DTYPES.dtype_for(name))
        for name, shape in _state_field_shapes(grid).items()
    }).ensure_conditional_leaves(mp_physics=8)
    carry = initial_operational_carry(state)
    xy = jnp.full((grid.ny, grid.nx), 1.234e-4, dtype=jnp.float64)
    carry = carry.replace(
        cumulus_carry=(jnp.full_like(state.theta, 0.123), jnp.full_like(xy, 54, dtype=jnp.int32)),
        cumulus_tendencies=(*(jnp.full_like(state.theta, 1.2e-4) for _ in range(6)), xy),
        radiation_diagnostics=RRTMGRadiationDiagnostics(*(xy if i != 10 else jnp.ones_like(xy, dtype=jnp.int32) for i in range(15))),
        census=initial_census(),
    )
    namelist = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=DT, acoustic_substeps=4)
    return grid, namelist, carry


def _exact_tree(left, right):
    import jax
    a, ta = jax.tree.flatten(left)
    b, tb = jax.tree.flatten(right)
    assert ta == tb
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert (x.dtype, x.shape, x.tobytes()) == (y.dtype, y.shape, y.tobytes())


def _sentinel_step(carry, step):
    """Held-tendency + timer pattern: refresh a held rate on its cadence, apply every step."""
    held = carry.cumulus_tendencies
    if (step - 1) % RAD_CADENCE == 0:
        rate = carry.state.theta * 1.0e-6 + np.float32(step % 7) * 1.0e-3
        held = (rate.astype(held[0].dtype), *held[1:])
    theta = carry.state.theta + held[0].astype(carry.state.theta.dtype)
    census = carry.census._replace(work=carry.census.work + jnp.uint64(1))
    return carry.replace(state=carry.state.replace(_cast=False, theta=theta), cumulus_tendencies=held, census=census)


def _install_driver(monkeypatch, seed, finals):
    """Real single-domain driver orchestration + NetCDF writer; sentinel stepping."""
    import gpuwrf.diagnostics.census as census
    import gpuwrf.integration.nested_pipeline as pipeline
    import gpuwrf.io.gen2_accessor as accessor
    import gpuwrf.io.wrfout_writer as netcdf
    import gpuwrf.runtime.restart_store as storage
    from gpuwrf.runtime.domain_tree import DomainTreeResult

    grid, namelist, carry0 = seed
    run_start = datetime(2023, 1, 15)
    bundles = {NAME: SimpleNamespace(state=carry0.state, grid=grid, namelist=namelist)}
    tree = SimpleNamespace(persistent_state_bytes=lambda: {NAME: carry0.state.bytes()})
    monkeypatch.setattr(pipeline.DomainTree, "from_domains", lambda *a, **k: tree)
    monkeypatch.setattr(pipeline, "_load_domains", lambda config, names, **k: (
        SimpleNamespace(order=names, nests=()), bundles, {}, run_start, {NAME: DT}, {NAME: carry0}))
    monkeypatch.setattr(accessor, "Gen2Run", lambda _: None)
    monkeypatch.setattr(pipeline, "_output_cadence_steps_by_domain", lambda *a: ({NAME: 67}, {NAME: 60}))
    monkeypatch.setattr(pipeline, "maybe_prewarm_defused_nest", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_prepare_operational_domain_tree_runtime", lambda *a, **k: None)
    monkeypatch.setattr(storage, "nested_identity", lambda config, *a: {"test": "single-domain", "output_dir": str(Path(config.output_dir).resolve())})
    for flag in ("GPUWRF_NESTED_ASYNC_OUTPUT", "GPUWRF_NEST_OUTPUT_PIPELINE", "GPUWRF_NEST_PERF_TIMERS"):
        monkeypatch.setenv(flag, "0")
    source = accessor.Gen2GridSpec(
        id=NAME, dx_m=3000., dy_m=3000., e_we=grid.nx + 1, e_sn=grid.ny + 1, e_vert=grid.nz + 1,
        mass_nx=grid.nx, mass_ny=grid.ny, mass_nz=grid.nz, grid_proj="lambert", map_proj_id=1,
        cen_lat=46.8, cen_lon=8.2, truelat1=45., truelat2=48., stand_lon=8.2,
        parent_id=1, parent_grid_ratio=1, i_parent_start=1, j_parent_start=1,
        znu=tuple(range(grid.nz)), znw=tuple(range(grid.nz + 1)), top_pressure_pa=5000.,
        source_wrfout="authenticated-test", source_namelist="authenticated-test")
    authorities = {NAME: netcdf.bind_wrfout_domain_authority(NAME, source, grid)}
    writers = []

    class Writer(pipeline._PerDomainWrfoutWriter):
        # Stub case setup/diagnostics only; keep the actual callback, step-0
        # routing, materialize/submit, journal publication and NetCDF writer.
        def __init__(self, *, output_dir, **kwargs):
            self.output_dir = output_dir
            self.run_start = kwargs["run_start"]
            self.bundles = kwargs["bundles"]
            self.dt_by_domain = kwargs["dt_by_domain"]
            self.written = {NAME: []}
            self.writer_static_latlon_metadata = {}
            self.writer_diagnostics = {}
            self.domain_authorities = authorities
            self.census_io_ledger = census.WriterIoLedger()
            self._census_persist_info = {}
            self._perf_timers = pipeline._NestedOutputPerfTimers.disabled()
            self._variable_subset = None
            self._full_variable_set = False
            self._async_writer = kwargs.get("async_writer")
            self._output_pipeline = None
            self._surface_diagnostics_for_output = lambda *a, **k: {}
            self._merge_output_diagnostics = lambda a, b: b
            self._initial_surface_fields = lambda name: {}
            writers.append(self)

    monkeypatch.setattr(pipeline, "_wrfout_path", lambda folder, name, valid: folder / f"wrfout_{name}_{round((valid - run_start).total_seconds() / DT)}")

    def prepare(state, grid, namelist, path, **kwargs):
        return netcdf.PreparedWrfout(
            path, netcdf._dimension_sizes(nx=grid.nx, ny=grid.ny, nz=grid.nz, namelist=namelist),
            {"T2": np.asarray(state.theta[0], dtype=np.float32)}, run_start, kwargs["valid_time"],
            kwargs["lead_hours"], grid, namelist, kwargs["domain"], kwargs["domain_authority"])

    monkeypatch.setattr(pipeline, "prepare_wrfout_payload", prepare)
    monkeypatch.setattr(pipeline, "_PerDomainWrfoutWriter", Writer)

    def advance(tree, *, root_steps, carries, initial_own_steps, output, output_alarm_steps, **kwargs):
        carry, step0 = carries[NAME], initial_own_steps[NAME]
        for step in range(step0 + 1, step0 + root_steps + 1):
            carry = _sentinel_step(carry, step)
            if step in output_alarm_steps[NAME]:
                output(NAME, step, carry)
        finals.append(carry)
        own = {NAME: step0 + root_steps}
        return DomainTreeResult({NAME: carry}, {NAME: carry.state}, own, (), (), cascade_counts={})

    monkeypatch.setattr(pipeline, "run_operational_domain_tree", advance)
    return pipeline, storage, writers


def _netcdf_payload(path):
    from netCDF4 import Dataset
    with Dataset(path) as data:
        data.set_auto_mask(False)
        return {key: (np.asarray(var[...]).dtype.str, np.asarray(var[...]).tobytes()) for key, var in data.variables.items()}


def _assert_same_history(control_dir, candidate_dir):
    names = sorted(p.name for p in control_dir.glob("wrfout_*"))
    assert names == sorted(p.name for p in candidate_dir.glob("wrfout_*"))
    for name in names:
        assert _netcdf_payload(control_dir / name) == _netcdf_payload(candidate_dir / name), name


# --------------------------------------------------------------------------- CLI


def _cli_capture(monkeypatch):
    from gpuwrf.integration import daily_pipeline as daily
    from gpuwrf.integration import nested_pipeline as shared
    configs = []

    def execute(config):
        configs.append(config)
        return {"verdict": "PIPELINE_GREEN", "wrfout_files": [], "metadata": {}}

    monkeypatch.setattr(shared, "execute_nested_pipeline", execute)
    monkeypatch.setattr(daily, "execute_daily_pipeline", lambda config: pytest.fail("native restart fell back to daily"))
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
    return configs


def test_cli_native_single_domain_accepts_checkpoint_and_resume(tmp_path, monkeypatch, capsys):
    from gpuwrf import cli
    configs = _cli_capture(monkeypatch)
    base = ["run", "--input-dir", str(SWISS), "--output-dir", str(tmp_path / "out"), "--hours", "3",
            "--scratch-dir", str(tmp_path / "scratch")]
    assert cli.main([*base, "--checkpoint-dir", str(tmp_path / "ck"), "--checkpoint-interval-steps", "100"]) == 0
    assert cli.main([*base, "--resume-checkpoint", str(tmp_path / "ck" / "gen"), "--hours", "6", "--extend-run"]) == 0
    first, resumed = configs
    assert (first.max_dom, first.checkpoint_dir, first.checkpoint_interval_steps) == (1, tmp_path / "ck", 100)
    assert first.emit_initial_history is True and first.extend_end_time is False
    assert (resumed.max_dom, resumed.resume_checkpoint, resumed.hours, resumed.extend_end_time) == (1, tmp_path / "ck" / "gen", 6, True)
    capsys.readouterr()
    assert cli.main([*base, "--checkpoint-dir", str(tmp_path / "ck"), "--checkpoint-interval-steps", "100", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["run_type"] == "single_domain" and plan["restart"]["checkpoint_interval_steps"] == 100


def test_cli_refuses_restart_where_no_restart_store_exists(tmp_path, monkeypatch, capsys):
    from gpuwrf import cli
    _cli_capture(monkeypatch)
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "namelist.input").write_bytes((SWISS / "namelist.input").read_bytes())
    for name in ("wrfinput_d01", "wrfbdy_d01"):
        (replay / name).symlink_to(SWISS / name)
    for hour in (0, 1):  # CPU-WRF replay mode (filename detection)
        (replay / f"wrfout_d01_2023-01-15_{hour:02d}:00:00").touch()
    common = ["--output-dir", str(tmp_path / "out"), "--hours", "1", "--scratch-dir", str(tmp_path / "s")]
    assert cli.main(["run", "--input-dir", str(replay), *common, "--checkpoint-dir", str(tmp_path / "c"),
                     "--checkpoint-interval-steps", "10"]) == 2
    assert "CPU-WRF replay" in capsys.readouterr().err
    assert cli.main(["run", "--input-dir", str(SWISS), *common, "--domain", "d02",
                     "--resume-checkpoint", str(tmp_path / "g")]) == 2
    assert "requires the native driver" in capsys.readouterr().err
    assert cli.main(["run", "--input-dir", str(SWISS), *common, "--extend-run"]) == 2
    assert "--extend-run requires --resume-checkpoint" in capsys.readouterr().err


# ----------------------------------------------------------------- driver/store

CRASHES = ("kill_after_verified", "capture", "data_write", "publish_rename", "after_publish",
           "history_after_link", "history_intent")


@pytest.mark.parametrize("crash", CRASHES)
def test_single_domain_crash_injection_resume_matches_uninterrupted(seed, tmp_path, monkeypatch, crash):
    finals = []
    pipeline, storage, writers = _install_driver(monkeypatch, seed, finals)
    import gpuwrf.io.wrfout_writer as netcdf
    config = pipeline.NestedPipelineConfig(tmp_path / "in", tmp_path / "control", tmp_path / "proof", 3, 1,
                                           emit_initial_history=True)
    control = pipeline.execute_nested_pipeline(config)
    assert control["verdict"] == "PIPELINE_GREEN"
    expected_carry, control_ledger = finals[-1], writers[-1].census_io_ledger.snapshot()
    assert [Path(p).name for p in writers[-1].written[NAME]] == [f"wrfout_d01_{s}" for s in (0, 67, 134, 200)]

    config = replace(config, output_dir=tmp_path / "run", checkpoint_dir=tmp_path / "ck",
                     checkpoint_interval_steps=50, checkpoint_reserve_bytes=0)
    save, publish, link, rename = (storage.RestartStore.save, storage.HistoryJournal.publish,
                                   netcdf._publish_wrfout_noreplace, storage.os.rename)
    payload, original_open = storage._carry_to_payload, Path.open
    second = {"own_step": 100}

    def manifest_step(temporary):
        return json.loads((Path(temporary) / "manifest.json").read_text())["own_steps"][NAME]

    if crash == "kill_after_verified":
        def kill(self, carries, own_steps, *a, **k):
            result = save(self, carries, own_steps, *a, **k)
            if own_steps[NAME] == second["own_step"]:
                raise OSError("injected interruption after verified generation")
            return result
        monkeypatch.setattr(storage.RestartStore, "save", kill)
    elif crash == "capture":
        calls = []

        def capture(carry):
            calls.append(1)
            if len(calls) == 2:  # second checkpoint: device->host capture of the carry
                raise OSError("injected crash during carry capture")
            return payload(carry)
        monkeypatch.setattr(storage, "_carry_to_payload", capture)
    elif crash == "data_write":
        def disk_full(path, *args, **kwargs):
            if path.name == "snapshot.pkl" and args and args[0] == "xb" and path.parent.name.startswith(".partial-") \
                    and len(list(path.parent.parent.glob("generation-*"))) == 1:
                raise OSError("injected disk full during snapshot data write")
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", disk_full)
    elif crash == "publish_rename":
        def fail_rename(source, target):
            if Path(source).name.startswith(".partial-") and manifest_step(source) == second["own_step"]:
                raise OSError("injected crash at generation publication (rename)")
            return rename(source, target)
        monkeypatch.setattr(storage.os, "rename", fail_rename)
    elif crash == "after_publish":
        def published_then_crash(source, target):
            step = manifest_step(source) if Path(source).name.startswith(".partial-") else None
            rename(source, target)
            if step == second["own_step"]:
                raise OSError("injected crash after generation rename, before readback")
        monkeypatch.setattr(storage.os, "rename", published_then_crash)
    elif crash == "history_after_link":
        def crash_after_frame(self, temporary, target):
            publish(self, temporary, target)
            if Path(target).name == "wrfout_d01_134":
                raise OSError("injected crash after history publication")
        monkeypatch.setattr(storage.HistoryJournal, "publish", crash_after_frame)
    else:
        def crash_after_intent(temporary, target):
            if Path(target).name == "wrfout_d01_134":
                raise OSError("injected crash after durable history intent")
            return link(temporary, target)
        monkeypatch.setattr(netcdf, "_publish_wrfout_noreplace", crash_after_intent)

    with pytest.raises(OSError, match="injected"):
        pipeline.execute_nested_pipeline(config)
    for target, value in ((storage.RestartStore, ("save", save)), (storage.HistoryJournal, ("publish", publish)),
                          (netcdf, ("_publish_wrfout_noreplace", link)), (storage.os, ("rename", rename)),
                          (storage, ("_carry_to_payload", payload)), (Path, ("open", original_open))):
        monkeypatch.setattr(target, *value)

    store = storage.RestartStore(config.checkpoint_dir, 10**8, 2, 0)
    latest = store.latest()
    latest_step = store.read(latest, device=False)[1]["own_steps"][NAME]
    # Interrupted publication never yields a torn successor: either the previous
    # verified generation stays latest, or the fully written successor is.
    assert latest_step == (50 if crash in ("capture", "data_write", "publish_rename") else 100)
    if crash in ("data_write", "publish_rename"):
        assert list(config.checkpoint_dir.glob(".partial-*")), "failed publication evidence is retained"
    committed = {p: (p.stat().st_ino, p.read_bytes()) for p in config.output_dir.glob("wrfout_*")
                 if int(p.name.rsplit("_", 1)[1]) <= latest_step}

    result = pipeline.execute_nested_pipeline(replace(config, resume_checkpoint=latest))
    assert result["verdict"] == "PIPELINE_GREEN"
    assert result["hierarchy"]["observed_own_steps"] == {NAME: 200}
    assert result["restart"]["end_time"] == {"checkpoint_hours": 3.0, "hours": 3.0, "extended": False}
    # No lost or duplicated history: lead zero is not re-emitted, every alarm exactly once.
    assert [Path(p).name for p in writers[-1].written[NAME]] == [f"wrfout_d01_{s}" for s in (0, 67, 134, 200)]
    assert result["per_domain"][NAME]["wrfout_count"] == result["per_domain"][NAME]["expected_wrfout_count"] == 4
    assert writers[-1].census_io_ledger.snapshot() == control_ledger
    for path, (inode, data) in committed.items():
        assert (path.stat().st_ino, path.read_bytes()) == (inode, data), "committed history must stay untouched"
    _assert_same_history(tmp_path / "control", config.output_dir)
    _exact_tree(expected_carry, finals[-1])


def test_single_domain_controlled_end_time_extension(seed, tmp_path, monkeypatch):
    finals = []
    pipeline, storage, writers = _install_driver(monkeypatch, seed, finals)
    base = pipeline.NestedPipelineConfig(tmp_path / "in", tmp_path / "control", tmp_path / "proof", 6, 1,
                                         emit_initial_history=True, checkpoint_dir=tmp_path / "ck_control",
                                         checkpoint_interval_steps=100, checkpoint_reserve_bytes=0)
    assert pipeline.execute_nested_pipeline(base)["verdict"] == "PIPELINE_GREEN"
    expected = finals[-1]
    alarms = (0, 67, 134, 200, 267, 334, 400)
    assert [Path(p).name for p in writers[-1].written[NAME]] == [f"wrfout_d01_{s}" for s in alarms]

    short = replace(base, hours=3, output_dir=tmp_path / "run", checkpoint_dir=tmp_path / "ck")
    first = pipeline.execute_nested_pipeline(short)
    assert first["verdict"] == "PIPELINE_GREEN" and first["hierarchy"]["observed_own_steps"] == {NAME: 200}
    store = storage.RestartStore(short.checkpoint_dir, 10**8, 2, 0)
    end = store.latest()
    assert store.read(end, device=False)[0]["driver_state"]["forecast_hours"] == 3
    files_before = {p: p.read_bytes() for p in short.output_dir.glob("wrfout_*")}

    with pytest.raises(ValueError, match="pass --extend-run"):
        pipeline.execute_nested_pipeline(replace(short, hours=6, resume_checkpoint=end))
    with pytest.raises(ValueError, match="cannot shorten"):
        pipeline.execute_nested_pipeline(replace(short, hours=2, resume_checkpoint=end))
    with pytest.raises(ValueError, match="requires a resume checkpoint"):
        pipeline.execute_nested_pipeline(replace(short, hours=6, extend_end_time=True))
    assert {p: p.read_bytes() for p in short.output_dir.glob("wrfout_*")} == files_before, "refusals write nothing"

    extended = pipeline.execute_nested_pipeline(replace(short, hours=6, resume_checkpoint=end, extend_end_time=True))
    assert extended["verdict"] == "PIPELINE_GREEN"
    assert extended["restart"]["end_time"] == {"checkpoint_hours": 3.0, "hours": 6.0, "extended": True}
    assert extended["hierarchy"]["observed_own_steps"] == {NAME: 400}
    assert [Path(p).name for p in writers[-1].written[NAME]] == [f"wrfout_d01_{s}" for s in alarms]
    _assert_same_history(tmp_path / "control", short.output_dir)
    _exact_tree(expected, finals[-1])
    # Later generations of the extended stream record the new end time.
    assert store.read(store.latest(), device=False)[0]["driver_state"]["forecast_hours"] == 6


def test_identity_excludes_end_time_but_binds_the_stream(tmp_path, monkeypatch):
    import gpuwrf.runtime.restart_store as storage
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    bundles = {NAME: SimpleNamespace(namelist=SimpleNamespace())}
    monkeypatch.setattr("gpuwrf.runtime.aot_cheap_key.canonical_digest", lambda value: "digest")
    monkeypatch.setattr("gpuwrf.physics.rrtmg_mp_re.mp_re_config", lambda value: None)
    config = NestedPipelineConfig(SWISS, tmp_path / "out", tmp_path / "proof", 3, 1, emit_initial_history=True)
    start = datetime(2023, 1, 15)
    identity = storage.nested_identity(config, bundles, start)
    assert "hours" not in identity
    assert storage.nested_identity(replace(config, hours=24), bundles, start) == identity
    assert storage.nested_identity(replace(config, output_dir=tmp_path / "other"), bundles, start) != identity
    assert identity["inputs"].keys() >= {"wrfinput_d01", "wrfbdy_d01", "namelist.input"}
