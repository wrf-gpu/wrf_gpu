"""Restart safety checks; CPU structural evidence, not forecast fidelity."""
from dataclasses import fields
import json
from pathlib import Path
import pickle
import shutil
import os
import subprocess
import sys
import hashlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics
from gpuwrf.diagnostics.census import initial_census
from gpuwrf.io.restart import write_restart, read_restart, _CARRY_FIELD_ORDER
from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore


@pytest.fixture
def initialized():
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.asarray(np.arange(np.prod(shape)).reshape(shape) / 991 + 1, dtype=DEFAULT_DTYPES.dtype_for(name)) for name, shape in _state_field_shapes(grid).items()})
    carry = initial_operational_carry(state)
    xy = jnp.full((grid.ny, grid.nx), 1.234e-4, dtype=jnp.float64)
    carry = carry.replace(
        cumulus_carry=(jnp.full_like(state.theta, 0.123), jnp.full_like(xy, 54, dtype=jnp.int32)),
        cumulus_tendencies=(*(jnp.full_like(state.theta, 1.2e-4) for _ in range(6)), xy),
        radiation_diagnostics=RRTMGRadiationDiagnostics(*(xy if i != 10 else jnp.ones_like(xy, dtype=jnp.int32) for i in range(15))),
        census=initial_census()._replace(work=initial_census().work + jnp.uint64(7)),
        slab_rad=(xy, xy), px_rad=(xy, xy),
    )
    namelist = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=54, acoustic_substeps=4)
    return grid, namelist, carry


def exact(left, right):
    a, ta = jax.tree.flatten(left)
    b, tb = jax.tree.flatten(right)
    assert ta == tb
    for x, y in zip(a, b, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype
        assert x.tobytes() == y.tobytes()


def test_all_carry_roundtrips_in_both_formats(initialized, tmp_path):
    grid, namelist, carry = initialized
    assert _CARRY_FIELD_ORDER == tuple(f.name for f in fields(OperationalCarry))
    path = tmp_path / "carry.pkl"
    write_restart(carry, namelist, grid, 11, path)
    restored, _, _, step = read_restart(path)
    assert step == 11
    exact(carry, restored)
    path = tmp_path / "carry.nc"
    write_wrfrst_carry(carry, grid, {}, path, valid_time="2026-07-25_18:09:54", run_start="2026-07-25_18:00:00", step_index=11)
    restored, _ = read_wrfrst_carry(path)
    exact(carry, restored)


def test_rotation_hash_and_synchronized_domain_set(initialized, tmp_path):
    _, _, carry = initialized
    store = RestartStore(tmp_path, 10**8, 2, 0)
    generations = []
    for step in range(1, 4):
        generation, receipt = store.save({"d01": carry, "d02": carry}, {"d01": step, "d02": step * 3}, {"d01": 54., "d02": 18.}, {"run": "P1"})
        generations.append(generation)
        assert receipt["retained_count"] <= 2
        assert receipt["retained_bytes"] <= store.max_bytes
        assert receipt["publication_peak_bytes"] <= store.max_bytes
    assert not generations[0].exists()
    assert store.latest() == generations[-1]
    snapshot, _ = store.read(generations[-1], expected_identity={"run": "P1"})
    exact(carry, snapshot["carries"]["d02"])
    path = generations[-1] / "snapshot.pkl"
    with path.open("r+b") as stream:
        stream.seek(40)
        stream.write(b"CORRUPT")
    with pytest.raises(ValueError, match="hash"):
        store.read(generations[-1])
    assert store.latest() == generations[-2]
    assert generations[-1].exists(), "corrupt evidence must be retained"
    with pytest.raises(ValueError, match="identity"):
        store.read(generations[-2], expected_identity={"run": "OTHER"})
    with pytest.raises(ValueError, match="synchronized"):
        store.save({"d01": carry, "d02": carry}, {"d01": 1, "d02": 2}, {"d01": 54., "d02": 18.}, {})


@pytest.mark.parametrize("failure", ["reserve", "budget", "partial", "readback"])
def test_failed_successor_retains_last_verified(initialized, tmp_path, monkeypatch, failure):
    _, _, carry = initialized
    store = RestartStore(tmp_path, 10**8, 2, 100)
    previous, _ = store.save({"d01": carry}, {"d01": 1}, {"d01": 54.}, {})
    if failure == "reserve":
        monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    elif failure == "budget":
        store = RestartStore(tmp_path, 1, 2, 0)
    elif failure == "partial":
        original = Path.open
        def fail(path, *args, **kwargs):
            if path.name == "manifest.json" and args and args[0] == "xb":
                raise OSError("injected disk full during write")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", fail)
    else:
        original = RestartStore.read
        def fail(self, path, **kwargs):
            if Path(path).name.startswith(".partial-"):
                raise ValueError("injected readback failure")
            return original(self, path, **kwargs)
        monkeypatch.setattr(RestartStore, "read", fail)
    with pytest.raises((OSError, ValueError)):
        store.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {})
    assert store.latest() == previous
    assert previous.exists()
    if failure in {"partial", "readback"}:
        assert list(tmp_path.glob(".partial-*"))


def test_mixed_domain_manifest_rejected(initialized, tmp_path):
    _, _, carry = initialized
    store = RestartStore(tmp_path, 10**8, 2, 0)
    generation, _ = store.save({"d01": carry, "d02": carry}, {"d01": 1, "d02": 3}, {"d01": 54., "d02": 18.}, {})
    path = generation / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["domains"] = ["d01"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing or mixed"):
        store.read(generation)


def test_corrupt_evidence_counts_toward_budget(initialized, tmp_path):
    _, _, carry = initialized
    store = RestartStore(tmp_path, 10**8, 2, 0)
    generation, receipt = store.save({"d01": carry}, {"d01": 1}, {"d01": 54.}, {})
    with (generation / "snapshot.pkl").open("r+b") as stream:
        stream.write(b"CORRUPT")
    bounded = RestartStore(tmp_path, receipt["bytes"] * 2 - 1, 2, 0)
    with pytest.raises(OSError, match="byte budget"):
        bounded.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {})
    assert generation.exists()
    bounded = RestartStore(tmp_path, 10**8, 1, 0)
    with pytest.raises(OSError):
        bounded.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {})


def test_store_refuses_rotation_of_other_run(initialized, tmp_path):
    _, _, carry = initialized
    store = RestartStore(tmp_path, 10**8, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 1}, {"d01": 54.}, {"run": "A"})
    with pytest.raises(ValueError, match="different run"):
        store.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {"run": "B"})
    assert store.latest() == generation


def test_publication_budget_preserves_latest_with_tight_capacity(initialized, tmp_path):
    _, _, carry = initialized
    store = RestartStore(tmp_path, 10**8, 2, 0)
    latest, receipt = store.save({"d01": carry}, {"d01": 1}, {"d01": 54.}, {})
    # Space for only one generation cannot fund a safe successor.
    bounded = RestartStore(tmp_path, receipt["bytes"] * 2 - 1, 2, 0)
    with pytest.raises(OSError, match="budget"):
        bounded.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {})
    assert bounded.latest() == latest
    with pytest.raises(OSError, match="budget"):
        RestartStore(tmp_path, 10**8, 1, 0).save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {})
    assert latest.exists()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_mixed_precision_state_and_noah_land_preserve_stored_dtype(initialized, tmp_path, dtype):
    from gpuwrf.contracts.noahmp_state import NoahMPLandState
    grid, namelist, carry = initialized
    soil = {"tslb", "smois", "sh2o"}
    snow = {"tsno", "snice", "snliq"}
    values = {}
    for name in NoahMPLandState.__slots__:
        shape = ((4,) if name in soil else (3,) if name in snow else (7,) if name == "zsnso" else ()) + (grid.ny, grid.nx)
        values[name] = jnp.full(shape, 1.25, dtype=jnp.int32 if name == "isnow" else dtype)
    state = carry.state.replace(_cast=False, theta=carry.state.theta.astype(jnp.float32), u_bdy=carry.state.u_bdy.astype(jnp.float64))
    carry = carry.replace(state=state, noahmp_land=NoahMPLandState(**values))
    path = tmp_path / "mixed.pkl"
    write_restart(carry, namelist, grid, 1, path)
    exact(carry, read_restart(path)[0])
    path = tmp_path / "mixed.nc"
    write_wrfrst_carry(carry, grid, {}, path, valid_time="2026-07-25_18:00:54", run_start="2026-07-25_18:00:00", step_index=1)
    exact(carry, read_wrfrst_carry(path)[0])


def test_nonfinite_carry_refused_before_publication(initialized, tmp_path):
    _, _, carry = initialized
    store = RestartStore(tmp_path, 10**8, 2, 0)
    previous, _ = store.save({"d01": carry}, {"d01": 1}, {"d01": 54.}, {})
    broken = carry.replace(state=carry.state.replace(_cast=False, theta=carry.state.theta.at[0, 0, 0].set(jnp.nan)))
    with pytest.raises(ValueError, match="nonfinite"):
        store.save({"d01": broken}, {"d01": 2}, {"d01": 54.}, {})
    assert store.latest() == previous
    assert not list(tmp_path.glob(".partial-*"))


def test_cpu_fresh_process_and_legacy_v1(initialized, tmp_path):
    import gpuwrf
    grid, namelist, carry = initialized
    path = tmp_path / "fresh.pkl"
    write_restart(carry, namelist, grid, 11, path)
    expected = [hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest() for leaf in jax.tree.leaves(carry)]
    script = """import hashlib,json,sys,jax,numpy as np
from gpuwrf.io.restart import read_restart
c,_,_,step=read_restart(sys.argv[1]); assert step==11
print(json.dumps([hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in jax.tree.leaves(c)]))
"""
    # pytest adds src to the parent's sys.path, which a fresh interpreter does
    # not inherit. Bind the child to the same imported checkout explicitly.
    source = str(Path(gpuwrf.__file__).resolve().parents[1])
    pythonpath = os.pathsep.join(filter(None, (source, os.environ.get("PYTHONPATH", ""))))
    result = subprocess.run([sys.executable, "-c", script, str(path)], env={**os.environ, "PYTHONPATH": pythonpath, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(result.stdout) == expected
    # A real v1 payload has no extra trees; preserve its absent fields exactly.
    legacy = initial_operational_carry(carry.state)
    write_restart(legacy, namelist, grid, 1, path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    payload["format_version"] = 1
    del payload["carry"]["carry_field_order"], payload["carry"]["extra_fields"]
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    restored, _, _, _ = read_restart(path)
    exact(legacy, restored)


def test_output_seam_receipts(initialized, tmp_path):
    from gpuwrf.runtime.restart_store import output_receipts, restore_output_receipts
    path = tmp_path / "wrfout"
    path.write_bytes(b"committed history frame")
    receipts = output_receipts({"d01": [str(path)]})
    writer = SimpleNamespace(written={"d01": []})
    restore_output_receipts(writer, receipts)
    assert writer.written == {"d01": [str(path)]}
    path.write_bytes(b"partial frame")
    with pytest.raises(ValueError, match="history frame"):
        restore_output_receipts(writer, receipts)


def test_production_segment_hooks_resume_clock_and_history(initialized, tmp_path, monkeypatch):
    """Actual driver orchestration with a stub step/writer; no physics claim."""
    from dataclasses import replace
    import gpuwrf.integration.nested_pipeline as pipeline
    import gpuwrf.io.gen2_accessor as accessor
    import gpuwrf.runtime.restart_store as storage
    from gpuwrf.runtime.domain_tree import DomainTreeResult
    import gpuwrf.diagnostics.census as census
    grid, namelist, seed = initialized
    names = ("d01", "d02")
    bundles = {name: SimpleNamespace(state=seed.state, grid=grid, namelist=namelist) for name in names}
    tree = SimpleNamespace(persistent_state_bytes=lambda: {name: seed.state.bytes() for name in names})
    monkeypatch.setattr(pipeline.DomainTree, "from_domains", lambda *a, **k: tree)
    monkeypatch.setattr(pipeline, "_load_domains", lambda *a, **k: (SimpleNamespace(order=names, nests=()), bundles, {}, __import__("datetime").datetime(2026, 7, 25), {"d01": 54., "d02": 18.}, {name: seed for name in names}))
    monkeypatch.setattr(accessor, "Gen2Run", lambda _: None)
    monkeypatch.setattr(pipeline, "_output_cadence_steps_by_domain", lambda *a: ({"d01": 67, "d02": 200}, dict.fromkeys(names, 60)))
    alarms = {"d01": (67, 134, 200), "d02": (200, 400, 600)}
    monkeypatch.setattr(pipeline, "_output_alarm_steps_by_domain", lambda *a: (alarms, alarms))
    monkeypatch.setattr(pipeline, "maybe_prewarm_defused_nest", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_prepare_operational_domain_tree_runtime", lambda *a, **k: None)
    monkeypatch.setattr(storage, "nested_identity", lambda *a: {"test": "driver-orchestration", "output_dir": str(a[0].output_dir.resolve())})
    monkeypatch.setattr(census, "write_segment", lambda *a: None)
    monkeypatch.setenv("GPUWRF_NESTED_ASYNC_OUTPUT", "0")
    monkeypatch.setenv("GPUWRF_NEST_OUTPUT_PIPELINE", "0")
    monkeypatch.setenv("GPUWRF_NEST_PERF_TIMERS", "0")
    class Writer:
        def __init__(self, *, output_dir, **kwargs):
            self.output_dir = output_dir
            self.written = {name: [] for name in names}
            self.writer_static_latlon_metadata = {}
        def __call__(self, name, step, carry):
            path = self.output_dir / f"wrfout_{name}_{step}"
            path.write_bytes(np.asarray(carry.state.theta).tobytes())
            self.written[name].append(str(path))
    monkeypatch.setattr(pipeline, "_PerDomainWrfoutWriter", Writer)
    final_carries = []
    def advance(tree, *, root_steps, carries, initial_own_steps, output, **kwargs):
        own_steps = {name: initial_own_steps[name] + root_steps * (1 if name == "d01" else 3) for name in names}
        new = {name: carry.replace(state=carry.state.replace(_cast=False, theta=carry.state.theta + own_steps[name] - initial_own_steps[name])) for name, carry in carries.items()}
        for name in names:
            for step in alarms[name]:
                if initial_own_steps[name] < step <= own_steps[name]:
                    output(name, step, new[name])
        final_carries.append(new)
        return DomainTreeResult(new, {n: c.state for n, c in new.items()}, own_steps, (), (), cascade_counts={})
    monkeypatch.setattr(pipeline, "run_operational_domain_tree", advance)
    config = pipeline.NestedPipelineConfig(tmp_path / "inputs", tmp_path / "control", tmp_path / "proof", 3, 2)
    control = pipeline.execute_nested_pipeline(config)
    expected = final_carries[-1]
    interrupted_config = replace(config, output_dir=tmp_path / "interrupted", checkpoint_dir=tmp_path / "checkpoints", checkpoint_interval_steps=67, checkpoint_reserve_bytes=0)
    original_save = RestartStore.save
    def interrupted(*a, **k):
        original_save(*a, **k)
        raise OSError("external interruption after verified successor")
    monkeypatch.setattr(RestartStore, "save", interrupted)
    with pytest.raises(OSError, match="interruption"):
        pipeline.execute_nested_pipeline(interrupted_config)
    monkeypatch.setattr(RestartStore, "save", original_save)
    store = RestartStore(interrupted_config.checkpoint_dir, 10**8, 2, 0)
    resumed = pipeline.execute_nested_pipeline(replace(interrupted_config, resume_checkpoint=store.latest()))
    assert control["verdict"] == resumed["verdict"] == "PIPELINE_GREEN"
    assert resumed["hierarchy"]["observed_own_steps"] == {"d01": 200, "d02": 600}
    assert all(d["wrfout_count"] == 3 for d in resumed["per_domain"].values())
    for name in names:
        exact(expected[name], final_carries[-1][name])
    assert "resume_to_first_segment_ready_s" in resumed["restart"]


@pytest.mark.parametrize("crash", ["one", "both", "save", "replay", "intent", "async", "relative", "foreign", "tampered", "receipt"])
def test_actual_writer_history_crash_recovery(initialized, tmp_path, monkeypatch, crash):
    """Real NetCDF writer/publication and driver; sentinel stepping, no physics claim."""
    from dataclasses import replace
    from datetime import datetime, timedelta
    from netCDF4 import Dataset
    import gpuwrf.integration.nested_pipeline as pipeline
    import gpuwrf.io.gen2_accessor as accessor
    import gpuwrf.runtime.restart_store as storage
    import gpuwrf.io.wrfout_writer as netcdf
    import gpuwrf.diagnostics.census as census
    from gpuwrf.runtime.domain_tree import DomainTreeResult
    grid, namelist, seed = initialized
    names = ("d01", "d02")
    bundles = {n: SimpleNamespace(state=seed.state, grid=grid, namelist=namelist) for n in names}
    tree = SimpleNamespace(persistent_state_bytes=lambda: {n: seed.state.bytes() for n in names})
    run_start = datetime(2026, 7, 25)
    monkeypatch.setattr(pipeline.DomainTree, "from_domains", lambda *a, **k: tree)
    monkeypatch.setattr(pipeline, "_load_domains", lambda *a, **k: (SimpleNamespace(order=names, nests=()), bundles, {}, run_start, {"d01": 54., "d02": 18.}, {n: seed for n in names}))
    monkeypatch.setattr(accessor, "Gen2Run", lambda _: None)
    monkeypatch.setattr(pipeline, "_output_cadence_steps_by_domain", lambda *a: ({"d01": 67, "d02": 200}, dict.fromkeys(names, 60)))
    alarms = {"d01": (67, 134, 200), "d02": (200, 400, 600)}
    monkeypatch.setattr(pipeline, "_output_alarm_steps_by_domain", lambda *a: (alarms, alarms))
    monkeypatch.setattr(pipeline, "maybe_prewarm_defused_nest", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_prepare_operational_domain_tree_runtime", lambda *a, **k: None)
    monkeypatch.setattr(storage, "nested_identity", lambda config, *a: {"test": "actual-writer", "output_dir": str(config.output_dir.resolve())})
    monkeypatch.setattr(census, "write_segment", lambda *a: None)
    for flag in ("GPUWRF_NESTED_ASYNC_OUTPUT", "GPUWRF_NEST_OUTPUT_PIPELINE", "GPUWRF_NEST_PERF_TIMERS"):
        monkeypatch.setenv(flag, "0")
    authorities = {}
    for n in names:
        source = accessor.Gen2GridSpec(id=n, dx_m=3000., dy_m=3000., e_we=grid.nx+1, e_sn=grid.ny+1, e_vert=grid.nz+1,
            mass_nx=grid.nx, mass_ny=grid.ny, mass_nz=grid.nz, grid_proj="lambert", map_proj_id=1,
            cen_lat=28.3, cen_lon=-16.1, truelat1=25., truelat2=30., stand_lon=-16.4,
            parent_id=1, parent_grid_ratio=1 if n=="d01" else 3, i_parent_start=1, j_parent_start=1,
            znu=tuple(range(grid.nz)), znw=tuple(range(grid.nz+1)), top_pressure_pa=5000.,
            source_wrfout="authenticated-test", source_namelist="authenticated-test")
        authorities[n] = netcdf.bind_wrfout_domain_authority(n, source, grid)
    ledgers = []
    class Writer(pipeline._PerDomainWrfoutWriter):
        # Stub case setup/diagnostics only; retain actual callback, materialize/
        # submit, restart-payload attachment and synchronous/asynchronous writer.
        def __init__(self, *, output_dir, **kwargs):
            self.output_dir = output_dir
            self.run_start = kwargs['run_start']
            self.bundles = kwargs['bundles']
            self.dt_by_domain = kwargs['dt_by_domain']
            self.written = {n: [] for n in names}
            self.writer_static_latlon_metadata = {}
            self.writer_diagnostics = {}
            self.domain_authorities = authorities
            self.census_io_ledger = census.WriterIoLedger()
            self._census_persist_info = {}
            self._perf_timers = pipeline._NestedOutputPerfTimers.disabled()
            self._variable_subset = None
            self._full_variable_set = False
            self._async_writer = kwargs.get('async_writer')
            self._output_pipeline = None
            self._surface_diagnostics_for_output = lambda *a, **k: {}
            self._merge_output_diagnostics = lambda a,b: b
            if self._async_writer is not None:
                self._async_writer._write_timing_callback = lambda path, seconds: self._record_census_persist(path)
            ledgers.append(self)
    monkeypatch.setattr(pipeline, '_wrfout_path', lambda folder, name, valid: folder/f"wrfout_{name}_{round((valid-run_start).total_seconds()/(54 if name=='d01' else 18))}")
    def prepare(state, grid, namelist, path, **kwargs):
        return netcdf.PreparedWrfout(path, netcdf._dimension_sizes(nx=grid.nx, ny=grid.ny, nz=grid.nz, namelist=namelist),
            {'T2': np.asarray(state.theta[0],dtype=np.float32)}, run_start, kwargs['valid_time'], kwargs['lead_hours'],
            grid, namelist, kwargs['domain'], kwargs['domain_authority'])
    monkeypatch.setattr(pipeline, 'prepare_wrfout_payload', prepare)
    if crash=='async':
        monkeypatch.setenv('GPUWRF_NESTED_ASYNC_OUTPUT','1')
    monkeypatch.setattr(pipeline, "_PerDomainWrfoutWriter", Writer)
    def advance(tree, *, root_steps, carries, initial_own_steps, output, **kwargs):
        own = {n: initial_own_steps[n]+root_steps*(1 if n=="d01" else 3) for n in names}
        new = {n: c.replace(state=c.state.replace(_cast=False, theta=c.state.theta+own[n]-initial_own_steps[n])) for n,c in carries.items()}
        for n in names:
            for step in alarms[n]:
                if initial_own_steps[n] < step <= own[n]:
                    output(n, step, new[n])
        return DomainTreeResult(new, {n:c.state for n,c in new.items()}, own, (), (), cascade_counts={})
    monkeypatch.setattr(pipeline, "run_operational_domain_tree", advance)
    config = pipeline.NestedPipelineConfig(tmp_path/"inputs", tmp_path/"control", tmp_path/"proof", 3, 2)
    if crash=='relative':
        monkeypatch.chdir(tmp_path)
        config = replace(config, output_dir=Path('relative/../control'))
    pipeline.execute_nested_pipeline(config)
    control_ledger = ledgers[-1].census_io_ledger.snapshot()
    config = replace(config, output_dir=Path('relative/../interrupted') if crash=='relative' else tmp_path/"interrupted", checkpoint_dir=tmp_path/"checkpoints", checkpoint_interval_steps=67, checkpoint_reserve_bytes=0)
    save, publish, link = storage.RestartStore.save, storage.HistoryJournal.publish, netcdf._publish_wrfout_noreplace
    rename = storage.os.rename
    def fail_checkpoint_publish(temporary, target):
        if Path(temporary).name.startswith('.partial-'):
            manifest = json.loads((Path(temporary)/'manifest.json').read_text())
            if manifest['own_steps']['d01']==134:
                raise OSError('injected checkpoint publication failure after fsync')
        return rename(temporary, target)
    def fail_save(self, carries, own_steps, *a, **k):
        if own_steps['d01']==134:
            raise OSError("injected checkpoint publication failure")
        return save(self, carries, own_steps, *a, **k)
    def fail_publish(self, temporary, target):
        publish(self, temporary, target)
        if Path(target).name == ("wrfout_d01_134" if crash=="one" else "wrfout_d02_400"):
            raise OSError("injected asymmetric publish crash")
    def fail_link(temporary, target):
        if Path(target).name=="wrfout_d01_134":
            raise OSError("injected crash after durable intent")
        return link(temporary, target)
    if crash in ("one", "both"):
        monkeypatch.setattr(storage.HistoryJournal, "publish", fail_publish)
    elif crash=="intent":
        monkeypatch.setattr(netcdf, "_publish_wrfout_noreplace", fail_link)
    elif crash=='save':
        monkeypatch.setattr(storage.os, 'rename', fail_checkpoint_publish)
    else:
        monkeypatch.setattr(storage.RestartStore, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        pipeline.execute_nested_pipeline(config)
    monkeypatch.setattr(storage.RestartStore, "save", save)
    monkeypatch.setattr(storage.HistoryJournal, "publish", publish)
    monkeypatch.setattr(netcdf, "_publish_wrfout_noreplace", link)
    monkeypatch.setattr(storage.os, 'rename', rename)
    store = storage.RestartStore(config.checkpoint_dir, 10**8, 2, 0)
    checkpoint = store.latest()
    assert store.read(checkpoint, device=False)[1]["own_steps"]=={"d01":67,"d02":201}
    committed = {p: (p.stat().st_ino, p.read_bytes()) for p in config.output_dir.glob("wrfout*") if p.name in ("wrfout_d01_67","wrfout_d02_200")}
    if crash in ("foreign", "tampered", "receipt"):
        target = config.output_dir/"wrfout_d02_400"
        if crash=="foreign":
            other = target.with_suffix('.foreign');other.write_bytes(target.read_bytes());os.replace(other, target)
        elif crash=="tampered":
            target.write_bytes(b'corrupt later output')
        else:
            entry = next(p for p in (config.checkpoint_dir).glob('history-*/*.json') if p.name!='owner.json')
            record = json.loads(entry.read_text());record['signature']='0'*64;entry.write_text(json.dumps(record))
        before = {p:p.read_bytes() for p in config.output_dir.glob('wrfout*')}
        with pytest.raises(ValueError, match="foreign or corrupt|authentication"):
            pipeline.execute_nested_pipeline(replace(config, resume_checkpoint=checkpoint))
        assert before=={p:p.read_bytes() for p in config.output_dir.glob('wrfout*')}
        return
    if crash=="replay":
        def replay_crash(self, temporary, target):
            publish(self, temporary, target)
            if Path(target).name=="wrfout_d01_134":
                raise OSError("injected second crash during replay")
        monkeypatch.setattr(storage.HistoryJournal, 'publish', replay_crash)
        with pytest.raises(OSError, match="second crash"):
            pipeline.execute_nested_pipeline(replace(config, resume_checkpoint=checkpoint))
        assert store.latest()==checkpoint
        monkeypatch.setattr(storage.HistoryJournal, 'publish', publish)
    result = pipeline.execute_nested_pipeline(replace(config, resume_checkpoint=checkpoint))
    assert result['verdict']=='PIPELINE_GREEN'
    assert result['hierarchy']['observed_own_steps']=={'d01':200,'d02':600}
    assert ledgers[-1].census_io_ledger.snapshot()==control_ledger
    assert {n:[Path(p).name for p in paths] for n,paths in ledgers[-1].written.items()}=={n:[f'wrfout_{n}_{s}' for s in alarms[n]] for n in names}
    for path, (inode,data) in committed.items():
        assert (path.stat().st_ino,path.read_bytes())==(inode,data)
    for path in (tmp_path/'control').glob('wrfout*'):
        with Dataset(path) as a, Dataset(config.output_dir/path.name) as b:
            a.set_auto_mask(False);b.set_auto_mask(False)
            assert a.variables.keys()==b.variables.keys()
            for key in a.variables:
                x,y=np.asarray(a.variables[key][...]),np.asarray(b.variables[key][...])
                assert (x.dtype,x.shape,x.tobytes())==(y.dtype,y.shape,y.tobytes())
