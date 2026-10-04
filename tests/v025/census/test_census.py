"""Fault injection and transparency gates (regression, not WRF fidelity)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.diagnostics import census as cs
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry


def fixture_state():
    path = Path(__file__).resolve().parents[2] / "test_v013_operational_smoke.py"
    spec = importlib.util.spec_from_file_location("census_smoke_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grid = module._grid(nz=4)
    return grid, module._base_state(grid)


def assert_fields_exact(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()


def test_injected_real_guard_path_is_bit_identical_and_counts_every_site(monkeypatch):
    grid, state = fixture_state()
    # Finite origins; all fields have room for independent NaN, Inf and range faults.
    origin = state.replace(mu_total=jnp.full_like(state.mu_total, 1000),
                           mu_perturbation=jnp.zeros_like(state.mu_perturbation))
    faults = {name: getattr(origin, name).at[0, 0, 0].set(jnp.nan)
              for name in set(cs.MOISTURE_FIELDS + cs.BOUNDARY_FINITE_FIELDS)}
    faults["theta"] = origin.theta.at[0, 0, 0].set(jnp.nan).at[0, 0, 1].set(1001)
    for name in cs.MOISTURE_FIELDS:
        faults[name] = faults[name].at[0, 0, 1].set(-0.1).at[0, 0, 2].set(0.051)
    faults["mu_total"] = origin.mu_total.at[0, 0].set(jnp.nan).at[0, 1].set(0.5)
    faults["mu_perturbation"] = origin.mu_perturbation.at[0, 2].set(jnp.inf)
    raw = origin.replace(**faults)
    monkeypatch.setattr(om, "_physics_step_forcing", lambda carry, *a, **k:
                        om._PhysicsStepForcing(carry.state, carry, om.DryPhysicsTendencies(), False))
    monkeypatch.setattr(om, "_rk_scan_step", lambda carry, *a, **k: carry.replace(state=raw))
    monkeypatch.setattr(om, "apply_lateral_boundaries", lambda *a, **k: raw)
    monkeypatch.setattr(om, "_enforce_operational_precision", lambda state, **k: state)
    namelist = SimpleNamespace(dt_s=54, disable_guards=False, run_boundary=True, run_physics=False,
        boundary_config=SimpleNamespace(nested_frozen_wrf_boundary_bundle=False),
        metrics=grid.metrics, grid=grid, lower_boundary=None, force_fp64=True,
        acoustic_precision_mode="fp64_default")
    carry = initial_operational_carry(origin)
    run = jax.jit(lambda c: om._physics_boundary_step_with_limiter_diagnostics(
        c, namelist, jnp.asarray(1), run_radiation=False)[0])
    off = run(carry)
    on = run(carry.replace(census=cs.initial_census()))
    assert_fields_exact(off, on.replace(census=None))
    counts = np.asarray(on.census.guards)
    for name in cs.GUARDS:
        nf, outside, repaired = counts[cs.GUARDS.index(name)]
        assert nf == 1, name
        assert repaired >= 1, name
        if name in ("dynamics.mu_total", "boundary.mu_total"):
            assert outside == 1 and repaired == 3
        elif name.endswith("mu_perturbation"):
            assert outside == 0 and repaired == 3
        elif name == "dynamics.theta":
            assert outside == 1
        elif name.split(".")[1] in cs.MOISTURE_FIELDS:
            assert outside == 2
        else:
            assert outside == 0
    assert int(on.census.work[cs.WORK.index("steps")]) == 1
    assert_fields_exact(off.state, run(on).state)
    np.testing.assert_array_equal(run(on).census.guards, counts * 2)


def test_disabled_has_no_leaves_or_device_work_and_uint64_does_not_wrap():
    assert jax.tree.leaves(cs.count_guard(None, "dynamics.theta", jnp.array([jnp.nan]))) == []
    assert not jax.make_jaxpr(lambda x: cs.count_guard(None, "dynamics.theta", x))(jnp.ones(3)).jaxpr.eqns
    value = cs.initial_census()._replace(work=jnp.full((len(cs.WORK),), 2**32, jnp.uint64))
    assert int(jax.jit(lambda c: cs.count_work(c, "steps"))(value).work[0]) == 2**32 + 1


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_runtime_fault_operands_count_on_device_without_uint32_wrap(dtype):
    candidate = jnp.asarray([jnp.nan, jnp.inf, -jnp.inf, -0.1, 0.051, 0, 0.05], dtype=dtype)
    counters = cs.initial_census()._replace(
        guards=jnp.full((len(cs.GUARDS), len(cs.EVENTS)), 2**32, dtype=jnp.uint64))
    run = jax.jit(lambda c, x: cs.count_guard(c, "dynamics.qv", x, lower=0, upper=0.05))
    actual = run(counters, candidate)
    expected = np.asarray([2**32 + 3, 2**32 + 2, 2**32 + 5], dtype=np.uint64)
    np.testing.assert_array_equal(actual.guards[cs.GUARDS.index("dynamics.qv")], expected)


def test_native_mass_sink_is_resident_cumulative_and_free_when_disabled():
    events = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32)
    index = cs.WORK.index("native_mass_guard_events")
    counters = cs.initial_census()._replace(work=cs.initial_census().work.at[index].set(2**32))
    run = jax.jit(cs.count_native_mass_guard_events)
    first = run(counters, events)
    second = run(first, events)
    assert int(first.work[index]) == 2**32 + 2
    assert int(second.work[index]) == 2**32 + 4
    np.testing.assert_array_equal(second.guards, counters.guards)
    assert run(None, events) is None
    assert not jax.make_jaxpr(lambda x: cs.count_native_mass_guard_events(None, x))(events).jaxpr.eqns


@pytest.mark.parametrize("name", ["radiation_init_calls", "radiation_output_calls", "output_writes"])
def test_initialization_output_counter_names_accumulate_without_transfers(name):
    run = jax.jit(lambda c: cs.count_work(c, name))
    first = run(cs.initial_census())
    second = run(first)
    assert int(second.work[cs.WORK.index(name)]) == 2
    assert run(None) is None


def test_writer_io_ledger_folds_persistence_without_mutating_carry(tmp_path):
    ledger = cs.WriterIoLedger()
    assert ledger.snapshot() == {}
    ledger.record_persist("d01", radiation_output_calls=1)
    first = ledger.snapshot()
    ledger.record_persist("d01", radiation_output_calls=0)
    ledger.record_persist("d02", radiation_output_calls=2)
    assert first["d01"] == {"radiation_output_calls": 1, "output_writes": 1}
    assert ledger.snapshot()["d01"] == {"radiation_output_calls": 1, "output_writes": 2}
    with pytest.raises(ValueError):
        ledger.record_persist("d01", radiation_output_calls=-1)
    namelist = SimpleNamespace(dt_s=54, acoustic_substeps=10, radiation_cadence_steps=33)
    counters = cs.count_work(cs.initial_census(), "steps", 67)
    writer = SimpleNamespace(written={"d01": ["a", "b", "pending"]}, census_io_ledger=ledger)
    cs.write_segment(tmp_path, {"d01": SimpleNamespace(census=counters)},
                     {"d01": SimpleNamespace(namelist=namelist)}, {"d01": 67}, writer, {"d01": 67})
    record = json.loads((tmp_path / "census.json").read_text())["domains"]["d01"]
    assert record["actual_work"]["radiation_output_calls"] == 1
    assert record["actual_work"]["output_writes"] == 2
    assert record["output_submissions"] == 3
    assert int(counters.work[cs.WORK.index("output_writes")]) == 0


def test_real_acoustic_loop_counts_executed_trips_and_preserves_fields():
    grid, state = fixture_state()
    shapes = ((grid.nz, grid.ny, grid.nx + 1), (grid.nz, grid.ny + 1, grid.nx),
              (grid.nz + 1, grid.ny, grid.nx), (grid.nz, grid.ny, grid.nx))
    z = lambda shape: jnp.zeros(shape, jnp.float64)
    tendencies = om.Tendencies(z(shapes[0]), z(shapes[1]), z(shapes[2]), z(shapes[3]),
                              z(shapes[3]), z(shapes[3]), z(shapes[2]), z((grid.ny, grid.nx)))
    namelist = om.OperationalNamelist(grid=grid, tendencies=tendencies, metrics=grid.metrics,
        dt_s=0.01, acoustic_substeps=4, run_physics=False, run_boundary=False, force_fp64=True)
    carry = om._initial_carry_for_run(state, namelist)
    run = lambda c: om._advance_chunk(c, namelist, 1, n_steps=2, cadence=60)
    off = run(carry)
    on = run(carry.replace(census=cs.initial_census()))
    assert_fields_exact(off, on.replace(census=None))
    assert int(on.census.work[cs.WORK.index("steps")]) == 2
    assert int(on.census.work[cs.WORK.index("acoustic_trips")]) == 14


@pytest.mark.parametrize("predicate", [False, True])
def test_radiation_counts_are_inside_executed_branch(monkeypatch, predicate):
    grid, state = fixture_state()
    monkeypatch.setattr(om, "rrtmg_radiation_diagnostics", lambda *a, **k:
        SimpleNamespace(swnorm=jnp.ones((grid.ny, grid.nx)), glw=jnp.ones((grid.ny, grid.nx)),
                        coszen=jnp.ones((grid.ny, grid.nx))))
    namelist = SimpleNamespace(dt_s=54, radiation_cadence_steps=33, grid=grid,
        time_utc="2026-07-25T18:00:00Z", radiation_static=None, topo_shading=0,
        slope_rad=0, topo_shadow_length_m=0, ra_sw_physics=4, ra_lw_physics=4)
    held = tuple(jnp.zeros((grid.ny, grid.nx)) for _ in range(3))
    run = jax.jit(lambda c, p: om._refresh_noahmp_rad(state, namelist, jnp.array(54.), p,
                                                     held, census=c))
    on, counts = run(cs.initial_census(), jnp.asarray(predicate))
    off = run(None, jnp.asarray(predicate))
    assert_fields_exact(on, off)
    assert int(counts.work[cs.WORK.index("radiation_surface_calls")]) == int(predicate)


def test_manifest_matches_runtime_values_and_device_counts(tmp_path):
    namelist = SimpleNamespace(dt_s=54, acoustic_substeps=10, rk_order=3,
        radiation_cadence_steps=33, cumulus_cadence_steps=6)
    counters = cs.count_work(cs.initial_census(), "steps", 67)
    counters = cs.count_work(counters, "acoustic_trips", 1072)
    carries = {"d01": SimpleNamespace(census=counters)}
    bundles = {"d01": SimpleNamespace(namelist=namelist)}
    writer = SimpleNamespace(written={"d01": ["out1"]}, _variable_subset=("T2", "U10"))
    cs.write_segment(tmp_path, carries, bundles, {"d01": 67}, writer, {"d01": 67})
    payload = json.loads((tmp_path / "census.json").read_text())
    import jsonschema
    schema = json.loads((Path(cs.__file__).with_name("census_schema.json")).read_text())
    jsonschema.validate(payload, schema)
    record = payload["domains"]["d01"]
    assert record["resolved_namelist"]["stepcu"] == 6
    assert record["resolved_namelist"]["radt_minutes"] == 29.7
    assert record["resolved_namelist"]["output_set"] == ["T2", "U10"]
    assert record["actual_work"]["acoustic_trips"] == 1072
    assert record["output_submissions"] == 1
    cs.write_segment(tmp_path / "disabled", {"d01": SimpleNamespace(census=None)}, {}, {}, writer, {})
    assert not (tmp_path / "disabled").exists()
    with pytest.raises(ValueError, match="missing from one or more domains"):
        cs.write_segment(tmp_path, {**carries, "d02": SimpleNamespace(census=None)},
                         bundles, {"d01": 67, "d02": 201}, writer, {"d01": 67})


def test_bench_attachment_preserves_measured_counts_and_rejects_lost_carry(tmp_path):
    from gpuwrf.diagnostics.census_report import attach_bench
    run = tmp_path / "arm"
    (run / "wrfout").mkdir(parents=True)
    namelist = SimpleNamespace(dt_s=54, acoustic_substeps=10, rk_order=3,
        radiation_cadence_steps=33, cumulus_cadence_steps=6)
    carries = {"d01": SimpleNamespace(census=cs.count_work(cs.initial_census(), "steps", 200))}
    writer = SimpleNamespace(written={"d01": ["a", "b", "c"]})
    cs.write_segment(run / "wrfout", carries, {"d01": SimpleNamespace(namelist=namelist)},
                     {"d01": 200}, writer, {"d01": 67})
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"arms": {"warm1": {"receipt": str(run / "receipt.json"),
                           "env": {"GPUWRF_CENSUS": "1"}}}}))
    manifests = attach_bench(bench)
    assert manifests["warm1"] == str(tmp_path / "warm1.census.json")
    payload = json.loads((run / "wrfout" / "census.json").read_text())
    assert payload == json.loads((tmp_path / "warm1.census.json").read_text())
    payload["domains"]["d01"]["actual_work"]["steps"] = 199
    (run / "wrfout" / "census.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="device step count differs"):
        attach_bench(bench)
    (run / "wrfout" / "census.json").unlink()
    with pytest.raises(ValueError, match="no manifest"):
        attach_bench(bench)


@pytest.mark.parametrize("strict", [True, False])
def test_attach_bench_puts_f2_counts_into_the_receipt(tmp_path, strict):
    """Release receipt: strict guard rows are would-repair candidates, legacy rows are applied repairs."""
    from gpuwrf.diagnostics.census_report import attach_bench
    run = tmp_path / "arm"
    (run / "wrfout").mkdir(parents=True)
    namelist = SimpleNamespace(dt_s=54, acoustic_substeps=10, rk_order=3, radiation_cadence_steps=33,
                               cumulus_cadence_steps=6, disable_guards=strict)
    census = cs.count_work(cs.initial_census(), "steps", 200)
    guards = np.array(census.guards)
    guards[cs.GUARDS.index("dynamics.qc")] = [0, 7, 7]
    guards[cs.GUARDS.index("boundary.qv")] = [2, 0, 0]
    cs.write_segment(run / "wrfout", {"d01": SimpleNamespace(census=census._replace(guards=jnp.asarray(guards)))},
                     {"d01": SimpleNamespace(namelist=namelist)}, {"d01": 200},
                     SimpleNamespace(written={"d01": ["a", "b", "c"]}), {"d01": 67})
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"arms": {"warm1": {"receipt": str(run / "receipt.json"),
                                                    "env": {"GPUWRF_CENSUS": "1"}}}}))
    attach_bench(bench)
    receipt = json.loads(bench.read_text())
    summary = receipt["census_summary"]["warm1"]["d01"]
    assert receipt["arms"]["warm1"]["census_summary"]["d01"] == summary
    assert summary["strict_guards"] is strict and summary["nonfinite"] == 2 and summary["out_of_range"] == 7
    assert (summary["applied_repairs"], summary["would_repair_candidates"]) == ((0, 7) if strict else (7, 0))


@pytest.mark.parametrize("mutation", [
    "all_guards_deleted", "one_guard_deleted", "unknown_guard", "null_acoustic",
    "zero_acoustic", "null_stepcu", "zero_stepcu", "negative_radt", "text_radt",
    "null_output_cadence", "zero_output_cadence", "nonstring_output_set", "negative_steps",
])
def test_attach_bench_rejects_incomplete_or_invalid_manifest(tmp_path, mutation):
    import jsonschema
    from gpuwrf.diagnostics.census_report import attach_bench

    run = tmp_path / "run"
    (run / "wrfout").mkdir(parents=True)
    namelist = SimpleNamespace(dt_s=54, acoustic_substeps=10, rk_order=3,
        radiation_cadence_steps=33, cumulus_cadence_steps=6)
    cs.write_segment(run / "wrfout",
        {"d01": SimpleNamespace(census=cs.count_work(cs.initial_census(), "steps", 200))},
        {"d01": SimpleNamespace(namelist=namelist)}, {"d01": 200},
        SimpleNamespace(written={"d01": ["a", "b", "c"]}), {"d01": 67})
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"arms": {"warm1": {"receipt": str(run / "receipt.json"),
                        "env": {"GPUWRF_CENSUS": "1"}}}}))
    # Exercise the real attachment path: valid writer output passes first.
    attach_bench(bench)
    manifest = run / "wrfout" / "census.json"
    payload = json.loads(manifest.read_text())
    record = payload["domains"]["d01"]
    if mutation == "all_guards_deleted":
        record["guards"] = {}
    elif mutation == "one_guard_deleted":
        del record["guards"]["boundary.mu_total"]
    elif mutation == "unknown_guard":
        record["guards"]["silent.guard"] = dict(record["guards"]["boundary.mu_total"])
    elif mutation == "negative_steps":
        record["actual_work"]["steps"] = -1
    else:
        field, value = {
            "null_acoustic": ("acoustic_substeps", None),
            "zero_acoustic": ("acoustic_substeps", 0),
            "null_stepcu": ("stepcu", None),
            "zero_stepcu": ("stepcu", 0),
            "negative_radt": ("radt_minutes", -1),
            "text_radt": ("radt_minutes", "30"),
            "null_output_cadence": ("output_cadence_steps", None),
            "zero_output_cadence": ("output_cadence_steps", 0),
            "nonstring_output_set": ("output_set", [1]),
        }[mutation]
        record["resolved_namelist"][field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(jsonschema.ValidationError):
        attach_bench(bench)
    from gpuwrf.diagnostics.census_report import main
    with pytest.raises(jsonschema.ValidationError):
        main(["--attach-bench", str(bench)])
    assert json.loads(bench.read_text())["ok"] is False
