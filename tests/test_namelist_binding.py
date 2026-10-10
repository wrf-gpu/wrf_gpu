"""v0.3.4 P0 (o1-nlbind): namelist.input physics/dynamics options reach the run.

<= v0.3.3 both CLI pipelines built ``OperationalNamelist.from_grid`` and never read
mp/bl/sfclay/ra_lw/ra_sw (nor epssm/damping/6th-order knobs): a validated selection
silently ran Thompson/MYNN/RRTMG.  These gates run the REAL loaders on the real WN3
namelist fixture with per-domain edits; only the heavy case builder / device
initializers are substituted (pattern of test_host_sync_b36_loader_seam).  Removing
the ``bind_namelist_options`` call in either pipeline fails them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf import cli
from gpuwrf.integration import daily_pipeline
from gpuwrf.integration import nested_pipeline
from gpuwrf.io import namelist_binding as nb
from gpuwrf.io.gen2_accessor import Gen2Run, parse_namelist

FIXTURE = Path(__file__).parent / "fixtures/wn3_20260227_cadence.namelist"
REPO = Path(__file__).resolve().parents[1]

# Per-domain edits (WRF-legal: module_check_a_mundo equal / equal-or-zero rules); every bound
# key gets a non-release value.
EDITS = {
    "mp_physics": "6, 6, 6",
    "bl_pbl_physics": "1, 0, 1",
    "sf_sfclay_physics": "1, 1, 1",
    "ra_lw_physics": "1, 1, 1",
    "ra_sw_physics": "2, 2, 2",
    "cu_physics": "3, 3, 0",
    "epssm": "0.5, 0.3, 0.2",
    "zdamp": "5000., 6000., 7000.",
    "dampcoef": "0.2, 0.1, 0.05",
    "diff_6th_opt": "2, 0, 2",
    "diff_6th_factor": "0.12, 0.12, 0.25",
    "damp_opt": "0",
    "w_damping": "0",
}
_COMMON = dict(mp_physics=6, sf_sfclay_physics=1, ra_lw_physics=1, ra_sw_physics=2, damp_opt=0, w_damping=0)
EXPECT = {
    "d01": dict(_COMMON, bl_pbl_physics=1, cu_physics=3, epssm=0.5, zdamp=5000.0, dampcoef=0.2,
                diff_6th_opt=2, diff_6th_factor=0.12),
    "d02": dict(_COMMON, bl_pbl_physics=0, cu_physics=3, epssm=0.3, zdamp=6000.0, dampcoef=0.1,
                diff_6th_opt=0, diff_6th_factor=0.12),
    "d03": dict(_COMMON, bl_pbl_physics=1, cu_physics=0, epssm=0.2, zdamp=7000.0, dampcoef=0.05,
                diff_6th_opt=2, diff_6th_factor=0.25),
}


def _edited(text: str, edits: dict[str, str]) -> str:
    for key, value in edits.items():
        pattern = re.compile(rf"^\s*{key}\s*=.*$", re.M)
        assert pattern.search(text), key
        text = pattern.sub(f" {key} = {value},", text)
    return text


def _write_case(tmp_path: Path, edits: dict[str, str], extra_physics: str = "") -> Path:
    case = tmp_path / "case"
    case.mkdir()
    text = _edited(FIXTURE.read_text(), edits)
    if extra_physics:
        text = re.sub(r"(&physics\s*\n)", lambda m: m.group(1) + extra_physics, text, count=1)
    (case / "namelist.input").write_text(text)
    return case


def _load_nested(monkeypatch, tmp_path, edits, extra_physics=""):
    from gpuwrf.validation.moving_nest_testbed import (
        _zero_tendencies,
        build_flat_grid,
        build_neutral_state,
    )

    case = _write_case(tmp_path, edits, extra_physics)
    run = Gen2Run(case)
    run.grid = lambda name: SimpleNamespace(**{  # no wrfinput: domain geometry from the namelist
        key: nested_pipeline._domain_int(run, "domains", key, name, 1)
        for key in ("parent_id", "parent_grid_ratio", "i_parent_start", "j_parent_start")})
    grid = build_flat_grid(nx=9, ny=8, nz=3, dx_m=3000.0)

    def fake_case(_source, *, domain, **_kwargs):
        return SimpleNamespace(run=run, grid=grid, metrics=grid.metrics, tendencies=_zero_tendencies(grid),
                               state=build_neutral_state(grid),
                               metadata={"run_start_label": "2026-02-27_18:00:00"})

    class _Carry(SimpleNamespace):
        def replace(self, **updates):
            return _Carry(**{**vars(self), **updates})

    monkeypatch.setattr(nested_pipeline, "build_replay_case", fake_case)
    monkeypatch.setattr(nested_pipeline, "load_radiation_static", lambda *_a, **_k: (None, {}))
    monkeypatch.setattr(nested_pipeline, "_domain_gwd_opt", lambda *_a, **_k: 0)
    real_physics_int = nested_pipeline._domain_physics_int
    monkeypatch.setattr(
        nested_pipeline, "_domain_physics_int",
        lambda run_, key, *rest, **kw: 0 if key in ("slope_rad", "topo_shading")
        else real_physics_int(run_, key, *rest, **kw),
    )
    monkeypatch.setattr(nested_pipeline, "_domain_sf_surface_physics", lambda *_a, **_k: 0)
    monkeypatch.setattr(nested_pipeline, "_initial_carry_for_run",
                        lambda state, *_a, **_k: _Carry(state=state, noahmp_rad=None, cumulus_carry=None,
                                                        cumulus_tendencies=None, radiation_diagnostics=None))
    monkeypatch.setattr(nested_pipeline, "_commit_to_operational_device", lambda carry, *_a, **_k: carry)
    monkeypatch.setattr(nested_pipeline, "_root_boundary_cadence_override", lambda nml, *_a, **_k: nml)
    import gpuwrf.io.lower_boundary as lower_boundary

    monkeypatch.setattr(lower_boundary, "load_lower_boundary", lambda *_a, **_k: None)
    config = nested_pipeline.NestedPipelineConfig(
        input_dir=case, output_dir=tmp_path / "out", proof_dir=tmp_path / "proof", hours=1, max_dom=3)
    _h, bundles, *_rest = nested_pipeline._load_domains(config, ("d01", "d02", "d03"))
    return bundles


def test_nested_loader_binds_every_scheme_and_dynamics_option_per_domain(monkeypatch, tmp_path):
    bundles = _load_nested(monkeypatch, tmp_path, EDITS)
    for domain, expected in EXPECT.items():
        nl = bundles[domain].namelist
        got = {key: getattr(nl, key) for key in expected}
        assert got == expected, domain


def test_nested_loader_release_namelist_keeps_pipeline_values(monkeypatch, tmp_path):
    bundles = _load_nested(monkeypatch, tmp_path, {})
    for domain in ("d01", "d02", "d03"):
        nl = bundles[domain].namelist
        assert (nl.mp_physics, nl.bl_pbl_physics, nl.sf_sfclay_physics, nl.ra_lw_physics,
                nl.ra_sw_physics) == (8, 5, 5, 4, 4)
        assert (nl.epssm, nl.damp_opt, nl.zdamp, nl.dampcoef, nl.w_damping, nl.diff_6th_opt,
                nl.diff_6th_factor) == (0.5, 3, 5000.0, 0.2, 1, 2, 0.12)


def test_bind_returns_the_same_object_when_nothing_changes(tmp_path):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _NL:
        mp_physics: int = 8
        bl_pbl_physics: int = 5
        sf_sfclay_physics: int = 5
        ra_lw_physics: int = 4
        ra_sw_physics: int = 4
        cu_physics: int = 1
        sf_urban_physics: int = 0
        sf_lake_physics: int = 0
        epssm: float = 0.5
        damp_opt: int = 3
        zdamp: float = 5000.0
        dampcoef: float = 0.2
        w_damping: int = 1
        diff_6th_opt: int = 2
        diff_6th_factor: float = 0.12
        khdif: float = 0.0
        kvdif: float = 0.0
        c_s: float = 0.25
        c_k: float = 0.15
        mix_isotropic: int = 0
        mix_upper_bound: float = 0.1
        tke_upper_bound: float = 1000.0
        cam_abs_freq_s: float = 21600.0
        topo_shadow_length_m: float = 25000.0

    wrf = parse_namelist(FIXTURE)
    base = _NL()
    assert nb.bind_namelist_options(base, wrf, "d01") is base
    bound = nb.bind_namelist_options(base, {"physics": {"bl_pbl_physics": [5, 1, 1]}}, "d02")
    assert bound.bl_pbl_physics == 1 and bound.mp_physics == 8


def test_physics_suite_fills_omitted_scheme_keys_like_wrf():
    wrf = {"physics": {"physics_suite": "'CONUS'", "mp_physics": [-1, 28]}}
    assert nb.bound_options(wrf, "d01")["mp_physics"] == 8
    assert nb.bound_options(wrf, "d02")["mp_physics"] == 28
    assert nb.bound_options(wrf, "d02")["bl_pbl_physics"] == 2
    tropical = {"physics": {"physics_suite": "tropical"}}
    assert nb.bound_options(tropical, "d01")["sf_sfclay_physics"] == 91


def test_single_entry_registry_keys_use_the_first_value_on_every_domain():
    # WRF Registry: w_damping/damp_opt are single-entry (one value for all domains);
    # epssm is max_domains (per domain).
    wrf = {"dynamics": {"w_damping": [1, 0], "damp_opt": [3, 0], "epssm": [0.5, 0.2]}}
    assert nb.bound_options(wrf, "d02")["w_damping"] == 1
    assert nb.bound_options(wrf, "d02")["damp_opt"] == 3
    assert nb.bound_options(wrf, "d02")["epssm"] == 0.2


def _build_daily_case(monkeypatch, tmp_path, physics: str, dynamics: str = ""):
    from test_v024_c2_daily_binding import _cpu_tendencies, _MinimalState

    from gpuwrf.contracts.grid import GridSpec

    (tmp_path / "namelist.input").write_text(
        "&domains\n max_dom = 1,\n/\n&physics\n" + physics + "/\n&dynamics\n" + dynamics + "/\n")
    run = Gen2Run(tmp_path)
    run.grid = lambda _domain: SimpleNamespace()
    grid = GridSpec.canary_3km_template()
    replay = SimpleNamespace(run=run, state=_MinimalState(), tendencies=_cpu_tendencies(grid), grid=grid,
                             metrics=grid.metrics,
                             metadata={"run_id": "nlbind", "run_start_label": "2026-07-24_00:00:00",
                                       "grid": {}, "boundary": {}, "standalone_native_init": True})
    monkeypatch.setattr(daily_pipeline, "build_replay_case", lambda _run_dir, domain: replay)
    monkeypatch.setattr(daily_pipeline, "dealias_state_buffers", lambda state: state)
    monkeypatch.setattr(daily_pipeline, "load_radiation_static", lambda *_a, **_k: (None, {"status": "test"}))
    monkeypatch.setattr(daily_pipeline, "_load_static_latlon_writer_diagnostics",
                        lambda *_a, **_k: (None, {"status": "test"}))
    monkeypatch.setattr(daily_pipeline, "bind_wrfout_domain_authority", lambda *_a, **_k: SimpleNamespace())
    case, _ = daily_pipeline._build_real_case(daily_pipeline.DailyPipelineConfig(run_id=str(tmp_path),
                                                                                 domain="d01"))
    return case.namelist


def test_daily_replay_driver_binds_schemes_and_cadences(monkeypatch, tmp_path):
    nl = _build_daily_case(
        monkeypatch, tmp_path,
        " mp_physics = 6,\n bl_pbl_physics = 1,\n sf_sfclay_physics = 1,\n ra_lw_physics = 1,\n"
        " ra_sw_physics = 2,\n cu_physics = 1,\n radt = 10,\n cudt = 5,\n",
        " epssm = 0.3,\n top_lid = .true.,\n")
    assert (nl.mp_physics, nl.bl_pbl_physics, nl.sf_sfclay_physics, nl.ra_lw_physics, nl.ra_sw_physics,
            nl.cu_physics) == (6, 1, 1, 1, 2, 1)
    assert nl.epssm == 0.3 and nl.top_lid is True
    # replay dt is the driver's fixed 10 s: radt 10 min -> 60 steps, cudt 5 min -> STEPCU 30
    assert (nl.radiation_cadence_steps, nl.cumulus_cadence_steps, nl.cudt_minutes) == (60, 30, 5.0)


def test_daily_replay_driver_without_keys_is_unchanged(monkeypatch, tmp_path):
    nl = _build_daily_case(monkeypatch, tmp_path, " isfflx = 1,\n")
    assert (nl.mp_physics, nl.bl_pbl_physics, nl.ra_sw_physics, nl.cu_physics) == (8, 5, 4, 0)
    assert (nl.radiation_cadence_steps, nl.cumulus_cadence_steps, nl.top_lid) == (180, 1, False)


@pytest.mark.parametrize("path", [
    REPO / "examples/switzerland_d01/namelist.input",
    FIXTURE,
    Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725/namelist.input"),
    Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/namelist.input"),
])
def test_release_namelists_are_fully_honoured(path):
    if not path.is_file():
        pytest.skip(f"{path} not mounted")
    from gpuwrf.io.namelist_check import _coerce_config

    nl = _coerce_config(path)
    max_dom = int(nl.get("domains", {}).get("max_dom", [1])[0])
    errors, warnings = nb.namelist_honour_report(nl, tuple(f"d{i:02d}" for i in range(1, max_dom + 1)))
    assert errors == [] and warnings == []


@pytest.mark.parametrize("line,group,needle", [
    (" bl_mynn_mixlength = 2,\n", "physics", "bl_mynn_mixlength"),  # catalog now refuses it too
    (" smdiv = 0.2,\n", "dynamics", "smdiv=0.2"),
    (" rk_ord = 2,\n", "dynamics", "rk_ord=2"),
    (" non_hydrostatic = .false.,\n", "dynamics", "non_hydrostatic=False"),
    (" dveg = 2,\n", "noah_mp", "dveg=2"),
    (" sf_surface_physics = 2,\n", "physics", "sf_surface_physics=[2]"),
    (" not_a_wrf_key = 1,\n", "physics", "not_a_wrf_key"),
])
def test_cli_refuses_unhonoured_values_before_jax(tmp_path, capsys, line, group, needle):
    text = FIXTURE.read_text()
    key = line.split("=")[0].strip()
    existing = re.compile(rf"^\s*{key}\s*=.*$", re.M)
    if existing.search(text):
        text = existing.sub(line.rstrip("\n"), text)
    else:
        if f"&{group}" not in text:
            text += f"&{group}\n/\n"
        text = re.sub(rf"(&{group}\s*\n)", lambda m: m.group(1) + line, text, count=1)
    (tmp_path / "namelist.input").write_text(text)
    rc = cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                   "--domains-from-namelist", "--hours", "3", "--dry-run"])
    err = capsys.readouterr().err
    assert rc != 0, err
    assert needle in err and ("would NOT honour" in err or "not bound" in err), err


@pytest.mark.parametrize("key,values", [("mp_physics", "8, 8, 0"), ("ra_sw_physics", "4, 1, 4"),
                                        ("bl_pbl_physics", "5, 1, 5")])
def test_cli_refuses_per_domain_selections_wrf_would_not_run(tmp_path, capsys, key, values):
    (tmp_path / "namelist.input").write_text(_edited(FIXTURE.read_text(), {key: values}))
    rc = cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                   "--domains-from-namelist", "--hours", "3", "--dry-run"])
    err = capsys.readouterr().err
    assert rc != 0 and key in err and "module_check_a_mundo" in err, err


def test_cli_legal_per_domain_edits_pass(tmp_path, capsys):
    (tmp_path / "namelist.input").write_text(_edited(FIXTURE.read_text(), EDITS))
    rc = cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                   "--domains-from-namelist", "--hours", "3", "--dry-run"])
    assert rc == 0, capsys.readouterr().err


def test_cli_release_fixture_dry_run_passes(tmp_path, capsys):
    (tmp_path / "namelist.input").write_text(FIXTURE.read_text())
    rc = cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                   "--domains-from-namelist", "--hours", "3", "--dry-run"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert json.loads(out.out)["effective_max_dom"] == 3


def test_registry_table_matches_pristine_wrf(tmp_path):
    reg = REPO / "data/wrf_pristine/WRF/Registry"
    if not (reg / "Registry.EM").is_file():
        pytest.skip("pristine WRF Registry not linked")
    import importlib.util

    spec = importlib.util.spec_from_file_location("gen", REPO / "proofs/o1_nlbind/gen_registry_defaults.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = tmp_path / "defaults.py"
    gen.main(str(reg), str(out))
    assert out.read_text() == (REPO / "src/gpuwrf/io/wrf_registry_defaults.py").read_text()


def test_pbl_sfclay_pairing_mirror_matches_the_dispatcher():
    from gpuwrf.coupling.physics_dispatch import _PBL_REQUIRES_REVISED_MM5_SFCLAY

    assert nb.PBL_REQUIRES_REVISED_MM5_SFCLAY == _PBL_REQUIRES_REVISED_MM5_SFCLAY
    errors, _ = nb.namelist_honour_report(
        {"physics": {"bl_pbl_physics": [7], "sf_sfclay_physics": [7]}}, ("d01",))
    assert any("revised-MM5" in e for e in errors)
    errors, _ = nb.namelist_honour_report(
        {"physics": {"bl_pbl_physics": [7], "sf_sfclay_physics": [1]}}, ("d01",))
    assert errors == []


def test_omitted_dynamics_knobs_run_the_wrf_registry_default():
    bound = nb.bound_options({"physics": {"mp_physics": [8]}}, "d01")
    assert (bound["epssm"], bound["w_damping"], bound["diff_6th_opt"]) == (0.1, 0, 0)
    assert (bound["damp_opt"], bound["zdamp"], bound["dampcoef"], bound["diff_6th_factor"]) == (3, 5000.0, 0.2, 0.12)
    assert "mp_physics" in bound and "bl_pbl_physics" not in bound  # omitted scheme keys stay unbound
    _, warnings = nb.namelist_honour_report({"physics": {"mp_physics": [8]}}, ("d01",))
    assert any("bl_pbl_physics" in w and "omitted" in w for w in warnings)


@pytest.mark.parametrize("diff,km,ok", [(1, 4, True), (0, 0, True), (0, 4, True), (2, 3, True), (2, 2, False),
                                        (2, 5, False), (2, 1, False), (1, 1, False), (2, 4, False), (1, 0, False)])
def test_diffusion_pairs_without_an_operational_path_are_refused(diff, km, ok):
    # v0.3.4 probe: diff_opt=2/km_opt=1/khdif=100 traced exactly the diff_opt=0 program
    errors, _ = nb.namelist_honour_report({"dynamics": {"diff_opt": [diff], "km_opt": [km]}}, ("d01",))
    assert (not any("km_opt" in e for e in errors)) is ok


def test_cam_radiation_and_cam_abs_freq_s_are_bound(monkeypatch, tmp_path):
    # o1-camrad: ra_lw/ra_sw=3 CAM + WRF cam_abs_freq_s (single-entry &physics, Registry 21600 s)
    assert nb.bound_options({"physics": {}}, "d01")["cam_abs_freq_s"] == 21600.0
    bundles = _load_nested(monkeypatch, tmp_path, {"ra_lw_physics": "3, 3, 3", "ra_sw_physics": "3, 3, 3"},
                           extra_physics=" cam_abs_freq_s = 10800.,\n")
    for domain in ("d01", "d02", "d03"):
        nl = bundles[domain].namelist
        assert (nl.ra_lw_physics, nl.ra_sw_physics, nl.cam_abs_freq_s) == (3, 3, 10800.0), domain
    errors, _ = nb.namelist_honour_report({"physics": {"cam_abs_freq_s": [10800]}}, ("d01",))
    assert errors == []



@pytest.mark.parametrize("spelling", ["rk_ord", "rk_order"])
@pytest.mark.parametrize("value,ok", [(3, True), (2, False)])
def test_rk_ord_and_port_spelling_rk_order_share_semantics(tmp_path, capsys, spelling, value, ok):
    # rv-nlbind F1: WRF's key is rk_ord (Registry description says "rk_order"); the
    # port's docs used rk_order. Both spellings: =3 runs (WRF RK3), =2 refused pre-JAX.
    text = re.sub(r"(&dynamics\s*\n)", lambda m: m.group(1) + f" {spelling} = {value},\n",
                  FIXTURE.read_text(), count=1)
    (tmp_path / "namelist.input").write_text(text)
    rc = cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                   "--domains-from-namelist", "--hours", "3", "--dry-run"])
    err = capsys.readouterr().err
    assert (rc == 0) is ok, err
    if not ok:
        assert spelling in err and ("RK3" in err or "RK2" in err), err
    errors, _ = nb.namelist_honour_report({"dynamics": {spelling: [value]}}, ("d01",))
    assert (errors == []) is ok
    assert ok or "RK3" in errors[0]



@pytest.mark.parametrize("km", [2, 5])
def test_km_opt_2_and_5_are_refused_pre_jax(tmp_path, capsys, km):
    # manager decision (rel034 18:08Z): unqualified v022 scaffolds, NaN under the release REAL carry
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

    support = classify_scheme("km_opt", km)
    assert support.status is SupportStatus.RECOGNIZED_FAIL_CLOSED and "NaN" in support.reason
    errors, _ = nb.namelist_honour_report({"dynamics": {"diff_opt": [2], "km_opt": [km]}}, ("d01",))
    assert any("NaN" in e and f"km_opt={km}" in e for e in errors)
    (tmp_path / "namelist.input").write_text(_edited(FIXTURE.read_text(), {"diff_opt": "2, 2, 2",
                                                                          "km_opt": f"{km}, {km}, {km}"}))
    rc = cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                   "--domains-from-namelist", "--hours", "3", "--dry-run"])
    err = capsys.readouterr().err
    assert rc != 0 and "km_opt" in err and "NaN" in err, err


def test_km_opt_3_and_release_diffusion_still_run(tmp_path, capsys):
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

    assert classify_scheme("km_opt", 3).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("km_opt", 4).status is SupportStatus.IMPLEMENTED
    (tmp_path / "namelist.input").write_text(_edited(FIXTURE.read_text(), {"diff_opt": "2, 2, 2",
                                                                          "km_opt": "3, 3, 3"}))
    rc = cli.main(["run", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                   "--domains-from-namelist", "--hours", "3", "--dry-run"])
    assert rc == 0, capsys.readouterr().err
