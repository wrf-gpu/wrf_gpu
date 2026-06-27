"""GPUWRF_WRF_ROOT portability for WRF table/source/oracle lookups."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from gpuwrf.config.paths import wrf_phys_path, wrf_root, wrf_run_path
from gpuwrf.io import lsm_static_extract


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"_gpuwrf_portability_{script_name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_wrf_tree(root: Path) -> None:
    (root / "phys" / "MYNN-EDMF" / "misc").mkdir(parents=True)
    (root / "phys" / "MYNN-EDMF").mkdir(parents=True, exist_ok=True)
    (root / "run").mkdir(parents=True)
    (root / "_build_gen2_dmpar" / "CMakeFiles" / "WRF_Core.dir" / "phys").mkdir(parents=True)

    (root / "phys" / "module_mp_thompson.F.pre").write_text("END MODULE module_mp_thompson\n", encoding="utf-8")
    (root / "phys" / "module_ra_rrtmg_sw.F").write_text("module sw\nend module sw\n", encoding="utf-8")
    (root / "phys" / "module_ra_rrtmg_lw.F").write_text("module lw\nend module lw\n", encoding="utf-8")
    (root / "phys" / "MYNN-EDMF" / "misc" / "module_bl_mynn.F90").write_text("module mynn\nend module mynn\n", encoding="utf-8")
    (root / "phys" / "MYNN-EDMF" / "module_bl_mynnedmf.F90").write_text("module edmf\nend module edmf\n", encoding="utf-8")

    (root / "run" / "LANDUSE.TBL").write_text(
        "\n".join(
            [
                "MODIFIED_IGBP_MODIS_NOAH",
                "1 'portable test'",
                "SUMMER",
                "1 0.10 0.20 0.93 12.5 4.0 0 0",
                "WINTER",
                "1 0.20 0.30 0.94 13.5 5.0 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "run" / "VEGPARM.TBL").write_text(
        "\n".join(
            [
                "MODIFIED_IGBP_MODIS_NOAH",
                "1 'portable test'",
                "1 0.50 2 111.0 30.0 36.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "run" / "RRTMG_SW_DATA").write_bytes(b"sw")
    (root / "run" / "RRTMG_LW_DATA").write_bytes(b"lw")


def test_wrf_root_env_relocates_tbl_and_source_lookups(monkeypatch, tmp_path: Path) -> None:
    relocated = tmp_path / "relocated" / "WRF"
    _write_minimal_wrf_tree(relocated)
    monkeypatch.setenv("GPUWRF_WRF_ROOT", str(relocated))
    monkeypatch.delenv("WRF_BUILD", raising=False)

    assert wrf_root() == relocated
    assert wrf_run_path("LANDUSE.TBL") == relocated / "run" / "LANDUSE.TBL"
    assert wrf_phys_path("module_ra_rrtmg_lw.F") == relocated / "phys" / "module_ra_rrtmg_lw.F"

    landuse = lsm_static_extract._parse_landuse_table(wrf_root())
    vegparm = lsm_static_extract._parse_vegparm_rsmin(wrf_root())
    assert landuse is not None
    assert vegparm is not None
    assert landuse[1][0] == (0.93, 12.5, 4.0)
    assert vegparm[1] == 111.0

    thompson_extract = _load_script_module("extract_thompson_tables")
    assert thompson_extract.WRF_ROOT == relocated
    assert thompson_extract._wrf_source() == relocated / "phys" / "module_mp_thompson.F.pre"

    thompson_fixture = _load_script_module("m5_generate_thompson_fixture")
    assert thompson_fixture.WRF_SOURCE == relocated / "phys" / "module_mp_thompson.F.pre"

    mynn_fixture = _load_script_module("m5_generate_mynn_fixture")
    assert mynn_fixture.WRF_SOURCE == relocated / "phys" / "MYNN-EDMF" / "misc" / "module_bl_mynn.F90"
    assert mynn_fixture.WRF_EDMF_OBJECT == relocated / "phys" / "module_bl_mynnedmf.o"

    rrtmg_extract = _load_script_module("extract_rrtmg_tables")
    assert rrtmg_extract.WRF_ROOT == relocated
    assert rrtmg_extract.SW_SOURCE == relocated / "phys" / "module_ra_rrtmg_sw.F"
    assert relocated / "run" / "RRTMG_LW_DATA" in rrtmg_extract.LW_DATA_CANDIDATES

    rrtmg_fixture = _load_script_module("m5_generate_rrtmg_fixture")
    assert rrtmg_fixture.WRF_ROOT == relocated
    assert rrtmg_fixture.WRF_SW_OBJECT == (
        relocated / "_build_gen2_dmpar" / "CMakeFiles" / "WRF_Core.dir" / "phys" / "module_ra_rrtmg_sw.F.o"
    )


def test_harness_shell_scripts_name_gpuwrf_wrf_root() -> None:
    for rel in (
        "scripts/wrf_thompson_harness_build.sh",
        "scripts/wrf_mynn_harness_build.sh",
        "scripts/wrf_sfclay_harness_build.sh",
        "scripts/wrf_rrtmg_harness_build.sh",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "GPUWRF_WRF_ROOT" in text
        for fixed_prefix in ("/" + "mnt" + "/", "/" + "home" + "/"):
            assert fixed_prefix not in text


def test_noahmp_oracle_helpers_use_configured_wrf_root() -> None:
    for rel in (
        "proofs/noahmp/energy_savepoint_gate.py",
        "proofs/noahmp/water_savepoint_gate.py",
        "proofs/noahmp/phenology_savepoint_gate.py",
        "proofs/noahmp/integration_step_gate.py",
        "proofs/noahmp/coldstart_landonly_diag.py",
        "proofs/noahmp/build_noahmp_savepoints.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "gpuwrf.config.paths" in text
        assert 'ROOT.parent / "wrf_pristine"' not in text
        assert 'HERE.parents[2] / "wrf_pristine"' not in text
