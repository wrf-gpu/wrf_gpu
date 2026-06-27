"""Proof: WRF auxiliary-history (``auxhist``) secondary output stream.

Falsifiable claims this test pins down:

1. OFF BY DEFAULT / MAIN STREAM BYTE-UNCHANGED. A run with ``auxhist=None`` writes
   ONLY the hourly ``wrfout`` files and NO second stream; and a run *with* an
   auxhist stream writes the SAME main ``wrfout`` field values as the off run
   (the auxhist stream is purely additive -- it does not perturb the main stream).

2. DISTINCT, SCHEMA-VALID SECOND STREAM. With a 15-min surface-variable stream,
   the run writes a separate NetCDF stream whose files (a) are named per WRF
   ``auxhist{N}_d<domain>_<date>`` semantics, (b) carry ONLY the requested variable
   subset plus the ``Times``/``XTIME`` time coordinates, (c) are valid NetCDF4 with
   WRF dimensions/attrs, and (d) land at the requested 15/30/45/60-min timestamps.

3. GENUINE SUB-HOUR FRAMES. The 15-min frames carry distinct model values
   (forecast advanced proportionally to elapsed lead) -- they are real sub-hour GPU
   snapshots, not duplicated/interpolated hour boundaries.

4. WRF NAMELIST SEMANTICS. The filename/interval/frame-index helpers follow WRF
   ``auxhist{N}_outname`` / ``auxhist{N}_interval`` / ``frames_per_auxhist{N}``.

Pure CPU, no GPU: a synthetic in-place ``forecast_fn`` advances the state by a
``hours``-proportional increment so sub-hour snapshots are genuinely distinct.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset, chartostring

from gpuwrf.integration.daily_pipeline import (
    DailyCase,
    DailyPipelineConfig,
    _run_forecast_sequence,
)
from gpuwrf.io.auxhist_stream import (
    AuxhistStreamConfig,
    auxhist_output_boundaries,
)
from gpuwrf.io.wrfout_writer import FULL_WRFOUT_VARIABLES, MINIMUM_WRFOUT_VARIABLES


AUXHIST_VARS = ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC", "SWDOWN")


def _synthetic_state(nx: int = 3, ny: int = 2, nz: int = 2) -> SimpleNamespace:
    y2, x2 = np.indices((ny, nx), dtype=np.float32)
    z3 = np.arange(nz, dtype=np.float32)[:, None, None]
    zf = np.arange(nz + 1, dtype=np.float32)[:, None, None]
    terrain = 10.0 + y2 + x2
    pb = 90_000.0 - 500.0 * z3 + np.zeros((nz, ny, nx), dtype=np.float32)
    p_pert = 100.0 + np.zeros((nz, ny, nx), dtype=np.float32)
    phb = 9.81 * (terrain[None, :, :] + 500.0 * zf)
    ph_pert = np.ones((nz + 1, ny, nx), dtype=np.float32)
    mub = 85_000.0 + terrain
    mu_pert = 10.0 + np.zeros((ny, nx), dtype=np.float32)
    return SimpleNamespace(
        u=np.full((nz, ny, nx + 1), 4.0, dtype=np.float32),
        v=np.full((nz, ny + 1, nx), 1.0, dtype=np.float32),
        w=np.zeros((nz + 1, ny, nx), dtype=np.float32),
        theta=300.0 + z3 + np.zeros((nz, ny, nx), dtype=np.float32),
        qv=np.full((nz, ny, nx), 0.008, dtype=np.float32),
        qc=np.zeros((nz, ny, nx), dtype=np.float32),
        qi=np.zeros((nz, ny, nx), dtype=np.float32),
        qr=np.zeros((nz, ny, nx), dtype=np.float32),
        p_total=pb + p_pert,
        p_perturbation=p_pert,
        ph_total=phb + ph_pert,
        ph_perturbation=ph_pert,
        mu_total=mub + mu_pert,
        mu_perturbation=mu_pert,
        u10=np.full((ny, nx), 4.0, dtype=np.float32),
        v10=np.full((ny, nx), 1.0, dtype=np.float32),
        t2=np.full((ny, nx), 289.0, dtype=np.float32),
        q2=np.full((ny, nx), 0.007, dtype=np.float32),
        psfc=pb[0] + p_pert[0],
        rainc=np.zeros((ny, nx), dtype=np.float32),
        rain_acc=np.zeros((ny, nx), dtype=np.float32),
        rainsh=np.zeros((ny, nx), dtype=np.float32),
        swdown=np.full((ny, nx), 500.0, dtype=np.float32),
        glw=np.full((ny, nx), 300.0, dtype=np.float32),
        pblh=np.full((ny, nx), 800.0, dtype=np.float32),
        ustar=np.full((ny, nx), 0.3, dtype=np.float32),
        hfx=np.full((ny, nx), 20.0, dtype=np.float32),
        lh=np.full((ny, nx), 70.0, dtype=np.float32),
        t_skin=np.full((ny, nx), 290.0, dtype=np.float32),
        cldfra=np.zeros((nz, ny, nx), dtype=np.float32),
        landmask=np.ones((ny, nx), dtype=np.float32),
        lu_index=np.ones((ny, nx), dtype=np.float32) * 2.0,
    )


def _synthetic_case(run_dir: Path) -> DailyCase:
    state = _synthetic_state()
    grid = SimpleNamespace(
        nx=3,
        ny=2,
        nz=2,
        projection=SimpleNamespace(kind="lambert", lat_0=28.3, lon_0=-16.1, dx_m=3000.0, dy_m=3000.0),
        vertical=SimpleNamespace(nz=2, top_pressure_pa=5000.0),
        terrain_height=np.ones((2, 3), dtype=np.float32) * 10.0,
    )
    namelist = {
        "title": " OUTPUT FROM GPUWRF AUXHIST PROOF",
        "truelat1": 25.0,
        "truelat2": 30.0,
        "stand_lon": -16.4,
        "moad_cen_lat": 28.3,
        "cen_lat": 28.3,
        "cen_lon": -16.1,
        "soil_layers_stag": 4,
    }
    # ``standalone_native_init`` + a non-replay source disable the hourly CPU-WRF
    # land re-snap (no corpus on disk in the test) without touching output logic.
    return DailyCase(
        state=state,
        grid=grid,
        namelist=namelist,
        run_start=datetime(2026, 5, 21, 18, tzinfo=timezone.utc),
        metadata={"run_id": "auxhist-proof", "run_dir": str(run_dir), "source": "synthetic"},
    )


def _advancing_forecast_fn(state: SimpleNamespace, namelist, hours: float) -> SimpleNamespace:
    """Advance the state by an increment proportional to the elapsed lead.

    Makes every sub-hour snapshot genuinely distinct so the test can prove the
    auxhist frames are not duplicated hour boundaries. In-place mutation mirrors
    the existing daily-pipeline test stub; ``hours`` is honoured (the real GPU path
    integrates ``hours`` of model time per call).
    """

    del namelist
    bump = 10.0 * float(hours)  # 10 K/hr -> 2.5 K per 15-min segment: clearly distinct
    state.t2 = state.t2 + bump
    state.theta = state.theta + bump
    state.u10 = state.u10 + 1.0 * float(hours)
    return state


def _run(
    tmp_path: Path,
    *,
    auxhist: AuxhistStreamConfig | None,
    hours: int,
    tag: str,
    async_output: bool = False,
    full_wrfout_variables: bool = False,
):
    run_dir = tmp_path / f"run_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = DailyPipelineConfig(
        run_id="auxhist-proof",
        hours=hours,
        output_dir=tmp_path / f"out_{tag}",
        proof_dir=tmp_path / f"proof_{tag}",
        score=False,
        refresh_land_state_hourly=False,
        async_output=async_output,
        auxhist=auxhist,
        full_wrfout_variables=full_wrfout_variables,
    )
    return _run_forecast_sequence(
        config,
        output_dir=config.output_dir,
        forecast_fn=_advancing_forecast_fn,
        case_builder=lambda cfg: (_synthetic_case(run_dir), run_dir),
    )


# --------------------------------------------------------------------------- #
# Claim 4: WRF auxhist namelist semantics (filename / interval / frame index). #
# --------------------------------------------------------------------------- #
def test_auxhist_config_follows_wrf_namelist_semantics() -> None:
    aux = AuxhistStreamConfig(stream_id=2, interval_minutes=15, variables=AUXHIST_VARS)
    # Default outname pattern == WRF auxhist{N}_d<domain>_<date>.
    assert aux.outname_pattern == "auxhist2_d<domain>_<date>"
    valid = datetime(2026, 5, 21, 18, 15, 0)
    assert aux.filename(valid, "d02") == "auxhist2_d02_2026-05-21_18:15:00"
    assert aux.filename(valid, "02") == "auxhist2_d02_2026-05-21_18:15:00"  # both domain forms
    # Custom outname pattern still substitutes both tokens.
    custom = AuxhistStreamConfig(stream_id=5, interval_minutes=30, variables=AUXHIST_VARS,
                                 outname="surf_<domain>_<date>.nc")
    assert custom.filename(valid, "d03") == "surf_03_2026-05-21_18:15:00.nc"
    # Interval cadence: fires at multiples of the interval, never at t=0.
    assert not aux.fires_at(0.0)
    assert aux.fires_at(15.0) and aux.fires_at(30.0) and aux.fires_at(60.0)
    assert not aux.fires_at(20.0)
    assert aux.frame_index(45.0) == 3
    # Boundary enumeration over 1 h at 15 min -> 4 frames at :15/:30/:45/:00.
    bounds = auxhist_output_boundaries(datetime(2026, 5, 21, 18, 0, 0), 1.0, aux)
    assert [k for k, _, _ in bounds] == [1, 2, 3, 4]
    assert [vt.strftime("%H:%M") for _, _, vt in bounds] == ["18:15", "18:30", "18:45", "19:00"]


# --------------------------------------------------------------------------- #
# Claim 1: off by default + main stream byte-unchanged.                        #
# --------------------------------------------------------------------------- #
def test_no_auxhist_writes_only_main_stream(tmp_path: Path) -> None:
    result = _run(tmp_path, auxhist=None, hours=1, tag="off")
    assert len(result.output_files) == 1
    assert result.auxhist_files == []
    # The output dir contains exactly the one main wrfout file, nothing else.
    written = sorted(p.name for p in result.output_dir.iterdir() if p.is_file())
    assert written == [result.output_files[0].name]
    assert written[0].startswith("wrfout_d02_")


def test_auxhist_does_not_change_main_stream(tmp_path: Path) -> None:
    off = _run(tmp_path, auxhist=None, hours=1, tag="m_off")
    on = _run(
        tmp_path,
        auxhist=AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=AUXHIST_VARS),
        hours=1,
        tag="m_on",
    )
    # Same number of main wrfouts, named identically, with identical field values:
    # the auxhist stream is purely additive.
    assert [p.name for p in off.output_files] == [p.name for p in on.output_files]
    for off_path, on_path in zip(off.output_files, on.output_files):
        with Dataset(off_path) as a, Dataset(on_path) as b:
            assert set(a.variables) == set(b.variables)
            for name in a.variables:
                av = np.asarray(a[name][:])
                bv = np.asarray(b[name][:])
                if av.dtype.kind in {"S", "U"}:
                    assert np.array_equal(av, bv), name
                else:
                    np.testing.assert_array_equal(av, bv, err_msg=name)


# --------------------------------------------------------------------------- #
# Claim 2 + 3: distinct schema-valid 2nd stream w/ requested vars at requested  #
# timestamps; genuine sub-hour frames.                                          #
# --------------------------------------------------------------------------- #
def test_auxhist_emits_distinct_subset_stream_at_requested_cadence(tmp_path: Path) -> None:
    aux = AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=AUXHIST_VARS)
    result = _run(tmp_path, auxhist=aux, hours=1, tag="on")

    # --- One main wrfout (hourly) + four auxhist frames (every 15 min). ---
    assert len(result.output_files) == 1
    assert len(result.auxhist_files) == 4

    # --- WRF-named filenames at the exact requested timestamps. ---
    expected_names = [
        "auxhist1_d02_2026-05-21_18:15:00",
        "auxhist1_d02_2026-05-21_18:30:00",
        "auxhist1_d02_2026-05-21_18:45:00",
        "auxhist1_d02_2026-05-21_19:00:00",
    ]
    assert [p.name for p in result.auxhist_files] == expected_names
    for path in result.auxhist_files:
        assert path.is_file() and path.stat().st_size > 0

    # --- Auxhist files are NOT confused with the main stream (distinct files). ---
    main_names = {p.name for p in result.output_files}
    assert main_names.isdisjoint(set(expected_names))

    requested = set(AUXHIST_VARS)
    t2_per_frame = []
    for path, name in zip(result.auxhist_files, expected_names):
        with Dataset(path) as ds:
            # (a) valid NetCDF4 with WRF dims.
            assert ds.file_format == "NETCDF4"
            assert len(ds.dimensions["Time"]) == 1
            assert len(ds.dimensions["west_east"]) == 3
            assert len(ds.dimensions["south_north"]) == 2
            assert ds.getncattr("Conventions") == "WRF-ARW"
            # (b) carries EXACTLY the requested subset + the time coordinates.
            variables = set(ds.variables)
            assert requested <= variables, f"missing requested vars in {name}"
            extra = variables - requested - {"Times", "XTIME"}
            assert extra == set(), f"auxhist file {name} carries non-requested vars {extra}"
            # (c) the timestamp embedded in the file matches the filename stamp.
            stamp = name.split("_d02_", 1)[1]
            assert chartostring(ds["Times"][:])[0] == stamp
            # (d) requested fields are present, real, finite arrays (not fabricated).
            for var in requested:
                arr = np.asarray(ds[var][:])
                assert arr.shape == (1, 2, 3)
                assert np.isfinite(arr).all()
            t2_per_frame.append(float(np.asarray(ds["T2"][:]).mean()))

    # --- Genuine sub-hour frames: T2 strictly increases frame-to-frame (the
    #     forecast advanced 2.5 K per 15-min segment). Not duplicated boundaries. ---
    assert t2_per_frame == sorted(t2_per_frame)
    assert all(b - a > 1.0 for a, b in zip(t2_per_frame[:-1], t2_per_frame[1:]))


def test_auxhist_main_stream_keeps_full_field_set(tmp_path: Path) -> None:
    """The auxhist subset restriction must NOT leak into the main wrfout stream."""

    aux = AuxhistStreamConfig(stream_id=1, interval_minutes=30, variables=("T2", "U10"))
    result = _run(tmp_path, auxhist=aux, hours=1, tag="full")
    with Dataset(result.output_files[0]) as ds:
        missing = [name for name in MINIMUM_WRFOUT_VARIABLES if name not in ds.variables]
        assert missing == [], f"main wrfout lost fields under auxhist: {missing}"
    # The 30-min auxhist stream carries only the 2 requested vars (+ time coords).
    with Dataset(result.auxhist_files[0]) as ds:
        assert set(ds.variables) - {"Times", "XTIME"} == {"T2", "U10"}


def test_full_wrfout_opt_in_pipeline_writes_full_main_and_auxhist(tmp_path: Path) -> None:
    """A small-grid forecast run can opt into the full WRF history schema."""

    aux = AuxhistStreamConfig(stream_id=1, interval_minutes=60, variables=FULL_WRFOUT_VARIABLES)
    result = _run(
        tmp_path,
        auxhist=aux,
        hours=1,
        tag="full_wrfout",
        full_wrfout_variables=True,
    )
    assert len(result.output_files) == 1
    assert len(result.auxhist_files) == 1
    stream_meta = result.metadata["auxhist_streams"][0]
    assert stream_meta["prepared_full_variable_set"] is True
    assert stream_meta["full_variable_count"] == 375

    with Dataset(result.output_files[0]) as main, Dataset(result.auxhist_files[0]) as auxhist_ds:
        assert list(main.variables) == list(FULL_WRFOUT_VARIABLES)
        assert list(auxhist_ds.variables) == list(FULL_WRFOUT_VARIABLES)
        assert len(main.variables) == 375
        assert len(auxhist_ds.variables) == 375


def test_auxhist_hourly_interval_matches_main_cadence(tmp_path: Path) -> None:
    """A 60-min auxhist stream fires once per hour (one sub-step), aligned to wrfout."""

    aux = AuxhistStreamConfig(stream_id=3, interval_minutes=60, variables=("T2", "PSFC"))
    result = _run(tmp_path, auxhist=aux, hours=2, tag="hourly")
    assert len(result.output_files) == 2
    assert [p.name for p in result.auxhist_files] == [
        "auxhist3_d02_2026-05-21_19:00:00",
        "auxhist3_d02_2026-05-21_20:00:00",
    ]


def test_auxhist_async_writer_path_emits_distinct_subset_frames(tmp_path: Path) -> None:
    """The background (async) writer serializes BOTH streams correctly.

    Exercises ``AsyncWrfoutWriter.submit_subset``: the auxhist frame reuses the same
    host payload (no extra device pull) and is written on the same writer thread as
    the main stream, with the requested subset and genuinely distinct sub-hour T2.
    """

    aux = AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=("U10", "T2", "PSFC"))
    result = _run(tmp_path, auxhist=aux, hours=1, tag="async", async_output=True)
    assert len(result.output_files) == 1
    assert [p.name for p in result.auxhist_files] == [
        "auxhist1_d02_2026-05-21_18:15:00",
        "auxhist1_d02_2026-05-21_18:30:00",
        "auxhist1_d02_2026-05-21_18:45:00",
        "auxhist1_d02_2026-05-21_19:00:00",
    ]
    t2_means = []
    for path in result.auxhist_files:
        with Dataset(path) as ds:
            assert set(ds.variables) - {"Times", "XTIME"} == {"U10", "T2", "PSFC"}
            t2_means.append(float(np.asarray(ds["T2"][:]).mean()))
    assert all(b - a > 1.0 for a, b in zip(t2_means[:-1], t2_means[1:]))
