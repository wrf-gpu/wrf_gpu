"""Proof: WRF MULTI-stream auxiliary-history (``auxhist1..N``) secondary output.

WRF drives up to 24 INDEPENDENT auxhist streams (``auxhist1``..``auxhist24``),
each with its OWN interval / outname / frames_per_file / variable subset. This
proof generalizes the single-stream proof in ``test_auxhist_stream.py`` to N
configurable streams. Falsifiable claims pinned here:

1. TWO STREAMS AT DIFFERENT CADENCES. A run configured with TWO auxhist streams
   -- a 15-min SURFACE-SUBSET stream (stream 1) and a 60-min FULL-FIELD stream
   (stream 3) over a 2 h forecast -- writes EACH stream as its OWN distinct,
   schema-valid NetCDF series at its OWN cadence/timestamps: stream 1 at
   :15/:30/:45/:00 x2h (8 frames), stream 3 hourly (2 frames). The two series do
   NOT collide (distinct WRF ``auxhist{N}_*`` filenames + variable sets).

2. MAIN WRFOUT BYTE-UNCHANGED. The main hourly ``wrfout`` field values are
   IDENTICAL to a no-auxhist run -- the multi-stream config is purely additive.

3. OFF BY DEFAULT. ``auxhist=None`` -> the run writes ONLY the main wrfout files
   and NO second stream (``auxhist_files == []``); the empty list is the same.

4. GENUINE SUB-HOUR FRAMES + SHARED-BOUNDARY. The 15-min stream's frames carry
   distinct (advancing) model values (real sub-hour snapshots). At the hourly
   :00 boundary BOTH streams fire and each writes its own file from the SAME
   model state -- the shared-state values agree across the two streams.

5. BACK-COMPAT. A single ``AuxhistStreamConfig`` (not wrapped in a list) still
   works exactly as before; ``DailyPipelineConfig.auxhist_streams`` normalizes
   the single / list / None forms; duplicate ``stream_id`` is rejected.

Pure CPU, no GPU: reuses the synthetic state + advancing ``forecast_fn`` from
``test_auxhist_stream``; the writer emits real NetCDF bytes from host numpy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset, chartostring

from gpuwrf.integration.daily_pipeline import (
    DailyPipelineConfig,
    _run_forecast_sequence,
)
from gpuwrf.io.auxhist_stream import (
    AuxhistStreamConfig,
    auxhist_substeps_per_hour,
    coerce_auxhist_streams,
)
from gpuwrf.io.wrfout_writer import MINIMUM_WRFOUT_VARIABLES

# Reuse the synthetic case + advancing forecast stub from the single-stream proof
# (same host-only, no-GPU machinery) so this proof shares the exact fixtures.
from test_auxhist_stream import (
    _advancing_forecast_fn,
    _synthetic_case,
)


SURFACE_SUBSET = ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC", "SWDOWN")
FULL_SUBSET = ("T2", "PSFC", "U10", "V10", "Q2", "PBLH", "HFX", "LH", "TSK", "SWDOWN", "GLW")


def _run(
    tmp_path: Path,
    *,
    auxhist,
    hours: int,
    tag: str,
    async_output: bool = False,
):
    run_dir = tmp_path / f"run_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = DailyPipelineConfig(
        run_id="auxhist-multi-proof",
        hours=hours,
        output_dir=tmp_path / f"out_{tag}",
        proof_dir=tmp_path / f"proof_{tag}",
        score=False,
        refresh_land_state_hourly=False,
        async_output=async_output,
        auxhist=auxhist,
    )
    return _run_forecast_sequence(
        config,
        output_dir=config.output_dir,
        forecast_fn=_advancing_forecast_fn,
        case_builder=lambda cfg: (_synthetic_case(run_dir), run_dir),
    )


# --------------------------------------------------------------------------- #
# Claim 5 (pure unit): coercion + cadence helpers + duplicate-id rejection.    #
# --------------------------------------------------------------------------- #
def test_coerce_and_substeps_helpers() -> None:
    s1 = AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=SURFACE_SUBSET)
    s3 = AuxhistStreamConfig(stream_id=3, interval_minutes=60, variables=("T2", "PSFC"))
    # None -> (); single -> 1-tuple; list -> tuple (order preserved).
    assert coerce_auxhist_streams(None) == ()
    assert coerce_auxhist_streams(s1) == (s1,)
    assert coerce_auxhist_streams([s1, s3]) == (s1, s3)
    # Substeps: gcd(60, 15, 60) == 15 -> 4 segments/hour (serves both streams).
    assert auxhist_substeps_per_hour(None) == 1
    assert auxhist_substeps_per_hour([s1, s3]) == 4
    # gcd(60, 20, 12) == 4 -> 15 segments/hour.
    s_a = AuxhistStreamConfig(stream_id=2, interval_minutes=20, variables=("T2",))
    s_b = AuxhistStreamConfig(stream_id=4, interval_minutes=12, variables=("T2",))
    assert auxhist_substeps_per_hour([s_a, s_b]) == 15
    # Duplicate stream_id across the list is rejected (WRF: each stream once).
    dup = AuxhistStreamConfig(stream_id=1, interval_minutes=30, variables=("T2",))
    with pytest.raises(ValueError, match="duplicate auxhist stream_id"):
        coerce_auxhist_streams([s1, dup])
    # Pipeline config normalizes too.
    cfg = DailyPipelineConfig(run_id="x", hours=1, auxhist=[s1, s3])
    assert cfg.auxhist_streams == (s1, s3)


# --------------------------------------------------------------------------- #
# Claim 3: OFF by default (None) AND the empty-list form.                      #
# --------------------------------------------------------------------------- #
def test_off_by_default_none_and_empty_list(tmp_path: Path) -> None:
    for tag, aux in (("none", None), ("empty", [])):
        result = _run(tmp_path, auxhist=aux, hours=1, tag=f"off_{tag}")
        assert result.auxhist_files == [], tag
        # Output dir holds exactly the one main wrfout file, nothing else.
        written = sorted(p.name for p in result.output_dir.iterdir() if p.is_file())
        assert written == [result.output_files[0].name], tag
        assert written[0].startswith("wrfout_d02_"), tag
        assert "auxhist_streams" not in result.metadata, tag


# --------------------------------------------------------------------------- #
# Claims 1 + 4: TWO streams at different cadences, each its own distinct series. #
# --------------------------------------------------------------------------- #
def _assert_two_stream_run(result) -> None:
    # --- 2 main hourly wrfouts. ---
    assert len(result.output_files) == 2

    # --- Stream 1 (15 min over 2 h) -> 8 frames; stream 3 (60 min) -> 2 frames. ---
    s1_names = [
        "auxhist1_d02_2026-05-21_18:15:00",
        "auxhist1_d02_2026-05-21_18:30:00",
        "auxhist1_d02_2026-05-21_18:45:00",
        "auxhist1_d02_2026-05-21_19:00:00",
        "auxhist1_d02_2026-05-21_19:15:00",
        "auxhist1_d02_2026-05-21_19:30:00",
        "auxhist1_d02_2026-05-21_19:45:00",
        "auxhist1_d02_2026-05-21_20:00:00",
    ]
    s3_names = [
        "auxhist3_d02_2026-05-21_19:00:00",
        "auxhist3_d02_2026-05-21_20:00:00",
    ]
    by_name = {p.name: p for p in result.auxhist_files}
    # All 10 auxhist files written, distinct, non-empty.
    assert sorted(by_name) == sorted(s1_names + s3_names)
    assert len(result.auxhist_files) == 10
    for p in result.auxhist_files:
        assert p.is_file() and p.stat().st_size > 0

    # --- Per-stream metadata reports each stream's own cadence/frames/files. ---
    meta = {m["stream_id"]: m for m in result.metadata["auxhist_streams"]}
    assert set(meta) == {1, 3}
    assert meta[1]["interval_minutes"] == 15 and meta[1]["frame_count"] == 8
    assert meta[3]["interval_minutes"] == 60 and meta[3]["frame_count"] == 2
    assert [Path(f).name for f in meta[1]["files"]] == s1_names
    assert [Path(f).name for f in meta[3]["files"]] == s3_names
    # Multi-stream run does NOT emit the singular back-compat key.
    assert "auxhist_stream" not in result.metadata

    # --- Stream 1: schema-valid surface subset at the 15-min timestamps. ---
    s1_req = set(SURFACE_SUBSET)
    t2_per_frame = []
    for name in s1_names:
        with Dataset(by_name[name]) as ds:
            assert ds.file_format == "NETCDF4"
            assert len(ds.dimensions["Time"]) == 1
            assert ds.getncattr("Conventions") == "WRF-ARW"
            variables = set(ds.variables)
            assert s1_req <= variables
            assert variables - s1_req - {"Times", "XTIME"} == set()
            stamp = name.split("_d02_", 1)[1]
            assert chartostring(ds["Times"][:])[0] == stamp
            t2_per_frame.append(float(np.asarray(ds["T2"][:]).mean()))
    # Genuine sub-hour frames: T2 strictly increases (not duplicated boundaries).
    assert t2_per_frame == sorted(t2_per_frame)
    assert all(b - a > 1.0 for a, b in zip(t2_per_frame[:-1], t2_per_frame[1:]))

    # --- Stream 3: its own (larger) subset at the hourly timestamps. ---
    s3_req = set(FULL_SUBSET)
    for name in s3_names:
        with Dataset(by_name[name]) as ds:
            variables = set(ds.variables)
            assert s3_req <= variables
            assert variables - s3_req - {"Times", "XTIME"} == set()
            stamp = name.split("_d02_", 1)[1]
            assert chartostring(ds["Times"][:])[0] == stamp

    # --- The two streams carry DIFFERENT variable sets (independent subsets). ---
    with Dataset(by_name[s1_names[0]]) as a, Dataset(by_name[s3_names[0]]) as b:
        a_vars = set(a.variables) - {"Times", "XTIME"}
        b_vars = set(b.variables) - {"Times", "XTIME"}
        assert a_vars != b_vars
        assert "RAINNC" in a_vars and "RAINNC" not in b_vars  # surface-only field
        assert "PBLH" in b_vars and "PBLH" not in a_vars       # full-stream-only field

    # --- Claim 4 (shared :00 boundary): at 19:00 BOTH streams fire from the SAME
    #     model state, so their shared fields (T2, PSFC) agree exactly. ---
    with Dataset(by_name["auxhist1_d02_2026-05-21_19:00:00"]) as a, \
         Dataset(by_name["auxhist3_d02_2026-05-21_19:00:00"]) as b:
        for shared in ("T2", "PSFC", "U10"):
            np.testing.assert_array_equal(
                np.asarray(a[shared][:]), np.asarray(b[shared][:]), err_msg=shared
            )


def test_two_streams_distinct_cadence_sync(tmp_path: Path) -> None:
    aux = [
        AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=SURFACE_SUBSET),
        AuxhistStreamConfig(stream_id=3, interval_minutes=60, variables=FULL_SUBSET),
    ]
    result = _run(tmp_path, auxhist=aux, hours=2, tag="two_sync")
    _assert_two_stream_run(result)


def test_two_streams_distinct_cadence_async(tmp_path: Path) -> None:
    """Same two-stream proof through the BACKGROUND (async) writer path."""

    aux = [
        AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=SURFACE_SUBSET),
        AuxhistStreamConfig(stream_id=3, interval_minutes=60, variables=FULL_SUBSET),
    ]
    result = _run(tmp_path, auxhist=aux, hours=2, tag="two_async", async_output=True)
    _assert_two_stream_run(result)


# --------------------------------------------------------------------------- #
# Claim 2: main wrfout byte-unchanged vs a no-auxhist run.                      #
# --------------------------------------------------------------------------- #
def test_multistream_does_not_change_main_wrfout(tmp_path: Path) -> None:
    off = _run(tmp_path, auxhist=None, hours=2, tag="m_off")
    on = _run(
        tmp_path,
        auxhist=[
            AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=SURFACE_SUBSET),
            AuxhistStreamConfig(stream_id=3, interval_minutes=60, variables=FULL_SUBSET),
        ],
        hours=2,
        tag="m_on",
    )
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
    # The main wrfout keeps the FULL field set even though streams carry subsets.
    with Dataset(on.output_files[0]) as ds:
        missing = [n for n in MINIMUM_WRFOUT_VARIABLES if n not in ds.variables]
        assert missing == [], f"main wrfout lost fields under multi-stream: {missing}"


# --------------------------------------------------------------------------- #
# Claim 5: a single (un-listed) config still works AND keeps the legacy key.   #
# --------------------------------------------------------------------------- #
def test_single_config_back_compat(tmp_path: Path) -> None:
    aux = AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=SURFACE_SUBSET)
    result = _run(tmp_path, auxhist=aux, hours=1, tag="single")
    assert len(result.auxhist_files) == 4
    # Single-stream run still emits the legacy singular metadata key AND the new
    # list key (with one entry).
    assert result.metadata["auxhist_stream"]["stream_id"] == 1
    assert len(result.metadata["auxhist_streams"]) == 1
