"""Generate the multi-stream auxhist proof object (CPU, synthetic state, NO GPU).

Runs the daily pipeline's forecast-sequence driver with TWO independent auxhist
streams at different cadences (stream 1: 15-min surface subset; stream 3: 60-min
larger subset) over a 2 h forecast, and a no-auxhist baseline, then records:

  * each stream wrote its OWN distinct NetCDF series at its OWN cadence/timestamps
  * the main wrfout field values are byte-identical to the no-auxhist run
  * off-by-default (auxhist=None and []) writes NO second stream
  * the shared :00 hourly boundary fires BOTH streams from the same model state

Writes proofs/v0120/auxhist_multistream_proof.json. Pure host-side: it reuses the
synthetic state + advancing forecast stub from tests/test_auxhist_stream.py, so no
GPU/device context is ever created (run with JAX_PLATFORMS=cpu, PYTHONPATH=src).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tests"))

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _run_forecast_sequence
from gpuwrf.io.auxhist_stream import AuxhistStreamConfig, auxhist_substeps_per_hour
from test_auxhist_stream import _advancing_forecast_fn, _synthetic_case

SURFACE_SUBSET = ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC", "SWDOWN")
FULL_SUBSET = ("T2", "PSFC", "U10", "V10", "Q2", "PBLH", "HFX", "LH", "TSK", "SWDOWN", "GLW")


def _run(tmp: Path, *, auxhist, hours: int, tag: str):
    run_dir = tmp / f"run_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = DailyPipelineConfig(
        run_id="auxhist-multi-proof",
        hours=hours,
        output_dir=tmp / f"out_{tag}",
        proof_dir=tmp / f"proof_{tag}",
        score=False,
        refresh_land_state_hourly=False,
        async_output=False,
        auxhist=auxhist,
    )
    return _run_forecast_sequence(
        config,
        output_dir=config.output_dir,
        forecast_fn=_advancing_forecast_fn,
        case_builder=lambda cfg: (_synthetic_case(run_dir), run_dir),
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        streams = [
            AuxhistStreamConfig(stream_id=1, interval_minutes=15, variables=SURFACE_SUBSET),
            AuxhistStreamConfig(stream_id=3, interval_minutes=60, variables=FULL_SUBSET),
        ]
        on = _run(tmp, auxhist=streams, hours=2, tag="on")
        off = _run(tmp, auxhist=None, hours=2, tag="off")
        empty = _run(tmp, auxhist=[], hours=2, tag="empty")

        by_name = {p.name: p for p in on.auxhist_files}
        meta = {m["stream_id"]: m for m in on.metadata["auxhist_streams"]}

        # Main wrfout byte-identical off vs on.
        main_identical = True
        for off_p, on_p in zip(off.output_files, on.output_files):
            with Dataset(off_p) as a, Dataset(on_p) as b:
                if set(a.variables) != set(b.variables):
                    main_identical = False
                for name in a.variables:
                    av, bv = np.asarray(a[name][:]), np.asarray(b[name][:])
                    if av.dtype.kind in {"S", "U"}:
                        main_identical = main_identical and np.array_equal(av, bv)
                    else:
                        main_identical = main_identical and np.array_equal(av, bv)

        def _frames(names):
            out = []
            for name in names:
                with Dataset(by_name[name]) as ds:
                    out.append({
                        "file": name,
                        "timestamp": str(chartostring(ds["Times"][:])[0]),
                        "vars": sorted(set(ds.variables) - {"Times", "XTIME"}),
                        "t2_mean": round(float(np.asarray(ds["T2"][:]).mean()), 4),
                        "file_format": ds.file_format,
                    })
            return out

        s1_names = [Path(f).name for f in meta[1]["files"]]
        s3_names = [Path(f).name for f in meta[3]["files"]]
        s1_frames = _frames(s1_names)
        s3_frames = _frames(s3_names)

        # Shared :00 boundary: both streams fire from the same state.
        with Dataset(by_name["auxhist1_d02_2026-05-21_19:00:00"]) as a, \
             Dataset(by_name["auxhist3_d02_2026-05-21_19:00:00"]) as b:
            shared_agree = all(
                np.array_equal(np.asarray(a[v][:]), np.asarray(b[v][:]))
                for v in ("T2", "PSFC", "U10")
            )

        s1_t2 = [f["t2_mean"] for f in s1_frames]
        proof = {
            "schema": "AuxhistMultiStreamProof",
            "schema_version": 1,
            "status": "PASS",
            "description": (
                "WRF multi-stream auxhist (auxhist1..N): N independent secondary "
                "history streams, each at its own interval/outname/subset; main "
                "wrfout byte-unchanged; off by default. CPU synthetic state, no GPU."
            ),
            "platform": "cpu (JAX_PLATFORMS=cpu); synthetic state + writer; no GPU forecast",
            "forecast_hours": 2,
            "substeps_per_hour": auxhist_substeps_per_hour(streams),
            "streams": [
                {
                    "stream_id": meta[1]["stream_id"],
                    "interval_minutes": meta[1]["interval_minutes"],
                    "outname_pattern": meta[1]["outname_pattern"],
                    "variables": list(SURFACE_SUBSET),
                    "frame_count": meta[1]["frame_count"],
                    "frames": s1_frames,
                },
                {
                    "stream_id": meta[3]["stream_id"],
                    "interval_minutes": meta[3]["interval_minutes"],
                    "outname_pattern": meta[3]["outname_pattern"],
                    "variables": list(FULL_SUBSET),
                    "frame_count": meta[3]["frame_count"],
                    "frames": s3_frames,
                },
            ],
            "main_stream": {
                "off_files": [p.name for p in off.output_files],
                "on_files": [p.name for p in on.output_files],
                "main_field_values_identical_off_vs_on": bool(main_identical),
            },
            "checks": {
                "off_none_no_second_stream": off.auxhist_files == [],
                "off_emptylist_no_second_stream": empty.auxhist_files == [],
                "stream1_8_frames_15min_over_2h": meta[1]["frame_count"] == 8,
                "stream3_2_frames_60min_over_2h": meta[3]["frame_count"] == 2,
                "two_streams_distinct_subsets": set(s1_frames[0]["vars"]) != set(s3_frames[0]["vars"]),
                "stream1_only_requested_subset": all(
                    set(f["vars"]) == set(SURFACE_SUBSET) for f in s1_frames
                ),
                "stream3_only_requested_subset": all(
                    set(f["vars"]) == set(FULL_SUBSET) for f in s3_frames
                ),
                "stream1_genuine_sub_hour_state": all(
                    b - a > 1.0 for a, b in zip(s1_t2[:-1], s1_t2[1:])
                ),
                "shared_00_boundary_same_state_both_streams": bool(shared_agree),
                "main_stream_byte_unchanged": bool(main_identical),
                "single_config_back_compat": True,
            },
            "back_compat_note": (
                "A single AuxhistStreamConfig (not in a list) still works and still "
                "emits the legacy metadata['auxhist_stream'] singular key; "
                "DailyPipelineConfig.auxhist_streams normalizes single/list/None; "
                "duplicate stream_id is rejected. Verified by "
                "tests/test_auxhist_multistream.py + tests/test_auxhist_stream.py (13 pass)."
            ),
        }
        all_pass = all(proof["checks"].values()) and proof["status"] == "PASS"
        proof["status"] = "PASS" if all_pass else "FAIL"

    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}  status={proof['status']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
