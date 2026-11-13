"""Legacy consumer adapters for the V1 extract (explicit conversions, never under the V1 schema id).

* ``sfc_npz``    -> ALISIOS ``ops/nightly_9nest/extract9.py`` ``dNN_sfc.npz`` (keys, order, dtypes)
* ``cloud_npz``  -> ``ops/nightly_9nest/cloud9.py`` ``d03_cloud.npz``
* ``wt1_profiles`` -> WT1 ``teacher_extract_v2.PROFILE_SCHEMA`` table (``points_profile.parquet``)

The legacy time conventions are reproduced, not reinterpreted: extract9/cloud9 ``lead_h`` int16 of
the hour-rounded actual time minus issue, ``tau_h`` = frame index (int16), ``time_offset_seconds`` f32;
WT1 ``lead_h`` = round(lead) with offset in [0, 60) s, ``tau_h`` = lead_h - 6. Every adapter refuses an
input whose V1 time axis disagrees with the legacy reading (e.g. not hourly from WRF start).
CLI: ``python -m gpuwrf.extract.legacy {sfc,cloud,wt1} IN.nc OUT [--issue ...] [--attempt N]``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

STATIC = ("XLAT", "XLONG", "HGT", "LANDMASK", "COSALPHA", "SINALPHA")
DYN = ("U10", "V10", "T2", "Q2", "PSFC")
EXTRA = ("RAINC", "RAINNC", "SWDOWN", "GLW", "HFX", "LH", "PBLH")
CLOUD = ("cloud_max", "cloud_low_max", "LWP")
WT1_FIELDS = ("T_K", "theta_K", "pressure_Pa", "QV_kg_kg", "U_ms", "V_ms", "QCLOUD_kg_kg", "CLDFRA",
              "z_msl_m", "z_agl_m")


def _issue(ds) -> datetime:
    return datetime.fromisoformat(ds.getncattr("issue_utc")).astimezone(timezone.utc)


def _status_ok(var) -> bool:
    return json.loads(var.getncattr("field_status")).get("status") == "OK"


def _legacy_hourly_axis(ds) -> dict:
    """extract9.py time block on the V1 axis; refuses anything the legacy reader would read differently."""
    actual = np.asarray(ds["actual_time_unix_s"][:], np.int64)
    issue = _issue(ds)
    times = [datetime.fromtimestamp(round(int(a) / 3600.0) * 3600, tz=timezone.utc) for a in actual]
    offs = [int(a) - int(t.timestamp()) for a, t in zip(actual, times)]
    if any((b - a) != timedelta(hours=1) for a, b in zip(times, times[1:])) or any(abs(o) >= 60 for o in offs):
        raise ValueError("legacy NPZ needs hourly frames with |offset| < 60 s")
    nominal = np.asarray(ds["nominal_time_unix_s"][:], np.int64)
    if nominal.tolist() != [int(t.timestamp()) for t in times]:
        raise ValueError("V1 nominal times differ from the legacy hour rounding")
    tau = np.asarray(ds["tau_h"][:], np.float64)
    if tau.tolist() != list(range(len(actual))):
        raise ValueError("legacy tau_h is the frame index: frames must be hourly from WRF start (tau_h 0,1,..)")
    lead = np.array([(t - issue).total_seconds() / 3600 for t in times], dtype=np.int16)
    return {"lead_h": lead, "tau_h": np.arange(len(actual), dtype=np.int16), "actual_time_unix_s": actual,
            "time_offset_seconds": np.array(offs, dtype=np.float32)}


def sfc_npz(extract_nc: Path) -> tuple[dict, dict]:
    """Arrays of extract9 ``dNN_sfc.npz`` in its key order, plus a report (omitted extras)."""
    import netCDF4

    with netCDF4.Dataset(extract_nc) as ds:
        ds.set_auto_maskandscale(False)
        axis = _legacy_hourly_axis(ds)
        st = {v: np.asarray(ds[v][:], np.float32) for v in STATIC}
        for v in DYN:
            if not _status_ok(ds[v]):
                raise ValueError(f"legacy NPZ needs {v} (status {ds[v].getncattr('field_status')})")
        extras = [v for v in EXTRA if _status_ok(ds[v])]
        dyn = {v: np.asarray(ds[v][:], np.float32) for v in DYN + tuple(extras)}
        ue = np.asarray(ds["u10_east"][:], np.float32)
        vn = np.asarray(ds["v10_north"][:], np.float32)
        spd = np.asarray(ds["wind_speed"][:], np.float32)
    for v in DYN:
        if not np.isfinite(dyn[v]).all():
            raise ValueError(f"legacy NPZ: non-finite {v}")
    # extract9 direction (no calm NaN): recomputed from the identical f4 earth-relative components.
    wdir = (np.degrees(np.arctan2(-ue, -vn)) + 360.0) % 360.0
    arrays = dict(**st, lead_h=axis["lead_h"], tau_h=axis["tau_h"],
                  actual_time_unix_s=axis["actual_time_unix_s"], time_offset_seconds=axis["time_offset_seconds"],
                  **dyn, u10_east=ue, v10_north=vn, wind_speed=spd, wind_direction_from_deg=wdir)
    report = {"format": "extract9 dNN_sfc.npz", "extras": extras,
              "omitted_extras_not_OK": [v for v in EXTRA if v not in extras],
              "note": "wind_direction_from_deg recomputed with the extract9 formula (calm not NaN)"}
    return arrays, report


def cloud_npz(extract_nc: Path) -> tuple[dict, dict]:
    import netCDF4

    with netCDF4.Dataset(extract_nc) as ds:
        ds.set_auto_maskandscale(False)
        axis = _legacy_hourly_axis(ds)
        arrs = {k: np.asarray(ds[k][:], np.float32) for k in CLOUD}
        status = {k: json.loads(ds[k].getncattr("field_status")) for k in CLOUD}
    if any(s.get("status") != "OK" for s in status.values()):
        raise ValueError(f"legacy cloud NPZ needs OK cloud fields: {status}")
    if not (np.isfinite(arrs["cloud_max"]).all() and np.isfinite(arrs["LWP"]).all()):
        raise ValueError("legacy cloud NPZ: non-finite cloud_max/LWP")
    arrays = dict(**arrs, lead_h=axis["lead_h"], tau_h=axis["tau_h"],
                  actual_time_unix_s=axis["actual_time_unix_s"])
    return arrays, {"format": "cloud9 d03_cloud.npz"}


def write_npz(arrays: dict, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)


def wt1_schema():
    import pyarrow as pa
    return pa.schema([
        ('phys_id', pa.string()), ('role', pa.string()), ('issue', pa.string()),
        ('attempt', pa.int32()), ('valid_utc', pa.string()), ('actual_utc', pa.string()),
        ('lead_h', pa.int16()), ('tau_h', pa.int16()), ('time_offset_seconds', pa.float32()),
        ('level', pa.int16()), ('grid_y', pa.int32()), ('grid_x', pa.int32()),
        ('T_K', pa.float32()), ('theta_K', pa.float32()), ('pressure_Pa', pa.float32()),
        ('QV_kg_kg', pa.float32()), ('U_ms', pa.float32()), ('V_ms', pa.float32()),
        ('QCLOUD_kg_kg', pa.float32()), ('CLDFRA', pa.float32()),
        ('z_msl_m', pa.float32()), ('z_agl_m', pa.float32())])


def wt1_profiles(profiles_nc: Path, *, attempt: int = 0, roles: dict | None = None) -> list:
    """One WT1 PROFILE_SCHEMA table per frame (WT1 writes one row group per frame).

    ``role`` = ``roles[point_id]`` when given, else the point ``kind`` of the frozen CSV (explicit).
    Refused unless every frame's WT1 lead (round, offset in [0, 60) s) equals the V1 nominal axis and
    the leads are unique (WT1 keys rows by lead_h: two frames of one hour would duplicate rows)."""
    return _wt1_tables(profiles_nc, attempt=attempt, roles=roles, validate_axis=True)


def _wt1_tables(profiles_nc: Path, *, attempt: int, roles: dict | None, validate_axis: bool) -> list:
    """``validate_axis=False`` only for per-frame field parity tests on sub-hourly fixtures."""
    import netCDF4
    import pyarrow as pa

    tables = []
    with netCDF4.Dataset(profiles_nc) as ds:
        ds.set_auto_maskandscale(False)
        issue_dt = _issue(ds)
        issue = issue_dt.strftime("%Y%m%d_%Hz")
        start = datetime.fromisoformat(ds.getncattr("wrf_start_utc")).astimezone(timezone.utc)
        if start - issue_dt != timedelta(hours=6):
            raise ValueError("WT1 tau_h = lead_h - 6 needs WRF start = issue + 6 h")
        ids = list(ds["point_id"][:])
        kinds = list(ds["point_kind"][:])
        gy = np.asarray(ds["grid_y"][:], np.int32)
        gx = np.asarray(ds["grid_x"][:], np.int32)
        actual = np.asarray(ds["actual_time_unix_s"][:], np.int64)
        if validate_axis:
            nominal_v1 = np.asarray(ds["nominal_time_unix_s"][:], np.int64)
            issue_s = int(issue_dt.timestamp())
            leads = [int(round((int(a) - issue_s) / 3600)) for a in actual]
            if [issue_s + 3600 * h for h in leads] != nominal_v1.tolist():
                raise ValueError("WT1 lead (rounded actual time) differs from the V1 nominal time axis")
            if len(set(leads)) != len(leads):
                raise ValueError(f"WT1 leads not unique {leads}: two frames would write duplicate rows")
        for f in WT1_FIELDS:
            if json.loads(ds[f].getncattr("field_status")).get("status") != "OK":
                raise ValueError(f"WT1 profile field {f} not OK")
        nz = ds.dimensions["level"].size
        for t, a in enumerate(actual):
            act = datetime.fromtimestamp(int(a), tz=timezone.utc)
            lead_s = (act - issue_dt).total_seconds()
            nominal = int(round(lead_s / 3600))
            offset = lead_s - nominal * 3600
            if not 0 <= offset < 60:
                raise ValueError("actual time outside [nominal, nominal+60s) (WT1 time_coordinates)")
            raster = issue_dt + timedelta(hours=nominal)
            vals = {f: np.asarray(ds[f][t]) for f in WT1_FIELDS}
            data = {k: [] for k in wt1_schema().names}
            for p, pid in enumerate(ids):
                common = {"phys_id": pid, "role": (roles or {}).get(pid, kinds[p]), "issue": issue,
                          "attempt": attempt, "valid_utc": raster.isoformat(), "actual_utc": act.isoformat(),
                          "lead_h": nominal, "tau_h": nominal - 6,
                          "time_offset_seconds": (act - raster).total_seconds(),
                          "grid_y": int(gy[p]), "grid_x": int(gx[p])}
                for k, v in common.items():
                    data[k].extend([v] * nz)
                data["level"].extend(range(nz))
                for f in WT1_FIELDS:
                    data[f].extend(vals[f][p].tolist())
            tables.append(pa.Table.from_pydict(data, schema=wt1_schema()))
    return tables


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("sfc", "cloud", "wt1"))
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--attempt", type=int, default=0)
    args = ap.parse_args(argv)
    if args.kind in ("sfc", "cloud"):
        arrays, report = (sfc_npz if args.kind == "sfc" else cloud_npz)(args.input)
        write_npz(arrays, args.output)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if args.output.exists():
            raise FileExistsError(args.output)
        tables = wt1_profiles(args.input, attempt=args.attempt)
        with pq.ParquetWriter(args.output, wt1_schema(), compression="zstd") as w:
            for t in tables:
                if t.num_rows:
                    w.write_table(t, row_group_size=t.num_rows)
        report = {"format": "WT1 points_profile.parquet", "frames": len(tables),
                  "rows": int(sum(t.num_rows for t in tables))}
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
