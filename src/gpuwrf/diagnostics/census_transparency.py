"""CPU-only census on/off regression scoring against predeclared bounds."""
import argparse
import json
from pathlib import Path

import numpy as np


def verify_receipts(off, on, hours):
    """Require the same immutable source, inputs and arithmetic settings."""
    for name in ("git_head", "src_tree"):
        if not off.get(name) or off[name] != on.get(name):
            raise ValueError(f"different/missing receipt {name}")
    if off.get("rc") != 0 or on.get("rc") != 0:
        raise ValueError("incomplete/failed forecast receipt")
    for option, expected in (("--input-dir", None), ("--hours", str(hours))):
        def value(receipt):
            argv = receipt.get("argv", [])
            return argv[argv.index(option) + 1] if option in argv else None
        a, b = value(off), value(on)
        if a is None or a != b or (expected is not None and a != expected):
            raise ValueError(f"different/missing receipt {option}")
    env_off, env_on = off.get("env", {}), on.get("env", {})
    truth = lambda value: str(value).lower() in {"1", "true", "yes", "on"}
    if truth(env_off.get("GPUWRF_CENSUS", "0")) or not truth(env_on.get("GPUWRF_CENSUS", "0")):
        raise ValueError("expected census OFF/ON receipts")
    for key in set(env_off) | set(env_on):
        arithmetic = (key.startswith("GPUWRF_") and "CACHE" not in key
                      and not key.startswith(("GPUWRF_GPU_LOCK", "GPUWRF_LOCK"))
                      and key != "GPUWRF_CENSUS") or key in {"XLA_FLAGS", "JAX_ENABLE_X64", "JAX_PLATFORMS"}
        if arithmetic and env_off.get(key) != env_on.get(key):
            raise ValueError(f"different arithmetic environment key {key}")
    return {"git_head": off["git_head"], "src_tree": off["src_tree"], "hours": hours}


def compare_array(name, off, on, bounds):
    off, on = np.asarray(off), np.asarray(on)
    record = {"shape": list(off.shape), "dtype": str(off.dtype)}
    if off.shape != on.shape or off.dtype != on.dtype:
        return {**record, "pass": False, "reason": "shape/dtype mismatch"}
    if not np.issubdtype(off.dtype, np.floating):
        return {**record, "pass": bool(np.array_equal(off, on)), "policy": "exact"}
    if not np.all(np.isfinite(off)) or not np.all(np.isfinite(on)):
        return {**record, "pass": False, "reason": "nonfinite off/on values"}
    left, right = off.astype(np.float64), on.astype(np.float64)
    delta = right - left
    maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    field = bounds["carry_aliases"].get(name, name)
    if name.startswith("base_state."):
        limits = {"max_abs": 0.0, "rmse": 0.0}
    elif field in bounds["fields"]:
        limits = bounds["fields"][field]
    else:
        fallback = bounds["other_floating_fields"]
        absolute, relative = fallback["absolute_floor"], fallback["relative_to_off"]
        limits = {"max_abs": absolute + relative * (float(np.max(np.abs(left))) if left.size else 0),
                  "rmse": absolute + relative * (float(np.sqrt(np.mean(left * left))) if left.size else 0)}
    return {**record, "max_abs": maximum, "rmse": rms, "limits": limits,
            "pass": maximum <= limits["max_abs"] and rms <= limits["rmse"]}


def compare_wrfout(off_dir, on_dir, bounds, expected_hours):
    """Read paired same-source files; require both domains and full hour coverage."""
    from datetime import datetime
    from gpuwrf.io.netcdf_lock import Dataset

    records = {}
    for domain in ("d01", "d02"):
        left = {p.name: p for p in off_dir.glob(f"wrfout_{domain}_*") if p.is_file()}
        right = {p.name: p for p in on_dir.glob(f"wrfout_{domain}_*") if p.is_file()}
        if not left or set(left) != set(right):
            raise ValueError(f"{domain}: missing/mismatched file coverage")
        times = [datetime.strptime(n.removeprefix(f"wrfout_{domain}_"), "%Y-%m-%d_%H:%M:%S") for n in sorted(left)]
        # Initial output is mandatory; no absent hourly checkpoint may pass.
        if len(times) < expected_hours + 1 or (times[-1] - times[0]).total_seconds() < expected_hours * 3600:
            raise ValueError(f"{domain}: incomplete {expected_hours} h coverage")
        if any((b - a).total_seconds() > 3660 for a, b in zip(times, times[1:])):
            raise ValueError(f"{domain}: missing hourly checkpoint")
        records[domain] = {}
        for filename in sorted(left):
            with Dataset(left[filename]) as a, Dataset(right[filename]) as b:
                if set(a.variables) != set(b.variables):
                    raise ValueError(f"{domain}/{filename}: different field inventory")
                if not set(bounds["required_output_fields"]).issubset(a.variables):
                    raise ValueError(f"{domain}/{filename}: missing mandatory D6 fields")
                fields = {}
                for name in a.variables:
                    # Masked fill values are failures, never silently removed.
                    def read(var):
                        data = var[:]
                        if np.ma.isMaskedArray(data) and np.any(np.ma.getmaskarray(data)):
                            raise ValueError(f"{domain}/{filename}/{name}: masked values")
                        return np.asarray(data)
                    fields[name] = compare_array(name, read(a[name]), read(b[name]), bounds)
                records[domain][filename] = fields
    return {"scope": "census-on/off regression, not WRF fidelity", "domains": records,
            "pass": all(r["pass"] for files in records.values() for fields in files.values() for r in fields.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--bounds", type=Path, required=True)
    parser.add_argument("--hours", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bounds = json.loads(args.bounds.read_text())
    import jsonschema
    try:
        off_receipt = json.loads((args.off.parent / "receipt.json").read_text())
        on_receipt = json.loads((args.on.parent / "receipt.json").read_text())
        provenance = verify_receipts(off_receipt, on_receipt, args.hours)
        result = compare_wrfout(args.off, args.on, bounds, args.hours)
        census = json.loads((args.on / "census.json").read_text())
        jsonschema.validate(census, json.loads(Path(__file__).with_name("census_schema.json").read_text()))
        for domain in ("d01", "d02"):
            record = census["domains"][domain]
            if record["actual_work"]["steps"] != record["own_steps"]:
                raise ValueError(f"{domain}: device step count differs from scheduler")
        result.update(provenance=provenance, own_steps={d: census["domains"][d]["own_steps"] for d in ("d01", "d02")})
    except (ValueError, OSError, KeyError, jsonschema.ValidationError) as error:
        result = {"pass": False, "reason": str(error)}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
