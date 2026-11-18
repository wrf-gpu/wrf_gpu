#!/usr/bin/env python3
"""Build / verify the static per-grid z0bare control file (EXPERIMENTAL, not WRF-faithful; CPU only).

  build  --input-dir <case dir with wrfinput_d01..> --out <file.npz> [--evidence <json>]
  verify --controls <file.npz> --input-dir <case dir> [--evidence <json>]

Rule (Q162, z0bgate PREREG): Noah-MP ENERGY bare-ground Z0 = 0.002 m * 10 = 0.02 m on IVGTYP == ISBARREN (16,
MODIFIED_IGBP_MODIS_NOAH) cells of d03..d09; every other cell keeps exactly float32(0.002); d01/d02 carry no
control (default path). The file embeds a grid lock (gpuwrf.surface_params.grid_lock): any consumer — this
verify command and `gpuwrf run --surface-controls` — refuses a run whose wrfinput grid differs.
Outputs are written exclusively (an existing target is refused). Run with JAX_PLATFORMS=cpu.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

DOMAINS = tuple(f"d{i:02d}" for i in range(3, 10))
RULE = {"control": "z0bare", "class_field": "IVGTYP", "class": 16, "factor": 10, "base_m": 0.002,
        "value_m": 0.02, "domains": list(DOMAINS), "default_domains": ["d01", "d02"],
        "status": "EXPERIMENTAL_NOT_WRF_FAITHFUL"}


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cpu():
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        sys.exit("refused: run with JAX_PLATFORMS=cpu")


def expected_field(lu, np):
    return np.where(lu == RULE["class"], np.float64(RULE["base_m"]) * RULE["factor"],
                    np.float64(RULE["base_m"])).astype("f4")


def _exclusive_json(path, value):
    path = Path(path)
    with open(path, "x") as f:
        f.write(json.dumps(value, indent=1, sort_keys=True) + "\n")


def build(input_dir: Path, out: Path, evidence: Path | None) -> dict:
    _cpu()
    import numpy as np
    import netCDF4 as nc
    from gpuwrf._fast_defaults import apply_fast_path_defaults
    apply_fast_path_defaults()
    import jax
    assert jax.devices()[0].platform == "cpu"
    import gpuwrf
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state, build_noahmp_params
    from gpuwrf.physics.noahmp.precision import real_dtype
    from gpuwrf.surface_params import controls as sc
    from gpuwrf.surface_params.grid_lock import make_lock, require_grid_lock

    input_dir = Path(input_dir).resolve()
    out = Path(out)
    if out.exists() or (evidence and Path(evidence).exists()):
        raise SystemExit(f"refused: target exists ({out} / {evidence})")
    if np.dtype(real_dtype()) != np.dtype("float32"):
        raise SystemExit("refused: Noah-MP real dtype is not float32")
    fields, proof = {}, {}
    for dom in DOMAINS:
        f = input_dir / f"wrfinput_{dom}"
        with nc.Dataset(f) as ds:
            lu = np.asarray(ds["LU_INDEX"][0])
            iv = np.asarray(ds["IVGTYP"][0])
            land = np.asarray(ds["LANDMASK"][0]) > 0.5
        if not np.array_equal(lu, iv):
            raise SystemExit(f"refused: {dom} LU_INDEX != IVGTYP")
        state, static, _ = build_noahmp_land_state(input_dir, dom)
        ep, rp, _ = build_noahmp_params(static)
        if int(static.parameters.isbarren) != RULE["class"]:
            raise SystemExit(f"refused: {dom} ISBARREN {int(static.parameters.isbarren)} != 16")
        nl = SimpleNamespace(use_noahmp=True, noahmp_static=static, noahmp_energy_params=ep,
                             noahmp_rad_params=rp, noahmp_options=None)
        field = sc.expand_controls(nl, {"z0bare": {RULE["class"]: RULE["factor"]}}, device=False)["z0bare"]
        exp = expected_field(iv, np)
        if field.dtype != np.dtype("float32") or field.tobytes() != exp.tobytes():
            raise SystemExit(f"refused: {dom} expanded field differs from the rule")
        sc.validate_controls({"z0bare": field}, nl)
        _, bound, _ = sc.apply_surface_controls(static, ep, rp, {"z0bare": field})
        if np.asarray(bound.z0bare).tobytes() != field.tobytes():
            raise SystemExit(f"refused: {dom} bound field differs")
        fields[dom] = {"z0bare": field}
        proof[dom] = {"shape": list(field.shape), "LU16_cells": int((iv == 16).sum()),
                      "LU16_land_cells": int((land & (iv == 16)).sum()), "non_LU16_exact_0.002": True,
                      "LU16_exact_0.02": True, "array_sha256": hashlib.sha256(field.tobytes()).hexdigest(),
                      "controls_digest": sc.controls_digest(fields[dom]), "wrfinput_sha256": sha(f),
                      "real_validate_and_bind": "PASS"}
    lock = make_lock(input_dir, DOMAINS, RULE)
    provenance = {"rule": RULE, "grid_lock": lock, "builder": "scripts/z0bare_grid_controls.py",
                  "builder_sha256": sha(__file__), "gpuwrf_version": getattr(gpuwrf, "__version__", None),
                  "source_dir": str(input_dir),
                  "wrfinput_sha256": {d: sha(input_dir / f"wrfinput_{d}") for d in ("d01", "d02", *DOMAINS)}}
    tmp = out.with_name(out.name + ".building.npz")
    if tmp.exists():
        raise SystemExit(f"refused: stale temporary {tmp}")
    sc.save_controls(tmp, fields, provenance=provenance)
    os.link(tmp, out)  # exclusive publish: fails if out appeared meanwhile
    tmp.unlink()
    loaded, manifest = sc.load_controls(out)
    if set(loaded) != set(DOMAINS) or any(sc.controls_digest(loaded[d]) != proof[d]["controls_digest"] for d in DOMAINS):
        raise SystemExit("refused: readback differs")
    check = require_grid_lock(manifest, input_dir, {f"d{i:02d}" for i in range(1, 10)})
    os.chmod(out, 0o444)
    record = {"schema": "z0bare.grid_controls.build.v1", "status": "PASS", "file": str(out), "sha256": sha(out),
              "grid_lock_digest": lock["digest"], "grid_lock_check": check, "rule": RULE, "domains": proof,
              "source_dir": str(input_dir), "gpu": False, "model_compiles": 0}
    if evidence:
        _exclusive_json(evidence, record)
        os.chmod(evidence, 0o444)
    return record


def verify(controls: Path, input_dir: Path, evidence: Path | None) -> dict:
    _cpu()
    import numpy as np
    import netCDF4 as nc
    from gpuwrf.surface_params import controls as sc
    from gpuwrf.surface_params.grid_lock import require_grid_lock, lock_of

    loaded, manifest = sc.load_controls(controls)
    record = {"schema": "z0bare.grid_controls.verify.v1", "file": str(controls), "sha256": sha(controls),
              "input_dir": str(Path(input_dir).resolve()), "status": "FAIL"}
    try:
        lock = lock_of(manifest)
        if lock is None:
            raise ValueError("control file has no grid lock")
        if lock.get("rule") != RULE or set(loaded) != set(DOMAINS):
            raise ValueError("control file rule/domains differ from the z0bare Nightly rule")
        record["grid_lock"] = require_grid_lock(manifest, input_dir, {f"d{i:02d}" for i in range(1, 10)})
        for dom in DOMAINS:
            with nc.Dataset(Path(input_dir) / f"wrfinput_{dom}") as ds:
                iv = np.asarray(ds["IVGTYP"][0])
            if loaded[dom]["z0bare"].tobytes() != expected_field(iv, np).tobytes():
                raise ValueError(f"{dom}: control values differ from the rule on this grid")
        record["status"] = "PASS"
    except (OSError, ValueError, KeyError) as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
    if evidence:
        _exclusive_json(evidence, record)
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--input-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True); b.add_argument("--evidence", type=Path)
    v = sub.add_parser("verify"); v.add_argument("--controls", type=Path, required=True)
    v.add_argument("--input-dir", type=Path, required=True); v.add_argument("--evidence", type=Path)
    a = p.parse_args(argv)
    r = build(a.input_dir, a.out, a.evidence) if a.cmd == "build" else verify(a.controls, a.input_dir, a.evidence)
    print(json.dumps({k: r[k] for k in ("status", "file", "sha256") if k in r} | ({"reason": r["reason"]} if "reason" in r else {})))
    return 0 if r["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
