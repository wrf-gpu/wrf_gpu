"""Seeded boundary-layer theta/QV initial-condition noise in private wrfinput copies (v0.5.0 item 0.2a).

WRF-EXTERNAL and therefore disclosed: original WRF has no such option. The recipe only edits input files, so
the GPU model program is untouched and the SAME perturbed inputs can be run by CPU-WRF for a matched reference.

Per member m and domain d, NumPy PCG64 seeded with ``SeedSequence([seed, m, d])`` draws two standard-normal
fields on the mass grid (theta first, then QV), clipped at +-``clip_sigma``. With the height of mass levels above
ground z (from the file's own PH+PHB and HGT) the vertical weight is a(z) = 1 for z <= full_m, a linear taper to 0
at top_m, 0 above. Cells inside ``spec_bdy_width`` of the domain edge are never changed (WRF relaxes/forces them).

  dry theta  T'    = T + a * theta_std_k * n_theta
  vapour     QV'   = QV * (1 + a * qv_rel_std * n_qv)
  moist      THM'  = THM + [(T'+300)(1+Rv/Rd QV') - (T+300)(1+Rv/Rd QV)]   when USE_THETA_M=1, else THM' = T'

WRF reads THM (Registry ``t``, the model's theta_m) and the port reads dry T, so both are kept consistent; every
unperturbed cell keeps its original bytes. Dry mass, PH, P and all winds are unchanged: the result is NOT balanced
(no MU/PH rebalance), which is why the amplitudes are meant to be small and the recipe is labelled research-only.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil

import netCDF4
import numpy as np

from gpuwrf.ens.members import identity
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

SCHEMA = "gpuwrf.perturb.thermo.v1"
_KEYS = {"schema", "seed", "theta_std_k", "qv_rel_std", "full_m", "top_m", "clip_sigma", "domains"}
_G = 9.81           # WRF module_model_constants g
_RVOVRD = 461.6 / 287.0
_XYZ = ("Time", "bottom_top", "south_north", "west_east")
RECIPE = ("PCG64(SeedSequence([seed, member, domain])) standard normals theta then QV, clipped at clip_sigma; "
          "a(z)=1 below full_m, linear to 0 at top_m (z above ground of mass levels from PH+PHB, HGT); "
          "outside spec_bdy_width only; T+=a*theta_std_k*n, QV*=1+a*qv_rel_std*n, THM by the real.exe "
          "theta_m increment (USE_THETA_M=1) else THM=T")


def load_config(path):
    """Strict JSON recipe: every key is required except ``domains`` (default all) and ``clip_sigma`` (3)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) - _KEYS:
        raise ValueError(f"thermo config: unknown keys {sorted(set(data) - _KEYS) if isinstance(data, dict) else data}")
    if data.get("schema") != SCHEMA:
        raise ValueError(f"thermo config: schema must be {SCHEMA}")
    seed = data.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
        raise ValueError("thermo config: seed must be an integer in [0, 2**63)")
    def number(key, low, high, default=None):
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"thermo config: {key} must be finite in [{low}, {high}]")
        return float(value)
    cfg = {"schema": SCHEMA, "seed": int(seed),
           "theta_std_k": number("theta_std_k", 0., 2.), "qv_rel_std": number("qv_rel_std", 0., .2),
           "full_m": number("full_m", 0., 5000.), "top_m": number("top_m", 0., 5000.),
           "clip_sigma": number("clip_sigma", 1., 4., 3.)}
    if cfg["theta_std_k"] == 0 and cfg["qv_rel_std"] == 0:
        raise ValueError("thermo config: theta_std_k and qv_rel_std are both zero (nothing to perturb)")
    if not cfg["top_m"] > cfg["full_m"]:
        raise ValueError("thermo config: top_m must exceed full_m")
    domains = data.get("domains")
    if domains is not None and (not isinstance(domains, list) or not domains or
                                any(isinstance(d, bool) or not isinstance(d, Integral) or not 1 <= d <= 99 for d in domains)
                                or len(set(domains)) != len(domains)):
        raise ValueError("thermo config: domains must be a list of distinct domain numbers")
    cfg["domains"] = None if domains is None else sorted(int(d) for d in domains)
    return cfg


def vertical_weight(z, full_m, top_m):
    return np.clip((top_m - z) / (top_m - full_m), 0., 1.)


def noise(seed, member, domain, shape, clip_sigma):
    """Two clipped standard-normal fields (theta, QV) for one member/domain, float64."""
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, member, domain])))
    theta = generator.standard_normal(shape)
    qv = generator.standard_normal(shape)
    return np.clip(theta, -clip_sigma, clip_sigma), np.clip(qv, -clip_sigma, clip_sigma)


def _read(dataset, name, dims):
    if name not in dataset.variables:
        raise ValueError(f"missing {name}")
    variable = dataset[name]
    if variable.dimensions != dims or variable.shape[0] != 1:
        raise ValueError(f"{name}: expected one-time WRF layout {dims}")
    if np.dtype(variable.dtype) != np.dtype("float32") or any(
            a in variable.ncattrs() for a in ("scale_factor", "add_offset")):
        raise ValueError(f"{name}: expected unpacked WRF REAL float32")
    value = variable[:]
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError(f"{name}: masked values")
    value = np.asarray(value)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name}: empty or NaN/Inf")
    # Callers read with auto-masking OFF (byte-exact copies), so a fill/missing value arrives as a large
    # finite number: refuse the declared and the NetCDF default float fill explicitly (critic finding 7).
    fills = {float(np.float32(netCDF4.default_fillvals["f4"]))}
    for attribute in ("_FillValue", "missing_value"):
        if attribute in variable.ncattrs():
            fills.update(float(np.float32(v)) for v in np.atleast_1d(variable.getncattr(attribute)))
    if np.isin(value, np.asarray(sorted(fills), np.float32)).any():
        raise ValueError(f"{name}: fill/missing values in the input")
    return value


def perturb_file(path, cfg, *, member, domain, bdy_width):
    """Edit one private wrfinput copy in place; return change statistics."""
    with Dataset(path, "r+") as ds:
        ds.set_auto_maskandscale(False)
        t = _read(ds, "T", _XYZ)
        qv = _read(ds, "QVAPOR", _XYZ)
        thm = _read(ds, "THM", _XYZ)
        ph = _read(ds, "PH", ("Time", "bottom_top_stag", "south_north", "west_east"))
        phb = _read(ds, "PHB", ("Time", "bottom_top_stag", "south_north", "west_east"))
        hgt = _read(ds, "HGT", ("Time", "south_north", "west_east"))
        theta_m = int(getattr(ds, "USE_THETA_M", -1))
        if theta_m not in (0, 1):
            raise ValueError(f"{path.name}: global attribute USE_THETA_M must be 0 or 1")
        if np.any(qv < 0):
            raise ValueError(f"{path.name}: negative source QVAPOR")
        total = (ph.astype(np.float64) + phb) / _G
        z = 0.5 * (total[:, :-1] + total[:, 1:]) - hgt[:, None].astype(np.float64)
        weight = vertical_weight(z, cfg["full_m"], cfg["top_m"])
        _, nz, ny, nx = t.shape
        if min(nx, ny) <= 2 * bdy_width:
            raise ValueError(f"{path.name}: domain has no interior inside spec_bdy_width={bdy_width}")
        interior = np.zeros((ny, nx), bool)
        interior[bdy_width:ny - bdy_width, bdy_width:nx - bdy_width] = True
        weight = np.where(interior[None, None], weight, 0.)
        n_theta, n_qv = noise(cfg["seed"], member, domain, t.shape, cfg["clip_sigma"])
        touched = weight > 0
        t_new = t.copy()
        qv_new = qv.copy()
        t_new[touched] = (t.astype(np.float64) + weight * cfg["theta_std_k"] * n_theta)[touched].astype(np.float32)
        qv_new[touched] = (qv.astype(np.float64) * (1 + weight * cfg["qv_rel_std"] * n_qv))[touched].astype(np.float32)
        if np.any(qv_new < 0) or not np.isfinite(t_new).all() or not np.isfinite(qv_new).all():
            raise ValueError(f"{path.name}: perturbation produced negative/nonfinite values")
        thm_new = thm.copy()
        if theta_m:
            # Increment of WRF real.exe's REAL conversion (t+T0)*(1+rvovrd*qv)-T0, applied to THM itself so that
            # an unchanged cell keeps its stored THM bytes exactly.
            f32 = np.float32
            def moist(tt, qq):
                return (tt + f32(300)) * (f32(1) + f32(_RVOVRD) * qq) - f32(300)
            increment = moist(t_new, qv_new).astype(np.float64) - moist(t, qv)
            thm_new[touched] = (thm.astype(np.float64) + increment)[touched].astype(np.float32)
        else:
            thm_new[touched] = t_new[touched]
        if not np.isfinite(thm_new).all():
            raise ValueError(f"{path.name}: computed THM is nonfinite")
        rows = {}
        for name, before, after in (("T", t, t_new), ("QVAPOR", qv, qv_new), ("THM", thm, thm_new)):
            changed = before.view(np.uint32) != after.view(np.uint32)
            if np.any(changed):
                ds[name][:] = after
            rows[name] = {"cells": int(before.size), "changed": int(changed.sum()),
                          "max_abs_change": float(np.max(np.abs(after.astype(np.float64) - before)))}
        ds.sync()
        levels = np.flatnonzero(np.any(touched, axis=(0, 2, 3)))
        rows["levels_touched"] = [int(levels.min()), int(levels.max())] if levels.size else None
        rows["use_theta_m"] = theta_m
        return rows


def prepare(case, output, *, members, config):
    """New member case directories with perturbed private wrfinput copies; every other input is linked."""
    cfg = load_config(config)
    if isinstance(members, bool) or not isinstance(members, Integral) or not 2 <= members <= 1000:
        raise ValueError("members must be an integer in [2,1000]")
    case = Path(case).resolve(strict=True)
    output = Path(output).absolute()
    if not case.is_dir():
        raise ValueError("case must be a directory")
    if output.exists() or output.is_symlink():
        raise ValueError("output exists; refusing overwrite")
    namelist = parse_namelist(case / "namelist.input")
    count = namelist.get("domains", {}).get("max_dom", 1)
    count = count[0] if isinstance(count, list) else count
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 99:
        raise ValueError("namelist max_dom must be an integer in [1,99]")
    width = namelist.get("bdy_control", {}).get("spec_bdy_width", 5)
    width = width[0] if isinstance(width, list) else width
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 50:
        raise ValueError("spec_bdy_width must be an integer in [1,50]")
    domains = cfg["domains"] or list(range(1, count + 1))
    if max(domains) > count:
        raise ValueError(f"thermo config names domain {max(domains)} beyond max_dom={count}")
    edited = [case / f"wrfinput_d{d:02d}" for d in domains]
    for path in edited:
        if not path.is_file():
            raise ValueError(f"missing {path.name}")
    # Same inputs as the wind recipe: namelist, lateral and lower boundary files, unedited nests.
    linked = [case / "namelist.input", *sorted(case.glob("wrfbdy_d[0-9][0-9]")),
              *sorted(case.glob("wrflowinp_d[0-9][0-9]")),
              *(case / f"wrfinput_d{d:02d}" for d in range(1, count + 1) if d not in domains)]
    for path in linked[1:]:
        if not path.is_file():
            raise ValueError(f"missing {path.name}")
    source_ids = {p.name: identity(p) for p in (*edited, *linked)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        rows = []
        for member in range(members):
            name = f"member-{member:03d}"
            dest = output / name
            dest.mkdir()
            changes = {}
            for number, path in zip(domains, edited):
                target = dest / path.name
                shutil.copyfile(path, target)
                if identity(target)["sha256"] != source_ids[path.name]["sha256"]:
                    raise ValueError(f"source changed while copying {path.name}")
                changes[path.name] = perturb_file(target, cfg, member=member, domain=number, bdy_width=width)
            for path in linked:
                (dest / path.name).symlink_to(path.resolve(strict=True))
            if not any(row[v]["changed"] for row in changes.values() for v in ("T", "QVAPOR")):
                # e.g. a perturbation layer below the first mass level, or only boundary-zone cells:
                # a "member" identical to the source is no ensemble (critic finding 6).
                raise ValueError(f"{name}: the recipe changes no T/QVAPOR cell (perturbation layer "
                                 f"{cfg['full_m']}-{cfg['top_m']} m AGL inside spec_bdy_width={width}); refusing")
            rows.append({"id": name, "case": str(dest.resolve()), "changes": changes,
                         "prepared": {p.name: identity(dest / p.name) for p in edited}})
        if any(identity(case / n)["sha256"] != item["sha256"] for n, item in source_ids.items()):
            raise ValueError("source case changed during preparation")
        result = {"schema": SCHEMA, "recipe": RECIPE, "config": cfg,
                  "config_file": identity(config), "spec_bdy_width": width, "domains": domains,
                  "source": source_ids, "members": rows,
                  "disclosure": "WRF-external seeded IC noise; unbalanced (no MU/PH rebalance); research use only. "
                                "CPU-WRF can run the identical member inputs for a matched reference."}
        (output / "perturb_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        shutil.rmtree(output)  # exclusively created above; inputs are never removed
        raise
