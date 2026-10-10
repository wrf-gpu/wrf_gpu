"""Loaders for the v034 NSSL 2-moment oracle (proofs/v034/f2_oracles/nssl_2mom).

* ``load_case(mode, case)``     -> savepoint dict (schema wrf-v034-o1-nssl2mom-column-savepoint-v2)
* ``stage_an(sp, 'S2')``        -> numpy (NA+1, KX) state stack, Fortran species index on axis 0
* ``module_vars(mode)``          -> {name: value or list} dumped after nssl_2mom_init (case 1)
* ``load_gs_dump(mode, case, 'G1')`` -> gs-internal locals (DEV ONLY; raw text from the lane work dir)
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SAVEPOINTS = REPO / "proofs" / "v034" / "f2_oracles" / "nssl_2mom"
ORACLE_WORK = Path(os.environ.get("NSSL_O1_ORACLE_WORK", "<USER_HOME>/wrf_gpu2_lanes/o1-nssl/oracle_v4"))
NA = 18
CASES = tuple(range(1, 18))  # 1-14 + 15 supercooled rain, 16 graupel->hail wet growth, 17 dry-air depletion


@lru_cache(maxsize=None)
def load_case(mode: str, case: int) -> dict:
    prefix = "nssl" if mode == "fp32" else "nssl_fp64"
    return json.loads((SAVEPOINTS / f"{prefix}_case_{case}.json").read_text())


def stage_an(sp: dict, stage: str) -> np.ndarray:
    an = np.asarray(sp["stages"][stage]["an"], dtype=np.float64)  # (NA, KX), Fortran il=1..NA
    out = np.zeros((NA + 1, an.shape[1]), dtype=np.float64)
    out[1:] = an
    return out


def stage_col(sp: dict, stage: str, name: str) -> np.ndarray:
    return np.asarray(sp["stages"][stage][name], dtype=np.float64)


def module_vars(mode: str) -> dict:
    return load_case(mode, 1)["module_vars"]


# ---------------------------------------------------------------- gs-internal dumps (dev only)
_IDX = dict(lv=2, lc=3, lr=4, li=5, ls=6, lh=7, lhl=8, lhab=8, lqmx=30)


def _bounds(dim: str):
    lo, _, hi = dim.partition(":")
    if not hi:
        return 1, int(_IDX.get(lo, lo)) if not lo.isdigit() else int(lo)
    return int(_IDX.get(lo, lo)) if not lo.lstrip('-').isdigit() else int(lo), int(_IDX.get(hi, hi)) if not hi.lstrip('-').isdigit() else int(hi)


@lru_cache(maxsize=None)
def _gs_decls() -> dict:
    out = {}
    for line in (ORACLE_WORK / "gs_vars_dims.txt").read_text().split("\n"):
        if line.strip():
            name, typ, kind, par, dims = line.split()
            out[name] = (typ, kind, dims)
    return out


def _conv(tok: str, typ: str):
    if typ in ("real", "doubleprecision"):
        return float(tok)
    if typ == "integer":
        return int(tok)
    return tok == "T"


@lru_cache(maxsize=None)
def load_gs_dump(mode: str, case: int, tag: str) -> dict:
    """Return {name: value}; arrays dimensioned (ngs, ...) are cut to the ngscnt gathered points
    and species/extra axes become dict keys (Fortran indices): qx -> {il: (ngscnt,)}."""
    decls = _gs_decls()
    path = ORACLE_WORK / f"build_{mode}_instr" / f"case_{case}.txt"
    raw = {}
    pat = re.compile(rf"^{tag}:([A-Za-z0-9_]+)=(.*)$")
    with open(path) as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                if m.group(1) == "kgs" and "kgs" in raw:
                    continue  # keep the explicit kgs(1:ngscnt) line, not the generic 64-entry local dump
                raw[m.group(1)] = m.group(2).split()
    n = int(raw.pop("ngscnt")[0])
    out = {"ngscnt": n, "kgs": np.array([int(x) for x in raw.pop("kgs")])}
    for name, toks in raw.items():
        typ, kind, dims = decls[name]
        vals = [_conv(t, typ) for t in toks]
        if kind == "scalar":
            out[name] = vals[0]
            continue
        dl = dims.split(",")
        if dl[0] != "ngs":
            out[name] = np.array(vals)
            continue
        extra = [_bounds(d) for d in dl[1:]]
        shape = [64] + [hi - lo + 1 for lo, hi in extra]
        arr = np.array(vals).reshape(shape, order="F")
        if not extra:
            out[name] = arr[:n]
        elif len(extra) == 1:
            lo, hi = extra[0]
            out[name] = {i: arr[:n, i - lo] for i in range(lo, hi + 1)}
        else:
            (lo1, hi1), (lo2, hi2) = extra
            out[name] = {(i, j): arr[:n, i - lo1, j - lo2] for i in range(lo1, hi1 + 1) for j in range(lo2, hi2 + 1)}
    return out
