"""Module constants of WRF ``module_mp_nssl_2mom.F`` after ``nssl_2mom_init`` (default mp=18).

Values are frozen from the PRISTINE init (``data/init_constants.json``, built by
``proofs/v034/oracle/nssl2mom/make_constants_asset.py``) for both the WRF REAL=fp32 build and the
``-fdefault-real-8`` fp64 oracle build.  Arrays keep FORTRAN indexing: an array declared
``x(lc:lqmx)`` becomes a numpy array of length ``lqmx + 1`` with ``x[il]`` == Fortran ``x(il)``
(entries below the lower bound are zero).  REAL values carry the build's REAL dtype
(np.float32 for fp32), DOUBLE PRECISION values np.float64.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ASSET = Path(__file__).resolve().parent / "data" / "init_constants.json"


def _bound(tok: str, ns: dict) -> int:
    tok = tok.strip()
    try:
        return int(tok)
    except ValueError:
        pass
    sign = -1 if tok.startswith("-") else 1
    return sign * int(ns[tok.lstrip("-")])


@lru_cache(maxsize=None)
def get_constants(mode: str) -> SimpleNamespace:
    """Return the module constants of one build (``mode`` in {"fp32", "fp64"})."""
    raw = json.loads(_ASSET.read_text())
    vals = raw[mode]
    decls = raw["decls"]
    R = np.float32 if mode == "fp32" else np.float64
    ns: dict = {}
    # scalars first (integer bounds such as lc, lqmx, ngm0 are scalars)
    for name, v in vals.items():
        typ, kind, _dims = decls[name]
        if kind != "scalar":
            continue
        x = v[0]
        if typ == "real":
            ns[name] = R(x)
        elif typ == "doubleprecision":
            ns[name] = np.float64(x)
        elif typ == "integer":
            ns[name] = int(x)
        else:
            ns[name] = bool(x)
    for name, v in vals.items():
        typ, kind, dims = decls[name]
        if kind == "scalar":
            continue
        bounds = []
        for d in dims.split(","):
            lo, sep, hi = d.partition(":")
            bounds.append((_bound(lo, ns), _bound(hi, ns)) if sep else (1, _bound(lo, ns)))
        shape = [hi - lo + 1 for lo, hi in bounds]
        dt = {"real": R, "doubleprecision": np.float64, "integer": np.int64}.get(typ, bool)
        arr = np.array(v, dtype=np.float64 if dt is not bool else bool).reshape(shape, order="F")
        full = np.zeros([hi + 1 for lo, hi in bounds], dtype=dt)
        full[tuple(slice(lo, hi + 1) for lo, hi in bounds)] = arr.astype(dt)
        ns[name] = full
    ns["mode"] = mode
    ns["R"] = R
    return SimpleNamespace(**ns)
