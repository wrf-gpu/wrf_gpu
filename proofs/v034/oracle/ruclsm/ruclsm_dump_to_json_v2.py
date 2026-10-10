#!/usr/bin/env python3
"""Parse the flat v0.34 RUC LSM oracle dump (ruclsm_oracle_v2.f90) into JSON.

Line grammar (one value per line):

* ``NAME=value``               global scalar (int, float or string)
* ``REGIME_NAME[r]=string``    regime label
* ``IN_NAME[r]=value``         regime input scalar (state before the first call)
* ``IN_NAME[r][k]=value``      regime input profile (soil level k, or land/soil
                               category k for LANDUSEF/SOILCTOP)
* ``NAME[r][s]=value``         value after LSMRUC call s (s = 1..NSTEPS)
* ``NAME[r][s][k]=value``      soil profile after call s (k = 1..NSL[r])
* ``RUN_RC[r]=int``            exit code of the per-regime process (appended by
                               the build script; 0 = all NSTEPS calls done)

The input is the concatenation of one dump per regime (each regime runs in
its own process); repeated global header lines must agree.  Any other line
(WRF table-reader chatter, WRF fatal messages) is kept verbatim in the
regime's ``messages`` list.  A regime with RUN_RC != 0 gets
``status = "wrf_fatal"`` and only its completed steps.

Reals are written with ES24.16E3; the parser also accepts gfortran's
exponent form without 'E' (``1.0-321``).  Integers (I0) stay ints.

Output::

  {"scalars": {...},
   "regimes": [{"index": r, "name": str, "status": "ok" | "wrf_fatal",
                "warning": str | null,
                "run_rc": int, "messages": [str],
                "inputs": {VAR: value | [values]},
                "steps":  [{VAR: value | [values]}, ...]}, ...]}

Optional third argument: the dump of the ``-finit-integer=2`` twin build.
WRF's sfctmp local INTEGER ``ilnb`` is read uninitialized for thin snow
(snhei < snth; module_sf_ruclsm.F:5716 / :4410).  The deliverable build pins
it with ``-finit-integer=0``; every step whose dumped state differs in the
twin is listed in ``ilnb_sensitive_steps`` (1-based) of its regime.

Usage: ruclsm_dump_to_json_v2.py <dump.txt> <out.json> [<ilnb2_twin_dump.txt>]
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

NAME = r"([A-Z][A-Z0-9_]*)"
REGIME_RE = re.compile(r"^REGIME_NAME\[(\d+)\]=(.*)$")
RC_RE = re.compile(r"^RUN_RC\[(\d+)\]=\s*([+-]?\d+)\s*$")
IN_PROF_RE = re.compile(rf"^IN_{NAME}\[(\d+)\]\[(\d+)\]=(.+)$")
IN_SCAL_RE = re.compile(rf"^IN_{NAME}\[(\d+)\]=(.+)$")
ST_PROF_RE = re.compile(rf"^{NAME}\[(\d+)\]\[(\d+)\]\[(\d+)\]=(.+)$")
ST_SCAL_RE = re.compile(rf"^{NAME}\[(\d+)\]\[(\d+)\]=(.+)$")
GLOBAL_RE = re.compile(rf"^{NAME}=(.*)$")
INT_RE = re.compile(r"^[+-]?\d+$")
# gfortran drops the 'E' for 3-digit exponents when no Ee field is given.
MISSING_E_RE = re.compile(r"^([+-]?\d*\.\d*)([+-]\d{3})$")


def parse_number(token: str) -> int | float:
    tok = token.strip()
    if INT_RE.match(tok):
        return int(tok)
    try:
        return float(tok)
    except ValueError:
        m = MISSING_E_RE.match(tok)
        if m:
            return float(f"{m.group(1)}E{m.group(2)}")
        raise


def parse_global(token: str) -> int | float | str:
    tok = token.strip()
    try:
        return parse_number(tok)
    except ValueError:
        return tok


def _ordered(d: dict[int, object], what: str) -> list:
    keys = sorted(d)
    if keys != list(range(1, len(keys) + 1)):
        raise ValueError(f"non-contiguous indices for {what}: {keys}")
    return [d[k] for k in keys]


STEP_KEY_RE = re.compile(rf"^{NAME}\[(\d+)\]\[(\d+)\](?:\[\d+\])?=")


def raw_step_values(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if STEP_KEY_RE.match(line) and not line.startswith("IN_"):
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v.strip()
    return out


def sensitive_steps(ref_path: str, twin_path: str) -> dict[int, list[int]]:
    """(regime -> steps) whose dumped state differs between ref and twin."""
    a, b = raw_step_values(ref_path), raw_step_values(twin_path)
    hits: dict[int, set[int]] = {}
    for k in set(a) | set(b):
        if a.get(k) != b.get(k):
            m = STEP_KEY_RE.match(k + "=")
            hits.setdefault(int(m.group(2)), set()).add(int(m.group(3)))
    return {r: sorted(v) for r, v in hits.items()}


def main(infile: str, outfile: str, twin: str | None = None) -> None:
    scalars: dict[str, object] = {}
    names: dict[int, str] = {}
    order: list[int] = []
    in_s: dict[int, dict[str, object]] = {}
    in_p: dict[int, dict[str, dict[int, object]]] = {}
    st_s: dict[int, dict[int, dict[str, object]]] = {}
    st_p: dict[int, dict[int, dict[str, dict[int, object]]]] = {}
    nonfinite: list[str] = []
    rcs: dict[int, int] = {}
    msgs: dict[int, list[str]] = {}
    current = 0

    def note(line: str, v: object) -> None:
        if isinstance(v, float) and not math.isfinite(v):
            nonfinite.append(line)

    with Path(infile).open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            m = REGIME_RE.match(line)
            if m:
                r = int(m.group(1))
                names[r] = m.group(2).strip()
                if r in order:
                    raise ValueError(f"regime {r} appears twice")
                order.append(r)
                current = r
                continue
            m = RC_RE.match(line)
            if m:
                rcs[int(m.group(1))] = int(m.group(2))
                current = 0
                continue
            m = IN_PROF_RE.match(line)
            if m:
                v = parse_number(m.group(4)); note(line, v)
                in_p.setdefault(int(m.group(2)), {}).setdefault(m.group(1), {})[int(m.group(3))] = v
                continue
            m = IN_SCAL_RE.match(line)
            if m:
                v = parse_number(m.group(3)); note(line, v)
                in_s.setdefault(int(m.group(2)), {})[m.group(1)] = v
                continue
            m = ST_PROF_RE.match(line)
            if m:
                v = parse_number(m.group(5)); note(line, v)
                r, s, k = int(m.group(2)), int(m.group(3)), int(m.group(4))
                st_p.setdefault(r, {}).setdefault(s, {}).setdefault(m.group(1), {})[k] = v
                continue
            m = ST_SCAL_RE.match(line)
            if m:
                v = parse_number(m.group(4)); note(line, v)
                r, s = int(m.group(2)), int(m.group(3))
                st_s.setdefault(r, {}).setdefault(s, {})[m.group(1)] = v
                continue
            m = GLOBAL_RE.match(line)
            if m:
                val = parse_global(m.group(2))
                if m.group(1) in scalars and scalars[m.group(1)] != val:
                    raise ValueError(f"global {m.group(1)} differs between processes")
                scalars[m.group(1)] = val
                current = 0
                continue
            # WRF chatter / fatal diagnostics: keep with the regime that emitted it.
            msgs.setdefault(current, []).append(line.rstrip())

    nsteps = int(scalars["NSTEPS"])
    sens = sensitive_steps(infile, twin) if twin else None
    regimes = []
    step_keys_ref: set[str] | None = None
    for r in order:
        inputs: dict[str, object] = dict(in_s.get(r, {}))
        for var, byk in in_p.get(r, {}).items():
            inputs[var] = _ordered(byk, f"IN_{var}[{r}]")
        nsl = int(inputs["NSL"])
        for var in ("ZS", "TSO", "SOILMOIS", "SH2O", "SMFR3D", "KEEPFR3DFLAG"):
            if len(inputs[var]) != nsl:
                raise ValueError(f"regime {r}: IN_{var} has {len(inputs[var])} != NSL {nsl}")
        steps = []
        rc = rcs.get(r)
        if rc is None:
            raise ValueError(f"regime {r}: missing RUN_RC line")
        done = sorted(st_s.get(r, {}))
        if done != list(range(1, len(done) + 1)):
            raise ValueError(f"regime {r}: non-contiguous steps {done}")
        if rc == 0 and len(done) != nsteps:
            raise ValueError(f"regime {r}: rc 0 but steps {done} != 1..{nsteps}")
        for s in done:
            rec: dict[str, object] = dict(st_s[r][s])
            for var, byk in st_p[r][s].items():
                rec[var] = _ordered(byk, f"{var}[{r}][{s}]")
                if len(rec[var]) != nsl:
                    raise ValueError(f"regime {r} step {s}: {var} length != NSL")
            keys = set(rec)
            if step_keys_ref is None:
                step_keys_ref = keys
            elif keys != step_keys_ref:
                raise ValueError(f"regime {r} step {s}: key set differs: {keys ^ step_keys_ref}")
            steps.append(rec)
        warning = None
        if inputs.get("ZSHALF_NROOT1_OOB") == 1:
            warning = ("LSMRUC reads zshalf(nroot+1) with nroot = NSL (forest iforest<=2 on a grid "
                       "whose zs(nsl-1) < 1.1 m): out-of-bounds read = undefined behaviour in WRF "
                       "itself; values (and the step of the fatal) depend on memory layout. "
                       "NOT a valid oracle.")
        ilnb_steps = None if sens is None else sens.get(r, [])
        regimes.append({"index": r, "name": names[r], "warning": warning,
                        "ilnb_sensitive_steps": ilnb_steps,
                        "status": "ok" if rc == 0 else "wrf_fatal", "run_rc": rc,
                        "steps_completed": len(steps), "messages": msgs.get(r, []),
                        "inputs": inputs, "steps": steps})

    scalars["NREG_DUMPED"] = len(regimes)
    scalars["NONFINITE_COUNT"] = len(nonfinite)
    scalars["NREG_WRF_FATAL"] = sum(1 for g in regimes if g["status"] != "ok")
    scalars["ILNB_CONVENTION"] = (
        "uninitialized sfctmp ilnb pinned to 0 (-finit-integer=0): thin-snow TSNAV uses the "
        "one-layer formula; steps differing under -finit-integer=2 are listed per regime")
    scalars["ILNB_TWIN_COMPARED"] = bool(twin)
    if int(scalars["NREG"]) != len(regimes):
        print(f"WARNING: NREG {scalars['NREG']} != dumped {len(regimes)}", file=sys.stderr)
    out = {"scalars": scalars, "regimes": regimes}

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {path}: regimes={len(regimes)} steps/regime={nsteps} "
          f"step_vars={len(step_keys_ref or ())} nonfinite={len(nonfinite)} "
          f"wrf_fatal={[g['index'] for g in regimes if g['status'] != 'ok']} "
          f"ilnb_sensitive={ {g['index']: g['ilnb_sensitive_steps'] for g in regimes if g['ilnb_sensitive_steps']} }")
    for line in nonfinite[:20]:
        print(f"NONFINITE: {line}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
