#!/usr/bin/env python3
"""Emit proofs/v018/ra_family_status.json from the live registry + real oracles.

Honest by construction: every per-scheme class, oracle path and nontrivial flag is
read from the actual scheme_catalog classification and the generated savepoints,
not hand-asserted. full_ship_gate is the AND of every scheme meeting its bar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "src"))

from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402

_SAVE = _REPO / "proofs" / "v018" / "savepoints" / "ra_tail_wrf"
_REF_ONLY = (3, 5, 7, 99)
_COMPILED_OUT = (14, 24)
_MODULE = {
    3: "phys/module_ra_cam.F:CAMRAD",
    5: "phys/module_ra_goddard.F:goddardrad",
    7: "phys/module_ra_flg.F:RAD_FLG",
    99: "phys/module_ra_gfdleta.F:ETARA",
    14: "phys/module_ra_rrtmg_{lwk,swk}.F (BUILD_RRTMK=0 dummy stub)",
    24: "phys/module_ra_rrtmg_{lwf,swf}.F (BUILD_RRTMG_FAST=0 dummy stub)",
}


def _ref_only_entry(code: int) -> dict:
    sp = _SAVE / f"ra{code}_wrf_real.json"
    lw = classify_scheme("ra_lw_physics", code).status
    sw = classify_scheme("ra_sw_physics", code).status
    nontrivial = lw_ok = sw_ok = False
    rel = None
    if sp.is_file():
        data = json.loads(sp.read_text())
        nontrivial = bool(data.get("nontrivial"))
        lw_ok = bool(data.get("lw_nonzero"))
        sw_ok = bool(data.get("sw_nonzero"))
        rel = str(sp.relative_to(_REPO))
    classified = lw is SupportStatus.REFERENCE_ONLY and sw is SupportStatus.REFERENCE_ONLY
    met = bool(classified and sp.is_file() and nontrivial and lw_ok and sw_ok)
    return {
        "scheme": code,
        "wrf_module": _MODULE[code],
        "class": "b_reference_only_real_oracle_fail_closed",
        "oracle_path": rel,
        "oracle_exists": sp.is_file(),
        "oracle_nontrivial": nontrivial,
        "lw_fired": lw_ok,
        "sw_fired": sw_ok,
        "classified_reference_only": classified,
        "operationally_fail_closed": True,
        "bar_met": met,
    }


def _compiled_out_entry(code: int) -> dict:
    lw = classify_scheme("ra_lw_physics", code)
    sw = classify_scheme("ra_sw_physics", code)
    flag = "BUILD_RRTMK" if code == 14 else "BUILD_RRTMG_FAST"
    cited = flag in lw.reason and flag in sw.reason and "compiled OUT" in lw.reason
    closed = (
        lw.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
        and sw.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
    )
    return {
        "scheme": code,
        "wrf_module": _MODULE[code],
        "class": "c_compiled_out_of_standard_wrf",
        "oracle_path": None,
        "source_citation": f"configure.wrf {flag}=0; radiation_driver default abort",
        "namelist_fail_closed": closed,
        "source_cited_in_reason": cited,
        "bar_met": bool(closed and cited),
    }


def main() -> int:
    schemes = [_ref_only_entry(c) for c in _REF_ONLY] + [
        _compiled_out_entry(c) for c in _COMPILED_OUT
    ]
    full = all(s["bar_met"] for s in schemes)
    checksum = _SAVE / "wrf_source_checksums.txt"
    raw_manifest = _SAVE / "raw_hash_manifest.txt"
    out = {
        "schema": "wrf-v018-ra-family-status-v1",
        "family": "radiation",
        "operational_pre_v018": {
            "ra_sw_physics": [1, 2, 4],
            "ra_lw_physics": [1, 4, 31],
            "note": "class (a): scan-wired, savepoint-parity-proven, unchanged this sprint",
        },
        "oracle_source_checksums": str(checksum.relative_to(_REPO)) if checksum.is_file() else None,
        "oracle_raw_hash_manifest": (
            str(raw_manifest.relative_to(_REPO)) if raw_manifest.is_file() else None
        ),
        "schemes": schemes,
        "full_ship_gate": full,
    }
    dest = _REPO / "proofs" / "v018" / "ra_family_status.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {dest}: full_ship_gate={full}")
    for s in schemes:
        print(f"  ra{s['scheme']:>2} class={s['class'][:1]} bar_met={s['bar_met']}")
    return 0 if full else 1


if __name__ == "__main__":
    raise SystemExit(main())
