"""Build REF_PACKAGE_V2: nested &stoch CPU-WRF reference namelists for ALISIOS, rand_perturb OFF (critic FIX 1:
pristine WRF 4.7.1 dmpar SIGSEGVs in UPDATE_STOCH when RAND starts before tiles are set, start_em.F:330/992,
module_stoch.F:1176-1183). Lane v50-stoch, v0.5.0 item 0.2b.

Usage (CPU only):
  JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/v50_stoch/ref_package_v2.py --output LANE/REF_PACKAGE_V2

Reads the D15 case namelist READ-ONLY, writes member namelists that differ from it only in &time_control
run length (24 h) and a new &stoch group, the exact diffs, the WRF seed words, and the port's own pre-run
admission verdicts (namelist + nested + stochastic + spectral). Never touches the case directory.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

CASE = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
HOURS = 24
SCHEMES = ("skebs", "sppt", "spp")
MEMBERS = 2  # nens = 1 (reference run) and nens = 2 (optional second member)


def main():
    from gpuwrf.ens.presets import (_set_keys, build_members, check_member_config, parse_start,
                                    wrf_seed_words)
    from gpuwrf.io.namelist_check import refuse_unqualified_nested_options, validate_operational_namelist
    from gpuwrf.io.namelist_parser import parse_namelist
    from gpuwrf.io.stochastic_config import (SEED_DEFAULTS, SEED_KEYS, bind_stochastic, d01_children, stochastic_refusals,
                                             with_executed_d01_children)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = CASE / "namelist.input"
    base_text = source.read_text(encoding="utf-8")
    base = parse_namelist(source)
    max_dom = int(base["domains"]["max_dom"])
    start = parse_start(base)
    end = start + timedelta(hours=HOURS)
    timed = _set_keys(base_text, "time_control", {
        "run_days": 0, "run_hours": HOURS, "run_minutes": 0, "run_seconds": 0,
        "end_year": [end.year]*max_dom, "end_month": [end.month]*max_dom,
        "end_day": [end.day]*max_dom, "end_hour": [end.hour]*max_dom})
    timed_cfg = parse_namelist(_write(out / ".base_24h.namelist.input", timed))
    built = build_members("stochastic", timed, timed_cfg, members=MEMBERS, schemes=SCHEMES, start=start)
    prediction = {}
    receipt = {"case": str(source), "case_namelist_sha256": _sha(source.read_bytes()), "hours": HOURS,
               "start": start.isoformat(), "end": end.isoformat(), "members": []}
    for index, (text, info) in enumerate(built):
        # rand_perturb explicitly OFF on every domain (V2). Seeds written explicitly (= Registry defaults)
        # so the reference never depends on a build's Registry.
        text = _set_keys(text, "stoch", {**info["stoch"], "rand_perturb": [0]*max_dom,
                                         **dict(zip(SEED_KEYS, SEED_DEFAULTS))})
        name = f"namelist.input.nens{info['nens']}"
        path = _write(out / name, text)
        member = parse_namelist(path)
        check_member_config("stochastic", timed_cfg, member, SCHEMES)
        validate_operational_namelist(member)
        refuse_unqualified_nested_options(member, max_dom)
        refusals = stochastic_refusals(member)
        assert not refusals, refusals
        configs = {f"d{i:02d}": bind_stochastic(member, f"d{i:02d}") for i in range(1, max_dom+1)}
        root = configs["d01"]
        seeds = {key: list(wrf_seed_words(int(member["stoch"].get(key, default)), info["nens"], start))
                 for key, default in zip(SEED_KEYS, SEED_DEFAULTS)}
        diff = "".join(difflib.unified_diff(base_text.splitlines(True), text.splitlines(True),
                                            "case/namelist.input", name))
        (out / f"{name}.diff").write_text(diff)
        receipt["members"].append({
            "namelist": name, "sha256": _sha(path.read_bytes()), "nens": info["nens"],
            "stoch": member["stoch"], "wrf_seed_words": seeds,
            "admission": {"validate_operational_namelist": "PASS", "refuse_unqualified_nested_options": "PASS",
                          "stochastic_refusals": refusals, "spectral_check_d01": "PASS",
                          "preset_only_changes_stoch": "PASS"},
            "bound": {d: None if c is None else {"root": c.root, "produce": list(c.produce),
                                                 "apply_pbl": c.apply_pbl,
                                                 # namelist bind never counts nests (FIX1 finding 3); the nested
                                                 # loader sets the executed-tree count; it only touches the RAND
                                                 # stream, which V2 disables (no effect on this reference).
                                                 "nest_init_rounds_namelist_bind": c.nest_init_rounds,
                                                 "nest_init_rounds_executed_tree": with_executed_d01_children(
                                                     c, d, tuple(f"d{i:02d}" for i in d01_children(member))
                                                     if d == "d01" else ()).nest_init_rounds,
                                                 "rand_stream_active": c.patterns[3] is not None,
                                                 "patterns": [None if p is None else [p.variable, p.vertical, p.stddev,
                                                              p.cutoff, p.lengthscale, p.timescale]
                                                              for p in c.patterns]}
                      for d, c in configs.items()},
        })
        assert not configs["d02"].root and not configs["d03"].root
        assert all(c is None or "rand_perturb" not in str(c.produce).lower() for c in configs.values()), root.produce
        # t0 seed prediction: d01 words from the WRF rand_seed recipe; RAND stays 0 (switch off, Registry
        # initial value); every child seed array 0 (ISEEDARR is allocated with size 0 on children and written 0).
        words = {"iseed_skebs": "ISEEDARR_SKEBS", "iseed_sppt": "ISEEDARR_SPPT",
                 "iseed_rand_pert": "ISEEDARR_RAND_PERTURB", "iseed_spp_conv": "ISEEDARRAY_SPP_CONV",
                 "iseed_spp_pbl": "ISEEDARRAY_SPP_PBL", "iseed_spp_lsm": "ISEEDARRAY_SPP_LSM"}
        d01 = {words[k]: ([0]*8 if k == "iseed_rand_pert" else [int(x) for x in seeds[k]]) for k in SEED_KEYS}
        prediction[f"nens{info['nens']}"] = {"d01": d01, **{f"d{i:02d}": {n: [0]*8 for n in d01}
                                                            for i in range(2, max_dom+1)}}
    (out / ".base_24h.namelist.input").unlink()
    (out / "T0_SEED_PREDICTION.json").write_text(json.dumps(prediction, indent=1) + "\n")
    (out / "PACKAGE_RECEIPT.json").write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print(json.dumps({m["namelist"]: m["sha256"] for m in receipt["members"]}, indent=1))


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
