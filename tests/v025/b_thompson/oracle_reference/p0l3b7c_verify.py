# Provenance: source commit 83c7ff126; original SHA256 5f7386af1a3c9ed2be9dbf242f3f2aefa9cdbaebc0f1274d2a35e1ac8340c40d
# Source: closure/k7/p0l3b7c_verify.py
"""P0 L3b r7 WRF/port columns: the ONE verdict rule (stdlib only; contract P0_L3B_R7_COLUMN_PACKAGE_CONTRACT.md,
manager MANAGER_L3B_R7_CRITIC_DECISION.md).

Used twice: p0l3b7c_runner.py imports it (hash-pinned) to compute the verdict it writes, and p0l3b7c_launch.sh runs it
on every written result to RE-DERIVE that verdict from the stored per-level arrays; any difference between the stored
and the re-derived decision, an input/tolerance/code hash that is not the pinned one, an omitted field or mutant, or a
K6c numeric entry is a BINDING_MISMATCH (the package stops).  Rules:
  manifest  exactly the committed r7 manifest 31f7e9d4 (no other sha is accepted): schema /3, COMPLETE, blockers [],
            every numeric negative covered, the six IDs; on every registered (quantity, level) bound = effect/10 and
            bound >= 10 x max(floor fp64, floor production); fall stage: reference margin >= 0.10 (or no cloud) and the
            mutant margin/exempt level lists [] -- except entry_rebalance_nc (K6b), where BOTH must equal [k6b_target]
            (F1 of the manager decision); K6b targets A6a level 1 and A6b level 2, each with its registered target
            quantities bounded at that level.  K6c raw_sed_n is structural only: never a column discriminator, its
            coverage == k6c_structural.admissible_columns (five IDs, A2 outside), per-column k6c_dstar.admissible agrees.
  tolerances Gate-2 5bdc03da and mp28 1601d68b, both full-sha checked; the mp28 file must embed the Gate-2 sha and
            carry the Gate-2 atol/rtol unchanged on the six shared fields.
  shared    fixed arm vs WRF on the warm levels (WRF T_in > 275 K): err - (atol + rtol |wrf|) <= 0 for every GATED
            field (the frozen Gate-2 statistic; non-finite fails); max abs / rel / fp32-ulp residual reported per field
            and level.
  bounds    the fixed arm within EVERY registered bound of every DISCRIMINATING negative of the column.
  mutants   every DISCRIMINATING negative must exceed its bound at EVERY registered (quantity, level) -- no "somewhere"
            acceptance.  Status: INEFFECTIVE (bit-identical to the fixed arm on qc/nc/nwfa), K6B_TARGET_MISS (K6b not
            exceeding at a k6b_target entry), OFF_TARGET (differs, no registered entry exceeded), BELOW_BOUND (some but
            not all registered entries exceeded) or PASS.  Only PASS counts.
  verdict   PASS iff shared, bounds and every mutant PASS; otherwise FAIL (a valid result).
Resources (manager review MANAGER_P0L3B7C_REVIEW.md): every file is size-checked (lstat, regular, <= cap) BEFORE it is
read and read bounded (result 16 MiB, opens 2 MiB, manifest/tolerances 1 MiB); hashes stream.  `verify` runs ONLY as
its own launcher step: it refuses unless its cgroup is app.slice/<label>-<column>-<arm>-verify.scope, and binds its
verdict to the exact result/opens bytes it read in ONE line written O_EXCL to --out:
  P0L3B7C_VERDICT <column> <dtype> <PASS|FAIL> result=<sha256> opens=<sha256>
Usage: python3 p0l3b7c_verify.py check-manifest --manifest M --manifest-sha256 S
       python3 p0l3b7c_verify.py verify --unit U --out V --manifest M --manifest-sha256 S --result R --column C
              --state-dtype D --tolerances T --tolerances-sha256 S --gate2-tolerances G --gate2-tolerances-sha256 S
              --opens O --rc N --expect KEY=SHA ...      (npz_sha256, wrf_output_sha256, compare_sha256, port:<file>)
       python3 p0l3b7c_verify.py self-test --manifest M
Prints PASS | FAIL | MANIFEST OK | BINDING_MISMATCH: <reason>.  Exit: verify 0 PASS / 1 FAIL / 2 mismatch or refusal
(no --out written); check-manifest and self-test 0.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

MANIFEST_SHA = "31f7e9d4993895aee5f746a19c91d27d483b1c0bd55b21b02117f8523d677ff7"
GATE2_TOL_SHA = "5bdc03da9cd0099bce48e30de88e6c570114480728f0f12551cacaf137b48b41"
MP28_TOL_SHA = "1601d68be1d7c4206b7f54ef27168304edc10a10d16878951eaa62efa677ceab"
COLUMNS = ("A1_kept", "A2_full_evap", "A3_mixed", "A5_band_w", "A6a_k6b_rh090", "A6b_k6b_rh095")
SHARED = ("T", "qv", "qc", "dT", "dtheta_dry", "theta_m")
GATED = SHARED + ("nc", "nwfa")
DISC_Q = ("qc", "nc", "nwfa")
K6B, K6C = "entry_rebalance_nc", "raw_sed_n"
K6B_TARGET = {"A6a_k6b_rh090": 1, "A6b_k6b_rh095": 2}          # manager decision: exempt levels [1] and [2]
NUMERIC = ("drop_cloud_sed_on", "vtc_only_mask", K6B, "mass_w", "gate_off", "all_lqc", "post_branch", "no_ksed")
T_WARM_K = 275.0            # p0_gate2_compare.T_WARM_K ("warm_T_in_gt_275K"); the runner asserts equality
XDC_MARGIN = 0.10           # p0c_aero_columns.XDC_MARGIN (fall-stage rebalance-edge distance)
LEVEL_MASK = "warm_T_in_gt_275K"
RESULT_MAX = 16 * 1048576   # measured schema footprint ~0.1-0.5 MB (44 levels, <= 3 mutants)
OPENS_MAX = 2 * 1048576     # audit log: a few hundred snapshot paths
SMALL_MAX = 1048576         # manifest (~45 KB) and tolerance files (~2 KB); both also hash-pinned
UNIT_RE = r"p0l3b7c_[a-z0-9]+-[A-Za-z0-9_]+-(prod|fp64)-verify"
# R2: the exact dtype vector of the State entering the mp28 adapter, per arm.  fp64 = the shipped force_fp64 carry;
# production = the ADR-007 fp32-gated MEASUREMENT carry (DEFAULT_DTYPES), not the shipped default
_DT = ("theta", "qv", "qc", "Nc", "nwfa", "nifa", "p_total")
STATE_DTYPES = {"fp64": {k: "float64" for k in _DT},
                "production": {**{k: "float32" for k in _DT[:-1]}, "p_total": "float64"}}


class Refuse(Exception):
    pass


def need(ok, why):
    if not ok:
        raise Refuse(why)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_capped(path, cap, what):
    """The bytes of a regular non-symlink file of at most cap bytes: size checked BEFORE reading, read bounded."""
    st = os.lstat(path)
    need(stat.S_ISREG(st.st_mode), f"{what} is not a regular file")
    need(st.st_size <= cap, f"{what} is {st.st_size} B > cap {cap} B (fail closed, not read)")
    with open(path, "rb") as fh:
        data = fh.read(cap + 1)
    need(len(data) <= cap, f"{what} grew beyond cap {cap} B while being read")
    return data


def in_unit(cgroup_text, unit):
    """True iff the unit is a verifier step name and /proc/self/cgroup places this process in exactly that scope."""
    own = cgroup_text.strip().split("::", 1)[-1]
    return bool(re.fullmatch(UNIT_RE, unit or "")) and own.endswith(f"/app.slice/{unit}.scope")


def _strict_json(text):
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(Refuse(f"non-finite JSON constant {c}")))


def enc(x):
    """A float for JSON (allow_nan=False): finite as is, else the string nan / inf / -inf."""
    x = float(x)
    return x if math.isfinite(x) else ("nan" if math.isnan(x) else ("inf" if x > 0 else "-inf"))


def dec(v):
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise Refuse(f"not a number: {v!r}")
    if isinstance(v, str):
        need(v in ("nan", "inf", "-inf"), f"not a number: {v!r}")
    return float(v)


def _finite_pos(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def ulp32(x):
    """IEEE binary32 spacing at |x| (the WRF output is REAL(4)); 2**-149 at zero / subnormal."""
    x = abs(float(x))
    if not math.isfinite(x):
        return math.inf
    if x < 2.0 ** -126:
        return 2.0 ** -149
    return 2.0 ** (math.frexp(x)[1] - 24)


# ------------------------------------------------------------------ manifest
def check_manifest(m):
    need(isinstance(m, dict) and m.get("schema") == "p0c_aero_column_manifest/3", "manifest schema")
    need(m.get("status") == "COMPLETE" and m.get("blockers") == [], "manifest status/blockers (no waiver)")
    need(m.get("frozen_tolerances_sha256") == GATE2_TOL_SHA, "manifest frozen Gate-2 tolerance sha")
    cols = m.get("columns") or {}
    need(set(cols) == set(COLUMNS), f"manifest column IDs {sorted(cols)}")
    nz = m.get("nz")
    need(type(nz) is int and nz > 0, "manifest nz")
    cov = m.get("coverage") or {}
    need(set(cov) == set(NUMERIC) | {K6C}, "manifest coverage keys")
    adm = (m.get("k6c_structural") or {}).get("admissible_columns")
    need(isinstance(adm, list) and adm and sorted(adm) == sorted(cov[K6C]) and set(adm) <= set(COLUMNS)
         and "A2_full_evap" not in adm, "K6c scope misuse: raw_sed_n coverage != the structural D* admissible columns")
    for cid, c in cols.items():
        d = c.get("discriminators") or {}
        need(K6C not in d, f"K6c scope misuse: raw_sed_n is a numeric discriminator on {cid}")
        need(set(d) <= set(NUMERIC), f"unknown discriminator on {cid}: {sorted(set(d) - set(NUMERIC))}")
        need(bool((c.get("k6c_dstar") or {}).get("admissible")) == (cid in adm), f"K6c D* admissibility of {cid}")
        tgt = c.get("k6b_target")
        if cid in K6B_TARGET:
            lv = K6B_TARGET[cid]
            need(isinstance(tgt, dict) and tgt.get("accepted") is True and tgt.get("level_1based") == lv
                 and (c.get("k6b_level_T") or {}).get("level_1based") == lv, f"K6b target of {cid} is not level {lv}")
            tq = tgt.get("quantities")
            need(isinstance(tq, list) and tq and set(tq) <= set(DISC_Q), f"K6b target quantities of {cid}")
            need((d.get(K6B) or {}).get("status") == "DISCRIMINATING", f"K6b not DISCRIMINATING on {cid}")
        else:
            need(tgt is None and K6B not in d, f"K6b target/discriminator outside the pair on {cid}")
        for name, x in d.items():
            st = x.get("status")
            need(st in ("DISCRIMINATING", "UNDISCRIMINABLE"), f"status of {name} on {cid}")
            if st == "UNDISCRIMINABLE":
                need([o for o in cov[name] if o != cid], f"unresolved UNDISCRIMINABLE {name} on {cid}")
                continue
            qs = x.get("quantities") or {}
            need(qs and set(qs) <= set(DISC_Q), f"quantities of {name} on {cid}")
            for q, spec in qs.items():
                b, e, f = spec.get("bound"), spec.get("effect"), spec.get("floor")
                need(isinstance(b, dict) and b and isinstance(e, dict) and isinstance(f, dict)
                     and set(b) == set(e) == set(f), f"bound/effect/floor keys of {name} {q} on {cid}")
                for lev, bv in b.items():
                    need(lev.isdigit() and 1 <= int(lev) <= nz, f"level {lev!r} of {name} {q} on {cid}")
                    fl = f[lev]
                    need(_finite_pos(bv) and _finite_pos(e[lev]) and bv == e[lev] / 10.0,
                         f"bound != effect/10 for {name} {q}@{lev} on {cid}")
                    need(isinstance(fl, dict) and set(fl) == {"fp64", "production"}
                         and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in fl.values())
                         and bv >= 10.0 * max(fl.values()), f"bound < 10 x floor for {name} {q}@{lev} on {cid}")
            fs = x.get("fall_stage") or {}
            rmm = fs.get("reference_min_margin")
            need(rmm is None or (isinstance(rmm, (int, float)) and math.isfinite(rmm) and rmm >= XDC_MARGIN),
                 f"reference fall-stage margin of {name} on {cid}")
            want = [K6B_TARGET[cid]] if name == K6B else []
            need(fs.get("mutant_levels_within_margin_1based") == want and fs.get("mutant_exempt_levels_1based") == want,
                 f"fall-stage levels of {name} on {cid} != {want}")
            if name == K6B:
                for q in c["k6b_target"]["quantities"]:
                    need(str(K6B_TARGET[cid]) in qs.get(q, {}).get("bound", {}),
                         f"K6b target {q}@{K6B_TARGET[cid]} not bounded on {cid}")
    for name in NUMERIC:
        disc = sorted(cid for cid, c in cols.items()
                      if (c.get("discriminators") or {}).get(name, {}).get("status") == "DISCRIMINATING")
        need(disc and sorted(cov[name]) == disc, f"coverage of {name} != its DISCRIMINATING columns")
    pair = m.get("k6b_pair") or {}
    need(pair.get("columns") == list(K6B_TARGET) and pair.get("both_discriminating") is True
         and pair.get("independent") is True, "K6b pair")
    return m


def load_manifest(path, sha):
    need(sha == MANIFEST_SHA, "manifest sha is not the committed r7 manifest 31f7e9d4")
    data = read_capped(path, SMALL_MAX, "manifest")
    need(hashlib.sha256(data).hexdigest() == sha, "manifest file sha256 mismatch")
    return check_manifest(_strict_json(data.decode()))


# ------------------------------------------------------------------ tolerances
def check_tolerances(mp28, g2):
    need(g2.get("schema") == "p0_gate2_tolerances/1" and g2.get("level_mask") == LEVEL_MASK
         and set(g2.get("fields", {})) == set(SHARED), "Gate-2 tolerance schema/fields/mask")
    need(mp28.get("schema") == "p0c_mp28_tolerances/1" and mp28.get("level_mask") == LEVEL_MASK
         and set(mp28.get("fields", {})) == set(GATED), "mp28 tolerance schema/fields/mask")
    need(mp28.get("frozen_gate2_tolerances_sha256") == GATE2_TOL_SHA, "mp28 tolerances do not embed Gate-2 5bdc03da")
    out = {}
    for k, v in mp28["fields"].items():
        a, r = v.get("atol"), v.get("rtol")
        need(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                 for x in (a, r)), f"tolerance {k}")
        if k in SHARED:
            need(g2["fields"][k] == v, f"mp28 tolerance {k} differs from the frozen Gate-2 value")
        out[k] = (float(a), float(r))
    return out


def load_tolerances(mp28_path, mp28_sha, g2_path, g2_sha):
    need(mp28_sha == MP28_TOL_SHA and g2_sha == GATE2_TOL_SHA, "tolerance pins are not 1601d68b / 5bdc03da")
    mp, g = read_capped(mp28_path, SMALL_MAX, "mp28 tolerances"), read_capped(g2_path, SMALL_MAX, "Gate-2 tolerances")
    need(hashlib.sha256(mp).hexdigest() == mp28_sha, "mp28 tolerance file sha256 mismatch")
    need(hashlib.sha256(g).hexdigest() == g2_sha, "Gate-2 tolerance file sha256 mismatch")
    return check_tolerances(_strict_json(mp.decode()), _strict_json(g.decode()))


# ------------------------------------------------------------------ decision
def numeric_discriminators(col):
    return sorted(n for n, x in col["discriminators"].items() if x["status"] == "DISCRIMINATING")


def _vec(arrays, part, name, nz):
    v = arrays.get(part, {}).get(name)
    need(isinstance(v, list) and len(v) == nz, f"omitted or malformed field {part}.{name}")
    return [dec(x) for x in v]


def _max(vals, gated):
    best, lev = -math.inf, None
    for k, v in enumerate(vals):
        if gated[k]:
            v = math.inf if math.isnan(v) else v
            if v > best:
                best, lev = v, k + 1
    return {"value": enc(best), "level_1based": lev}


def decide(cid, col, arrays, tol):
    """The verdict from the stored arrays: T_in, wrf/fixed per GATED field, mutants per DISC_Q (all per level)."""
    need(isinstance(arrays, dict) and set(arrays) == {"T_in", "wrf", "fixed", "mutants"}, "arrays keys")
    nz = len(arrays["T_in"]) if isinstance(arrays["T_in"], list) else 0
    need(nz > 0, "arrays T_in")
    T_in = [dec(x) for x in arrays["T_in"]]
    gated = [t > T_WARM_K for t in T_in]
    need(any(gated), "level mask selects no level")
    for part in ("wrf", "fixed"):
        need(set(arrays[part]) == set(GATED), f"{part} fields {sorted(arrays[part])} != {list(GATED)}")
    wrf = {f: _vec(arrays, "wrf", f, nz) for f in GATED}
    fix = {f: _vec(arrays, "fixed", f, nz) for f in GATED}
    shared = {}
    for f in GATED:
        a, r = tol[f]
        ab, rel, ul, ok = [], [], [], True
        for k in range(nz):
            e = abs(fix[f][k] - wrf[f][k])
            ab.append(e)
            rel.append(e / abs(wrf[f][k]) if wrf[f][k] != 0 else (0.0 if e == 0 else math.inf))
            ul.append(e / ulp32(wrf[f][k]))
            if gated[k] and not (math.isfinite(e) and e - (a + r * abs(wrf[f][k])) <= 0.0):
                ok = False
        shared[f] = {"within": ok, "atol": a, "rtol": r, "max_abs": _max(ab, gated), "max_rel": _max(rel, gated),
                     "max_ulp32": _max(ul, gated), "abs": [enc(x) for x in ab], "rel": [enc(x) for x in rel],
                     "ulp32": [enc(x) for x in ul]}
    names = numeric_discriminators(col)
    muts = arrays["mutants"]
    need(isinstance(muts, dict), "mutants")
    need(K6C not in muts, "K6c scope misuse: raw_sed_n reported as a numeric mutant")
    need(sorted(muts) == names, f"mutants {sorted(muts)} != the column's DISCRIMINATING negatives {names}")
    bounds, mres = {}, {}
    for name in names:
        d = col["discriminators"][name]
        mq = {q: _vec(muts, name, q, nz) for q in DISC_Q}
        need(set(muts[name]) == set(DISC_Q), f"mutant {name} quantities")
        tgt = set()
        if name == K6B:
            tgt = {f"{q}@{K6B_TARGET[cid]}" for q in col["k6b_target"]["quantities"]}
        bx, ent = {}, {}
        for q, spec in sorted(d["quantities"].items()):
            for lev, b in sorted(spec["bound"].items(), key=lambda kv: int(kv[0])):
                k = int(lev) - 1
                need(k < nz, "bound level beyond nz")
                ef, em = abs(fix[q][k] - wrf[q][k]), abs(mq[q][k] - wrf[q][k])
                key = f"{q}@{lev}"
                bx[key] = {"err": enc(ef), "bound": b, "within": bool(math.isfinite(ef) and ef <= b)}
                ent[key] = {"err": enc(em), "bound": b, "exceeds": bool(math.isfinite(em) and em > b),
                            "k6b_target": key in tgt, "port_effect": enc(abs(mq[q][k] - fix[q][k])),
                            "manifest_effect": spec["effect"][lev]}
        need(tgt <= set(ent), f"K6b target entries {sorted(tgt - set(ent))} not registered")
        same = all(mq[q][k] == fix[q][k] for q in DISC_Q for k in range(nz))
        differs = sorted({k + 1 for q in DISC_Q for k in range(nz) if not mq[q][k] == fix[q][k]})
        hit = [v["exceeds"] for v in ent.values()]
        if same:
            st = "INEFFECTIVE"
        elif tgt and not all(ent[t]["exceeds"] for t in tgt):
            st = "K6B_TARGET_MISS"
        elif all(hit):
            st = "PASS"
        elif not any(hit):
            st = "OFF_TARGET"
        else:
            st = "BELOW_BOUND"
        bounds[name] = bx
        mres[name] = {"status": st, "entries": ent, "bit_identical": same, "differs_levels_1based": differs}
    shared_ok = all(v["within"] for v in shared.values())
    bounds_ok = all(v["within"] for b in bounds.values() for v in b.values())
    mut_ok = all(v["status"] == "PASS" for v in mres.values())
    return {"verdict": "PASS" if (shared_ok and bounds_ok and mut_ok) else "FAIL", "shared_ok": shared_ok,
            "bounds_ok": bounds_ok, "mutants_ok": mut_ok, "gated_levels_1based": [k + 1 for k in range(nz) if gated[k]],
            "shared": shared, "bounds": bounds, "mutants": mres}


def k6c_record(m, cid):
    return {"structural_only": True, "numeric": False,
            "admissible": cid in m["k6c_structural"]["admissible_columns"],
            "evidence": {k: m["k6c_structural"][k] for k in ("gate_test_sha256", "gate_log_sha256", "lemma_json_sha256")}}


def canon(x):
    return json.dumps(x, sort_keys=True, allow_nan=False)


def verify(a):
    """Launcher-step re-derivation; returns (verdict, result sha256, opens sha256) or raises Refuse."""
    need(in_unit(Path("/proc/self/cgroup").read_text(), a.unit),
         "unscoped verification refused: not inside its named <label>-<column>-<arm>-verify step scope")
    need(a.out and not os.path.lexists(a.out), "--out missing or already exists")
    m = load_manifest(a.manifest, a.manifest_sha256)
    tol = load_tolerances(a.tolerances, a.tolerances_sha256, a.gate2_tolerances, a.gate2_tolerances_sha256)
    need(a.column in COLUMNS and a.state_dtype in ("production", "fp64"), "column/dtype")
    rb, ob = read_capped(a.result, RESULT_MAX, "result"), read_capped(a.opens, OPENS_MAX, "opens")
    r = _strict_json(rb.decode())
    o = _strict_json(ob.decode())
    exp = dict(kv.split("=", 1) for kv in a.expect)
    need(set(exp) == {"npz_sha256", "wrf_output_sha256", "compare_sha256", "port:physics_couplers.py",
                      "port:thompson_aero_column.py"}, "verifier --expect keys")
    need(r.get("schema") == "p0l3b7c_result/1" and r.get("column") == a.column and r.get("state_dtype") == a.state_dtype,
         "result schema/column/dtype")
    i = r.get("inputs") or {}
    want = {"manifest_sha256": MANIFEST_SHA, "tolerances_sha256": MP28_TOL_SHA, "gate2_tolerances_sha256": GATE2_TOL_SHA,
            "verify_sha256": sha256_file(__file__), "npz_sha256": exp["npz_sha256"],
            "wrf_output_sha256": exp["wrf_output_sha256"], "compare_sha256": exp["compare_sha256"]}
    for k, v in want.items():
        need(i.get(k) == v, f"result input {k} is not the pinned/expected one")
    need(r.get("port_code_sha256") == {"physics_couplers.py": exp["port:physics_couplers.py"],
                                       "thompson_aero_column.py": exp["port:thompson_aero_column.py"]},
         "port code is not the snapshot's")
    need(r.get("k6c") == k6c_record(m, a.column), "K6c scope misuse: result k6c record is not structural-only")
    need(r.get("state_dtypes") == STATE_DTYPES[a.state_dtype],
         f"state dtype vector {r.get('state_dtypes')!r} != the {a.state_dtype} arm's exact {STATE_DTYPES[a.state_dtype]}")
    need(o.get("compare_exit") == int(a.rc) and not o.get("forbidden_opens")
         and "data/fixtures/thompson-aero-tables-v1.npz" in o.get("snapshot_opens", []), "audit opens/exit")
    dd = decide(a.column, m["columns"][a.column], r.get("arrays"), tol)
    need(canon(dd) == canon(r.get("decision")), "recorded decision != re-derived decision")
    need(r.get("verdict") == dd["verdict"] and (dd["verdict"] == "PASS") == (int(a.rc) == 0)
         and int(a.rc) in (0, 1), "verdict/exit code")
    return dd["verdict"], hashlib.sha256(rb).hexdigest(), hashlib.sha256(ob).hexdigest()


# ------------------------------------------------------------------ self-test (synthetic arrays on the real manifest)
def _synthetic(m, cid, bump=2.0):
    col = m["columns"][cid]
    nz = m["nz"]
    T_in = [290.0 - 2.0 * k for k in range(nz)]                        # warm below ~level 8, cold above
    base = {"T": [t + 0.1 for t in T_in], "qv": [1e-2] * nz, "qc": [1e-4] * nz, "dT": [0.1] * nz,
            "dtheta_dry": [0.1] * nz, "theta_m": [300.0] * nz, "nc": [1e8] * nz, "nwfa": [1e9] * nz}
    muts = {}
    for name in numeric_discriminators(col):
        mq = {q: list(base[q]) for q in DISC_Q}
        for q, spec in col["discriminators"][name]["quantities"].items():
            for lev, b in spec["bound"].items():
                mq[q][int(lev) - 1] = base[q][int(lev) - 1] + bump * b
        muts[name] = mq
    return {"T_in": T_in, "wrf": copy.deepcopy(base), "fixed": copy.deepcopy(base), "mutants": muts}


def self_test(manifest_path):
    n = 0
    m = load_manifest(manifest_path, MANIFEST_SHA); n += 1
    g2 = {"schema": "p0_gate2_tolerances/1", "level_mask": LEVEL_MASK,
          "fields": {"T": {"atol": 0.03, "rtol": 0.0}, "qv": {"atol": 1.5e-5, "rtol": 0.0},
                     "qc": {"atol": 1.5e-5, "rtol": 0.0}, "dT": {"atol": 0.03, "rtol": 0.0},
                     "dtheta_dry": {"atol": 0.03, "rtol": 0.0}, "theta_m": {"atol": 0.04, "rtol": 0.0}}}
    mp = {"schema": "p0c_mp28_tolerances/1", "level_mask": LEVEL_MASK, "frozen_gate2_tolerances_sha256": GATE2_TOL_SHA,
          "fields": {**copy.deepcopy(g2["fields"]), "nc": {"atol": 1.0, "rtol": 5e-4}, "nwfa": {"atol": 1.0, "rtol": 2e-4}}}
    tol = check_tolerances(mp, g2); n += 1

    def refused(fn, frag):
        try:
            fn()
        except Refuse as e:
            assert frag in str(e), f"refusal {e!r} lacks {frag!r}"
            return 1
        raise AssertionError(f"no refusal ({frag})")
    for bad, frag in ((dict(frozen_gate2_tolerances_sha256="0" * 64), "embed Gate-2"),
                      (dict(fields={**mp["fields"], "qc": {"atol": 3e-5, "rtol": 0.0}}), "differs from the frozen")):
        n += refused(lambda: check_tolerances({**mp, **bad}, g2), frag)
    n += refused(lambda: load_tolerances(manifest_path, "0" * 64, manifest_path, GATE2_TOL_SHA), "tolerance pins")
    # every column: the synthetic exact-bound design passes; each negative is caught at the registered entry
    for cid in COLUMNS:
        d = decide(cid, m["columns"][cid], _synthetic(m, cid), tol)
        assert d["verdict"] == "PASS" and d["mutants"], (cid, d["verdict"]); n += 1
    col = m["columns"]["A6b_k6b_rh095"]
    s = _synthetic(m, "A6b_k6b_rh095")
    for q in DISC_Q:                                                   # K6b exceeds ONLY off-target (level 1)
        s["mutants"][K6B][q][1] = s["fixed"][q][1]
    d = decide("A6b_k6b_rh095", col, s, tol)
    assert d["mutants"][K6B]["status"] == "K6B_TARGET_MISS" and d["verdict"] == "FAIL"; n += 1
    s = _synthetic(m, "A6a_k6b_rh090")                                 # K6b differs only at an unregistered level
    for q in DISC_Q:
        s["mutants"][K6B][q] = list(s["fixed"][q])
    s["mutants"][K6B]["nc"][2] += 1e9
    d = decide("A6a_k6b_rh090", m["columns"]["A6a_k6b_rh090"], s, tol)
    assert d["mutants"][K6B]["status"] == "K6B_TARGET_MISS" and d["mutants"][K6B]["differs_levels_1based"] == [3]; n += 1
    s = _synthetic(m, "A1_kept")                                       # gate_off differs only off its entries
    s["mutants"]["gate_off"] = {q: list(s["fixed"][q]) for q in DISC_Q}
    s["mutants"]["gate_off"]["qc"][10] += 1.0
    assert decide("A1_kept", m["columns"]["A1_kept"], s, tol)["mutants"]["gate_off"]["status"] == "OFF_TARGET"; n += 1
    s = _synthetic(m, "A1_kept")                                       # one registered entry inside its bound
    s["mutants"]["gate_off"]["nc"][2] = s["fixed"]["nc"][2] + 0.5 * m["columns"]["A1_kept"]["discriminators"][
        "gate_off"]["quantities"]["nc"]["bound"]["3"]
    assert decide("A1_kept", m["columns"]["A1_kept"], s, tol)["mutants"]["gate_off"]["status"] == "BELOW_BOUND"; n += 1
    s = _synthetic(m, "A3_mixed")                                      # bit-identical mutant
    s["mutants"]["all_lqc"] = {q: list(s["fixed"][q]) for q in DISC_Q}
    d = decide("A3_mixed", m["columns"]["A3_mixed"], s, tol)
    assert d["mutants"]["all_lqc"]["status"] == "INEFFECTIVE" and d["verdict"] == "FAIL"; n += 1
    s = _synthetic(m, "A5_band_w")                                     # fixed arm outside a registered bound
    s["fixed"]["qc"][2] += 2.0 * m["columns"]["A5_band_w"]["discriminators"]["no_ksed"]["quantities"]["qc"]["bound"]["3"]
    d = decide("A5_band_w", m["columns"]["A5_band_w"], s, tol)
    assert not d["bounds_ok"] and d["verdict"] == "FAIL"; n += 1
    s = _synthetic(m, "A2_full_evap")                                  # shared field: warm level fails, cold level masked
    s["fixed"]["T"][0] += 0.05
    assert decide("A2_full_evap", m["columns"]["A2_full_evap"], s, tol)["verdict"] == "FAIL"; n += 1
    s = _synthetic(m, "A2_full_evap")
    s["fixed"]["T"][40] += 5.0
    d = decide("A2_full_evap", m["columns"]["A2_full_evap"], s, tol)
    assert d["verdict"] == "PASS" and d["shared"]["T"]["max_abs"]["level_1based"] != 41; n += 1
    s = _synthetic(m, "A2_full_evap")
    s["fixed"]["nwfa"][0] = "nan"
    assert decide("A2_full_evap", m["columns"]["A2_full_evap"], s, tol)["verdict"] == "FAIL"; n += 1
    for frac, within in ((0.5, True), (2.0, False)):                 # nc: atol + rtol |wrf| (the rtol term counts)
        s = _synthetic(m, "A2_full_evap")
        s["fixed"]["nc"][0] = s["wrf"]["nc"][0] + frac * (1.0 + 5e-4 * s["wrf"]["nc"][0])
        assert decide("A2_full_evap", m["columns"]["A2_full_evap"], s, tol)["shared"]["nc"]["within"] is within; n += 1
    assert ulp32(1.0) == 2.0 ** -23 and ulp32(0.0) == 2.0 ** -149 and ulp32(3.0) == 2.0 ** -22; n += 1
    s = _synthetic(m, "A1_kept")                                       # omitted field / mutant, K6c numeric misuse
    del s["fixed"]["nwfa"]
    n += refused(lambda: decide("A1_kept", m["columns"]["A1_kept"], s, tol), "fixed fields")
    s = _synthetic(m, "A1_kept")
    del s["mutants"]["gate_off"]
    n += refused(lambda: decide("A1_kept", m["columns"]["A1_kept"], s, tol), "DISCRIMINATING negatives")
    s = _synthetic(m, "A1_kept")
    s["mutants"][K6C] = {q: list(s["fixed"][q]) for q in DISC_Q}
    n += refused(lambda: decide("A1_kept", m["columns"]["A1_kept"], s, tol), "K6c scope misuse")
    # manifest-level: K6c misuse, K6b target drift, fall-stage exemption drift, effect/floor, status
    def mut(fn):
        mm = copy.deepcopy(m)
        fn(mm)
        return lambda: check_manifest(mm)
    n += refused(mut(lambda mm: mm["columns"]["A1_kept"]["discriminators"].__setitem__(
        K6C, {"status": "DISCRIMINATING", "quantities": {}})), "K6c scope misuse")
    n += refused(mut(lambda mm: mm["coverage"][K6C].append("A2_full_evap")), "K6c scope misuse")
    n += refused(mut(lambda mm: mm["columns"]["A6b_k6b_rh095"]["k6b_target"].__setitem__("level_1based", 1)),
                 "K6b target")
    n += refused(mut(lambda mm: mm["columns"]["A6b_k6b_rh095"]["discriminators"][K6B]["fall_stage"].__setitem__(
        "mutant_exempt_levels_1based", [1, 2])), "fall-stage levels")
    n += refused(mut(lambda mm: mm["columns"]["A3_mixed"]["discriminators"]["gate_off"]["fall_stage"].__setitem__(
        "mutant_levels_within_margin_1based", [1])), "fall-stage levels")
    n += refused(mut(lambda mm: mm["columns"]["A1_kept"]["discriminators"]["gate_off"]["quantities"]["nc"]["bound"]
                     .__setitem__("3", 1.0)), "bound != effect/10")
    n += refused(mut(lambda mm: mm["columns"]["A1_kept"]["discriminators"]["gate_off"]["quantities"]["nc"]["floor"]
                     ["3"].__setitem__("fp64", 1e9)), "10 x floor")
    n += refused(mut(lambda mm: mm.__setitem__("status", "BLOCKED")), "status/blockers")
    n += refused(lambda: load_manifest(manifest_path, "0" * 64), "committed r7 manifest")
    # launcher-side re-derivation: a runner that records PASS over arrays that miss the K6b target is caught
    s = _synthetic(m, "A6b_k6b_rh095")
    good = decide("A6b_k6b_rh095", m["columns"]["A6b_k6b_rh095"], s, tol)
    for q in DISC_Q:
        s["mutants"][K6B][q][1] = s["fixed"][q][1]
    assert canon(decide("A6b_k6b_rh095", m["columns"]["A6b_k6b_rh095"], s, tol)) != canon(good); n += 1
    # resource guards: size before read, symlink refused; the verifier-scope predicate
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "r.json"
        f.write_bytes(b"x" * 64)
        assert read_capped(f, 64, "r") == b"x" * 64; n += 1
        n += refused(lambda: read_capped(f, 63, "r"), "fail closed, not read")
        (Path(td) / "l.json").symlink_to(f)
        n += refused(lambda: read_capped(Path(td) / "l.json", 64, "l"), "not a regular file")
    cgt = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/p0l3b7c_r1-A6b_k6b_rh095-fp64-verify.scope\n"
    assert in_unit(cgt, "p0l3b7c_r1-A6b_k6b_rh095-fp64-verify"); n += 1
    assert not in_unit(cgt, "p0l3b7c_r1-A6b_k6b_rh095-prod-verify") and not in_unit(
        cgt.replace("-verify.scope", ".scope"), "p0l3b7c_r1-A6b_k6b_rh095-fp64") and not in_unit(
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/p0l3b7c_fakes_x-suite.scope\n",
        "p0l3b7c_r1-A1_kept-prod-verify"); n += 1
    print(f"P0L3B7C_VERIFY SELF-TEST PASS checks={n}")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("check-manifest", "verify", "self-test"))
    for k in ("manifest", "manifest-sha256", "result", "column", "state-dtype", "tolerances", "tolerances-sha256",
              "gate2-tolerances", "gate2-tolerances-sha256", "opens", "rc", "unit", "out"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--expect", action="append", default=[])
    a = ap.parse_args(argv)
    if a.mode == "self-test":                  # an AssertionError/Refuse here is a crash: no PASS line, exit != 0
        self_test(a.manifest)
        return 0
    try:
        if a.mode == "check-manifest":
            load_manifest(a.manifest, a.manifest_sha256)
            print("MANIFEST OK")
            return 0
        verdict, rsha, osha = verify(a)
        fd = os.open(a.out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(f"P0L3B7C_VERDICT {a.column} {a.state_dtype} {verdict} result={rsha} opens={osha}\n")
        print(verdict)
        return 0 if verdict == "PASS" else 1
    except Exception as exc:                  # any malformed result/binding is a mismatch, never a verdict
        print(f"BINDING_MISMATCH: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
