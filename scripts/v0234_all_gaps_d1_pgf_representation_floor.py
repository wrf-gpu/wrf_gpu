#!/usr/bin/env python3
"""D1 — PGF incoming-input representation-floor verification (v0234 all-gaps).

Preregistered in `.agent/sprints/2026-07-19-v0234-all-remaining-gaps-fable5/
PREREGISTRATION-D1-D2.md` (commit 690d32b4) as amended by
PREREGISTRATION-AMENDMENT-01.md (commit cca4a428) BEFORE execution.

CPU-only, fp64 NumPy, no JAX, no gpuwrf import, read-only on all inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# The carry pickle references gpuwrf state classes, so gpuwrf (and its jax
# dependency) load for DESERIALIZATION ONLY under JAX_PLATFORMS=cpu; every
# numeric result below is computed in NumPy fp64 after immediate conversion.

SPRINT = Path(__file__).resolve().parent.parent / ".agent/sprints/2026-07-19-v0234-all-remaining-gaps-fable5"
GPT_SPRINT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-s1-residual-closure-gpt/.agent/sprints/2026-07-18-v0234-s1-residual-closure-gpt")
DUMPS = Path("<DATA_ROOT>/wrf_gpu2/v0234_s1_residual_closure_gpt/operator_ledger1/dumps_pgf_inputs1")
CARRY = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_1500_science_60659a2e_terminal2/science-runtime/failure/last-healthy-d03-step-0.pkl"
)
CARRY_SHA = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
WRFINPUT = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs/control/run/wrfinput_d03")
WRFINPUT_SHA = "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"

NX_M, NY_M, NZ_M = 111, 93, 44

FIELDS_3D = ["p", "pb", "ph", "php", "al", "alt"]
FIELDS_2D = ["mu"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_meta(path: Path) -> dict:
    text = path.read_text()
    dom = re.search(r"ids_ide_jds_jde_kds_kde\s+(-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)", text)
    mem = re.search(r"ims_ime_jms_jme_kms_kme\s+(-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)", text)
    pat = re.search(r"ips_ipe_jps_jpe_kps_kpe\s+(-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)", text)
    if not (dom and mem and pat):
        raise SystemExit(f"STOP: malformed meta {path}")
    g = lambda m: [int(x) for x in m.groups()]
    return {"dom": g(dom), "mem": g(mem), "pat": g(pat)}


def reassemble(name: str, three_d: bool) -> np.ndarray:
    out = np.full((NZ_M, NY_M, NX_M) if three_d else (NY_M, NX_M), np.nan, dtype=np.float64)
    ranks = sorted(DUMPS.glob("rank[0-9][0-9][0-9][0-9]"))
    if len(ranks) != 12:
        raise SystemExit(f"STOP: expected 12 ranks, found {len(ranks)}")
    for rank_dir in ranks:
        meta = parse_meta(rank_dir / "meta.txt")
        ims, ime, jms, jme, kms, kme = meta["mem"]
        ips, ipe, jps, jpe, _, _ = meta["pat"]
        raw = np.fromfile(rank_dir / f"step000001_op_pgf_in__{name}.f64", dtype=">f8")
        ni = ime - ims + 1
        nj = jme - jms + 1
        nk = kme - kms + 1
        if three_d:
            if raw.size != ni * nj * nk:
                raise SystemExit(f"STOP: size mismatch {name} {rank_dir.name}")
            win = raw.reshape(nj, nk, ni)
        else:
            if raw.size != ni * nj:
                raise SystemExit(f"STOP: size mismatch {name} {rank_dir.name}")
            win = raw.reshape(nj, ni)
        i_lo, i_hi = ips, min(ipe, NX_M)
        j_lo, j_hi = jps, min(jpe, NY_M)
        isl = slice(i_lo - ims, i_hi - ims + 1)
        jsl = slice(j_lo - jms, j_hi - jms + 1)
        if three_d:
            out[:, j_lo - 1 : j_hi, i_lo - 1 : i_hi] = np.transpose(win[jsl, 0:NZ_M, isl], (1, 0, 2))
        else:
            out[j_lo - 1 : j_hi, i_lo - 1 : i_hi] = win[jsl, isl]
    if not np.isfinite(out).all():
        raise SystemExit(f"STOP: reassembly holes/nonfinite in {name}")
    return out


def ulp32(t: np.ndarray) -> np.ndarray:
    mag = np.maximum(np.abs(t), 2.0 ** -126)
    return np.exp2(np.floor(np.log2(mag)) - 23.0)


def stats(a: np.ndarray) -> dict:
    return {
        "rms": float(np.sqrt(np.mean(a * a))),
        "max_abs": float(np.max(np.abs(a))),
        "finite": bool(np.isfinite(a).all()),
    }


def read_wrfinput_hgt() -> np.ndarray:
    import netCDF4

    with netCDF4.Dataset(WRFINPUT) as ds:
        hgt = np.array(ds.variables["HGT"][0], dtype=np.float64)
    if hgt.shape != (NY_M, NX_M):
        raise SystemExit(f"STOP: HGT shape {hgt.shape}")
    return hgt


def main() -> None:
    # Authorities.
    carry_sha = sha256_file(CARRY)
    if carry_sha != CARRY_SHA:
        raise SystemExit(f"STOP: carry hash {carry_sha}")
    wrfinput_sha = sha256_file(WRFINPUT)
    if wrfinput_sha != WRFINPUT_SHA:
        raise SystemExit(f"STOP: wrfinput hash {wrfinput_sha}")
    ledger = json.loads((GPT_SPRINT / "pgf-component-ledger.json").read_text())
    manifest = {row["relative_path"]: row["file_sha256"] for row in ledger["fresh_wrf"]["raw_manifest"]["rows"]}
    consumed = []
    for rank_dir in sorted(DUMPS.glob("rank[0-9][0-9][0-9][0-9]")):
        for fname in ["meta.txt"] + [f"step000001_op_pgf_in__{f}.f64" for f in FIELDS_3D + FIELDS_2D]:
            rel = f"{rank_dir.name}/{fname}"
            actual = sha256_file(rank_dir / fname)
            expected = manifest.get(rel)
            if expected is None or actual != expected:
                raise SystemExit(f"STOP: dump hash mismatch {rel}")
            consumed.append(rel)

    with open(CARRY, "rb") as fh:
        carry = pickle.load(fh)
    st = carry.state
    gpu = {
        "p": np.asarray(st.p_perturbation, dtype=np.float64),
        "p_total": np.asarray(st.p_total, dtype=np.float64),
        "mu": np.asarray(st.mu_perturbation, dtype=np.float64),
        "mu_total": np.asarray(st.mu_total, dtype=np.float64),
        "ph": np.asarray(st.ph_perturbation, dtype=np.float64),
        "ph_total": np.asarray(st.ph_total, dtype=np.float64),
    }

    wrf = {name: reassemble(name, True) for name in FIELDS_3D}
    wrf["mu"] = reassemble("mu", False)

    # GPU pb equivalent and identity checks.
    gpu_pb = gpu["p_total"] - gpu["p"]
    delta_pb = gpu_pb - wrf["pb"]
    pb_exact_zero = bool(np.all(delta_pb == 0.0))

    # Decisive field: p, ULP-normalized against WRF total pressure.
    p_total_wrf = wrf["p"] + wrf["pb"]
    delta_p = gpu["p"] - wrf["p"]
    n_p = np.abs(delta_p) / ulp32(p_total_wrf)
    hgt = read_wrfinput_hgt()
    hgt3 = np.broadcast_to(hgt[None, :, :], n_p.shape)
    corr = float(np.corrcoef(n_p.ravel(), hgt3.ravel())[0, 1])
    frac_gt64 = float(np.mean(n_p > 64.0))
    gates = {
        "median_n_p": float(np.median(n_p)),
        "p95_n_p": float(np.percentile(n_p, 95.0)),
        "p999_n_p": float(np.percentile(n_p, 99.9)),
        "max_n_p": float(np.max(n_p)),
        "frac_n_p_gt_64": frac_gt64,
        "corr_n_p_vs_hgt": corr,
        "pb_delta_exact_zero": pb_exact_zero,
    }
    verdict_green = (
        gates["median_n_p"] <= 1.0
        and gates["p95_n_p"] <= 4.0
        and gates["p999_n_p"] <= 16.0
        and frac_gt64 < 0.001
        and abs(corr) < 0.5
        and pb_exact_zero
    )

    # Cross-check against the sealed ledger delta stats (pipeline validation).
    ledger_deltas = ledger["input_deltas_gpu_minus_wrf"]
    cross = {
        "p_rms_here": stats(delta_p)["rms"],
        "p_rms_ledger": ledger_deltas["p"]["rms"],
        "p_max_here": stats(delta_p)["max_abs"],
        "p_max_ledger": ledger_deltas["p"]["max_abs"],
    }
    pipeline_ok = (
        math.isclose(cross["p_rms_here"], cross["p_rms_ledger"], rel_tol=1e-9)
        and math.isclose(cross["p_max_here"], cross["p_max_ledger"], rel_tol=1e-9)
    )

    descriptive = {}
    eps32 = 2.0 ** -23
    for name, gpu_arr, wrf_arr, scale_arr in [
        ("mu", gpu["mu"], wrf["mu"], gpu["mu_total"]),
        ("ph", gpu["ph"][:44], wrf["ph"][:44] if wrf["ph"].shape[0] >= 44 else wrf["ph"], None),
        ("php", gpu["ph"][:44], wrf["php"][:44] if wrf["php"].shape[0] >= 44 else wrf["php"], None),
    ]:
        delta = gpu_arr - wrf_arr
        entry = stats(delta)
        if scale_arr is not None:
            entry["rms_over_eps32_scale"] = float(entry["rms"] / (eps32 * np.sqrt(np.mean(scale_arr**2))))
        descriptive[name] = entry
    descriptive["al_ledger_rms"] = ledger_deltas["al"]["rms"]
    descriptive["alt_ledger_rms"] = ledger_deltas["alt"]["rms"]

    envelope_restated = {
        "intrinsic_nonspec_rmse_committed": 0.0017099604943289304,
        "member_envelope_max_committed": 0.0026994,
        "intrinsic_le_envelope": 0.0017099604943289304 <= 0.0026994,
    }

    payload = {
        "schema": "gpuwrf.v0234.all-gaps-d1-pgf-representation-floor.v1",
        "preregistration_commits": ["690d32b4", "cca4a428"],
        "authorities": {
            "carry_sha256": carry_sha,
            "wrfinput_sha256": wrfinput_sha,
            "ledger_canonical": "c49881d06fcf581505891efaac7b88a033a11327d63cd293b1bcc76d4de3d20a",
            "dump_files_consumed": len(consumed),
        },
        "gates": gates,
        "pipeline_cross_check": {**cross, "matches_sealed_ledger": pipeline_ok},
        "descriptive": descriptive,
        "envelope_restated": envelope_restated,
        "verdict": "D1_REPRESENTATION_FLOOR_CONFIRMED" if (verdict_green and pipeline_ok) else "D1_RED_REAL_INPUT_DIFFERENCE_OR_PIPELINE",
        "resource_attestation": {
            "cpu_only": True,
            "gpu_queries": 0,
            "gpu_locks": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "jax_imported": "jax" in sys.modules,
            "gpuwrf_imported": "gpuwrf" in sys.modules,
        },
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["self_sha256"] = hashlib.sha256(body).hexdigest()
    out = SPRINT / "d1-pgf-representation-floor.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": payload["verdict"], "gates": gates, "cross": cross, "self": payload["self_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
