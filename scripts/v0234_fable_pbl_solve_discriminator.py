#!/usr/bin/env python3
"""Independent audit + decisive discriminator for the v0234 MYNN SP2
lower-span localization (Fable sprint 2026-07-19-v0234-fable-pbl-solve-discriminator).

Phase 1 (this script) uses ONLY sealed captured arrays and numpy:

  A. Verify every SHA-256 named in the handoff brief (KIMI_QUESTION.md).
  B. Independently reload the WRF dump tree and the 28-array GPU-port CPU
     reference bundle and reproduce the sealed comparator's baseline and
     lower-span numbers.
  C. Reconstruct the WRF a,b,c,d systems from dumped operands with code written
     directly from the sealed pristine WRF source (independent re-derivation of
     the bit-exactness claim).
  D. Cross-solve captured WRF and GPU systems with an independent numpy Thomas
     reference (WRF tridiag2 recurrence, float32 for WRF / float64 for GPU) and
     verify each captured x.
  E. Run the pre-registered mechanism experiments:
       E0  null control: re-solve the captured GPU system unchanged.
       E1  surgery: b[0] <- b[0] + a[0] on the captured GPU system (removes the
           port's structural extra surface-interface diffusion term from the
           momentum bottom-row diagonal; both sides define
           a[0] = -dtz0*kmdz0*rhoinv0 so the spurious term equals -a[0]).
       E5  converse: apply the port bug to the WRF system (b[0] <- b[0] - a[0])
           and check it reproduces the sign/structure of the baseline error.
     Each experiment reports tendency SSE ratio vs the sealed baseline, RMS
     reduction, and the comparator's structure partitions (k0-4/k5-top/land/sea).
  F. Row/level partitions of coefficient error, plus operand assumption checks
     (onoff==1, sd_aw==0, sub/det==0, uoce/voce==0, delt==6).

No GPU, no JAX, no WRF/MPI. Read-only on all sealed roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


BRIEF_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084")
DUMP_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/evidence-dumps-fresh-d03-runtime"
)
REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-kimi-pbl-solve-discriminator")
GPT_SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-pristine-pbl-entry-closure-gpt"

EXPECTED_SHAS = {
    str(GPT_SPRINT / "g3-corrected-capture-execution-receipt.json"):
        "459ecb9cbadb232d362678ccb59cbecf7e5c4ffb27b1cf699ffd2a3ed3d99068",
    str(GPT_SPRINT / "g3-corrected-capture-validation.json"):
        "aae01455dbb794c273c25cf5fd96eebfbd3e1ec97b11067963fb250e0230e209",
    str(GPT_SPRINT / "g4-comparator-dispatch.json"):
        "cf79b5033af493e1ef79baaa25e3f98b3ee599b7d2191eb6089e572be91296bb",
    str(BRIEF_ROOT / "cpu-reference/pbl-sp2-reference-28.npz"):
        "a6416b7245d26f23f0df398dd6a3a926a1749cba2069dc3d0ea39c98bc3d2566",
    str(BRIEF_ROOT / "cpu-reference/pbl-sp2-reference-validation.json"):
        "acebb09e607df32e0e4d81d83930068ed5e9b7c0c1ddcc599154703fea03fbe3",
    str(BRIEF_ROOT / "comparator/scientific-comparison.json"):
        "b582d39042eacb8e3a96b78e5cc91814bbd99f76a5e802aa03eb4d5094cbc0ba",
    str(BRIEF_ROOT / "comparator/comparison-terminal.json"):
        "9faec6e0ff2ad1566ab490a096de049f96024b69ea2ec7ed7b7713b4dc33c52c",
    str(BRIEF_ROOT / "comparator/reference-adapter.json"):
        "06c6a5461cea0d9df403b3351bff7b9389277f429ee2a8c6de66ca12f6f70e4a",
}
EXPECTED_CANONICAL = {
    str(GPT_SPRINT / "g3-corrected-capture-validation.json"):
        "69347cb2fe4b3ac289cfa16aadc3bc0b0f4ae12b742d7e42b32df8a2adaf67e3",
    str(BRIEF_ROOT / "cpu-reference/pbl-sp2-reference-validation.json"):
        "40cc191f5f69d2a403d0e964d6a1111338d12b46bcc08c207c30737696a88599",
}
EXPECTED_DUMP_TREE = "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
RANKS = 6
DELT = 6.0


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = {k: v for k, v in value.items() if k != "canonical_payload_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def audit_shas() -> dict:
    result = {"files": {}, "canonical": {}, "dump_tree": {}}
    ok = True
    for path_str, expected in EXPECTED_SHAS.items():
        actual = sha256_file(Path(path_str))
        result["files"][path_str] = {"expected": expected, "actual": actual, "match": actual == expected}
        ok = ok and actual == expected
    for path_str, expected in EXPECTED_CANONICAL.items():
        actual = canonical_payload_sha(Path(path_str))
        result["canonical"][path_str] = {"expected": expected, "actual": actual, "match": actual == expected}
        ok = ok and actual == expected
    files: dict[str, str] = {}
    for path in sorted(DUMP_ROOT.rglob("*")):
        if path.is_symlink():
            raise AuditError(f"symlink in dump tree: {path}")
        if path.is_file():
            files[path.relative_to(DUMP_ROOT).as_posix()] = sha256_file(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    tree_sha = hashlib.sha256(canonical).hexdigest()
    result["dump_tree"] = {
        "expected": EXPECTED_DUMP_TREE,
        "actual": tree_sha,
        "file_count": len(files),
        "match": tree_sha == EXPECTED_DUMP_TREE,
    }
    ok = ok and tree_sha == EXPECTED_DUMP_TREE
    result["all_match"] = ok
    if not ok:
        raise AuditError("sealed evidence SHA mismatch — refusing to continue")
    return result


# ---------------------------------------------------------------------------
# WRF dump loading (independent implementation from meta.txt documentation)
# ---------------------------------------------------------------------------

def parse_meta(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key in {"rank", "real_bytes", "ids_ide_jds_jde_kds_kde",
                   "ims_ime_jms_jme_kms_kme", "ips_ipe_jps_jpe_kps_kpe"}:
            result[key] = [int(v) for v in parts[1:]]
        else:
            result[key] = " ".join(parts[1:])
    if result.get("schema") != "wrfgpu2-mynn-sp2-v1" or result["real_bytes"][0] != 4:
        raise AuditError(f"unexpected dump schema/real size at {path}")
    return result


class WrfDump:
    def __init__(self, root: Path):
        self.root = root
        self.metas = [parse_meta(root / "mynnsp2" / f"rank{r:04d}" / "meta.txt") for r in range(RANKS)]
        ids, ide, jds, jde, kds, kde = self.metas[0]["ids_ide_jds_jde_kds_kde"]
        self.gi = range(ids, ide)
        self.gj = range(jds, jde)
        self.gk = range(kds, kde)
        self.dtype = np.dtype(">f4")

    def outer(self, tag: str, dims: int) -> np.ndarray:
        gi, gj, gk = self.gi, self.gj, self.gk
        shape = (len(gj), len(gi)) if dims == 2 else (len(gk), len(gj), len(gi))
        result = np.full(shape, np.nan, dtype=np.float64)
        filled = np.zeros(shape, dtype=bool)
        for rank, meta in enumerate(self.metas):
            ims, ime, jms, jme, kms, kme = meta["ims_ime_jms_jme_kms_kme"]
            ips, ipe, jps, jpe, kps, kpe = meta["ips_ipe_jps_jpe_kps_kpe"]
            raw = np.fromfile(
                self.root / "mynnsp2" / f"rank{rank:04d}" / f"step000001_outer__{tag}.bin",
                dtype=self.dtype,
            )
            ni, nj, nk = ime - ims + 1, jme - jms + 1, kme - kms + 1
            if dims == 2:
                if raw.size != ni * nj:
                    raise AuditError(f"outer 2d size mismatch {tag} rank {rank}")
                mem = raw.reshape((ni, nj), order="F")
                for j in range(max(jps, gj.start), min(jpe, gj.stop - 1) + 1):
                    for i in range(max(ips, gi.start), min(ipe, gi.stop - 1) + 1):
                        t = (j - gj.start, i - gi.start)
                        if filled[t]:
                            raise AuditError(f"duplicate cell {tag} {t}")
                        result[t] = mem[i - ims, j - jms]
                        filled[t] = True
            else:
                if raw.size != ni * nk * nj:
                    raise AuditError(f"outer 3d size mismatch {tag} rank {rank}")
                mem = raw.reshape((ni, nk, nj), order="F")
                for j in range(max(jps, gj.start), min(jpe, gj.stop - 1) + 1):
                    for k in range(max(kps, gk.start), min(kpe, gk.stop - 1) + 1):
                        for i in range(max(ips, gi.start), min(ipe, gi.stop - 1) + 1):
                            t = (k - gk.start, j - gj.start, i - gi.start)
                            if filled[t]:
                                raise AuditError(f"duplicate cell {tag} {t}")
                            result[t] = mem[i - ims, k - kms, j - jms]
                            filled[t] = True
        if not filled.all():
            raise AuditError(f"coverage gap in outer {tag}")
        return result

    def columns(self, tag: str) -> np.ndarray:
        gi, gj = self.gi, self.gj
        columns: dict[tuple[int, int], np.ndarray] = {}
        bounds = None
        for rank in range(RANKS):
            path = self.root / "mynnsp2" / f"rank{rank:04d}" / f"step000001_columns__{tag}.bin"
            with path.open("rb") as stream:
                while True:
                    header = np.fromfile(stream, dtype=">i4", count=5)
                    if header.size == 0:
                        break
                    i, j, lower, upper, count = (int(v) for v in header)
                    values = np.fromfile(stream, dtype=self.dtype, count=count)
                    if values.size != count or upper - lower + 1 != count:
                        raise AuditError(f"bad column record {tag} ({i},{j})")
                    if (i, j) in columns:
                        raise AuditError(f"duplicate column {tag} ({i},{j})")
                    if bounds is None:
                        bounds = (lower, upper)
                    elif bounds != (lower, upper):
                        raise AuditError(f"bounds drift {tag}")
                    columns[(i, j)] = values.astype("<f4", copy=False)
        if bounds is None or len(columns) != len(gi) * len(gj):
            raise AuditError(f"column coverage mismatch {tag}: {len(columns)}")
        nz = bounds[1] - bounds[0] + 1
        out = np.empty((nz, len(gj), len(gi)), dtype=np.float32)
        for (i, j), values in columns.items():
            out[:, j - gj.start, i - gi.start] = values
        return out


# ---------------------------------------------------------------------------
# Independent WRF coefficient reconstruction (written from the sealed source,
# module_bl_mynnedmf.F90 U/V blocks; float32 kind_phys)
# ---------------------------------------------------------------------------

def reconstruct_wrf_system(w: dict, component: str) -> dict:
    f32 = np.float32
    dtz, rho, kmdz = w["bc_dtz"], w["bc_rho"], w["bc_kmdz"]
    rhoinv = np.empty_like(rho)
    rhoinv[0] = f32(1.0) / rho[0]
    rhoinv[1:] = f32(1.0) / np.maximum(rho[1:], f32(1.0e-4))
    ops = w["bc_lower_operands"]
    delt, ust, wspd = ops[5], ops[6], ops[7]
    uoce, voce, rhosfc, onoff = ops[8], ops[9], ops[10], ops[17]
    s_aw, sd_aw = w["bc_s_aw"], w["bc_sd_aw"]
    if component == "u":
        state, ocean = w["bc_u"], uoce
        s_awx, sd_awx = w["bc_s_awu"], w["bc_sd_awu"]
        subs, detr = w["bc_sub_u"], w["bc_det_u"]
    else:
        state, ocean = w["bc_v"], voce
        s_awx, sd_awx = w["bc_s_awv"], w["bc_sd_awv"]
        subs, detr = w["bc_sub_v"], w["bc_det_v"]
    nz = state.shape[0]
    a = np.empty_like(state)
    b = np.empty_like(state)
    c = np.empty_like(state)
    d = np.empty_like(state)
    half, one = f32(0.5), f32(1.0)
    ust2 = ust * ust
    # k = kts  (drag replaces kmdz(kts) on the diagonal)
    a[0] = -dtz[0] * kmdz[0] * rhoinv[0]
    b[0] = (one + dtz[0] * (kmdz[1] + rhosfc * ust2 / wspd) * rhoinv[0]
            - half * dtz[0] * rhoinv[0] * s_aw[1] * onoff
            - half * dtz[0] * rhoinv[0] * sd_aw[1] * onoff)
    c[0] = (-dtz[0] * kmdz[1] * rhoinv[0]
            - half * dtz[0] * rhoinv[0] * s_aw[1] * onoff
            - half * dtz[0] * rhoinv[0] * sd_aw[1] * onoff)
    d[0] = (state[0] + dtz[0] * ocean * ust2 / wspd
            - dtz[0] * rhoinv[0] * s_awx[1] * onoff
            + dtz[0] * rhoinv[0] * sd_awx[1] * onoff
            + subs[0] * delt + detr[0] * delt)
    sl = slice(1, nz - 1)
    a[sl] = (-dtz[sl] * kmdz[1:nz - 1] * rhoinv[sl]
             + half * dtz[sl] * rhoinv[sl] * s_aw[1:nz - 1] * onoff
             + half * dtz[sl] * rhoinv[sl] * sd_aw[1:nz - 1] * onoff)
    b[sl] = (one + dtz[sl] * (kmdz[1:nz - 1] + kmdz[2:nz]) * rhoinv[sl]
             + half * dtz[sl] * rhoinv[sl] * (s_aw[1:nz - 1] - s_aw[2:nz]) * onoff
             + half * dtz[sl] * rhoinv[sl] * (sd_aw[1:nz - 1] - sd_aw[2:nz]) * onoff)
    c[sl] = (-dtz[sl] * kmdz[2:nz] * rhoinv[sl]
             - half * dtz[sl] * rhoinv[sl] * s_aw[2:nz] * onoff
             - half * dtz[sl] * rhoinv[sl] * sd_aw[2:nz] * onoff)
    d[sl] = (state[sl]
             + dtz[sl] * rhoinv[sl] * (s_awx[1:nz - 1] - s_awx[2:nz]) * onoff
             - dtz[sl] * rhoinv[sl] * (sd_awx[1:nz - 1] - sd_awx[2:nz]) * onoff
             + subs[sl] * delt + detr[sl] * delt)
    a[-1], b[-1], c[-1], d[-1] = f32(0.0), one, f32(0.0), state[-1]
    return {"a": a, "b": b, "c": c, "d": d}


def thomas(a, b, c, d) -> np.ndarray:
    """WRF tridiag2 recurrence, elementwise in the input dtype (a[0] unused)."""
    nz = a.shape[0]
    cp = np.empty_like(b)
    dp = np.empty_like(b)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for k in range(1, nz):
        m = b[k] - cp[k - 1] * a[k]
        cp[k] = c[k] / m
        dp[k] = (d[k] - dp[k - 1] * a[k]) / m
    x = np.empty_like(b)
    x[-1] = dp[-1]
    for k in range(nz - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def metrics(left, right) -> dict:
    delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    if not np.isfinite(delta).all():
        raise AuditError("nonfinite delta")
    return {
        "exact_fraction": float(np.mean(np.asarray(left) == np.asarray(right))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "sse": float(np.sum(delta * delta)),
    }


def sse(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sum(x * x))


def structure_partitions(residual, baseline, xland) -> dict:
    land = np.asarray(xland) < 1.5
    out = {
        "k0_4": sse(residual[:5]) / sse(baseline[:5]),
        "k5_top": sse(residual[5:]) / sse(baseline[5:]),
        "land": sse(residual[:, land]) / sse(baseline[:, land]),
        "sea": sse(residual[:, ~land]) / sse(baseline[:, ~land]),
    }
    out["gate"] = all(v < 1.0 or v == 0.0 for v in out.values())
    return out


def experiment(name, x_solved, u_before, wrf_tend, baseline_diff, xland) -> dict:
    tend = (np.asarray(x_solved, np.float64) - u_before) / DELT
    residual = tend - wrf_tend
    base_sse = sse(baseline_diff)
    ratio = sse(residual) / base_sse
    base_rms = math.sqrt(float(np.mean(baseline_diff ** 2)))
    res_rms = math.sqrt(float(np.mean(residual ** 2)))
    return {
        "experiment": name,
        "sse_ratio_vs_baseline": ratio,
        "explained_sse_fraction": 1.0 - ratio,
        "rms_reduction": math.inf if res_rms == 0.0 else base_rms / res_rms,
        "residual_rms": res_rms,
        "structure": structure_partitions(residual, baseline_diff, xland),
        "decisive_gate": bool(
            ratio <= 0.10
            and (base_rms / max(res_rms, 1e-300)) >= 10.0
            and structure_partitions(residual, baseline_diff, xland)["gate"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        print("REFUSE: output exists", file=sys.stderr)
        return 74

    proof: dict = {"schema": "wrfgpu2-v0234-fable-pbl-solve-discriminator-v1"}

    # --- A. SHA audit -------------------------------------------------------
    proof["sha_audit"] = audit_shas()

    # --- load sealed arrays -------------------------------------------------
    dump = WrfDump(DUMP_ROOT)
    wrf: dict = {}
    for tag in ("bc_lower_operands", "bc_rho", "bc_dfm", "bc_u", "bc_v", "bc_dtz",
                "bc_rhoz", "bc_kmdz", "bc_s_aw", "bc_sd_aw", "bc_s_awu", "bc_sd_awu",
                "bc_s_awv", "bc_sd_awv", "bc_sub_u", "bc_det_u", "bc_sub_v", "bc_det_v",
                "solve_u_a", "solve_u_b", "solve_u_c", "solve_u_d", "solve_u_x",
                "solve_v_a", "solve_v_b", "solve_v_c", "solve_v_d", "solve_v_x",
                "solve_u_du", "solve_v_dv"):
        wrf[tag] = dump.columns(tag)
    wrf_rublten = dump.outer("rublten_exit", 3)
    wrf_rvblten = dump.outer("rvblten_exit", 3)

    gpu = np.load(BRIEF_ROOT / "cpu-reference/pbl-sp2-reference-28.npz", allow_pickle=False)

    # --- operand assumption checks -----------------------------------------
    ops = wrf["bc_lower_operands"]
    proof["operand_checks"] = {
        "delt_unique": sorted(set(np.unique(ops[5]).tolist())),
        "onoff_unique": sorted(set(np.unique(ops[17]).tolist())),
        "max_abs_uoce": float(np.max(np.abs(ops[8]))),
        "max_abs_voce": float(np.max(np.abs(ops[9]))),
        "max_abs_sd_aw": float(np.max(np.abs(wrf["bc_sd_aw"]))),
        "max_abs_sub_u": float(np.max(np.abs(wrf["bc_sub_u"]))),
        "max_abs_det_u": float(np.max(np.abs(wrf["bc_det_u"]))),
        "max_abs_sub_v": float(np.max(np.abs(wrf["bc_sub_v"]))),
        "max_abs_det_v": float(np.max(np.abs(wrf["bc_det_v"]))),
        "kmdz0_min_max": [float(wrf["bc_kmdz"][0].min()), float(wrf["bc_kmdz"][0].max())],
    }

    # --- B. reproduce sealed comparator numbers ----------------------------
    sealed = json.loads((BRIEF_ROOT / "comparator/scientific-comparison.json").read_text())
    repro = {"baseline": {}, "lower_bc": {}, "match": True}
    baseline_u = np.asarray(gpu["sp2_rublten"], np.float64) - wrf_rublten
    baseline_v = np.asarray(gpu["sp2_rvblten"], np.float64) - wrf_rvblten
    wrf_tend_by_field = {"rublten": wrf_rublten, "rvblten": wrf_rvblten}
    for field in ("rublten", "rvblten"):
        mine = metrics(gpu[f"sp2_{field}"], wrf_tend_by_field[field])
        sealed_vals = sealed["quantitative_attribution"]["baseline"][field]
        repro["baseline"][field] = {
            "mine": {k: mine[k] for k in ("rms", "sse", "max_abs")},
            "sealed": {k: sealed_vals[k] for k in ("rms", "sse", "max_abs")},
        }
        for key in ("rms", "sse", "max_abs"):
            if not math.isclose(mine[key], sealed_vals[key], rel_tol=1e-12, abs_tol=0.0):
                repro["match"] = False
    for name in ("solve_u_a", "solve_u_b", "solve_u_c", "solve_u_d", "solve_u_x",
                 "solve_v_a", "solve_v_b", "solve_v_c", "solve_v_d", "solve_v_x"):
        mine = metrics(gpu[name], wrf[name])
        sealed_vals = sealed["field_comparisons"]["lower_bc"][name]
        repro["lower_bc"][name] = {"mine": mine, "sealed_rms": sealed_vals["rms"]}
        for key in ("rms", "sse", "max_abs", "exact_fraction"):
            if not math.isclose(mine[key], sealed_vals[key], rel_tol=1e-12, abs_tol=0.0):
                repro["match"] = False
    # span sse ratios from sealed residual arrays
    repro["span_sse_ratios"] = {}
    for span in ("surface", "mixing", "lower_bc"):
        for field, base in (("rublten", baseline_u), ("rvblten", baseline_v)):
            mine_ratio = sse(np.asarray(gpu[f"residual_after_{span}_{field}"], np.float64)) / sse(base)
            sealed_ratio = sealed["quantitative_attribution"]["spans"][span][field]["sse_ratio"]
            repro["span_sse_ratios"][f"{span}_{field}"] = {
                "mine": mine_ratio, "sealed": sealed_ratio,
            }
            if not math.isclose(mine_ratio, sealed_ratio, rel_tol=1e-12, abs_tol=0.0):
                repro["match"] = False
    proof["comparator_reproduction"] = repro

    # --- C. independent WRF reconstruction ---------------------------------
    recon = {}
    for component in ("u", "v"):
        rebuilt = reconstruct_wrf_system(wrf, component)
        for coeff, values in rebuilt.items():
            name = f"solve_{component}_{coeff}"
            recon[name] = bool(np.array_equal(
                values.view(np.uint32), wrf[name].view(np.uint32)))
    proof["wrf_reconstruction_bit_exact_independent"] = recon
    if not all(recon.values()):
        raise AuditError("independent WRF reconstruction is NOT bit-exact")

    # --- D. cross-solve verification ---------------------------------------
    solver = {}
    for component in ("u", "v"):
        aw, bw, cw, dw = (wrf[f"solve_{component}_{n}"] for n in ("a", "b", "c", "d"))
        xw = thomas(aw, bw, cw, dw)
        xw_dumped = wrf[f"solve_{component}_x"]
        solver[f"wrf_{component}_thomas_f32_vs_dumped_x"] = metrics(xw, xw_dumped)
        ag, bg, cg, dg = (np.asarray(gpu[f"solve_{component}_{n}"], np.float64)
                          for n in ("a", "b", "c", "d"))
        xg = thomas(ag, bg, cg, dg)
        solver[f"gpu_{component}_thomas_f64_vs_captured_x"] = metrics(
            xg, np.asarray(gpu[f"solve_{component}_x"], np.float64))
        solver[f"gpu_vs_wrf_{component}_x_baseline"] = metrics(
            gpu[f"solve_{component}_x"], xw_dumped)
    proof["solver_verification"] = solver

    # --- tendency identities ------------------------------------------------
    ident = {}
    for component, tend_wrf in (("u", wrf_rublten), ("v", wrf_rvblten)):
        du = wrf[f"solve_{component}_du" if component == "u" else "solve_v_dv"]
        ident[f"wrf_{component}_du_equals_rblten_exit"] = bool(
            np.array_equal(np.asarray(du, np.float64), tend_wrf))
        state = wrf[f"bc_{component}"]
        x = wrf[f"solve_{component}_x"]
        # WRF: du = (x - u)/delt in f32
        du_re = (x - state) / np.float32(DELT)
        ident[f"wrf_{component}_du_reconstruction_bit_exact"] = bool(
            np.array_equal(du_re.view(np.uint32), du.view(np.uint32)))
    # GPU pre-solve state recovery: u_before = x - dt*tend (coupler identity)
    u_before_g = np.asarray(gpu["solve_u_x"], np.float64) - DELT * np.asarray(gpu["sp2_rublten"], np.float64)
    v_before_g = np.asarray(gpu["solve_v_x"], np.float64) - DELT * np.asarray(gpu["sp2_rvblten"], np.float64)
    # top row: d(top) = state(top) on both sides -> consistency check
    ident["gpu_u_before_top_vs_d_top"] = metrics(u_before_g[-1], np.asarray(gpu["solve_u_d"], np.float64)[-1])
    ident["gpu_v_before_top_vs_d_top"] = metrics(v_before_g[-1], np.asarray(gpu["solve_v_d"], np.float64)[-1])
    proof["tendency_identities"] = ident

    # --- E. mechanism experiments ------------------------------------------
    xland = np.asarray(gpu["xland"])
    experiments = {}
    for component, tend_wrf, base, u_before in (
        ("u", wrf_rublten, baseline_u, u_before_g),
        ("v", wrf_rvblten, baseline_v, v_before_g),
    ):
        ag, bg, cg, dg = (np.asarray(gpu[f"solve_{component}_{n}"], np.float64)
                          for n in ("a", "b", "c", "d"))
        # E0 null control: unchanged system, independent solver
        x0 = thomas(ag, bg, cg, dg)
        experiments[f"E0_null_{component}"] = experiment(
            "E0_null", x0, u_before, tend_wrf, base, xland)
        # E1 surgery: remove the port's extra surface-interface diffusion term
        # from the momentum bottom-row diagonal: b[0] += a[0]
        b_fix = bg.copy()
        b_fix[0] = bg[0] + ag[0]
        x1 = thomas(ag, b_fix, cg, dg)
        experiments[f"E1_b0_surgery_{component}"] = experiment(
            "E1_b0_surgery", x1, u_before, tend_wrf, base, xland)
        # coefficient-level row-0 comparison before/after
        bw = np.asarray(wrf[f"solve_{component}_b"], np.float64)
        experiments[f"E1_row0_coeff_{component}"] = {
            "b0_gpu_vs_wrf_rms": float(np.sqrt(np.mean((bg[0] - bw[0]) ** 2))),
            "b0_fixed_vs_wrf_rms": float(np.sqrt(np.mean((b_fix[0] - bw[0]) ** 2))),
            "b0_gpu_vs_wrf_max": float(np.max(np.abs(bg[0] - bw[0]))),
            "b0_fixed_vs_wrf_max": float(np.max(np.abs(b_fix[0] - bw[0]))),
        }
        # E5 converse: bug the WRF system the way the port is structured
        aw, bwf, cw, dw = (wrf[f"solve_{component}_{n}"] for n in ("a", "b", "c", "d"))
        b_bug = bwf.copy()
        b_bug[0] = bwf[0] - aw[0]
        x_bug = thomas(aw, b_bug, cw, dw)
        tend_bug = (np.asarray(x_bug, np.float64) - np.asarray(wrf[f"bc_{component}"], np.float64)) / DELT
        diff_bug = tend_bug - np.asarray(tend_wrf, np.float64)
        cos = float(np.sum(diff_bug * base) / max(
            np.sqrt(np.sum(diff_bug ** 2)) * np.sqrt(np.sum(base ** 2)), 1e-300))
        experiments[f"E5_converse_{component}"] = {
            "sse_bugged_wrf_vs_baseline_sse": sse(diff_bug) / sse(base),
            "cosine_similarity_with_baseline_error": cos,
        }
    proof["experiments"] = experiments

    # --- F. per-level error profiles ---------------------------------------
    profiles = {}
    for component in ("u", "v"):
        per_k = {}
        for coeff in ("a", "b", "c", "d", "x"):
            g = np.asarray(gpu[f"solve_{component}_{coeff}"], np.float64)
            wv = np.asarray(wrf[f"solve_{component}_{coeff}"], np.float64)
            per_k[coeff] = [float(np.sqrt(np.mean((g[k] - wv[k]) ** 2))) for k in range(5)] + [
                float(np.sqrt(np.mean((g[5:] - wv[5:]) ** 2)))]
        profiles[component] = {"rms_k0_to_k4_then_k5plus": per_k}
    proof["per_level_rms_profiles"] = profiles

    canonical = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    proof["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.write_text(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "sha_audit_all_match": proof["sha_audit"]["all_match"],
        "comparator_reproduction_match": repro["match"],
        "wrf_reconstruction_bit_exact": all(recon.values()),
        "E1_u": experiments["E1_b0_surgery_u"]["sse_ratio_vs_baseline"],
        "E1_u_gate": experiments["E1_b0_surgery_u"]["decisive_gate"],
        "E1_v": experiments["E1_b0_surgery_v"]["sse_ratio_vs_baseline"],
        "E1_v_gate": experiments["E1_b0_surgery_v"]["decisive_gate"],
        "E0_u": experiments["E0_null_u"]["sse_ratio_vs_baseline"],
        "E0_v": experiments["E0_null_v"]["sse_ratio_vs_baseline"],
        "proof": str(output),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
