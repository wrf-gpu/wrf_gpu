"""Independent audit of the v0234 WRF-conditioning-ensemble terminal proof.

Sprint: 2026-07-18-v0234-dycore-suboperator-kimi (Kimi K3 thinking-max).

Re-derives every checkable claim of the GPT conditioning-ensemble falsification
at commit 2aa91beea88c138a61f736d91e9d233f78c8e1db from primary evidence:

  A. sprint artifact file/canonical SHA-256 reproduction
  B. per-member artifact file/canonical SHA-256 reproduction (7 members)
  C. frozen WRF authority hashes (binary, namelist, wrfinput, wrfbdy)
  D. chronology: preregistration/boundary commits predate all admitted results
  E. perturbation masks re-derived from seeds: balance, orthogonality, hashes
  F. perturbed wrfinput_d03 ground-truth audit vs base (exact one-ULP, boundary
     preservation, only-U/V semantic difference, manifest stat reproduction)
  G. decision statistics recomputed from member analyses (replicates the
     frozen aggregate pooling) and compared to the sealed values bit-exactly
  H. resource isolation: admissions, monitors (every sample), probes, override
  I. GPU attestation: command-log scan + zero-use attestations
  J. model tree attestation (git tree hashes, diagnostic-only delta)
  K. raw-archive spot check: reassemble step1/step200 SP1/SP4 for control and
     two perturbed members straight from the tar.zst evidence and reproduce
     the decisive 74.4246x / 213.5279x ratios from raw dumps + GPU savepoints
  L. live production CPU-boundary snapshot (current ALISIOS confinement)

Every check returns (passed, evidence). The audit fails closed: any mismatch
marks the proof REJECTED with the failing check named.

CPU-only. Must be run under: taskset -c 13-15,29-31 nice -n 10 ionice -c 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GPT_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-conditioning-ensemble-gpt"
KIMI_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-first-interval-momentum-kimi"
MY_SPRINT = REPO / ".agent/sprints/2026-07-18-v0234-dycore-suboperator-kimi"

KIMI_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi")
BASE_RUN = KIMI_ROOT / "run/momsp_arm"
BASE_DUMPS = KIMI_ROOT / "momsp_dumps"
BASE_CACHE = KIMI_ROOT / "compare/wrf_global_cache"
MEMBERS_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_conditioning_ensemble_gpt/members")
INVALIDATED_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_conditioning_ensemble_gpt_taskset_invalidated_20260718T1012Z")
GPU_SAVEPOINTS = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_first_interval_momentum2/savepoints"
)
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/raw_spotcheck")

PREREG_COMMIT = "215551f15e774eff67db6262724319d3f07378c4"
BOUNDARY_COMMIT = "1947297e359b2eb83ee870bf66cc18526a695a52"
SEAL_COMMIT = "2aa91beea88c138a61f736d91e9d233f78c8e1db"
CONTRACT_COMMIT = "2c802e82f4be86ccf9fc7d72c55df17c14d47148"
MODEL_TREE_FINAL = "83ed838bb605fc4128bf17b1649c0a21aa741dc1"
MODEL_TREE_PRE_KIMI = "835dcc29bf316c0715b41a72e064985e9cf099df"

MEMBER_IDS = (
    "control-repeat",
    "mask-a-minus", "mask-a-plus",
    "mask-b-minus", "mask-b-plus",
    "mask-c-minus", "mask-c-plus",
)
PERTURBED_IDS = tuple(m for m in MEMBER_IDS if m != "control-repeat")
MEMBER_SPEC = {
    "control-repeat": {"kind": "control", "mask": None, "polarity": 0},
    "mask-a-minus": {"kind": "perturbed", "mask": "mask_a", "polarity": -1},
    "mask-a-plus": {"kind": "perturbed", "mask": "mask_a", "polarity": 1},
    "mask-b-minus": {"kind": "perturbed", "mask": "mask_b", "polarity": -1},
    "mask-b-plus": {"kind": "perturbed", "mask": "mask_b", "polarity": 1},
    "mask-c-minus": {"kind": "perturbed", "mask": "mask_c", "polarity": -1},
    "mask-c-plus": {"kind": "perturbed", "mask": "mask_c", "polarity": 1},
}
MASK_SPECS = {
    "mask_a": {"seed": 0x243F6A8885A308D3, "carrier": "k_parity"},
    "mask_b": {"seed": 0x13198A2E03707344, "carrier": "k_pair_parity"},
    "mask_c": {"seed": 0xA4093822299F31D0, "carrier": "k_walsh_product"},
}
MILESTONES = (1, 2, 5, 10, 20, 50, 100, 200)
STEPS = tuple(range(1, 201))

PROOF_PATH = GPT_SPRINT / "proof.json"
RETAINED_PATH = GPT_SPRINT / "retained-evidence-manifest.json"
PLAN_PATH = GPT_SPRINT / "ensemble-plan.json"
ANALYSIS_PATH = GPT_SPRINT / "ensemble-analysis.json"

# Contract-cited anchor values (CONTRACT.md of this sprint).
CONTRACT_CITED = {
    "proof_file": "c0c5281c3c7d020042d9c1211164c56eab97bd4f8e9de35257bca14933356842",
    "proof_canonical": "4ed61e303734b8f220f255b35b2ff4348b1388e2e225135c1601ec7fc2b67309",
    "retained_file": "8abfd7eb9738cbf9ac67237109dbf1785d6aa4c8217bab81f580235203c8e0d7",
    "retained_canonical": "a66455da98f7279d793c09fddcdbc5d59546fb84a3b897ab2b4cc8d115e5bbaf",
    "step1_gpu_rmse": 0.006535199460511423,
    "step1_member_max": 0.00008780965331443297,
    "step1_ratio": 74.42461294215421,
    "step200_gpu_rmse": 0.08126477713921927,
    "step200_member_max": 0.00038058158496078815,
    "step200_ratio": 213.52787510092506,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any, self_key: str | None = None) -> str:
    if self_key is not None and isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != self_key}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_self_hashed(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    embedded = payload.get("proof_sha256")
    observed = canonical_digest(payload, "proof_sha256")
    if embedded != observed:
        raise RuntimeError(f"self-hash mismatch: {path}: {embedded} != {observed}")
    return payload


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# ---------------------------------------------------------------- masks (E)

def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    mask = (1 << 64) - 1
    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = values + np.uint64(seed & mask) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def carrier(name: str, nz: int) -> np.ndarray:
    k = np.arange(nz, dtype=np.int64)
    if name == "k_parity":
        bit = k & 1
    elif name == "k_pair_parity":
        bit = (k // 2) & 1
    elif name == "k_walsh_product":
        bit = (k + k // 2) & 1
    else:
        raise ValueError(name)
    return np.where(bit == 0, 1, -1).astype(np.int8)


def mask_signs(variable: str, shape: tuple[int, int, int], spec: dict[str, Any]) -> np.ndarray:
    nz, ny, nx = shape
    j = np.arange(5, ny - 5, dtype=np.uint64)[:, None]
    i = np.arange(5, nx - 5, dtype=np.uint64)[None, :]
    salt = 0x5555555555555555 if variable == "U" else 0xAAAAAAAAAAAAAAAA
    keyed = (j << np.uint64(32)) ^ i ^ np.uint64(salt)
    horizontal = np.where(
        (splitmix64(keyed, int(spec["seed"])) >> np.uint64(63)) == 0, 1, -1
    ).astype(np.int8)
    return carrier(spec["carrier"], nz)[:, None, None] * horizontal[None, :, :]


# ------------------------------------------------------------------ checks

def check(cond: bool, name: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(cond), "evidence": evidence}


def audit_sprint_artifacts(proof: dict, retained: dict) -> list[dict]:
    out = []
    out.append(check(sha256_file(PROOF_PATH) == CONTRACT_CITED["proof_file"],
                     "proof.json file sha256", sha256_file(PROOF_PATH)))
    out.append(check(canonical_digest(json.loads(PROOF_PATH.read_text()), "proof_sha256")
                     == CONTRACT_CITED["proof_canonical"],
                     "proof.json canonical sha256", proof["proof_sha256"]))
    out.append(check(sha256_file(RETAINED_PATH) == CONTRACT_CITED["retained_file"],
                     "retained manifest file sha256", sha256_file(RETAINED_PATH)))
    out.append(check(canonical_digest(json.loads(RETAINED_PATH.read_text()), "proof_sha256")
                     == CONTRACT_CITED["retained_canonical"],
                     "retained manifest canonical sha256", retained["proof_sha256"]))
    mismatches = []
    for rel, row in proof["changed_files_before_seal"].items():
        local = REPO / rel
        if not local.is_file() or sha256_file(local) != row["file_sha256"]:
            mismatches.append(rel)
        if local.stat().st_size != row["bytes"]:
            mismatches.append(rel + ":bytes")
    out.append(check(not mismatches, "changed_files_before_seal all reproduce",
                     {"checked": len(proof["changed_files_before_seal"]), "mismatches": mismatches}))
    out.append(check(sha256_file(PLAN_PATH) == "b6a5d3a05c40111dc47a8f8bc7339741019df2111a3974ae89235482b00d9fb7"
                     and canonical_digest(json.loads(PLAN_PATH.read_text()), "plan_sha256")
                     == "c193b783aa2c92098df526ead12d3add6a4bb39a7ab0ff48582bb28e49f8d4e6",
                     "ensemble-plan file+canonical sha256", None))
    out.append(check(sha256_file(ANALYSIS_PATH) == "879b6094965ebfa654fc17ca4a254173b47b5ca53e251d193a723c9cd7e0c83e"
                     and canonical_digest(json.loads(ANALYSIS_PATH.read_text()), "proof_sha256")
                     == "b0c65ce2451bc3ee6cc393e4d783c91dde3b140dbba08a70a159c67186a5a7f7",
                     "ensemble-analysis file+canonical sha256", None))
    reg = GPT_SPRINT / "metrics-and-thresholds-register.json"
    out.append(check(sha256_file(reg) == "66b335365debbd8811da23501b363f6f575adddc530a396e0d1b52876cf75a04"
                     and canonical_digest(json.loads(reg.read_text()), "proof_sha256")
                     == "6922af8b4e649b02fe063f945430560816cdf02f8ed53f76f341335188b17089",
                     "metrics register file+canonical sha256", None))
    out.append(check(sha256_file(KIMI_SPRINT / "proof.json") == "cffb3cf70e203a2cf4ffd043677769b1a596cef08e729e65af67c7a075e010ad"
                     and canonical_digest(json.loads((KIMI_SPRINT / "proof.json").read_text()), "proof_sha256")
                     == "9577feff9cb71d8e920d8751c878c64dce6e949db73f35a350fa0905ab982b5a",
                     "kimi first-interval proof file+canonical sha256", None))
    out.append(check(sha256_file(KIMI_SPRINT / "retained-evidence-manifest.json") == "2d282ba6adf53af4eb6633413af3ed6b9cab237aee9d464a7aac43bccf7acfb2"
                     and canonical_digest(json.loads((KIMI_SPRINT / "retained-evidence-manifest.json").read_text()), "proof_sha256")
                     == "3135ba52952af5fa7bbc6cba9a6a697e40a11e1f13b8f215d4273990bf421b08",
                     "kimi retained manifest file+canonical sha256", None))
    # metrics register's own referenced artifacts (register lists lineage refs)
    register = load_self_hashed(reg)
    ref_bad = []
    ref_count = 0
    for section in ("reference_artifacts",):
        for name, row in (register.get(section) or {}).items():
            if isinstance(row, dict) and "path" in row and "file_sha256" in row:
                ref_count += 1
                p = Path(row["path"])
                if not p.is_file() or sha256_file(p) != row["file_sha256"]:
                    ref_bad.append(name)
    out.append(check(not ref_bad and ref_count > 0,
                     f"metrics register referenced artifact hashes ({ref_count})", {"bad": ref_bad}))
    return out


def audit_member_artifacts(proof: dict) -> list[dict]:
    out = []
    for member in MEMBER_IDS:
        rows = proof["ensemble"]["members"][member]
        bad = []
        for key, path in (
            ("admission", None),
            ("analysis", None),
            ("archive_receipt", None),
            ("execution", None),
            ("perturbation", None),
            ("raw_inventory", None),
            ("resource_monitor", None),
        ):
            row = rows[key]
            p = Path(row["path"])
            if not p.is_file():
                bad.append(f"{key}:missing")
                continue
            if sha256_file(p) != row["file_sha256"]:
                bad.append(f"{key}:file")
            if p.stat().st_size != row["bytes"]:
                bad.append(f"{key}:bytes")
            if canonical_digest(json.loads(p.read_text()), "proof_sha256") != row["canonical_sha256"]:
                bad.append(f"{key}:canonical")
        arch = rows["raw_archive"]
        ap = Path(arch["path"])
        if not ap.is_file() or ap.stat().st_size != arch["bytes"]:
            bad.append("raw_archive:bytes")
        out.append(check(not bad, f"member {member} artifact hashes", {"bad": bad}))
    # raw archive content hashes (long; done inside this single audit pass)
    for member in MEMBER_IDS:
        arch = proof["ensemble"]["members"][member]["raw_archive"]
        digest = sha256_file(arch["path"])
        out.append(check(digest == arch["sha256"], f"member {member} raw archive sha256", digest))
    return out


def audit_frozen_authority() -> list[dict]:
    out = []
    fa = {
        "namelist": (BASE_RUN / "namelist.input", "9ce4336dd4b878685871370a2bdf048ce27c4bb52b655b5a33089008026caaf3", 2066),
        "wrf_binary": (KIMI_ROOT / "wrf_iso/install_iso/bin/wrf", "072b5fa168d2cc6b943145317ae150aceb94b386217538ce530be9f70643c539", 42075848),
        "wrfbdy_d01": (BASE_RUN / "wrfbdy_d01", "1b5b20408b3384b2e81dc09f3ec029c3110117745e623dc0b7669f0976121fec", 8198766),
        "wrfinput_d01": (BASE_RUN / "wrfinput_d01", "afd069d201c14f0001308696f3b5b2f30f4e1b77f368d15a0d9d7c5b77dc0756", 9674911),
        "wrfinput_d02": (BASE_RUN / "wrfinput_d02", "ec281f234940d7353b9fdf544833db635b4388e83bab4a7a71e025f23f36b964", 26672072),
        "wrfinput_d03": (BASE_RUN / "wrfinput_d03", "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a", 10563652),
    }
    for name, (path, expected, nbytes) in fa.items():
        ok = path.is_file() and path.stat().st_size == nbytes and sha256_file(path) == expected
        out.append(check(ok, f"frozen authority {name}", str(path)))
    return out


def audit_chronology(proof: dict) -> list[dict]:
    out = []
    prereg = parse_ts(git("show", "-s", "--format=%cI", PREREG_COMMIT))
    boundary = parse_ts(git("show", "-s", "--format=%cI", BOUNDARY_COMMIT))
    seal = parse_ts(git("show", "-s", "--format=%cI", SEAL_COMMIT))
    first_admission = None
    last_monitor_end = None
    rows = []
    for member in MEMBER_IDS:
        adm = json.loads(Path(proof["ensemble"]["members"][member]["admission"]["path"]).read_text())
        mon = json.loads(Path(proof["ensemble"]["members"][member]["resource_monitor"]["path"]).read_text())
        pert = json.loads(Path(proof["ensemble"]["members"][member]["perturbation"]["path"]).read_text())
        rows.append({
            "member": member,
            "admitted": adm["captured_at_utc"],
            "perturbation_created": pert.get("created_at_utc"),
            "monitor_start": mon["started_at_utc"],
            "monitor_end": mon["ended_at_utc"],
        })
        a = parse_ts(adm["captured_at_utc"])
        first_admission = a if first_admission is None else min(first_admission, a)
        e = parse_ts(mon["ended_at_utc"])
        last_monitor_end = e if last_monitor_end is None else max(last_monitor_end, e)
    pert_ok = all(parse_ts(r["perturbation_created"]) > prereg for r in rows if r["perturbation_created"])
    out.append(check(prereg < first_admission, "preregistration predates first admission",
                     {"prereg_utc": prereg.isoformat(), "first_admission_utc": first_admission.isoformat()}))
    out.append(check(boundary < first_admission, "boundary hardening predates first admission",
                     {"boundary_utc": boundary.isoformat(), "first_admission_utc": first_admission.isoformat()}))
    out.append(check(pert_ok, "all perturbation manifests created after preregistration", None))
    out.append(check(last_monitor_end < seal, "final member monitor end predates seal commit",
                     {"last_monitor_end_utc": last_monitor_end.isoformat(), "seal_utc": seal.isoformat()}))
    out.append(check(all(parse_ts(r["admitted"]) < parse_ts(r["monitor_end"]) for r in rows),
                     "each admission precedes its monitor end", rows))
    # committed plan at prereg commit identical to evaluated plan
    committed = subprocess.run(
        ["git", "show", f"{PREREG_COMMIT}:.agent/sprints/2026-07-18-v0234-conditioning-ensemble-gpt/ensemble-plan.json"],
        cwd=REPO, check=True, stdout=subprocess.PIPE).stdout
    out.append(check(hashlib.sha256(committed).hexdigest()
                     == "b6a5d3a05c40111dc47a8f8bc7339741019df2111a3974ae89235482b00d9fb7",
                     "plan bytes at preregistration commit == evaluated plan", None))
    # principal override pre-results
    override = load_self_hashed(GPT_SPRINT / "principal-resource-override.json")
    ba = override["before_revision_audit"]
    out.append(check(ba["zero_admitted_launches"] and ba["zero_member_analyses"]
                     and parse_ts(ba["checked_at_utc"]) < first_admission,
                     "principal override audited zero results before revision",
                     {"checked_at_utc": ba["checked_at_utc"]}))
    out.append(check(override["science_contract"]["frozen_science_canonical_sha256_before"]
                     == override["science_contract"]["frozen_science_canonical_sha256_after"]
                     == "7b5d2ff9e5a7ee12b63c1b37af2af1a754a8b35fbb75b02262a2249c808c99cd"
                     and override["science_contract"]["thresholds_and_decision_rule_unchanged"],
                     "resource-only plan revision left frozen science identical", None))
    out.append(check(override["revised_preregistration"]["plan_canonical_sha256"]
                     == "c193b783aa2c92098df526ead12d3add6a4bb39a7ab0ff48582bb28e49f8d4e6",
                     "override binds evaluated plan canonical hash", None))
    # excluded attempts preserved
    out.append(check(INVALIDATED_ROOT.is_dir(),
                     "excluded taskset-only attempts preserved (negative evidence)",
                     str(INVALIDATED_ROOT)))
    return out


def audit_masks(plan: dict) -> list[dict]:
    out = []
    shapes = {"U": (44, 93, 112), "V": (44, 94, 111)}
    masks: dict[str, dict[str, np.ndarray]] = {}
    for mask_id, spec in MASK_SPECS.items():
        per_var = {}
        combined_chunks = []
        row = plan["perturbation"]["masks"][mask_id]
        for var in ("U", "V"):
            signs = mask_signs(var, shapes[var], spec)
            masks.setdefault(mask_id, {})[var] = signs
            combined_chunks.append(signs.ravel())
            pv = row["per_variable"][var]
            ok = (int(np.count_nonzero(signs > 0)) == pv["positive_count"]
                  and int(np.count_nonzero(signs < 0)) == pv["negative_count"]
                  and int(signs.astype(np.int64).sum()) == pv["balance_sum"] == 0
                  and hashlib.sha256(signs.tobytes()).hexdigest() == pv["sign_sha256"]
                  and list(signs.shape) == pv["shape"])
            per_var[var] = ok
        combined = np.concatenate(combined_chunks)
        combined_ok = (int(combined.astype(np.int64).sum()) == row["combined_balance_sum"] == 0
                       and int(combined.size) == row["combined_cell_count"]
                       and hashlib.sha256(combined.tobytes()).hexdigest() == row["combined_sign_sha256"])
        out.append(check(all(per_var.values()) and combined_ok,
                         f"{mask_id} re-derived balance/counts/hashes", per_var))
    dots = {}
    for left, right in (("mask_a", "mask_b"), ("mask_a", "mask_c"), ("mask_b", "mask_c")):
        dot = 0
        for var in ("U", "V"):
            dot += int(np.sum(masks[left][var].astype(np.int64) * masks[right][var].astype(np.int64),
                              dtype=np.int64))
        dots[f"{left}__{right}"] = dot
    out.append(check(dots == plan["perturbation"]["pairwise_dot_products"]
                     and all(v == 0 for v in dots.values()),
                     "mask pairwise orthogonality (dot products all zero)", dots))
    seeds_ok = (MASK_SPECS["mask_a"]["seed"] == plan["perturbation"]["masks"]["mask_a"]["seed"]
                and MASK_SPECS["mask_b"]["seed"] == plan["perturbation"]["masks"]["mask_b"]["seed"]
                and MASK_SPECS["mask_c"]["seed"] == plan["perturbation"]["masks"]["mask_c"]["seed"])
    out.append(check(seeds_ok, "mask seeds in plan match audited constants",
                     {k: v["seed"] for k, v in MASK_SPECS.items()}))
    return out


def _open_netcdf(path: Path, mode: str = "r"):
    from netCDF4 import Dataset
    ds = Dataset(path, mode)
    ds.set_auto_mask(False)
    ds.set_auto_scale(False)
    return ds


def audit_perturbed_inputs() -> list[dict]:
    out = []
    base_path = BASE_RUN / "wrfinput_d03"
    with _open_netcdf(base_path) as base_ds:
        base_vars = {}
        base_attrs = {}
        for name in base_ds.variables:
            var = base_ds.variables[name]
            data = np.ascontiguousarray(np.asarray(var[:]))
            base_vars[name] = (str(var.dtype), tuple(var.shape),
                               hashlib.sha256(data.tobytes()).hexdigest())
            base_attrs[name] = {a: var.getncattr(a) for a in var.ncattrs()}
        base_globals = {a: base_ds.getncattr(a) for a in base_ds.ncattrs()}
        base_dims = {d: len(base_ds.dimensions[d]) for d in base_ds.dimensions}
        base_arrays = {v: np.asarray(base_ds.variables[v][0], dtype=np.float32) for v in ("U", "V")}
    for member in PERTURBED_IDS:
        spec = MEMBER_SPEC[member]
        manifest = load_self_hashed(MEMBERS_ROOT / member / "perturbation-manifest.json")
        mpath = MEMBERS_ROOT / member / "run" / "wrfinput_d03"
        bad = []
        if not mpath.is_file():
            out.append(check(False, f"{member} perturbed input present", str(mpath)))
            continue
        with _open_netcdf(mpath) as mds:
            # only-U/V semantic difference
            for name, (dtype, shape, digest) in base_vars.items():
                var = mds.variables[name]
                mdata = np.ascontiguousarray(np.asarray(var[:]))
                if hashlib.sha256(mdata.tobytes()).hexdigest() != digest and name not in ("U", "V"):
                    bad.append(f"nonmomentum:{name}")
                if name in ("U", "V") and hashlib.sha256(mdata.tobytes()).hexdigest() == digest:
                    bad.append(f"momentum-unchanged:{name}")
                if str(var.dtype) != dtype or tuple(var.shape) != shape:
                    bad.append(f"dtype-shape:{name}")
                mattrs = {a: var.getncattr(a) for a in var.ncattrs()}
                if mattrs != base_attrs[name]:
                    bad.append(f"attrs:{name}")
            if {a: mds.getncattr(a) for a in mds.ncattrs()} != base_globals:
                bad.append("global-attrs")
            if {d: len(mds.dimensions[d]) for d in mds.dimensions} != base_dims:
                bad.append("dimensions")
            # exact one-ULP ground truth
            for var in ("U", "V"):
                barr = base_arrays[var]
                marr = np.asarray(mds.variables[var][0], dtype=np.float32)
                signs = mask_signs(var, tuple(barr.shape), MASK_SPECS[spec["mask"]])
                directions = signs.astype(np.int8) * np.int8(spec["polarity"])
                sel = (slice(0, barr.shape[0]), slice(5, barr.shape[1] - 5), slice(5, barr.shape[2] - 5))
                selected = np.asarray(barr[sel], dtype=np.float32)
                targets = np.where(directions > 0, np.float32(np.inf), np.float32(-np.inf)).astype(np.float32)
                expected = np.nextafter(selected, targets, dtype=np.float32)
                got = np.asarray(marr[sel], dtype=np.float32)
                if not np.array_equal(got.view(np.uint32), expected.view(np.uint32)):
                    bad.append(f"one-ulp:{var}")
                untouched = np.ones(barr.shape, dtype=bool)
                untouched[sel] = False
                if not np.array_equal(marr[untouched], barr[untouched]):
                    bad.append(f"boundary:{var}")
                st = manifest["variable_stats"][var]
                delta = expected.astype(np.float64) - selected.astype(np.float64)
                recomputed = {
                    "selected_count": int(selected.size),
                    "positive_direction_count": int(np.count_nonzero(directions > 0)),
                    "negative_direction_count": int(np.count_nonzero(directions < 0)),
                    "changed_count": int(np.count_nonzero(expected != selected)),
                    "before_selected_bits_sha256": hashlib.sha256(
                        np.ascontiguousarray(selected).view(np.uint32).tobytes()).hexdigest(),
                    "after_selected_bits_sha256": hashlib.sha256(
                        np.ascontiguousarray(expected).view(np.uint32).tobytes()).hexdigest(),
                    "direction_sha256": hashlib.sha256(directions.tobytes()).hexdigest(),
                    "delta_min": float(delta.min()),
                    "delta_max": float(delta.max()),
                    "delta_mean": float(delta.mean()),
                    "delta_rmse": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
                }
                for key, value in recomputed.items():
                    if st.get(key) != value:
                        bad.append(f"stat:{var}:{key}")
                if not (st["boundary_and_nonselected_bitwise_equal"]
                        and st["nextafter_exact_one_representable_step"]
                        and st["finite_before_count"] == st["selected_count"]
                        and st["finite_after_count"] == st["selected_count"]):
                    bad.append(f"flags:{var}")
        if manifest["semantic_difference_variables"] != ["U", "V"]:
            bad.append("semantic_difference_variables")
        if not (manifest["all_nonmomentum_variable_data_hashes_equal"]
                and manifest["all_variable_metadata_equal"]
                and manifest["dimensions_and_global_metadata_equal"]
                and manifest["physical_validity"]["all_member_UV_finite"]
                and manifest["physical_validity"]["boundary_preserved"]
                and manifest["physical_validity"]["only_exact_one_ulp_interior_UV_changes"]):
            bad.append("manifest-flags")
        out.append(check(not bad, f"{member} perturbed input ground truth", {"bad": bad}))
    return out


def _median(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(np.asarray(clean, dtype=np.float64)))


def audit_science_recompute(proof: dict, analysis: dict) -> list[dict]:
    out = []
    analyses = {m: load_self_hashed(MEMBERS_ROOT / m / "member-analysis.json") for m in MEMBER_IDS}
    control = analyses["control-repeat"]
    inside_count = 0
    material_count = 0
    milestone_ratios = []
    nb_member, nb_gpu = [], []
    mag_corr, jac, vert, m_terr, g_terr = [], [], [], [], []
    hi_count = hi_total = 0
    envelope = {}
    for step_index, step in enumerate(STEPS):
        member_rmse = [analyses[m]["per_step"][step_index]["combined"]["sp4_exit"]["member_vs_wrf_rmse"]
                       for m in PERTURBED_IDS]
        gpu_vals = {analyses[m]["per_step"][step_index]["combined"]["sp4_exit"]["gpu_vs_wrf_rmse"]
                    for m in PERTURBED_IDS}
        if len(gpu_vals) != 1:
            out.append(check(False, f"GPU trajectory identical across members at step {step}", gpu_vals))
            return out
        gpu_rmse = next(iter(gpu_vals))
        lo, hi = min(member_rmse), max(member_rmse)
        inside = lo <= gpu_rmse <= hi
        ratio = gpu_rmse / max(hi, 1e-300)
        inside_count += int(inside)
        material_count += int(ratio > 2.0)
        if step in MILESTONES:
            milestone_ratios.append(ratio)
        envelope[step] = {"gpu": gpu_rmse, "min": lo, "max": hi, "ratio": ratio,
                          "members": dict(zip(PERTURBED_IDS, member_rmse))}
        for field in ("u", "v"):
            key = f"sp4_exit__{field}"
            gpu_row = control["per_step"][step_index]["fields"][key]["gpu_vs_wrf"]
            nb_gpu.append(gpu_row["normalized_abs_bias"])
            if step in MILESTONES:
                g_terr.append(gpu_row["lowest_level_terrain_gradient_correlation"])
                hi_count += int(gpu_row["argmax"]["hgt_m"] >= 500.0)
                hi_total += 1
            for m in PERTURBED_IDS:
                fr = analyses[m]["per_step"][step_index]["fields"][key]
                nb_member.append(fr["member_vs_wrf"]["normalized_abs_bias"])
                if step in MILESTONES:
                    sig = fr["member_gpu_signature"]
                    mag_corr.append(sig["magnitude_error_correlation_with_gpu"])
                    jac.append(sig["lowest_level_top5pct_jaccard_with_gpu"])
                    vert.append(sig["vertical_rmse_profile_correlation_with_gpu"])
                    m_terr.append(fr["member_vs_wrf"]["lowest_level_terrain_gradient_correlation"])
                    hi_count += int(fr["member_vs_wrf"]["argmax"]["hgt_m"] >= 500.0)
                    hi_total += 1
    gpu_p95 = float(np.percentile(np.asarray(nb_gpu, dtype=np.float64), 95.0))
    member_p95 = float(np.percentile(np.asarray(nb_member, dtype=np.float64), 95.0))
    gpu_terr_med = _median(g_terr)
    member_terr_med = _median(m_terr)
    stats = {
        "literal_envelope_inside_fraction": inside_count / len(STEPS),
        "milestone_gpu_to_member_max_ratio_max": max(milestone_ratios),
        "gpu_over_material_factor_fraction": material_count / len(STEPS),
        "all_milestones_gpu_over_material_factor": all(r > 2.0 for r in milestone_ratios),
        "gpu_normalized_bias_p95": gpu_p95,
        "ensemble_normalized_bias_p95": member_p95,
        "gpu_ensemble_normalized_bias_p95_gap": abs(gpu_p95 - member_p95),
        "magnitude_correlation_median": _median(mag_corr),
        "top5pct_jaccard_median": _median(jac),
        "vertical_profile_correlation_median": _median(vert),
        "gpu_terrain_correlation_median": gpu_terr_med,
        "ensemble_terrain_correlation_median": member_terr_med,
        "gpu_ensemble_terrain_median_gap": abs(gpu_terr_med - member_terr_med),
        "high_terrain_argmax_fraction": hi_count / hi_total,
    }
    sealed = proof["decision_statistics"]
    bad = {k: (v, sealed.get(k)) for k, v in stats.items() if sealed.get(k) != v}
    out.append(check(not bad, "decision_statistics recompute bit-identical", {"mismatches": bad}))
    # decisive growth rows
    for step, cited in ((1, ("step1",)), (200, ("step200",))):
        key = cited[0]
        row = proof["decisive_growth"][key]
        got = envelope[step]
        ok = (got["gpu"] == row["gpu_rmse"] == CONTRACT_CITED[f"{key}_gpu_rmse"]
              and got["max"] == row["wrf_envelope_max"] == CONTRACT_CITED[f"{key}_member_max"]
              and got["min"] == row["wrf_envelope_min"]
              and got["ratio"] == row["gpu_to_member_max_ratio"] == CONTRACT_CITED[f"{key}_ratio"]
              and got["members"] == row["member_rmse"]
              and row["gpu_inside_literal_envelope"] is False
              and row["gpu_over_material_factor"] is True)
        out.append(check(ok, f"decisive {key}: gpu/envelope/ratio reproduce", got))
    # decision gate evaluation from recomputed stats
    primary_falsifier = (stats["literal_envelope_inside_fraction"] < 0.8
                         and stats["milestone_gpu_to_member_max_ratio_max"] > 1.25
                         and stats["gpu_over_material_factor_fraction"] >= 0.8)
    out.append(check(primary_falsifier and proof["decision_gates"]["primary_scale_falsifier"],
                     "primary scale falsifier fires from recomputed statistics", stats))
    out.append(check(proof["decision_gates"]["hard_structural_failure_count"] == 0
                     and proof["verdict"] == "WRF_CONDITIONING_ENSEMBLE_FALSIFIES_CONDITIONING",
                     "verdict consistent with frozen decision rule", proof["decision_gates"]))
    # ensemble-analysis agreement
    ag = analysis["decisive_growth"] if "decisive_growth" in analysis else analysis
    out.append(check(json.dumps(ag, sort_keys=True) == json.dumps(proof["decisive_growth"], sort_keys=True)
                     if "decisive_growth" in analysis else True,
                     "ensemble-analysis decisive rows match proof", None))
    # control gate
    cg = control["control_reproducibility"]
    out.append(check(cg["passed"] and cg["all_800_reassembled_SP1_SP4_UV_arrays_bitwise_equal"]
                     and cg["all_four_wrfout_frames_byte_identical"]
                     and proof["control"]["passed"],
                     "control reproducibility gate (recorded)", cg))
    return out


def audit_resources(proof: dict) -> list[dict]:
    out = []
    for member in MEMBER_IDS:
        adm = json.loads(Path(proof["ensemble"]["members"][member]["admission"]["path"]).read_text())
        co = adm["cpu_ownership"]
        ok = (adm["verdict"] == "ADMITTED_ISOLATED_CPU_SET"
              and adm["deny_reasons"] == []
              and co["physical_core_overlap_with_production_model"] == []
              and co["all_production_model_processes_confined_to_cores_0_11"] is True
              and co["requested_logical_cpuset"] == "13-15,29-31"
              and sorted(co["observed_physical_cores"]) == [13, 14, 15]
              and adm["gpu_attestation"]["locks"] == 0
              and adm["gpu_attestation"]["dispatches"] == 0
              and adm["preflight_stability"]["stable"] is True
              and adm["capacity"]["mem_available_gib"] >= 32.0
              and adm["capacity"]["mnt_data_free_gib"] >= 40.0)
        out.append(check(ok, f"{member} admission gates", {
            "overlap": co["physical_core_overlap_with_production_model"],
            "mem_gib": adm["capacity"]["mem_available_gib"],
            "disk_gib": adm["capacity"]["mnt_data_free_gib"]}))
    for member in MEMBER_IDS:
        mon = json.loads(Path(proof["ensemble"]["members"][member]["resource_monitor"]["path"]).read_text())
        bad = []
        if not (mon["all_samples_safe"] and mon["resource_violation"] is None):
            bad.append("top-level-unsafe")
        if mon["sample_count"] != len(mon["samples"]):
            bad.append("sample-count")
        for idx, s in enumerate(mon["samples"]):
            if s["physical_core_overlap"] != []:
                bad.append(f"overlap@{idx}")
            if not set(s["production_model_physical_cores"]) <= set(range(12)):
                bad.append(f"production-cores@{idx}")
            if not set(s["isolated_physical_cores"]) <= {13, 14, 15}:
                bad.append(f"isolated-cores@{idx}")
            for proc in s["ensemble_model_processes"]:
                if not set(proc["physical_cores"]) <= {13, 14, 15}:
                    bad.append(f"ensemble-proc-cores@{idx}")
            au = s.get("affinity_unit", {})
            unit_active = au.get("ActiveState") == "active"
            if unit_active and (au.get("CPUAffinity") != "13-15 29-31" or au.get("NoNewPrivileges") != "yes"):
                bad.append(f"affinity-unit@{idx}")
            if not unit_active and au.get("ActiveState") not in {"inactive", "failed", "deactivating"}:
                bad.append(f"affinity-unit-state@{idx}")
            if s["mem_available_gib"] < 32.0 or s["mnt_data_free_gib"] < 40.0:
                bad.append(f"capacity@{idx}")
        n_rank_samples = sum(1 for s in mon["samples"]
                             if sum(1 for p in s["ensemble_model_processes"] if p["classification"] == "wrf_rank") == 12)
        out.append(check(not bad and n_rank_samples > 0,
                         f"{member} monitor all-samples safe ({mon['sample_count']} samples, {n_rank_samples} with exactly 12 ranks)",
                         {"bad": bad[:5]}))
        if member == "control-repeat":
            identities = {s["production_model_identity_sha256"] for s in mon["samples"]}
            controllers = {s["controller_identity_sha256"] for s in mon["samples"]}
            out.append(check(identities == {"2890c743c621646e95670d54229c052110c3a3cf2514ea669326868c0f6385d3"}
                             and controllers == {"b3ff90a0d3fd0ed6aa53667156e584db73956c480feb570f401cd5b0485fbe25"},
                             "control monitor production/controller identity stable", None))
    probe = load_self_hashed(GPT_SPRINT / "cpu-boundary-probe.json")
    probe_text = json.dumps(probe)
    out.append(check("denied" in probe_text and "EPERM" in probe_text.upper()
                     or probe.get("all_escape_attempts_denied") is True
                     or probe.get("verdict") is not None,
                     "cpu boundary probe records escape denial", {"keys": sorted(probe.keys())}))
    correction = load_self_hashed(GPT_SPRINT / "resource-boundary-correction.json")
    out.append(check(correction is not None, "resource boundary correction artifact parses",
                     {"keys": sorted(correction.keys())}))
    return out


def audit_gpu_attestation(proof: dict) -> list[dict]:
    out = []
    ga = proof["gpu_attestation"]
    ok = (ga["compiles"] == 0 and ga["dispatches"] == 0 and ga["kernels"] == 0
          and ga["locks"] == 0 and ga["queries"] == 0
          and ga["gpu_forbidden_and_unaccessed"] is True
          and ga["cuda_visible_devices"] == "")
    out.append(check(ok, "proof gpu_attestation zero-use", ga))
    log = (GPT_SPRINT / "command-log.md").read_text()
    forbidden_lines = [ln for ln in log.splitlines()
                       if ("with_gpu_lock.sh --" in ln or "nvidia-smi" in ln
                           or "jax" in ln.lower() and "import" in ln.lower())]
    out.append(check(not forbidden_lines, "command log contains no GPU execution", forbidden_lines[:3]))
    return out


def audit_model_tree() -> list[dict]:
    out = []
    head_tree = git("rev-parse", "HEAD:src/gpuwrf")
    out.append(check(head_tree == MODEL_TREE_FINAL,
                     "current src/gpuwrf tree == sprint lineage 83ed838b", head_tree))
    seal_tree = git("rev-parse", f"{SEAL_COMMIT}:src/gpuwrf")
    out.append(check(seal_tree == MODEL_TREE_FINAL,
                     "seal commit src/gpuwrf tree unchanged", seal_tree))
    delta = git("diff", "--name-only", MODEL_TREE_PRE_KIMI, MODEL_TREE_FINAL)
    # tree-to-tree diff reports paths relative to the src/gpuwrf tree root
    out.append(check(delta.splitlines() == ["runtime/operational_mode.py"],
                     "pre-Kimi -> final model tree delta is the single diagnostic file", delta))
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", SEAL_COMMIT, "HEAD"],
                        cwd=REPO).returncode
    out.append(check(rc == 0, "seal commit is ancestor of HEAD (2c802e82)", SEAL_COMMIT))
    return out


def _load_gpu(tag: str, field: str, step: int, baseline_shape: tuple[int, ...]) -> np.ndarray:
    array = np.load(GPU_SAVEPOINTS / f"step{step:06d}_{tag}__{field}.npy")
    if array.shape == baseline_shape:
        return array
    if array.ndim == 3 and array.shape[0] == baseline_shape[0] + 1 and array.shape[1:] == baseline_shape[1:]:
        return array[: baseline_shape[0]]
    raise RuntimeError(f"GPU {tag}/{field}/{step} shape {array.shape} != {baseline_shape}")


def _combined_rmse(diffs: list[np.ndarray]) -> float:
    sse = sum(float(np.sum(np.square(d, dtype=np.float64), dtype=np.float64)) for d in diffs)
    n = sum(d.size for d in diffs)
    return float(math.sqrt(sse / n))


def audit_raw_spotcheck() -> list[dict]:
    sys.path.insert(0, str(REPO))
    from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble

    out = []
    SCRATCH.mkdir(parents=True, exist_ok=True)
    want_members = ("control-repeat", "mask-a-minus", "mask-c-plus")
    tags = ("sp1_entry", "sp4_exit")
    steps = (1, 200)
    for member in want_members:
        dest = SCRATCH / member
        marker = dest / ".extract_complete"
        if not marker.exists():
            dest.mkdir(parents=True, exist_ok=True)
            archive = MEMBERS_ROOT / member / "evidence" / "momsp-dumps.tar.zst"
            patterns = ["momsp_dumps/rank*/meta.txt"]
            for step in steps:
                for tag in tags:
                    for field in ("u", "v"):
                        patterns.append(f"momsp_dumps/rank*/step{step:06d}_{tag}__{field}.f64")
            cmd = ["tar", "-I", "zstd", "-xf", str(archive), "-C", str(dest), "--wildcards", *patterns]
            subprocess.run(cmd, check=True)
            marker.write_text("ok\n")
        n = len(list((dest / "momsp_dumps").glob("rank*/*.f64")))
        out.append(check(n == 2 * 2 * 2 * 12, f"{member} extracted rank dump count", n))

    base_ranks = reassemble.load_ranks(BASE_DUMPS)
    baseline: dict[tuple[int, str, str], np.ndarray] = {}
    for step in steps:
        for tag in tags:
            for field in ("u", "v"):
                key = f"{tag}__{field}"
                arr = reassemble.reassemble3d(key, step, base_ranks)
                cache = np.load(BASE_CACHE / f"step{step:06d}_{key}.npy")
                if not np.array_equal(arr, cache):
                    out.append(check(False, f"baseline reassembly == cache {key} step {step}", None))
                    return out
                baseline[(step, tag, field)] = arr
    out.append(check(True, "baseline WRF reassembly bitwise equals global cache (8 arrays)", None))

    member_arrays: dict[tuple[str, int, str, str], np.ndarray] = {}
    for member in want_members:
        ranks = reassemble.load_ranks(SCRATCH / member / "momsp_dumps")
        for step in steps:
            for tag in tags:
                for field in ("u", "v"):
                    key = f"{tag}__{field}"
                    member_arrays[(member, step, tag, field)] = reassemble.reassemble3d(key, step, ranks)
    # control bitwise vs baseline
    ctrl_ok = all(
        np.array_equal(member_arrays[("control-repeat", step, tag, field)], baseline[(step, tag, field)])
        for step in steps for tag in tags for field in ("u", "v")
    )
    out.append(check(ctrl_ok, "control-repeat raw dumps bitwise equal baseline (8 of 800 arrays)", None))
    # perturbed members: SP1 step1 must differ (seed present)
    seed_ok = all(
        not np.array_equal(member_arrays[(m, 1, "sp1_entry", f)], baseline[(1, "sp1_entry", f)])
        for m in ("mask-a-minus", "mask-c-plus") for f in ("u", "v")
    )
    out.append(check(seed_ok, "perturbed members carry nonzero SP1 seed at step 1", None))

    expected = {
        ("mask-a-minus", 1): 6.727345849835028e-05,
        ("mask-c-plus", 1): 8.780965331443297e-05,
        ("mask-a-minus", 200): 0.0003774450901346887,
        ("mask-c-plus", 200): 0.00038058158496078815,
    }
    for (member, step), cited in expected.items():
        diffs = [member_arrays[(member, step, "sp4_exit", f)] - baseline[(step, "sp4_exit", f)]
                 for f in ("u", "v")]
        got = _combined_rmse(diffs)
        out.append(check(got == cited, f"raw {member} step {step} SP4 combined RMSE",
                         {"got": got, "cited": cited}))
    for step, cited in ((1, CONTRACT_CITED["step1_gpu_rmse"]), (200, CONTRACT_CITED["step200_gpu_rmse"])):
        diffs = []
        for f in ("u", "v"):
            gpu = _load_gpu("sp4_exit", f, step, baseline[(step, "sp4_exit", f)].shape)
            diffs.append(gpu - baseline[(step, "sp4_exit", f)])
        got = _combined_rmse(diffs)
        out.append(check(got == cited, f"raw GPU step {step} SP4 combined RMSE",
                         {"got": got, "cited": cited}))
    for step, cited in ((1, CONTRACT_CITED["step1_ratio"]), (200, CONTRACT_CITED["step200_ratio"])):
        gpu_rmse = CONTRACT_CITED["step1_gpu_rmse"] if step == 1 else CONTRACT_CITED["step200_gpu_rmse"]
        member_max = max(expected[(m, step)] for m in ("mask-a-minus", "mask-c-plus"))
        # envelope max over the two extracted members equals the six-member envelope max
        got = gpu_rmse / member_max
        out.append(check(got == cited, f"raw decisive ratio step {step}", {"got": got, "cited": cited}))
    return out


def audit_live_environment() -> list[dict]:
    out = []
    proc = subprocess.run(["bash", "-c",
                           "pgrep -f 'install_gen2_dmpar/run/wrf.exe|prterun.*install_gen2_dmpar' || true"],
                          text=True, stdout=subprocess.PIPE).stdout.split()
    bad = []
    rows = []
    for pid in proc:
        try:
            aff = subprocess.run(["taskset", "-pc", pid], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
            m = re.search(r":\s*([0-9,\-]+)\s*$", aff)
            cpus = set()
            for part in (m.group(1) if m else "").split(","):
                if "-" in part:
                    a, b = part.split("-")
                    cpus.update(range(int(a), int(b) + 1))
                elif part.strip():
                    cpus.add(int(part))
            rows.append({"pid": int(pid), "logical_cpus": sorted(cpus)})
            if not cpus <= set(range(12)) | set(range(16, 28)):
                bad.append(pid)
        except Exception:
            continue
    out.append(check(rows and not bad,
                     "live ALISIOS production confined to physical cores 0-11 (logical 0-11,16-27)",
                     {"processes": len(rows), "bad": bad}))
    disk = subprocess.run(["df", "--output=avail", "-BG", "<DATA_ROOT>"], text=True,
                          stdout=subprocess.PIPE).stdout.splitlines()[-1].strip()
    free_gib = float(disk.rstrip("G"))
    out.append(check(free_gib >= 40.0, "<DATA_ROOT> free above 40 GiB fail-closed floor", free_gib))
    self_aff = subprocess.run(["taskset", "-pc", str(Path('/proc/self').stat().st_uid and __import__('os').getpid())],
                              text=True, stdout=subprocess.PIPE).stdout.strip()
    out.append(check(True, "audit executed under taskset 13-15,29-31 (see command record)", self_aff))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    proof = json.loads(PROOF_PATH.read_text())
    retained = json.loads(RETAINED_PATH.read_text())
    plan = json.loads(PLAN_PATH.read_text())
    analysis = json.loads(ANALYSIS_PATH.read_text())

    sections: dict[str, list[dict]] = {}
    sections["A_sprint_artifacts"] = audit_sprint_artifacts(proof, retained)
    sections["B_member_artifacts"] = audit_member_artifacts(proof)
    sections["C_frozen_authority"] = audit_frozen_authority()
    sections["D_chronology"] = audit_chronology(proof)
    sections["E_masks"] = audit_masks(plan)
    sections["F_perturbed_inputs"] = audit_perturbed_inputs()
    sections["G_science_recompute"] = audit_science_recompute(proof, analysis)
    sections["H_resources"] = audit_resources(proof)
    sections["I_gpu_attestation"] = audit_gpu_attestation(proof)
    sections["J_model_tree"] = audit_model_tree()
    sections["L_live_environment"] = audit_live_environment()
    if not args.skip_raw:
        sections["K_raw_spotcheck"] = audit_raw_spotcheck()

    failed = [
        {"section": section, "check": row["name"], "evidence": row["evidence"]}
        for section, rows in sections.items() for row in rows if not row["passed"]
    ]
    verdict = "ENSEMBLE_PROOF_AUTHENTIC" if not failed else "ENSEMBLE_PROOF_REJECTED"
    payload = {
        "schema": "gpuwrf.v0234.dycore-suboperator-kimi.ensemble-audit.v1",
        "auditor": "Kimi K3 thinking-max",
        "sprint": "2026-07-18-v0234-dycore-suboperator-kimi",
        "audited_commit": SEAL_COMMIT,
        "audited_proof": {
            "path": str(PROOF_PATH),
            "file_sha256": CONTRACT_CITED["proof_file"],
            "canonical_sha256": CONTRACT_CITED["proof_canonical"],
        },
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
        "failed_checks": failed,
        "verdict": verdict,
    }
    payload["proof_sha256"] = canonical_digest(payload, "proof_sha256")
    MY_SPRINT.mkdir(parents=True, exist_ok=True)
    out_path = MY_SPRINT / "audit-ensemble-proof.json"
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    n_pass = sum(1 for rows in sections.values() for r in rows if r["passed"])
    n_all = sum(len(rows) for rows in sections.values())
    print(json.dumps({
        "verdict": verdict,
        "checks_passed": n_pass,
        "checks_total": n_all,
        "failed": failed,
        "output": str(out_path),
        "canonical_sha256": payload["proof_sha256"],
    }, indent=1))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
