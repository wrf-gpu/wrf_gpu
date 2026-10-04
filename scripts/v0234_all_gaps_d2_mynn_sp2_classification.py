#!/usr/bin/env python3
"""D2 — MYNN SP2 lane classification (v0234 all-gaps, mechanism M5).

Preregistered in `.agent/sprints/2026-07-19-v0234-all-remaining-gaps-fable5/
PREREGISTRATION-D1-D2.md` (commit 690d32b4) as amended by
PREREGISTRATION-AMENDMENT-01.md (commit cca4a428) BEFORE execution.

Arms: B0 harness gate (CPU capture reproduces retained GPU SP2), B1
cross-backend sample, B2 six 1-fp32-ULP input-dither members, R target
(retained GPU SP2 vs WRF SP2). CPU-only; zero GPU actions; no production
file edited.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPRINT = ROOT / ".agent/sprints/2026-07-19-v0234-all-remaining-gaps-fable5"
EVIDENCE_BASE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a"
)
CARRY = EVIDENCE_BASE / "v0234_1500_science_60659a2e_terminal2/science-runtime/failure/last-healthy-d03-step-0.pkl"
CARRY_SHA = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
FIX_NS = EVIDENCE_BASE / "nested_stage_omega_transport_470e6111_s1_diff6_fix_validation_gpt1"
FIX_PROOF = FIX_NS / "s1-diff6-validation-terminal-proof.json"
FIX_ROWS_SHA = "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a"
WRF_CACHE = Path("<DATA_ROOT>/wrf_gpu2/v0234_first_interval_momentum_kimi/compare/wrf_global_cache")

REQUIRED_ENV = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "JAX_CPU_ENABLE_ASYNC_DISPATCH": "false",
    "GPUWRF_JAX_CACHE": "0",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_NESTED_FROZEN_WRF_BOUNDARY_BUNDLE": "1",
}
ADMITTED_CPUS = {13, 14, 15, 29, 30, 31}

B0_GATE_RMS = 1e-5
CONDITIONING_FACTOR = 2.0
SYSTEMATIC_FACTOR = 10.0
KPROFILE_CORR_MIN = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rms(a) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype=np.float64)))))


def k_profile(a) -> list[float]:
    import numpy as np

    return [float(v) for v in np.mean(np.abs(np.asarray(a, dtype=np.float64)), axis=(1, 2))]


def main() -> int:
    started = time.time()
    actual_env = {name: os.environ.get(name) for name in REQUIRED_ENV}
    if actual_env != REQUIRED_ENV:
        raise SystemExit(f"STOP: environment mismatch {actual_env!r}")
    affinity = set(os.sched_getaffinity(0))
    if not affinity <= ADMITTED_CPUS:
        raise SystemExit(f"STOP: affinity {sorted(affinity)} outside admitted lane")
    if sha256_file(CARRY) != CARRY_SHA:
        raise SystemExit("STOP: carry hash mismatch")

    fix_proof = json.loads(FIX_PROOF.read_text())
    manifest = fix_proof["savepoint_manifest"]
    if manifest["rows_sha256"] != FIX_ROWS_SHA:
        raise SystemExit("STOP: fixed-run savepoint rows hash mismatch")
    rows = {(row["step"], row["tag"], row["field"]): row for row in manifest["rows"]}

    import numpy as np

    def load_fixed(step: int, tag: str, field: str) -> np.ndarray:
        row = rows[(step, tag, field)]
        path = Path(row["path"])
        if sha256_file(path) != row["file_sha256"]:
            raise SystemExit(f"STOP: savepoint hash mismatch {path.name}")
        arr = np.load(path)
        if list(arr.shape) != list(row["shape"]):
            raise SystemExit(f"STOP: savepoint shape mismatch {path.name}")
        return np.asarray(arr, dtype=np.float64)

    gpu_sp2 = {f: load_fixed(1, "sp2_pbl", f) for f in ("rublten", "rvblten")}
    wrf_sp2 = {}
    for f in ("rublten", "rvblten"):
        arr = np.load(WRF_CACHE / f"step000001_sp2_pbl__{f}.npy")
        wrf_sp2[f] = np.asarray(arr, dtype=np.float64)
        if wrf_sp2[f].shape != gpu_sp2[f].shape:
            raise SystemExit(f"STOP: WRF/GPU sp2 shape mismatch {f}")

    import jax

    jax.config.update("jax_cpu_enable_async_dispatch", False)
    import jax.numpy as jnp

    if jax.default_backend() != "cpu":
        raise SystemExit(f"STOP: backend {jax.default_backend()}")

    import gpuwrf.contracts.state as state_contract

    state_contract._gpu_device = lambda: jax.devices("cpu")[0]
    import gpuwrf.runtime.operational_mode as runtime
    from scripts import v0234_corrected_ni_ordinary_bisection as ordinary

    scratch = Path(tempfile.mkdtemp(prefix="v0234-all-gaps-d2-"))
    tree, names, _initial, dt_by_domain, _load_authority = ordinary.load_corrected_tree(scratch / "load")
    if names != ("d01", "d02", "d03") or dt_by_domain != {"d01": 54.0, "d02": 18.0, "d03": 6.0}:
        raise SystemExit("STOP: canonical hierarchy changed")
    namelist = tree.domains["d03"].namelist
    clock = runtime.build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)

    with open(CARRY, "rb") as fh:
        carry = pickle.load(fh)

    def capture_sp2(c) -> dict[str, np.ndarray]:
        result = runtime.advance_one_step_with_first_interval_capture(
            c, namelist, 1, clock, cadence=cadence
        )
        record = result.record
        return {
            "rublten": np.asarray(jax.device_get(record.rublten), dtype=np.float64),
            "rvblten": np.asarray(jax.device_get(record.rvblten), dtype=np.float64),
        }

    print("D2: compiling + running baseline capture", flush=True)
    t0 = time.time()
    cpu_sp2 = capture_sp2(carry)
    baseline_seconds = time.time() - t0
    print(f"D2: baseline capture done in {baseline_seconds:.1f}s", flush=True)

    for name, arr in cpu_sp2.items():
        if not np.isfinite(arr).all():
            raise SystemExit(f"STOP: nonfinite CPU capture {name}")

    # B0 harness gate + B1 cross-backend sample.
    r_backend = {f: rms(cpu_sp2[f] - gpu_sp2[f]) for f in cpu_sp2}
    b0_green = all(v <= B0_GATE_RMS for v in r_backend.values())
    if not b0_green:
        payload = {
            "schema": "gpuwrf.v0234.all-gaps-d2-mynn-sp2.v1",
            "verdict": "D2_HARNESS_BLOCKER_B0_RED",
            "r_backend_rms": r_backend,
            "b0_gate": B0_GATE_RMS,
        }
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["self_sha256"] = hashlib.sha256(body).hexdigest()
        (SPRINT / "d2-mynn-sp2-classification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=1))
        return 3

    # B2 dither ensemble: one fp32 ULP on fp64 storage, full domain.
    def ulp32(x: np.ndarray) -> np.ndarray:
        mag = np.maximum(np.abs(x), 2.0 ** -126)
        return np.exp2(np.floor(np.log2(mag)) - 23.0)

    st = carry.state
    members = []
    for leaf, sign in (("theta", +1), ("theta", -1), ("qv", +1), ("qv", -1), ("p_perturbation", +1), ("p_perturbation", -1)):
        base = np.asarray(getattr(st, leaf), dtype=np.float64)
        dithered = base + sign * ulp32(base)
        new_state = st.replace(**{leaf: jnp.asarray(dithered)})
        members.append((f"{'plus' if sign > 0 else 'minus'}_{leaf}", carry.replace(state=new_state)))

    envelope = {}
    for label, member_carry in members:
        t0 = time.time()
        member_sp2 = capture_sp2(member_carry)
        envelope[label] = {
            f: rms(member_sp2[f] - cpu_sp2[f]) for f in member_sp2
        }
        envelope[label]["seconds"] = time.time() - t0
        envelope[label]["kprofile_rublten"] = k_profile(member_sp2["rublten"] - cpu_sp2["rublten"])
        print(f"D2: member {label} done ({envelope[label]['rublten']:.3e}/{envelope[label]['rvblten']:.3e})", flush=True)

    E = {f: max(envelope[m][f] for m in envelope) for f in ("rublten", "rvblten")}
    comparator = {f: max(E[f], r_backend[f]) for f in E}

    # R target: retained GPU SP2 vs WRF SP2 (step 1).
    delta_model = {f: gpu_sp2[f] - wrf_sp2[f] for f in gpu_sp2}
    r_model = {f: rms(delta_model[f]) for f in delta_model}

    xland = np.asarray(st.xland, dtype=np.float64)
    land = xland < 1.5
    structure = {}
    for f, d in delta_model.items():
        prof = k_profile(d)
        structure[f] = {
            "kprofile": prof,
            "k0_4_rms": rms(d[:5]),
            "k5_up_rms": rms(d[5:]),
            "land_rms": rms(d[:, land]),
            "sea_rms": rms(d[:, ~land]),
        }

    # Structural comparison: correlation of |delta| k-profiles, model vs the
    # strongest envelope member.
    strongest = max(envelope, key=lambda m: envelope[m]["rublten"])
    prof_model = np.asarray(structure["rublten"]["kprofile"])
    prof_member = np.asarray(envelope[strongest]["kprofile_rublten"])
    kprofile_corr = float(np.corrcoef(prof_model, prof_member)[0, 1])

    ratio = {f: (r_model[f] / comparator[f] if comparator[f] > 0 else math.inf) for f in r_model}
    max_ratio = max(ratio.values())
    min_ratio = min(ratio.values())

    if max_ratio <= CONDITIONING_FACTOR and kprofile_corr >= KPROFILE_CORR_MIN:
        verdict = "D2_SP2_LANE_CONDITIONING_BOUNDED"
    elif min_ratio >= SYSTEMATIC_FACTOR:
        verdict = "D2_SP2_LANE_SYSTEMATIC_SOURCE_DIFFERENCE"
    else:
        verdict = "D2_SP2_LANE_MIXED_UNRESOLVED"

    payload = {
        "schema": "gpuwrf.v0234.all-gaps-d2-mynn-sp2.v1",
        "preregistration_commits": ["690d32b4", "cca4a428"],
        "authorities": {
            "carry_sha256": CARRY_SHA,
            "fixed_rows_sha256": FIX_ROWS_SHA,
            "wrf_cache": str(WRF_CACHE),
            "namelist_radiation_cadence": cadence,
        },
        "b0_harness_gate": {"green": True, "gate_rms": B0_GATE_RMS, "r_backend_rms": r_backend},
        "b2_envelope": envelope,
        "comparator_max_E_or_backend": comparator,
        "r_model_rms": r_model,
        "ratio_model_over_comparator": ratio,
        "kprofile_corr_model_vs_strongest_member": kprofile_corr,
        "structure": structure,
        "decision_rule": {
            "conditioning_max_ratio": CONDITIONING_FACTOR,
            "systematic_min_ratio": SYSTEMATIC_FACTOR,
            "kprofile_corr_min": KPROFILE_CORR_MIN,
        },
        "verdict": verdict,
        "timing_seconds": {"baseline": baseline_seconds, "total": time.time() - started},
        "resource_attestation": {
            "cpu_only": True,
            "jax_backend": jax.default_backend(),
            "gpu_queries": 0,
            "gpu_locks": 0,
            "gpu_compiles": 0,
            "gpu_dispatches": 0,
            "affinity": sorted(affinity),
        },
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["self_sha256"] = hashlib.sha256(body).hexdigest()
    (SPRINT / "d2-mynn-sp2-classification.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": verdict,
        "r_backend": r_backend,
        "E": E,
        "r_model": r_model,
        "ratio": ratio,
        "kprofile_corr": kprofile_corr,
        "self": payload["self_sha256"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
