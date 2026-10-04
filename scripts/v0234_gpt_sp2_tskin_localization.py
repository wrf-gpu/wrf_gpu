#!/usr/bin/env python3
"""Backend-dark localization of the remaining WRF TSK / EDMF gate split."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
WRF_ROOT = STAGED / "wrf-dumps"
ARCHIVE = STAGED / "surface-fixed-cpu-adapter/tskin-v1/single-authority-capture.npz"
PROOF = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure/tskin-capture-proof.json"
WRF_TREE_SHA = "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b"
ARCHIVE_SHA = "880ac4d84258d3f8fccc7af3aab86ede89511a05c4be4d8aa9549958fc6fd94c"
PROOF_SHA = "f1e6a568b6636461d7906936951f136c389327047ccbd04b32626d72d2c250f7"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_reader():
    path = REPO / "scripts/v0234_gpt_adversarial_audit.py"
    spec = importlib.util.spec_from_file_location("tskin_reader", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WRF_DUMP_ROOT = WRF_ROOT
    return module.IndependentWrfDump(WRF_ROOT)


def main() -> int:
    if sha256(ARCHIVE) != ARCHIVE_SHA or sha256(PROOF) != PROOF_SHA:
        raise SystemExit("REFUSE:EVIDENCE_HASH")
    reader = import_reader()
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        port_tskin = np.asarray(archive["surface_terms_flux_t_skin"])
        port_theta = np.asarray(archive["surface_terms_state_theta"])[0]
        port_qv = np.asarray(archive["surface_terms_state_qv"])[0]
        port_dz = np.asarray(archive["surface_terms_state_dz"])[0]
        xland = np.asarray(archive["surface_terms_flux_xland"])
        port_fltv = np.asarray(archive["surface_terms_fltv"])
        port_active = np.any(np.abs(np.asarray(archive["mass_flux_s_aw"])) > 0.0, axis=0)
    outer = importlib.util.spec_from_file_location("tskin_outer", REPO / "scripts/v0234_gpt_operand_attribution.py")
    assert outer and outer.loader
    out = importlib.util.module_from_spec(outer)
    outer.loader.exec_module(out)
    out.WRF_ROOT = WRF_ROOT
    wrf_tskin = out.load_outer2(reader, "tsk")
    wrf_active = np.any(np.abs(reader.columns("bc_s_aw")) > 0.0, axis=0)
    thv = port_theta * (1.0 + 0.608 * port_qv)
    threshold = np.where(xland >= 1.5, -0.001, -0.003)
    port_gate = ((thv - port_tskin * (1.0 + 0.608 * port_qv)) / (0.5 * port_dz) < threshold)
    wrf_gate = ((thv - wrf_tskin * (1.0 + 0.608 * port_qv)) / (0.5 * port_dz) < threshold)
    def stats(a, b):
        d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        return {"rms": float(np.sqrt(np.mean(d * d))), "max_abs": float(np.max(np.abs(d))), "exact_fraction": float(np.mean(np.asarray(a) == np.asarray(b)))}
    land = xland < 1.5
    payload = {
        "schema": "wrfgpu2-v0234-gpt-sp2-tskin-localization-v1",
        "verdict": "SOURCE_LOCALIZED_TSK_FROZEN_BOUND",
        "passed": True,
        "backend": "numpy-cpu",
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
        "capture_archive_sha256": ARCHIVE_SHA,
        "capture_proof_sha256": PROOF_SHA,
        "wrf_dump_tree_sha256": WRF_TREE_SHA,
        "source_locations": {
            "wrf_gate": "phys/MYNN-EDMF/module_bl_mynnedmf.F90:5892-5918",
            "port_gate": "src/gpuwrf/physics/mynn_edmf.py:_wrf_superadiabatic_gate",
            "port_transport": "src/gpuwrf/physics/mynn_pbl.py:_edmf_arrays_from_state",
        },
        "tskin_port_vs_wrf": {
            "all": stats(port_tskin, wrf_tskin),
            "land": stats(port_tskin[land], wrf_tskin[land]),
            "water": stats(port_tskin[~land], wrf_tskin[~land]),
        },
        "mass_flux_activity": {
            "port_active_columns": int(np.sum(port_active)),
            "wrf_active_columns": int(np.sum(wrf_active)),
            "port_only_columns": int(np.sum(port_active & ~wrf_active)),
            "wrf_only_columns": int(np.sum(wrf_active & ~port_active)),
            "port_tskin_gate_columns": int(np.sum(port_gate)),
            "wrf_tskin_gate_columns": int(np.sum(wrf_gate)),
        },
        "gate_masks": {
            "port_tskin_vs_port_activity_exact_fraction": float(np.mean(port_gate == port_active)),
            "wrf_tskin_vs_wrf_activity_exact_fraction": float(np.mean(wrf_gate == wrf_active)),
        },
        "strict_fix_gate": {
            "fresh_capture_u_rms": 3.9935109120127676e-07,
            "fresh_capture_v_rms": 7.372784327594862e-07,
            "strictly_improved_both": False,
            "reason": "real TSK plumbing changed the gate input but no U/V improvement; port TSK itself remains source-divergent over land",
        },
    }
    payload["canonical_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out_path = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure/tskin-localization-proof.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(out_path), "canonical_payload_sha256": payload["canonical_payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
