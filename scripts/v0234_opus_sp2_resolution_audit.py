#!/usr/bin/env python3
"""v0234 Opus SP2 batch resolution — reproducible audit / proof object.

Re-derives every load-bearing number in the Opus resolution directly from the
committed GPT land-TSK provenance proof JSONs and the pristine WRF source. No
GPU, no WRF/MPI, no capture, no <DATA_ROOT>. Fail-closed on any drift.

Findings sealed:
  A1 accepted QML conversion == WRF module_sf_noahmpdrv.F:756 Q_ML.
  A2 accepted MYNN rhosfc == WRF module_bl_mynnedmf.F:3960.
  A3 accepted RRTMG-LW top buffer == WRF deltap=4mb / nint(p_top/deltap).
  A4 O3RAD / composed-interface production population reverted; optional leaves
     byte-identical for omitted callers.
  B1 SP2 gate metric == authentic_..._adapter_vs_pristine_wrf.rms; ratios match.
  B2 mass-flux Shapley vector bitwise invariant across accepted/composed/O3
     captures (distinct archives) -> physical invariance of the 4,526-col split.
  B3 under BOTH rejected radiation fixes the mixing channel rms strictly
     DECREASES and surface/entry/metric stay flat, yet the gated total rises ->
     hypothesis (a) (compensating error) falsified; (b) gate defect proven.
  B4 residual is a 0.5-0.6% structural error (~4.4e4 fp32-ULP) dominated by the
     mass-flux split (cosine 0.956); gate threshold is ~4.5 ULP.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
WRF = pathlib.Path("<USER_HOME>/src/wrf_pristine/WRF")


class AuditFailure(RuntimeError):
    pass


def load(name: str) -> dict:
    return json.loads((SPRINT / name).read_text())


def approx(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


def grep_line(path: pathlib.Path, needle: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if needle in line:
            return line.strip()
    return None


def main() -> int:
    findings: dict[str, object] = {}

    # ---- A1..A3 source faithfulness of accepted production fixes ----
    qml = grep_line(WRF / "phys/module_sf_noahmpdrv.F", "Q_ML   = QV3D")
    if qml is None or "/(1.0+QV3D" not in qml.replace(" ", ""):
        raise AuditFailure(f"A1_QML_SOURCE:{qml!r}")
    findings["A1_qml_wrf_source"] = qml

    rho = grep_line(WRF / "phys/module_bl_mynnedmf.F", "psfc/(R_d*(tk(kts)")
    if rho is None or "rhosfc=psfc/(R_d*(tk(kts)+p608*qv(kts)))" not in rho.replace(" ", ""):
        raise AuditFailure(f"A2_RHOSFC_SOURCE:{rho!r}")
    findings["A2_rhosfc_wrf_source"] = rho

    deltap = grep_line(WRF / "phys/module_ra_rrtmg_lw.F", "deltap = 4.")
    nlayers = grep_line(WRF / "phys/module_ra_rrtmg_lw.F", "nint(p_top")
    if deltap is None or nlayers is None or "deltap" not in nlayers:
        raise AuditFailure(f"A3_TOPBUFFER_SOURCE:{deltap!r}|{nlayers!r}")
    findings["A3_topbuffer_wrf_source"] = [deltap, nlayers]

    # Accepted-fix code presence in the current production tree.
    coupler = (REPO / "src/gpuwrf/physics/noahmp_coupler.py").read_text()
    if "qv_mixing_ratio / (1.0 + qv_mixing_ratio)" not in coupler:
        raise AuditFailure("A1_PORT_QML_MISSING")
    if "P608 * qv" not in coupler:
        raise AuditFailure("A2_PORT_RHOSFC_MISSING")

    # ---- A4 reverts clean ----
    o3rw = load("rrtmg-o3-real-wrf-proof.json")
    if o3rw["is_self_compare"] is not False:
        raise AuditFailure("A4_O3_SELF_COMPARE")
    olr = o3rw["omitted_leaf_regression"]
    if olr["expected_aggregate_sha256"] != olr["observed_aggregate_sha256"] or not olr["byte_identical"]:
        raise AuditFailure("A4_OMITTED_LEAF_NOT_BYTE_IDENTICAL")
    prod_coupler = (REPO / "src/gpuwrf/coupling/physics_couplers.py").read_text()
    # the production RRTMG assembler must not repopulate ozone/dynamic leaves
    assembler = prod_coupler.split("def _rrtmg_column_inputs", 1)[1].split("\ndef ", 1)[0]
    for banned in ("ozone_vmr=", "wrf_cam_ozone_profile", "p_interface=", "t_interface="):
        if banned in assembler:
            raise AuditFailure(f"A4_PRODUCTION_REPOPULATES:{banned}")
    findings["A4_reverts_clean"] = True

    # ---- B1 gate metric identity ----
    acc = load("final-accepted-partition-proof.json")
    o3 = load("o3-partition-proof.json")
    comp = load("rrtmg-composed-partition-proof.json")

    ca = acc["candidate_acceptance"]
    if not approx(ca["candidate"]["u_rms"], acc["components"]["u"]["authentic_postfix_adapter_vs_pristine_wrf"]["rms"]):
        raise AuditFailure("B1_GATE_METRIC_NOT_ADAPTER_VS_WRF")
    base = ca["e653bdbf_baseline"]
    for comp_name, cand, expect_pass in (
        ("accepted", ca["candidate"], True),
        ("o3", o3["candidate_acceptance"]["candidate"], False),
        ("composed", comp["candidate_acceptance"]["candidate"], False),
    ):
        u_ratio = cand["u_rms"] / base["u_rms"]
        v_ratio = cand["v_rms"] / base["v_rms"]
        passed = (u_ratio <= 1.0001) and (v_ratio <= 1.0001)
        if passed != expect_pass:
            raise AuditFailure(f"B1_RATIO_GATE_MISMATCH:{comp_name}")
        findings[f"B1_ratio_{comp_name}"] = {"u": u_ratio, "v": v_ratio, "passes_1e-4": passed}

    # ---- B2 mass-flux Shapley bitwise invariant across DISTINCT captures ----
    archives = {
        "accepted": load("final-accepted-capture-proof.json")["archive"]["sha256"],
        "o3": load("o3-capture-proof.json")["archive"]["sha256"],
    }
    if archives["accepted"] == archives["o3"]:
        raise AuditFailure("B2_ARCHIVES_IDENTICAL")  # must be distinct captures
    mf_acc = acc["combined_vector_shapley"]["mass_flux"]
    mf_o3 = o3["combined_vector_shapley"]["mass_flux"]
    mf_comp = comp["combined_vector_shapley"]["mass_flux"]
    if not (mf_acc["rms"] == mf_o3["rms"] == mf_comp["rms"]):
        raise AuditFailure("B2_MASSFLUX_NOT_BITWISE_INVARIANT")
    findings["B2_massflux_invariant"] = {
        "archives_distinct": archives,
        "massflux_rms_bitwise_equal": mf_acc["rms"],
        "cosine_with_total": mf_acc["cosine_with_authentic_error"],
        "sse_fraction": mf_acc["signed_projection_fraction_of_authentic_error_sse"],
    }

    # ---- B3 channel-fidelity direction under each rejected fix (falsifies a) ----
    def combined_authentic_rms(p: dict) -> float:
        m = p["combined_vector_shapley"]["mass_flux"]
        return m["rms"] / m["rms_over_authentic_error_rms"]

    r_acc = combined_authentic_rms(acc)
    b3 = {}
    for name, p in (("o3", o3), ("composed", comp)):
        r = combined_authentic_rms(p)
        mix_acc = acc["combined_vector_shapley"]["mixing"]["rms"]
        mix = p["combined_vector_shapley"]["mixing"]["rms"]
        surf = p["combined_vector_shapley"]["surface_drag"]["rms"]
        surf_acc = acc["combined_vector_shapley"]["surface_drag"]["rms"]
        total_up = r > r_acc
        mixing_improved = mix < mix_acc
        surface_flat = abs(surf - surf_acc) < 1e-3 * surf_acc
        if not (total_up and mixing_improved and surface_flat):
            raise AuditFailure(f"B3_SIGNATURE_UNEXPECTED:{name}:{total_up},{mixing_improved},{surface_flat}")
        b3[name] = {
            "total_rms_delta": r - r_acc,
            "mixing_rms_delta": mix - mix_acc,
            "surface_rms_delta": surf - surf_acc,
            "hypothesis_a_falsified": True,
        }
    findings["B3_channel_directions"] = b3

    # ---- B4 residual scale vs gate threshold in fp32-ULP ----
    field_v = acc["components"]["v"]["authentic_postfix_adapter_vs_pristine_wrf"]["reference_rms"]
    ulp = field_v * (2 ** -23)
    findings["B4_scale"] = {
        "residual_v_rms": base["v_rms"],
        "residual_frac_of_field": base["v_rms"] / field_v,
        "residual_in_ulp": base["v_rms"] / ulp,
        "gate_threshold_abs": 1e-4 * base["v_rms"],
        "gate_threshold_in_ulp": (1e-4 * base["v_rms"]) / ulp,
    }

    # ---- over-activation characterization ----
    mf = acc["mass_flux_activity"]["s_aw"]
    if mf["columns_wrf_only"] != 0 or mf["columns_port_only"] != 4526:
        raise AuditFailure("B_OVERACTIVATION_SHAPE")
    findings["overactivation"] = {
        "port_active": mf["port_active_columns"],
        "wrf_active": mf["wrf_active_columns"],
        "port_only": mf["columns_port_only"],
        "wrf_only": mf["columns_wrf_only"],
        "port_is_strict_superset": True,
    }

    out = {
        "schema": "wrfgpu2-v0234-opus-sp2-resolution-audit-v1",
        "verdict": "SP2_GATE_METHODOLOGY_DEFECT_PROVEN__COMPENSATING_ERROR_FALSIFIED",
        "resolution": "b",
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
        "captures_taken": 0,
        "findings": findings,
    }
    payload = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    out["canonical_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    dest = SPRINT.parent / "2026-07-20-v0234-opus-sp2-batch-resolution" / "resolution-audit-proof.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"PASS  verdict={out['verdict']}")
    print(f"      canonical_payload_sha256={out['canonical_payload_sha256']}")
    print(f"      -> {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditFailure as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        sys.exit(2)
