"""Check scorer coverage before describing its result in release documentation."""
from __future__ import annotations


def gate_summary(js: dict, max_lead: int = 24) -> dict:
    summary, inventory = js["summaries"], js["inventory"]
    incomplete = [issue for issue in summary.get("coverage_issues", []) if issue.get("kind") != "gpu_only"]
    if incomplete or inventory.get("cpu_only_count") or inventory.get("incompatible_common"):
        raise ValueError("D6 report has incomplete field coverage")
    common = set(inventory["common"])
    if not common or common != set(js["field_summaries"]):
        raise ValueError("D6 summaries do not cover every common field")
    expected = list(range(max_lead + 1))
    if js["pairing"]["common_leads_h"] != expected:
        raise ValueError(f"D6 report needs every lead 0..{max_lead}")
    if not js["tolerances"]["supplied"]:
        raise ValueError("D6 report has no frozen tolerance manifest")
    bounded, numeric, finite, frame_failures = 0, 0, True, 0
    for name, fs in js["field_summaries"].items():
        if fs.get("missing_leads") or fs.get("incompatible_leads"):
            raise ValueError(f"D6 incomplete field: {name}")
        rows = fs.get("by_lead")
        if rows is None:
            checks = fs.get("checks")
            if not checks or [r["lead_h"] for r in checks] != expected:
                raise ValueError(f"D6 incomplete metadata: {name}")
            if not fs.get("all_equal"):
                raise ValueError(f"D6 metadata differs: {name}")
            continue
        numeric += 1
        if [r["lead_h"] for r in rows] != expected:
            raise ValueError(f"D6 incomplete lead coverage: {name}")
        finite &= all(r.get("n", 0) > 0 and r.get("finite_gpu") == r["n"] for r in rows)
        frame_failures += sum((r.get("tolerance_result") or {}).get("pass") is False for r in rows)
        if (fs.get("tolerance_result") or {}).get("supplied"):
            bounded += 1
    if numeric != int(summary["comparable_field_count"]):
        raise ValueError("D6 numeric coverage count disagrees with field rows")
    if bounded != int(js["tolerances"]["field_count"]):
        raise ValueError("D6 report omits fields from the frozen tolerance manifest")
    return {"fields": len(common), "numeric_fields": numeric, "bounded_fields": bounded,
            "frames": len(expected), "finite_gpu": bool(finite),
            "failures": max(int(summary["tolerance_failure_count"]), frame_failures)}


def integrity_summary(js: dict, min_frames: int = 25, require_full_header: bool = False,
                      domains: tuple[str, ...] = ("d01", "d02", "d03")) -> dict:
    """Require the strict missing-variable/degenerate/global-attribute report."""
    legacy_rule = "missing CPU variables + degenerate fields + WRF globals"
    full_rule = legacy_rule + " (full CPU header, every paired frame)"
    full_header = js.get("rule") == full_rule
    if js.get("rule") not in (legacy_rule, full_rule):
        raise ValueError("need the strict output-integrity rule")
    if require_full_header and not full_header:
        raise ValueError("final output integrity needs the full CPU header at every paired frame")
    if set(js["domains"]) != set(domains):
        scope = "WN3" if domains == ("d01", "d02", "d03") else "requested"
        raise ValueError(f"output integrity needs every {scope} domain")
    missing, degenerate, attrs, cpu_attrs, global_frames = 0, 0, 0, 0, 0
    passed = js["pass"] is True
    for d, row in js["domains"].items():
        if row["frames_paired"] < min_frames or not row["frames_checked"]:
            raise ValueError(f"output integrity has incomplete frame coverage: {d}")
        missing += len(row["missing_variables_vs_cpu"])
        degenerate += len(row["degenerate_fields"])
        header = row["global_attrs"]
        attrs += len(header["missing_required"])
        if full_header:
            if (row["pairing_ok"] is not True or row["cpu_frames"] != row["frames_paired"]
                    or header["frames_checked"] != row["frames_paired"]):
                raise ValueError(f"output integrity has incomplete full-header frame coverage: {d}")
            cpu_attrs += len(header["missing_vs_cpu"])
            global_frames += len(header["by_frame"])
        passed &= row["pass"] is True
    return {"pass": bool(passed and missing == degenerate == attrs == cpu_attrs == global_frames == 0),
            "missing_fields": missing, "degenerate_fields": degenerate, "missing_required_attrs": attrs,
            "missing_cpu_attrs": cpu_attrs, "global_frames_failing": global_frames, "full_cpu_header": full_header}


def approved_integrity_summary(js: dict, exception: dict, approval_reference: str,
                               delivery_note: str) -> dict:
    """Retain raw FAIL while checking an explicitly approved FINAL-b disclosure.

    The published delivery or release-gate note must name the case and approval; the exception
    must reproduce every degenerate event in this exact 24 h strict report.
    Structural failures are always rejected.
    """
    from pathlib import Path

    result = integrity_summary(js, require_full_header=True)
    if result["pass"] or not result["degenerate_fields"]:
        raise ValueError("disclosed exception requires a raw degenerate-field FAIL")
    if any(result[key] for key in ("missing_fields", "missing_required_attrs",
                                    "missing_cpu_attrs", "global_frames_failing")):
        raise ValueError("structural integrity failures cannot use a disclosed exception")
    case = exception.get("case")
    if not case or case not in Path(js["gpu_dir"]).parts:
        raise ValueError("integrity exception belongs to different GPU outputs")
    actual = {domain: row["degenerate_fields"] for domain, row in js["domains"].items()
              if row["degenerate_fields"]}
    if actual != exception.get("fields", {}).get("integrity_24h"):
        raise ValueError("integrity exception does not match every raw strict event")
    current_approval = ("RE-DELIVERY on FINAL-b" in delivery_note
                        or "RELEASE GATE (WN3 part) ACCEPTED on FINAL-b" in delivery_note)
    if (not approval_reference or f"manager approval {approval_reference}" not in delivery_note
            or case[:12] not in delivery_note or not current_approval):
        raise ValueError("need the FINAL-b delivery note or release-gate note with explicit manager approval")
    return {**result, "accepted_with_exception": True, "approval_reference": approval_reference}
