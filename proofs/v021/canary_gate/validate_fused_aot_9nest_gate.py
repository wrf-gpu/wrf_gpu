"""Validate the v0.21 fused+AOT 9-nest bounded-phase gate logs.

The 9-nest fused d02 cascade is allowed to have a small finite set of runtime
aval phases. The gate passes only if the cold process materializes that bounded
set and the fresh warm process loads the same fused phase keys from AOT blobs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


STATUS_RE = re.compile(r"^\[gpuwrf:nested-aot\] domain=(?P<domain>\S+) (?P<rest>.*)$")
KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")


def _kv(rest: str) -> dict[str, str]:
    return {m.group("key"): m.group("value") for m in KV_RE.finditer(rest)}


def _read(path: Path) -> str:
    return path.read_text(errors="replace")


def _status_records(text: str, *, domain: str = "fused/d02") -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for line in text.splitlines():
        m = STATUS_RE.match(line)
        if not m or m.group("domain") != domain:
            continue
        rec = _kv(m.group("rest"))
        rec["domain"] = m.group("domain")
        rec["line"] = line
        out.append(rec)
    return out


def _summary(text: str) -> dict[str, str] | None:
    for line in text.splitlines():
        if not line.startswith("SUMMARY ") or " mode=fixed " not in f" {line} ":
            continue
        return _kv(line)
    return None


def _bit_id(text: str) -> dict[str, str] | None:
    for line in text.splitlines():
        if line.startswith("BIT_ID "):
            return _kv(line)
    return None


def _ref_compare(text: str) -> dict[str, str] | None:
    for line in text.splitlines():
        if line.startswith("REF_COMPARE "):
            return _kv(line)
    return None


def _bool(value: str | None) -> bool:
    return str(value).lower() == "true"


def _float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _sorted_keys(records: list[dict[str, str]], sources: set[str] | None = None) -> list[str]:
    keys: set[str] = set()
    for rec in records:
        key = rec.get("cheap_key")
        if not key:
            continue
        if sources is not None and rec.get("source") not in sources:
            continue
        keys.add(key)
    return sorted(keys)


def _validate_summary(
    label: str,
    summary: dict[str, str] | None,
    errors: list[str],
) -> dict[str, Any]:
    if summary is None:
        errors.append(f"{label}: missing fixed-mode SUMMARY")
        return {}

    s_per_step = _float(summary.get("s_per_step"))
    s_per_fc_hour = _float(summary.get("s_per_fc_hour"))
    if not math.isfinite(s_per_step) or s_per_step <= 0:
        errors.append(f"{label}: invalid s_per_step={summary.get('s_per_step')}")
    if not math.isfinite(s_per_fc_hour) or s_per_fc_hour <= 0:
        errors.append(f"{label}: invalid s_per_fc_hour={summary.get('s_per_fc_hour')}")
    for field in ("all_finite", "vram_flat", "rss_flat"):
        if not _bool(summary.get(field)):
            errors.append(f"{label}: {field}={summary.get(field)}")
    return {
        "s_per_step": s_per_step,
        "s_per_fc_hour": s_per_fc_hour,
        "all_finite": _bool(summary.get("all_finite")),
        "vram_flat": _bool(summary.get("vram_flat")),
        "rss_flat": _bool(summary.get("rss_flat")),
        "raw": summary,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    cold_text = _read(args.cold_log)
    warm_text = _read(args.warm_log)
    phase_text = _read(args.phase_log) if args.phase_log else cold_text
    errors: list[str] = []

    for label, text in (("phase", phase_text), ("cold", cold_text), ("warm", warm_text)):
        if "DRIVER RAISED" in text:
            errors.append(f"{label}: DRIVER RAISED present")
        if "fallback:fused-cached-call-error" in text:
            errors.append(f"{label}: fused cached-call error present")
        if "fallback:fused-aot-exec-error" in text:
            errors.append(f"{label}: fused AOT exec error present")
        if "fallback:fused-jit-compile-exception" in text:
            errors.append(f"{label}: fused jit compile exception present")
        if "fallback:fused-lower-error" in text:
            errors.append(f"{label}: fused lower error present")

    if "Compiling module jit_fused" in warm_text:
        errors.append("warm: jit_fused recompile/re-lower marker present")

    phase_records = _status_records(phase_text)
    warm_records = _status_records(warm_text)

    phase_all_keys = _sorted_keys(phase_records)
    phase_resident_keys = _sorted_keys(
        phase_records,
        {"aot_blob", "fallback:fused-jit-compiled+aot-captured"},
    )
    phase_missing_keys = _sorted_keys(phase_records, {"fallback:missing"})
    warm_all_keys = _sorted_keys(warm_records)
    warm_loaded_keys = _sorted_keys(warm_records, {"aot_blob"})
    warm_fallback_keys = _sorted_keys(
        [rec for rec in warm_records if str(rec.get("source", "")).startswith("fallback:")]
    )

    if len(phase_resident_keys) != args.expect_fused_phases:
        errors.append(
            f"phase: resident fused phase count {len(phase_resident_keys)} "
            f"!= expected {args.expect_fused_phases}"
        )
    if len(phase_resident_keys) > args.max_fused_phases:
        errors.append(
            f"phase: resident fused phase count {len(phase_resident_keys)} "
            f"> max {args.max_fused_phases}"
        )
    unresolved_missing = sorted(set(phase_missing_keys) - set(phase_resident_keys))
    if unresolved_missing:
        errors.append(f"phase: missing fused keys not materialized: {unresolved_missing}")
    if sorted(phase_resident_keys) != sorted(warm_loaded_keys):
        errors.append(
            "warm: loaded fused phase keys do not equal phase resident set "
            f"phase={phase_resident_keys} warm={warm_loaded_keys}"
        )
    if sorted(warm_all_keys) != sorted(warm_loaded_keys):
        errors.append(f"warm: non-aot fused key statuses present: {warm_fallback_keys}")

    cold_summary = _validate_summary("cold", _summary(cold_text), errors)
    warm_summary = _validate_summary("warm", _summary(warm_text), errors)
    cold_s_per_step = cold_summary.get("s_per_step")
    warm_s_per_step = warm_summary.get("s_per_step")
    if (
        isinstance(cold_s_per_step, float)
        and isinstance(warm_s_per_step, float)
        and math.isfinite(cold_s_per_step)
        and math.isfinite(warm_s_per_step)
        and cold_s_per_step > 0
    ):
        warm_cold_ratio = warm_s_per_step / cold_s_per_step
        if warm_cold_ratio > args.max_warm_cold_s_per_step_ratio:
            errors.append(
                "timing: warm s_per_step regressed versus cold "
                f"ratio={warm_cold_ratio:.4f} "
                f"> {args.max_warm_cold_s_per_step_ratio:.4f} "
                f"(cold={cold_s_per_step:.4f}, warm={warm_s_per_step:.4f})"
            )
    else:
        warm_cold_ratio = None

    cold_bit = _bit_id(cold_text)
    warm_bit = _bit_id(warm_text)
    if cold_bit is None:
        errors.append("cold: missing BIT_ID")
    if warm_bit is None:
        errors.append("warm: missing BIT_ID")
    if cold_bit and warm_bit and cold_bit.get("digest_sha256") != warm_bit.get("digest_sha256"):
        errors.append(
            "digest: cold/warm digest mismatch "
            f"cold={cold_bit.get('digest_sha256')} warm={warm_bit.get('digest_sha256')}"
        )

    cold_ref = _ref_compare(cold_text)
    warm_ref = _ref_compare(warm_text)
    ref = warm_ref or cold_ref
    if ref is None:
        errors.append("identity: missing REF_COMPARE in cold/warm logs")
    elif not _bool(ref.get("ref_equal")):
        errors.append(f"identity: REF_COMPARE ref_equal={ref.get('ref_equal')}")

    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "phase": {
            "log": str(args.phase_log or args.cold_log),
            "all_fused_keys": phase_all_keys,
            "resident_fused_keys": phase_resident_keys,
            "missing_fused_keys": phase_missing_keys,
        },
        "cold": {
            "summary": cold_summary,
            "bit_id": cold_bit,
            "ref_compare": cold_ref,
        },
        "warm": {
            "all_fused_keys": warm_all_keys,
            "loaded_fused_keys": warm_loaded_keys,
            "fallback_fused_keys": warm_fallback_keys,
            "summary": warm_summary,
            "bit_id": warm_bit,
            "ref_compare": warm_ref,
        },
        "limits": {
            "expect_fused_phases": args.expect_fused_phases,
            "max_fused_phases": args.max_fused_phases,
            "max_warm_cold_s_per_step_ratio": args.max_warm_cold_s_per_step_ratio,
            "timing_model": "warm-vs-cold-no-regression",
            "max_s_per_fc_hour": args.max_s_per_fc_hour,
        },
        "timing": {
            "warm_cold_s_per_step_ratio": warm_cold_ratio,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold-log", type=Path, required=True)
    parser.add_argument("--warm-log", type=Path, required=True)
    parser.add_argument(
        "--phase-log",
        type=Path,
        help="Optional phase-discovery log. Use when cold-log is a reduced-K JIT reference.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-fused-phases", type=int, default=2)
    parser.add_argument("--max-fused-phases", type=int, default=4)
    parser.add_argument(
        "--max-s-per-fc-hour",
        type=float,
        default=0.0,
        help="Recorded for back-compat only; 9-nest timing is judged warm-vs-cold.",
    )
    parser.add_argument("--max-warm-cold-s-per-step-ratio", type=float, default=1.25)
    args = parser.parse_args(argv)

    result = validate(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
