#!/usr/bin/env python3
"""FAST-v025 case definition, namelist derivation, and staging.

FAST-v025 (contract §5.2) is *derived* from the completed ALISIOS `20260725_18z`
production bundle: the same real inputs, restricted to domain 1 and to a single
forecast hour. The source bundle's `namelist.input` describes the full 2-domain
162-hour production run, so the derived namelist necessarily has a different
hash — this module owns that derivation, states it explicitly, and records both
hashes so a critic can regenerate the derived file and compare byte-for-byte.

Nothing here weakens the case: every d01 physics option, dynamics option,
boundary setting, and cadence value is carried over unmodified. Only the domain
count and the run length change.

The dt=54 s / 3600 s non-divisibility is load-bearing and handled explicitly:
WRF asked for one hour advances 67 steps to 01:00:18 (3618 s), not 3600 s. See
``wrf_timing`` for why every rate must normalise by the actual advance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

SOURCE_BUNDLE = Path(
    "<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/"
    "alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case"
)

# Contract §5.2 pre-recorded hashes. These are asserted, never trusted blindly:
# ``verify_source_inputs`` recomputes them and fails closed on any mismatch.
EXPECTED_SOURCE_HASHES = {
    "namelist.input": "5ef3a96be3324deb712b0e0114288244bcf5f53edb21461fee7374aa766cef4b",
    "wrfinput_d01": "1024ddb7d6badc958fb1a4f385117ce5c46d20249b75bcce2ea4261d34f648f0",
    "wrfbdy_d01": "296cef912e17dc0dfff4b78103dedfb1329cd184de5bc2cd0d44cc6edf823599",
}

WRF_RUN_DIR = Path(
    "<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/run"
)
WRF_EXE = WRF_RUN_DIR / "wrf.exe"
MPIRUN = Path(
    "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/mpirun"
)

# Frozen FAST-v025 geometry (contract §5.2).
FAST_START = datetime(2026, 7, 26, 0, 0, 0)
FAST_DT_SECONDS = 54
FAST_SCHEDULED_WINDOW_SECONDS = 3600
FAST_RANKS = 12
FAST_NPROC_X = 4
FAST_NPROC_Y = 3

# 3600/54 is not an integer: WRF must take the first step at or past the
# boundary, so a one-hour job runs 67 steps and lands at 01:00:18.
FAST_STEPS = -(-FAST_SCHEDULED_WINDOW_SECONDS // FAST_DT_SECONDS)  # ceil = 67
FAST_ACTUAL_ADVANCE_SECONDS = FAST_STEPS * FAST_DT_SECONDS  # 3618
FAST_END = FAST_START + timedelta(seconds=FAST_SCHEDULED_WINDOW_SECONDS)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_source_inputs(bundle: Path = SOURCE_BUNDLE) -> dict[str, str]:
    """Recompute the contract's three pre-recorded hashes. Fail closed."""
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        path = bundle / name
        if not path.exists():
            mismatches.append(f"{name}: MISSING at {path}")
            continue
        got = sha256_file(path)
        observed[name] = got
        if got != expected:
            mismatches.append(f"{name}: expected {expected}, got {got}")
    if mismatches:
        raise RuntimeError(
            "FAST-v025 source inputs do not match the frozen contract hashes:\n  "
            + "\n  ".join(mismatches)
        )
    return observed


# --- namelist handling ---------------------------------------------------

_ENTRY_RE = re.compile(r"^\s*(?P<key>[A-Za-z_0-9]+)\s*=\s*(?P<value>.*?),?\s*$")


def parse_namelist(text: str) -> dict[str, dict[str, str]]:
    """Minimal Fortran-namelist reader: ``{group: {key: raw_value}}``."""
    groups: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("&"):
            current = stripped[1:].strip()
            groups[current] = {}
            continue
        if stripped == "/":
            current = None
            continue
        if current is None or not stripped:
            continue
        match = _ENTRY_RE.match(line)
        if match:
            groups[current][match["key"].lower()] = match["value"].strip().rstrip(",")
    return groups


def first_column(raw: str) -> str:
    """The domain-1 element of a possibly per-domain namelist value."""
    return raw.split(",")[0].strip()


@dataclass
class NamelistDerivation:
    """One recorded edit from the production namelist to the FAST-v025 one."""

    group: str
    key: str
    source: str
    derived: str
    reason: str


def derive_fast_namelist(
    source_text: str, *, max_dom: int = 1, hours: int = 1
) -> tuple[str, list[NamelistDerivation]]:
    """Return the FAST-v025 namelist text plus the explicit derivation record.

    Every per-domain value collapses to its domain-1 column; every other value
    is carried through untouched. The edits are returned, not just applied, so
    the contract's "same active production d01 schemes, options, boundary data,
    and cadence" claim is auditable rather than asserted.

    ``max_dom=2`` produces the *legacy-probe* variant instead: the same one-hour
    window on the full production 2-domain grid. That is the configuration the
    2026-07-24 rank probe behind the 86.45 s/model-hour scalar actually ran, so
    reconstructing that number requires it. Per-domain lists are then kept whole.
    """
    groups = parse_namelist(source_text)
    edits: list[NamelistDerivation] = []

    # The window: one forecast hour starting at the production start time.
    # ``run_*`` are scalar in WRF; ``end_*`` are per-domain and MUST carry one
    # column per domain. Emitting a single end column for a 2-domain namelist
    # leaves d02 with no integration window and it silently never runs -- which
    # is exactly how a "2-domain" probe can quietly measure only d01.
    def _cols(value: int) -> str:
        return ", ".join([str(value)] * max_dom)

    end = FAST_START + timedelta(hours=hours)
    window = {
        "run_days": ("0", "FAST-v025 is a short window, not 6d18h"),
        "run_hours": (str(hours), f"window is {hours} forecast hour(s)"),
        "run_minutes": ("0", "unchanged in intent, pinned explicitly"),
        "run_seconds": ("0", "unchanged in intent, pinned explicitly"),
        "end_year": (_cols(end.year), "end = start + 1 h, one column per domain"),
        "end_month": (_cols(end.month), "end = start + 1 h, one column per domain"),
        "end_day": (_cols(end.day), "end = start + 1 h, one column per domain"),
        "end_hour": (_cols(end.hour), "end = start + 1 h, one column per domain"),
    }

    lines_out: list[str] = []
    current: str | None = None
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("&"):
            current = stripped[1:].strip()
            lines_out.append(line)
            continue
        if stripped == "/" or not stripped or current is None:
            lines_out.append(line)
            continue

        match = _ENTRY_RE.match(line)
        if not match:
            lines_out.append(line)
            continue

        key = match["key"].lower()
        raw = match["value"].strip().rstrip(",")

        if key == "max_dom":
            new = str(max_dom)
            if raw != new:
                edits.append(
                    NamelistDerivation(
                        current,
                        key,
                        raw,
                        new,
                        "FAST-v025 is d01 only"
                        if max_dom == 1
                        else "legacy probe keeps the production 2-domain grid",
                    )
                )
            lines_out.append(f" {match['key']:<35} = {new},")
            continue

        if current == "time_control" and key in window:
            new, reason = window[key]
            if raw != new:
                edits.append(NamelistDerivation(current, key, raw, new, reason))
            lines_out.append(f" {match['key']:<35} = {new},")
            continue

        # Collapse per-domain lists to their d01 column. This is what makes the
        # "identical d01 configuration" claim literally true. The legacy probe
        # keeps both columns, because reproducing 86.45 means running what it ran.
        if "," in raw and max_dom == 1:
            d01 = first_column(raw)
            edits.append(
                NamelistDerivation(
                    current, key, raw, d01, "per-domain list collapsed to its d01 column"
                )
            )
            lines_out.append(f" {match['key']:<35} = {d01},")
            continue

        lines_out.append(line)

    return "\n".join(lines_out) + "\n", edits


# --- staging -------------------------------------------------------------

# Everything WRF needs in its run directory that is not the inputs themselves.
_STATIC_SKIP_PREFIXES = ("rsl.", "wrfout_", "wrfrst_", "namelist.output")
_STATIC_SKIP_NAMES = {"namelist.input", "pairs_manifest.json"}
# d01-only runs must not see a d02 input at all, or WRF's nest bookkeeping and
# the staged file set stop matching the namelist they are paired with.
_D01_ONLY_SKIP_NAMES = {"wrfinput_d02"}


@dataclass
class StagedCase:
    root: Path
    namelist_path: Path
    namelist_sha256: str
    source_namelist_sha256: str
    derivations: list[NamelistDerivation] = field(default_factory=list)
    linked: list[str] = field(default_factory=list)


def stage_fast_case(
    dest: Path,
    *,
    bundle: Path = SOURCE_BUNDLE,
    verify: bool = True,
    max_dom: int = 1,
    hours: int = 1,
) -> StagedCase:
    """Materialise a fresh, runnable FAST-v025 directory at ``dest``.

    Static tables and the large real inputs are symlinked (they are read-only to
    WRF); only the derived namelist is a real file. ``dest`` must not already
    exist — every arm gets its own directory so no run can inherit another's
    output, which is what makes the fresh-CPU rule enforceable.
    """
    if verify:
        verify_source_inputs(bundle)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; each FAST-v025 arm needs a fresh directory")
    dest.mkdir(parents=True)

    skip = set(_STATIC_SKIP_NAMES)
    if max_dom == 1:
        skip |= _D01_ONLY_SKIP_NAMES
    linked: list[str] = []
    for entry in sorted(bundle.iterdir()):
        name = entry.name
        if name in skip or name.startswith(_STATIC_SKIP_PREFIXES):
            continue
        target = entry.resolve() if entry.is_symlink() else entry
        os.symlink(target, dest / name)
        linked.append(name)

    source_namelist = bundle / "namelist.input"
    text, edits = derive_fast_namelist(
        source_namelist.read_text(), max_dom=max_dom, hours=hours
    )
    namelist_path = dest / "namelist.input"
    namelist_path.write_text(text)

    return StagedCase(
        root=dest,
        namelist_path=namelist_path,
        namelist_sha256=sha256_file(namelist_path),
        source_namelist_sha256=sha256_file(source_namelist),
        derivations=edits,
        linked=linked,
    )


def case_descriptor() -> dict:
    """The frozen, machine-readable FAST-v025 definition."""
    return {
        "name": "FAST-v025",
        "source_bundle": str(SOURCE_BUNDLE),
        "domains": 1,
        "grid_namelist_we_sn": [121, 71],
        "grid_mass_we_sn": [120, 70],
        "e_vert": 45,
        "mass_levels": 44,
        "dt_seconds": FAST_DT_SECONDS,
        "ranks": FAST_RANKS,
        "decomposition": [FAST_NPROC_X, FAST_NPROC_Y],
        "start_time": FAST_START.strftime("%Y-%m-%d_%H:%M:%S"),
        "scheduled_window_seconds": FAST_SCHEDULED_WINDOW_SECONDS,
        "steps": FAST_STEPS,
        "actual_model_advance_seconds": FAST_ACTUAL_ADVANCE_SECONDS,
        "end_stamp_expected": (
            FAST_START + timedelta(seconds=FAST_ACTUAL_ADVANCE_SECONDS)
        ).strftime("%Y-%m-%d_%H:%M:%S"),
        "non_divisible_boundary_note": (
            f"3600/{FAST_DT_SECONDS} is not an integer; a one-hour job takes "
            f"{FAST_STEPS} steps and advances {FAST_ACTUAL_ADVANCE_SECONDS}s, "
            f"overshooting the scheduled window by "
            f"{FAST_ACTUAL_ADVANCE_SECONDS - FAST_SCHEDULED_WINDOW_SECONDS}s "
            f"({100 * (FAST_ACTUAL_ADVANCE_SECONDS / FAST_SCHEDULED_WINDOW_SECONDS - 1):.3f}%). "
            "Every rate normalises by the actual advance."
        ),
        "expected_source_hashes": dict(EXPECTED_SOURCE_HASHES),
        "wrf_exe": str(WRF_EXE),
        "mpirun": str(MPIRUN),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=None, help="stage a case here")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    if args.verify_hashes:
        print(json.dumps(verify_source_inputs(), indent=2, sort_keys=True))
    if args.describe:
        print(json.dumps(case_descriptor(), indent=2, sort_keys=True))
    if args.stage:
        staged = stage_fast_case(args.stage)
        print(
            json.dumps(
                {
                    "root": str(staged.root),
                    "namelist_sha256": staged.namelist_sha256,
                    "source_namelist_sha256": staged.source_namelist_sha256,
                    "linked_entries": len(staged.linked),
                    "derivations": [vars(d) for d in staged.derivations],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
