#!/usr/bin/env python3
"""v020_make_namelist.py — write a P4-ladder namelist variant (dt rung) from a base
namelist, changing ONLY time_step (and, when env-plumbing lands, recording the target
acoustic_substeps).

Pure text rewrite of a WRF namelist.input. NO GPU, NO gpuwrf import. The P4 dt ladder's
lever is the root `time_step` (every nest dt = root/ratio, so one field cuts the whole
cascade's step count). acoustic_substeps is NOT in the all-7 namelist and is currently
hardcoded in nested_pipeline.py (see REPORT.md "P4 n_sound plumbing gap"); this tool
records the intended n_sound per rung so the driver can pass it via env the instant the
one-line override lands, but it does NOT fabricate a namelist field the runtime ignores.

Usage:
  python v020_make_namelist.py --base BASE.input --out VARIANT.input --time-step 24
                               [--n-sound 14]   # recorded as a comment + sidecar only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def set_time_step(text: str, dt: int) -> str:
    pat = re.compile(r"^(\s*time_step\s*=\s*)(\d+)(.*)$", re.M)
    if not pat.search(text):
        raise ValueError("base namelist has no 'time_step =' line to set")
    return pat.sub(rf"\g<1>{dt}\g<3>", text, count=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--time-step", required=True, type=int)
    ap.add_argument("--n-sound", type=int, default=None,
                    help="target acoustic_substeps for this rung (recorded only; the "
                         "runtime override is GPUWRF_ACOUSTIC_SUBSTEPS once plumbed)")
    args = ap.parse_args(argv)

    if not args.base.is_file():
        print(f"v020_make_namelist: base not found {args.base}", file=sys.stderr)
        return 2
    text = args.base.read_text()
    try:
        new = set_time_step(text, args.time_step)
    except ValueError as e:
        print(f"v020_make_namelist: {e}", file=sys.stderr)
        return 2

    banner = (f"! v020 P4 ladder variant: time_step={args.time_step}"
              + (f" target_acoustic_substeps={args.n_sound} "
                 f"(apply via GPUWRF_ACOUSTIC_SUBSTEPS once plumbed)" if args.n_sound else "")
              + "\n")
    args.out.write_text(banner + new)
    # sidecar manifest the driver reads back
    side = args.out.with_suffix(args.out.suffix + ".rung.json")
    import json
    side.write_text(json.dumps({"time_step": args.time_step,
                                "target_acoustic_substeps": args.n_sound,
                                "base": str(args.base), "out": str(args.out)}, indent=2) + "\n")
    print(f"wrote {args.out} (time_step={args.time_step}"
          + (f", n_sound target={args.n_sound}" if args.n_sound else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
