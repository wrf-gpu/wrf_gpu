"""Scale a WRF namelist's root domains.time_step by a factor (TEST INFRA ONLY).

K2 lossless lever (config-level, NO core dycore edit): the only edit is the
root ``time_step`` in &domains. The child timesteps follow automatically via the
parent_grid_ratio chain (nested_pipeline._dt_by_domain), so a single root edit
scales the whole cascade by the same factor -- exactly the K2 prescription.

We do a minimal TEXT edit of just the ``time_step`` line (and optionally clamp
``max_dom``), preserving every other byte so the run config is otherwise the
shipped real-case namelist.

Usage:  K2_FACTOR=1.5 [K2_MAXDOM=3] python scale_namelist.py SRC DST
The scaled root dt is rounded to the nearest integer second (WRF time_step is an
integer); we also set time_step_fract_* to 0 to keep it exact.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def scale(src: str, dst: str, factor: float, max_dom: int | None) -> dict:
    text = Path(src).read_text()
    # Find the FIRST time_step assignment in &domains (root). WRF packs it as
    #   time_step = 18,
    m = re.search(r"(?im)^(\s*time_step\s*=\s*)(\d+)(\s*,?.*)$", text)
    if not m:
        raise SystemExit("could not find 'time_step =' in namelist")
    old_dt = int(m.group(2))
    new_dt = int(round(old_dt * factor))
    if new_dt < 1:
        raise SystemExit(f"scaled dt {new_dt} < 1")
    text = text[: m.start()] + f"{m.group(1)}{new_dt}{m.group(3)}" + text[m.end():]

    # Force integer dt (fract num/den -> 0) so the cascade stays exact.
    text = re.sub(r"(?im)^(\s*time_step_fract_num\s*=\s*)\d+", r"\g<1>0", text)
    text = re.sub(r"(?im)^(\s*time_step_fract_den\s*=\s*)\d+", r"\g<1>1", text)

    if max_dom is not None:
        text = re.sub(
            r"(?im)^(\s*max_dom\s*=\s*)\d+", rf"\g<1>{int(max_dom)}", text
        )

    Path(dst).write_text(text)
    return {"src": src, "dst": dst, "factor": factor, "old_dt": old_dt, "new_dt": new_dt, "max_dom": max_dom}


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    factor = float(os.environ.get("K2_FACTOR", "1.0"))
    max_dom_env = os.environ.get("K2_MAXDOM")
    max_dom = int(max_dom_env) if max_dom_env else None
    info = scale(src, dst, factor, max_dom)
    print(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
