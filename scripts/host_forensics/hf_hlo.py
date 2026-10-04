"""Dump HLO of the production _advance_chunk_fori segment and grep for
host-round-trip ops (outfeed/send-recv/custom-call/copy-to-host).

NOTE: XLA_FLAGS must be set BEFORE the backend initializes (first device op),
so this script sets it immediately at import time.
"""
import dataclasses
import os
import sys
from pathlib import Path

DUMP = "/tmp/hf_hlo"
WORKTREE_SRC = "<USER_HOME>/src/wrf_gpu2_wt/hostforensic/src"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={DUMP} --xla_dump_hlo_as_text"
)
sys.path.insert(0, WORKTREE_SRC)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.integration import daily_pipeline as daily  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")

config = daily.DailyPipelineConfig(run_id=str(RUN_DIR), run_root=RUN_DIR.parent, hours=1, domain="d01")
case, resolved = daily._build_real_case(config)
state = case.state
capture = daily._capture_boundary_leaves(state, case.namelist)
cadence = daily._boundary_window_cadence_s(case.namelist)
record_cadence = float((case.metadata.get("boundary") or {}).get("interval_seconds") or cadence)
if capture:
    state = daily._rewindow_boundary_leaves(state, capture, segment_start_s=0.0,
                                            record_cadence_s=record_cadence, window_s=cadence)
nl = dataclasses.replace(case.namelist, dt_s=54.0)

carry = om._committed_initial_carry_for_run(state, nl)
cb = om.build_clock_base(nl)

lowered = jax.jit(
    lambda c, s, ss, clock: om._advance_chunk_fori(
        c, s, ss, clock, n_steps=34, cadence=int(nl.radiation_cadence_steps))
).lower(carry, nl, jnp.asarray(1, dtype=jnp.int32), cb)
comp = lowered.compile()
text = lowered.compiler_ir(dialect="hlo")
s = str(text)
Path(DUMP).mkdir(parents=True, exist_ok=True)
Path(DUMP, "advance_chunk_fori.hlo").write_text(s)
print("HLO chars:", len(s))
import re
for pat in ("outfeed", "infeed", "send", "recv", "custom-call", "after-all", "fft",
            "sparse", "gtsv", "OptimizationBarrier", "copy-start", "collective"):
    hits = re.findall(rf".*{pat}.*", s, flags=re.IGNORECASE)
    uniq = sorted(set(h.strip()[:160] for h in hits))
    print(f"== {pat}: {len(hits)} lines, {len(uniq)} uniq")
    for u in uniq[:6]:
        print("   ", u)
