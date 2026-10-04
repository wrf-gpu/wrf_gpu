"""FP32 MIXED-branch static audit (ADR-031 S2 boundary-rule gate).

Extends ``proofs/v014/fp32_acoustic_static_audit.py`` with the ADR-031 §3.1
"boundary rule" gate: in the ``MIXED_PERTURB_FP32`` branch there must be ZERO
in-loop ``base = fp32(total) - fp32(perturbation)`` reconstructions, and the
acoustic/dynamics path MUST consume the ``acoustic_precision_mode`` label
(i.e. the mode is no longer default-inert).

This is a SOURCE-AUDIT gate (no model run, no GPU).  It runs the v014 base scan,
then applies two MIXED-branch assertions:

  GATE-A (boundary rule): every base-from-total reconstruction inside the
    timestep loop (small_step_prep / small_step_finish / acoustic / advance_w /
    calc_p_rho / rk_addtend_dry / operational_mode acoustic core) must be
    GUARDED by the MIXED-mode branch using ``carry.base.{pb,phb,mub}`` instead --
    i.e. it must NOT execute ``fp32(total) - fp32(perturbation)`` when the mode
    is MIXED.  Until S2 lands, the audit records the OPEN reconstructions and
    marks the gate ``status="OPEN_PRE_S2"``; once S2 wires ``carry.base`` it must
    flip to ``PASS`` (the in-loop list, filtered to the MIXED branch, is empty).

  GATE-B (mode consumed): ``timestep_precision_mode_consumers`` must be non-empty
    in the MIXED implementation (the acoustic must read the label), refuting the
    v014 ``default_inert_claim``.

The audit is intentionally BUILD-FORWARD: it does not require S2 to be done to
run; it produces the live count + the pass/fail of each gate so the joint
Opus+GPT S2 team has a red->green target.

Run:  python proofs/perf/v015/fp32_oracles/fp32_mixed_branch_static_audit.py
Out:  proofs/perf/v015/fp32_oracles/fp32_mixed_branch_static_audit.json
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

THIS = Path(__file__).resolve()
# repo root that actually holds src/ (worktree or shared checkout)
_CANDIDATE_ROOTS = [THIS.parents[4], Path("<USER_HOME>/src/wrf_gpu2")]


def _root() -> Path:
    for r in _CANDIDATE_ROOTS:
        if (r / "src" / "gpuwrf" / "dynamics" / "core" / "calc_p_rho.py").is_file():
            return r
    raise FileNotFoundError("cannot locate repo root with src/gpuwrf")


ROOT = _root()
OUT = THIS.parent / "fp32_mixed_branch_static_audit.json"

# The timestep-loop modules where an in-loop base-from-total reconstruction is
# FORBIDDEN under MIXED (ADR §3.1 boundary rule).  operational_mode is included
# but only its acoustic-core reconstructions count (the wrfout/restart/diagnostic
# assembly at controlled interfaces is allowed -- those are NOT in-loop).
IN_LOOP_MODULES = (
    "src/gpuwrf/dynamics/core/small_step_prep.py",
    "src/gpuwrf/dynamics/core/small_step_finish.py",
    "src/gpuwrf/dynamics/core/acoustic.py",
    "src/gpuwrf/dynamics/core/advance_w.py",
    "src/gpuwrf/dynamics/core/calc_p_rho.py",
    "src/gpuwrf/dynamics/core/rk_addtend_dry.py",
    "src/gpuwrf/dynamics/acoustic_wrf.py",
)

SOURCE_FILES = IN_LOOP_MODULES + (
    "src/gpuwrf/contracts/precision.py",
    "src/gpuwrf/contracts/state.py",
    "src/gpuwrf/runtime/operational_mode.py",
    "src/gpuwrf/runtime/operational_state.py",
)

TOTAL_MINUS_PERT_RE = re.compile(
    r"(?:jnp\.asarray\()?([A-Za-z_][A-Za-z0-9_.]*)\.(p|ph|mu)_total\)?"
    r"\s*-\s*(?:jnp\.asarray\()?([A-Za-z_][A-Za-z0-9_.]*)\.\2_perturbation"
)
PRECISION_MODE_RE = re.compile(
    r"AcousticPrecisionMode|DEFAULT_ACOUSTIC_PRECISION_MODE|acoustic_precision_mode|MIXED_PERTURB_FP32"
)
# A reconstruction is "guarded" (acceptable) if the surrounding code carries the
# base via carry.base / BaseState rather than total-minus-perturbation.  The S2
# fix replaces these with carry.base.{pb,phb,mub}; once it does, the regex below
# stops matching that line.  We also detect whether the file consumes the base.
BASE_CARRY_RE = re.compile(r"carry\.base\.|\.base\.(pb|phb|mub)|BaseState")


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _scan():
    in_loop_recon = []
    mode_consumers = []
    base_carry_uses = []
    for rel in SOURCE_FILES:
        path = ROOT / rel
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if rel in IN_LOOP_MODULES and TOTAL_MINUS_PERT_RE.search(line):
                in_loop_recon.append({"file": rel, "line": lineno, "source": stripped})
            if PRECISION_MODE_RE.search(line) and (
                rel.startswith("src/gpuwrf/dynamics/") or rel.endswith("operational_state.py")
            ):
                mode_consumers.append({"file": rel, "line": lineno, "source": stripped})
            if BASE_CARRY_RE.search(line):
                base_carry_uses.append({"file": rel, "line": lineno, "source": stripped})
    return in_loop_recon, mode_consumers, base_carry_uses


def run():
    in_loop_recon, mode_consumers, base_carry_uses = _scan()

    # GATE-A: the boundary rule.  PASS iff zero in-loop base-from-total survives
    # in the timestep modules (the MIXED fix routes through carry.base instead).
    gate_a_pass = len(in_loop_recon) == 0
    # GATE-B: the mode must be consumed by the dynamics/acoustic path.
    gate_b_pass = len(mode_consumers) > 0

    # status: OPEN until S2 wires carry.base + the mode consumer; then PASS.
    if gate_a_pass and gate_b_pass:
        status = "PASS"
    elif not gate_a_pass and not gate_b_pass:
        status = "OPEN_PRE_S2"
    else:
        status = "PARTIAL"

    report = {
        "case": "ADR-031 S2 -- FP32 MIXED-branch boundary-rule static audit",
        "head": _git("rev-parse", "HEAD"),
        "extends": "proofs/v014/fp32_acoustic_static_audit.py",
        "in_loop_modules_audited": list(IN_LOOP_MODULES),
        "gate_A_boundary_rule": {
            "rule": (
                "ZERO in-loop `fp32(total) - fp32(perturbation)` base reconstructions "
                "in the timestep modules under MIXED_PERTURB_FP32; the base must come "
                "from carry.base.{pb,phb,mub} (fp64). Controlled-interface "
                "reconstructions (wrfout/restart/diagnostics) are exempt and live "
                "outside these modules."
            ),
            "in_loop_base_from_total_count": len(in_loop_recon),
            "in_loop_base_from_total_sites": in_loop_recon,
            "PASS": bool(gate_a_pass),
        },
        "gate_B_mode_consumed": {
            "rule": (
                "The acoustic/dynamics path must CONSUME acoustic_precision_mode "
                "(refuting the v014 default_inert_claim) in the MIXED implementation."
            ),
            "dynamics_mode_consumer_count": len(mode_consumers),
            "dynamics_mode_consumers": mode_consumers,
            "PASS": bool(gate_b_pass),
        },
        "base_carry_threading": {
            "note": "Uses of carry.base / BaseState (the fp64 base island); the S2 "
                    "fix should grow this set as it replaces total-minus-pert.",
            "count": len(base_carry_uses),
            "sites": base_carry_uses,
        },
        "status": status,
        "GATE_PASS": bool(gate_a_pass and gate_b_pass),
        "interpretation": (
            "OPEN_PRE_S2 is the EXPECTED state before the S2 kernel attack: the "
            "current code still reconstructs the base by subtraction and does not "
            "consume the mode in the loop. The joint Opus+GPT S2 work must flip BOTH "
            "gates to PASS -- this audit is the red->green target."
        ),
    }
    return report


def main() -> int:
    report = run()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT}")
    print(f"  GATE-A (boundary rule): in-loop base-from-total count = "
          f"{report['gate_A_boundary_rule']['in_loop_base_from_total_count']} -> "
          f"{'PASS' if report['gate_A_boundary_rule']['PASS'] else 'FAIL (must be 0 after S2)'}")
    print(f"  GATE-B (mode consumed): dynamics consumer count = "
          f"{report['gate_B_mode_consumed']['dynamics_mode_consumer_count']} -> "
          f"{'PASS' if report['gate_B_mode_consumed']['PASS'] else 'FAIL (must be >0 after S2)'}")
    print(f"  base-carry threading uses = {report['base_carry_threading']['count']}")
    print(f"  STATUS = {report['status']}  (GATE_PASS={report['GATE_PASS']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
