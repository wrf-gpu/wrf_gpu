"""v0.4.0 S5 — 24h native-init FORECAST gate: READY (NOT RUN).

The honest end-to-end standalone proof for v0.4.0 is: native real-init ->
native GPU forecast (NO CPU-WRF replay; the NATIVE wrfbdy is the only LBC) ->
24h -> per-lead full-field comparison to the CPU-WRF wrfout. That step is
GPU-BOUND and MANAGER-SCHEDULED (single GPU shared with the v0.3.0 parity
campaign + per-case backfill replays — ONE GPU job at a time).

This module does NOT run any GPU forecast. It (1) regenerates the forecast-gate
PLAN artifact wired to the ASSEMBLED ``driver.build_real_init`` factory and the
S5 usable case set, and (2) verifies the CPU-WRF reference roots + the
continuous-gate metric module exist, so the gate is provably READY for the
manager to trigger.

Manager invocation when the GPU is free (ONE job at a time):

    from gpuwrf.init.real_init import comparator as C
    from proofs.v040.s5_native_init_parity import make_factory  # the integrated factory
    C.run_forecast_gate(make_factory(lbc_intervals=1), execute=True,
                        out_path="proofs/v040/s5_forecast_gate_report.json")

(``execute=True`` is a guarded NotImplementedError in the comparator until the
manager wires the GPU forecast-pipeline body at that single-GPU serialization
point; the metric set + spec + case list + references are already frozen here.)

Usage (NON-GPU, cores 0-3):
    taskset -c 0-3 env JAX_PLATFORM_NAME=cpu PYTHONPATH=src python \
        proofs/v040/s5_forecast_gate_ready.py \
        --out proofs/v040/s5_forecast_gate_spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpuwrf.init.real_init import comparator as C
from gpuwrf.init.real_init.comparator import ForecastGateSpec

# Reuse the S5 integrated factory + usable-case discovery.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s5_native_init_parity as s5  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="proofs/v040/s5_forecast_gate_spec.json")
    args = ap.parse_args()

    # The usable cases (oracle wrfinput/wrfbdy + matching wps met_em + retained
    # CPU-WRF wrfout for the 24h reference).
    all_cases = C.discover_oracle_cases(
        require_domains=("d01",), require_wrfbdy=True)
    usable = []
    for oc in all_cases:
        wps = s5._wps_dir_for(oc.case_id)
        if wps is None or len(s5._ordered_metem_paths(wps, "d01")) < 2:
            continue
        # require a retained CPU-WRF wrfout_d01 at the init time (24h ref source)
        init_vt = s5._init_valid_time(oc.case_id)
        wrfout0 = oc.run_dir / f"wrfout_d01_{init_vt}"
        if not wrfout0.is_file():
            continue
        usable.append(oc)

    spec = ForecastGateSpec()
    plan = C.run_forecast_gate(
        s5.make_factory(lbc_intervals=1),
        spec=spec,
        cases=usable,
        out_path=None,
        execute=False,  # READY, NOT RUN — GPU step is manager-scheduled
    )

    # readiness checks (no GPU)
    cpu_roots = [Path(r) for r in spec.cpu_wrfout_roots]
    cg = Path("proofs/m20/continuous_gate.py")
    readiness = {
        "factory_wired": "driver.build_real_init via s5_native_init_parity.make_factory",
        "n_ready_cases": len(usable),
        "ready_case_ids": [oc.case_id for oc in usable],
        "cpu_wrfout_backfill_roots_exist": {
            str(r): r.is_dir() for r in cpu_roots},
        "retained_per_case_wrfout_present": all(
            (oc.run_dir / f"wrfout_d01_{s5._init_valid_time(oc.case_id)}").is_file()
            for oc in usable) if usable else False,
        "continuous_gate_module_present": cg.is_file(),
        "manager_invocation": (
            "C.run_forecast_gate(make_factory(lbc_intervals=1), execute=True, "
            "out_path='proofs/v040/s5_forecast_gate_report.json')  # GPU, ONE job at a time"
        ),
        "executed": False,
        "gpu_bound": True,
        "blocked_reason": (
            "single-GPU serialization point (shared with v0.3.0 parity + backfill "
            "replays); manager schedules ONE GPU job at a time"
        ),
    }
    plan["s5_readiness"] = readiness

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"FORECAST GATE READY (NOT RUN): n_ready_cases={len(usable)} "
          f"cont_gate={cg.is_file()} "
          f"backfill_root={all(r.is_dir() for r in cpu_roots)}")
    print(f"  wrote {out}")
    # ready iff we have cases + the metric module + a reference source
    ready = (len(usable) >= 1 and cg.is_file()
             and (any(r.is_dir() for r in cpu_roots)
                  or readiness["retained_per_case_wrfout_present"]))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
