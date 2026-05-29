"""F7L official idealized-case proof run: Straka 900s + warm bubble 500s.

Writes the F7L proof artifacts (JSON + verdict + plots) via the operational
case runners.  Straka now carries WRF-faithful const-K ν=75 on u,v,w,θ.
"""

from __future__ import annotations

import json
import sys

from jax import config

config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import run_density_current_case, run_warm_bubble_case


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    results = {}
    if which in ("straka", "both"):
        print("=== STRAKA density current -> 900s ===", flush=True)
        r = run_density_current_case(proof_dir="proofs/f7l", require_gpu=True)
        results["density_current"] = {"verdict": r.verdict, "status": r.status}
        print(f"STRAKA verdict={r.verdict} status={r.status}", flush=True)
        for name, row in r.checks.items():
            print(f"  {name}: value={row.get('value')} passed={row.get('passed')}", flush=True)
    if which in ("bubble", "both"):
        print("=== Skamarock warm bubble -> 500s (no-regression) ===", flush=True)
        r = run_warm_bubble_case(proof_dir="proofs/f7l", require_gpu=True)
        results["warm_bubble"] = {"verdict": r.verdict, "status": r.status}
        print(f"BUBBLE verdict={r.verdict} status={r.status}", flush=True)
        for name, row in r.checks.items():
            print(f"  {name}: value={row.get('value')} passed={row.get('passed')}", flush=True)
    print("SUMMARY " + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
