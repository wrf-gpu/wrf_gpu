#!/usr/bin/env python
"""Gate M6-S7 Tier-4 probtest prototype artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.io.proof_schemas import Tier4ProbtestTolerances
from gpuwrf.validation.tier4_probtest import DEFAULT_LEADS_H, DEFAULT_VARIABLES, PROTOTYPE_LABEL


DEFAULT_DIR = ROOT / "artifacts" / "m6" / "tier4"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, object]:
    tolerance_path = Path(args.tolerances)
    heldout_path = Path(args.heldout)
    cost_path = Path(args.cost_model)
    if not tolerance_path.is_absolute():
        tolerance_path = ROOT / tolerance_path
    if not heldout_path.is_absolute():
        heldout_path = ROOT / heldout_path
    if not cost_path.is_absolute():
        cost_path = ROOT / cost_path

    tolerances = Tier4ProbtestTolerances.validate_file(tolerance_path)
    if tolerances["prototype_label"] != PROTOTYPE_LABEL:
        raise RuntimeError(f"Tier-4 prototype label mismatch: {tolerances['prototype_label']!r}")
    if tolerances["sample_size"] != 10:
        raise RuntimeError(f"Tier-4 sample size must be 10 for M6-S7, got {tolerances['sample_size']}")
    if "min(raw, cap)" in json.dumps(tolerances):
        raise RuntimeError("Tier-4 artifact contains forbidden min(raw, cap) fudge language")

    required_strata = {"land", "sea"}
    if not required_strata.issubset(set(tolerances["strata"])):
        raise RuntimeError(f"Tier-4 strata missing {sorted(required_strata - set(tolerances['strata']))}")
    if not any(stratum.startswith("elevation_band_") for stratum in tolerances["strata"]):
        raise RuntimeError("Tier-4 strata missing elevation bands")

    missing: list[str] = []
    for variable in DEFAULT_VARIABLES:
        if variable not in tolerances["tolerances"]:
            missing.append(variable)
            continue
        for lead_h in DEFAULT_LEADS_H:
            lead_key = f"{lead_h}h"
            if lead_key not in tolerances["tolerances"][variable]:
                missing.append(f"{variable}.{lead_key}")
                continue
            for stratum in tolerances["strata"]:
                if stratum not in tolerances["tolerances"][variable][lead_key]:
                    missing.append(f"{variable}.{lead_key}.{stratum}")
    if missing:
        raise RuntimeError(f"Tier-4 tolerance table incomplete: {missing[:10]}")

    heldout = _load(heldout_path)
    cost_model = _load(cost_path)
    if args.require_heldout_pass and heldout.get("status") != "PASS":
        raise RuntimeError(f"Tier-4 held-out candidate gate failed: {heldout.get('status')} failures={heldout.get('failures', [])[:5]}")
    if cost_model.get("prototype_label") != PROTOTYPE_LABEL:
        raise RuntimeError("Tier-4 cost model missing prototype label")
    if cost_model.get("recommended_m7_ensemble_size") is None:
        raise RuntimeError("Tier-4 cost model missing M7 ensemble recommendation")
    return {
        "status": "PASS",
        "tolerances": str(tolerance_path),
        "heldout_status": heldout.get("status"),
        "cost_model_status": cost_model.get("status"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerances", default=str(DEFAULT_DIR / "probtest_tolerances.json"))
    parser.add_argument("--heldout", default=str(DEFAULT_DIR / "heldout_candidate_validation.json"))
    parser.add_argument("--cost-model", default=str(DEFAULT_DIR / "cost_model.json"))
    parser.add_argument("--allow-heldout-fail", dest="require_heldout_pass", action="store_false")
    parser.set_defaults(require_heldout_pass=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
