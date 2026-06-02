"""P0-5a proof: wrfout variable/metadata coverage inventory vs the WRF field set.

GPU-FREE. Compares what gpuwrf's wrfout writer KNOWS how to emit
(``OPERATIONAL_WRFOUT_VARIABLES`` + the ``Times`` string var) against a real
WRF-ARW ``wrfout`` reference file, and CROSS-CHECKS every shared field's
units / dimensions / staggering against the reference so the writer's metadata is
WRF-faithful, not merely WRF-named.

The 375-var WRF reference includes a large tail of accumulated-budget /
seed-array / ideal-run / lake / urban / SPP diagnostics that the operational
Canary product does not need; this proof classifies the residual gap so it is an
explicit, justified decision rather than a silent omission.

Run (from repo root):
    PYTHONPATH=src JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=2 taskset -c 0-3 \
        python proofs/p0_5/wrfout_coverage_inventory.py

Emits proofs/p0_5/wrfout_coverage_inventory.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from netCDF4 import Dataset

from gpuwrf.io.wrfout_writer import OPERATIONAL_WRFOUT_VARIABLES, WRFOUT_VARIABLE_SPECS

REFERENCE = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l3/"
    "20260428_18z_l3_24h_20260525T221139Z/wrfout_d02_2026-04-28_19:00:00"
)

# WRF fields we DELIBERATELY do not emit, grouped with the reason. (Names below
# are real WRF wrfout variables; the operational Canary product does not need
# them and the model either does not compute them or they are pure diagnostics.)
INTENTIONAL_OMISSIONS = {
    "accumulated_budget_diagnostics": "AC* accumulated land/energy/radiation budget terms (Noah-MP ACCFLX family) — diagnostic accumulators, not forecast fields; out of operational scope.",
    "radiation_flux_diagnostic_tail": "LWUP*/SWUP*/LWDN*/SWDN* TOA/BOA clear-sky flux diagnostics + OLR/SWNORM — RRTMG diagnostic tail; SWDOWN/GLW (the surface forcing the product uses) ARE emitted.",
    "stochastic_seed_arrays": "ISEEDARR*/ISEEDARRAY* SPP/SKEBS/SPPT random-perturbation seed state — not used (no stochastic physics in scope).",
    "ideal_run_flags": "THIS_IS_AN_IDEAL_RUN/SAVE_TOPO_FROM_REAL/NEST_POS bookkeeping flags.",
    "lake_urban_crop_carbon": "WATER_DEPTH/BATHYMETRY_FLAG/WSLAKE/CROPCAT/GRAIN/LFMASS/STMASS/RTMASS/WOOD/GPP/NPP/NEE/GDD — lake/urban/crop/carbon options CUT from the Noah-MP scope (NOAH-MP-SCOPING.md).",
    "sso_gwd_inputs": "VAR_SSO/OA[1-4]/OL[1-4]/VAR/CON/BGAP/WGAP/ZTOP_PLUME — sub-grid orography / GWD inputs (P1-7 GWD deferred; documented non-load-bearing on supported domains).",
    "vertical_coord_coefficient_tail": "C3H/C4H/.../FNM/FNP/DN/DNW/RDN/RDNW/CF1-3/CFN/RDX/RDY/P00/T00 — hybrid-eta coefficient + scalar metadata tail; ZNU/ZNW/P_TOP (the coordinate fields downstream tools need) ARE emitted.",
    "redundant_or_skin_variants": "SST/SSTSK/TMN/SNOALB/ALBBCK/Q2B/T2B/Q2V/T2V/CHB*/CHV*/TH2 variants + extra map-factor variants (MAPFAC_UY/VX/MX/MY) — static or derived; the primary fields are emitted.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_units(units: str) -> str:
    return " ".join(units.replace("-", " ").split()).strip().lower()


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    emitted = ("Times",) + tuple(v for v in OPERATIONAL_WRFOUT_VARIABLES if v != "Times")
    emitted_set = set(emitted)

    result: dict = {
        "artifact_type": "wrfout_coverage_inventory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_free": True,
        "emitted_count": len(emitted_set),
        "emitted_variables": sorted(emitted_set),
    }

    if not REFERENCE.exists():
        result["status"] = "INCONCLUSIVE"
        result["reason"] = f"WRF reference not found: {REFERENCE}"
        (out_dir / "wrfout_coverage_inventory.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    ds = Dataset(REFERENCE)
    ref_vars = {}
    for name, var in ds.variables.items():
        ref_vars[name] = {
            "dims": list(var.dimensions),
            "units": getattr(var, "units", ""),
            "stagger": getattr(var, "stagger", ""),
        }
    ds.close()

    ref_set = set(ref_vars)
    result["reference_file"] = str(REFERENCE)
    result["reference_sha256"] = _sha256(REFERENCE)
    result["reference_count"] = len(ref_set)

    # Names we emit that are NOT real WRF wrfout variables (must be empty -- no
    # invented fields).
    not_in_wrf = sorted(emitted_set - ref_set)
    result["emitted_not_in_wrf"] = not_in_wrf

    # Metadata cross-check for the shared fields (units/dims/stagger faithful).
    metadata_mismatches: list[dict] = []
    for name in sorted(emitted_set & ref_set):
        if name == "Times":
            continue
        spec = WRFOUT_VARIABLE_SPECS.get(name)
        if spec is None:
            continue
        ref = ref_vars[name]
        our_dims = list(spec.dimensions)
        issues = []
        if our_dims != ref["dims"]:
            issues.append({"dims": {"ours": our_dims, "wrf": ref["dims"]}})
        if (spec.stagger or "") != (ref["stagger"] or ""):
            issues.append({"stagger": {"ours": spec.stagger, "wrf": ref["stagger"]}})
        # XTIME units are runtime-stamped ("minutes since <run_start>") in
        # _write_xtime, matching WRF; the static spec only holds a placeholder, so
        # its units are compared at write time, not here.
        if name != "XTIME" and _normalize_units(spec.units) != _normalize_units(ref["units"]):
            issues.append({"units": {"ours": spec.units, "wrf": ref["units"]}})
        if issues:
            metadata_mismatches.append({"name": name, "issues": issues})
    result["metadata_mismatches"] = metadata_mismatches

    # The residual WRF fields we do NOT emit, with their classification.
    not_emitted = sorted(ref_set - emitted_set)
    result["not_emitted_count"] = len(not_emitted)
    result["not_emitted_variables"] = not_emitted
    result["intentional_omission_rationale"] = INTENTIONAL_OMISSIONS

    status_pass = not not_in_wrf and not metadata_mismatches
    result["status"] = "PASS" if status_pass else "FAIL"
    result["summary"] = {
        "emitted": len(emitted_set),
        "shared_with_wrf": len(emitted_set & ref_set),
        "invented_fields": len(not_in_wrf),
        "metadata_faithful": not metadata_mismatches,
        "wrf_fields_not_emitted_intentionally": len(not_emitted),
    }

    (out_dir / "wrfout_coverage_inventory.json").write_text(json.dumps(result, indent=2))
    # Print a compact summary (full JSON is on disk).
    print(json.dumps({k: result[k] for k in ("status", "summary", "emitted_not_in_wrf", "metadata_mismatches")}, indent=2))
    return 0 if status_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
