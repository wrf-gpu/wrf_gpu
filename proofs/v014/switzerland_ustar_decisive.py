#!/usr/bin/env python
"""V0.14 Switzerland venting — DECISIVE forecast experiment for the ustar root.

The flux-localizer + ustar-drag-root proofs localized the depth-8 venting to a
domain-wide low-level westerly bias driven by a sfclayrev/MYNN-surface ``ustar``
that is only ~61 % of the WRF h36 UST, so the explicit MYNN surface momentum
drag (``rhosfc*ust^2/wspd``) is ~37 % of WRF's and the low-level wind is
under-braked. Those proofs only established the SUBSTEP sign-flip.

This script adjudicates the root at the FORECAST level: it runs a 2 h open-top
forecast from the h36 reinit with the surface-layer ``ustar`` (and the
consistent momentum stress ``tau``) scaled by a runtime knob, then measures the
depth-8 hourly excess outflux against the CPU truth using the SAME
``switzerland_hpg_native_face_fix.budget_between`` control surface as every prior
venting number.

This is a DECISIVE EXPERIMENT, not a fix: the scale is a falsifiable knob. If the
h37 excess collapses from -26.5 toward ~0, the surface-layer ustar magnitude is
the venting root at the forecast level and the next step is the WRF-faithful
closure fix in ``src/gpuwrf/physics/surface_layer.py``.

The ``ustar`` scale is injected by wrapping
``gpuwrf.physics.noahmp_coupler.surface_layer_with_diagnostics`` (the live
Switzerland NoahMP surface path) so the returned diagnostics carry
``ustar*=s`` and ``tau_{u,v}*=s^2`` (tau ~ ustar^2). No model source is changed.

Usage (GPU; wrap with scripts/with_gpu_lock.sh):

    python proofs/v014/switzerland_ustar_decisive.py --scale 1.64 --hours 2 \
        --label x164
    python proofs/v014/switzerland_ustar_decisive.py --budget-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_HPG_SPEC = importlib.util.spec_from_file_location(
    "hpg_native_face_proof", Path(__file__).with_name("switzerland_hpg_native_face_fix.py")
)
hpg = importlib.util.module_from_spec(_HPG_SPEC)
_HPG_SPEC.loader.exec_module(hpg)  # type: ignore[union-attr]

CPU = hpg.CPU
PROBE_ROOT = hpg.PROBE_ROOT
OUT_JSON = ROOT / "proofs/v014/switzerland_ustar_decisive.json"


def _patch_ustar_scale(scale: float):
    """Monkeypatch the LIVE bulk surface path to scale the sfclay ustar/tau.

    The Switzerland d01 forecast runs ``use_noahmp=False`` + ``sf_sfclay_physics=5``
    (not in SFCLAY_SCAN_ADAPTERS), so the surface slot dispatches to
    ``physics_couplers.surface_adapter`` -> ``physics_couplers.surface_layer``
    (NOT the noahmp_coupler path). Patch ``physics_couplers.surface_layer`` so the
    momentum handles written into the state (ustar, tau_u, tau_v) are scaled.

    Returns a restore callable. ``ustar`` scales by ``s`` and ``tau_{u,v}`` by
    ``s^2`` so the kinematic stress stays consistent with ``tau ~ -ustar^2*u/|U|``
    (the MYNN bottom drag rebuilds from ustar / this stress). theta/qv fluxes are
    LEFT UNCHANGED -- this is a pure momentum-drag probe.
    """

    import gpuwrf.coupling.physics_couplers as pc

    orig = pc.surface_layer
    s = float(scale)

    def wrapped(state_view, *, first_timestep=False):
        flux = orig(state_view, first_timestep=first_timestep)
        return flux._replace(
            ustar=flux.ustar * s,
            tau_u=flux.tau_u * (s * s),
            tau_v=flux.tau_v * (s * s),
        )

    pc.surface_layer = wrapped

    def restore():
        pc.surface_layer = orig

    return restore


def run_forecast(scale: float, hours: int, label: str) -> Path:
    from gpuwrf.integration import daily_pipeline as dp

    out_dir = PROBE_ROOT / f"gpu_output_ustar_{label}"
    config = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=int(hours),
        output_dir=out_dir,
        proof_dir=PROBE_ROOT / f"proofs_ustar_{label}",
        run_root=PROBE_ROOT,
        domain="d01",
    )

    restore = _patch_ustar_scale(scale)
    try:
        result = dp._run_forecast_sequence(config, output_dir=config.output_dir)
    finally:
        restore()
    print(
        f"forecast ustar_scale={scale} label={label}: status={result.status} "
        f"hours={result.hours} output_dir={result.output_dir}",
        flush=True,
    )
    return out_dir


def budget_report(labels: dict[str, Path]) -> dict:
    cpu37 = hpg.budget_between(CPU, 36, CPU, 37, depth=8)
    out = {
        "schema": "v014_switzerland_ustar_decisive",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_reference": str(CPU),
        "depth": 8,
        "cpu_budget_h36_h37": cpu37,
        "baseline_phys_tendf_excess_h37": -26.544642857142847,
        "variants": {},
    }
    cpu38 = None
    if hpg.fn(CPU, 38).exists():
        cpu38 = hpg.budget_between(CPU, 36, CPU, 38, depth=8)
        out["cpu_budget_h36_h38"] = cpu38
    for name, base in labels.items():
        rec: dict = {"path": str(base)}
        if hpg.fn(base, 37).exists():
            b37 = hpg.budget_between(CPU, 36, base, 37, depth=8)
            rec["budget_h37"] = b37
            rec["excess_h37"] = float(
                b37["net_influx_pa_per_cell_h"] - cpu37["net_influx_pa_per_cell_h"]
            )
            rec["metrics_h37"] = hpg.field_metrics(base, 37)
        if cpu38 is not None and hpg.fn(base, 38).exists():
            b38 = hpg.budget_between(CPU, 36, base, 38, depth=8)
            rec["budget_h38"] = b38
            rec["excess_h38"] = float(
                b38["net_influx_pa_per_cell_h"] - cpu38["net_influx_pa_per_cell_h"]
            )
            rec["metrics_h38"] = hpg.field_metrics(base, 38)
        out["variants"][name] = rec
    return out


def discover_variants() -> dict[str, Path]:
    labels: dict[str, Path] = {
        "phys_tendf_baseline": PROBE_ROOT / "gpu_output_phys_tendf",
    }
    for d in sorted(PROBE_ROOT.glob("gpu_output_ustar_*")):
        labels[d.name.replace("gpu_output_", "")] = d
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--budget-only", action="store_true")
    args = parser.parse_args()

    if args.scale is not None and not args.budget_only:
        label = args.label or f"x{int(round(args.scale * 100)):03d}"
        run_forecast(args.scale, args.hours, label)

    labels = discover_variants()
    report = budget_report(labels)
    OUT_JSON.write_text(json.dumps(report, indent=1, default=float))
    summary = {
        "cpu_excess_ref": 0.0,
        "baseline_phys_tendf_excess_h37": report["baseline_phys_tendf_excess_h37"],
        "variants": {
            n: {"excess_h37": v.get("excess_h37"), "excess_h38": v.get("excess_h38"),
                "u_rmse_h37": (v.get("metrics_h37") or {}).get("u", {}).get("rmse"),
                "u_bias_h37": (v.get("metrics_h37") or {}).get("u", {}).get("bias"),
                "finite_h37": (v.get("metrics_h37") or {}).get("u", {}).get("finite")}
            for n, v in report["variants"].items()
        },
    }
    print(json.dumps(summary, indent=1, default=float))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
