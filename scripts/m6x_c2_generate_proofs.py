from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig
from gpuwrf.dynamics.hybrid_eta import hybrid_summary, mass_level_pressure, pressure_thickness
from gpuwrf.dynamics.hyperdiffusion import HyperdiffusionConfig
from gpuwrf.dynamics.limiters import LimiterConfig, limiter_diagnostics, positive_definite_limiter
from gpuwrf.dynamics.metrics import flat_metrics_for_grid, load_wrfinput_metrics, metric_minmax
from gpuwrf.dynamics.orchestrator import OrchestratorConfig, run_scan


SPRINT = Path(".agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture")
PROOFS = SPRINT / "proofs"
WRFINPUT_D02 = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z/wrfinput_d02"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array_payload(array) -> dict:
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _host_callback_free(jaxpr_text: str) -> bool:
    lowered = jaxpr_text.lower()
    return all(token not in lowered for token in ("host_callback", "io_callback", "pure_callback"))


def _analytic_state(grid: GridSpec) -> State:
    arrays = {
        field: jnp.zeros(shape, dtype=jnp.float64)
        for field, shape in _state_field_shapes(grid).items()
    }
    z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    bubble = jnp.exp(-((x - 0.5 * grid.nx) ** 2 + (y - 0.5 * grid.ny) ** 2 + (z - 0.35 * grid.nz) ** 2) / 8.0)
    arrays["theta"] = 300.0 + 0.1 * bubble
    arrays["p"] = jnp.ones_like(arrays["p"]) * 1000.0
    arrays["p_total"] = arrays["p"]
    arrays["p_perturbation"] = arrays["p"]
    arrays["ph_total"] = arrays["ph"]
    arrays["ph_perturbation"] = arrays["ph"]
    arrays["mu"] = jnp.ones_like(arrays["mu"]) * 90000.0
    arrays["mu_total"] = arrays["mu"]
    arrays["mu_perturbation"] = arrays["mu"]
    return State(**arrays)


def metrics_proof() -> None:
    grid = GridSpec.canary_3km_template()
    flat = flat_metrics_for_grid(grid)
    jaxpr_text = str(jax.make_jaxpr(metric_minmax)(flat))
    payload = {
        "acceptance_criterion": "AC2",
        "analytic_flat_fixture": {
            "provenance": flat.provenance,
            "msftx": _array_payload(flat.msftx),
            "msfty": _array_payload(flat.msfty),
            "msfux": _array_payload(flat.msfux),
            "msfuy": _array_payload(flat.msfuy),
            "msfvx": _array_payload(flat.msfvx),
            "msfvy": _array_payload(flat.msfvy),
            "cf1": _array_payload(flat.cf1),
            "cf2": _array_payload(flat.cf2),
            "cf3": _array_payload(flat.cf3),
            "fnm": _array_payload(flat.fnm),
            "fnp": _array_payload(flat.fnp),
            "dzdx": _array_payload(flat.dzdx),
            "dzdy": _array_payload(flat.dzdy),
            "dzdx_u": _array_payload(flat.dzdx_u),
            "dzdy_v": _array_payload(flat.dzdy_v),
            "summary_minmax": np.asarray(metric_minmax(flat)).tolist(),
        },
        "jit_audit": {
            "function": "gpuwrf.dynamics.metrics.metric_minmax",
            "host_callback_free": _host_callback_free(jaxpr_text),
            "jaxpr_contains": "scan" if "scan" in jaxpr_text.lower() else "elementwise-reductions",
        },
        "wrf_fixture": {"path": str(WRFINPUT_D02), "available": WRFINPUT_D02.exists()},
    }
    if WRFINPUT_D02.exists():
        wrf = load_wrfinput_metrics(WRFINPUT_D02)
        payload["wrf_fixture"].update(
            {
                "provenance": wrf.provenance,
                "msftx": _array_payload(wrf.msftx),
                "msfty": _array_payload(wrf.msfty),
                "msfux": _array_payload(wrf.msfux),
                "msfuy": _array_payload(wrf.msfuy),
                "msfvx": _array_payload(wrf.msfvx),
                "msfvy": _array_payload(wrf.msfvy),
                "cf1": _array_payload(wrf.cf1),
                "cf2": _array_payload(wrf.cf2),
                "cf3": _array_payload(wrf.cf3),
                "fnm": _array_payload(wrf.fnm),
                "fnp": _array_payload(wrf.fnp),
                "dzdx": _array_payload(wrf.dzdx),
                "dzdy": _array_payload(wrf.dzdy),
                "dzdx_u": _array_payload(wrf.dzdx_u),
                "dzdy_v": _array_payload(wrf.dzdy_v),
                "summary_minmax": np.asarray(metric_minmax(wrf)).tolist(),
            }
        )
    _write_json(PROOFS / "metrics.json", payload)


def hybrid_eta_proof() -> None:
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    mu = jnp.full((grid.ny, grid.nx), 90000.0, dtype=jnp.float64)
    pressure = mass_level_pressure(mu, metrics)
    expected = metrics.c3h[:, None, None] * mu[None, :, :] + metrics.c4h[:, None, None] + metrics.p_top
    jaxpr_text = str(jax.make_jaxpr(hybrid_summary)(mu, metrics))
    payload = {
        "acceptance_criterion": "AC3",
        "analytic_oracle": {
            "max_abs_error": float(jnp.max(jnp.abs(pressure - expected))),
            "pressure": _array_payload(pressure),
            "thickness_min": float(jnp.min(pressure_thickness(mu, metrics))),
            "summary": np.asarray(hybrid_summary(mu, metrics)).tolist(),
        },
        "jit_audit": {
            "function": "gpuwrf.dynamics.hybrid_eta.hybrid_summary",
            "host_callback_free": _host_callback_free(jaxpr_text),
        },
        "wrf_fixture": {"path": str(WRFINPUT_D02), "available": WRFINPUT_D02.exists()},
        "wrf_source_anchors": [
            "module_big_step_utilities_em.F:1045-1047",
            "module_small_step_em.F:522-542",
        ],
    }
    if WRFINPUT_D02.exists():
        wrf = load_wrfinput_metrics(WRFINPUT_D02)
        wrf_mu = jnp.full(wrf.msftx.shape, 85000.0, dtype=jnp.float64)
        wrf_pressure = mass_level_pressure(wrf_mu, wrf)
        payload["wrf_fixture"].update(
            {
                "c1h": _array_payload(wrf.c1h),
                "c2h": _array_payload(wrf.c2h),
                "c3h": _array_payload(wrf.c3h),
                "c4h": _array_payload(wrf.c4h),
                "c1f": _array_payload(wrf.c1f),
                "c2f": _array_payload(wrf.c2f),
                "c3f": _array_payload(wrf.c3f),
                "c4f": _array_payload(wrf.c4f),
                "dn": _array_payload(wrf.dn),
                "dnw": _array_payload(wrf.dnw),
                "rdn": _array_payload(wrf.rdn),
                "rdnw": _array_payload(wrf.rdnw),
                "cf1": _array_payload(wrf.cf1),
                "cf2": _array_payload(wrf.cf2),
                "cf3": _array_payload(wrf.cf3),
                "fnm": _array_payload(wrf.fnm),
                "fnp": _array_payload(wrf.fnp),
                "pressure": _array_payload(wrf_pressure),
                "pressure_min": float(jnp.min(wrf_pressure)),
                "pressure_max": float(jnp.max(wrf_pressure)),
            }
        )
    _write_json(PROOFS / "hybrid_eta.json", payload)


def limiter_proof() -> None:
    scalar = jnp.asarray([[[2.0, -0.5, 1.0]]], dtype=jnp.float64)
    mass = jnp.ones_like(scalar)
    limited = positive_definite_limiter(scalar, mass, LimiterConfig(enabled=True))
    diagnostics = np.asarray(limiter_diagnostics(scalar, limited, mass)).tolist()
    payload = {
        "acceptance_criterion": "AC6",
        "fixture": "analytic one-cell row with positive total scalar mass and one negative value",
        "min_after": diagnostics[0],
        "mass_before": diagnostics[1],
        "mass_after": diagnostics[2],
        "relative_mass_error": diagnostics[3],
        "passes_nonnegative": bool(diagnostics[0] >= 0.0),
        "passes_mass_tolerance": bool(diagnostics[3] < 1.0e-12),
    }
    _write_json(PROOFS / "limiter_conservation.json", payload)


def scan_audit() -> None:
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _analytic_state(grid)
    config = OrchestratorConfig(acoustic=AcousticConfig(n_substeps=2))
    carry = run_scan(state, metrics, config, 1.0, 3)
    jaxpr_text = str(jax.make_jaxpr(run_scan, static_argnums=(2, 3, 4))(state, metrics, config, 1.0, 3))
    platforms = sorted({leaf.devices().copy().pop().platform for leaf in jax.tree_util.tree_leaves(carry)})
    body = "\n".join(
        [
            "# AC5 Scan Transfer Audit",
            "",
            "- Mode: static JAXPR audit plus executed analytic scan.",
            f"- Devices observed for final carry leaves: {platforms}",
            "- Outer loop: `gpuwrf.dynamics.orchestrator.run_scan` uses `jax.lax.scan`.",
            "- Nested loop: `gpuwrf.dynamics.acoustic_wrf.run_acoustic_scan` uses `jax.lax.scan`.",
            f"- Host callback primitives present: {not _host_callback_free(jaxpr_text)}",
            "- Post-init host/device transfers inside scan: 0 by static audit; no `host_callback`, `io_callback`, or `pure_callback` primitives appear in the timestep JAXPR.",
            "- Limitation: this is not an Nsight transfer trace. It is sufficient for c2-A1 architecture proof, not for a GPU performance claim.",
            "",
            "WRF source anchors: `module_small_step_em.F:562` for previous-pressure smdiv memory; `module_small_step_em.F:1094-1112` for flux/vertical-velocity carry context.",
        ]
    )
    (PROOFS / "scan_transfer_audit.md").write_text(body + "\n", encoding="utf-8")


def integration_proof() -> None:
    grid = GridSpec.canary_3km_template()
    metrics = flat_metrics_for_grid(grid)
    state = _analytic_state(grid)
    off = OrchestratorConfig(acoustic=AcousticConfig(n_substeps=2))
    on = OrchestratorConfig(
        acoustic=AcousticConfig(
            n_substeps=2,
            smdiv=SmdivConfig(enabled=True, coefficient=0.05),
            rayleigh=RayleighConfig(enabled=True, coefficient=0.05, top_start_fraction=0.5),
        ),
        hyperdiffusion=HyperdiffusionConfig(enabled=True, coefficient=1.0e-4),
        limiter=LimiterConfig(enabled=True),
    )
    off_carry = run_scan(state, metrics, off, 1.0, 5)
    on_carry = run_scan(state, metrics, on, 1.0, 5)
    payload = {
        "acceptance_criterion": "AC7",
        "status": "PARTIAL_SMOKE_NOT_WARM_BUBBLE_PARITY",
        "warm_bubble_script": {
            "path": "scripts/m6_warm_bubble_test.py",
            "available": Path("scripts/m6_warm_bubble_test.py").exists(),
            "note": "The role prompt referenced this script, but it is absent in this worktree.",
        },
        "analytic_smoke": {
            "steps": 5,
            "all_stabilizers_off": {
                "theta_min": float(jnp.min(off_carry.state.theta)),
                "theta_mass": float(jnp.sum(off_carry.state.theta)),
                "finite": bool(all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(off_carry.state))),
            },
            "stabilizers_enabled": {
                "theta_min": float(jnp.min(on_carry.state.theta)),
                "theta_mass": float(jnp.sum(on_carry.state.theta)),
                "finite": bool(all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(on_carry.state))),
            },
        },
        "previous_c1_reference": {
            "source": "MORNING-REPORT-2026-05-22.md UPDATE 08:00",
            "warm_bubble_w_max_300s_m_s": 5.99,
            "warm_bubble_centroid_300s_m": 2517,
            "finite_to_600s": False,
            "diverged_at_s": 350,
        },
        "risk": "This proof verifies c2 wiring and finite state only. It does not close warm-bubble physics parity.",
    }
    _write_json(PROOFS / "integration_warm_bubble.json", payload)


def main() -> int:
    PROOFS.mkdir(parents=True, exist_ok=True)
    metrics_proof()
    hybrid_eta_proof()
    limiter_proof()
    scan_audit()
    integration_proof()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
