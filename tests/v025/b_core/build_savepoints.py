"""CPU-only real-history stage fixtures; no synthetic atmosphere.

These are derived stage inputs, not internal pristine-WRF substep dumps. The
hourly tendency interpolation is the accepted cancellation-map construction;
its provenance and limitation accompany the payloads.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/v025")]
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["GPUWRF_FUSED_VERTICAL"] = "0"
import cpu_guard  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from real_state import load_real_snapshot, stage_advanced_state  # noqa: E402
from gpuwrf.contracts.state import Tendencies  # noqa: E402
from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG  # noqa: E402
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients  # noqa: E402
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, acoustic_substep_core  # noqa: E402
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf  # noqa: E402
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf  # noqa: E402
from gpuwrf.runtime.operational_mode import _acoustic_core_state_from_prep  # noqa: E402

UPDATED = ("u", "v", "w", "ph", "mu", "muts", "muave", "mudf", "ww",
           "theta", "theta_coupled_work", "theta_ave", "t_2ave", "p", "al",
           "pm1", "ru_m", "rv_m", "ww_m")


def build(domain, out):
    run = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/" +
               ("fastbind_r1" if domain == "d01" else "legacybind_r1"))
    histories = sorted(run.glob(f"wrfout_{domain}_*"))
    before = load_real_snapshot(run, domain=domain, wrfout_name=histories[0].name)
    snap = load_real_snapshot(run, domain=domain, wrfout_name=histories[1].name)
    dt, dx = (54., 9000.) if domain == "d01" else (18., 3000.)
    advanced, pair = stage_advanced_state(before, snap, dt)
    prep = small_step_prep_wrf(advanced, 2, dt, metrics=snap.metrics,
                              reference_state=snap.state, base_state=snap.base_state)
    pressure = calc_p_rho_wrf(prep)
    # The operational helper builds the real frozen pg_buoy_w, Coriolis,
    # curvature, rhs_ph and moist cqw fields from this real stage.
    grid = SimpleNamespace(projection=SimpleNamespace(dx_m=dx, dy_m=dx))
    nl = SimpleNamespace(grid=grid, metrics=snap.metrics, dt_s=dt,
                         top_lid=False, run_boundary=False,
                         h_sca_adv_order=2, boundary_config=DEFAULT_BOUNDARY_CONFIG)
    tendencies = Tendencies(u=jnp.zeros_like(prep.u_work), v=jnp.zeros_like(prep.v_work),
                            w=jnp.zeros_like(prep.w_work), theta=jnp.zeros_like(prep.theta_work),
                            ph=jnp.zeros_like(prep.ph_work), mu=jnp.zeros_like(prep.mu_work),
                            qv=jnp.zeros_like(prep.theta_work), p=jnp.zeros_like(prep.theta_work))
    carry = SimpleNamespace(ww=prep.ww_save, mudf=jnp.zeros_like(prep.mu_work))
    state = _acoustic_core_state_from_prep(carry, prep, pressure, nl, tendencies).replace(mu_work=prep.mu_work)
    cfg = AcousticCoreConfig(dt=dt/7., dx=dx, dy=dx, epssm=snap.config["epssm"],
                             dt_full=dt, periodic_x=False, specified=domain == "d01",
                             nested=domain == "d02", w_damping=1, damp_opt=3,
                             dampcoef=.2, zdamp=5000.)
    coefficients = calc_coef_w_wrf_coefficients(state.mut, snap.metrics,
                      dt=cfg.dt, epssm=cfg.epssm, cqw=state.cqw, c2a=state.c2a)
    arrays = {"state_" + name: np.asarray(value) for name, value in state.to_dict().items()
              if value is not None}
    arrays.update({name: np.asarray(value) for name, value in
                   zip(("a", "alpha", "gamma"), coefficients, strict=True)})
    fn = jax.jit(lambda s: acoustic_substep_core(s, a=coefficients[0],
                      alpha=coefficients[1], gamma=coefficients[2], cfg=cfg))
    current = state
    for step in range(1, 12):
        current = fn(current)
        if step in (1, 11):
            for name in UPDATED:
                arrays[f"ref{step}_" + name] = np.asarray(getattr(current, name))
    path = out / (domain + ".npz")
    np.savez(path, **arrays)
    manifest = dict(domain=domain, shape=list(state.theta.shape), config=asdict(cfg),
                    previous=before.provenance, current=snap.provenance, stage_pair=pair,
                    payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    fields=list(arrays), all_finite=all(np.isfinite(x).all() for x in arrays.values()),
                    limitations="Real-history-derived RK2 inputs; zero horizontal/theta external tendencies; "
                    "real operational pg_buoy/rhs_ph; no boundary forcing. Not WRF internal dumps.")
    (out / (domain + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("domain", "shape", "all_finite", "payload_sha256")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for domain in ("d01", "d02"):
        build(domain, args.out)
