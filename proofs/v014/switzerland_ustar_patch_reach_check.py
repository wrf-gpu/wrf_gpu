#!/usr/bin/env python
"""Fast check: does patching physics_couplers.surface_layer change the state
ustar/tau AND the post-PBL winds after one operational physics step?

Confirms (a) the patch reaches the live surface->PBL path and (b) whether the
scaled momentum drag actually changes the prognostic winds. If the winds are
byte-identical with tau scaled x2.69, the MYNN momentum drag is inert in the
live dynamics path (the venting is NOT surface-drag-driven at the forecast level).
"""
from __future__ import annotations
import dataclasses, importlib.util, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
P = ROOT / "proofs/v014"
_H = importlib.util.spec_from_file_location("hpg", P / "switzerland_hpg_native_face_fix.py")
hpg = importlib.util.module_from_spec(_H); _H.loader.exec_module(hpg)


def run(scale):
    import jax.numpy as jnp
    import gpuwrf.coupling.physics_couplers as pc
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry

    case, state0, _ = hpg._build_state(hpg.PROBE_ROOT)
    nl = dataclasses.replace(case.namelist, dt_s=18.0, acoustic_substeps=4)

    orig = pc.surface_layer
    if scale != 1.0:
        s = float(scale)
        def wrapped(view, *, first_timestep=False):
            f = orig(view, first_timestep=first_timestep)
            return f._replace(ustar=f.ustar * s, tau_u=f.tau_u * (s * s), tau_v=f.tau_v * (s * s))
        pc.surface_layer = wrapped
    try:
        carry0 = initial_operational_carry(state0)
        lead = jnp.asarray(1, jnp.int32).astype(jnp.float64) * float(nl.dt_s)
        c1 = om._physics_step_forcing(carry0, nl, lead, run_radiation=bool(nl.run_physics), first_timestep=False)
    finally:
        pc.surface_layer = orig
    st = c1.state if hasattr(c1, "state") else c1[0].state
    dry = c1.dry if hasattr(c1, "dry") else c1[2]
    rut = np.asarray(dry.ru_tendf)
    return {
        "ustar_mean": float(np.nanmean(np.asarray(st.ustar))),
        "tau_u_mean": float(np.nanmean(np.asarray(st.tau_u))),
        "u_mean": float(np.nanmean(np.asarray(st.u))),
        "u_k0_mean": float(np.nanmean(np.asarray(st.u)[0])),
        "ru_tendf_k0_meanabs": float(np.nanmean(np.abs(rut[0]))),
        "ru_tendf_k0_mean": float(np.nanmean(rut[0])),
        "ru_tendf_meanabs": float(np.nanmean(np.abs(rut))),
    }


if __name__ == "__main__":
    base = run(1.0)
    scaled = run(1.64)
    print("BASE  ", base)
    print("SCALED", scaled)
    print("ustar ratio:", scaled["ustar_mean"] / base["ustar_mean"])
    print("tau_u ratio:", scaled["tau_u_mean"] / base["tau_u_mean"])
    print("u_k0 delta:", scaled["u_k0_mean"] - base["u_k0_mean"])
    print("ru_tendf_k0_meanabs ratio:",
          scaled["ru_tendf_k0_meanabs"] / max(base["ru_tendf_k0_meanabs"], 1e-30))
    print("ru_tendf_k0_meanabs base/scaled:",
          base["ru_tendf_k0_meanabs"], scaled["ru_tendf_k0_meanabs"])
    print("ru_tendf_meanabs base/scaled:",
          base["ru_tendf_meanabs"], scaled["ru_tendf_meanabs"])
