"""o1-smag3d mutant check: each mutation of the literal operator must break pristine-oracle parity.

usage: JAX_PLATFORMS=cpu python proofs/o1_smag3d/mutants.py <out.json>
Uses the committed fixture and the same gate as tests/dynamics/test_les3d_smagorinsky.py.  A mutant is KILLED
only when the gate reports violations (assertion-level signature: violating field names), never on exceptions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "dynamics"))
import test_les3d_smagorinsky as T  # noqa: E402

from gpuwrf.dynamics import les3d_smagorinsky as les  # noqa: E402


def m_ring(orig):
    """smag_km writes the outer mass ring too (WRF leaves it at its zero init)."""
    def f(m, periodic, *a, **k):
        saved = (m.ids, m.ide, m.jds, m.jde)
        m.ids, m.ide, m.jds, m.jde = m.ids - 1, m.ide + 1, m.jds - 1, m.jde + 1
        try:
            return orig(m, periodic, *a, **k)
        finally:
            m.ids, m.ide, m.jds, m.jde = saved
    return f


def m_noflux(orig):
    """vertical_diffusion_2 without the surface stress / heat / moisture flux terms."""
    def f(m, cfg, periodic, ru, rv, rt, mt, u, v, thp, th_phy, moist, iqv, xkhv, rho, fnm, fnp, rdz, dnw, *rest):
        out = [les._vertical_diffusion_s(m, periodic, mt[n], q, xkhv, rho, fnm, fnp, rdz, dnw) for n, q in enumerate(moist)]
        return ru, rv, les._vertical_diffusion_s(m, periodic, rt, thp, xkhv, rho, fnm, fnp, rdz, dnw), out
    return f


def m_bc_east(orig):
    """set_physical_bc3d without the 'd'/'e' east overwrite dat(ide)=dat(ide-1) (treat like 'u')."""
    def f(m, a, var, periodic):
        return orig(m, a, "u" if var in ("d", "e") else var, periodic)
    return f


MUTANTS = {
    "ring_written": ("_smag_km", m_ring, "real/spec_vert_isfflx1"),
    "no_surface_fluxes": ("_vertical_surface_and_scalars", m_noflux, "real/spec_vert_isfflx1"),
    "bc_east_stagger": ("_bc3", m_bc_east, "real/spec_vert_isfflx1"),
}


def main(out: str) -> int:
    import numpy as np

    zz = np.load(T.FIXTURE)
    meta = json.loads(bytes(zz["meta_json"]).decode())
    z = {k: zz[k] for k in zz.files if k != "meta_json"}
    report = {}
    for name, (attr, make, key) in MUTANTS.items():
        crop, cfg = key.split("/")[0], meta["configs"][key]
        r64 = T._run(z, crop, cfg, jnp.float64)
        orig = getattr(les, attr)
        setattr(les, attr, make(orig))
        jax.clear_caches()
        try:
            r32 = T._run(z, crop, cfg, jnp.float32)
        finally:
            setattr(les, attr, orig)
            jax.clear_caches()
        bad = T._violations(z, key, r32, r64)
        report[name] = {"killed": bool(bad), "violations": [v[0] for v in bad]}
        print(name, report[name], flush=True)
    Path(out).write_text(json.dumps(report, indent=1))
    return 0 if all(r["killed"] for r in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
