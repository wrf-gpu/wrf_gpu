"""Probe the MYNN mechanism: does the face<->mass<->face round-trip corrupt w
(especially the rigid-lid top face k=nz which the dycore keeps at 0)?

MYNN reads w via _w_mass (face->mass average), passes it through unchanged (MYNN
solves u/v/theta/qv, NOT w), then writes it back via _mass_to_w_face (mass->face
average).  face->mass->face is NOT identity, so even though MYNN doesn't change the
physical w, the stored C-grid w field is altered every MYNN step.  This probe
quantifies the alteration, focusing on the TOP face (k=nz) where top_lid=True is
supposed to enforce w=0.
"""
from __future__ import annotations
import numpy as np, jax
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision
from gpuwrf.coupling.physics_couplers import (
    _w_mass, _mass_to_w_face, _to_columns, _from_columns, mynn_adapter)

cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
case, _ = _build_real_case(cfg)
state = _enforce_operational_precision(case.state, force_fp64=True)

w0 = np.asarray(jax.device_get(state.w), dtype=np.float64)  # (nz+1, ny, nx)
nz = w0.shape[0] - 1
print(f"w shape {w0.shape}, top face index k={nz}")
print(f"BEFORE MYNN: top-face w[k={nz}] absmax = {np.abs(w0[nz]).max():.6e} "
      f"(rigid lid expects ~0); interior w absmax = {np.abs(w0[:nz]).max():.4f}")

# Pure reconstruction round-trip (no MYNN physics) -- isolates the staggering loss.
wm = _w_mass(state)                       # (nz, ny, nx)
w_rt = np.asarray(jax.device_get(_mass_to_w_face(wm)), dtype=np.float64)  # (nz+1,ny,nx)
d_rt = w_rt - w0
print(f"\nface->mass->face round-trip (pure, no physics):")
print(f"  top-face w[k={nz}] after = {np.abs(w_rt[nz]).max():.6e}  (was {np.abs(w0[nz]).max():.2e})")
print(f"  max |delta w| over column = {np.abs(d_rt).max():.4f} at k={int(np.argmax(np.abs(d_rt).reshape(nz+1,-1).max(axis=1)))}")
print(f"  top-face |delta| = {np.abs(d_rt[nz]).max():.6e}  near-top k{nz-1} |delta| = {np.abs(d_rt[nz-1]).max():.4f}")

# Full MYNN adapter (one call) -- what actually happens in the timestep.
out = mynn_adapter(state, 10.0, case.namelist.grid)
w1 = np.asarray(jax.device_get(out.w), dtype=np.float64)
dw = w1 - w0
print(f"\nfull MYNN adapter (one 10s call):")
print(f"  top-face w[k={nz}] after MYNN = {np.abs(w1[nz]).max():.6e}  (was {np.abs(w0[nz]).max():.2e})")
print(f"  max |delta w| = {np.abs(dw).max():.4f} at k={int(np.argmax(np.abs(dw).reshape(nz+1,-1).max(axis=1)))}")
print(f"  per-top-3-levels |delta w|: k{nz-2}={np.abs(dw[nz-2]).max():.4f} "
      f"k{nz-1}={np.abs(dw[nz-1]).max():.4f} k{nz}={np.abs(dw[nz]).max():.6e}")
print("DONE", flush=True)
