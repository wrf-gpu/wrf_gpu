"""Native real.exe-equivalent initialization (v0.4.0).

This package replaces WRF ``real.exe`` (``dyn_em/module_initialize_real.F`` +
``main/real_em.F``). It consumes the FROZEN v0.3.0 metgrid-equivalent artifact
(:class:`gpuwrf.init.metgrid_schema.MetEmArtifact`) and produces, with no
CPU-WRF dependency:

* a ``wrfinput``-equivalent initial :class:`gpuwrf.contracts.state.State` +
  :class:`gpuwrf.contracts.state.BaseState` + the static
  :class:`gpuwrf.contracts.grid.DycoreMetrics` (S1 dynamics + S2 surface), and
* a ``wrfbdy``-equivalent :class:`gpuwrf.contracts.state.BoundaryState` carrying
  per-side specified values + tendencies over the forcing intervals (S3 LBC).

LANE OWNERSHIP (v0.4.0 S0 freeze; see ``.agent/decisions/V0.4.0-S0-PLAN.md``):

* S1 (Opus) — :mod:`gpuwrf.init.real_init.vertical_coord`,
  :mod:`gpuwrf.init.real_init.base_state`,
  :mod:`gpuwrf.init.real_init.hydrostatic`,
  :mod:`gpuwrf.init.real_init.vinterp`.
* S2 (GPT)  — :mod:`gpuwrf.init.real_init.surface_init`,
  :mod:`gpuwrf.init.real_init.soil_init`.
* S3 (Opus) — :mod:`gpuwrf.init.real_init.lateral_bc`.
* S4 (GPT)  — :mod:`gpuwrf.init.real_init.comparator` (oracle harness; OWNS no
  production module, only the test/comparison harness).
* S5 (manager-merge) — :mod:`gpuwrf.init.real_init.driver` (the only file the
  manager edits to wire the lanes; the four lanes deliver against its frozen
  signatures, they do NOT edit it).

The FROZEN handoff types live in :mod:`gpuwrf.init.real_init.types`. No lane may
edit ``types.py`` or ``driver.py`` after this freeze without a manager sign-off
recorded in the S0 plan doc.
"""

from __future__ import annotations

__all__ = [
    "REAL_INIT_INTERFACE_VERSION",
]

# Bump (with a manager-signed note in V0.4.0-S0-PLAN.md) only on a breaking
# change to any frozen signature in types.py / the lane module entry points.
REAL_INIT_INTERFACE_VERSION = "0.4.0-S0-frozen-2026-06-02"
