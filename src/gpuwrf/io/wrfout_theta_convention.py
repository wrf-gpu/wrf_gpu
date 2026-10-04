"""WRF history potential-temperature output convention (numpy only; no JAX, no gpuwrf imports).

WRF v4 (Registry/Registry.EM_COMMON:209-211, share/output_wrf.F:679):
  * ``T``   = ``th_phy_m_t0`` = DRY theta - T0 under both ``use_theta_m`` values;
  * ``THM`` = the prognostic state ``t``: MOIST theta_m - T0 when ``use_theta_m=1``, DRY theta - T0 when ``0``;
  * the integer global attribute ``USE_THETA_M``.
The port carries theta_m internally for both options (P0 option-0 ingest), so only the THM source and the attribute
depend on the option.  Kept dependency-free so the two-option contract is unit-testable without importing JAX.
"""

from __future__ import annotations

from typing import Any

USE_THETA_M_ALLOWED = (0, 1)


def parse_use_theta_m(value: Any) -> int:
    """WRF ``&dynamics use_theta_m`` (default 1).  Anything other than an integral 0/1 refuses (no silent default)."""

    if value is None:
        return 1
    if isinstance(value, bool):
        raise ValueError(f"WRFOUT_USE_THETA_M_INVALID:{value!r}")
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"WRFOUT_USE_THETA_M_INVALID:{value!r}") from None
    if as_int != value or as_int not in USE_THETA_M_ALLOWED:
        raise ValueError(f"WRFOUT_USE_THETA_M_INVALID:{value!r}")
    return as_int


def thm_source(theta_m: Any, theta_dry: Any, use_theta_m: int) -> Any:
    """Full-theta array WRF writes as ``THM`` (before subtracting T0): theta_m for option 1, dry theta for 0."""

    if use_theta_m not in USE_THETA_M_ALLOWED:
        raise ValueError(f"WRFOUT_USE_THETA_M_INVALID:{use_theta_m!r}")
    return theta_m if use_theta_m == 1 else theta_dry
