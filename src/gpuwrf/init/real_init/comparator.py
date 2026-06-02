"""S4 (GPT) — real.exe wrfinput/wrfbdy parity comparator harness.

FROZEN ENTRY SIGNATURES. S4 OWNS this comparator (the harness + the field
extraction + the gate logic) — it does NOT own any production lane module. The
harness:

  1. Reads a real.exe oracle pair (wrfinput_d0N + wrfbdy_d01) from the corpus
     (``/mnt/data/canairy_meteo/runs/{wrf_l2,wrf_l3}/<case>/``).
  2. Builds the native :class:`RealInitProduct` via
     :func:`gpuwrf.init.real_init.driver.build_real_init` from the matching
     met_em (wps_cases) for the SAME case/time/domain.
  3. Compares field-by-field against the FROZEN tolerances
     ``types.WRFINPUT_TOLS`` / ``types.WRFBDY_TOLS`` with the recorded masking
     policy (land/water/below-ground per V0.4.0-S0-PLAN.md section 5). NO
     post-hoc loosening.
  4. Emits a per-case, per-field JSON proof object (rmse, maxabs, pass/fail) and
     a campaign roll-up across the ≥10 cases (d01/d02/d03).

It ALSO drives the 24h native-init FORECAST gate: feed the native wrfinput/wrfbdy
into the existing GPU forecast pipeline (NO replay), run 24h, and compare the
full-field forecast to the CPU-WRF wrfout for the same case (reusing the
existing per-case continuous gate ``proofs/m20/continuous_gate.py`` metric set:
T2/U10/V10/PBLH/precip per lead). Conservation + restart gates must still pass.

The forecast gate is the only GPU-bound S4 step; it serializes onto the single
GPU (one job at a time) and runs in S5/integration, not in the parallel S4 lane.

FILE OWNERSHIP: this file + the tests under ``tests/init/real_init/`` are S4's
exclusive files.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpuwrf.init.real_init.types import RealInitProduct


@dataclass(frozen=True)
class FieldParityResult:
    """Per-field comparison record for the proof object."""

    name: str
    rmse: float
    maxabs: float
    rmse_tol: float
    maxabs_tol: float
    n_points: int
    passed: bool


@dataclass(frozen=True)
class CaseParityResult:
    """Per-(case, domain) roll-up of field parity vs the real.exe oracle."""

    case: str
    domain: str
    kind: str  # "wrfinput" | "wrfbdy"
    fields: tuple[FieldParityResult, ...]
    passed: bool


def compare_wrfinput(
    product: RealInitProduct,
    oracle_wrfinput_path: str,
) -> CaseParityResult:
    """Compares a native init product to a real.exe wrfinput oracle file."""

    raise NotImplementedError("v0.4.0 S4 (GPT): compare_wrfinput — frozen stub")


def compare_wrfbdy(
    product: RealInitProduct,
    oracle_wrfbdy_path: str,
) -> CaseParityResult:
    """Compares the native LBC to a real.exe wrfbdy oracle file."""

    raise NotImplementedError("v0.4.0 S4 (GPT): compare_wrfbdy — frozen stub")
