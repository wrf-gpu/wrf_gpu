"""v0.25 S3: date-correct CLWRF gases inside compiled programs.

The step and M9 programs are cached under a date-blind jit/AOT key (#114), so
every date-dependent value they use must arrive as an operand.  These tests pin
the traced CLWRF gas clock against the host transcription, its dtype contract,
its production threading, and the same-process date switch that used to reuse
the first date's gases.  Hermetic: a real SSP245 table slice is written to a
temporary WRF root.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.wrf_clwrf_ghg import (
    clwrf_gas_clock,
    clwrf_gases_at_lead,
    clwrf_ssp245_gases_for_time,
)

# Four fp32 epsilons: the traced blend may round a record weight one fp32 ulp
# differently from NumPy (XLA FMA / divide rewrites).  One day of CO2 growth is
# ~1.9e-5 relative, so this bound is ~40 s of gas drift.
_RTOL = 4.0 * float(np.finfo(np.float32).eps)

# Rows 2018-2032 of WRF's run/CAMtr_volume_mixing_ratio.SSP245 (verbatim).
_SSP245_SLICE = """\
## year[1] | co2 (ppmv) [2] | n2o (ppbv)[3] | ch4 (ppbv)[4] | cfc11 (ppbv)[5] | cfc12 (pppbv)[6]
## Non values are given by  -9999.999 values SSP245
2018  408.632    330.541   1887.041    223.642    504.263
2019  411.506    331.302   1899.411    220.948    499.657
2020  414.390    332.068   1910.971    218.231    495.016
2021  417.287    332.839   1921.791    215.496    490.353
2022  420.198    333.616   1932.061    212.745    485.679
2023  423.125    334.398   1941.941    209.981    481.003
2024  426.069    335.186   1951.461    207.208    476.333
2025  429.030    335.980   1960.651    204.429    471.673
2026  432.011    336.779   1969.541    201.646    467.030
2027  435.012    337.583   1978.151    198.861    462.408
2028  438.033    338.393   1986.491    196.078    457.809
2029  441.077    339.208   1994.600    193.298    453.238
2030  444.143    340.029   2002.480    190.523    448.695
2031  447.232    340.855   2010.150    187.756    444.183
2032  450.333    341.683   2017.230    184.997    439.704
"""

# Far-apart init dates: leap day, Dec 31 (year crossing), mid-June record, 06Z.
_INITS = (
    datetime(2024, 2, 29, 12, tzinfo=timezone.utc),
    datetime(2024, 9, 1, tzinfo=timezone.utc),
    datetime(2025, 12, 25, 6, tzinfo=timezone.utc),
    datetime(2025, 12, 31, 23, tzinfo=timezone.utc),
    datetime(2026, 6, 14, 12, tzinfo=timezone.utc),
    datetime(2026, 7, 26, tzinfo=timezone.utc),
)


@pytest.fixture()
def _isolated_gas_dispatch_cache():
    """Absolute cache counts need space in JAX's shared bounded dispatch cache.

    TH13d reproduced both zero-count failures at its 8192-entry capacity while
    JIT and numerical checks still worked. Keep the cache intact across every
    date within each test, so re-specialization still fails the original gate.
    """
    jax.clear_caches()
    try:
        yield
    finally:
        jax.clear_caches()


@pytest.fixture()
def wrf_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "WRF"
    (root / "run").mkdir(parents=True)
    (root / "run" / "CAMtr_volume_mixing_ratio").write_text(_SSP245_SLICE, encoding="ascii")
    monkeypatch.setenv("GPUWRF_WRF_ROOT", str(root))
    return root


def _table(root: Path) -> Path:
    return root / "run" / "CAMtr_volume_mixing_ratio"


def _host(table: Path, init: datetime, lead: float) -> np.ndarray:
    return np.asarray(
        clwrf_ssp245_gases_for_time(init + timedelta(seconds=float(lead)), table_path=table)
    )


def _leads(init: datetime) -> np.ndarray:
    """Hourly-at-dt54 over three days, daily for a year, and both record/year edges."""

    fine = np.arange(0.0, 3 * 86400.0 + 1.0, 54.0 * 20)
    daily = np.arange(0.0, 366.0) * 86400.0
    new_year = datetime(init.year + 1, 1, 1, tzinfo=timezone.utc)
    record = datetime(init.year + (init.month > 6), 6, 14, 12, tzinfo=timezone.utc)
    edges = []
    for edge in (new_year, record):
        base = (edge - init).total_seconds()
        edges.extend(base + offset for offset in (-54.0, -1.0, 0.0, 1.0, 54.0))
    return np.concatenate([fine, daily, np.asarray([lead for lead in edges if lead >= 0.0])])


def test_gas_clock_matches_host_clwrf_along_valid_time_sequences(
    wrf_root: Path, _isolated_gas_dispatch_cache,
) -> None:
    """Jitted and eager traced gases equal the host transcription at init + lead,
    for far-apart dates, a daily sequence and the record/year edges, with ONE
    compiled evaluator for every date."""

    table = _table(wrf_root)
    # Give this cache-count gate its own callable. Earlier callers may have
    # compiled the same function in JAX's shared dispatch cache.
    evaluate = jax.jit(lambda clock, lead: clwrf_gases_at_lead(clock, lead))
    for init in _INITS:
        clock = clwrf_gas_clock(init, table_path=table)
        leads = _leads(init)
        got = np.asarray(
            [np.asarray(evaluate(clock, jnp.float64(lead))) for lead in leads]
        )
        want = np.asarray([_host(table, init, lead) for lead in leads])
        np.testing.assert_allclose(got, want, rtol=_RTOL, atol=0.0, err_msg=str(init))
        eager = np.asarray(
            [[float(gas) for gas in clwrf_gases_at_lead(clock, float(lead))] for lead in leads[:40]]
        )
        np.testing.assert_allclose(eager, want[:40], rtol=_RTOL, atol=0.0, err_msg=str(init))
    assert evaluate._cache_size() == 1, "a new date re-specialized the gas evaluator"


def test_gas_clock_advances_with_valid_time_and_poisons_outside_window(wrf_root: Path) -> None:
    table = _table(wrf_root)
    init = datetime(2026, 7, 26, tzinfo=timezone.utc)
    clock = clwrf_gas_clock(init, table_path=table)
    day0 = np.asarray(clwrf_gases_at_lead(clock, 0.0))
    day1 = np.asarray(clwrf_gases_at_lead(clock, 86400.0))
    assert day1[0] > day0[0] and day1[3] < day0[3]  # CO2 rises, CFC-11 falls
    # The window is [Jan 1 of the init year, Jan 1 two years later).
    first = (datetime(2026, 1, 1, tzinfo=timezone.utc) - init).total_seconds()
    last = (datetime(2028, 1, 1, tzinfo=timezone.utc) - init).total_seconds()
    for lead in (first, last - 1.0):
        assert np.isfinite(np.asarray(clwrf_gases_at_lead(clock, lead))).all()
    for lead in (first - 1.0, last):
        assert np.isnan(np.asarray(clwrf_gases_at_lead(clock, lead))).all()


def test_gas_clock_is_weak_float64_like_the_host_floats_it_replaces(wrf_root: Path) -> None:
    clock = clwrf_gas_clock(_INITS[1], table_path=_table(wrf_root))
    columns = jnp.ones((3,), dtype=jnp.float32)

    @jax.jit
    def scaled(clock, lead, columns):
        gas = clwrf_gases_at_lead(clock, lead).co2_vmr
        return columns * gas, gas

    product, gas = scaled(clock, jnp.float64(3600.0), columns)
    assert gas.dtype == jnp.float64
    assert product.dtype == jnp.float32, "gas promoted fp32 column amounts to fp64"
    assert jax.typeof(clwrf_gases_at_lead(clock, 0.0).co2_vmr).weak_type


def test_gas_clock_refuses_extrapolation_and_missing_records(tmp_path: Path) -> None:
    table = tmp_path / "CAMtr"
    table.write_text(_SSP245_SLICE, encoding="ascii")
    with pytest.raises(ValueError, match="extrapolation"):
        clwrf_gas_clock(datetime(2018, 3, 1, tzinfo=timezone.utc), table_path=table)
    with pytest.raises(ValueError, match="extrapolation"):
        clwrf_gas_clock(datetime(2031, 3, 1, tzinfo=timezone.utc), table_path=table)
    missing = tmp_path / "CAMtr_missing"  # new path: tables are cached per path
    missing.write_text(
        _SSP245_SLICE.replace("2026  432.011    336.779", "2026  432.011   -9999.999"),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="missing"):
        clwrf_gas_clock(datetime(2026, 3, 1, tzinfo=timezone.utc), table_path=missing)


# Compiled WRF ``read_CAMgases`` on ``_SSP245_SLICE`` (v0.25 S3 calendar
# amendment): WRF's preprocessed module_ra_clWRF_support.f90 (sha256 b8e02d7e...)
# built with WRF's gfortran flags, julian = REAL(dayOfYear_r8) - 1.0.  Oracle and
# build script: .agent/sprints/2026-09-23-v0250-s3-date-correct-ghg-cache/
# wrf_clwrf_oracle.  Covers the leap day, both sides of the leap and ordinary
# mid-June bracket switch, the year boundaries (incl. Dec 31 23:59:59, where WRF's
# REAL julian rounds to 365.0) and the frozen production case valid times.
_WRF_ORACLE_SLICE = {
    "2024-02-29T12:00:00": (
        0.0004252123431570939, 3.349566954363405e-07, 1.9486907808505084e-06,
        2.0801489673960234e-10, 4.776918915840553e-10,
    ),
    "2024-06-15T00:00:00": (
        0.0004260689989242792, 3.351859905202925e-07, 1.9514609448089136e-06,
        2.0720799917203769e-10, 4.763329980966673e-10,
    ),
    "2024-06-15T11:59:59": (
        0.0004260730285694327, 3.3518707314922793e-07, 1.9514739859399222e-06,
        2.07204214707974e-10, 4.763266270214834e-10,
    ),
    "2024-06-15T12:00:00": (
        0.0004260730674890001, 3.3518708794671967e-07, 1.951473590644976e-06,
        2.0720419835318327e-10, 4.763266283979944e-10,
    ),
    "2024-12-31T23:00:00": (
        0.0004276911139865631, 3.356209583791203e-07, 1.9564954539360935e-06,
        2.0568557049126379e-10, 4.737800911784183e-10,
    ),
    "2025-01-01T00:00:00": (
        0.0004276914646615192, 3.3562105899789715e-07, 1.956496561067335e-06,
        2.0568525946344016e-10, 4.737795734683605e-10,
    ),
    "2025-06-15T11:00:00": (
        0.0004290337080621382, 3.3598098056249836e-07, 1.960662443590821e-06,
        2.0442550551373832e-10, 4.716671370576836e-10,
    ),
    "2025-06-15T12:00:00": (
        0.00042903409496496163, 3.359810947966874e-07, 1.960663179693626e-06,
        2.0442519280395592e-10, 4.716666515686627e-10,
    ),
    "2025-12-31T23:59:59": (
        0.00043066342355804364, 3.3641779870137075e-07, 1.965522177251654e-06,
        2.0290406769374766e-10, 4.691288885553675e-10,
    ),
    "2026-07-26T00:00:00": (
        0.0004323480975405232, 3.368693028019775e-07, 1.9705080949607793e-06,
        2.0133316357725485e-10, 4.665108145712518e-10,
    ),
    "2026-07-26T13:00:00": (
        0.0004323525221429312, 3.3687047337846464e-07, 1.970520740382053e-06,
        2.0132901703510225e-10, 4.665039240724623e-10,
    ),
}


def test_host_and_traced_clwrf_equal_compiled_wrf_at_calendar_boundaries(
    wrf_root: Path,
) -> None:
    table = _table(wrf_root)
    evaluate = jax.jit(clwrf_gases_at_lead)
    for text, expected in _WRF_ORACLE_SLICE.items():
        when = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
        host = np.asarray(clwrf_ssp245_gases_for_time(when, table_path=table))
        np.testing.assert_array_equal(host, np.asarray(expected), err_msg=text)
        clock = clwrf_gas_clock(when - timedelta(days=3), table_path=table)
        traced = np.asarray(evaluate(clock, jnp.float64(3 * 86400.0)))
        np.testing.assert_allclose(traced, expected, rtol=_RTOL, atol=0.0, err_msg=text)


# --------------------------------------------------------------------------- #
# Production threading: build_clock_base -> _rad_clock_base -> RRTMG columns.
# --------------------------------------------------------------------------- #
def _fixture(time_utc):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
    from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base

    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid)
    state = State(
        **{
            name: jnp.asarray(np.zeros(shape), dtype=DEFAULT_DTYPES.dtype_for(name))
            for name, shape in shapes.items()
        }
    )
    renamed = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tendencies = Tendencies(
        **{
            name: jnp.zeros(shapes[renamed.get(name, name)], dtype=DEFAULT_DTYPES.dtype_for(name))
            for name in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
        }
    )
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=tendencies,
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=6,
        time_utc=time_utc,
    )
    return state, namelist, build_clock_base(namelist)


@jax.jit
def _production_gas_probe(state, namelist, lead_seconds, clock_base):
    """The RRTMG column-input path of the step/M9 programs, gases only."""

    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.runtime import operational_mode as op

    sw, lw, *_ = pc._rrtmg_column_inputs(
        state,
        namelist.grid,
        time_utc=namelist.time_utc,
        lead_seconds=lead_seconds,
        clock_base=op._rad_clock_base(clock_base),
        radiation_static=namelist.radiation_static,
    )
    return jnp.stack(
        [lw.co2_vmr, lw.n2o_vmr, lw.ch4_vmr, lw.cfc11_vmr, lw.cfc12_vmr,
         sw.co2_vmr, sw.n2o_vmr, sw.ch4_vmr]
    )


def test_build_clock_base_threads_a_date_invariant_gas_clock(wrf_root: Path) -> None:
    from gpuwrf.runtime.operational_mode import build_clock_base

    _, namelist_a, clock_a = _fixture("2024-09-01_00:00:00")
    _, _, clock_b = _fixture("2025-12-25_06:00:00")
    assert clock_a.ghg_clock is not None
    leaves_a, tree_a = jax.tree_util.tree_flatten(clock_a)
    leaves_b, tree_b = jax.tree_util.tree_flatten(clock_b)
    assert tree_a == tree_b
    assert [(x.shape, x.dtype) for x in leaves_a] == [(x.shape, x.dtype) for x in leaves_b]
    assert any(not np.array_equal(x, y) for x, y in zip(leaves_a, leaves_b))
    # Undated runs keep the legacy constants: no anchor, no table read.
    assert build_clock_base(dataclasses.replace(namelist_a, time_utc=None)).ghg_clock is None


def test_same_process_date_switch_uses_each_dates_valid_time_gases(
    wrf_root: Path, _isolated_gas_dispatch_cache,
) -> None:
    """One compiled production gas path serves three dates with their own gases.

    Detector liveness: the pre-S3 mechanism (host gases from ``time_utc`` at trace
    time, still used by clock-less legacy callers) is shown to return the FIRST
    date's gases for later dates under the same date-blind jit key."""

    table = _table(wrf_root)
    lead = 7.0 * 86400.0 + 3618.0
    dates = ("2024-09-01_00:00:00", "2025-12-25_06:00:00", "2024-02-29_12:00:00")
    served = []
    for text in dates:
        state, namelist, clock = _fixture(text)
        got = np.asarray(_production_gas_probe(state, namelist, jnp.float64(lead), clock))
        want = _host(table, datetime.fromisoformat(text.replace("_", "T")).replace(
            tzinfo=timezone.utc), lead)
        np.testing.assert_allclose(got[:5], want, rtol=_RTOL, atol=0.0, err_msg=text)
        np.testing.assert_array_equal(got[5:], got[:3])
        served.append(got)
    assert _production_gas_probe._cache_size() == 1, "date switch re-traced the program"
    assert not np.allclose(served[0], served[1], rtol=1.0e-4)

    legacy = jax.jit(lambda state, namelist, lead: _production_gas_probe.__wrapped__(
        state, namelist, lead, None))
    stale = [
        np.asarray(legacy(*_fixture(text)[:2], jnp.float64(lead)))[:5] for text in dates[:2]
    ]
    np.testing.assert_array_equal(stale[1], stale[0])  # pre-S3: stale by construction
    assert not np.allclose(stale[1], served[1][:5], rtol=1.0e-4)


def test_traced_clock_without_gas_clock_fails_closed_for_dated_rrtmg(wrf_root: Path) -> None:
    from gpuwrf.coupling import physics_couplers as pc

    state, namelist, clock = _fixture("2024-09-01_00:00:00")
    radiation_clock = pc.RadiationClock(clock.rad_julian, clock.rad_minute, None)
    with pytest.raises(ValueError, match="no CLWRF gas clock"):
        pc._rrtmg_column_inputs(
            state,
            namelist.grid,
            time_utc=namelist.time_utc,
            lead_seconds=0.0,
            clock_base=radiation_clock,
        )
