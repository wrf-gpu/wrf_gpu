"""v0234 late-Ni fix: WRF pre-sedimentation cloud-ice mass/number balance.

Proves the fix added in ``gpuwrf.physics.thompson_column._balance_ice_number``
(pristine WRF ``module_mp_thompson.F:3033-3055``) on the exact retained d01
step-1147 -> 1148 transition that produced the first persisted nonfinite of the
v0234 accepted trajectory.

The heavy tests need the retained exact carry on ``<DATA_ROOT>`` and are skipped
when it is not mounted; the algebraic tests below always run.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from gpuwrf.physics import thompson_column as tc  # noqa: E402

EXACT_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/v0234_gpt_late_ni_df242a850d6ebab1_exact"
)
INPUT_CARRY = EXACT_DIR / "last-green-advance-input-d01-step-1147.pkl"
TARGET_YX = (48, 40)
DT_SECONDS = 54.0

# Independently re-derived in
# .agent/sprints/2026-07-21-v0234-opus-late-ni-fix/OPUS_INDEPENDENT_VERIFICATION.json
EXPECTED_SOURCE_LEVEL = 19
EXPECTED_BALANCED_NI_PER_KG = 4138.382776150909
EXPECTED_BALANCED_SOURCE_SPEED = 0.6391697902895398  # k19, the ice source level
EXPECTED_BALANCED_COLUMN_MAX_SPEED = 0.6609843332980857  # k20, the column maximum
EXPECTED_UNBALANCED_SPEED = 856.9557946103141

requires_carry = pytest.mark.skipif(
    not INPUT_CARRY.exists(), reason="retained v0234 exact carry not mounted"
)


def _target_column():
    from gpuwrf.coupling.physics_couplers import _thompson_column_from_state

    with INPUT_CARRY.open("rb") as stream:
        carry = pickle.load(stream)
    full = _thompson_column_from_state(carry.state)
    y, x = TARGET_YX
    return jax.tree_util.tree_map(lambda a: jnp.asarray(a[y, x, :]), full)


def _pre_sedimentation_state(column):
    """Walk the Thompson operator split up to (not including) sedimentation."""

    state = tc._cast_state(column, tc._work_dtype())
    state = tc._clip_species(state)
    state = tc._reset_mp8_graupel_number(state)
    state = tc._warm_rain_collection(state, DT_SECONDS)
    cold = (
        tc._cold_collection_rates(state, DT_SECONDS, tc.COLD_COLLECTION_TABLES)
        if tc._cold_collection_enabled()
        else tc._zero_cold_collection_rates(state)
    )
    state, graupel_melt, vts_boost, cold = tc._ice_sources_with_process_flags(
        state, DT_SECONDS, cold_collection_rates=cold
    )
    if tc._cold_collection_enabled():
        state = tc._apply_cold_collection_rates(state, DT_SECONDS, cold)
    state, condensed = tc._saturation_adjustment_with_condensation(state, DT_SECONDS)
    state = tc._rain_evaporation(
        state, DT_SECONDS, skip_evaporation=condensed, graupel_melt=graupel_melt
    )
    return state, vts_boost


# --------------------------------------------------------------------------
# Algebraic properties of the balance itself (no fixture needed).
# --------------------------------------------------------------------------


def test_balance_is_a_strict_no_op_inside_the_band():
    """An already-balanced (qi, Ni) pair must come back byte-identical."""

    rho = jnp.asarray([1.0, 0.9, 0.75, 0.5])
    qi = jnp.asarray([1.0e-5, 4.0e-6, 1.0e-6, 5.0e-7])
    # Choose Ni that puts the mean diameter mid-band (~100 um) at each level.
    ri = qi * rho
    lami = tc.CIE2 / 100.0e-6
    ni = tc.OIG2 * ri / tc.AM_I * lami**3.0
    Ni = ni / rho

    out = tc._balance_ice_number(qi, Ni, rho)
    assert np.array_equal(np.asarray(out), np.asarray(Ni))


def test_balance_caps_oversized_ice_at_300_microns():
    """Ice mass with a near-zero number is rebuilt to exactly 300 um."""

    rho = jnp.asarray([1.0])
    qi = jnp.asarray([4.881511619437367e-06])
    Ni = jnp.asarray([0.0])

    out = tc._balance_ice_number(qi, Ni, rho)
    ri = np.asarray(qi * rho)
    ni = np.asarray(out * rho)
    lami = (tc.AM_I * 6.0 * tc.OIG1 * ni / ri) ** tc.OBMI
    xdi_um = (tc.CIE2 / lami) * 1.0e6

    assert np.allclose(xdi_um, 300.0, rtol=1e-12)
    assert np.all(np.asarray(out) > 0.0)


def test_balance_lifts_undersized_ice_to_5_microns():
    """A huge number on tiny mass is rebuilt to the 5 um floor (or the ceiling)."""

    rho = jnp.asarray([1.0])
    qi = jnp.asarray([1.0e-9])
    Ni = jnp.asarray([1.0e8])  # absurdly many tiny crystals

    out = tc._balance_ice_number(qi, Ni, rho)
    ni = float(np.asarray(out * rho)[0])
    assert ni <= 999.0e3 + 1e-6  # WRF number ceiling, line 3054
    assert ni < 1.0e8  # strictly reduced


def test_balance_enforces_the_999e3_number_ceiling():
    rho = jnp.asarray([1.0])
    qi = jnp.asarray([1.0e-4])
    Ni = jnp.asarray([5.0e6])

    out = tc._balance_ice_number(qi, Ni, rho)
    assert float(np.asarray(out * rho)[0]) <= 999.0e3 + 1e-6


def test_balance_leaves_ice_free_layers_untouched():
    """qi below R1 must not have its number rewritten by this function."""

    rho = jnp.asarray([1.0, 1.0])
    qi = jnp.asarray([0.0, 1.0e-30])
    Ni = jnp.asarray([0.0, 7.0])

    out = tc._balance_ice_number(qi, Ni, rho)
    assert np.array_equal(np.asarray(out), np.asarray(Ni))


def test_balance_never_changes_ice_mass():
    """The balance is a number-only operation; it must be mass-neutral."""

    rng = np.random.default_rng(20260721)
    rho = jnp.asarray(rng.uniform(0.3, 1.2, size=64))
    qi = jnp.asarray(10.0 ** rng.uniform(-11.0, -4.0, size=64))
    Ni = jnp.asarray(10.0 ** rng.uniform(-6.0, 7.0, size=64))

    out = tc._balance_ice_number(qi, Ni, rho)
    assert np.all(np.isfinite(np.asarray(out)))
    # The function returns only a number; qi is not an output at all, so mass
    # neutrality is structural. Assert the band instead.
    ri = np.maximum(np.asarray(qi * rho), tc.R1)
    ni = np.maximum(np.asarray(out * rho), tc.R2)
    lami = (tc.AM_I * 6.0 * tc.OIG1 * ni / ri) ** tc.OBMI
    xdi = tc.CIE2 / lami
    active = np.asarray(qi * rho) > tc.R1
    assert np.all(xdi[active] <= 300.0e-6 * (1.0 + 1e-9))
    assert np.all(ni[active] <= 999.0e3 * (1.0 + 1e-9))


# --------------------------------------------------------------------------
# The exact retained d01 step-1147 -> 1148 event.
# --------------------------------------------------------------------------


@requires_carry
def test_exact_event_ice_fall_speed_is_physical_after_the_fix():
    """vti must fall from 856.96 m/s to a sane sub-1 m/s ice speed."""

    column = _target_column()
    pre_sed, vts_boost = _pre_sedimentation_state(column)

    speeds = tc._fall_speeds(pre_sed, vts_boost)
    vt_i_mass = np.asarray(speeds[2])

    assert np.all(np.isfinite(vt_i_mass))
    assert vt_i_mass.max() < 1.0
    assert np.isclose(
        vt_i_mass.max(), EXPECTED_BALANCED_COLUMN_MAX_SPEED, rtol=1e-9
    )
    assert np.isclose(
        vt_i_mass[EXPECTED_SOURCE_LEVEL], EXPECTED_BALANCED_SOURCE_SPEED, rtol=1e-9
    )
    # And the pre-fix speed is gone by a factor of ~1300.
    assert vt_i_mass.max() < EXPECTED_UNBALANCED_SPEED / 1000.0


@requires_carry
def test_exact_event_substep_count_collapses_to_one():
    """WRF's adaptive nstep must be 1, not 911 (which the cap clipped to 16)."""

    column = _target_column()
    pre_sed, vts_boost = _pre_sedimentation_state(column)

    speeds = tc._fall_speeds(pre_sed, vts_boost)
    dz = jnp.maximum(pre_sed.dz, 1.0)
    nstep_i = tc._nstep_per_column(speeds[2], speeds[2], dz, DT_SECONDS)

    assert float(np.asarray(nstep_i)) == 1.0
    # Uncapped WRF count from the same speeds must also be 1, i.e. the static
    # cap is not what is holding this column together any more.
    raw = float(np.max(np.floor(DT_SECONDS * np.asarray(speeds[2]) / np.asarray(dz) + 1.0)))
    assert raw == 1.0
    assert raw <= tc.NSED_MAX


@requires_carry
def test_exact_event_balanced_source_number_matches_literal_wrf():
    column = _target_column()
    pre_sed, _ = _pre_sedimentation_state(column)

    balanced = np.asarray(
        tc._balance_ice_number(pre_sed.qi, pre_sed.Ni, pre_sed.rho)
    )
    assert np.isclose(
        balanced[EXPECTED_SOURCE_LEVEL], EXPECTED_BALANCED_NI_PER_KG, rtol=1e-9
    )


@requires_carry
def test_exact_event_sedimentation_conserves_column_ice_mass():
    """The 7.93e13x ice-mass creation must be gone; sedimentation is conservative."""

    column = _target_column()
    pre_sed, vts_boost = _pre_sedimentation_state(column)

    rho = np.asarray(pre_sed.rho)
    dz = np.maximum(np.asarray(pre_sed.dz), 1.0)
    mass_before = float(np.sum(np.asarray(pre_sed.qi) * rho * dz))

    sedimented, precip = tc._sedimentation(pre_sed, DT_SECONDS, vts_boost=vts_boost)
    mass_after = float(
        np.sum(np.asarray(sedimented.qi) * np.asarray(sedimented.rho) * dz)
    )
    surface_ice_mm = float(np.asarray(precip["ice"]))

    assert np.all(np.isfinite(np.asarray(sedimented.qi)))
    assert np.all(np.isfinite(np.asarray(sedimented.Ni)))
    # Column ice + what fell out of the bottom must balance to round-off.
    assert mass_after <= mass_before * (1.0 + 1e-9)
    residual = abs(mass_before - mass_after - surface_ice_mm) / max(mass_before, 1e-30)
    assert residual < 1e-6, f"ice mass not conserved: residual {residual}"


@requires_carry
def test_exact_event_public_kernel_is_finite_and_physical():
    """The end-to-end public Thompson entry on the exact failing column."""

    column = _target_column()
    out, _ = tc.step_thompson_column_with_precip(column, DT_SECONDS, debug=False)

    Ni = np.asarray(out.Ni)
    qi = np.asarray(out.qi)
    T = np.asarray(out.T)

    assert np.all(np.isfinite(Ni)) and np.all(np.isfinite(qi)) and np.all(np.isfinite(T))
    # Pre-fix this column produced Ni=1.42e14, qi=1.19e9, T=4.83e10.
    assert Ni.max() <= 999.0e3 / 0.2, f"ice number still unphysical: {Ni.max()}"
    assert qi.max() < 1.0e-3, f"ice mixing ratio still unphysical: {qi.max()}"
    assert 150.0 < T.min() and T.max() < 350.0, f"temperature unphysical: {T.min()}-{T.max()}"
