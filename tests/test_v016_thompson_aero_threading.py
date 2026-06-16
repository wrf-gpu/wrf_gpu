"""v0.16 aerosol-aware Thompson (mp_physics=28) state-threading tests (CPU).

Locks the append-only State extension (nwfa/nifa), the registry/dispatch
acceptance of mp=28, the adapter smoke behaviour on a tiny synthetic State, the
jnp inline climatological cold-start/emission against the FROZEN NumPy
reference (``thompson_aero_column.climatological_aerosol_profiles``), and the
wrfout/wrfrst variable threading (QNWFA/QNIFA).

No GPU, no forecast, no savepoint store required (the kernel-parity gate lives
in tests/test_v016_thompson_aero_oracle.py).
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts import physics_registry as registry
from gpuwrf.contracts.precision import FP32_GATED, PRECISION_MATRIX, STATE_FIELD_ORDER
from gpuwrf.contracts.state import State, _state_field_shapes


P0_PA, R_D, C_P, GRAVITY = 1.0e5, 287.0, 1004.0, 9.80665


class _GridShim:
    """Minimal grid stand-in: _state_field_shapes only reads nz/ny/nx."""

    def __init__(self, nz: int, ny: int, nx: int) -> None:
        self.nz, self.ny, self.nx = nz, ny, nx


def _tiny_state(nz: int = 24, ny: int = 4, nx: int = 4, seed: int = 3) -> State:
    """Deep, moist column State (the v013 operational-smoke b2 pattern)."""

    rng = np.random.default_rng(seed)
    grid = _GridShim(nz, ny, nx)
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    z_iface = np.arange(nz + 1) * 300.0
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    theta_col = 300.0 + 0.004 * z_mid
    p_col = P0_PA * (1.0 - GRAVITY * z_mid / (C_P * 290.0)) ** (C_P / R_D)

    def m3(base, noise):
        return jnp.asarray(base[:, None, None] + noise * rng.standard_normal((nz, ny, nx)), dtype=jnp.float64)

    fields["theta"] = m3(theta_col, 0.3)
    fields["p"] = m3(p_col, 50.0)
    fields["p_total"] = fields["p"]
    fields["qv"] = jnp.clip(m3(0.012 * np.exp(-z_mid / 3000.0), 5.0e-4), 0.0, None)
    fields["qc"] = jnp.clip(m3(np.where(z_mid < 4000.0, 4.0e-4, 0.0), 2.0e-5), 0.0, None)
    fields["qr"] = jnp.clip(m3(np.where((z_mid > 500.0) & (z_mid < 3000.0), 1.0e-4, 0.0), 1.0e-5), 0.0, None)
    fields["qi"] = jnp.clip(m3(np.where(z_mid > 6000.0, 5.0e-5, 0.0), 1.0e-6), 0.0, None)
    fields["qs"] = jnp.clip(m3(np.where(z_mid > 5000.0, 3.0e-5, 0.0), 1.0e-6), 0.0, None)
    fields["Ni"] = jnp.clip(m3(np.where(z_mid > 6000.0, 5.0e3, 0.0), 1.0e2), 0.0, None)
    fields["Nr"] = jnp.clip(m3(np.where((z_mid > 500.0) & (z_mid < 3000.0), 1.0e4, 0.0), 1.0e2), 0.0, None)
    fields["Nc"] = jnp.clip(m3(np.where(z_mid < 4000.0, 1.0e8, 0.0), 1.0e6), 0.0, None)
    ph = jnp.asarray(np.broadcast_to(GRAVITY * z_iface[:, None, None], (nz + 1, ny, nx)), dtype=jnp.float64)
    fields["ph"] = ph
    fields["ph_total"] = ph
    fields["mu_total"] = jnp.full((ny, nx), 1.0e5, dtype=jnp.float64)
    fields["mu"] = jnp.full((ny, nx), 9.0e4, dtype=jnp.float64)
    return State(**fields)


# ============================================================================
# 1. Registry: mp=28 is a consistent first-class accepted scheme.
# ============================================================================
def test_registry_consistent_with_mp28() -> None:
    registry.assert_registry_consistent()
    assert 28 in registry.ACCEPTED_MP_PHYSICS
    assert registry.MP_SCHEMES[28].wrf_package == "thompson_aero"
    assert registry.MP_MOIST_MEMBERS[28] == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert registry.MP_NUMBER_MEMBERS[28] == ("Ni", "Nr", "Nc", "nwfa", "nifa")
    assert registry.NUMBER_REGISTRY_MEMBER["nwfa"] == "qnwfa"
    assert registry.NUMBER_REGISTRY_MEMBER["nifa"] == "qnifa"
    assert registry.NUMBER_WRFOUT_NAME["nwfa"] == "QNWFA"
    assert registry.NUMBER_WRFOUT_NAME["nifa"] == "QNIFA"
    # The field-spec generator picked the new leaves up with the Nc/Nn flags.
    for leaf in ("nwfa", "nifa"):
        spec = registry.FIELD_SPECS_BY_LEAF[leaf]
        assert spec.additive_state and not spec.existing_state
        assert spec.restart_required and spec.wrfout_required
        assert spec.nest_forcedown and spec.nest_feedback
        assert spec.lateral_bc is False
    assert "QNWFA" in registry.wrfout_names_for_mp(28)
    assert "QNIFA" in registry.wrfout_names_for_mp(28)


def test_nest_field_list_carries_aerosol_numbers_for_mp28() -> None:
    entries = {e.leaf: e for e in registry.nest_field_list(mp_physics=28, bl_pbl_physics=5)}
    for leaf in ("Ni", "Nr", "Nc", "nwfa", "nifa"):
        assert leaf in entries, leaf
        assert entries[leaf].forcedown and entries[leaf].feedback
        assert entries[leaf].lateral_bc is False


# ============================================================================
# 2. State: append-only pytree extension.
# ============================================================================
def test_state_field_order_preserves_v016_aerosol_then_appends_v017_hail_tail() -> None:
    assert STATE_FIELD_ORDER[-7:] == ("nwfa", "nifa", "qh", "Nh", "qvolg", "qvolh", "hail_acc")
    assert State.__slots__[-7:] == ("nwfa", "nifa", "qh", "Nh", "qvolg", "qvolh", "hail_acc")
    # The pre-v0.16 prefix is untouched (append-only): nwfa/nifa still sit AFTER
    # the v0.6.0 (Nc/Nn/rainc_acc) and v0.15 MYNN SGS-cloud
    # (qsq/qc_bl/qi_bl/cldfra_bl) leaves; v0.17 only appends the hail tail after
    # the v0.16 aerosol leaves.
    assert State.__slots__[-14:] == (
        "Nc", "Nn", "rainc_acc", "qsq", "qc_bl", "qi_bl", "cldfra_bl",
        "nwfa", "nifa", "qh", "Nh", "qvolg", "qvolh", "hail_acc",
    )
    assert PRECISION_MATRIX["nwfa"] == (FP32_GATED, True)
    assert PRECISION_MATRIX["nifa"] == (FP32_GATED, True)


def test_state_constructs_with_and_without_aerosol_kwargs() -> None:
    state = _tiny_state()
    # Default: zeros templated on qc, fp32-gated dtype.
    assert state.nwfa.shape == state.qc.shape
    assert state.nifa.shape == state.qc.shape
    assert state.nwfa.dtype == jnp.float32
    assert state.nifa.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(state.nwfa))) == 0.0
    assert float(jnp.max(jnp.abs(state.nifa))) == 0.0

    # Explicit kwargs: value carried, dtype canonicalised to the matrix dtype.
    nwfa = jnp.full(state.qc.shape, 1.0e8, dtype=jnp.float64)
    nifa = jnp.full(state.qc.shape, 1.0e6, dtype=jnp.float64)
    replaced = state.replace(nwfa=nwfa, nifa=nifa)
    assert float(jnp.max(replaced.nwfa)) == pytest.approx(1.0e8, rel=1e-6)
    assert float(jnp.max(replaced.nifa)) == pytest.approx(1.0e6, rel=1e-6)
    assert replaced.nwfa.dtype == jnp.float32  # replace casts to the live dtype


def test_state_pytree_round_trip_preserves_leaf_count_and_order() -> None:
    state = _tiny_state()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    # Consolidated v0.16 release schema: 53 original + v0.6.0 (3) + v0.15 MYNN
    # (4) + v0.16 aerosol (nwfa/nifa, 2) + v0.17 hail tail (5) = 67 leaves.
    assert len(leaves) == len(State.__slots__) == 67
    # tree_flatten emits leaves in __slots__ order; v0.16 leaves are preserved
    # immediately before the v0.17 hail tail.
    assert leaves[-7] is state.nwfa
    assert leaves[-6] is state.nifa
    assert leaves[-5] is state.qh
    assert leaves[-4] is state.Nh
    assert leaves[-3] is state.qvolg
    assert leaves[-2] is state.qvolh
    assert leaves[-1] is state.hail_acc
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in State.__slots__:
        assert getattr(rebuilt, name) is getattr(state, name), name
    leaves2, treedef2 = jax.tree_util.tree_flatten(rebuilt)
    assert treedef2 == treedef
    assert len(leaves2) == len(leaves)


# ============================================================================
# 3. Dispatch: mp=28 routes to the aero adapter.
# ============================================================================
def test_dispatch_resolves_mp28_to_thompson_aero_adapter() -> None:
    from gpuwrf.coupling.physics_dispatch import resolve_physics_suite, scheme_entry

    entry = scheme_entry("microphysics", 28)
    assert entry.owner_module == "gpuwrf.coupling.physics_couplers"
    assert entry.entrypoint == "thompson_aero_adapter"
    assert entry.convention == "state_adapter"
    assert entry.gpu_runnable is True
    assert set(("Nc", "nwfa", "nifa")) <= set(entry.writes_state)

    suite = resolve_physics_suite({"mp_physics": 28})
    assert suite.microphysics.option == 28
    assert suite.gpu_gate_ready is True

    module = importlib.import_module(entry.owner_module)
    assert hasattr(module, entry.entrypoint)


def test_namelist_layer_accepts_mp28_and_constrains_aerosol_options() -> None:
    from gpuwrf.io.namelist_check import (
        UnsupportedSchemeError,
        validate_namelist,
        validate_supported_namelist,
    )

    validate_supported_namelist({"physics": {"mp_physics": [28]}})
    validate_namelist(
        {"physics": {"mp_physics": [28], "use_aero_icbc": ".false.",
                     "wif_input_opt": 1, "aer_init_opt": 1}}
    )
    # use_aero_icbc=.true. (aerosol ICs/BCs from WPS) is NOT wired: fail closed.
    with pytest.raises(UnsupportedSchemeError, match="use_aero_icbc"):
        validate_namelist({"physics": {"mp_physics": [28], "use_aero_icbc": ".true."}})
    with pytest.raises(UnsupportedSchemeError, match="wif_input_opt"):
        validate_namelist({"physics": {"wif_input_opt": 0}})
    with pytest.raises(UnsupportedSchemeError, match="aer_init_opt"):
        validate_namelist({"physics": {"aer_init_opt": 2}})


# ============================================================================
# 4. Cold start + emission: jnp inline == frozen NumPy climatology.
# ============================================================================
def test_coldstart_matches_frozen_numpy_climatology() -> None:
    from gpuwrf.coupling.physics_couplers import (
        WRF_PHYSICS_G_M_S2,
        _aerosol_surface_emission_columns,
        thompson_aero_coldstart_init,
    )
    from gpuwrf.physics.thompson_aero_column import climatological_aerosol_profiles

    state = _tiny_state()
    assert float(jnp.max(state.nwfa)) == 0.0
    seeded = thompson_aero_coldstart_init(state, None)

    # Reference: the FROZEN NumPy helper on the same mass-level heights.
    z_face = np.asarray(state.ph, dtype=np.float64) / WRF_PHYSICS_G_M_S2
    z_mass = 0.5 * (z_face[:-1] + z_face[1:])  # (nz, ny, nx)
    hgt = np.moveaxis(z_mass, 0, -1)  # vertical LAST
    nwfa_ref, nifa_ref, nwfa2d_ref = climatological_aerosol_profiles(hgt)

    nwfa_got = np.moveaxis(np.asarray(seeded.nwfa, dtype=np.float64), 0, -1)
    nifa_got = np.moveaxis(np.asarray(seeded.nifa, dtype=np.float64), 0, -1)
    # fp32 storage tolerance on O(1e8) numbers.
    np.testing.assert_allclose(nwfa_got, nwfa_ref, rtol=2.0e-6)
    np.testing.assert_allclose(nifa_got, nifa_ref, rtol=2.0e-6)

    nwfa2d, nifa2d = _aerosol_surface_emission_columns(state, None)
    np.testing.assert_allclose(np.asarray(nwfa2d, dtype=np.float64), nwfa2d_ref, rtol=1.0e-12)
    assert float(jnp.max(jnp.abs(nifa2d))) == 0.0

    # A state that already carries aerosol is left untouched (restart path).
    again = thompson_aero_coldstart_init(seeded, None)
    assert again is seeded


# ============================================================================
# 5. Adapter smoke: one CPU step, finite fields, correct shapes/dtypes.
# ============================================================================
def test_thompson_aero_adapter_smoke_one_step() -> None:
    from gpuwrf.coupling.physics_couplers import (
        thompson_aero_adapter,
        thompson_aero_coldstart_init,
    )

    state = thompson_aero_coldstart_init(_tiny_state(), None)
    out = thompson_aero_adapter(state, 20.0)

    assert isinstance(out, State)
    for field in ("theta", "qv", "qc", "qr", "qi", "qs", "qg",
                  "Ni", "Nr", "Ns", "Ng", "Nc", "nwfa", "nifa",
                  "rain_acc", "snow_acc", "graupel_acc", "ice_acc"):
        before = getattr(state, field)
        after = getattr(out, field)
        arr = np.asarray(after)
        assert np.all(np.isfinite(arr)), f"{field} produced non-finite values"
        assert after.shape == before.shape, field
        assert after.dtype == before.dtype, field
    # The aerosol-aware prognostics actually move (microphysics + surface
    # emission), and precipitation accumulates somewhere in the moist column.
    assert not np.array_equal(np.asarray(out.nwfa), np.asarray(state.nwfa))
    assert not np.array_equal(np.asarray(out.Nc), np.asarray(state.Nc))
    assert not np.array_equal(np.asarray(out.qv), np.asarray(state.qv))
    # Surface emission adds nwfa on the LOWEST level only for skipped columns,
    # so level 0 must have grown somewhere.
    assert float(jnp.max(out.nwfa[0] - state.nwfa[0])) > 0.0

    # Tendency side channel mirrors the mp=8 contract.
    from gpuwrf.coupling.physics_couplers import ThompsonTendencySideChannel

    out2, side = thompson_aero_adapter(state, 20.0, return_tendencies=True)
    assert isinstance(side, ThompsonTendencySideChannel)
    for leaf in jax.tree_util.tree_leaves(side):
        assert np.all(np.isfinite(np.asarray(leaf)))


# ============================================================================
# 6. wrfout / wrfrst threading.
# ============================================================================
def test_wrfout_writer_knows_qnwfa_qnifa() -> None:
    from gpuwrf.io.wrfout_writer import (
        MICROPHYSICS_EXTRA_VARIABLES,
        OPERATIONAL_WRFOUT_VARIABLES,
        WRFOUT_VARIABLE_SPECS,
    )

    for name in ("QNWFA", "QNIFA"):
        assert name in MICROPHYSICS_EXTRA_VARIABLES
        assert name in OPERATIONAL_WRFOUT_VARIABLES
        spec = WRFOUT_VARIABLE_SPECS[name]
        assert spec.memory_order == "XYZ"
        assert "aerosol number con" in spec.description
    # Registry forward names for mp=28 are all writable wrfout variables.
    for name in registry.wrfout_names_for_mp(28):
        assert name in WRFOUT_VARIABLE_SPECS, name


def test_wrfrst_restart_threading_covers_nwfa_nifa() -> None:
    from gpuwrf.io.wrfrst_netcdf import (
        STANDARD_RESTART_FIELDS,
        STATE_EXACT_DIMENSIONS,
        WRF_STANDARD_RESTART_VARIABLES,
        _units_for_leaf,
    )

    assert "QNWFA" in WRF_STANDARD_RESTART_VARIABLES
    assert "QNIFA" in WRF_STANDARD_RESTART_VARIABLES
    by_name = {f.spec.name: f for f in STANDARD_RESTART_FIELDS}
    state = _tiny_state()
    assert by_name["QNWFA"].value(state) is state.nwfa
    assert by_name["QNIFA"].value(state) is state.nifa
    # The GPUWRF-exact section iterates State.__slots__: dims + units present.
    assert STATE_EXACT_DIMENSIONS["nwfa"] == ("Time", "bottom_top", "south_north", "west_east")
    assert STATE_EXACT_DIMENSIONS["nifa"] == ("Time", "bottom_top", "south_north", "west_east")
    assert _units_for_leaf("nwfa") == "kg-1"
    assert _units_for_leaf("nifa") == "kg-1"
