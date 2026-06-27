"""v0.17 ADR-032 graupel/hail (qh) State substrate -- structural + inertness gate.

This is the substrate enabler for the hail-heavy microphysics family (WSM7=24,
WDM7=26, UDM=27, Goddard-4ice/NUWRF=7, NSSL=17-22, Thompson-graupel/hail=38).
The schemes themselves are NOT implemented here; these tests assert only that

  * the four leaves ``qh``/``Nh``/``qvolg``/``qvolh`` are appended in the additive
    State tail (append-only; on the v0.18 trunk they sit just before the v0.16
    nwfa/nifa aerosol leaves and the hail_acc accumulator),
  * the flatten/unflatten round-trip is the identity and the treedef is stable,
  * the leaves cold-start at zero and are FP32_GATED (ADR-007),
  * the registry/precision/IO/advection wiring is internally consistent, and
  * the hail MP family stays fail-closed (no scheme is implemented).

No GPU is required: States are built with explicit ``jnp`` arrays. One GPU-gated
check covers ``State.zeros``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import (
    DEFAULT_DTYPES,
    FP32_GATED,
    PRECISION_MATRIX,
    STATE_FIELD_ORDER,
)
from gpuwrf.contracts.state import (
    AEROSOL_CONDITIONAL_LEAVES,
    CONDITIONAL_STATE_LEAVES,
    HAIL_CONDITIONAL_LEAVES,
    SCALAR_BOUNDARY_OPTIONAL_LEAVES,
    State,
    _state_field_shapes,
)


HAIL_LEAVES = ("qh", "Nh", "qvolg", "qvolh")


def _full_state(grid: GridSpec) -> State:
    """Build a fully-populated State with distinct per-field patterns on CPU."""

    fields = {}
    for index, (field, shape) in enumerate(_state_field_shapes(grid, include_all_conditional=True).items(), start=1):
        values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape) + index
        fields[field] = jnp.asarray(values, dtype=DEFAULT_DTYPES.dtype_for(field))
    return State(**fields)


def test_hail_leaves_appended_at_end_append_only() -> None:
    # v0.18 trunk consolidated additive tail (set-UNION of every lane): 53
    # original + 3 v0.6.0 (Nc/Nn/rainc_acc) + 4 v0.15 MYNN + 4 v0.17 hail
    # substrate (qh/Nh/qvolg/qvolh) + 2 v0.16 aerosol-aware Thompson (nwfa/nifa)
    # + 1 v0.17 hail surface accumulator (hail_acc) = 67, minus the 3 legacy
    # p/ph/mu duplicate aliases removed in v0.20 S1 = 64, plus 7 v0.22 optional
    # standalone wrfbdy scalar leaves = 71.
    assert len(State.__slots__) == 71
    # The four 3-D hail substrate leaves sit just before the v0.16 aerosol leaves
    # and hail_acc, followed only by the v0.22 optional wrfbdy scalar leaves.
    assert State.__slots__[-14:-10] == HAIL_LEAVES
    assert STATE_FIELD_ORDER[-14:-10] == HAIL_LEAVES
    assert State.__slots__[-10:-7] == ("nwfa", "nifa", "hail_acc")
    assert STATE_FIELD_ORDER[-10:-7] == ("nwfa", "nifa", "hail_acc")
    assert State.__slots__[-7:] == SCALAR_BOUNDARY_OPTIONAL_LEAVES
    assert STATE_FIELD_ORDER[-7:] == SCALAR_BOUNDARY_OPTIONAL_LEAVES
    # Every leaf BEFORE the hail block keeps its exact position (append-only):
    # the prefix up to the v0.15 cldfra_bl is unchanged.
    assert State.__slots__[-15] == "cldfra_bl"
    # STATE_FIELD_ORDER (precision/storage order) and __slots__ (pytree order)
    # are deliberately distinct orderings in the middle, but they cover the SAME
    # leaf set and BOTH end with the hail/aerosol tail + optional wrfbdy scalars.
    assert set(STATE_FIELD_ORDER) == set(State.__slots__)
    assert len(STATE_FIELD_ORDER) == len(State.__slots__) == 71


def test_hail_leaves_precision_fp32_gated() -> None:
    for leaf in HAIL_LEAVES:
        dtype, gate_required = PRECISION_MATRIX[leaf]
        assert dtype == FP32_GATED, leaf
        assert gate_required is True, leaf


def test_hail_leaves_absent_by_default_and_materialized_for_hail_mp() -> None:
    grid = GridSpec.canary_3km_template()
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    shapes = _state_field_shapes(grid, mp_physics=8)
    fields = {
        k: jnp.zeros(v, dtype=DEFAULT_DTYPES.dtype_for(k))
        for k, v in shapes.items()
    }
    state = State(**fields)
    assert state.active_field_names() == tuple(name for name in State.__slots__ if name not in CONDITIONAL_STATE_LEAVES)
    assert len(jax.tree_util.tree_leaves(state)) == len(State.__slots__) - len(CONDITIONAL_STATE_LEAVES) == 57
    for leaf in CONDITIONAL_STATE_LEAVES:
        assert getattr(state, leaf) is None, leaf

    hail_state = state.ensure_conditional_leaves(mp_physics=24)
    assert hail_state.active_field_names()[-5:] == HAIL_CONDITIONAL_LEAVES
    for leaf in AEROSOL_CONDITIONAL_LEAVES:
        assert getattr(hail_state, leaf) is None, leaf
    for leaf in HAIL_LEAVES:
        arr = getattr(hail_state, leaf)
        assert arr.shape == (nz, ny, nx), leaf
        assert float(np.asarray(arr).sum()) == 0.0, leaf
        assert arr.dtype == DEFAULT_DTYPES.dtype_for(leaf), leaf
    assert hail_state.hail_acc.shape == (ny, nx)
    assert float(np.asarray(hail_state.hail_acc).sum()) == 0.0
    assert len(jax.tree_util.tree_leaves(hail_state)) == 62  # v0.20 S1: 57 base + 5 hail


def test_flatten_unflatten_identity_and_treedef_stable() -> None:
    grid = GridSpec.canary_3km_template()
    state = _full_state(grid)
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 64  # 57 base + 7 materialized hail/aerosol leaves
    # The hail substrate leaves sit before the v0.16 aerosol leaves + hail_acc.
    assert State.__slots__[-14:-10] == HAIL_LEAVES
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    # Round-trip is the structural identity for every leaf, hail included.
    for leaf in State.__slots__:
        a = np.asarray(getattr(state, leaf))
        b = np.asarray(getattr(rebuilt, leaf))
        assert a.shape == b.shape, leaf
        assert np.array_equal(a, b), leaf
    # Treedef is stable across a second flatten (carry-in == carry-out).
    leaves2, treedef2 = jax.tree_util.tree_flatten(rebuilt)
    assert treedef2 == treedef


def test_unflatten_does_not_recanonicalise_hail_leaves() -> None:
    # tree_unflatten must assign children verbatim (no __init__ re-cast / None
    # default), or a scan carry treedef can diverge. Feeding a non-array sentinel
    # leaf must NOT raise (JAX uses this to format treedef-mismatch messages).
    grid = GridSpec.canary_3km_template()
    state = _full_state(grid)
    _, treedef = jax.tree_util.tree_flatten(state)
    sentinel = jax.tree_util.tree_unflatten(treedef, [object()] * 64)
    assert sentinel.qvolh.__class__ is object  # verbatim, not coerced


def test_state_with_hail_inert_byte_identical_to_zero_hail() -> None:
    # Enabling the hail substrate materializes zero hail leaves while leaving
    # every pre-hail base leaf byte-identical.
    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid, mp_physics=8)
    base_fields = {
        k: jnp.asarray(np.arange(int(np.prod(v)), dtype=np.float64).reshape(v) + i,
                       dtype=DEFAULT_DTYPES.dtype_for(k))
        for i, (k, v) in enumerate(shapes.items(), start=1)
    }
    default_state = State(**base_fields)
    hail_state = default_state.ensure_conditional_leaves(mp_physics=24)
    for leaf in default_state.active_field_names():
        a = np.asarray(getattr(default_state, leaf))
        b = np.asarray(getattr(hail_state, leaf))
        assert np.array_equal(a, b), leaf
    for leaf in HAIL_CONDITIONAL_LEAVES:
        assert float(np.asarray(getattr(hail_state, leaf)).sum()) == 0.0, leaf
    for leaf in AEROSOL_CONDITIONAL_LEAVES:
        assert getattr(hail_state, leaf) is None, leaf


def test_registry_and_io_names_consistent() -> None:
    from gpuwrf.contracts.physics_registry import (
        MOIST_WRFOUT_NAME,
        NUMBER_REGISTRY_MEMBER,
        NUMBER_WRFOUT_NAME,
        VOLUME_WRFOUT_NAME,
        assert_registry_consistent,
    )
    from gpuwrf.io.wrfout_writer import MICROPHYSICS_EXTRA_VARIABLES, WRFOUT_VARIABLE_SPECS
    from gpuwrf.io.wrfrst_netcdf import STATE_EXACT_DIMENSIONS, WRF_STANDARD_RESTART_VARIABLES

    assert_registry_consistent()
    # WRF Registry names (verbatim from Registry.EM_COMMON).
    assert MOIST_WRFOUT_NAME["qh"] == "QHAIL"
    assert NUMBER_REGISTRY_MEMBER["Nh"] == "qnh"
    assert NUMBER_WRFOUT_NAME["Nh"] == "QNHAIL"
    assert VOLUME_WRFOUT_NAME["qvolg"] == "QVGRAUPEL"
    assert VOLUME_WRFOUT_NAME["qvolh"] == "QVHAIL"
    for wrf in ("QHAIL", "QNHAIL", "QVGRAUPEL", "QVHAIL"):
        assert wrf in MICROPHYSICS_EXTRA_VARIABLES
        assert wrf in WRFOUT_VARIABLE_SPECS
        assert wrf in WRF_STANDARD_RESTART_VARIABLES
    # exact-state restart dimensions exist for all four leaves (else the restart
    # write loop over State.__slots__ would KeyError).
    for leaf in HAIL_LEAVES:
        assert leaf in STATE_EXACT_DIMENSIONS


def test_advection_selector_static_and_hail_gated() -> None:
    from gpuwrf.runtime.operational_mode import (
        _HAIL_MP_FAMILY,
        _MOISTURE_SPECIES,
        _advected_scalar_species,
    )

    class _NL:
        def __init__(self, mp: int) -> None:
            self.mp_physics = mp

    # Every NON-hail mp gets EXACTLY the core moist set -> byte-identical.
    for mp in (0, 1, 2, 3, 4, 6, 8, 10, 14, 16):
        assert _advected_scalar_species(_NL(mp)) == _MOISTURE_SPECIES, mp
    # Hail family gets the core set PLUS the scheme's hail extras.
    assert _advected_scalar_species(_NL(24)) == _MOISTURE_SPECIES + ("qh",)
    assert _advected_scalar_species(_NL(26)) == _MOISTURE_SPECIES + ("qh",)
    assert _advected_scalar_species(_NL(28)) == _MOISTURE_SPECIES + ("nwfa", "nifa")
    assert _advected_scalar_species(_NL(38)) == _MOISTURE_SPECIES + ("qvolg",)
    assert _advected_scalar_species(_NL(18)) == _MOISTURE_SPECIES + ("qh", "qvolg", "qvolh")
    assert 24 in _HAIL_MP_FAMILY and 8 not in _HAIL_MP_FAMILY


def test_hail_mp_family_unwired_members_stay_fail_closed() -> None:
    # v0.17 wires the WSM7/WDM7 hail schemes through their own scan adapters
    # (coupling.scan_adapters.MP_SCAN_ADAPTERS), so those are accepted +
    # scan-wired. Every OTHER hail-family id (Goddard-4ice/NSSL/UDM/Thompson-
    # graupel-hail) must still fail-closed -- the substrate is not scheme support.
    # The wired set is derived from the registry itself so this stays correct as
    # WSM7 (24) lands first and WDM7 (26) follows.
    import pytest

    from gpuwrf.contracts.physics_registry import ACCEPTED_MP_PHYSICS
    from gpuwrf.coupling.physics_couplers import HAIL_MP_FAMILY, hail_mp_adapter
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS
    from gpuwrf.runtime.operational_mode import _SCAN_WIRED_OPTIONS

    wired = HAIL_MP_FAMILY & set(MP_SCAN_ADAPTERS)
    assert 24 in wired, "WSM7 (mp=24) must be scan-wired in v0.17"
    for mp in sorted(HAIL_MP_FAMILY - wired):
        assert mp not in ACCEPTED_MP_PHYSICS, mp
        assert mp not in _SCAN_WIRED_OPTIONS["mp_physics"], mp
    # Every wired hail scheme IS accepted + scan-wired.
    for mp in sorted(wired):
        assert mp in ACCEPTED_MP_PHYSICS, mp
        assert mp in _SCAN_WIRED_OPTIONS["mp_physics"], mp
    # The shared fail-closed adapter slot still fails closed for an UNwired id
    # (e.g. NSSL mp=18); WSM7/WDM7 never route through it.
    with pytest.raises(NotImplementedError):
        hail_mp_adapter(None, 1.0, mp_physics=18)


def test_state_zeros_hail_leaves_zero_on_gpu() -> None:
    # GPU-gated: State.zeros allocates on the GPU; the hail leaves must be zero
    # and on the device like every other leaf.
    import pytest

    try:
        default_state = State.zeros(GridSpec.canary_3km_template())
        state = State.zeros(GridSpec.canary_3km_template(), mp_physics=24)
    except RuntimeError as exc:  # no GPU backend in a CPU-only run
        pytest.skip(f"GPU-required test on a CPU-only run: {exc}")
    for leaf in CONDITIONAL_STATE_LEAVES:
        assert getattr(default_state, leaf) is None, leaf
    for leaf in HAIL_LEAVES:
        arr = getattr(state, leaf)
        assert float(jnp.sum(arr)) == 0.0, leaf
        assert arr.devices().pop().platform == "gpu", leaf
    assert float(jnp.sum(state.hail_acc)) == 0.0
    assert state.hail_acc.devices().pop().platform == "gpu"
