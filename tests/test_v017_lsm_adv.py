"""v0.17 advanced LSM wiring + oracle integrity: RUC (sf_surface_physics=3) and
SSiB (sf_surface_physics=8), both REFERENCE-ONLY.

These two large land-surface models are wired through the full registry /
interface / dispatch / catalog / operational seam exactly like the other Tier-3
reference-only schemes (Grell-3D, KSAS, Goddard-LW): each has a fp64 pristine-WRF
single-column oracle staged (built from the UNMODIFIED WRF Fortran -- NOT a
self-compare), but its faithful traceable JAX column kernel is a documented
carry-over, so each is namelist-accepted (selectable for a single-column
reference comparison) and FAIL-CLOSES in the operational GPU scan.

This module proves:
  1. the scheme is registry/interface/dispatch/catalog consistent and classified
     REFERENCE_ONLY (never over-claimed as IMPLEMENTED);
  2. the operational scan fail-closes it with a scheme-specific named reason;
  3. the staged Fortran oracle savepoints exist, are finite, regime-varying, and
     checksum the UNMODIFIED pristine WRF source (the load-bearing evidence);
  4. the JAX column-kernel stub raises rather than silently returning a wrong
     land state, and freezes the land-carry shape the future port must thread.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

from gpuwrf.contracts.physics_interfaces import (
    assert_interfaces_consistent,
    scheme_step_spec,
)
from gpuwrf.contracts.physics_registry import (
    ACCEPTED_SF_SURFACE_PHYSICS,
    LAND_CARRY_MEMBERS,
    SURFACE_SCHEMES,
    assert_registry_consistent,
)
from gpuwrf.coupling.physics_dispatch import (
    UnsupportedSchemeSelection,
    resolve_physics_suite,
    scheme_entry,
)
from gpuwrf.io.scheme_catalog import (
    SupportStatus,
    assert_catalog_consistent,
    classify_scheme,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# (option, name-substring, JAX module, column entrypoint, oracle savepoint glob,
#  the unmodified WRF source the oracle checksums)
_REF_LSMS = [
    (
        3,
        "RUC",
        "gpuwrf.physics.lsm_ruc",
        "ruc_column",
        "proofs/v017/savepoints/ruclsm/fp64",
        "module_sf_ruclsm.F",
    ),
    (
        8,
        "SSiB",
        "gpuwrf.physics.lsm_ssib",
        "ssib_column",
        "proofs/v017/savepoints/ssib/fp64",
        "module_sf_ssib.F",
    ),
]


def test_registry_interface_catalog_consistent_with_ruc_ssib() -> None:
    """All three freeze authorities stay consistent after adding RUC(3)/SSiB(8)."""

    assert_registry_consistent()
    assert_interfaces_consistent()
    assert_catalog_consistent()
    assert set(ACCEPTED_SF_SURFACE_PHYSICS) == {0, 1, 2, 3, 4, 7, 8}


@pytest.mark.parametrize("opt, name, _mod, _ep, _sp, _src", _REF_LSMS)
def test_reference_only_classification(opt, name, _mod, _ep, _sp, _src) -> None:
    """RUC/SSiB are REFERENCE_ONLY -- accepted at the namelist layer, never
    over-claimed as IMPLEMENTED."""

    assert classify_scheme("sf_surface_physics", opt).status is SupportStatus.REFERENCE_ONLY
    assert name.lower() in SURFACE_SCHEMES[opt].name.lower()
    # The land carry shape is frozen for the future port.
    assert LAND_CARRY_MEMBERS[opt]


@pytest.mark.parametrize("opt, name, _mod, _ep, _sp, _src", _REF_LSMS)
def test_dispatch_routes_but_not_gpu_runnable(opt, name, _mod, _ep, _sp, _src) -> None:
    """The dispatcher routes the scheme (accepted) but marks it not GPU-runnable,
    so any suite selecting it is excluded from the integrated GPU gate."""

    entry = scheme_entry("land_surface", opt)
    assert entry.option == opt
    assert entry.gpu_runnable is False
    suite = resolve_physics_suite({"sf_surface_physics": opt})
    assert suite.land_surface.option == opt
    assert suite.gpu_gate_ready is False


@pytest.mark.parametrize("opt, name, _mod, _ep, _sp, _src", _REF_LSMS)
def test_operational_scan_fails_closed_with_named_reason(opt, name, _mod, _ep, _sp, _src) -> None:
    """The operational scan must fail-close RUC/SSiB with a scheme-specific reason
    (never silently substituted by another LSM)."""

    from gpuwrf.runtime.operational_mode import OperationalNamelist, _resolve_operational_suite

    nml = _minimal_namelist(sf_surface_physics=opt)
    with pytest.raises(UnsupportedSchemeSelection) as excinfo:
        _resolve_operational_suite(nml)
    assert f"sf_surface_physics={opt}" in str(excinfo.value)
    assert "oracle staged" in str(excinfo.value)


@pytest.mark.parametrize("opt, name, mod, ep, _sp, _src", _REF_LSMS)
def test_kernel_stub_raises_not_silent(opt, name, mod, ep, _sp, _src) -> None:
    """The JAX column-kernel stub raises NotImplementedError (carry-over) rather
    than silently returning a wrong land state, and the dispatch entrypoint
    resolves to it."""

    entry = scheme_entry("land_surface", opt)
    assert entry.owner_module == mod and entry.entrypoint == ep
    module = importlib.import_module(mod)
    fn = getattr(module, ep)
    with pytest.raises(NotImplementedError, match="REFERENCE-ONLY"):
        fn()
    # The kernel's frozen land-carry NamedTuple fields match the registry carry.
    carry_cls = next(
        getattr(module, n)
        for n in dir(module)
        if n.endswith("LandState") and hasattr(getattr(module, n), "_fields")
    )
    assert tuple(carry_cls._fields) == tuple(LAND_CARRY_MEMBERS[opt])


@pytest.mark.parametrize("opt, name, _mod, _ep, savepoint_dir, src", _REF_LSMS)
def test_oracle_savepoints_are_real_non_self_compare_evidence(
    opt, name, _mod, _ep, savepoint_dir, src
) -> None:
    """The staged Fortran oracle savepoints exist, are finite, regime-varying, and
    were produced from the UNMODIFIED pristine WRF source (checksummed). This is
    the load-bearing non-self-compare evidence; the JAX-vs-oracle parity test is a
    documented carry-over (the kernel is not yet ported)."""

    sp_dir = _REPO_ROOT / savepoint_dir
    assert sp_dir.is_dir(), f"missing oracle savepoint dir {sp_dir}"
    jsons = sorted(p for p in sp_dir.glob("*.json"))
    assert jsons, f"no oracle savepoint JSON under {sp_dir}"

    # The oracle checksums the UNMODIFIED pristine WRF source it compiled.
    checks = list(sp_dir.glob("*checksums*.txt"))
    assert checks, f"missing WRF source checksum file under {sp_dir}"
    checksum_text = checks[0].read_text()
    assert src in checksum_text, f"{src} not recorded in {checks[0]}"
    pristine = Path(
        os.environ.get("GPUWRF_WRF_ROOT", "/home/user/src/wrf_pristine/WRF")
    ) / "phys" / src
    if pristine.exists():
        import hashlib

        digest = hashlib.sha256(pristine.read_bytes()).hexdigest()
        assert digest in checksum_text, (
            f"oracle checksum for {src} does not match the current pristine source "
            "(oracle must be rebuilt against the unmodified WRF tree)"
        )

    # Every numeric output field is finite, and at least one key flux varies across
    # the regimes (the oracle is not a degenerate constant -- real physics).
    hfx_samples: list[float] = []
    for jf in jsons:
        data = json.loads(jf.read_text())
        flat = _iter_numbers(data)
        assert all(_isfinite(v) for v in flat), f"non-finite value in {jf.name}"
        hfx = _find_field(data, ("HFX", "XSHF", "HFLUX"))
        if hfx is not None:
            hfx_samples.extend(hfx if isinstance(hfx, list) else [hfx])
    assert hfx_samples, f"{name} oracle savepoints expose no sensible-heat-flux field"
    assert max(hfx_samples) - min(hfx_samples) > 1.0, (
        f"{name} oracle HFX does not vary across regimes (degenerate evidence)"
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _minimal_namelist(*, sf_surface_physics: int):
    """A minimal CPU OperationalNamelist pinned to a land-surface option.

    Mirrors the operational-smoke ``_namelist`` helper but trimmed to what
    ``_resolve_operational_suite`` reads, so this test stays light + CPU-only.
    """

    import dataclasses

    import jax.numpy as jnp

    from gpuwrf.contracts.grid import (
        BCMetadata,
        DycoreMetrics,
        GridSpec,
        Projection,
        TerrainProvenance,
        VerticalCoord,
    )
    from gpuwrf.contracts.state import Tendencies
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    nz, ny, nx = 24, 4, 4
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="v017-lsm-adv", sha256="v017-lsm-adv", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="v017-lsm-adv-flat",
    )
    grid = GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)

    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    tend = Tendencies(
        z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)),
        z((nz, ny, nx)), z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)),
    )
    base = OperationalNamelist.from_grid(grid, dt_s=20.0, tendencies=tend)
    return dataclasses.replace(
        base, run_physics=True, sf_surface_physics=sf_surface_physics
    )


def _iter_numbers(obj):
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_iter_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_iter_numbers(v))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    return out


def _isfinite(v: float) -> bool:
    return v == v and abs(v) != float("inf")


def _find_field(data, names):
    """Find the first matching named field anywhere in the nested savepoint dict."""

    if isinstance(data, dict):
        for key, val in data.items():
            if key in names:
                return val
        for val in data.values():
            found = _find_field(val, names)
            if found is not None:
                return found
    return None
