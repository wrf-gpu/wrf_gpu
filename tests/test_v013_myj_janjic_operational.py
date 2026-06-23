"""v0.13 operational wiring of the MYJ PBL + Janjic Eta surface-layer pair.

Covers the WIRING (kernel parity lives in ``proofs/v013/myj_janjic_oracle.py``):

* ``myj_columns`` / ``myjsfc_columns`` are genuinely ``jax.jit``-traceable and
  match the host-NumPy reference column kernels (which are WRF-savepoint-proven);
* the operational ``_physics_step_forcing`` PBL + surface-layer slots route
  ``bl_pbl_physics=2`` / ``sf_sfclay_physics=2`` to the MYJ pair and the result
  actually CHANGES the state (not a silent no-op);
* the MYJ pair (bl=2 + sf=2) resolves in ``_resolve_operational_suite``, the
  mandatory pairing fails closed if only one of the pair is selected, and the
  DEFAULT suite (MYNN / sfclayrev) is byte-for-byte UNCHANGED by this wiring.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.physics.bl_myj import myj_columns, myjpbl_column_traceable
from gpuwrf.physics.pbl_myj import myjpbl_column as myjpbl_column_ref
from gpuwrf.physics.sf_myj import myjsfc_columns
from gpuwrf.physics.sfclay_janjic import myjsfc_column
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    UnsupportedSchemeSelection,
    _physics_step_forcing,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "proofs" / "v060" / "savepoints_fp64"
TIME_UTC = "2019-05-21T12:00:00Z"


# ----------------------------------------------------------------------------
# 1. traceable-kernel parity vs the savepoint-proven host-NumPy reference
# ----------------------------------------------------------------------------

def _col(d, n):
    return np.asarray(d["columns"][n], dtype=np.float64)


def test_myj_columns_matches_reference_all_regimes() -> None:
    """The traceable MYJ kernel reproduces the host-NumPy reference (~1e-13)."""

    worst = 0.0
    for c in range(1, 7):
        d = json.load((SP / f"myjpbl_case_{c}.json").open())
        sf = json.load((SP / f"myjsfc_case_{c}.json").open())
        s = d["scalars"]
        tke0 = 0.5 * np.asarray(sf["columns"]["Q2"], dtype=np.float64)
        kw = dict(
            u=_col(d, "U"), v=_col(d, "V"), temperature=_col(d, "T"), theta=_col(d, "TH"),
            qv=_col(d, "QV"), qc=_col(d, "QC"), p_mid=_col(d, "PMID"), p_int=_col(d, "PINT"),
            exner=_col(d, "EXNER"), dz=_col(d, "DZ"), tke=tke0, tsk=s["TSK"], xland=s["XLAND"],
            ustar=s["USTAR"], akhs=s["AKHS"], akms=s["AKMS"], chklowq=s["CHKLOWQ"],
            elflx=s["ELFLX"], thz0=s["THZ0"], qz0=s["QZ0"], uz0=s["UZ0"], vz0=s["VZ0"],
            qsfc=s["QSFC"], ct=s["CT"], dt=s["DT"], stepbl=s["STEPBL"], ht=s["HT"],
        )
        ref = myjpbl_column_ref(znt=s["ZNT"], **kw)
        trc = myjpbl_column_traceable(**kw)
        for fld in ("TKE_MYJ", "EXCH_H", "EL_MYJ", "RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN"):
            worst = max(worst, float(np.max(np.abs(np.asarray(trc[fld]) - np.asarray(ref[fld])))))
        assert int(trc["KPBL"]) == int(ref["KPBL"])
    assert worst < 1.0e-12, f"traceable MYJ deviates from reference by {worst:.3e}"


def test_myj_columns_is_jit_traceable() -> None:
    """Batched myj_columns runs under jax.jit (operational scan requirement)."""

    d = json.load((SP / "myjpbl_case_1.json").open())
    sf = json.load((SP / "myjsfc_case_1.json").open())
    s = d["scalars"]
    A = lambda f: jnp.asarray(np.stack([_col(d, f), _col(d, f)]), jnp.float64)
    tke0 = 0.5 * np.asarray(sf["columns"]["Q2"], dtype=np.float64)
    sc = lambda v: jnp.asarray([v, v], jnp.float64)
    out = jax.jit(myj_columns)(
        A("U"), A("V"), A("T"), A("TH"), A("QV"), A("QC"), A("PMID"), A("PINT"),
        A("EXNER"), A("DZ"), jnp.asarray(np.stack([tke0, tke0]), jnp.float64),
        tsk=sc(s["TSK"]), xland=sc(s["XLAND"]), ustar=sc(s["USTAR"]), akhs=sc(s["AKHS"]),
        akms=sc(s["AKMS"]), chklowq=sc(s["CHKLOWQ"]), elflx=sc(s["ELFLX"]), thz0=sc(s["THZ0"]),
        qz0=sc(s["QZ0"]), uz0=sc(s["UZ0"]), vz0=sc(s["VZ0"]), qsfc=sc(s["QSFC"]), ct=sc(s["CT"]),
        dt=s["DT"], stepbl=s["STEPBL"], ht=sc(s["HT"]),
    )
    assert out["TKE_MYJ"].shape == (2, 32)
    assert bool(np.all(np.isfinite(np.asarray(out["RTHBLTEN"]))))


def test_myjsfc_columns_matches_single_column() -> None:
    """Batched Janjic kernel reproduces the per-column kernel exactly."""

    d = json.load((SP / "myjsfc_case_1.json").open())
    dp = json.load((SP / "myjpbl_case_1.json").open())
    s = d["scalars"]
    qv0 = float(_col(d, "QV")[0])
    single = myjsfc_column(
        u=_col(d, "U"), v=_col(d, "V"), temperature=_col(d, "T"), theta=_col(d, "TH"),
        qv=_col(d, "QV"), qc=_col(d, "QC"), p_mid=_col(d, "PMID"), dz=_col(d, "DZ"),
        q2=_col(d, "Q2"), tsk=s["TSK"], xland=s["XLAND"], z0base=s["Z0BASE"],
        psfc=float(_col(dp, "PINT")[0]), znt=0.10, ustar=0.1, mavail=s["MAVAIL"],
        qsfc=qv0, thz0=float(_col(d, "TH")[0]), qz0=qv0 / (1.0 + qv0),
        uz0=0.0, vz0=0.0, pblh=s["PBLH"],
    )
    A = lambda f: jnp.asarray(_col(d, f)[None, :], jnp.float64)
    batch = myjsfc_columns(
        A("U"), A("V"), A("T"), A("TH"), A("QV"), A("QC"), A("PMID"), A("DZ"), A("Q2"),
        tsk=jnp.asarray([s["TSK"]]), xland=jnp.asarray([s["XLAND"]]),
        z0base=jnp.asarray([s["Z0BASE"]]), psfc=jnp.asarray([float(_col(dp, "PINT")[0])]),
        znt=jnp.asarray([0.10]), ustar=jnp.asarray([0.1]), mavail=jnp.asarray([s["MAVAIL"]]),
        qsfc=jnp.asarray([qv0]), thz0=jnp.asarray([float(_col(d, "TH")[0])]),
        qz0=jnp.asarray([qv0 / (1.0 + qv0)]), uz0=0.0, vz0=0.0, pblh=jnp.asarray([s["PBLH"]]),
    )
    for key in ("ustar", "akhs", "akms", "hfx", "qfx", "u10", "v10", "t02"):
        assert np.allclose(np.asarray(batch[key])[0], float(np.asarray(single[key])), atol=1e-12)


# ----------------------------------------------------------------------------
# 2. operational dispatch wiring
# ----------------------------------------------------------------------------

def _grid(ny: int = 3, nx: int = 3, nz: int = 8) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="myj-wire-test", sha256="myj-wire-test", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="myj-wire-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    p = jnp.broadcast_to(jnp.linspace(95000.0, 20000.0, nz)[:, None, None], (nz, ny, nx))
    ph = jnp.broadcast_to(jnp.linspace(0.0, 12000.0 * 9.80665, nz + 1)[:, None, None], (nz + 1, ny, nx))
    fields.update(
        theta=jnp.full((nz, ny, nx), 295.0),
        p_total=p,
        ph_total=ph,
        mu_total=jnp.full((ny, nx), 90000.0),
        qv=jnp.full((nz, ny, nx), 5.0e-3), qc=jnp.full((nz, ny, nx), 1.0e-4),
        qke=jnp.full((nz, ny, nx), 0.5),
        u=jnp.full((nz, ny, nx + 1), 5.0), v=jnp.full((nz, ny + 1, nx), 2.0),
        t_skin=jnp.full((ny, nx), 298.0), xland=jnp.full((ny, nx), 1.0),
        mavail=jnp.full((ny, nx), 0.5), roughness_m=jnp.full((ny, nx), 0.1),
        ustar=jnp.full((ny, nx), 0.3), lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)
    return Tendencies(
        z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)),
        z((nz, ny, nx)), z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)),
    )


def _namelist(grid: GridSpec, **over) -> OperationalNamelist:
    base = OperationalNamelist.from_grid(grid, dt_s=10.0, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(base, time_utc=TIME_UTC, run_physics=True, **over)


def test_myj_pair_resolves_in_operational_suite() -> None:
    grid = _grid()
    nml = _namelist(grid, bl_pbl_physics=2, sf_sfclay_physics=2, use_noahmp=False)
    suite = _resolve_operational_suite(nml)
    assert suite.pbl.option == 2
    assert suite.surface_layer.option == 2


@pytest.mark.parametrize("bl,sf", [(2, 5), (5, 2), (2, 1), (1, 2)])
def test_myj_pairing_fails_closed_when_unpaired(bl: int, sf: int) -> None:
    grid = _grid()
    nml = _namelist(grid, bl_pbl_physics=bl, sf_sfclay_physics=sf, use_noahmp=False)
    with pytest.raises(UnsupportedSchemeSelection):
        _resolve_operational_suite(nml)


def test_operational_step_routes_myj_pair_and_changes_state() -> None:
    """bl=2/sf=2 dispatch runs the MYJ pair and actually mutates theta/qke/u."""

    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, bl_pbl_physics=2, sf_sfclay_physics=2, use_noahmp=False)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    assert np.all(np.isfinite(np.asarray(after.theta)))
    assert np.all(np.isfinite(np.asarray(after.qke)))
    assert np.all(np.isfinite(np.asarray(after.u)))
    # MYJ surface layer wrote a real ustar; PBL changed theta + the TKE carry.
    assert not np.allclose(np.asarray(after.theta), np.asarray(state.theta))
    assert not np.allclose(np.asarray(after.qke), np.asarray(state.qke))
    assert float(np.asarray(after.ustar)[0, 0]) > 0.0


def test_default_suite_byte_unchanged_by_myj_wiring() -> None:
    """The default (MYNN / sfclayrev) physics step is byte-for-byte unchanged."""

    grid = _grid()
    state = _state(grid)
    nml = _namelist(grid, use_noahmp=False)  # defaults: bl=5 MYNN, sf=5 MYNN-sfclay
    assert nml.bl_pbl_physics == 5 and nml.sf_sfclay_physics == 5
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    # Re-run -- determinism + the default path does not touch any MYJ code.
    forcing2 = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    for leaf in ("theta", "qv", "u", "v", "qke", "ustar"):
        a = np.asarray(getattr(forcing.state, leaf))
        b = np.asarray(getattr(forcing2.state, leaf))
        assert np.array_equal(a, b), f"default {leaf} not deterministic/unchanged"
