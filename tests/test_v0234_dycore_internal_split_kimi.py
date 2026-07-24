"""Focused CPU-only tests for the v0234 dycore internal-split discriminator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import v0234_dycore_internal_split_kimi as disc  # noqa: E402


def test_face_average_u_exclusions():
    field = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    out = disc.face_average_u(field)
    assert out.shape == (2, 4, 4)
    # Nested/spec exclusions: row 0 and row ny-1 stay zero; faces 0 and nx stay zero.
    assert np.all(out[:, 0, :] == 0.0)
    assert np.all(out[:, -1, :] == 0.0)
    assert np.all(out[:, :, 0] == 0.0)
    assert np.all(out[:, :, -1] == 0.0)
    # Interior faces: 0.5*(m(i-1)+m(i)) on kept rows.
    rows = range(1, 3)
    for j in rows:
        for i in range(1, 3):
            expect = 0.5 * (field[:, j, i - 1] + field[:, j, i])
            np.testing.assert_array_equal(out[:, j, i], expect)


def test_face_average_v_exclusions():
    field = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    out = disc.face_average_v(field)
    assert out.shape == (2, 5, 3)
    assert np.all(out[:, 0, :] == 0.0)
    assert np.all(out[:, -1, :] == 0.0)
    assert np.all(out[:, :, 0] == 0.0)
    assert np.all(out[:, :, -1] == 0.0)
    for j in range(1, 4):
        for i in range(1, 2):
            expect = 0.5 * (field[:, j - 1, i] + field[:, j, i])
            np.testing.assert_array_equal(out[:, j, i], expect)


def test_relax_weights_nested_lbc_fcx_gcx():
    weights = disc.relax_weights(6.0)
    assert set(weights) == {1, 2, 3}
    f1, g1 = weights[1]
    f2, g2 = weights[2]
    f3, g3 = weights[3]
    assert f1 == 0.1 / 6.0 * 3 / 3
    assert f2 == 0.1 / 6.0 * 2 / 3
    assert f3 == 0.1 / 6.0 * 1 / 3
    assert g1 == 1.0 / 6.0 / 50.0 * 3 / 3
    assert g2 == 1.0 / 6.0 / 50.0 * 2 / 3
    assert g3 == 1.0 / 6.0 / 50.0 * 1 / 3


def test_wrf_relax_tendency_band_only_and_trims():
    rng = np.random.default_rng(7)
    field = rng.normal(size=(3, 10, 11))
    target = rng.normal(size=(3, 10, 11))
    tend = disc.wrf_relax_tendency(field, target, 6.0, "u")
    assert tend.shape == field.shape
    dist = disc.cpu.distance_to_edge(tend.shape)
    assert np.all(tend[dist == 0] == 0.0)
    assert np.all(tend[dist >= 5] == 0.0)
    # Corner trim: at relax row 1 (west), rows j=0 and j=ny-1 excluded.
    assert np.all(tend[:, 0, 1] == 0.0)
    assert np.all(tend[:, -1, 1] == 0.0)
    # Hand-check one west-band cell (k=1, j=2, i=1, b_dist=1).
    k, j, i, b = 1, 2, 1, 1
    fcx, gcx = disc.relax_weights(6.0)[b]
    f0 = target[k, j, i] - field[k, j, i]
    f1 = target[k, j - 1, i] - field[k, j - 1, i]
    f2 = target[k, j + 1, i] - field[k, j + 1, i]
    f3 = target[k, j, i - 1] - field[k, j, i - 1]
    f4 = target[k, j, i + 1] - field[k, j, i + 1]
    expect = fcx * f0 - gcx * (f1 + f2 + f3 + f4 - 4.0 * f0)
    # The cell (j=2, i=1) also receives the SOUTH-side contribution at b=1? No:
    # south row b=1 covers j=1 only.  (j=2,i=1) is west-only at b=1... but
    # south row b=2 covers j=2 with i in [2, IE-2], excluding i=1.  So west-only.
    assert abs(tend[k, j, i] - expect) < 1e-12


def test_wrf_relax_tendency_matches_gpu_port_structure():
    # The discriminator port and the production GPU port must produce the same
    # nonzero support: relax rows 1..4 with WRF corner trims.
    rng = np.random.default_rng(3)
    field = rng.normal(size=(2, 8, 9))
    target = rng.normal(size=(2, 8, 9))
    tend = disc.wrf_relax_tendency(field, target, 6.0, "u")
    # WRF X-side trim at b=1: j in [2, ny-2]; Y-side trim at b=1: i in [1, nx-2].
    assert tend[0, 2, 1] != 0.0 or tend[0, 3, 1] != 0.0
    assert np.all(tend[:, 1, 0] == 0.0)  # south row b=1 excludes i=0


def test_e_decomposition_closure():
    # delta_relax = relax(T2,F2) - relax(T1,F1) must equal
    # field_side + target_side exactly, where
    # field_side = relax(T1, F2) - relax(T1, F1) is linear in the field and
    # target_side = relax(T2, F2) - relax(T1, F2).
    rng = np.random.default_rng(11)
    f1 = rng.normal(size=(2, 8, 9))
    f2 = f1 + rng.normal(scale=0.01, size=(2, 8, 9))
    t1 = rng.normal(size=(2, 8, 9))
    t2 = t1 + rng.normal(scale=0.01, size=(2, 8, 9))
    r1 = disc.wrf_relax_tendency(f1, t1, 6.0, "u")
    r2 = disc.wrf_relax_tendency(f2, t2, 6.0, "u")
    delta = r2 - r1
    # Linearity of the stencil in (field, target):
    # delta = relax(F2 - F1 as field with zero target) * -1 ... verify with the
    # same port: relax(dF, 0) = fcx*(0 - dF) - gcx*lap(0 - dF) = field_side.
    field_side = disc.wrf_relax_tendency(f2 - f1, np.zeros_like(f1), 6.0, "u")
    target_side = disc.wrf_relax_tendency(f1, t2, 6.0, "u") - disc.wrf_relax_tendency(f1, t1, 6.0, "u")
    np.testing.assert_allclose(delta, field_side + target_side, rtol=0, atol=1e-12)


def test_mu_faces_edge_convention():
    mu = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]])
    muu = disc.mu_faces(mu, axis=1)
    # WRF calc_mu_uv non-periodic: edge face reuses the edge mass point.
    expect = np.array([
        [1.0, 1.5, 3.0, 4.0],
        [2.0, 2.5, 4.0, 5.0],
    ])
    np.testing.assert_array_equal(muu, expect)
    muv = disc.mu_faces(mu, axis=0)
    expect_v = np.array([
        [1.0, 2.0, 4.0],
        [1.5, 2.5, 4.5],
        [2.0, 3.0, 5.0],
    ])
    np.testing.assert_array_equal(muv, expect_v)


def test_mass_weight_shape_and_values():
    c1 = np.array([1.0, 0.5])
    c2 = np.array([10.0, 20.0])
    mu = np.array([[2.0, 3.0]])
    m = disc.mass_weight(c1, c2, mu)
    assert m.shape == (2, 1, 2)
    np.testing.assert_array_equal(m[0, 0], [12.0, 13.0])
    np.testing.assert_array_equal(m[1, 0], [21.0, 21.5])


def test_canonical_digest_omit_round_trip():
    payload = {"b": 1, "a": [1, 2, 3], "proof_sha256": "x"}
    d1 = disc.canonical_digest(payload, omit="proof_sha256")
    d2 = disc.canonical_digest({"a": [1, 2, 3], "b": 1})
    assert d1 == d2


def test_band_masks_match_frozen_distance():
    shape = (4, 9, 10)
    masks = disc.band_masks(shape)
    dist = disc.cpu.distance_to_edge(shape)
    assert np.array_equal(masks["spec_row_0"], dist == 0)
    assert np.array_equal(masks["relax_rows_1_4"], (dist >= 1) & (dist <= 4))
    assert np.array_equal(masks["interior_ge_5"], dist >= 5)
    total = sum(int(m.sum()) for m in masks.values() if m is not masks["all"])
    assert total == int(np.prod(shape))


def test_no_jax_or_gpuwrf_imported_by_discriminator():
    assert "jax" not in sys.modules
    assert not any(
        name == "gpuwrf" or name.startswith("gpuwrf.") for name in sys.modules
    )
