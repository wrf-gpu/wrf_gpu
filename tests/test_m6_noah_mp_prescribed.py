from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.gen2_accessor import DEFAULT_M6_GEN2_RUN_DIR, Gen2Run
from gpuwrf.io.land_state import build_land_state_manifest, load_prescribed_land_state
from gpuwrf.physics.noah_mp import mavail_from_prescribed_fields, prescribe_noah_mp_state, roughness_from_prescribed_fields


RUN_PATH = DEFAULT_M6_GEN2_RUN_DIR


def test_roughness_surrogate_uses_water_and_land_masks_when_cm_is_zero():
    xland = jnp.asarray([[1.0, 2.0]])
    landmask = jnp.asarray([[1.0, 0.0]])
    vegfra = jnp.asarray([[50.0, 0.0]])
    cm = jnp.zeros((1, 2))
    z0 = roughness_from_prescribed_fields(xland, landmask, vegfra=vegfra, cm=cm)
    assert float(z0[0, 0]) > float(z0[0, 1])
    assert 0.02 <= float(z0[0, 0]) <= 0.20
    assert 1.0e-4 <= float(z0[0, 1]) <= 5.0e-3


def test_landuse_table_initializes_roughness_and_mavail_when_lu_index_is_available():
    xland = jnp.asarray([[1.0, 2.0]])
    landmask = jnp.asarray([[1.0, 0.0]])
    lu_index = jnp.asarray([[13.0, 17.0]])
    z0 = roughness_from_prescribed_fields(xland, landmask, lu_index=lu_index)
    mavail = mavail_from_prescribed_fields(xland, landmask, jnp.ones((1, 1, 2)), lu_index=lu_index)

    assert float(z0[0, 0]) == 0.8
    assert float(z0[0, 1]) == 1.0e-4
    assert float(mavail[0, 0]) == 0.1
    assert float(mavail[0, 1]) == 1.0


def test_prescribed_noah_mp_state_is_bounded_and_non_prognostic():
    state = prescribe_noah_mp_state(
        t_skin=np.asarray([[350.0, 250.0]]),
        smois=np.asarray([[[1.2, -0.1]]]),
        sh2o=np.asarray([[[0.5, 2.0]]]),
        tslb=np.asarray([[[360.0, 170.0]]]),
        xland=np.asarray([[1.0, 2.0]]),
        landmask=np.asarray([[1.0, 0.0]]),
        lakemask=np.asarray([[0.0, 0.0]]),
        ivgtyp=np.asarray([[7, 16]]),
        isltyp=np.asarray([[4, 14]]),
        lu_index=np.asarray([[7.0, 16.0]]),
        sst=np.asarray([[294.0, 295.0]]),
        vegfra=np.asarray([[20.0, 0.0]]),
    )
    assert float(jnp.max(state.t_skin)) == 340.0
    assert float(jnp.min(state.soil_moisture)) == 0.0
    assert float(jnp.max(state.soil_liquid)) == 1.0
    assert state.soil_moisture.shape == (1, 1, 2)
    assert state.roughness_m.shape == (1, 2)


def test_gen2_prescribed_land_state_manifest_when_fixture_available():
    if not RUN_PATH.exists():
        pytest.skip("Gen2 run fixture unavailable")
    run = Gen2Run(RUN_PATH)
    state = load_prescribed_land_state(run, "d02", 0)
    manifest = build_land_state_manifest(run, "d02", 0, state)
    assert manifest["status"] == "PASS"
    assert manifest["variables"]["TSK"]["available"] is True
    assert manifest["variables"]["SMOIS"]["available"] is True
    assert manifest["variables"]["XLAND"]["available"] is True
    assert manifest["summaries"]["roughness_m"]["finite"] is True
    assert manifest["source_sha256"]
