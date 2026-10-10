"""v0.3.4 o1-grell: Grell-3D (cu=5) / Grell-Devenyi (cu=93) vs pristine WRF.

Gate: the JAX ports in ``gpuwrf.physics._grell_cup_jax`` reproduce the
UNMODIFIED pristine-WRF ``G3DRV`` + ``conv_grell_spread3d`` / ``GRELLDRV``
tile oracles (``proofs/v034/oracle/cumulus_grell``; savepoints staged by
``proofs/v034/make_grell_savepoints.py``) on 12x12 tiles of varied soundings
(deep / capped-shallow / stable / marginal / cold, land + water, day + night):

* fp64 inputs vs the ``-fdefault-real-8`` oracle: machine precision (the
  scheme's own conditioning amplifies round-off ~1e3; observed <= 5e-11);
* fp32 inputs vs the WRF REAL build: within the scheme's REAL conditioning.

Inputs are regenerated from the recorded seed and checked by sha256, so the
oracle outputs are tied to exactly the inputs the port sees.  Every case is
census-checked for non-trivial activity first (E154).  CPU only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / "proofs" / "v034"
SAVE = PROOFS / "savepoints" / "cumulus_grell"
sys.path.insert(0, str(PROOFS))

import grell_oracle_io as gio  # noqa: E402
import grell_parity_lib as gpl  # noqa: E402

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics import _grell_cup_jax as G  # noqa: E402

TENDENCIES = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
G3_FIELDS = TENDENCIES + ("CUGD_TTEN", "CUGD_QVTEN", "CUGD_QCTEN", "CUGD_TTENS", "CUGD_QVTENS",
                          "GDC", "GDC2", "RAINCV", "PRATEC", "DRV_RAINCV", "DRV_PRATEC", "HTOP",
                          "HBOT", "KTOP_DEEP", "EDT_OUT", "XMB_SHALLOW", "K22_SHALLOW",
                          "KBCON_SHALLOW", "KTOP_SHALLOW", "XF_ENS", "PR_ENS")
GD_FIELDS = TENDENCIES + ("GDC", "GDC2", "RAINCV", "PRATEC", "HTOP", "HBOT", "KTOP_DEEP",
                          "XF_ENS", "PR_ENS")


EXPECTED_SAVEPOINTS = ("g3_deep_r4", "g3_deep_r8", "g3_highres_r8", "g3_ichoice1_kx33_r8",
                       "g3_shallow_r8", "gd_r4_s2", "gd_r8_s1", "gd_r8_s2")


def test_savepoints_present():
    """Loud guard: the staged oracle cases must exist (a missing/ignored *.npz would make
    the parametrized gates below silently collect zero cases)."""
    found = tuple(sorted(p.stem for p in SAVE.glob("*.npz")))
    assert found == EXPECTED_SAVEPOINTS, f"staged Grell savepoints missing/unexpected: {found}"


def _load(name):
    data = np.load(SAVE / f"{name}.npz")
    meta = json.loads(str(data["meta"]))
    oracle = {k: data[k] for k in data.files if k != "meta"}
    tile = gio.make_tile(meta["seed"], nx=meta["nx"], ny=meta["ny"], kx=meta["kx"])
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        gio.write_input(fh.name, tile, **meta["opts"])
        sha = hashlib.sha256(Path(fh.name).read_bytes()).hexdigest()
    assert sha == meta["input_sha256"], f"{name}: regenerated inputs differ from the oracle's"
    return meta, oracle, tile


def _run(meta, tile):
    dtype = jnp.float64 if meta["precision"] == "r8" else jnp.float32
    opts = dict(meta["opts"])
    inp = gpl.tile_to_jax_inputs(tile, dtype)
    if meta["scheme"] == "g3":
        fn = jax.jit(lambda x: G.g3drv_tile(
            **x, dt=opts["dt"], dx=opts["dx"], cugd_avedx=opts.get("cugd_avedx", 1),
            ishallow=opts.get("ishallow", 0), ichoice=opts.get("ichoice", 0)))
        out = fn(inp)
    else:
        inp.pop("kpbl")
        ht = jnp.asarray(tile["f2"]["htop"].T, dtype)
        hb = jnp.asarray(tile["f2"]["hbot"].T, dtype)
        fn = jax.jit(lambda x, a, b: G.grelldrv_tile(**x, dt=opts["dt"], htop=a, hbot=b))
        out = fn(inp, ht, hb)
    return gpl.jax_out_to_oracle_layout(jax.tree_util.tree_map(np.asarray, out))


def _rel(oracle, port, key):
    a = np.asarray(oracle[key], np.float64)
    b = np.asarray(port[key], np.float64)
    assert np.isfinite(b).all(), f"{key}: non-finite port output"
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


@pytest.mark.parametrize("name", EXPECTED_SAVEPOINTS)
def test_savepoint_census_is_active(name):
    """E154: every staged oracle case exercises active deep convection, null
    columns and both ice/liquid detrainment (no vacuous gate)."""
    meta, oracle, tile = _load(name)
    interior = np.zeros((meta["nx"], meta["ny"]), bool)
    interior[4:-4, 4:-4] = True
    rain = oracle["RAINCV"] > 0
    assert rain.sum() >= 4, f"{name}: too few raining columns"
    assert (interior & ~rain).sum() >= 2, f"{name}: no null interior column"
    assert np.abs(oracle["RQICUTEN"]).max() > 0 and np.abs(oracle["RQCCUTEN"]).max() > 0
    land = tile["f2"]["xland"] < 1.5
    assert (rain & land).any() and (rain & ~land).any(), f"{name}: rain not on land+water"
    if meta["opts"].get("ishallow", 0) == 1:
        assert (oracle["XMB_SHALLOW"] > 0).sum() >= 4, f"{name}: shallow convection inactive"


FP64_G3 = ("g3_deep_r8", "g3_shallow_r8", "g3_highres_r8", "g3_ichoice1_kx33_r8")
FP64_GD = ("gd_r8_s1", "gd_r8_s2")


@pytest.mark.parametrize("name", FP64_G3 + FP64_GD)
def test_fp64_parity_with_pristine_wrf(name):
    meta, oracle, tile = _load(name)
    port = _run(meta, tile)
    fields = G3_FIELDS if meta["scheme"] == "g3" else GD_FIELDS
    rel = {k: _rel(oracle, port, k) for k in fields}
    # shallow-closure xmb is a ratio of a tiny work-function difference
    # ((xaa3-aa3)/mbdt_s): its round-off amplification is larger than the rest.
    limits = {k: (1e-8 if k == "XMB_SHALLOW" else 1e-9) for k in fields}
    bad = {k: v for k, v in rel.items() if v > limits[k]}
    assert not bad, f"{name}: fields beyond fp64 machine-precision bound: {bad}"
    for k in ("KTOP_DEEP", "HTOP", "HBOT"):
        assert rel[k] == 0.0, f"{name}: integer cloud levels differ ({k})"


def test_fp64_closure_diagnostics_g3():
    """APR_* closure-group precipitation diagnostics (default clos_choice=0)."""
    meta, oracle, tile = _load("g3_shallow_r8")
    port = _run(meta, tile)
    for k in gpl.APR_NAMES:
        assert _rel(oracle, port, k) < 1e-9, k


@pytest.mark.parametrize("name", ("g3_deep_r4", "gd_r4_s2"))
def test_fp32_tracks_wrf_real_build(name):
    """fp32 (WRF REAL) port vs the REAL=4 oracle build: same active columns and
    cloud levels; amplitudes within the scheme's REAL conditioning (WRF REAL vs
    WRF fp64 itself differs by up to ~2e-2 on these closures)."""
    meta, oracle, tile = _load(name)
    port = _run(meta, tile)
    assert ((port["RAINCV"] > 0) == (oracle["RAINCV"] > 0)).all()
    assert (port["KTOP_DEEP"] == oracle["KTOP_DEEP"]).all()
    for k in TENDENCIES + ("RAINCV", "PRATEC"):
        assert _rel(oracle, port, k) < 1e-2, k
