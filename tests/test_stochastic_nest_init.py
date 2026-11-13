"""Nested original-WRF stochastic initialization and nested admission (lane v50-stoch).

med_nest_initial (share/mediation_integrate.F:845) re-calls start_domain(d01) once per d01 child,
after set_tiles. INITIALIZE_STOCH then re-seeds every stream and, unlike the first call from main,
really spins RAND_PERT_UPDATE('R') ten times (module_stoch.F:242-270). The fixture is the original
module_stoch.F/FFTPACK5 compiled by proofs/u2_stoch/build_oracle.py with
proofs/v50_stoch/nest_spin_oracle.f90 (manifest tests/fixtures/v50_nest_spin_oracle.json).
"""
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.io.stochastic_config import (bind_stochastic, d01_children, stochastic_refusals,
                                          with_executed_d01_children)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GRID = SimpleNamespace(nx=17, ny=15, nz=4, dx=9000., dy=9000.)
START = datetime(2026, 2, 28, 0)
CONTROL = {"rand_perturb": 1, "nens": 3, "iseed_rand_pert": 17, "gridpt_stddev_rand_pert": .5,
           "stddev_cutoff_rand_pert": 2., "lengthscale_rand_pert": 15000., "rand_pert_vertstruc": 0}


def _oracle():
    manifest = json.loads((FIXTURES / "v50_nest_spin_oracle.json").read_text())
    path = FIXTURES / "v50_nest_spin_oracle.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["npz_sha256"]
    return np.load(path)


def _close(actual, expected, what):
    actual = np.asarray(actual)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    scale = float(np.max(np.abs(expected)))
    # Same pre-registered bound as the u2/w41 original-WRF pattern gates.
    np.testing.assert_allclose(actual, expected, rtol=8e-6, atol=64*np.finfo(np.float32).eps*scale, err_msg=what)


def _nested(max_dom, parents, **extra):
    namelist = {"stoch": dict(CONTROL), "domains": {"max_dom": max_dom, "parent_id": parents, "e_vert": [5]*max_dom}}
    namelist["stoch"].update(extra)
    return namelist


@pytest.mark.parametrize("rounds", [0, 1, 2, 3])
def test_nest_init_rounds_match_original_wrf(rounds):
    from gpuwrf.physics.stochastic_driver import initial_stochastic
    oracle = _oracle()
    cfg = with_executed_d01_children(bind_stochastic(_nested(4, [0, 1, 1, 1]), "d01"), "d01", ("d02", "d03", "d04"))
    assert cfg.nest_init_rounds == 3
    carry = initial_stochastic(replace(cfg, nest_init_rounds=rounds), GRID, 54, START)
    p = carry.patterns[3]
    if rounds == 0:
        assert not np.any(oracle["sine"][0]) and not np.any(oracle["cosine"][0])
    _close(p.sine, oracle["sine"][rounds], f"sine round {rounds}")
    _close(p.cosine, oracle["cosine"][rounds], f"cosine round {rounds}")
    np.testing.assert_array_equal(np.asarray(carry.seeds[2]), oracle["seed"][rounds])
    # The spins have no spectral projection: the held RAND field stays zero.
    assert not np.any(np.asarray(p.field)) and not np.any(oracle["field"][rounds])


def test_first_step_after_three_children_matches_original_wrf_and_old_init_does_not():
    from gpuwrf.physics.stochastic_driver import advance_stochastic, initial_stochastic
    oracle = _oracle()
    cfg = with_executed_d01_children(bind_stochastic(_nested(4, [0, 1, 1, 1]), "d01"), "d01", ("d02", "d03", "d04"))
    stepped = advance_stochastic(initial_stochastic(cfg, GRID, 54, START), cfg, dx=9000., dy=9000.)
    _close(np.asarray(stepped.patterns[3].field)[0], oracle["field"][4][:, 0, :], "first-step RAND_PERT")
    _close(stepped.patterns[3].sine, oracle["sine"][4], "first-step sine")
    np.testing.assert_array_equal(np.asarray(stepped.seeds[2]), oracle["seed"][4])
    # Deletion sensitivity: the pre-fix single-domain init misses the nested truth.
    old = advance_stochastic(initial_stochastic(replace(cfg, nest_init_rounds=0), GRID, 54, START),
                             cfg, dx=9000., dy=9000.)
    assert not np.array_equal(np.asarray(old.seeds[2]), oracle["seed"][4])
    assert np.max(np.abs(np.asarray(old.patterns[3].field)[0] - oracle["field"][4][:, 0, :])) > .1


def test_rounds_count_d01_children_only_and_never_change_the_step_program():
    wn3 = _nested(3, [1, 1, 2])
    nine = _nested(9, [1, 1, 2, 2, 2, 2, 2, 2, 2])
    assert d01_children(wn3) == (2,) and d01_children(nine) == (2,)
    assert d01_children({"domains": {"max_dom": 1}}) == ()
    assert d01_children(_nested(3, [1, 1, 1])) == (2, 3)
    # The namelist alone never binds rounds (critic finding 3): a d01-only run from the same max_dom=3
    # files must keep the single-domain RAND stream. The nested loader counts the EXECUTED d01 edges.
    bound = [bind_stochastic(n, "d01") for n in ({"stoch": dict(CONTROL)}, wn3, _nested(3, [1, 1, 1]))]
    assert [c.nest_init_rounds for c in bound] == [0, 0, 0]
    roots = [with_executed_d01_children(c, "d01", kids) for c, kids in zip(bound, ((), ("d02",), ("d02", "d03")))]
    assert [c.nest_init_rounds for c in roots] == [0, 1, 2]
    assert with_executed_d01_children(bound[1], "d01", ()) is bound[1]          # d01-only run from wn3 files
    assert with_executed_d01_children(bind_stochastic(wn3, "d02"), "d02", ("d03",)).nest_init_rounds == 0
    # Static-switch layout (hash/eq) is what selects the compiled step program.
    assert roots[0] == roots[1] == roots[2] and len({hash(c) for c in roots}) == 1
    assert bind_stochastic(wn3, "d02").nest_init_rounds == 0
    assert bind_stochastic(wn3, "d03").root is False


def test_rounds_only_touch_the_rand_stream():
    from gpuwrf.physics.stochastic_driver import initial_stochastic
    control = dict(skebs=1, sppt=1, spp=1, rand_perturb=1, nens=3, lengthscale_sppt=15000.,
                   lengthscale_rand_pert=15000., lengthscale_spp_conv=15000., lengthscale_spp_pbl=15000.,
                   lengthscale_spp_lsm=15000.)
    cfg = bind_stochastic({"stoch": control, "domains": {"max_dom": 2, "parent_id": [0, 1]}}, "d01")
    one, zero = (initial_stochastic(replace(cfg, nest_init_rounds=r), GRID, 54, START) for r in (1, 0))
    for index, (a, b) in enumerate(zip(one.patterns, zero.patterns)):
        same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in
                   zip((a.sine, a.cosine, a.amplitude), (b.sine, b.cosine, b.amplitude)))
        assert same == (index != 3), index
    for index, (a, b) in enumerate(zip(one.seeds, zero.seeds)):
        assert np.array_equal(np.asarray(a), np.asarray(b)) == (index != 2), index


def test_nested_admission_refuses_late_d01_child_and_mixed_levels_only_when_active():
    base = {"stoch": {"skebs": 1}, "domains": {"max_dom": 3, "parent_id": [1, 1, 2], "e_vert": [45]*3},
            "time_control": {"start_hour": [0, 0, 0]}}
    assert stochastic_refusals(base) == []
    late = json.loads(json.dumps(base))
    late["time_control"]["start_hour"] = [0, 6, 0]
    assert any("d02" in e and "re-seeds" in e for e in stochastic_refusals(late))
    late_grandchild = json.loads(json.dumps(base))
    late_grandchild["time_control"]["start_hour"] = [0, 0, 6]
    assert stochastic_refusals(late_grandchild) == []  # start_domain(d02) has no stochastic body
    levels = json.loads(json.dumps(base))
    levels["domains"]["e_vert"] = [45, 45, 60]
    assert any("e_vert" in e for e in stochastic_refusals(levels))
    off = json.loads(json.dumps(late))
    off["stoch"] = {"skebs": 0}
    assert stochastic_refusals(off) == []


def test_child_history_writes_spp_spectral_coefficients_as_wrf_alloc_zero():
    """registry.stoch SPFORCC2/3/4, SPFORCS2/3/4 are 'irh' in the global spp_on package."""
    from gpuwrf.io.stochastic_schema import SPECTRAL_FIELDS, history_fields
    from gpuwrf.physics.stochastic_driver import initial_stochastic
    control = dict(skebs=1, spp=1, nens=3, lengthscale_spp_conv=15000., lengthscale_spp_pbl=15000.,
                   lengthscale_spp_lsm=15000.)
    namelist = {"stoch": control, "domains": {"max_dom": 2, "parent_id": [0, 1]}}
    child_grid = SimpleNamespace(nx=9, ny=9, nz=4, dx=3000., dy=3000.)
    child = initial_stochastic(bind_stochastic(namelist, "d02"), child_grid, 18, START)
    fields = history_fields(child, 4, 9, 9)
    assert set(SPECTRAL_FIELDS) <= set(fields)
    for name in SPECTRAL_FIELDS:
        value = np.asarray(fields[name])
        assert value.shape == (10, 10) and value.dtype == np.float32 and not value.any(), name
    root = initial_stochastic(bind_stochastic(namelist, "d01"), GRID, 54, START)
    assert np.asarray(history_fields(root, 4, 15, 17)["SPFORCC3"]).shape == (16, 18)
    no_spp = initial_stochastic(bind_stochastic({"stoch": {"skebs": 1}}, "d02"), child_grid, 18, START)
    assert not set(SPECTRAL_FIELDS) & set(history_fields(no_spp, 4, 9, 9))
