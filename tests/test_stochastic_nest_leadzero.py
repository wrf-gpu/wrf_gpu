"""Child lead-zero stochastic history at original-WRF phase (critic FIX v50-stoch finding 2).

Original WRF writes a child's first history after the parent's first solve and force-down
(frame/module_integrate.F:375,393-430): RAND-off WRF 4.7.1 dmpar on WN3 0227 writes d02/d03 lead-zero
max|RSTOCH| 0.0495/0.0258 while d01 is 0 (REF_PACKAGE_V2/SMOKE_PROOF.json). The port used to write the
zero-initialized child carry.
"""
from dataclasses import dataclass, replace
from datetime import datetime
from types import SimpleNamespace

import jax
import numpy as np

from gpuwrf.integration.nested_pipeline import _emit_initial_history_frames
from gpuwrf.io.stochastic_config import bind_stochastic
from gpuwrf.nesting.boundary_construction import build_nest_force_weights
from gpuwrf.nesting.stochastic import force_stochastic
from gpuwrf.physics.stochastic_driver import advance_stochastic, initial_stochastic

START = datetime(2026, 2, 28)
CONTROL = dict(skebs=1, sppt=1, spp=1, rand_perturb=0, nens=1,
               **{"lengthscale_" + s: 15000. for s in ("sppt", "spp_conv", "spp_pbl", "spp_lsm")})
GRIDS = {"d01": SimpleNamespace(nx=17, ny=15, nz=4, dx=9000., dy=9000.),
         "d02": SimpleNamespace(nx=10, ny=10, nz=4, dx=3000., dy=3000.),
         "d03": SimpleNamespace(nx=7, ny=7, nz=4, dx=1000., dy=1000.)}
DT = {"d01": 54, "d02": 18, "d03": 6}


@dataclass(frozen=True)
class Carry:
    stochastic: object
    tag: str

    def replace(self, **kw):
        return replace(self, **kw)


def _setup():
    configs = {d: bind_stochastic({"stoch": dict(CONTROL)}, d) for d in GRIDS}
    carries = {d: Carry(initial_stochastic(configs[d], GRIDS[d], DT[d], START), d) for d in GRIDS}
    bundles = {d: SimpleNamespace(namelist=SimpleNamespace(grid=GRIDS[d], stochastic_config=configs[d])) for d in GRIDS}
    w12 = build_nest_force_weights(parent_grid_ratio=3, i_parent_start=5, j_parent_start=4,
                                   parent_grid=GRIDS["d01"], child_grid=GRIDS["d02"])
    w23 = build_nest_force_weights(parent_grid_ratio=3, i_parent_start=3, j_parent_start=3,
                                   parent_grid=GRIDS["d02"], child_grid=GRIDS["d03"])
    edge = lambda p, c, w: SimpleNamespace(parent=p, child=c, weights=w, parent_grid_ratio=3)
    tree = SimpleNamespace(edges={"d01": (edge("d01", "d02", w12),), "d02": (edge("d02", "d03", w23),)})
    return configs, carries, bundles, tree, w12, w23


def _leaves(stoch):
    return [np.asarray(x) for x in jax.tree.leaves(stoch.patterns)]


def test_child_lead_zero_frames_carry_parent_first_step_force_down():
    configs, carries, bundles, tree, w12, w23 = _setup()
    before = {d: _leaves(c.stochastic) for d, c in carries.items()}
    frames = dict(_emit_initial_history_frames(lambda name, step, carry: (name, carry), tuple(GRIDS), carries,
                                               enabled=True, tree=tree, bundles=bundles))
    # Independent recomputation of the original-WRF phase.
    d01_step1 = advance_stochastic(carries["d01"].stochastic, configs["d01"], dx=9000., dy=9000.)
    d02_t0 = force_stochastic(carries["d02"].stochastic, d01_step1, w12, 3)
    d02_step1 = advance_stochastic(d02_t0, configs["d02"], dx=3000., dy=3000.)
    d03_t0 = force_stochastic(carries["d03"].stochastic, d02_step1, w23, 3)
    for got, want in ((frames["d01"].stochastic, carries["d01"].stochastic),
                      (frames["d02"].stochastic, d02_t0), (frames["d03"].stochastic, d03_t0)):
        for a, b in zip(_leaves(got), _leaves(want)):
            # jitted (pipeline) vs eager (reference) rounding: the pre-registered pattern bound.
            scale = float(np.max(np.abs(b))) if b.size else 0.
            np.testing.assert_allclose(a, b, rtol=8e-6, atol=64 * np.finfo(np.float32).eps * scale)
    rstoch = lambda s: float(np.max(np.abs(np.asarray(s.patterns[2].field))))
    assert rstoch(frames["d01"].stochastic) == 0.0                        # d01 writes before its first solve
    assert rstoch(frames["d02"].stochastic) > 0 and rstoch(frames["d03"].stochastic) > 0
    # The numerical carries themselves are untouched (only the lead-zero frame view changes).
    for d, c in carries.items():
        for a, b in zip(before[d], _leaves(c.stochastic)):
            np.testing.assert_array_equal(a, b)


def test_without_stochastic_carries_the_frames_are_the_initial_carries():
    carries = {d: Carry(None, d) for d in GRIDS}
    _, _, bundles, tree, _, _ = _setup()
    frames = dict(_emit_initial_history_frames(lambda name, step, carry: (name, carry), tuple(GRIDS), carries,
                                               enabled=True, tree=tree, bundles=bundles))
    assert all(frames[d] is carries[d] for d in GRIDS)
