"""AOT cheap key is value-blind for stochastic members (critic FIX v50-stoch finding 10).

nens, seeds, pattern scales and the host-only nest_init_rounds only change initial carry values; the
traced step reads the static switch layout alone (StochasticConfig._layout, the holder's own hash/eq).
The persistent key used to hash the whole holder, so every ensemble member split the AOT cache.
"""
from dataclasses import replace

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.io.stochastic_config import bind_stochastic, with_executed_d01_children
from gpuwrf.runtime.aot_cheap_key import static_config_hash
from test_noahmp_checkpoint_v2 import _namelist

BASE = dict(skebs=1, sppt=1, spp=1, nens=1)


def _key(nl, stoch, **edit):
    cfg = bind_stochastic({"stoch": {**BASE, **stoch}}, "d01")
    if edit:
        cfg = replace(cfg, **edit)
    return static_config_hash(replace(nl, stochastic_config=cfg))


def test_member_controls_share_the_static_key_and_layout_still_splits_it():
    nl = _namelist(GridSpec.canary_3km_template())
    ref = _key(nl, {})
    assert _key(nl, {"nens": 2}) == ref
    assert _key(nl, {"iseed_sppt": 99, "iseed_skebs": 7}) == ref
    assert _key(nl, {"gridpt_stddev_sppt": .3, "lengthscale_sppt": 300000.}) == ref
    cfg = bind_stochastic({"stoch": BASE}, "d01")
    assert static_config_hash(replace(nl, stochastic_config=with_executed_d01_children(cfg, "d01", ("d02",)))) == ref
    # Layout switches select a different step program and must keep splitting the key.
    assert _key(nl, {"sppt_vertstruc": 1}) != ref
    assert _key(nl, {"spp": 0}) != ref
    assert static_config_hash(nl) != ref
