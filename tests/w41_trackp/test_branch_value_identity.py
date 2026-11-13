"""Checkpoint-branch series: control VALUES are declared digests; family/name/shape changes stay refused.

The restart identity binds two views of the controls: the per-domain value digest (``surface_controls``,
declarable in a branch manifest) and the static namelist digest (``namelists``), which sees only the
content-blind ``SurfaceControlsAux`` signature. A value-only variant therefore needs exactly the declared
digest; enabling the family on another domain or changing names is an undeclared ``namelists`` change,
even when its value digest is declared. CPU structural evidence, not a forecast result.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from w41_trackp_fixtures import _fixture, _namelist, pytestmark  # noqa: F401
from gpuwrf.runtime.aot_cheap_key import canonical_digest, static_config_hash
from gpuwrf.runtime.checkpoint_branch import branch_identity
from gpuwrf.runtime.restart_parameters import static_namelist
from gpuwrf.surface_params import controls as sc


def _base():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    return OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18., mp_physics=0, ra_sw_physics=0,
                               ra_lw_physics=0, sf_sfclay_physics=0, cu_physics=0,
                               time_utc=datetime(2026, 6, 1, tzinfo=timezone.utc))


def _identity(namelists, tmp, out):
    """The two control-related restart identity entries, as nested_identity builds them."""
    return {"output_dir": str(tmp / out), "source_sha256": "s",
            "namelists": {d: canonical_digest(static_namelist(nl)) for d, nl in namelists.items()},
            "surface_controls": {d: sc.restart_controls_digest(nl) for d, nl in namelists.items()}}


@pytest.fixture
def setup(tmp_path):
    _land, _forcing, static = _fixture()
    nl = _namelist(static)
    base = _base()
    controls = {f: sc.expand_controls(nl, {"z0mvt": f}) for f in (1.0, 2.0)}
    with_names = sc.expand_controls(nl, {"z0mvt": 1.0, "laim": 1.0})
    bound = lambda c: dataclasses.replace(base, surface_controls=c)  # noqa: E731
    parent = {"d01": base, "d02": bound(controls[1.0])}  # d01 ran without controls, d02 neutral
    generation = tmp_path / "ckpt" / "generation-1"
    generation.mkdir(parents=True)
    manifest = {"branch_schema_version": 1, "identity": _identity(parent, tmp_path, "parent"), "sha256": "p",
                "domains": ["d01", "d02"], "own_steps": {"d01": 1, "d02": 3}}
    (generation / "manifest.json").write_text("{}")
    return dict(tmp=tmp_path, base=base, bound=bound, controls=controls, with_names=with_names,
                manifest=manifest, generation=generation)


def test_value_only_variant_shares_static_identity_and_needs_its_digest(setup):
    s = setup
    child = {"d01": s["base"], "d02": s["bound"](s["controls"][2.0])}
    current = _identity(child, s["tmp"], "child")
    parent = s["manifest"]["identity"]
    assert current["namelists"] == parent["namelists"]  # values never reach the static/compile view
    assert static_config_hash(child["d02"]) == static_config_hash(s["bound"](s["controls"][1.0]))
    assert current["surface_controls"]["d02"] != parent["surface_controls"]["d02"]
    declared = {"schema_version": 1, "parameters": {"surface_controls": {"d02": current["surface_controls"]["d02"]}}}
    assert branch_identity(s["manifest"], current, declared, s["generation"])["surface_controls"] == current["surface_controls"]
    with pytest.raises(ValueError, match=r"undeclared branch identity change: \['surface_controls'\]"):
        branch_identity(s["manifest"], current, {"schema_version": 1}, s["generation"])


@pytest.mark.parametrize("change", ["enable_on_new_domain", "add_control_name"])
def test_family_or_name_change_is_refused_even_when_value_digest_is_declared(setup, change):
    s = setup
    if change == "enable_on_new_domain":
        child = {"d01": s["bound"](s["controls"][1.0]), "d02": s["bound"](s["controls"][1.0])}
        domain = "d01"
    else:
        child = {"d01": s["base"], "d02": s["bound"](s["with_names"])}
        domain = "d02"
    current = _identity(child, s["tmp"], "child")
    declared = {"schema_version": 1, "parameters": {"surface_controls": {domain: current["surface_controls"][domain]}}}
    with pytest.raises(ValueError, match=r"undeclared branch identity change: \['namelists'\]"):
        branch_identity(s["manifest"], current, declared, s["generation"])
