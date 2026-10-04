"""The split W phase aliases only the pointwise temperature-average output."""
import jax
import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig

from gpuwrf.kernels import dyn_acoustic_fp32 as da


@pytest.mark.parametrize("flag, expected", [("1", ((1, 0),)), ("0", ())])
def test_pointwise_w_alias_dispatch(monkeypatch, flag, expected):
    monkeypatch.setenv("GPUWRF_ACOUSTIC_ALIAS", flag)
    shape = jax.ShapeDtypeStruct((4, 3, 5), jnp.float32)

    def kernel(inputs, outputs):
        outputs["t_2ave"][...] = inputs["t_2ave"][...]
        outputs["rhs"][...] = inputs["rhs"][...]

    def call(rhs, average):
        return da._call(
            kernel, {"rhs": rhs, "t_2ave": average},
            {"t_2ave": shape, "rhs": shape}, (1,),
            name="b_core_w_p1_fp32", interpret=True,
        )

    equations = jax.make_jaxpr(call)(shape, shape).jaxpr.eqns
    calls = [e for e in equations if e.primitive.name == "pallas_call"]
    assert len(calls) == 1
    assert calls[0].params["input_output_aliases"] == expected


def test_recurrent_w_is_not_pointwise_aliased():
    # W2PASS=0 preserves old ring values after its stores; W2PASS=3's column
    # recurrence has already consumed t_2ave in the separate pointwise phase.
    assert "b_core_w_fp32" not in da._POINTWISE_ALIASES


def test_split_w_does_not_keep_old_average_live_in_part2(monkeypatch):
    old = jnp.zeros((4, 3, 5), jnp.float32)
    new = jnp.ones_like(old)
    face = jnp.zeros((5, 3, 5), jnp.float32)
    state = SimpleNamespace(t_2ave=old, w=face, w_save=face,
                            replace=lambda **updates: updates)
    args = dict.fromkeys(("a", "alpha", "gamma", "w_save", "ph_1", "phb", "w", "ph",
                          "c1f", "c2f", "mut", "muts", "msfty", "s"), face)
    args["t_2ave"] = old
    calls = []

    def call(kernel, inputs, outputs, grid, *, name, **kwargs):
        calls.append(name)
        if name == "b_core_w_p1_fp32":
            return {"t_2ave": new, "rhs": face}
        if name == "b_core_w_p2_fp32":
            assert "t_2ave" not in inputs
            assert inputs["t2a"] is new
            return {"wupd": face}
        return {"w": face, "ph": face}

    monkeypatch.setattr(da, "_call", call)
    result = da._w_phase_split(state, args, AcousticCoreConfig(dt=1., dx=1., dy=1.),
                               nz=4, ny=3, nx=5, interpret=True)
    assert calls == ["b_core_w_p1_fp32", "b_core_w_p2_fp32", "b_core_w_fp32"]
    assert result["t_2ave"] is new
    assert args["t_2ave"] is old  # P1's input table retains its original value.
