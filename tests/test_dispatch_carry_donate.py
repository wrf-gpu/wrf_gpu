"""CPU gates for GPUWRF_CARRY_DONATE (dispatch D10 donation candidate).

The flag is resolved at import: OFF keeps the plain ``_advance_chunk_fori`` jit
(no input/output aliasing); ON donates the carry (aliased outputs). The resolved
value is registered in ``aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS`` so ON and OFF
can never share a cheap key.

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_dispatch_carry_donate.py``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")

_SRC = str(Path(__file__).resolve().parents[1] / "src")

_CHILD = r"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
assert jax.default_backend() == "cpu"
from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime import operational_mode as om
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes

grid = GridSpec.canary_3km_template()
shapes = _state_field_shapes(grid)
sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}


def build():
    # Fresh State per call: a donating check deletes the buffers it was given.
    state = State(**{f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f)) for f, s in shapes.items()})
    tend = Tendencies(**{k: jnp.zeros(shapes[sk.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
                         for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")})
    nl = om.OperationalNamelist(grid=grid, tendencies=tend, metrics=grid.metrics, dt_s=10.0,
                                acoustic_substeps=6, time_utc="2024-09-01_00:00:00")
    return om._initial_carry_for_run(state, nl), nl


carry, nl = build()
text = om._advance_chunk_fori.lower(carry, nl, jnp.asarray(1, jnp.int32), om.build_clock_base(nl),
                                     n_steps=1, cadence=1).as_text()
def _repeats(tree):
    ids = [id(x) for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]
    return len(ids) - len(set(ids))
def _donate_ok(tree):
    probe = jax.jit(lambda c: jax.tree.map(lambda x: x + 0, c), donate_argnums=(0,))
    try:
        jax.block_until_ready(probe(tree))
        return True
    except Exception as exc:  # noqa: BLE001
        assert "donate the same buffer twice" in str(exc), exc
        return False
unaliased = om.unalias_donated_carry(carry)
same = [a is b for a, b in zip(jax.tree_util.tree_leaves(carry), jax.tree_util.tree_leaves(unaliased))]
equal = all(np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(jax.tree_util.tree_leaves(carry), jax.tree_util.tree_leaves(unaliased)))
repeats_initial, repeats_unaliased = _repeats(carry), _repeats(unaliased)
donate_ok_unaliased = _donate_ok(unaliased)
import inspect
from gpuwrf.runtime import domain_tree as dt
# A device_put of an uncommitted array is a committed ALIAS: a distinct object on the same buffer.
def aliased_carry():
    flat, tdef = jax.tree_util.tree_flatten(om.unalias_donated_carry(build()[0]))
    if not om._CARRY_DONATE:  # OFF: unalias is identity, take distinct buffers first
        flat = [jnp.copy(x) if isinstance(x, jax.Array) else x for x in flat]
    first = next(i for i, x in enumerate(flat) if isinstance(x, jax.Array) and x.size > 0)
    last = max(i for i, x in enumerate(flat) if isinstance(x, jax.Array) and x.shape == flat[first].shape
               and x.dtype == flat[first].dtype and i != first)
    flat[last] = jax.device_put(flat[first], jax.devices()[0])
    return jax.tree_util.tree_unflatten(tdef, flat), flat[last] is not flat[first]
alias_by_id, distinct = aliased_carry()
alias_by_buffer, _ = aliased_carry()
shared = om.unalias_donated_carry(build()[0])
seen = set()
two = {"a": om.unalias_donated_carry(shared, seen=seen, by_buffer=True),
       "b": om.unalias_donated_carry(shared, seen=seen, by_buffer=True)}
# review-b: _advance_chunk is the entry every runner shares (single-domain loops at the
# operational runners, tree fallbacks). Route a REAL donating jit through it.
def through_entry(tree):
    received = []
    def donating_fori(c, nl_, start, clock, *, n_steps, cadence):
        received.append(c)
        return _donate_ok(c)
    real, om._advance_chunk_fori = om._advance_chunk_fori, donating_fori
    try:
        ok = om._advance_chunk(tree, nl, jnp.asarray(1, jnp.int32), None, n_steps=1, cadence=1)
    finally:
        om._advance_chunk_fori = real
    return {"donate_ok": ok, "passed_through": received[0] is tree}
entry = {"same_object": through_entry(build()[0]), "buffer_alias": through_entry(aliased_carry()[0]),
         "clean": through_entry(om.unalias_donated_carry(aliased_carry()[0], by_buffer=True))}
if not om._CARRY_DONATE:  # OFF: unalias is identity -> give the clean case distinct buffers itself
    flat, tdef = jax.tree_util.tree_flatten(build()[0])
    entry["clean"] = through_entry(jax.tree_util.tree_unflatten(
        tdef, [jnp.copy(x) if isinstance(x, jax.Array) else x for x in flat]))
# Inside a trace (vmapped batch lanes) a repeated tracer is not a buffer: no copy op.
traced_copies = len(jax.make_jaxpr(lambda x: om.unalias_donated_carry({"a": x, "b": x}))(jnp.ones(3)).eqns)
print(json.dumps({
    "entry": entry,
    "traced_copies": traced_copies,
    "alias_distinct_object": distinct,
    "donate_ok_alias_by_id": _donate_ok(om.unalias_donated_carry(alias_by_id)),
    "donate_ok_alias_by_buffer": _donate_ok(om.unalias_donated_carry(alias_by_buffer, by_buffer=True)),
    "cross_domain_repeat_copied": _repeats(two) == 0 and _donate_ok(two),
    "entry_wired": "unalias_donated_carry(carry, seen=_entering, by_buffer=True)" in inspect.getsource(dt.run_domain_tree_callbacks),
    "repeats_initial": repeats_initial,
    "repeats_unaliased": repeats_unaliased,
    "kept_identical": sum(same),
    "values_equal": equal,
    "donate_ok_unaliased": donate_ok_unaliased,
    "advance_wired": "carry = unalias_donated_carry(carry)" in inspect.getsource(dt._operational_advance_factory),
    "flag": om._CARRY_DONATE,
    "aliased_args": text.count("tf.aliasing_output"),
    "carry_array_leaves": sum(1 for x in jax.tree_util.tree_leaves(carry) if hasattr(x, "shape")),
    "const_hash": ck.module_const_env_hash(),
    "registered": ("gpuwrf.runtime.operational_mode", "_CARRY_DONATE") in ck.IMPORT_TIME_ENV_CONSTANTS,
}))
"""


def _child(donate: str | None) -> dict:
    env = dict(os.environ)
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", GPUWRF_JAX_CACHE="0")
    env.pop("GPUWRF_CARRY_DONATE", None)
    env.pop("GPUWRF_ADVANCE_CHUNK_LOOP", None)
    if donate is not None:
        env["GPUWRF_CARRY_DONATE"] = donate
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_SRC, env.get("PYTHONPATH")) if p)
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def off():
    return _child("0")  # explicit: the release defaults turn donation on


@pytest.fixture(scope="module")
def on():
    return _child("1")


def test_off_keeps_plain_jit_without_aliasing(off):
    assert off["flag"] is False
    assert off["aliased_args"] == 0


def test_on_donates_the_carry(on):
    assert on["flag"] is True
    # Every aliasable carry buffer is donated (outputs may alias at most one input each).
    assert 0 < on["aliased_args"] <= on["carry_array_leaves"], on


def test_flag_is_registered_and_splits_the_key(on, off):
    assert on["registered"] and off["registered"]
    assert on["const_hash"] != off["const_hash"]


def test_initial_carry_repeats_are_copied_once_when_donating(on, off):
    # The initial carry holds some State arrays at two positions (D13: "Attempt to
    # donate the same buffer twice"); ON copies each repeat, OFF passes it through.
    assert on["repeats_initial"] > 0 and off["repeats_initial"] == on["repeats_initial"]
    assert on["repeats_unaliased"] == 0 and on["values_equal"]
    assert on["kept_identical"] == on["carry_array_leaves"] - on["repeats_initial"]
    assert on["donate_ok_unaliased"] is True
    assert off["repeats_unaliased"] == off["repeats_initial"] and off["donate_ok_unaliased"] is False
    assert on["advance_wired"]


def test_shared_buffers_are_split_where_carries_enter_a_run(on, off):
    # D13 run 2: distinct arrays on ONE buffer (device_put alias) still failed the
    # id check; the run entry compares buffer pointers across all domains' carries.
    assert on["alias_distinct_object"]
    assert on["donate_ok_alias_by_id"] is False and on["donate_ok_alias_by_buffer"] is True
    assert on["cross_domain_repeat_copied"] is True
    assert on["entry_wired"] and off["entry_wired"]


def test_advance_chunk_entry_splits_repeats_for_every_runner(on, off):
    # review-b: the single-domain/public runners call _advance_chunk directly; a
    # repeated object AND a distinct alias of one buffer must both reach the
    # donating jit with one owner per buffer. OFF passes the carry through untouched.
    for case in ("same_object", "buffer_alias"):
        assert on["entry"][case] == {"donate_ok": True, "passed_through": False}, (case, on["entry"])
        assert off["entry"][case] == {"donate_ok": False, "passed_through": True}, (case, off["entry"])
    assert on["entry"]["clean"] == off["entry"]["clean"] == {"donate_ok": True, "passed_through": True}
    assert on["traced_copies"] == 0 and off["traced_copies"] == 0


def _resolved_flag(**extra: str) -> bool:
    env = dict(os.environ)
    env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", GPUWRF_JAX_CACHE="0", **extra)
    env.pop("GPUWRF_CARRY_DONATE", None)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_SRC, env.get("PYTHONPATH")) if p)
    code = "import gpuwrf\nfrom gpuwrf.runtime import operational_mode as om\nprint(om._CARRY_DONATE)"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout.strip().splitlines()[-1] == "True"


def test_release_defaults_turn_donation_on_and_legacy_keeps_it_off():
    assert _resolved_flag(GPUWRF_FAST_DEFAULTS="1") is True
    assert _resolved_flag(GPUWRF_FAST_DEFAULTS="0") is False
