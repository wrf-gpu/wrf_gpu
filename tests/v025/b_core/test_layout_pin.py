"""GPUWRF_LAYOUT_PIN (b-core P1, default off): row-major layout constraints at dycore<->physics seams.

Layout only: values must be bitwise unchanged (compiled, E95); every pinned seam must emit its
LayoutConstraint custom calls when its part is on (deletion-sensitive) and none when it is off
(the OFF trace is the unpinned program)."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.dynamics import flux_advection as fa
from gpuwrf.kernels import layout_pin
from gpuwrf.runtime import operational_mode as om


def _field(shape, seed=0, lo=0.5, hi=1.5):
    return jnp.asarray(np.random.default_rng(seed).uniform(lo, hi, shape).astype(np.float32))


def _constraints(fn, *args):
    return jax.jit(fn).lower(*args).as_text().count("LayoutConstraint")


def _both(monkeypatch, parts, fn, *args):
    """(constraint count, compiled outputs) with ``parts`` enabled, then with the flag off."""
    out = {}
    for key, enabled in (("on", frozenset(parts)), ("off", frozenset())):
        monkeypatch.setattr(layout_pin, "PARTS", enabled)
        jax.clear_caches()
        out[key] = (_constraints(fn, *args), np.asarray(jax.jit(fn)(*args)))
    return out


def test_parse_parts():
    assert layout_pin.parse_parts("") == layout_pin.parse_parts("0") == ()
    assert layout_pin.parse_parts("1") == ("ac", "carry", "cols", "cum", "nested")
    assert layout_pin.parse_parts("cols_ac") == layout_pin.parse_parts("ac_cols") == ("ac", "cols")


def test_parts_cheap_key_is_process_independent():
    # PARTS is in IMPORT_TIME_ENV_CONSTANTS: its digest must not depend on the string-hash seed (a frozenset
    # did: every GPUWRF_LAYOUT_PIN=1 process missed the AOT cache, b-core P2G).
    import os, subprocess, sys
    import gpuwrf
    src = os.path.dirname(os.path.dirname(os.path.abspath(gpuwrf.__file__)))
    pythonpath = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p)
    code = ("from gpuwrf.kernels import layout_pin as L; from gpuwrf.runtime.aot_cheap_key import canonical_digest; "
            "print(canonical_digest(('module-const-env', {'p': L.parse_parts('1')})))")
    digests = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed, "JAX_PLATFORMS": "cpu",
                                   "PYTHONPATH": pythonpath}).stdout.strip()
               for seed in ("1", "2", "3")}
    assert len(digests) == 1


def test_pin_is_identity_when_part_disabled(monkeypatch):
    x = _field((4, 3, 5))
    monkeypatch.setattr(layout_pin, "PARTS", frozenset(("cum",)))
    assert layout_pin.pin("cols", x) is x
    tree = {"a": x, "b": (x, 1.0)}
    assert layout_pin.pin("ac", tree) is tree


def test_column_views_pinned_and_bitwise(monkeypatch):
    x = _field((6, 4, 5))
    r = _both(monkeypatch, ("cols",), pc._to_columns, x)
    assert r["on"][0] == 2 and r["off"][0] == 0
    np.testing.assert_array_equal(r["on"][1], r["off"][1])
    np.testing.assert_array_equal(r["on"][1], np.moveaxis(np.asarray(x), 0, -1))
    c = jnp.moveaxis(x, 0, -1)
    r = _both(monkeypatch, ("cols",), pc._from_columns, c)
    assert r["on"][0] == 2 and r["off"][0] == 0
    np.testing.assert_array_equal(r["on"][1], np.asarray(x))
    # parts are independent: the cumsum part does not pin column views
    monkeypatch.setattr(layout_pin, "PARTS", frozenset(("cum", "ac")))
    jax.clear_caches()
    assert _constraints(pc._to_columns, x) == 0


def test_stage_omega_cumsum_pinned_and_bitwise(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "0")  # XLA cumsum path (not the omega column kernel)
    nz, ny, nx = 6, 4, 5
    u, v, mu = _field((nz, ny, nx + 1), 1), _field((nz, ny + 1, nx), 2), _field((ny, nx), 3, 9e4, 1e5)
    kw = dict(c1h=_field((nz,), 4), c2h=_field((nz,), 5), dnw=-_field((nz,), 6, 0.01, 0.05), rdx=1 / 3000.0, rdy=1 / 3000.0,
              msfuy=_field((ny, nx + 1), 7, 0.9, 1.1), msfvx=_field((ny + 1, nx), 8, 0.9, 1.1), msftx=_field((ny, nx), 9, 0.9, 1.1))
    r = _both(monkeypatch, ("cum",), lambda a, b, m: fa.stage_omega_specified(a, b, m, **kw), u, v, mu)
    assert r["on"][0] == 2 and r["off"][0] == 0
    np.testing.assert_array_equal(r["on"][1], r["off"][1])


def test_nested_scope_pins_only_nested_child_steps(monkeypatch):
    from gpuwrf.kernels.ring_select import nested_step
    x = _field((6, 4, 5))
    monkeypatch.setattr(layout_pin, "PARTS", frozenset(("cols", "nested")))
    jax.clear_caches()
    assert _constraints(pc._to_columns, x) == 0  # root (d01) step: unpinned
    for nested, expected in ((False, 0), (True, 2)):
        jax.clear_caches()
        with nested_step(nested):
            assert _constraints(pc._to_columns, x) == expected
    with nested_step(True):
        np.testing.assert_array_equal(np.asarray(jax.jit(pc._to_columns)(x)), np.moveaxis(np.asarray(x), 0, -1))


def test_acoustic_scan_payload_seams_present():
    # The acoustic scan seed, body input and body output are the three "ac" seams (whole-scan
    # lowering needs a full carry; the deviceless d02 gate measures the effect).
    src = inspect.getsource(om._acoustic_scan)
    assert src.count('_layout_pin("ac", ') == 5  # seed, body in/out, scan constants (template, coefficients)
    assert 'seed_value = _layout_pin("ac", encode_acoustic(acoustic))' in src
    assert 'encoded_next = _layout_pin("ac", encode_acoustic(next_acoustic))' in src


def test_step_loop_carry_seams_present():
    src = inspect.getsource(om._advance_chunk_fori)
    assert src.count('_layout_pin("carry", ') == 2
