"""CPU IDENTITY-PROOF for the cheap-key AOT manifest -- the falsification engine.

The cheap-key manifest (``gpuwrf.runtime.aot_cheap_key``) lets the eager Step-C
loop locate a serialized ``_advance_chunk_fori`` executable by a metadata-only
hash WITHOUT lowering (the ~30-54 min wall the AOT preview otherwise still paid).

THE LOAD-BEARING RISK = KEY COMPLETENESS: if ``cheap_key`` misses ANY input baked
into the lowered HLO, two distinct executables can share a key -> wrong blob loads
-> SILENT WRONG RESULT. This test PROVES, on CPU, with NO GPU, that the map
``cheap_key -> hlo_sha256`` is a function (INJECTIVE: one key never maps to two
HLOs) across a determinant matrix, and that a deliberately-incomplete key is
CAUGHT (collision detected). It lowers the REAL ``_advance_chunk_fori`` (feasible
on CPU in ~5 s; we compare HLO *text*, never execute -- the XLA:CPU machine-type
SIGILL warning is benign because we never run the program).

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_aot_cheap_key.py``

Each lowering is ~190 MB of HLO text and a few seconds, so the matrix is kept
focused and every distinct config is lowered AT MOST ONCE (module-scoped cache).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gpuwrf.runtime import aot_cheap_key as ck

pytestmark = pytest.mark.filterwarnings("ignore")


# Fresh-process children must run THIS checkout's gpuwrf, never whatever an
# ambient PYTHONPATH points at (S3-A/B critic R1). The child's source root is
# pinned first on PYTHONPATH, and the child fails fast unless ``gpuwrf.__file__``
# resolves under it; the parent re-checks the reported origin. Children are
# CPU-only: an override of the backend env is refused before spawning.
_CANDIDATE_SRC = str(Path(__file__).resolve().parents[1] / "src")
_CHILD_ORIGIN_CHECK = """
import os, sys, jax, gpuwrf
assert jax.default_backend() == "cpu", jax.default_backend()
_GPUWRF_FILE = os.path.realpath(gpuwrf.__file__)
assert _GPUWRF_FILE.startswith(sys.argv[2] + os.sep), (
    f"child imported gpuwrf from {_GPUWRF_FILE}, not {sys.argv[2]}")
"""


def _run_pinned_child(code: str, payload: dict, env_overrides=None, src_root=_CANDIDATE_SRC) -> dict:
    """Run ``code`` in a fresh CPU process on ``src_root`` (the candidate tree); return its JSON."""
    import json
    import subprocess
    import sys

    env = dict(os.environ)
    env.update(env_overrides or {})
    for k, v in (("JAX_PLATFORMS", "cpu"), ("CUDA_VISIBLE_DEVICES", "")):
        assert (env_overrides or {}).get(k, v) == v, f"fresh-process child must stay CPU-only: {k}"
        env[k] = v
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (src_root, env.get("PYTHONPATH")) if p)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_ORIGIN_CHECK + code, json.dumps(payload),
         src_root],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["gpuwrf_file"].startswith(src_root + os.sep), out["gpuwrf_file"]
    print(f"child gpuwrf origin: {out['gpuwrf_file']}")
    return out


@pytest.mark.parametrize("override", [{"JAX_PLATFORMS": "cuda"}, {"CUDA_VISIBLE_DEVICES": "0"}])
def test_pinned_child_refuses_non_cpu_backend_override(override):
    with pytest.raises(AssertionError, match="CPU-only"):
        _run_pinned_child("raise SystemExit(0)", {}, override)


# --------------------------------------------------------------------------- #
# Fixture builders (real GridSpec + namelist + carry; CPU-constructible).
# --------------------------------------------------------------------------- #
def _build_call(nl_kwargs=None, time_utc="2024-09-01_00:00:00"):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
    from gpuwrf.runtime.operational_mode import (
        OperationalNamelist,
        _initial_carry_for_run,
        build_clock_base,
    )

    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid)
    fields = {
        f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f))
        for f, s in shapes.items()
    }
    state = State(**fields)
    sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tend = Tendencies(
        **{
            k: jnp.zeros(shapes[sk.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
            for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
        }
    )
    kw = dict(
        grid=grid,
        tendencies=tend,
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=6,
        time_utc=time_utc,
    )
    if nl_kwargs:
        kw.update(nl_kwargs)
    namelist = OperationalNamelist(**kw)
    carry = _initial_carry_for_run(state, namelist)
    clock_base = build_clock_base(namelist)
    return carry, namelist, clock_base


def _real_hlo_sha256(carry, namelist, clock_base, *, n_steps=1, cadence=1):
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    lowered = _advance_chunk_fori.lower(
        carry,
        namelist,
        jnp.asarray(1, dtype=jnp.int32),
        clock_base,
        n_steps=int(n_steps),
        cadence=int(cadence),
    )
    return hashlib.sha256(lowered.as_text().encode("utf-8")).hexdigest()


def _cheap_key(carry, namelist, clock_base, *, n_steps=1, cadence=1):
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    return ck.cheap_key(
        _advance_chunk_fori,
        (carry, namelist, jnp.asarray(1, dtype=jnp.int32), clock_base),
        {"n_steps": int(n_steps), "cadence": int(cadence)},
        namelist,
    )


# A trace-time GPUWRF_* env knob is read INSIDE the jitted body, but JAX caches a
# lowering in-process by its (avals, statics) key -- which (by design, the bug we
# fix) does NOT include the env. So a second in-process ``.lower()`` after an env
# change returns the STALE cached HLO. To observe the env's true effect on the
# HLO (and on the cheap_key), the env-axis cells must lower in a FRESH process.
def _lower_in_subprocess(
    env_overrides: dict[str, str], nl_kwargs=None, call_kwargs=None, *, src_root=_CANDIDATE_SRC
):
    """Return (cheap_key, hlo_sha256) computed in a FRESH process under ``env``.

    Fresh process => empty JAX lowering cache => the env knobs are read at trace
    time and their true effect on both the HLO and the cheap_key is observed."""
    out = _lower_record_in_subprocess(env_overrides, nl_kwargs, call_kwargs, src_root=src_root)
    return out["key"], out["hlo"], out["incomplete_key"]


def _lower_record_in_subprocess(
    env_overrides: dict[str, str],
    nl_kwargs=None,
    call_kwargs=None,
    *,
    bake_init_date_gases: bool = False,
    src_root: str = _CANDIDATE_SRC,
) -> dict:
    """Fresh-process lowering record: key, hlo, incomplete_key, clock digest.

    ``bake_init_date_gases`` is the deliberate S3 mutation: it restores the
    pre-S3 program, where the host interpolated the CLWRF gases for the init
    date at trace time and baked them into the step HLO."""
    import textwrap

    payload = {
        "env": env_overrides,
        "nl_kwargs": nl_kwargs or {},
        "call_kwargs": call_kwargs or {},
        "bake_init_date_gases": bool(bake_init_date_gases),
    }
    code = textwrap.dedent(
        """
        import os, sys, json, hashlib
        import numpy as np, jax, jax.numpy as jnp
        from gpuwrf.contracts.grid import GridSpec
        from gpuwrf.contracts.precision import DEFAULT_DTYPES
        from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
        from gpuwrf.runtime.operational_mode import (
            OperationalNamelist, _initial_carry_for_run, build_clock_base,
            _advance_chunk_fori,
        )
        from gpuwrf.runtime import aot_cheap_key as ck
        spec = json.loads(sys.argv[1])
        nl_kwargs = dict(spec["nl_kwargs"])
        call_kwargs = dict(spec["call_kwargs"])
        time_utc = nl_kwargs.pop("time_utc", "2024-09-01_00:00:00")
        grid = GridSpec.canary_3km_template()
        shapes = _state_field_shapes(grid)
        fields = {f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f))
                  for f, s in shapes.items()}
        state = State(**fields)
        sk = {"p":"p_total","ph":"ph_total","mu":"mu_total"}
        tend = Tendencies(**{k: jnp.zeros(shapes[sk.get(k,k)],
                             dtype=DEFAULT_DTYPES.dtype_for(k))
                             for k in ("u","v","w","theta","qv","p","ph","mu")})
        kw = dict(grid=grid, tendencies=tend, metrics=grid.metrics,
                  dt_s=10.0, acoustic_substeps=6, time_utc=time_utc)
        kw.update(nl_kwargs)
        nl = OperationalNamelist(**kw)
        carry = _initial_carry_for_run(state, nl)
        cb = build_clock_base(nl)
        if spec["bake_init_date_gases"]:
            from gpuwrf.coupling import physics_couplers as pc
            host_gases = pc.clwrf_ssp245_gases_for_time
            pc.clwrf_gases_at_lead = lambda clock, lead: host_gases(time_utc)
        n_steps = int(call_kwargs.get("n_steps", 1))
        cadence = int(call_kwargs.get("cadence", 1))
        key = ck.cheap_key(_advance_chunk_fori,
                           (carry, nl, jnp.asarray(1, jnp.int32), cb),
                           {"n_steps": n_steps, "cadence": cadence}, nl)
        # An env-BLIND key (component 5 / global_trace_env_hash omitted) for the
        # collision-detection proof.
        incomplete = ck.canonical_digest((
            ck.KEY_SCHEMA,
            ck.version_fingerprint_hash(),
            ck.fn_identity_hash(_advance_chunk_fori),
            ck.static_config_hash(nl),
            ck.carry_aval_hash(
                (carry, nl, jnp.asarray(1, jnp.int32), cb),
                {"n_steps": n_steps, "cadence": cadence}),
        ))
        low = _advance_chunk_fori.lower(carry, nl, jnp.asarray(1, jnp.int32), cb,
                                        n_steps=n_steps, cadence=cadence)
        hlo = hashlib.sha256(low.as_text().encode("utf-8")).hexdigest()
        clock = hashlib.sha256(b"".join(
            np.asarray(leaf).tobytes() for leaf in jax.tree_util.tree_leaves(cb)
        )).hexdigest()
        print(json.dumps({"key": key, "hlo": hlo, "incomplete_key": incomplete,
                          "clock": clock, "gpuwrf_file": _GPUWRF_FILE}))
        """
    )
    return _run_pinned_child(code, payload, env_overrides, src_root)


# The determinant matrix. Each cell is (name, build_kwargs_or_env, call_kwargs).
# `env` cells set a GPUWRF_* knob for the lowering (read at trace time); the
# fixture restores the env afterwards. Cells are chosen so SOME share an HLO
# (date, n_steps, an inert option under a zero state) and SOME differ (dt_s,
# epssm, the env knobs) -- both classes are needed to exercise injectivity AND
# the no-over-fragment direction.
_MATRIX = [
    ("base", {}, {}, {}),
    # NOTE: the DATE axis is NOT an in-process cell. The namelist treedef is
    # date-blind (#114), so an in-process lower of a second date returns the first
    # date's cached HLO and can never expose a baked date constant (v0.25 S3 F1).
    # Date cells lower in FRESH SUBPROCESSES in test_date_axis_* below.
    # n_steps / cadence vary -> traced int32 -> key + HLO IDENTICAL to base.
    ("nsteps9", {}, {"n_steps": 9, "cadence": 3}, {}),
    # dt_s changes a baked scalar -> key + HLO MUST DIFFER.
    ("dt18", {"dt_s": 18.0}, {}, {}),
    # epssm changes a baked acoustic coefficient -> key + HLO MUST DIFFER.
    ("epssm", {"epssm": 0.3}, {}, {}),
    # acoustic_substeps changes the loop bound -> key + HLO MUST DIFFER.
    ("subs10", {"acoustic_substeps": 10}, {}, {}),
    # NOTE: the ENV-AXIS cells (GPUWRF_MOIST_CQW / GPUWRF_ACOUSTIC_UNROLL) are NOT
    # in this in-process matrix -- JAX caches a lowering by avals+statics (which by
    # design excludes the env), so a second in-process lower after an env change
    # returns the STALE HLO. They are exercised via FRESH SUBPROCESSES in
    # test_env_axis_* below (which is also the more faithful trace-time scenario).
]


@pytest.fixture(scope="module")
def matrix_results():
    """Lower every matrix cell once; return [(name, cheap_key, hlo_sha256), ...]."""
    results = []
    for name, nl_kwargs, call_kwargs, _env in _MATRIX:
        nl_kwargs = dict(nl_kwargs)
        time_utc = nl_kwargs.pop("time_utc", "2024-09-01_00:00:00")
        carry, namelist, clock_base = _build_call(nl_kwargs, time_utc=time_utc)
        key = _cheap_key(carry, namelist, clock_base, **call_kwargs)
        hlo = _real_hlo_sha256(carry, namelist, clock_base, **call_kwargs)
        assert key is not None, f"cheap_key returned None for {name}"
        assert hlo is not None, f"lowering produced no HLO for {name}"
        results.append((name, key, hlo))
    return results


# --------------------------------------------------------------------------- #
# (1) INJECTIVITY -- catches a MISSED determinant = the silent-wrong bug.
# --------------------------------------------------------------------------- #
def test_cheap_key_is_injective_over_hlo(matrix_results):
    """One cheap_key must never map to two different HLOs (no silent wrong load)."""
    by_key: dict[str, set[str]] = {}
    names_by_key: dict[str, list[str]] = {}
    for name, key, hlo in matrix_results:
        by_key.setdefault(key, set()).add(hlo)
        names_by_key.setdefault(key, []).append(name)
    collisions = {
        key: (names_by_key[key], sorted(h[:12] for h in hlos))
        for key, hlos in by_key.items()
        if len(hlos) > 1
    }
    assert not collisions, (
        "CHEAP_KEY COLLISION (a determinant is missing from the key): "
        f"{collisions}"
    )


# --------------------------------------------------------------------------- #
# (2) date / n_steps / cadence are NOT determinants -> key AND HLO invariant.
# --------------------------------------------------------------------------- #
def test_nsteps_is_key_and_hlo_invariant(matrix_results):
    """base and nsteps9 must share ONE key AND ONE HLO (date: see test_date_axis_*)."""
    by_name = {name: (key, hlo) for name, key, hlo in matrix_results}
    base_key, base_hlo = by_name["base"]
    for name in ("nsteps9",):
        key, hlo = by_name[name]
        assert hlo == base_hlo, (
            f"{name}: HLO unexpectedly differs from base -- the matrix assumption "
            "that this axis is HLO-inert is wrong"
        )
        assert key == base_key, (
            f"{name}: cheap_key OVER-FRAGMENTS on an HLO-inert axis "
            "(would silently return the 30-min cost as 'warm but slow')"
        )


# --------------------------------------------------------------------------- #
# (3) HLO-changing config + env knobs -> key MUST differ (completeness).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["dt18", "epssm", "subs10"])
def test_hlo_changing_config_axes_change_the_key(matrix_results, name):
    """Each namelist cell that changes the HLO MUST change the cheap_key."""
    by_name = {n: (key, hlo) for n, key, hlo in matrix_results}
    base_key, base_hlo = by_name["base"]
    key, hlo = by_name[name]
    assert hlo != base_hlo, (
        f"{name}: precondition -- this cell was expected to change the HLO but did "
        "not; the test cannot prove the key responds to it"
    )
    assert key != base_key, (
        f"{name}: HLO changed but cheap_key did NOT -> SILENT WRONG LOAD risk "
        "(a determinant is missing from the key)"
    )


# --------------------------------------------------------------------------- #
# (3b) ENV-AXIS (the load-bearing completeness cells) via FRESH SUBPROCESSES.
#      These trace-time GPUWRF_* knobs branch the HLO but live OUTSIDE the
#      namelist/carry; the cheap_key MUST respond to them (else silent wrong load).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "env",
    [
        {"GPUWRF_MOIST_CQW": "0"},
        {"GPUWRF_ACOUSTIC_UNROLL": "4"},
    ],
)
def test_env_axis_changes_both_hlo_and_key(env):
    """A trace-time env knob that changes the HLO MUST change the cheap_key."""
    base_key, base_hlo, _ = _lower_in_subprocess({})
    env_key, env_hlo, _ = _lower_in_subprocess(env)
    assert env_hlo != base_hlo, (
        f"precondition: {env} must change the HLO (trace-time env determinant)"
    )
    assert env_key != base_key, (
        f"{env}: HLO changed but cheap_key did NOT -> SILENT WRONG LOAD risk "
        "(component 5 / global_trace_env_hash is missing this knob)"
    )


# --------------------------------------------------------------------------- #
# (3c) DATE AXIS (v0.25 S3) via FRESH SUBPROCESSES.
#      The key is date-blind by design (#114): that is correct ONLY if the
#      lowered step is date-independent, with every date-derived value (solar
#      clock, CLWRF gases, Noah-MP clock) an operand. Each date lowers in its own
#      fresh process so no in-process cache can hide a baked date constant.
# --------------------------------------------------------------------------- #
_DATE_CELLS = ("2024-09-01_00:00:00", "2025-12-25_06:00:00", "2024-02-29_12:00:00")


def test_date_axis_fresh_processes_share_key_and_hlo():
    """Far-apart dates (incl. a leap day) -> ONE cheap_key, ONE HLO, and
    genuinely different clock operands."""
    records = [
        _lower_record_in_subprocess({}, {"time_utc": date}) for date in _DATE_CELLS
    ]
    assert len({r["key"] for r in records}) == 1, "cheap_key over-fragments on date"
    assert len({r["hlo"] for r in records}) == 1, (
        "CHEAP_KEY COLLISION on the date axis: one key, different step HLOs "
        f"{[r['hlo'][:12] for r in records]} -- a date constant is baked in"
    )
    assert len({r["clock"] for r in records}) == len(records), (
        "precondition: the dates must reach the program as different operands"
    )


def test_date_axis_detector_catches_baked_init_date_gases():
    """Deliberate wrong-key mutation: bake init-date CLWRF gases into the step
    (the pre-S3 program). The date-blind key then COLLIDES -- one key, two HLOs --
    exactly what test_date_axis_fresh_processes_share_key_and_hlo rejects."""
    first, second = (
        _lower_record_in_subprocess({}, {"time_utc": date}, bake_init_date_gases=True)
        for date in _DATE_CELLS[:2]
    )
    assert first["key"] == second["key"]
    assert first["hlo"] != second["hlo"], (
        "the baked-gas mutation did not change the HLO; the date-axis test "
        "would not detect a date constant"
    )


_GAS_PROBE_CODE = """
import json, sys
import numpy as np, jax, jax.numpy as jnp
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.runtime import aot_cheap_key as ck, aot_executable as aotx
from gpuwrf.runtime import aot_precompile as aotp, operational_mode as op
spec = json.loads(sys.argv[1])
grid = GridSpec.canary_3km_template()
shapes = _state_field_shapes(grid)
state = State(**{f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f))
                 for f, s in shapes.items()})
sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
tend = Tendencies(**{k: jnp.zeros(shapes[sk.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
                     for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")})
nl = op.OperationalNamelist(grid=grid, tendencies=tend, metrics=grid.metrics,
                            dt_s=10.0, acoustic_substeps=6, time_utc=spec["time_utc"])
cb = op.build_clock_base(nl)
if spec["bake"]:
    host_gases = pc.clwrf_ssp245_gases_for_time
    pc.clwrf_gases_at_lead = lambda clock, lead: host_gases(spec["time_utc"])

@jax.jit
def gas_probe(state, namelist, lead_seconds, clock_base):
    sw, lw, *_ = pc._rrtmg_column_inputs(
        state, namelist.grid, time_utc=namelist.time_utc, lead_seconds=lead_seconds,
        clock_base=op._rad_clock_base(clock_base), radiation_static=namelist.radiation_static)
    return jnp.stack([lw.co2_vmr, lw.n2o_vmr, lw.ch4_vmr, lw.cfc11_vmr, lw.cfc12_vmr])

args = (state, nl, jnp.asarray(spec["lead"], dtype=jnp.float64), cb)
key = ck.cheap_key(gas_probe, args, {}, nl)
lowered = gas_probe.lower(*args)
out = {"key": key, "hlo": aotx.hlo_sha256_from_lowered(lowered),
       "gpuwrf_file": _GPUWRF_FILE}
if spec["mode"] == "write":
    status = aotp._serialize_domain_blob("s3gas", lowered.compile(), spec["cache"],
                                         lowered=lowered, cheap_key=key,
                                         key_schema=ck.KEY_SCHEMA)
    out["written"] = bool(status["aot_written"])
else:
    call, status = aotp.load_domain_blob("s3gas", spec["cache"], cheap_key=key,
                                         return_status=True)
    out.update(loaded=call is not None, source=status["source"],
               meta_hlo=status["meta_hlo_sha256"],
               gases=np.asarray(call(*args)).tolist() if call is not None else None)
print(json.dumps(out))
"""


def _gas_probe_in_subprocess(mode: str, time_utc: str, cache: str, lead: float, bake: bool):
    payload = {"mode": mode, "time_utc": time_utc, "cache": cache, "lead": lead,
               "bake": bake}
    return _run_pinned_child(_GAS_PROBE_CODE, payload)


@pytest.mark.parametrize("bake", [False, True], ids=["s3", "baked-mutation"])
def test_prior_date_aot_blob_serves_current_date_gases(tmp_path, bake):
    """Persisted AOT across dates: a blob written by a date-1 process is found by
    a FRESH date-2 process under the same cheap_key and must return date-2 valid
    time gases. The baked mutation shows the pre-S3 failure: the date-2 process
    loads the date-1 program and silently serves date-1 gases."""
    from datetime import datetime, timedelta, timezone

    from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time

    cache = str(tmp_path / "aot")
    lead = 3.0 * 86400.0 + 3618.0
    dates = ("2024-09-01_00:00:00", "2025-12-25_06:00:00")
    writer = _gas_probe_in_subprocess("write", dates[0], cache, lead, bake)
    reader = _gas_probe_in_subprocess("read", dates[1], cache, lead, bake)
    assert writer["written"] and reader["loaded"], reader
    assert reader["source"] == "aot_blob" and reader["key"] == writer["key"]

    def host(date):
        valid = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        return np.asarray(clwrf_ssp245_gases_for_time(valid + timedelta(seconds=lead)))

    served = np.asarray(reader["gases"])
    rtol = 4.0 * float(np.finfo(np.float32).eps)
    if not bake:
        assert reader["hlo"] == writer["hlo"] == reader["meta_hlo"]
        np.testing.assert_allclose(served, host(dates[1]), rtol=rtol, atol=0.0)
    else:
        assert reader["hlo"] != reader["meta_hlo"], "mutation must collide"
        np.testing.assert_allclose(
            served, np.asarray(clwrf_ssp245_gases_for_time(dates[0])), rtol=rtol, atol=0.0
        )
        assert not np.allclose(served, host(dates[1]), rtol=1.0e-4)


# --------------------------------------------------------------------------- #
# (4) Cross-process stability -- proves _StaticHolder is content-hashed, not id().
# --------------------------------------------------------------------------- #
def test_cheap_key_is_process_stable():
    """The same config must yield the same cheap_key in TWO fresh processes.

    If any component used ``id()``/``hash(str)`` (PYTHONHASHSEED-salted), a fresh
    run would compute a different key and miss every prewarmed blob (100% fallback,
    zero warm win). We compute the key under two DIFFERENT PYTHONHASHSEEDs in
    separate processes and require an exact match -- the strongest salt check.
    """
    key_a, _, _ = _lower_in_subprocess({"PYTHONHASHSEED": "0"})
    key_b, _, _ = _lower_in_subprocess({"PYTHONHASHSEED": "12345"})
    assert key_a and key_b
    assert key_a == key_b, (
        "cheap_key is NOT process-stable (a component uses id()/salted hash): "
        f"seed0={key_a[:16]} seed12345={key_b[:16]}"
    )


# --------------------------------------------------------------------------- #
# (4a) GPU-LOCK-ENV INVARIANCE -- the v0.21.0 cross-process WARM-LOAD blocker.
#
# scripts/with_gpu_lock.sh exports GPUWRF_GPU_LOCK_{HELD,FD,FILE,HOLDER_FILE,
# LABEL,TOKEN}; LABEL and TOKEN are UNIQUE per invocation. The fail-SAFE env
# auto-discovery (global_trace_env_hash) folded ALL GPUWRF_* vars into the key, so
# every lock-wrapped process computed a DIFFERENT cheap_key -> the warm run looked
# under a key the cold run never wrote -> "fallback:missing" for an artifact that
# DID exist (just under a different filename). The CPU suite was not lock-wrapped,
# so it was process-stable and MISSED this -- this test injects the lock env and
# would have caught it. (TWO fresh processes, each with a different lock token.)
# --------------------------------------------------------------------------- #
def test_gpu_lock_env_does_not_fragment_cheap_key():
    """The per-invocation GPU-lock bookkeeping env MUST NOT change the cheap_key."""
    lock_a = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": "9",
        "GPUWRF_GPU_LOCK_FILE": "/tmp/wrf_gpu2_gpu.lock",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": "/tmp/wrf_gpu2_gpu.lock.holder",
        "GPUWRF_GPU_LOCK_LABEL": "coldA",
        "GPUWRF_GPU_LOCK_TOKEN": "gpuwrf-lock-111-222-333",
    }
    lock_b = dict(lock_a)
    lock_b["GPUWRF_GPU_LOCK_LABEL"] = "warmB"  # different invocation
    lock_b["GPUWRF_GPU_LOCK_TOKEN"] = "gpuwrf-lock-444-555-666"  # unique token
    key_none, _, _ = _lower_in_subprocess({})  # no lock env (the canonical key)
    key_a, _, _ = _lower_in_subprocess(lock_a)
    key_b, _, _ = _lower_in_subprocess(lock_b)
    assert key_a == key_b, (
        "GPU-lock bookkeeping env fragments the cheap_key (per-invocation TOKEN/"
        f"LABEL leaks in): cold={key_a[:16]} warm={key_b[:16]} -> cross-process "
        "WARM LOAD breaks (fallback:missing)"
    )
    assert key_a == key_none, (
        "the GPU-lock env must be inert vs no-lock too (it is pure infra "
        f"bookkeeping): lock={key_a[:16]} no-lock={key_none[:16]}"
    )


def test_gpu_arm_cpu_pinning_does_not_fragment_cheap_key():
    """with_gpu_lock's core plan (GPUWRF_GPU_ARM_CPUS) is process placement, not HLO."""
    import os as _os

    saved = _os.environ.get("GPUWRF_GPU_ARM_CPUS")
    try:
        _os.environ.pop("GPUWRF_GPU_ARM_CPUS", None)
        base = ck.global_trace_env_hash()
        for cpus in ("10,11,26,27", "10,11,12,13,26,27,28,29", ""):
            _os.environ["GPUWRF_GPU_ARM_CPUS"] = cpus
            assert ck.global_trace_env_hash() == base, (
                f"GPUWRF_GPU_ARM_CPUS={cpus!r} leaked into global_trace_env_hash: a cold "
                "and a warm arm with different core plans miss each other's AOT blobs"
            )
    finally:
        if saved is None:
            _os.environ.pop("GPUWRF_GPU_ARM_CPUS", None)
        else:
            _os.environ["GPUWRF_GPU_ARM_CPUS"] = saved


@pytest.mark.parametrize(
    "name,values",
    [
        ("GPUWRF_BENCH_FLAGS_ENV", ("scripts/bench/flags_lw9.env", "/abs/flags_lw10b.env", "")),
        ("GPUWRF_MIN_FREE_VRAM_GIB", ("24", "8")),
        ("GPUWRF_MIN_FREE_VRAM_FRACTION", ("0.5", "0.1")),
    ],
)
def test_bench_and_preflight_infra_env_does_not_fragment_cheap_key(monkeypatch, name, values):
    """No traced reader: the bench flag-file path and the preflight VRAM threshold."""
    monkeypatch.delenv(name, raising=False)
    base = ck.global_trace_env_hash()
    for value in values:
        monkeypatch.setenv(name, value)
        assert ck.global_trace_env_hash() == base, f"{name}={value!r} leaked into the cheap key"
    assert ck.trace_env_is_inert(name)


def test_trace_env_is_inert_is_the_hash_rule(monkeypatch):
    for name in ("GPUWRF_GPU_ARM_CPUS", "GPUWRF_JAX_CACHE_DIR", "GPUWRF_GPU_LOCK_FUTURE_FIELD"):
        assert ck.trace_env_is_inert(name), name
    assert not ck.trace_env_is_inert("GPUWRF_MOIST_CQW")
    monkeypatch.delenv("GPUWRF_MOIST_CQW", raising=False)
    base = ck.global_trace_env_hash()
    monkeypatch.setenv("GPUWRF_MOIST_CQW", "0")
    assert ck.global_trace_env_hash() != base, "positive control: a physics knob must stay keyed"


def test_process_infra_env_is_a_narrow_subset_of_the_hlo_denylist():
    """Both keys drop process infra; only the cheap key drops 'HLO keyed elsewhere' knobs."""
    assert ck.PROCESS_INFRA_ENV <= ck.HLO_AFFECTING_ENV_DENYLIST
    for name in (*ck.PROCESS_INFRA_ENV, "GPUWRF_GPU_LOCK_TOKEN", "GPUWRF_GPU_LOCK_FUTURE_FIELD"):
        assert ck.is_process_infra_env(name) and ck.trace_env_is_inert(name), name
    for name in ("GPUWRF_NESTED_FUSE", "GPUWRF_NESTED_DEFUSE_COMPILE", "GPUWRF_ADVANCE_CHUNK_LOOP",
                 "GPUWRF_TRAINING_OUTPUT_SUBSET", "GPUWRF_NESTED_AOT", "GPUWRF_PROFILE",
                 "GPUWRF_AOT_VERIFY"):
        assert ck.trace_env_is_inert(name) and not ck.is_process_infra_env(name), name
    assert not ck.is_process_infra_env("GPUWRF_MOIST_CQW")


def test_aot_prewarm_env_does_not_fragment_cheap_key():
    """The AOT prewarm orchestration knob is not a lowered-HLO determinant."""

    key_none, _, _ = _lower_in_subprocess({})
    key_off, _, _ = _lower_in_subprocess({"GPUWRF_NESTED_AOT_PREWARM": "0"})
    key_on, _, _ = _lower_in_subprocess({"GPUWRF_NESTED_AOT_PREWARM": "1"})
    assert key_none == key_off == key_on, (
        "GPUWRF_NESTED_AOT_PREWARM leaked into cheap_key despite being an AOT "
        f"orchestration knob: none={key_none[:16]} off={key_off[:16]} on={key_on[:16]}"
    )


def test_gpu_lock_env_prefix_is_denylisted():
    """The lock bookkeeping vars are excluded by name AND by inert prefix."""
    import os as _os

    saved = {k: _os.environ.get(k) for k in (
        "GPUWRF_GPU_LOCK_TOKEN", "GPUWRF_GPU_LOCK_SOMETHING_NEW",
    )}
    try:
        base = ck.global_trace_env_hash()
        # A NEW (not individually denylisted) var under the inert prefix must also
        # be excluded by the prefix guard, so a future lock var cannot re-break it.
        _os.environ["GPUWRF_GPU_LOCK_TOKEN"] = "unique-per-run-xyz"
        _os.environ["GPUWRF_GPU_LOCK_SOMETHING_NEW"] = "future-bookkeeping"
        assert ck.global_trace_env_hash() == base, (
            "a GPUWRF_GPU_LOCK_* var leaked into global_trace_env_hash (prefix "
            "guard not applied)"
        )
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


# --------------------------------------------------------------------------- #
# (4b) PREWARM-vs-EAGER carry REPRESENTATION INVARIANCE.
#
# The parallel-prewarm worker keys a domain off a ShapeDtypeStruct carry
# (aot_precompile._to_shape_dtype_tree -- picklable, no device buffers) while the
# eager warm loop keys off the CONCRETE runtime carry. _to_shape_dtype_tree lowers
# to the IDENTICAL HLO (documented invariant), so the cheap_key MUST match too --
# else the prewarm-serialized blob is unloadable by the eager loop and the WHOLE
# parallel-prewarm wall-win is defeated. The MetaTy strings (sharding/committed/
# is_jax_array) differ between the two representations, so they must NOT be hashed
# raw; carry_aval_hash hashes a canonical placement class instead.
# --------------------------------------------------------------------------- #
def test_prewarm_shape_carry_and_concrete_carry_share_cheap_key():
    """ShapeDtypeStruct (prewarm) and concrete (eager) carries -> SAME cheap_key."""
    from gpuwrf.runtime.aot_precompile import _to_shape_dtype_tree
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori, build_clock_base

    carry, namelist, clock_base = _build_call()
    start = jnp.asarray(1, jnp.int32)
    cadence = int(namelist.radiation_cadence_steps)

    k_concrete = ck.cheap_key(
        _advance_chunk_fori, (carry, namelist, start, clock_base),
        {"n_steps": 1, "cadence": cadence}, namelist,
    )
    carry_shapes = _to_shape_dtype_tree(carry)
    clock_shapes = _to_shape_dtype_tree(build_clock_base(namelist))
    k_shape = ck.cheap_key(
        _advance_chunk_fori, (carry_shapes, namelist, start, clock_shapes),
        {"n_steps": 1, "cadence": cadence}, namelist,
    )
    assert k_concrete and k_shape
    assert k_concrete == k_shape, (
        "prewarm shape-dtype carry and concrete carry compute DIFFERENT cheap_keys "
        f"(concrete={k_concrete[:16]} shape={k_shape[:16]}) -> the prewarm-written "
        "blob is unloadable by the eager loop; the MetaTy placement fields must be "
        "canonicalized (placement_class), not hashed raw"
    )


def test_carry_aval_hash_is_representation_invariant():
    """carry_aval_hash matches for ShapeDtypeStruct vs concrete carry."""
    from gpuwrf.runtime.aot_precompile import _to_shape_dtype_tree
    from gpuwrf.runtime.operational_mode import build_clock_base

    carry, namelist, clock_base = _build_call()
    start = jnp.asarray(1, jnp.int32)
    args_c = (carry, namelist, start, clock_base)
    args_s = (
        _to_shape_dtype_tree(carry), namelist, start,
        _to_shape_dtype_tree(build_clock_base(namelist)),
    )
    kw = {"n_steps": 1, "cadence": 1}
    assert ck.carry_aval_hash(args_c, kw) == ck.carry_aval_hash(args_s, kw)


def test_placement_class_still_distinguishes_real_placement_changes():
    """The canonicalization keeps the GPT-critic SAFETY: a genuinely different
    placement (committed array / non-single-device sharding) is NOT collapsed."""
    default_abstract = ck._placement_class({"committed": None, "sharding": None})
    default_concrete = ck._placement_class(
        {"committed": False, "sharding": "SingleDeviceSharding(device=CpuDevice(id=0))"}
    )
    committed = ck._placement_class(
        {"committed": True, "sharding": "SingleDeviceSharding(device=CpuDevice(id=0))"}
    )
    sharded = ck._placement_class(
        {"committed": False, "sharding": "NamedSharding(mesh=m, spec=PartitionSpec('x'))"}
    )
    # The abstract (prewarm) and uncommitted single-device (eager) cases MUST agree.
    assert default_abstract == default_concrete == "default"
    # A genuinely different placement MUST be distinguished (no silent reuse).
    assert committed != "default" and committed != default_concrete
    assert sharded != "default" and sharded != committed


# --------------------------------------------------------------------------- #
# (5) COLLISION DETECTION -- a deliberately-incomplete key IS caught.
#
# This is the adversarial proof: build a key that OMITS the trace-time env
# component (component 5). Under it, two configs that differ ONLY in an env knob
# (which DOES change the HLO) collide -> same key, different HLO. The injectivity
# check MUST flag it. This proves the test has teeth: it would FAIL if a real
# determinant were dropped, so the PASS in test (1) is meaningful.
# --------------------------------------------------------------------------- #
def test_incomplete_key_collision_is_detected():
    """An env-blind key collides on the env axis -> the collision IS caught.

    Two FRESH-PROCESS lowerings that differ only in GPUWRF_ACOUSTIC_UNROLL have
    DIFFERENT HLOs (it changes the acoustic-loop unroll STRUCTURE) while leaving the
    namelist AND the carry avals UNCHANGED -- so it is captured ONLY by component 5
    (global_trace_env_hash). The env-BLIND key (component 5 omitted) therefore gives
    the two lowerings the SAME key. We assert (a) the HLOs really differ (the knob
    is a genuine determinant captured by no other component), (b) the incomplete key
    COLLIDES (the exact silent-wrong bug), and (c) the FULL key does NOT collide
    (the real key closes the gap). This proves the injectivity test has teeth: it
    would FAIL if component 5 were dropped.
    """
    full_on, hlo_on, incomplete_on = _lower_in_subprocess({})
    full_off, hlo_off, incomplete_off = _lower_in_subprocess(
        {"GPUWRF_ACOUSTIC_UNROLL": "4"}
    )

    # The env knob is a genuine HLO determinant captured by NO other component.
    assert hlo_on != hlo_off, (
        "precondition: GPUWRF_ACOUSTIC_UNROLL must change the HLO for this test to "
        "mean anything"
    )
    # The INCOMPLETE (env-blind) key COLLIDES -> exactly the silent-wrong bug.
    assert incomplete_on == incomplete_off, (
        "the deliberately-incomplete (env-blind) key did NOT collide; the "
        "collision-detection test is not exercising the env gap it claims to"
    )
    # The FULL key does NOT collide -> the real key closes the gap.
    assert full_on != full_off, (
        "the FULL cheap_key collided on the env axis -- component 5 "
        "(global_trace_env_hash) is not capturing GPUWRF_ACOUSTIC_UNROLL"
    )


# =========================================================================== #
# GPT-CRITIC SAFETY FIXES (KEY_SCHEMA v2) -- the P0/P1/P2 closures.
# =========================================================================== #
import dataclasses  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402


# --------------------------------------------------------------------------- #
# A tiny REAL jit + serialize harness (no GPU State needed) reused by the P0-1
# collision-overwrite test. Two DIFFERENT compiled programs are forced under ONE
# cheap_key to construct a real two-HLO same-key collision on disk.
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _MiniNamelist:
    radiation_cadence_steps: int
    scale: object
    time_utc: object = datetime(2024, 9, 1, tzinfo=timezone.utc)
    noahmp_julian: float = 1.0
    noahmp_yearlen: float = 365.0

    def tree_flatten(self):
        return (self.scale,), (
            int(self.radiation_cadence_steps),
            self.time_utc,
            float(self.noahmp_julian),
            float(self.noahmp_yearlen),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        cadence, time_utc, julian, yearlen = aux
        return cls(
            radiation_cadence_steps=cadence,
            scale=children[0],
            time_utc=time_utc,
            noahmp_julian=julian,
            noahmp_yearlen=yearlen,
        )


def _mini_advance():
    from gpuwrf.runtime import domain_tree as dt  # build_clock_base lives here

    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {"y": carry["a"] + carry["b"]}

    return advance_like, dt


def _compile_mini(mult: int):
    """Compile a small jit that lowers to a DIFFERENT HLO per ``mult`` (a*mult)."""
    from gpuwrf.runtime import domain_tree as dt

    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {"y": carry["a"] * mult + carry["b"]}

    namelist = _MiniNamelist(radiation_cadence_steps=7, scale=jnp.asarray(1, jnp.int32))
    clock_base = dt.build_clock_base(namelist)
    carry = {"a": jnp.arange(4.0), "b": jnp.ones((4,))}
    lowered = advance_like.lower(
        carry, namelist, jnp.asarray(1, jnp.int32), clock_base, n_steps=1, cadence=7
    )
    from gpuwrf.runtime import aot_executable as aotx

    hlo = aotx.hlo_sha256_from_lowered(lowered)
    compiled = lowered.compile()
    return compiled, hlo


def _compile_mini_with_lowered(mult: int):
    """Like :func:`_compile_mini` but ALSO returns the ``lowered`` object.

    The HLO-digest fix needs the lowered StableHLO to assert the persisted
    ``meta.hlo_sha256`` equals ``hlo_sha256_from_lowered(lowered)`` and to drive
    the no-``hlo_sha256``-arg serialize path (the cross-process GPU bug)."""
    from gpuwrf.runtime import aot_executable as aotx
    from gpuwrf.runtime import domain_tree as dt

    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {"y": carry["a"] * mult + carry["b"]}

    namelist = _MiniNamelist(radiation_cadence_steps=7, scale=jnp.asarray(1, jnp.int32))
    clock_base = dt.build_clock_base(namelist)
    carry = {"a": jnp.arange(4.0), "b": jnp.ones((4,))}
    lowered = advance_like.lower(
        carry, namelist, jnp.asarray(1, jnp.int32), clock_base, n_steps=1, cadence=7
    )
    hlo = aotx.hlo_sha256_from_lowered(lowered)
    compiled = lowered.compile()
    return compiled, lowered, hlo


@pytest.fixture()
def _cache(monkeypatch, tmp_path):
    from tests._jax_cache_isolation import private_jax_cache

    with private_jax_cache(monkeypatch, tmp_path / "jit") as cache_dir:
        yield cache_dir


# --------------------------------------------------------------------------- #
# KEY_SCHEMA v3: source location is inert; terrain identity/geometry is not.
# --------------------------------------------------------------------------- #
def test_v3_terrain_source_path_is_inert_but_content_and_geometry_remain_keyed():
    from gpuwrf.contracts.grid import TerrainProvenance

    assert ck.KEY_SCHEMA == "GPUWRF-AOTKEY-v4"
    base = TerrainProvenance(
        source_path="/namespace/a/terrain.nc",
        sha256="a" * 64,
        shape=(93, 195),
        units="m",
        projection_transform="lambert",
        max_elevation_m=3715.0,
        coastline_sanity_check_passed=True,
    )
    relocated = dataclasses.replace(base, source_path="/namespace/b/terrain.nc")
    assert ck.canonical_digest(base) == ck.canonical_digest(relocated)

    mutations = {
        "sha256": "b" * 64,
        "shape": (94, 195),
        "units": "km",
        "projection_transform": "mercator",
        "max_elevation_m": 3716.0,
        "coastline_sanity_check_passed": False,
    }
    base_digest = ck.canonical_digest(base)
    for field, value in mutations.items():
        changed = dataclasses.replace(base, **{field: value})
        assert ck.canonical_digest(changed) != base_digest, field


def test_v4_wrfinput_provenance_path_is_inert_but_prefix_and_geometry_are_keyed():
    carry, namelist, clock_base = _build_call()
    del carry, clock_base

    def with_provenance(value):
        metrics = dataclasses.replace(namelist.metrics)
        object.__setattr__(metrics, "provenance", value)
        return dataclasses.replace(namelist, metrics=metrics)

    key = ck._structure_fingerprint  # metrics provenance enters via the call treedef
    case_a = with_provenance("wrfinput:/cases/20260227_18z_a1/wrfinput_d01:nz=44:eta=(45,)")
    case_b = with_provenance("wrfinput:/cases/20260502_18z_a1/wrfinput_d01:nz=44:eta=(45,)")
    assert key(case_a) == key(case_b)
    for other in ("wrfinput:/cases/a/wrfinput_d01:nz=45:eta=(46,)",
                  "analytic-flat",
                  "wrfinput:/cases/20260227_18z_a1/wrfinput_d01:nz=44:eta=(45,):x-sharded"):
        assert key(with_provenance(other)) != key(case_a), other
    assert key(with_provenance("wrfinput:/x/wrfinput_d01:nz=44:eta=(45,):x-sharded")) != key(
        with_provenance("wrfinput:/y/wrfinput_d01:nz=44:eta=(45,):x-sharded"))


def test_v3_real_static_config_hash_is_terrain_path_invariant():
    carry, namelist, clock_base = _build_call()
    del carry, clock_base
    relocated_terrain = dataclasses.replace(
        namelist.grid.terrain,
        source_path="/different/process/namespace/canary_terrain.nc",
    )
    relocated_grid = dataclasses.replace(namelist.grid, terrain=relocated_terrain)
    relocated = dataclasses.replace(namelist, grid=relocated_grid)
    assert ck.static_config_hash(relocated) == ck.static_config_hash(namelist)


# --------------------------------------------------------------------------- #
# (P0-1) Collision-overwrite is FAIL-CLOSED: a REAL two-HLO same-cheap_key
#        collision does NOT overwrite k_<cheap_key>; the load fails OPEN.
# --------------------------------------------------------------------------- #
def test_p0_1_collision_does_not_overwrite_cheap_key_blob_and_load_fails_open(_cache):
    """Two distinct HLOs forced under ONE cheap_key -> quarantine, no overwrite.

    Constructs the exact silent-wrong scenario: program A is serialized under
    cheap_key K; then program B (a DIFFERENT HLO) is serialized under the SAME K.
    The write-guard must (a) NOT overwrite A's blob with B, (b) quarantine K, and
    (c) make the subsequent load of K fail OPEN (return None) -- never silently
    serve B (or a half-overwritten blob)."""
    from gpuwrf.runtime import aot_precompile as aotp

    compiled_a, hlo_a = _compile_mini(mult=2)
    compiled_b, hlo_b = _compile_mini(mult=5)
    assert hlo_a and hlo_b and hlo_a != hlo_b, "preconditions: A and B are distinct HLOs"

    forged_key = "collision" + "0" * 56  # one shared cheap_key for both programs

    # 1) Write program A under the forged cheap_key (clean, first write).
    sa = aotp._serialize_domain_blob(
        "d01", compiled_a, str(_cache),
        hlo_sha256=hlo_a, cheap_key=forged_key, key_schema=ck.KEY_SCHEMA,
    )
    assert sa["aot_written"] is True and sa.get("cheap_key") == forged_key, sa
    blob_path, meta_path = aotp._aot_blob_paths(
        "d01", str(_cache), cheap_key=forged_key
    )
    a_blob_sha = sa["blob_sha256"]
    assert blob_path.is_file()
    a_hlo_blob, _ = aotp._aot_blob_paths("d01", str(_cache), hlo_sha256=hlo_a)
    assert a_hlo_blob.is_file(), "safe writes must publish the exact-HLO address"

    # 2) Write program B under the SAME forged cheap_key -> COLLISION.
    sb = aotp._serialize_domain_blob(
        "d01", compiled_b, str(_cache),
        hlo_sha256=hlo_b, cheap_key=forged_key, key_schema=ck.KEY_SCHEMA,
    )
    # The write must be marked quarantined and must NOT have written the cheap-key
    # address (cheap_key in the status is None -> only the hlo-addressed fallback).
    assert sb.get("cheap_key_quarantined") is True, sb
    assert sb.get("cheap_key") is None, sb

    # 3) The cheap-key blob/meta are REMOVED (poisoned), and the key is quarantined.
    assert aotp.cheap_key_is_quarantined("d01", forged_key, str(_cache))
    assert not blob_path.is_file(), "ambiguous cheap-key blob was NOT removed"
    assert not meta_path.is_file(), "ambiguous cheap-key meta was NOT removed"

    # 4) The load of the quarantined cheap_key FAILS OPEN (no silent wrong blob).
    call, status = aotp.load_domain_blob(
        "d01", str(_cache), cheap_key=forged_key, return_status=True
    )
    assert call is None
    assert status["source"] == "fallback:cheap-key-quarantined", status

    # 5) Program B's HLO-addressed fallback DID land (keyed by exact HLO -> safe).
    b_blob_path, _ = aotp._aot_blob_paths("d01", str(_cache), hlo_sha256=hlo_b)
    assert b_blob_path.is_file(), "the hlo-addressed fallback for B should exist"
    # And A's exact-HLO artifact survived cheap-key quarantine untouched.
    assert a_hlo_blob.is_file(), "quarantine removed the unambiguous A HLO artifact"
    assert a_blob_sha != sb["blob_sha256"]


def test_p0_1_quarantine_blocks_future_cheap_key_writes(_cache):
    """Once quarantined, even a re-write of the matching program skips the cheap key."""
    from gpuwrf.runtime import aot_precompile as aotp

    compiled_a, hlo_a = _compile_mini(mult=3)
    forged_key = "poison" + "0" * 58
    aotp.quarantine_cheap_key("d01", forged_key, str(_cache), reason="test")
    assert aotp.cheap_key_is_quarantined("d01", forged_key, str(_cache))
    s = aotp._serialize_domain_blob(
        "d01", compiled_a, str(_cache),
        hlo_sha256=hlo_a, cheap_key=forged_key, key_schema=ck.KEY_SCHEMA,
    )
    # Wrote (the hlo fallback) but NOT under the quarantined cheap key.
    assert s["aot_written"] is True
    assert s.get("cheap_key") is None, s
    assert s.get("cheap_key_quarantined") is True, s
    blob_path, _ = aotp._aot_blob_paths("d01", str(_cache), cheap_key=forged_key)
    assert not blob_path.is_file()


# --------------------------------------------------------------------------- #
# (P1-3) Load metadata enforcement: a cheap-key load requires the full contract.
# --------------------------------------------------------------------------- #
def test_p1_3_cheap_key_load_requires_metadata_contract(_cache):
    """meta.cheap_key/key_schema/hlo_sha256/blob_sha256 must all positively match."""
    import pickle

    from gpuwrf.runtime import aot_executable as aotx
    from gpuwrf.runtime import aot_precompile as aotp

    compiled, hlo = _compile_mini(mult=2)
    blob, meta = aotx.serialize(
        compiled, hlo_sha256=hlo, cheap_key="goodkey" + "0" * 57, key_schema=ck.KEY_SCHEMA
    )
    good_key = "goodkey" + "0" * 57

    def _write(m):
        bp, mp = aotp._aot_blob_paths("d01", str(_cache), cheap_key=good_key)
        aotp._atomic_write_bytes(bp, blob)
        aotp._atomic_write_bytes(mp, pickle.dumps(m))
        return bp, mp

    # (a) wrong key_schema -> reject.
    _write(dataclasses.replace(meta, key_schema="GPUWRF-AOTKEY-OLD"))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st

    # (b) missing hlo_sha256 -> reject.
    _write(dataclasses.replace(meta, hlo_sha256=None))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st

    # (c) missing blob_sha256 -> reject (cannot attest integrity).
    _write(dataclasses.replace(meta, blob_sha256=None))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st

    # (d) meta.cheap_key None (not recorded) -> reject (was silently ACCEPTED before).
    _write(dataclasses.replace(meta, cheap_key=None))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st


# --------------------------------------------------------------------------- #
# (P0-2) Transitive source identity: a source-fingerprint change -> different key.
# --------------------------------------------------------------------------- #
def test_p0_2_source_fingerprint_change_changes_program_key(monkeypatch):
    """If source_fingerprint_hash changes, program_key + cheap_key change too.

    Proves a TRACED-CALLEE source edit (modeled here by perturbing the source
    fingerprint, which a real edit to dynamics/physics would do) invalidates the
    key even with __version__ unbumped -- closing the silent stale-blob reuse."""
    carry, namelist, clock_base = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    args = (carry, namelist, jnp.asarray(1, jnp.int32), clock_base)
    kw = {"n_steps": 1, "cadence": 1}
    pk0 = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek0 = ck.exec_key(_advance_chunk_fori, args, kw, namelist)

    # Model a source edit: clear the lru_cache and force a different fingerprint.
    ck.source_fingerprint_hash.cache_clear()
    monkeypatch.setattr(
        ck, "source_fingerprint_hash", lambda: "EDITED-SOURCE-FINGERPRINT"
    )
    pk1 = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek1 = ck.exec_key(_advance_chunk_fori, args, kw, namelist)
    assert pk0 != pk1, "program_key did NOT respond to a source-fingerprint change"
    assert ek0 != ek1, "exec_key did NOT respond to a source-fingerprint change"


def test_p0_2_source_fingerprint_is_nonempty_and_stable():
    """The real source fingerprint resolves to a stable 64-hex digest."""
    ck.source_fingerprint_hash.cache_clear()
    a = ck.source_fingerprint_hash()
    b = ck.source_fingerprint_hash()
    assert a == b and len(a) == 64, a


# --------------------------------------------------------------------------- #
# (P0-2 SCOPE) source_fingerprint is scoped to TRACE-REACHABLE modules only.
#
# THE v0.21.0 9-NEST cold->warm BLOCKER: source_fingerprint_hash content-hashed
# the WHOLE src/gpuwrf tree, so a concurrent UNCOMMITTED edit to an ORCHESTRATION
# module (domain_tree.py -- the de-fuse-flip agent's diff that landed between the
# cold compile and the warm load) shifted ALL 9 domains' cheap_keys while the
# lowered _advance_chunk_fori HLO was byte-identical -> the warm process looked
# under keys the cold process never wrote -> 100% fallback:missing, full ~60-min
# re-lower, headline defeated. The 3-nest passed only because its cold+warm ran in
# ONE unedited window (NOT a max_dom bug). Fix: scope the digest to the STATIC
# IMPORT CLOSURE of the traced body (gpuwrf.runtime.operational_mode), a provable
# SUPERSET of the trace-reachable set -> an orchestration/IO edit is invariant,
# but a genuine traced-callee (dynamics/physics/coupling) edit still shifts it.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _src_copy(tmp_path_factory):
    """Private copy of the candidate source tree for the source-EDIT tests (E125).

    Those tests model a working-tree edit; doing it in src/ in place races any
    concurrent run in the same checkout and can leave the edit behind. data/ and
    scripts/ are linked next to the copy for package-relative lookups."""
    import shutil

    repo = Path(_CANDIDATE_SRC).parent
    tree = tmp_path_factory.mktemp("src_copy")
    shutil.copytree(
        Path(_CANDIDATE_SRC) / "gpuwrf",
        tree / "src" / "gpuwrf",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("data", "scripts"):
        (tree / name).symlink_to((repo / name).resolve())
    return tree / "src"


@pytest.fixture()
def _fingerprint_on_copy(_src_copy, monkeypatch):
    """Point source_fingerprint_hash at the private copy; yields the copy's src root."""
    monkeypatch.setattr(ck, "_gpuwrf_package_root", lambda: _src_copy / "gpuwrf")
    ck.source_fingerprint_hash.cache_clear()
    yield _src_copy
    ck.source_fingerprint_hash.cache_clear()


def _patch_one_source_file_byte(src_root, rel_path: str):
    """Context manager: append a harmless comment to one file under ``src_root``, then restore.

    Models a real working-tree edit (the exact failure mode: a concurrent agent
    edits a .py between the cold and warm runs) on a private copy, never src/.
    Returns a contextmanager."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        target = src_root / rel_path
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# regression-test transient edit\n")
            ck.source_fingerprint_hash.cache_clear()
            yield
        finally:
            target.write_bytes(original)
            ck.source_fingerprint_hash.cache_clear()

    return _ctx()


def test_source_fingerprint_invariant_to_orchestration_edit(_fingerprint_on_copy):
    """An edit to a NON-traced orchestration module must NOT shift the fingerprint.

    domain_tree.py / nested_pipeline.py / cli.py / aot_*.py import the traced body
    but are NOT imported BY it -> outside the import closure -> cannot change the
    lowered HLO -> the cache key MUST be invariant. This is the exact 9-nest
    cold->warm blocker: a domain_tree.py edit between cold and warm must no longer
    desync the keys. Would have FAILED before the scope fix (whole-tree digest)."""
    ck.source_fingerprint_hash.cache_clear()
    base = ck.source_fingerprint_hash()
    for orchestration in (
        "gpuwrf/runtime/domain_tree.py",
        "gpuwrf/integration/nested_pipeline.py",
        "gpuwrf/cli.py",
        "gpuwrf/runtime/aot_precompile.py",
    ):
        with _patch_one_source_file_byte(_fingerprint_on_copy, orchestration):
            shifted = ck.source_fingerprint_hash()
        assert shifted == base, (
            f"source_fingerprint shifted on an HLO-IRRELEVANT edit to {orchestration} "
            f"(the 9-nest cold->warm blocker): base={base[:16]} edited={shifted[:16]}. "
            "The orchestration module must be OUTSIDE the trace-import closure."
        )


@pytest.mark.parametrize(
    "traced_module",
    [
        "gpuwrf/runtime/operational_mode.py",  # the traced body itself
        "gpuwrf/dynamics/core/acoustic.py",  # a deep traced dycore callee
        "gpuwrf/physics/__init__.py",  # a traced physics module
        "gpuwrf/nesting/interp.py",  # fused cascade force-down interpolation
        "gpuwrf/coupling/boundary_feedback.py",  # fused cascade feedback
        "gpuwrf/coupling/physics_couplers.py",  # a traced coupling callee
        "gpuwrf/contracts/state.py",  # a traced contract (carry struct)
    ],
)
def test_source_fingerprint_responds_to_traced_callee_edit(traced_module, _fingerprint_on_copy):
    """An edit to a TRACE-REACHABLE module MUST shift the fingerprint (SAFETY).

    The safety side of the scope fix: a real source edit to a module that CAN
    change the lowered _advance_chunk_fori HLO must invalidate the cache key, so a
    stale blob is never silently reused. Under-scoping here = SILENT WRONG RESULT;
    this positive control guards against an over-narrow closure."""
    ck.source_fingerprint_hash.cache_clear()
    base = ck.source_fingerprint_hash()
    with _patch_one_source_file_byte(_fingerprint_on_copy, traced_module):
        shifted = ck.source_fingerprint_hash()
    assert shifted != base, (
        f"source_fingerprint did NOT respond to an edit of the trace-reachable "
        f"module {traced_module} -- it is missing from the import closure "
        "(under-scoped = silent stale-blob reuse risk)."
    )


def test_trace_closure_includes_traced_excludes_orchestration():
    """The import closure structurally contains every traced subtree and NO
    orchestration/IO module -- a direct assertion on the scope set itself."""
    from pathlib import Path

    pkg_root = Path(ck.__file__).resolve().parent.parent  # .../src/gpuwrf
    files = ck._trace_reachable_source_files(pkg_root)
    assert files is not None and files, "import closure could not be computed"
    rels = {f.relative_to(pkg_root).as_posix() for f in files}
    # MUST contain traced callees (HLO-affecting source).
    for needle in (
        "runtime/operational_mode.py",
        "contracts/state.py",
        "contracts/grid.py",
    ):
        assert any(r.endswith(needle) for r in rels), f"closure missing traced {needle}"
    assert any(r.startswith("dynamics/") for r in rels), "closure missing dynamics/"
    assert any(r.startswith("physics/") for r in rels), "closure missing physics/"
    assert any(r.startswith("coupling/") for r in rels), "closure missing coupling/"
    # The fused cascade traces the nest force-down and feedback too.
    for needle in (
        "nesting/boundary_construction.py",
        "nesting/interp.py",
        "coupling/boundary_feedback.py",
    ):
        assert needle in rels, f"closure missing fused-cascade callee {needle}"
    # MUST NOT contain orchestration/IO (the bug source).
    for forbidden in (
        "runtime/domain_tree.py",
        "integration/nested_pipeline.py",
        "cli.py",
        "runtime/aot_precompile.py",
        "runtime/aot_cheap_key.py",
    ):
        assert not any(r.endswith(forbidden) for r in rels), (
            f"orchestration/IO module {forbidden} is INSIDE the closure -- it would "
            "re-introduce the 9-nest cold->warm key churn"
        )
    # The closure is a strict subset of the whole tree (proves it actually narrows).
    whole = {
        p.relative_to(pkg_root).as_posix()
        for p in pkg_root.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert rels < whole, "closure is not a strict subset of the whole tree"


def test_cheap_key_stable_across_process_and_concurrent_orchestration_edit(_src_copy):
    """END-TO-END regression for the 9-nest blocker, on CPU, two fresh processes.

    Process A (cold): computes the cheap_key with a clean tree.
    Process B (warm): computes it under a DIFFERENT PYTHONHASHSEED *and* with an
    HLO-irrelevant orchestration edit applied (domain_tree.py) -- exactly the
    cold->warm window that broke the GPU 9-nest. The keys MUST match. Before the
    scope fix B's key would differ (whole-tree digest shifted by the edit), so this
    test would have CAUGHT the blocker. Both processes run on a private copy of the
    tree, which receives the edit (E125: never src/ in place)."""
    src_root = str(_src_copy)
    key_a, hlo_a, _ = _lower_in_subprocess({"PYTHONHASHSEED": "0"}, src_root=src_root)

    orchestration = _src_copy / "gpuwrf/runtime/domain_tree.py"
    original = orchestration.read_bytes()
    try:
        orchestration.write_bytes(original + b"\n# concurrent de-fuse-flip edit\n")
        key_b, hlo_b, _ = _lower_in_subprocess({"PYTHONHASHSEED": "31337"}, src_root=src_root)
    finally:
        orchestration.write_bytes(original)

    assert key_a and key_b
    assert hlo_a == hlo_b, (
        "the orchestration edit changed the lowered HLO -- test fixture is wrong "
        f"(should be HLO-irrelevant): {hlo_a[:12]} vs {hlo_b[:12]}"
    )
    assert key_a == key_b, (
        "cheap_key is NOT stable across a cold->warm window with a concurrent "
        f"HLO-irrelevant orchestration edit (the 9-nest blocker): cold={key_a[:16]} "
        f"warm={key_b[:16]}. source_fingerprint must be scoped to trace-reachable "
        "modules only."
    )


# --------------------------------------------------------------------------- #
# (P1-4) program_key vs exec_key: same HLO program, different compile env ->
#        SAME program_key but DIFFERENT exec_key (the blob address splits).
# --------------------------------------------------------------------------- #
def test_p1_4_exec_key_splits_on_compile_options_but_program_key_does_not():
    """A change in an exec-only determinant (XLA_FLAGS) must move exec_key only."""
    carry, namelist, clock_base = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    args = (carry, namelist, jnp.asarray(1, jnp.int32), clock_base)
    kw = {"n_steps": 1, "cadence": 1}

    pk_base = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek_base = ck.exec_key(_advance_chunk_fori, args, kw, namelist)

    # exec_env_hash reads XLA_FLAGS live; mutate it and recompute (program is inert).
    old = os.environ.get("XLA_FLAGS")
    os.environ["XLA_FLAGS"] = (old or "") + " --xla_force_host_platform_device_count=3"
    try:
        pk_flag = ck.program_key(_advance_chunk_fori, args, kw, namelist)
        ek_flag = ck.exec_key(_advance_chunk_fori, args, kw, namelist)
    finally:
        if old is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = old

    assert pk_flag == pk_base, "program_key MUST be invariant to XLA_FLAGS (exec-only)"
    assert ek_flag != ek_base, "exec_key MUST split on XLA_FLAGS (blob-address determinant)"


# --------------------------------------------------------------------------- #
# (P1-5) Resolved import-time env constants: the key captures the RESOLVED module
#        constant, so an env mutation AFTER import cannot desync it from the HLO.
# --------------------------------------------------------------------------- #
def test_p1_5_module_const_env_hash_is_import_resolved_not_live_env(monkeypatch):
    """module_const_env_hash hashes the RESOLVED constant, not live os.environ.

    The module constant (boundary_apply.NORMAL_BDY_RELAX_STRENGTH) is resolved at
    import; mutating GPUWRF_NORMAL_BDY_RELAX_STRENGTH in this (already-imported)
    process must NOT change the hash -- proving the key tracks what the HLO baked,
    not the live env (the desync the GPT critic flagged)."""
    h0 = ck.module_const_env_hash()
    monkeypatch.setenv("GPUWRF_NORMAL_BDY_RELAX_STRENGTH", "999.0")
    h1 = ck.module_const_env_hash()
    assert h0 == h1, (
        "module_const_env_hash changed on a POST-import env mutation -- it must hash "
        "the resolved import-time constant, not live os.environ"
    )


def test_p1_5_module_const_env_changes_when_resolved_constant_changes(monkeypatch):
    """If the resolved module constant differs, the hash differs (positive control)."""
    import gpuwrf.coupling.boundary_apply as bdy

    real = bdy.NORMAL_BDY_RELAX_STRENGTH
    h0 = ck.module_const_env_hash()
    monkeypatch.setattr(bdy, "NORMAL_BDY_RELAX_STRENGTH", real + 7.0)
    h1 = ck.module_const_env_hash()
    assert h0 != h1, "hash did not respond to a changed resolved module constant"


# --------------------------------------------------------------------------- #
# (P1-5 CI scanner) Flag trace-reachable import-time env-derived module constants
#        that are NOT covered by IMPORT_TIME_ENV_CONSTANTS.
# --------------------------------------------------------------------------- #
def _reads_env(node, helpers=frozenset()):
    """``os.environ.get/pop/setdefault``, ``os.getenv``, ``os.environ[...]``,
    ``... in os.environ`` or a call (``f(...)`` or ``mod.f(...)``) to an env helper."""
    import ast

    for c in ast.walk(node):
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute):
            if c.func.attr == "getenv" or (
                c.func.attr in ("get", "pop", "setdefault")
                and isinstance(c.func.value, ast.Attribute)
                and c.func.value.attr == "environ"
            ):
                return True
        if isinstance(c, ast.Call) and getattr(c.func, "id", getattr(c.func, "attr", None)) in helpers:
            return True
        if isinstance(c, ast.Subscript) and isinstance(c.value, ast.Attribute) and c.value.attr == "environ":
            return True
        if isinstance(c, ast.Compare) and any(
            isinstance(x, ast.Attribute) and x.attr == "environ" for x in c.comparators
        ):
            return True
    return False


# Import-time env-derived bindings that are deliberately NOT in the registry.
_IMPORT_TIME_ENV_NOT_HLO = {
    "gpuwrf/__init__.py::_JAX_X64_FORCE_STATUS": "resolved jax_enable_x64 is in program_config_hash",
    "gpuwrf/__init__.py::_COMMAND_BUFFER_STATUS": "XLA_FLAGS is in exec_env_hash",
    "gpuwrf/__init__.py::_FAST_DEFAULTS_STATUS": "release switch VALUES land in os.environ; global_trace_env_hash hashes them",
    "gpuwrf/__init__.py::_JAX_CACHE_STATUS": "cache location, not a program determinant",
    "gpuwrf/physics/ra_lw_rrtm.py::WRF_ROOT": "known residual: ra_lw_physics=1 table path (not PROD)",
}


# Import-time env-reading calls whose effect is keyed elsewhere.
_IMPORT_TIME_ENV_SIDE_EFFECTS = {
    "configure_jax_x64": "resolved jax_enable_x64 is in program_config_hash",
}


def test_no_uncovered_import_time_env_constants():
    """CI scan: every module-level constant resolved from env at IMPORT in the
    traced import closure (the same file set the source fingerprint hashes) must be
    in IMPORT_TIME_ENV_CONSTANTS, so the key hashes its RESOLVED value. A NEW one
    trips this test -> add it to the registry.

    Detection: a module-level (column-0) assignment whose RHS reads env directly or
    through an env-reading helper ANYWHERE in the closure, transitively and through
    import aliases (``_env_bool(...)``, ``column_tiling.env_bool(...)``,
    ``sgs_cloud_enabled()``). Bare-name matching over-approximates (safe).
    Env reads inside functions resolve at TRACE time and are covered by
    ``global_trace_env_hash`` (live env) instead."""
    import ast

    import gpuwrf

    pkg_root = Path(gpuwrf.__file__).resolve().parent
    src_root = pkg_root.parent
    files = ck._trace_reachable_source_files(pkg_root)
    assert files, "trace closure unavailable"
    covered = {
        f"{m.replace('.', '/')}.py::{a}" for (m, a) in ck.IMPORT_TIME_ENV_CONSTANTS
    }
    trees = {py: ast.parse(py.read_text(), filename=str(py)) for py in files}
    nodes = [n for t in trees.values() for n in ast.walk(t)]
    defs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    aliases = [a for n in nodes if isinstance(n, ast.ImportFrom) for a in n.names if a.asname]
    helpers: frozenset[str] = frozenset()
    while True:  # fixpoint: env helpers, helpers of helpers, and their import aliases
        grown = helpers | {f.name for f in defs if _reads_env(f, helpers)}
        grown |= {a.asname for a in aliases if a.name in grown}
        if grown == helpers:
            break
        helpers = frozenset(grown)
    offenders: list[str] = []
    for py, tree in trees.items():
        rel = py.relative_to(src_root).as_posix()
        for node in tree.body:  # MODULE-LEVEL statements only (import-time)
            # Other import-time env reads: bare statements/branches, class bodies,
            # def-time default arguments. Each must be an allowlisted side effect.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parts = [d for d in node.args.defaults + node.args.kw_defaults if d is not None]
            elif isinstance(node, ast.ClassDef):
                parts = [b for b in node.body if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
            elif not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
                parts = [node]
            else:
                parts = []
            for part in parts:
                calls = {getattr(c.func, "id", getattr(c.func, "attr", None))
                         for c in ast.walk(part) if isinstance(c, ast.Call)}
                if _reads_env(part, helpers) and not calls & set(_IMPORT_TIME_ENV_SIDE_EFFECTS):
                    offenders.append(f"{rel}::<{type(node).__name__}@{node.lineno}>")
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            if not _reads_env(node.value, helpers):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    ident = f"{py.relative_to(src_root).as_posix()}::{t.id}"
                    if ident not in covered and ident not in _IMPORT_TIME_ENV_NOT_HLO:
                        offenders.append(ident)
    assert not offenders, (
        "UNCOVERED import-time env-derived module constants in trace-reachable code "
        "(add each to aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS so the cheap_key hashes "
        f"its RESOLVED value): {sorted(offenders)}"
    )


def test_import_time_env_registry_entries_resolve():
    """Every registered constant exists, so the key never hashes a stale sentinel."""
    import importlib

    for module, attr in ck.IMPORT_TIME_ENV_CONSTANTS:
        assert hasattr(importlib.import_module(module), attr), f"{module}:{attr}"


# --------------------------------------------------------------------------- #
# (P2) Determinant matrix expansion: GridSpec array content + MetaTy.
# --------------------------------------------------------------------------- #
def test_p2_gridspec_array_content_is_content_hashed():
    """GridSpec STATIC-aux array CONTENT (eta_levels / terrain_height) is hashed.

    Proves the static config is CONTENT-hashed (not by salted __hash__/id), so a
    GridSpec static-array content edit cannot share a key with the original. (The
    GridSpec's TRACED children -- e.g. ``vertical`` arrays passed as device leaves
    -- are correctly handled by carry_aval_hash's shape/dtype instead, so we
    perturb the AUX arrays here.)"""
    import dataclasses as _dc

    from gpuwrf.contracts.grid import GridSpec

    g0 = GridSpec.canary_3km_template()
    d0 = ck.canonical_digest(g0)

    # (a) top-level eta_levels (a baked static array) content change.
    e = np.asarray(g0.eta_levels)
    assert e.size > 3
    e2 = e.copy()
    e2[1] = float(e2[1]) * 0.999 + 1e-4
    g_eta = _dc.replace(g0, eta_levels=jnp.asarray(e2, dtype=e.dtype))
    assert ck.canonical_digest(g_eta) != d0, "eta_levels content not hashed"

    # (b) terrain_height (a baked static field) content change.
    if g0.terrain_height is not None:
        th = np.asarray(g0.terrain_height)
        if th.size:
            th2 = th.copy()
            th2.flat[0] = float(th2.flat[0]) + 1.0
            g_th = _dc.replace(g0, terrain_height=jnp.asarray(th2, dtype=th.dtype))
            assert ck.canonical_digest(g_th) != d0, "terrain_height content not hashed"

    # Content equality holds for value-identical copies (no id()/salt leakage).
    g_same = _dc.replace(g0, eta_levels=jnp.asarray(e.copy(), dtype=e.dtype))
    assert ck.canonical_digest(g_same) == d0, "value-identical GridSpec digest differs"


def test_p2_carry_metaty_fields_present_in_aval_record():
    """carry_aval_hash leaf records carry the MetaTy fields (sharding/committed/...)."""
    carry, namelist, clock_base = _build_call()
    leaf = jnp.arange(4.0)
    rec = ck._leaf_metaty_record(leaf)
    for field in ("shape", "dtype", "weak_type", "sharding", "committed", "is_jax_array"):
        assert field in rec, f"MetaTy field {field} missing from leaf record"


def test_p2_clock_base_none_fails_open():
    """clock_base=None (date-static path) must NOT crash; cheap_key fails open or keys.

    The contract requires a non-None clock_base; a None must fail OPEN (return None),
    never raise, so the caller compiles. We assert no exception escapes and the
    result is either a valid key or None (both are fail-safe)."""
    carry, namelist, _cb = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    key = ck.cheap_key(
        _advance_chunk_fori,
        (carry, namelist, jnp.asarray(1, jnp.int32), None),
        {"n_steps": 1, "cadence": 1},
        namelist,
    )
    assert key is None or (isinstance(key, str) and len(key) == 64)


def test_p2_program_key_and_exec_key_are_distinct_and_stable():
    """program_key != exec_key (different folds) and both are stable 64-hex digests."""
    carry, namelist, clock_base = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    args = (carry, namelist, jnp.asarray(1, jnp.int32), clock_base)
    kw = {"n_steps": 1, "cadence": 1}
    pk = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek = ck.exec_key(_advance_chunk_fori, args, kw, namelist)
    assert len(pk) == 64 and len(ek) == 64 and pk != ek
    # cheap_key IS exec_key.
    assert ck.cheap_key(_advance_chunk_fori, args, kw, namelist) == ek


# --------------------------------------------------------------------------- #
# CROSS-PROCESS HLO-DIGEST FIX: the persisted meta.hlo_sha256 MUST be the
# lower-only StableHLO digest (== hlo_sha256_from_lowered(lowered)), so a fresh
# warm process loads the cheap-key blob and verify-mode confirms it. Regression
# guard for the GPU bug where the on-disk meta.hlo_sha256 was None (the compiled
# fallback returns None / a DIFFERENT digest) -> fallback:cheap-key-meta-mismatch
# + verify-error.
# --------------------------------------------------------------------------- #
def test_serialize_persists_lowered_hlo_digest_in_meta(_cache):
    """``aot_executable.serialize`` writes the LOWERED digest to ``meta.hlo_sha256``.

    Even when the explicit ``hlo_sha256`` arg is None, passing ``lowered=`` makes
    the persisted digest equal ``hlo_sha256_from_lowered(lowered)`` -- NOT the
    compiled-HLO fallback (which is None on GPU / a non-matching digest on CPU)."""
    from gpuwrf.runtime import aot_executable as aotx

    compiled, lowered, hlo = _compile_mini_with_lowered(mult=2)
    assert hlo, "precondition: lowered HLO digest is available on this backend"

    # (a) explicit hlo arg omitted, but lowered provided -> persisted == lowered digest.
    _blob, meta = aotx.serialize(compiled, lowered=lowered)
    assert meta.hlo_sha256 == hlo, (
        "serialize(lowered=...) must persist the lowered StableHLO digest, "
        f"got {meta.hlo_sha256!r} expected {hlo!r}"
    )

    # (b) the misleading compiled fallback is a DIFFERENT (or None) digest -> proves
    #     the lowered source is what we must persist, never the compiled one.
    compiled_digest = aotx._compiled_hlo_sha256(compiled)
    assert compiled_digest != hlo, (
        "compiled-HLO digest unexpectedly equals the lowered digest; the fix's "
        "premise (they differ) would be moot"
    )


def test_serialize_domain_blob_threads_lowered_so_on_disk_meta_has_hlo(_cache):
    """``_serialize_domain_blob(..., lowered=...)`` -> on-disk meta has a real hlo.

    This is the cold-write contract: the blob is written under the cheap_key, and
    the PICKLED meta on disk carries a nonempty ``hlo_sha256`` == the lowered
    digest. Without the fix the on-disk meta.hlo_sha256 was None."""
    import pickle

    from gpuwrf.runtime import aot_precompile as aotp

    compiled, lowered, hlo = _compile_mini_with_lowered(mult=3)
    cheap = "hlofix" + "0" * 58

    # Simulate the EXACT cold-write GPU bug condition: the caller's hlo_sha256 came
    # back None on this backend, but we thread the lowered object through.
    status = aotp._serialize_domain_blob(
        "d01",
        compiled,
        str(_cache),
        hlo_sha256=None,
        lowered=lowered,
        cheap_key=cheap,
        key_schema=ck.KEY_SCHEMA,
    )
    assert status["aot_written"] is True, status
    assert status.get("cheap_key") == cheap, status
    assert status.get("hlo_sha256") == hlo, status

    # Read the ON-DISK meta back (a fresh process only ever sees this).
    _blob_path, meta_path = aotp._aot_blob_paths("d01", str(_cache), cheap_key=cheap)
    with open(meta_path, "rb") as fh:
        on_disk_meta = pickle.load(fh)
    assert on_disk_meta.hlo_sha256 == hlo, (
        "on-disk meta.hlo_sha256 must equal the lowered digest; "
        f"got {on_disk_meta.hlo_sha256!r}"
    )


def test_safe_serialize_publishes_cheap_and_exact_hlo_addresses(_cache):
    """Dual addresses share bytes/inode and both pass their metadata contracts."""
    from gpuwrf.runtime import aot_precompile as aotp

    compiled, lowered, hlo = _compile_mini_with_lowered(mult=9)
    cheap = "dualaddr" + "0" * 56
    status = aotp._serialize_domain_blob(
        "d04",
        compiled,
        str(_cache),
        lowered=lowered,
        cheap_key=cheap,
        key_schema=ck.KEY_SCHEMA,
    )
    assert status["aot_written"] is True, status
    assert status["aot_addresses"] == ["hlo", "cheap_key"], status
    cheap_blob, cheap_meta = aotp._aot_blob_paths(
        "d04", str(_cache), cheap_key=cheap
    )
    hlo_blob, hlo_meta = aotp._aot_blob_paths(
        "d04", str(_cache), hlo_sha256=hlo
    )
    assert all(path.is_file() for path in (cheap_blob, cheap_meta, hlo_blob, hlo_meta))
    assert cheap_blob.read_bytes() == hlo_blob.read_bytes()
    if status["aot_alias_mode"] == "hardlink":
        assert cheap_blob.stat().st_dev == hlo_blob.stat().st_dev
        assert cheap_blob.stat().st_ino == hlo_blob.stat().st_ino
    else:
        assert status["aot_alias_mode"] == "copy"

    cheap_call, cheap_status = aotp.load_domain_blob(
        "d04", str(_cache), cheap_key=cheap, return_status=True
    )
    hlo_call, hlo_status = aotp.load_domain_blob(
        "d04", str(_cache), hlo_sha256=hlo, return_status=True
    )
    assert cheap_call is not None and cheap_status["address"] == "cheap_key"
    assert hlo_call is not None and hlo_status["address"] == "hlo"


def test_fresh_process_cheap_key_load_succeeds_after_lowered_serialize(_cache):
    """A fresh-process-style cheap-key load + verify-style cross-check both PASS.

    ``load_domain_blob`` keeps NO in-memory state -- it reads the pickled meta from
    disk every call, so calling it after the serialize models a fresh warm process.
    With the fix the P1-3 metadata-enforcement path accepts the blob (loaded, not
    ``fallback:cheap-key-meta-mismatch``) and the verify-style cross-check
    (``meta.hlo_sha256 == hlo_sha256_from_lowered(live_lowered)``) confirms -- no
    verify-error."""
    from gpuwrf.runtime import aot_executable as aotx
    from gpuwrf.runtime import aot_precompile as aotp

    compiled, lowered, hlo = _compile_mini_with_lowered(mult=4)
    cheap = "warmload" + "0" * 56

    # Cold write (hlo_sha256 arg None on this backend; lowered threaded through).
    status = aotp._serialize_domain_blob(
        "d02",
        compiled,
        str(_cache),
        hlo_sha256=None,
        lowered=lowered,
        cheap_key=cheap,
        key_schema=ck.KEY_SCHEMA,
    )
    assert status["aot_written"] and status.get("cheap_key") == cheap, status

    # Fresh-process-style warm load via the metadata-enforcement (P1-3) path.
    call, ld = aotp.load_domain_blob(
        "d02", str(_cache), cheap_key=cheap, return_status=True
    )
    assert call is not None, f"warm cheap-key load failed: {ld}"
    assert ld["source"] == "aot_blob", ld
    assert ld["source"] != "fallback:cheap-key-meta-mismatch", ld
    assert ld["meta_hlo_sha256"] == hlo, ld

    # Verify-mode style cross-check: the recorded meta hlo MUST match the digest a
    # fresh process recomputes by re-lowering the SAME program (so verify confirms,
    # never `missing HLO digest for verify`).
    live_hlo = aotx.hlo_sha256_from_lowered(lowered)
    assert live_hlo and ld["meta_hlo_sha256"] == live_hlo, (
        "verify-mode cross-check would fail: meta hlo "
        f"{ld['meta_hlo_sha256']!r} != live lowered hlo {live_hlo!r}"
    )
