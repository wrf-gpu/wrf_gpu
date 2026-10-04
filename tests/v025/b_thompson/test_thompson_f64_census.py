"""Exact f64 census of the native-REAL Thompson full-column kernel (E145).

The frozen WRF oracles cannot see REAL/DOUBLE kind choices (S2-MP mutants passed them), so the
kind map is pinned here: f64-result equations of the kernel jaxpr per innermost thompson function,
split into conditional / unconditional. A change must be justified against module_mp_thompson.F
declarations (audit: wrf_gpu2_lanes/b-thompson/TH06/thompson_f64_audit.md) and re-pinned.
"""
import collections
import inspect
import os
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import source_info_util

from gpuwrf.kernels.phys_thompson_full import full_column
from gpuwrf.physics import thompson_column as tc

C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0")
SKIP = {"convert_element_type", "broadcast_in_dim", "reshape", "select_n", "squeeze", "concatenate", "slice",
        "dynamic_slice", "copy", "pjit"}
PINNED = {  # JAX 0.10, TH08 tree (S2-MP + B52 fixes): 664 f64 eqns, 51 unconditional
    "_apply_cold_collection_rates|uncond": 14, "_apply_rain_evaporation|cond": 10, "_apply_warm_rain_rates|cond": 12,
    "_balance_ice_number|cond": 3, "_clamp_rain_number|cond": 24, "_cloud_distribution|cond": 2,
    "_cloud_water_fall_speed|cond": 4, "_cloud_water_freezing_rates|cond": 14, "_cold_collection_rates|cond": 33,
    "_dpow|cond": 30, "_fall_speeds|cond": 21, "_graupel_distribution|cond": 13,
    "_ice_collection_rates_from_moments|cond": 16, "_ice_distribution|cond": 4, "_ice_sources_with_process_flags|cond": 160,
    "_inactive_sedimentation.<locals>.sedimentation|uncond": 1, "_inactive_source_branches.<locals>.cold|uncond": 1, "_inactive_source_branches.<locals>.ice|uncond": 1,
    "_load|cond": 210, "_load|uncond": 14, "_lookup_digit_index|cond": 21,
    "_rain_distribution|cond": 6, "_rain_evaporation|cond": 14, "_scale_cold_collection_rain_rates|cond": 3,
    "_store|uncond": 20, "_warm_rain_collection|cond": 13,
}


def f64_census():
    x = np.load(Path(__file__).with_name("fixtures") / "exit_columns.npz")
    state = tc.ThompsonColumnState(**{k: jnp.asarray(x[k][:1]) for k in tc.ThompsonColumnState.__slots__})
    jaxpr = jax.make_jaxpr(lambda s: full_column(s, 18.0))(state).jaxpr
    inner = next(e.params for e in jaxpr.eqns if e.primitive.name == "pallas_call")["jaxpr"]
    counts = collections.Counter()

    def walk(j, in_cond):
        for e in j.eqns:
            for param in e.params.values():
                for sub in (param if isinstance(param, (list, tuple)) else [param]):
                    body = sub.jaxpr if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns") else (
                        sub if hasattr(sub, "eqns") else None)
                    if body is not None:
                        walk(body, in_cond or e.primitive.name == "cond")
            if e.primitive.name in SKIP or not any(
                    str(getattr(getattr(v, "aval", None), "dtype", "")) == "float64" for v in e.outvars):
                continue
            frames = [f for f in source_info_util.user_frames(e.source_info.traceback) if "thompson" in f.file_name]
            counts[f"{frames[0].function_name if frames else '?'}|{'cond' if in_cond else 'uncond'}"] += 1

    walk(inner, False)
    return dict(counts)


def _with_env(fn):
    saved = {key: os.environ.get(key) for key in C24}
    os.environ.update(C24)
    jax.clear_caches()
    try:
        return fn()
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        jax.clear_caches()


def test_native_real_thompson_f64_census_is_pinned():
    assert _with_env(f64_census) == PINNED


def test_census_sees_a_real_to_double_kind_change(monkeypatch):
    """Mutant: drop the REAL rounding of WRF's REAL ice sump -> the census must change."""
    src = textwrap.dedent(inspect.getsource(tc._ice_sources_with_process_flags))
    line = "        ice_sump = ice_sump.astype(jnp.float32)  # WRF REAL sump\n"
    assert src.count(line) == 1
    namespace = dict(vars(tc))
    exec(compile(src.replace(line, "        pass\n"), tc.__file__, "exec"), namespace)
    monkeypatch.setattr(tc, "_ice_sources_with_process_flags", namespace["_ice_sources_with_process_flags"])
    assert _with_env(f64_census) != PINNED
