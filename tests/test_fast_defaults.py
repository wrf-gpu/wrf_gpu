"""v0.3 release defaults: no flag file == the LW9 release flag set (+ ring select, REAL_ALL pair)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from gpuwrf._fast_defaults import FAST_PATH_DEFAULTS

REPO = Path(__file__).resolve().parents[1]
FLAGS_FILE = REPO / "scripts" / "bench" / "flags_lw9.env"
_NOT_IN_LW9_FILE = {"GPUWRF_SPEC_RING_SELECT", "GPUWRF_DYN_REAL_ALL", "GPUWRF_CARRY_REAL_ALL",
                    "GPUWRF_MYNN_PLUME_SUMS", "GPUWRF_MYNN_TRIDIAG_REAL", "GPUWRF_RRTMG_LW_BAND_SUMS",
                    "GPUWRF_MCICA_KISS_KERNEL", "GPUWRF_ACOUSTIC_W2PASS", "GPUWRF_ACOUSTIC_MASS_BLOCK",
                    "GPUWRF_ACOUSTIC_UV_MASKED", "GPUWRF_DYN_PD_SPECIES_LOOP", "GPUWRF_DYN_GLUE_FUSED",
                    "GPUWRF_RRTMG_SW_BAND_SUMS", "GPUWRF_CARRY_DONATE",
                    "GPUWRF_ACOUSTIC_W_RECUR_SL", "GPUWRF_NOAHMP_LAYER_LISTS", "GPUWRF_NOAHMP_COLUMN_KERNELS",
                    "GPUWRF_LAYOUT_PIN",
                    "GPUWRF_RRTMG_MAXRAND", "GPUWRF_RRTMG_MP_RE", "GPUWRF_MYNN_SGS_MIXING_RATIO", "GPUWRF_MYNN_SCALE_AWARE", "GPUWRF_W_SURFACE_RESET", "GPUWRF_MYNN_SFC_WSPD", "GPUWRF_SPEC_W_WORK_COPY", "GPUWRF_MYNN_FLTV_WRF", "GPUWRF_MYNN_DHEAT", "GPUWRF_MYNN_PSIQ_FLUX_WRF", "GPUWRF_W_DAMP_STAGE", "GPUWRF_ACOUSTIC_NO_MU_FLOOR", "GPUWRF_NOAHMP_JULIAN_ADVANCE", "GPUWRF_NEST_O3_FROM_PARENT", "GPUWRF_ROOT_SCALAR_BDY_RK1", "GPUWRF_THOMPSON_MIXED_PHASE_WRF",
                    "GPUWRF_MYNN_DMP_KTOP_BOUND", "GPUWRF_MYNN_ELB_MF",
                    "GPUWRF_MYNN_PHY_EXNER", "GPUWRF_MYNN_PSIG_CLAMP", "GPUWRF_MYNN_QNI_MIXING"}


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in FAST_PATH_DEFAULTS and k not in {"GPUWRF_FAST_DEFAULTS", "XLA_FLAGS"}}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(REPO / "src"))
    env.update(extra)
    return env


def _probe(env: dict[str, str], body: str, *, source_flags: bool = False) -> dict:
    code = ("import json, os, gpuwrf\n"
            "from gpuwrf._fast_defaults import FAST_PATH_DEFAULTS, FAST_DEFAULTS_STATUS\n" + body)
    cmd = [sys.executable, "-c", code]
    if source_flags:
        cmd = ["bash", "-c", 'source "$1"; shift; exec "$@"', "bash", str(FLAGS_FILE), *cmd]
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=300)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _flag_file_values() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    out = subprocess.run(["bash", "-c", 'source "$1"; env -0', "bash", str(FLAGS_FILE)],
                         env=env, check=True, capture_output=True, text=True)
    pairs = (item.split("=", 1) for item in out.stdout.split("\0") if "=" in item)
    return {k: v for k, v in pairs if k.startswith("GPUWRF_") or k.startswith("JAX_")}


def test_defaults_equal_the_release_flag_file():
    file_values = _flag_file_values()
    assert file_values, FLAGS_FILE
    expected = {k: v for k, v in FAST_PATH_DEFAULTS.items() if k not in _NOT_IN_LW9_FILE}
    assert file_values == expected


def test_fresh_import_applies_every_default():
    got = _probe(_clean_env(), "print(json.dumps({k: os.environ.get(k) for k in FAST_PATH_DEFAULTS}))")
    assert got == FAST_PATH_DEFAULTS


def test_explicit_value_wins_per_flag():
    got = _probe(_clean_env(GPUWRF_DYN_FP32="0", GPUWRF_RRTMG_COLUMN_TILE_COLS="1024"),
                 "print(json.dumps({'env': {k: os.environ[k] for k in ('GPUWRF_DYN_FP32', "
                 "'GPUWRF_RRTMG_COLUMN_TILE_COLS', 'GPUWRF_DYN_RK_FP32')}, "
                 "'kept': FAST_DEFAULTS_STATUS['kept']}))")
    assert got["env"] == {"GPUWRF_DYN_FP32": "0", "GPUWRF_RRTMG_COLUMN_TILE_COLS": "1024",
                          "GPUWRF_DYN_RK_FP32": "1"}
    assert got["kept"] == {"GPUWRF_DYN_FP32": "0", "GPUWRF_RRTMG_COLUMN_TILE_COLS": "1024"}


def test_master_opt_out_restores_legacy_environment():
    got = _probe(_clean_env(GPUWRF_FAST_DEFAULTS="0"),
                 "print(json.dumps({'set': sorted(k for k in FAST_PATH_DEFAULTS if k in os.environ "
                 "and k not in ('JAX_PLATFORMS',)), 'enabled': FAST_DEFAULTS_STATUS['enabled']}))")
    assert got["enabled"] is False
    # gpuwrf's own x64 hook may still export JAX_ENABLE_X64; no release switch is set.
    assert [k for k in got["set"] if k.startswith("GPUWRF_")] == []


def test_cheap_key_env_parity_with_the_flag_file():
    body = ("from gpuwrf.runtime import aot_cheap_key as ck\n"
            "print(json.dumps({'trace': ck.global_trace_env_hash(), 'consts': ck.module_const_env_hash(), "
            "'xla': os.environ.get('XLA_FLAGS', '')}))")
    no_file = _probe(_clean_env(), body)
    with_file = _probe(_clean_env(**{k: FAST_PATH_DEFAULTS[k] for k in _NOT_IN_LW9_FILE}), body,
                       source_flags=True)
    assert no_file["trace"] == with_file["trace"]
    assert no_file["consts"] == with_file["consts"]
    master_on = _probe(_clean_env(GPUWRF_FAST_DEFAULTS="1"), body)
    assert master_on["trace"] == no_file["trace"]


def test_gpu_command_buffer_injection_equals_the_flag_file_xla_string():
    # On a CPU pin nothing is injected (and the flag file still exports its tokens), so compare
    # the GPU-path injection string itself: configure_command_buffers(default_on=True) appends
    # exactly these candidates to an empty XLA_FLAGS on a GPU host.
    import inspect
    from gpuwrf.runtime import xla_autotune

    src = inspect.getsource(xla_autotune.configure_command_buffers)
    injected = [tok for tok in ("--xla_gpu_enable_command_buffer=+CONDITIONAL",
                                "--xla_enable_command_buffers_during_profiling=true") if tok in src]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    out = subprocess.run(["bash", "-c", 'unset XLA_FLAGS; source "$1"; printf %s "$XLA_FLAGS"', "bash",
                          str(FLAGS_FILE)], env=env, check=True, capture_output=True, text=True)
    assert " ".join(injected) == out.stdout
