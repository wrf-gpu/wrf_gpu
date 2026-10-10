"""cu_physics=4 (scale-aware GFS SAS, ARW) -- parity of the JAX port vs pristine WRF 4.7.1.

Oracle: ``proofs/v034/scalesas`` -- the UNMODIFIED ``module_cu_scalesas.F`` (+ GFS funcphys /
physcons) called by a multi-column ``CU_SCALESAS`` driver exactly like ARW ``cumulus_driver``
(only ``DY`` is passed explicitly: ARW forwards the absent optional ``DYNMM``, which segfaults).
Two builds: ``r4`` = the WRF build semantics (RWORDSIZE=4) and ``r8`` = ``-fdefault-real-8``.
455 columns x 2 configurations (dt 54 s/STEPCU 5/DY 9 km; dt 18 s/STEPCU 1/DY 3 km); the
MANIFEST census (from a debug-instrumented copy whose outputs are byte-identical) lists the
exercised paths.

Gates (pre-registered for this lane):
  * r4 port vs r4 oracle: every output within 1 float32 ulp and >= 99.9 % bitwise identical
    (measured: 100 % bitwise on both configs, eager AND jax.jit); HBOT/HTOP exact.
  * r8 port vs r8 oracle: <= 1e-9 relative to the column max (measured <= 1.5e-11 with XLA:CPU
    FMA contraction, <= 2.4e-13 with ``--xla_cpu_max_isa=AVX``).
  * Knife-edge columns (rain-free up to an O(1e-20) roundoff residual in BOTH arms, not both
    exactly 0: the residual's sign decides restore vs keep) are excluded and capped at 5 %.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.physics.cumulus_scalesas import cu_scalesas_columns  # noqa: E402

SP = Path(__file__).resolve().parents[1] / "proofs" / "v034" / "scalesas" / "savepoints"
CONFIGS = ("cfg_dt54_stepcu5", "cfg_dt18_stepcu1")
OUTPUTS = ("RAINCV", "PRATEC", "HBOT", "HTOP", "SCALEFUN", "SIGMU",
           "RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
KNIFE = 1.0e-12


def _manifest():
    return json.loads((SP / "MANIFEST.json").read_text())


def _run(cfg: str, mode: str, compiled: bool = False):
    import functools

    inp = np.load(SP / "columns_inputs.npz")
    meta = _manifest()["configs"][cfg]
    fn = functools.partial(cu_scalesas_columns, dy=meta["DY"], dt=meta["DT"], stepcu=meta["STEPCU"],
                           literals=mode)
    if compiled:  # whole-program XLA compilation, as in the operational scan
        fn = jax.jit(fn)
    out = fn(inp["T"], inp["QV"], inp["QC"], inp["QI"], inp["P"], inp["PI"], inp["RHO"], inp["DZ"],
             inp["U"], inp["V"], inp["W"], inp["P8W"][:, 0], inp["XLAND"], inp["DX2D"])
    return {k: np.asarray(v) for k, v in out.items()}


def _oracle(cfg: str, mode: str):
    return dict(np.load(SP / f"{cfg}_oracle_{mode}.npz"))


def _knife_edge(port, ora):
    """Columns that are rain-free in both arms up to a roundoff residual but not identically 0.

    When the evaporation cap removes all rain, ``rn`` ends as an O(1e-20) residual whose sign
    is roundoff; ``rn>0`` keeps the (non-restored) tendencies, ``rn<=0`` restores the column.
    """

    p = np.abs(port["RAINCV"].astype(np.float64))
    o = np.abs(ora["RAINCV"].astype(np.float64))
    return (p < KNIFE) & (o < KNIFE) & ~((p == 0.0) & (o == 0.0))


def test_savepoints_are_sha_pinned_and_cover_the_regimes():
    import hashlib

    man = _manifest()
    for name, sha in man["sha256"].items():
        assert hashlib.sha256((SP / name).read_bytes()).hexdigest() == sha, name
    for cfg in CONFIGS:
        c = man["configs"][cfg]["census_r4"]
        # E154: non-trivial active regimes and every reachable rejection/closure path.
        assert c["active_QE"] >= 40 and c["active_nonQE"] >= 40
        assert c["ktconn_active"] >= 10 and c["htop_km"] >= 3 and c["restore_rn<=0"] >= 3
        for path in ("kill_kbcon", "kill_cinpcr", "kill_dthk", "kill_cthk", "kill_jmin_aa1", "kill_wc_ddaa1"):
            assert c[path] >= 1, (cfg, path)
    assert man["configs"]["cfg_dt54_stepcu5"]["census_r4"]["kill_closure"] >= 1


def test_savepoints_present_and_tracked_for_clean_checkouts():
    """R1 (rv-sas2): *.npz is gitignored repo-wide -- every MANIFEST savepoint must exist AND be
    tracked, else a clean checkout cannot run the parity gate."""

    import subprocess

    names = sorted(_manifest()["sha256"])
    assert len(names) == 5
    for name in names:
        assert (SP / name).is_file(), name
    root = SP.parents[3]
    try:
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--", str(SP.relative_to(root))],
                                 capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    for name in names:
        assert f"proofs/v034/scalesas/savepoints/{name}" in tracked, f"{name} is not tracked by git"


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "jit"))
@pytest.mark.parametrize("cfg", CONFIGS)
def test_r4_port_is_bitwise_with_wrf_real4_build(cfg, compiled):
    port, ora = _run(cfg, "r4", compiled), _oracle(cfg, "r4")
    knife = _knife_edge(port, ora)
    assert knife.sum() <= 23, f"too many knife-edge columns: {np.where(knife)[0]}"
    keep = ~knife
    active = ora["RAINCV"][keep] > 0
    assert active.sum() >= 140
    assert np.array_equal(port["HBOT"][keep], ora["HBOT"][keep])
    assert np.array_equal(port["HTOP"][keep], ora["HTOP"][keep])
    for k in OUTPUTS:
        a = port[k][keep].astype(np.float32)
        b = ora[k][keep].astype(np.float32)
        ulp = np.spacing(np.maximum(np.abs(a), np.abs(b)).astype(np.float32))
        assert np.all(np.abs(a.astype(np.float64) - b) <= ulp), k
        assert np.mean(a == b) >= 0.999, (k, np.mean(a == b))


@pytest.mark.parametrize("cfg", CONFIGS)
def test_r8_port_matches_promoted_build_to_machine_precision(cfg):
    port, ora = _run(cfg, "r8"), _oracle(cfg, "r8")
    knife = _knife_edge(port, ora)
    assert knife.sum() <= 23, f"too many knife-edge columns: {np.where(knife)[0]}"
    keep = ~knife
    assert np.array_equal(port["HBOT"][keep], ora["HBOT"][keep])
    for k in OUTPUTS:
        a, b = port[k][keep], ora[k][keep]
        if a.ndim == 1:
            scale = np.abs(b) + 1e-300
        else:
            scale = np.max(np.abs(b), axis=1, keepdims=True) + 1e-300
        assert np.all(np.abs(a - b) <= 1e-9 * scale), (k, float(np.max(np.abs(a - b) / scale)))


def test_wrf_real4_literals_are_load_bearing():
    """Deletion check of the r4 literal mode: the promoted-literal port must NOT reproduce the
    WRF REAL build (it differs in trigger flags and tendencies), i.e. the oracle discriminates."""

    port = _run("cfg_dt18_stepcu1", "r8")
    ora = _oracle("cfg_dt18_stepcu1", "r4")
    differs = np.mean(port["RTHCUTEN"].astype(np.float32) != ora["RTHCUTEN"])
    assert differs > 0.05
