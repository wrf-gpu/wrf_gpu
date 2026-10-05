"""fid-q2 R01 (alisios §3.4): WRF lead-zero nest state of the blend ring, on real WN3 0227 operands (CPU).

WRF ``med_nest_initial`` → ``adjust_tempqv`` → ``start_domain(nest)`` (AL/ALT/P from the input MU, then
``press_adj`` with start_em's reference-profile ALB) and, when the nest's own child opens on its first
step, ``start_domain(parent=nest)`` again ("kludge: 20040604"), which re-derives AL/ALT/P from the
post-``press_adj`` MU.  WRF history ``T`` is ``th_phy_m_t0``: the input dry theta at lead zero.
Truth = the CPU-WRF lead-zero frames of the same case (read-only).
"""
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.integration import d02_replay as rp

CASE = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1")
CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
FRAME = "2026-02-28_00:00:00"


def _rings(ny, nx):
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    return np.minimum(np.minimum(j, ny - 1 - j), np.minimum(i, nx - 1 - i))


def _f32(ds, name):
    return np.asarray(ds.variables[name][0], np.float32)


@pytest.fixture(scope="module")
def wn3():
    if not (CASE / "wrfinput_d03").is_file() or not (CPU / f"wrfout_d03_{FRAME}").is_file():
        pytest.skip("WN3 0227 case or its CPU-WRF lead-zero frames unavailable")
    return rp.Gen2Run(CASE)


def _start_domain(run, domain, *, child_opens):
    """Port start_domain transcription on WRF's own lead-zero blended base + adjusted THM."""
    with Dataset(CASE / f"wrfinput_{domain}") as w, Dataset(CPU / f"wrfout_{domain}_{FRAME}") as c:
        inp = {n: _f32(w, n) for n in ("HGT", "PH", "MU", "W", "U", "V")}
        cpu = {n: _f32(c, n) for n in ("HGT", "PB", "MUB", "PHB", "THM", "P", "MU")}
    grid = run.grid(domain).as_grid_spec()
    grid = dataclasses.replace(grid, terrain_height=jnp.asarray(cpu["HGT"], jnp.float64))
    metrics = rp.load_wrfinput_metrics(CASE / f"wrfinput_{domain}")
    p, mu, _w, meta = rp._wrf_live_nest_start_domain_perturb_init(
        run, domain=domain, grid=grid, metrics=metrics,
        ph_perturbation=inp["PH"], mu_perturbation=inp["MU"],
        theta_full=cpu["THM"].astype(np.float64) + 300.0,
        w=inp["W"], u=inp["U"], v=inp["V"], ht_fine=inp["HGT"],
        base_pb=cpu["PB"], base_mub=cpu["MUB"], base_phb=cpu["PHB"],
        child_opens_at_start=child_opens,
    )
    return np.asarray(p), np.asarray(mu), cpu, meta


@pytest.mark.parametrize("domain,child_opens", [("d02", True), ("d03", False)])
def test_lead_zero_p_and_mu_match_cpu_wrf_in_blend_ring(wn3, domain, child_opens):
    assert jax.devices()[0].platform == "cpu"
    p, mu, cpu, meta = _start_domain(wn3, domain, child_opens=child_opens)
    ring = _rings(*mu.shape)
    blend = ring < 10
    # REAL(4) round-off (measured oracle: P <= 0.1 Pa, MU <= 8.5e-4 Pa); before R01: MU 0.07, d02 P 44 Pa
    assert np.abs(mu - cpu["MU"])[blend].max() <= 2e-3
    assert np.abs(mu - cpu["MU"]).max() <= 2e-3
    assert np.abs(p - cpu["P"]).max() <= 0.5
    assert np.abs(mu - cpu["MU"])[ring == 0].max() < np.abs(cpu["MU"]).max()  # non-vacuous ring
    if child_opens:
        assert any("post-press_adj" in s for s in meta["surfaces"])


def test_middle_nest_without_child_open_rederive_misses_cpu_wrf_p(wn3):
    """Deletion check (F2): WRF re-derives d02 P after press_adj because d03 opens with it."""
    p, _mu, cpu, _ = _start_domain(wn3, "d02", child_opens=False)
    ring = _rings(*p.shape[1:])
    assert np.abs(p - cpu["P"])[:, ring < 5].max() > 10.0


def test_press_adj_with_inverted_phb_alb_misses_cpu_wrf_mu(wn3, monkeypatch):
    """Deletion check (F1): an ALB inverted from PHB (hypsometric 1) instead of start_em's profile ALB."""

    def inverted(pb, scalars):
        t_init, _alb = original(pb, scalars)
        return t_init, np.asarray(rp._wrf_base_alb_from_loaded_state(
            phb=jnp.asarray(phb), mub=jnp.asarray(mub), metrics=metrics), np.float32)

    original = rp._wrf_start_domain_t_init_alb32
    with Dataset(CPU / f"wrfout_d02_{FRAME}") as c:
        phb, mub = _f32(c, "PHB").astype(np.float64), _f32(c, "MUB").astype(np.float64)
    metrics = rp.load_wrfinput_metrics(CASE / "wrfinput_d02")
    monkeypatch.setattr(rp, "_wrf_start_domain_t_init_alb32", inverted)
    _p, mu, cpu, _ = _start_domain(wn3, "d02", child_opens=True)
    assert np.abs(mu - cpu["MU"]).max() > 0.03


def test_nest_open_timing_follows_namelist_start(wn3):
    from types import SimpleNamespace

    assert rp.wrf_nest_opens_with_parent_start(wn3, "d02", "d03")
    tc = {**wn3.namelist["time_control"], "start_hour": [0, 0, 6]}
    late = SimpleNamespace(namelist={**wn3.namelist, "time_control": tc})
    assert not rp.wrf_nest_opens_with_parent_start(late, "d02", "d03")


def test_loader_flags_only_a_nest_whose_child_opens_with_it(wn3, monkeypatch, tmp_path):
    """Wiring: nested_pipeline passes child_opens_at_start=True for WN3 d02 (child d03, same start)."""
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.integration import nested_pipeline as pipeline

    class Stop(Exception):
        pass

    seen = {}
    real = pipeline.build_replay_case

    def spy(run_dir, *, domain, **kwargs):
        if kwargs.get("live_nest_parent") is None:
            return real(run_dir, domain=domain, **kwargs)
        seen[domain] = kwargs.get("live_nest_child_opens_at_start")
        raise Stop

    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
    monkeypatch.setattr(pipeline, "build_replay_case", spy)
    config = pipeline.NestedPipelineConfig(CASE, tmp_path / "out", tmp_path / "proof", hours=1, max_dom=3)
    with pytest.raises(Stop):
        pipeline._load_domains(config, ("d01", "d02", "d03"))
    assert seen == {"d02": True}
