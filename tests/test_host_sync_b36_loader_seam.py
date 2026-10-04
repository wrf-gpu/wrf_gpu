"""B36 integration seam: the real nested loader seeds the seven live-child scalar records.

``_load_domains`` must allocate qc/qr/qi/qs/qg/Ni/Nr two-time boundary records
on every live coupled child BEFORE its initial carry is built (carry/AOT
interface), and must leave the root alone.  The loader, the real PROD (2-domain)
and WN3 (3-domain) namelists and the seeding predicate run unchanged; only the
heavy replay-case builder and device/land initializers are substituted, because
``State.zeros`` is GPU-only in this repository.  Deleting the seeding in
``_load_domains`` fails this test (review-b36 deletion check).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf.integration import nested_pipeline
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.runtime.domain_tree import _coupled_forcedown_enabled
from gpuwrf.validation.moving_nest_testbed import (
    _zero_tendencies,
    build_flat_grid,
    build_neutral_state,
)

SPECIES = ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr")
CASES = {  # name: (read-only case dir, domains, run start label)
    "PROD": (Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725"), ("d01", "d02"),
             "2026-07-26_00:00:00"),
    "WN3": (Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run"),
            ("d01", "d02", "d03"), "2026-02-27_18:00:00"),
}


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_real_loader_seeds_live_child_scalar_records(case_name, tmp_path, monkeypatch):
    source, names, start_label = CASES[case_name]
    if not (source / "namelist.input").is_file():
        pytest.skip(f"{case_name} case not mounted")
    run = Gen2Run(source)
    grid = build_flat_grid(nx=9, ny=8, nz=3, dx_m=3000.0)
    carried = {}

    def fake_case(_source, *, domain, **_kwargs):
        return SimpleNamespace(
            run=run, grid=grid, metrics=grid.metrics, tendencies=_zero_tendencies(grid),
            state=build_neutral_state(grid),
            metadata={"run_start_label": start_label},
        )

    class _Carry(SimpleNamespace):
        # Carry stand-in with the OperationalCarry surface the loader touches after seeding
        # (GPUWRF_CARRY_REAL_ALL re-seeds the held leaves through ``replace``).
        def replace(self, **updates):
            return _Carry(**{**vars(self), **updates})

    def capture_carry(state, *_args, **_kwargs):
        carried[len(carried)] = state
        return _Carry(state=state, noahmp_rad=None, cumulus_carry=None,
                      cumulus_tendencies=None, radiation_diagnostics=None)

    monkeypatch.setattr(nested_pipeline, "build_replay_case", fake_case)
    monkeypatch.setattr(nested_pipeline, "load_radiation_static", lambda *_a, **_k: (None, {}))
    monkeypatch.setattr(nested_pipeline, "_domain_gwd_opt", lambda *_a, **_k: 0)
    real_physics_int = nested_pipeline._domain_physics_int
    monkeypatch.setattr(
        nested_pipeline, "_domain_physics_int",
        lambda run_, key, *rest, **kw: 0 if key in ("slope_rad", "topo_shading")
        else real_physics_int(run_, key, *rest, **kw),
    )
    monkeypatch.setattr(nested_pipeline, "_domain_sf_surface_physics", lambda *_a, **_k: 0)
    monkeypatch.setattr(nested_pipeline, "_initial_carry_for_run", capture_carry)
    monkeypatch.setattr(nested_pipeline, "_commit_to_operational_device", lambda carry, *_a, **_k: carry)
    monkeypatch.setattr(nested_pipeline, "_root_boundary_cadence_override", lambda nml, *_a, **_k: nml)
    import gpuwrf.io.lower_boundary as lower_boundary

    monkeypatch.setattr(lower_boundary, "load_lower_boundary", lambda *_a, **_k: None)
    config = NestedPipelineConfig(
        input_dir=source, output_dir=tmp_path / "out", proof_dir=tmp_path / "proof",
        hours=1, max_dom=len(names),
    )
    _hierarchy, bundles, *_rest = _load_domains(config, names)

    root = bundles[names[0]].state
    assert all(getattr(root, f"{n}_bdy") is None for n in SPECIES)
    for name in names[1:]:
        bundle = bundles[name]
        assert _coupled_forcedown_enabled(bundle.namelist), name
        qv = bundle.state.qv_bdy
        for species in SPECIES:
            leaf = getattr(bundle.state, f"{species}_bdy")
            assert leaf is not None, (name, species)
            assert tuple(leaf.shape) == (2, *qv.shape[1:]), (name, species, leaf.shape)
            # State contract: the seven scalar records are FP32-gated WRF REAL.
            assert str(leaf.dtype) == "float32", (name, species, leaf.dtype)
    # Seeding happens before the initial carry (fixed carry/AOT interface).
    seeded = [s for s in carried.values() if s.qc_bdy is not None]
    assert len(seeded) == len(names) - 1
