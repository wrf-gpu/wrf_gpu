"""#14 Noah-MP launch count (CPU).

GPUWRF_NOAHMP_LAYER_LISTS: the snow-water column and the SOILWATER sub-step loop on per-layer
lists of 2-D fields -> bitwise vs the layer-select path. GPUWRF_NOAHMP_COLUMN_KERNELS: those two
plus the canopy/bare Newton loops as single Pallas column kernels (Pallas interpreter on CPU via
GPUWRF_NOAHMP_COLUMN_INTERPRET) -> the registered pristine-WRF REAL gates (NS01 snow) still pass.
Snow checks run eagerly (jax.disable_jit, 1x1 columns): XLA:CPU needs > 30 min to compile the list-form
snow column; the compiled bitwise proof is the GPU evidence (tests/v025/b_core/noahmp_columns_evidence.json).
"""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

GATE = Path(__file__).resolve().parents[1] / "b_noahmp" / "snow_real_gate.py"


def _flags(monkeypatch, lists, cols):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    monkeypatch.setenv("GPUWRF_NOAHMP_LAYER_SELECT", "1")
    monkeypatch.setenv("GPUWRF_NOAHMP_LAYER_LISTS", lists)
    monkeypatch.setenv("GPUWRF_NOAHMP_COLUMN_KERNELS", cols)
    monkeypatch.setenv("GPUWRF_NOAHMP_COLUMN_INTERPRET", cols)
    jax.clear_caches()


def _gate():
    spec = importlib.util.spec_from_file_location("ns01_snow_real_gate_b14", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snow_outputs(gate):
    """Raw outputs of the public noahmp_snow and of the internal SNOWWATER path, all 14 NS01 scenarios."""
    out = {}
    for s in sorted(gate.r4):
        pre, f = gate.r4[s]["PRE"], gate.sp.SCEN_FORCING[s]
        gate.sp.SCEN_TG = f["tg"]
        land = gate.sp.build_land_state(pre)
        imelt = np.zeros((gate.NSNOW + gate.sp.NSOIL, 1, 1), np.int32)
        if s == 6:
            imelt[1, 0, 0] = imelt[2, 0, 0] = 1
        imelt = jnp.asarray(imelt)
        public = gate.snowmod.noahmp_snow(land, gate.sp.build_forcing(f), gate.static, jnp.asarray([[f["qsnow"]]]),
                                          imelt, jnp.asarray([[f["qrain"]]]), gate.sp.DT)
        out[s] = [np.asarray(v) for v in jax.tree.leaves(public)]
        internal = gate.internal_real(land, f, imelt)
        out[(s, "internal")] = [np.asarray(v) for k, v in sorted(internal.items()) if k != "_dtypes"]
    return out


def test_snow_lists_bitwise_and_dispatched(monkeypatch):
    assert jax.devices()[0].platform == "cpu"
    gate = _gate()
    _flags(monkeypatch, "0", "0")
    with jax.disable_jit():
        base = _snow_outputs(gate)
    calls = []
    original = gate.snowmod._snowwater_column_lists
    monkeypatch.setattr(gate.snowmod, "_snowwater_column_lists", lambda *a, **k: calls.append(1) or original(*a, **k))
    _flags(monkeypatch, "1", "0")
    with jax.disable_jit():
        lists = _snow_outputs(gate)
    jax.clear_caches()
    assert calls, "list path not dispatched"
    for key, ref in base.items():
        for i, (a, b) in enumerate(zip(ref, lists[key], strict=True)):
            np.testing.assert_array_equal(b, a, err_msg=f"scenario {key} leaf {i}")


@pytest.mark.parametrize("lists,cols", [("1", "0"), ("1", "1")])
def test_snow_gate_ns01_on_new_paths(monkeypatch, lists, cols):
    gate = _gate()
    _flags(monkeypatch, lists, cols)
    calls = []
    if cols == "1":
        original = gate.snowmod.column_call
        monkeypatch.setattr(gate.snowmod, "column_call", lambda *a, **k: calls.append(k["name"]) or original(*a, **k))
    with jax.disable_jit():
        summary = gate.run_gate()
    jax.clear_caches()
    assert summary["n"] == 14
    assert summary["all_pass"], [s["scenario"] for s in summary["scenarios"] if not s["pass"]]
    if cols == "1":
        assert "b_core_noah_snowwater" in calls


def _soil_case(seed=3, ny=4, nx=8):
    rng = np.random.default_rng(seed)
    f32 = np.float32

    def layers(lo, hi):
        return jnp.asarray(rng.uniform(lo, hi, (4, ny, nx)).astype(f32))

    smcmax = layers(0.40, 0.48)
    sice = jnp.where(layers(0.0, 1.0) > 0.5, layers(0.02, 0.2), 0.0)
    # half the columns ice-saturated (sh2o = smcmax - sice) -> the SSTEP overflow cascade fires in every layer
    saturated = jnp.asarray(rng.uniform(size=(ny, nx)) > 0.5)[None]
    sh2o = jnp.where(saturated, smcmax - sice, jnp.minimum(layers(0.05, 0.45), smcmax - sice))
    smc = sh2o + sice
    zsoil = jnp.broadcast_to(jnp.asarray([-0.1, -0.4, -1.0, -2.0], f32)[:, None, None], (4, ny, nx))
    dzs = jnp.broadcast_to(jnp.asarray([0.1, 0.3, 0.6, 1.0], f32)[:, None, None], (4, ny, nx))
    qinsur = jnp.asarray(np.where(rng.uniform(size=(ny, nx)) > 0.3, rng.uniform(1e-4, 5e-3, (ny, nx)), 0.0).astype(f32))
    return dict(qinsur=qinsur, qseva=jnp.asarray(rng.uniform(0, 2e-8, (ny, nx)).astype(f32)),
                etrani=layers(0.0, 1e-8), sh2o=sh2o, smc=smc, sice=sice, zsoil=zsoil, dzs=dzs,
                smcwtd=jnp.full((ny, nx), 0.3, f32), bexp=layers(4.0, 8.0), smcmax=smcmax,
                smcref=layers(0.30, 0.38), smcwlt=layers(0.05, 0.12), dksat=layers(1e-6, 2e-5),
                dwsat=layers(1e-5, 3e-5), kdt=jnp.full((ny, nx), 3.0, f32), frzx=jnp.full((ny, nx), 0.15, f32),
                slope=jnp.full((ny, nx), 0.1, f32), dt=jnp.float32(18.0))


@pytest.mark.parametrize("cols", ["0", "1"])
def test_soilwater_lists_and_kernel_bitwise(monkeypatch, cols):
    from gpuwrf.physics.noahmp import water_hydro as wh
    case = _soil_case()
    _flags(monkeypatch, "0", "0")
    ref = [np.asarray(v) for v in jax.jit(lambda c: wh._soilwater(**c))(case)]
    calls = []
    original = wh._soilwater_loop_core
    monkeypatch.setattr(wh, "_soilwater_loop_core", lambda *a, **k: calls.append(1) or original(*a, **k))
    _flags(monkeypatch, "1", cols)
    got = [np.asarray(v) for v in jax.jit(lambda c: wh._soilwater(**c))(case)]
    jax.clear_caches()
    assert calls, "list loop not dispatched"
    assert int(np.asarray(case["qinsur"] * 18.0 > case["dzs"][0] * case["smcmax"][0]).sum()) > 0  # 6-trip columns exist
    for name, a, b in zip(("sh2o", "smc", "runsrf", "qdrain", "runsub"), ref, got, strict=True):
        if cols == "0":
            np.testing.assert_array_equal(b, a, err_msg=name)
        else:  # Pallas interpreter: same expressions, XLA:CPU contraction may differ at ulp level
            np.testing.assert_allclose(b, a, rtol=2e-6, atol=1e-12, err_msg=name)


def _snow_case(seed=5, ny=8, nx=8):
    """Real-structured random snow columns (E95: PROD has no snow): every ISNOW state, thin (<=0.1 kg/m2)
    and thick layers, liquid, cold/warm layers, snowfall/rain/frost/sublimation forcing."""
    from gpuwrf.contracts.noahmp_state import NSNOW
    rng = np.random.default_rng(seed)
    f32 = np.float32
    isnow = rng.integers(-3, 1, (ny, nx)).astype(np.int32)
    k = np.arange(NSNOW)[:, None, None]
    active = k >= (NSNOW + isnow)[None]
    snice = np.where(active, np.where(rng.uniform(size=(NSNOW, ny, nx)) > 0.7, rng.uniform(0.0, 0.12, (NSNOW, ny, nx)),
                                      rng.uniform(0.5, 60.0, (NSNOW, ny, nx))), 0.0)
    snliq = np.where(active, rng.uniform(0.0, 0.4, (NSNOW, ny, nx)) * snice, 0.0)
    dz = np.where(active, (snice / 250.0 + snliq / 1000.0) * rng.uniform(1.0, 4.0, (NSNOW, ny, nx)), 0.0)
    stc = np.where(active, rng.uniform(255.0, 273.2, (NSNOW, ny, nx)), 0.0)
    sneqv = (snice + snliq).sum(axis=0) + np.where(isnow == 0, rng.uniform(0.0, 8.0, (ny, nx)), 0.0)
    snowh = dz.sum(axis=0) + np.where(isnow == 0, rng.uniform(0.0, 0.04, (ny, nx)), 0.0)
    wx = snice + snliq
    ficeold = np.where(wx > 0, snice / np.where(wx > 0, wx, 1.0), 0.0)

    def a(x, dtype=f32):
        return jnp.asarray(np.asarray(x, dtype))

    return (a(isnow, np.int32), a(snowh), a(sneqv), a(snice), a(snliq), a(rng.uniform(0.1, 0.4, (ny, nx))),
            a(rng.uniform(0.0, 0.05, (ny, nx))), a(stc), a([-0.1, -0.4, -1.0, -2.0]),
            a(np.where(rng.uniform(size=(ny, nx)) > 0.5, rng.uniform(0, 2e-3, (ny, nx)), 0.0)),
            a(rng.uniform(0, 2e-5, (ny, nx))), a(rng.uniform(0, 1e-5, (ny, nx))), a(rng.uniform(0, 1e-5, (ny, nx))),
            a(np.where(rng.uniform(size=(ny, nx)) > 0.5, rng.uniform(0, 3e-3, (ny, nx)), 0.0)),
            a(rng.uniform(260.0, 280.0, (ny, nx))), a(ficeold),
            a(rng.integers(0, 3, (NSNOW, ny, nx)), np.int32), a(dz), 18.0)


@pytest.mark.parametrize("cols", ["0", "1"])
def test_snowwater_column_random_lists_bitwise(monkeypatch, cols):
    from gpuwrf.physics.noahmp import snow
    case = _snow_case()
    _flags(monkeypatch, "0", "0")
    with jax.disable_jit():
        ref = [np.asarray(v) for v in snow._snowwater_column(*case)]
    _flags(monkeypatch, "1", cols)
    with jax.disable_jit():
        got = [np.asarray(v) for v in snow._snowwater_column(*case)]
    jax.clear_caches()
    isnow_in, isnow_out = np.asarray(case[0]), ref[0]
    assert (isnow_out != isnow_in).sum() > 3  # combine/divide/new-layer branches actually changed layer counts
    for i, (x, y) in enumerate(zip(ref, got, strict=True)):
        if cols == "0":
            np.testing.assert_array_equal(y, x, err_msg=f"output {i}")
        else:
            np.testing.assert_allclose(y, x, rtol=2e-6, atol=1e-9, err_msg=f"output {i}")


def test_canopy_kernel_matches_xla_on_energy_savepoints(monkeypatch):
    """VEGE_FLUX + BARE_FLUX in one column kernel (sweep 1 peeled, sweeps 2..NITERC in an in-kernel loop)
    vs the unrolled XLA loops on the 11 pristine-WRF ENERGY savepoint columns; the frozen gate passes."""
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("energy_gate_b14", root / "proofs/noahmp/energy_savepoint_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    import json
    from gpuwrf.physics.noahmp import energy
    columns = json.load(open(gate.HERE / "savepoints_energy.json"))["columns"]
    _flags(monkeypatch, "0", "0")
    with jax.disable_jit():
        ref = [gate.run_column(col) for col in columns]
    calls = []
    original = energy.column_call
    monkeypatch.setattr(energy, "column_call", lambda *a, **k: calls.append(k["name"]) or original(*a, **k))
    _flags(monkeypatch, "1", "1")
    with jax.disable_jit():
        got = [gate.run_column(col) for col in columns]
    jax.clear_caches()
    assert "b_core_noah_canopy_flux" in calls
    # Same expressions; compiler contraction differences pass through the |DTV|<=0.01 Newton exit, so the
    # bar is the frozen pristine-WRF gate: the kernel path passes it, and |kernel - XLA| <= 1 % of its tolerance.
    for col, a, b in zip(columns, ref, got, strict=True):
        truth = gate.reference(col)
        for field, (atol, rtol) in gate.TOL.items():
            if field in ("qsfc", "erreng") or truth.get(field) is None:
                continue
            tol = atol + rtol * abs(truth[field])
            assert abs(b[field] - truth[field]) <= tol, (col["name"], field, b[field], truth[field])
            assert abs(b[field] - a[field]) <= 0.01 * tol, (col["name"], field, b[field], a[field], tol)
        assert abs(b["erreng"]) <= gate.TOL["erreng"][0], (col["name"], b["erreng"])


def test_canopy_kernel_history_outputs(monkeypatch):
    """history=True (writer land history: PSN/RSSUN/RSSHA from sweep 1, CHLEAF = last sweep's CVH, CHUC, CHV2)
    reaches the routine inside the column kernel and matches the XLA loop on the ENERGY savepoint columns."""
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("energy_gate_b14h", root / "proofs/noahmp/energy_savepoint_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    import json
    from gpuwrf.kernels.phys_noahmp_columns import column_call
    from gpuwrf.physics.noahmp import energy
    columns = json.load(open(gate.HERE / "savepoints_energy.json"))["columns"]
    _flags(monkeypatch, "0", "0")
    captured, original = [], energy._vege_flux
    monkeypatch.setattr(energy, "_vege_flux", lambda *a, **k: captured.append(a) or original(*a, **k))
    with jax.disable_jit():
        for col in columns:
            gate.run_column(col)
    monkeypatch.setattr(energy, "_vege_flux", original)
    keys = ("chleaf", "chuc", "chv2", "rssun", "rssha", "psnsun", "psnsha")
    active = 0
    with jax.disable_jit():
        for args in captured:
            ref = energy._vege_flux(*args, history=True)
            got = column_call(lambda va: energy._vege_flux(*va, history=True), (args,),
                              grid_shape=jnp.shape(args[0].tv), name="b_core_noah_canopy_flux", interpret=True)
            for key in keys:
                a, b = np.asarray(ref[key], np.float64), np.asarray(got[key], np.float64)
                # same expressions (STOMATA in the peeled ITER==1 sweep); the kernel body is compiled as one
                # program, so values agree to a few ulp, not bitwise (rssun 1033.5396 vs 1033.5393 on CPU)
                assert np.allclose(b, a, rtol=1e-5, atol=1e-12, equal_nan=True), (key, a, b)
            active += bool(np.any(np.asarray(ref["psnsun"]) != 0))
            assert "chleaf" not in energy._vege_flux(*args)  # history-only outputs absent when history is off
    assert captured and active  # photosynthesis is nonzero on some savepoint columns (deletion-sensitive)


def test_column_call_leaves_and_sentinel(monkeypatch):
    from gpuwrf.kernels.phys_noahmp_columns import column_call
    monkeypatch.setenv("GPUWRF_NOAHMP_COLUMN_INTERPRET", "1")
    ny, nx = 3, 5
    a = jnp.arange(ny * nx, dtype=jnp.float32).reshape(ny, nx)
    flag = a > 6.0
    count = (jnp.arange(ny * nx, dtype=jnp.int32) % 3 - 2).reshape(ny, nx)
    scalar = jnp.float32(2.5)
    layered = jnp.ones((4, ny, nx), jnp.float32)

    def fn(x, f, n, s, py, lay):
        del lay
        return {"y": jnp.where(f, x * s + py, -x), "m": (n * 2).astype(jnp.float32), "b": f}

    direct = fn(a, flag, count, scalar, 0.5, layered)
    got = column_call(fn, (a, flag, count, scalar, 0.5, layered), grid_shape=(ny, nx), name="t_cols")
    for k in direct:
        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(direct[k]), err_msg=k)
    with pytest.raises(TypeError, match="non-column leaf"):
        column_call(lambda lay: {"z": lay.sum(axis=0)}, (layered,), grid_shape=(ny, nx), name="t_bad")


def test_column_call_interprets_on_cpu_backend(monkeypatch):
    """Release defaults turn the column kernels ON: on a CPU backend column_call must use the Pallas interpreter
    without GPUWRF_NOAHMP_COLUMN_INTERPRET (GPU runs lower to Triton)."""
    from gpuwrf.kernels.phys_noahmp_columns import column_call
    assert jax.default_backend() == "cpu"
    monkeypatch.delenv("GPUWRF_NOAHMP_COLUMN_INTERPRET", raising=False)
    a = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    got = jax.jit(lambda x: column_call(lambda v: {"y": v * 2.0 + 1.0}, (x,), grid_shape=(3, 5), name="t_cpu"))(a)
    np.testing.assert_array_equal(np.asarray(got["y"]), np.asarray(a * 2.0 + 1.0))
