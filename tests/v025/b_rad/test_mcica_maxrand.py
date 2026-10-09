"""RE01: WRF cldovrlp=2 maximum-random McICA (GPUWRF_RRTMG_MAXRAND) vs a literal REAL4 oracle.

The oracle re-states pristine module_ra_rrtmg_lw.F:2481-2490 (= module_ra_rrtmg_sw.F:1803-1812)
level by level in numpy float32 on the WRF (nsubcol, ncol, nlay) layout, with WRF's serial KISS
draws seeded from the literal REAL4 pmid = (p3d/100.)*1.e2; every product path (KISS lanes,
jump-ahead, serial scan, legacy fp64 conversion) must give the same masks bitwise, and the
flag-off masks must keep the random-overlap graph. Test pressures round-trip exactly through
p/100.*100. and p*0.01*100 in REAL4 (as on all 186 RE01 columns), so the oracle tests the overlap,
not the port's seed arithmetic (XLA folds (p*0.01)*100 to p; a separate finding). The product's
uniform conversion kiss*2.328306e-10+0.5 may be FMA-contracted (one rounding, WRF rounds twice):
draws are checked to one result ulp (2**-24), the chain/threshold bitwise on the product's own draws.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import rad_mcica
from gpuwrf.physics import rrtmg_lw, rrtmg_sw


def _serial_words(seeds, steps):
    s1, s2, s3, s4 = [s.copy() for s in seeds]
    out = []
    for _ in range(steps):
        s1 = s1 * np.uint32(69069) + np.uint32(1327217885)
        s2 ^= s2 << np.uint32(13)
        s2 ^= s2 >> np.uint32(17)
        s2 ^= s2 << np.uint32(5)
        s3 = np.uint32(18000) * (s3 & np.uint32(65535)) + (s3 >> np.uint32(16))
        s4 = np.uint32(30903) * (s4 & np.uint32(65535)) + (s4 >> np.uint32(16))
        out.append(s1 + s2 + (s3 << np.uint32(16)) + s4)
    return np.stack(out)


def _wrf_seeds(p):
    """seed = (pmid - int(pmid)) * 1e9 with play = p3d/100. and pmid = play*1.e2, all REAL4."""
    pmid = (np.asarray(p, np.float32)[:, :4] / np.float32(100.0)).astype(np.float32) * np.float32(100.0)
    frac = pmid - np.trunc(pmid)
    return tuple((frac * np.float32(1.0e9)).astype(np.uint32)[:, i] for i in range(4))


def _wrf_cdf(p, first, nsubcol):
    """WRF kissvec draws as CDF(isubcol, i, ilev): isubcol outer, ilev inner (irng=0)."""
    seeds = _wrf_seeds(p)
    ncol, nlay = p.shape
    words = _serial_words(seeds, first + nsubcol * nlay - 1)[first - 1:]  # (nsubcol*nlay, ncol)
    rand = words.view(np.int32).astype(np.float32) * np.float32(2.328306e-10) + np.float32(0.5)
    return np.transpose(rand.reshape(nsubcol, nlay, ncol), (0, 2, 1)).copy()


def _wrf_maxrand_literal(cdf, cld, *, strict=True):
    """mcica_subcol_gen_lw/_sw: cldmin floor, case(2) chain, isCloudy; real(kind=rb) = REAL4."""
    one = np.float32(1.0)
    cldf = np.where(cld < np.float32(1.0e-20), np.float32(0.0), cld).astype(np.float32)
    cdf = cdf.copy()
    nlay = cdf.shape[-1]
    for ilev in range(1, nlay):
        clear = one - cldf[:, ilev - 1]
        above = cdf[:, :, ilev - 1] > clear if strict else cdf[:, :, ilev - 1] >= clear
        cdf[:, :, ilev] = np.where(above, cdf[:, :, ilev - 1], cdf[:, :, ilev] * clear)
    return cdf >= (one - cldf)[None]


def _wrf_random_literal(cdf, cld):
    one = np.float32(1.0)
    cldf = np.where(cld < np.float32(1.0e-20), np.float32(0.0), cld).astype(np.float32)
    return cdf >= (one - cldf)[None]


def _columns(ncol=32, nlay=45, seed=20261005):
    # ncol*ng multiple of the 128-lane KISS block: CPU interpret mode aliases masked tail lanes (NaN draws)
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(50.0, 101500.0, (ncol, nlay)), axis=1)[:, ::-1].astype(np.float32)
    for _ in range(64):  # nudge seed layers until every REAL4 seed formula agrees (see module doc)
        q = p[:, :4]
        bad = ((q / np.float32(100)).astype(np.float32) * np.float32(100) != q) | \
              ((q * np.float32(0.01)).astype(np.float32) * np.float32(100) != q)
        if not bad.any():
            break
        p[:, :4] = np.where(bad, np.nextafter(q, np.float32(0)), q)
    cf = rng.choice(np.float32([0, 1e-25, 1e-20, 0.05, 0.2, 0.5, 0.75, 0.999999, 1]), size=(ncol, nlay))
    cf[0] = 0.0                                                    # clear column
    cf[1] = np.where(np.arange(nlay) % 7 < 3, np.float32(0.4), np.float32(0.0))  # stacked partial layers
    cf[2, 5:20] = np.float32(1.0)                                  # overcast slab
    return p, cf.astype(np.float32)


def _sw_reduce(cloudy_global, gpoint_mask):
    reduced = np.take(np.transpose(cloudy_global, (1, 2, 0)), np.asarray(rrtmg_sw._SW_GLOBAL_GPOINT_INDEX), axis=-1)
    return reduced.astype(np.float32) * gpoint_mask


GM = np.asarray([[g < cnt for g in range(12)] for cnt in rrtmg_sw._SW_GPOINT_COUNTS], dtype=np.float32)
NSW = sum(rrtmg_sw._SW_GPOINT_COUNTS)


def _product_cdf(p, first, ng, sw=False):
    """The product's own draws as WRF CDF(isubcol, i, ilev), after checking them against WRF's."""
    draws = np.asarray(jax.jit(lambda p: rad_mcica._random_values(p, first=first, ng=ng, legacy_fp64=False, sw=sw))(p))
    wrf = np.transpose(_wrf_cdf(p, first, ng), (1, 2, 0))
    # WRF rounds the product and the sum, an FMA once: at most one result ulp (2**-24 in [0.5, 1))
    assert np.abs(draws.astype(np.float64) - wrf).max() <= 2.0 ** -24
    return np.transpose(draws, (2, 0, 1)).copy()


@pytest.mark.parametrize("kiss_kernel", ["0", "1"])
def test_maxrand_masks_match_wrf_literal(monkeypatch, kiss_kernel):
    monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", kiss_kernel)
    p, cf = _columns()
    lw = jax.jit(lambda p, c: rad_mcica.lw_cloud_mask(p, c, legacy_fp64=False, maxrand=True))(p, cf)
    lw_cdf = _product_cdf(p, 151, 140)
    np.testing.assert_array_equal(np.asarray(lw), np.transpose(_wrf_maxrand_literal(lw_cdf, cf), (1, 2, 0)))
    sw = jax.jit(lambda p, c, g: rad_mcica.sw_cloud_mask(p, c, g, legacy_fp64=False, maxrand=True))(p, cf, GM)
    np.testing.assert_array_equal(np.asarray(sw), _sw_reduce(_wrf_maxrand_literal(_product_cdf(p, 2, NSW, True), cf), GM))
    # the chain changed something on the stacked-cloud columns, nothing on the clear column
    lw_rand = np.transpose(_wrf_random_literal(lw_cdf, cf), (1, 2, 0))
    assert (np.asarray(lw) != lw_rand)[1:3].any()
    np.testing.assert_array_equal(np.asarray(lw)[0], lw_rand[0])


def test_maxrand_mutants_are_detected():
    p, cf = _columns()
    cdf = _wrf_cdf(p, 151, 140)
    ref = _wrf_maxrand_literal(cdf, cf)
    assert (ref != _wrf_random_literal(cdf, cf)).any()           # deletion mutant (no chain)
    # strictness mutant: '>=' differs once a carried CDF sits exactly on 1-cldf
    edge = cdf.copy()
    edge[:, :, 0] = np.float32(0.5)
    cf_edge = cf.copy()
    cf_edge[:, 0] = np.float32(0.5)
    assert (_wrf_maxrand_literal(edge, cf_edge) != _wrf_maxrand_literal(edge, cf_edge, strict=False)).any()
    chain = np.asarray(rad_mcica.max_random_cdf(jnp.asarray(np.transpose(edge, (1, 2, 0))),
                                                jnp.asarray(np.where(cf_edge < 1e-20, 0, cf_edge).astype(np.float32))))
    ref_edge = _wrf_maxrand_literal(edge, cf_edge)
    np.testing.assert_array_equal(chain >= (1 - np.where(cf_edge < 1e-20, 0, cf_edge))[..., None],
                                  np.transpose(ref_edge, (1, 2, 0)))


def _mutant_chain(kind):
    """Pre-registered mutants (mass-opus gate 3) of rad_mcica.max_random_cdf."""
    from jax import lax

    def chain(random, cldf):
        dtype = random.dtype
        clear = jnp.moveaxis(dtype.type(1.0) - cldf.astype(dtype), -1, 0)
        draws = jnp.moveaxis(random, -2, 0)
        if kind == "top_down":
            draws, clear = draws[::-1], clear[::-1]
        thr = clear[1:] if kind == "same_level" else clear[:-1]

        def step(prev, xs):
            draw, t = xs
            above = prev >= t[..., None] if kind == "ge" else prev > t[..., None]
            cur = jnp.where(above, prev, draw * t[..., None])
            return cur, cur

        _, rest = lax.scan(step, draws[0], (draws[1:], thr))
        out = jnp.concatenate((draws[:1], rest), axis=0)
        if kind == "top_down":
            out = out[::-1]
        return jnp.moveaxis(out, 0, -2)
    return chain


@pytest.mark.parametrize("kind", ["ge", "same_level", "top_down", "deleted"])
def test_maxrand_registered_mutants_are_killed(monkeypatch, kind):
    """Each mutant patched into the product changes masks vs the literal oracle. ('chain before the cldmin floor' is an
    equivalent mutant in REAL4: 1 - cldf rounds to 1 for every cldf < 1e-20, so the floored and raw chains coincide.)"""
    p, cf = _columns()
    if kind == "ge":  # force an exact tie CDF(1) == 1 - cldf(1) (Sterbenz: exact for a draw d >= 0.5) under cldf(2) = 0.5
        draws = _product_cdf(p, 151, 140)  # (g, col, lay); cloud fraction does not change the draws
        col = 3
        g = int(np.flatnonzero((draws[:, col, 0] >= 0.5) & (draws[:, col, 1] * draws[:, col, 0] < 0.5))[0])
        cf = cf.copy()
        cf[col, 0] = np.float32(1.0) - draws[g, col, 0]
        cf[col, 1] = np.float32(0.5)
        assert np.float32(1.0) - cf[col, 0] == draws[g, col, 0]
    lit = np.transpose(_wrf_maxrand_literal(_product_cdf(p, 151, 140), cf), (1, 2, 0))
    good = jax.jit(lambda p, c: rad_mcica.lw_cloud_mask(p, c, legacy_fp64=False, maxrand=True))(p, cf)
    np.testing.assert_array_equal(np.asarray(good), lit)
    chain = (lambda random, cldf: random) if kind == "deleted" else _mutant_chain(kind)
    monkeypatch.setattr(rad_mcica, "max_random_cdf", chain)
    bad = jax.jit(lambda p, c: rad_mcica.lw_cloud_mask(p, c, legacy_fp64=False, maxrand=True))(p, cf)
    assert (np.asarray(bad) != lit).any()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_maxrand_serial_and_legacy_paths_agree(monkeypatch, dtype):
    """Serial-scan solver masks (JUMPAHEAD off) == jump-ahead legacy conversion, both with the chain."""
    monkeypatch.setattr(rad_mcica, "_MAXRAND", True)
    monkeypatch.setattr(rrtmg_lw, "_MCICA_JUMPAHEAD", False)
    monkeypatch.setattr(rrtmg_sw, "_MCICA_JUMPAHEAD", False)
    p, cf = _columns()
    p, cf = jnp.asarray(p, dtype), jnp.asarray(cf, dtype)
    lw_ref = jax.jit(lambda p, c: rrtmg_lw._lw_mcica_random_cloud_mask(p, c, output_dtype=dtype))(p, cf)
    lw_new = jax.jit(lambda p, c: rad_mcica.lw_cloud_mask(p, c, output_dtype=dtype, legacy_fp64=True))(p, cf)
    np.testing.assert_array_equal(lw_new, lw_ref)
    gm = jnp.asarray(GM, jnp.float64)
    sw_ref = jax.jit(rrtmg_sw._mcica_random_overlap_mask)(p, cf, gm)
    sw_new = jax.jit(lambda p, c, g: rad_mcica.sw_cloud_mask(p, c, g, legacy_fp64=True))(p, cf, gm)
    np.testing.assert_array_equal(sw_new, sw_ref)
    if dtype == jnp.float32:  # REAL4 inputs: the serial scan path is the chain literal on its own draws
        cdf = np.transpose(np.asarray(jax.jit(lambda p: rad_mcica._random_values(
            p, first=151, ng=140, legacy_fp64=False))(p)), (2, 0, 1))
        lit = _wrf_maxrand_literal(cdf, np.asarray(cf))
        np.testing.assert_array_equal(np.asarray(lw_ref), np.transpose(lit, (1, 2, 0)).astype(np.float32))


def test_flag_off_keeps_random_overlap_graph():
    p, cf = _columns()
    assert rad_mcica._MAXRAND is False
    for fn in (rad_mcica.lw_cloud_mask, lambda p, c, **k: rad_mcica.sw_cloud_mask(p, c, GM, **k)):
        off = jax.jit(lambda p, c: fn(p, c, legacy_fp64=False)).lower(p, cf).as_text()
        on = jax.jit(lambda p, c: fn(p, c, legacy_fp64=False, maxrand=True)).lower(p, cf).as_text()
        assert off.count("stablehlo.while") < on.count("stablehlo.while")
        assert off == jax.jit(lambda p, c: fn(p, c, legacy_fp64=False, maxrand=False)).lower(p, cf).as_text()
    lw_off = jax.jit(lambda p, c: rad_mcica.lw_cloud_mask(p, c, legacy_fp64=False))(p, cf)
    np.testing.assert_array_equal(np.asarray(lw_off), np.transpose(_wrf_random_literal(_product_cdf(p, 151, 140), cf), (1, 2, 0)))


def test_maxrand_flag_is_in_the_cheap_key():
    from gpuwrf.runtime import aot_cheap_key

    assert ("gpuwrf.kernels.rad_mcica", "_MAXRAND") in aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS


@pytest.mark.parametrize(
    "namelist,maxrand,outcome",
    [
        ({"physics": {"ra_lw_physics": [4], "ra_sw_physics": [4]}}, "0", "warn"),
        ({"physics": {"ra_lw_physics": [4], "cldovrlp": [2]}}, "1", "pass"),
        ({"physics": {"ra_lw_physics": [4], "cldovrlp": [1]}}, "0", "pass"),
        ({"physics": {"ra_lw_physics": [4], "cldovrlp": [1]}}, "1", "pass"),
        ({"physics": {"ra_lw_physics": [4], "cldovrlp": [1, 2]}}, "1", "fail"),
        ({"physics": {"ra_lw_physics": [4], "cldovrlp": [0]}}, "0", "fail"),
        ({"physics": {"ra_sw_physics": [4], "cldovrlp": [3]}}, "0", "fail"),
        ({"physics": {"cldovrlp": [4]}}, "1", "fail"),
        ({"physics": {"cldovrlp": [5]}}, "0", "fail"),
        ({"physics": {"ra_lw_physics": [1], "ra_sw_physics": [1], "cldovrlp": [4]}}, "0", "pass"),
    ],
)
def test_cldovrlp_namelist(monkeypatch, namelist, maxrand, outcome):
    from gpuwrf.io import namelist_check

    monkeypatch.setattr(rad_mcica, "_MAXRAND", maxrand == "1")
    if outcome == "fail":
        with pytest.raises(namelist_check.UnsupportedSchemeError, match="cldovrlp"):
            namelist_check.validate_namelist(namelist)
        return
    namelist_check.validate_namelist(namelist)
    warned = any("cldovrlp" in w for w in namelist_check.collect_namelist_warnings(namelist))
    assert warned == (outcome == "warn")


def test_apply_cldovrlp_pins_random_overlap(monkeypatch):
    from gpuwrf.io import namelist_check

    nl = {"physics": {"ra_lw_physics": [4], "ra_sw_physics": [4], "cldovrlp": [1]}}
    monkeypatch.setenv("GPUWRF_RRTMG_MAXRAND", "1")
    monkeypatch.setattr(rad_mcica, "_MAXRAND", False)
    assert namelist_check.apply_cldovrlp(nl) is not None
    assert namelist_check.os.environ["GPUWRF_RRTMG_MAXRAND"] == "0"
    monkeypatch.setenv("GPUWRF_RRTMG_MAXRAND", "1")
    assert namelist_check.apply_cldovrlp({"physics": {"cldovrlp": [2]}}) is None
    assert namelist_check.apply_cldovrlp({"physics": {}}) is None
    monkeypatch.setattr(rad_mcica, "_MAXRAND", True)  # radiation already imported with icld=2
    with pytest.raises(RuntimeError, match="cldovrlp=1"):
        namelist_check.apply_cldovrlp(nl)


def _assert_finite_pair_equal(candidate, reference):
    """E200: reject any non-finite candidate OR reference value before the exact comparison."""
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    assert candidate.shape == reference.shape, (candidate.shape, reference.shape)
    assert np.isfinite(candidate).all(), "non-finite candidate draws"
    assert np.isfinite(reference).all(), "non-finite reference draws"
    np.testing.assert_array_equal(candidate, reference)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["candidate", "reference", "both"])
def test_finite_pair_check_rejects_single_bad_values(bad, where):
    good = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    cand, ref = good.copy(), good.copy()
    if where in ("candidate", "both"):
        cand[1, 2] = bad
    if where in ("reference", "both"):
        ref[1, 2] = bad
    with pytest.raises(AssertionError):
        _assert_finite_pair_equal(cand, ref)
    _assert_finite_pair_equal(good, good.copy())


def test_cpu_kiss_kernel_request_is_finite_on_unaligned_shapes(monkeypatch):
    """ncol*ng % 128 != 0 (9 columns): the CPU path must not return the interpret-mode tail NaNs."""
    monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "1")
    p, _ = _columns(ncol=9)
    for first, ng, sw in ((151, 140, False), (2, NSW, True)):
        r = np.asarray(jax.jit(lambda p: rad_mcica._random_values(p, first=first, ng=ng, legacy_fp64=False, sw=sw))(p))
        monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "0")
        j = np.asarray(jax.jit(lambda p: rad_mcica._random_values(p, first=first, ng=ng, legacy_fp64=False, sw=sw))(p))
        monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "1")
        _assert_finite_pair_equal(r, j)
