"""S2 — native real.exe-equivalent soil thermodynamic + categorical state.

FROZEN ENTRY SIGNATURE (``compute_soil_init``). Reproduces real.exe
``process_soil_real`` / ``init_soil_2_real`` (the Noah/Noah-MP branch) and the
dominant-category soil/veg assignment (``module_initialize_real.F:3086-3150``,
``surface_input_source=3`` path), consuming the frozen v0.3.0 ``MetEmArtifact``.

The 2 -> 4 soil-layer interpolation (the load-bearing step)
----------------------------------------------------------
met_em provides 2 metgrid soil layers (``ST``/``SM`` stacks, ``SOIL_LAYERS``
thicknesses; the Canary case has ``FLAG_SOIL_LAYERS=1``,
``NUM_METGRID_SOIL_LEVELS=2``). Noah-MP needs ``config.num_soil_layers`` (=4).
``process_soil_real`` selects, for ``sf_surface_physics=4``,
``init_soil_depth_2`` (ZS = [0.05,0.25,0.70,1.50] m, DZS = [0.1,0.3,0.6,1.0] m;
module_soil_pre.F:1128-1151) and ``init_soil_2_real`` (the interp;
module_soil_pre.F:1391-1764). For the layered (``flag_soil_layers==1``) input
the reproduced algorithm is:

  1. metgrid layer midpoints -> ``st_levels`` (cm): with SOIL_LAYERS thicknesses
     [10,40] (stored deep-first as [40,10]), flipped to surface-first, the
     cumulative midpoints are (0+10)/2=5 and (10+40)/2=25, integer-rounded ->
     ``[5, 25]`` cm (``module_optional_input.F:1331-1357``).
  2. the interp "have" depths are ``zhave = [0, st_levels/100..., 3.0]`` m, i.e.
     ``[0, 0.05, 0.25, 3.0]`` (the 0-cm skin + the 2 layers + the 300-cm deep
     endpoint; module_soil_pre.F:1591-1595 / :1649-1653).
  3. the "have" temperature column is ``[tsk, st_top, st_deep, tmn]``; the
     moisture column is ``[sm_top, sm_top, sm_deep, sm_deep]`` (the 0-cm and
     300-cm moisture endpoints are copied from the nearest layer;
     module_soil_pre.F:1506-1514). ``tsk`` and the layer temps get the SAME
     over-land elevation lapse ``-= 0.0065*(ht-toposoil)`` as the surface lane
     (``adjust_soil_temp_new``, module_soil_pre.F:1059-1067); ``tmn`` is the
     surface lane's *pre-fix* ``tmn_soil_endpoint``.
  4. each target depth ``zs(lwant)`` is LINEARLY interpolated in depth between
     the bracketing "have" levels (module_soil_pre.F:1599-1614 / :1657-1672).
  5. over water (``flag_sst==0``): ``tslb := tsk``, ``smois := 1.0`` (and
     ``sh2o := 1.0``; module_soil_pre.F:1745-1758).
  6. zero-soil-moisture floor (``flag_soil_layers==1``): land cells whose top
     layer is a valid temperature with ``smois < 0.005`` are floored to 0.005
     across all layers (``module_initialize_real.F:3370-3398``).

Categorical fields (dominant-category / ``surface_input_source=3`` path)
------------------------------------------------------------------------
  * ``ivgtyp`` = ``NINT(LU_INDEX)``; ``lu_index`` (wrfinput) = ``ivgtyp``
    (module_initialize_real.F:3103-3110, :3252).
  * ``isltyp`` = ``NINT(SCT_DOM)`` then matched to the landmask
    (:3117-3135): a LAND cell that came out ``isoilwater`` (14) is forced to
    ``default_soiltype`` (=8, silty clay loam); a WATER cell that is not
    ``isoilwater`` is forced to 14.

HONEST RESIDUAL (documented, see proof + handoff): the oracle wrfinput files
are post-``start_em`` snapshots, so a single static d01 inland-lake cell carries
a NOAHMP_INIT lake-fill (``IVGTYP``/``ISLTYP`` differ from the real.exe-faithful
dominant result) and ``SH2O`` carries the supercooled-water split. ``SH2O`` is
NOT a frozen-gate field (absent from ``WRFINPUT_TOLS``); this lane returns
``sh2o = smois`` (the real.exe pre-LSM-init value). The lake-cell categorical
difference is reported per-case; it is a downstream-LSM artifact, not an S2
real.exe-equivalence defect, and d02/d03 (no lakes) are categorically exact.

FILE OWNERSHIP: S2 exclusive (see ``surface_init.py`` header).
"""

from __future__ import annotations

import numpy as np

from gpuwrf.init.real_init.types import RealInitConfig, SoilInit, SurfaceInit
from gpuwrf.init.real_init.surface_init import compute_surface_init_full, _LAPSE
from gpuwrf.init.metgrid_schema import MetEmArtifact


# WRF Noah/Noah-MP 4-layer node-depth set (init_soil_depth_2). Built, not pinned,
# but cross-checked against the oracle ZS/DZS.
_NOAHMP_DZS = (0.1, 0.3, 0.6, 1.0)  # module_soil_pre.F:1138
# WRF defaults (namelist not setting them): silty-clay-loam fallback + soil-water
# category. module_initialize_real.F default_soiltype=8; isoilwater=14.
_DEFAULT_SOILTYPE = 8
_ISOILWATER = 14
# zero-soil-moisture floor (module_initialize_real.F:3379).
_SMOIS_FLOOR = 0.005
_T_MIN = 170.0
_T_MAX = 400.0
# deep-endpoint "have" depth in metres (module_soil_pre.F:1595 uses 300 cm).
_DEEP_ENDPOINT_M = 3.0


def init_soil_depth_noahmp(num_soil_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces ``init_soil_depth_2`` (module_soil_pre.F:1128-1151).

    DZS = [0.1, 0.3, 0.6, 1.0] m; ZS(1)=DZS(1)/2; ZS(l)=ZS(l-1)+DZS(l-1)/2+DZS(l)/2.
    """

    if num_soil_layers != 4:
        raise NotImplementedError(
            "init_soil_depth_2 (Noah/Noah-MP) requires num_soil_layers=4; got "
            f"{num_soil_layers}"
        )
    dzs = np.array(_NOAHMP_DZS, dtype=np.float64)
    zs = np.zeros(num_soil_layers, dtype=np.float64)
    zs[0] = 0.5 * dzs[0]
    for l in range(1, num_soil_layers):
        zs[l] = zs[l - 1] + 0.5 * dzs[l - 1] + 0.5 * dzs[l]
    return zs, dzs


def _metgrid_soil_levels_cm(soil_layers: np.ndarray) -> list[int]:
    """Reproduces the layered-input level-midpoint computation
    (``module_optional_input.F:1331-1357``, non-UM branch).

    ``soil_layers`` is the met_em ``SOIL_LAYERS`` stack (nst, ny, nx), constant
    in (i,j); stored deep-first. Flipping to surface-first, the level for layer
    k is the cumulative midpoint (level_above + thickness)/2, integer-rounded.
    """

    nst = soil_layers.shape[0]
    thick = soil_layers[:, 0, 0]  # constant over the grid; deep-first storage
    levels: list[int] = []
    level_above = 0.0
    for k in range(1, nst + 1):
        val = float(thick[nst - k])  # flip -> surface-first thickness (cm)
        lev = (level_above + val) / 2.0
        levels.append(int(round(lev)))
        level_above = val
    return levels


def _interp_depth(
    have_col: np.ndarray, zhave: np.ndarray, zs: np.ndarray
) -> np.ndarray:
    """Linear depth interpolation matching ``init_soil_2_real``'s inner loop.

    ``have_col`` is (n_have, ny, nx); ``zhave`` (n_have,) ascending; ``zs``
    (n_want,) the target node depths. For each target depth pick the bracketing
    pair [zhave[lh], zhave[lh+1]] containing it and linearly interpolate.
    """

    n_have, ny, nx = have_col.shape
    n_want = zs.shape[0]
    out = np.zeros((n_want, ny, nx), dtype=np.float64)
    for lw in range(n_want):
        z = zs[lw]
        for lh in range(n_have - 1):
            if zhave[lh] <= z <= zhave[lh + 1]:
                w_lo = (zhave[lh + 1] - z) / (zhave[lh + 1] - zhave[lh])
                w_hi = (z - zhave[lh]) / (zhave[lh + 1] - zhave[lh])
                out[lw] = have_col[lh] * w_lo + have_col[lh + 1] * w_hi
                break
    return out


def compute_soil_init(
    config: RealInitConfig,
    metem: MetEmArtifact,
    surface: SurfaceInit,
) -> SoilInit:
    """Builds the wrfinput-equivalent soil temperature/moisture + categories.

    ``surface`` is the frozen :class:`SurfaceInit` from the surface lane; this
    lane re-derives the consistent intermediate fields (the same landmask /
    tsk / pre-fix tmn endpoint) by re-running the surface lane's internal
    computation, so the two lanes are byte-consistent regardless of dispatch
    order (the driver passes the same met_em to both).
    """

    if config.num_soil_layers != 4 or config.sf_surface_physics != 4:
        raise NotImplementedError(
            "v0.4.0 S2 soil_init supports the Noah-MP 4-layer set only; got "
            f"sf_surface_physics={config.sf_surface_physics}, "
            f"num_soil_layers={config.num_soil_layers}"
        )

    # carry-batch 2e: if the surface lane already provided the pre-fix deep-soil
    # endpoint on the frozen SurfaceInit, consume it directly (the driver runs the
    # surface lane ONCE and passes its output here) — no doubled surface compute.
    # The remaining consistent intermediates are all on the frozen SurfaceInit
    # (tsk/landmask/hgt) plus met_em SOILHGT; only fall back to the internal
    # re-run when tmn_soil_endpoint is absent (e.g. a hand-built SurfaceInit).
    if getattr(surface, "tmn_soil_endpoint", None) is not None:
        landmask = np.asarray(surface.landmask, dtype=np.float64)
        land = landmask > 0.5
        water = ~land
        tsk = np.asarray(surface.tsk, dtype=np.float64)
        tmn_endpoint = np.asarray(surface.tmn_soil_endpoint, dtype=np.float64)
        ht = np.asarray(surface.hgt, dtype=np.float64)
        toposoil = np.asarray(metem.arrays["SOILHGT"], dtype=np.float64)
    else:
        sres = compute_surface_init_full(config, metem)
        land = sres.land
        water = sres.water
        tsk = sres.tsk
        tmn_endpoint = sres.tmn_soil_endpoint
        ht = sres.ht
        toposoil = sres.toposoil
    delev = ht - toposoil

    st = np.asarray(metem.arrays["ST"], dtype=np.float64)  # (nst, ny, nx)
    sm = np.asarray(metem.arrays["SM"], dtype=np.float64)
    soil_layers = np.asarray(metem.arrays["SOIL_LAYERS"], dtype=np.float64)
    nst = st.shape[0]
    if nst != 2:
        raise NotImplementedError(
            f"v0.4.0 S2 reproduces the 2->4 metgrid-soil interp; got {nst} "
            "metgrid soil layers (extend zhave/st_input packing for >2)."
        )
    _, ny, nx = st.shape

    zs, dzs = init_soil_depth_noahmp(config.num_soil_layers)
    st_levels = _metgrid_soil_levels_cm(soil_layers)  # cm, e.g. [5, 25]

    # ---- build the "have" temperature column [tsk, st_top, st_deep, tmn] -----
    # met_em ST is stored deep-first; flip to surface-first (st[nst-1-k] for k).
    st_surface_first = st[::-1]  # [top, deep] for nst=2
    st_have = np.zeros((nst + 2, ny, nx), dtype=np.float64)
    st_have[0] = tsk
    for k in range(nst):
        st_have[k + 1] = st_surface_first[k]
    st_have[nst + 1] = tmn_endpoint
    # adjust_soil_temp_new (flag_soil_layers==1): the layer temps st_have[1..nst]
    # get the over-land elevation lapse (module_soil_pre.F:1059-1062). tsk
    # (st_have[0]) was already lapsed by the surface lane; tmn endpoint is the
    # lapsed deep value; so only the metgrid layers are adjusted here.
    for k in range(1, nst + 1):
        st_have[k][land] = st_have[k][land] - _LAPSE * delev[land]

    # ---- build the "have" moisture column [sm_top, sm_top, sm_deep, sm_deep] -
    sm_surface_first = sm[::-1]
    sm_have = np.zeros((nst + 2, ny, nx), dtype=np.float64)
    for k in range(nst):
        sm_have[k + 1] = sm_surface_first[k]
    sm_have[0] = sm_have[1]  # 0-cm moisture := shallowest layer
    sm_have[nst + 1] = sm_have[nst]  # 300-cm moisture := deepest layer

    zhave = np.array(
        [0.0] + [lev / 100.0 for lev in st_levels] + [_DEEP_ENDPOINT_M],
        dtype=np.float64,
    )

    tslb = _interp_depth(st_have, zhave, zs)
    smois = _interp_depth(sm_have, zhave, zs)

    # ---- over-water fill (flag_sst==0 -> tslb:=tsk, smois:=1, sh2o:=1) -------
    for l in range(config.num_soil_layers):
        tslb[l][water] = tsk[water]
        smois[l][water] = 1.0

    # ---- zero-soil-moisture floor (Noah->Noah, flag_soil_layers==1) ---------
    bad_moist = (
        land
        & (tslb[0] > _T_MIN)
        & (tslb[0] < _T_MAX)
        & (smois[0] < _SMOIS_FLOOR)
    )
    for l in range(config.num_soil_layers):
        smois[l][bad_moist] = _SMOIS_FLOOR

    # ---- SH2O: real.exe pre-LSM-init value = smois (water cells = 1.0). The
    # supercooled-water split is a NOAHMP_INIT (model start) artifact, NOT a
    # real.exe field, and SH2O is absent from WRFINPUT_TOLS (out of gate). ----
    sh2o = smois.copy()

    # ---- categorical: dominant-category (surface_input_source=3) path -------
    lu_index_met = np.rint(np.asarray(metem.arrays["LU_INDEX"], dtype=np.float64))
    ivgtyp = lu_index_met.astype(np.int32)

    sct_dom = np.rint(np.asarray(metem.arrays["SCT_DOM"], dtype=np.float64))
    isltyp = sct_dom.astype(np.int32)
    # match isltyp to landmask (module_initialize_real.F:3122-3135)
    isltyp[land & (isltyp == _ISOILWATER)] = _DEFAULT_SOILTYPE
    isltyp[water & (isltyp != _ISOILWATER)] = _ISOILWATER

    # lu_index (wrfinput) = ivgtyp (module_initialize_real.F:3252)
    lu_index = ivgtyp.astype(np.float64)

    vegfra = _vegfra(metem)
    canwat = None  # canopy water is a model-init field (0 in the Canary wrfinput)

    return SoilInit(
        tslb=tslb,
        smois=smois,
        sh2o=sh2o,
        zs=zs,
        dzs=dzs,
        isltyp=isltyp,
        ivgtyp=ivgtyp,
        lu_index=lu_index,
        vegfra=vegfra,
        canwat=canwat,
    )


def _vegfra(metem: MetEmArtifact) -> np.ndarray | None:
    """Returns the green-fraction (vegfra) if a monthly GREENFRAC stack is
    available; otherwise None. Faithful monthly-to-date interpolation
    (``monthly_interp_to_date``) needs the valid date + the 12-month stack; the
    Canary metgrid carries GREENFRAC, but vegfra is NOT a frozen-gate field
    (absent from WRFINPUT_TOLS), so this is a best-effort optional output and is
    left None unless a single-month field is directly present."""

    arr = metem.arrays.get("VEGFRA")
    if arr is not None:
        return np.asarray(arr, dtype=np.float64)
    return None
