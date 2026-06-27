"""Noah-MP snow water / compaction / albedo aging (Sprint S3).

Faithful JAX port of the pristine-WRF Noah-MP snow column
(``/home/user/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F``):

  SNOWWATER  (:6398-6535)  driver: SNOWFALL -> COMPACT -> COMBINE -> DIVIDE ->
                           SNOWH2O, then empty-layer zeroing + ZSNSO/DZSNSO rebuild
  SNOWFALL   (:6539-6606)  new-snowfall accumulation + first-layer creation
  COMPACT    (:6974-7081)  destructive/overburden/melt metamorphism (multi-layer)
  COMBINE    (:6610-6788)  thin-layer collapse (COMBO enthalpy merge, ISNOW<-1)
  DIVIDE     (:6792-6916)  layer subdivision into <=3 layers
  COMBO      (:6920-6970)  enthalpy-conserving two-node merge
  SNOWH2O    (:7085-7230)  sublimation/frost + liquid percolation -> QSNBOT
  SNOW_AGE   (:3119-3167)  BATS/Yang97 non-dimensional snow age (TAUSS)
  SNOWALB_CLASS (:3226-3275, opt_alb=2)  CLASS broadband snow albedo -> ALBOLD

Active options: opt_snf=1 (Jordan rain/snow handled upstream; SNOWHIN/QSNOW are
inputs), opt_alb=2 (CLASS albedo). NSNOW=3, ISNOW in {0,-1,-2,-3}.

GPU REQUIREMENT — BRANCH-FREE MASKED KERNEL
-------------------------------------------
The variable snow-layer count (ISNOW in {0,-1,-2,-3}) is realised WITHOUT any
python-level layer-count branching. Every column carries fixed length-3 snow
arrays whose *active* slots are the WRF indices ``ISNOW+1 .. 0`` mapped to the
top-aligned local positions ``0 .. NSNOW-1``. A per-layer boolean mask (derived
purely from ``isnow``) gates every update with ``jnp.where``; empty slots hold
exact zeros. COMBINE / DIVIDE are expressed as masked, fixed-trip-count layer
operations so the whole routine is one vmapped, jit-friendly kernel with no data
dependent control flow. fp64 throughout.

Layer index convention (per column)
------------------------------------
WRF stores snow in STC/SNICE/SNLIQ/DZSNSO at indices ``-NSNOW+1 .. 0`` with the
*surface* layer at index 0 and the active layers contiguous at the bottom of that
range (``ISNOW+1 .. 0``). We use a fixed local index ``k = 0 .. NSNOW-1`` that is
**top-aligned**: the surface layer is ALWAYS local k=NSNOW-1 (WRF index 0), and
active layers occupy ``k = NSNOW+ISNOW .. NSNOW-1``. ``active[k] = k >= NSNOW +
isnow``. When ISNOW=0 no slot is active; when ISNOW=-1 only k=NSNOW-1 (the
surface) is active; when ISNOW=-3 all 3 slots are active. WRF layer index of
local k = ``ISNOW+1 + (k - (NSNOW+ISNOW))`` for active k.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPForcing

configure_jax_x64()

# --- pristine-WRF physical constants (module_sf_noahmplsm.F:204-220) ---------
TFRZ = 273.16        # freezing/melting point [K]
HFUS = 0.3336e06     # latent heat of fusion [J/kg]
CWAT = 4.188e06      # volumetric heat capacity of water [J/m3/K]
CICE = 2.094e06      # volumetric heat capacity of ice [J/m3/K]
DENH2O = 1000.0      # density of water [kg/m3]
DENICE = 917.0       # density of ice [kg/m3]

# --- COMPACT parameters (module_sf_noahmplsm.F:7000-7007) --------------------
C2 = 21.0e-3         # [m3/kg]
C3 = 2.5e-6          # [1/s]
C4 = 0.04            # [1/K]
C5 = 2.0
DM = 100.0           # destructive-metamorphism density cap [kg/m3]
ETA0 = 1.33e06       # viscosity coefficient [kg-s/m2] (He et al. 2021)

# --- DIVIDE / COMBINE limits (module_sf_noahmplsm.F:6650, :6839, :6866, :6888)
DZMIN = (0.025, 0.025, 0.1)   # COMBINE minimum top-layer thickness per slot
DZ_SPLIT1 = 0.05              # DIVIDE: target top-layer thickness [m]
DZ_SPLIT2 = 0.20             # DIVIDE: target second-layer thickness [m]

# --- SNOWH2O (module_sf_noahmplsm.F:7135) ------------------------------------
MAX_LIQ_MASS_FRACTION = 0.4

# --- scoped Noah-MP parameter defaults (run/MPTABLE.TBL) ---------------------
# Non-category snow scalars; pinned here so snow.py oracle-tests without the
# (separately owned) tables.py loader. The driver/coupler will pass them through
# ``static.parameters`` once tables.py lands; until then snow uses these WRF
# defaults (opt_snf=1 / opt_alb=2 corpus namelist).
SSI = 0.03                # liquid water holding capacity [m3/m3]
SNOW_RET_FAC = 5.0e-5     # snowpack water-release timescale factor [1/s]
SWEMX = 1.00              # new-snow mass to fully cover old snow [mm]
TAU0 = 1.0e6              # Yang97 eqn.10a
GRAIN_GROWTH = 5000.0     # Yang97 eqn.10b
EXTRA_GROWTH = 10.0       # Yang97 eqn.10c
DIRT_SOOT = 0.3           # Yang97 eqn.10d


def _combo(dz1, wliq1, wice1, t1, dz2, wliq2, wice2, t2):
    """Enthalpy-conserving merge of two snow nodes (COMBO, :6920-6970).

    Returns combined ``(dz, wliq, wice, t)``. Branch-free over the three enthalpy
    regimes via ``jnp.where``. fp64.
    """
    dzc = dz1 + dz2
    wicec = wice1 + wice2
    wliqc = wliq1 + wliq2
    h = (CICE * wice1 + CWAT * wliq1) * (t1 - TFRZ) + HFUS * wliq1
    h2 = (CICE * wice2 + CWAT * wliq2) * (t2 - TFRZ) + HFUS * wliq2
    hc = h + h2
    cap = CICE * wicec + CWAT * wliqc
    cap_safe = jnp.where(cap > 0.0, cap, 1.0)
    tc_cold = TFRZ + hc / cap_safe
    tc_warm = TFRZ + (hc - HFUS * wliqc) / cap_safe
    tfrz = jnp.asarray(TFRZ, hc.dtype)
    tc = jnp.where(hc < 0.0, tc_cold, jnp.where(hc <= HFUS * wliqc, tfrz, tc_warm))
    return dzc, wliqc, wicec, tc


def _active_mask(isnow):
    """Top-aligned per-layer active mask: local k active iff k >= NSNOW+isnow."""
    ks = jnp.arange(NSNOW).reshape((NSNOW,) + (1,) * isnow.ndim)
    return ks >= (NSNOW + isnow)[None, ...]


def _snowfall(isnow, snowh, sneqv, dzsnso, stc, snice, snliq,
              qsnow, snowhin, sfctmp, dt):
    """SNOWFALL (:6539-6606), branch-free. Surface layer is local NSNOW-1."""
    surf = NSNOW - 1
    dtype = stc.dtype
    no_layer = isnow == 0

    # shallow snow / no layer: accumulate bulk SNOWH/SNEQV (:6579-6582)
    add_bulk = no_layer & (qsnow > 0.0)
    snowh = jnp.where(add_bulk, snowh + snowhin * dt, snowh)
    sneqv = jnp.where(add_bulk, sneqv + qsnow * dt, sneqv)

    # create a new layer when bulk depth >= 0.025 m (:6588-6596)
    newnode = no_layer & (snowh >= 0.025)
    isnow = jnp.where(newnode, jnp.int32(-1), isnow)
    dzsnso = dzsnso.at[surf].set(jnp.where(newnode, snowh, dzsnso[surf]))
    stc = stc.at[surf].set(jnp.where(newnode, jnp.minimum(TFRZ, sfctmp), stc[surf]))
    snice = snice.at[surf].set(jnp.where(newnode, sneqv, snice[surf]))
    snliq = snliq.at[surf].set(jnp.where(newnode, jnp.asarray(0.0, dtype), snliq[surf]))
    snowh = jnp.where(newnode, jnp.asarray(0.0, dtype), snowh)

    # snow onto an existing top layer (WRF ISNOW+1 = local NSNOW+isnow) (:6600-6603)
    add_to_top = (isnow < 0) & (~newnode) & (qsnow > 0.0)
    top = NSNOW + isnow                    # local index of WRF layer ISNOW+1
    for k in range(NSNOW):
        hit = add_to_top & (top == k)
        snice = snice.at[k].set(jnp.where(hit, snice[k] + qsnow * dt, snice[k]))
        dzsnso = dzsnso.at[k].set(jnp.where(hit, dzsnso[k] + snowhin * dt, dzsnso[k]))
    return isnow, snowh, sneqv, dzsnso, stc, snice, snliq


def _compact(isnow, dt, stc, snice, snliq, imelt, ficeold, dzsnso):
    """COMPACT (:6974-7081). Multi-layer only (gated by mask), branch-free.

    Sweeps WRF top->surface (local 0 -> NSNOW-1) accumulating BURDEN. The
    per-layer body is masked to active, non-saturated, ice-bearing nodes exactly
    as the Fortran inner IF (:7032).
    """
    multilayer = isnow < 0
    active = _active_mask(isnow)
    burden = jnp.zeros_like(dzsnso[0])
    dz_new = dzsnso
    for k in range(NSNOW):
        wx = snice[k] + snliq[k]
        wx_safe = jnp.where(wx > 0.0, wx, 1.0)
        fice = snice[k] / wx_safe
        dz_k = dzsnso[k]
        dz_safe = jnp.where(dz_k > 0.0, dz_k, 1.0)
        void = 1.0 - (snice[k] / DENICE + snliq[k] / DENH2O) / dz_safe

        do_compact = active[k] & multilayer & (void > 0.001) & (snice[k] > 0.1)

        bi = snice[k] / dz_safe
        td = jnp.maximum(0.0, TFRZ - stc[k])
        dexpf = jnp.exp(-C4 * td)
        ddz1 = -C3 * dexpf
        ddz1 = jnp.where(bi > DM, ddz1 * jnp.exp(-46.0e-3 * (bi - DM)), ddz1)
        ddz1 = jnp.where(snliq[k] > 0.01 * dz_k, ddz1 * C5, ddz1)
        ddz2 = -(burden + 0.5 * wx) * jnp.exp(-0.08 * td - C2 * bi) / ETA0
        ficeold_safe = jnp.maximum(1.0e-6, ficeold[k])
        ddz3_melt = jnp.maximum(0.0, (ficeold[k] - fice) / ficeold_safe)
        ddz3 = jnp.where(imelt[k] == 1, -ddz3_melt / dt, 0.0)
        pdzdtc = jnp.maximum(-0.5, (ddz1 + ddz2 + ddz3) * dt)

        dz_compacted = dz_k * (1.0 + pdzdtc)
        dz_compacted = jnp.maximum(dz_compacted, snice[k] / DENICE + snliq[k] / DENH2O)
        dz_compacted = jnp.minimum(
            jnp.maximum(dz_compacted, (snice[k] + snliq[k]) / 500.0),
            (snice[k] + snliq[k]) / 50.0,
        )
        dz_new = dz_new.at[k].set(jnp.where(do_compact, dz_compacted, dz_k))
        burden = burden + wx
    return dz_new


def _combine(isnow, sh2o0, sice0, stc, snice, snliq, dzsnso, snowh, sneqv, dz1_soil):
    """COMBINE (:6610-6788). Branch-free, fixed trip count.

    Phase 1: collapse thin/empty layers (SNICE<=0.1), merging into a neighbour and
    shifting the column. Phase 2: re-sum SWE/SNOWH and, if total depth < 0.025 m,
    collapse to ISNOW=0 (ponding). Phase 3: COMBO-based DZMIN re-merge (ISNOW<-1).
    Only the top soil layer (``sh2o0``/``sice0``) is mutated (over-sublimation).
    """
    surf = NSNOW - 1
    dtype = stc.dtype
    ponding1 = jnp.zeros_like(snowh)
    ponding2 = jnp.zeros_like(snowh)
    ks = jnp.arange(NSNOW).reshape((NSNOW,) + (1,) * isnow.ndim)
    big = NSNOW + 5

    # ---- Phase 1: thin-layer (SNICE<=0.1) collapse & shift -------------------
    for _ in range(NSNOW):
        active = _active_mask(isnow)
        removable = active & (snice <= 0.1)
        first_k = jnp.min(jnp.where(removable, ks, big), axis=0)
        has_removable = first_k < big

        j_is_surface = first_k == surf
        merge_down = has_removable & (~j_is_surface)
        merge_up = has_removable & j_is_surface & (isnow < -1)
        clear_single = has_removable & j_is_surface & (isnow == -1)

        for k in range(NSNOW):
            into_down = merge_down & (first_k == k)
            into_up = merge_up & (first_k == k)
            if k + 1 < NSNOW:
                nb = k + 1
                snliq = snliq.at[nb].set(jnp.where(into_down, snliq[nb] + snliq[k], snliq[nb]))
                snice = snice.at[nb].set(jnp.where(into_down, snice[nb] + snice[k], snice[nb]))
                dzsnso = dzsnso.at[nb].set(jnp.where(into_down, dzsnso[nb] + dzsnso[k], dzsnso[nb]))
            if k - 1 >= 0:
                nb = k - 1
                snliq = snliq.at[nb].set(jnp.where(into_up, snliq[nb] + snliq[k], snliq[nb]))
                snice = snice.at[nb].set(jnp.where(into_up, snice[nb] + snice[k], snice[nb]))
                dzsnso = dzsnso.at[nb].set(jnp.where(into_up, dzsnso[nb] + dzsnso[k], dzsnso[nb]))

        # case C: single surface layer -> ponding (:6667-6682)
        snice_pos = clear_single & (snice[surf] >= 0.0)
        snice_neg = clear_single & (snice[surf] < 0.0)
        ponding1 = jnp.where(snice_pos, snliq[surf], ponding1)
        sneqv = jnp.where(snice_pos, snice[surf], sneqv)
        snowh = jnp.where(snice_pos, dzsnso[surf], snowh)
        pond_neg = snliq[surf] + snice[surf]
        sub_to_soil = snice_neg & (pond_neg < 0.0)
        sice0 = jnp.where(sub_to_soil, sice0 + pond_neg / (dz1_soil * 1000.0), sice0)
        ponding1 = jnp.where(snice_neg, jnp.where(pond_neg < 0.0, jnp.asarray(0.0, dtype), pond_neg), ponding1)
        sneqv = jnp.where(snice_neg, jnp.asarray(0.0, dtype), sneqv)
        snowh = jnp.where(snice_neg, jnp.asarray(0.0, dtype), snowh)

        for k in range(NSNOW):
            hit = has_removable & (first_k == k)
            snliq = snliq.at[k].set(jnp.where(hit, jnp.asarray(0.0, dtype), snliq[k]))
            snice = snice.at[k].set(jnp.where(hit, jnp.asarray(0.0, dtype), snice[k]))
            dzsnso = dzsnso.at[k].set(jnp.where(hit, jnp.asarray(0.0, dtype), dzsnso[k]))

        # shift slots above removed toward surface (:6688-6696)
        do_shift = has_removable & (first_k > (NSNOW + isnow)) & (isnow < -1)
        firstactive = NSNOW + isnow
        for k in range(NSNOW - 1, 0, -1):
            in_range = do_shift & (k <= first_k) & (k >= firstactive + 1)
            stc = stc.at[k].set(jnp.where(in_range, stc[k - 1], stc[k]))
            snliq = snliq.at[k].set(jnp.where(in_range, snliq[k - 1], snliq[k]))
            snice = snice.at[k].set(jnp.where(in_range, snice[k - 1], snice[k]))
            dzsnso = dzsnso.at[k].set(jnp.where(in_range, dzsnso[k - 1], dzsnso[k]))

        isnow = jnp.where(has_removable, isnow + 1, isnow)

    # over-sublimation soil correction (:6703-6706)
    neg_sice = sice0 < 0.0
    sh2o0 = jnp.where(neg_sice, sh2o0 + sice0, sh2o0)
    sice0 = jnp.where(neg_sice, jnp.asarray(0.0, dtype), sice0)

    still_multi = isnow < 0

    # ---- re-sum SWE/SNOWH & total-collapse check (:6710-6731) ----------------
    active = _active_mask(isnow)
    sneqv_sum = jnp.sum(jnp.where(active, snice + snliq, 0.0), axis=0)
    snowh_sum = jnp.sum(jnp.where(active, dzsnso, 0.0), axis=0)
    zwice = jnp.sum(jnp.where(active, snice, 0.0), axis=0)
    zwliq = jnp.sum(jnp.where(active, snliq, 0.0), axis=0)
    sneqv = jnp.where(still_multi, sneqv_sum, sneqv)
    snowh = jnp.where(still_multi, snowh_sum, snowh)

    collapse = still_multi & (snowh < 0.025)
    isnow = jnp.where(collapse, jnp.int32(0), isnow)
    sneqv = jnp.where(collapse, zwice, sneqv)
    ponding2 = jnp.where(collapse, zwliq, ponding2)
    snowh = jnp.where(collapse & (zwice <= 0.0), jnp.asarray(0.0, dtype), snowh)

    # ---- phase 3: DZMIN COMBO re-merge when ISNOW < -1 (:6736-6786) ----------
    for _ in range(NSNOW):
        active = _active_mask(isnow)
        firstactive = NSNOW + isnow
        pos = ks - firstactive[None, ...]
        dzmin_vec = jnp.asarray(DZMIN, dtype).reshape((NSNOW,) + (1,) * isnow.ndim)
        pos_clamped = jnp.clip(pos, 0, NSNOW - 1)
        thr = jnp.take_along_axis(dzmin_vec * jnp.ones_like(dzsnso), pos_clamped, axis=0)
        thin = active & (dzsnso < thr) & (isnow < -1)
        first_k = jnp.min(jnp.where(thin, ks, big), axis=0)
        has_thin = first_k < big

        is_top = first_k == firstactive
        is_surf = first_k == surf
        dz_im1 = jnp.zeros_like(snowh)
        dz_i = jnp.zeros_like(snowh)
        dz_ip1 = jnp.zeros_like(snowh)
        for k in range(NSNOW):
            sel = has_thin & (first_k == k)
            dz_i = jnp.where(sel, dzsnso[k], dz_i)
            if k - 1 >= 0:
                dz_im1 = jnp.where(sel, dzsnso[k - 1], dz_im1)
            if k + 1 < NSNOW:
                dz_ip1 = jnp.where(sel, dzsnso[k + 1], dz_ip1)
        prefer_up = (dz_im1 + dz_i) < (dz_ip1 + dz_i)
        neibor_up = is_surf | ((~is_top) & prefer_up)

        for k in range(NSNOW):
            sel = has_thin & (first_k == k)
            up = sel & neibor_up
            if k - 1 >= 0:
                dzc, wlc, wic, tc = _combo(
                    dzsnso[k], snliq[k], snice[k], stc[k],
                    dzsnso[k - 1], snliq[k - 1], snice[k - 1], stc[k - 1])
                dzsnso = dzsnso.at[k].set(jnp.where(up, dzc, dzsnso[k]))
                snliq = snliq.at[k].set(jnp.where(up, wlc, snliq[k]))
                snice = snice.at[k].set(jnp.where(up, wic, snice[k]))
                stc = stc.at[k].set(jnp.where(up, tc, stc[k]))
            down = sel & (~neibor_up)
            if k + 1 < NSNOW:
                dzc, wlc, wic, tc = _combo(
                    dzsnso[k + 1], snliq[k + 1], snice[k + 1], stc[k + 1],
                    dzsnso[k], snliq[k], snice[k], stc[k])
                dzsnso = dzsnso.at[k + 1].set(jnp.where(down, dzc, dzsnso[k + 1]))
                snliq = snliq.at[k + 1].set(jnp.where(down, wlc, snliq[k + 1]))
                snice = snice.at[k + 1].set(jnp.where(down, wic, snice[k + 1]))
                stc = stc.at[k + 1].set(jnp.where(down, tc, stc[k + 1]))

        l_local = jnp.where(neibor_up, first_k - 1, first_k)
        do_shift = has_thin
        for k in range(NSNOW - 1, 0, -1):
            in_range = do_shift & (k <= l_local) & (k >= firstactive + 1)
            stc = stc.at[k].set(jnp.where(in_range, stc[k - 1], stc[k]))
            snliq = snliq.at[k].set(jnp.where(in_range, snliq[k - 1], snliq[k]))
            snice = snice.at[k].set(jnp.where(in_range, snice[k - 1], snice[k]))
            dzsnso = dzsnso.at[k].set(jnp.where(in_range, dzsnso[k - 1], dzsnso[k]))

        isnow = jnp.where(has_thin, isnow + 1, isnow)

    return isnow, snowh, sneqv, sh2o0, sice0, ponding1, ponding2, stc, snice, snliq, dzsnso


def _divide(isnow, stc, snice, snliq, dzsnso):
    """DIVIDE (:6792-6916). Branch-free subdivision to up-to-3 layers.

    WRF copies active layers into a *top-down* working set: ``DZ(J)=DZSNSO(J+
    ISNOW)`` so working index 1 (==w[0]) is the topmost active layer (WRF index
    ISNOW+1) and working index |ISNOW| (==w[|ISNOW|-1]) is the surface (WRF index
    0). After subdivision, write back ``DZSNSO(J)=DZ(J-ISNOW)``. Working arrays
    ``w[0..2]`` therefore run TOP -> surface. Gather/scatter are firstactive-
    relative (firstactive = NSNOW+ISNOW = local index of the topmost active layer)
    and realised branch-free with ``jnp.where`` since firstactive varies per col.
    """
    dtype = stc.dtype
    surf = NSNOW - 1
    firstactive = NSNOW + isnow            # local index of topmost active layer

    def _gather(field):
        # w[m] = field[firstactive + m], top-down; inactive -> 0
        cols = []
        for m in range(NSNOW):
            src = firstactive + m
            acc = jnp.zeros_like(field[0])
            for k in range(NSNOW):
                acc = jnp.where(src == k, field[k], acc)
            cols.append(acc)
        return jnp.stack(cols, axis=0)

    dz = _gather(dzsnso)
    swice = _gather(snice)
    swliq = _gather(snliq)
    tsno = _gather(stc)

    msno = -isnow   # ABS(ISNOW)

    # MSNO==1: split top if DZ(1)>0.05 (:6837-6848)
    split1 = (msno == 1) & (dz[0] > DZ_SPLIT1)
    half_dz = dz[0] / 2.0
    half_wi = swice[0] / 2.0
    half_wl = swliq[0] / 2.0
    dz = dz.at[0].set(jnp.where(split1, half_dz, dz[0]))
    swice = swice.at[0].set(jnp.where(split1, half_wi, swice[0]))
    swliq = swliq.at[0].set(jnp.where(split1, half_wl, swliq[0]))
    dz = dz.at[1].set(jnp.where(split1, half_dz, dz[1]))
    swice = swice.at[1].set(jnp.where(split1, half_wi, swice[1]))
    swliq = swliq.at[1].set(jnp.where(split1, half_wl, swliq[1]))
    tsno = tsno.at[1].set(jnp.where(split1, tsno[0], tsno[1]))
    msno = jnp.where(split1, jnp.int32(2), msno)

    # MSNO>1: trim top to 0.05, COMBO excess into layer 2 (:6851-6884)
    do2 = (msno > 1) & (dz[0] > DZ_SPLIT1)
    dz0_safe = jnp.where(dz[0] > 0.0, dz[0], 1.0)
    drr = dz[0] - DZ_SPLIT1
    propor = drr / dz0_safe
    zwice = propor * swice[0]
    zwliq = propor * swliq[0]
    propor2 = DZ_SPLIT1 / dz0_safe
    new_swice0 = propor2 * swice[0]
    new_swliq0 = propor2 * swliq[0]
    dzc, wlc, wic, tc = _combo(dz[1], swliq[1], swice[1], tsno[1], drr, zwliq, zwice, tsno[0])
    swice = swice.at[0].set(jnp.where(do2, new_swice0, swice[0]))
    swliq = swliq.at[0].set(jnp.where(do2, new_swliq0, swliq[0]))
    dz = dz.at[0].set(jnp.where(do2, jnp.asarray(DZ_SPLIT1, dtype), dz[0]))
    dz = dz.at[1].set(jnp.where(do2, dzc, dz[1]))
    swliq = swliq.at[1].set(jnp.where(do2, wlc, swliq[1]))
    swice = swice.at[1].set(jnp.where(do2, wic, swice[1]))
    tsno = tsno.at[1].set(jnp.where(do2, tc, tsno[1]))

    # subdivide layer 2 if MSNO<=2 and DZ(2)>0.20 (:6866-6883)
    split2 = do2 & (msno <= 2) & (dz[1] > DZ_SPLIT2)
    dz1_safe = jnp.where((dz[0] + dz[1]) > 0.0, (dz[0] + dz[1]), 1.0)
    dtdz = (tsno[0] - tsno[1]) / (dz1_safe / 2.0)
    half_dz2 = dz[1] / 2.0
    half_wi2 = swice[1] / 2.0
    half_wl2 = swliq[1] / 2.0
    dz = dz.at[1].set(jnp.where(split2, half_dz2, dz[1]))
    swice = swice.at[1].set(jnp.where(split2, half_wi2, swice[1]))
    swliq = swliq.at[1].set(jnp.where(split2, half_wl2, swliq[1]))
    dz = dz.at[2].set(jnp.where(split2, half_dz2, dz[2]))
    swice = swice.at[2].set(jnp.where(split2, half_wi2, swice[2]))
    swliq = swliq.at[2].set(jnp.where(split2, half_wl2, swliq[2]))
    tsno3 = tsno[1] - dtdz * half_dz2 / 2.0
    warm3 = tsno3 >= TFRZ
    tsno = tsno.at[2].set(jnp.where(split2, jnp.where(warm3, tsno[1], tsno3), tsno[2]))
    tsno = tsno.at[1].set(jnp.where(split2 & (~warm3), tsno[1] + dtdz * half_dz2 / 2.0, tsno[1]))
    msno = jnp.where(split2, jnp.int32(3), msno)

    # MSNO>2: trim layer2 to 0.2, COMBO excess into layer 3 (:6887-6900)
    do3 = (msno > 2) & (dz[1] > DZ_SPLIT2)
    dz1b_safe = jnp.where(dz[1] > 0.0, dz[1], 1.0)
    drr3 = dz[1] - DZ_SPLIT2
    propor3 = drr3 / dz1b_safe
    zwice3 = propor3 * swice[1]
    zwliq3 = propor3 * swliq[1]
    propor3b = DZ_SPLIT2 / dz1b_safe
    new_swice1 = propor3b * swice[1]
    new_swliq1 = propor3b * swliq[1]
    dzc3, wlc3, wic3, tc3 = _combo(dz[2], swliq[2], swice[2], tsno[2], drr3, zwliq3, zwice3, tsno[1])
    swice = swice.at[1].set(jnp.where(do3, new_swice1, swice[1]))
    swliq = swliq.at[1].set(jnp.where(do3, new_swliq1, swliq[1]))
    dz = dz.at[1].set(jnp.where(do3, jnp.asarray(DZ_SPLIT2, dtype), dz[1]))
    dz = dz.at[2].set(jnp.where(do3, dzc3, dz[2]))
    swliq = swliq.at[2].set(jnp.where(do3, wlc3, swliq[2]))
    swice = swice.at[2].set(jnp.where(do3, wic3, swice[2]))
    tsno = tsno.at[2].set(jnp.where(do3, tc3, tsno[2]))

    isnow_new = -msno

    # write back: WRF DSNSO(J)=DZ(J-ISNOW) for J=ISNOW_new+1..0. Local slot
    # `local` (active) <- working index m = local - firstactive_new (top-down).
    firstactive_new = NSNOW + isnow_new
    active_new = _active_mask(isnow_new)
    for local in range(NSNOW):
        act = active_new[local]
        m_idx = local - firstactive_new      # per-column working index
        dz_l = jnp.zeros_like(dzsnso[0])
        wi_l = jnp.zeros_like(dzsnso[0])
        wl_l = jnp.zeros_like(dzsnso[0])
        t_l = jnp.zeros_like(dzsnso[0])
        for m in range(NSNOW):
            sel = m_idx == m
            dz_l = jnp.where(sel, dz[m], dz_l)
            wi_l = jnp.where(sel, swice[m], wi_l)
            wl_l = jnp.where(sel, swliq[m], wl_l)
            t_l = jnp.where(sel, tsno[m], t_l)
        dzsnso = dzsnso.at[local].set(jnp.where(act, dz_l, dzsnso[local]))
        snice = snice.at[local].set(jnp.where(act, wi_l, snice[local]))
        snliq = snliq.at[local].set(jnp.where(act, wl_l, snliq[local]))
        stc = stc.at[local].set(jnp.where(act, t_l, stc[local]))

    return isnow_new, stc, snice, snliq, dzsnso


def _snowh2o(isnow, dzsnso, snowh, sneqv, snice, snliq, sh2o0, sice0, stc,
             qsnfro, qsnsub, qrain, dz1_soil, dt):
    """SNOWH2O (:7085-7230). Sublimation/frost + liquid percolation.

    The embedded COMBINE re-call (:7183-7186) is folded in via a single masked
    COMBINE pass on the WGDIF<1e-6 condition.
    """
    dtype = stc.dtype
    ponding1 = jnp.zeros_like(snowh)
    ponding2 = jnp.zeros_like(snowh)

    # SNEQV==0 after COMBINE: frost/sublimation onto top soil (:7140-7146)
    no_swe = sneqv == 0.0
    sice0 = jnp.where(no_swe, sice0 + (qsnfro - qsnsub) * dt / (dz1_soil * 1000.0), sice0)
    neg = no_swe & (sice0 < 0.0)
    sh2o0 = jnp.where(neg, sh2o0 + sice0, sh2o0)
    sice0 = jnp.where(neg, jnp.asarray(0.0, dtype), sice0)

    # shallow snow without a layer (:7153-7169)
    shallow = (isnow == 0) & (sneqv > 0.0)
    temp = sneqv
    sneqv_s = sneqv - qsnsub * dt + qsnfro * dt
    temp_safe = jnp.where(temp > 0.0, temp, 1.0)
    propor = sneqv_s / temp_safe
    snowh_s = jnp.maximum(0.0, propor * snowh)
    snowh_s = jnp.minimum(jnp.maximum(snowh_s, sneqv_s / 500.0), sneqv_s / 50.0)
    sneqv = jnp.where(shallow, sneqv_s, sneqv)
    snowh = jnp.where(shallow, snowh_s, snowh)
    neg_swe = shallow & (sneqv < 0.0)
    sice0 = jnp.where(neg_swe, sice0 + sneqv / (dz1_soil * 1000.0), sice0)
    sneqv = jnp.where(neg_swe, jnp.asarray(0.0, dtype), sneqv)
    snowh = jnp.where(neg_swe, jnp.asarray(0.0, dtype), snowh)
    neg2 = shallow & (sice0 < 0.0)
    sh2o0 = jnp.where(neg2, sh2o0 + sice0, sh2o0)
    sice0 = jnp.where(neg2, jnp.asarray(0.0, dtype), sice0)

    # clamp tiny snow to zero (:7171-7174)
    tiny = (snowh <= 1.0e-8) | (sneqv <= 1.0e-6)
    snowh = jnp.where(tiny, jnp.asarray(0.0, dtype), snowh)
    sneqv = jnp.where(tiny, jnp.asarray(0.0, dtype), sneqv)

    # deep snow: top-layer sublimation/frost (:7178-7194)
    multilayer = isnow < 0
    top = NSNOW + isnow
    wgdif_lt = jnp.zeros_like(snowh, dtype=bool)
    for k in range(NSNOW):
        is_top = multilayer & (top == k)
        wgdif = snice[k] - qsnsub * dt + qsnfro * dt
        snice = snice.at[k].set(jnp.where(is_top, wgdif, snice[k]))
        wgdif_lt = wgdif_lt | (is_top & (wgdif < 1.0e-6))

    # embedded COMBINE when WGDIF<1e-6 (:7182-7187), masked re-call
    need_combine = multilayer & wgdif_lt
    (isnow_c, snowh_c, sneqv_c, sh2o0_c, sice0_c, p1_c, p2_c,
     stc_c, snice_c, snliq_c, dzsnso_c) = _combine(
        isnow, sh2o0, sice0, stc, snice, snliq, dzsnso, snowh, sneqv, dz1_soil)
    nc2 = need_combine
    nc3 = need_combine[None, ...]
    isnow = jnp.where(nc2, isnow_c, isnow)
    snowh = jnp.where(nc2, snowh_c, snowh)
    sneqv = jnp.where(nc2, sneqv_c, sneqv)
    sh2o0 = jnp.where(nc2, sh2o0_c, sh2o0)
    sice0 = jnp.where(nc2, sice0_c, sice0)
    ponding1 = jnp.where(nc2, p1_c, ponding1)
    ponding2 = jnp.where(nc2, p2_c, ponding2)
    stc = jnp.where(nc3, stc_c, stc)
    snice = jnp.where(nc3, snice_c, snice)
    snliq = jnp.where(nc3, snliq_c, snliq)
    dzsnso = jnp.where(nc3, dzsnso_c, dzsnso)

    # rain into top layer if still multi-layer (:7189-7192)
    multilayer = isnow < 0
    top = NSNOW + isnow
    for k in range(NSNOW):
        is_top = multilayer & (top == k)
        snliq = snliq.at[k].set(jnp.where(is_top, jnp.maximum(0.0, snliq[k] + qrain * dt), snliq[k]))

    # porosity + percolation top->surface (:7198-7224), branch-free sweep
    active = _active_mask(isnow)
    dz_safe = jnp.where(dzsnso > 0.0, dzsnso, 1.0)
    vol_ice = jnp.minimum(1.0, snice / (dz_safe * DENICE))
    epore = 1.0 - vol_ice

    qin = jnp.zeros_like(snowh)
    qout = jnp.zeros_like(snowh)
    for k in range(NSNOW):
        act = active[k]
        snliq_k = snliq[k] + qin
        vol_liq = snliq_k / (dz_safe[k] * DENH2O)
        qout_k = jnp.maximum(0.0, (vol_liq - SSI * epore[k]) * dzsnso[k])
        if k == (NSNOW - 1):  # surface (WRF J==0) special hold (:7210-7212)
            qout_k = jnp.maximum((vol_liq - epore[k]) * dzsnso[k], SNOW_RET_FAC * dt * qout_k)
        qout_k = qout_k * DENH2O
        snliq_k = snliq_k - qout_k
        denom = snice[k] + snliq_k
        denom_safe = jnp.where(denom > 0.0, denom, 1.0)
        over = (snliq_k / denom_safe) > MAX_LIQ_MASS_FRACTION
        excess = snliq_k - MAX_LIQ_MASS_FRACTION / (1.0 - MAX_LIQ_MASS_FRACTION) * snice[k]
        qout_k = jnp.where(over, qout_k + excess, qout_k)
        snliq_k = jnp.where(over, MAX_LIQ_MASS_FRACTION / (1.0 - MAX_LIQ_MASS_FRACTION) * snice[k], snliq_k)
        snliq = snliq.at[k].set(jnp.where(act, snliq_k, snliq[k]))
        qout = jnp.where(act, qout_k, qout)
        qin = jnp.where(act, qout_k, qin)

    floor = snliq / DENH2O + snice / DENICE
    dzsnso = jnp.where(active, jnp.maximum(dzsnso, floor), dzsnso)

    qsnbot = qout / dt
    return isnow, snowh, sneqv, snice, snliq, sh2o0, sice0, stc, dzsnso, qsnbot, ponding1, ponding2


def _snowwater_column(isnow, snowh, sneqv, snice, snliq, sh2o, sice, stc_snow,
                      zsoil, qsnow, snowhin, qsnfro, qsnsub, qrain, sfctmp,
                      ficeold, imelt, dzsnso, dt):
    """SNOWWATER driver (:6398-6535). All snow arrays top-aligned (axis 0)."""
    surf = NSNOW - 1
    dz1_soil = -zsoil[0]            # DZSNSO(1) = ZSOIL(1) magnitude

    isnow, snowh, sneqv, dzsnso, stc_snow, snice, snliq = _snowfall(
        isnow, snowh, sneqv, dzsnso, stc_snow, snice, snliq, qsnow, snowhin, sfctmp, dt)

    dzsnso = _compact(isnow, dt, stc_snow, snice, snliq, imelt, ficeold, dzsnso)

    (isnow, snowh, sneqv, sh2o, sice, p1, p2,
     stc_snow, snice, snliq, dzsnso) = _combine(
        isnow, sh2o, sice, stc_snow, snice, snliq, dzsnso, snowh, sneqv, dz1_soil)

    multi = isnow < 0
    isnow_d, stc_d, snice_d, snliq_d, dzsnso_d = _divide(isnow, stc_snow, snice, snliq, dzsnso)
    m3 = multi[None, ...]
    isnow = jnp.where(multi, isnow_d, isnow)
    stc_snow = jnp.where(m3, stc_d, stc_snow)
    snice = jnp.where(m3, snice_d, snice)
    snliq = jnp.where(m3, snliq_d, snliq)
    dzsnso = jnp.where(m3, dzsnso_d, dzsnso)

    (isnow, snowh, sneqv, snice, snliq, sh2o, sice, stc_snow, dzsnso,
     qsnbot, p1b, p2b) = _snowh2o(
        isnow, dzsnso, snowh, sneqv, snice, snliq, sh2o, sice, stc_snow,
        qsnfro, qsnsub, qrain, dz1_soil, dt)
    ponding1 = p1 + p1b
    ponding2 = p2 + p2b

    # set empty snow layers to zero (:6480-6486)
    active = _active_mask(isnow)
    snice = jnp.where(active, snice, 0.0)
    snliq = jnp.where(active, snliq, 0.0)
    stc_snow = jnp.where(active, stc_snow, 0.0)
    dzsnso = jnp.where(active, dzsnso, 0.0)

    # glacier equilibrium cap (:6490-6496)
    glacier = sneqv > 5000.0
    dz_surf_safe = jnp.where(dzsnso[surf] > 0.0, dzsnso[surf], 1.0)
    bdsnow = snice[surf] / dz_surf_safe
    bdsnow_safe = jnp.where(bdsnow > 0.0, bdsnow, 1.0)
    snoflow_mass = sneqv - 5000.0
    snice = snice.at[surf].set(jnp.where(glacier, snice[surf] - snoflow_mass, snice[surf]))
    dzsnso = dzsnso.at[surf].set(jnp.where(glacier, dzsnso[surf] - snoflow_mass / bdsnow_safe, dzsnso[surf]))
    snoflow = jnp.where(glacier, snoflow_mass / dt, 0.0)

    # re-sum SWE for layered snow (:6500-6505)
    multilayer = isnow < 0
    active = _active_mask(isnow)
    sneqv_sum = jnp.sum(jnp.where(active, snice + snliq, 0.0), axis=0)
    sneqv = jnp.where(multilayer, sneqv_sum, sneqv)

    # rebuild full ZSNSO (snow + soil) cumulatively from the topmost active snow
    # layer (:6507-6525). Snow thicknesses are negated; soil thicknesses are the
    # (already negative) ZSOIL differences. ZSNSO is the running cumulative;
    # inactive snow slots stay zero. SNOWH(multi) = sum of active snow dz.
    nx_shape = dzsnso.shape[1:]
    dz_soil0 = zsoil[0]                       # = ZSOIL(1) (negative)
    dz_soil_rest = zsoil[1:] - zsoil[:-1]     # ZSOIL(IZ)-ZSOIL(IZ-1) (negative)
    dz_soil = jnp.concatenate([dz_soil0[None], dz_soil_rest], axis=0)   # (NSOIL,)
    dz_soil_col = jnp.broadcast_to(dz_soil[:, None, None], (NSOIL,) + nx_shape)

    neg_snow = jnp.where(active, -dzsnso, 0.0)         # (NSNOW,) negative snow dz
    full_neg = jnp.concatenate([neg_snow, dz_soil_col], axis=0)  # (NSNOW+NSOIL,)
    zsnso_full = jnp.cumsum(full_neg, axis=0)
    # zero the inactive snow slots in ZSNSO (WRF leaves them 0 via :6480-6486)
    active_full = jnp.concatenate(
        [active, jnp.ones((NSOIL,) + nx_shape, dtype=bool)], axis=0)
    zsnso_full = jnp.where(active_full, zsnso_full, 0.0)

    snowh_ml = jnp.sum(jnp.where(active, dzsnso, 0.0), axis=0)
    snowh = jnp.where(multilayer, snowh_ml, snowh)

    return (isnow, snowh, sneqv, snice, snliq, sh2o, sice, stc_snow, dzsnso,
            zsnso_full, qsnbot, snoflow, ponding1, ponding2)


def _snow_age(dt, tg, sneqvo, sneqv, tauss):
    """SNOW_AGE (:3119-3167). Returns (tauss, fage). Branch-free."""
    dela0 = dt / TAU0
    tg_safe = jnp.where(tg > 0.0, tg, TFRZ)
    arg = GRAIN_GROWTH * (1.0 / TFRZ - 1.0 / tg_safe)
    age1 = jnp.exp(arg)
    age2 = jnp.exp(jnp.minimum(0.0, EXTRA_GROWTH * arg))
    age3 = DIRT_SOOT
    tage = age1 + age2 + age3
    dela = dela0 * tage
    dels = jnp.maximum(0.0, sneqv - sneqvo) / SWEMX
    sge = (tauss + dela) * (1.0 - dels)
    tauss_new = jnp.where(sneqv <= 0.0, jnp.asarray(0.0, tauss.dtype), jnp.maximum(0.0, sge))
    fage = tauss_new / (tauss_new + 1.0)
    return tauss_new, fage


def _snowalb_class(qsnow, dt, albold):
    """SNOWALB_CLASS (:3226-3275, opt_alb=2). Returns new broadband ALB."""
    alb = 0.55 + (albold - 0.55) * jnp.exp(-0.01 * dt / 3600.0)
    fresh = qsnow > 0.0
    alb_fresh = alb + jnp.minimum(qsnow, SWEMX / dt) * (0.84 - alb) / (SWEMX / dt)
    return jnp.where(fresh, alb_fresh, alb)


@partial(jax.jit, static_argnames=("dt",))
def noahmp_snow(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    qsnow: jax.Array,
    imelt: jax.Array,
    qmelt: jax.Array,
    dt: float,
) -> NoahMPLandState:
    """Advance the snow column one ``dt`` (SNOWWATER + albedo aging).

    Faithful to pristine-WRF Noah-MP ``SNOWWATER`` + ``SNOW_AGE`` +
    ``SNOWALB_CLASS`` (opt_alb=2). Branch-free masked variable-layer kernel; fp64.

    Parameters
    ----------
    land_state : NoahMPLandState
        Prognostic land carry. Snow fields (``isnow``/``tsno``/``snice``/``snliq``/
        ``zsnso``/``snowh``/``sneqv``/``sneqvo``/``tauss``/``albold``) and the top
        soil layer of ``sh2o``/``smois`` are updated; all else returned unchanged.
    forcing : NoahMPForcing
        Uses ``sfctmp`` (new-layer STC + BDFALL) and ``cosz`` (CLASS-albedo guard).
    static : NoahMPStatic
        Uses ``zsoil`` (soil interface depths, <0) to rebuild ZSNSO/DZSNSO.
    qsnow : jax.Array
        Snowfall onto ground [mm/s], (ny, nx).
    imelt : jax.Array
        Per-layer phase-change flag (NSNOW+NSOIL, ny, nx); snow rows 0..NSNOW-1.
    qmelt : jax.Array
        Snowmelt rate [mm/s], (ny, nx) — routed in as QRAIN reaching the pack.
    dt : float
        Physics timestep [s].

    Returns
    -------
    NoahMPLandState
        Land carry with snow + albedo-aging fields advanced.
    """
    dtype = land_state.snice.dtype

    # SNOWHIN = QSNOW / BDFALL (PRECIP_HEAT :1216, opt_snf=1). BDFALL reconstructed
    # here from SFCTMP so accumulation depth matches WRF until the precip-heat
    # sprint supplies SNOWHIN directly.
    sfctmp = forcing.sfctmp
    bdfall = jnp.minimum(120.0, 67.92 + 51.25 * jnp.exp((sfctmp - TFRZ) / 2.59))
    snowhin = jnp.where(qsnow > 0.0, qsnow / bdfall, 0.0)

    # QRAIN / QSNSUB / QSNFRO are produced by the energy/water sprint; until then
    # QRAIN = qmelt (melt + throughfall reaching the pack), no sublimation/frost.
    qrain = qmelt
    qsnsub = jnp.zeros_like(qsnow)
    qsnfro = jnp.zeros_like(qsnow)

    imelt_snow = imelt[:NSNOW].astype(jnp.int32)

    # FICEOLD from the entering pack (prior-step ice fraction)
    wx_old = land_state.snice + land_state.snliq
    wx_old_safe = jnp.where(wx_old > 0.0, wx_old, 1.0)
    ficeold = jnp.where(wx_old > 0.0, land_state.snice / wx_old_safe, 0.0)

    isnow = land_state.isnow.astype(jnp.int32)
    snowh = land_state.snowh.astype(dtype)
    sneqv = land_state.sneqv.astype(dtype)
    snice = land_state.snice.astype(dtype)
    snliq = land_state.snliq.astype(dtype)
    stc_snow = land_state.tsno.astype(dtype)

    # snow DZSNSO from current snow ZSNSO (positive thickness, cumulative)
    zsnso = land_state.zsnso.astype(dtype)
    active0 = _active_mask(isnow)
    zsnso_snow = zsnso[:NSNOW]
    prev = jnp.concatenate([jnp.zeros_like(zsnso_snow[:1]), zsnso_snow[:-1]], axis=0)
    dzsnso_snow = jnp.where(active0, prev - zsnso_snow, 0.0)

    sh2o_top = land_state.sh2o[0].astype(dtype)
    sice_top = (land_state.smois[0] - land_state.sh2o[0]).astype(dtype)
    zsoil = static.zsoil.astype(dtype)

    (isnow_n, snowh_n, sneqv_n, snice_n, snliq_n, sh2o_top_n, sice_top_n,
     stc_snow_n, dzsnso_snow_n, zsnso_full, qsnbot, snoflow,
     ponding1, ponding2) = _snowwater_column(
        isnow, snowh, sneqv, snice, snliq, sh2o_top, sice_top, stc_snow,
        zsoil, qsnow, snowhin, qsnfro, qsnsub, qrain, sfctmp,
        ficeold, imelt_snow, dzsnso_snow, dt)

    # albedo aging
    sneqvo = land_state.sneqv.astype(dtype)
    tauss_n, _fage = _snow_age(dt, land_state.tg.astype(dtype), sneqvo, sneqv_n,
                               land_state.tauss.astype(dtype))
    alb_new = _snowalb_class(qsnow, dt, land_state.albold.astype(dtype))
    albold_n = jnp.where(forcing.cosz > 0.0, alb_new, land_state.albold.astype(dtype))

    smois_new = land_state.smois.at[0].set(sh2o_top_n + sice_top_n)
    sh2o_new = land_state.sh2o.at[0].set(sh2o_top_n)

    return land_state.replace(
        isnow=isnow_n.astype(land_state.isnow.dtype),
        tsno=stc_snow_n,
        snice=snice_n,
        snliq=snliq_n,
        zsnso=zsnso_full,
        snowh=snowh_n,
        sneqv=sneqv_n,
        sneqvo=sneqvo,
        tauss=tauss_n,
        albold=albold_n,
        smois=smois_new,
        sh2o=sh2o_new,
    )


__all__ = ["noahmp_snow"]
