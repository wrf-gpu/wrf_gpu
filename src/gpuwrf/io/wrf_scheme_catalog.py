"""Authoritative WRF v4 physics scheme catalog (code -> name).

This module is the *full WRF v4 ARW enumeration* of the valid integer codes for
the core physics namelist parameters, transcribed faithfully from the WRF v4
namelist documentation:

    /home/user/src/wrf_pristine/WRF/run/README.namelist  (verified 2026-06-04)

It is a pure-data, dependency-free reference table. It makes **no** claim about
what the GPU port implements -- that is owned by
``gpuwrf.contracts.physics_registry`` (the ``ACCEPTED_*`` sets) and consumed by
``gpuwrf.io.namelist_check``.

The catalog exists so the namelist checker can distinguish three cases for any
selected scheme value:

  (a) IMPLEMENTED / accepted   -> passes the port's support check;
  (b) recognized WRF v4 scheme but NOT YET IMPLEMENTED in the port
                               -> fail-closed with a *specific* message naming
                                  the WRF scheme and its status;
  (c) NOT a recognized WRF v4 option at all
                               -> fail-closed with "not a recognized WRF v4 ..."

Each ``WrfSchemeName`` carries the integer code, the human-readable WRF scheme
name, and the README.namelist line range it was transcribed from, so the table
can be re-audited against a future WRF release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


WRF_README_SOURCE = "/home/user/src/wrf_pristine/WRF/run/README.namelist (v4, audited 2026-06-04)"


@dataclass(frozen=True)
class WrfSchemeName:
    """One valid WRF v4 namelist option code and its WRF scheme name."""

    code: int
    name: str
    source_lines: str


# --------------------------------------------------------------------------- #
# mp_physics -- microphysics (README.namelist lines 473-536)                  #
# --------------------------------------------------------------------------- #
WRF_MP_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no microphysics", "474"),
    1: WrfSchemeName(1, "Kessler warm-rain", "475"),
    2: WrfSchemeName(2, "Lin et al. (Purdue-Lin)", "476"),
    3: WrfSchemeName(3, "WSM3 simple-ice", "477"),
    4: WrfSchemeName(4, "WSM5", "478"),
    5: WrfSchemeName(5, "Ferrier (new Eta), HRW", "479"),
    6: WrfSchemeName(6, "WSM6 graupel", "480"),
    7: WrfSchemeName(7, "Goddard 4-ice", "481"),
    8: WrfSchemeName(8, "Thompson", "482"),
    9: WrfSchemeName(9, "Milbrandt-Yau 2-moment", "483"),
    10: WrfSchemeName(10, "Morrison 2-moment", "484"),
    11: WrfSchemeName(11, "CAM 5.1 microphysics", "485"),
    13: WrfSchemeName(13, "SBU-YLin", "486"),
    14: WrfSchemeName(14, "WDM5", "487"),
    16: WrfSchemeName(16, "WDM6", "488"),
    17: WrfSchemeName(17, "NSSL (legacy)", "497"),
    18: WrfSchemeName(18, "NSSL 2-moment 4-ice w/ predicted CCN", "489"),
    19: WrfSchemeName(19, "NSSL (legacy)", "497"),
    21: WrfSchemeName(21, "NSSL (legacy)", "497"),
    22: WrfSchemeName(22, "NSSL (legacy)", "497"),
    24: WrfSchemeName(24, "WSM7 (separate hail/graupel)", "498"),
    26: WrfSchemeName(26, "WDM7 (separate hail/graupel)", "499"),
    27: WrfSchemeName(27, "UDM 7-class", "500"),
    28: WrfSchemeName(28, "aerosol-aware Thompson (water/ice-friendly)", "501"),
    29: WrfSchemeName(29, "RCON (Thompson aerosol-aware, liquid-phase mods)", "514"),
    30: WrfSchemeName(30, "HUJI spectral-bin (fast)", "516"),
    32: WrfSchemeName(32, "HUJI spectral-bin (full)", "518"),
    38: WrfSchemeName(38, "Thompson w/ 2-moment graupel/hail", "519"),
    40: WrfSchemeName(40, "Morrison 2-moment w/ CESM-NCSU RCP4.5 aerosol", "520"),
    50: WrfSchemeName(50, "P3 1-ice, 1-moment cloud water", "522"),
    51: WrfSchemeName(51, "P3 1-ice + double-moment cloud water", "523"),
    52: WrfSchemeName(52, "P3 2-ice + double-moment cloud water", "524"),
    53: WrfSchemeName(53, "P3 1-ice 3-moment + double-moment cloud water", "525"),
    55: WrfSchemeName(55, "Jensen-ISHMAEL", "526"),
    56: WrfSchemeName(56, "NTU multi-moment", "528"),
    95: WrfSchemeName(95, "Ferrier (old Eta)", "534"),
    96: WrfSchemeName(96, "Madwrf", "535"),
    97: WrfSchemeName(97, "Goddard GCE", "536"),
}


# --------------------------------------------------------------------------- #
# ra_lw_physics -- longwave radiation (README.namelist lines 580-601)         #
# --------------------------------------------------------------------------- #
WRF_RA_LW_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no longwave radiation", "581"),
    1: WrfSchemeName(1, "RRTM", "582"),
    3: WrfSchemeName(3, "CAM", "586"),
    4: WrfSchemeName(4, "RRTMG", "588"),
    5: WrfSchemeName(5, "Goddard longwave", "597"),
    7: WrfSchemeName(7, "FLG (UCLA)", "598"),
    14: WrfSchemeName(14, "RRTMG-K (KIAPS)", "592"),
    24: WrfSchemeName(24, "fast RRTMG (GPU/MIC)", "593"),
    31: WrfSchemeName(31, "Earth Held-Suarez forcing", "599"),
    99: WrfSchemeName(99, "GFDL (Eta) longwave (semi-supported)", "600"),
}


# --------------------------------------------------------------------------- #
# ra_sw_physics -- shortwave radiation (README.namelist lines 603-615)        #
# --------------------------------------------------------------------------- #
WRF_RA_SW_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no shortwave radiation", "604"),
    1: WrfSchemeName(1, "Dudhia", "605"),
    2: WrfSchemeName(2, "Goddard shortwave", "606"),
    3: WrfSchemeName(3, "CAM", "607"),
    4: WrfSchemeName(4, "RRTMG", "609"),
    5: WrfSchemeName(5, "Goddard shortwave (new)", "612"),
    7: WrfSchemeName(7, "FLG (UCLA)", "613"),
    14: WrfSchemeName(14, "RRTMG-K (KIAPS)", "610"),
    24: WrfSchemeName(24, "fast RRTMG (GPU/MIC)", "611"),
    99: WrfSchemeName(99, "GFDL (Eta) (semi-supported)", "614"),
}


# --------------------------------------------------------------------------- #
# sf_sfclay_physics -- surface-layer (README.namelist lines 686-695)          #
# --------------------------------------------------------------------------- #
WRF_SF_SFCLAY_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no surface-layer", "687"),
    1: WrfSchemeName(1, "revised MM5 Monin-Obukhov", "688"),
    2: WrfSchemeName(2, "Monin-Obukhov (Janjic Eta)", "689"),
    3: WrfSchemeName(3, "NCEP GFS surface layer", "690"),
    4: WrfSchemeName(4, "QNSE surface layer", "691"),
    5: WrfSchemeName(5, "MYNN surface layer", "692"),
    7: WrfSchemeName(7, "Pleim-Xiu surface layer", "693"),
    10: WrfSchemeName(10, "TEMF surface layer", "694"),
    91: WrfSchemeName(91, "old MM5 surface layer", "695"),
}


# --------------------------------------------------------------------------- #
# sf_surface_physics -- land-surface (README.namelist lines 697-707)          #
# --------------------------------------------------------------------------- #
WRF_SF_SURFACE_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no surface temp prediction", "698"),
    1: WrfSchemeName(1, "thermal diffusion (5-layer slab)", "699"),
    2: WrfSchemeName(2, "Unified Noah LSM", "700"),
    3: WrfSchemeName(3, "RUC LSM", "701"),
    4: WrfSchemeName(4, "Noah-MP LSM", "702"),
    5: WrfSchemeName(5, "CLM4 (Community Land Model v4)", "703"),
    6: WrfSchemeName(6, "CTSM (Community Terrestrial Systems Model)", "704"),
    7: WrfSchemeName(7, "Pleim-Xiu LSM", "705"),
    8: WrfSchemeName(8, "SSiB (Simplified Simple Biosphere)", "706"),
}


# --------------------------------------------------------------------------- #
# bl_pbl_physics -- boundary-layer (README.namelist lines 740-758)            #
# --------------------------------------------------------------------------- #
WRF_BL_PBL_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no boundary-layer", "741"),
    1: WrfSchemeName(1, "YSU", "742"),
    2: WrfSchemeName(2, "Mellor-Yamada-Janjic (MYJ) TKE", "743"),
    3: WrfSchemeName(3, "Hybrid EDMF GFS (ACM-GFS)", "744"),
    4: WrfSchemeName(4, "QNSE-EDMF (Quasi-Normal Scale Elimination)", "745"),
    5: WrfSchemeName(5, "MYNN TKE", "746"),
    7: WrfSchemeName(7, "ACM2 (Pleim)", "748"),
    8: WrfSchemeName(8, "Bougeault-Lacarrere (BouLac)", "749"),
    9: WrfSchemeName(9, "UW (CAM5)", "750"),
    10: WrfSchemeName(10, "TEMF (Total Energy Mass Flux)", "751"),
    11: WrfSchemeName(11, "Shin-Hong scale-aware", "753"),
    12: WrfSchemeName(12, "Grenier-Bretherton-McCaa", "754"),
    16: WrfSchemeName(16, "TKE + TKE-dissipation (epsilon)", "755"),
    17: WrfSchemeName(17, "TKE + TKE-dissipation + TPE", "757"),
    99: WrfSchemeName(99, "MRF", "758"),
}


# --------------------------------------------------------------------------- #
# cu_physics -- cumulus (README.namelist lines 825-842)                       #
# --------------------------------------------------------------------------- #
WRF_CU_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no cumulus", "826"),
    1: WrfSchemeName(1, "Kain-Fritsch (new Eta)", "827"),
    2: WrfSchemeName(2, "Betts-Miller-Janjic (BMJ)", "828"),
    3: WrfSchemeName(3, "Grell-Freitas ensemble", "829"),
    4: WrfSchemeName(4, "Scale-aware GFS SAS", "830"),
    5: WrfSchemeName(5, "Grell 3D ensemble", "831"),
    6: WrfSchemeName(6, "Modified Tiedtke", "832"),
    7: WrfSchemeName(7, "Zhang-McFarlane (CAM5)", "833"),
    10: WrfSchemeName(10, "Modified Kain-Fritsch (PDF trigger)", "834"),
    11: WrfSchemeName(11, "Multi-scale Kain-Fritsch (MSKF)", "835"),
    14: WrfSchemeName(14, "KIM Simplified Arakawa-Schubert (KSAS)", "836"),
    16: WrfSchemeName(16, "New Tiedtke", "837"),
    93: WrfSchemeName(93, "Grell-Devenyi ensemble", "841"),
    94: WrfSchemeName(94, "2015 GFS SAS (HWRF)", "838"),
    95: WrfSchemeName(95, "previous GFS SAS (HWRF)", "839"),
    96: WrfSchemeName(96, "previous new GFS SAS (YSU)", "840"),
    99: WrfSchemeName(99, "previous Kain-Fritsch", "842"),
}


# --------------------------------------------------------------------------- #
# Dynamics options that the port's support-checker also gates.                #
# Full WRF v4 enumeration so "valid-but-unimplemented" is distinguished from   #
# "not a recognized WRF option". (README.namelist lines noted per entry.)      #
# --------------------------------------------------------------------------- #
WRF_DIFF_OPT: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no turbulence / explicit spatial filters off", "1575"),
    1: WrfSchemeName(1, "2nd-order diffusion on coordinate surfaces", "1577"),
    2: WrfSchemeName(2, "mixing in physical space (stress form)", "1582"),
}

WRF_KM_OPT: Mapping[int, WrfSchemeName] = {
    1: WrfSchemeName(1, "constant K (khdif/kvdif)", "1587"),
    2: WrfSchemeName(2, "1.5-order TKE closure (3D)", "1588"),
    3: WrfSchemeName(3, "Smagorinsky first-order closure (3D)", "1589"),
    4: WrfSchemeName(4, "horizontal Smagorinsky first-order closure", "1591"),
    5: WrfSchemeName(5, "SMS-3DTKE scale-adaptive LES/PBL", "1593"),
}

WRF_DAMP_OPT: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no upper-level damping", "1597"),
    1: WrfSchemeName(1, "diffusive damping", "1598"),
    2: WrfSchemeName(2, "Rayleigh damping (idealized only)", "1600"),
    3: WrfSchemeName(3, "w-Rayleigh damping (real-data)", "1602"),
}

WRF_DIFF_6TH_OPT: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no 6th-order diffusion", "1614"),
    1: WrfSchemeName(1, "6th-order numerical diffusion", "1615"),
    2: WrfSchemeName(2, "6th-order diffusion, no up-gradient", "1616"),
}

WRF_RK_ORD: Mapping[int, WrfSchemeName] = {
    2: WrfSchemeName(2, "Runge-Kutta 2nd order", "1565"),
    3: WrfSchemeName(3, "Runge-Kutta 3rd order", "1566"),
}

WRF_W_DAMPING: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "vertical-velocity damping off", "1572"),
    1: WrfSchemeName(1, "vertical-velocity damping on", "1573"),
}

WRF_SF_URBAN_PHYSICS: Mapping[int, WrfSchemeName] = {
    0: WrfSchemeName(0, "no urban canopy", "710"),
    1: WrfSchemeName(1, "single-layer UCM", "711"),
    2: WrfSchemeName(2, "multi-layer BEP", "712"),
    3: WrfSchemeName(3, "multi-layer BEM", "714"),
}


# Map every gated namelist key -> its full WRF v4 code catalog.
WRF_SCHEME_CATALOG: Mapping[str, Mapping[int, WrfSchemeName]] = {
    "mp_physics": WRF_MP_PHYSICS,
    "ra_lw_physics": WRF_RA_LW_PHYSICS,
    "ra_sw_physics": WRF_RA_SW_PHYSICS,
    "sf_sfclay_physics": WRF_SF_SFCLAY_PHYSICS,
    "sf_surface_physics": WRF_SF_SURFACE_PHYSICS,
    "bl_pbl_physics": WRF_BL_PBL_PHYSICS,
    "cu_physics": WRF_CU_PHYSICS,
    "diff_opt": WRF_DIFF_OPT,
    "km_opt": WRF_KM_OPT,
    "damp_opt": WRF_DAMP_OPT,
    "diff_6th_opt": WRF_DIFF_6TH_OPT,
    "rk_order": WRF_RK_ORD,
    "w_damping": WRF_W_DAMPING,
    "sf_urban_physics": WRF_SF_URBAN_PHYSICS,
}

# Human-readable parameter labels for error messages.
WRF_PARAM_LABEL: Mapping[str, str] = {
    "mp_physics": "microphysics",
    "ra_lw_physics": "longwave-radiation",
    "ra_sw_physics": "shortwave-radiation",
    "sf_sfclay_physics": "surface-layer",
    "sf_surface_physics": "land-surface",
    "bl_pbl_physics": "PBL",
    "cu_physics": "cumulus",
    "diff_opt": "diffusion",
    "km_opt": "eddy-coefficient",
    "damp_opt": "upper-level-damping",
    "diff_6th_opt": "6th-order-diffusion",
    "rk_order": "time-integration",
    "w_damping": "w-damping",
    "sf_urban_physics": "urban-canopy",
}


def wrf_scheme_name(key: str, code: int) -> WrfSchemeName | None:
    """Return the WRF v4 scheme metadata for ``key=code``, or ``None``.

    ``None`` means ``code`` is not a recognized WRF v4 option for ``key``.
    """

    catalog = WRF_SCHEME_CATALOG.get(key)
    if catalog is None:
        return None
    return catalog.get(int(code))


def is_recognized_wrf_option(key: str, code: int) -> bool:
    """True iff ``code`` is a valid WRF v4 option for namelist parameter ``key``."""

    return wrf_scheme_name(key, code) is not None


__all__ = [
    "WRF_README_SOURCE",
    "WrfSchemeName",
    "WRF_MP_PHYSICS",
    "WRF_RA_LW_PHYSICS",
    "WRF_RA_SW_PHYSICS",
    "WRF_SF_SFCLAY_PHYSICS",
    "WRF_SF_SURFACE_PHYSICS",
    "WRF_BL_PBL_PHYSICS",
    "WRF_CU_PHYSICS",
    "WRF_DIFF_OPT",
    "WRF_KM_OPT",
    "WRF_DAMP_OPT",
    "WRF_DIFF_6TH_OPT",
    "WRF_RK_ORD",
    "WRF_W_DAMPING",
    "WRF_SF_URBAN_PHYSICS",
    "WRF_SCHEME_CATALOG",
    "WRF_PARAM_LABEL",
    "wrf_scheme_name",
    "is_recognized_wrf_option",
]
