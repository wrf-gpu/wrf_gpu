# Third-party notices

wrf_gpu's own code and documentation are released under the MIT License (see [LICENSE](LICENSE)).
wrf_gpu is a re-implementation of the Weather Research and Forecasting (WRF) model in JAX/XLA/Pallas, and
some files translate, contain or are derived from third-party code and data. **Those files keep their upstream
terms, listed below; the MIT License does not relicense them.** wrf_gpu is not affiliated with or endorsed by
UCAR/NCAR, AER or NOAA. Questions or corrections: please open an issue.

## Summary

| Component | Files in this repository | Upstream terms |
|---|---|---|
| WRF model code and run data (UCAR/NCAR) | Translations throughout `src/gpuwrf` not listed below; verbatim or near-verbatim WRF Fortran used as validation oracles under `proofs/` and `tests/` (e.g. `proofs/v023/oracle/nssl2mom/`, `proofs/v090/module_sf_mynn_pristine.f90`, `proofs/noahmp/oracle/snow_routines.inc`, `proofs/**/*_wrf_patch.diff`); WRF run data `data/fixtures/wrf-cam-ozone-v1.*`, `data/fixtures/thompson-*` | Public domain (UCAR notice below) |
| RRTMG / RRTM radiation (AER) | `src/gpuwrf/physics/rrtmg_lw.py`, `rrtmg_sw.py`, `rrtmg_tables.py`, `ra_lw_rrtm.py`, `ra_lw_rrtm_jax.py`; `src/gpuwrf/kernels/rad_lw_transfer.py`, `rad_sw_band_sums.py`, `rad_sw_quadrature.py`, `rad_mcica.py` (McICA logic; the KISS generator itself is public domain); `data/fixtures/rrtmg-tables-v1.*`, `data/fixtures/rrtmg-intermediate-oracle-v1.*`, `data/fixtures/analytic-rrtmg-*`, `fixtures/samples/analytic-rrtmg-*`; `scripts/wrf_rrtmg_harness.f90` | AER terms below (**may not be sold**; notice must be reproduced) |
| NCAR MMM physics (`physics_mmm`) | `src/gpuwrf/physics/gwd_gwdo.py`, `src/gpuwrf/kernels/phys_gwdo_column.py`, `physics/pbl_ysu.py`, `physics/cumulus_ntiedtke.py`, `physics/cumulus_ntiedtke_jax.py`, `physics/microphysics_wsm6.py`, `physics/wsm6_constants.py`, `physics/sfclay_revised_mm5.py`, the revised-MM5 constants in `physics/surface_constants.py`; MYNN code where it derives from WRF's `physics_mmm` copy | BSD 3-Clause, NCAR (text below) |
| Ooura error function | `src/gpuwrf/physics/_morrison_aero_cold.py` (DERF1 coefficients) | Ooura terms below |
| Switzerland example inputs | `examples/switzerland_d01/wrfinput_d01`, `wrfbdy_d01` | NCEP GFS analysis (US Government, public domain) processed with WPS/real.exe; WPS static geography datasets per their providers |
| Fixtures from the maintainer's own WRF runs | `data/fixtures/m6/`, `fixtures/samples/canary-*`, a few test/proof `.npz` files | Forcing data from public NWP sources (GFS public domain; where ECMWF open data was used: © ECMWF, CC BY 4.0) |

The project does not vendor third-party Python, JavaScript, CSS or fonts; runtime dependencies (JAX, NumPy, netCDF4, …)
are installed separately under their own licenses. Noah-MP and MYNN-EDMF are translated from the copies distributed
with WRF v4; their upstream repositories (NCAR/noahmp, NOAA GSL) apply.

## UCAR / NCAR — WRF

> WRF was developed at the National Center for Atmospheric Research (NCAR) which is operated by the University
> Corporation for Atmospheric Research (UCAR). NCAR and UCAR make no proprietary claims, either statutory or
> otherwise, to this version and release of WRF and consider WRF to be in the public domain for use by any person
> or entity for any purpose without any fee or charge. UCAR requests that any WRF user include this notice on any
> partial or full copies of WRF. WRF is provided on an "AS IS" basis and any warranties, either express or implied,
> including but not limited to implied warranties of non-infringement, originality, merchantability and fitness for
> a particular purpose, are disclaimed. In no event shall UCAR be liable for any damages, whatsoever, whether
> direct, indirect, consequential or special, that arise out of or in connection with the access, use or
> performance of WRF, including infringement actions.
>
> WRF® is a registered trademark of the University Corporation for Atmospheric Research (UCAR).

wrf_gpu is "WRF-compatible"; it is not WRF and does not use the WRF name as its own brand.

## Atmospheric & Environmental Research, Inc. (AER) — RRTMG / RRTM

The RRTMG long- and shortwave code and coefficient data distributed with WRF carry the following notice
(year ranges 2002-2008, 2002-2009 and 2006-2008 in the individual WRF modules):

> Copyright 2002-2009, Atmospheric & Environmental Research, Inc. (AER).
> This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is
> reproduced on each copy made. This model is provided as is without any express or implied warranties.
> (http://www.rtweb.aer.com/)

The files listed under "RRTMG / RRTM radiation" above are translations of, or contain data from, that code. They are
distributed under these AER terms, not under the MIT License: in particular they may not be sold.

## NCAR MMM physics — BSD 3-Clause License

```
BSD 3-Clause License

Copyright (c) 2022, National Center for Atmospheric Research
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Takuya Ooura — error function coefficients

> Copyright(C) 1996 Takuya OOURA. You may use, copy, modify this code for any purpose and without fee.

## KISS random number generator

The KISS generator used for McICA cloud sampling is marked as public domain code in WRF's RRTMG modules.
