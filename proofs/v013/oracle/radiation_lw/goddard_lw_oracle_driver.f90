! =====================================================================
! v0.13 Tier-3 RADIATION batch2: GSFC/Goddard NUWRF LONGWAVE
! (ra_lw_physics=5) single-column fp64 ORACLE driver.
!
! Drives the UNMODIFIED-PHYSICS WRF Goddard longwave core
! ``phys/module_ra_goddard.F:lwrad`` (the Chou-Suarez 1994 10-band
! correlated-k IR radiative-transfer kernel) on prescribed
! single-column soundings and dumps, per regime:
!
!   * the full input column (pressure levels pl(np+1), layer T/qv/o3,
!     cloud fraction fcld, surface skin temperature ts, near-surface
!     air temperature tb, per-band emissivity emiss(nband_lw)),
!   * the lwrad outputs: net all-sky LW flux profile flx(np+1),
!     downwelling acflxd(np+1) and upwelling acflxu(np+1) (both
!     W m^-2, top-down, kts=TOA .. kte=BOA),
!   * the WRF-derived surface downwelling GLW = acflxd(np+1)
!     (the boundary flux a ra_lw_physics=5 forecast feeds the LSM),
!     the TOA upwelling OLR = acflxu(1), and the layer heating-rate
!     tten(np) = -(0.01*g/cp)*(flx(k-1)-flx(k))/(pl(k-1)-pl(k))
!     [K s^-1], exactly the goddardrad LW heating-rate map.
!
! The Goddard module's internal working precision (fp_kind) is ALREADY
! double (SELECTED_REAL_KIND(15,307)), so this is the canonical fp64
! reference for the (fp64) JAX port -- there is NO JAX-vs-JAX
! self-compare. The aerosol optics (taual/ssaal/asyal) are passed ZERO,
! matching the operational GPU build's aer_ra_feedback/=1 default
! (identical convention to the GSFC SW oracle); the cloud optics
! (taucl/ssacl/asycl) are passed for the cloudy regimes from a simple
! WRF-style liquid/ice optical-depth diagnostic so the cloud-overlap
! (cldovlp) path is exercised.
!
! lwrad is PRIVATE in the pristine module; the build script applies a
! single-line, checksummed VISIBILITY-ONLY shim (`public :: lwrad`)
! that adds NO physics change -- the kernel source is byte-identical.
! Both the pristine checksum and the post-shim checksum are recorded in
! the build manifest so the shim is auditable.
!
! Compiled fp64 (the module is already double). CPU-only, cores 0-3,
! conda env `wrfbuild`.  Usage: ./goddard_lw_oracle <case_id>;
! output = flat key=value text consumed by dump_to_json.py.
!
! STATUS: this is STAGED oracle INFRASTRUCTURE. The faithful JAX
! ra_lw=5 column kernel is a documented v0.13 carry-over (the combined
! SW+LW Goddard module is ~12.5k LOC / ~11.8k hardcoded LW coefficients
! -- too large for a faithful single-session port without becoming a
! self-compare / happy-path). This driver + build script + dumper are
! delivered so the future faithful port has a ready non-self-compare
! fp64 reference. See proofs/v013/t3_gsfc_lw_oracle.py for the
! reference-only classification + carry-over rationale.
! =====================================================================
PROGRAM goddard_lw_oracle
  USE module_ra_goddard, ONLY : lwrad
  IMPLICIT NONE

  ! Goddard LW band count (module_ra_goddard.F: nband_lw = 10).
  INTEGER, PARAMETER :: NBAND_LW = 10
  ! Goddard vector chunk (module_ra_goddard.F: chunk = 16). We solve a
  ! single physical column ic=1; irestrict=1 masks the padded lanes.
  INTEGER, PARAMETER :: CHUNK = 16
  INTEGER, PARAMETER :: DP = SELECTED_REAL_KIND(15, 307)

  ! Physical constants (WRF module_model_constants.F values).
  REAL(DP), PARAMETER :: G  = 9.80665_DP
  REAL(DP), PARAMETER :: CP = 1004.0_DP

  INTEGER, PARAMETER :: NP = 40        ! number of model layers (top-down)
  INTEGER :: case_id
  CHARACTER(LEN=32) :: arg

  ! lwrad arguments (top-down: index 1 = TOA layer, np = BOA layer;
  ! interface levels 1..np+1, level 1 = TOA, np+1 = surface).
  INTEGER :: ict(CHUNK), icb(CHUNK), irestrict
  REAL(DP) :: pl(CHUNK, NP+1), ta(CHUNK, NP), wa(CHUNK, NP), oa(CHUNK, NP)
  REAL(DP) :: tb(CHUNK), ts(CHUNK), fcld(CHUNK, NP)
  REAL(DP) :: emiss(CHUNK, NBAND_LW)
  REAL(DP) :: taucl(CHUNK, NP, NBAND_LW), ssacl(CHUNK, NP, NBAND_LW), asycl(CHUNK, NP, NBAND_LW)
  REAL(DP) :: taual(CHUNK, NP, NBAND_LW), ssaal(CHUNK, NP, NBAND_LW), asyal(CHUNK, NP, NBAND_LW)
  ! lwrad output flux args are DEFAULT real in the pristine source.
  REAL :: flx_out(CHUNK, NP+1), acflxd_out(CHUNK, NP+1), acflxu_out(CHUNK, NP+1)

  REAL(DP) :: tten(NP), glw, olr, dp_lev
  INTEGER :: k, ib

  IF (COMMAND_ARGUMENT_COUNT() < 1) THEN
    WRITE(*,*) 'usage: goddard_lw_oracle <case_id 1..6>'
    STOP 1
  END IF
  CALL GET_COMMAND_ARGUMENT(1, arg)
  READ(arg, *) case_id

  CALL build_sounding(case_id)

  ! Aerosol optics OFF (operational aer_ra_feedback/=1 default), like the SW oracle.
  taual = 0.0_DP; ssaal = 0.0_DP; asyal = 0.0_DP

  irestrict = 1                       ! one physical column (lane ic=1)
  ict(:) = 0; icb(:) = 0
  ict(1) = high_mid_split()           ! ~400 mb high/middle cloud group split
  icb(1) = mid_low_split()            ! ~700 mb middle/low cloud group split

  CALL lwrad(np=NP, emiss=emiss, tb=tb, ts=ts, ict=ict, icb=icb, &
             pl=pl, ta=ta, wa=wa, oa=oa, fcld=fcld, &
             taucl=taucl, ssacl=ssacl, asycl=asycl, &
             taual=taual, ssaal=ssaal, asyal=asyal, &
             flx_out=flx_out, acflxd_out=acflxd_out, acflxu_out=acflxu_out, &
             irestrict=irestrict)

  ! Heating rate (K/s), the goddardrad LW map: tten over the np layers.
  DO k = 1, NP
    ! interface k is "above" layer k (top-down); use the same finite
    ! difference goddardrad applies, kt=kts..kte over the model layers.
    dp_lev = pl(1, k) - pl(1, k+1)       ! mb across the layer (negative; matches WRF sign)
    tten(k) = -(0.01_DP * G / CP) * (REAL(flx_out(1, k), DP) - REAL(flx_out(1, k+1), DP)) / dp_lev
  END DO
  glw = REAL(acflxd_out(1, NP+1), DP)    ! surface downwelling LW [W/m2]
  olr = REAL(acflxu_out(1, 1), DP)       ! TOA upwelling LW (OLR) [W/m2]

  ! ---- emit flat key=value -----------------------------------------
  WRITE(*,'(A,I0)') 'case=', case_id
  WRITE(*,'(A,I0)') 'np=', NP
  WRITE(*,'(A,I0)') 'nband_lw=', NBAND_LW
  WRITE(*,'(A,I0)') 'ict=', ict(1)
  WRITE(*,'(A,I0)') 'icb=', icb(1)
  WRITE(*,'(A,ES23.15)') 'tb=', tb(1)
  WRITE(*,'(A,ES23.15)') 'ts=', ts(1)
  CALL emit_d('emiss', emiss(1, :), NBAND_LW)
  CALL emit_d('pl', pl(1, :), NP+1)
  CALL emit_d('ta', ta(1, :), NP)
  CALL emit_d('wa', wa(1, :), NP)
  CALL emit_d('oa', oa(1, :), NP)
  CALL emit_d('fcld', fcld(1, :), NP)
  CALL emit_d('taucl_b1', taucl(1, :, 1), NP)
  CALL emit_r('flx', flx_out(1, :), NP+1)
  CALL emit_r('acflxd', acflxd_out(1, :), NP+1)
  CALL emit_r('acflxu', acflxu_out(1, :), NP+1)
  CALL emit_d('tten', tten, NP)
  WRITE(*,'(A,ES23.15)') 'glw=', glw
  WRITE(*,'(A,ES23.15)') 'olr=', olr

CONTAINS

  INTEGER FUNCTION high_mid_split() RESULT(idx)
    ! top-down index nearest 400 mb (high/middle cloud group boundary).
    INTEGER :: kk
    idx = 2
    DO kk = 1, NP
      IF (0.5_DP * (pl(1, kk) + pl(1, kk+1)) >= 400.0_DP) THEN
        idx = kk
        EXIT
      END IF
    END DO
  END FUNCTION high_mid_split

  INTEGER FUNCTION mid_low_split() RESULT(idx)
    ! top-down index nearest 700 mb (middle/low cloud group boundary).
    INTEGER :: kk
    idx = NP - 1
    DO kk = 1, NP
      IF (0.5_DP * (pl(1, kk) + pl(1, kk+1)) >= 700.0_DP) THEN
        idx = kk
        EXIT
      END IF
    END DO
  END FUNCTION mid_low_split

  SUBROUTINE emit_d(name, arr, n)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: n
    REAL(DP), INTENT(IN) :: arr(n)
    INTEGER :: ii
    WRITE(*,'(A,A)', ADVANCE='NO') name, '='
    DO ii = 1, n
      WRITE(*,'(ES23.15)', ADVANCE='NO') arr(ii)
      IF (ii < n) WRITE(*,'(A)', ADVANCE='NO') ','
    END DO
    WRITE(*,*)
  END SUBROUTINE emit_d

  SUBROUTINE emit_r(name, arr, n)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: n
    REAL, INTENT(IN) :: arr(n)
    INTEGER :: ii
    WRITE(*,'(A,A)', ADVANCE='NO') name, '='
    DO ii = 1, n
      WRITE(*,'(ES23.15)', ADVANCE='NO') REAL(arr(ii), DP)
      IF (ii < n) WRITE(*,'(A)', ADVANCE='NO') ','
    END DO
    WRITE(*,*)
  END SUBROUTINE emit_r

  ! ------------------------------------------------------------------
  ! Idealized soundings (top-down: k=1 TOA .. k=NP BOA). Pressure is a
  ! simple sigma-like profile from ~10 mb (TOA interface) to ps surface;
  ! temperature is a tropospheric lapse + isothermal stratosphere; qv a
  ! decaying-with-height profile scaled by a regime RH; o3 a small mid-
  ! stratospheric bump. Regimes vary surface T, moisture, and cloud:
  !   1 clear tropical moist     2 clear mid-lat dry
  !   3 clear polar cold         4 single thick low water cloud
  !   5 high thin ice cloud      6 deep multi-layer cloud
  ! ------------------------------------------------------------------
  SUBROUTINE build_sounding(cid)
    INTEGER, INTENT(IN) :: cid
    REAL(DP) :: ps, t_sfc, rh, lapse, sigma, pmid, tlay, qsat
    REAL(DP) :: cloud_tau
    INTEGER :: kk, ktop, kbot

    pl = 0.0_DP; ta = 0.0_DP; wa = 0.0_DP; oa = 0.0_DP; fcld = 0.0_DP
    taucl = 0.0_DP; ssacl = 0.0_DP; asycl = 0.0_DP; emiss = 1.0_DP

    SELECT CASE (cid)
    CASE (1); ps = 1010.0_DP; t_sfc = 300.0_DP; rh = 0.80_DP; lapse = 6.5_DP
    CASE (2); ps = 1000.0_DP; t_sfc = 288.0_DP; rh = 0.35_DP; lapse = 6.5_DP
    CASE (3); ps =  990.0_DP; t_sfc = 250.0_DP; rh = 0.60_DP; lapse = 5.0_DP
    CASE (4); ps = 1008.0_DP; t_sfc = 295.0_DP; rh = 0.85_DP; lapse = 6.5_DP
    CASE (5); ps = 1005.0_DP; t_sfc = 292.0_DP; rh = 0.55_DP; lapse = 6.5_DP
    CASE (6); ps = 1010.0_DP; t_sfc = 298.0_DP; rh = 0.90_DP; lapse = 6.5_DP
    CASE DEFAULT
      WRITE(*,*) 'unknown case', cid; STOP 1
    END SELECT

    ! interface pressures (top-down): pl(1)=10 mb .. pl(NP+1)=ps.
    DO kk = 1, NP+1
      sigma = REAL(kk-1, DP) / REAL(NP, DP)
      pl(1, kk) = 10.0_DP + (ps - 10.0_DP) * sigma
    END DO
    ! layer mid temperatures, qv, o3.
    DO kk = 1, NP
      pmid = 0.5_DP * (pl(1, kk) + pl(1, kk+1))
      ! temperature: surface value, lapse to ~tropopause then isothermal.
      tlay = t_sfc - lapse * 0.001_DP * height_of(pmid, ps, t_sfc)
      tlay = MAX(tlay, 200.0_DP)
      ta(1, kk) = tlay
      ! water vapor mixing ratio [g/g] -> Goddard wants specific-humidity-like.
      qsat = 0.622_DP * 6.112_DP * EXP(17.67_DP * (tlay - 273.15_DP) / (tlay - 29.65_DP)) / pmid
      wa(1, kk) = MAX(rh * qsat * (pmid / ps)**1.0_DP, 1.0e-7_DP)
      ! ozone [g/g]: small mid-stratospheric peak near 30 mb.
      oa(1, kk) = 6.0e-6_DP * EXP(-((LOG(pmid) - LOG(25.0_DP))**2) / 1.5_DP) + 1.0e-8_DP
    END DO
    ts(1) = t_sfc
    tb(1) = ta(1, NP)
    emiss(1, :) = 0.98_DP

    ! cloudy regimes: place liquid/ice cloud with a WRF-style optical depth.
    SELECT CASE (cid)
    CASE (4)   ! thick low water cloud ~850-950 mb
      ktop = level_at(900.0_DP); kbot = level_at(950.0_DP)
      cloud_tau = 20.0_DP
      DO kk = ktop, kbot
        fcld(1, kk) = 0.95_DP
        DO ib = 1, NBAND_LW
          taucl(1, kk, ib) = cloud_tau
          ssacl(1, kk, ib) = 0.0_DP    ! LW: absorption-dominated
          asycl(1, kk, ib) = 0.85_DP
        END DO
      END DO
    CASE (5)   ! high thin ice cloud ~250-300 mb
      ktop = level_at(250.0_DP); kbot = level_at(320.0_DP)
      cloud_tau = 1.5_DP
      DO kk = ktop, kbot
        fcld(1, kk) = 0.6_DP
        DO ib = 1, NBAND_LW
          taucl(1, kk, ib) = cloud_tau
          ssacl(1, kk, ib) = 0.0_DP
          asycl(1, kk, ib) = 0.9_DP
        END DO
      END DO
    CASE (6)   ! deep multi-layer cloud: low + mid + high
      CALL set_cloud_layer(880.0_DP, 950.0_DP, 0.9_DP, 18.0_DP)
      CALL set_cloud_layer(550.0_DP, 650.0_DP, 0.7_DP, 6.0_DP)
      CALL set_cloud_layer(250.0_DP, 320.0_DP, 0.5_DP, 1.2_DP)
    END SELECT
  END SUBROUTINE build_sounding

  SUBROUTINE set_cloud_layer(ptop, pbot, frac, tau)
    REAL(DP), INTENT(IN) :: ptop, pbot, frac, tau
    INTEGER :: kk, ktop, kbot
    ktop = level_at(ptop); kbot = level_at(pbot)
    DO kk = ktop, kbot
      fcld(1, kk) = frac
      DO ib = 1, NBAND_LW
        taucl(1, kk, ib) = tau
        ssacl(1, kk, ib) = 0.0_DP
        asycl(1, kk, ib) = 0.85_DP
      END DO
    END DO
  END SUBROUTINE set_cloud_layer

  INTEGER FUNCTION level_at(p_target) RESULT(idx)
    REAL(DP), INTENT(IN) :: p_target
    REAL(DP) :: pmid
    INTEGER :: kk
    idx = NP
    DO kk = 1, NP
      pmid = 0.5_DP * (pl(1, kk) + pl(1, kk+1))
      IF (pmid >= p_target) THEN
        idx = kk
        EXIT
      END IF
    END DO
  END FUNCTION level_at

  REAL(DP) FUNCTION height_of(pmid, ps, t_sfc) RESULT(z)
    ! crude hypsometric height [m] from surface to pmid (for the lapse).
    REAL(DP), INTENT(IN) :: pmid, ps, t_sfc
    z = (287.0_DP * t_sfc / G) * LOG(ps / pmid)
  END FUNCTION height_of

END PROGRAM goddard_lw_oracle
