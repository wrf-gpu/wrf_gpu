! =====================================================================
! v0.13 NON-5000-PTOP classic RRTM longwave (ra_lw_physics=1) oracle.
!
! Companion to rrtm_lw_oracle_driver.f90.  Drives the SAME UNMODIFIED WRF
! phys/module_ra_rrtm.F:RRTMLWRAD but at a model-top pressure p_top /= 5000
! Pa, to exercise the grid-aware above-model-top buffer sizing
! (NLAYERS = kme + nint(p_top*0.01/deltap) - 1, module_ra_rrtm.F:6781).
! WRF itself always sizes the buffer from the runtime p_top; the JAX port
! previously hardcoded p_top=5000 Pa (Finding F1).  This oracle proves the
! grid-aware JAX port matches WRF at a non-default top.
!
! The column is laid out directly in LOG-PRESSURE from psfc down to p_top so
! the top model-interface pressure equals p_top EXACTLY.
!
! Usage:  rrtm_lw_ptop_oracle <case_id>
!   case 1 = low-top (p_top=10000 Pa=100mb): hardcoded nbuf=13 UNDERSHOOTS TOA
!   case 2 = high-top (p_top= 2000 Pa= 20mb): hardcoded nbuf=13 OVERSHOOTS to
!            NEGATIVE pressure (the Finding-F2 masking-clamp regime in JAX)
! =====================================================================
PROGRAM rrtm_lw_ptop_oracle
  USE module_ra_rrtm, ONLY : RRTMLWRAD, rrtminit
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: P1000  = 1.0E5

  INTEGER, PARAMETER :: KX = 40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: T3D,T8W,QV3D,QC3D,QR3D,QI3D,QS3D,QG3D
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: P3D,P8W,PI3D,DZ8W,RHO3D,CLDFRA3D
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RTHRATEN,RTHRATENC
  REAL, DIMENSION(ims:ime,jms:jme) :: GLW,OLR,EMISS,TSK
  CHARACTER(LEN=32) :: arg, regime
  INTEGER :: case_id, icloud, ghg_input, yr
  REAL :: p_top, julian_d
  LOGICAL :: warm_rain, f_qv,f_qc,f_qr,f_qi,f_qs,f_qg

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  SELECT CASE (case_id)
  CASE (1)
    p_top  = 10000.0      ! Pa = 100 mb (low-top: hardcoded nbuf=13 undershoots)
    regime = 'lowtop_ptop100mb'
  CASE DEFAULT
    p_top  = 2000.0       ! Pa = 20 mb (high-top: hardcoded nbuf=13 -> neg p)
    regime = 'hightop_ptop20mb'
  END SELECT

  icloud    = 1
  ghg_input = 0             ! pre-V3.5 fixed trace gases path
  yr        = 2009
  julian_d  = 171.5
  warm_rain = .FALSE.
  f_qv=.TRUE.; f_qc=.TRUE.; f_qr=.TRUE.; f_qi=.TRUE.; f_qs=.TRUE.; f_qg=.TRUE.

  ! Load the RRTM k-distribution tables; rrtminit sizes NLAYERS from p_top.
  CALL rrtminit(p_top, .TRUE., &
                ids,ide,jds,jde,kds,kde, &
                ims,ime,jms,jme,kms,kme, &
                its,ite,jts,jte,kts,kte)

  CALL build_case(p_top, T3D, T8W, QV3D, QC3D, QR3D, QI3D, QS3D, QG3D, &
                  P3D, P8W, PI3D, DZ8W, RHO3D, CLDFRA3D, EMISS, TSK)

  RTHRATEN = 0.0; RTHRATENC = 0.0
  GLW = 0.0; OLR = 0.0

  CALL RRTMLWRAD(p_top=p_top, rthraten=RTHRATEN, rthratenc=RTHRATENC, &
                 glw=GLW, olr=OLR, emiss=EMISS, p8w=P8W, p3d=P3D, pi3d=PI3D, &
                 dz8w=DZ8W, tsk=TSK, t3d=T3D, t8w=T8W, rho3d=RHO3D, r=R_D, g=G, &
                 icloud=icloud, warm_rain=warm_rain, &
                 ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
                 ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
                 its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte, &
                 qv3d=QV3D, qc3d=QC3D, qr3d=QR3D, qi3d=QI3D, qs3d=QS3D, qg3d=QG3D, &
                 cldfra3d=CLDFRA3D, &
                 f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg, &
                 yr=yr, julian=julian_d, ghg_input=ghg_input)

  CALL dump_case(case_id, regime, T3D, T8W, QV3D, QC3D, QR3D, QI3D, QS3D, QG3D, &
                 P3D, P8W, PI3D, DZ8W, RHO3D, CLDFRA3D, EMISS, TSK, icloud, &
                 RTHRATEN, GLW, OLR, p_top, yr, julian_d)

CONTAINS

  SUBROUTINE build_case(ptop, Tt, Tw, Qv, Qc, Qr, Qi, Qs, Qg, Pp, Pw, &
                        Exner, Dz, Rho, Cldf, EmA, TskA)
    REAL, INTENT(IN) :: ptop
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: &
        Tt,Tw,Qv,Qc,Qr,Qi,Qs,Qg,Pp,Pw,Exner,Dz,Rho,Cldf
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: EmA,TskA
    REAL, DIMENSION(KX+1) :: pint, tint, zint
    REAL, DIMENSION(KX) :: theta, qprof, temp, pfull
    REAL :: psfc0, tsfc0, lapse_trop, q0, qscale, ztrop, tstrat
    REAL :: em0, tsk0, tv, znom, logp_top, logp_sfc, frac
    INTEGER :: k

    ! Mid-latitude clearsky sounding built directly on TEMPERATURE (not theta) so
    ! the stratosphere stays realistic up to a 20 mb / 100 mb top -- a tropospheric
    ! lapse to a tropopause then an isothermal stratosphere (WRF's buffer-temp
    ! table extrapolation needs a physical top T, else RTRN diverges). Clouds off.
    psfc0=100000.0; tsfc0=290.0
    lapse_trop=0.0065      ! K/m troposphere lapse
    ztrop=11000.0          ! tropopause height (m)
    tstrat=290.0-0.0065*11000.0   ! ~218.5 K isothermal stratosphere
    q0=0.0100; qscale=2500.0; em0=0.98; tsk0=292.0

    ! Interface pressures in EXACT log spacing psfc -> ptop so Pw(kme)=ptop.
    logp_sfc = LOG(psfc0)
    logp_top = LOG(ptop)
    DO k = 1, KX+1
      frac = REAL(k-1)/REAL(KX)
      pint(k) = EXP(logp_sfc + frac*(logp_top - logp_sfc))
    END DO

    ! Temperature from a US-standard-atmosphere-like T(z): troposphere lapse to a
    ! tropopause, then isothermal stratosphere. z is a nominal hydrostatic height.
    tint(1) = tsk0
    DO k = 1, KX
      pfull(k) = 0.5*(pint(k)+pint(k+1))
      znom = -7000.0*LOG(pfull(k)/psfc0)   ! approx geometric height (m)
      IF (znom <= ztrop) THEN
        temp(k) = tsfc0 - lapse_trop*znom
      ELSE
        temp(k) = tstrat
      END IF
      qprof(k) = MAX(q0*EXP(-znom/qscale), 1.0E-6)
      theta(k) = temp(k)*(P1000/pfull(k))**ROVCP
    END DO
    DO k = 2, KX
      tint(k) = 0.5*(temp(k-1)+temp(k))
    END DO
    tint(KX+1) = temp(KX) + (temp(KX)-temp(KX-1))*0.5

    ! Layer thickness from hydrostatic dz = -R_D*Tv/g * dln(p).
    Tt=0.0; Tw=0.0; Qv=0.0; Qc=0.0; Qr=0.0; Qi=0.0; Qs=0.0; Qg=0.0
    Pp=0.0; Pw=0.0; Exner=0.0; Dz=0.0; Rho=0.0; Cldf=0.0
    DO k = 1, KX
      tv = temp(k)*(1.0 + EP1*qprof(k))
      Tt(1,k,1)   = temp(k)
      Qv(1,k,1)   = qprof(k)
      Pp(1,k,1)   = pfull(k)
      Exner(1,k,1)= (pfull(k)/P1000)**ROVCP
      Dz(1,k,1)   = (R_D*tv/G)*LOG(pint(k)/pint(k+1))
      Rho(1,k,1)  = pfull(k)/(R_D*temp(k)*(1.0+EP1*qprof(k)))
    END DO
    DO k = 1, KX+1
      Pw(1,k,1) = pint(k)
      Tw(1,k,1) = tint(k)
    END DO
    EmA(1,1)=em0; TskA(1,1)=tsk0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(label, arr)
    CHARACTER(LEN=*), INTENT(IN) :: label
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') label,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col_interface(label, arr)
    CHARACTER(LEN=*), INTENT(IN) :: label
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte+1
      WRITE(*,'(A,A,I0,A,ES23.15)') label,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col_interface

  SUBROUTINE dump_case(cid, name, Tt, Tw, Qv, Qc, Qr, Qi, Qs, Qg, Pp, Pw, &
                       Exner, Dz, Rho, Cldf, EmA, TskA, icld, &
                       Rth, GlwA, OlrA, ptop, yrA, julA)
    INTEGER, INTENT(IN) :: cid, icld, yrA
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        Tt,Tw,Qv,Qc,Qr,Qi,Qs,Qg,Pp,Pw,Exner,Dz,Rho,Cldf,Rth
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: EmA,TskA,GlwA,OlrA
    REAL, INTENT(IN) :: ptop, julA
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'ICLOUD=', icld
    WRITE(*,'(A,I0)') 'YR=', yrA
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES23.15)') 'PTOP=', ptop
    WRITE(*,'(A,ES23.15)') 'JULIAN=', julA
    WRITE(*,'(A,ES23.15)') 'EMISS=', EmA(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', TskA(1,1)
    WRITE(*,'(A,ES23.15)') 'GLW=', GlwA(1,1)
    WRITE(*,'(A,ES23.15)') 'OLR=', OlrA(1,1)
    CALL dump_col('T', Tt)
    CALL dump_col_interface('T8W', Tw)
    CALL dump_col('QV', Qv)
    CALL dump_col('QC', Qc)
    CALL dump_col('QR', Qr)
    CALL dump_col('QI', Qi)
    CALL dump_col('QS', Qs)
    CALL dump_col('QG', Qg)
    CALL dump_col('CLDFRA', Cldf)
    CALL dump_col('P', Pp)
    CALL dump_col_interface('P8W', Pw)
    CALL dump_col('PI', Exner)
    CALL dump_col('DZ', Dz)
    CALL dump_col('RHO', Rho)
    CALL dump_col('RTHRATEN', Rth)
  END SUBROUTINE dump_case
END PROGRAM rrtm_lw_ptop_oracle
