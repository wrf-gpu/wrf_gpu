! =====================================================================
! v0.6.0 single-column classic RRTM longwave (ra_lw_physics=1) oracle.
!
! Drives the UNMODIFIED WRF phys/module_ra_rrtm.F:RRTMLWRAD (which calls
! the 16-band RRTM k-distribution column model) on prescribed
! single-column soundings. The k-distribution lookup tables are loaded
! once from the unmodified RRTM_DATA asset via rrtminit(allowed_to_read=T).
! The dump captures inputs plus the WRF LW potential-temperature tendency
! RTHRATEN and the surface downwelling LW flux GLW and TOA OLR for JAX
! savepoint parity. This is a real WRF-module oracle, not a JAX
! self-compare; it is not a full coupled wrf.exe run.
! =====================================================================
PROGRAM rrtm_lw_oracle
  USE module_ra_rrtm, ONLY : RRTMLWRAD, rrtminit
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: P1000  = 1.0E5

  INTEGER, PARAMETER :: KX = 32
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

  p_top     = 5000.0        ! Pa
  icloud    = 1
  ghg_input = 0             ! pre-V3.5 fixed trace gases path
  yr        = 2009
  julian_d  = 171.5
  warm_rain = .FALSE.
  f_qv=.TRUE.; f_qc=.TRUE.; f_qr=.TRUE.; f_qi=.TRUE.; f_qs=.TRUE.; f_qg=.TRUE.

  ! Load the RRTM k-distribution tables from the unmodified RRTM_DATA asset.
  CALL rrtminit(p_top, .TRUE., &
                ids,ide,jds,jde,kds,kde, &
                ims,ime,jms,jme,kms,kme, &
                its,ite,jts,jte,kts,kte)

  CALL build_case(case_id, regime, T3D, T8W, QV3D, QC3D, QR3D, QI3D, QS3D, QG3D, &
                  P3D, P8W, PI3D, DZ8W, RHO3D, CLDFRA3D, EMISS, TSK, icloud)

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

  SUBROUTINE build_case(cid, name, Tt, Tw, Qv, Qc, Qr, Qi, Qs, Qg, Pp, Pw, &
                        Exner, Dz, Rho, Cldf, EmA, TskA, icld)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: &
        Tt,Tw,Qv,Qc,Qr,Qi,Qs,Qg,Pp,Pw,Exner,Dz,Rho,Cldf
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: EmA,TskA
    INTEGER, INTENT(OUT) :: icld
    REAL, DIMENSION(KX+1) :: zint, pint, tint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, z, tv, pim
    REAL :: em0, tsk0, qc0, qi0, qr0, qs0, qg0, cf0, cldbot, cldtop
    INTEGER :: k

    ztop = 16000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.15
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    icld = 1
    qc0=0.0; qi0=0.0; qr0=0.0; qs0=0.0; qg0=0.0; cf0=0.0; cldbot=0.0; cldtop=0.0

    SELECT CASE (cid)
    CASE (1)
      name='clearsky_warm_land'
      psfc0=100000.0; theta0=300.0; zml=1200.0; lapse_ml=0.0004; lapse_ft=0.0045
      q0=0.0120; qscale=2500.0; em0=0.98; tsk0=301.0
    CASE (2)
      name='clearsky_cold_dry'
      psfc0=100500.0; theta0=270.0; zml=400.0; lapse_ml=0.0040; lapse_ft=0.0055
      q0=0.0030; qscale=1800.0; em0=0.95; tsk0=268.0
    CASE (3)
      name='nocturnal_inversion'
      psfc0=100500.0; theta0=288.0; zml=200.0; lapse_ml=0.0080; lapse_ft=0.0050
      q0=0.0080; qscale=2200.0; em0=0.97; tsk0=285.0
    CASE (4)
      name='thick_warm_cloud'
      psfc0=100800.0; theta0=298.0; zml=1100.0; lapse_ml=0.0004; lapse_ft=0.0042
      q0=0.0150; qscale=2700.0; em0=0.98; tsk0=299.0
      qc0=8.0E-4; qr0=2.0E-4; cf0=1.0; cldbot=500.0; cldtop=3000.0
    CASE (5)
      name='ice_cloud_marine'
      psfc0=101200.0; theta0=295.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0038
      q0=0.0110; qscale=2600.0; em0=0.985; tsk0=296.0
      qi0=3.0E-4; qs0=1.5E-4; cf0=0.9; cldbot=5000.0; cldtop=9000.0
    CASE (6)
      name='moist_tropical'
      psfc0=101000.0; theta0=302.0; zml=1500.0; lapse_ml=0.0003; lapse_ft=0.0040
      q0=0.0190; qscale=3000.0; em0=0.99; tsk0=303.0
    CASE DEFAULT
      name='snow_graupel_clouds'
      psfc0=99500.0; theta0=297.0; zml=1000.0; lapse_ml=0.0006; lapse_ft=0.0040
      q0=0.0130; qscale=2400.0; em0=0.97; tsk0=298.0
      qc0=4.0E-4; qs0=3.0E-4; qg0=2.0E-4; qr0=1.0E-4; cf0=1.0; cldbot=800.0; cldtop=6000.0
    END SELECT

    pint(1) = psfc0
    tint(1) = tsk0
    DO k = 1, KX
      z = zmid(k)
      IF (z <= zml) THEN
        theta(k) = theta0 + lapse_ml*z
      ELSE
        theta(k) = theta0 + lapse_ml*zml + lapse_ft*(z-zml)
      END IF
      qprof(k) = MAX(q0*EXP(-z/qscale), 1.0E-6)
      pim = (MAX(pint(k), 1000.0)/P1000)**ROVCP
      temp(k) = theta(k)*pim
      tv = temp(k)*(1.0 + EP1*qprof(k))
      pint(k+1) = pint(k)*EXP(-G*(zint(k+1)-zint(k))/(R_D*MAX(tv, 180.0)))
      pfull(k) = 0.5*(pint(k)+pint(k+1))
      pim = (pfull(k)/P1000)**ROVCP
      temp(k) = theta(k)*pim
    END DO
    ! interface temperatures (t8w) by linear interp of layer temps in z.
    DO k = 2, KX
      tint(k) = 0.5*(temp(k-1)+temp(k))
    END DO
    tint(KX+1) = temp(KX) + (temp(KX)-temp(KX-1))*0.5

    Tt=0.0; Tw=0.0; Qv=0.0; Qc=0.0; Qr=0.0; Qi=0.0; Qs=0.0; Qg=0.0
    Pp=0.0; Pw=0.0; Exner=0.0; Dz=0.0; Rho=0.0; Cldf=0.0
    DO k = 1, KX
      Tt(1,k,1)  = temp(k)
      Qv(1,k,1)  = qprof(k)
      Pp(1,k,1)  = pfull(k)
      Exner(1,k,1) = (pfull(k)/P1000)**ROVCP
      Dz(1,k,1)  = zint(k+1)-zint(k)
      Rho(1,k,1) = pfull(k)/(R_D*temp(k)*(1.0+EP1*qprof(k)))
      IF (zmid(k) >= cldbot .AND. zmid(k) <= cldtop) THEN
        Qc(1,k,1)=qc0; Qr(1,k,1)=qr0; Qi(1,k,1)=qi0
        Qs(1,k,1)=qs0; Qg(1,k,1)=qg0; Cldf(1,k,1)=cf0
      END IF
    END DO
    ! interface (full-level) pressure/temperature on kme levels.
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
END PROGRAM rrtm_lw_oracle
