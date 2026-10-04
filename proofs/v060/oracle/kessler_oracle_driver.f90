! =====================================================================
! v0.6.0 single-column Kessler (WRF mp_physics=1) oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_mp_kessler.F subroutine kessler
! on prescribed warm-rain columns and dumps full inputs and outputs.
! The project-authored code here only builds columns and serializes values;
! the physics oracle is the real WRF Fortran source copied at build time.
! =====================================================================
PROGRAM kessler_oracle
  USE module_mp_kessler, ONLY : kessler
  IMPLICIT NONE

  REAL, PARAMETER :: G       = 9.81
  REAL, PARAMETER :: R_D     = 287.0
  REAL, PARAMETER :: CP      = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V     = 461.6
  REAL, PARAMETER :: XLV     = 2.5E6
  REAL, PARAMETER :: EP2     = R_D/R_V
  REAL, PARAMETER :: SVP1    = 0.6112
  REAL, PARAMETER :: SVP2    = 17.67
  REAL, PARAMETER :: SVP3    = 29.65
  REAL, PARAMETER :: SVPT0   = 273.15
  REAL, PARAMETER :: RHOWATER= 1000.0
  REAL, PARAMETER :: P1000   = 1.0E5
  REAL, PARAMETER :: ROVCP   = R_D/CP

  INTEGER, PARAMETER :: KX=32
  INTEGER, PARAMETER :: ids=1, ide=1, jds=1, jde=1, kds=1, kde=KX
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: th, qv, qc, qr, rho, pii, z, dz8w
  REAL, DIMENSION(kts:kte) :: th0, qv0, qc0, qr0, rho0, pii0, z0, dz0
  REAL, DIMENSION(ims:ime,jms:jme) :: rainnc, rainncv
  REAL :: dt
  INTEGER :: k, case_id
  CHARACTER(LEN=32) :: arg

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  dt = 90.0
  CALL build_column(case_id, th, qv, qc, qr, rho, pii, z, dz8w)

  DO k = kts, kte
    th0(k)  = th(1,k,1)
    qv0(k)  = qv(1,k,1)
    qc0(k)  = qc(1,k,1)
    qr0(k)  = qr(1,k,1)
    rho0(k) = rho(1,k,1)
    pii0(k) = pii(1,k,1)
    z0(k)   = z(1,k,1)
    dz0(k)  = dz8w(1,k,1)
  END DO

  rainnc = 0.0
  rainncv = 0.0

  CALL kessler(t=th, qv=qv, qc=qc, qr=qr, rho=rho, pii=pii, dt_in=dt, z=z, &
       xlv=XLV, cp=CP, ep2=EP2, svp1=SVP1, svp2=SVP2, svp3=SVP3,          &
       svpt0=SVPT0, rhowater=RHOWATER, dz8w=dz8w, rainnc=rainnc,           &
       rainncv=rainncv, ids=ids, ide=ide, jds=jds, jde=jde, kds=kds,       &
       kde=kde, ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,      &
       its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', dt
  WRITE(*,'(A,L1)') 'FULL_WRF_EXE=', .FALSE.
  CALL dump_col('THETA_IN', th0)
  CALL dump_col('QV_IN', qv0)
  CALL dump_col('QC_IN', qc0)
  CALL dump_col('QR_IN', qr0)
  CALL dump_col('RHO', rho0)
  CALL dump_col('PII', pii0)
  CALL dump_col('Z', z0)
  CALL dump_col('DZ8W', dz0)
  CALL dump_col3('THETA_OUT', th)
  CALL dump_col3('QV_OUT', qv)
  CALL dump_col3('QC_OUT', qc)
  CALL dump_col3('QR_OUT', qr)
  WRITE(*,'(A,ES23.15)') 'RAINNC=', rainnc(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=', rainncv(1,1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col3(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col3

  SUBROUTINE build_column(cid, th, qv, qc, qr, rho, pii, z, dz8w)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: th, qv, qc, qr
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: rho, pii, z, dz8w
    REAL :: psfc, tsfc, theta_sfc, ztop, zml, lapse, rh_ml, rh_top
    REAL :: th_k, t_k, p_k, p_mp, tv_k, z_k, es, qsw, rh_k
    REAL, DIMENSION(KX) :: zz
    INTEGER :: kk

    ztop = 12000.0
    DO kk = 1, KX
      zz(kk) = ztop * ((REAL(kk)-0.5)/REAL(KX))**1.10
    END DO

    SELECT CASE (cid)
    CASE (1)   ! condensation: supersaturated warm cloud, no initial rain
      psfc=1000.0E2; tsfc=294.0; zml=2200.0; lapse=4.0E-3; rh_ml=1.06; rh_top=0.78
    CASE (2)   ! autoconversion: cloud water above 1 g/kg threshold
      psfc=1000.0E2; tsfc=296.0; zml=1800.0; lapse=4.5E-3; rh_ml=0.99; rh_top=0.70
    CASE (3)   ! accretion: cloud and rain together
      psfc=980.0E2;  tsfc=292.0; zml=1600.0; lapse=5.0E-3; rh_ml=0.98; rh_top=0.68
    CASE (4)   ! evaporation: rain falling into dry air
      psfc=940.0E2;  tsfc=286.0; zml=500.0;  lapse=5.5E-3; rh_ml=0.55; rh_top=0.35
    CASE (5)   ! sedimentation/fall stress case
      psfc=1000.0E2; tsfc=298.0; zml=1000.0; lapse=5.5E-3; rh_ml=0.92; rh_top=0.55
    CASE DEFAULT
      psfc=1000.0E2; tsfc=294.0; zml=1000.0; lapse=5.0E-3; rh_ml=0.95; rh_top=0.60
    END SELECT

    theta_sfc = tsfc * (P1000/psfc)**ROVCP
    p_k = psfc
    DO kk = 1, KX
      z_k = zz(kk)
      IF (z_k <= zml) THEN
        th_k = theta_sfc
        rh_k = rh_ml
      ELSE
        th_k = theta_sfc + lapse*(z_k-zml)
        rh_k = rh_top + (rh_ml-rh_top)*EXP(-(z_k-zml)/2800.0)
      END IF
      IF (kk == 1) THEN
        t_k = th_k*(psfc/P1000)**ROVCP
        tv_k = t_k
        p_k = psfc*EXP(-G*zz(1)/(R_D*tv_k))
      ELSE
        t_k = th_k*(p_k/P1000)**ROVCP
        tv_k = t_k*(1.0+0.608*qv(1,kk-1,1))
        p_k = p_k*EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      pii(1,kk,1) = (p_k/P1000)**ROVCP
      t_k = th_k*pii(1,kk,1)
      p_mp = P1000*(pii(1,kk,1)**(1004.0/287.0))
      es = 1000.0*SVP1*EXP(SVP2*(t_k-SVPT0)/(t_k-SVP3))
      qsw = EP2*es/(p_mp-es)
      qv(1,kk,1) = MAX(rh_k*qsw, 1.0E-8)
      th(1,kk,1) = th_k
      tv_k = t_k*(1.0+0.608*qv(1,kk,1))
      rho(1,kk,1) = p_k/(R_D*tv_k)
      z(1,kk,1) = zz(kk)
      IF (kk == 1) THEN
        dz8w(1,kk,1) = 2.0*zz(1)
      ELSE
        dz8w(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
    END DO

    qc = 0.0
    qr = 0.0
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)
        IF (z_k < 2500.0) qc(1,kk,1) = 2.0E-4*EXP(-((z_k-1100.0)/850.0)**2)
      CASE (2)
        IF (z_k < 3200.0) qc(1,kk,1) = 2.1E-3*EXP(-((z_k-1400.0)/950.0)**2)
      CASE (3)
        IF (z_k < 3500.0) qc(1,kk,1) = 1.1E-3*EXP(-((z_k-1600.0)/1100.0)**2)
        IF (z_k < 4300.0) qr(1,kk,1) = 7.5E-4*EXP(-((z_k-1900.0)/1300.0)**2)
      CASE (4)
        IF (z_k < 5000.0) qr(1,kk,1) = 6.5E-4*EXP(-((z_k-2600.0)/1700.0)**2)
      CASE (5)
        IF (z_k < 2800.0) qc(1,kk,1) = 6.0E-4*EXP(-((z_k-1200.0)/1000.0)**2)
        IF (z_k < 5200.0) qr(1,kk,1) = 1.8E-3*EXP(-((z_k-1800.0)/1600.0)**2)
      END SELECT
    END DO
  END SUBROUTINE build_column

END PROGRAM kessler_oracle
