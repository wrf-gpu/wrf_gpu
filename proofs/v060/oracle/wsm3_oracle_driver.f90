! v0.6.0 WSM3 single-column oracle driver.
!
! Calls the UNMODIFIED WRF classic WSM3 simple-ice scheme in
! phys/module_mp_wsm3.F. WSM3's Registry footprint is moist:qv,qc,qr; the
! scheme's qci/qrs arrays are those qc/qr leaves whose phase interpretation
! depends on temperature.
PROGRAM wsm3_oracle
  USE module_model_constants, ONLY : G, R_D, CP, R_V, CPV, CLIQ, CICE, PSAT, &
       XLV, XLS, XLF, SVPT0, EP_1, EP_2, EPSILON, RHOAIR0, RHOWATER, RHOSNOW, &
       P1000MB, RCP, RE_QC_BG, RE_QI_BG, RE_QS_BG
  USE module_mp_wsm3, ONLY : wsm3, wsm3init
  IMPLICIT NONE

  INTEGER, PARAMETER :: KX=40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: th, qv, qci, qrs
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: w, den, pii, p, delz
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: re_cloud, re_ice, re_snow
  REAL, DIMENSION(ims:ime,jms:jme) :: rain, rainncv, snow, snowncv, sr
  REAL, DIMENSION(kts:kte) :: t0, qv0, qc0, qr0, pii0, w0
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

  CALL wsm3init(RHOAIR0, RHOWATER, RHOSNOW, CLIQ, CPV, .false.)
  CALL build_column(case_id, th, qv, qci, qrs, w, den, pii, p, delz)

  DO k = kts, kte
    t0(k) = th(1,k,1) * pii(1,k,1)
    qv0(k) = qv(1,k,1)
    qc0(k) = qci(1,k,1)
    qr0(k) = qrs(1,k,1)
    pii0(k) = pii(1,k,1)
    w0(k) = w(1,k,1)
  END DO

  rain=0.; rainncv=0.; snow=0.; snowncv=0.; sr=0.
  re_cloud=RE_QC_BG; re_ice=RE_QI_BG; re_snow=RE_QS_BG

  CALL wsm3(th=th, q=qv, qci=qci, qrs=qrs, w=w, den=den, pii=pii, p=p, delz=delz, &
       delt=dt, g=G, cpd=CP, cpv=CPV, rd=R_D, rv=R_V, t0c=SVPT0, ep1=EP_1, ep2=EP_2, &
       qmin=EPSILON, XLS=XLS, XLV0=XLV, XLF0=XLF, den0=RHOAIR0, denr=RHOWATER, &
       cliq=CLIQ, cice=CICE, psat=PSAT, rain=rain, rainncv=rainncv, snow=snow, &
       snowncv=snowncv, sr=sr, has_reqc=1, has_reqi=1, has_reqs=1, &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
       ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, ims=ims, ime=ime, &
       jms=jms, jme=jme, kms=kms, kme=kme, its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', dt
  CALL dump_col('T_IN', t0)
  CALL dump_col('QV_IN', qv0)
  CALL dump_col('QC_IN', qc0)
  CALL dump_col('QR_IN', qr0)
  CALL dump_col('W_IN', w0)
  CALL dump_col('PII', pii0)
  CALL dump_3d('DEN', den)
  CALL dump_3d('P', p)
  CALL dump_3d('DELZ', delz)
  CALL dump_temp('T_OUT', th, pii)
  CALL dump_3d('QV_OUT', qv)
  CALL dump_3d('QC_OUT', qci)
  CALL dump_3d('QR_OUT', qrs)
  CALL dump_3d('RE_CLOUD', re_cloud)
  CALL dump_3d('RE_ICE', re_ice)
  CALL dump_3d('RE_SNOW', re_snow)
  WRITE(*,'(A,ES23.15)') 'RAIN=', rain(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINNCV=', rainncv(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOW=', snow(1,1)
  WRITE(*,'(A,ES23.15)') 'SNOWNCV=', snowncv(1,1)
  WRITE(*,'(A,ES23.15)') 'SR=', sr(1,1)

CONTAINS
  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(kts:kte), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(kk)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_3d(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_3d

  SUBROUTINE dump_temp(name, theta, exner)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: theta, exner
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', theta(1,kk,1) * exner(1,kk,1)
    END DO
  END SUBROUTINE dump_temp

  SUBROUTINE build_column(cid, thh, qq, qcc, qrr, ww, dd, exner, pp, dz)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: thh, qq, qcc, qrr, ww
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: dd, exner, pp, dz
    REAL, DIMENSION(KX) :: zz
    REAL :: psfc, tsfc, zml, lapse, rh_ml, rh_trop, theta_sfc
    REAL :: th_k, t_k, p_k, tv_k, z_k, es, qsw, rh_k, ztop
    INTEGER :: kk

    ztop = 16000.0
    DO kk = 1, KX
      zz(kk) = ztop * ((REAL(kk)-0.5)/REAL(KX))**1.15
    END DO

    SELECT CASE (cid)
    CASE (1)
      psfc=1000.0E2; tsfc=298.0; zml=1500.0; lapse=5.0E-3; rh_ml=1.02; rh_trop=0.40
    CASE (2)
      psfc=1000.0E2; tsfc=287.0; zml=600.0; lapse=6.0E-3; rh_ml=0.98; rh_trop=0.60
    CASE (3)
      psfc=850.0E2; tsfc=258.0; zml=400.0; lapse=5.5E-3; rh_ml=1.05; rh_trop=0.70
    CASE (4)
      psfc=1000.0E2; tsfc=296.0; zml=1000.0; lapse=6.5E-3; rh_ml=1.00; rh_trop=0.65
    CASE (5)
      psfc=950.0E2; tsfc=283.0; zml=300.0; lapse=6.0E-3; rh_ml=0.55; rh_trop=0.30
    CASE DEFAULT
      psfc=1000.0E2; tsfc=295.0; zml=2000.0; lapse=5.0E-3; rh_ml=1.01; rh_trop=0.50
    END SELECT

    theta_sfc = tsfc * (P1000MB/psfc)**RCP
    p_k = psfc
    DO kk = 1, KX
      z_k = zz(kk)
      IF (z_k <= zml) THEN
        th_k = theta_sfc
        rh_k = rh_ml
      ELSE
        th_k = theta_sfc + lapse*(z_k-zml)
        rh_k = rh_trop + (rh_ml-rh_trop)*EXP(-(z_k-zml)/3000.0)
      END IF
      IF (kk == 1) THEN
        t_k = th_k*(psfc/P1000MB)**RCP
        p_k = psfc * EXP(-G*zz(1)/(R_D*t_k))
      ELSE
        t_k = th_k*(p_k/P1000MB)**RCP
        tv_k = t_k*(1.0+0.608*qq(1,kk-1,1))
        p_k = p_k * EXP(-G*(zz(kk)-zz(kk-1))/(R_D*tv_k))
      END IF
      t_k = th_k*(p_k/P1000MB)**RCP
      es = 610.78*EXP(17.27*(t_k-273.15)/(t_k-35.86))
      qsw = 0.622*es/(p_k-es)
      qq(1,kk,1) = MAX(rh_k*qsw, 1.0E-8)
      thh(1,kk,1) = th_k
      pp(1,kk,1) = p_k
      tv_k = t_k*(1.0+0.608*qq(1,kk,1))
      dd(1,kk,1) = p_k/(R_D*tv_k)
      exner(1,kk,1) = (p_k/P1000MB)**RCP
      IF (kk == 1) THEN
        dz(1,kk,1) = 2.0*zz(1)
      ELSE
        dz(1,kk,1) = zz(kk)-zz(kk-1)
      END IF
    END DO

    qcc=0.; qrr=0.; ww=0.
    DO kk = 1, KX
      z_k = zz(kk)
      SELECT CASE (cid)
      CASE (1)
        IF (z_k < 3000.0) qcc(1,kk,1) = 1.5E-3*EXP(-((z_k-1200.0)/900.0)**2)
        IF (z_k < 4000.0) qrr(1,kk,1) = 5.0E-4*EXP(-((z_k-1500.0)/1200.0)**2)
      CASE (2)
        IF (z_k < 4000.0) qcc(1,kk,1) = 8.0E-4*EXP(-((z_k-1500.0)/1500.0)**2)
        IF (z_k > 3500.0) qcc(1,kk,1) = qcc(1,kk,1) + 3.0E-4*EXP(-((z_k-7000.0)/2500.0)**2)
        qrr(1,kk,1) = 8.0E-4*EXP(-((z_k-5500.0)/3000.0)**2)
        ww(1,kk,1) = 2.0*EXP(-((z_k-4500.0)/1800.0)**2)
      CASE (3)
        qcc(1,kk,1) = 2.0E-4*EXP(-((z_k-6000.0)/3000.0)**2)
        qrr(1,kk,1) = 5.0E-4*EXP(-((z_k-5000.0)/3000.0)**2)
      CASE (4)
        IF (z_k < 5000.0) qcc(1,kk,1) = 1.2E-3*EXP(-((z_k-2000.0)/2000.0)**2)
        qrr(1,kk,1) = 1.2E-3*EXP(-((z_k-4500.0)/2600.0)**2)
        ww(1,kk,1) = 3.0*EXP(-((z_k-4200.0)/1600.0)**2)
      CASE (5)
        qrr(1,kk,1) = 7.0E-4*EXP(-((z_k-4200.0)/2300.0)**2)
      CASE DEFAULT
        IF (z_k < 3000.0) qcc(1,kk,1) = 1.0E-5*EXP(-((z_k-1500.0)/1000.0)**2)
      END SELECT
    END DO
  END SUBROUTINE build_column
END PROGRAM wsm3_oracle
