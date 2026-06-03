! Standalone Grell-Freitas (WRF cu_physics=3) single-column oracle driver.
!
! The scheme sources are copied verbatim from /home/enric/src/wrf_pristine/WRF
! by build_and_run.sh. This driver supplies prescribed columns to GFDRV and
! dumps inputs plus WRF tendencies. It is not a full wrf.exe integration.
PROGRAM gf_oracle
  USE module_cu_gf_wrfdrv, ONLY : GFDRV
  IMPLICIT NONE

  REAL, PARAMETER :: G = 9.81
  REAL, PARAMETER :: R_D = 287.0
  REAL, PARAMETER :: R_V = 461.6
  REAL, PARAMETER :: CP = 1004.0
  REAL, PARAMETER :: XLV = 2.5E6
  REAL, PARAMETER :: P1000 = 1.0E5
  REAL, PARAMETER :: ROVCP = R_D / CP

  INTEGER, PARAMETER :: KX = 45
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,W,T,QV,P,PI,DZ8W,P8W,RHO
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RQVFTEN,RTHFTEN,RTHRATEN,RQVBLTEN,RTHBLTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RUCUTEN,RVCUTEN,GDC,GDC2
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: PATTERN_SPP_CONV,FIELD_CONV
  REAL, DIMENSION(ims:ime,jms:jme) :: RAINCV,PRATEC,HTOP,HBOT,HT,HFX,QFX,XLAND,XMB_SHALLOW
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL,K22_SHALLOW,KBCON_SHALLOW,KTOP_SHALLOW,KTOP_DEEP

  REAL :: DT, DX
  INTEGER :: case_id, k, spp_conv, ichoice, ishallow_g3
  LOGICAL :: periodic_x, periodic_y
  CHARACTER(LEN=32) :: arg
  REAL, DIMENSION(KX) :: zlev
  CHARACTER(LEN=48) :: regime

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_sounding(case_id, T, QV, P, DZ8W, RHO, U, V, W, zlev, DT, DX, HFX(1,1), QFX(1,1), KPBL(1,1), XLAND(1,1), regime)

  DO k=kts,kte
    PI(1,k,1) = (P(1,k,1) / P1000) ** ROVCP
    P8W(1,k,1) = P(1,k,1)
  END DO

  RTHCUTEN=0.0; RQVCUTEN=0.0; RQCCUTEN=0.0; RQICUTEN=0.0
  RUCUTEN=0.0; RVCUTEN=0.0; RQVFTEN=0.0; RTHFTEN=0.0; RTHRATEN=0.0
  RQVBLTEN=0.0; RTHBLTEN=0.0; GDC=0.0; GDC2=0.0
  PATTERN_SPP_CONV=0.0; FIELD_CONV=0.0
  RAINCV=0.0; PRATEC=0.0; HTOP=0.0; HBOT=REAL(KTE); HT=0.0
  XMB_SHALLOW=0.0; K22_SHALLOW=0; KBCON_SHALLOW=0; KTOP_SHALLOW=0; KTOP_DEEP=0

  ! Weak PBL forcing side input for the shallow branch. These are WRF driver
  ! inputs, not outputs from this oracle.
  DO k=1,MIN(KPBL(1,1), KX)
    IF (case_id == 2) THEN
      RTHBLTEN(1,k,1) = 1.2E-4
      RQVBLTEN(1,k,1) = 1.0E-8
    ELSE IF (case_id == 1 .OR. case_id == 4 .OR. case_id == 5) THEN
      RTHBLTEN(1,k,1) = 7.5E-5
      RQVBLTEN(1,k,1) = 8.0E-9
    END IF
  END DO

  spp_conv = 0
  ichoice = 0
  ishallow_g3 = 1
  periodic_x = .TRUE.
  periodic_y = .TRUE.

  CALL GFDRV(spp_conv, PATTERN_SPP_CONV, FIELD_CONV, &
       DT=DT, DX=DX, RHO=RHO, RAINCV=RAINCV, PRATEC=PRATEC, &
       U=U, V=V, T=T, W=W, Q=QV, P=P, PI=PI, DZ8W=DZ8W, P8W=P8W, &
       HTOP=HTOP, HBOT=HBOT, KTOP_DEEP=KTOP_DEEP, HT=HT, HFX=HFX, QFX=QFX, XLAND=XLAND, &
       GDC=GDC, GDC2=GDC2, KPBL=KPBL, K22_SHALLOW=K22_SHALLOW, &
       KBCON_SHALLOW=KBCON_SHALLOW, KTOP_SHALLOW=KTOP_SHALLOW, XMB_SHALLOW=XMB_SHALLOW, &
       ICHOICE=ichoice, ISHALLOW_G3=ishallow_g3, &
       IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, &
       IMS=ims, IME=ime, JMS=jms, JME=jme, KMS=kms, KME=kme, &
       ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte, &
       PERIODIC_X=periodic_x, PERIODIC_Y=periodic_y, &
       RQVCUTEN=RQVCUTEN, RQCCUTEN=RQCCUTEN, RQICUTEN=RQICUTEN, &
       RQVFTEN=RQVFTEN, RTHFTEN=RTHFTEN, RTHCUTEN=RTHCUTEN, RTHRATEN=RTHRATEN, &
       RQVBLTEN=RQVBLTEN, RTHBLTEN=RTHBLTEN, DUDT_PHY=RUCUTEN, DVDT_PHY=RVCUTEN)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,A)') 'REGIME=', TRIM(regime)
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  WRITE(*,'(A,ES23.15)') 'DX=', DX
  WRITE(*,'(A,ES23.15)') 'HFX=', HFX(1,1)
  WRITE(*,'(A,ES23.15)') 'QFX=', QFX(1,1)
  WRITE(*,'(A,I0)') 'KPBL=', KPBL(1,1)
  WRITE(*,'(A,ES23.15)') 'XLAND=', XLAND(1,1)
  CALL dump_col('T', T)
  CALL dump_col('QV', QV)
  CALL dump_col('P', P)
  CALL dump_col('PI', PI)
  CALL dump_col('DZ', DZ8W)
  CALL dump_col('RHO', RHO)
  CALL dump_col('U', U)
  CALL dump_col('V', V)
  CALL dump_col('W', W)
  CALL dump_col('RTHBLTEN', RTHBLTEN)
  CALL dump_col('RQVBLTEN', RQVBLTEN)
  CALL dump_col('RTHCUTEN', RTHCUTEN)
  CALL dump_col('RQVCUTEN', RQVCUTEN)
  CALL dump_col('RQCCUTEN', RQCCUTEN)
  CALL dump_col('RQICUTEN', RQICUTEN)
  CALL dump_col('RUCUTEN', RUCUTEN)
  CALL dump_col('RVCUTEN', RVCUTEN)
  WRITE(*,'(A,ES23.15)') 'RAINCV=', RAINCV(1,1)
  WRITE(*,'(A,ES23.15)') 'PRATEC=', PRATEC(1,1)
  WRITE(*,'(A,ES23.15)') 'HTOP=', HTOP(1,1)
  WRITE(*,'(A,ES23.15)') 'HBOT=', HBOT(1,1)
  WRITE(*,'(A,I0)') 'KTOP_DEEP=', KTOP_DEEP(1,1)
  WRITE(*,'(A,ES23.15)') 'XMB_SHALLOW=', XMB_SHALLOW(1,1)
  WRITE(*,'(A,I0)') 'K22_SHALLOW=', K22_SHALLOW(1,1)
  WRITE(*,'(A,I0)') 'KBCON_SHALLOW=', KBCON_SHALLOW(1,1)
  WRITE(*,'(A,I0)') 'KTOP_SHALLOW=', KTOP_SHALLOW(1,1)

CONTAINS

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk=kts,kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  REAL FUNCTION qsat_liq(temp, pres)
    REAL, INTENT(IN) :: temp, pres
    REAL :: es
    es = 611.2 * EXP(17.67 * (temp - 273.15) / (temp - 29.65))
    es = MIN(es, 0.95 * pres)
    qsat_liq = 0.622 * es / (pres - es)
  END FUNCTION qsat_liq

  SUBROUTINE build_sounding(cid, Tt, Qq, Pp, Dz, Rr, Uu, Vv, Ww, zz, dt, dx, hfx, qfx, kpbl, xland, regime)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Tt,Qq,Pp,Dz,Rr,Uu,Vv,Ww
    REAL, DIMENSION(KX), INTENT(OUT) :: zz
    REAL, INTENT(OUT) :: dt, dx, hfx, qfx, xland
    INTEGER, INTENT(OUT) :: kpbl
    CHARACTER(LEN=48), INTENT(OUT) :: regime
    REAL :: psfc, tsfc, zml, lapse_theta, rh_ml, rh_free, ushr, wpeak, cap_amp
    REAL :: theta_sfc, ztop, z_k, th_k, t_k, q_k, p_k, tv_k, rh_k, qs
    INTEGER :: kk

    ztop = 19000.0
    DO kk=1,KX
      zz(kk) = ztop * ((REAL(kk)-0.5)/REAL(KX))**1.18
    END DO

    SELECT CASE (cid)
    CASE (1)
      regime='deep_convective'
      psfc=100800.0; tsfc=302.0; zml=1200.0; lapse_theta=3.5E-3
      rh_ml=0.92; rh_free=0.62; ushr=8.0; wpeak=1.1; cap_amp=0.0
      dx=9000.0; dt=54.0; hfx=420.0; qfx=3.0E-4; kpbl=6; xland=1.0
    CASE (2)
      regime='shallow_convective'
      psfc=100500.0; tsfc=301.0; zml=900.0; lapse_theta=4.7E-3
      rh_ml=0.89; rh_free=0.38; ushr=3.0; wpeak=0.42; cap_amp=3.2
      dx=9000.0; dt=54.0; hfx=340.0; qfx=2.4E-4; kpbl=5; xland=1.0
    CASE (3)
      regime='stable_nontriggering'
      psfc=100000.0; tsfc=286.0; zml=250.0; lapse_theta=1.05E-2
      rh_ml=0.36; rh_free=0.12; ushr=0.0; wpeak=-0.08; cap_amp=0.0
      dx=9000.0; dt=54.0; hfx=0.0; qfx=0.0; kpbl=2; xland=1.0
    CASE (4)
      regime='scale_aware_coarse_15km'
      psfc=100800.0; tsfc=302.0; zml=1200.0; lapse_theta=3.5E-3
      rh_ml=0.92; rh_free=0.62; ushr=8.0; wpeak=1.1; cap_amp=0.0
      dx=15000.0; dt=90.0; hfx=420.0; qfx=3.0E-4; kpbl=6; xland=1.0
    CASE (5)
      regime='scale_aware_fine_3km'
      psfc=100800.0; tsfc=302.0; zml=1200.0; lapse_theta=3.5E-3
      rh_ml=0.92; rh_free=0.62; ushr=8.0; wpeak=1.1; cap_amp=0.0
      dx=3000.0; dt=18.0; hfx=420.0; qfx=3.0E-4; kpbl=6; xland=1.0
    CASE DEFAULT
      regime='default_deep'
      psfc=100800.0; tsfc=302.0; zml=1200.0; lapse_theta=3.5E-3
      rh_ml=0.92; rh_free=0.62; ushr=8.0; wpeak=1.1; cap_amp=0.0
      dx=9000.0; dt=54.0; hfx=420.0; qfx=3.0E-4; kpbl=6; xland=1.0
    END SELECT

    theta_sfc = tsfc * (P1000 / psfc) ** ROVCP
    p_k = psfc
    DO kk=1,KX
      z_k = zz(kk)
      IF (z_k <= zml) THEN
        th_k = theta_sfc
        rh_k = rh_ml
      ELSE
        th_k = theta_sfc + lapse_theta * (z_k - zml)
        IF (cap_amp > 0.0 .AND. z_k > 1100.0 .AND. z_k < 2600.0) THEN
          th_k = th_k + cap_amp * EXP(-((z_k - 1700.0)/550.0)**2)
        END IF
        rh_k = rh_free + (rh_ml-rh_free) * EXP(-(z_k-zml)/2300.0)
      END IF

      IF (kk == 1) THEN
        t_k = th_k * (psfc / P1000) ** ROVCP
        tv_k = t_k
        p_k = psfc * EXP(-G * zz(1) / (R_D * tv_k))
      ELSE
        t_k = th_k * (p_k / P1000) ** ROVCP
        tv_k = t_k * (1.0 + 0.608 * Qq(1,kk-1,1))
        p_k = p_k * EXP(-G * (zz(kk)-zz(kk-1)) / (R_D * tv_k))
      END IF

      t_k = th_k * (p_k / P1000) ** ROVCP
      qs = qsat_liq(t_k, p_k)
      q_k = MAX(1.0E-7, rh_k * qs)
      tv_k = t_k * (1.0 + 0.608 * q_k)

      Tt(1,kk,1) = t_k
      Qq(1,kk,1) = q_k
      Pp(1,kk,1) = p_k
      Rr(1,kk,1) = p_k / (R_D * tv_k)
      IF (kk == 1) THEN
        Dz(1,kk,1) = 2.0 * zz(1)
      ELSE
        Dz(1,kk,1) = zz(kk) - zz(kk-1)
      END IF
      Uu(1,kk,1) = ushr * MIN(1.0, z_k / 10000.0)
      Vv(1,kk,1) = 0.0
      Ww(1,kk,1) = wpeak * EXP(-((z_k - 1700.0)/1300.0)**2)
    END DO
  END SUBROUTINE build_sounding

END PROGRAM gf_oracle
