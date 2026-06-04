! =====================================================================
! v0.6.0 single-column Betts-Miller-Janjic (WRF cu_physics=2) oracle.
!
! Drives the UNMODIFIED pristine WRF module_cu_bmj.F through BMJINIT+BMJDRV on
! predeclared single-column soundings and dumps:
!   * input T, QV(mixing ratio), PMID, DZ8W, RHO, PI, TH
!   * output RTHCUTEN, RQVCUTEN, RAINCV, PRATEC, CUTOP, CUBOT, CLDEFI
!
! This is a WRF-module oracle, not a JAX self-compare.  The soundings are short
! offline columns so the build/run stays within the v060 resource rule.
! =====================================================================
PROGRAM bmj_oracle
  USE module_cu_bmj, ONLY : BMJDRV, BMJINIT
  IMPLICIT NONE

  REAL, PARAMETER :: G     = 9.81
  REAL, PARAMETER :: R_D   = 287.0
  REAL, PARAMETER :: CP_D  = 7.0*R_D/2.0
  REAL, PARAMETER :: P1000 = 100000.0
  REAL, PARAMETER :: ROVCP = R_D/CP_D
  REAL, PARAMETER :: XLV   = 2.5E6
  REAL, PARAMETER :: XLS   = 2.85E6
  REAL, PARAMETER :: TFRZ  = 273.16
  REAL, PARAMETER :: D608  = 0.608
  REAL, PARAMETER :: SVP1  = 0.6112
  REAL, PARAMETER :: SVP2  = 17.67
  REAL, PARAMETER :: SVP3  = 29.65
  REAL, PARAMETER :: SVPT0 = 273.15

  INTEGER, PARAMETER :: KX = 40
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: TH,T,QV,DZ8W,PMID,PINT,RHO,PI
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CCLDFRA,QCCONV,QICONV
  REAL, DIMENSION(ims:ime,jms:jme) :: CLDEFI,RAINCV,PRATEC,CONVCLD,CUBOT,CUTOP,XLAND
  INTEGER, DIMENSION(ims:ime,jms:jme) :: KPBL,LOWLYR
  LOGICAL, DIMENSION(ims:ime,jms:jme) :: CU_ACT_FLAG

  REAL :: DT
  INTEGER :: STEPCU, ITIMESTEP, case_id, k
  CHARACTER(LEN=32) :: arg
  CHARACTER(LEN=16) :: regime

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  DT = 54.0
  STEPCU = 1
  ITIMESTEP = 100

  RTHCUTEN=0.0; RQVCUTEN=0.0; RQCCUTEN=0.0; RQRCUTEN=0.0
  CCLDFRA=0.0; QCCONV=0.0; QICONV=0.0
  CLDEFI=0.0; RAINCV=0.0; PRATEC=0.0; CONVCLD=0.0
  CUBOT=0.0; CUTOP=0.0; XLAND=1.0; CU_ACT_FLAG=.TRUE.
  KPBL=5; LOWLYR=1

  CALL BMJINIT(RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN,CLDEFI,LOWLYR,CP_D,R_D, &
               .FALSE.,.TRUE.,                                           &
               ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme,          &
               its,ite,jts,jte,kts,kte)

  CALL build_sounding(case_id, T, QV, PMID, PINT, DZ8W, RHO, TH, PI, XLAND, KPBL)

  CALL BMJDRV( &
       ids,ide,jds,jde,kds,kde, &
       ims,ime,jms,jme,kms,kme, &
       its,ite,jts,jte,kts,kte, &
       DT,ITIMESTEP,STEPCU,CCLDFRA,CONVCLD, &
       RAINCV,PRATEC,CUTOP,CUBOT,KPBL, &
       TH,T,QV,QCCONV,QICONV,.FALSE., &
       PINT,PMID,PI,RHO,DZ8W, &
       CP_D,R_D,XLV,XLS,G,TFRZ,D608, &
       CLDEFI,LOWLYR,XLAND,CU_ACT_FLAG, &
       RTHCUTEN,RQVCUTEN )

  regime = classify_regime(RAINCV(1,1), RTHCUTEN, RQVCUTEN)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,I0)') 'KX=', KX
  WRITE(*,'(A,A)') 'REGIME=', TRIM(regime)
  WRITE(*,'(A,ES23.15)') 'DT=', DT
  WRITE(*,'(A,I0)') 'STEPCU=', STEPCU
  WRITE(*,'(A,ES23.15)') 'XLAND=', XLAND(1,1)
  WRITE(*,'(A,I0)') 'KPBL=', KPBL(1,1)
  WRITE(*,'(A,I0)') 'LOWLYR=', LOWLYR(1,1)
  WRITE(*,'(A,ES23.15)') 'CLDEFI_OUT=', CLDEFI(1,1)
  WRITE(*,'(A,ES23.15)') 'RAINCV=', RAINCV(1,1)
  WRITE(*,'(A,ES23.15)') 'PRATEC=', PRATEC(1,1)
  WRITE(*,'(A,ES23.15)') 'CUTOP=', CUTOP(1,1)
  WRITE(*,'(A,ES23.15)') 'CUBOT=', CUBOT(1,1)
  CALL dump_col('T', T, KX)
  CALL dump_col('QV', QV, KX)
  CALL dump_col('P', PMID, KX)
  CALL dump_col('DZ', DZ8W, KX)
  CALL dump_col('RHO', RHO, KX)
  CALL dump_col('TH', TH, KX)
  CALL dump_col('PI', PI, KX)
  CALL dump_col('RTHCUTEN', RTHCUTEN, KX)
  CALL dump_col('RQVCUTEN', RQVCUTEN, KX)
  DO k=1,KX+1
    WRITE(*,'(A,I0,A,ES23.15)') 'PINT[',k,']=', PINT(1,k,1)
  END DO

CONTAINS

  CHARACTER(LEN=16) FUNCTION classify_regime(rain, rth, rqv)
    REAL, INTENT(IN) :: rain
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: rth, rqv
    REAL :: max_tend
    max_tend = MAX(MAXVAL(ABS(rth)), MAXVAL(ABS(rqv)))
    IF (rain > 1.0E-8) THEN
      classify_regime = 'deep'
    ELSE IF (max_tend > 1.0E-12) THEN
      classify_regime = 'shallow'
    ELSE
      classify_regime = 'nonconvective'
    END IF
  END FUNCTION classify_regime

  SUBROUTINE dump_col(name, arr, nk)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: nk
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk=1,nk
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE build_sounding(cid, Tt, Qq, Pm, PiFace, Dz, Rr, Th, Exner, Xl, Kpbl)
    INTEGER, INTENT(IN) :: cid
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: Tt,Qq,Pm,PiFace,Dz,Rr,Th,Exner
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: Xl
    INTEGER, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: Kpbl
    REAL, DIMENSION(KX+1) :: zint, pint_local
    REAL :: psfc, tsfc, theta_sfc, ztop, zml, theta_lapse, rh_ml, rh_top
    REAL :: cap_amp, cap_bot, cap_top, xland_val
    REAL :: z, zmid, th_k, t_k, p_k, es, qs, qmix, rh_k, dz_k, tv_k
    INTEGER :: kk

    ztop = 20000.0
    DO kk=1,KX+1
      zint(kk) = ztop * (REAL(kk-1)/REAL(KX))**1.18
    END DO

    SELECT CASE (cid)
    CASE (1)   ! moist tropical, weak lapse: expected deep BMJ adjustment
      psfc=100600.0; tsfc=301.0; zml=1100.0; theta_lapse=3.8E-3
      rh_ml=0.90; rh_top=0.58; cap_amp=0.0; cap_bot=0.0; cap_top=0.0; xland_val=1.0
    CASE (2)   ! capped moist lower troposphere: expected shallow adjustment
      psfc=100000.0; tsfc=298.0; zml=850.0; theta_lapse=5.0E-3
      rh_ml=0.82; rh_top=0.28; cap_amp=4.0; cap_bot=1800.0; cap_top=4200.0; xland_val=1.0
    CASE (3)   ! very moist warm column over sea: strong deep adjustment
      psfc=100900.0; tsfc=303.0; zml=1300.0; theta_lapse=3.4E-3
      rh_ml=0.92; rh_top=0.62; cap_amp=0.0; cap_bot=0.0; cap_top=0.0; xland_val=2.0
    CASE (4)   ! dry stable land column: nonconvective
      psfc=100000.0; tsfc=287.0; zml=250.0; theta_lapse=8.5E-3
      rh_ml=0.35; rh_top=0.08; cap_amp=0.0; cap_bot=0.0; cap_top=0.0; xland_val=1.0
    CASE (5)   ! marginal warm/dry midlevel: shallow or suppressed
      psfc=100300.0; tsfc=299.0; zml=700.0; theta_lapse=6.2E-3
      rh_ml=0.72; rh_top=0.18; cap_amp=2.5; cap_bot=1500.0; cap_top=3500.0; xland_val=1.0
    CASE DEFAULT
      psfc=100500.0; tsfc=300.0; zml=1000.0; theta_lapse=4.0E-3
      rh_ml=0.88; rh_top=0.55; cap_amp=0.0; cap_bot=0.0; cap_top=0.0; xland_val=1.0
    END SELECT

    theta_sfc = tsfc * (P1000/psfc)**ROVCP
    Xl(1,1) = xland_val
    Kpbl(1,1) = 6

    DO kk=1,KX+1
      z = zint(kk)
      th_k = theta_sfc + MAX(0.0, z - zml) * theta_lapse
      IF (cap_amp > 0.0 .AND. z >= cap_bot .AND. z <= cap_top) THEN
        th_k = th_k + cap_amp * SIN(3.14159265*(z-cap_bot)/(cap_top-cap_bot))
      END IF
      t_k = th_k - 0.0060*z
      t_k = MAX(180.0, t_k)
      tv_k = t_k
      pint_local(kk) = psfc * EXP(-G*z/(R_D*MAX(tv_k,180.0)))
      PiFace(1,kk,1) = pint_local(kk)
    END DO

    DO kk=1,KX
      zmid = 0.5*(zint(kk)+zint(kk+1))
      dz_k = MAX(1.0, zint(kk+1)-zint(kk))
      p_k = 0.5*(pint_local(kk)+pint_local(kk+1))
      th_k = theta_sfc + MAX(0.0, zmid - zml) * theta_lapse
      IF (cap_amp > 0.0 .AND. zmid >= cap_bot .AND. zmid <= cap_top) THEN
        th_k = th_k + cap_amp * SIN(3.14159265*(zmid-cap_bot)/(cap_top-cap_bot))
      END IF
      t_k = th_k*(p_k/P1000)**ROVCP
      IF (zmid <= zml) THEN
        rh_k = rh_ml
      ELSE
        rh_k = rh_top + (rh_ml-rh_top)*EXP(-(zmid-zml)/3000.0)
      END IF
      es = (SVP1*1000.0)*EXP((SVP2*t_k - SVP2*SVPT0)/(t_k - SVP3))
      qs = 0.622*es/MAX(1.0, p_k-es)
      qmix = MAX(1.0E-8, rh_k*qs)
      tv_k = t_k*(1.0 + D608*qmix)

      Tt(1,kk,1) = t_k
      Qq(1,kk,1) = qmix
      Pm(1,kk,1) = p_k
      Dz(1,kk,1) = dz_k
      Rr(1,kk,1) = p_k/(R_D*tv_k)
      Exner(1,kk,1) = (p_k/P1000)**ROVCP
      Th(1,kk,1) = t_k/Exner(1,kk,1)
    END DO

    DO kk=KX+1,kme
      Tt(1,kk,1)=Tt(1,KX,1)
      Qq(1,kk,1)=Qq(1,KX,1)
      Pm(1,kk,1)=Pm(1,KX,1)
      Dz(1,kk,1)=Dz(1,KX,1)
      Rr(1,kk,1)=Rr(1,KX,1)
      Exner(1,kk,1)=Exner(1,KX,1)
      Th(1,kk,1)=Th(1,KX,1)
    END DO
  END SUBROUTINE build_sounding

END PROGRAM bmj_oracle
