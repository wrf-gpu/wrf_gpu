! =====================================================================
! GSFC (Chou-Suarez) shortwave (ra_sw_physics=2) single-column oracle.
!
! Drives the UNMODIFIED WRF phys/module_ra_gsfcsw.F:GSFCSWRAD on
! prescribed single-column soundings and dumps inputs plus the WRF SW
! potential-temperature tendency RTHRATEN, the surface net SW flux GSW
! and the TOA upward SW RSWTOA for JAX savepoint parity. This is a real
! WRF-module oracle, not a JAX self-compare; it is not a full coupled
! wrf.exe run.
!
! GSFCSWRAD flips K internally (NK=kme-1-K+kms inside the j-loop), so
! K=kts is the model BOTTOM in the public 3-D arrays. We build/dump
! everything in natural bottom-to-top model order (k=1 lowest layer) and
! let GSFCSWRAD do its own internal flip, exactly as the operational
! radiation driver does.
!
! The default operational moisture path is exercised:
!   warm_rain=.FALSE., F_QI=.TRUE.  (Thompson-class microphysics: ice is a
!   prognostic species, so cwc(:,:,1)=QI and cwc(:,:,2)=QC), F_QNDROP absent.
! Aerosol feedback (WRF_CHEM) is OFF (the optional aerosol args are absent),
! matching the operational GPU build.
! =====================================================================
PROGRAM gsfcsw_oracle
  USE module_ra_gsfcsw, ONLY : GSFCSWRAD, gsfc_swinit
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: P1000  = 1.0E5
  REAL, PARAMETER :: DEGRAD = 3.1415926/180.

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: T3D,QV3D,QC3D,QR3D,QI3D,QS3D,QG3D
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: QNDROP3D
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: P3D,P8W3D,PI3D,DZ8W,RHO_PHY
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: CLDFRA3D,RTHRATEN
  REAL, DIMENSION(ims:ime,jms:jme) :: ALBEDO,GSW,RSWTOA,COSZEN,OBSCUR
  CHARACTER(LEN=32) :: arg, regime
  INTEGER :: case_id, julday
  REAL :: solcon, center_lat
  LOGICAL :: warm_rain, f_qv,f_qc,f_qr,f_qi,f_qs,f_qg

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  julday    = 172            ! near solstice
  solcon    = 1370.0         ! solar constant W/m^2 (date-scaled in driver)
  warm_rain = .FALSE.
  obscur    = 0.0
  center_lat= 28.0           ! mid-latitude -> iprof selection (gsfc_swinit)
  ! flags mark which moist species are present (Thompson-class default path)
  f_qv=.TRUE.; f_qc=.TRUE.; f_qr=.TRUE.; f_qi=.TRUE.; f_qs=.TRUE.; f_qg=.TRUE.

  CALL gsfc_swinit(center_lat, .TRUE.)

  CALL build_case(case_id, regime, T3D, QV3D, QC3D, QR3D, QI3D, QS3D, QG3D, &
                  QNDROP3D, P3D, P8W3D, PI3D, DZ8W, RHO_PHY, CLDFRA3D, &
                  ALBEDO, COSZEN, solcon)

  RTHRATEN = 0.0
  GSW      = 0.0
  RSWTOA   = 0.0

  CALL GSFCSWRAD(rthraten=RTHRATEN, gsw=GSW, &
       dz8w=DZ8W, rho_phy=RHO_PHY, alb=ALBEDO, &
       t3d=T3D, qv3d=QV3D, qc3d=QC3D, qr3d=QR3D, &
       qi3d=QI3D, qs3d=QS3D, qg3d=QG3D, qndrop3d=QNDROP3D, &
       p3d=P3D, p8w3d=P8W3D, pi3d=PI3D, cldfra3d=CLDFRA3D, rswtoa=RSWTOA, &
       cp=CP, g=G, julday=JULDAY, solcon=solcon, &
       warm_rain=warm_rain, &
       f_qv=f_qv, f_qc=f_qc, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs, f_qg=f_qg, &
       coszen=COSZEN, obscur=OBSCUR, &
       ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
       ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
       its=its,ite=ite,jts=jte,jte=jte,kts=kts,kte=kte)

  CALL dump_case(case_id, regime, T3D, QV3D, QC3D, QR3D, QI3D, QS3D, QG3D, &
                 P3D, P8W3D, PI3D, DZ8W, RHO_PHY, CLDFRA3D, ALBEDO, COSZEN, &
                 solcon, julday, center_lat, RTHRATEN, GSW, RSWTOA)

CONTAINS

  SUBROUTINE build_case(cid, name, Tt, Qv, Qc, Qr, Qi, Qs, Qg, Qnd, &
                        Pp, P8w, Exner, Dz, Rho, Cf, AlbA, CosA, scon)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: &
        Tt,Qv,Qc,Qr,Qi,Qs,Qg,Qnd,Pp,P8w,Exner,Dz,Rho,Cf
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: AlbA,CosA
    REAL, INTENT(IN) :: scon
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, z, tv, pim
    REAL :: alb0, cos0, qc0, qi0, qr0, qs0, qg0, cldtop, cldbot, fcld0
    INTEGER :: k

    ztop = 16000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.15
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    qc0=0.0; qi0=0.0; qr0=0.0; qs0=0.0; qg0=0.0
    cldbot=0.0; cldtop=0.0; fcld0=0.0

    SELECT CASE (cid)
    CASE (1)
      name='clearsky_highsun_land'
      psfc0=100000.0; theta0=300.0; zml=1200.0; lapse_ml=0.0004; lapse_ft=0.0045
      q0=0.0120; qscale=2500.0; alb0=0.20; cos0=0.95
    CASE (2)
      name='clearsky_lowsun_land'
      psfc0=100000.0; theta0=296.0; zml=900.0; lapse_ml=0.0006; lapse_ft=0.0045
      q0=0.0090; qscale=2300.0; alb0=0.22; cos0=0.18
    CASE (3)
      name='nighttime_zerosun'
      psfc0=100500.0; theta0=290.0; zml=200.0; lapse_ml=0.0080; lapse_ft=0.0050
      q0=0.0070; qscale=2000.0; alb0=0.20; cos0=0.0
      qc0=2.0E-4; cldbot=600.0; cldtop=2000.0; fcld0=0.9
    CASE (4)
      name='thick_warm_cloud_highsun'
      psfc0=100800.0; theta0=298.0; zml=1100.0; lapse_ml=0.0004; lapse_ft=0.0042
      q0=0.0150; qscale=2700.0; alb0=0.18; cos0=0.85
      qc0=8.0E-4; qr0=2.0E-4; cldbot=500.0; cldtop=3000.0; fcld0=0.95
    CASE (5)
      name='ice_cloud_midsun_marine'
      psfc0=101200.0; theta0=295.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0038
      q0=0.0110; qscale=2600.0; alb0=0.06; cos0=0.55
      qi0=3.0E-4; qs0=1.5E-4; cldbot=5000.0; cldtop=9000.0; fcld0=0.8
    CASE (6)
      name='snow_graupel_clouds_highsun'
      psfc0=99500.0; theta0=297.0; zml=1000.0; lapse_ml=0.0006; lapse_ft=0.0040
      q0=0.0130; qscale=2400.0; alb0=0.30; cos0=0.78
      qc0=4.0E-4; qs0=3.0E-4; qg0=2.0E-4; qr0=1.0E-4; cldbot=800.0; cldtop=6000.0; fcld0=0.85
    CASE DEFAULT
      name='terminator_lowsun_humid'
      psfc0=100200.0; theta0=294.0; zml=800.0; lapse_ml=0.0008; lapse_ft=0.0044
      q0=0.0160; qscale=2800.0; alb0=0.15; cos0=0.05
    END SELECT

    pint(1) = psfc0
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

    Tt=0.0; Qv=0.0; Qc=0.0; Qr=0.0; Qi=0.0; Qs=0.0; Qg=0.0; Qnd=0.0
    Pp=0.0; P8w=0.0; Exner=0.0; Dz=0.0; Rho=0.0; Cf=0.0
    DO k = 1, KX
      Tt(1,k,1)  = temp(k)
      Qv(1,k,1)  = qprof(k)
      Pp(1,k,1)  = pfull(k)
      P8w(1,k,1) = pint(k)
      Exner(1,k,1) = (pfull(k)/P1000)**ROVCP
      Dz(1,k,1)  = zint(k+1)-zint(k)
      Rho(1,k,1) = pfull(k)/(R_D*temp(k)*(1.0+EP1*qprof(k)))
      ! cloud / hydrometeor layer placement (model z order, k=1 bottom)
      IF (zmid(k) >= cldbot .AND. zmid(k) <= cldtop) THEN
        Qc(1,k,1) = qc0
        Qr(1,k,1) = qr0
        Qi(1,k,1) = qi0
        Qs(1,k,1) = qs0
        Qg(1,k,1) = qg0
        Cf(1,k,1) = fcld0
      END IF
    END DO
    ! Top interface pressure (k=KX+1 = model top), needed by P8W3D reversal.
    P8w(1,KX+1,1) = pint(KX+1)

    AlbA(1,1)=alb0; CosA(1,1)=cos0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(label, arr)
    CHARACTER(LEN=*), INTENT(IN) :: label
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') label,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col8w(label, arr)
    CHARACTER(LEN=*), INTENT(IN) :: label
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte+1
      WRITE(*,'(A,A,I0,A,ES23.15)') label,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col8w

  SUBROUTINE dump_case(cid, name, Tt, Qv, Qc, Qr, Qi, Qs, Qg, Pp, P8w, Exner, &
                       Dz, Rho, Cf, AlbA, CosA, scon, jd, clat, Rth, GswA, RswtoaA)
    INTEGER, INTENT(IN) :: cid, jd
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        Tt,Qv,Qc,Qr,Qi,Qs,Qg,Pp,P8w,Exner,Dz,Rho,Cf,Rth
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: AlbA,CosA,GswA,RswtoaA
    REAL, INTENT(IN) :: scon, clat
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,I0)') 'JULDAY=', jd
    WRITE(*,'(A,ES23.15)') 'SOLCON=', scon
    WRITE(*,'(A,ES23.15)') 'CENTER_LAT=', clat
    WRITE(*,'(A,ES23.15)') 'ALBEDO=', AlbA(1,1)
    WRITE(*,'(A,ES23.15)') 'COSZEN=', CosA(1,1)
    WRITE(*,'(A,ES23.15)') 'GSW=', GswA(1,1)
    WRITE(*,'(A,ES23.15)') 'RSWTOA=', RswtoaA(1,1)
    CALL dump_col('T', Tt)
    CALL dump_col('QV', Qv)
    CALL dump_col('QC', Qc)
    CALL dump_col('QR', Qr)
    CALL dump_col('QI', Qi)
    CALL dump_col('QS', Qs)
    CALL dump_col('QG', Qg)
    CALL dump_col('P', Pp)
    CALL dump_col8w('P8W', P8w)
    CALL dump_col('PI', Exner)
    CALL dump_col('DZ', Dz)
    CALL dump_col('RHO', Rho)
    CALL dump_col('CLDFRA', Cf)
    CALL dump_col('RTHRATEN', Rth)
  END SUBROUTINE dump_case
END PROGRAM gsfcsw_oracle
