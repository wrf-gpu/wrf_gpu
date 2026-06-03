! =====================================================================
! v0.6.0 single-column Dudhia shortwave (ra_sw_physics=1) oracle driver.
!
! Drives the UNMODIFIED WRF phys/module_ra_sw.F:SWRAD (which calls the
! internal SWPARA Stephens-1984 broadband column kernel) on prescribed
! single-column soundings. The dump captures inputs plus the WRF SW
! potential-temperature tendency RTHRATEN and the surface net SW flux
! GSW for JAX savepoint parity. This is a real WRF-module oracle, not a
! JAX self-compare; it is not a full coupled wrf.exe run.
!
! WRF flips K inside SWRAD (NK = kme-1-K+kms), so K=kts is the model
! BOTTOM in the public arrays. We build/dump everything in the natural
! bottom-to-top model order (k=1 lowest layer) and let SWRAD do its own
! internal flip, exactly as the operational radiation driver does.
! =====================================================================
PROGRAM dudhia_oracle
  USE module_ra_sw, ONLY : SWRAD, swinit
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
  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: P3D,PI3D,DZ8W,RHO_PHY,RTHRATEN
  REAL, DIMENSION(ims:ime,jms:jme) :: XLAT,XLONG,ALBEDO,GSW,COSZEN,OBSCUR
  CHARACTER(LEN=256) :: errmsg
  CHARACTER(LEN=32) :: arg, regime
  INTEGER :: case_id, julday, icloud, ghg_input
  REAL :: gmt, xtime, declin, solcon, radfrq, julian_d, swrad_scat
  LOGICAL :: warm_rain, f_qv,f_qc,f_qr,f_qi,f_qs,f_qg

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  ! WRF default SW namelist constants used by the radiation driver.
  swrad_scat = 1.0          ! namelist default; cssca = swrad_scat*1.e-5
  CALL swinit(swrad_scat, .TRUE., &
              ids,ide,jds,jde,kds,kde, &
              ims,ime,jms,jme,kms,kme, &
              its,ite,jts,jte,kts,kte)

  gmt      = 12.0
  xtime    = 0.0
  julday   = 172            ! near solstice
  julian_d = 171.5
  radfrq   = 30.0
  ghg_input= 0
  warm_rain= .FALSE.
  obscur   = 0.0
  ! flags mark which moist species are present in the 4-D moist array.
  f_qv=.TRUE.; f_qc=.TRUE.; f_qr=.TRUE.; f_qi=.TRUE.; f_qs=.TRUE.; f_qg=.TRUE.

  CALL build_case(case_id, regime, T3D, QV3D, QC3D, QR3D, QI3D, QS3D, QG3D, &
                  P3D, PI3D, DZ8W, RHO_PHY, XLAT, XLONG, ALBEDO, COSZEN, &
                  declin, solcon, icloud)

  RTHRATEN = 0.0
  GSW      = 0.0

  CALL SWRAD(dt=radfrq*60.0, RTHRATEN=RTHRATEN, GSW=GSW, &
             XLAT=XLAT, XLONG=XLONG, ALBEDO=ALBEDO, &
             rho_phy=RHO_PHY, T3D=T3D, QV3D=QV3D, QC3D=QC3D, QR3D=QR3D, &
             QI3D=QI3D, QS3D=QS3D, QG3D=QG3D, P3D=P3D, pi3D=PI3D, dz8w=DZ8W, &
             GMT=GMT, R=R_D, CP=CP, G=G, JULDAY=JULDAY, GHG_INPUT=ghg_input, &
             XTIME=XTIME, DECLIN=declin, SOLCON=solcon, &
             F_QV=f_qv, F_QC=f_qc, F_QR=f_qr, F_QI=f_qi, F_QS=f_qs, F_QG=f_qg, &
             pm2_5_dry=DZ8W*0.0, pm2_5_water=DZ8W*0.0, pm2_5_dry_ec=DZ8W*0.0, &
             RADFRQ=radfrq, ICLOUD=icloud, DEGRAD=DEGRAD, warm_rain=warm_rain, &
             ids=ids,ide=ide,jds=jds,jde=jde,kds=kds,kde=kde, &
             ims=ims,ime=ime,jms=jms,jme=jme,kms=kms,kme=kme, &
             its=its,ite=ite,jts=jts,jte=jte,kts=kts,kte=kte, &
             coszen=COSZEN, julian=julian_d, obscur=OBSCUR)

  CALL dump_case(case_id, regime, T3D, QV3D, QC3D, QR3D, QI3D, QS3D, QG3D, &
                 P3D, PI3D, DZ8W, RHO_PHY, XLAT, XLONG, ALBEDO, COSZEN, &
                 declin, solcon, icloud, radfrq, RTHRATEN, GSW)

  IF (errmsg /= "") CONTINUE  ! silence unused-warning path

CONTAINS

  SUBROUTINE build_case(cid, name, Tt, Qv, Qc, Qr, Qi, Qs, Qg, Pp, Exner, &
                        Dz, Rho, LatA, LonA, AlbA, CosA, dec, scon, icld)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=32), INTENT(OUT) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(OUT) :: &
        Tt,Qv,Qc,Qr,Qi,Qs,Qg,Pp,Exner,Dz,Rho
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(OUT) :: LatA,LonA,AlbA,CosA
    REAL, INTENT(OUT) :: dec, scon
    INTEGER, INTENT(OUT) :: icld
    REAL, DIMENSION(KX+1) :: zint, pint
    REAL, DIMENSION(KX) :: zmid, theta, qprof, temp, pfull
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, z, tv, pim
    REAL :: alb0, cos0, qc0, qi0, qr0, qs0, qg0, cldtop, cldbot
    INTEGER :: k

    ztop = 16000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.15
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    ! defaults
    dec = 0.4090      ! ~solstice declination (rad)
    scon = 1370.0     ! solar constant W/m^2
    icld = 1          ! icloud on
    qc0=0.0; qi0=0.0; qr0=0.0; qs0=0.0; qg0=0.0; cldbot=0.0; cldtop=0.0

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
      qc0=2.0E-4; cldbot=600.0; cldtop=2000.0
    CASE (4)
      name='thick_warm_cloud_highsun'
      psfc0=100800.0; theta0=298.0; zml=1100.0; lapse_ml=0.0004; lapse_ft=0.0042
      q0=0.0150; qscale=2700.0; alb0=0.18; cos0=0.85
      qc0=8.0E-4; qr0=2.0E-4; cldbot=500.0; cldtop=3000.0
    CASE (5)
      name='ice_cloud_midsun_marine'
      psfc0=101200.0; theta0=295.0; zml=700.0; lapse_ml=0.0010; lapse_ft=0.0038
      q0=0.0110; qscale=2600.0; alb0=0.06; cos0=0.55
      qi0=3.0E-4; qs0=1.5E-4; cldbot=5000.0; cldtop=9000.0
    CASE (6)
      name='snow_graupel_clouds_highsun'
      psfc0=99500.0; theta0=297.0; zml=1000.0; lapse_ml=0.0006; lapse_ft=0.0040
      q0=0.0130; qscale=2400.0; alb0=0.30; cos0=0.78
      qc0=4.0E-4; qs0=3.0E-4; qg0=2.0E-4; qr0=1.0E-4; cldbot=800.0; cldtop=6000.0
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

    Tt=0.0; Qv=0.0; Qc=0.0; Qr=0.0; Qi=0.0; Qs=0.0; Qg=0.0
    Pp=0.0; Exner=0.0; Dz=0.0; Rho=0.0
    DO k = 1, KX
      Tt(1,k,1)  = temp(k)
      Qv(1,k,1)  = qprof(k)
      Pp(1,k,1)  = pfull(k)
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
      END IF
    END DO

    LatA(1,1)=28.0*DEGRAD; LonA(1,1)=-16.0*DEGRAD
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

  SUBROUTINE dump_case(cid, name, Tt, Qv, Qc, Qr, Qi, Qs, Qg, Pp, Exner, &
                       Dz, Rho, LatA, LonA, AlbA, CosA, dec, scon, icld, &
                       radf, Rth, GswA)
    INTEGER, INTENT(IN) :: cid, icld
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: &
        Tt,Qv,Qc,Qr,Qi,Qs,Qg,Pp,Exner,Dz,Rho,Rth
    REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: LatA,LonA,AlbA,CosA,GswA
    REAL, INTENT(IN) :: dec, scon, radf
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'ICLOUD=', icld
    WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
    WRITE(*,'(A,ES23.15)') 'DT=', radf*60.0
    WRITE(*,'(A,ES23.15)') 'RADFRQ=', radf
    WRITE(*,'(A,ES23.15)') 'DECLIN=', dec
    WRITE(*,'(A,ES23.15)') 'SOLCON=', scon
    WRITE(*,'(A,ES23.15)') 'GMT=', 12.0
    WRITE(*,'(A,ES23.15)') 'XLAT=', LatA(1,1)
    WRITE(*,'(A,ES23.15)') 'XLONG=', LonA(1,1)
    WRITE(*,'(A,ES23.15)') 'ALBEDO=', AlbA(1,1)
    WRITE(*,'(A,ES23.15)') 'COSZEN=', CosA(1,1)
    WRITE(*,'(A,ES23.15)') 'GSW=', GswA(1,1)
    CALL dump_col('T', Tt)
    CALL dump_col('QV', Qv)
    CALL dump_col('QC', Qc)
    CALL dump_col('QR', Qr)
    CALL dump_col('QI', Qi)
    CALL dump_col('QS', Qs)
    CALL dump_col('QG', Qg)
    CALL dump_col('P', Pp)
    CALL dump_col('PI', Exner)
    CALL dump_col('DZ', Dz)
    CALL dump_col('RHO', Rho)
    CALL dump_col('RTHRATEN', Rth)
  END SUBROUTINE dump_case
END PROGRAM dudhia_oracle
