PROGRAM slab_oracle
  ! Single-column oracle for the WRF 5-layer thermal-diffusion slab LSM
  ! (UNMODIFIED phys/module_sf_slab.F, SUBROUTINE SLAB). Emits gold savepoints
  ! (inputs + outputs) as flat key=value lines for slab_dump_to_json.py.
  USE module_sf_slab, ONLY : slab
  IMPLICIT NONE

  ! Constants WRF passes into SLAB (module_surface_driver.F:2659-2665 +
  ! share/module_model_constants.F).
  REAL, PARAMETER :: CP = 1004.0
  REAL, PARAMETER :: R  = 287.0
  REAL, PARAMETER :: ROVCP = R / CP            ! rcp
  REAL, PARAMETER :: XLV = 2.5E6
  REAL, PARAMETER :: SVP1 = 0.6112
  REAL, PARAMETER :: SVP2 = 17.67
  REAL, PARAMETER :: SVP3 = 29.65
  REAL, PARAMETER :: SVPT0 = 273.15
  REAL, PARAMETER :: EP2 = 0.622
  REAL, PARAMETER :: KARMAN = 0.4
  REAL, PARAMETER :: EOMEG = 7.2921E-5
  REAL, PARAMETER :: STBOLT = 5.67051E-8
  REAL, PARAMETER :: P1000MB = 100000.0

  INTEGER, PARAMETER :: N = 8
  INTEGER, PARAMETER :: NSOIL = 5
  INTEGER, PARAMETER :: ids=1, ide=N+1, jds=1, jde=2, kds=1, kde=2
  INTEGER, PARAMETER :: ims=1, ime=N,   jms=1, jme=1, kms=1, kme=1
  INTEGER, PARAMETER :: its=1, ite=N,   jts=1, jte=1, kts=1, kte=1

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: T3D, QV3D, P3D
  REAL, DIMENSION(ims:ime,jms:jme) :: FLHC, FLQC, PSFC, XLAND, TMN, HFX, QFX, LH
  REAL, DIMENSION(ims:ime,jms:jme) :: TSK, QSFC, CHKLOWQ, GSW, GLW, CAPG, THC
  REAL, DIMENSION(ims:ime,jms:jme) :: SNOWC, EMISS, MAVAIL
  REAL, DIMENSION(ims:ime,NSOIL,jms:jme) :: TSLB
  REAL, DIMENSION(NSOIL) :: ZS, DZS
  REAL :: DELTSM, DTMIN
  INTEGER :: IFSNOW
  LOGICAL :: radiation

  ! Saved inputs for dump.
  REAL, DIMENSION(its:ite) :: TSK_IN, HFX_IN, QFX_IN
  REAL, DIMENSION(its:ite,NSOIL) :: TSLB_IN

  INTEGER :: case_id, i, k
  CHARACTER(LEN=32) :: arg, precision_mode
  CHARACTER(LEN=64) :: regime_name

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
     CALL GET_COMMAND_ARGUMENT(1, arg)
     READ(arg,*) case_id
  ELSE
     case_id = 1
  END IF
  IF (COMMAND_ARGUMENT_COUNT() >= 2) THEN
     CALL GET_COMMAND_ARGUMENT(2, precision_mode)
  ELSE
     precision_mode = 'fp32'
  END IF

  CALL build_case(case_id, regime_name)

  ! Save inputs.
  DO i = its, ite
     TSK_IN(i) = TSK(i,1)
     HFX_IN(i) = HFX(i,1)
     QFX_IN(i) = QFX(i,1)
     DO k = 1, NSOIL
        TSLB_IN(i,k) = TSLB(i,k,1)
     END DO
  END DO

  CALL slab(T3D,QV3D,P3D,FLHC,FLQC,                       &
            PSFC,XLAND,TMN,HFX,QFX,LH,TSK,QSFC,CHKLOWQ,   &
            GSW,GLW,CAPG,THC,SNOWC,EMISS,MAVAIL,          &
            DELTSM,ROVCP,XLV,DTMIN,IFSNOW,                &
            SVP1,SVP2,SVP3,SVPT0,EP2,                     &
            KARMAN,EOMEG,STBOLT,                          &
            TSLB,ZS,DZS,NSOIL,radiation,                  &
            P1000MB,                                      &
            ids,ide, jds,jde, kds,kde,                    &
            ims,ime, jms,jme, kms,kme,                    &
            its,ite, jts,jte, kts,kte)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,A)')  'REGIME=', TRIM(regime_name)
  WRITE(*,'(A,A)')  'PRECISION_MODE=', TRIM(precision_mode)
  WRITE(*,'(A,I0)') 'N=', N
  WRITE(*,'(A,I0)') 'NSOIL=', NSOIL
  WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
  WRITE(*,'(A,I0)') 'IFSNOW=', IFSNOW
  WRITE(*,'(A,ES23.15)') 'DELTSM=', DELTSM
  WRITE(*,'(A,ES23.15)') 'DTMIN=', DTMIN
  DO k = 1, NSOIL
     CALL dumpk('ZS', k, ZS(k))
     CALL dumpk('DZS', k, DZS(k))
  END DO

  DO i = its, ite
     ! inputs
     CALL dump('T', i, T3D(i,1,1))
     CALL dump('QV', i, QV3D(i,1,1))
     CALL dump('P', i, P3D(i,1,1))
     CALL dump('FLHC', i, FLHC(i,1))
     CALL dump('FLQC', i, FLQC(i,1))
     CALL dump('PSFC', i, PSFC(i,1))
     CALL dump('XLAND', i, XLAND(i,1))
     CALL dump('TMN', i, TMN(i,1))
     CALL dump('GSW', i, GSW(i,1))
     CALL dump('GLW', i, GLW(i,1))
     CALL dump('THC', i, THC(i,1))
     CALL dump('SNOWC', i, SNOWC(i,1))
     CALL dump('EMISS', i, EMISS(i,1))
     CALL dump('MAVAIL', i, MAVAIL(i,1))
     CALL dump('TSK_IN', i, TSK_IN(i))
     CALL dump('HFX_IN', i, HFX_IN(i))
     CALL dump('QFX_IN', i, QFX_IN(i))
     DO k = 1, NSOIL
        CALL dump2('TSLB_IN', i, k, TSLB_IN(i,k))
     END DO
     ! outputs
     CALL dump('TSK', i, TSK(i,1))
     CALL dump('HFX', i, HFX(i,1))
     CALL dump('QFX', i, QFX(i,1))
     CALL dump('LH', i, LH(i,1))
     CALL dump('QSFC', i, QSFC(i,1))
     CALL dump('CHKLOWQ', i, CHKLOWQ(i,1))
     CALL dump('CAPG', i, CAPG(i,1))
     DO k = 1, NSOIL
        CALL dump2('TSLB', i, k, TSLB(i,k,1))
     END DO
  END DO

CONTAINS

  SUBROUTINE dump(name, idx, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', value
  END SUBROUTINE dump

  SUBROUTINE dump2(name, idx, kk, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx, kk
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,I0,A,ES23.15)') TRIM(name),'[',idx,',',kk,']=', value
  END SUBROUTINE dump2

  SUBROUTINE dumpk(name, kk, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: kk
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',kk,']=', value
  END SUBROUTINE dumpk

  SUBROUTINE init_common()
    T3D=0.0; QV3D=0.0; P3D=0.0
    FLHC=0.0; FLQC=0.0; PSFC=0.0; XLAND=1.0; TMN=285.0
    HFX=0.0; QFX=0.0; LH=0.0; TSK=288.0; QSFC=0.0; CHKLOWQ=0.0
    GSW=0.0; GLW=0.0; CAPG=0.0; THC=0.04; SNOWC=0.0; EMISS=0.98; MAVAIL=1.0
    TSLB=285.0
    ! WRF 5-layer slab ZS/DZS (the registry defaults; ZS = cumulative centers).
    DZS = (/ 0.01, 0.02, 0.04, 0.08, 0.16 /)
    ZS(1) = 0.5*DZS(1)
    ZS(2) = ZS(1) + 0.5*(DZS(1)+DZS(2))
    ZS(3) = ZS(2) + 0.5*(DZS(2)+DZS(3))
    ZS(4) = ZS(3) + 0.5*(DZS(3)+DZS(4))
    ZS(5) = ZS(4) + 0.5*(DZS(4)+DZS(5))
    radiation = .TRUE.
    IFSNOW = 0
    DELTSM = 60.0
    DTMIN = DELTSM / 60.0
  END SUBROUTINE init_common

  SUBROUTINE setcol(i, tval, qvval, pval, flhcval, flqcval, psfcval, xlandval, &
                    tmnval, gswval, glwval, thcval, snowcval, emissval, mavailval, &
                    tskval, hfxval, qfxval, tslbval)
    INTEGER, INTENT(IN) :: i
    REAL, INTENT(IN) :: tval, qvval, pval, flhcval, flqcval, psfcval, xlandval
    REAL, INTENT(IN) :: tmnval, gswval, glwval, thcval, snowcval, emissval, mavailval
    REAL, INTENT(IN) :: tskval, hfxval, qfxval, tslbval
    INTEGER :: kk
    T3D(i,1,1)=tval; QV3D(i,1,1)=qvval; P3D(i,1,1)=pval
    FLHC(i,1)=flhcval; FLQC(i,1)=flqcval; PSFC(i,1)=psfcval; XLAND(i,1)=xlandval
    TMN(i,1)=tmnval; GSW(i,1)=gswval; GLW(i,1)=glwval; THC(i,1)=thcval
    SNOWC(i,1)=snowcval; EMISS(i,1)=emissval; MAVAIL(i,1)=mavailval
    TSK(i,1)=tskval; HFX(i,1)=hfxval; QFX(i,1)=qfxval
    DO kk = 1, NSOIL
       TSLB(i,kk,1)=tslbval
    END DO
  END SUBROUTINE setcol

  SUBROUTINE build_case(cid, name)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(OUT) :: name
    CALL init_common()
    SELECT CASE (cid)
    CASE (1)
       name = 'daytime_heating_land_and_water'
       ! Strong daytime SW: warm skin, positive net radiation -> heating.
       CALL setcol(1, 292.0, 0.0080, 95000.0, 0.020, 0.018, 95450.0, 1.0, 285.0, 600.0, 330.0, 0.04, 0.0, 0.98, 0.70, 298.0, 60.0, 0.0001, 290.0)
       CALL setcol(2, 294.0, 0.0100, 99800.0, 0.025, 0.022, 100250.0, 2.0, 290.0, 700.0, 360.0, 0.06, 0.0, 0.98, 1.00, 297.0, 20.0, 0.0002, 291.0)
       CALL setcol(3, 289.0, 0.0065, 92000.0, 0.015, 0.013, 92450.0, 1.0, 283.0, 500.0, 320.0, 0.03, 0.0, 0.97, 0.55, 295.0, 15.0, 0.00005, 286.0)
       CALL setcol(4, 291.5, 0.0095, 101000.0, 0.022, 0.020, 101450.0, 2.0, 288.0, 650.0, 350.0, 0.05, 0.0, 0.98, 1.00, 294.5, 5.0, 0.0001, 289.0)
       CALL setcol(5, 300.0, 0.0120, 96000.0, 0.030, 0.027, 96450.0, 1.0, 290.0, 800.0, 380.0, 0.045, 0.0, 0.96, 0.80, 305.0, 120.0, 0.0003, 296.0)
       CALL setcol(6, 296.0, 0.0110, 100500.0, 0.018, 0.016, 100950.0, 2.0, 291.0, 720.0, 365.0, 0.06, 0.0, 0.98, 1.00, 299.0, 0.0, 0.0, 293.0)
       CALL setcol(7, 285.0, 0.0040, 88000.0, 0.012, 0.010, 88450.0, 1.0, 280.0, 450.0, 300.0, 0.025, 0.0, 0.95, 0.60, 291.0, 30.0, 0.00004, 282.0)
       CALL setcol(8, 293.0, 0.0090, 99000.0, 0.024, 0.021, 99450.0, 2.0, 289.0, 680.0, 355.0, 0.055, 0.0, 0.98, 1.00, 296.0, 10.0, 0.0002, 290.0)
    CASE (2)
       name = 'nocturnal_cooling_land'
       ! No SW, net LW loss -> cooling skin/soil.
       CALL setcol(1, 286.0, 0.0070, 95000.0, 0.012, 0.011, 95450.0, 1.0, 286.0, 0.0, 300.0, 0.04, 0.0, 0.98, 0.65, 282.0, -20.0, -0.00001, 285.0)
       CALL setcol(2, 285.0, 0.0060, 92000.0, 0.010, 0.009, 92450.0, 1.0, 284.0, 0.0, 290.0, 0.03, 0.0, 0.97, 0.50, 280.0, -30.0, -0.00002, 283.5)
       CALL setcol(3, 288.0, 0.0080, 100000.0, 0.014, 0.012, 100450.0, 1.0, 287.0, 0.0, 310.0, 0.05, 0.0, 0.98, 0.70, 283.5, -15.0, -0.00001, 286.0)
       CALL setcol(4, 290.0, 0.0090, 101000.0, 0.016, 0.014, 101450.0, 1.0, 289.0, 0.0, 320.0, 0.06, 0.0, 0.98, 0.80, 285.0, -10.0, -0.00001, 287.0)
       CALL setcol(5, 283.0, 0.0050, 90000.0, 0.009, 0.008, 90450.0, 1.0, 282.0, 0.0, 280.0, 0.025, 0.0, 0.95, 0.45, 278.0, -25.0, -0.00002, 281.0)
       CALL setcol(6, 291.0, 0.0100, 99500.0, 0.015, 0.013, 99950.0, 1.0, 290.0, 0.0, 315.0, 0.055, 0.0, 0.98, 0.75, 287.0, -12.0, -0.00001, 288.0)
       CALL setcol(7, 287.0, 0.0075, 96000.0, 0.013, 0.011, 96450.0, 1.0, 286.0, 0.0, 305.0, 0.045, 0.0, 0.97, 0.60, 282.5, -18.0, -0.00001, 285.5)
       CALL setcol(8, 289.0, 0.0085, 98000.0, 0.014, 0.012, 98450.0, 1.0, 288.0, 0.0, 312.0, 0.05, 0.0, 0.98, 0.70, 284.0, -14.0, -0.00001, 287.0)
    CASE (3)
       name = 'all_ocean_unchanged'
       ! All water columns: SLAB must leave TSK/TSLB unchanged (XLD1<0.5 skip).
       CALL setcol(1, 292.0, 0.0080, 95000.0, 0.020, 0.018, 95450.0, 2.0, 290.0, 600.0, 330.0, 0.06, 0.0, 0.98, 1.00, 297.0, 50.0, 0.0001, 291.0)
       CALL setcol(2, 294.0, 0.0100, 99800.0, 0.025, 0.022, 100250.0, 2.0, 291.0, 700.0, 360.0, 0.06, 0.0, 0.98, 1.00, 298.0, 40.0, 0.0002, 292.0)
       CALL setcol(3, 289.0, 0.0065, 92000.0, 0.015, 0.013, 92450.0, 2.0, 288.0, 500.0, 320.0, 0.06, 0.0, 0.98, 1.00, 295.0, 30.0, 0.00005, 290.0)
       CALL setcol(4, 291.5, 0.0095, 101000.0, 0.022, 0.020, 101450.0, 2.0, 289.0, 650.0, 350.0, 0.06, 0.0, 0.98, 1.00, 296.0, 20.0, 0.0001, 291.0)
       CALL setcol(5, 300.0, 0.0120, 96000.0, 0.030, 0.027, 96450.0, 2.0, 292.0, 800.0, 380.0, 0.06, 0.0, 0.98, 1.00, 301.0, 60.0, 0.0003, 293.0)
       CALL setcol(6, 296.0, 0.0110, 100500.0, 0.018, 0.016, 100950.0, 2.0, 291.0, 720.0, 365.0, 0.06, 0.0, 0.98, 1.00, 299.0, 10.0, 0.0001, 293.0)
       CALL setcol(7, 285.0, 0.0040, 88000.0, 0.012, 0.010, 88450.0, 2.0, 283.0, 450.0, 300.0, 0.06, 0.0, 0.98, 1.00, 290.0, 25.0, 0.00004, 287.0)
       CALL setcol(8, 293.0, 0.0090, 99000.0, 0.024, 0.021, 99450.0, 2.0, 290.0, 680.0, 355.0, 0.06, 0.0, 0.98, 1.00, 296.0, 15.0, 0.0002, 291.0)
    CASE (4)
       name = 'snow_cover_ifsnow_limit'
       IFSNOW = 1
       CALL setcol(1, 270.0, 0.0020, 91000.0, 0.012, 0.011, 91450.0, 1.0, 270.0, 300.0, 250.0, 0.03, 1.0, 0.98, 0.30, 275.0, 10.0, 0.00001, 274.0)
       CALL setcol(2, 272.0, 0.0030, 98000.0, 0.014, 0.012, 98450.0, 1.0, 271.0, 350.0, 260.0, 0.04, 1.0, 0.98, 0.35, 276.0, 12.0, 0.00002, 274.5)
       CALL setcol(3, 268.0, 0.0015, 88000.0, 0.010, 0.009, 88450.0, 1.0, 267.0, 250.0, 240.0, 0.025, 1.0, 0.97, 0.25, 273.0, 8.0, 0.00001, 272.0)
       CALL setcol(4, 274.0, 0.0035, 101000.0, 0.016, 0.014, 101450.0, 1.0, 272.0, 400.0, 270.0, 0.045, 1.0, 0.98, 0.40, 277.0, 15.0, 0.00002, 275.0)
       CALL setcol(5, 271.0, 0.0025, 94000.0, 0.013, 0.011, 94450.0, 1.0, 270.0, 320.0, 255.0, 0.035, 1.0, 0.96, 0.30, 275.5, 11.0, 0.00001, 274.0)
       CALL setcol(6, 273.0, 0.0030, 100500.0, 0.015, 0.013, 100950.0, 1.0, 272.0, 360.0, 262.0, 0.04, 1.0, 0.98, 0.35, 276.0, 9.0, 0.00001, 274.5)
       CALL setcol(7, 266.0, 0.0010, 85000.0, 0.008, 0.007, 85450.0, 1.0, 265.0, 220.0, 235.0, 0.02, 1.0, 0.95, 0.20, 272.0, 6.0, 0.00001, 271.0)
       CALL setcol(8, 271.0, 0.0028, 99000.0, 0.014, 0.012, 99450.0, 1.0, 270.0, 340.0, 258.0, 0.04, 1.0, 0.98, 0.32, 275.0, 10.0, 0.00001, 274.0)
    CASE DEFAULT
       name = 'mixed_land_water_strong_substep'
       ! High THC + long-ish DELTSM stress the soil substep cadence.
       DELTSM = 180.0
       DTMIN = DELTSM / 60.0
       CALL setcol(1, 292.0, 0.0080, 95000.0, 0.020, 0.018, 95450.0, 1.0, 285.0, 600.0, 330.0, 0.12, 0.0, 0.98, 0.70, 298.0, 60.0, 0.0001, 290.0)
       CALL setcol(2, 294.0, 0.0100, 99800.0, 0.025, 0.022, 100250.0, 2.0, 290.0, 700.0, 360.0, 0.12, 0.0, 0.98, 1.00, 297.0, 20.0, 0.0002, 291.0)
       CALL setcol(3, 289.0, 0.0065, 92000.0, 0.015, 0.013, 92450.0, 1.0, 283.0, 500.0, 320.0, 0.10, 0.0, 0.97, 0.55, 295.0, 15.0, 0.00005, 286.0)
       CALL setcol(4, 291.5, 0.0095, 101000.0, 0.022, 0.020, 101450.0, 2.0, 288.0, 650.0, 350.0, 0.12, 0.0, 0.98, 1.00, 294.5, 5.0, 0.0001, 289.0)
       CALL setcol(5, 300.0, 0.0120, 96000.0, 0.030, 0.027, 96450.0, 1.0, 290.0, 800.0, 380.0, 0.11, 0.0, 0.96, 0.80, 305.0, 120.0, 0.0003, 296.0)
       CALL setcol(6, 296.0, 0.0110, 100500.0, 0.018, 0.016, 100950.0, 2.0, 291.0, 720.0, 365.0, 0.12, 0.0, 0.98, 1.00, 299.0, 0.0, 0.0, 293.0)
       CALL setcol(7, 285.0, 0.0040, 88000.0, 0.012, 0.010, 88450.0, 1.0, 280.0, 450.0, 300.0, 0.10, 0.0, 0.95, 0.60, 291.0, 30.0, 0.00004, 282.0)
       CALL setcol(8, 293.0, 0.0090, 99000.0, 0.024, 0.021, 99450.0, 2.0, 289.0, 680.0, 355.0, 0.11, 0.0, 0.98, 1.00, 296.0, 10.0, 0.0002, 290.0)
    END SELECT
  END SUBROUTINE build_case

END PROGRAM slab_oracle
