PROGRAM pxsfclay_oracle
  USE module_sf_pxsfclay, ONLY : pxsfclay
  IMPLICIT NONE

  REAL, PARAMETER :: CP = 1004.0
  REAL, PARAMETER :: G = 9.81
  REAL, PARAMETER :: R = 287.0
  REAL, PARAMETER :: ROVCP = R / CP
  REAL, PARAMETER :: XLV = 2.5E6
  REAL, PARAMETER :: SVP1 = 0.6112
  REAL, PARAMETER :: SVP2 = 17.67
  REAL, PARAMETER :: SVP3 = 29.65
  REAL, PARAMETER :: SVPT0 = 273.15
  REAL, PARAMETER :: EP1 = 0.608
  REAL, PARAMETER :: EP2 = 0.622
  REAL, PARAMETER :: KARMAN = 0.40
  REAL, PARAMETER :: P1000MB = 100000.0

  INTEGER, PARAMETER :: N = 8
  INTEGER, PARAMETER :: ids=1, ide=N+1, jds=1, jde=2, kds=1, kde=2
  INTEGER, PARAMETER :: ims=1, ime=N,   jms=1, jme=1, kms=1, kme=1
  INTEGER, PARAMETER :: its=1, ite=N,   jts=1, jte=1, kts=1, kte=1

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U3D,V3D,T3D,TH3D,QV3D,P3D,DZ8W
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,CHS,CHS2,CQS2,CPM,ZNT,UST,PBLH,MAVAIL,ZOL,MOL
  REAL, DIMENSION(ims:ime,jms:jme) :: REGIME,PSIM,PSIH,XLAND,HFX,QFX,LH,TSK,FLHC,FLQC,QGH
  REAL, DIMENSION(ims:ime,jms:jme) :: QSFC,RMOL,U10,V10,GZ1OZ0,WSPD,BR
  REAL, DIMENSION(ims:ime,jms:jme) :: TH2,T2,Q2
  REAL, DIMENSION(ims:ime) :: UST_IN_SAVE, MOL_IN_SAVE, HFX_IN_SAVE, QFX_IN_SAVE, QSFC_IN_SAVE, ZNT_IN_SAVE
  REAL :: DX
  INTEGER :: case_id, i, itimestep, isfflx
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

  CALL build_case(case_id, regime_name, itimestep, isfflx, DX)

  CALL pxsfclay(U3D,V3D,T3D,TH3D,QV3D,P3D,dz8w, &
       CP,G,ROVCP,R,XLV,PSFC,CHS,CHS2,CQS2,CPM, &
       ZNT,UST,PBLH,MAVAIL,ZOL,MOL,REGIME,PSIM,PSIH, &
       XLAND,HFX,QFX,LH,TSK,FLHC,FLQC,QGH,QSFC,RMOL, &
       U10,V10, &
       GZ1OZ0,WSPD,BR,isfflx,DX, &
       SVP1,SVP2,SVP3,SVPT0,EP1,EP2,KARMAN, &
       itimestep, &
       ids,ide,jds,jde,kds,kde, ims,ime,jms,jme,kms,kme, &
       its,ite,jts,jte,kts,kte)

  CALL derive_2m_diagnostics()

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,A)') 'REGIME_NAME=', TRIM(regime_name)
  WRITE(*,'(A,A)') 'PRECISION_MODE=', TRIM(precision_mode)
  WRITE(*,'(A,I0)') 'N=', N
  WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
  WRITE(*,'(A,I0)') 'ISFFLX=', isfflx
  WRITE(*,'(A,I0)') 'ITIMESTEP=', itimestep
  WRITE(*,'(A,ES23.15)') 'DT=', 0.0

  DO i = its, ite
     CALL dump('U', i, U3D(i,1,1))
     CALL dump('V', i, V3D(i,1,1))
     CALL dump('T', i, T3D(i,1,1))
     CALL dump('THETA', i, TH3D(i,1,1))
     CALL dump('QV', i, QV3D(i,1,1))
     CALL dump('P', i, P3D(i,1,1))
     CALL dump('DZ', i, DZ8W(i,1,1))
     CALL dump('PSFC', i, PSFC(i,1))
     CALL dump('TSK', i, TSK(i,1))
     CALL dump('XLAND', i, XLAND(i,1))
     CALL dump('MAVAIL', i, MAVAIL(i,1))
     CALL dump('PBLH', i, PBLH(i,1))
     CALL dump('DX', i, DX)
     CALL dump('UST_IN', i, UST_IN_SAVE(i))
     CALL dump('MOL_IN', i, MOL_IN_SAVE(i))
     CALL dump('HFX_IN', i, HFX_IN_SAVE(i))
     CALL dump('QFX_IN', i, QFX_IN_SAVE(i))
     CALL dump('QSFC_IN', i, QSFC_IN_SAVE(i))
     CALL dump('ZNT_IN', i, ZNT_IN_SAVE(i))
     CALL dump('UST', i, UST(i,1))
     CALL dump('TSTAR', i, MOL(i,1))
     CALL dump('QSTAR', i, -QFX(i,1) / MAX(PSFC(i,1)/(R*T3D(i,1,1)*(1.0+EP1*QV3D(i,1,1))) * UST(i,1), 1.0E-12))
     CALL dump('HFX', i, HFX(i,1))
     CALL dump('QFX', i, QFX(i,1))
     CALL dump('LH', i, LH(i,1))
     CALL dump('U10', i, U10(i,1))
     CALL dump('V10', i, V10(i,1))
     CALL dump('TH2', i, TH2(i,1))
     CALL dump('T2', i, T2(i,1))
     CALL dump('Q2', i, Q2(i,1))
     CALL dump('CHS', i, CHS(i,1))
     CALL dump('CHS2', i, CHS2(i,1))
     CALL dump('CQS2', i, CQS2(i,1))
     CALL dump('CPM', i, CPM(i,1))
     CALL dump('FLHC', i, FLHC(i,1))
     CALL dump('FLQC', i, FLQC(i,1))
     CALL dump('QSFC', i, QSFC(i,1))
     CALL dump('QGH', i, QGH(i,1))
     CALL dump('ZNT', i, ZNT(i,1))
     CALL dump('ZOL', i, ZOL(i,1))
     CALL dump('MOL', i, MOL(i,1))
     CALL dump('RMOL', i, RMOL(i,1))
     CALL dump('REGIME', i, REGIME(i,1))
     CALL dump('PSIM', i, PSIM(i,1))
     CALL dump('PSIH', i, PSIH(i,1))
     CALL dump('BR', i, BR(i,1))
     CALL dump('WSPD', i, WSPD(i,1))
     CALL dump('GZ1OZ0', i, GZ1OZ0(i,1))
  END DO

CONTAINS

  SUBROUTINE dump(name, idx, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', value
  END SUBROUTINE dump

  SUBROUTINE derive_2m_diagnostics()
    REAL :: rho2
    DO i = its, ite
       rho2 = PSFC(i,1) / (R * TSK(i,1))
       IF (CQS2(i,1) .LT. 1.E-5) THEN
          Q2(i,1) = QSFC(i,1)
       ELSE
          Q2(i,1) = QSFC(i,1) - QFX(i,1) / (rho2 * CQS2(i,1))
       ENDIF
       IF (CHS2(i,1) .LT. 1.E-5) THEN
          T2(i,1) = TSK(i,1)
       ELSE
          T2(i,1) = TSK(i,1) - HFX(i,1) / (rho2 * CP * CHS2(i,1))
       ENDIF
       TH2(i,1) = T2(i,1) * (P1000MB / PSFC(i,1))**ROVCP
    ENDDO
  END SUBROUTINE derive_2m_diagnostics

  SUBROUTINE init_common()
    U3D=0.0; V3D=0.0; T3D=0.0; TH3D=0.0; QV3D=0.0; P3D=0.0; DZ8W=0.0
    PSFC=0.0; CHS=0.0; CHS2=0.0; CQS2=0.0; CPM=CP
    ZNT=0.0; UST=0.0; PBLH=800.0; MAVAIL=1.0; ZOL=0.0; MOL=0.0
    REGIME=0.0; PSIM=0.0; PSIH=0.0; XLAND=1.0
    HFX=0.0; QFX=0.0; LH=0.0; TSK=0.0; FLHC=0.0; FLQC=0.0; QGH=0.0
    QSFC=-1.0; RMOL=0.0; U10=0.0; V10=0.0; TH2=0.0; T2=0.0; Q2=0.0
    GZ1OZ0=0.0; WSPD=0.0; BR=0.0
    UST_IN_SAVE=0.0; MOL_IN_SAVE=0.0; HFX_IN_SAVE=0.0; QFX_IN_SAVE=0.0; QSFC_IN_SAVE=-1.0; ZNT_IN_SAVE=0.0
  END SUBROUTINE init_common

  SUBROUTINE setcol(i,uval,vval,tval,qvval,pval,dzval,tskval,xlandval,zntval,ustval,molval,hfxval,qfxval,qsfcval,mavailval,pblhval)
    INTEGER, INTENT(IN) :: i
    REAL, INTENT(IN) :: uval,vval,tval,qvval,pval,dzval,tskval,xlandval,zntval,ustval,molval
    REAL, INTENT(IN) :: hfxval,qfxval,qsfcval,mavailval,pblhval
    U3D(i,1,1)=uval; V3D(i,1,1)=vval; T3D(i,1,1)=tval; QV3D(i,1,1)=qvval; P3D(i,1,1)=pval; DZ8W(i,1,1)=dzval
    TH3D(i,1,1)=tval * (P1000MB / pval)**ROVCP
    PSFC(i,1)=pval - 450.0
    TSK(i,1)=tskval; XLAND(i,1)=xlandval; ZNT(i,1)=zntval; UST(i,1)=ustval; MOL(i,1)=molval
    HFX(i,1)=hfxval; QFX(i,1)=qfxval; QSFC(i,1)=qsfcval; MAVAIL(i,1)=mavailval; PBLH(i,1)=pblhval
    UST_IN_SAVE(i)=ustval; MOL_IN_SAVE(i)=molval; HFX_IN_SAVE(i)=hfxval; QFX_IN_SAVE(i)=qfxval
    QSFC_IN_SAVE(i)=qsfcval; ZNT_IN_SAVE(i)=zntval
  END SUBROUTINE setcol

  SUBROUTINE build_case(cid, name, l_itimestep, l_isfflx, l_dx)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(OUT) :: name
    INTEGER, INTENT(OUT) :: l_itimestep, l_isfflx
    REAL, INTENT(OUT) :: l_dx
    CALL init_common()
    l_isfflx = 1
    l_itimestep = 2
    l_dx = 3000.0
    SELECT CASE (cid)
    CASE (1)
       name = 'unstable_convective_land_water'
       l_itimestep = 1
       CALL setcol(1, 6.0,  1.0, 292.0, 0.0080, 95000.0, 70.0, 298.0, 1.0, 0.0800, 0.10, 0.0,  80.0, 0.00008, -1.0, 0.70, 1200.0)
       CALL setcol(2, 8.0, -1.5, 294.0, 0.0100, 99800.0, 58.0, 297.0, 2.0, 0.0016, 0.12, 0.0,  20.0, 0.00005, -1.0, 1.00,  900.0)
       CALL setcol(3, 3.0,  0.5, 289.0, 0.0065, 92000.0, 80.0, 295.0, 1.0, 0.1200, 0.05, 0.0,  15.0, 0.00002, -1.0, 0.55, 1500.0)
       CALL setcol(4, 5.0,  4.0, 291.5, 0.0095,101000.0, 54.0, 294.5, 2.0, 0.0025, 0.20, 0.0,   5.0, 0.00003, -1.0, 1.00,  750.0)
       CALL setcol(5,10.0,  0.2, 300.0, 0.0120, 96000.0, 66.0, 305.0, 1.0, 0.0400, 0.18, 0.0, 120.0, 0.00012, -1.0, 0.80, 1800.0)
       CALL setcol(6, 4.5, -3.0, 296.0, 0.0110,100500.0, 60.0, 299.0, 2.0, 0.0012, 0.15, 0.0,   0.0, 0.00000, -1.0, 1.00,  650.0)
       CALL setcol(7, 7.5,  2.0, 285.0, 0.0040, 88000.0, 90.0, 291.0, 1.0, 0.2000, 0.08, 0.0,  30.0, 0.00004, -1.0, 0.60, 1100.0)
       CALL setcol(8, 9.0,  1.0, 293.0, 0.0090, 99000.0, 55.0, 296.0, 2.0, 0.0018, 0.11, 0.0,  10.0, 0.00002, -1.0, 1.00,  800.0)
    CASE (2)
       name = 'stable_nocturnal_land_water'
       CALL setcol(1, 3.5,  0.4, 292.0, 0.0070, 95000.0, 70.0, 286.0, 1.0, 0.0800, 0.08, 0.0, -20.0,-0.00001, -1.0, 0.65, 350.0)
       CALL setcol(2, 6.0, -1.2, 295.0, 0.0100,100000.0, 58.0, 291.0, 2.0, 0.0016, 0.10, 0.0,  -5.0,-0.00001, -1.0, 1.00, 300.0)
       CALL setcol(3, 2.0,  0.2, 289.0, 0.0055, 92000.0, 80.0, 283.5, 1.0, 0.1200, 0.04, 0.0, -40.0,-0.00002, -1.0, 0.45, 250.0)
       CALL setcol(4, 4.5,  2.0, 294.0, 0.0090,101000.0, 54.0, 290.0, 2.0, 0.0025, 0.12, 0.0,  -3.0,-0.00001, -1.0, 1.00, 280.0)
       CALL setcol(5, 8.0,  0.3, 299.0, 0.0110, 96000.0, 66.0, 294.0, 1.0, 0.0400, 0.15, 0.0, -15.0,-0.00002, -1.0, 0.75, 400.0)
       CALL setcol(6, 2.8, -2.5, 296.0, 0.0100,100500.0, 60.0, 292.0, 2.0, 0.0012, 0.08, 0.0,  -2.0,-0.00001, -1.0, 1.00, 260.0)
       CALL setcol(7, 5.0,  1.4, 285.0, 0.0040, 88000.0, 90.0, 280.0, 1.0, 0.2000, 0.06, 0.0, -60.0,-0.00002, -1.0, 0.55, 220.0)
       CALL setcol(8, 7.0,  0.8, 293.0, 0.0090, 99000.0, 55.0, 289.5, 2.0, 0.0018, 0.09, 0.0,  -4.0,-0.00001, -1.0, 1.00, 300.0)
    CASE (3)
       name = 'near_neutral_persisted_land_qsfc'
       CALL setcol(1, 5.0, 1.0, 292.0, 0.0080, 95000.0, 70.0, 291.6039447, 1.0, 0.0800, 0.10, 0.0, 0.0, 0.0, 0.0080, 0.70, 800.0)
       CALL setcol(2, 7.0, 0.0, 294.0, 0.0100,100000.0, 58.0, 292.8026107, 2.0, 0.0016, 0.12, 0.0, 0.0, 0.0, 0.0100, 1.00, 800.0)
       CALL setcol(3, 3.5, 0.8, 289.0, 0.0060, 92000.0, 80.0, 288.5952091, 1.0, 0.1200, 0.07, 0.0, 0.0, 0.0, 0.0060, 0.60, 800.0)
       CALL setcol(4, 6.0, 2.0, 296.0, 0.0110,101000.0, 54.0, 294.6790261, 2.0, 0.0025, 0.11, 0.0, 0.0, 0.0, 0.0110, 1.00, 800.0)
       CALL setcol(5, 8.0, 1.0, 300.0, 0.0120, 96000.0, 66.0, 299.5973396, 1.0, 0.0400, 0.16, 0.0, 0.0, 0.0, 0.0120, 0.80, 800.0)
       CALL setcol(6, 4.0,-1.0, 296.0, 0.0100,100500.0, 60.0, 294.5142979, 2.0, 0.0012, 0.10, 0.0, 0.0, 0.0, 0.0100, 1.00, 800.0)
       CALL setcol(7, 5.5, 1.5, 285.0, 0.0040, 88000.0, 90.0, 284.5826336, 1.0, 0.2000, 0.08, 0.0, 0.0, 0.0, 0.0040, 0.55, 800.0)
       CALL setcol(8, 7.0, 0.5, 293.0, 0.0090, 99000.0, 55.0, 291.7661183, 2.0, 0.0018, 0.10, 0.0, 0.0, 0.0, 0.0090, 1.00, 800.0)
    CASE (4)
       name = 'very_stable_low_wind_guard'
       l_dx = 9000.0
       CALL setcol(1, 0.6, 0.1, 292.0, 0.0070, 95000.0, 70.0, 274.0, 1.0, 0.0800, 0.02, 0.0, -20.0,-0.00001, -1.0, 0.70, 250.0)
       CALL setcol(2, 0.8, 0.1, 295.0, 0.0100,100000.0, 58.0, 280.0, 2.0, 0.0016, 0.02, 0.0,  -5.0,-0.00001, -1.0, 1.00, 250.0)
       CALL setcol(3, 0.5, 0.0, 289.0, 0.0060, 92000.0, 80.0, 270.5, 1.0, 0.1200, 0.01, 0.0, -40.0,-0.00002, -1.0, 0.50, 220.0)
       CALL setcol(4, 0.9, 0.2, 294.0, 0.0090,101000.0, 54.0, 281.0, 2.0, 0.0025, 0.02, 0.0,  -3.0,-0.00001, -1.0, 1.00, 230.0)
       CALL setcol(5, 1.0, 0.1, 299.0, 0.0110, 96000.0, 66.0, 284.0, 1.0, 0.0400, 0.02, 0.0, -15.0,-0.00002, -1.0, 0.75, 250.0)
       CALL setcol(6, 0.7,-0.2, 296.0, 0.0100,100500.0, 60.0, 282.0, 2.0, 0.0012, 0.02, 0.0,  -2.0,-0.00001, -1.0, 1.00, 220.0)
       CALL setcol(7, 0.6, 0.2, 285.0, 0.0040, 88000.0, 90.0, 268.0, 1.0, 0.2000, 0.01, 0.0, -60.0,-0.00002, -1.0, 0.55, 220.0)
       CALL setcol(8, 1.1, 0.1, 293.0, 0.0090, 99000.0, 55.0, 281.0, 2.0, 0.0018, 0.02, 0.0,  -4.0,-0.00001, -1.0, 1.00, 220.0)
    CASE (5)
       name = 'first_timestep_ice_saturation'
       l_itimestep = 1
       CALL setcol(1, 3.0,0.5,268.0,0.0020, 91000.0,70.0,266.0,1.0,0.0800,0.05,0.0, -8.0,-0.00001,0.0060,0.70,500.0)
       CALL setcol(2, 4.0,0.3,272.0,0.0030, 98000.0,58.0,269.0,2.0,0.0016,0.06,0.0, -4.0,-0.00001,0.0060,1.00,500.0)
       CALL setcol(3, 2.0,0.2,265.0,0.0015, 88000.0,80.0,263.0,1.0,0.1200,0.04,0.0,-15.0,-0.00002,0.0040,0.50,400.0)
       CALL setcol(4, 5.0,0.5,275.0,0.0035,101000.0,54.0,270.0,2.0,0.0025,0.08,0.0, -2.0,-0.00001,0.0060,1.00,450.0)
       CALL setcol(5, 4.5,0.4,270.0,0.0025, 94000.0,66.0,267.0,1.0,0.0400,0.05,0.0,-10.0,-0.00002,0.0050,0.75,450.0)
       CALL setcol(6, 3.5,0.3,273.0,0.0030,100500.0,60.0,271.0,2.0,0.0012,0.06,0.0, -1.0,-0.00001,0.0060,1.00,400.0)
       CALL setcol(7, 2.5,0.2,262.0,0.0010, 85000.0,90.0,260.0,1.0,0.2000,0.03,0.0,-20.0,-0.00002,0.0030,0.55,350.0)
       CALL setcol(8, 4.0,0.2,271.0,0.0028, 99000.0,55.0,268.0,2.0,0.0018,0.06,0.0, -2.0,-0.00001,0.0060,1.00,400.0)
    CASE DEFAULT
       name = 'mixed_regime_persisted_qsfc'
       l_dx = 12000.0
       CALL setcol(1, 6.0,1.0,292.0,0.0080, 95000.0,70.0,298.0,1.0,0.0800,0.10,0.0, 80.0,0.00008,0.0100,0.70,1200.0)
       CALL setcol(2, 6.0,0.0,295.0,0.0100,100000.0,58.0,291.0,2.0,0.0016,0.10,0.0, -5.0,-0.00001,0.0100,1.00, 300.0)
       CALL setcol(3, 4.0,1.0,289.0,0.0060, 92000.0,80.0,289.0,1.0,0.1200,0.07,0.0,  0.0,0.00000,0.0065,0.60, 800.0)
       CALL setcol(4, 5.0,1.0,294.0,0.0090,101000.0,54.0,290.0,2.0,0.0025,0.12,0.0, -3.0,-0.00001,0.0100,1.00, 280.0)
       CALL setcol(5, 8.0,1.0,300.0,0.0120, 96000.0,66.0,305.0,1.0,0.0400,0.16,0.0,120.0,0.00012,0.0140,0.80,1800.0)
       CALL setcol(6, 4.0,1.0,296.0,0.0100,100500.0,60.0,296.0,2.0,0.0012,0.10,0.0,  0.0,0.00000,0.0110,1.00, 800.0)
       CALL setcol(7, 5.0,1.0,285.0,0.0040, 88000.0,90.0,280.0,1.0,0.2000,0.06,0.0,-60.0,-0.00002,0.0050,0.55, 220.0)
       CALL setcol(8, 7.0,1.0,293.0,0.0090, 99000.0,55.0,296.0,2.0,0.0018,0.10,0.0,  0.0,0.00000,0.0100,1.00, 800.0)
    END SELECT
  END SUBROUTINE build_case

END PROGRAM pxsfclay_oracle
