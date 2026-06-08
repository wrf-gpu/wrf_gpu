PROGRAM sfclay_old_mm5_oracle
  ! Single-column oracle for the WRF old-MM5 Monin-Obukhov surface layer
  ! (UNMODIFIED phys/module_sf_sfclay.F, SUBROUTINE SFCLAY). Emits gold
  ! savepoints (inputs + outputs) for sfclay_old_mm5_dump_to_json.py.
  USE module_sf_sfclay, ONLY : sfclay, sfclayinit
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
  REAL, PARAMETER :: EOMEG = 7.2921E-5
  REAL, PARAMETER :: STBOLT = 5.67051E-8
  REAL, PARAMETER :: P1000MB = 100000.0

  INTEGER, PARAMETER :: N = 8
  INTEGER, PARAMETER :: ids=1, ide=N+1, jds=1, jde=2, kds=1, kde=2
  INTEGER, PARAMETER :: ims=1, ime=N,   jms=1, jme=1, kms=1, kme=1
  INTEGER, PARAMETER :: its=1, ite=N,   jts=1, jte=1, kts=1, kte=1

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U3D,V3D,T3D,QV3D,P3D,DZ8W
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,CHS,CHS2,CQS2,CPM,ZNT,UST,PBLH,MAVAIL,ZOL,MOL
  REAL, DIMENSION(ims:ime,jms:jme) :: REGIME,PSIM,PSIH,FM,FH,XLAND,HFX,QFX,LH,TSK,FLHC,FLQC,QGH
  REAL, DIMENSION(ims:ime,jms:jme) :: QSFC,RMOL,U10,V10,TH2,T2,Q2,GZ1OZ0,WSPD,BR,LAKEMASK
  REAL, DIMENSION(ims:ime) :: DXCOL
  REAL, DIMENSION(its:ite) :: UST_IN, MOL_IN, ZNT_IN, QSFC_IN, HFX_IN, QFX_IN
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

  CALL sfclayinit(.TRUE.)
  CALL build_case(case_id, regime_name, itimestep, isfflx, DX)
  DO i = its, ite
     DXCOL(i) = DX
     UST_IN(i)=UST(i,1); MOL_IN(i)=MOL(i,1); ZNT_IN(i)=ZNT(i,1)
     QSFC_IN(i)=QSFC(i,1); HFX_IN(i)=HFX(i,1); QFX_IN(i)=QFX(i,1)
  END DO

  CALL sfclay(U3D,V3D,T3D,QV3D,P3D,DZ8W,                       &
       CP,G,ROVCP,R,XLV,PSFC,CHS,CHS2,CQS2,CPM,                &
       ZNT,UST,PBLH,MAVAIL,ZOL,MOL,REGIME,PSIM,PSIH,           &
       FM,FH,                                                  &
       XLAND,HFX,QFX,LH,TSK,FLHC,FLQC,QGH,QSFC,RMOL,           &
       U10,V10,TH2,T2,Q2,                                      &
       GZ1OZ0,WSPD,BR,isfflx,DXCOL,                            &
       SVP1,SVP2,SVP3,SVPT0,EP1,EP2,                           &
       KARMAN,EOMEG,STBOLT,                                    &
       P1000MB,LAKEMASK,                                       &
       ids,ide, jds,jde, kds,kde,                              &
       ims,ime, jms,jme, kms,kme,                              &
       its,ite, jts,jte, kts,kte)

  WRITE(*,'(A,I0)') 'CASE=', case_id
  WRITE(*,'(A,A)')  'REGIME=', TRIM(regime_name)
  WRITE(*,'(A,A)')  'PRECISION_MODE=', TRIM(precision_mode)
  WRITE(*,'(A,I0)') 'N=', N
  WRITE(*,'(A,I0)') 'FULL_WRF_EXE=', 0
  WRITE(*,'(A,I0)') 'ISFFLX=', isfflx
  WRITE(*,'(A,I0)') 'ITIMESTEP=', itimestep

  DO i = its, ite
     ! inputs
     CALL dump('U', i, U3D(i,1,1))
     CALL dump('V', i, V3D(i,1,1))
     CALL dump('T', i, T3D(i,1,1))
     CALL dump('QV', i, QV3D(i,1,1))
     CALL dump('P', i, P3D(i,1,1))
     CALL dump('DZ', i, DZ8W(i,1,1))
     CALL dump('PSFC', i, PSFC(i,1))
     CALL dump('TSK', i, TSK(i,1))
     CALL dump('XLAND', i, XLAND(i,1))
     CALL dump('LAKEMASK', i, LAKEMASK(i,1))
     CALL dump('MAVAIL', i, MAVAIL(i,1))
     CALL dump('PBLH', i, PBLH(i,1))
     CALL dump('DX', i, DX)
     CALL dump('UST_IN', i, UST_IN(i))
     CALL dump('MOL_IN', i, MOL_IN(i))
     CALL dump('ZNT_IN', i, ZNT_IN(i))
     CALL dump('QSFC_IN', i, QSFC_IN(i))
     CALL dump('HFX_IN', i, HFX_IN(i))
     CALL dump('QFX_IN', i, QFX_IN(i))
     ! outputs
     CALL dump('UST', i, UST(i,1))
     CALL dump('ZNT', i, ZNT(i,1))
     CALL dump('MOL', i, MOL(i,1))
     CALL dump('RMOL', i, RMOL(i,1))
     CALL dump('REGIME', i, REGIME(i,1))
     CALL dump('PSIM', i, PSIM(i,1))
     CALL dump('PSIH', i, PSIH(i,1))
     CALL dump('FM', i, FM(i,1))
     CALL dump('FH', i, FH(i,1))
     CALL dump('QSFC', i, QSFC(i,1))
     CALL dump('QGH', i, QGH(i,1))
     CALL dump('CPM', i, CPM(i,1))
     CALL dump('HFX', i, HFX(i,1))
     CALL dump('QFX', i, QFX(i,1))
     CALL dump('LH', i, LH(i,1))
     CALL dump('FLHC', i, FLHC(i,1))
     CALL dump('FLQC', i, FLQC(i,1))
     CALL dump('CHS', i, CHS(i,1))
     CALL dump('CHS2', i, CHS2(i,1))
     CALL dump('CQS2', i, CQS2(i,1))
     CALL dump('U10', i, U10(i,1))
     CALL dump('V10', i, V10(i,1))
     CALL dump('TH2', i, TH2(i,1))
     CALL dump('T2', i, T2(i,1))
     CALL dump('Q2', i, Q2(i,1))
     CALL dump('BR', i, BR(i,1))
     CALL dump('WSPD', i, WSPD(i,1))
     CALL dump('GZ1OZ0', i, GZ1OZ0(i,1))
     CALL dump('ZOL', i, ZOL(i,1))
  END DO

CONTAINS

  SUBROUTINE dump(name, idx, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', value
  END SUBROUTINE dump

  SUBROUTINE init_common()
    U3D=0.0; V3D=0.0; T3D=0.0; QV3D=0.0; P3D=0.0; DZ8W=0.0
    PSFC=0.0; CHS=0.0; CHS2=0.0; CQS2=0.0; CPM=CP
    ZNT=0.1; UST=0.1; PBLH=1000.0; MAVAIL=1.0; ZOL=0.0; MOL=0.0
    REGIME=0.0; PSIM=0.0; PSIH=0.0; FM=0.0; FH=0.0; XLAND=1.0
    HFX=0.0; QFX=0.0; LH=0.0; TSK=0.0; FLHC=0.0; FLQC=0.0; QGH=0.0
    QSFC=0.0; RMOL=0.0; U10=0.0; V10=0.0; TH2=0.0; T2=0.0; Q2=0.0
    GZ1OZ0=0.0; WSPD=0.0; BR=0.0; LAKEMASK=0.0
  END SUBROUTINE init_common

  SUBROUTINE setcol(i,uval,vval,tval,qvval,pval,dzval,tskval,xlandval,zntval, &
                    ustval,molval,mavailval,pblhval,qsfcval,hfxval,qfxval)
    INTEGER, INTENT(IN) :: i
    REAL, INTENT(IN) :: uval,vval,tval,qvval,pval,dzval,tskval,xlandval,zntval
    REAL, INTENT(IN) :: ustval,molval,mavailval,pblhval,qsfcval,hfxval,qfxval
    U3D(i,1,1)=uval; V3D(i,1,1)=vval; T3D(i,1,1)=tval; QV3D(i,1,1)=qvval
    P3D(i,1,1)=pval; DZ8W(i,1,1)=dzval
    PSFC(i,1)=pval - 450.0
    TSK(i,1)=tskval; XLAND(i,1)=xlandval; ZNT(i,1)=zntval; UST(i,1)=ustval
    MOL(i,1)=molval; MAVAIL(i,1)=mavailval; PBLH(i,1)=pblhval
    QSFC(i,1)=qsfcval; HFX(i,1)=hfxval; QFX(i,1)=qfxval
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
    ! setcol args:  i, u, v, t, qv, p, dz, tsk, xland, znt, ust, mol,
    !               mavail, pblh, qsfc, hfx, qfx   (qsfc=-1.0 -> recomputed)
    SELECT CASE (cid)
    CASE (1)
       name = 'unstable_convective_land_water'
       CALL setcol(1, 6.0,  1.0, 292.0, 0.0080, 95000.0, 70.0, 298.0, 1.0, 0.0800, 0.10, -1.0, 0.70, 1200.0, -1.0, 0.0, 0.0)
       CALL setcol(2, 8.0, -1.5, 294.0, 0.0100, 99800.0, 58.0, 297.0, 2.0, 0.0016, 0.12, -1.0, 1.00,  900.0, -1.0, 0.0, 0.0)
       CALL setcol(3, 3.0,  0.5, 289.0, 0.0065, 92000.0, 80.0, 295.0, 1.0, 0.1200, 0.05, -1.0, 0.55, 1500.0, -1.0, 0.0, 0.0)
       CALL setcol(4, 5.0,  4.0, 291.5, 0.0095,101000.0, 54.0, 294.5, 2.0, 0.0025, 0.20, -1.0, 1.00,  750.0, -1.0, 0.0, 0.0)
       CALL setcol(5,10.0,  0.2, 300.0, 0.0120, 96000.0, 66.0, 305.0, 1.0, 0.0400, 0.18, -1.0, 0.80, 1800.0, -1.0, 0.0, 0.0)
       CALL setcol(6, 4.5, -3.0, 296.0, 0.0110,100500.0, 60.0, 299.0, 2.0, 0.0012, 0.15, -1.0, 1.00,  650.0, -1.0, 0.0, 0.0)
       CALL setcol(7, 7.5,  2.0, 285.0, 0.0040, 88000.0, 90.0, 291.0, 1.0, 0.2000, 0.08, -1.0, 0.60, 1100.0, -1.0, 0.0, 0.0)
       CALL setcol(8, 9.0,  1.0, 293.0, 0.0090, 99000.0, 55.0, 296.0, 2.0, 0.0018, 0.11, -1.0, 1.00,  800.0, -1.0, 0.0, 0.0)
    CASE (2)
       name = 'stable_nocturnal_land_water'
       CALL setcol(1, 3.5,  0.4, 292.0, 0.0070, 95000.0, 70.0, 286.0, 1.0, 0.0800, 0.05, -1.0, 0.65, 350.0, -1.0, 0.0, 0.0)
       CALL setcol(2, 6.0, -1.2, 295.0, 0.0100,100000.0, 58.0, 291.0, 2.0, 0.0016, 0.05, -1.0, 1.00, 300.0, -1.0, 0.0, 0.0)
       CALL setcol(3, 2.0,  0.2, 289.0, 0.0055, 92000.0, 80.0, 283.5, 1.0, 0.1200, 0.04, -1.0, 0.45, 250.0, -1.0, 0.0, 0.0)
       CALL setcol(4, 4.5,  2.0, 294.0, 0.0090,101000.0, 54.0, 290.0, 2.0, 0.0025, 0.05, -1.0, 1.00, 280.0, -1.0, 0.0, 0.0)
       CALL setcol(5, 8.0,  0.3, 299.0, 0.0110, 96000.0, 66.0, 294.0, 1.0, 0.0400, 0.06, -1.0, 0.75, 400.0, -1.0, 0.0, 0.0)
       CALL setcol(6, 2.8, -2.5, 296.0, 0.0100,100500.0, 60.0, 292.0, 2.0, 0.0012, 0.05, -1.0, 1.00, 260.0, -1.0, 0.0, 0.0)
       CALL setcol(7, 5.0,  1.4, 285.0, 0.0040, 88000.0, 90.0, 280.0, 1.0, 0.2000, 0.04, -1.0, 0.55, 220.0, -1.0, 0.0, 0.0)
       CALL setcol(8, 7.0,  0.8, 293.0, 0.0090, 99000.0, 55.0, 289.5, 2.0, 0.0018, 0.05, -1.0, 1.00, 300.0, -1.0, 0.0, 0.0)
    CASE (3)
       name = 'near_neutral_persisted_qsfc'
       CALL setcol(1, 5.0, 1.0, 292.0, 0.0080, 95000.0, 70.0, 291.60, 1.0, 0.0800, 0.10, 0.0, 0.70, 800.0, 0.0080, 0.0, 0.0)
       CALL setcol(2, 7.0, 0.0, 294.0, 0.0100,100000.0, 58.0, 292.80, 2.0, 0.0016, 0.12, 0.0, 1.00, 800.0, 0.0100, 0.0, 0.0)
       CALL setcol(3, 3.5, 0.8, 289.0, 0.0060, 92000.0, 80.0, 288.59, 1.0, 0.1200, 0.07, 0.0, 0.60, 800.0, 0.0060, 0.0, 0.0)
       CALL setcol(4, 6.0, 2.0, 296.0, 0.0110,101000.0, 54.0, 294.67, 2.0, 0.0025, 0.11, 0.0, 1.00, 800.0, 0.0110, 0.0, 0.0)
       CALL setcol(5, 8.0, 1.0, 300.0, 0.0120, 96000.0, 66.0, 299.59, 1.0, 0.0400, 0.16, 0.0, 0.80, 800.0, 0.0120, 0.0, 0.0)
       CALL setcol(6, 4.0,-1.0, 296.0, 0.0100,100500.0, 60.0, 294.51, 2.0, 0.0012, 0.10, 0.0, 1.00, 800.0, 0.0100, 0.0, 0.0)
       CALL setcol(7, 5.5, 1.5, 285.0, 0.0040, 88000.0, 90.0, 284.58, 1.0, 0.2000, 0.08, 0.0, 0.55, 800.0, 0.0040, 0.0, 0.0)
       CALL setcol(8, 7.0, 0.5, 293.0, 0.0090, 99000.0, 55.0, 291.76, 2.0, 0.0018, 0.10, 0.0, 1.00, 800.0, 0.0090, 0.0, 0.0)
    CASE (4)
       name = 'very_stable_low_wind'
       l_dx = 9000.0
       CALL setcol(1, 0.6, 0.1, 292.0, 0.0070, 95000.0, 70.0, 274.0, 1.0, 0.0800, 0.05, 0.30, 0.70, 250.0, -1.0, 0.0, 0.0)
       CALL setcol(2, 0.8, 0.1, 295.0, 0.0100,100000.0, 58.0, 280.0, 2.0, 0.0016, 0.05, 0.30, 1.00, 250.0, -1.0, 0.0, 0.0)
       CALL setcol(3, 0.5, 0.0, 289.0, 0.0060, 92000.0, 80.0, 270.5, 1.0, 0.1200, 0.05, 0.30, 0.50, 220.0, -1.0, 0.0, 0.0)
       CALL setcol(4, 0.9, 0.2, 294.0, 0.0090,101000.0, 54.0, 281.0, 2.0, 0.0025, 0.05, 0.30, 1.00, 230.0, -1.0, 0.0, 0.0)
       CALL setcol(5, 1.0, 0.1, 299.0, 0.0110, 96000.0, 66.0, 284.0, 1.0, 0.0400, 0.05, 0.30, 0.75, 250.0, -1.0, 0.0, 0.0)
       CALL setcol(6, 0.7,-0.2, 296.0, 0.0100,100500.0, 60.0, 282.0, 2.0, 0.0012, 0.05, 0.30, 1.00, 220.0, -1.0, 0.0, 0.0)
       CALL setcol(7, 0.6, 0.2, 285.0, 0.0040, 88000.0, 90.0, 268.0, 1.0, 0.2000, 0.05, 0.30, 0.55, 220.0, -1.0, 0.0, 0.0)
       CALL setcol(8, 1.1, 0.1, 293.0, 0.0090, 99000.0, 55.0, 281.0, 2.0, 0.0018, 0.05, 0.30, 1.00, 220.0, -1.0, 0.0, 0.0)
    CASE DEFAULT
       name = 'convective_with_prev_fluxes'
       l_dx = 12000.0
       CALL setcol(1, 6.0,1.0,292.0,0.0080, 95000.0,70.0,300.0,1.0,0.0800,0.20,-0.2,0.70,1200.0,-1.0, 150.0, 0.0001)
       CALL setcol(2, 6.0,0.0,295.0,0.0100,100000.0,58.0,298.0,2.0,0.0016,0.15,-0.1,1.00, 900.0,-1.0,  80.0, 0.00008)
       CALL setcol(3, 4.0,1.0,289.0,0.0060, 92000.0,80.0,296.0,1.0,0.1200,0.10,-0.2,0.60,1500.0,-1.0, 120.0, 0.00009)
       CALL setcol(4, 5.0,1.0,294.0,0.0090,101000.0,54.0,297.0,2.0,0.0025,0.15,-0.1,1.00, 750.0,-1.0,  60.0, 0.00007)
       CALL setcol(5, 8.0,1.0,300.0,0.0120, 96000.0,66.0,307.0,1.0,0.0400,0.20,-0.3,0.80,1800.0,-1.0, 200.0, 0.00012)
       CALL setcol(6, 4.0,1.0,296.0,0.0100,100500.0,60.0,300.0,2.0,0.0012,0.15,-0.1,1.00, 650.0,-1.0,  40.0, 0.00006)
       CALL setcol(7, 5.0,1.0,285.0,0.0040, 88000.0,90.0,293.0,1.0,0.2000,0.12,-0.2,0.60,1100.0,-1.0,  90.0, 0.00005)
       CALL setcol(8, 7.0,1.0,293.0,0.0090, 99000.0,55.0,298.0,2.0,0.0018,0.15,-0.1,1.00, 800.0,-1.0,  70.0, 0.00008)
    END SELECT
  END SUBROUTINE build_case

END PROGRAM sfclay_old_mm5_oracle
