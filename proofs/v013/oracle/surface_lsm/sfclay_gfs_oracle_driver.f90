PROGRAM sfclay_gfs_oracle
  ! Single-column oracle for the WRF NCEP-GFS surface layer
  ! (UNMODIFIED phys/module_sf_gfs.F, SUBROUTINE SF_GFS). Emits gold savepoints
  ! (inputs + outputs) for sfclay_gfs_dump_to_json.py.
  USE module_sf_gfs, ONLY : sf_gfs
  IMPLICIT NONE

  REAL, PARAMETER :: CP = 1004.0
  REAL, PARAMETER :: R  = 287.0
  REAL, PARAMETER :: ROVCP = R / CP
  REAL, PARAMETER :: XLV = 2.5E6
  REAL, PARAMETER :: EP1 = 0.608
  REAL, PARAMETER :: EP2 = 0.622
  REAL, PARAMETER :: KARMAN = 0.40

  INTEGER, PARAMETER :: N = 8
  INTEGER, PARAMETER :: ids=1, ide=N+1, jds=1, jde=2, kds=1, kde=2
  INTEGER, PARAMETER :: ims=1, ime=N,   jms=1, jme=1, kms=1, kme=1
  INTEGER, PARAMETER :: its=1, ite=N,   jts=1, jte=1, kts=1, kte=1

  REAL, DIMENSION(ims:ime,kms:kme,jms:jme) :: U3D,V3D,T3D,QV3D,P3D
  REAL, DIMENSION(ims:ime,jms:jme) :: PSFC,CHS,CHS2,CQS2,CPM,ZNT,UST,PSIM,PSIH
  REAL, DIMENSION(ims:ime,jms:jme) :: XLAND,HFX,QFX,LH,TSK,FLHC,FLQC,QGH,QSFC
  REAL, DIMENSION(ims:ime,jms:jme) :: U10,V10,GZ1OZ0,WSPD,BR
  REAL, DIMENSION(its:ite) :: UST_IN, ZNT_IN
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

  CALL build_case(case_id, regime_name, itimestep, isfflx)
  DO i = its, ite
     UST_IN(i)=UST(i,1); ZNT_IN(i)=ZNT(i,1)
  END DO

  CALL sf_gfs(U3D,V3D,T3D,QV3D,P3D,                            &
       CP,ROVCP,R,XLV,PSFC,CHS,CHS2,CQS2,CPM,                  &
       ZNT,UST,PSIM,PSIH,                                      &
       XLAND,HFX,QFX,LH,TSK,FLHC,FLQC,                         &
       QGH,QSFC,U10,V10,                                       &
       GZ1OZ0,WSPD,BR,isfflx,                                  &
       EP1,EP2,KARMAN,itimestep,                               &
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
     CALL dump('U', i, U3D(i,1,1))
     CALL dump('V', i, V3D(i,1,1))
     CALL dump('T', i, T3D(i,1,1))
     CALL dump('QV', i, QV3D(i,1,1))
     CALL dump('P', i, P3D(i,1,1))
     CALL dump('PSFC', i, PSFC(i,1))
     CALL dump('TSK', i, TSK(i,1))
     CALL dump('XLAND', i, XLAND(i,1))
     CALL dump('UST_IN', i, UST_IN(i))
     CALL dump('ZNT_IN', i, ZNT_IN(i))
     ! outputs
     CALL dump('UST', i, UST(i,1))
     CALL dump('ZNT', i, ZNT(i,1))
     CALL dump('CHS', i, CHS(i,1))
     CALL dump('CHS2', i, CHS2(i,1))
     CALL dump('CQS2', i, CQS2(i,1))
     CALL dump('CPM', i, CPM(i,1))
     CALL dump('PSIM', i, PSIM(i,1))
     CALL dump('PSIH', i, PSIH(i,1))
     CALL dump('QGH', i, QGH(i,1))
     CALL dump('QSFC', i, QSFC(i,1))
     CALL dump('U10', i, U10(i,1))
     CALL dump('V10', i, V10(i,1))
     CALL dump('WSPD', i, WSPD(i,1))
     CALL dump('BR', i, BR(i,1))
     CALL dump('GZ1OZ0', i, GZ1OZ0(i,1))
     CALL dump('FLHC', i, FLHC(i,1))
     CALL dump('FLQC', i, FLQC(i,1))
     CALL dump('HFX', i, HFX(i,1))
     CALL dump('QFX', i, QFX(i,1))
     CALL dump('LH', i, LH(i,1))
  END DO

CONTAINS

  SUBROUTINE dump(name, idx, value)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER, INTENT(IN) :: idx
    REAL, INTENT(IN) :: value
    WRITE(*,'(A,A,I0,A,ES23.15)') TRIM(name),'[',idx,']=', value
  END SUBROUTINE dump

  SUBROUTINE init_common()
    U3D=0.0; V3D=0.0; T3D=0.0; QV3D=0.0; P3D=0.0
    PSFC=0.0; CHS=0.0; CHS2=0.0; CQS2=0.0; CPM=CP
    ZNT=0.1; UST=0.1; PSIM=0.0; PSIH=0.0; XLAND=1.0
    HFX=0.0; QFX=0.0; LH=0.0; TSK=288.0; FLHC=0.0; FLQC=0.0; QGH=0.0; QSFC=0.0
    U10=0.0; V10=0.0; GZ1OZ0=0.0; WSPD=0.0; BR=0.0
  END SUBROUTINE init_common

  SUBROUTINE setcol(i,uval,vval,tval,qvval,pval,tskval,xlandval,zntval,ustval)
    INTEGER, INTENT(IN) :: i
    REAL, INTENT(IN) :: uval,vval,tval,qvval,pval,tskval,xlandval,zntval,ustval
    U3D(i,1,1)=uval; V3D(i,1,1)=vval; T3D(i,1,1)=tval; QV3D(i,1,1)=qvval
    P3D(i,1,1)=pval
    PSFC(i,1)=pval + 450.0
    TSK(i,1)=tskval; XLAND(i,1)=xlandval; ZNT(i,1)=zntval; UST(i,1)=ustval
  END SUBROUTINE setcol

  SUBROUTINE build_case(cid, name, l_itimestep, l_isfflx)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(OUT) :: name
    INTEGER, INTENT(OUT) :: l_itimestep, l_isfflx
    CALL init_common()
    l_isfflx = 1
    l_itimestep = 2
    ! setcol args: i, u, v, t, qv, p, tsk, xland, znt, ust
    SELECT CASE (cid)
    CASE (1)
       name = 'unstable_convective_land_water'
       CALL setcol(1, 6.0,  1.0, 292.0, 0.0080, 95000.0, 298.0, 1.0, 0.0800, 0.30)
       CALL setcol(2, 8.0, -1.5, 294.0, 0.0100, 99800.0, 297.0, 2.0, 0.0016, 0.30)
       CALL setcol(3, 3.0,  0.5, 289.0, 0.0065, 92000.0, 295.0, 1.0, 0.1200, 0.30)
       CALL setcol(4, 5.0,  4.0, 291.5, 0.0095,101000.0, 294.5, 2.0, 0.0025, 0.30)
       CALL setcol(5,10.0,  0.2, 300.0, 0.0120, 96000.0, 305.0, 1.0, 0.0400, 0.30)
       CALL setcol(6, 4.5, -3.0, 296.0, 0.0110,100500.0, 299.0, 2.0, 0.0012, 0.30)
       CALL setcol(7, 7.5,  2.0, 285.0, 0.0040, 88000.0, 291.0, 1.0, 0.2000, 0.30)
       CALL setcol(8, 9.0,  1.0, 293.0, 0.0090, 99000.0, 296.0, 2.0, 0.0018, 0.30)
    CASE (2)
       name = 'stable_nocturnal_land_water'
       CALL setcol(1, 3.5,  0.4, 292.0, 0.0070, 95000.0, 286.0, 1.0, 0.0800, 0.20)
       CALL setcol(2, 6.0, -1.2, 295.0, 0.0100,100000.0, 291.0, 2.0, 0.0016, 0.20)
       CALL setcol(3, 2.0,  0.2, 289.0, 0.0055, 92000.0, 283.5, 1.0, 0.1200, 0.20)
       CALL setcol(4, 4.5,  2.0, 294.0, 0.0090,101000.0, 290.0, 2.0, 0.0025, 0.20)
       CALL setcol(5, 8.0,  0.3, 299.0, 0.0110, 96000.0, 294.0, 1.0, 0.0400, 0.20)
       CALL setcol(6, 2.8, -2.5, 296.0, 0.0100,100500.0, 292.0, 2.0, 0.0012, 0.20)
       CALL setcol(7, 5.0,  1.4, 285.0, 0.0040, 88000.0, 280.0, 1.0, 0.2000, 0.20)
       CALL setcol(8, 7.0,  0.8, 293.0, 0.0090, 99000.0, 289.5, 2.0, 0.0018, 0.20)
    CASE (3)
       name = 'near_neutral_land_water'
       CALL setcol(1, 5.0, 1.0, 292.0, 0.0080, 95000.0, 291.6, 1.0, 0.0800, 0.25)
       CALL setcol(2, 7.0, 0.0, 294.0, 0.0100,100000.0, 293.9, 2.0, 0.0016, 0.25)
       CALL setcol(3, 3.5, 0.8, 289.0, 0.0060, 92000.0, 288.6, 1.0, 0.1200, 0.25)
       CALL setcol(4, 6.0, 2.0, 296.0, 0.0110,101000.0, 295.7, 2.0, 0.0025, 0.25)
       CALL setcol(5, 8.0, 1.0, 300.0, 0.0120, 96000.0, 299.6, 1.0, 0.0400, 0.25)
       CALL setcol(6, 4.0,-1.0, 296.0, 0.0100,100500.0, 295.5, 2.0, 0.0012, 0.25)
       CALL setcol(7, 5.5, 1.5, 285.0, 0.0040, 88000.0, 284.6, 1.0, 0.2000, 0.25)
       CALL setcol(8, 7.0, 0.5, 293.0, 0.0090, 99000.0, 292.8, 2.0, 0.0018, 0.25)
    CASE (4)
       name = 'cold_ice_saturation'
       CALL setcol(1, 3.0, 0.5, 268.0, 0.0020, 91000.0, 266.0, 1.0, 0.0800, 0.20)
       CALL setcol(2, 4.0, 0.3, 272.0, 0.0030, 98000.0, 269.0, 2.0, 0.0016, 0.20)
       CALL setcol(3, 2.0, 0.2, 265.0, 0.0015, 88000.0, 263.0, 1.0, 0.1200, 0.20)
       CALL setcol(4, 5.0, 0.5, 275.0, 0.0035,101000.0, 270.0, 2.0, 0.0025, 0.20)
       CALL setcol(5, 4.5, 0.4, 270.0, 0.0025, 94000.0, 267.0, 1.0, 0.0400, 0.20)
       CALL setcol(6, 3.5, 0.3, 273.0, 0.0030,100500.0, 271.0, 2.0, 0.0012, 0.20)
       CALL setcol(7, 2.5, 0.2, 262.0, 0.0010, 85000.0, 260.0, 1.0, 0.2000, 0.20)
       CALL setcol(8, 4.0, 0.2, 271.0, 0.0028, 99000.0, 268.0, 2.0, 0.0018, 0.20)
    CASE DEFAULT
       name = 'strong_unstable_high_wind'
       CALL setcol(1, 12.0, 2.0, 295.0, 0.0090, 95000.0, 303.0, 1.0, 0.0800, 0.50)
       CALL setcol(2, 14.0,-2.0, 297.0, 0.0110, 99800.0, 301.0, 2.0, 0.0016, 0.60)
       CALL setcol(3, 11.0, 1.0, 292.0, 0.0075, 92000.0, 300.0, 1.0, 0.1200, 0.45)
       CALL setcol(4, 13.0, 3.0, 296.0, 0.0100,101000.0, 302.0, 2.0, 0.0025, 0.55)
       CALL setcol(5, 16.0, 1.0, 301.0, 0.0130, 96000.0, 308.0, 1.0, 0.0400, 0.70)
       CALL setcol(6, 10.0,-2.0, 298.0, 0.0115,100500.0, 303.0, 2.0, 0.0012, 0.50)
       CALL setcol(7, 12.0, 2.0, 288.0, 0.0050, 88000.0, 296.0, 1.0, 0.2000, 0.48)
       CALL setcol(8, 15.0, 1.0, 295.0, 0.0095, 99000.0, 302.0, 2.0, 0.0018, 0.62)
    END SELECT
  END SUBROUTINE build_case

END PROGRAM sfclay_gfs_oracle
