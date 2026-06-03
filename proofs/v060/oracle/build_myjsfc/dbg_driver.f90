! =====================================================================
! v0.6.0 single-column Janjic (MYJ) surface-layer oracle driver.
!
! Drives the UNMODIFIED WRF module_sf_myjsfc.F (MYJSFCINIT to build the
! PSIM/PSIH lookup tables, then MYJSFC) on prescribed single-column
! soundings, then dumps the surface-layer exchange coefficients, fluxes,
! and surface-roughness for JAX savepoint parity. This is a real
! WRF-module oracle, not a JAX self-compare; not a full wrf.exe run.
!
! sf_sfclay_physics=2 MUST pair with bl_pbl_physics=2 (Janjic Eta sfc +
! MYJ PBL). This driver exercises the surface-layer half of that pair.
! =====================================================================
PROGRAM myjsfc_oracle_dbg
  USE MODULE_SF_MYJSFC_DBG, ONLY : MYJSFC, MYJSFCINIT
  IMPLICIT NONE

  REAL, PARAMETER :: G      = 9.81
  REAL, PARAMETER :: R_D    = 287.0
  REAL, PARAMETER :: CP     = 7.0*R_D/2.0
  REAL, PARAMETER :: R_V    = 461.6
  REAL, PARAMETER :: ROVCP  = R_D/CP
  REAL, PARAMETER :: EP1    = R_V/R_D - 1.0
  REAL, PARAMETER :: P1000  = 1.0E5

  INTEGER, PARAMETER :: KX = 32
  INTEGER, PARAMETER :: ids=1, ide=2, jds=1, jde=2, kds=1, kde=KX+1
  INTEGER, PARAMETER :: ims=1, ime=1, jms=1, jme=1, kms=1, kme=KX+1
  INTEGER, PARAMETER :: its=1, ite=1, jts=1, jte=1, kts=1, kte=KX

  REAL,DIMENSION(ims:ime,kms:kme,jms:jme) :: U,V,T,TH,QV,QC,PMID,PINT,DZ,Q2
  REAL,DIMENSION(ims:ime,jms:jme) :: HT,MAVAIL,TSK,XLAND,Z0BASE
  REAL,DIMENSION(ims:ime,jms:jme) :: QSFC,THZ0,QZ0,UZ0,VZ0
  REAL,DIMENSION(ims:ime,jms:jme) :: USTAR,ZNT,PBLH,RMOL,AKHS,AKMS,RIB
  REAL,DIMENSION(ims:ime,jms:jme) :: CHS,CHS2,CQS2,HFX,QFX,FLX_LH,FLHC,FLQC
  REAL,DIMENSION(ims:ime,jms:jme) :: QGH,CPM,CT
  REAL,DIMENSION(ims:ime,jms:jme) :: U10,V10,T02,TH02,TSHLTR,TH10,Q02,QSHLTR,Q10,PSHLTR
  REAL,DIMENSION(ims:ime,jms:jme) :: U10E,V10E
  REAL,DIMENSION(ims:ime,jms:jme) :: SEAMASKD,XICE
  INTEGER,DIMENSION(ims:ime,jms:jme) :: LOWLYR,IVGTYP
  INTEGER :: ISURBAN, IZ0TLND, ITIMESTEP, case_id
  CHARACTER(LEN=256) :: regime
  CHARACTER(LEN=32) :: arg
  LOGICAL :: restart, allowed_to_read

  IF (COMMAND_ARGUMENT_COUNT() >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg,*) case_id
  ELSE
    case_id = 1
  END IF

  CALL build_case(case_id, regime, U, V, T, TH, QV, QC, PMID, PINT, DZ, Q2, &
                  HT, MAVAIL, TSK, XLAND, Z0BASE, QSFC, THZ0, QZ0, UZ0, VZ0, &
                  USTAR, ZNT, IVGTYP)

  ISURBAN = 1
  IZ0TLND = 0
  ITIMESTEP = 2          ! second-step path (NTSD>1) exercises the full surface logic
  restart = .FALSE.
  allowed_to_read = .FALSE.
  LOWLYR = 1
  RMOL = 0.0
  AKHS = 0.0; AKMS = 0.0
  PBLH = -1.0
  CT = 0.0
  SEAMASKD = XLAND
  XICE = 0.0

  CALL MYJSFCINIT(LOWLYR,USTAR,ZNT,SEAMASKD,XICE,IVGTYP,restart, &
                  allowed_to_read, &
                  ids,ide,jds,jde,kds,kde, &
                  ims,ime,jms,jme,kms,kme, &
                  its,ite,jts,jte,kts,kte)

  CALL MYJSFC(ITIMESTEP,HT,DZ, &
              PMID,PINT,TH,T,QV,QC,U,V,Q2, &
              TSK,QSFC,THZ0,QZ0,UZ0,VZ0, &
              LOWLYR,XLAND,IVGTYP,ISURBAN,IZ0TLND, &
              USTAR,ZNT,Z0BASE,PBLH,MAVAIL,RMOL, &
              AKHS,AKMS, &
              RIB, &
              CHS,CHS2,CQS2,HFX,QFX,FLX_LH,FLHC,FLQC, &
              QGH,CPM,CT, &
              U10,V10,T02,TH02,TSHLTR,TH10,Q02,QSHLTR,Q10,PSHLTR, &
              P1000,U10E,V10E, &
              ids,ide,jds,jde,kds,kde, &
              ims,ime,jms,jme,kms,kme, &
              its,ite,jts,jte,kts,kte)

  CALL dump_case(case_id, regime, U, V, T, TH, QV, QC, PMID, PINT, DZ, Q2, &
                 HT, MAVAIL, TSK, XLAND, Z0BASE, QSFC, THZ0, QZ0, UZ0, VZ0, &
                 USTAR, ZNT, PBLH, RMOL, AKHS, AKMS, RIB, &
                 CHS, CHS2, CQS2, HFX, QFX, FLX_LH, FLHC, FLQC, QGH, CPM, CT, &
                 U10, V10, T02, TH02, TSHLTR, TH10, Q02, QSHLTR, Q10, PSHLTR, &
                 U10E, V10E, IVGTYP, ITIMESTEP)

CONTAINS

  SUBROUTINE build_case(cid, name, Uu, Vv, Tt, Tht, Qq, Qcw, Pp, Pii, Dzz, Q2a, &
                        Htt, Mav, Tsk_a, Xl, Z0b, Qsf, Thz, Qz, Uz, Vz, &
                        Ust_a, Znt_a, Ivg)
    INTEGER, INTENT(IN) :: cid
    CHARACTER(LEN=*), INTENT(OUT) :: name
    REAL,DIMENSION(ims:ime,kms:kme,jms:jme),INTENT(OUT) :: Uu,Vv,Tt,Tht,Qq,Qcw,Pp,Pii,Dzz,Q2a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(OUT) :: Htt,Mav,Tsk_a,Xl,Z0b,Qsf,Thz,Qz,Uz,Vz
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(OUT) :: Ust_a,Znt_a
    INTEGER,DIMENSION(ims:ime,jms:jme),INTENT(OUT) :: Ivg
    REAL,DIMENSION(KX+1) :: zint, pint
    REAL,DIMENSION(KX) :: zmid, theta, qprof, uprof, vprof, temp, pfull, q2prof
    REAL :: psfc0, ztop, zml, theta0, lapse_ml, lapse_ft, q0, qscale, shear, z, tv, pi_mass
    REAL :: tsk0, ust0, znt0, xland0, mav0, q2sfc, dtheta_sfc
    INTEGER :: k, ivgtyp0

    ztop = 12000.0
    zint(1) = 0.0
    DO k = 1, KX
      zint(k+1) = ztop * (REAL(k)/REAL(KX))**1.18
      zmid(k) = 0.5*(zint(k)+zint(k+1))
    END DO

    SELECT CASE (cid)
    CASE (1)
      name='unstable_daytime_land'
      psfc0=100000.0; theta0=300.0; zml=1100.0; lapse_ml=0.0003; lapse_ft=0.0045
      q0=0.0140; qscale=2300.0; shear=0.0015; tsk0=305.0; dtheta_sfc=5.0
      ust0=0.45; znt0=0.10; xland0=1.0; mav0=0.6; q2sfc=0.6; ivgtyp0=10
    CASE (2)
      name='strong_unstable_land'
      psfc0=100800.0; theta0=302.0; zml=1400.0; lapse_ml=0.0002; lapse_ft=0.0038
      q0=0.0160; qscale=2600.0; shear=0.0020; tsk0=310.0; dtheta_sfc=8.0
      ust0=0.60; znt0=0.08; xland0=1.0; mav0=0.5; q2sfc=1.0; ivgtyp0=7
    CASE (3)
      name='stable_nocturnal_land'
      psfc0=100500.0; theta0=287.0; zml=150.0; lapse_ml=0.0100; lapse_ft=0.0060
      q0=0.0060; qscale=1600.0; shear=0.0060; tsk0=283.0; dtheta_sfc=-4.0
      ust0=0.20; znt0=0.05; xland0=1.0; mav0=0.4; q2sfc=0.05; ivgtyp0=10
    CASE (4)
      name='stable_marine'
      psfc0=101200.0; theta0=289.0; zml=250.0; lapse_ml=0.0040; lapse_ft=0.0032
      q0=0.0100; qscale=2600.0; shear=0.0010; tsk0=288.0; dtheta_sfc=-1.0
      ust0=0.18; znt0=0.001; xland0=2.0; mav0=1.0; q2sfc=0.1; ivgtyp0=16
    CASE (5)
      name='neutral_land'
      psfc0=100000.0; theta0=296.0; zml=800.0; lapse_ml=0.0010; lapse_ft=0.0030
      q0=0.0090; qscale=2200.0; shear=0.0025; tsk0=296.5; dtheta_sfc=0.5
      ust0=0.40; znt0=0.08; xland0=1.0; mav0=0.6; q2sfc=0.3; ivgtyp0=10
    CASE DEFAULT
      name='unstable_marine'
      psfc0=101000.0; theta0=293.0; zml=600.0; lapse_ml=0.0010; lapse_ft=0.0035
      q0=0.0150; qscale=2400.0; shear=0.0015; tsk0=296.0; dtheta_sfc=3.0
      ust0=0.25; znt0=0.0015; xland0=2.0; mav0=1.0; q2sfc=0.4; ivgtyp0=16
    END SELECT

    pint(1) = psfc0
    DO k = 1, KX
      z = zmid(k)
      IF (z <= zml) THEN
        theta(k) = theta0 + lapse_ml*z
      ELSE
        theta(k) = theta0 + lapse_ml*zml + lapse_ft*(z-zml)
      END IF
      qprof(k) = MAX(q0*EXP(-z/qscale), 1.0E-5)
      uprof(k) = 4.0 + shear*z
      vprof(k) = 1.0 + 0.25*shear*z
      pi_mass = (MAX(pint(k), 1000.0)/P1000)**ROVCP
      temp(k) = theta(k)*pi_mass
      tv = temp(k)*(1.0 + EP1*qprof(k))
      pint(k+1) = pint(k)*EXP(-G*(zint(k+1)-zint(k))/(R_D*MAX(tv, 180.0)))
      pfull(k) = 0.5*(pint(k)+pint(k+1))
      pi_mass = (pfull(k)/P1000)**ROVCP
      temp(k) = theta(k)*pi_mass
      ! TKE profile (m2 s-2): decays with height; lowest layer set by q2sfc
      q2prof(k) = MAX(q2sfc*EXP(-z/600.0), 1.0E-3)
    END DO

    Uu=0.; Vv=0.; Tt=0.; Tht=0.; Qq=0.; Qcw=0.; Pp=0.; Pii=0.; Dzz=0.; Q2a=0.
    DO k = 1, KX
      Uu(1,k,1) = uprof(k)
      Vv(1,k,1) = vprof(k)
      Tt(1,k,1) = temp(k)
      Tht(1,k,1) = theta(k)
      Qq(1,k,1) = qprof(k)
      Qcw(1,k,1) = 0.0
      Pp(1,k,1) = pfull(k)
      Pii(1,k,1) = pint(k)
      Dzz(1,k,1) = zint(k+1)-zint(k)
      Q2a(1,k,1) = q2prof(k)
    END DO
    Pii(1,KX+1,1) = pint(KX+1)

    Htt(1,1)=0.0
    Mav(1,1)=mav0
    Tsk_a(1,1)=theta0*( (pint(1)/P1000)**ROVCP ) + dtheta_sfc
    Xl(1,1)=xland0
    Z0b(1,1)=znt0
    Ust_a(1,1)=ust0
    Znt_a(1,1)=znt0
    Ivg(1,1)=ivgtyp0
    ! Surface-state INOUT seeds (consistent with a warm start, NTSD>1 path)
    Qsf(1,1)=qprof(1)
    Thz(1,1)=theta(1)
    Qz(1,1)=qprof(1)/(1.+qprof(1))
    Uz(1,1)=0.0
    Vz(1,1)=0.0
  END SUBROUTINE build_case

  SUBROUTINE dump_col(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col

  SUBROUTINE dump_col_interface(name, arr)
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL, DIMENSION(ims:ime,kms:kme,jms:jme), INTENT(IN) :: arr
    INTEGER :: kk
    DO kk = kts, kte+1
      WRITE(*,'(A,A,I0,A,ES23.15)') name,'[',kk,']=', arr(1,kk,1)
    END DO
  END SUBROUTINE dump_col_interface

  SUBROUTINE dump_case(cid, name, Uu, Vv, Tt, Tht, Qq, Qcw, Pp, Pii, Dzz, Q2a, &
                       Htt, Mav, Tsk_a, Xl, Z0b, Qsf, Thz, Qz, Uz, Vz, &
                       Ust_a, Znt_a, Pblh_a, Rmol_a, Akhs_a, Akms_a, Rib_a, &
                       Chs_a, Chs2_a, Cqs2_a, Hfx_a, Qfx_a, Flxlh_a, Flhc_a, Flqc_a, &
                       Qgh_a, Cpm_a, Ct_a, U10_a, V10_a, T02_a, Th02_a, Tshltr_a, &
                       Th10_a, Q02_a, Qshltr_a, Q10_a, Pshltr_a, U10e_a, V10e_a, &
                       Ivg, itime)
    INTEGER, INTENT(IN) :: cid, itime
    CHARACTER(LEN=*), INTENT(IN) :: name
    REAL,DIMENSION(ims:ime,kms:kme,jms:jme),INTENT(IN) :: Uu,Vv,Tt,Tht,Qq,Qcw,Pp,Pii,Dzz,Q2a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Htt,Mav,Tsk_a,Xl,Z0b,Qsf,Thz,Qz,Uz,Vz
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Ust_a,Znt_a,Pblh_a,Rmol_a,Akhs_a,Akms_a,Rib_a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Chs_a,Chs2_a,Cqs2_a,Hfx_a,Qfx_a,Flxlh_a,Flhc_a,Flqc_a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Qgh_a,Cpm_a,Ct_a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: U10_a,V10_a,T02_a,Th02_a,Tshltr_a,Th10_a
    REAL,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Q02_a,Qshltr_a,Q10_a,Pshltr_a,U10e_a,V10e_a
    INTEGER,DIMENSION(ims:ime,jms:jme),INTENT(IN) :: Ivg
    WRITE(*,'(A,I0)') 'CASE=', cid
    WRITE(*,'(A,A)') 'REGIME=', TRIM(name)
    WRITE(*,'(A,I0)') 'KX=', KX
    WRITE(*,'(A,I0)') 'ITIMESTEP=', itime
    WRITE(*,'(A,I0)') 'IVGTYP=', Ivg(1,1)
    WRITE(*,'(A,ES23.15)') 'HT=', Htt(1,1)
    WRITE(*,'(A,ES23.15)') 'MAVAIL=', Mav(1,1)
    WRITE(*,'(A,ES23.15)') 'TSK=', Tsk_a(1,1)
    WRITE(*,'(A,ES23.15)') 'XLAND=', Xl(1,1)
    WRITE(*,'(A,ES23.15)') 'Z0BASE=', Z0b(1,1)
    ! Surface-layer outputs/INOUT
    WRITE(*,'(A,ES23.15)') 'USTAR=', Ust_a(1,1)
    WRITE(*,'(A,ES23.15)') 'ZNT=', Znt_a(1,1)
    WRITE(*,'(A,ES23.15)') 'AKHS=', Akhs_a(1,1)
    WRITE(*,'(A,ES23.15)') 'AKMS=', Akms_a(1,1)
    WRITE(*,'(A,ES23.15)') 'RMOL=', Rmol_a(1,1)
    WRITE(*,'(A,ES23.15)') 'RIB=', Rib_a(1,1)
    WRITE(*,'(A,ES23.15)') 'CHS=', Chs_a(1,1)
    WRITE(*,'(A,ES23.15)') 'CHS2=', Chs2_a(1,1)
    WRITE(*,'(A,ES23.15)') 'CQS2=', Cqs2_a(1,1)
    WRITE(*,'(A,ES23.15)') 'HFX=', Hfx_a(1,1)
    WRITE(*,'(A,ES23.15)') 'QFX=', Qfx_a(1,1)
    WRITE(*,'(A,ES23.15)') 'FLX_LH=', Flxlh_a(1,1)
    WRITE(*,'(A,ES23.15)') 'FLHC=', Flhc_a(1,1)
    WRITE(*,'(A,ES23.15)') 'FLQC=', Flqc_a(1,1)
    WRITE(*,'(A,ES23.15)') 'QGH=', Qgh_a(1,1)
    WRITE(*,'(A,ES23.15)') 'CPM=', Cpm_a(1,1)
    WRITE(*,'(A,ES23.15)') 'CT=', Ct_a(1,1)
    WRITE(*,'(A,ES23.15)') 'QSFC=', Qsf(1,1)
    WRITE(*,'(A,ES23.15)') 'THZ0=', Thz(1,1)
    WRITE(*,'(A,ES23.15)') 'QZ0=', Qz(1,1)
    WRITE(*,'(A,ES23.15)') 'UZ0=', Uz(1,1)
    WRITE(*,'(A,ES23.15)') 'VZ0=', Vz(1,1)
    WRITE(*,'(A,ES23.15)') 'PBLH=', Pblh_a(1,1)
    WRITE(*,'(A,ES23.15)') 'U10=', U10_a(1,1)
    WRITE(*,'(A,ES23.15)') 'V10=', V10_a(1,1)
    WRITE(*,'(A,ES23.15)') 'T02=', T02_a(1,1)
    WRITE(*,'(A,ES23.15)') 'TH02=', Th02_a(1,1)
    WRITE(*,'(A,ES23.15)') 'TSHLTR=', Tshltr_a(1,1)
    WRITE(*,'(A,ES23.15)') 'TH10=', Th10_a(1,1)
    WRITE(*,'(A,ES23.15)') 'Q02=', Q02_a(1,1)
    WRITE(*,'(A,ES23.15)') 'QSHLTR=', Qshltr_a(1,1)
    WRITE(*,'(A,ES23.15)') 'Q10=', Q10_a(1,1)
    WRITE(*,'(A,ES23.15)') 'PSHLTR=', Pshltr_a(1,1)
    WRITE(*,'(A,ES23.15)') 'U10E=', U10e_a(1,1)
    WRITE(*,'(A,ES23.15)') 'V10E=', V10e_a(1,1)
    CALL dump_col('U', Uu)
    CALL dump_col('V', Vv)
    CALL dump_col('T', Tt)
    CALL dump_col('TH', Tht)
    CALL dump_col('QV', Qq)
    CALL dump_col('QC', Qcw)
    CALL dump_col('PMID', Pp)
    CALL dump_col_interface('PINT', Pii)
    CALL dump_col('DZ', Dzz)
    CALL dump_col('Q2', Q2a)
  END SUBROUTINE dump_case
END PROGRAM myjsfc_oracle_dbg
