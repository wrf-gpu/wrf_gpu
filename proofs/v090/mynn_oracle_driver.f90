PROGRAM mynn_driver
  USE module_sf_mynn, ONLY: sfclay1d_mynn, mynn_sf_init_driver
  IMPLICIT NONE
  ! WRF constants (module_model_constants / SFCLAY_mynn defaults)
  REAL, PARAMETER :: CP=1004.5, G=9.81, R=287., XLV=2.5E6
  REAL, PARAMETER :: SVP1=0.6112, SVP2=17.67, SVP3=29.65, SVPT0=273.15
  REAL, PARAMETER :: EP1=0.60776, EP2=0.62175  ! ep1=r_v/r_d-1, ep2=r_d/r_v
  REAL, PARAMETER :: KARMAN=0.4, ROVCP=287./1004.5

  INTEGER, PARAMETER :: N=512   ! max columns
  INTEGER :: ncol, i, itimestep, isfflx, isftcflx, iz0tlnd, spp_pbl
  REAL :: DX
  ! per-column inputs
  REAL, DIMENSION(N) :: U1D,V1D,T1D,QV1D,P1D,dz8w1d,rho1d,U1D2,V1D2,dz2w1d
  REAL, DIMENSION(N) :: MAVAIL,PBLH,XLAND,TSK,PSFCPA,QCG,SNOWH
  REAL, DIMENSION(N) :: REGIME,HFX,QFX,LH,MOL,RMOL,QGH,QSFC,ZNT,ZOL,UST,CPM
  REAL, DIMENSION(N) :: CHS2,CQS2,CHS,CH,FLHC,FLQC,GZ1OZ0,WSPD,BR,PSIM,PSIH
  REAL, DIMENSION(N) :: U10,V10,TH2,T2,Q2,wstar,qstar,rstoch1D
  REAL, DIMENSION(N) :: ustm,ck,cka,cd,cda

  ! read scalar header then per-column rows from stdin
  read(*,*) ncol, itimestep, isfflx, isftcflx, iz0tlnd, spp_pbl, DX
  do i=1,ncol
     read(*,*) U1D(i),V1D(i),T1D(i),QV1D(i),P1D(i),dz8w1d(i),rho1d(i), &
               U1D2(i),V1D2(i),dz2w1d(i), &
               MAVAIL(i),PBLH(i),XLAND(i),TSK(i),PSFCPA(i),QCG(i),SNOWH(i), &
               ZNT(i),UST(i),MOL(i),QSFC(i),HFX(i),QFX(i)
  end do

  CALL mynn_sf_init_driver(.true.)

  rstoch1D=0.0; REGIME=0.; LH=0.; RMOL=0.; QGH=0.; ZOL=0.; CPM=0.
  CHS2=0.; CQS2=0.; CHS=0.; CH=0.; FLHC=0.; FLQC=0.; GZ1OZ0=0.; WSPD=0.
  BR=0.; PSIM=0.; PSIH=0.; U10=0.; V10=0.; TH2=0.; T2=0.; Q2=0.
  wstar=0.; qstar=0.; ustm=0.; ck=0.; cka=0.; cd=0.; cda=0.

  CALL SFCLAY1D_mynn( &
       1,U1D,V1D,T1D,QV1D,P1D,dz8w1d,rho1d, U1D2,V1D2,dz2w1d, &
       CP,G,ROVCP,R,XLV,PSFCPA,CHS,CHS2,CQS2,CPM, &
       PBLH,RMOL,ZNT,UST,MAVAIL,ZOL,MOL,REGIME, &
       PSIM,PSIH,XLAND,HFX,QFX,TSK, &
       U10,V10,TH2,T2,Q2,FLHC,FLQC,SNOWH,QGH, &
       QSFC,LH,GZ1OZ0,WSPD,BR,ISFFLX,DX, &
       SVP1,SVP2,SVP3,SVPT0,EP1,EP2, &
       KARMAN,ch,qcg, itimestep, &
       wstar,qstar, spp_pbl,rstoch1D, &
       1,N, 1,1, 1,1, 1,N, 1,1, 1,1, 1,ncol, 1,1, 1,1, &
       isftcflx, iz0tlnd, ustm,ck,cka,cd,cda )

  ! header line then per-column outputs
  write(*,'(A)') '#OUTPUT'
  do i=1,ncol
     write(*,'(I5,1X,30(ES22.14,1X))') i, &
       UST(i),MOL(i),RMOL(i),ZOL(i),REGIME(i),PSIM(i),PSIH(i),BR(i), &
       FLHC(i),FLQC(i),HFX(i),QFX(i),LH(i),QSFC(i),QGH(i), &
       CHS(i),CHS2(i),CQS2(i),CH(i),WSPD(i),GZ1OZ0(i), &
       U10(i),V10(i),TH2(i),T2(i),Q2(i),CPM(i),wstar(i),qstar(i),ZNT(i)
  end do
END PROGRAM mynn_driver
