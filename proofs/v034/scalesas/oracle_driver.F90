! v0.3.4 o1-sas: multi-column pristine-WRF oracle for CU_SCALESAS (cu_physics=4, ARW).
!
! Calls the UNMODIFIED phys/module_cu_scalesas.F (WRF 4.7.1) exactly as the ARW
! cumulus_driver does (keyword args; MOMMIX/sas_mass_flux/shalconv/shal_pgcon/
! HPBL2D/EVAP2D/HEAT2D/pert_sas absent), except DY: ARW passes the absent optional
! DYNMM as the non-optional DY (upstream bug, segfaults with gfortran), so the
! oracle passes DY explicitly (read from the input file).
!
! Input (stdin, list-directed):  N KX / DT STEPCU ITIMESTEP DY PGCON / per column:
!   XLAND DX2D / T(KX) QV QC QI PCPS PI RHO DZ8W U V / P8W(KX+1) / W(KX+1)
! Output (stdout): per column one header line + 4 tendency lines (KX values each).
PROGRAM scalesas_oracle
  USE module_cu_scalesas, ONLY : cu_scalesas
  IMPLICIT NONE
  INTEGER :: N, KX, i, k, STEPCU, ITIMESTEP
  REAL :: DT, DY, PGCON
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: U,V,W,T,QV,QC,QI,PII,RHO,DZ8W,PCPS,P8W
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQICUTEN
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: RUCUTEN,RVCUTEN
  REAL, ALLOCATABLE, DIMENSION(:,:) :: RAINCV,PRATEC,XLAND,HBOT,HTOP,DX2D
  REAL, ALLOCATABLE, DIMENSION(:,:) :: SCALEFUN,SCALEFUN1,SIGMU,SIGMU1
  LOGICAL, ALLOCATABLE :: CU_ACT_FLAG(:,:)

  READ(*,*) N, KX
  READ(*,*) DT, STEPCU, ITIMESTEP, DY, PGCON
  ALLOCATE(U(N,KX+1,1),V(N,KX+1,1),W(N,KX+1,1),T(N,KX+1,1),QV(N,KX+1,1),QC(N,KX+1,1))
  ALLOCATE(QI(N,KX+1,1),PII(N,KX+1,1),RHO(N,KX+1,1),DZ8W(N,KX+1,1),PCPS(N,KX+1,1),P8W(N,KX+1,1))
  ALLOCATE(RTHCUTEN(N,KX+1,1),RQVCUTEN(N,KX+1,1),RQCCUTEN(N,KX+1,1),RQICUTEN(N,KX+1,1))
  ALLOCATE(RUCUTEN(N,1,KX+1),RVCUTEN(N,1,KX+1))
  ALLOCATE(RAINCV(N,1),PRATEC(N,1),XLAND(N,1),HBOT(N,1),HTOP(N,1),DX2D(N,1))
  ALLOCATE(SCALEFUN(N,1),SCALEFUN1(N,1),SIGMU(N,1),SIGMU1(N,1),CU_ACT_FLAG(N,1))
  U=0.; V=0.; W=0.; T=0.; QV=0.; QC=0.; QI=0.; PII=1.; RHO=0.; DZ8W=0.; PCPS=0.; P8W=0.
  RTHCUTEN=0.; RQVCUTEN=0.; RQCCUTEN=0.; RQICUTEN=0.; RUCUTEN=0.; RVCUTEN=0.
  RAINCV=0.; PRATEC=0.; HBOT=0.; HTOP=0.; SCALEFUN=0.; SCALEFUN1=0.; SIGMU=0.; SIGMU1=0.
  CU_ACT_FLAG=.TRUE.
  DO i=1,N
    READ(*,*) XLAND(i,1), DX2D(i,1)
    READ(*,*) (T(i,k,1),k=1,KX)
    READ(*,*) (QV(i,k,1),k=1,KX)
    READ(*,*) (QC(i,k,1),k=1,KX)
    READ(*,*) (QI(i,k,1),k=1,KX)
    READ(*,*) (PCPS(i,k,1),k=1,KX)
    READ(*,*) (PII(i,k,1),k=1,KX)
    READ(*,*) (RHO(i,k,1),k=1,KX)
    READ(*,*) (DZ8W(i,k,1),k=1,KX)
    READ(*,*) (U(i,k,1),k=1,KX)
    READ(*,*) (V(i,k,1),k=1,KX)
    READ(*,*) (P8W(i,k,1),k=1,KX+1)
    READ(*,*) (W(i,k,1),k=1,KX+1)
  ENDDO

  CALL cu_scalesas(DT=DT,ITIMESTEP=ITIMESTEP,STEPCU=STEPCU,                 &
       RTHCUTEN=RTHCUTEN,RQVCUTEN=RQVCUTEN,RQCCUTEN=RQCCUTEN,RQICUTEN=RQICUTEN, &
       RUCUTEN=RUCUTEN,RVCUTEN=RVCUTEN,RAINCV=RAINCV,PRATEC=PRATEC,         &
       HTOP=HTOP,HBOT=HBOT,U3D=U,V3D=V,W=W,T3D=T,QV3D=QV,QC3D=QC,QI3D=QI,   &
       PI3D=PII,RHO3D=RHO,DZ8W=DZ8W,PCPS=PCPS,P8W=P8W,XLAND=XLAND,          &
       CU_ACT_FLAG=CU_ACT_FLAG,P_QC=2,PGCON=PGCON,                          &
       P_QI=3,P_FIRST_SCALAR=2,DX2D=DX2D,DY=DY,                             &
       SCALEFUN=SCALEFUN,SCALEFUN1=SCALEFUN1,SIGMU=SIGMU,SIGMU1=SIGMU1,     &
       ids=1,ide=N+1,jds=1,jde=2,kds=1,kde=KX+1,                            &
       ims=1,ime=N,jms=1,jme=1,kms=1,kme=KX+1,                              &
       its=1,ite=N,jts=1,jte=1,kts=1,kte=KX)

  DO i=1,N
    WRITE(*,'(I7,6ES27.17E3)') i, RAINCV(i,1), PRATEC(i,1), HBOT(i,1), HTOP(i,1), &
         SCALEFUN(i,1), SIGMU(i,1)
    WRITE(*,'(*(ES27.17E3))') (RTHCUTEN(i,k,1),k=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RQVCUTEN(i,k,1),k=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RQCCUTEN(i,k,1),k=1,KX)
    WRITE(*,'(*(ES27.17E3))') (RQICUTEN(i,k,1),k=1,KX)
  ENDDO
END PROGRAM scalesas_oracle
