program whole_mynn_oracle
use module_bl_mynnedmf, only: mynnedmf
use module_bl_mynnedmf_common, only: kind_phys,cp,p608,rcp,r_d,p1000mb
implicit none
integer :: nz,nb,k,col,kpbl,errflg
real(kind_phys) :: dt,dx,ust,flt,flq,fltv_input,xland,ts,rhosfc,ps,hfx,qfx,wind,pblh,maxwidth,maxmf,ztop,denom
real(kind_phys),allocatable :: buf(:,:),chem(:,:),vdep(:),qv_input(:),qc_input(:),qi_input(:),qs_input(:)
character(len=1024):: input_path,output_path,errmsg
call get_command_argument(1,input_path)
call get_command_argument(2,output_path)
open(10,file=trim(input_path),status='old')
open(20,file=trim(output_path),status='replace')
read(10,*) nz,nb,dt,dx
allocate(buf(nz,62),chem(nz,1),vdep(1),qv_input(nz),qc_input(nz),qi_input(nz),qs_input(nz))
write(20,*) nz,nb,dt,dx
do col=1,nb
 buf=0.; chem=0.; vdep=0.
 read(10,*) ust,flt,flq,fltv_input,xland,ts,rhosfc
 do k=1,nz
  read(10,*) buf(k,2), buf(k,3), buf(k,4), buf(k,5), buf(k,6), buf(k,20), buf(k,16), buf(k,18), buf(k,1), buf(k,7), buf(k,8), buf(k,9), buf(k,28)
  qv_input(k)=buf(k,6)
  qc_input(k)=buf(k,7)
  qi_input(k)=buf(k,8)
  qs_input(k)=buf(k,9)
  denom=1.+max(buf(k,6),0.)
  buf(k,6)=buf(k,6)/denom
  buf(k,7)=buf(k,7)/denom
  buf(k,8)=buf(k,8)/denom
  buf(k,9)=buf(k,9)/denom
  buf(k,20)=2.*buf(k,20)
  buf(k,21)=buf(k,20)
  buf(k,17)=(buf(k,16)/p1000mb)**rcp
  buf(k,19)=buf(k,5)*buf(k,17)
 enddo
 ! Invert the retained kinematic-flux/rhosfc interface into WRF caller fields.
 hfx=buf(1,18)*cp*(1.+.84*max(buf(1,6),1.e-8))*flt
 qfx=buf(1,18)*flq
 ps=rhosfc*r_d*(buf(1,19)+p608*qv_input(1))
 wind=max(sqrt(buf(1,2)**2+buf(1,3)**2),.2_kind_phys)
 pblh=0.;kpbl=1;maxwidth=0.;maxmf=0.;ztop=0.
 call mynnedmf( &
  i=1, j=1, initflag=0, restart=.true. , &
  cycling=.false., delt=dt, dz1=buf(:,1), dx=dx , &
  u1=buf(:,2), v1=buf(:,3), w1=buf(:,4), th1=buf(:,5) , &
  sqv1=buf(:,6), sqc1=buf(:,7), sqi1=buf(:,8), sqs1=buf(:,9) , &
  qnc1=buf(:,10), qni1=buf(:,11), qnwfa1=buf(:,12), qnifa1=buf(:,13) , &
  qnbca1=buf(:,14), ozone1=buf(:,15), p1=buf(:,16), ex1=buf(:,17) , &
  rho1=buf(:,18), tk1=buf(:,19), xland=xland, ts=ts , &
  qsfc=0._kind_phys, ps=ps, ust=ust, ch=0._kind_phys , &
  hfx=hfx, qfx=qfx, znt=.1_kind_phys, wspd=wind , &
  uoce=0._kind_phys, voce=0._kind_phys, qke1=buf(:,20), qke_adv1=buf(:,21) , &
  el1=buf(:,22), sh1=buf(:,23), sm1=buf(:,24), kh1=buf(:,25) , &
  km1=buf(:,26), nchem=1, kdvel=0, ndvel=1 , &
  chem=chem, vdep=vdep, frp=0._kind_phys, emis_ant_no=0._kind_phys , &
  mix_chem=.false., enh_mix=.false., rrfs_sd=.false., smoke_dbg=.false. , &
  tsq1=buf(:,27), qsq1=buf(:,28), cov1=buf(:,29), du1=buf(:,30) , &
  dv1=buf(:,31), dth1=buf(:,32), dqv1=buf(:,33), dqc1=buf(:,34) , &
  dqi1=buf(:,35), dqnc1=buf(:,36), dqni1=buf(:,37), dqs1=buf(:,38) , &
  dqnwfa1=buf(:,39), dqnifa1=buf(:,40), dqnbca1=buf(:,41), dozone1=buf(:,42) , &
  rthraten1=buf(:,43), pblh=pblh, kpbl=kpbl, maxwidth=maxwidth , &
  maxmf=maxmf, ztop_plume=ztop, dqke1=buf(:,44), qwt1=buf(:,45) , &
  qshear1=buf(:,46), qbuoy1=buf(:,47), qdiss1=buf(:,48), qc_bl1=buf(:,49) , &
  qi_bl1=buf(:,50), cldfra_bl1=buf(:,51), bl_mynn_tkeadvect=.false., tke_budget=0 , &
  bl_mynn_cloudpdf=2, bl_mynn_mixlength=1, icloud_bl=1, closure=2.6_kind_phys , &
  bl_mynn_edmf=1, bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0, bl_mynn_mixscalars=1 , &
  bl_mynn_output=1, bl_mynn_cloudmix=1, bl_mynn_mixqt=0, edmf_a1=buf(:,52) , &
  edmf_w1=buf(:,53), edmf_qt1=buf(:,54), edmf_thl1=buf(:,55), edmf_ent1=buf(:,56) , &
  edmf_qc1=buf(:,57), sub_thl1=buf(:,58), sub_sqv1=buf(:,59), det_thl1=buf(:,60) , &
  det_sqv1=buf(:,61), spp_pbl=0, pattern_spp_pbl1=buf(:,62), flag_qc=.true. , &
  flag_qi=.true., flag_qnc=.false., flag_qni=.false., flag_qs=.false. , &
  flag_qnwfa=.false., flag_qnifa=.false., flag_qnbca=.false., flag_ozone=.false. , &
  kts=1, kte=nz, errmsg=errmsg, errflg=errflg)
 if (errflg/=0) then
  write(*,*) 'WRF error ',errflg,trim(errmsg)
  stop 3
 endif
 call mynnedmf_post_run(nz,.true.,.true.,.false.,dt,qv_input,qc_input,qi_input,qs_input, &
  buf(:,33),buf(:,34),buf(:,35),buf(:,38),errmsg,errflg)
 write(20,*) pblh,maxmf
 do k=1,nz
  write(20,*) buf(k,30), buf(k,31), buf(k,32), buf(k,33), buf(k,20), buf(k,25), buf(k,26), buf(k,22), buf(k,28), buf(k,49), buf(k,50), buf(k,51), buf(k,34), buf(k,35)
 enddo
enddo
close(10)
close(20)
contains
 subroutine mynnedmf_post_run(kte,f_qc,f_qi,f_qs,delt,qv,qc,qi,qs,dqv,dqc,dqi,dqs,errmsg,errflg)
!=================================================================================================================

!---  input arguments:
 logical,intent(in):: &
    f_qc, &                   ! if true,the physics package includes the cloud liquid water mixing ratio.
    f_qi, &                   ! if true,the physics package includes the cloud ice mixing ratio.
    f_qs                      ! if true,the physics package includes the snow mixing ratio.

 integer,intent(in):: kte

 real(kind=kind_phys),intent(in):: &
    delt                      !

 real(kind=kind_phys),intent(in),dimension(1:kte):: &
    qv,   &                   !
    qc,   &                   !
    qi,   &                   !
    qs                        !

!---  inout arguments:
 real(kind=kind_phys),intent(inout),dimension(1:kte):: &
    dqv,  &                   !
    dqc,  &                   !
    dqi,  &                   !
    dqs                       !

!---  output arguments:
 character(len=*),intent(out):: errmsg
 integer,intent(out):: errflg


!---  local variables:
 integer:: k
 integer,parameter::kts=1
 real(kind=kind_phys):: rq,sq,tem
 real(kind=kind_phys),dimension(1:kte):: sqv,sqc,sqi,sqs
!-----------------------------------------------------------------------------------------------------------------
!---  initialization:
 do k = kts,kte
    sq = qv(k)/(1.+qv(k))      !conversion of qv at time-step n from mixing ratio to specific humidity.
    sqv(k) = sq + dqv(k)*delt  !calculation of specific humidity at time-step n+1.
    rq = sqv(k)/(1.-sqv(k))    !conversion of qv at time-step n+1 from specific humidity to mixing ratio.
    dqv(k) = (rq - qv(k))/delt !calculation of the tendency.
 enddo

 if (f_qc) then
    do k = kts,kte
       sq = qc(k)/(1.+qv(k))
       sqc(k) = sq + dqc(k)*delt
       rq  = sqc(k)*(1.+sqv(k))
       dqc(k) = (rq - qc(k))/delt
    enddo
 endif

 if (f_qi) then
    do k = kts,kte
       sq = qi(k)/(1.+qv(k))
       sqi(k) = sq + dqi(k)*delt
       rq = sqi(k)*(1.+sqv(k))
       dqi(k) = (rq - qi(k))/delt
    enddo
 endif

 if (f_qs) then
    do k = kts,kte
       sq = qs(k)/(1.+qv(k))
       sqs(k) = sq + dqs(k)*delt
       rq = sqs(k)*(1.+sqv(k))
       dqs(k) = (rq - qs(k))/delt
    enddo
 endif

!--- output error flag and message:
 errmsg = " "
 errflg = 0

 end subroutine mynnedmf_post_run

end program
