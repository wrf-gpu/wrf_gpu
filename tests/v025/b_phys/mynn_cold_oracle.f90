! Original WRF routines and fresh-call argument mapping, with deterministic
! rmol=0 matching the retained initializer (WRF driver leaves it uninitialized).
program cold_oracle
use module_bl_mynnedmf, only: get_pblh,scale_aware,mym_initialize
use module_bl_mynnedmf_common, only: kind_phys,p608,rcp,p1000mb,xlvcp,xlscp
implicit none
integer :: nz,nb,col,k,kpbl
real(kind_phys) :: dt,dx,ust,flt,flq,fltv,xland,ts,rhosfc,pblh,psig,psig_shcu,denom
real(kind_phys),allocatable :: a(:,:),zw(:),th(:),thv(:),thl(:),thlv(:),sqv(:),qke(:),work(:,:),zero_arrays(:,:)
character(len=1024) :: input_path,output_path
call get_command_argument(1,input_path)
call get_command_argument(2,output_path)
open(10,file=trim(input_path),status='old')
open(20,file=trim(output_path),status='replace')
read(10,*) nz,nb,dt,dx
allocate(a(nz,13),zw(nz+1),th(nz),thv(nz),thl(nz),thlv(nz),sqv(nz),qke(nz),work(nz,6),zero_arrays(nz,6))
write(20,*) nz,nb
do col=1,nb
 read(10,*) ust,flt,flq,fltv,xland,ts,rhosfc
 do k=1,nz
  read(10,*) a(k,:)
 enddo
 zw(1)=0.;work=0.;zero_arrays=0.
 do k=1,nz
  denom=1.+max(a(k,5),0.)
  sqv(k)=a(k,5)/denom
  th(k)=a(k,4)
  thv(k)=th(k)*(1.+p608*sqv(k))
  thl(k)=th(k)-xlvcp/((a(k,7)/p1000mb)**rcp)*a(k,10)/denom &
              -xlscp/((a(k,7)/p1000mb)**rcp)*a(k,11)/denom
  thlv(k)=thl(k)*(1.+p608*sqv(k))
  zw(k+1)=zw(k)+a(k,9)
  qke(k)=5.*ust*max((ust*700.-zw(k))/(max(ust,.01)*700.),.01)
 enddo
 call get_pblh(1,nz,pblh,thv,qke,zw,a(:,9),xland,kpbl)
 call scale_aware(dx,pblh,psig,psig_shcu)
 ! Distinct SH/SM/EL/TSQ/QSQ/COV storage prevents caller operand aliasing.
 ! WRF driver passes SQV (vapor), not SQW (total water), in this call.
 call mym_initialize(1,nz,xland,a(:,9),dx,zw,a(:,1),a(:,2),thl,sqv, &
  pblh,th,thv,thlv,work(:,1),work(:,2),ust,0._kind_phys,work(:,3),qke, &
  work(:,4),work(:,5),work(:,6),psig,zero_arrays(:,1),1,zero_arrays(:,2),zero_arrays(:,3), &
  zero_arrays(:,4),zero_arrays(:,5),.true.,0,zero_arrays(:,6))
 write(20,*) pblh
 do k=1,nz
  write(20,*) qke(k)
 enddo
enddo
close(10);close(20)
end program
