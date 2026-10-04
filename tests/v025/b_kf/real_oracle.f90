! Direct KF_eta_PARA oracle linked to UNMODIFIED pristine WRF module_cu_kfeta.F.
! Matches the trigger=1 CPS call after its U/V/T/QV/P/DZ/RHO/W0AVG copies.
! Input is column-major binary REAL*4: 8 x KX values for each column, then EOF.
! Output: 6 tendency profiles followed by RAINCV/PRATEC/NCA/CUTOP/CUBOT/ISHALL/TIMEC.
program kf_real_oracle
  use module_cu_kfeta, only: kf_eta_init, kf_eta_para
  implicit none
  integer, parameter :: kx=44
  real, parameter :: g=9.81, r=287., cp=1004.5, ep2=287./461.6
  real, parameter :: xlv0=3.15e6, xlv1=2370., xls0=2.905e6, xls1=259.532
  real, parameter :: svp1=.6112, svp2=17.67, svp3=29.65, svpt0=273.15
  real :: t(kx),qv(kx),p(kx),dz(kx),rho(kx),w0(kx),u(kx),v(kx),zero(kx),pii(kx)
  real :: dtdt(kx),dqdt(kx),dqcdt(kx),dqrdt(kx),dqidt(kx),dqsdt(kx)
  real :: rth(1,kx,1),rqv(1,kx,1),rqc(1,kx,1),rqr(1,kx,1),rqi(1,kx,1),rqs(1,kx,1)
  real :: cldd(1,kx,1),clds(1,kx,1),qc(1,kx,1),qi(1,kx,1)
  real :: udr(1,kx,1),ddr(1,kx,1),uer(1,kx,1),der(1,kx,1),wavg(1,kx,1)
  real :: rain(1,1),prate(1,1),nca(1,1),cutop(1,1),cubot(1,1),timec(1,1)
  integer :: ishall,ios,ncolumn
  character(len=1024) :: input_file,output_file
  call get_command_argument(1,input_file)
  call get_command_argument(2,output_file)
  call kf_eta_init(rth,rqv,rqc,rqr,rqi,rqs,nca,wavg,2,3,svp1,svp2,svp3,svpt0,1, &
                  .false.,.true., 1,2,1,2,1,kx+1, 1,1,1,1,1,kx, 1,1,1,1,1,kx)
  open(11,file=trim(input_file),access='stream',form='unformatted',status='old')
  open(12,file=trim(output_file),access='stream',form='unformatted',status='replace')
  zero=0.;ncolumn=0
  do
    read(11,iostat=ios) t,qv,p,dz,rho,w0,u,v
    if(ios<0)exit
    if(ios/=0)stop 2
    dtdt=0.;dqdt=0.;dqcdt=0.;dqrdt=0.;dqidt=0.;dqsdt=0.
    rain=0.;prate=0.;nca=-100.;cutop=1.;cubot=kx+1.;timec=0.;ishall=2
    cldd=0.;clds=0.;qc=0.;qi=0.;udr=0.;ddr=0.;uer=0.;der=0.
    call kf_eta_para(1,1,u,v,t,qv,p,dz,w0,zero,zero,1,54.,9000.,9000.**2,rho, &
                    xlv0,xlv1,xls0,xls1,cp,r,g,ep2,svp1,svp2,svp3,svpt0, &
                    dqdt,dqidt,dqcdt,dqrdt,dqsdt,dtdt,rain,prate,nca,.true.,.true., &
                    .false.,cutop,cubot,ishall,0., &
                    1,2,1,2,1,kx+1, 1,1,1,1,1,kx, 1,1,1,1,1,kx, &
                    cldd,clds,qc,qi,udr,ddr,uer,der,timec,1)
    pii=(p/1.e5)**(r/cp)
    write(12) dtdt/pii,dqdt,dqcdt,dqrdt,dqidt,dqsdt, &
              rain,prate,nca,cutop,cubot,real(ishall),timec
    ncolumn=ncolumn+1
  enddo
  close(11);close(12)
  print *, 'columns=',ncolumn
end program kf_real_oracle
