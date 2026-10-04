! Batch driver for the pristine WRF bl_gwdo_run (phys/physics_mmm/bl_gwdo.F90).
! Reads float32 columns in Fortran (ncol, nk) order from gwdo_in.bin, calls the
! unmodified routine once over the whole batch (its=1, ite=ncol, kts=1, kte=nk),
! and writes the routine's outputs to gwdo_out.bin. rublten/rvblten enter as 0
! (INOUT: the result is the GWDO tendency alone).
program gwdo_driver
  use ccpp_kind_types, only: kind_phys
  use bl_gwdo, only: bl_gwdo_run
  implicit none
  integer :: ncol, nk, u, errflg
  real(kind_phys) :: deltim
  real(kind_phys), allocatable, dimension(:,:) :: uproj, vproj, t1, q1, prsl, prslk, zl, prsi
  real(kind_phys), allocatable, dimension(:) :: var, oc1, oa1, oa2, oa3, oa4, ol1, ol2, ol3, ol4, &
                                                 sina, cosa, dxmeter
  real(kind_phys), allocatable, dimension(:,:) :: rublten, rvblten, dtaux3d, dtauy3d
  real(kind_phys), allocatable, dimension(:) :: dusfcg, dvsfcg
  character(len=256) :: errmsg

  open(newunit=u, file='gwdo_in.bin', access='stream', form='unformatted', status='old')
  read(u) ncol, nk, deltim
  allocate(uproj(ncol,nk), vproj(ncol,nk), t1(ncol,nk), q1(ncol,nk), prsl(ncol,nk), &
           prslk(ncol,nk), zl(ncol,nk), prsi(ncol,nk+1))
  allocate(var(ncol), oc1(ncol), oa1(ncol), oa2(ncol), oa3(ncol), oa4(ncol), ol1(ncol), &
           ol2(ncol), ol3(ncol), ol4(ncol), sina(ncol), cosa(ncol), dxmeter(ncol))
  read(u) uproj, vproj, t1, q1, prsl, prslk, zl, prsi
  read(u) var, oc1, oa1, oa2, oa3, oa4, ol1, ol2, ol3, ol4, sina, cosa, dxmeter
  close(u)

  allocate(rublten(ncol,nk), rvblten(ncol,nk), dtaux3d(ncol,nk), dtauy3d(ncol,nk), &
           dusfcg(ncol), dvsfcg(ncol))
  rublten = 0.
  rvblten = 0.
  call bl_gwdo_run(sina=sina, cosa=cosa,                          &
                   rublten=rublten, rvblten=rvblten,              &
                   dtaux3d=dtaux3d, dtauy3d=dtauy3d,              &
                   dusfcg=dusfcg, dvsfcg=dvsfcg,                  &
                   uproj=uproj, vproj=vproj, t1=t1, q1=q1,        &
                   prsi=prsi, prsl=prsl, prslk=prslk, zl=zl,      &
                   var=var, oc1=oc1,                              &
                   oa2d1=oa1, oa2d2=oa2, oa2d3=oa3, oa2d4=oa4,    &
                   ol2d1=ol1, ol2d2=ol2, ol2d3=ol3, ol2d4=ol4,    &
                   g_=9.81, cp_=7.*287./2., rd_=287., rv_=461.6,  &
                   fv_=461.6/287.-1., pi_=3.141592653,            &
                   dxmeter=dxmeter, deltim=deltim,                &
                   its=1, ite=ncol, kte=nk, kme=nk+1,             &
                   errmsg=errmsg, errflg=errflg)
  if (errflg /= 0) stop 3

  open(newunit=u, file='gwdo_out.bin', access='stream', form='unformatted', status='replace')
  write(u) rublten, rvblten, dtaux3d, dtauy3d, dusfcg, dvsfcg
  close(u)
end program gwdo_driver
