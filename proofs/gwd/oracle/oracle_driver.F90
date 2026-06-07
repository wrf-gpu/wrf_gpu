! Standalone oracle: drive WRF bl_gwdo_run on a controlled column batch and dump
! inputs + outputs as plain text for the JAX-port comparison.
program oracle_driver
  use ccpp_kind_types, only: kind_phys
  use bl_gwdo, only: bl_gwdo_run
  implicit none

  integer, parameter :: its = 1, ite = 3, kte = 32, kme = kte + 1
  real(kind=kind_phys), parameter :: g_=9.81, cp_=7.*287./2., rd_=287., &
                                     rv_=461.6, fv_=461.6/287.-1., pi_=3.141592653
  real(kind=kind_phys), parameter :: deltim = 60.0

  real(kind=kind_phys), dimension(its:ite) :: sina, cosa, dxmeter
  real(kind=kind_phys), dimension(its:ite) :: var, oc1, oa2d1, oa2d2, oa2d3, oa2d4
  real(kind=kind_phys), dimension(its:ite) :: ol2d1, ol2d2, ol2d3, ol2d4
  real(kind=kind_phys), dimension(its:ite) :: dusfcg, dvsfcg
  real(kind=kind_phys), dimension(its:ite, kte) :: uproj, vproj, t1, q1, prsl, prslk, zl
  real(kind=kind_phys), dimension(its:ite, kte) :: rublten, rvblten, dtaux3d, dtauy3d
  real(kind=kind_phys), dimension(its:ite, kme) :: prsi
  character(len=512) :: errmsg
  integer :: errflg, i, k
  real(kind=kind_phys) :: dz, psfc, ptop, zlev, lapse, tsfc, u0, v0

  ! --- build three columns: 0 flat(var=0), 1 mountain westerly, 2 mountain SW
  dz = 300.0; psfc = 100000.0; ptop = 5000.0; lapse = 0.0065; tsfc = 288.0
  do i = its, ite
    do k = 1, kte
      zlev = (real(k) - 0.5) * dz
      zl(i,k) = zlev
      t1(i,k) = tsfc - lapse * zlev
      q1(i,k) = 0.004
      prslk(i,k) = 0.0  ! set below
    end do
    do k = 1, kme
      prsi(i,k) = psfc + (ptop - psfc) * real(k-1) / real(kme-1)
    end do
    do k = 1, kte
      prsl(i,k) = 0.5 * (prsi(i,k) + prsi(i,k+1))
      prslk(i,k) = (prsl(i,k) / 1.0e5) ** (287.0/1004.5)
    end do
  end do

  ! winds
  do k = 1, kte
    uproj(1,k) = 15.0; vproj(1,k) = 0.0    ! flat col
    uproj(2,k) = 20.0; vproj(2,k) = 0.0    ! westerly over mountain
    uproj(3,k) = 12.0; vproj(3,k) = 12.0   ! SW flow over mountain
  end do

  ! statics
  var   = (/ 0.0, 300.0, 400.0 /)
  oc1   = (/ 1.0, 1.0, 1.2 /)
  oa2d1 = (/ 0.4, 0.4, 0.3 /); oa2d2 = (/ 0.0, 0.0, 0.2 /)
  oa2d3 = (/ 0.0, 0.0, 0.1 /); oa2d4 = (/ 0.0, 0.0, 0.0 /)
  ol2d1 = (/ 0.3, 0.3, 0.25 /); ol2d2 = (/ 0.3, 0.3, 0.35 /)
  ol2d3 = (/ 0.3, 0.3, 0.30 /); ol2d4 = (/ 0.3, 0.3, 0.20 /)
  sina = 0.0; cosa = 1.0; dxmeter = 3000.0

  rublten = 0.0; rvblten = 0.0

  call bl_gwdo_run(sina=sina, cosa=cosa, rublten=rublten, rvblten=rvblten, &
       dtaux3d=dtaux3d, dtauy3d=dtauy3d, dusfcg=dusfcg, dvsfcg=dvsfcg, &
       uproj=uproj, vproj=vproj, t1=t1, q1=q1, prsi=prsi, prsl=prsl, &
       prslk=prslk, zl=zl, var=var, oc1=oc1, &
       oa2d1=oa2d1, oa2d2=oa2d2, oa2d3=oa2d3, oa2d4=oa2d4, &
       ol2d1=ol2d1, ol2d2=ol2d2, ol2d3=ol2d3, ol2d4=ol2d4, &
       g_=g_, cp_=cp_, rd_=rd_, rv_=rv_, fv_=fv_, pi_=pi_, &
       dxmeter=dxmeter, deltim=deltim, its=its, ite=ite, kte=kte, kme=kme, &
       errmsg=errmsg, errflg=errflg)

  ! dump: one block per column
  open(unit=10, file='/tmp/gwdo_oracle/oracle_out.txt', status='replace')
  write(10,'(A)') '# GWDO oracle: kte, ncol'
  write(10,'(I0,1X,I0)') kte, ite
  do i = its, ite
    write(10,'(A,I0)') '# COL ', i
    write(10,'(A)') '# inputs: k uproj vproj t1 q1 prsl prsi(k) prslk zl'
    do k = 1, kte
      write(10,'(I0,9(1X,ES20.12))') k, uproj(i,k), vproj(i,k), t1(i,k), q1(i,k), &
           prsl(i,k), prsi(i,k), prslk(i,k), zl(i,k)
    end do
    write(10,'(A,1X,ES20.12)') '# prsi_top', prsi(i,kme)
    write(10,'(A)') '# statics: var oc1 oa1 oa2 oa3 oa4 ol1 ol2 ol3 ol4 sina cosa dx'
    write(10,'(13(ES20.12,1X))') var(i), oc1(i), oa2d1(i), oa2d2(i), oa2d3(i), &
         oa2d4(i), ol2d1(i), ol2d2(i), ol2d3(i), ol2d4(i), sina(i), cosa(i), dxmeter(i)
    write(10,'(A)') '# outputs: k rublten rvblten dtaux3d dtauy3d'
    do k = 1, kte
      write(10,'(I0,4(1X,ES20.12))') k, rublten(i,k), rvblten(i,k), dtaux3d(i,k), dtauy3d(i,k)
    end do
    write(10,'(A,2(1X,ES20.12))') '# dusfcg dvsfcg', dusfcg(i), dvsfcg(i)
  end do
  close(10)
  write(*,'(A,1X,I0,1X,A)') 'errflg=', errflg, trim(errmsg)
  print *, 'wrote /tmp/gwdo_oracle/oracle_out.txt'
end program oracle_driver
