program wrf_mynn_harness
  implicit none

  integer :: nz, k, argc
  real(8) :: dt
  character(len=512) :: input_path, output_path
  real(8), allocatable :: u(:), v(:), w(:), theta(:), qv(:), tke(:), p(:), rho(:), dz(:)
  real(8), allocatable :: uo(:), vo(:), wo(:), thetao(:), qvo(:), tkeo(:)
  real(8), allocatable :: km(:), kh(:), shear(:), buoy(:)

  argc = command_argument_count()
  if (argc /= 2) then
     write(*,*) 'usage: wrf_mynn_harness input.dat output.dat'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10,*) nz, dt
  allocate(u(nz), v(nz), w(nz), theta(nz), qv(nz), tke(nz), p(nz), rho(nz), dz(nz))
  allocate(uo(nz), vo(nz), wo(nz), thetao(nz), qvo(nz), tkeo(nz), km(nz), kh(nz), shear(nz), buoy(nz))
  do k = 1, nz
     read(10,*) u(k), v(k), w(k), theta(k), qv(k), tke(k), p(k), rho(k), dz(k)
  end do
  close(10)

  call source_derived_mynn(nz, dt, u, v, w, theta, qv, tke, p, rho, dz, uo, vo, wo, thetao, qvo, tkeo, km, kh, shear, buoy)

  open(unit=20, file=trim(output_path), status='replace', action='write')
  write(20,'(I6,1X,ES24.16E3)') nz, dt
  do k = 1, nz
     write(20,'(14(ES24.16E3,1X))') uo(k), vo(k), wo(k), thetao(k), qvo(k), tkeo(k), p(k), rho(k), dz(k), km(k), kh(k), shear(k), buoy(k), 0.0d0
  end do
  close(20)

contains

  subroutine source_derived_mynn(nz, dt, u, v, w, theta, qv, tke, p, rho, dz, uo, vo, wo, thetao, qvo, tkeo, km, kh, shear, buoy)
    integer, intent(in) :: nz
    real(8), intent(in) :: dt
    real(8), intent(in) :: u(nz), v(nz), w(nz), theta(nz), qv(nz), tke(nz), p(nz), rho(nz), dz(nz)
    real(8), intent(out) :: uo(nz), vo(nz), wo(nz), thetao(nz), qvo(nz), tkeo(nz), km(nz), kh(nz), shear(nz), buoy(nz)
    integer :: k
    real(8), parameter :: grav=9.81d0, tref=300.0d0, karman=0.4d0, b1=24.0d0, qkemin=1.0d-5
    real(8) :: z, dzk, du, dv, dth, ri, sh, sm, qkw, el, prod, diss, ustar, wind

    uo = u
    vo = v
    wo = w
    thetao = theta
    qvo = max(qv, 0.0d0)
    tkeo = max(tke, 0.5d0*qkemin)
    km = 0.0d0
    kh = 0.0d0
    shear = 0.0d0
    buoy = 0.0d0

    z = 0.0d0
    do k = 2, nz
       z = z + dz(k-1)
       dzk = max(0.5d0*(dz(k)+dz(k-1)), 1.0d0)
       du = (u(k)-u(k-1))/dzk
       dv = (v(k)-v(k-1))/dzk
       dth = ((theta(k)*(1.0d0+0.608d0*qv(k))) - (theta(k-1)*(1.0d0+0.608d0*qv(k-1))))/dzk
       ri = (grav/tref)*dth/max(du*du + dv*dv, 1.0d-10)
       sh = max(0.02d0, min(4.0d0, 0.74d0/(1.0d0 + 5.0d0*max(ri, 0.0d0))))
       sm = max(0.0d0, min(4.0d0, sh*min(0.76d0 + 4.0d0*max(ri, 0.0d0), 5.0d0)))
       qkw = sqrt(max(2.0d0*0.5d0*(tke(k)+tke(k-1)), qkemin))
       el = min(400.0d0, max(0.1d0, (karman*z)/(1.0d0 + karman*z/120.0d0)))
       km(k) = el*qkw*sm
       kh(k) = el*qkw*sh
       shear(k) = km(k)*(du*du + dv*dv)
       buoy(k) = -kh(k)*(grav/tref)*dth
    end do

    do k = 2, nz-1
       uo(k) = u(k) + dt*0.5d0*((km(k+1)*(u(k+1)-u(k))/max(dz(k),1.0d0)) - (km(k)*(u(k)-u(k-1))/max(dz(k-1),1.0d0)))/max(dz(k),1.0d0)
       vo(k) = v(k) + dt*0.5d0*((km(k+1)*(v(k+1)-v(k))/max(dz(k),1.0d0)) - (km(k)*(v(k)-v(k-1))/max(dz(k-1),1.0d0)))/max(dz(k),1.0d0)
       thetao(k) = theta(k) + dt*0.5d0*((kh(k+1)*(theta(k+1)-theta(k))/max(dz(k),1.0d0)) - (kh(k)*(theta(k)-theta(k-1))/max(dz(k-1),1.0d0)))/max(dz(k),1.0d0)
       qvo(k) = max(0.0d0, qv(k) + dt*0.5d0*((kh(k+1)*(qv(k+1)-qv(k))/max(dz(k),1.0d0)) - (kh(k)*(qv(k)-qv(k-1))/max(dz(k-1),1.0d0)))/max(dz(k),1.0d0))
    end do

    wind = max(sqrt(u(1)*u(1) + v(1)*v(1)), 0.2d0)
    ustar = sqrt(1.3d-3)*wind
    do k = 1, nz
       qkw = sqrt(max(2.0d0*tke(k), qkemin))
       el = max(1.0d0, 0.5d0*(max(km(k),0.0d0)+max(kh(k),0.0d0))/(max(qkw,1.0d-8)*0.4d0))
       prod = shear(k) + buoy(k)
       diss = max(tke(k), 0.5d0*qkemin)**1.5d0/(b1*el)
       tkeo(k) = max(0.5d0*qkemin, tke(k) + dt*(prod - diss))
    end do
    tkeo(1) = max(0.5d0*qkemin, tkeo(1) + dt*ustar**3/max(0.5d0*dz(1),1.0d0))
  end subroutine source_derived_mynn
end program wrf_mynn_harness
