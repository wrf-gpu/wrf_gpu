! Standalone oracle: exact copy of WRF share/module_llxy.F arithmetic for
! Mercator (set_merc/llij_merc/ijll_merc) and polar-stereographic
! (set_ps/llij_ps/ijll_ps). Single-precision REAL, EARTH_RADIUS_M=6370000,
! matching the WRF module constants. Emits i,j (forward) and lat,lon (inverse)
! for a fixed point list so the Python port can be checked against the
! independent Fortran reference (NOT a self-compare).
program wrf_proj_oracle
  implicit none
  real, parameter :: PI = 3.141592653589793
  real, parameter :: DEG_PER_RAD = 180./PI
  real, parameter :: RAD_PER_DEG = PI/180.
  real, parameter :: RE = 6370000.

  ! ---- Mercator config (matches Python MERC_KW with knowni/j=3,7) ----
  real :: m_tl1, m_lat1, m_lon1, m_dx, m_ki, m_kj
  real :: m_dlon, m_rsw
  ! ---- Polar config (matches Python PS_KW with knowni/j=4,9) ----
  real :: p_tl1, p_stdlon, p_lat1, p_lon1, p_dx, p_ki, p_kj
  real :: p_hemi, p_rebydx, p_scale_top, p_polei, p_polej
  ! ---- Polar SH config ----
  real :: s_tl1, s_stdlon, s_lat1, s_lon1, s_dx, s_ki, s_kj
  real :: s_hemi, s_rebydx, s_scale_top, s_polei, s_polej

  real :: lat, lon, i, j, ala1, alo1
  integer :: k
  real, dimension(6) :: tlat = (/ 0.0, 28.3, -15.0, 45.0, 60.0, -55.0 /)
  real, dimension(6) :: tlon = (/ -30.0, -16.4, 10.0, -100.0, 120.0, 5.0 /)

  ! ===== Mercator setup (set_merc) =====
  m_tl1=20.0; m_lat1=0.0; m_lon1=-30.0; m_dx=12000.0; m_ki=3.0; m_kj=7.0
  m_dlon = m_dx / (RE * cos(RAD_PER_DEG*m_tl1))
  m_rsw = 0.
  if (m_lat1 .ne. 0.) m_rsw = (alog(tan(0.5*((m_lat1+90.)*RAD_PER_DEG))))/m_dlon

  ! ===== Polar NH setup (set_ps) =====
  p_tl1=60.0; p_stdlon=-90.0; p_lat1=45.0; p_lon1=-120.0; p_dx=25000.0; p_ki=4.0; p_kj=9.0
  if (p_tl1 .lt. 0.) then; p_hemi=-1.0; else; p_hemi=1.0; end if
  p_rebydx = RE / p_dx
  p_scale_top = 1. + p_hemi*sin(p_tl1*RAD_PER_DEG)
  ala1 = p_lat1*RAD_PER_DEG
  p_rebydx = RE / p_dx
  ! rsw to SW corner -> pole point
  call ps_pole(p_hemi,p_rebydx,p_scale_top,p_lat1,p_lon1,p_stdlon,p_ki,p_kj,p_polei,p_polej)

  ! ===== Polar SH setup =====
  s_tl1=-71.0; s_stdlon=0.0; s_lat1=-60.0; s_lon1=0.0; s_dx=30000.0; s_ki=1.0; s_kj=1.0
  if (s_tl1 .lt. 0.) then; s_hemi=-1.0; else; s_hemi=1.0; end if
  s_rebydx = RE / s_dx
  s_scale_top = 1. + s_hemi*sin(s_tl1*RAD_PER_DEG)
  call ps_pole(s_hemi,s_rebydx,s_scale_top,s_lat1,s_lon1,s_stdlon,s_ki,s_kj,s_polei,s_polej)

  ! ----- Forward: Mercator -----
  do k=1,6
     lat=tlat(k); lon=tlon(k)
     call llij_merc(lat,lon,m_lon1,m_dlon,m_rsw,m_ki,m_kj,i,j)
     write(*,'(A,I1,A,2F18.10)') 'MERC_FWD ',k,' ',i,j
  end do
  ! ----- Inverse: Mercator (round-trip the forward i/j) -----
  do k=1,6
     lat=tlat(k); lon=tlon(k)
     call llij_merc(lat,lon,m_lon1,m_dlon,m_rsw,m_ki,m_kj,i,j)
     call ijll_merc(i,j,m_lon1,m_dlon,m_rsw,m_ki,m_kj,lat,lon)
     write(*,'(A,I1,A,2F18.10)') 'MERC_INV ',k,' ',lat,lon
  end do

  ! ----- Forward: Polar NH -----
  do k=1,6
     lat=tlat(k); lon=tlon(k)
     call llij_ps(lat,lon,p_hemi,p_rebydx,p_scale_top,p_stdlon,p_tl1,p_polei,p_polej,i,j)
     write(*,'(A,I1,A,2F18.10)') 'PSNH_FWD ',k,' ',i,j
  end do
  do k=1,6
     lat=tlat(k); lon=tlon(k)
     call llij_ps(lat,lon,p_hemi,p_rebydx,p_scale_top,p_stdlon,p_tl1,p_polei,p_polej,i,j)
     call ijll_ps(i,j,p_hemi,p_rebydx,p_scale_top,p_stdlon,p_polei,p_polej,lat,lon)
     write(*,'(A,I1,A,2F18.10)') 'PSNH_INV ',k,' ',lat,lon
  end do

  ! pole point
  write(*,'(A,2F18.10)') 'PSNH_POLE_IJ ', p_polei, p_polej
  call llij_ps(90.0,0.0,p_hemi,p_rebydx,p_scale_top,p_stdlon,p_tl1,p_polei,p_polej,i,j)
  write(*,'(A,2F18.10)') 'PSNH_FWD_POLE ', i, j

contains

  subroutine ps_pole(hemi,rebydx,scale_top,lat1,lon1,stdlon,ki,kj,polei,polej)
    real, intent(in) :: hemi,rebydx,scale_top,lat1,lon1,stdlon,ki,kj
    real, intent(out) :: polei,polej
    real :: reflon, ala1, alo1, rsw
    reflon = stdlon + 90.
    ala1 = lat1*RAD_PER_DEG
    rsw = rebydx*cos(ala1)*scale_top/(1.+hemi*sin(ala1))
    alo1 = (lon1-reflon)*RAD_PER_DEG
    polei = ki - rsw*cos(alo1)
    polej = kj - hemi*rsw*sin(alo1)
  end subroutine ps_pole

  subroutine llij_merc(lat,lon,lon1,dlon,rsw,ki,kj,i,j)
    real, intent(in) :: lat,lon,lon1,dlon,rsw,ki,kj
    real, intent(out) :: i,j
    real :: deltalon
    deltalon = lon - lon1
    if (deltalon .lt. -180.) deltalon = deltalon + 360.
    if (deltalon .gt. 180.) deltalon = deltalon - 360.
    i = ki + (deltalon/(dlon*DEG_PER_RAD))
    j = kj + (alog(tan(0.5*((lat+90.)*RAD_PER_DEG))))/dlon - rsw
  end subroutine llij_merc

  subroutine ijll_merc(i,j,lon1,dlon,rsw,ki,kj,lat,lon)
    real, intent(in) :: i,j,lon1,dlon,rsw,ki,kj
    real, intent(out) :: lat,lon
    lat = 2.0*atan(exp(dlon*(rsw+j-kj)))*DEG_PER_RAD - 90.
    lon = (i-ki)*dlon*DEG_PER_RAD + lon1
    if (lon .gt. 180.) lon = lon - 360.
    if (lon .lt. -180.) lon = lon + 360.
  end subroutine ijll_merc

  subroutine llij_ps(lat,lon,hemi,rebydx,scale_top,stdlon,tl1,polei,polej,i,j)
    real, intent(in) :: lat,lon,hemi,rebydx,scale_top,stdlon,tl1,polei,polej
    real, intent(out) :: i,j
    real :: reflon, ala, alo, rm
    reflon = stdlon + 90.
    ala = lat*RAD_PER_DEG
    rm = rebydx*cos(ala)*scale_top/(1.+hemi*sin(ala))
    alo = (lon-reflon)*RAD_PER_DEG
    i = polei + rm*cos(alo)
    j = polej + hemi*rm*sin(alo)
  end subroutine llij_ps

  subroutine ijll_ps(i,j,hemi,rebydx,scale_top,stdlon,polei,polej,lat,lon)
    real, intent(in) :: i,j,hemi,rebydx,scale_top,stdlon,polei,polej
    real, intent(out) :: lat,lon
    real :: reflon, xx, yy, r2, gi2, arccos
    reflon = stdlon + 90.
    xx = i - polei
    yy = (j - polej)*hemi
    r2 = xx**2 + yy**2
    if (r2 .eq. 0.) then
       lat = hemi*90.
       lon = reflon
    else
       gi2 = (rebydx*scale_top)**2.
       lat = DEG_PER_RAD*hemi*asin((gi2-r2)/(gi2+r2))
       arccos = acos(xx/sqrt(r2))
       if (yy .gt. 0) then
          lon = reflon + DEG_PER_RAD*arccos
       else
          lon = reflon - DEG_PER_RAD*arccos
       end if
    end if
    if (lon .gt. 180.) lon = lon - 360.
    if (lon .lt. -180.) lon = lon + 360.
  end subroutine ijll_ps

end program wrf_proj_oracle
