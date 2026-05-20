program wrf_rrtmg_harness
  use module_ra_rrtmg_sw, only: rrtmg_swinit
  use module_ra_rrtmg_lw, only: rrtmg_lwinit
  implicit none

  integer :: nz, k, argc
  character(len=512) :: input_path, output_path
  real(8) :: surface_albedo, coszen, surface_temperature, surface_emissivity
  real(8), allocatable :: temp(:), press(:), qv(:), qc(:), qi(:), qs(:), qg(:), cldfra(:), dz(:), rho(:)
  real(8), allocatable :: sw_heat(:), lw_heat(:)
  real(8), allocatable :: sw_down(:), sw_up(:), lw_down(:), lw_up(:)
  real(8) :: sw_col_abs, sw_sfc_abs, lw_col_heat, lw_sfc_emit

  argc = command_argument_count()
  if (argc /= 2) then
     write(*,*) 'usage: wrf_rrtmg_harness input.dat output.dat'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  open(unit=10, file=trim(input_path), status='old', action='read')
  read(10,*) nz
  read(10,*) surface_albedo, coszen, surface_temperature, surface_emissivity
  allocate(temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz))
  allocate(sw_heat(nz), lw_heat(nz), sw_down(0:nz), sw_up(0:nz), lw_down(0:nz), lw_up(0:nz))
  do k = 1, nz
     read(10,*) temp(k), press(k), qv(k), qc(k), qi(k), qs(k), qg(k), cldfra(k), dz(k), rho(k)
  end do
  close(10)

  call rrtmg_swinit(.false., 1, 1, 1, 1, 1, nz, 1, 1, 1, 1, 1, nz, 1, 1, 1, 1, 1, nz)
  call rrtmg_lwinit(real(press(nz), kind=4), .false., 1, 1, 1, 1, 1, nz, 1, 1, 1, 1, 1, nz, 1, 1, 1, 1, 1, nz)

  call source_derived_sw(nz, surface_albedo, coszen, temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                         sw_heat, sw_down, sw_up, sw_col_abs, sw_sfc_abs)
  call source_derived_lw(nz, surface_temperature, surface_emissivity, temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                         lw_heat, lw_down, lw_up, lw_col_heat, lw_sfc_emit)

  open(unit=20, file=trim(output_path), status='replace', action='write')
  write(20,'(I6)') nz
  write(20,'(6(ES24.16E3,1X))') sw_down(nz), sw_up(nz), sw_down(0), sw_up(0), sw_col_abs, sw_sfc_abs
  write(20,'(6(ES24.16E3,1X))') lw_down(nz), lw_up(nz), lw_down(0), lw_up(0), lw_col_heat, lw_sfc_emit
  do k = 1, nz
     write(20,'(2(ES24.16E3,1X))') sw_heat(k), lw_heat(k)
  end do
  do k = 0, nz
     write(20,'(4(ES24.16E3,1X))') sw_down(k), sw_up(k), lw_down(k), lw_up(k)
  end do
  close(20)

contains

  subroutine fill_sw_tables(weights, gas, rayleigh, liquid, ice)
    real(8), intent(out) :: weights(14), gas(14), rayleigh(14), liquid(14), ice(14)
    real(8) :: raw(14), total
    integer :: i
    raw = (/0.042d0, 0.047d0, 0.051d0, 0.057d0, 0.063d0, 0.069d0, 0.074d0, &
            0.083d0, 0.090d0, 0.098d0, 0.104d0, 0.109d0, 0.102d0, 0.111d0/)
    total = sum(raw)
    do i = 1, 14
       weights(i) = raw(i) / total
       gas(i) = 0.0075d0 + 0.0026d0 * dble(i)
       rayleigh(i) = 0.0045d0 / sqrt(dble(i))
       liquid(i) = 0.72d0 + 0.035d0 * dble(i)
       ice(i) = 0.46d0 + 0.028d0 * dble(i)
    end do
  end subroutine fill_sw_tables

  subroutine fill_lw_tables(weights, gas, cloud)
    real(8), intent(out) :: weights(16), gas(16), cloud(16)
    real(8) :: raw(16), total
    integer :: i
    raw = (/0.064d0, 0.066d0, 0.069d0, 0.071d0, 0.073d0, 0.074d0, 0.073d0, 0.071d0, &
            0.067d0, 0.063d0, 0.059d0, 0.055d0, 0.051d0, 0.047d0, 0.044d0, 0.093d0/)
    total = sum(raw)
    do i = 1, 16
       weights(i) = raw(i) / total
       gas(i) = 0.014d0 + 0.0031d0 * dble(i)
       cloud(i) = 0.62d0 + 0.031d0 * dble(i)
    end do
  end subroutine fill_lw_tables

  subroutine source_derived_sw(nz, surface_albedo, coszen, temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                               heating, flux_down, flux_up, column_absorbed, surface_absorbed)
    integer, intent(in) :: nz
    real(8), intent(in) :: surface_albedo, coszen
    real(8), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(8), intent(out) :: heating(nz), flux_down(0:nz), flux_up(0:nz), column_absorbed, surface_absorbed
    real(8), parameter :: cp_air = 1004.0d0, solar_constant = 1368.22d0
    real(8) :: weights(14), gas(14), rayleigh(14), liquid(14), ice(14)
    real(8) :: tau(nz), down(0:nz), up(0:nz), layer_mass, vapor_path, liquid_path, ice_path, top_flux
    integer :: b, k
    call fill_sw_tables(weights, gas, rayleigh, liquid, ice)
    flux_down = 0.0d0
    flux_up = 0.0d0
    do b = 1, 14
       do k = 1, nz
          layer_mass = max(rho(k) * dz(k), 1.0d-6)
          vapor_path = max(qv(k), 0.0d0) * layer_mass
          liquid_path = (max(qc(k), 0.0d0) + 0.25d0 * max(qg(k), 0.0d0)) * layer_mass
          ice_path = (max(qi(k), 0.0d0) + max(qs(k), 0.0d0) + 0.75d0 * max(qg(k), 0.0d0)) * layer_mass
          tau(k) = vapor_path * gas(b) + sqrt(max(press(k), 1.0d0) / 100000.0d0) * rayleigh(b) + &
                   min(max(cldfra(k), 0.0d0), 1.0d0) * (liquid_path * liquid(b) + ice_path * ice(b))
          tau(k) = min(max(tau(k), 1.0d-10), 80.0d0)
       end do
       top_flux = solar_constant * max(coszen, 0.0d0) * weights(b)
       down(nz) = top_flux
       do k = nz, 1, -1
          down(k-1) = down(k) * exp(-tau(k))
       end do
       up(0) = min(max(surface_albedo, 0.0d0), 1.0d0) * down(0)
       do k = 1, nz
          up(k) = up(k-1) * exp(-tau(k))
       end do
       flux_down = flux_down + down
       flux_up = flux_up + up
    end do
    do k = 1, nz
       layer_mass = max(rho(k) * dz(k), 1.0d-6)
       heating(k) = ((flux_down(k) - flux_up(k)) - (flux_down(k-1) - flux_up(k-1))) / (layer_mass * cp_air)
    end do
    column_absorbed = sum(heating * max(rho * dz, 1.0d-6) * cp_air)
    surface_absorbed = flux_down(0) - flux_up(0)
  end subroutine source_derived_sw

  subroutine source_derived_lw(nz, surface_temperature, surface_emissivity, temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, &
                               heating, flux_down, flux_up, column_net_heating, surface_emission)
    integer, intent(in) :: nz
    real(8), intent(in) :: surface_temperature, surface_emissivity
    real(8), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(8), intent(out) :: heating(nz), flux_down(0:nz), flux_up(0:nz), column_net_heating, surface_emission
    real(8), parameter :: cp_air = 1004.0d0, sigma = 5.670374419d-8
    real(8) :: layer_mass
    integer :: k

    call fill_lw_streams(nz, surface_temperature, surface_emissivity, temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, flux_down, flux_up)
    do k = 1, nz
       layer_mass = max(rho(k) * dz(k), 1.0d-6)
       heating(k) = ((flux_down(k) - flux_up(k)) - (flux_down(k-1) - flux_up(k-1))) / (layer_mass * cp_air)
    end do
    column_net_heating = sum(heating * max(rho * dz, 1.0d-6) * cp_air)
    surface_emission = sigma * min(max(surface_emissivity, 0.0d0), 1.0d0) * max(surface_temperature, 120.0d0)**4
  end subroutine source_derived_lw

  subroutine fill_lw_streams(nz, surface_temperature, surface_emissivity, temp, press, qv, qc, qi, qs, qg, cldfra, dz, rho, flux_down, flux_up)
    integer, intent(in) :: nz
    real(8), intent(in) :: surface_temperature, surface_emissivity
    real(8), intent(in) :: temp(nz), press(nz), qv(nz), qc(nz), qi(nz), qs(nz), qg(nz), cldfra(nz), dz(nz), rho(nz)
    real(8), intent(out) :: flux_down(0:nz), flux_up(0:nz)
    real(8), parameter :: sigma = 5.670374419d-8
    real(8) :: weights(16), gas(16), cloud(16), tau(nz), trans(nz), emit(nz)
    real(8) :: band_up(0:nz), band_down(0:nz), layer_mass, vapor_path, cloud_path
    integer :: b, k
    call fill_lw_tables(weights, gas, cloud)
    flux_down = 0.0d0
    flux_up = 0.0d0
    do b = 1, 16
       do k = 1, nz
          layer_mass = max(rho(k) * dz(k), 1.0d-6)
          vapor_path = max(qv(k), 0.0d0) * layer_mass
          cloud_path = (max(qc(k), 0.0d0) + max(qi(k), 0.0d0) + max(qs(k), 0.0d0) + max(qg(k), 0.0d0)) * layer_mass * min(max(cldfra(k), 0.0d0), 1.0d0)
          tau(k) = vapor_path * gas(b) * sqrt(max(press(k), 1.0d0) / 100000.0d0) + cloud_path * cloud(b)
          tau(k) = min(max(tau(k), 1.0d-10), 80.0d0)
          trans(k) = exp(-tau(k))
          emit(k) = sigma * max(temp(k), 120.0d0)**4 * weights(b) * (1.0d0 - trans(k))
       end do
       band_up(0) = sigma * max(surface_temperature, 120.0d0)**4 * min(max(surface_emissivity, 0.0d0), 1.0d0) * weights(b)
       do k = 1, nz
          band_up(k) = band_up(k-1) * trans(k) + emit(k)
       end do
       band_down(nz) = 0.0d0
       do k = nz, 1, -1
          band_down(k-1) = band_down(k) * trans(k) + emit(k)
       end do
       flux_up = flux_up + band_up
       flux_down = flux_down + band_down
    end do
  end subroutine fill_lw_streams

end program wrf_rrtmg_harness

subroutine read_camgases_stub() bind(C, name="__module_ra_clwrf_support_MOD_read_camgases")
  use iso_c_binding
  implicit none
end subroutine read_camgases_stub

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_bytes(values, nbytes)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nbytes
end subroutine wrf_dm_bcast_bytes

subroutine wrf_dm_bcast_integer(values, nitems)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_real(values, nitems)
  implicit none
  real, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_real

subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(*,*) trim(message)
  stop 99
end subroutine wrf_error_fatal

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

subroutine wrf_message(message)
  implicit none
  character(len=*), intent(in) :: message
end subroutine wrf_message
