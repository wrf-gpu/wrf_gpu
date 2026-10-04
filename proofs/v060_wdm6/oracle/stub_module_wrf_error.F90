! Minimal stub of module_wrf_error for the WDM6 single-column oracle.
!
! The REAL WRF module_mp_radar.F (compiled UNMODIFIED here so that
! refl10cm_wdm6 + radar_init resolve their many internal symbols) `use`s
! module_wrf_error ONLY for wrf_debug. The full WRF module_wrf_error pulls in
! the DM/MPI stack; for a serial single-column oracle we provide no-op
! messaging shims. None of these feed the validated microphysics tendencies.
module module_wrf_error
  implicit none
  public
  character(len=256) :: wrf_err_message
contains
  subroutine wrf_debug(level, str)
    integer, intent(in) :: level
    character(len=*), intent(in) :: str
  end subroutine wrf_debug
  subroutine wrf_message(str)
    character(len=*), intent(in) :: str
  end subroutine wrf_message
  subroutine wrf_error_fatal(str)
    character(len=*), intent(in) :: str
    write(*,'(A)') 'WRF_ERROR_FATAL: '//trim(str)
    stop 9
  end subroutine wrf_error_fatal
end module module_wrf_error
