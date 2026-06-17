logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_bytes(buf, nbytes)
  implicit none
  integer, intent(in) :: nbytes
  integer :: buf(*)
  if (nbytes < 0) then
    return
  end if
end subroutine wrf_dm_bcast_bytes

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
  if (level < 0) then
    print *, trim(message)
  end if
end subroutine wrf_debug

subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  print *, 'WRF_FATAL: ', trim(message)
  stop 2
end subroutine wrf_error_fatal
