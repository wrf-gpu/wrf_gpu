! Minimal serial stubs for the WRF logging externals referenced by the
! unmodified module_ra_gsfcsw.F when compiled standalone for the GSFC SW
! single-column oracle. These never alter the GSFC physics; they only
! resolve link symbols. The default operational path (no aerosol feedback,
! sane optical depths) never reaches wrf_error_fatal.
SUBROUTINE wrf_message(msg)
  CHARACTER(LEN=*), INTENT(IN) :: msg
  WRITE(*,'(A)') TRIM(msg)
END SUBROUTINE wrf_message

SUBROUTINE wrf_debug(level, msg)
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: msg
END SUBROUTINE wrf_debug

SUBROUTINE wrf_error_fatal(msg)
  CHARACTER(LEN=*), INTENT(IN) :: msg
  WRITE(*,'(A)') 'FATAL: '//TRIM(msg)
  STOP 1
END SUBROUTINE wrf_error_fatal
