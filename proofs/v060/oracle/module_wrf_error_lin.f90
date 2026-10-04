! Minimal stub of module_wrf_error so module_mp_lin.F + module_mp_radar.F can be
! compiled and linked standalone for the v0.6.0 single-column Purdue-Lin oracle.
! The real WRF module routes messages through the model logging subsystem; for
! an offline single-column driver we just print to stdout / stop on fatal.
! ``wrf_err_message`` is the module-level message buffer the real scheme writes
! into before calling ``wrf_error_fatal`` (only on the ggamma overflow path,
! which the oracle soundings never hit). These stubs exist only so the
! unresolved externals resolve at link time and never alter scheme physics.
MODULE module_wrf_error
  IMPLICIT NONE
  CHARACTER(LEN=512) :: wrf_err_message
CONTAINS
  SUBROUTINE wrf_message(msg)
    CHARACTER(LEN=*), INTENT(IN) :: msg
    WRITE(*,'(A)') TRIM(msg)
  END SUBROUTINE wrf_message

  SUBROUTINE wrf_debug(level, msg)
    INTEGER, INTENT(IN) :: level
    CHARACTER(LEN=*), INTENT(IN) :: msg
    ! no-op
  END SUBROUTINE wrf_debug

  SUBROUTINE wrf_error_fatal(msg)
    CHARACTER(LEN=*), INTENT(IN) :: msg
    WRITE(*,'(A)') 'FATAL: '//TRIM(msg)
    STOP 1
  END SUBROUTINE wrf_error_fatal

  SUBROUTINE wrf_error_fatal3(fname, line, msg)
    CHARACTER(LEN=*), INTENT(IN) :: fname, msg
    INTEGER, INTENT(IN) :: line
    WRITE(*,'(A)') 'FATAL: '//TRIM(msg)
    STOP 1
  END SUBROUTINE wrf_error_fatal3
END MODULE module_wrf_error
