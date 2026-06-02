! Minimal stub of module_wrf_error so module_cu_kfeta.F can be compiled and
! linked standalone for the P0-4 single-column KF-eta oracle.
! The real WRF module routes messages through the model logging subsystem;
! for an offline single-column driver we just print to stdout / stop on fatal.
! These stubs are NEVER exercised on the convecting code path of the oracle
! soundings (no OUT-OF-BOUNDS / mass-budget error is hit); they exist only so
! the unresolved externals resolve at link time.
MODULE module_wrf_error
  IMPLICIT NONE
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
