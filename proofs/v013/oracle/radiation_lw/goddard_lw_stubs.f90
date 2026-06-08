! Minimal serial stub modules for the WRF framework dependencies that the
! UNMODIFIED phys/module_ra_goddard.F references via `use` when compiled
! standalone for the GSFC/Goddard LONGWAVE single-column oracle.
!
! These resolve link/USE symbols ONLY -- they never alter the Goddard
! physics. The LW-no-aerosol oracle never reaches wrf_error_fatal (it is
! only called from the netcdf LUT readers / iskip validation, which the
! lwrad-only driver does not exercise).
!
! module_checkerror (phys/module_checkerror.F) is self-contained and is
! compiled from the pristine source directly by the build script; only
! module_wrf_error (frame/module_wrf_error.F, which pulls in ESMF + MPI)
! is stubbed here.

MODULE module_wrf_error
  IMPLICIT NONE
  INTEGER :: wrf_debug_level = 0
  CHARACTER(LEN=256) :: wrf_err_message
CONTAINS
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

  SUBROUTINE wrf_error_fatal3(file, line, msg)
    CHARACTER(LEN=*), INTENT(IN) :: file, msg
    INTEGER, INTENT(IN) :: line
    WRITE(*,'(A)') 'FATAL: '//TRIM(msg)
    STOP 1
  END SUBROUTINE wrf_error_fatal3
END MODULE module_wrf_error

! Serial DM (distributed-memory) externals referenced ONLY by the netcdf
! LUT-broadcast / monitor-detection paths in module_ra_goddard.F. The
! lwrad-only oracle never calls these; the stubs just resolve link symbols.
SUBROUTINE wrf_dm_bcast_bytes(buf, nbytes)
  INTEGER, INTENT(IN) :: nbytes
  REAL, INTENT(INOUT) :: buf(*)
END SUBROUTINE wrf_dm_bcast_bytes

LOGICAL FUNCTION wrf_dm_on_monitor() RESULT(is_monitor)
  is_monitor = .TRUE.
END FUNCTION wrf_dm_on_monitor

! Bare EXTERNAL logging symbols. module_checkerror.F calls wrf_message /
! wrf_error_fatal as plain external subroutines (NOT via the module), so
! the linker needs the un-mangled `wrf_message_` / `wrf_error_fatal_`.
SUBROUTINE wrf_message(msg)
  CHARACTER(LEN=*), INTENT(IN) :: msg
  WRITE(*,'(A)') TRIM(msg)
END SUBROUTINE wrf_message

SUBROUTINE wrf_error_fatal(msg)
  CHARACTER(LEN=*), INTENT(IN) :: msg
  WRITE(*,'(A)') 'FATAL: '//TRIM(msg)
  STOP 1
END SUBROUTINE wrf_error_fatal
