! Minimal serial stubs for the WRF distributed-memory externals referenced by
! the unmodified module_ra_rrtm.F / module_ra_clWRF_support.F when compiled
! standalone for the single-column classic-RRTM longwave oracle. The single
! column runs on one rank, so the monitor is always true and the broadcast is
! a no-op. These never alter the RRTM physics; they only resolve link symbols.
LOGICAL FUNCTION wrf_dm_on_monitor()
  wrf_dm_on_monitor = .TRUE.
END FUNCTION wrf_dm_on_monitor

SUBROUTINE wrf_dm_bcast_bytes(buf, nbytes)
  ! callers pass INTEGER scalars and REAL arrays through the same implicit
  ! external interface; compiled with -fallow-argument-mismatch. single-rank
  ! no-op broadcast.
  INTEGER :: buf
  INTEGER :: nbytes
END SUBROUTINE wrf_dm_bcast_bytes

! module_ra_rrtm.F calls wrf_message / wrf_debug as bare externals (the
! `USE module_wrf_error` is only inside rrtm_lookuptable's contained scope, not
! in mm5atm/rrtmlwrad), so provide free-standing externals here.
SUBROUTINE wrf_message(msg)
  CHARACTER(LEN=*), INTENT(IN) :: msg
  WRITE(*,'(A)') TRIM(msg)
END SUBROUTINE wrf_message

SUBROUTINE wrf_debug(level, msg)
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: msg
END SUBROUTINE wrf_debug

INTEGER FUNCTION get_unused_unit()
  INTEGER :: i
  LOGICAL :: opened
  get_unused_unit = -1
  DO i = 31, 99
    INQUIRE(i, OPENED=opened)
    IF (.NOT. opened) THEN
      get_unused_unit = i
      RETURN
    END IF
  END DO
END FUNCTION get_unused_unit
