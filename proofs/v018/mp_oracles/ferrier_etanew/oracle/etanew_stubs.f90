! =====================================================================
! Minimal standalone stubs so the UNMODIFIED WRF Ferrier source
! phys/module_mp_etanew.F can be compiled+linked in isolation, without the
! full WRF framework / DM (distributed-memory) layer.
!
! module_mp_etanew.F has NO `USE` statements; it only references the GLOBAL
! externals wrf_debug, wrf_error_fatal, wrf_dm_on_monitor, wrf_dm_bcast_bytes.
! In a single-rank offline driver:
!   * wrf_dm_on_monitor() is always .TRUE. (rank 0),
!   * wrf_dm_bcast_bytes is a no-op (nothing to broadcast),
!   * wrf_debug is silent, wrf_error_fatal prints + stops.
! The scheme source is copied VERBATIM and is NOT edited.
! =====================================================================

SUBROUTINE wrf_debug( level, str )
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: str
END SUBROUTINE wrf_debug

SUBROUTINE wrf_error_fatal( str )
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: str
  WRITE(*,'(A)') 'WRF_ERROR_FATAL: '//TRIM(str)
  STOP 9
END SUBROUTINE wrf_error_fatal

LOGICAL FUNCTION wrf_dm_on_monitor()
  IMPLICIT NONE
  wrf_dm_on_monitor = .TRUE.
END FUNCTION wrf_dm_on_monitor

SUBROUTINE wrf_dm_bcast_bytes( buf, nbytes )
  IMPLICIT NONE
  ! single-rank: nothing to broadcast. ``buf`` is INTENT-agnostic here; the
  ! caller passes real/integer arrays by reference. We never touch it.
  INTEGER, INTENT(IN) :: nbytes
  REAL :: buf
END SUBROUTINE wrf_dm_bcast_bytes
