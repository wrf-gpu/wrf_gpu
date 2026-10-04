SUBROUTINE wrf_message(msg)
  CHARACTER(LEN=*), INTENT(IN) :: msg
  WRITE(*,'(A)') TRIM(msg)
END SUBROUTINE wrf_message

SUBROUTINE wrf_debug(level, msg)
  INTEGER, INTENT(IN) :: level
  CHARACTER(LEN=*), INTENT(IN) :: msg
END SUBROUTINE wrf_debug

SUBROUTINE wrf_error_fatal3(fname, line, msg)
  CHARACTER(LEN=*), INTENT(IN) :: fname, msg
  INTEGER, INTENT(IN) :: line
  WRITE(0,'(A,A)') 'WRF_FATAL: ', TRIM(msg)
  STOP 2
END SUBROUTINE wrf_error_fatal3
