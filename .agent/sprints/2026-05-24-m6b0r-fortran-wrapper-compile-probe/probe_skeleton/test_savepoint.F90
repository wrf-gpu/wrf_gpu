! test_savepoint.F90 — drives savepoint_wrapper end-to-end + measures wall time
!
! Writes a deterministic 4x4x4 REAL(8) field to /tmp/wrapprobe_test.h5 once,
! then re-writes it `N_LOOPS` times to /tmp/wrapprobe_perf.h5 for a per-call
! wall-time estimate. The Python verifier reads /tmp/wrapprobe_test.h5.

program test_savepoint
   use savepoint_wrapper
   use iso_fortran_env, only: real64, int32, int64
   implicit none

   integer, parameter :: NX = 4, NY = 4, NZ = 4
   integer, parameter :: N_LOOPS = 200
   real(real64) :: arr(NX, NY, NZ)
   integer :: i, j, k, ierr
   integer(int64) :: tic, toc, rate
   real(real64) :: per_call_us

   ! Deterministic fill: arr(i,j,k) = 100*i + 10*j + k  (1-indexed)
   do k = 1, NZ
      do j = 1, NY
         do i = 1, NX
            arr(i, j, k) = 100.0_real64 * real(i, real64) &
                         +  10.0_real64 * real(j, real64) &
                         +              real(k, real64)
         end do
      end do
   end do

   ! ---- Correctness write (read back by verify_roundtrip.py) -----------------
   call sp_write_real8_3d('/tmp/wrapprobe_test.h5', &
                          name    = 'probe_field', &
                          units   = 'arbitrary',   &
                          stagger = 'C',           &
                          rkstage = 1,             &
                          acstep  = 0,             &
                          arr     = arr,           &
                          ierr_out = ierr)
   if (ierr /= 0) then
      write(*,'(A,I0)') 'FAIL: sp_write_real8_3d returned ierr=', ierr
      stop 1
   end if
   write(*,'(A)') 'OK: wrote /tmp/wrapprobe_test.h5'

   ! ---- Perf loop ------------------------------------------------------------
   call system_clock(count_rate=rate)
   call system_clock(tic)
   do i = 1, N_LOOPS
      call sp_write_real8_3d('/tmp/wrapprobe_perf.h5', &
                             name    = 'probe_field', &
                             units   = 'arbitrary',   &
                             stagger = 'C',           &
                             rkstage = 1,             &
                             acstep  = i,             &
                             arr     = arr,           &
                             ierr_out = ierr)
      if (ierr /= 0) then
         write(*,'(A,I0,A,I0)') 'FAIL on loop ', i, ' ierr=', ierr
         stop 2
      end if
   end do
   call system_clock(toc)

   per_call_us = 1.0e6_real64 * real(toc - tic, real64) / real(rate, real64) / real(N_LOOPS, real64)
   write(*,'(A,I0,A,F10.2,A)') 'PERF: ', N_LOOPS, ' calls, ', per_call_us, ' us/call'

end program test_savepoint
