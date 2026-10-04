program control_oracle
implicit none
integer :: itimestep, stepcu, stepra, ra_call_offset, ios
integer :: i,j,k,k_start,k_end,P_QI,P_QS,PARAM_FIRST_SCALAR
real :: dt,cudt_pass,curr_secs_pass,cudtacttime_pass,radt,curr_secs,radtacttime
real :: nca(1,1),rthcuten(1,1,1),rqvcuten(1,1,1),rqccuten(1,1,1)
real :: rqrcuten(1,1,1),rqicuten(1,1,1),rqscuten(1,1,1)
real :: rainc(1,1),pratec(1,1),qv_increment
logical :: decided,run_param,doing_adapt_dt,cu_run,rad_run
i=1; j=1; k_start=1; k_end=1; P_QI=1; P_QS=1; PARAM_FIRST_SCALAR=1
ra_call_offset=0; doing_adapt_dt=.false.
curr_secs=0; curr_secs_pass=0; radtacttime=0; cudtacttime_pass=0
 do
 read(*,*,iostat=ios) itimestep,dt,stepcu,stepra,cudt_pass,radt,nca(1,1),rqvcuten(1,1,1),pratec(1,1)
 if(ios/=0)exit
 rthcuten=0; rqccuten=0; rqrcuten=0; rqicuten=0; rqscuten=0; rainc=0
   decided = .FALSE.
   run_param = .FALSE.
   IF ( ( .NOT. decided ) .AND. &
        ( itimestep .EQ. 1 ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
   END IF

   IF ( ( .NOT. decided ) .AND. &
        ( ( cudt_pass .EQ. 0. ) .OR. ( stepcu .EQ. 1 ) ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
   END IF

   IF ( ( .NOT. decided ) .AND. &
        ( .NOT. doing_adapt_dt ) .AND. &
        ( MOD(itimestep,stepcu) .EQ. 0 ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
   END IF

   IF ( ( .NOT. decided ) .AND. &
        ( doing_adapt_dt ) .AND. &
        ( curr_secs_pass .GE. cudtacttime_pass ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
      cudtacttime_pass = curr_secs_pass + cudt_pass*60
   END IF


 cu_run=run_param
   run_param = .FALSE.
   decided = .FALSE.
   IF ( ( .NOT. decided ) .AND. &
        ( itimestep .EQ. 1 ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
   END IF

   IF ( ( .NOT. decided ) .AND. &
        ( ( radt .EQ. 0. ) .OR. ( stepra .EQ. 1 ) ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
   END IF

   IF ( ( .NOT. decided ) .AND. &
        ( .NOT. doing_adapt_dt ) .AND. &
        ( MOD(itimestep,stepra) .EQ. 1+ra_call_offset ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
   END IF

   IF ( ( .NOT. decided ) .AND. &
        ( doing_adapt_dt ) .AND. &
        ( curr_secs .GE. radtacttime ) ) THEN
      run_param   = .TRUE.
      decided     = .TRUE.
      radtacttime = curr_secs + radt*60
   END IF


 rad_run=run_param
 ! solve_em.F finishes Runge_Kutta_loop before advance_ppt: the incoming rate
 ! has already contributed to this step. This scalar records that order only.
 qv_increment=dt*rqvcuten(1,1,1)
      RAINC(I,J)  = RAINC(I,J) + PRATEC(I,J)*DT
           IF ( NCA(I,J) .GT. 0 ) THEN

              IF ( NINT(NCA(I,J) / DT) .LE. 1 ) THEN

              ! set tendency to zero
!                PRATEC(I,J)=0.
!                RAINCV(I,J)=0.
                 DO k = k_start,k_end
                    RTHCUTEN(i,k,j)=0.
                    RQVCUTEN(i,k,j)=0.
                    RQCCUTEN(i,k,j)=0.
                    RQRCUTEN(i,k,j)=0.
                    if (P_QI .ge. PARAM_FIRST_SCALAR) RQICUTEN(i,k,j)=0.
                    if (P_QS .ge. PARAM_FIRST_SCALAR) RQSCUTEN(i,k,j)=0.
                 ENDDO
              ENDIF

              NCA(I,J)=NCA(I,J)-DT ! Decrease NCA
!              NCA(I,J)=NCA(I,J)-1. ! Decrease NCA

           ENDIF
 print *, merge(1,0,cu_run),merge(1,0,rad_run),qv_increment,rqvcuten(1,1,1),nca(1,1),rainc(1,1)
 enddo
end program control_oracle
