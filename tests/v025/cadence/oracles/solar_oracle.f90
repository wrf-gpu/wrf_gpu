program solar_oracle
implicit none
real :: julian,xtime,gmt,radt,degrad,declin,solcon,dpd
real :: xlat(1,1),xlon(1,1),coszen(1,1),hrang(1,1)
integer :: ios
degrad=acos(-1.)/180.;dpd=360./365.
do
read(*,*,iostat=ios) julian,xtime,gmt,radt,xlat(1,1),xlon(1,1)
if(ios/=0)exit
call radconst(xtime,declin,solcon,julian,degrad,dpd)
call calc_coszen(1,1,1,1,1,1,1,1,julian,xtime+radt*.5,gmt,declin,degrad,xlon,xlat,coszen,hrang)
print *, declin,coszen(1,1),hrang(1,1),solcon
enddo
end program solar_oracle
   SUBROUTINE radconst(XTIME,DECLIN,SOLCON,JULIAN,                   &
                       DEGRAD,DPD                                    )
!---------------------------------------------------------------------
! module_wrf_error unused by extracted radconst
   IMPLICIT NONE
!---------------------------------------------------------------------

! !ARGUMENTS:
   REAL, INTENT(IN   )      ::       DEGRAD,DPD,XTIME,JULIAN
   REAL, INTENT(OUT  )      ::       DECLIN,SOLCON
   REAL                     ::       OBECL,SINOB,SXLONG,ARG,  &
                                     DECDEG,DJUL,RJUL,ECCFAC
!
! !DESCRIPTION:
! Compute terms used in radiation physics 
!EOP

! for short wave radiation

   DECLIN=0.
   SOLCON=0.

!-----OBECL : OBLIQUITY = 23.5 DEGREE.
        
   OBECL=23.5*DEGRAD
   SINOB=SIN(OBECL)
        
!-----CALCULATE LONGITUDE OF THE SUN FROM VERNAL EQUINOX:
        
   IF(JULIAN.GE.80.)SXLONG=DPD*(JULIAN-80.)
   IF(JULIAN.LT.80.)SXLONG=DPD*(JULIAN+285.)
   SXLONG=SXLONG*DEGRAD
   ARG=SINOB*SIN(SXLONG)
   DECLIN=ASIN(ARG)
   DECDEG=DECLIN/DEGRAD
!----SOLAR CONSTANT ECCENTRICITY FACTOR (PALTRIDGE AND PLATT 1976)
   DJUL=JULIAN*360./365.
   RJUL=DJUL*DEGRAD
   ECCFAC=1.000110+0.034221*COS(RJUL)+0.001280*SIN(RJUL)+0.000719*  &
          COS(2*RJUL)+0.000077*SIN(2*RJUL)
   SOLCON=1370.*ECCFAC
   
   END SUBROUTINE radconst
   SUBROUTINE calc_coszen(ims,ime,jms,jme,its,ite,jts,jte,  &
                          julian,xtime,gmt, &
                          declin,degrad,xlon,xlat,coszen,hrang)
       ! Added Equation of Time correction : jararias, 2013/08/10
       implicit none
       integer, intent(in) :: ims,ime,jms,jme,its,ite,jts,jte
       real, intent(in)    :: julian,declin,xtime,gmt,degrad
       real, dimension(ims:ime,jms:jme), intent(in)    :: xlat,xlon
       real, dimension(ims:ime,jms:jme), intent(inout) :: coszen,hrang

       integer :: i,j
       real    :: da,eot,xt24,tloctm,xxlat

       da=6.2831853071795862*(julian-1)/365.
       eot=(0.000075+0.001868*cos(da)-0.032077*sin(da) &
            -0.014615*cos(2*da)-0.04089*sin(2*da))*(229.18)
       xt24=mod(xtime,1440.)+eot
       do j=jts,jte
          do i=its,ite
             tloctm=gmt+xt24/60.+xlon(i,j)/15.
             hrang(i,j)=15.*(tloctm-12.)*degrad
             xxlat=xlat(i,j)*degrad
             coszen(i,j)=sin(xxlat)*sin(declin) &
                        +cos(xxlat)*cos(declin) *cos(hrang(i,j))
             coszen(i, j) = min (max (coszen(i, j), -1.0), 1.0)
          enddo
       enddo
   END SUBROUTINE calc_coszen
