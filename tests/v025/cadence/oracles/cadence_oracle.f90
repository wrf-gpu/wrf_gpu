! Extracted from pristine WRF solve_em.F and module_physics_init.F.
program oracle
implicit none
real :: dt, spacing, radt, cudt
integer :: num_sound_steps, stepra, stepcu, ios
do
 read(*,*,iostat=ios) dt, spacing, radt, cudt
 if (ios /= 0) exit
 num_sound_steps = max ( 2 * ( INT (300. * dt /  spacing             - 0.01 ) + 1 ), 4 )
 STEPRA = nint(RADT*60./DT)
 STEPRA = max(STEPRA,1)
 STEPCU = nint(CUDT*60./DT)
 STEPCU = max(STEPCU,1)
 print *, num_sound_steps, stepra, stepcu
end do
end program oracle
