program wrf_savepoint_instrumented
  use hdf5
  implicit none
  integer :: hdferr
  call h5open_f(hdferr)
  call h5close_f(hdferr)
  print *, "M6B0-R WRF_SAVEPOINT CPU emission shim"
  print *, "Savepoint extraction is orchestrated by scripts/m6b0r_wrf_savepoint_extract.py"
end program wrf_savepoint_instrumented

module savepoint_wrapper
  implicit none
contains
  subroutine sp_calc_coef_w_pre()
  end subroutine sp_calc_coef_w_pre
  subroutine sp_calc_coef_w_post()
  end subroutine sp_calc_coef_w_post
  subroutine sp_small_step_prep_post()
  end subroutine sp_small_step_prep_post
  subroutine sp_advance_mu_t_pre(rkstage, acstep, mu, mut, mudf, muts, muave, ww_in, theta_in)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: mu(:, :), mut(:, :), mudf(:, :), muts(:, :), muave(:, :)
    real, intent(in) :: ww_in(:, :, :), theta_in(:, :, :)
  end subroutine sp_advance_mu_t_pre
  subroutine sp_advance_mu_t_post(rkstage, acstep, mu, mut, mudf, muts, muave, ww_out, theta_out, ph_tend)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: mu(:, :), mut(:, :), mudf(:, :), muts(:, :), muave(:, :)
    real, intent(in) :: ww_out(:, :, :), theta_out(:, :, :), ph_tend(:, :, :)
  end subroutine sp_advance_mu_t_post
  subroutine sp_t_2ave_update_pre(rkstage, acstep, t_old, t_new, t_2ave)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: t_old(:, :, :), t_new(:, :, :), t_2ave(:, :, :)
  end subroutine sp_t_2ave_update_pre
  subroutine sp_t_2ave_update_post(rkstage, acstep, t_2ave)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: t_2ave(:, :, :)
  end subroutine sp_t_2ave_update_post
  subroutine sp_ww_update_pre(rkstage, acstep, ww_old, ww_new)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: ww_old(:, :, :), ww_new(:, :, :)
  end subroutine sp_ww_update_pre
  subroutine sp_ww_update_post(rkstage, acstep, ww_out)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: ww_out(:, :, :)
  end subroutine sp_ww_update_post
  subroutine sp_muave_update_pre(rkstage, acstep, mu_old, mu_new, mut, muave, muts)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: mu_old(:, :), mu_new(:, :), mut(:, :), muave(:, :), muts(:, :)
  end subroutine sp_muave_update_pre
  subroutine sp_muave_update_post(rkstage, acstep, muave, muts)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: muave(:, :), muts(:, :)
  end subroutine sp_muave_update_post
  subroutine sp_ph_tend_accumulate_pre(rkstage, acstep, ph_tend, ph_tend_increment)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: ph_tend(:, :, :), ph_tend_increment(:, :, :)
  end subroutine sp_ph_tend_accumulate_pre
  subroutine sp_ph_tend_accumulate_post(rkstage, acstep, ph_tend)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: ph_tend(:, :, :)
  end subroutine sp_ph_tend_accumulate_post
  subroutine sp_substep_save_state_pre(rkstage, acstep, u, v, w, t, ph, mu, ww)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: u(:, :, :), v(:, :, :), w(:, :, :), t(:, :, :), ph(:, :, :), ww(:, :, :)
    real, intent(in) :: mu(:, :)
  end subroutine sp_substep_save_state_pre
  subroutine sp_substep_save_state_post(rkstage, acstep, u_save, v_save, w_save, t_save, ph_save, mu_save, ww_save)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: u_save(:, :, :), v_save(:, :, :), w_save(:, :, :), t_save(:, :, :)
    real, intent(in) :: ph_save(:, :, :), ww_save(:, :, :)
    real, intent(in) :: mu_save(:, :)
  end subroutine sp_substep_save_state_post
  subroutine sp_acoustic_substep_complete(rkstage, substep, mu, mut, mudf, muts, muave, ww, theta, ph_tend, u, v, w, ph, p, t_2ave)
    integer, intent(in) :: rkstage, substep
    real, intent(in) :: mu(:, :), mut(:, :), mudf(:, :), muts(:, :), muave(:, :)
    real, intent(in) :: ww(:, :, :), theta(:, :, :), ph_tend(:, :, :)
    real, intent(in) :: u(:, :, :), v(:, :, :), w(:, :, :), ph(:, :, :), p(:, :, :), t_2ave(:, :, :)
  end subroutine sp_acoustic_substep_complete
  subroutine sp_acoustic_loop_complete(rkstage, mu, mut, mudf, muts, muave, ww, theta, ph_tend, u, v, w, ph, p, t_2ave)
    integer, intent(in) :: rkstage
    real, intent(in) :: mu(:, :), mut(:, :), mudf(:, :), muts(:, :), muave(:, :)
    real, intent(in) :: ww(:, :, :), theta(:, :, :), ph_tend(:, :, :)
    real, intent(in) :: u(:, :, :), v(:, :, :), w(:, :, :), ph(:, :, :), p(:, :, :), t_2ave(:, :, :)
  end subroutine sp_acoustic_loop_complete
  subroutine sp_advance_uv_post()
  end subroutine sp_advance_uv_post
  subroutine sp_advance_w_rhs_ready()
  end subroutine sp_advance_w_rhs_ready
  subroutine sp_advance_w_raw_w()
  end subroutine sp_advance_w_raw_w
  subroutine sp_advance_w_tridiag_fwd_pre(rkstage, acstep, a, alpha, gamma, rhs)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: a(:, :, :), alpha(:, :, :), gamma(:, :, :), rhs(:, :)
  end subroutine sp_advance_w_tridiag_fwd_pre
  subroutine sp_advance_w_tridiag_fwd_post(rkstage, acstep, a, alpha, gamma, w_fwd)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: a(:, :, :), alpha(:, :, :), gamma(:, :, :), w_fwd(:, :, :)
  end subroutine sp_advance_w_tridiag_fwd_post
  subroutine sp_advance_w_tridiag_back_pre(rkstage, acstep, gamma, w_fwd)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: gamma(:, :, :), w_fwd(:, :, :)
  end subroutine sp_advance_w_tridiag_back_pre
  subroutine sp_advance_w_tridiag_back_post(rkstage, acstep, gamma, w_solved)
    integer, intent(in) :: rkstage, acstep
    real, intent(in) :: gamma(:, :, :), w_solved(:, :, :)
  end subroutine sp_advance_w_tridiag_back_post
  subroutine sp_advance_w_rayleigh()
  end subroutine sp_advance_w_rayleigh
  subroutine sp_advance_w_ph_final()
  end subroutine sp_advance_w_ph_final
  subroutine sp_calc_p_rho_post()
  end subroutine sp_calc_p_rho_post
  subroutine sp_small_step_finish_post()
  end subroutine sp_small_step_finish_post
  subroutine sp_acoustic_substep_boundary()
  end subroutine sp_acoustic_substep_boundary
  subroutine sp_rk_stage_boundary()
  end subroutine sp_rk_stage_boundary
end module savepoint_wrapper
