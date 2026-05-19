#include <Kokkos_Core.hpp>

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using DoubleView = Kokkos::View<double*, ExecSpace>;
using ConstDoubleView = Kokkos::View<const double*, ExecSpace>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using MemberType = TeamPolicy::member_type;

constexpr int kTeamSize = 64;

struct ColumnThermoKernel {
    ConstDoubleView temperature_initial;
    ConstDoubleView qv_initial;
    ConstDoubleView pressure_initial;
    ConstDoubleView saturation_qv;
    DoubleView temperature_next;
    DoubleView qv_next;
    DoubleView pressure_next;
    DoubleView mse_delta;
    int levels;

    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType& team) const {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, levels), [&](const int k) {
            const double t0 = temperature_initial(k);
            const double q0 = qv_initial(k);
            const double p0 = pressure_initial(k);
            const double sat = saturation_qv(k);
            const double excess = Kokkos::fmax(q0 - sat, 0.0);
            const double deficit = Kokkos::fmax(0.72 * sat - q0, 0.0);
            const double condensation = 0.32 * excess;
            const double evaporation = Kokkos::fmin(0.04 * deficit, 0.18 * q0);
            const double q1 = Kokkos::fmax(q0 - condensation + evaporation, 1.0e-8);

            constexpr double cp_d = 1004.0;
            constexpr double lv = 2.5e6;
            const double latent_mass = condensation - evaporation;
            const double t1 = t0 + (lv / cp_d) * latent_mass;

            temperature_next(k) = t1;
            qv_next(k) = q1;
            pressure_next(k) = p0;
            mse_delta(k) = cp_d * (t1 - t0) + lv * (q1 - q0);
        });
    }
};

}  // namespace

void kokkos_column_thermo(
    ConstDoubleView temperature_initial,
    ConstDoubleView qv_initial,
    ConstDoubleView pressure_initial,
    ConstDoubleView saturation_qv,
    DoubleView temperature_next,
    DoubleView qv_next,
    DoubleView pressure_next,
    DoubleView mse_delta,
    int levels) {
    Kokkos::parallel_for(
        "ColumnThermoKernel",
        TeamPolicy(1, kTeamSize),
        ColumnThermoKernel{
            temperature_initial,
            qv_initial,
            pressure_initial,
            saturation_qv,
            temperature_next,
            qv_next,
            pressure_next,
            mse_delta,
            levels});
}

int kokkos_column_threads_per_team() {
    return kTeamSize;
}
