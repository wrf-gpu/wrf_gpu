#include <Kokkos_Core.hpp>

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using DoubleView = Kokkos::View<double*, ExecSpace>;
using ConstDoubleView = Kokkos::View<const double*, ExecSpace>;
using ConstFloatView = Kokkos::View<const float*, ExecSpace>;

constexpr int kTileK = 1;
constexpr int kTileY = 8;
constexpr int kTileX = 16;

KOKKOS_INLINE_FUNCTION
int wrap_index(int value, int size) {
    value %= size;
    return value < 0 ? value + size : value;
}

KOKKOS_INLINE_FUNCTION
double diffusivity_for_level(int k) {
    constexpr double values[8] = {
        18.0,
        19.414213562373095,
        20.0,
        19.414213562373095,
        18.0,
        16.585786437626904,
        16.0,
        16.585786437626904,
    };
    return values[k & 7];
}

struct StencilAdvdiffKernel {
    ConstDoubleView phi_initial;
    ConstFloatView u_face;
    ConstFloatView v_face;
    ConstFloatView w_face;
    DoubleView phi_next;
    int nx;
    int ny;
    int nz;
    double dx;
    double dy;
    double dz;
    double dt;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int k, const int y, const int x) const {
        const int idx = (k * ny + y) * nx + x;
        const int xp1 = wrap_index(x + 1, nx);
        const int xp2 = wrap_index(x + 2, nx);
        const int xm1 = wrap_index(x - 1, nx);
        const int xm2 = wrap_index(x - 2, nx);
        const int yp1 = wrap_index(y + 1, ny);
        const int yp2 = wrap_index(y + 2, ny);
        const int ym1 = wrap_index(y - 1, ny);
        const int ym2 = wrap_index(y - 2, ny);
        const int kp = wrap_index(k + 1, nz);
        const int km = wrap_index(k - 1, nz);

        const double center = phi_initial(idx);
        const double phi_xp1 = phi_initial((k * ny + y) * nx + xp1);
        const double phi_xp2 = phi_initial((k * ny + y) * nx + xp2);
        const double phi_xm1 = phi_initial((k * ny + y) * nx + xm1);
        const double phi_xm2 = phi_initial((k * ny + y) * nx + xm2);
        const double phi_yp1 = phi_initial((k * ny + yp1) * nx + x);
        const double phi_yp2 = phi_initial((k * ny + yp2) * nx + x);
        const double phi_ym1 = phi_initial((k * ny + ym1) * nx + x);
        const double phi_ym2 = phi_initial((k * ny + ym2) * nx + x);
        const double phi_zp1 = phi_initial((kp * ny + y) * nx + x);
        const double phi_zm1 = phi_initial((km * ny + y) * nx + x);

        const double ddx4 = (-phi_xp2 + 8.0 * phi_xp1 - 8.0 * phi_xm1 + phi_xm2) / (12.0 * dx);
        const double ddy4 = (-phi_yp2 + 8.0 * phi_yp1 - 8.0 * phi_ym1 + phi_ym2) / (12.0 * dy);
        const double ddz2 = (phi_zp1 - phi_zm1) / (2.0 * dz);
        const double lapx4 =
            (-phi_xp2 + 16.0 * phi_xp1 - 30.0 * center + 16.0 * phi_xm1 - phi_xm2) / (12.0 * dx * dx);
        const double lapy4 =
            (-phi_yp2 + 16.0 * phi_yp1 - 30.0 * center + 16.0 * phi_ym1 - phi_ym2) / (12.0 * dy * dy);
        const double lapz2 = (phi_zp1 - 2.0 * center + phi_zm1) / (dz * dz);

        const double u_mass = 0.5 * (static_cast<double>(u_face((k * ny + y) * (nx + 1) + x)) +
                                     static_cast<double>(u_face((k * ny + y) * (nx + 1) + x + 1)));
        const double v_mass = 0.5 * (static_cast<double>(v_face((k * (ny + 1) + y) * nx + x)) +
                                     static_cast<double>(v_face((k * (ny + 1) + y + 1) * nx + x)));
        const double w_mass = 0.5 * (static_cast<double>(w_face((k * ny + y) * nx + x)) +
                                     static_cast<double>(w_face(((k + 1) * ny + y) * nx + x)));
        const double advection = u_mass * ddx4 + v_mass * ddy4 + w_mass * ddz2;
        const double diffusion = diffusivity_for_level(k) * (lapx4 + lapy4 + lapz2);
        phi_next(idx) = center + dt * (-advection + diffusion);
    }
};

}  // namespace

void kokkos_stencil_advdiff(
    ConstDoubleView phi_initial,
    ConstFloatView u_face,
    ConstFloatView v_face,
    ConstFloatView w_face,
    DoubleView phi_next,
    int nx,
    int ny,
    int nz) {
    using Policy = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;
    Kokkos::parallel_for(
        "StencilAdvdiffKernel",
        Policy({0, 0, 0}, {nz, ny, nx}, {kTileK, kTileY, kTileX}),
        StencilAdvdiffKernel{phi_initial, u_face, v_face, w_face, phi_next, nx, ny, nz, 900.0, 900.0, 120.0, 3.0});
}

int kokkos_stencil_threads_per_team() {
    return kTileK * kTileY * kTileX;
}
