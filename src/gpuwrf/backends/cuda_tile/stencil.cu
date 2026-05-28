#include <cuda_runtime.h>

namespace {

constexpr int kTileX = 16;
constexpr int kTileY = 8;
constexpr int kHalo = 2;
__device__ __forceinline__ int wrap_index(int value, int size) {
    value %= size;
    return value < 0 ? value + size : value;
}

__device__ __forceinline__ double diffusivity_for_level(int k) {
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

__global__ void stencil_advdiff_kernel(
    const double* __restrict__ phi_initial,
    const float* __restrict__ u_face,
    const float* __restrict__ v_face,
    const float* __restrict__ w_face,
    double* __restrict__ phi_next,
    int nx,
    int ny,
    int nz,
    double dx,
    double dy,
    double dz,
    double dt) {
    extern __shared__ double tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int k = blockIdx.z;
    const int tile_w = blockDim.x + 2 * kHalo;
    const int tile_h = blockDim.y + 2 * kHalo;

    for (int local_y = ty; local_y < tile_h; local_y += blockDim.y) {
        const int y = wrap_index(by + local_y - kHalo, ny);
        for (int local_x = tx; local_x < tile_w; local_x += blockDim.x) {
            const int x = wrap_index(bx + local_x - kHalo, nx);
            tile[local_y * tile_w + local_x] = phi_initial[(k * ny + y) * nx + x];
        }
    }
    __syncthreads();

    const int x = bx + tx;
    const int y = by + ty;
    if (x >= nx || y >= ny || k >= nz) {
        return;
    }

    const int lx = tx + kHalo;
    const int ly = ty + kHalo;
    const int idx = (k * ny + y) * nx + x;
    const int kp = wrap_index(k + 1, nz);
    const int km = wrap_index(k - 1, nz);
    const double center = tile[ly * tile_w + lx];
    const double phi_xp1 = tile[ly * tile_w + lx + 1];
    const double phi_xp2 = tile[ly * tile_w + lx + 2];
    const double phi_xm1 = tile[ly * tile_w + lx - 1];
    const double phi_xm2 = tile[ly * tile_w + lx - 2];
    const double phi_yp1 = tile[(ly + 1) * tile_w + lx];
    const double phi_yp2 = tile[(ly + 2) * tile_w + lx];
    const double phi_ym1 = tile[(ly - 1) * tile_w + lx];
    const double phi_ym2 = tile[(ly - 2) * tile_w + lx];
    const double phi_zp1 = phi_initial[(kp * ny + y) * nx + x];
    const double phi_zm1 = phi_initial[(km * ny + y) * nx + x];

    const double ddx4 = (-phi_xp2 + 8.0 * phi_xp1 - 8.0 * phi_xm1 + phi_xm2) / (12.0 * dx);
    const double ddy4 = (-phi_yp2 + 8.0 * phi_yp1 - 8.0 * phi_ym1 + phi_ym2) / (12.0 * dy);
    const double ddz2 = (phi_zp1 - phi_zm1) / (2.0 * dz);
    const double lapx4 = (-phi_xp2 + 16.0 * phi_xp1 - 30.0 * center + 16.0 * phi_xm1 - phi_xm2) / (12.0 * dx * dx);
    const double lapy4 = (-phi_yp2 + 16.0 * phi_yp1 - 30.0 * center + 16.0 * phi_ym1 - phi_ym2) / (12.0 * dy * dy);
    const double lapz2 = (phi_zp1 - 2.0 * center + phi_zm1) / (dz * dz);

    const double u_mass = 0.5 * (static_cast<double>(u_face[(k * ny + y) * (nx + 1) + x]) +
                                 static_cast<double>(u_face[(k * ny + y) * (nx + 1) + x + 1]));
    const double v_mass = 0.5 * (static_cast<double>(v_face[(k * (ny + 1) + y) * nx + x]) +
                                 static_cast<double>(v_face[(k * (ny + 1) + y + 1) * nx + x]));
    const double w_mass = 0.5 * (static_cast<double>(w_face[(k * ny + y) * nx + x]) +
                                 static_cast<double>(w_face[((k + 1) * ny + y) * nx + x]));
    const double diffusivity = diffusivity_for_level(k);
    const double advection = u_mass * ddx4 + v_mass * ddy4 + w_mass * ddz2;
    const double diffusion = diffusivity * (lapx4 + lapy4 + lapz2);
    phi_next[idx] = center + dt * (-advection + diffusion);
}

}  // namespace

extern "C" cudaError_t launch_stencil_advdiff(
    const double* phi_initial,
    const float* u_face,
    const float* v_face,
    const float* w_face,
    double* phi_next,
    int nx,
    int ny,
    int nz,
    cudaStream_t stream) {
    const dim3 block(kTileX, kTileY, 1);
    const dim3 grid((nx + kTileX - 1) / kTileX, (ny + kTileY - 1) / kTileY, nz);
    const size_t shared_bytes = static_cast<size_t>(kTileX + 2 * kHalo) *
                                static_cast<size_t>(kTileY + 2 * kHalo) *
                                sizeof(double);
    stencil_advdiff_kernel<<<grid, block, shared_bytes, stream>>>(
        phi_initial, u_face, v_face, w_face, phi_next, nx, ny, nz, 900.0, 900.0, 120.0, 3.0);
    return cudaGetLastError();
}

extern "C" cudaError_t stencil_theoretical_occupancy_pct(double* occupancy_pct, int nx, int ny) {
    const dim3 block(kTileX, kTileY, 1);
    const size_t shared_bytes = static_cast<size_t>(kTileX + 2 * kHalo) *
                                static_cast<size_t>(kTileY + 2 * kHalo) *
                                sizeof(double);
    int blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, stencil_advdiff_kernel, block.x * block.y * block.z, shared_bytes);
    if (err != cudaSuccess) {
        return err;
    }
    int max_threads_per_sm = 0;
    err = cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    if (err != cudaSuccess) {
        return err;
    }
    *occupancy_pct = 100.0 * static_cast<double>(blocks_per_sm * block.x * block.y * block.z) /
                     static_cast<double>(max_threads_per_sm);
    (void)nx;
    (void)ny;
    return cudaSuccess;
}
