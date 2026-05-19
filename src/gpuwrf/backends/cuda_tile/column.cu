#include <cuda_runtime.h>

namespace {

__global__ void column_thermo_kernel(
    const double* __restrict__ temperature_initial,
    const double* __restrict__ qv_initial,
    const double* __restrict__ pressure_initial,
    const double* __restrict__ saturation_qv,
    double* __restrict__ temperature_next,
    double* __restrict__ qv_next,
    double* __restrict__ pressure_next,
    double* __restrict__ mse_delta,
    int levels) {
    extern __shared__ double shared[];
    double* sat_shared = shared;

    const int k = threadIdx.x;
    if (k < levels) {
        sat_shared[k] = saturation_qv[k];
    }
    __syncthreads();

    if (k >= levels) {
        return;
    }

    const double t0 = temperature_initial[k];
    const double q0 = qv_initial[k];
    const double p0 = pressure_initial[k];
    const double sat = sat_shared[k];
    const double excess = fmax(q0 - sat, 0.0);
    const double deficit = fmax(0.72 * sat - q0, 0.0);
    const double condensation = 0.32 * excess;
    const double evaporation = fmin(0.04 * deficit, 0.18 * q0);
    const double q1 = fmax(q0 - condensation + evaporation, 1.0e-8);

    constexpr double cp_d = 1004.0;
    constexpr double lv = 2.5e6;
    const double latent_mass = condensation - evaporation;
    const double t1 = t0 + (lv / cp_d) * latent_mass;

    temperature_next[k] = t1;
    qv_next[k] = q1;
    pressure_next[k] = p0;
    mse_delta[k] = cp_d * (t1 - t0) + lv * (q1 - q0);
}

}  // namespace

extern "C" cudaError_t launch_column_thermo(
    const double* temperature_initial,
    const double* qv_initial,
    const double* pressure_initial,
    const double* saturation_qv,
    double* temperature_next,
    double* qv_next,
    double* pressure_next,
    double* mse_delta,
    int levels,
    cudaStream_t stream) {
    const int threads = 64;
    column_thermo_kernel<<<1, threads, static_cast<size_t>(levels) * sizeof(double), stream>>>(
        temperature_initial,
        qv_initial,
        pressure_initial,
        saturation_qv,
        temperature_next,
        qv_next,
        pressure_next,
        mse_delta,
        levels);
    return cudaGetLastError();
}

extern "C" cudaError_t column_theoretical_occupancy_pct(double* occupancy_pct, int levels) {
    const int threads = 64;
    const size_t shared_bytes = static_cast<size_t>(levels) * sizeof(double);
    int blocks_per_sm = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, column_thermo_kernel, threads, shared_bytes);
    if (err != cudaSuccess) {
        return err;
    }
    int max_threads_per_sm = 0;
    err = cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    if (err != cudaSuccess) {
        return err;
    }
    *occupancy_pct = 100.0 * static_cast<double>(blocks_per_sm * threads) /
                     static_cast<double>(max_threads_per_sm);
    return cudaSuccess;
}
