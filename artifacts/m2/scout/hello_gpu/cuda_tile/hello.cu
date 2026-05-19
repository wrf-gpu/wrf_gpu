#include <cuda_runtime.h>

#include <iostream>
#include <vector>

__global__ void times_two(const float* x, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 2.0f;
}

int main() {
  constexpr int n = 4;
  std::vector<float> host_x{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> host_y(n, 0.0f);
  float* x = nullptr;
  float* y = nullptr;
  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));
  cudaMemcpy(x, host_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  times_two<<<1, 32>>>(x, y, n);
  cudaDeviceSynchronize();
  cudaMemcpy(host_y.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(x);
  cudaFree(y);

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "candidate=cuda_tile cuda_runtime=" << CUDART_VERSION << "\n";
  std::cout << "device=" << prop.name << "\n";
  std::cout << "result=[";
  for (int i = 0; i < n; ++i) {
    if (i) std::cout << ", ";
    std::cout << host_y[i];
    if (host_y[i] != static_cast<float>((i + 1) * 2)) return 1;
  }
  std::cout << "]\n";
  return 0;
}
