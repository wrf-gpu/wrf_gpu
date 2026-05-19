#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float*> x("x", 4);
    Kokkos::View<float*> y("y", 4);
    auto x_host = Kokkos::create_mirror_view(x);
    for (int i = 0; i < 4; ++i) {
      x_host(i) = static_cast<float>(i + 1);
    }
    Kokkos::deep_copy(x, x_host);
    Kokkos::parallel_for(
        "times_two", Kokkos::RangePolicy<Kokkos::Cuda>(0, 4),
        KOKKOS_LAMBDA(const int i) { y(i) = x(i) * 2.0f; });
    Kokkos::fence();
    auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
    std::cout << "candidate=kokkos version=" << KOKKOS_VERSION << "\n";
    std::cout << "execution_space=" << Kokkos::DefaultExecutionSpace::name()
              << "\n";
    std::cout << "result=[";
    for (int i = 0; i < 4; ++i) {
      if (i) std::cout << ", ";
      std::cout << y_host(i);
      if (y_host(i) != static_cast<float>((i + 1) * 2)) return 1;
    }
    std::cout << "]\n";
  }
  Kokkos::finalize();
  return 0;
}
