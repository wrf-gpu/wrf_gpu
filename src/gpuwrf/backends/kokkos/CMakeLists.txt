cmake_minimum_required(VERSION 3.22)
project(gpuwrf_m2_kokkos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED PATHS "${CMAKE_PREFIX_PATH}" NO_DEFAULT_PATH)
find_package(ZLIB REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(bench host.cpp stencil.cpp column.cpp)
target_link_libraries(bench PRIVATE Kokkos::kokkos ZLIB::ZLIB CUDA::cudart)
