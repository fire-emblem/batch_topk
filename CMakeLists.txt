cmake_minimum_required(VERSION 3.24)

project(batch_radix_topk LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(batch_topk STATIC
  src/batch_topk.cu
)
target_include_directories(batch_topk PUBLIC include)
target_link_libraries(batch_topk PUBLIC CUDA::cudart)

add_executable(test_batch_topk
  test/test_batch_topk.cu
)
target_link_libraries(test_batch_topk PRIVATE batch_topk)
add_test(NAME test_batch_topk COMMAND test_batch_topk)

add_executable(bench_batch_topk
  bench/bench_batch_topk.cu
)
target_link_libraries(bench_batch_topk PRIVATE batch_topk)
