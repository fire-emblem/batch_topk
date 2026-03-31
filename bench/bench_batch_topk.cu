#include <cstdio>

#include "batch_topk.cuh"

int main() {
  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(128, 10000, 50);
  half* d_input = nullptr;
  half* d_values = nullptr;
  int* d_indices = nullptr;
  void* d_workspace = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 128) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_values), sizeof(half) * 128) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_indices), sizeof(int) * 128) != cudaSuccess ||
      cudaMalloc(&d_workspace, workspace_size) != cudaSuccess) {
    return 1;
  }
  const cudaError_t status = radix_topk::batch_topk_half(
      d_input, 128, 10000, 50, d_values, d_indices, d_workspace, workspace_size, 0);
  if (status != cudaErrorNotSupported) {
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return 1;
  }
  puts("batch_topk smoke benchmark stub: implementation not enabled yet");
  cudaFree(d_workspace);
  cudaFree(d_indices);
  cudaFree(d_values);
  cudaFree(d_input);
  return 0;
}
