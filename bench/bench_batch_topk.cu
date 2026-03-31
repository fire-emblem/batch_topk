#include <cstdio>

#include "batch_topk.cuh"

int main() {
  const int seg_num = 128;
  const int seg_len = 10000;
  const int k = 50;
  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(seg_num, seg_len, k);
  half* d_input = nullptr;
  half* d_values = nullptr;
  int* d_indices = nullptr;
  void* d_workspace = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * static_cast<size_t>(seg_num) * seg_len) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_values),
                 sizeof(half) * static_cast<size_t>(seg_num) * k) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_indices),
                 sizeof(int) * static_cast<size_t>(seg_num) * k) !=
          cudaSuccess ||
      cudaMalloc(&d_workspace, workspace_size) != cudaSuccess) {
    return 1;
  }
  const cudaError_t status = radix_topk::batch_topk_half(
      d_input, seg_num, seg_len, k, d_values, d_indices, d_workspace,
      workspace_size, 0);
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
