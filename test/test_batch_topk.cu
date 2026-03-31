#include <cassert>

#include "batch_topk.cuh"

int main() {
  assert(radix_topk::batch_topk_half_workspace_size(0, 10000, 50) == 0);
  assert(radix_topk::batch_topk_half_workspace_size(1, 10001, 50) == 0);
  assert(radix_topk::batch_topk_half_workspace_size(1, 10000, 129) == 0);
  assert(radix_topk::batch_topk_half_workspace_size(1, 10, 11) == 0);

  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(1, 10000, 50);
  assert(workspace_size > 0);

  half* d_input = nullptr;
  half* d_values = nullptr;
  int* d_indices = nullptr;
  void* d_workspace = nullptr;
  assert(cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half)) == cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&d_values), sizeof(half)) == cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&d_indices), sizeof(int)) == cudaSuccess);
  assert(cudaMalloc(&d_workspace, workspace_size) == cudaSuccess);

  assert(radix_topk::batch_topk_half(
             nullptr, 1, 10000, 50, d_values, d_indices, d_workspace, workspace_size, 0) ==
         cudaErrorInvalidValue);
  assert(radix_topk::batch_topk_half(
             d_input, 1, 10000, 50, nullptr, d_indices, d_workspace, workspace_size, 0) ==
         cudaErrorInvalidValue);
  assert(radix_topk::batch_topk_half(
             d_input, 1, 10000, 50, d_values, nullptr, d_workspace, workspace_size, 0) ==
         cudaErrorInvalidValue);
  assert(radix_topk::batch_topk_half(
             d_input, 1, 10000, 50, d_values, d_indices, nullptr, workspace_size, 0) ==
         cudaErrorInvalidValue);
  assert(radix_topk::batch_topk_half(
             d_input, 1, 10, 11, d_values, d_indices, d_workspace,
             radix_topk::batch_topk_half_workspace_size(1, 10, 11), 0) ==
         cudaErrorInvalidValue);

  const cudaError_t status = radix_topk::batch_topk_half(
      d_input, 1, 10000, 50, d_values, d_indices, d_workspace, workspace_size, 0);
  assert(status == cudaErrorNotSupported);

  assert(cudaFree(d_workspace) == cudaSuccess);
  assert(cudaFree(d_indices) == cudaSuccess);
  assert(cudaFree(d_values) == cudaSuccess);
  assert(cudaFree(d_input) == cudaSuccess);
  return 0;
}
