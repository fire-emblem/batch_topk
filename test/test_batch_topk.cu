#include <cassert>

#include "batch_topk.cuh"

int main() {
  assert(radix_topk::batch_topk_half_workspace_size(0, 10000, 50) == 0);
  assert(radix_topk::batch_topk_half_workspace_size(1, 10001, 50) == 0);
  assert(radix_topk::batch_topk_half_workspace_size(1, 10000, 129) == 0);

  alignas(16) half input[1] = {};
  alignas(16) half values[1] = {};
  int indices[1] = {};
  char workspace[256] = {};

  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(1, 10000, 50);
  assert(workspace_size > 0);
  assert(radix_topk::batch_topk_half(
             nullptr, 1, 10000, 50, values, indices, workspace, workspace_size, 0) ==
         cudaErrorInvalidValue);
  assert(radix_topk::batch_topk_half(
             input, 1, 10000, 50, nullptr, indices, workspace, workspace_size, 0) ==
         cudaErrorInvalidValue);
  assert(radix_topk::batch_topk_half(
             input, 1, 10000, 50, values, nullptr, workspace, workspace_size, 0) ==
         cudaErrorInvalidValue);
  assert(radix_topk::batch_topk_half(
             input, 1, 10000, 50, values, indices, nullptr, workspace_size, 0) ==
         cudaErrorInvalidValue);

  const cudaError_t status = radix_topk::batch_topk_half(
      input, 1, 10000, 50, values, indices, workspace, workspace_size, 0);
  assert(status == cudaErrorNotSupported);
  return 0;
}
