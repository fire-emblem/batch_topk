#include <cstdio>

#include "batch_topk.cuh"

int main() {
  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(128, 10000, 50);
  alignas(16) half input[128] = {};
  alignas(16) half values[128] = {};
  int indices[128] = {};
  char workspace[128 * 256] = {};
  const cudaError_t status = radix_topk::batch_topk_half(
      input, 128, 10000, 50, values, indices, workspace, workspace_size, 0);
  if (status != cudaErrorNotSupported) {
    return 1;
  }
  puts("batch_topk smoke benchmark stub: implementation not enabled yet");
  return 0;
}
