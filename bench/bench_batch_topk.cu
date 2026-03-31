#include "batch_topk.cuh"

int main() {
  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(128, 10000, 50);
  const cudaError_t status = radix_topk::batch_topk_half(
      nullptr, 128, 10000, 50, nullptr, nullptr, nullptr, workspace_size, 0);
  return status == cudaErrorNotSupported ? 0 : 1;
}
