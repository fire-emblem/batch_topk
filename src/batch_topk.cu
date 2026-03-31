#include "batch_topk.cuh"

namespace radix_topk {

cudaError_t batch_topk_half(
    const half*,
    int,
    int,
    int,
    half*,
    int*,
    void*,
    size_t,
    cudaStream_t) {
  return cudaErrorNotSupported;
}

size_t batch_topk_half_workspace_size(int, int, int) {
  return 0;
}

}  // namespace radix_topk
