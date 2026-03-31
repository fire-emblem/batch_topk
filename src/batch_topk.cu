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

size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k) {
  return (seg_num > 0 && seg_len > 0 && k > 0) ? 1u : 0u;
}

}  // namespace radix_topk
