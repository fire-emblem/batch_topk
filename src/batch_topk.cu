#include "batch_topk.cuh"

namespace radix_topk {

size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k) {
  if (seg_num <= 0 || seg_len <= 0 || seg_len > 10000 || k <= 0 || k > 128) {
    return 0u;
  }
  return static_cast<size_t>(seg_num) * 256u;
}

cudaError_t batch_topk_half(const half* d_input,
                            int seg_num,
                            int seg_len,
                            int k,
                            half* d_output_values,
                            int* d_output_indices,
                            void* d_workspace,
                            size_t workspace_bytes,
                            cudaStream_t) {
  const size_t required_workspace =
      batch_topk_half_workspace_size(seg_num, seg_len, k);
  if (seg_num <= 0 || seg_len <= 0 || seg_len > 10000 || k <= 0 || k > 128) {
    return cudaErrorInvalidValue;
  }
  if (!d_input || !d_output_values || !d_output_indices || !d_workspace) {
    return cudaErrorInvalidValue;
  }
  if (workspace_bytes < required_workspace) {
    return cudaErrorInvalidValue;
  }
  return cudaErrorNotSupported;
}

}  // namespace radix_topk
