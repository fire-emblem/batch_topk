#include "batch_topk.cuh"
#include "batch_topk_types.cuh"
#include "final_block_sort.cuh"

namespace radix_topk {

static bool is_supported_shape(int seg_num, int seg_len, int k) {
  return seg_num > 0 && seg_len > 0 && seg_len <= 10000 && k > 0 &&
         k <= 128 && k <= seg_len;
}

static bool has_valid_arguments(const half* d_input,
                                int seg_num,
                                int seg_len,
                                int k,
                                half* d_output_values,
                                int* d_output_indices,
                                void* d_workspace,
                                size_t workspace_bytes) {
  if (!is_supported_shape(seg_num, seg_len, k)) {
    return false;
  }
  if (!d_input || !d_output_values || !d_output_indices || !d_workspace) {
    return false;
  }
  return workspace_bytes >= batch_topk_half_workspace_size(seg_num, seg_len, k);
}

size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k) {
  if (!is_supported_shape(seg_num, seg_len, k)) {
    return 0u;
  }
  return static_cast<size_t>(seg_num) * 4096u;
}

cudaError_t batch_topk_half(const half* d_input,
                            int seg_num,
                            int seg_len,
                            int k,
                            half* d_output_values,
                            int* d_output_indices,
                            void* d_workspace,
                            size_t workspace_bytes,
                            cudaStream_t stream) {
  if (!has_valid_arguments(d_input,
                           seg_num,
                           seg_len,
                           k,
                           d_output_values,
                           d_output_indices,
                           d_workspace,
                           workspace_bytes)) {
    return cudaErrorInvalidValue;
  }
  (void)d_workspace;
  direct_segment_topk_kernel<<<seg_num, 1, 0, stream>>>(
      d_input, seg_len, k, d_output_values, d_output_indices);
  return cudaGetLastError();
}

}  // namespace radix_topk
