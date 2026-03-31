#include "batch_topk.cuh"
#include "batch_topk_types.cuh"
#include "candidate_compact.cuh"
#include "final_block_sort.cuh"
#include "radix_histogram.cuh"
#include "radix_select_state.cuh"

#include <vector>

namespace radix_topk {

static bool is_supported_shape(int seg_num, int seg_len, int k) {
  return seg_num > 0 && seg_len > 0 && seg_len <= kMaxSupportedSegLen && k > 0 &&
         k <= kMaxSupportedK && k <= seg_len;
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

static SegmentSelectState select_state_from_histogram(const unsigned int* histogram,
                                                      int k) {
  SegmentSelectState state{};
  int selected_count = 0;
  for (int bucket = 0; bucket < 256; ++bucket) {
    const int bucket_count = static_cast<int>(histogram[static_cast<size_t>(bucket)]);
    if (selected_count + bucket_count < k) {
      selected_count += bucket_count;
      continue;
    }
    state.prefix = static_cast<uint16_t>(bucket << 8);
    state.prefix_mask = 0xff00u;
    state.selected_count = selected_count;
    state.live_count = bucket_count;
    state.boundary_digit = bucket;
    return state;
  }

  state.prefix = 0xffffu;
  state.prefix_mask = 0xffffu;
  state.selected_count = selected_count;
  state.live_count = 0;
  state.boundary_digit = 255;
  return state;
}

static bool build_segment_select_states(const unsigned int* histograms,
                                       int seg_num,
                                       int k,
                                       std::vector<SegmentSelectState>* states) {
  states->resize(static_cast<size_t>(seg_num));
  for (int seg = 0; seg < seg_num; ++seg) {
    const unsigned int* histogram = histograms + static_cast<size_t>(seg) * 256u;
    (*states)[static_cast<size_t>(seg)] = select_state_from_histogram(histogram, k);
  }
  return true;
}

size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k) {
  if (!is_supported_shape(seg_num, seg_len, k)) {
    return 0u;
  }
  return candidate_compaction_workspace_bytes(seg_num);
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

  const CandidateCompactionWorkspaceView workspace =
      make_candidate_compaction_workspace(d_workspace, seg_num);
  if (!workspace.histograms_hi || !workspace.states || !workspace.candidate_counts ||
      !workspace.candidate_indices) {
    return cudaErrorInvalidValue;
  }

  histogram_pass_kernel<<<seg_num, 256, 0, stream>>>(
      d_input, seg_len, 8, 0u, 0u, workspace.histograms_hi);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    return status;
  }

  std::vector<unsigned int> host_histograms(static_cast<size_t>(seg_num) * 256u);
  status = cudaMemcpy(host_histograms.data(),
                      workspace.histograms_hi,
                      sizeof(unsigned int) * host_histograms.size(),
                      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return status;
  }

  std::vector<SegmentSelectState> host_states;
  build_segment_select_states(host_histograms.data(), seg_num, k, &host_states);
  status = cudaMemcpy(workspace.states,
                      host_states.data(),
                      sizeof(SegmentSelectState) * host_states.size(),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    return status;
  }

  compact_candidates_kernel<<<seg_num, 256, 0, stream>>>(
      d_input, seg_len, workspace.states, workspace.candidate_indices,
      workspace.candidate_counts);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  final_candidate_sort_kernel<<<seg_num, 1, 0, stream>>>(
      d_input,
      seg_len,
      workspace.candidate_indices,
      workspace.candidate_counts,
      k,
      d_output_values,
      d_output_indices);
  return cudaGetLastError();
}

}  // namespace radix_topk
