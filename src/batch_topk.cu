#include "batch_topk.cuh"
#include "batch_topk_types.cuh"
#include "candidate_compact.cuh"
#include "dispatch_policy.cuh"
#include "final_block_sort.cuh"
#include "final_topk50.cuh"
#include "radix_boundary_select.cuh"
#include "radix_histogram.cuh"
#include "radix_select_state.cuh"

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
  if (!workspace.histograms_hi || !workspace.histograms_lo ||
      !workspace.partial_histograms || !workspace.states || !workspace.candidate_counts ||
      !workspace.candidate_indices) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = cudaSuccess;
  if (seg_len == kOptimizedSegLen && k == kOptimizedK) {
    const int ctas_per_segment = histogram_ctas_per_segment(seg_num);
    if (ctas_per_segment < 1 ||
        ctas_per_segment > kPartialHistogramMaxCtasPerSegment) {
      // partial_histograms is provisioned for at most 4 split CTAs per segment.
      return cudaErrorInvalidValue;
    }

    if (ctas_per_segment == 1) {
      histogram_high_byte_topk50_kernel<<<seg_num, 256, 0, stream>>>(
          d_input, seg_len, workspace.histograms_hi);
    } else {
      histogram_high_byte_splitk_kernel<<<seg_num * ctas_per_segment, 256, 0, stream>>>(
          d_input, seg_num, seg_len, ctas_per_segment, workspace.partial_histograms);
      status = cudaGetLastError();
      if (status != cudaSuccess) {
        return status;
      }
      reduce_partial_histograms_kernel<<<seg_num, 256, 0, stream>>>(
          workspace.partial_histograms, seg_num, ctas_per_segment, workspace.histograms_hi);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }

    select_high_byte_boundary_kernel<<<(seg_num + 127) / 128, 128, 0, stream>>>(
        workspace.histograms_hi, seg_num, k, workspace.states);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }

    if (ctas_per_segment == 1) {
      histogram_low_byte_topk50_kernel<<<seg_num, 256, 0, stream>>>(
          d_input, seg_len, workspace.states, workspace.histograms_lo);
    } else {
      histogram_low_byte_splitk_kernel<<<seg_num * ctas_per_segment, 256, 0, stream>>>(
          d_input,
          seg_num,
          seg_len,
          ctas_per_segment,
          workspace.states,
          workspace.partial_histograms);
      status = cudaGetLastError();
      if (status != cudaSuccess) {
        return status;
      }
      reduce_partial_histograms_kernel<<<seg_num, 256, 0, stream>>>(
          workspace.partial_histograms, seg_num, ctas_per_segment, workspace.histograms_lo);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }

    finalize_cutoff_key_kernel<<<(seg_num + 127) / 128, 128, 0, stream>>>(
        workspace.histograms_lo, seg_num, k, workspace.states);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }

    compact_candidate_indices_topk50_warp_kernel<<<seg_num, 256, 0, stream>>>(
        d_input, seg_len, workspace.states, workspace.candidate_indices,
        workspace.candidate_counts);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }

    final_topk50_kernel<<<seg_num, kOptimizedCandidateCap, 0, stream>>>(
        d_input, seg_len, workspace.candidate_indices, workspace.candidate_counts,
        d_output_values, d_output_indices);
    return cudaGetLastError();
  }

  histogram_high_byte_kernel<<<seg_num, 256, 0, stream>>>(
      d_input, seg_len, workspace.histograms_hi);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  select_high_byte_boundary_kernel<<<(seg_num + 127) / 128, 128, 0, stream>>>(
      workspace.histograms_hi, seg_num, k, workspace.states);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  histogram_low_byte_kernel<<<seg_num, 256, 0, stream>>>(
      d_input, seg_len, workspace.states, workspace.histograms_lo);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  finalize_cutoff_key_kernel<<<(seg_num + 127) / 128, 128, 0, stream>>>(
      workspace.histograms_lo, seg_num, k, workspace.states);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  compact_candidate_indices_kernel<<<seg_num, 256, 0, stream>>>(
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
