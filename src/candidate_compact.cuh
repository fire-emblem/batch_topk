#pragma once

#include <cstddef>
#include <cstdint>

#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

// Runtime candidate storage must support the full public API range (k <= 128).
// The optimized k=50 path still uses kOptimizedCandidateCap as its own target cap.
inline constexpr int kCompactedCandidateCap = kMaxSupportedK;

inline size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

struct CandidateCompactionWorkspaceView {
  unsigned int* histograms_hi = nullptr;
  unsigned int* histograms_lo = nullptr;
  unsigned int* partial_histograms = nullptr;
  SegmentSelectState* states = nullptr;
  int* candidate_counts = nullptr;
  int* candidate_indices = nullptr;
};

inline size_t candidate_compaction_workspace_bytes(int seg_num) {
  if (seg_num <= 0) {
    return 0u;
  }

  size_t offset = 0u;
  offset = align_up(offset, alignof(unsigned int));
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  offset += static_cast<size_t>(seg_num) * 4u * 256u * sizeof(unsigned int);
  offset = align_up(offset, alignof(SegmentSelectState));
  offset += static_cast<size_t>(seg_num) * sizeof(SegmentSelectState);
  offset = align_up(offset, alignof(int));
  offset += static_cast<size_t>(seg_num) * sizeof(int);
  offset += static_cast<size_t>(seg_num) * kCompactedCandidateCap * sizeof(int);
  return offset;
}

inline CandidateCompactionWorkspaceView make_candidate_compaction_workspace(void* workspace,
                                                                           int seg_num) {
  CandidateCompactionWorkspaceView view{};
  if (!workspace || seg_num <= 0) {
    return view;
  }

  size_t offset = 0u;
  offset = align_up(offset, alignof(unsigned int));
  view.histograms_hi = reinterpret_cast<unsigned int*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  view.histograms_lo = reinterpret_cast<unsigned int*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  view.partial_histograms = reinterpret_cast<unsigned int*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * 4u * 256u * sizeof(unsigned int);
  offset = align_up(offset, alignof(SegmentSelectState));
  view.states = reinterpret_cast<SegmentSelectState*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * sizeof(SegmentSelectState);
  offset = align_up(offset, alignof(int));
  view.candidate_counts = reinterpret_cast<int*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * sizeof(int);
  view.candidate_indices = reinterpret_cast<int*>(
      static_cast<std::byte*>(workspace) + offset);
  return view;
}

__global__ inline void compact_candidate_indices_kernel(const half* input,
                                                        int seg_len,
                                                        const SegmentSelectState* states,
                                                        int* candidate_indices,
                                                        int* candidate_counts) {
  using BlockScan = cub::BlockScan<int, 256>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int better_written;
  __shared__ int equal_written;
  __shared__ int shared_better_base;
  __shared__ int shared_equal_base;

  const int seg = blockIdx.x;
  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  int* segment_candidates =
      candidate_indices + static_cast<size_t>(seg) * kCompactedCandidateCap;

  if (threadIdx.x == 0) {
    better_written = 0;
    equal_written = 0;
  }
  __syncthreads();

  for (int base = 0; base < seg_len; base += blockDim.x) {
    const int idx = base + threadIdx.x;
    const bool in_range = idx < seg_len;
    const uint16_t key = in_range ? encode_half_desc(segment_input[idx]) : 0xffffu;

    const int better_flag = in_range && key < state.cutoff_key ? 1 : 0;
    int better_prefix = 0;
    int better_total = 0;
    BlockScan(scan_storage).ExclusiveSum(better_flag, better_prefix, better_total);
    __syncthreads();
    if (threadIdx.x == 0) {
      shared_better_base = better_written;
      better_written += better_total;
    }
    __syncthreads();
    if (better_flag) {
      const int better_rank = shared_better_base + better_prefix;
      if (better_rank < kCompactedCandidateCap) {
        segment_candidates[better_rank] = idx;
      }
    }
    __syncthreads();

    const int equal_flag = in_range && key == state.cutoff_key ? 1 : 0;
    int equal_prefix = 0;
    int equal_total = 0;
    BlockScan(scan_storage).ExclusiveSum(equal_flag, equal_prefix, equal_total);
    __syncthreads();
    if (threadIdx.x == 0) {
      shared_equal_base = equal_written;
      equal_written += equal_total;
    }
    __syncthreads();
    if (equal_flag) {
      const int equal_rank = shared_equal_base + equal_prefix;
      const int output_slot = state.strictly_better_count + equal_rank;
      if (equal_rank < state.remaining_slots && output_slot < kCompactedCandidateCap) {
        segment_candidates[output_slot] = idx;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const int total = state.strictly_better_count + state.remaining_slots;
    candidate_counts[seg] = total < kCompactedCandidateCap ? total : kCompactedCandidateCap;
  }
}

}  // namespace radix_topk
