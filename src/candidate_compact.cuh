#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

// Correctness-first cap: the current supported path allows seg_len up to 10000,
// so compaction must be able to retain the full segment when radix reduction
// is not yet selective enough.
inline constexpr int kCompactedCandidateCap = 10000;

inline size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

struct CandidateCompactionWorkspaceView {
  unsigned int* histograms = nullptr;
  SegmentSelectState* states = nullptr;
  int* candidate_counts = nullptr;
  Candidate* candidates = nullptr;
};

inline size_t candidate_compaction_workspace_bytes(int seg_num) {
  if (seg_num <= 0) {
    return 0u;
  }

  size_t offset = 0u;
  offset = align_up(offset, alignof(unsigned int));
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  offset = align_up(offset, alignof(SegmentSelectState));
  offset += static_cast<size_t>(seg_num) * sizeof(SegmentSelectState);
  offset = align_up(offset, alignof(int));
  offset += static_cast<size_t>(seg_num) * sizeof(int);
  offset = align_up(offset, alignof(Candidate));
  offset += static_cast<size_t>(seg_num) * kCompactedCandidateCap * sizeof(Candidate);
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
  view.histograms = reinterpret_cast<unsigned int*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  offset = align_up(offset, alignof(SegmentSelectState));
  view.states = reinterpret_cast<SegmentSelectState*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * sizeof(SegmentSelectState);
  offset = align_up(offset, alignof(int));
  view.candidate_counts = reinterpret_cast<int*>(
      static_cast<std::byte*>(workspace) + offset);
  offset += static_cast<size_t>(seg_num) * sizeof(int);
  offset = align_up(offset, alignof(Candidate));
  view.candidates = reinterpret_cast<Candidate*>(
      static_cast<std::byte*>(workspace) + offset);
  return view;
}

__global__ inline void compact_candidates_kernel(const half* input,
                                                 int seg_len,
                                                 const SegmentSelectState* states,
                                                 Candidate* candidates,
                                                 int* candidate_counts) {
  const int seg = blockIdx.x;
  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  Candidate* segment_candidates =
      candidates + static_cast<size_t>(seg) * kCompactedCandidateCap;

  __shared__ int block_count;
  if (threadIdx.x == 0) {
    block_count = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    const half value = segment_input[i];
    const uint16_t encoded = encode_half_desc(value);
    if ((encoded & state.prefix_mask) <= state.prefix) {
      const int slot = atomicAdd(&block_count, 1);
      if (slot < kCompactedCandidateCap) {
        segment_candidates[slot] = Candidate{encoded, value, i};
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    candidate_counts[seg] =
        block_count < kCompactedCandidateCap ? block_count : kCompactedCandidateCap;
  }
}

}  // namespace radix_topk
