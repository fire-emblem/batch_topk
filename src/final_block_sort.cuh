#pragma once

#include "candidate_compact.cuh"
#include "batch_topk_types.cuh"

namespace radix_topk {

static_assert(kMaxSupportedK == 128, "final sort kernel expects 128-slot storage");

__device__ inline void insert_candidate_sorted(Candidate* topk,
                                               int& count,
                                               int k,
                                               const Candidate& candidate) {
  if (count < k) {
    topk[count] = candidate;
    int pos = count;
    ++count;
    while (pos > 0 && candidate_better(topk[pos], topk[pos - 1])) {
      const Candidate tmp = topk[pos];
      topk[pos] = topk[pos - 1];
      topk[pos - 1] = tmp;
      --pos;
    }
    return;
  }

  if (!candidate_better(candidate, topk[count - 1])) {
    return;
  }

  topk[count - 1] = candidate;
  int pos = count - 1;
  while (pos > 0 && candidate_better(topk[pos], topk[pos - 1])) {
    const Candidate tmp = topk[pos];
    topk[pos] = topk[pos - 1];
    topk[pos - 1] = tmp;
    --pos;
  }
}

__global__ inline void final_candidate_sort_kernel(const half* input,
                                                   int seg_len,
                                                   const int* candidate_indices,
                                                   const int* candidate_counts,
                                                   int k,
                                                   half* output_values,
                                                   int* output_indices) {
  static_assert(kMaxSupportedK == 128, "final sort kernel relies on 128-slot storage");

  const int seg = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }

  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  const int* segment_candidate_indices =
      candidate_indices + static_cast<size_t>(seg) * kCompactedCandidateCap;
  half* segment_values = output_values + static_cast<size_t>(seg) * k;
  int* segment_indices = output_indices + static_cast<size_t>(seg) * k;

  Candidate topk[kMaxSupportedK];
  int count = 0;
  const int candidate_count = candidate_counts[seg];
  for (int i = 0; i < candidate_count; ++i) {
    const int index = segment_candidate_indices[i];
    const half value = segment_input[index];
    const Candidate candidate{encode_half_desc(value), value, index};
    insert_candidate_sorted(topk, count, k, candidate);
  }

  for (int i = 0; i < k; ++i) {
    segment_values[i] = topk[i].value;
    segment_indices[i] = topk[i].index;
  }
}

}  // namespace radix_topk
