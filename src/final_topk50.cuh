#pragma once

#include <climits>

#include "batch_topk_types.cuh"
#include "candidate_compact.cuh"

namespace radix_topk {

__device__ inline void compare_swap(Candidate* items, int lhs, int rhs, bool ascending) {
  const bool lhs_better = candidate_better(items[lhs], items[rhs]);
  if ((ascending && lhs_better) || (!ascending && !lhs_better)) {
    const Candidate tmp = items[lhs];
    items[lhs] = items[rhs];
    items[rhs] = tmp;
  }
}

__global__ inline void final_topk50_kernel(const half* input,
                                           int seg_len,
                                           const int* candidate_indices,
                                           const int* candidate_counts,
                                           half* output_values,
                                           int* output_indices) {
  __shared__ Candidate items[kOptimizedCandidateCap];

  const int seg = blockIdx.x;
  const int lane = threadIdx.x;
  const int count = candidate_counts[seg];
  const int offset = seg * kCompactedCandidateCap;

  Candidate item{};
  if (lane < count) {
    const int index = candidate_indices[offset + lane];
    item.index = index;
    item.value = input[static_cast<size_t>(seg) * seg_len + index];
    item.encoded_key = encode_half_desc(item.value);
  } else {
    item.index = INT_MAX;
    item.value = __float2half(-INFINITY);
    item.encoded_key = 0xffffu;
  }
  if (lane < kOptimizedCandidateCap) {
    items[lane] = item;
  }
  __syncthreads();

  for (int size = 2; size <= kOptimizedCandidateCap; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      const int partner = lane ^ stride;
      if (lane < kOptimizedCandidateCap && partner < kOptimizedCandidateCap &&
          lane < partner) {
        // candidate_better defines "better" as larger value, then smaller index.
        // Using ascending for the upper half of each bitonic block pushes worse
        // items toward higher lanes so the final order is descending in that relation.
        const bool ascending = (lane & size) != 0;
        compare_swap(items, lane, partner, ascending);
      }
      __syncthreads();
    }
  }

  if (lane < kOptimizedK) {
    output_values[static_cast<size_t>(seg) * kOptimizedK + lane] = items[lane].value;
    output_indices[static_cast<size_t>(seg) * kOptimizedK + lane] = items[lane].index;
  }
}

}  // namespace radix_topk
