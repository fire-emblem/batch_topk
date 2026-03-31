#pragma once

#include "batch_topk_types.cuh"

namespace radix_topk {

static_assert(kMaxSupportedK == 128, "direct baseline kernel expects 128-slot storage");

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

__global__ inline void direct_segment_topk_kernel(const half* input,
                                                  int seg_len,
                                                  int k,
                                                  half* output_values,
                                                  int* output_indices) {
  if (threadIdx.x != 0) {
    return;
  }

  const int seg = blockIdx.x;
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  half* segment_values = output_values + static_cast<size_t>(seg) * k;
  int* segment_indices = output_indices + static_cast<size_t>(seg) * k;

  Candidate topk[kMaxSupportedK];
  int count = 0;
  for (int i = 0; i < seg_len; ++i) {
    Candidate candidate{};
    candidate.encoded_key = encode_half_desc(segment_input[i]);
    candidate.value = segment_input[i];
    candidate.index = i;
    insert_candidate_sorted(topk, count, k, candidate);
  }

  for (int i = 0; i < k; ++i) {
    segment_values[i] = topk[i].value;
    segment_indices[i] = topk[i].index;
  }
}

}  // namespace radix_topk
