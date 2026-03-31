#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_fp16.h>

namespace radix_topk {

struct Candidate {
  uint32_t encoded_key = 0;
  half value = __float2half(0.0f);
  int index = 0;
};

struct ReferenceTopKResult {
  std::vector<float> values;
  std::vector<int> indices;
};

__host__ __device__ uint16_t normalize_half_bits(uint16_t bits);
__host__ __device__ uint16_t encode_half_asc(half value);
__host__ __device__ uint16_t encode_half_desc(half value);
__host__ __device__ bool candidate_better(const Candidate& lhs,
                                          const Candidate& rhs);

inline ReferenceTopKResult cpu_reference_topk(const std::vector<float>& values,
                                              int seg_len,
                                              int k) {
  ReferenceTopKResult result;
  if (seg_len <= 0 || k <= 0) {
    return result;
  }

  const int bounded_seg_len =
      std::min(seg_len, static_cast<int>(values.size()));
  if (bounded_seg_len <= 0) {
    return result;
  }

  const int actual_k = std::min(k, bounded_seg_len);
  std::vector<int> order(static_cast<size_t>(bounded_seg_len));
  for (int i = 0; i < bounded_seg_len; ++i) {
    order[static_cast<size_t>(i)] = i;
  }

  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    const float lhs_value = values[static_cast<size_t>(lhs)];
    const float rhs_value = values[static_cast<size_t>(rhs)];
    const bool lhs_nan = std::isnan(lhs_value);
    const bool rhs_nan = std::isnan(rhs_value);
    if (lhs_nan != rhs_nan) {
      return rhs_nan;
    }
    if (!lhs_nan && lhs_value != rhs_value) {
      return lhs_value > rhs_value;
    }
    return lhs < rhs;
  });

  result.values.reserve(static_cast<size_t>(actual_k));
  result.indices.reserve(static_cast<size_t>(actual_k));
  for (int i = 0; i < actual_k; ++i) {
    const int index = order[static_cast<size_t>(i)];
    result.values.push_back(values[static_cast<size_t>(index)]);
    result.indices.push_back(index);
  }
  return result;
}

}  // namespace radix_topk

#include "type_codec.cuh"
