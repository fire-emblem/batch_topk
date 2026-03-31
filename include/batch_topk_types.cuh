#pragma once

#include <algorithm>
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

inline ReferenceTopKResult cpu_reference_topk(const std::vector<float>& values,
                                              int seg_len,
                                              int k) {
  ReferenceTopKResult result;
  if (seg_len <= 0 || k <= 0) {
    return result;
  }

  const int actual_k = std::min(k, seg_len);
  std::vector<int> order(static_cast<size_t>(seg_len));
  for (int i = 0; i < seg_len; ++i) {
    order[static_cast<size_t>(i)] = i;
  }

  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    if (values[static_cast<size_t>(lhs)] != values[static_cast<size_t>(rhs)]) {
      return values[static_cast<size_t>(lhs)] > values[static_cast<size_t>(rhs)];
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
