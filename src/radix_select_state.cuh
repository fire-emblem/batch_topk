#pragma once

#include <vector>

#include "batch_topk_types.cuh"

namespace radix_topk {

inline SegmentSelectState simulate_radix_boundary(const std::vector<float>& values,
                                                  int seg_len,
                                                  int k) {
  SegmentSelectState state{};
  if (seg_len <= 0 || k <= 0) {
    return state;
  }

  const int bounded_seg_len =
      std::min(seg_len, static_cast<int>(values.size()));
  if (bounded_seg_len <= 0) {
    return state;
  }

  std::vector<int> counts(256, 0);
  for (int i = 0; i < bounded_seg_len; ++i) {
    const half value = __float2half(values[static_cast<size_t>(i)]);
    const uint16_t encoded = encode_half_desc(value);
    ++counts[static_cast<size_t>((encoded >> 8) & 0xffu)];
  }

  const int actual_k = std::min(k, bounded_seg_len);
  int selected_count = 0;
  int boundary_digit = 0;
  int live_count = counts[0];
  for (int bucket = 0; bucket < 256; ++bucket) {
    const int bucket_count = counts[static_cast<size_t>(bucket)];
    if (selected_count + bucket_count < actual_k) {
      selected_count += bucket_count;
      continue;
    }
    boundary_digit = bucket;
    live_count = bucket_count;
    break;
  }

  state.prefix = 0;
  state.prefix_mask = 0xff00u;
  state.selected_count = selected_count;
  state.live_count = live_count;
  state.boundary_digit = boundary_digit;
  return state;
}

}  // namespace radix_topk
