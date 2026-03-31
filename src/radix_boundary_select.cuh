#pragma once

#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

__global__ inline void select_high_byte_boundary_kernel(
    const unsigned int* histograms_hi,
    int seg_num,
    int k,
    SegmentSelectState* states) {
  const int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= seg_num) {
    return;
  }

  const unsigned int* histogram =
      histograms_hi + static_cast<size_t>(seg) * 256u;
  SegmentSelectState state{};
  int selected = 0;
  for (int bucket = 0; bucket < 256; ++bucket) {
    const int count = static_cast<int>(histogram[bucket]);
    if (selected + count < k) {
      selected += count;
      continue;
    }
    state.boundary_digit = bucket;
    state.prefix = static_cast<uint16_t>(bucket << 8);
    state.prefix_mask = 0xff00u;
    state.selected_count = selected;
    state.live_count = count;
    break;
  }
  states[seg] = state;
}

__global__ inline void finalize_cutoff_key_kernel(const unsigned int* histograms_lo,
                                                  int seg_num,
                                                  int k,
                                                  SegmentSelectState* states) {
  const int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= seg_num) {
    return;
  }

  const unsigned int* histogram =
      histograms_lo + static_cast<size_t>(seg) * 256u;
  SegmentSelectState state = states[seg];
  int selected = state.selected_count;
  for (int bucket = 0; bucket < 256; ++bucket) {
    const int count = static_cast<int>(histogram[bucket]);
    if (selected + count < k) {
      selected += count;
      continue;
    }
    state.cutoff_key =
        static_cast<uint16_t>((state.boundary_digit << 8) | bucket);
    state.strictly_better_count = selected;
    state.remaining_slots = k - selected;
    states[seg] = state;
    return;
  }
}

}  // namespace radix_topk
