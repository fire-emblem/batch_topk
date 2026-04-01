#pragma once

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

inline constexpr int kFirstNibbleBuckets = 16;
inline constexpr int kFirstNibbleWarpBatch = 20;
inline constexpr int kFirstNibbleBlockThreads = 256;
inline constexpr int kFirstNibbleWarpsPerBlock =
    kFirstNibbleBlockThreads / 32;

__global__ inline void first_nibble_select_kernel(const half* input,
                                                  int seg_len,
                                                  int k,
                                                  SegmentSelectState* states) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;

  __shared__ unsigned int warp_bucket_counts[kFirstNibbleWarpsPerBlock][kFirstNibbleBuckets];
  __shared__ unsigned int block_bucket_counts[kFirstNibbleBuckets];

  for (int i = tid; i < kFirstNibbleWarpsPerBlock * kFirstNibbleBuckets;
       i += blockDim.x) {
    reinterpret_cast<unsigned int*>(warp_bucket_counts)[i] = 0u;
  }
  for (int i = tid; i < kFirstNibbleBuckets; i += blockDim.x) {
    block_bucket_counts[i] = 0u;
  }
  __syncthreads();

  const int warp_batch_base = warp * kFirstNibbleWarpBatch;
  for (int base = warp_batch_base; base < seg_len;
       base += kFirstNibbleWarpsPerBlock * kFirstNibbleWarpBatch) {
    const int local = lane < kFirstNibbleWarpBatch ? lane : -1;
    const int idx = local >= 0 ? base + local : -1;
    if (idx >= 0 && idx < seg_len) {
      const uint16_t encoded = encode_half_desc(segment_input[idx]);
      const int bucket = (encoded >> 12) & 0xf;
      atomicAdd(&warp_bucket_counts[warp][bucket], 1u);
    }
  }
  __syncthreads();

  for (int bucket = tid; bucket < kFirstNibbleBuckets; bucket += blockDim.x) {
    unsigned int total = 0;
    for (int w = 0; w < kFirstNibbleWarpsPerBlock; ++w) {
      total += warp_bucket_counts[w][bucket];
    }
    block_bucket_counts[bucket] = total;
  }
  __syncthreads();

  if (tid == 0) {
    SegmentSelectState state{};
    int selected = 0;
    for (int bucket = 0; bucket < kFirstNibbleBuckets; ++bucket) {
      const int count = static_cast<int>(block_bucket_counts[bucket]);
      if (selected + count < k) {
        selected += count;
        continue;
      }
      state.prefix = static_cast<uint16_t>(bucket << 12);
      state.prefix_mask = 0xf000u;
      state.selected_count = selected;
      state.live_count = count;
      state.boundary_digit = bucket;
      states[seg] = state;
      return;
    }

    state.prefix = 0xffffu;
    state.prefix_mask = 0xffffu;
    state.selected_count = selected;
    state.live_count = 0;
    state.boundary_digit = 15;
    states[seg] = state;
  }
}

__global__ inline void histogram_second_nibble_kernel(const half* input,
                                                      int seg_len,
                                                      const SegmentSelectState* states,
                                                      unsigned int* histograms_hi) {
  const int seg = blockIdx.x;
  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  unsigned int* segment_histogram = histograms_hi + static_cast<size_t>(seg) * 256u;

  __shared__ unsigned int local[kFirstNibbleBuckets];
  for (int i = threadIdx.x; i < kFirstNibbleBuckets; i += blockDim.x) {
    local[i] = 0u;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded & state.prefix_mask) == state.prefix) {
      atomicAdd(&local[(encoded >> 8) & 0xf], 1u);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kFirstNibbleBuckets; i += blockDim.x) {
    segment_histogram[i] = local[i];
  }
}

__global__ inline void select_second_nibble_kernel(const unsigned int* histograms_hi,
                                                   int seg_num,
                                                   int k,
                                                   SegmentSelectState* states) {
  const int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= seg_num) {
    return;
  }

  const unsigned int* histogram = histograms_hi + static_cast<size_t>(seg) * 256u;
  SegmentSelectState state = states[seg];
  int selected = state.selected_count;
  const int high_nibble = static_cast<int>(state.prefix >> 12);
  for (int bucket = 0; bucket < kFirstNibbleBuckets; ++bucket) {
    const int count = static_cast<int>(histogram[bucket]);
    if (selected + count < k) {
      selected += count;
      continue;
    }
    state.boundary_digit = (high_nibble << 4) | bucket;
    state.prefix = static_cast<uint16_t>(state.boundary_digit << 8);
    state.prefix_mask = 0xff00u;
    state.selected_count = selected;
    state.live_count = count;
    states[seg] = state;
    return;
  }
  states[seg] = state;
}

cudaError_t batch_topk_half_first_nibble_round_for_test(
    const half* d_input,
    int seg_num,
    int seg_len,
    int k,
    void* d_workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

}  // namespace radix_topk
