#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

__global__ inline void histogram_high_byte_kernel(const half* input,
                                                  int seg_len,
                                                  unsigned int* histograms_hi) {
  const int seg = blockIdx.x;
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;

  __shared__ unsigned int local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    local[i] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    atomicAdd(&local[(encoded >> 8) & 0xffu], 1u);
  }
  __syncthreads();

  unsigned int* segment_histogram =
      histograms_hi + static_cast<size_t>(seg) * 256u;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    segment_histogram[i] = local[i];
  }
}

__global__ inline void histogram_low_byte_kernel(const half* input,
                                                 int seg_len,
                                                 const SegmentSelectState* states,
                                                 unsigned int* histograms_lo) {
  const int seg = blockIdx.x;
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  const SegmentSelectState state = states[seg];

  __shared__ unsigned int local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    local[i] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded >> 8) == state.boundary_digit) {
      atomicAdd(&local[encoded & 0xffu], 1u);
    }
  }
  __syncthreads();

  unsigned int* segment_histogram =
      histograms_lo + static_cast<size_t>(seg) * 256u;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    segment_histogram[i] = local[i];
  }
}

__global__ inline void histogram_high_byte_splitk_kernel(
    const half* input,
    int seg_num,
    int seg_len,
    int ctas_per_segment,
    unsigned int* partial_histograms) {
  const int global_cta = blockIdx.x;
  const int seg = global_cta / ctas_per_segment;
  const int tile = global_cta % ctas_per_segment;
  if (seg >= seg_num) {
    return;
  }

  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  __shared__ unsigned int local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    local[i] = 0;
  }
  __syncthreads();

  for (int i = tile * blockDim.x + threadIdx.x; i < seg_len;
       i += ctas_per_segment * blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    atomicAdd(&local[(encoded >> 8) & 0xffu], 1u);
  }
  __syncthreads();

  unsigned int* partial =
      partial_histograms + (static_cast<size_t>(seg) * ctas_per_segment + tile) * 256u;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    partial[i] = local[i];
  }
}

__global__ inline void histogram_low_byte_splitk_kernel(
    const half* input,
    int seg_num,
    int seg_len,
    int ctas_per_segment,
    const SegmentSelectState* states,
    unsigned int* partial_histograms) {
  const int global_cta = blockIdx.x;
  const int seg = global_cta / ctas_per_segment;
  const int tile = global_cta % ctas_per_segment;
  if (seg >= seg_num) {
    return;
  }

  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  __shared__ unsigned int local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    local[i] = 0;
  }
  __syncthreads();

  for (int i = tile * blockDim.x + threadIdx.x; i < seg_len;
       i += ctas_per_segment * blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded >> 8) == state.boundary_digit) {
      atomicAdd(&local[encoded & 0xffu], 1u);
    }
  }
  __syncthreads();

  unsigned int* partial =
      partial_histograms + (static_cast<size_t>(seg) * ctas_per_segment + tile) * 256u;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    partial[i] = local[i];
  }
}

__global__ inline void reduce_partial_histograms_kernel(
    const unsigned int* partial_histograms,
    int seg_num,
    int ctas_per_segment,
    unsigned int* merged_histograms) {
  const int seg = blockIdx.x;
  if (seg >= seg_num) {
    return;
  }

  for (int bucket = threadIdx.x; bucket < 256; bucket += blockDim.x) {
    unsigned int total = 0;
    for (int tile = 0; tile < ctas_per_segment; ++tile) {
      total += partial_histograms[(static_cast<size_t>(seg) * ctas_per_segment + tile) *
                                      256u +
                                  static_cast<size_t>(bucket)];
    }
    merged_histograms[static_cast<size_t>(seg) * 256u + static_cast<size_t>(bucket)] =
        total;
  }
}

static __global__ void histogram_pass_kernel(const half* input,
                                             int seg_len,
                                             int shift,
                                             uint16_t prefix,
                                             uint16_t prefix_mask,
                                             unsigned int* segment_histograms) {
  const int seg = blockIdx.x;
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;

  __shared__ unsigned int local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    local[i] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded & prefix_mask) == prefix) {
      atomicAdd(&local[static_cast<size_t>((encoded >> shift) & 0xffu)], 1u);
    }
  }
  __syncthreads();

  unsigned int* segment_histogram =
      segment_histograms + static_cast<size_t>(seg) * 256u;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    segment_histogram[i] = local[i];
  }
}

}  // namespace radix_topk
