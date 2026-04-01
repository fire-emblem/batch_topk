#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

inline constexpr int kHistogramBlockThreads = 256;
inline constexpr int kHistogramWarpSize = 32;
inline constexpr int kHistogramWarpsPerBlock =
    kHistogramBlockThreads / kHistogramWarpSize;
inline constexpr int kLowByteTopk50HistogramCacheEntries = 4;

// This register cache is only for the optimized single-CTA k=50 low-byte
// histogram path. It uses unsigned short counts with a fixed small cache, and
// full-cache eviction currently replaces slot 0.
struct HistogramCacheEntry {
  unsigned short bin = 0xffffu;
  unsigned short count = 0u;
};

__device__ inline void flush_histogram_cache(HistogramCacheEntry* cache,
                                             int cache_size,
                                             unsigned int* warp_histogram) {
  for (int i = 0; i < cache_size; ++i) {
    if (cache[i].count != 0u) {
      atomicAdd(&warp_histogram[cache[i].bin], static_cast<unsigned int>(cache[i].count));
      cache[i].bin = 0xffffu;
      cache[i].count = 0u;
    }
  }
}

__device__ inline void accumulate_histogram_cache(HistogramCacheEntry* cache,
                                                  int cache_size,
                                                  unsigned short bin,
                                                  unsigned int* warp_histogram) {
  for (int i = 0; i < cache_size; ++i) {
    if (cache[i].count != 0u && cache[i].bin == bin) {
      ++cache[i].count;
      return;
    }
  }
  for (int i = 0; i < cache_size; ++i) {
    if (cache[i].count == 0u) {
      cache[i].bin = bin;
      cache[i].count = 1u;
      return;
    }
  }

  atomicAdd(&warp_histogram[cache[0].bin], static_cast<unsigned int>(cache[0].count));
  cache[0].bin = bin;
  cache[0].count = 1u;
}

__global__ inline void histogram_high_byte_topk50_kernel(const half* input,
                                                         int seg_len,
                                                         unsigned int* histograms_hi) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / kHistogramWarpSize;
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;

  __shared__ unsigned int warp_histograms[kHistogramWarpsPerBlock][256];
  for (int i = tid; i < kHistogramWarpsPerBlock * 256; i += blockDim.x) {
    reinterpret_cast<unsigned int*>(warp_histograms)[i] = 0u;
  }
  __syncthreads();

  for (int i = tid; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    atomicAdd(&warp_histograms[warp][(encoded >> 8) & 0xffu], 1u);
  }
  __syncthreads();

  unsigned int* segment_histogram =
      histograms_hi + static_cast<size_t>(seg) * 256u;
  for (int bin = tid; bin < 256; bin += blockDim.x) {
    unsigned int total = 0;
    for (int w = 0; w < kHistogramWarpsPerBlock; ++w) {
      total += warp_histograms[w][bin];
    }
    segment_histogram[bin] = total;
  }
}

__global__ inline void histogram_low_byte_topk50_kernel(const half* input,
                                                        int seg_len,
                                                        const SegmentSelectState* states,
                                                        unsigned int* histograms_lo) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / kHistogramWarpSize;
  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;

  __shared__ unsigned int warp_histograms[kHistogramWarpsPerBlock][256];
  for (int i = tid; i < kHistogramWarpsPerBlock * 256; i += blockDim.x) {
    reinterpret_cast<unsigned int*>(warp_histograms)[i] = 0u;
  }
  __syncthreads();

  for (int i = tid; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded >> 8) == state.boundary_digit) {
      atomicAdd(&warp_histograms[warp][encoded & 0xffu], 1u);
    }
  }
  __syncthreads();

  unsigned int* segment_histogram =
      histograms_lo + static_cast<size_t>(seg) * 256u;
  for (int bin = tid; bin < 256; bin += blockDim.x) {
    unsigned int total = 0;
    for (int w = 0; w < kHistogramWarpsPerBlock; ++w) {
      total += warp_histograms[w][bin];
    }
    segment_histogram[bin] = total;
  }
}

__global__ inline void histogram_low_byte_topk50_v2_kernel(
    const half* input,
    int seg_len,
    const SegmentSelectState* states,
    unsigned int* histograms_lo) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid / kHistogramWarpSize;
  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;

  __shared__ unsigned int warp_histograms[kHistogramWarpsPerBlock][256];
  for (int i = tid; i < kHistogramWarpsPerBlock * 256; i += blockDim.x) {
    reinterpret_cast<unsigned int*>(warp_histograms)[i] = 0u;
  }
  __syncthreads();

  HistogramCacheEntry cache[kLowByteTopk50HistogramCacheEntries];
  for (int i = 0; i < kLowByteTopk50HistogramCacheEntries; ++i) {
    cache[i].bin = 0xffffu;
    cache[i].count = 0u;
  }

  unsigned int* warp_histogram = warp_histograms[warp];
  for (int i = tid; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded >> 8) == state.boundary_digit) {
      accumulate_histogram_cache(
          cache,
          kLowByteTopk50HistogramCacheEntries,
          static_cast<unsigned short>(encoded & 0xffu),
          warp_histogram);
    }
  }
  flush_histogram_cache(cache, kLowByteTopk50HistogramCacheEntries, warp_histogram);
  __syncthreads();

  unsigned int* segment_histogram =
      histograms_lo + static_cast<size_t>(seg) * 256u;
  for (int bin = tid; bin < 256; bin += blockDim.x) {
    unsigned int total = 0;
    for (int w = 0; w < kHistogramWarpsPerBlock; ++w) {
      total += warp_histograms[w][bin];
    }
    segment_histogram[bin] = total;
  }
}

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
