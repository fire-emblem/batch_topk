#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

inline constexpr int kHistogramBlockThreads = 256;
inline constexpr int kHistogramWarpSize = 32;
inline constexpr int kHistogramWarpsPerBlock =
    kHistogramBlockThreads / kHistogramWarpSize;
inline constexpr int kLargeBatchHistogramSampleWidth = 4;

struct HistogramMergeEntry {
  unsigned short bin = 0xffffu;
  unsigned short count = 0u;
};

// This helper is intentionally specialized for the fixed 4-sample large-batch
// histogram kernels; slot 0 is the overflow merge target.
__device__ inline void merge_histogram_samples(HistogramMergeEntry* entries,
                                               int entries_count,
                                               unsigned short bin) {
  for (int i = 0; i < entries_count; ++i) {
    if (entries[i].count != 0u && entries[i].bin == bin) {
      ++entries[i].count;
      return;
    }
  }
  for (int i = 0; i < entries_count; ++i) {
    if (entries[i].count == 0u) {
      entries[i].bin = bin;
      entries[i].count = 1u;
      return;
    }
  }

  // Fixed-size local merge for the 4-sample batch; overflow falls back to slot 0.
  ++entries[0].count;
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

__global__ inline void histogram_high_byte_topk50_large_kernel(
    const half* input,
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

  unsigned int* warp_histogram = warp_histograms[warp];
  for (int base = tid; base < seg_len;
       base += blockDim.x * kLargeBatchHistogramSampleWidth) {
    HistogramMergeEntry entries[kLargeBatchHistogramSampleWidth] = {};
#pragma unroll
    for (int item = 0; item < kLargeBatchHistogramSampleWidth; ++item) {
      const int idx = base + item * blockDim.x;
      if (idx < seg_len) {
        const uint16_t encoded = encode_half_desc(segment_input[idx]);
        merge_histogram_samples(
            entries,
            kLargeBatchHistogramSampleWidth,
            static_cast<unsigned short>((encoded >> 8) & 0xffu));
      }
    }
#pragma unroll
    for (int i = 0; i < kLargeBatchHistogramSampleWidth; ++i) {
      if (entries[i].count != 0u) {
        atomicAdd(&warp_histogram[entries[i].bin],
                  static_cast<unsigned int>(entries[i].count));
      }
    }
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

__global__ inline void histogram_low_byte_topk50_large_kernel(
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

  unsigned int* warp_histogram = warp_histograms[warp];
  for (int base = tid; base < seg_len;
       base += blockDim.x * kLargeBatchHistogramSampleWidth) {
    HistogramMergeEntry entries[kLargeBatchHistogramSampleWidth] = {};
#pragma unroll
    for (int item = 0; item < kLargeBatchHistogramSampleWidth; ++item) {
      const int idx = base + item * blockDim.x;
      if (idx < seg_len) {
        const uint16_t encoded = encode_half_desc(segment_input[idx]);
        if ((encoded >> 8) == state.boundary_digit) {
          merge_histogram_samples(
              entries,
              kLargeBatchHistogramSampleWidth,
              static_cast<unsigned short>(encoded & 0xffu));
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kLargeBatchHistogramSampleWidth; ++i) {
      if (entries[i].count != 0u) {
        atomicAdd(&warp_histogram[entries[i].bin],
                  static_cast<unsigned int>(entries[i].count));
      }
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
