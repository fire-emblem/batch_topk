#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

__global__ void histogram_pass_kernel(const half* input,
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
