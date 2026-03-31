#pragma once

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace radix_topk {

cudaError_t batch_topk_half(
    const half* d_input,
    int seg_num,
    int seg_len,
    int k,
    half* d_output_values,
    int* d_output_indices,
    void* d_workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k);

}  // namespace radix_topk
