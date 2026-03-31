#include <cstdio>
#include <vector>

#include "batch_topk.cuh"
#include "batch_topk_types.cuh"

namespace {

bool check_reference_ordering() {
  const std::vector<float> values = {5.0f, 7.0f, 7.0f, 1.0f};
  const radix_topk::ReferenceTopKResult result =
      radix_topk::cpu_reference_topk(values, 4, 2);
  return result.indices == std::vector<int>{1, 2} &&
         result.values == std::vector<float>{7.0f, 7.0f};
}

}  // namespace

int main() {
  const radix_topk::Candidate candidate{};
  (void)candidate;

  if (radix_topk::batch_topk_half_workspace_size(0, 10000, 50) != 0 ||
      radix_topk::batch_topk_half_workspace_size(1, 10001, 50) != 0 ||
      radix_topk::batch_topk_half_workspace_size(1, 10000, 129) != 0 ||
      radix_topk::batch_topk_half_workspace_size(1, 10, 11) != 0) {
    std::fprintf(stderr, "unexpected workspace size for invalid shapes\n");
    return 1;
  }

  if (!check_reference_ordering()) {
    std::fprintf(stderr, "cpu reference ordering is not deterministic\n");
    return 1;
  }

  const int seg_num = 1;
  const int seg_len = 10000;
  const int k = 50;
  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(seg_num, seg_len, k);
  if (workspace_size == 0) {
    std::fprintf(stderr, "workspace size must be nonzero for valid shapes\n");
    return 1;
  }
  if (workspace_size <= 1) {
    std::fprintf(stderr, "workspace size must support an insufficient-workspace check\n");
    return 1;
  }

  half* d_input = nullptr;
  half* d_values = nullptr;
  int* d_indices = nullptr;
  void* d_workspace = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * static_cast<size_t>(seg_len)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_values),
                 sizeof(half) * static_cast<size_t>(k)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_indices),
                 sizeof(int) * static_cast<size_t>(k)) != cudaSuccess ||
      cudaMalloc(&d_workspace, workspace_size) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed in test smoke path\n");
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return 1;
  }

  if (radix_topk::batch_topk_half(
          nullptr, seg_num, seg_len, k, d_values, d_indices, d_workspace,
          workspace_size, 0) != cudaErrorInvalidValue ||
      radix_topk::batch_topk_half(
          d_input, seg_num, seg_len, k, nullptr, d_indices, d_workspace,
          workspace_size, 0) != cudaErrorInvalidValue ||
      radix_topk::batch_topk_half(
          d_input, seg_num, seg_len, k, d_values, nullptr, d_workspace,
          workspace_size, 0) != cudaErrorInvalidValue ||
      radix_topk::batch_topk_half(
          d_input, seg_num, seg_len, k, d_values, d_indices, nullptr,
          workspace_size, 0) != cudaErrorInvalidValue ||
      radix_topk::batch_topk_half(
          d_input, 1, 10, 11, d_values, d_indices, d_workspace,
          radix_topk::batch_topk_half_workspace_size(1, 10, 11), 0) !=
          cudaErrorInvalidValue) {
    std::fprintf(stderr, "invalid-argument validation failed in test smoke path\n");
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return 1;
  }

  if (radix_topk::batch_topk_half(
          d_input, seg_num, seg_len, k, d_values, d_indices, d_workspace,
          workspace_size - 1, 0) != cudaErrorInvalidValue) {
    std::fprintf(stderr, "insufficient workspace validation failed\n");
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return 1;
  }

  const cudaError_t status = radix_topk::batch_topk_half(
      d_input, seg_num, seg_len, k, d_values, d_indices, d_workspace,
      workspace_size, 0);
  if (status != cudaErrorNotSupported) {
    std::fprintf(stderr, "expected cudaErrorNotSupported from smoke path\n");
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return 1;
  }

  if (cudaFree(d_workspace) != cudaSuccess || cudaFree(d_indices) != cudaSuccess ||
      cudaFree(d_values) != cudaSuccess || cudaFree(d_input) != cudaSuccess) {
    std::fprintf(stderr, "cudaFree failed in test smoke path\n");
    return 1;
  }
  return 0;
}
