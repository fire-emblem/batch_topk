#include <cmath>
#include <cstdio>
#include <vector>

#include "batch_topk.cuh"
#include "batch_topk_types.cuh"
#include "../src/radix_histogram.cuh"
#include "../src/radix_select_state.cuh"

namespace {

bool check_reference_ordering() {
  const std::vector<float> values = {5.0f, 7.0f, 7.0f, 1.0f};
  const radix_topk::ReferenceTopKResult result =
      radix_topk::cpu_reference_topk(values, 4, 2);
  return result.indices == std::vector<int>{1, 2} &&
         result.values == std::vector<float>{7.0f, 7.0f};
}

bool check_reference_nan_ordering() {
  const std::vector<float> values = {NAN, 3.0f, -1.0f, NAN};
  const radix_topk::ReferenceTopKResult result =
      radix_topk::cpu_reference_topk(values, 4, 3);
  return result.indices == std::vector<int>{1, 2, 0} &&
         result.values.size() == 3 &&
         result.values[0] == 3.0f &&
         result.values[1] == -1.0f &&
         std::isnan(result.values[2]);
}

bool check_reference_bounded_seg_len() {
  const std::vector<float> values = {2.0f, 1.0f};
  const radix_topk::ReferenceTopKResult result =
      radix_topk::cpu_reference_topk(values, 4, 4);
  return result.indices == std::vector<int>{0, 1} &&
         result.values == std::vector<float>{2.0f, 1.0f};
}

bool check_half_codec_special_values() {
  const uint16_t neg = radix_topk::encode_half_desc(__float2half(-1.0f));
  const uint16_t pos = radix_topk::encode_half_desc(__float2half(3.0f));
  const uint16_t nan = radix_topk::encode_half_desc(__float2half(NAN));
  return pos < neg && nan > neg;
}

bool check_candidate_tie_break() {
  radix_topk::Candidate lhs{};
  lhs.value = __float2half(5.0f);
  lhs.index = 2;

  radix_topk::Candidate rhs{};
  rhs.value = __float2half(5.0f);
  rhs.index = 1;

  return !radix_topk::candidate_better(lhs, rhs) &&
         radix_topk::candidate_better(rhs, lhs);
}

bool check_radix_boundary_state() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  const radix_topk::SegmentSelectState state =
      radix_topk::simulate_radix_boundary(values, 8, 3);
  return state.selected_count == 2 && state.live_count > 0 &&
         state.boundary_digit >= 0;
}

bool check_histogram_pass() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  std::vector<half> host_input(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host_input[i] = __float2half(values[i]);
  }

  half* d_input = nullptr;
  unsigned int* d_histograms = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_histograms),
                 sizeof(unsigned int) * 256) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_histograms);
    cudaFree(d_input);
    return false;
  }

  radix_topk::histogram_pass_kernel<<<1, 256>>>(d_input, 8, 8, d_histograms);
  std::vector<unsigned int> histograms(256, 0);
  const bool ok = cudaGetLastError() == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(histograms.data(),
                             d_histograms,
                             sizeof(unsigned int) * histograms.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_histograms);
  cudaFree(d_input);
  if (!ok) {
    return false;
  }

  unsigned int sum = 0;
  for (unsigned int count : histograms) {
    sum += count;
  }
  return sum == values.size();
}

bool check_gpu_small_correctness() {
  const int seg_num = 2;
  const int seg_len = 8;
  const int k = 3;
  const std::vector<float> host_values = {
      1.0f, 9.0f, 2.0f, 8.0f, 3.0f, 7.0f, 4.0f, 6.0f,
      5.0f, 4.0f, 9.0f, 1.0f, 9.0f, 2.0f, 0.0f, 8.0f};
  std::vector<half> host_input(host_values.size());
  for (size_t i = 0; i < host_values.size(); ++i) {
    host_input[i] = __float2half(host_values[i]);
  }

  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(seg_num, seg_len, k);
  half* d_input = nullptr;
  half* d_output_values = nullptr;
  int* d_output_indices = nullptr;
  void* d_workspace = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_output_values),
                 sizeof(half) * static_cast<size_t>(seg_num) * k) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_output_indices),
                 sizeof(int) * static_cast<size_t>(seg_num) * k) != cudaSuccess ||
      cudaMalloc(&d_workspace, workspace_size) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_workspace);
    cudaFree(d_output_indices);
    cudaFree(d_output_values);
    cudaFree(d_input);
    return false;
  }

  const cudaError_t status = radix_topk::batch_topk_half(
      d_input,
      seg_num,
      seg_len,
      k,
      d_output_values,
      d_output_indices,
      d_workspace,
      workspace_size,
      0);
  std::vector<half> output_values(static_cast<size_t>(seg_num) * k);
  std::vector<int> output_indices(static_cast<size_t>(seg_num) * k);
  const bool ok = status == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(output_values.data(),
                             d_output_values,
                             sizeof(half) * output_values.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(output_indices.data(),
                             d_output_indices,
                             sizeof(int) * output_indices.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_workspace);
  cudaFree(d_output_indices);
  cudaFree(d_output_values);
  cudaFree(d_input);
  if (!ok) {
    return false;
  }

  for (int seg = 0; seg < seg_num; ++seg) {
    const auto begin = host_values.begin() + static_cast<size_t>(seg) * seg_len;
    const auto end = begin + seg_len;
    const std::vector<float> segment(begin, end);
    const radix_topk::ReferenceTopKResult expected =
        radix_topk::cpu_reference_topk(segment, seg_len, k);
    for (int i = 0; i < k; ++i) {
      if (output_indices[static_cast<size_t>(seg) * k + i] != expected.indices[i] ||
          __half2float(output_values[static_cast<size_t>(seg) * k + i]) !=
              expected.values[i]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

__global__ void codec_smoke_kernel(uint16_t* keys_out, int* better_out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  keys_out[0] = radix_topk::encode_half_desc(__float2half(-1.0f));
  keys_out[1] = radix_topk::encode_half_desc(__float2half(3.0f));

  radix_topk::Candidate lhs{};
  lhs.value = __float2half(5.0f);
  lhs.index = 2;

  radix_topk::Candidate rhs{};
  rhs.value = __float2half(5.0f);
  rhs.index = 1;

  better_out[0] = radix_topk::candidate_better(rhs, lhs) ? 1 : 0;
}

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
  if (!check_reference_nan_ordering()) {
    std::fprintf(stderr, "cpu reference NaN handling is incorrect\n");
    return 1;
  }
  if (!check_reference_bounded_seg_len()) {
    std::fprintf(stderr, "cpu reference seg_len clamping is incorrect\n");
    return 1;
  }
  if (!check_half_codec_special_values()) {
    std::fprintf(stderr, "half codec special values are incorrect\n");
    return 1;
  }
  if (!check_candidate_tie_break()) {
    std::fprintf(stderr, "candidate tie break is incorrect\n");
    return 1;
  }
  if (!check_radix_boundary_state()) {
    std::fprintf(stderr, "radix boundary state check failed\n");
    return 1;
  }
  if (!check_histogram_pass()) {
    std::fprintf(stderr, "histogram pass check failed\n");
    return 1;
  }
  if (!check_gpu_small_correctness()) {
    std::fprintf(stderr, "gpu small correctness check failed\n");
    return 1;
  }

  uint16_t host_keys[2] = {};
  int host_better = 0;
  uint16_t* d_keys = nullptr;
  int* d_better = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_keys), sizeof(host_keys)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_better), sizeof(host_better)) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed in codec smoke path\n");
    cudaFree(d_better);
    cudaFree(d_keys);
    return 1;
  }
  codec_smoke_kernel<<<1, 1>>>(d_keys, d_better);
  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
      cudaMemcpy(host_keys, d_keys, sizeof(host_keys), cudaMemcpyDeviceToHost) !=
          cudaSuccess ||
      cudaMemcpy(&host_better, d_better, sizeof(host_better), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
    std::fprintf(stderr, "device codec smoke path failed\n");
    cudaFree(d_better);
    cudaFree(d_keys);
    return 1;
  }
  if (!(host_keys[1] < host_keys[0]) || host_better != 1) {
    std::fprintf(stderr, "device codec semantics are incorrect\n");
    cudaFree(d_better);
    cudaFree(d_keys);
    return 1;
  }
  if (cudaFree(d_better) != cudaSuccess || cudaFree(d_keys) != cudaSuccess) {
    std::fprintf(stderr, "cudaFree failed in codec smoke path\n");
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
  std::vector<half> smoke_input(static_cast<size_t>(seg_len), __float2half(0.0f));
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * static_cast<size_t>(seg_len)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_values),
                 sizeof(half) * static_cast<size_t>(k)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_indices),
                 sizeof(int) * static_cast<size_t>(k)) != cudaSuccess ||
      cudaMalloc(&d_workspace, workspace_size) != cudaSuccess ||
      cudaMemcpy(d_input,
                 smoke_input.data(),
                 sizeof(half) * smoke_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
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
  if (status != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "expected cudaSuccess from smoke path\n");
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
