#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include "batch_topk_benchmark.cuh"
#include "batch_topk.cuh"
#include "batch_topk_types.cuh"
#include "../src/batch_topk_stage_timing.cuh"
#include "../src/candidate_compact.cuh"
#include "../src/dispatch_policy.cuh"
#include "../src/final_topk50.cuh"
#include "../src/radix_boundary_select.cuh"
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
  return state.selected_count == 2 && state.live_count == 1 &&
         state.boundary_digit == 56 && state.prefix == 0x3800u &&
         state.prefix_mask == 0xff00u;
}

bool run_histogram_pass(const std::vector<float>& values,
                        uint16_t prefix,
                        uint16_t prefix_mask,
                        int shift,
                        std::vector<unsigned int>* histograms) {
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

  radix_topk::histogram_pass_kernel<<<1, 256>>>(
      d_input, static_cast<int>(values.size()), shift, prefix, prefix_mask,
      d_histograms);
  const bool ok = cudaGetLastError() == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(histograms->data(),
                             d_histograms,
                             sizeof(unsigned int) * histograms->size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_histograms);
  cudaFree(d_input);
  return ok;
}

bool run_histogram_topk50_pair(const std::vector<float>& values,
                               int k,
                               std::vector<unsigned int>* histograms_hi,
                               std::vector<unsigned int>* histograms_lo,
                               radix_topk::SegmentSelectState* state_out) {
  std::vector<half> host_input(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host_input[i] = __float2half(values[i]);
  }

  half* d_input = nullptr;
  unsigned int* d_histograms_hi = nullptr;
  unsigned int* d_histograms_lo = nullptr;
  radix_topk::SegmentSelectState* d_state = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_histograms_hi),
                 sizeof(unsigned int) * histograms_hi->size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_histograms_lo),
                 sizeof(unsigned int) * histograms_lo->size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_state),
                 sizeof(radix_topk::SegmentSelectState)) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_state);
    cudaFree(d_histograms_lo);
    cudaFree(d_histograms_hi);
    cudaFree(d_input);
    return false;
  }

  radix_topk::histogram_high_byte_topk50_kernel<<<1, 256>>>(
      d_input, static_cast<int>(values.size()), d_histograms_hi);
  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(d_state);
    cudaFree(d_histograms_lo);
    cudaFree(d_histograms_hi);
    cudaFree(d_input);
    return false;
  }
  radix_topk::select_high_byte_boundary_kernel<<<1, 128>>>(d_histograms_hi, 1, k, d_state);
  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(d_state);
    cudaFree(d_histograms_lo);
    cudaFree(d_histograms_hi);
    cudaFree(d_input);
    return false;
  }
  radix_topk::histogram_low_byte_topk50_kernel<<<1, 256>>>(
      d_input, static_cast<int>(values.size()), d_state, d_histograms_lo);
  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(d_state);
    cudaFree(d_histograms_lo);
    cudaFree(d_histograms_hi);
    cudaFree(d_input);
    return false;
  }
  radix_topk::finalize_cutoff_key_kernel<<<1, 128>>>(d_histograms_lo, 1, k, d_state);
  const bool ok =
      cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
      cudaMemcpy(histograms_hi->data(),
                 d_histograms_hi,
                 sizeof(unsigned int) * histograms_hi->size(),
                 cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(histograms_lo->data(),
                 d_histograms_lo,
                 sizeof(unsigned int) * histograms_lo->size(),
                 cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(state_out,
                 d_state,
                 sizeof(*state_out),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_state);
  cudaFree(d_histograms_lo);
  cudaFree(d_histograms_hi);
  cudaFree(d_input);
  return ok;
}

bool run_gpu_cutoff_selection(const std::vector<float>& values,
                              int k,
                              radix_topk::SegmentSelectState* state_out) {
  std::vector<half> host_input(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host_input[i] = __float2half(values[i]);
  }

  half* d_input = nullptr;
  unsigned int* d_hist_hi = nullptr;
  unsigned int* d_hist_lo = nullptr;
  radix_topk::SegmentSelectState* d_state = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_hist_hi), sizeof(unsigned int) * 256) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_hist_lo), sizeof(unsigned int) * 256) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_state), sizeof(radix_topk::SegmentSelectState)) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_state);
    cudaFree(d_hist_lo);
    cudaFree(d_hist_hi);
    cudaFree(d_input);
    return false;
  }

  radix_topk::histogram_high_byte_kernel<<<1, 256>>>(
      d_input, static_cast<int>(values.size()), d_hist_hi);
  const cudaError_t launch_status_hi = cudaGetLastError();
  radix_topk::select_high_byte_boundary_kernel<<<1, 128>>>(d_hist_hi, 1, k, d_state);
  const cudaError_t launch_status_select = cudaGetLastError();
  radix_topk::histogram_low_byte_kernel<<<1, 256>>>(
      d_input, static_cast<int>(values.size()), d_state, d_hist_lo);
  const cudaError_t launch_status_lo = cudaGetLastError();
  radix_topk::finalize_cutoff_key_kernel<<<1, 128>>>(d_hist_lo, 1, k, d_state);
  const cudaError_t launch_status_finalize = cudaGetLastError();

  const bool ok = launch_status_hi == cudaSuccess &&
                  launch_status_select == cudaSuccess &&
                  launch_status_lo == cudaSuccess &&
                  launch_status_finalize == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(state_out,
                             d_state,
                             sizeof(radix_topk::SegmentSelectState),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_state);
  cudaFree(d_hist_lo);
  cudaFree(d_hist_hi);
  cudaFree(d_input);
  return ok;
}

bool check_gpu_cutoff_selection() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  const uint16_t expected_cutoff =
      radix_topk::encode_half_desc(__float2half(7.0f));

  radix_topk::SegmentSelectState state{};
  if (!run_gpu_cutoff_selection(values, 3, &state)) {
    return false;
  }

  return state.strictly_better_count == 2 &&
         state.remaining_slots == 1 &&
         state.cutoff_key == expected_cutoff;
}

// Defensive unit-level check: this intentionally uses k > seg_len to force the
// sentinel fallback path in boundary selection kernels.
bool check_gpu_cutoff_selection_fallback() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  radix_topk::SegmentSelectState state{};
  if (!run_gpu_cutoff_selection(values, 9, &state)) {
    return false;
  }
  return state.prefix == 0xffffu &&
         state.prefix_mask == 0xffffu &&
         state.cutoff_key == 0xffffu &&
         state.live_count == 0 &&
         state.boundary_digit == 255 &&
         state.strictly_better_count == 8 &&
         state.remaining_slots == 1;
}

bool check_histogram_pass() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  const radix_topk::SegmentSelectState state =
      radix_topk::simulate_radix_boundary(values, 8, 3);
  std::vector<unsigned int> histograms(256, 0);
  if (!run_histogram_pass(values, 0, 0, 8, &histograms)) {
    return false;
  }

  if (histograms[55] != 2 || histograms[56] != 1 || histograms[57] != 1 ||
      histograms[58] != 1 || histograms[59] != 1 || histograms[61] != 1 ||
      histograms[63] != 1) {
    return false;
  }

  std::vector<unsigned int> filtered_histograms(256, 0);
  if (!run_histogram_pass(values, state.prefix, state.prefix_mask, 0,
                          &filtered_histograms)) {
    return false;
  }

  if (filtered_histograms[255] != 1) {
    return false;
  }
  for (int i = 0; i < 255; ++i) {
    if (filtered_histograms[static_cast<size_t>(i)] != 0) {
      return false;
    }
  }
  return true;
}

bool check_topk50_histogram_kernels() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  std::vector<unsigned int> histograms_hi(256, 0);
  std::vector<unsigned int> histograms_lo(256, 0);
  radix_topk::SegmentSelectState state{};
  if (!run_histogram_topk50_pair(values, 3, &histograms_hi, &histograms_lo, &state)) {
    return false;
  }

  if (histograms_hi[55] != 2 || histograms_hi[56] != 1 || histograms_hi[57] != 1 ||
      histograms_hi[58] != 1 || histograms_hi[59] != 1 || histograms_hi[61] != 1 ||
      histograms_hi[63] != 1) {
    return false;
  }

  if (state.boundary_digit != 56 || state.cutoff_key != 0x38ffu ||
      state.strictly_better_count != 2 || state.remaining_slots != 1) {
    return false;
  }

  if (histograms_lo[255] != 1) {
    return false;
  }
  for (int i = 0; i < 255; ++i) {
    if (histograms_lo[static_cast<size_t>(i)] != 0) {
      return false;
    }
  }
  return true;
}

bool check_benchmark_target_matrix() {
  constexpr auto cases = radix_topk::kPrimaryBenchmarkCases;
  if (cases.size() != 5) {
    return false;
  }

  static constexpr int kExpectedSegNums[5] = {128, 1500, 3000, 4500, 6000};
  static constexpr float kExpectedRefs[5] = {11.0f, 82.0f, 161.0f, 241.0f, 319.0f};
  for (size_t i = 0; i < cases.size(); ++i) {
    if (cases[i].seg_num != kExpectedSegNums[i] || cases[i].seg_len != 10000 ||
        cases[i].k != 50 || cases[i].ref_us != kExpectedRefs[i]) {
      return false;
    }
  }
  return true;
}

bool check_benchmark_measurement_contract() {
  return radix_topk::kPrimaryBenchmarkCases[0].seg_num == 128 &&
         radix_topk::kPrimaryBenchmarkCases[4].seg_num == 6000;
}

bool check_stage_timing_contract() {
  radix_topk::BatchTopkStageTiming timing{};
  return timing.high_byte_hist_us == 0.0f &&
         timing.high_byte_select_us == 0.0f &&
         timing.low_byte_hist_us == 0.0f &&
         timing.finalize_cutoff_us == 0.0f &&
         timing.compaction_us == 0.0f &&
         timing.final_topk_us == 0.0f;
}

bool check_optimized_state_contract() {
  radix_topk::SegmentSelectState state{};
  return state.prefix == 0u &&
         state.prefix_mask == 0u &&
         state.cutoff_key == 0xffffu &&
         state.selected_count == 0 &&
         state.live_count == 0 &&
         state.boundary_digit == 0 &&
         state.strictly_better_count == 0 &&
         state.remaining_slots == 0;
}

bool check_optimized_workspace_contract() {
  const size_t workspace_bytes =
      radix_topk::batch_topk_half_workspace_size(128, 10000, 50);
  // Baseline: legacy full Candidate-per-element storage at (128, 10000).
  const size_t legacy_full_candidate_workspace_bytes =
      static_cast<size_t>(128) * 10000 * sizeof(radix_topk::Candidate);
  return workspace_bytes > 0 &&
         workspace_bytes < legacy_full_candidate_workspace_bytes &&
         radix_topk::kOptimizedCandidateCap == 64 &&
         radix_topk::kCompactedCandidateCap >= radix_topk::kMaxSupportedK;
}

bool check_optimized_workspace_layout_contract() {
  const int seg_num = 3;
  const size_t workspace_bytes =
      radix_topk::candidate_compaction_workspace_bytes(seg_num);
  if (workspace_bytes == 0) {
    return false;
  }
  std::vector<std::byte> storage(workspace_bytes);
  const auto view =
      radix_topk::make_candidate_compaction_workspace(storage.data(), seg_num);
  if (!view.histograms_hi || !view.histograms_lo || !view.partial_histograms ||
      !view.states || !view.candidate_counts || !view.candidate_indices) {
    return false;
  }

  const auto* const base = storage.data();
  size_t offset = 0u;
  offset = radix_topk::align_up(offset, alignof(unsigned int));
  if (reinterpret_cast<std::byte*>(view.histograms_hi) != base + offset) {
    return false;
  }
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  if (reinterpret_cast<std::byte*>(view.histograms_lo) != base + offset) {
    return false;
  }
  offset += static_cast<size_t>(seg_num) * 256u * sizeof(unsigned int);
  if (reinterpret_cast<std::byte*>(view.partial_histograms) != base + offset) {
    return false;
  }
  offset += static_cast<size_t>(seg_num) *
            radix_topk::kPartialHistogramMaxCtasPerSegment * 256u *
            sizeof(unsigned int);
  offset = radix_topk::align_up(offset, alignof(radix_topk::SegmentSelectState));
  if (reinterpret_cast<std::byte*>(view.states) != base + offset) {
    return false;
  }
  offset += static_cast<size_t>(seg_num) * sizeof(radix_topk::SegmentSelectState);
  offset = radix_topk::align_up(offset, alignof(int));
  if (reinterpret_cast<std::byte*>(view.candidate_counts) != base + offset) {
    return false;
  }
  offset += static_cast<size_t>(seg_num) * sizeof(int);
  if (reinterpret_cast<std::byte*>(view.candidate_indices) != base + offset) {
    return false;
  }
  offset += static_cast<size_t>(seg_num) *
            radix_topk::kCompactedCandidateCap * sizeof(int);
  return offset == workspace_bytes;
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
    std::fprintf(stderr,
                 "random regression runtime failure: call=%s sync=%s\n",
                 cudaGetErrorString(status),
                 cudaGetErrorString(cudaDeviceSynchronize()));
    return false;
  }

  for (int seg = 0; seg < seg_num; ++seg) {
    const auto begin = host_values.begin() + static_cast<size_t>(seg) * seg_len;
    const auto end = begin + seg_len;
    const std::vector<float> segment(begin, end);
    const radix_topk::ReferenceTopKResult expected =
        radix_topk::cpu_reference_topk(segment, seg_len, k);
    for (int i = 0; i < k; ++i) {
      const int actual_index = output_indices[static_cast<size_t>(seg) * k + i];
      const float actual_value =
          __half2float(output_values[static_cast<size_t>(seg) * k + i]);
      if (actual_index != expected.indices[i] || actual_value != expected.values[i]) {
        std::fprintf(stderr,
                     "random segment %d mismatch at %d: got (%d, %.5f) expected (%d, %.5f)\n",
                     seg,
                     i,
                     actual_index,
                     actual_value,
                     expected.indices[i],
                     expected.values[i]);
        return false;
      }
    }
  }
  return true;
}

bool check_dispatch_policy() {
  return radix_topk::histogram_ctas_per_segment(1) == 4 &&
         radix_topk::histogram_ctas_per_segment(8) == 4 &&
         radix_topk::histogram_ctas_per_segment(32) == 2 &&
         radix_topk::histogram_ctas_per_segment(128) == 1;
}

bool check_partial_histogram_splitk_contract() {
  return radix_topk::kPartialHistogramMaxCtasPerSegment == 4 &&
         radix_topk::histogram_ctas_per_segment(1) <=
             radix_topk::kPartialHistogramMaxCtasPerSegment &&
         radix_topk::histogram_ctas_per_segment(8) <=
             radix_topk::kPartialHistogramMaxCtasPerSegment &&
         radix_topk::histogram_ctas_per_segment(32) <=
             radix_topk::kPartialHistogramMaxCtasPerSegment &&
         radix_topk::histogram_ctas_per_segment(128) <=
             radix_topk::kPartialHistogramMaxCtasPerSegment;
}

std::vector<float> make_random_input(int seg_num, int seg_len, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-64.0f, 64.0f);
  std::vector<float> values(static_cast<size_t>(seg_num) * seg_len);
  for (float& value : values) {
    value = __half2float(__float2half(dist(rng)));
  }
  return values;
}

std::vector<float> make_duplicate_heavy_input(int seg_num, int seg_len) {
  std::vector<float> values(static_cast<size_t>(seg_num) * seg_len);
  for (int seg = 0; seg < seg_num; ++seg) {
    const size_t base = static_cast<size_t>(seg) * seg_len;
    for (int i = 0; i < seg_len; ++i) {
      if (i < 64) {
        values[base + static_cast<size_t>(i)] =
            200.0f - static_cast<float>(seg * 64 + i);
      } else {
        const int lane = (i + seg) % 5;
        values[base + static_cast<size_t>(i)] = static_cast<float>(lane - 2);
      }
    }
  }
  return values;
}

float quantize_to_half_float(float value);

std::vector<float> make_topk50_all_equal_input(int seg_len) {
  return std::vector<float>(static_cast<size_t>(seg_len), quantize_to_half_float(5.0f));
}

std::vector<float> make_topk50_mixed_better_equal_input(int seg_len) {
  std::vector<float> values(static_cast<size_t>(seg_len), quantize_to_half_float(100.0f));
  constexpr int kBetterPrefix = 32;
  for (int i = 0; i < kBetterPrefix; ++i) {
    values[static_cast<size_t>(i)] = quantize_to_half_float(200.0f - static_cast<float>(i));
  }
  return values;
}

std::vector<int> make_sequential_indices(int count) {
  std::vector<int> indices(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    indices[static_cast<size_t>(i)] = i;
  }
  return indices;
}

float quantize_to_half_float(float value) {
  return __half2float(__float2half(value));
}

std::vector<float> make_optimized_special_values_input(int seg_num, int seg_len) {
  std::vector<float> values(static_cast<size_t>(seg_num) * seg_len);
  for (int seg = 0; seg < seg_num; ++seg) {
    const size_t base = static_cast<size_t>(seg) * seg_len;
    for (int i = 0; i < seg_len; ++i) {
      values[base + static_cast<size_t>(i)] =
          quantize_to_half_float(-512.0f + static_cast<float>((i + seg) % 113));
    }

    values[base + 3] = __half2float(__float2half(INFINITY));
    values[base + 9] = __half2float(__float2half(INFINITY));
    values[base + 15] = quantize_to_half_float(2048.0f - static_cast<float>(seg));
    values[base + 21] = quantize_to_half_float(2048.0f - static_cast<float>(seg));
    values[base + 27] = quantize_to_half_float(1024.0f + static_cast<float>(seg));
    values[base + 33] = __half2float(__float2half(-INFINITY));
    values[base + 39] = __half2float(__float2half(NAN));
    values[base + 45] = __half2float(__float2half(NAN));
  }
  return values;
}

bool run_compact_topk50_variant(const std::vector<float>& values,
                                const radix_topk::SegmentSelectState& state,
                                bool use_warp_reserved_equal,
                                std::vector<int>* candidate_indices_out,
                                int* candidate_count_out) {
  std::vector<half> host_input(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host_input[i] = __float2half(values[i]);
  }

  half* d_input = nullptr;
  radix_topk::SegmentSelectState* d_state = nullptr;
  int* d_candidate_indices = nullptr;
  int* d_candidate_counts = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * host_input.size()) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_state),
                 sizeof(radix_topk::SegmentSelectState)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_indices),
                 sizeof(int) *
                     static_cast<size_t>(radix_topk::kCompactedCandidateCap)) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_counts), sizeof(int)) !=
          cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_state, &state, sizeof(state), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_candidate_counts);
    cudaFree(d_candidate_indices);
    cudaFree(d_state);
    cudaFree(d_input);
    return false;
  }

  if (use_warp_reserved_equal) {
    radix_topk::compact_candidate_indices_topk50_warp_reserved_equal_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_state, d_candidate_indices,
        d_candidate_counts);
  } else {
    radix_topk::compact_candidate_indices_topk50_warp_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_state, d_candidate_indices,
        d_candidate_counts);
  }

  const bool ok =
      cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
      cudaMemcpy(candidate_indices_out->data(),
                 d_candidate_indices,
                 sizeof(int) * candidate_indices_out->size(),
                 cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(candidate_count_out,
                 d_candidate_counts,
                 sizeof(*candidate_count_out),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_candidate_counts);
  cudaFree(d_candidate_indices);
  cudaFree(d_state);
  cudaFree(d_input);
  return ok;
}

bool check_compact_topk50_warp_reserved_equivalence() {
  const int seg_len = 10000;
  const int k = 50;

  struct CompactTopk50Case {
    const char* name;
    std::vector<float> values;
    bool check_oracle;
    std::vector<int> expected_indices;
  };

  const std::vector<CompactTopk50Case> cases = {
      {"duplicate-heavy", make_duplicate_heavy_input(1, seg_len), false, {}},
      {"all-equal", make_topk50_all_equal_input(seg_len), true, make_sequential_indices(k)},
      {"mixed-better-equal", make_topk50_mixed_better_equal_input(seg_len), false, {}},
  };

  for (const CompactTopk50Case& test_case : cases) {
    radix_topk::SegmentSelectState state{};
    if (!run_gpu_cutoff_selection(test_case.values, k, &state)) {
      return false;
    }

    std::vector<int> old_indices(radix_topk::kCompactedCandidateCap, -1);
    std::vector<int> new_indices(radix_topk::kCompactedCandidateCap, -1);
    int old_count = 0;
    int new_count = 0;
    if (!run_compact_topk50_variant(test_case.values, state, false, &old_indices, &old_count) ||
        !run_compact_topk50_variant(test_case.values, state, true, &new_indices, &new_count)) {
      return false;
    }

    if (old_count != new_count) {
      return false;
    }
    old_indices.resize(static_cast<size_t>(old_count));
    new_indices.resize(static_cast<size_t>(new_count));
    std::sort(old_indices.begin(), old_indices.end());
    std::sort(new_indices.begin(), new_indices.end());
    if (old_indices != new_indices) {
      return false;
    }

    if (test_case.check_oracle) {
      const radix_topk::ReferenceTopKResult expected =
          radix_topk::cpu_reference_topk(test_case.values, seg_len, k);
      if (expected.indices != test_case.expected_indices) {
        return false;
      }

      std::vector<int> expected_sorted = expected.indices;
      std::sort(expected_sorted.begin(), expected_sorted.end());
      if (old_indices != expected_sorted || new_indices != expected_sorted) {
        return false;
      }
    }
  }
  return true;
}

bool check_compaction_warp_reserved_baseline_contract() {
  return radix_topk::kOptimizedSegLen == 10000 &&
         radix_topk::kOptimizedK == 50 &&
         radix_topk::kOptimizedCandidateCap == 64;
}

bool check_compact_topk50_warp_kernel_selection() {
  const int seg_num = 1;
  const int seg_len = 10000;
  const int k = 50;
  const std::vector<float> host_values = make_duplicate_heavy_input(seg_num, seg_len);
  std::vector<half> host_input(host_values.size());
  for (size_t i = 0; i < host_values.size(); ++i) {
    host_input[i] = __float2half(host_values[i]);
  }

  radix_topk::SegmentSelectState host_state{};
  if (!run_gpu_cutoff_selection(host_values, k, &host_state)) {
    return false;
  }

  half* d_input = nullptr;
  radix_topk::SegmentSelectState* d_state = nullptr;
  int* d_candidate_indices = nullptr;
  int* d_candidate_counts = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_state), sizeof(radix_topk::SegmentSelectState)) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_indices),
                 sizeof(int) * static_cast<size_t>(radix_topk::kOptimizedCandidateCap)) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_counts), sizeof(int)) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_state,
                 &host_state,
                 sizeof(host_state),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_candidate_counts);
    cudaFree(d_candidate_indices);
    cudaFree(d_state);
    cudaFree(d_input);
    return false;
  }

  radix_topk::compact_candidate_indices_topk50_warp_kernel<<<seg_num, 256>>>(
      d_input, seg_len, d_state, d_candidate_indices, d_candidate_counts);
  std::vector<int> candidate_indices(radix_topk::kOptimizedCandidateCap, -1);
  int candidate_count = 0;
  const bool ok =
      cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
      cudaMemcpy(candidate_indices.data(),
                 d_candidate_indices,
                 sizeof(int) * candidate_indices.size(),
                 cudaMemcpyDeviceToHost) == cudaSuccess &&
      cudaMemcpy(&candidate_count,
                 d_candidate_counts,
                 sizeof(candidate_count),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_candidate_counts);
  cudaFree(d_candidate_indices);
  cudaFree(d_state);
  cudaFree(d_input);
  if (!ok || candidate_count != k) {
    return false;
  }

  candidate_indices.resize(static_cast<size_t>(candidate_count));
  std::sort(candidate_indices.begin(), candidate_indices.end());

  const radix_topk::ReferenceTopKResult expected =
      radix_topk::cpu_reference_topk(host_values, seg_len, k);
  std::vector<int> expected_indices = expected.indices;
  std::sort(expected_indices.begin(), expected_indices.end());
  return candidate_indices == expected_indices;
}

bool run_random_gpu_case(int seg_num, int seg_len, int k, uint32_t seed) {
  const std::vector<float> host_values = make_random_input(seg_num, seg_len, seed);
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
  std::vector<int> candidate_counts(seg_num, 0);
  const auto workspace_view =
      radix_topk::make_candidate_compaction_workspace(d_workspace, seg_num);
  const bool ok = status == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(output_values.data(),
                             d_output_values,
                             sizeof(half) * output_values.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(output_indices.data(),
                             d_output_indices,
                             sizeof(int) * output_indices.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(candidate_counts.data(),
                             workspace_view.candidate_counts,
                             sizeof(int) * candidate_counts.size(),
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
      const int actual_index = output_indices[static_cast<size_t>(seg) * k + i];
      const float actual_value =
          __half2float(output_values[static_cast<size_t>(seg) * k + i]);
      if (actual_index != expected.indices[i] || actual_value != expected.values[i]) {
        std::fprintf(stderr,
                     "random case seed=%u shape=(%d,%d,%d) mismatch seg=%d rank=%d: got (%d, %.8g) expected (%d, %.8g)\n",
                     seed,
                     seg_num,
                     seg_len,
                     k,
                     seg,
                     i,
                     actual_index,
                     actual_value,
                     expected.indices[i],
                     expected.values[i]);
        return false;
      }
    }
  }
  return true;
}

bool check_gpu_small_batch_shapes() {
  return run_random_gpu_case(1, 10000, 50, 7u) &&
         run_random_gpu_case(8, 10000, 50, 17u) &&
         run_random_gpu_case(32, 10000, 50, 23u);
}

bool check_final_topk50_kernel_ordering() {
  const int seg_num = 1;
  const int seg_len = radix_topk::kOptimizedCandidateCap;
  std::vector<float> host_values(static_cast<size_t>(seg_len));
  for (int i = 0; i < seg_len; ++i) {
    host_values[static_cast<size_t>(i)] = quantize_to_half_float(-static_cast<float>(i));
  }
  host_values[0] = quantize_to_half_float(42.0f);
  host_values[1] = quantize_to_half_float(42.0f);
  host_values[2] = __half2float(__float2half(INFINITY));
  host_values[3] = __half2float(__float2half(INFINITY));
  host_values[4] = quantize_to_half_float(17.0f);
  host_values[5] = quantize_to_half_float(17.0f);
  host_values[6] = __half2float(__float2half(NAN));
  host_values[7] = __half2float(__float2half(-INFINITY));

  std::vector<half> host_input(host_values.size());
  for (size_t i = 0; i < host_values.size(); ++i) {
    host_input[i] = __float2half(host_values[i]);
  }
  std::vector<int> host_candidate_indices(radix_topk::kCompactedCandidateCap, 0);
  for (int i = 0; i < radix_topk::kOptimizedCandidateCap; ++i) {
    host_candidate_indices[static_cast<size_t>(i)] =
        radix_topk::kOptimizedCandidateCap - 1 - i;
  }
  const int host_candidate_count = radix_topk::kOptimizedCandidateCap;

  half* d_input = nullptr;
  int* d_candidate_indices = nullptr;
  int* d_candidate_counts = nullptr;
  half* d_output_values = nullptr;
  int* d_output_indices = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_indices),
                 sizeof(int) * host_candidate_indices.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_counts), sizeof(int)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_output_values),
                 sizeof(half) * static_cast<size_t>(radix_topk::kOptimizedK)) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_output_indices),
                 sizeof(int) * static_cast<size_t>(radix_topk::kOptimizedK)) !=
          cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_candidate_indices,
                 host_candidate_indices.data(),
                 sizeof(int) * host_candidate_indices.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_candidate_counts,
                 &host_candidate_count,
                 sizeof(host_candidate_count),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_output_indices);
    cudaFree(d_output_values);
    cudaFree(d_candidate_counts);
    cudaFree(d_candidate_indices);
    cudaFree(d_input);
    return false;
  }

  radix_topk::final_topk50_kernel<<<seg_num, radix_topk::kOptimizedCandidateCap>>>(
      d_input,
      seg_len,
      d_candidate_indices,
      d_candidate_counts,
      d_output_values,
      d_output_indices);
  std::vector<half> output_values(static_cast<size_t>(radix_topk::kOptimizedK));
  std::vector<int> output_indices(static_cast<size_t>(radix_topk::kOptimizedK));
  const bool ok = cudaGetLastError() == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(output_values.data(),
                             d_output_values,
                             sizeof(half) * output_values.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(output_indices.data(),
                             d_output_indices,
                             sizeof(int) * output_indices.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_output_indices);
  cudaFree(d_output_values);
  cudaFree(d_candidate_counts);
  cudaFree(d_candidate_indices);
  cudaFree(d_input);
  if (!ok) {
    return false;
  }

  const radix_topk::ReferenceTopKResult expected =
      radix_topk::cpu_reference_topk(host_values, seg_len, radix_topk::kOptimizedK);
  for (int i = 0; i < radix_topk::kOptimizedK; ++i) {
    const int actual_index = output_indices[static_cast<size_t>(i)];
    const float actual_value = __half2float(output_values[static_cast<size_t>(i)]);
    if (actual_index != expected.indices[i] || actual_value != expected.values[i]) {
      std::fprintf(stderr,
                   "final_topk50 mismatch at rank %d: got (%d, %.8g) expected (%d, %.8g)\n",
                   i,
                   actual_index,
                   actual_value,
                   expected.indices[i],
                   expected.values[i]);
      return false;
    }
  }
  return true;
}

bool check_gpu_optimized_nan_inf_regression() {
  const int seg_num = 2;
  const int seg_len = 10000;
  const int k = 50;
  const std::vector<float> host_values = make_optimized_special_values_input(seg_num, seg_len);
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
  std::vector<int> candidate_counts(seg_num, 0);
  const auto workspace_view =
      radix_topk::make_candidate_compaction_workspace(d_workspace, seg_num);
  const bool ok = status == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(output_values.data(),
                             d_output_values,
                             sizeof(half) * output_values.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(output_indices.data(),
                             d_output_indices,
                             sizeof(int) * output_indices.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(candidate_counts.data(),
                             workspace_view.candidate_counts,
                             sizeof(int) * candidate_counts.size(),
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
      const int actual_index = output_indices[static_cast<size_t>(seg) * k + i];
      const float actual_value =
          __half2float(output_values[static_cast<size_t>(seg) * k + i]);
      if (actual_index != expected.indices[i] || actual_value != expected.values[i]) {
        std::fprintf(stderr,
                     "optimized NaN/Inf mismatch seg=%d rank=%d: got (%d, %.8g) expected (%d, %.8g)\n",
                     seg,
                     i,
                     actual_index,
                     actual_value,
                     expected.indices[i],
                     expected.values[i]);
        return false;
      }
    }
    if (candidate_counts[seg] != k) {
      std::fprintf(stderr,
                   "optimized NaN/Inf segment %d candidate count must equal k (%d), got %d\n",
                   seg,
                   k,
                   candidate_counts[seg]);
      return false;
    }
  }
  return true;
}

bool check_gpu_k128_correctness() {
  const int seg_num = 3;
  const int seg_len = 256;
  const int k = 128;
  const std::vector<float> host_values = make_random_input(seg_num, seg_len, 2026u);
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
  std::vector<int> candidate_counts(seg_num, 0);
  const auto workspace_view =
      radix_topk::make_candidate_compaction_workspace(d_workspace, seg_num);
  const bool ok = status == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(output_values.data(),
                             d_output_values,
                             sizeof(half) * output_values.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(output_indices.data(),
                             d_output_indices,
                             sizeof(int) * output_indices.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(candidate_counts.data(),
                             workspace_view.candidate_counts,
                             sizeof(int) * candidate_counts.size(),
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
      const int actual_index = output_indices[static_cast<size_t>(seg) * k + i];
      const float actual_value =
          __half2float(output_values[static_cast<size_t>(seg) * k + i]);
      if (actual_index != expected.indices[i] || actual_value != expected.values[i]) {
        std::fprintf(stderr,
                     "k128 segment %d mismatch at %d: got (%d, %.5f) expected (%d, %.5f)\n",
                     seg,
                     i,
                     actual_index,
                     actual_value,
                     expected.indices[i],
                     expected.values[i]);
        return false;
      }
    }
    if (candidate_counts[seg] != k) {
      std::fprintf(stderr,
                   "k128 segment %d candidate count must equal k (%d), got %d\n",
                   seg,
                   k,
                   candidate_counts[seg]);
      return false;
    }
  }
  return true;
}

bool check_gpu_duplicate_heavy_k50() {
  const int seg_num = 4;
  const int seg_len = 10000;
  const int k = 50;
  const std::vector<float> host_values = make_duplicate_heavy_input(seg_num, seg_len);
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
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemset(d_workspace, 0xa5, workspace_size) != cudaSuccess) {
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
  std::vector<int> candidate_counts(seg_num, 0);
  const auto workspace_view =
      radix_topk::make_candidate_compaction_workspace(d_workspace, seg_num);
  const bool ok = status == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(output_values.data(),
                             d_output_values,
                             sizeof(half) * output_values.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(output_indices.data(),
                             d_output_indices,
                             sizeof(int) * output_indices.size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(candidate_counts.data(),
                             workspace_view.candidate_counts,
                             sizeof(int) * candidate_counts.size(),
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
      const int actual_index = output_indices[static_cast<size_t>(seg) * k + i];
      const float actual_value =
          __half2float(output_values[static_cast<size_t>(seg) * k + i]);
      if (actual_index != expected.indices[i] || actual_value != expected.values[i]) {
        std::fprintf(stderr,
                     "segment %d mismatch at %d: got (%d, %.3f) expected (%d, %.3f)\n",
                     seg,
                     i,
                     actual_index,
                     actual_value,
                     expected.indices[i],
                     expected.values[i]);
        return false;
      }
    }
    if (candidate_counts[seg] != k) {
      std::fprintf(stderr,
                   "segment %d candidate count must equal k (%d), got %d\n",
                   seg,
                   k,
                   candidate_counts[seg]);
      return false;
    }
  }
  return true;
}

bool check_gpu_large_random_regression() {
  const int seg_num = 128;
  const int seg_len = 10000;
  const int k = 50;
  const std::vector<float> host_values = make_random_input(seg_num, seg_len, 1234u);
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
  std::vector<int> candidate_counts(seg_num, 0);
  const auto workspace_view =
      radix_topk::make_candidate_compaction_workspace(d_workspace, seg_num);
  const cudaError_t sync_status = cudaDeviceSynchronize();
  const cudaError_t values_status = cudaMemcpy(output_values.data(),
                                               d_output_values,
                                               sizeof(half) * output_values.size(),
                                               cudaMemcpyDeviceToHost);
  const cudaError_t indices_status = cudaMemcpy(output_indices.data(),
                                                d_output_indices,
                                                sizeof(int) * output_indices.size(),
                                                cudaMemcpyDeviceToHost);
  const cudaError_t counts_status = cudaMemcpy(candidate_counts.data(),
                                               workspace_view.candidate_counts,
                                               sizeof(int) * candidate_counts.size(),
                                               cudaMemcpyDeviceToHost);
  const bool ok = status == cudaSuccess && sync_status == cudaSuccess &&
                  values_status == cudaSuccess && indices_status == cudaSuccess &&
                  counts_status == cudaSuccess;
  cudaFree(d_workspace);
  cudaFree(d_output_indices);
  cudaFree(d_output_values);
  cudaFree(d_input);
  if (!ok) {
    std::fprintf(stderr,
                 "random regression runtime failure: call=%s sync=%s values=%s indices=%s counts=%s\n",
                 cudaGetErrorString(status),
                 cudaGetErrorString(sync_status),
                 cudaGetErrorString(values_status),
                 cudaGetErrorString(indices_status),
                 cudaGetErrorString(counts_status));
    return false;
  }

  for (int seg = 0; seg < seg_num; ++seg) {
    const auto begin = host_values.begin() + static_cast<size_t>(seg) * seg_len;
    const auto end = begin + seg_len;
    const std::vector<float> segment(begin, end);
    const radix_topk::ReferenceTopKResult expected =
        radix_topk::cpu_reference_topk(segment, seg_len, k);
    for (int i = 0; i < k; ++i) {
      const int actual_index = output_indices[static_cast<size_t>(seg) * k + i];
      const float actual_value =
          __half2float(output_values[static_cast<size_t>(seg) * k + i]);
      if (actual_index != expected.indices[i] || actual_value != expected.values[i]) {
        std::fprintf(stderr,
                     "random segment %d mismatch at %d: got (%d, %.5f) expected (%d, %.5f), candidates=%d\n",
                     seg,
                     i,
                     actual_index,
                     actual_value,
                     expected.indices[i],
                     expected.values[i],
                     candidate_counts[seg]);
        return false;
      }
    }
    if (candidate_counts[seg] != k) {
      std::fprintf(stderr,
                   "random segment %d candidate count must equal k (%d), got %d\n",
                   seg,
                   k,
                   candidate_counts[seg]);
      return false;
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
  if (!check_topk50_histogram_kernels()) {
    std::fprintf(stderr, "topk50 histogram kernel check failed\n");
    return 1;
  }
  if (!check_benchmark_target_matrix()) {
    std::fprintf(stderr, "benchmark target matrix is incorrect\n");
    return 1;
  }
  if (!check_benchmark_measurement_contract()) {
    std::fprintf(stderr, "benchmark measurement contract is incorrect\n");
    return 1;
  }
  if (!check_stage_timing_contract()) {
    std::fprintf(stderr, "stage timing contract is incorrect\n");
    return 1;
  }
  if (!check_optimized_state_contract()) {
    std::fprintf(stderr, "optimized state contract is incorrect\n");
    return 1;
  }
  if (!check_optimized_workspace_contract()) {
    std::fprintf(stderr, "optimized workspace contract is incorrect\n");
    return 1;
  }
  if (!check_optimized_workspace_layout_contract()) {
    std::fprintf(stderr, "optimized workspace layout contract is incorrect\n");
    return 1;
  }
  if (!check_gpu_cutoff_selection()) {
    std::fprintf(stderr, "gpu cutoff selection check failed\n");
    return 1;
  }
  if (!check_gpu_cutoff_selection_fallback()) {
    std::fprintf(stderr, "gpu cutoff selection fallback check failed\n");
    return 1;
  }
  if (!check_gpu_small_correctness()) {
    std::fprintf(stderr, "gpu small correctness check failed\n");
    return 1;
  }
  if (!check_dispatch_policy()) {
    std::fprintf(stderr, "dispatch policy is incorrect\n");
    return 1;
  }
  if (!check_partial_histogram_splitk_contract()) {
    std::fprintf(stderr, "partial histogram split-k contract is incorrect\n");
    return 1;
  }
  if (!check_gpu_small_batch_shapes()) {
    std::fprintf(stderr, "gpu small batch shape check failed\n");
    return 1;
  }
  if (!check_compaction_warp_reserved_baseline_contract()) {
    std::fprintf(stderr, "compaction warp-reserved baseline contract is incorrect\n");
    return 1;
  }
  if (!check_compact_topk50_warp_reserved_equivalence()) {
    std::fprintf(stderr, "compact topk50 warp-reserved equivalence check failed\n");
    return 1;
  }
  if (!check_compact_topk50_warp_kernel_selection()) {
    std::fprintf(stderr, "compact topk50 warp kernel selection check failed\n");
    return 1;
  }
  if (!check_final_topk50_kernel_ordering()) {
    std::fprintf(stderr, "final_topk50 ordering check failed\n");
    return 1;
  }
  if (!check_gpu_optimized_nan_inf_regression()) {
    std::fprintf(stderr, "gpu optimized NaN/Inf regression failed\n");
    return 1;
  }
  if (!check_gpu_k128_correctness()) {
    std::fprintf(stderr, "gpu k128 correctness check failed\n");
    return 1;
  }
  if (!check_gpu_duplicate_heavy_k50()) {
    std::fprintf(stderr, "gpu duplicate-heavy k50 check failed\n");
    return 1;
  }
  if (!check_gpu_large_random_regression()) {
    std::fprintf(stderr, "gpu large random regression failed\n");
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
