#include <cstdlib>
#include <cstdio>
#include <vector>

#include "../src/batch_topk_stage_timing.cuh"
#include "batch_topk_benchmark.cuh"
#include "batch_topk.cuh"

namespace {

bool fill_input(std::vector<half>* host_input) {
  for (size_t i = 0; i < host_input->size(); ++i) {
    const int raw = static_cast<int>((i * 17u + 13u) % 2048u) - 1024;
    (*host_input)[i] = __float2half(static_cast<float>(raw) / 16.0f);
  }
  return true;
}

bool run_benchmark_case(const radix_topk::BenchmarkCase& benchmark_case,
                        float* latency_us,
                        radix_topk::BatchTopkStageTiming* stage_timing) {
  const int seg_num = benchmark_case.seg_num;
  const int seg_len = benchmark_case.seg_len;
  const int k = benchmark_case.k;
  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(seg_num, seg_len, k);
  if (workspace_size == 0) {
    return false;
  }

  std::vector<half> host_input(static_cast<size_t>(seg_num) * seg_len);
  fill_input(&host_input);

  half* d_input = nullptr;
  half* d_values = nullptr;
  int* d_indices = nullptr;
  void* d_workspace = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input),
                 sizeof(half) * static_cast<size_t>(seg_num) * seg_len) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_values),
                 sizeof(half) * static_cast<size_t>(seg_num) * k) !=
          cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_indices),
                 sizeof(int) * static_cast<size_t>(seg_num) * k) !=
          cudaSuccess ||
      cudaMalloc(&d_workspace, workspace_size) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return false;
  }

  constexpr int kWarmupIterations = 5;
  constexpr int kMeasureIterations = 20;
  for (int iter = 0; iter < kWarmupIterations; ++iter) {
    const cudaError_t status = radix_topk::batch_topk_half(
        d_input, seg_num, seg_len, k, d_values, d_indices, d_workspace,
        workspace_size, 0);
    if (status != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      cudaFree(d_workspace);
      cudaFree(d_indices);
      cudaFree(d_values);
      cudaFree(d_input);
      return false;
    }
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess ||
      cudaEventCreate(&stop) != cudaSuccess) {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return false;
  }

  if (cudaEventRecord(start) != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return false;
  }
  for (int iter = 0; iter < kMeasureIterations; ++iter) {
    const cudaError_t status = radix_topk::batch_topk_half(
        d_input, seg_num, seg_len, k, d_values, d_indices, d_workspace,
        workspace_size, 0);
    if (status != cudaSuccess) {
      cudaEventDestroy(stop);
      cudaEventDestroy(start);
      cudaFree(d_workspace);
      cudaFree(d_indices);
      cudaFree(d_values);
      cudaFree(d_input);
      return false;
    }
  }
  if (cudaEventRecord(stop) != cudaSuccess ||
      cudaEventSynchronize(stop) != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return false;
  }

  float elapsed_ms = 0.0f;
  if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return false;
  }
  *latency_us = elapsed_ms * 1000.0f / kMeasureIterations;

  if (stage_timing != nullptr) {
    *stage_timing = radix_topk::BatchTopkStageTiming{};
    radix_topk::set_batch_topk_stage_timing_sink(stage_timing);
    const cudaError_t status = radix_topk::batch_topk_half(
        d_input, seg_num, seg_len, k, d_values, d_indices, d_workspace,
        workspace_size, 0);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    radix_topk::set_batch_topk_stage_timing_sink(nullptr);
    if (status != cudaSuccess || sync_status != cudaSuccess) {
      cudaFree(d_workspace);
      cudaFree(d_indices);
      cudaFree(d_values);
      cudaFree(d_input);
      return false;
    }
  }

  if (cudaEventDestroy(stop) != cudaSuccess ||
      cudaEventDestroy(start) != cudaSuccess) {
    cudaFree(d_workspace);
    cudaFree(d_indices);
    cudaFree(d_values);
    cudaFree(d_input);
    return false;
  }

  if (cudaFree(d_workspace) != cudaSuccess || cudaFree(d_indices) != cudaSuccess ||
      cudaFree(d_values) != cudaSuccess || cudaFree(d_input) != cudaSuccess) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const bool print_stage_timing = std::getenv("BATCH_TOPK_STAGE_TIMING") != nullptr;
  for (const radix_topk::BenchmarkCase& benchmark_case :
       radix_topk::kPrimaryBenchmarkCases) {
    float latency_us = 0.0f;
    radix_topk::BatchTopkStageTiming stage_timing{};
    if (!run_benchmark_case(
            benchmark_case, &latency_us, print_stage_timing ? &stage_timing : nullptr)) {
      std::fprintf(stderr,
                   "benchmark failed for seg_num=%d seg_len=%d k=%d\n",
                   benchmark_case.seg_num,
                   benchmark_case.seg_len,
                   benchmark_case.k);
      return 1;
    }

    std::printf(
        "seg_num=%d seg_len=%d k=%d latency_us=%.2f ref_us=%.2f delta_us=%.2f\n",
        benchmark_case.seg_num,
        benchmark_case.seg_len,
        benchmark_case.k,
        latency_us,
        benchmark_case.ref_us,
        latency_us - benchmark_case.ref_us);

    if (print_stage_timing) {
      std::printf(
          "stage_us seg_num=%d high_hist=%.2f high_select=%.2f low_hist=%.2f finalize=%.2f compact=%.2f final=%.2f\n",
          benchmark_case.seg_num,
          stage_timing.high_byte_hist_us,
          stage_timing.high_byte_select_us,
          stage_timing.low_byte_hist_us,
          stage_timing.finalize_cutoff_us,
          stage_timing.compaction_us,
          stage_timing.final_topk_us);
    }
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    return 1;
  }
  return 0;
}
