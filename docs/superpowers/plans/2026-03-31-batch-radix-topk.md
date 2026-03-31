# Batch Radix TopK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CUDA batch top-k operator for `half` inputs with `seg_len <= 10000`, `k <= 128`, internal index generation, and a radix-select plus final-sort pipeline that is correct first and then tuned against the `11 / 82 / 161 / 241 / 319 us` reference targets.

**Architecture:** The implementation exposes a small public CUDA API, computes a deterministic CPU reference for tests, and builds the GPU operator in layers: type codec, fallback direct top-k path, radix histogram and threshold selection, candidate compaction, and final block sort. Benchmarking is a first-class part of the project and is wired in from the start so tuning can be driven by measurements instead of assumptions.

**Tech Stack:** CUDA C++, CMake, CTest, CUB, NVCC

---

### Task 1: Create Project Skeleton And Build Targets

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/batch_topk.cuh`
- Create: `include/batch_topk_types.cuh`
- Create: `src/batch_topk.cu`
- Create: `test/test_batch_topk.cu`
- Create: `bench/bench_batch_topk.cu`

- [ ] **Step 1: Write the failing build configuration**

```cmake
cmake_minimum_required(VERSION 3.24)
project(batch_radix_topk LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(batch_topk STATIC
  src/batch_topk.cu
)
target_include_directories(batch_topk PUBLIC include)

add_executable(test_batch_topk test/test_batch_topk.cu)
target_link_libraries(test_batch_topk PRIVATE batch_topk)
add_test(NAME test_batch_topk COMMAND test_batch_topk)

add_executable(bench_batch_topk bench/bench_batch_topk.cu)
target_link_libraries(bench_batch_topk PRIVATE batch_topk)
```

- [ ] **Step 2: Add the public API declarations**

```c++
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

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

size_t batch_topk_half_workspace_size(
    int seg_num,
    int seg_len,
    int k);
```

- [ ] **Step 3: Add a stub implementation that fails at runtime**

```c++
#include "batch_topk.cuh"

cudaError_t batch_topk_half(
    const half*,
    int,
    int,
    int,
    half*,
    int*,
    void*,
    size_t,
    cudaStream_t) {
  return cudaErrorNotSupported;
}

size_t batch_topk_half_workspace_size(int, int, int) {
  return 0;
}
```

- [ ] **Step 4: Add minimal test and benchmark entrypoints that depend on the API**

```c++
#include "batch_topk.cuh"

int main() {
  return batch_topk_half_workspace_size(1, 10000, 50) == 0 ? 1 : 0;
}
```

```c++
#include "batch_topk.cuh"

int main() {
  return batch_topk_half_workspace_size(128, 10000, 50) == 0 ? 1 : 0;
}
```

- [ ] **Step 5: Run the build to verify it fails for the expected reason**

Run: `cmake -S . -B build && cmake --build build -j`

Expected: build succeeds, but `ctest --test-dir build --output-on-failure` fails because the workspace query stub returns `0`

- [ ] **Step 6: Make the minimal code change to get the first test green**

```c++
size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k) {
  if (seg_num <= 0 || seg_len <= 0 || k <= 0) {
    return 0;
  }
  return static_cast<size_t>(seg_num) * 256;
}
```

- [ ] **Step 7: Run tests to verify green**

Run: `ctest --test-dir build --output-on-failure`

Expected: `test_batch_topk` passes

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt include/batch_topk.cuh src/batch_topk.cu test/test_batch_topk.cu bench/bench_batch_topk.cu
git commit -m "build: add batch topk project skeleton"
```

### Task 2: Add CPU Reference And API Validation Tests

**Files:**
- Modify: `include/batch_topk_types.cuh`
- Modify: `test/test_batch_topk.cu`
- Modify: `src/batch_topk.cu`

- [ ] **Step 1: Write failing tests for API validation and deterministic reference ordering**

```c++
static void test_workspace_validation() {
  if (batch_topk_half_workspace_size(1, 10001, 50) != 0) std::abort();
  if (batch_topk_half_workspace_size(1, 10000, 129) != 0) std::abort();
}

static void test_reference_tie_break() {
  std::vector<float> values = {5.0f, 7.0f, 7.0f, 1.0f};
  std::vector<int> expected_indices = {1, 2};
  auto actual = cpu_reference_topk(values, 4, 2);
  if (actual.indices != expected_indices) std::abort();
}
```

- [ ] **Step 2: Run the targeted test to verify red**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`

Expected: compile fails because `cpu_reference_topk` is not defined

- [ ] **Step 3: Add the shared candidate and reference result types**

```c++
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

struct Candidate {
  uint32_t encoded_key;
  half value;
  int index;
};

struct ReferenceTopKResult {
  std::vector<float> values;
  std::vector<int> indices;
};
```

- [ ] **Step 4: Implement CPU reference and argument validation**

```c++
static ReferenceTopKResult cpu_reference_topk(
    const std::vector<float>& values,
    int seg_len,
    int k) {
  std::vector<int> order(seg_len);
  for (int i = 0; i < seg_len; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    if (values[a] != values[b]) return values[a] > values[b];
    return a < b;
  });

  ReferenceTopKResult result;
  result.values.resize(k);
  result.indices.resize(k);
  for (int i = 0; i < k; ++i) {
    result.values[i] = values[order[i]];
    result.indices[i] = order[i];
  }
  return result;
}

size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k) {
  if (seg_num <= 0 || seg_len <= 0 || seg_len > 10000 || k <= 0 || k > 128) {
    return 0;
  }
  return static_cast<size_t>(seg_num) * 4096;
}
```

- [ ] **Step 5: Run tests to verify green**

Run: `ctest --test-dir build --output-on-failure`

Expected: validation and CPU-reference tests pass

- [ ] **Step 6: Commit**

```bash
git add include/batch_topk_types.cuh src/batch_topk.cu test/test_batch_topk.cu
git commit -m "test: add validation and cpu topk reference"
```

### Task 3: Implement Half Codec And Comparator Semantics

**Files:**
- Create: `src/type_codec.cuh`
- Modify: `include/batch_topk_types.cuh`
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Write failing tests for half ordering edge cases**

```c++
static void test_half_codec_special_values() {
  uint16_t neg = encode_half_desc(__float2half(-1.0f));
  uint16_t pos = encode_half_desc(__float2half(3.0f));
  uint16_t nan = encode_half_desc(__float2half(NAN));
  if (!(pos < neg)) std::abort();
  if (!(nan > neg)) std::abort();
}
```

- [ ] **Step 2: Run test to verify red**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`

Expected: compile fails because `encode_half_desc` is not defined

- [ ] **Step 3: Implement codec helpers and comparison policy**

```c++
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

inline uint16_t normalize_half_bits(uint16_t bits) {
  const uint16_t exp_mask = 0x7c00u;
  const uint16_t mantissa_mask = 0x03ffu;
  if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0) {
    return 0x0000u;
  }
  if ((bits & 0x7fffu) == 0) {
    return 0x0000u;
  }
  return bits;
}

inline uint16_t encode_half_asc(half value) {
  uint16_t bits = normalize_half_bits(*reinterpret_cast<uint16_t*>(&value));
  return (bits & 0x8000u) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits ^ 0x8000u);
}

inline uint16_t encode_half_desc(half value) {
  return static_cast<uint16_t>(~encode_half_asc(value));
}

inline bool candidate_better(const Candidate& lhs, const Candidate& rhs) {
  float lv = __half2float(lhs.value);
  float rv = __half2float(rhs.value);
  if (lv != rv) return lv > rv;
  return lhs.index < rhs.index;
}
```

- [ ] **Step 4: Expose codec declarations through the shared types header**

```c++
uint16_t encode_half_asc(half value);
uint16_t encode_half_desc(half value);
bool candidate_better(const Candidate& lhs, const Candidate& rhs);
```

- [ ] **Step 5: Run tests to verify green**

Run: `ctest --test-dir build --output-on-failure`

Expected: codec tests pass for sign, NaN, and tie-break behavior

- [ ] **Step 6: Commit**

```bash
git add include/batch_topk_types.cuh src/type_codec.cuh test/test_batch_topk.cu
git commit -m "core: add half codec and comparison helpers"
```

### Task 4: Build A Correct GPU Baseline With Direct Segment TopK

**Files:**
- Create: `src/final_block_sort.cuh`
- Modify: `src/batch_topk.cu`
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Write a failing end-to-end GPU correctness test**

```c++
static void test_gpu_small_correctness() {
  const int seg_num = 2;
  const int seg_len = 8;
  const int k = 3;
  const std::vector<float> host_values = {
      1, 9, 2, 8, 3, 7, 4, 6,
      5, 4, 9, 1, 9, 2, 0, 8};
  run_gpu_and_compare(host_values, seg_num, seg_len, k);
}
```

- [ ] **Step 2: Run the test to verify red**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`

Expected: runtime failure because `batch_topk_half` returns `cudaErrorNotSupported`

- [ ] **Step 3: Implement a minimal direct path that loads one segment per CTA and emits correct top-k**

```c++
__global__ void direct_segment_topk_kernel(
    const half* input,
    int seg_len,
    int k,
    half* output_values,
    int* output_indices) {
  const int seg = blockIdx.x;
  extern __shared__ Candidate shared[];

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    shared[i].value = input[seg * seg_len + i];
    shared[i].index = i;
    shared[i].encoded_key = encode_half_desc(shared[i].value);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 0; i < k; ++i) {
      int best = i;
      for (int j = i + 1; j < seg_len; ++j) {
        if (candidate_better(shared[j], shared[best])) best = j;
      }
      Candidate tmp = shared[i];
      shared[i] = shared[best];
      shared[best] = tmp;
      output_values[seg * k + i] = shared[i].value;
      output_indices[seg * k + i] = shared[i].index;
    }
  }
}
```

- [ ] **Step 4: Wire the direct kernel into the host API for supported inputs**

```c++
cudaError_t batch_topk_half(...) {
  if (!d_input || !d_output_values || !d_output_indices) return cudaErrorInvalidValue;
  if (seg_num <= 0 || seg_len <= 0 || seg_len > 10000 || k <= 0 || k > 128) return cudaErrorInvalidValue;
  if (workspace_bytes < batch_topk_half_workspace_size(seg_num, seg_len, k)) return cudaErrorInvalidValue;
  direct_segment_topk_kernel<<<seg_num, 256, sizeof(Candidate) * seg_len, stream>>>(
      d_input, seg_len, k, d_output_values, d_output_indices);
  return cudaGetLastError();
}
```

- [ ] **Step 5: Run tests to verify green**

Run: `ctest --test-dir build --output-on-failure`

Expected: small correctness tests pass against the CPU reference

- [ ] **Step 6: Commit**

```bash
git add src/final_block_sort.cuh src/batch_topk.cu test/test_batch_topk.cu
git commit -m "cuda: add direct topk baseline kernel"
```

### Task 5: Add Segment State, Histogram Pass, And Threshold Selection

**Files:**
- Create: `src/radix_histogram.cuh`
- Create: `src/radix_select_state.cuh`
- Modify: `include/batch_topk_types.cuh`
- Modify: `src/batch_topk.cu`
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Write failing tests for radix boundary selection**

```c++
static void test_radix_boundary_state() {
  std::vector<float> values = {9, 8, 7, 6, 5, 4, 3, 2};
  auto state = simulate_radix_boundary(values, 8, 3);
  if (state.selected_count != 3) std::abort();
}
```

- [ ] **Step 2: Run the tests to verify red**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`

Expected: compile fails because `simulate_radix_boundary` and radix state types are not defined

- [ ] **Step 3: Add the per-segment state type**

```c++
struct SegmentSelectState {
  uint16_t prefix;
  uint16_t prefix_mask;
  int selected_count;
  int live_count;
  int boundary_digit;
};
```

- [ ] **Step 4: Implement histogram and host-side simulation helper**

```c++
__global__ void histogram_pass_kernel(
    const half* input,
    int seg_len,
    int shift,
    unsigned int* segment_histograms) {
  const int seg = blockIdx.x;
  __shared__ unsigned int local[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    uint16_t key = encode_half_desc(input[seg * seg_len + i]);
    atomicAdd(&local[(key >> shift) & 0xff], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    segment_histograms[seg * 256 + i] = local[i];
  }
}

static SegmentSelectState simulate_radix_boundary(
    const std::vector<float>& values,
    int seg_len,
    int k) {
  std::vector<int> counts(256, 0);
  for (int i = 0; i < seg_len; ++i) {
    half h = __float2half(values[i]);
    counts[(encode_half_desc(h) >> 8) & 0xff] += 1;
  }
  int selected = 0;
  int bucket = 0;
  while (bucket < 256 && selected + counts[bucket] < k) {
    selected += counts[bucket];
    ++bucket;
  }
  return SegmentSelectState{0, 0xff00u, selected, counts[bucket], bucket};
}
```

- [ ] **Step 5: Run tests to verify green**

Run: `ctest --test-dir build --output-on-failure`

Expected: radix-boundary tests pass and existing GPU baseline still passes

- [ ] **Step 6: Commit**

```bash
git add include/batch_topk_types.cuh src/radix_histogram.cuh src/radix_select_state.cuh src/batch_topk.cu test/test_batch_topk.cu
git commit -m "radix: add histogram pass and boundary state"
```

### Task 6: Implement Candidate Compaction And Final Sorted Output

**Files:**
- Create: `src/candidate_compact.cuh`
- Modify: `src/final_block_sort.cuh`
- Modify: `src/batch_topk.cu`
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Write a failing GPU test for duplicate-heavy inputs and `k = 50`**

```c++
static void test_gpu_duplicate_heavy_k50() {
  const int seg_num = 4;
  const int seg_len = 10000;
  const int k = 50;
  auto host_values = make_duplicate_heavy_input(seg_num, seg_len);
  run_gpu_and_compare(host_values, seg_num, seg_len, k);
}
```

- [ ] **Step 2: Run the tests to verify red**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`

Expected: runtime correctness failure because the direct baseline path is still used instead of the radix-select pipeline

- [ ] **Step 3: Add compaction kernel and compacted-candidate sort**

```c++
__global__ void compact_candidates_kernel(
    const half* input,
    int seg_len,
    uint16_t prefix,
    uint16_t prefix_mask,
    Candidate* candidates,
    int* candidate_counts) {
  const int seg = blockIdx.x;
  __shared__ int block_count;
  if (threadIdx.x == 0) block_count = 0;
  __syncthreads();

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    half value = input[seg * seg_len + i];
    uint16_t key = encode_half_desc(value);
    if ((key & prefix_mask) == prefix) {
      int slot = atomicAdd(&block_count, 1);
      candidates[seg * 512 + slot] = Candidate{key, value, i};
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) candidate_counts[seg] = block_count;
}
```

```c++
__global__ void final_candidate_sort_kernel(
    const Candidate* candidates,
    const int* candidate_counts,
    int k,
    half* output_values,
    int* output_indices) {
  const int seg = blockIdx.x;
  extern __shared__ Candidate local[];
  const int count = candidate_counts[seg];
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    local[i] = candidates[seg * 512 + i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::sort(local, local + count, [](const Candidate& a, const Candidate& b) {
      return candidate_better(a, b);
    });
    for (int i = 0; i < k; ++i) {
      output_values[seg * k + i] = local[i].value;
      output_indices[seg * k + i] = local[i].index;
    }
  }
}
```

- [ ] **Step 4: Route the host implementation through radix histogram, boundary selection, compaction, and final sort**

```c++
cudaError_t batch_topk_half(...) {
  validate_arguments_or_return_error(...);
  Workspace ws = make_workspace(d_workspace, workspace_bytes, seg_num);
  histogram_pass_kernel<<<seg_num, 256, 0, stream>>>(d_input, seg_len, 8, ws.histograms);
  launch_boundary_selection(ws.histograms, seg_num, k, ws.states, stream);
  compact_candidates_kernel<<<seg_num, 256, 0, stream>>>(
      d_input, seg_len, ws.prefix, ws.prefix_mask, ws.candidates, ws.counts);
  final_candidate_sort_kernel<<<seg_num, 128, sizeof(Candidate) * 512, stream>>>(
      ws.candidates, ws.counts, k, d_output_values, d_output_indices);
  return cudaGetLastError();
}
```

- [ ] **Step 5: Run tests to verify green**

Run: `ctest --test-dir build --output-on-failure`

Expected: duplicate-heavy, random, and `k = 50` correctness tests pass

- [ ] **Step 6: Commit**

```bash
git add src/candidate_compact.cuh src/final_block_sort.cuh src/batch_topk.cu test/test_batch_topk.cu
git commit -m "cuda: add radix candidate compaction path"
```

### Task 7: Add Benchmark Harness And Track The 4080 Targets

**Files:**
- Modify: `bench/bench_batch_topk.cu`
- Modify: `test/test_batch_topk.cu`
- Modify: `src/batch_topk.cu`

- [ ] **Step 1: Write failing benchmark output checks**

```c++
static void expect_shape_metadata() {
  const std::vector<int> seg_nums = {128, 1500, 3000, 4500, 6000};
  const std::vector<int> refs_us = {11, 82, 161, 241, 319};
  if (seg_nums.size() != refs_us.size()) std::abort();
}
```

- [ ] **Step 2: Run the benchmark build to verify red**

Run: `cmake --build build -j`

Expected: benchmark compiles, but the executable does not yet emit per-shape latency and delta-vs-reference lines

- [ ] **Step 3: Implement the benchmark driver with the exact target matrix**

```c++
int main() {
  const int seg_len = 10000;
  const int k = 50;
  const std::vector<int> seg_nums = {128, 1500, 3000, 4500, 6000};
  const std::vector<float> refs_us = {11.0f, 82.0f, 161.0f, 241.0f, 319.0f};

  for (size_t i = 0; i < seg_nums.size(); ++i) {
    BenchmarkResult result = run_benchmark(seg_nums[i], seg_len, k);
    std::printf("seg_num=%d seg_len=%d k=%d latency_us=%.2f ref_us=%.2f delta_us=%.2f\n",
                seg_nums[i], seg_len, k, result.latency_us, refs_us[i], result.latency_us - refs_us[i]);
  }
  return 0;
}
```

- [ ] **Step 4: Add a random large-shape GPU correctness regression**

```c++
static void test_gpu_large_random_regression() {
  const int seg_num = 128;
  const int seg_len = 10000;
  const int k = 50;
  auto host_values = make_random_input(seg_num, seg_len, 1234);
  run_gpu_and_compare(host_values, seg_num, seg_len, k);
}
```

- [ ] **Step 5: Run tests and benchmark to verify green**

Run: `ctest --test-dir build --output-on-failure`

Expected: all correctness tests pass

Run: `./build/bench_batch_topk`

Expected: the benchmark prints the five target shapes with measured latency, reference latency, and delta

- [ ] **Step 6: Commit**

```bash
git add bench/bench_batch_topk.cu test/test_batch_topk.cu src/batch_topk.cu
git commit -m "bench: add 4080 target tracking"
```

### Task 8: Tune The Fast Path Toward The Reference Targets

**Files:**
- Modify: `src/radix_histogram.cuh`
- Modify: `src/candidate_compact.cuh`
- Modify: `src/final_block_sort.cuh`
- Modify: `src/batch_topk.cu`
- Modify: `bench/bench_batch_topk.cu`

- [ ] **Step 1: Write a failing benchmark expectation note into the tuning loop**

```c++
static const float kTargetUs[] = {11.0f, 82.0f, 161.0f, 241.0f, 319.0f};
```

- [ ] **Step 2: Run the benchmark and capture the current baseline**

Run: `./build/bench_batch_topk`

Expected: the direct implementation is slower than at least one of the target numbers, giving a measurable tuning baseline

- [ ] **Step 3: Replace generic choices with measured fast-path specializations**

```c++
if (k <= 32) {
  launch_final_sort<32>(...);
} else if (k <= 64) {
  launch_final_sort<64>(...);
} else {
  launch_final_sort<128>(...);
}
```

```c++
constexpr int kCompactionCap = 512;
constexpr int kHistogramThreads = 256;
constexpr int kFinalSortThreads = 128;
```

- [ ] **Step 4: Tune and re-run until the benchmark output clearly reports parity status**

Run: `./build/bench_batch_topk`

Expected: each line prints a negative, zero, or positive delta versus reference so progress is explicit

- [ ] **Step 5: Run the full correctness suite after each tuning round**

Run: `ctest --test-dir build --output-on-failure`

Expected: all correctness tests remain green after each optimization pass

- [ ] **Step 6: Commit**

```bash
git add src/radix_histogram.cuh src/candidate_compact.cuh src/final_block_sort.cuh src/batch_topk.cu bench/bench_batch_topk.cu
git commit -m "perf: tune batch radix topk fast path"
```

## Self-Review

- Spec coverage: the plan covers API shape, internal index generation, `half` support, deterministic tie-breaks, radix histogram/selection, candidate compaction, final sort, and explicit benchmarking against the five reference latency targets.
- Placeholder scan: no `TODO`, `TBD`, or deferred implementation markers are left inside the task steps.
- Type consistency: `Candidate`, `ReferenceTopKResult`, `SegmentSelectState`, and the `batch_topk_half` API use consistent names across tasks.
