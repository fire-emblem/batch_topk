# Batch TopK First-Nibble Select Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace only the first full-segment selection round in the optimized `(batch, 10000) -> top50` path with a warp-centric highest-4-bit selector and keep the rest of the specialized path stable.

**Architecture:** The existing optimized path already has specialized k=50 compaction and final sort. This plan inserts one new first-round selector that chooses the highest nibble of `encode_half_desc(value)` using warp-local bucket accumulation and shared reduction, then hands off to a light second-nibble refinement and the existing low-byte cutoff, compaction, and final top-k50 kernels. The generic fallback path remains unchanged.

**Tech Stack:** CUDA C++, CUB, CMake, CTest, NVCC, Nsight Systems

---

## File Responsibilities

- `src/radix_first_nibble_select.cuh`: New specialized first-round selector and second-nibble continuation kernels for the optimized path only.
- `src/batch_topk.cu`: Host-side orchestration for the optimized `seg_len == 10000 && k == 50` path. This file decides whether to use the new first-nibble path or the existing fallback path.
- `test/test_batch_topk.cu`: Kernel-level nibble-selector checks and end-to-end optimized-path regressions.
- `docs/superpowers/specs/2026-04-01-batch-topk-first-nibble-select-design.md`: The approved design reference for acceptance criteria and baseline numbers.

### Task 1: Add Failing Tests For First-Nibble Selection

**Files:**
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Write the failing kernel-level first-nibble selector test**

Add a small CPU-derived expectation helper and a direct GPU helper in `test/test_batch_topk.cu`:

```c++
bool run_first_nibble_select_gpu(const std::vector<float>& values,
                                 int k,
                                 radix_topk::SegmentSelectState* state_out) {
  std::vector<half> host_input(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host_input[i] = __float2half(values[i]);
  }

  half* d_input = nullptr;
  radix_topk::SegmentSelectState* d_state = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_state), sizeof(radix_topk::SegmentSelectState)) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_state);
    cudaFree(d_input);
    return false;
  }

  radix_topk::first_nibble_select_kernel<<<1, 256>>>(
      d_input, static_cast<int>(values.size()), k, d_state);
  const bool ok = cudaGetLastError() == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(state_out,
                             d_state,
                             sizeof(*state_out),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_state);
  cudaFree(d_input);
  return ok;
}

bool check_first_nibble_select_kernel() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  int counts[16] = {};
  for (float value : values) {
    const uint16_t encoded = radix_topk::encode_half_desc(__float2half(value));
    ++counts[(encoded >> 12) & 0xf];
  }

  int expected_bucket = -1;
  int expected_selected = 0;
  int expected_live = 0;
  for (int bucket = 0; bucket < 16; ++bucket) {
    if (expected_selected + counts[bucket] < 3) {
      expected_selected += counts[bucket];
      continue;
    }
    expected_bucket = bucket;
    expected_live = counts[bucket];
    break;
  }

  radix_topk::SegmentSelectState state{};
  if (!run_first_nibble_select_gpu(values, 3, &state)) {
    return false;
  }

  return state.prefix == static_cast<uint16_t>(expected_bucket << 12) &&
         state.prefix_mask == 0xf000u &&
         state.selected_count == expected_selected &&
         state.live_count == expected_live;
}
```

Wire it into `main()` before the current optimized-path regressions:

```c++
  if (!check_first_nibble_select_kernel()) {
    std::fprintf(stderr, "first nibble select kernel check failed\n");
    return 1;
  }
```

- [ ] **Step 2: Write the failing end-to-end optimized-path regression for the new first round**

Add one more optimized-path regression that will continue to pass only if the new first-nibble path still feeds the existing stages correctly:

```c++
bool check_gpu_first_nibble_optimized_regression() {
  return run_random_gpu_case(128, 10000, 50, 1234u);
}
```

Wire it into `main()`:

```c++
  if (!check_gpu_first_nibble_optimized_regression()) {
    std::fprintf(stderr, "gpu first nibble optimized regression failed\n");
    return 1;
  }
```

- [ ] **Step 3: Run the build to verify red**

Run: `cmake --build build -j`

Expected: compile fails because `first_nibble_select_kernel` does not exist.

- [ ] **Step 4: Commit the red tests once the failure is observed**

```bash
git add test/test_batch_topk.cu
git commit -m "test: add first nibble select regressions"
```

### Task 2: Implement The First-Nibble Selector And Second-Nibble Continuation

**Files:**
- Create: `src/radix_first_nibble_select.cuh`
- Modify: `src/batch_topk.cu`
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Add the new specialized selector header**

Create `src/radix_first_nibble_select.cuh`:

```c++
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "batch_topk_types.cuh"

namespace radix_topk {

inline constexpr int kFirstNibbleBuckets = 16;
inline constexpr int kFirstNibbleWarpBatch = 20;

__global__ inline void first_nibble_select_kernel(const half* input,
                                                  int seg_len,
                                                  int k,
                                                  SegmentSelectState* states) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;

  __shared__ unsigned int warp_bucket_counts[8][kFirstNibbleBuckets];
  __shared__ unsigned int block_bucket_counts[kFirstNibbleBuckets];

  for (int i = tid; i < 8 * kFirstNibbleBuckets; i += blockDim.x) {
    reinterpret_cast<unsigned int*>(warp_bucket_counts)[i] = 0u;
  }
  for (int i = tid; i < kFirstNibbleBuckets; i += blockDim.x) {
    block_bucket_counts[i] = 0u;
  }
  __syncthreads();

  const int warp_batch_base = (seg * 8 + warp) * kFirstNibbleWarpBatch;
  for (int base = warp_batch_base; base < seg_len; base += 8 * kFirstNibbleWarpBatch) {
    const int local = lane < 20 ? lane : -1;
    const int idx = local >= 0 ? base + local : -1;
    if (idx >= 0 && idx < seg_len) {
      const uint16_t encoded = encode_half_desc(segment_input[idx]);
      const int bucket = (encoded >> 12) & 0xf;
      atomicAdd(&warp_bucket_counts[warp][bucket], 1u);
    }
  }
  __syncthreads();

  for (int bucket = tid; bucket < kFirstNibbleBuckets; bucket += blockDim.x) {
    unsigned int total = 0;
    for (int w = 0; w < 8; ++w) {
      total += warp_bucket_counts[w][bucket];
    }
    block_bucket_counts[bucket] = total;
  }
  __syncthreads();

  if (tid == 0) {
    SegmentSelectState state{};
    int selected = 0;
    for (int bucket = 0; bucket < kFirstNibbleBuckets; ++bucket) {
      const int count = static_cast<int>(block_bucket_counts[bucket]);
      if (selected + count < k) {
        selected += count;
        continue;
      }
      state.prefix = static_cast<uint16_t>(bucket << 12);
      state.prefix_mask = 0xf000u;
      state.selected_count = selected;
      state.live_count = count;
      state.boundary_digit = bucket;
      states[seg] = state;
      return;
    }

    state.prefix = 0xffffu;
    state.prefix_mask = 0xffffu;
    state.selected_count = selected;
    state.live_count = 0;
    state.boundary_digit = 15;
    states[seg] = state;
  }
}

__global__ inline void histogram_second_nibble_kernel(const half* input,
                                                      int seg_len,
                                                      const SegmentSelectState* states,
                                                      unsigned int* histograms_hi) {
  const int seg = blockIdx.x;
  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  unsigned int* segment_histogram = histograms_hi + static_cast<size_t>(seg) * 256u;

  __shared__ unsigned int local[kFirstNibbleBuckets];
  for (int i = threadIdx.x; i < kFirstNibbleBuckets; i += blockDim.x) {
    local[i] = 0u;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded & state.prefix_mask) == state.prefix) {
      atomicAdd(&local[(encoded >> 8) & 0xf], 1u);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kFirstNibbleBuckets; i += blockDim.x) {
    segment_histogram[i] = local[i];
  }
}

__global__ inline void select_second_nibble_kernel(const unsigned int* histograms_hi,
                                                   int seg_num,
                                                   int k,
                                                   SegmentSelectState* states) {
  const int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= seg_num) {
    return;
  }

  const unsigned int* histogram = histograms_hi + static_cast<size_t>(seg) * 256u;
  SegmentSelectState state = states[seg];
  int selected = state.selected_count;
  const int high_nibble = static_cast<int>(state.prefix >> 12);
  for (int bucket = 0; bucket < kFirstNibbleBuckets; ++bucket) {
    const int count = static_cast<int>(histogram[bucket]);
    if (selected + count < k) {
      selected += count;
      continue;
    }
    state.boundary_digit = (high_nibble << 4) | bucket;
    state.prefix = static_cast<uint16_t>(state.boundary_digit << 8);
    state.prefix_mask = 0xff00u;
    state.selected_count = selected;
    state.live_count = count;
    states[seg] = state;
    return;
  }
}

}  // namespace radix_topk
```

- [ ] **Step 2: Integrate the new first-round selector into the optimized path**

In `src/batch_topk.cu`, include the new header and replace only the first high-byte stage for the optimized path:

```c++
#include "radix_first_nibble_select.cuh"
```

Replace the first half of the optimized `seg_len == kOptimizedSegLen && k == kOptimizedK` branch with:

```c++
    first_nibble_select_kernel<<<seg_num, 256, 0, stream>>>(
        d_input, seg_len, k, workspace.states);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }

    histogram_second_nibble_kernel<<<seg_num, 256, 0, stream>>>(
        d_input, seg_len, workspace.states, workspace.histograms_hi);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }

    select_second_nibble_kernel<<<(seg_num + 127) / 128, 128, 0, stream>>>(
        workspace.histograms_hi, seg_num, k, workspace.states);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }
```

Leave the later low-byte histogram, cutoff finalize, specialized compaction, and `final_topk50_kernel` unchanged.

- [ ] **Step 3: Run the full test binary to verify green**

Run: `cmake --build build -j && ./build/test_batch_topk`

Expected: the new nibble-selector tests pass, all existing optimized-path regressions remain green, and the generic fallback path still works.

- [ ] **Step 4: Commit**

```bash
git add src/radix_first_nibble_select.cuh src/batch_topk.cu test/test_batch_topk.cu
git commit -m "cuda: add first nibble selector for topk50"
```

### Task 3: Benchmark The First-Round Selector And Decide Keep Or Revert

**Files:**
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Add a tiny acceptance note check for the baseline values**

Add a lightweight regression note in `test/test_batch_topk.cu` so the baseline values are documented in code review context:

```c++
bool check_first_nibble_baseline_contract() {
  return radix_topk::kOptimizedSegLen == 10000 &&
         radix_topk::kOptimizedK == 50 &&
         radix_topk::kOptimizedCandidateCap == 64;
}
```

Wire it into `main()`:

```c++
  if (!check_first_nibble_baseline_contract()) {
    std::fprintf(stderr, "first nibble baseline contract is incorrect\n");
    return 1;
  }
```

- [ ] **Step 2: Rebuild, rerun tests, and benchmark**

Run: `cmake --build build -j && ./build/test_batch_topk && ./build/bench_batch_topk`

Expected:

- `./build/test_batch_topk` exits `0`
- `./build/bench_batch_topk` prints exactly five lines
- the new optimized path beats the current baseline on this machine:
  - `(128,10000,50)` better than `50.07 us`
  - `(6000,10000,50)` better than `917.04 us`

- [ ] **Step 3: Keep-or-revert decision**

If the benchmark beats the baseline, keep the implementation and commit the test-note change:

```bash
git add test/test_batch_topk.cu
git commit -m "test: document first nibble select baseline contract"
```

If the benchmark does not beat the baseline, revert the implementation commit from Task 2 and do not proceed to a fuller nibble-round design:

```bash
git restore test/test_batch_topk.cu
git revert --no-edit HEAD
```

Expected after revert: `cmake --build build -j && ./build/test_batch_topk` still passes and the branch returns to the previous verified baseline.

## Self-Review

- Spec coverage:
  - The plan covers only the first-round nibble selector, handoff to the existing later stages, required tests, and the keep-or-revert benchmark gate.
  - It does not commit to a full nibble-round state machine.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to Task N” placeholders remain.
  - Each code-changing step contains concrete code blocks and commands.
- Type consistency:
  - `first_nibble_select_kernel`, `histogram_second_nibble_kernel`, `select_second_nibble_kernel`, `SegmentSelectState`, `kOptimizedSegLen`, `kOptimizedK`, and `kOptimizedCandidateCap` are named consistently across tasks.
