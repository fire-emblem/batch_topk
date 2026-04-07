# Batch TopK Large-Batch Histogram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a large-batch-only histogram branch for the optimized `(batch, 10000) -> top50` path so that `seg_num >= 1500` can use more aggressive high/low histogram kernels without disturbing the already-stable small-batch path.

**Architecture:** The current optimized path on this branch is stable and instrumented by stage timing. Large-batch runs show histogram cost is large enough to justify a branch-specific optimization, while small-batch runs are already relatively strong. This plan adds two new large-batch histogram kernels and one dispatch split inside the optimized path. The rest of the pipeline stays unchanged, and the implementation is kept only if it improves large-batch benchmark points without materially hurting the `1500`-segment entry shape.

**Tech Stack:** CUDA C++, CMake, CTest, NVCC, Nsight Systems

---

## File Responsibilities

- `src/radix_histogram.cuh`: Keep the current histogram kernels and add large-batch-only high/low histogram variants plus tiny thread-local merge helpers.
- `src/batch_topk.cu`: Only the optimized `seg_len == 10000 && k == 50` branch changes, adding a `seg_num >= 1500` dispatch split for the new histogram kernels.
- `test/test_batch_topk.cu`: Add direct old/new histogram equivalence helpers and a small baseline contract check for the large-batch branch.

### Task 1: Add Failing Equivalence Tests For The Large-Batch Histogram Kernels

**Files:**
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Add old/new high-byte histogram helper**

Add a helper that launches either the current specialized high-byte kernel or the new large-batch version:

```c++
bool run_high_byte_histogram_variant(const std::vector<float>& values,
                                     bool use_large_kernel,
                                     std::vector<unsigned int>* histograms_out) {
  std::vector<half> host_input(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host_input[i] = __float2half(values[i]);
  }

  half* d_input = nullptr;
  unsigned int* d_histograms = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_histograms),
                 sizeof(unsigned int) * histograms_out->size()) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_histograms);
    cudaFree(d_input);
    return false;
  }

  if (use_large_kernel) {
    radix_topk::histogram_high_byte_topk50_large_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_histograms);
  } else {
    radix_topk::histogram_high_byte_topk50_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_histograms);
  }

  const bool ok = cudaGetLastError() == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(histograms_out->data(),
                             d_histograms,
                             sizeof(unsigned int) * histograms_out->size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_histograms);
  cudaFree(d_input);
  return ok;
}
```

- [ ] **Step 2: Add old/new low-byte histogram helper**

Add a matching helper for the low-byte kernels:

```c++
bool run_low_byte_histogram_large_variant(const std::vector<float>& values,
                                          const radix_topk::SegmentSelectState& state,
                                          bool use_large_kernel,
                                          std::vector<unsigned int>* histograms_out) {
  std::vector<half> host_input(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    host_input[i] = __float2half(values[i]);
  }

  half* d_input = nullptr;
  radix_topk::SegmentSelectState* d_state = nullptr;
  unsigned int* d_histograms = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_state), sizeof(radix_topk::SegmentSelectState)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_histograms),
                 sizeof(unsigned int) * histograms_out->size()) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_state,
                 &state,
                 sizeof(state),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_histograms);
    cudaFree(d_state);
    cudaFree(d_input);
    return false;
  }

  if (use_large_kernel) {
    radix_topk::histogram_low_byte_topk50_large_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_state, d_histograms);
  } else {
    radix_topk::histogram_low_byte_topk50_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_state, d_histograms);
  }

  const bool ok = cudaGetLastError() == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
                  cudaMemcpy(histograms_out->data(),
                             d_histograms,
                             sizeof(unsigned int) * histograms_out->size(),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  cudaFree(d_histograms);
  cudaFree(d_state);
  cudaFree(d_input);
  return ok;
}
```

- [ ] **Step 3: Add the failing equivalence tests and baseline contract**

Add deterministic old/new kernel equivalence tests for both high and low byte paths:

```c++
bool check_large_batch_high_histogram_equivalence() {
  const int seg_len = 10000;
  const std::vector<float> values = make_random_input(1, seg_len, 20260401u);
  std::vector<unsigned int> old_hist(256, 0);
  std::vector<unsigned int> new_hist(256, 0);
  if (!run_high_byte_histogram_variant(values, false, &old_hist) ||
      !run_high_byte_histogram_variant(values, true, &new_hist)) {
    return false;
  }
  return old_hist == new_hist;
}

bool check_large_batch_low_histogram_equivalence() {
  const int seg_len = 10000;
  const int k = 50;
  const std::vector<float> values = make_duplicate_heavy_input(1, seg_len);
  radix_topk::SegmentSelectState state{};
  if (!run_gpu_cutoff_selection(values, k, &state)) {
    return false;
  }

  std::vector<unsigned int> old_hist(256, 0);
  std::vector<unsigned int> new_hist(256, 0);
  if (!run_low_byte_histogram_large_variant(values, state, false, &old_hist) ||
      !run_low_byte_histogram_large_variant(values, state, true, &new_hist)) {
    return false;
  }
  return old_hist == new_hist;
}

bool check_large_batch_histogram_baseline_contract() {
  return radix_topk::kOptimizedSegLen == 10000 &&
         radix_topk::kOptimizedK == 50 &&
         radix_topk::kOptimizedCandidateCap == 64;
}
```

Wire them into `main()` before the existing optimized-path regressions:

```c++
  if (!check_large_batch_histogram_baseline_contract()) {
    std::fprintf(stderr, "large batch histogram baseline contract is incorrect\n");
    return 1;
  }
  if (!check_large_batch_high_histogram_equivalence()) {
    std::fprintf(stderr, "large batch high histogram equivalence check failed\n");
    return 1;
  }
  if (!check_large_batch_low_histogram_equivalence()) {
    std::fprintf(stderr, "large batch low histogram equivalence check failed\n");
    return 1;
  }
```

- [ ] **Step 4: Run the build to verify red**

Run: `cmake --build build -j`

Expected: compile fails because `histogram_high_byte_topk50_large_kernel` and `histogram_low_byte_topk50_large_kernel` do not exist.

- [ ] **Step 5: Commit the red tests**

```bash
git add test/test_batch_topk.cu
git commit -m "test: add large batch histogram regressions"
```

### Task 2: Implement The Large-Batch-Only Histogram Branch

**Files:**
- Modify: `src/radix_histogram.cuh`
- Modify: `src/batch_topk.cu`

- [ ] **Step 1: Add tiny thread-local merge helpers in `src/radix_histogram.cuh`**

Insert a small helper type and merge function:

```c++
struct HistogramMergeEntry {
  unsigned short bin = 0xffffu;
  unsigned short count = 0u;
};

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
```

- [ ] **Step 2: Add the new large-batch high-byte kernel**

Append:

```c++
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
  for (int base = tid; base < seg_len; base += blockDim.x * 4) {
    HistogramMergeEntry entries[4] = {};
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int idx = base + item * blockDim.x;
      if (idx < seg_len) {
        const uint16_t encoded = encode_half_desc(segment_input[idx]);
        merge_histogram_samples(entries, 4,
                                static_cast<unsigned short>((encoded >> 8) & 0xffu));
      }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (entries[i].count != 0u) {
        atomicAdd(&warp_histogram[entries[i].bin], static_cast<unsigned int>(entries[i].count));
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
```

- [ ] **Step 3: Add the new large-batch low-byte kernel**

Append:

```c++
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
  for (int base = tid; base < seg_len; base += blockDim.x * 4) {
    HistogramMergeEntry entries[4] = {};
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int idx = base + item * blockDim.x;
      if (idx < seg_len) {
        const uint16_t encoded = encode_half_desc(segment_input[idx]);
        if ((encoded >> 8) == state.boundary_digit) {
          merge_histogram_samples(entries, 4,
                                  static_cast<unsigned short>(encoded & 0xffu));
        }
      }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (entries[i].count != 0u) {
        atomicAdd(&warp_histogram[entries[i].bin], static_cast<unsigned int>(entries[i].count));
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
```

- [ ] **Step 4: Add the large-batch dispatch split in `src/batch_topk.cu`**

Inside the optimized branch only, split at `seg_num >= 1500`:

```c++
    const bool use_large_batch_hist = seg_num >= 1500;

    if (ctas_per_segment == 1) {
      if (use_large_batch_hist) {
        histogram_high_byte_topk50_large_kernel<<<seg_num, 256, 0, stream>>>(
            d_input, seg_len, workspace.histograms_hi);
      } else {
        histogram_high_byte_topk50_kernel<<<seg_num, 256, 0, stream>>>(
            d_input, seg_len, workspace.histograms_hi);
      }
    } else {
      ...
    }
```

and similarly for the low-byte stage:

```c++
    if (ctas_per_segment == 1) {
      if (use_large_batch_hist) {
        histogram_low_byte_topk50_large_kernel<<<seg_num, 256, 0, stream>>>(
            d_input, seg_len, workspace.states, workspace.histograms_lo);
      } else {
        histogram_low_byte_topk50_kernel<<<seg_num, 256, 0, stream>>>(
            d_input, seg_len, workspace.states, workspace.histograms_lo);
      }
    } else {
      ...
    }
```

Do not touch the generic fallback path or the compaction/final stages.

- [ ] **Step 5: Run tests to verify green**

Run: `cmake --build build -j && ./build/test_batch_topk`

Expected: both new histogram equivalence tests pass, all existing optimized-path regressions stay green, and the generic fallback path still works.

- [ ] **Step 6: Commit**

```bash
git add test/test_batch_topk.cu src/radix_histogram.cuh src/batch_topk.cu
git commit -m "cuda: add large batch histogram branch"
```

### Task 3: Benchmark Gate And Keep-Or-Revert Decision

**Files:**
- Modify: none

- [ ] **Step 1: Rebuild, rerun tests, and benchmark**

Run: `cmake --build build -j && ./build/test_batch_topk && ./build/bench_batch_topk`

Expected:

- `./build/test_batch_topk` exits `0`
- `./build/bench_batch_topk` prints exactly five lines
- the large-batch branch satisfies the gate:
  - `(1500,10000,50)` no worse than `205.52 us`
  - `(6000,10000,50)` better than `1079.45 us`

- [ ] **Step 2: Keep-or-revert decision**

If the benchmark satisfies the gate, keep the implementation and stop here.

If the benchmark does not satisfy the gate, revert the implementation commit from Task 2 and do not keep the branch:

```bash
git revert --no-edit HEAD
```

Expected after revert: `cmake --build build -j && ./build/test_batch_topk` still passes and the branch returns to the previous verified baseline.

## Self-Review

- Spec coverage:
  - The plan covers only the large-batch histogram branch and the keep-or-revert benchmark gate.
  - It does not commit to changes in compaction or final-sort behavior.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to Task N” placeholders remain.
  - Every code-changing step includes concrete code blocks and commands.
- Type consistency:
  - `histogram_high_byte_topk50_large_kernel`, `histogram_low_byte_topk50_large_kernel`, `HistogramMergeEntry`, `merge_histogram_samples`, `kOptimizedSegLen`, `kOptimizedK`, and `kOptimizedCandidateCap` are named consistently across tasks.
