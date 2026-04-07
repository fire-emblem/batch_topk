# Batch TopK Low-Byte Histogram Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the specialized `histogram_low_byte_topk50_kernel` with a lower-contention `v2` kernel for the optimized `(batch, 10000) -> top50` path and keep the rest of the pipeline unchanged.

**Architecture:** The current optimized path is already stable and benchmarked on `eaa713d`: high-byte histogram, high-byte boundary select, low-byte histogram, cutoff finalize, `k=50` compaction, and `final_topk50`. The current low-byte kernel already uses warp-private shared histograms, so this plan introduces a `v2` kernel that keeps the same output contract but reduces shared atomic pressure by adding a small per-thread register cache before flushing into the existing warp-private shared histograms. The generic fallback path remains unchanged, and the old low-byte kernel stays available for equivalence testing and rollback.

**Tech Stack:** CUDA C++, CMake, CTest, NVCC, Nsight Systems

---

## File Responsibilities

- `src/radix_histogram.cuh`: Keep the existing histogram kernels, add `histogram_low_byte_topk50_v2_kernel`, and local helper functions for the register-cache accumulation path.
- `src/batch_topk.cu`: Only the optimized `seg_len == 10000 && k == 50` branch changes, switching from the old low-byte kernel to the new `v2` kernel.
- `test/test_batch_topk.cu`: Add a kernel-level equivalence helper comparing old vs new low-byte histograms and a tiny benchmark contract note for this experiment.

### Task 1: Add Failing Equivalence Tests For The New Low-Byte Kernel

**Files:**
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Write the failing low-byte histogram equivalence helper**

Add a helper that launches either the old or new low-byte kernel against the same `SegmentSelectState`:

```c++
bool run_low_byte_histogram_variant(const std::vector<float>& values,
                                    const radix_topk::SegmentSelectState& state,
                                    bool use_v2,
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

  if (use_v2) {
    radix_topk::histogram_low_byte_topk50_v2_kernel<<<1, 256>>>(
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

- [ ] **Step 2: Write the failing kernel-level equivalence test**

Add a deterministic equivalence test:

```c++
bool check_low_byte_topk50_v2_equivalence() {
  const std::vector<float> values = {9.0f, 8.0f, 7.0f, 6.0f,
                                     5.0f, 4.0f, 3.0f, 2.0f};
  const radix_topk::SegmentSelectState state =
      radix_topk::simulate_radix_boundary(values, 8, 3);

  std::vector<unsigned int> old_hist(256, 0);
  std::vector<unsigned int> new_hist(256, 0);
  if (!run_low_byte_histogram_variant(values, state, false, &old_hist) ||
      !run_low_byte_histogram_variant(values, state, true, &new_hist)) {
    return false;
  }

  return old_hist == new_hist;
}
```

Wire it into `main()` before the optimized-path regressions:

```c++
  if (!check_low_byte_topk50_v2_equivalence()) {
    std::fprintf(stderr, "low byte topk50 v2 equivalence check failed\n");
    return 1;
  }
```

- [ ] **Step 3: Add the tiny baseline contract note**

Add:

```c++
bool check_low_byte_histogram_baseline_contract() {
  return radix_topk::kOptimizedSegLen == 10000 &&
         radix_topk::kOptimizedK == 50 &&
         radix_topk::kOptimizedCandidateCap == 64;
}
```

Wire it into `main()`:

```c++
  if (!check_low_byte_histogram_baseline_contract()) {
    std::fprintf(stderr, "low byte histogram baseline contract is incorrect\n");
    return 1;
  }
```

- [ ] **Step 4: Run the build to verify red**

Run: `cmake --build build -j`

Expected: compile fails because `histogram_low_byte_topk50_v2_kernel` does not exist.

- [ ] **Step 5: Commit the red tests**

```bash
git add test/test_batch_topk.cu
git commit -m "test: add low byte histogram v2 regressions"
```

### Task 2: Implement The Lower-Contention Low-Byte Kernel

**Files:**
- Modify: `src/radix_histogram.cuh`
- Modify: `src/batch_topk.cu`

- [ ] **Step 1: Add a small register-cache helper to `src/radix_histogram.cuh`**

Insert a tiny helper ahead of the specialized kernels:

```c++
struct HistogramCacheEntry {
  unsigned short bin = 0xffffu;
  unsigned short count = 0u;
};

__device__ inline void flush_histogram_cache(HistogramCacheEntry* cache,
                                             int cache_size,
                                             unsigned int* warp_histogram) {
  for (int i = 0; i < cache_size; ++i) {
    if (cache[i].count != 0u) {
      atomicAdd(&warp_histogram[cache[i].bin], static_cast<unsigned int>(cache[i].count));
      cache[i].bin = 0xffffu;
      cache[i].count = 0u;
    }
  }
}

__device__ inline void accumulate_histogram_cache(HistogramCacheEntry* cache,
                                                  int cache_size,
                                                  unsigned short bin,
                                                  unsigned int* warp_histogram) {
  for (int i = 0; i < cache_size; ++i) {
    if (cache[i].count != 0u && cache[i].bin == bin) {
      ++cache[i].count;
      return;
    }
  }
  for (int i = 0; i < cache_size; ++i) {
    if (cache[i].count == 0u) {
      cache[i].bin = bin;
      cache[i].count = 1u;
      return;
    }
  }

  atomicAdd(&warp_histogram[cache[0].bin], static_cast<unsigned int>(cache[0].count));
  cache[0].bin = bin;
  cache[0].count = 1u;
}
```

- [ ] **Step 2: Add the new `histogram_low_byte_topk50_v2_kernel`**

Append the new kernel next to the existing specialized low-byte kernel:

```c++
__global__ inline void histogram_low_byte_topk50_v2_kernel(
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

  HistogramCacheEntry cache[4];
  for (int i = 0; i < 4; ++i) {
    cache[i].bin = 0xffffu;
    cache[i].count = 0u;
  }

  unsigned int* warp_histogram = warp_histograms[warp];
  for (int i = tid; i < seg_len; i += blockDim.x) {
    const uint16_t encoded = encode_half_desc(segment_input[i]);
    if ((encoded >> 8) == state.boundary_digit) {
      accumulate_histogram_cache(
          cache, 4, static_cast<unsigned short>(encoded & 0xffu), warp_histogram);
    }
  }
  flush_histogram_cache(cache, 4, warp_histogram);
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

- [ ] **Step 3: Switch only the optimized path to the new kernel**

In `src/batch_topk.cu`, change only the specialized branch:

```c++
    if (ctas_per_segment == 1) {
      histogram_low_byte_topk50_v2_kernel<<<seg_num, 256, 0, stream>>>(
          d_input, seg_len, workspace.states, workspace.histograms_lo);
    } else {
      histogram_low_byte_splitk_kernel<<<seg_num * ctas_per_segment, 256, 0, stream>>>(
          d_input,
          seg_num,
          seg_len,
          ctas_per_segment,
          workspace.states,
          workspace.partial_histograms);
      status = cudaGetLastError();
      if (status != cudaSuccess) {
        return status;
      }
      reduce_partial_histograms_kernel<<<seg_num, 256, 0, stream>>>(
          workspace.partial_histograms, seg_num, ctas_per_segment, workspace.histograms_lo);
    }
```

Do not touch the generic fallback path.

- [ ] **Step 4: Run tests to verify green**

Run: `cmake --build build -j && ./build/test_batch_topk`

Expected: the equivalence test passes, all existing regressions stay green, and the fallback path still works.

- [ ] **Step 5: Commit**

```bash
git add src/radix_histogram.cuh src/batch_topk.cu
git commit -m "cuda: add low byte histogram v2 for topk50"
```

### Task 3: Benchmark Gate And Keep-Or-Revert Decision

**Files:**
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Rebuild, rerun tests, and benchmark**

Run: `cmake --build build -j && ./build/test_batch_topk && ./build/bench_batch_topk`

Expected:

- `./build/test_batch_topk` exits `0`
- `./build/bench_batch_topk` prints exactly five lines
- the specialized low-byte change beats the current baseline on this machine:
  - `(128,10000,50)` better than `49.10 us`
  - `(6000,10000,50)` better than `910.80 us`

- [ ] **Step 2: Keep-or-revert decision**

If the benchmark beats the baseline, keep the implementation and commit the baseline-note change:

```bash
git add test/test_batch_topk.cu
git commit -m "test: document low byte histogram baseline contract"
```

If the benchmark does not beat the baseline, revert the implementation commit from Task 2 and do not keep the new kernel:

```bash
git restore test/test_batch_topk.cu
git revert --no-edit HEAD
```

Expected after revert: `cmake --build build -j && ./build/test_batch_topk` still passes and the branch returns to the previous verified baseline.

## Self-Review

- Spec coverage:
  - The plan covers only the specialized low-byte histogram kernel, its equivalence test, and the keep-or-revert benchmark gate.
  - It does not commit to broader selection-structure changes.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to Task N” placeholders remain.
  - Every code-changing step includes concrete code blocks and commands.
- Type consistency:
  - `histogram_low_byte_topk50_v2_kernel`, `HistogramCacheEntry`, `flush_histogram_cache`, `accumulate_histogram_cache`, `kOptimizedSegLen`, `kOptimizedK`, and `kOptimizedCandidateCap` are named consistently across tasks.
