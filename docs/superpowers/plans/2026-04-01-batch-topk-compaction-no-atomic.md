# Batch TopK Compaction No-Atomic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the specialized `k=50` compaction kernel’s per-equal-item slot atomic with deterministic equal-slot writes while preserving the current stable optimized path structure.

**Architecture:** The current optimized path on this branch is already stable and benchmarked: histogram, cutoff selection, specialized `k=50` compaction, and `final_topk50`. This plan keeps the better-item path intact and changes only the equal-item path inside a new specialized compaction kernel variant. The generic fallback path remains unchanged, and the benchmark gate at the end decides whether the new kernel stays or is reverted.

**Tech Stack:** CUDA C++, CMake, CTest, NVCC, Nsight Systems

---

## File Responsibilities

- `src/candidate_compact.cuh`: Keep the current generic compaction kernels and add one new specialized `k=50` compaction kernel variant with separate better/equal accounting.
- `src/batch_topk.cu`: Only the optimized `seg_len == 10000 && k == 50` branch changes, switching from the current specialized compaction kernel to the new no-atomic-equal variant.
- `test/test_batch_topk.cu`: Add one direct old/new kernel equivalence test and a small baseline contract note for this experiment.

### Task 1: Implement The No-Atomic Equal-Path Kernel With TDD

**Files:**
- Modify: `test/test_batch_topk.cu`
- Modify: `src/candidate_compact.cuh`
- Modify: `src/batch_topk.cu`

- [ ] **Step 1: Write the failing direct old/new compaction helper in `test/test_batch_topk.cu`**

Add a helper that launches either the current specialized kernel or the new no-atomic variant against the same `SegmentSelectState`:

```c++
bool run_compact_topk50_variant(const std::vector<float>& values,
                                const radix_topk::SegmentSelectState& state,
                                bool use_no_atomic,
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
  if (cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * host_input.size()) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_state), sizeof(radix_topk::SegmentSelectState)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_indices),
                 sizeof(int) * static_cast<size_t>(radix_topk::kCompactedCandidateCap)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&d_candidate_counts), sizeof(int)) != cudaSuccess ||
      cudaMemcpy(d_input,
                 host_input.data(),
                 sizeof(half) * host_input.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_state,
                 &state,
                 sizeof(state),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_candidate_counts);
    cudaFree(d_candidate_indices);
    cudaFree(d_state);
    cudaFree(d_input);
    return false;
  }

  if (use_no_atomic) {
    radix_topk::compact_candidate_indices_topk50_no_atomic_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_state, d_candidate_indices, d_candidate_counts);
  } else {
    radix_topk::compact_candidate_indices_topk50_warp_kernel<<<1, 256>>>(
        d_input, static_cast<int>(values.size()), d_state, d_candidate_indices, d_candidate_counts);
  }

  const bool ok = cudaGetLastError() == cudaSuccess &&
                  cudaDeviceSynchronize() == cudaSuccess &&
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
```

- [ ] **Step 2: Write the failing old/new equivalence test and baseline note**

Add a deterministic equivalence test on the existing duplicate-heavy optimized-shape case and a small baseline contract note:

```c++
bool check_compact_topk50_no_atomic_equivalence() {
  const int seg_len = 10000;
  const int k = 50;
  const std::vector<float> values = make_duplicate_heavy_input(1, seg_len);
  radix_topk::SegmentSelectState state{};
  if (!run_gpu_cutoff_selection(values, k, &state)) {
    return false;
  }

  std::vector<int> old_indices(radix_topk::kCompactedCandidateCap, -1);
  std::vector<int> new_indices(radix_topk::kCompactedCandidateCap, -1);
  int old_count = 0;
  int new_count = 0;
  if (!run_compact_topk50_variant(values, state, false, &old_indices, &old_count) ||
      !run_compact_topk50_variant(values, state, true, &new_indices, &new_count)) {
    return false;
  }

  if (old_count != new_count) {
    return false;
  }
  old_indices.resize(static_cast<size_t>(old_count));
  new_indices.resize(static_cast<size_t>(new_count));
  return old_indices == new_indices;
}

bool check_compaction_no_atomic_baseline_contract() {
  return radix_topk::kOptimizedSegLen == 10000 &&
         radix_topk::kOptimizedK == 50 &&
         radix_topk::kOptimizedCandidateCap == 64;
}
```

Wire both into `main()` before the existing optimized-path regressions:

```c++
  if (!check_compaction_no_atomic_baseline_contract()) {
    std::fprintf(stderr, "compaction no-atomic baseline contract is incorrect\n");
    return 1;
  }
  if (!check_compact_topk50_no_atomic_equivalence()) {
    std::fprintf(stderr, "compact topk50 no-atomic equivalence check failed\n");
    return 1;
  }
```

- [ ] **Step 3: Run the build to verify red**

Run: `cmake --build build -j`

Expected: compile fails because `radix_topk::compact_candidate_indices_topk50_no_atomic_kernel` does not exist.

- [ ] **Step 4: Add the new no-atomic specialized kernel in `src/candidate_compact.cuh`**

Append the new kernel alongside the existing specialized one:

```c++
__global__ inline void compact_candidate_indices_topk50_no_atomic_kernel(
    const half* input,
    int seg_len,
    const SegmentSelectState* states,
    int* candidate_indices,
    int* candidate_counts) {
  const int seg = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kCompactionWarpSize - 1);
  const int warp = tid / kCompactionWarpSize;
  const unsigned int full_mask = 0xffffffffu;

  const SegmentSelectState state = states[seg];
  const half* segment_input = input + static_cast<size_t>(seg) * seg_len;
  int* segment_candidates =
      candidate_indices + static_cast<size_t>(seg) * kCompactedCandidateCap;

  __shared__ int better_written;
  __shared__ int equal_seen;
  __shared__ int warp_equal_counts[kCompactionWarpsPerBlock];
  __shared__ int warp_equal_bases[kCompactionWarpsPerBlock];
  __shared__ int tile_equal_base;

  if (tid == 0) {
    better_written = 0;
    equal_seen = 0;
  }
  __syncthreads();

  for (int base = 0; base < seg_len; base += blockDim.x) {
    const int idx = base + tid;
    const bool in_range = idx < seg_len;
    const uint16_t key = in_range ? encode_half_desc(segment_input[idx]) : 0xffffu;

    const bool better_flag = in_range && key < state.cutoff_key;
    const bool equal_flag = in_range && key == state.cutoff_key;
    const unsigned int better_mask = __ballot_sync(full_mask, better_flag);
    const unsigned int equal_mask = __ballot_sync(full_mask, equal_flag);
    const unsigned int lane_mask = (1u << lane) - 1u;
    const int better_count = __popc(better_mask);
    const int equal_count = __popc(equal_mask);

    int warp_better_base = 0;
    if (lane == 0) {
      if (better_count > 0) {
        warp_better_base = atomicAdd(&better_written, better_count);
      }
      warp_equal_counts[warp] = equal_count;
    }
    warp_better_base = __shfl_sync(full_mask, warp_better_base, 0);
    __syncthreads();

    if (warp == 0 && lane < kCompactionWarpsPerBlock) {
      int equal_prefix = 0;
      for (int i = 0; i < lane; ++i) {
        equal_prefix += warp_equal_counts[i];
      }
      warp_equal_bases[lane] = equal_prefix;
      if (lane == 0) {
        int tile_equal_total = 0;
        for (int i = 0; i < kCompactionWarpsPerBlock; ++i) {
          tile_equal_total += warp_equal_counts[i];
        }
        tile_equal_base = equal_seen;
        equal_seen += tile_equal_total;
      }
    }
    __syncthreads();

    if (better_flag) {
      const int better_rank = __popc(better_mask & lane_mask);
      const int output_slot = warp_better_base + better_rank;
      if (output_slot < kCompactedCandidateCap) {
        segment_candidates[output_slot] = idx;
      }
    }

    const int equal_rank =
        tile_equal_base + warp_equal_bases[warp] + __popc(equal_mask & lane_mask);
    if (equal_flag && equal_rank < state.remaining_slots) {
      const int output_slot = state.strictly_better_count + equal_rank;
      if (output_slot < kCompactedCandidateCap) {
        segment_candidates[output_slot] = idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int better_slots = better_written;
    if (better_slots < 0) {
      better_slots = 0;
    }
    if (better_slots > kCompactedCandidateCap) {
      better_slots = kCompactedCandidateCap;
    }

    int equal_slots = equal_seen;
    if (equal_slots < 0) {
      equal_slots = 0;
    }
    if (equal_slots > state.remaining_slots) {
      equal_slots = state.remaining_slots;
    }
    if (equal_slots > kCompactedCandidateCap - state.strictly_better_count) {
      equal_slots = kCompactedCandidateCap - state.strictly_better_count;
    }

    int total = better_slots;
    if (state.strictly_better_count <= total) {
      const int equal_end = state.strictly_better_count + equal_slots;
      if (equal_end > total) {
        total = equal_end;
      }
    }
    candidate_counts[seg] = total;
  }
}
```

- [ ] **Step 5: Switch only the optimized path in `src/batch_topk.cu`**

Change only the optimized branch:

```c++
    compact_candidate_indices_topk50_no_atomic_kernel<<<seg_num, 256, 0, stream>>>(
        d_input, seg_len, workspace.states, workspace.candidate_indices,
        workspace.candidate_counts);
```

Do not touch the generic fallback path.

- [ ] **Step 6: Run tests to verify green**

Run: `cmake --build build -j && ./build/test_batch_topk`

Expected: the new direct old/new equivalence test passes, all existing optimized-path regressions stay green, and the generic fallback path still works.

- [ ] **Step 7: Commit**

```bash
git add test/test_batch_topk.cu src/candidate_compact.cuh src/batch_topk.cu
git commit -m "cuda: add no-atomic k50 compaction"
```

### Task 2: Benchmark Gate And Keep-Or-Revert Decision

**Files:**
- Modify: none

- [ ] **Step 1: Rebuild, rerun tests, and benchmark**

Run: `cmake --build build -j && ./build/test_batch_topk && ./build/bench_batch_topk`

Expected:

- `./build/test_batch_topk` exits `0`
- `./build/bench_batch_topk` prints exactly five lines
- the specialized compaction change beats the current baseline on this machine:
  - `(128,10000,50)` better than `45.52 us`
  - `(6000,10000,50)` better than `909.67 us`

- [ ] **Step 2: Keep-or-revert decision**

If the benchmark beats the baseline, keep the implementation and stop here.

If the benchmark does not beat the baseline, revert the implementation commit from Task 1 and do not keep the new kernel:

```bash
git revert --no-edit HEAD
```

Expected after revert: `cmake --build build -j && ./build/test_batch_topk` still passes and the branch returns to the previous verified baseline.

## Self-Review

- Spec coverage:
  - The plan covers only the specialized compaction kernel’s equal-path change and the benchmark keep-or-revert gate.
  - It does not commit to changes in histogram or final-sort behavior.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to Task N” placeholders remain.
  - Every code-changing step includes concrete code blocks and commands.
- Type consistency:
  - `compact_candidate_indices_topk50_no_atomic_kernel`, `better_written`, `equal_seen`, `candidate_counts`, `kOptimizedSegLen`, `kOptimizedK`, and `kOptimizedCandidateCap` are named consistently across tasks.
