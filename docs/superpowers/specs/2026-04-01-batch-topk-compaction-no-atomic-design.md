# Batch TopK Compaction No-Atomic Design

## Goal

Optimize only the specialized `compact_candidate_indices_topk50_warp_kernel` in the existing optimized `(batch, 10000) -> top50` path by removing the fine-grained `atomicAdd(..., 1)` in the equal-value path.

This design is intentionally narrow:

- only the optimized `k=50` compaction kernel changes
- the generic fallback path remains unchanged
- histogram stages remain unchanged
- final top-k stages remain unchanged

The objective is to reduce compaction overhead without destabilizing the already-validated optimized pipeline.

## Current Baseline

At commit `02056a2` on `feature/batch-topk-deep-opt`, the verified benchmark on this machine is:

- `(128, 10000, 50)`: `45.52 us`
- `(1500, 10000, 50)`: `165.31 us`
- `(3000, 10000, 50)`: `313.24 us`
- `(4500, 10000, 50)`: `684.99 us`
- `(6000, 10000, 50)`: `909.67 us`

Recent `nsys` profiling on the same optimized path showed the main GPU-kernel time split is approximately:

- `compact_candidate_indices_topk50_warp_kernel`: `~45.9%`
- `histogram_low_byte_topk50_kernel`: `~24.0%`
- `histogram_high_byte_topk50_kernel`: `~22.7%`
- `final_topk50_kernel`: `~5.7%`

Compaction remains the largest single hotspot and is the only target of this design.

## Non-Goals

- Changing the selection structure
- Modifying histogram kernels
- Modifying the public API
- Changing the generic fallback path
- Rewriting `final_topk50_kernel`
- Changing benchmark methodology

## Recommended Scope

Keep the current optimized path structure unchanged except for the specialized compaction kernel:

1. high-byte histogram
2. high-byte boundary select
3. low-byte histogram
4. low-byte cutoff finalize
5. top-k50 compaction
6. final top-k50 sort

Only replace step 5 with a less atomic-heavy implementation.

## Proposed Kernel Direction

The current `compact_candidate_indices_topk50_warp_kernel` behaves like this:

- better items:
  - warp ballot
  - lane 0 reserves a contiguous output block with one `atomicAdd`
  - warp lanes write into that reserved block
- equal items:
  - each selected equal lane performs its own `atomicAdd(&selected_written, 1)`
  - output slot is assigned by that per-lane atomic

The replacement keeps the better path intact and changes only the equal path:

- better items stay exactly as they are now
- equal items no longer do per-lane output-slot atomics
- each equal item writes directly to:
  - `state.strictly_better_count + equal_rank`

where `equal_rank` is the deterministic rank of that equal item among all equal items seen so far.

## Counting Model

The kernel should maintain separate accounting:

- `better_written`
  - actual number of better items written into the prefix
- `equal_seen`
  - actual number of equal items encountered so far

Then:

- `equal_slots = min(equal_seen, state.remaining_slots)`
- `candidate_counts = better_written + equal_slots`

This replaces the current model where one shared counter mixes both “better block reservation” and “equal per-lane slot assignment”.

## Correctness Requirements

The new kernel must preserve:

1. all better items are written in ascending local-index order within the better prefix
2. equal items fill the next `remaining_slots` positions in ascending local-index order
3. `candidate_counts == k` for the optimized `k=50` path
4. the final `final_topk50_kernel` reads a contiguous initialized prefix

No change is allowed to:

- tie-break semantics
- `candidate_counts` expectations in existing optimized-path regressions

## Interface

The existing kernel name may be kept or replaced. The simplest option is to keep the same name:

```c++
__global__ inline void compact_candidate_indices_topk50_warp_kernel(
    const half* input,
    int seg_len,
    const SegmentSelectState* states,
    int* candidate_indices,
    int* candidate_counts);
```

Only the internal counting and equal-item write strategy changes.

## Integration Strategy

`src/batch_topk.cu` should not need any host-side orchestration changes if the kernel name and signature stay the same.

That is preferred for this design because:

- the change stays completely inside the kernel
- rollback is trivial
- benchmark attribution stays clean

## Testing Strategy

The implementation must preserve and pass the existing optimized-path regressions:

- duplicate-heavy `k=50`
- large-random `k=50`
- NaN/Inf optimized-path regression
- small-batch optimized-path regression
- `k=128` correctness regression for the generic path
- direct `compact_topk50` kernel selection regression

No new broad benchmark tests are required before implementation. The current regressions already lock:

- `candidate_counts == 50`
- end-to-end correctness
- deterministic tie-break behavior

## Acceptance Criteria

This design is considered successful only if all of the following hold:

- `./build/test_batch_topk` passes
- the optimized path remains gated to `seg_len == 10000 && k == 50`
- the generic fallback path remains unchanged in behavior
- benchmark improves beyond the current `02056a2` baseline on this machine, with special attention to:
  - `(128, 10000, 50)` better than `45.52 us`
  - `(6000, 10000, 50)` better than `909.67 us`

If the new compaction variant does not beat that baseline, the implementation should be discarded.

## Risks

The main risks are:

- `candidate_counts` no longer matches the contiguous initialized prefix
- equal-item direct indexing leaves holes when `better_written != state.strictly_better_count`
- the removed per-lane atomic does not pay for itself because the kernel remains dominated by ballots and synchronization

The design mitigates these risks by:

- preserving the better path
- changing only the equal path and final count calculation
- reusing the existing optimized-path regression suite

## Expected Outcome

If the design works, it should:

- reduce compaction overhead
- improve both small and large target-shape latency
- preserve the current stable path structure

If it does not work, it should be reverted cleanly without affecting the rest of the optimized path.
