# Batch TopK Compaction Warp-Reserved Design

## Goal

Optimize only the specialized `compact_candidate_indices_topk50_warp_kernel` in the existing optimized `(batch, 10000) -> top50` path by reducing equal-path atomics from per-element to per-warp.

This design is intentionally narrow:

- only the optimized `k=50` compaction kernel changes
- histogram stages remain unchanged
- final top-k stages remain unchanged
- generic fallback path remains unchanged

The objective is to reduce compaction overhead without destabilizing the already-validated optimized pipeline.

## Current Baseline

At commit `91c9f6e` on `feature/batch-topk-deep-opt`, the verified benchmark on this machine is approximately:

- `(128, 10000, 50)`: `46.28 us`
- `(1500, 10000, 50)`: `175.46 us`
- `(3000, 10000, 50)`: `334.13 us`
- `(4500, 10000, 50)`: `761.10 us`
- `(6000, 10000, 50)`: `958.36 us`

Recent stage timing shows that compaction remains the largest single optimized-path hotspot:

- `seg_num=128`: `compact ≈ 22.53 us`
- `seg_num=6000`: `compact ≈ 381.95 us`

The current specialized compaction kernel still performs one atomic operation per accepted equal-value element.

## Non-Goals

- Changing histogram kernels
- Changing selection structure or cutoff semantics
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

Only replace step 5 with a lower-atomic equal-path strategy.

## Proposed Kernel Direction

The current `compact_candidate_indices_topk50_warp_kernel` behaves like this:

- better items:
  - warp ballot
  - lane 0 reserves a contiguous block with one `atomicAdd`
  - warp lanes write into that block
- equal items:
  - each accepted equal lane performs its own `atomicAdd(&selected_written, 1)`
  - output slot is assigned one item at a time

The new design keeps the better path intact and changes only the equal path:

1. compute a ballot mask for accepted equal lanes:
   - `equal_flag && equal_rank < state.remaining_slots`
2. lane 0 performs one `atomicAdd` for the whole warp’s accepted equal count
3. all accepted equal lanes compute their local rank in that warp mask
4. each accepted equal lane writes into:
   - `warp_equal_base + local_equal_rank`

This reduces equal-path atomics from “one per item” to “one per participating warp”.

## Counting Model

The kernel should maintain separate accounting:

- `better_written`
  - actual number of better items written
- `equal_written`
  - actual number of equal items written, counted per warp reservation

Then:

- `candidate_counts = better_written + equal_written`

This is simpler than the previous no-atomic design because the output remains a truly contiguous append-only prefix.

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

The existing kernel name may be kept:

```c++
__global__ inline void compact_candidate_indices_topk50_warp_kernel(
    const half* input,
    int seg_len,
    const SegmentSelectState* states,
    int* candidate_indices,
    int* candidate_counts);
```

Only the internal equal-path reservation strategy changes.

## Integration Strategy

`src/batch_topk.cu` should not need any host-side orchestration changes if the kernel name and signature stay the same.

That is preferred because:

- the change stays completely inside one kernel
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
- benchmark improves beyond the current `91c9f6e` baseline on this machine, with special attention to:
  - `(128, 10000, 50)` better than `46.28 us`
  - `(6000, 10000, 50)` better than `958.36 us`

If the new compaction variant does not beat that baseline, the implementation should be discarded.

## Risks

The main risks are:

- equal-path warp reservations still do not pay for their added ballot/scan overhead
- the new warp-level append path accidentally breaks equal-item ordering
- the kernel becomes harder to reason about without enough comments

The design mitigates these risks by:

- preserving the better path exactly
- keeping the output model append-only and contiguous
- reusing the existing optimized-path regression suite

## Expected Outcome

If the design works, it should:

- reduce compaction overhead
- improve both small and large target-shape latency
- preserve the current stable optimized path structure

If it does not work, it should be reverted cleanly without affecting the rest of the optimized path.
