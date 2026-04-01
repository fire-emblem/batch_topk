# Batch TopK Low-Byte Histogram Optimization Design

## Goal

Optimize only the specialized `seg_len == 10000 && k == 50` low-byte histogram stage in the existing batch top-k fast path.

This design is intentionally narrow:

- only the optimized `k=50` path changes
- the generic fallback path remains unchanged
- compaction and final top-k stages remain unchanged
- benchmark/reporting logic remains unchanged

The objective is to reduce the cost of the second histogram round without destabilizing the already-validated optimized pipeline.

## Current Baseline

At commit `eaa713d` on `feature/batch-topk-deep-opt`, the verified benchmark on this machine is:

- `(128, 10000, 50)`: `49.10 us`
- `(1500, 10000, 50)`: `177.97 us`
- `(3000, 10000, 50)`: `336.18 us`
- `(4500, 10000, 50)`: `686.69 us`
- `(6000, 10000, 50)`: `910.80 us`

`nsys` profiling on the same path shows the main GPU-kernel time split is approximately:

- `compact_candidate_indices_topk50_warp_kernel`: `~45.9%`
- `histogram_low_byte_topk50_kernel`: `~24.0%`
- `histogram_high_byte_topk50_kernel`: `~22.7%`
- `final_topk50_kernel`: `~5.7%`

The low-byte histogram is the largest non-compaction hotspot and is the only target of this design.

## Non-Goals

- Replacing the selection algorithm structure
- Modifying the public API
- Changing the generic fallback path
- Rewriting compaction or final top-k
- Changing benchmark methodology

## Recommended Scope

Keep the current optimized path structure:

1. high-byte histogram
2. high-byte boundary select
3. low-byte histogram
4. low-byte cutoff finalize
5. top-k50 compaction
6. final top-k50 sort

Only replace step 3 with a lower-contention specialized kernel.

## Proposed Kernel Direction

The current `histogram_low_byte_topk50_kernel` uses:

- `1 CTA / segment`
- `256` bins
- shared-memory atomics
- one shared histogram array for the whole block

The replacement should remain semantically identical but reduce contention:

- keep `1 CTA / segment`
- keep `256` bins
- keep the same filtered condition:
  - only count values where `(encoded >> 8) == state.boundary_digit`
- accumulate counts per warp into warp-private shared histograms
- merge the warp-private histograms into the final `256`-bin output at the end

The expected win comes from avoiding block-wide contention on a single shared `256`-bin histogram.

## Interface

The new kernel should be added alongside the existing one, for example:

```c++
__global__ inline void histogram_low_byte_topk50_v2_kernel(
    const half* input,
    int seg_len,
    const SegmentSelectState* states,
    unsigned int* histograms_lo);
```

The old `histogram_low_byte_topk50_kernel` should remain available for equivalence testing and easy rollback.

## Integration Strategy

In `src/batch_topk.cu`, only the optimized branch should be changed:

- replace the current call to `histogram_low_byte_topk50_kernel(...)`
- with the new `histogram_low_byte_topk50_v2_kernel(...)`

All other stages remain untouched.

## Testing Strategy

The implementation must add:

1. A kernel-level equivalence test:
   - run the old low-byte kernel and the new low-byte kernel on the same input/state
   - compare all `256` bins for equality

2. Existing end-to-end optimized-path regressions must remain green:
   - duplicate-heavy `k=50`
   - large-random `k=50`
   - NaN/Inf optimized-path regression
   - small-batch optimized-path regression

The kernel-level test should be small and deterministic. It should not depend on benchmark timing.

## Acceptance Criteria

This design is considered successful only if all of the following hold:

- `./build/test_batch_topk` passes
- the optimized path remains gated to `seg_len == 10000 && k == 50`
- the generic fallback path remains unchanged in behavior
- the new low-byte histogram kernel is bin-for-bin equivalent to the old one
- benchmark improves beyond the current `eaa713d` baseline on this machine, with special attention to:
  - `(128, 10000, 50)` better than `49.10 us`
  - `(6000, 10000, 50)` better than `910.80 us`

If the new kernel does not beat that baseline, the implementation should be discarded.

## Risks

The main risks are:

- warp-private histograms use more shared memory and reduce occupancy
- final histogram merge cost cancels out the reduced contention
- filtered low-byte input distribution is already sparse enough that the current kernel is hard to beat

The design mitigates these risks by keeping the scope to one kernel only and requiring a keep-or-revert benchmark gate.

## Expected Outcome

If the design works, it should:

- reduce low-byte histogram time
- lower end-to-end latency on both small and large target shapes
- preserve the current stable optimized path structure

If it does not work, it should be reverted cleanly without affecting the rest of the optimized path.
