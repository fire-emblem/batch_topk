# Batch TopK Large-Batch Histogram Design

## Goal

Optimize only the large-batch histogram stages in the specialized `(batch, 10000) -> top50` path.

This design is intentionally narrow:

- only the optimized `seg_len == 10000 && k == 50` path changes
- only large batch counts are targeted first
- compaction and final top-k stages remain unchanged
- generic fallback path remains unchanged

The objective is to reduce histogram time for the large target shapes without risking the already-stable small-batch path.

## Current Baseline

At commit `03ecdfb` on `feature/batch-topk-deep-opt`, the verified benchmark on this machine is approximately:

- `(128, 10000, 50)`: `46.03 us`
- `(1500, 10000, 50)`: `205.52 us`
- `(3000, 10000, 50)`: `398.51 us`
- `(4500, 10000, 50)`: `812.61 us`
- `(6000, 10000, 50)`: `1079.45 us`

Stage timing gathered through the new internal timing hook shows:

- `seg_num=128`
  - `high_hist`: `8.19 us`
  - `high_select`: `7.17 us`
  - `low_hist`: `16.38 us`
  - `finalize`: `5.12 us`
  - `compact`: `28.67 us`
  - `final`: `7.17 us`

- `seg_num=6000`
  - `high_hist`: `267.26 us`
  - `high_select`: `6.14 us`
  - `low_hist`: `272.38 us`
  - `finalize`: `5.12 us`
  - `compact`: `402.50 us`
  - `final`: `46.08 us`

This shows the large-batch path is no longer dominated by a single kernel. Histogram cost is large enough to justify a batch-size-specific optimization.

## Non-Goals

- Changing selection structure or cutoff semantics
- Modifying the public API
- Modifying the generic fallback path
- Rewriting compaction or final top-k
- Optimizing the `seg_num < 1500` path in this iteration

## Recommended Scope

Add one more dispatch split inside the optimized path:

- `seg_num < 1500`: keep the current stable specialized histogram kernels
- `seg_num >= 1500 && seg_len == 10000 && k == 50`: use new large-batch histogram kernels

Only histogram stages change under that dispatch:

1. high-byte histogram
2. high-byte boundary select
3. low-byte histogram
4. low-byte cutoff finalize
5. top-k50 compaction
6. final top-k50 sort

This design replaces only stages 1 and 3 for large batches.

## Proposed Kernel Direction

The new large-batch histogram kernels should keep the same output contract:

- `256` bins
- `1 CTA / segment`
- same encoded key mapping
- same filtering rule for low byte:
  - only count elements where `(encoded >> 8) == state.boundary_digit`

The implementation change is thread-local aggregation:

- each thread loads `4` elements per loop step
- within those `4` elements, identical bins are merged locally before writing to shared memory
- shared memory still holds per-warp private `256`-bin histograms
- end of kernel still merges per-warp histograms into the final `256`-bin output

This is not a new histogram format. It is a lower-contention producer feeding the same final per-segment `256`-bin result.

## Interface

The design adds two new kernels next to the current specialized ones:

```c++
__global__ inline void histogram_high_byte_topk50_large_kernel(
    const half* input,
    int seg_len,
    unsigned int* histograms_hi);

__global__ inline void histogram_low_byte_topk50_large_kernel(
    const half* input,
    int seg_len,
    const SegmentSelectState* states,
    unsigned int* histograms_lo);
```

The current `histogram_high_byte_topk50_kernel` and `histogram_low_byte_topk50_kernel` remain intact for:

- small-batch optimized path
- kernel-level equivalence tests
- easy rollback

## Dispatch Strategy

Inside the optimized branch in `src/batch_topk.cu`:

- if `seg_num < 1500`, keep current kernels
- if `seg_num >= 1500`, switch to the new large-batch kernels

No change to:

- split-k path
- generic fallback path
- compaction or final kernel choice

## Testing Strategy

The implementation must add:

1. Kernel-level equivalence tests for the new large-batch kernels:
   - old vs new high-byte histogram
   - old vs new low-byte histogram
   - full `256`-bin equality for both

2. Existing end-to-end optimized-path regressions must remain green:
   - duplicate-heavy `k=50`
   - large-random `k=50`
   - NaN/Inf optimized-path regression
   - small-batch optimized-path regression
   - `k=128` correctness regression for the generic path

The kernel-level equivalence tests should be deterministic and should include at least one realistic optimized-shape case.

## Acceptance Criteria

This design is considered successful only if all of the following hold:

- `./build/test_batch_topk` passes
- the optimized path remains gated to `seg_len == 10000 && k == 50`
- the generic fallback path remains unchanged in behavior
- the new large-batch histogram kernels are bin-for-bin equivalent to the old kernels
- benchmark improves for the large-batch target set, with special attention to:
  - `(1500, 10000, 50)` no worse than `205.52 us`
  - `(6000, 10000, 50)` better than `1079.45 us`

The small-batch point `(128, 10000, 50)` should not materially regress, but it is not the primary gate for this design because the new dispatch explicitly avoids that shape.

If the new large-batch branch does not beat the large-batch baseline, the implementation should be discarded.

## Risks

The main risks are:

- per-thread micro-aggregation adds more instructions than it saves in shared atomic traffic
- the `4`-element local merge works poorly on real distributions and creates no net gain
- dispatch threshold `1500` is too low or too high for the actual machine

The design mitigates these risks by:

- restricting the branch to large batches only
- preserving the current small-batch path untouched
- requiring a keep-or-revert benchmark gate

## Expected Outcome

If the design works, it should:

- reduce histogram cost for large batches
- improve the high-batch benchmark points without destabilizing the small-batch path
- preserve the rest of the current optimized pipeline unchanged

If it does not work, it should be reverted cleanly without affecting the current stable path.
