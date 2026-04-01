# Batch TopK First-Nibble Select Design

## Goal

Add one more optimization stage to the existing specialized `(batch, 10000) -> top50` path by replacing only the first full-segment selection round with a warp-centric highest-4-bit selector.

This design is intentionally narrow:

- optimized path only: `seg_len == 10000 && k == 50`
- fallback path remains unchanged
- later selection, compaction, and final top-k stages remain in place unless explicitly described here

The immediate objective is to reduce the cost of the first full scan over all `10000` elements without destabilizing the rest of the pipeline.

## Current Baseline

At commit `825ed65` on `feature/batch-topk-deep-opt`, the verified benchmark on this machine is:

- `(128, 10000, 50)`: `50.07 us`
- `(1500, 10000, 50)`: `191.39 us`
- `(3000, 10000, 50)`: `338.07 us`
- `(4500, 10000, 50)`: `689.61 us`
- `(6000, 10000, 50)`: `917.04 us`

`nsys` profiling on the same path shows the main GPU-kernel time split is approximately:

- `compact_candidate_indices_topk50_warp_kernel`: `~46%`
- `histogram_high_byte_topk50_kernel`: `~23%`
- `histogram_low_byte_topk50_kernel`: `~24%`
- `final_topk50_kernel`: `~6%`

The first selection round still performs a full `10000`-element pass with histogram-style counting. That is the specific target of this design.

## Non-Goals

- Replacing the full optimized path with a brand-new multi-round selector in one step
- Modifying the public API
- Touching the generic fallback path for `k != 50` or `seg_len != 10000`
- Replacing the specialized `final_topk50_kernel`
- Changing benchmark methodology or output format

## Recommended Scope

Implement a new specialized first-round selector for the optimized path only:

1. Inspect the highest 4 bits of `encode_half_desc(value)`
2. Use warp-local bucket accumulation over `16` nibble buckets
3. Reduce warp-local counts into a shared `16`-bucket total
4. Select the winning nibble bucket containing the top-`k` boundary
5. Convert that nibble decision into the next `SegmentSelectState`
6. Hand off to the existing later rounds and downstream kernels

This is a “replace only the first round” design. It deliberately avoids committing to a full nibble-round state machine until the first-round prototype proves it is worthwhile.

## Data Model

The selector operates on the existing descending encoded key:

- `encoded = encode_half_desc(value)`
- first-round nibble = `(encoded >> 12) & 0xf`

Descending selection semantics remain unchanged:

1. Larger numeric value ranks first
2. For equal values, smaller local index ranks first

The first-round nibble selector does not resolve final tie-breaks. It only reduces the candidate key-space before the later stages continue.

## First-Round Selector

### Input

Per segment:

- all `10000` encoded values
- target `k = 50`

### Local Bucket Structure

Each warp processes a small fixed batch and accumulates into `16` nibble buckets.

The intended implementation model is:

- represent bucket counts in compact warp-local registers or packed lane-local structures
- treat the `16` nibble buckets as the first selection frontier
- aggregate warp results into shared memory once per local batch instead of issuing a shared atomic on every element

The user-provided direction is preserved:

- conceptually use two compact `u4 * 16` bucket groups
- the first group covers the first local subset of elements
- the second group covers the remaining local subset
- warp reduction and shared accumulation happen after the local batch is filled

The design does not require a literal source-level `u4` type. It requires the same logical behavior: compact local nibble bucket accumulation before shared aggregation.

### Shared Aggregation

After each warp builds its local nibble buckets:

- warp-level reduction produces one `16`-bucket summary per warp
- shared memory accumulates the per-warp totals for the whole block
- one warp performs a prefix scan over the `16` buckets to locate the boundary nibble

### Output State

The first-round selector produces:

- `prefix` updated with the selected highest nibble
- `prefix_mask` updated to cover that nibble
- `selected_count` equal to the number of elements strictly better than the selected nibble
- `live_count` equal to the number of elements in the selected nibble bucket

It does not produce the final `cutoff_key`. That remains the responsibility of the later stages.

## Integration Strategy

The optimized `seg_len == 10000 && k == 50` path in `src/batch_topk.cu` changes as follows:

Current structure:

1. high-byte histogram over full input
2. high-byte boundary select
3. low-byte histogram over filtered set
4. low-byte cutoff finalize
5. top-k50 compaction
6. final top-k50 sort

Proposed structure:

1. highest-4-bit warp-centric selector over full input
2. existing later selection stage adapted to continue from the nibble-selected state
3. top-k50 compaction
4. final top-k50 sort

The critical point is that only the first stage is replaced. The remainder of the path should be changed as little as possible in the first iteration.

## State Continuation

The first-round selector must hand off into the existing state model cleanly.

Required invariants:

- `selected_count + live_count >= k`
- `prefix_mask` exactly covers the bits that are fixed so far
- later stages can interpret the selected nibble as the first narrowed live set

If the continuation logic becomes awkward enough that the later stages need to be rewritten completely, that is a signal to stop and reassess before implementation expands further.

## Testing Strategy

The first implementation must add:

1. A kernel-level unit test that directly checks the first-round nibble selector against a CPU-derived expectation on a small synthetic segment.
2. An optimized-path end-to-end regression that confirms:
   - output still matches `cpu_reference_topk`
   - `candidate_counts == 50`
   - the specialized path remains deterministic

Existing optimized-path tests must stay green:

- duplicate-heavy `k=50`
- large-random `k=50`
- NaN/Inf specialized-path regression
- small-batch optimized-path regression

## Acceptance Criteria

This design is considered successful only if all of the following hold:

- `./build/test_batch_topk` passes
- the optimized path remains gated to `seg_len == 10000 && k == 50`
- the generic fallback path remains unchanged in behavior
- benchmark improves beyond the current `825ed65` baseline on this machine, with special attention to:
  - `(128, 10000, 50)` better than `50.07 us`
  - `(6000, 10000, 50)` better than `917.04 us`

If the first-round selector does not beat that baseline, the implementation should be discarded rather than extended into later rounds.

## Risks

The main risks are:

- the first-round selector produces a state that is awkward to hand off into the later existing stages
- packed bucket logic becomes too opaque to maintain
- the selector improves local arithmetic but loses the gain to extra synchronization or data movement

The design explicitly mitigates these risks by limiting scope to the first round only.

## Expected Outcome

If the design works, it should:

- reduce the cost of the first full-segment selection pass
- preserve the already-validated specialized compaction and final top-k50 stages
- provide evidence for whether a fuller nibble-round state machine is worth building later

If it does not work, it should fail cleanly without destabilizing the rest of the optimized path.
