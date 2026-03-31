# Batch Radix TopK Design

## Goal

Implement a very high performance CUDA batch top-k operator for fixed-shape segments using a radix-select plus merge-sort pipeline.

Primary target:

- Input shape: `(segNum, segLen)`
- Output shape: `(segNum, k)`
- First optimized path: `segLen <= 10000`, `k <= 128`
- Primary input type: `half`
- Input has values only
- Output returns top-k values and internally generated local indices in `[0, segLen)`

Target benchmark shapes:

- `(128, 10000) -> (128, 50)`
- `(1500, 10000) -> (1500, 50)`
- `(3000, 10000) -> (3000, 50)`
- `(4500, 10000) -> (4500, 50)`
- `(6000, 10000) -> (6000, 50)`

Reference latency targets in microseconds for the above shapes:

- `(128, 10000), k = 50`: `11 us`
- `(1500, 10000), k = 50`: `82 us`
- `(3000, 10000), k = 50`: `161 us`
- `(4500, 10000), k = 50`: `241 us`
- `(6000, 10000), k = 50`: `319 us`

Performance goal:

- Match and then exceed the above reference latencies on RTX 4080-class comparison runs
- Treat these numbers as first-class optimization targets during implementation

## Non-Goals

- Variable-length segments in the first version
- Stable sort semantics
- Full generic library coverage for arbitrary data types
- Full-sort implementation as the main execution path

## High-Level Approach

The operator is split into two stages:

1. Radix-select stage: identify a small candidate set that must contain the final top-k for each segment.
2. Final sort stage: compact the candidate set and run a block-level sort/merge network to produce ordered top-k outputs.

This design uses radix-select to avoid sorting all `segLen` elements and uses a small final sorting kernel because `k <= 128`.

## API

Proposed host API:

```c++
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

Notes:

- `seg_len` is runtime-visible but first implementation only supports `seg_len <= 10000`.
- `k` is runtime-visible but validated to `1 <= k <= 128`.
- `d_output_indices` stores per-segment local indices.
- Workspace is caller-provided to avoid repeated allocations on the fast path.

## Data Model

Each segment is independent.

- Input segment `s` occupies `d_input[s * seg_len : (s + 1) * seg_len]`
- Output values for segment `s` occupy `d_output_values[s * k : (s + 1) * k]`
- Output indices for segment `s` occupy `d_output_indices[s * k : (s + 1) * k]`

The index is not loaded from input. It is synthesized as the column offset within the segment.

Comparison order:

1. Larger value ranks first
2. For equal values, smaller original index ranks first

This tie-break rule makes the output deterministic.

## Numeric Semantics

The first version is optimized for `half` input.

Internal compare semantics:

- `half` values are converted into monotonic unsigned compare keys for radix processing
- Negative and positive values must preserve numeric ordering
- `-0` and `+0` are treated as equal values
- `+inf` and `-inf` follow standard numeric ordering
- `NaN` is treated as the lowest value and should never be selected when enough non-NaN values exist

The final output value remains `half`.

## Kernel Pipeline

### 1. Type Codec

Purpose:

- Convert `half` to a radix-comparable `uint16_t` or widened `uint32_t` key
- Provide helper comparison logic for final sort

Responsibilities:

- Monotonic encoding for descending top-k selection
- NaN normalization to a defined minimum rank

### 2. Segment Histogram Pass

Purpose:

- Process one radix digit over active elements of each segment
- Count digit populations without materializing a full sorted array

Execution model:

- Main path uses multiple CTAs per segment for histogram accumulation when `seg_num` is small or moderate
- When `seg_num` is very large, the implementation can fall back to one CTA per segment to keep scheduling simple

Digit width:

- Default to 8-bit digits for a 256-bin histogram
- For half inputs, this implies two radix rounds for the value bits if the full encoded key is used, plus any tie-handling logic outside radix

Use of CUB:

- CUB block load/store and block scan primitives are allowed inside the kernel implementation
- CUB device-wide full radix sort is not used as the main algorithm

### 3. Segment Threshold Selection

Purpose:

- Decide which digit bucket contains the top-k boundary
- Track how many items are strictly above the current bucket and how many remain tied in the boundary bucket

State tracked per segment:

- Current radix prefix
- Remaining candidate count
- Count of items already known to be in the top region
- Whether the segment is ready for compaction

The segment stays in radix-select mode until the candidate count shrinks below a configurable threshold.

### 4. Candidate Compaction

Purpose:

- Write candidate `(encoded_key, value, index)` triples into a compact workspace buffer per segment

Compaction trigger:

- Trigger once candidate count is less than or equal to `alpha * k`
- Initial design target uses `alpha` in the range `2` to `4`
- For the first version, the exact threshold is benchmark-tuned, with `4 * k` as the safe default

This stage uses internally generated local indices and writes only live candidates.

Use of CUB:

- CUB block scan primitives may be used to compute local write offsets

### 5. Final Sort / Merge

Purpose:

- Sort the compact candidate list and output the first `k` items

Execution model:

- One CTA per segment
- Small fixed-size network optimized for `k <= 128`
- Candidate count expected to be small enough to fit comfortably into shared memory

Sorting method:

- Use a block-local bitonic or merge network
- The network sorts `(value, index)` pairs using the deterministic comparison rule

Output:

- Top-k values in descending order
- Matching local indices

## Execution Strategy

The design is intentionally specialized around the target workload.

Fast-path assumptions:

- `dtype = half`
- `seg_len <= 10000`
- `k <= 128`

Priority tuning point:

- `k = 50` should receive direct attention during benchmarking because it is the target production-like case

Recommended specialization split:

- Runtime parameter for `seg_len`
- Runtime parameter for `k`, validated to `<= 128`
- Internal template specialization buckets for `k <= 32`, `k <= 64`, and `k <= 128`

This preserves usability while still allowing optimized final-sort kernels.

## Workspace Layout

Workspace contains:

- Per-segment histogram buffers
- Per-segment selection state
- Candidate counts
- Compacted candidate storage
- Optional temporary storage for CUB primitives

Suggested logical layout:

```text
[segment state][histograms][candidate counts][candidate triples][temp scratch]
```

Candidate triple layout:

```c++
struct Candidate {
  uint32_t encoded_key;
  half value;
  int index;
};
```

The exact packed layout can be revised during implementation if alignment or bandwidth measurements justify it.

## Fallback Paths

The first version should include a correctness-preserving fallback.

Cases:

- If `seg_len > 10000`, return `cudaErrorInvalidValue` in the first version
- If `k > 128`, return `cudaErrorInvalidValue`
- If workspace is insufficient, return `cudaErrorInvalidValue`

Deferred internal fallback:

- For very small `seg_len`, a future revision may bypass radix-select and run direct block sort

The first milestone does not implement this branch. The initial implementation always uses the radix-select plus final-sort pipeline for supported inputs.

## File Structure

Recommended initial layout:

- `include/batch_topk.cuh`: public API declarations
- `include/batch_topk_types.cuh`: candidate structs, compare helpers, type codec declarations
- `src/batch_topk.cu`: host launcher and workspace management
- `src/type_codec.cuh`: half-to-key conversion helpers
- `src/radix_histogram.cuh`: histogram kernels
- `src/radix_select_state.cuh`: threshold selection helpers
- `src/candidate_compact.cuh`: compaction kernels
- `src/final_block_sort.cuh`: final sort kernels
- `test/test_batch_topk.cu`: correctness tests
- `bench/bench_batch_topk.cu`: benchmark driver for target shapes
- `CMakeLists.txt`: build configuration

This split keeps the host entrypoint small and isolates the tuning-heavy kernels.

## Correctness Strategy

CPU reference behavior:

- For each segment, generate `(value, index)` pairs
- Sort by descending value, then ascending index
- Return first `k`

GPU implementation must match this exactly for non-NaN inputs.

For NaN:

- Treat as minimum value in both reference and GPU code

## Testing Strategy

### Unit Tests

- Type codec ordering for `half`
- Tie-break behavior for repeated values
- `NaN`, `inf`, `-inf`, `-0`, `+0`

### Correctness Tests

Test against CPU reference for:

- Random half inputs
- All-equal inputs
- Strictly increasing and strictly decreasing segments
- Duplicate-heavy inputs
- Mixed sign inputs
- Small and large `seg_num`
- `k` values including `1`, `32`, `50`, `64`, `128`

### Benchmark Tests

Required benchmark matrix:

- Shapes:
  - `(128, 10000), k = 50`
  - `(1500, 10000), k = 50`
  - `(3000, 10000), k = 50`
  - `(4500, 10000), k = 50`
  - `(6000, 10000), k = 50`
- Additional `k` sweep:
  - `k = 32`
  - `k = 64`
  - `k = 128`

Metrics:

- End-to-end kernel latency
- Effective throughput in elements per second
- Workspace bytes

Acceptance target for the primary benchmark set:

- The implementation must explicitly track the five reference latency targets
- Success means reaching parity first and then tuning beyond the reference numbers
- Benchmark reports should include both absolute latency and delta versus the reference targets

Comparison baselines:

- CPU reference for correctness only
- A simple GPU baseline, such as full segment sort or naive top-k, for performance context

## Error Handling

Host API validates:

- non-null pointers
- `seg_num > 0`
- `0 < seg_len <= 10000`
- `0 < k <= 128`
- sufficient workspace

Any kernel launch or CUDA runtime error is returned to the caller.

## Open Tuning Parameters

These are implementation-time tuning knobs, not API surface:

- radix digit width
- number of CTAs per segment in histogram stage
- compaction trigger threshold `alpha * k`
- shared-memory layout for candidate triples
- final sort network variant

The first version should keep these internal and choose defaults based on measurement.

## Milestones

### Milestone 1

- Buildable project skeleton
- Public API
- CPU reference
- Basic CUDA path that is correct for `half`, `seg_len <= 10000`, `k <= 128`

### Milestone 2

- Radix-select candidate reduction
- Compact candidate workspace
- Final block sort path

### Milestone 3

- Benchmark-driven tuning for the target shapes
- Improve occupancy and memory traffic
- Tune specifically for `k = 50`
- Track progress against the `11 / 82 / 161 / 241 / 319 us` reference targets

## Risks

- Multi-CTA per-segment histogram can add reduction overhead if tuned poorly
- Half-key ordering and NaN normalization must match reference exactly
- Overly large compact candidate pools will erase the benefit of radix-select
- Aggressive specialization can make code harder to extend if interfaces are not kept clean

## Decision Summary

The design intentionally favors the target workload over broad generality:

- Fixed-shape segments instead of ragged batches
- Half input first
- Internal index generation
- Radix-select to avoid full sort
- Small final sort specialized for `k <= 128`

This is the lowest-risk path to a very high performance batch top-k implementation for the requested shapes.
