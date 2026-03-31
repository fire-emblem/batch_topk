# Batch TopK Optimization Design

## Goal

Optimize the CUDA batch top-k operator for the fixed workload:

- Input shape: `(batch, 10000)`
- Per-batch operation: select top-`50`
- Output: top-`50` values and local indices within each batch
- Primary datatype: `half`

The design must improve both:

- Small-batch latency
- Large-batch throughput

Priority is given to large-batch throughput when a tradeoff is required.

## Current Baseline

As measured on April 1, 2026 in this repository:

- `./build/test_batch_topk` fails in the large-random regression because candidate counts can grow to `609`
- `./build/bench_batch_topk` reports:
  - `(128, 10000, 50)`: `490.74 us`
  - `(1500, 10000, 50)`: `4158.06 us`
  - `(3000, 10000, 50)`: `7454.29 us`
  - `(4500, 10000, 50)`: `10709.72 us`
  - `(6000, 10000, 50)`: `13496.75 us`

The current implementation in [src/batch_topk.cu](/home/cjxu/topk/radix_topk/src/batch_topk.cu) is still a correctness-first pipeline with three major bottlenecks:

1. It synchronizes the stream and copies histograms back to the host before boundary selection.
2. Candidate compaction can retain far more than `k` elements.
3. Final selection is still done by a single-thread insertion path.

## Scope

This design covers the optimized fast path for:

- `seg_len == 10000`
- `k == 50` as the primary tuned case
- `batch > 0`

The public API remains unchanged:

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

The first optimized implementation may continue to accept `k <= 128`, but `k == 50` is the only explicitly tuned case in this pass.

## Non-Goals

- Variable-length batches
- A full-sort fallback as the main supported path
- Generalization to arbitrary dtypes
- Silent fallback to a slower algorithm for supported inputs

## Recommended Approach

Use a GPU-only two-level radix-select pipeline:

1. High-byte histogram over encoded values
2. GPU boundary selection for each batch
3. Low-byte histogram inside the live high-byte bucket
4. Precise candidate compaction against a full cutoff key
5. CTA-local parallel final top-k specialized for `k = 50`

This approach is preferred over incremental patching because the current bottlenecks are architectural, not isolated kernel inefficiencies. It is also preferred over full sorting because sorting all `10000` elements per batch does unnecessary work for top-`50`.

## Execution Model

Each batch is independent and maps to one logical segment:

- `seg_num = batch`
- `seg_len = 10000`
- `k = 50`

Execution is split into two scheduling modes:

- Large-batch mode: one CTA per batch for steady throughput
- Small-batch mode: multiple CTAs per batch for histogram and reduction stages to keep the GPU occupied and reduce tail latency

The implementation must choose between these modes without changing API semantics.

## Ordering Semantics

Top-k ordering remains:

1. Larger value ranks first
2. For equal values, smaller local index ranks first

This rule is required for deterministic results and must be preserved in both compaction and final output.

## Cutoff and Candidate Control

The optimized path must not stop at a coarse high-byte boundary.

Instead, each batch tracks:

- `strictly_better_count`: number of elements guaranteed to be in the final top-`50`
- `remaining_slots`: number of slots still to be filled from elements equal to the cutoff key
- `cutoff_key`: the full value cutoff after high-byte and low-byte narrowing

Compaction rules:

- All elements better than `cutoff_key` are retained
- Elements equal to `cutoff_key` are retained only until `remaining_slots` is filled, using smaller local index first

The goal is to keep candidate counts close to `k`, rather than allowing the current behavior where counts can grow into the hundreds or even the full segment length.

## Kernel Breakdown

### Kernel A: High-Byte Histogram

Responsibilities:

- Compute a 256-bin histogram over the high byte of the encoded `half` key for each batch
- Support one-CTA-per-batch and multi-CTA-per-batch execution

Outputs:

- `hist_hi[batch, 256]`

### Kernel B: High-Byte Boundary Selection

Responsibilities:

- Scan `hist_hi`
- Determine the live high-byte bucket
- Compute `strictly_better_count` and live-bucket population

Outputs:

- per-batch selection state

This stage remains entirely on the GPU. The existing host synchronization and device-to-host histogram copy are removed.

### Kernel C: Low-Byte Histogram

Responsibilities:

- Revisit only values in the selected high-byte bucket
- Compute a 256-bin histogram over the low byte

Outputs:

- `hist_lo[batch, 256]`

This narrows the cutoff without rescanning all values blindly in the final stages.

### Kernel D: Precise Compaction

Responsibilities:

- Materialize only the candidates needed for final top-k
- Keep all values better than the full cutoff
- Keep only the needed number of cutoff-equal values, using ascending index order

Outputs:

- `candidate_count[batch]`
- compact candidate local indices

This kernel is responsible for enforcing a small fixed candidate cap rather than allowing degenerate full-segment retention.

### Kernel E: Final CTA-Local TopK

Responsibilities:

- Consume the compact candidate list for one batch
- Produce the final ordered top-`50`

Implementation direction:

- CTA-local parallel selection or small-sort network
- Specialized for `k = 50`

The current single-thread insertion path in [src/final_block_sort.cuh](/home/cjxu/topk/radix_topk/src/final_block_sort.cuh) is not sufficient for the target workload.

## Workspace Layout

The optimized workspace contains:

- `hist_hi[batch, 256]`
- `hist_lo[batch, 256]`
- `state[batch]`
- `candidate_count[batch]`
- `candidate_indices[batch, cap]`
- scratch for small-batch multi-CTA reduction when that execution mode is enabled

Design constraints:

- The candidate region must use a fixed cap near `k`, such as `64` or `96`
- The workspace must not reserve space for a full `10000` retained candidates per batch in the optimized path
- The workspace query function must continue to validate shape support before returning non-zero storage
- The final kernel reloads values from the input by local index rather than storing full candidate structs in the compacted buffer

## Error Handling

Argument validation remains explicit and strict:

- invalid pointers return `cudaErrorInvalidValue`
- unsupported shapes return `cudaErrorInvalidValue`
- insufficient workspace returns `cudaErrorInvalidValue`

The optimized path does not silently fall back to a different algorithm for supported shapes, because that would invalidate benchmark interpretation and mask performance regressions.

## Testing Strategy

Correctness tests continue to compare against the CPU reference, and the optimized path adds stronger regression checks:

- large-batch random input must keep candidates below the chosen small fixed cap
- duplicate-heavy input must preserve the value-first, index-second ordering rule
- small-batch cases such as `batch = 1`, `8`, and `32` must remain correct
- supported-shape validation must continue to reject unsupported inputs

The current failure in [test/test_batch_topk.cu](/home/cjxu/topk/radix_topk/test/test_batch_topk.cu) is treated as a real design signal: the candidate-control logic is too weak for the intended workload and must be fixed as part of this effort.

## Benchmark Strategy

Benchmarking continues to use the existing target matrix:

- `(128, 10000, 50)` with reference `11 us`
- `(1500, 10000, 50)` with reference `82 us`
- `(3000, 10000, 50)` with reference `161 us`
- `(4500, 10000, 50)` with reference `241 us`
- `(6000, 10000, 50)` with reference `319 us`

Acceptance is staged:

1. Restore correctness and remove the host round-trip bottleneck.
2. Reduce candidate counts to a bounded fixed cap.
3. Specialize the final top-k path for `k = 50`.
4. Tune large-batch throughput without regressing small-batch latency.

The benchmark output should continue to print `latency_us`, `ref_us`, and `delta_us`.

## Expected Outcome

After this optimization pass:

- the operator still serves `(batch, 10000) -> top50`
- correctness remains deterministic
- host synchronization is removed from boundary selection
- candidate counts become tightly bounded near `k`
- large-batch throughput improves materially
- small-batch latency no longer pays for a throughput-only design

This design intentionally focuses on the parts of the current implementation that dominate both throughput and latency rather than attempting superficial tuning of the existing pipeline.
