# Batch Radix TopK Structured Implementation Plan

## Goal Description

Organize the existing batch radix top-k draft into a repository-aligned implementation plan for a CUDA `half` operator that serves fixed-shape segments with `seg_len <= 10000` and `k <= 128`, generates local indices internally, and uses a radix-select plus final-sort pipeline for supported inputs.

The current repository already contains core source files, tests, and kernel helpers, but they are not yet fully aligned around one coherent baseline. The plan therefore focuses on reconciling build behavior, deterministic numeric semantics, GPU correctness, workspace handling, and benchmark reporting before performance tuning is treated as complete.

The intended end state is a buildable and testable operator with:

- a stable public CUDA API in `include/batch_topk.cuh`
- deterministic host and device ordering semantics
- a supported-path implementation in `src/batch_topk.cu` that routes through histogram, selection, compaction, and final sort
- correctness tests that compare GPU outputs against a CPU reference
- a benchmark driver that tracks the five RTX 4080 reference latencies and reports deltas explicitly

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: The public API, workspace query, and build targets are coherent for the supported fixed-shape `half` path.
  - Positive Tests (expected to PASS):
    - `cmake -S . -B build && cmake --build build -j` succeeds and produces both `test_batch_topk` and `bench_batch_topk`.
    - `batch_topk_half_workspace_size(1, 10000, 50)` returns a non-zero value.
    - A valid `batch_topk_half(...)` call with non-null buffers and sufficient workspace does not return `cudaErrorInvalidValue`.
  - Negative Tests (expected to FAIL):
    - `batch_topk_half_workspace_size(1, 10001, 50)`, `batch_topk_half_workspace_size(1, 10000, 129)`, and `batch_topk_half_workspace_size(1, 10, 11)` return `0`.
    - `batch_topk_half(...)` returns `cudaErrorInvalidValue` for null pointers, unsupported shapes, `k > seg_len`, or insufficient workspace.
    - The benchmark target must not remain a stub that expects `cudaErrorNotSupported` for supported inputs.
- AC-2: Reference ordering and `half` codec semantics are deterministic and shared across host and device paths.
  - Positive Tests (expected to PASS):
    - CPU reference ordering prefers larger values first and smaller local indices on ties.
    - `NaN` is treated as the lowest value, while `+inf`, `-inf`, `+0`, and `-0` follow the intended numeric ordering.
    - Host and device smoke checks for `encode_half_desc(...)` and `candidate_better(...)` agree on tie-break behavior.
  - Negative Tests (expected to FAIL):
    - A reference or codec implementation that lets `NaN` outrank finite values is rejected by tests.
    - A comparator that selects a larger index when values are equal is rejected by tests.
    - A codec that orders positive values below clearly smaller negative values is rejected by tests.
- AC-3: The GPU operator matches the CPU reference for supported inputs and returns deterministic local indices.
  - Positive Tests (expected to PASS):
    - Small multi-segment correctness tests match the CPU reference exactly for both values and indices.
    - Duplicate-heavy `k = 50` cases match the CPU reference exactly and preserve deterministic tie-break behavior.
    - Random, mixed-sign, monotonic, and representative large-shape regression inputs match the CPU reference for supported shapes.
  - Negative Tests (expected to FAIL):
    - Any output with unsorted values, mismatched local indices, or incorrect tie ordering is rejected by tests.
    - Any supported-path invocation that silently truncates results or emits fewer than `k` valid outputs is rejected by tests.
    - Any implementation that only passes tiny smoke inputs but fails `seg_len = 10000` regression shapes is rejected by tests.
- AC-4: The supported fast path is implemented as a radix-select plus final-sort pipeline with explicit candidate-state invariants.
  - Positive Tests (expected to PASS):
    - Histogram tests verify expected bucket counts for full and prefix-filtered passes.
    - Boundary-selection helpers produce the expected selected-count, live-count, prefix, and boundary digit for representative inputs.
    - Candidate compaction produces counts in `[k, kCompactedCandidateCap]` for supported duplicate-heavy and large-shape cases.
  - Negative Tests (expected to FAIL):
    - A path that bypasses the radix-select pipeline and performs full-segment sorting as the main supported path is not acceptable.
    - A compaction path that can overflow its candidate storage or produce fewer than `k` live candidates is rejected by tests.
    - A histogram or prefix-selection implementation that disagrees with the simulated boundary state is rejected by tests.
- AC-5: The benchmark harness reports the required target matrix and explicitly tracks delta versus the reference latencies.
  - Positive Tests (expected to PASS):
    - `./build/bench_batch_topk` prints one line for each of `(128,10000,50)`, `(1500,10000,50)`, `(3000,10000,50)`, `(4500,10000,50)`, and `(6000,10000,50)`.
    - Each benchmark line includes `seg_num`, `seg_len`, `k`, `latency_us`, `ref_us`, and `delta_us`.
    - Benchmark output is driven by actual supported-path execution rather than by a placeholder smoke message.
    - First-pass acceptance remains valid when `delta_us` is non-zero, provided correctness, validation, and required benchmark reporting all pass.
  - Negative Tests (expected to FAIL):
    - Missing target shapes, missing delta fields, or retained stub output are rejected.
    - Benchmark code that no longer reflects the supported operator contract is rejected.
    - Benchmark output that cannot be compared directly to the five reference numbers is rejected.
    - Treating absolute parity with the five reference numbers as the only initial pass/fail gate is rejected.
- AC-6: Performance tuning preserves correctness and makes fast-path progress visible for the primary `k = 50` workload.
  - Positive Tests (expected to PASS):
    - The implementation exposes or uses measured specialization buckets such as `k <= 32`, `k <= 64`, and `k <= 128` where beneficial.
    - After each tuning round, `ctest --test-dir build --output-on-failure` remains green.
    - Benchmark output makes parity status visible by showing positive, zero, or negative deltas against the references.
    - Parity with or improvement over the reference latencies is treated as a second-stage optimization objective after the first-pass acceptance gate is met.
  - Negative Tests (expected to FAIL):
    - Any optimization that regresses correctness, numeric semantics, or validation behavior is rejected.
    - Any tuning pass that removes explicit delta reporting or obscures comparison against the target matrix is rejected.
    - Any performance-only change that expands scope to unsupported dtypes or variable-length segments is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation fully aligns the existing repository around one supported fast path for `half`, keeps the public API stable, validates all supported-shape error paths, matches the CPU reference on deterministic correctness suites, reports the five primary benchmark shapes plus useful `k` specialization context, and applies measured fast-path tuning without breaking readability or testability.

This upper bound includes reconciling existing inconsistencies between `src/batch_topk.cu`, `test/test_batch_topk.cu`, and `bench/bench_batch_topk.cu`, and may include internal specialization for histogram, compaction, and final sort so long as the public API remains unchanged.

### Lower Bound (Minimum Acceptable Scope)

The implementation supports only the documented first-version workload: `half` inputs, fixed `seg_len <= 10000`, `1 <= k <= 128`, caller-provided workspace, internally generated local indices, deterministic tie-break behavior, and a supported path that still uses radix selection plus final sort for valid inputs.

The lower bound must still include:

- deterministic CPU reference coverage
- supported-shape GPU correctness tests
- explicit validation for invalid inputs and insufficient workspace
- benchmark output for the five reference shapes with delta reporting

### Allowed Choices

- Can use: the existing file split under `include/`, `src/`, `test/`, and `bench/`
- Can use: CUB block primitives, shared-memory staging, and host-side helper logic for initial state construction where needed
- Can use: internal specialization buckets for `k <= 32`, `k <= 64`, and `k <= 128`
- Can use: conservative candidate caps and benchmark-guided defaults for compaction thresholds
- Cannot use: device-wide full radix sort or full-segment sort as the main supported fast path
- Cannot use: expanding first-version scope to variable-length segments, generic dtype coverage, or externally supplied indices
- Cannot use: changing the public API to hide workspace allocation inside the operator
- Cannot use: leaving benchmark or test entrypoints in a placeholder state that disagrees with the supported operator behavior

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Treat the repository as a partially implemented radix-topk project rather than as a greenfield scaffold. Start by making build, smoke, and validation behavior consistent across the public API, tests, and benchmark driver. Once the baseline contract is coherent, lock down deterministic semantics for the CPU reference and the `half` codec. Then make the supported GPU path verifiably route through histogram, boundary selection, compaction, and final sort for valid inputs. Only after the supported path is benchmark-visible should tuning work begin.

One practical execution path is:

1. Audit and reconcile current behavior in `src/batch_topk.cu`, `test/test_batch_topk.cu`, and `bench/bench_batch_topk.cu`.
2. Harden shared ordering semantics in `include/batch_topk_types.cuh` and codec helpers.
3. Verify histogram and selection invariants before relying on large-shape correctness.
4. Replace benchmark stub assumptions with actual timing and delta reporting for the five target shapes.
5. Tune candidate caps, CTA sizing, and `k` specialization buckets while rerunning correctness after each change.

### Relevant References

- `docs/superpowers/specs/2026-03-31-batch-radix-topk-design.md` - design intent, target shapes, non-goals, and benchmark expectations
- `docs/superpowers/plans/2026-03-31-batch-radix-topk.md` - original draft plan with granular implementation steps
- `include/batch_topk.cuh` - public API surface that the structured plan must preserve
- `include/batch_topk_types.cuh` - shared data structures and ordering helpers
- `src/batch_topk.cu` - host launcher, validation path, and workspace orchestration
- `src/radix_histogram.cuh` - histogram kernel responsibilities and radix digit handling
- `src/radix_select_state.cuh` - threshold-selection helpers and segment state definitions
- `src/candidate_compact.cuh` - candidate workspace layout and compaction logic
- `src/final_block_sort.cuh` - final sorting path for compacted candidates
- `test/test_batch_topk.cu` - correctness, validation, and smoke coverage to be aligned with the plan
- `bench/bench_batch_topk.cu` - benchmark harness that must track the five reference latencies

## Dependencies and Sequence

### Milestones

1. Milestone 1: Baseline contract alignment
   - Phase A: Validate the current build graph and public API behavior against the documented supported shapes.
   - Phase B: Remove inconsistencies between operator, tests, and benchmark entrypoints so the repository has one coherent baseline.
2. Milestone 2: Deterministic semantics and correctness foundation
   - Phase A: Finalize CPU reference ordering, `half` codec semantics, and host/device tie-break behavior.
   - Phase B: Extend correctness coverage across small, duplicate-heavy, random, and large supported inputs.
3. Milestone 3: Radix pipeline completion
   - Phase A: Verify histogram and boundary-selection invariants.
   - Phase B: Complete candidate compaction and final-sort behavior for supported inputs.
4. Milestone 4: Benchmark visibility
   - Phase A: Replace any stub benchmark behavior with real operator execution.
   - Phase B: Emit the five target shapes with absolute latency and delta versus the RTX 4080 references.
5. Milestone 5: Performance tuning and stabilization
   - Phase A: Measure dominant bottlenecks for the `k = 50` workload and select specialization buckets.
   - Phase B: Iterate on tuning while preserving correctness and reporting parity status after each round.

The dependency order is strict: benchmark-driven tuning is only meaningful after baseline API behavior, deterministic semantics, and supported-path correctness are stable. Performance work depends on benchmark visibility, which in turn depends on the supported radix pipeline being functionally correct.

## Task Breakdown

Each task includes exactly one routing tag:
- `coding`: implemented in-repo
- `analyze`: investigation and measurement work

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Audit current repository state against the draft and design documents, with special focus on mismatches between `src/batch_topk.cu`, `test/test_batch_topk.cu`, and `bench/bench_batch_topk.cu`. | AC-1, AC-4, AC-5 | analyze | - |
| task2 | Align build targets, public API validation, workspace sizing, and supported-shape error handling so tests and benchmark use the same operator contract. | AC-1 | coding | task1 |
| task3 | Finalize shared deterministic semantics for CPU reference ordering, `half` codec normalization, tie-break behavior, and host/device smoke coverage. | AC-2 | coding | task2 |
| task4 | Verify and, if needed, correct histogram-pass behavior, boundary-state selection, workspace layout, and candidate-count invariants for the radix pipeline. | AC-4 | coding | task3 |
| task5 | Complete and validate end-to-end GPU correctness on small, duplicate-heavy, mixed-sign, random, and large supported shapes using the CPU reference as oracle. | AC-3, AC-4 | coding | task4 |
| task6 | Replace benchmark stubs with a real target-matrix runner that emits latency and delta-versus-reference fields for the five primary shapes. | AC-5 | coding | task5 |
| task7 | Measure benchmark results for the target matrix, identify dominant runtime bottlenecks, and summarize which internal knobs most affect the `k = 50` workload. | AC-5, AC-6 | analyze | task6 |
| task8 | Apply fast-path specialization and tuning changes, then rerun correctness and benchmark suites to confirm progress without regressions. | AC-6 | coding | task7 |

## Claude-Codex Deliberation

### Agreements

- The repository is already relevant to the draft, so the plan should adapt the draft to the current file layout instead of repeating greenfield "create file" steps.
- Correctness and deterministic semantics must be made explicit before tuning claims are considered meaningful.
- The benchmark harness must become a first-class artifact that reports the five target shapes and their deltas versus the reference numbers.

### Resolved Disagreements

- Draft framing vs repository reality: The draft reads like a fresh implementation sequence, while the repository already contains partial code and tests. The chosen resolution is to reinterpret the work as alignment, completion, and tuning of the existing codebase.
- Benchmark gate strictness: The draft emphasizes the `11 / 82 / 161 / 241 / 319 us` numbers strongly, but the repository does not yet have a trustworthy benchmark path. The chosen resolution is to require correctness, validation, and explicit `delta_us` reporting for first-pass acceptance, while treating absolute parity with the five reference numbers as a second-stage optimization goal.

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- None. Benchmark gate policy has been resolved in this revision: explicit target-matrix reporting and `delta_us` visibility are required for first-pass acceptance, while absolute parity with the five RTX 4080 references remains a tuning goal rather than a blocking initial gate.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must not contain plan-specific labels such as `AC-`, `Milestone`, `Phase`, `task1`, or similar workflow markers.
- Use domain-appropriate names such as `workspace_bytes`, `candidate_counts`, `histogram_pass_kernel`, or `batch_topk_half` rather than plan vocabulary.
- Keep the public namespace and API naming consistent with the current repository unless an explicit API change is approved separately.
- Keep validation logic, benchmark assumptions, and test expectations aligned around the same supported-shape contract.
- Benchmark output field names should remain stable and machine-readable once introduced.
