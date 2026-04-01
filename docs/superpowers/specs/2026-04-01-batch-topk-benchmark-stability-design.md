# Batch TopK Benchmark Stability Design

## Goal

Stabilize benchmark measurement so future optimization decisions are based on repeatable evidence instead of noisy single-run totals.

This design is intentionally narrow:

- no algorithm changes
- no public API changes
- no correctness-path changes
- only benchmark driver and benchmark-oriented validation change

The immediate objective is to make `bench_batch_topk` useful for shape-specific optimization work on the existing optimized path.

## Current Problem

The current benchmark is noisy enough to mislead optimization decisions.

Recent repeated runs on the same code revision showed large variation:

- `(1500, 10000, 50)`: roughly `175 us` to `290 us`
- `(3000, 10000, 50)`: roughly `350 us` to `900+ us`
- `(6000, 10000, 50)`: roughly `888 us` to `1488 us`

The stage-timing output drifts along with the total runtime, which means the instability is not just one isolated stage. A single five-shape sequential run is currently too noisy to serve as a strict optimization gate.

## Non-Goals

- Replacing CUDA event timing
- Changing the target matrix in `include/batch_topk_benchmark.cuh`
- Adding profiler-only tooling requirements to the normal benchmark path
- Optimizing any kernel in this design

## Recommended Scope

Extend the benchmark driver so it can:

1. run only one selected shape
2. repeat that shape multiple times
3. report aggregate statistics such as median and minimum
4. print stage timing for that one selected shape when the existing timing env var is enabled

This allows future tuning work to compare like-for-like runs on:

- `128 x 10000 x 50`
- `6000 x 10000 x 50`

without mixing in interference from the other benchmark shapes.

## Interface

The benchmark executable remains `./build/bench_batch_topk`.

The new behavior is controlled only through environment variables so the existing default output remains usable.

Suggested environment variables:

- `BATCH_TOPK_BENCH_SEG_NUM`
  - when set, run only the benchmark case whose `seg_num` matches this value
- `BATCH_TOPK_BENCH_REPEAT`
  - number of repeated benchmark runs for the selected shape
- `BATCH_TOPK_BENCH_PRINT_SUMMARY`
  - when set, print median/min/max in addition to the per-run latency line
- `BATCH_TOPK_STAGE_TIMING`
  - already exists and continues to print stage breakdown

## Default Behavior

If no environment variable is set:

- keep the current five-shape loop
- keep the current `latency_us ref_us delta_us` line format

This preserves the benchmark as a repository-level smoke/perf script.

## Selected-Shape Behavior

If `BATCH_TOPK_BENCH_SEG_NUM` is set:

- run only the matching case from `kPrimaryBenchmarkCases`
- fail if no matching case exists
- do not silently fall back to running all shapes

This makes accidental comparisons less likely.

## Repeat Behavior

If `BATCH_TOPK_BENCH_REPEAT` is set:

- run the selected shape that many times
- each repeat uses the existing `run_benchmark_case(...)`
- collect the resulting `latency_us` values into a vector

Suggested aggregate outputs:

- `median_us`
- `min_us`
- `max_us`

The median is the main optimization comparison signal.

## Reporting

The benchmark should still print the existing latency line for compatibility:

```text
seg_num=... seg_len=... k=... latency_us=... ref_us=... delta_us=...
```

When repeated runs are enabled, add one summary line after the repeats:

```text
summary seg_num=... runs=... median_us=... min_us=... max_us=...
```

If stage timing is also enabled, stage timing should be emitted for each run as it is today, because future tuning may still need run-by-run stage breakdown.

## Validation

The test suite should add a small contract check proving the benchmark driver’s new control surface exists and stays narrow.

This contract should verify:

- the primary matrix still contains the same five `seg_num` values
- the benchmark is still centered on the same `seg_len=10000` and `k=50` target matrix

No benchmark test should attempt to assert exact latency.

## Acceptance Criteria

This design is considered successful only if all of the following hold:

- `./build/test_batch_topk` passes
- `./build/bench_batch_topk` still prints the five default cases when no env vars are set
- `BATCH_TOPK_BENCH_SEG_NUM=<value>` runs only one case
- `BATCH_TOPK_BENCH_REPEAT=<n>` produces repeatable multi-run output plus a summary line
- future optimization work can compare median values for one shape without rerunning the whole five-shape matrix

## Risks

The main risks are:

- adding too many benchmark modes and making the driver hard to read
- accidentally changing the default output format
- mixing measurement logic with algorithm logic

The design mitigates these risks by:

- using only a small number of env vars
- preserving the default path unchanged
- keeping all changes inside the benchmark and test files only

## Expected Outcome

If this design works, it should:

- make benchmark results more stable for optimization work
- reduce false positives and false negatives during keep-or-revert decisions
- make future tuning iterations faster because only the target shape needs to run

If it does not work, it should be easy to revert because no runtime kernel logic changes.
