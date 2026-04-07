# batch_radix_topk

CUDA batch top-k prototype for fixed-shape segments, focused on the optimized path:

- input shape: `(seg_num, seg_len)`
- current tuned target: `seg_len == 10000`, `k == 50`
- primary datatype: `half`
- outputs: top-k values and local indices

## Repository Layout

- `src/`: CUDA implementation, including histogram, cutoff selection, compaction, and final top-k kernels
- `include/`: public API and shared types
- `test/test_batch_topk.cu`: correctness and regression coverage
- `bench/bench_batch_topk.cu`: benchmark driver for the five reference shapes
- `docs/superpowers/specs/`: design notes
- `docs/superpowers/plans/`: implementation plans

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Test

```bash
./build/test_batch_topk
```

This validates CPU reference ordering, CUDA kernel behavior, optimized-path regressions, and benchmark control contracts.

## Benchmark

Default benchmark:

```bash
./build/bench_batch_topk
```

Shape-specific repeated benchmark:

```bash
BATCH_TOPK_BENCH_SEG_NUM=128 \
BATCH_TOPK_BENCH_REPEAT=5 \
BATCH_TOPK_BENCH_PRINT_SUMMARY=1 \
./build/bench_batch_topk
```

Per-stage timing:

```bash
BATCH_TOPK_STAGE_TIMING=1 ./build/bench_batch_topk
```

## Public API

```c++
cudaError_t batch_topk_half(...);
size_t batch_topk_half_workspace_size(int seg_num, int seg_len, int k);
```

Workspace is caller-provided. Unsupported shapes return `0` workspace or `cudaErrorInvalidValue`.
