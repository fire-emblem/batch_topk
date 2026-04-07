# Batch TopK Benchmark Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shape-select, repeat, and summary controls to `bench_batch_topk` so future optimization work can compare stable per-shape medians instead of noisy single five-shape runs.

**Architecture:** Keep the benchmark executable and default output unchanged, but add a small env-controlled control surface to the benchmark driver. The implementation is split into two tasks: first make the tests actually validate the control surface behavior, then implement the benchmark driver changes. No runtime kernels or public APIs change.

**Tech Stack:** CUDA C++, CMake, CTest, NVCC

---

## File Responsibilities

- `test/test_batch_topk.cu`: Add subprocess-based smoke tests that invoke `./build/bench_batch_topk` with env vars and validate single-shape and repeat-summary behavior.
- `bench/bench_batch_topk.cu`: Add env parsing, shape filtering, repeat execution, and summary output while preserving the default five-case path.

### Task 1: Strengthen The Benchmark Control Tests

**Files:**
- Modify: `test/test_batch_topk.cu`

- [ ] **Step 1: Keep the existing matrix check and add a subprocess helper**

Add a helper like this:

```c++
bool run_benchmark_with_env(
    const std::vector<std::pair<std::string, std::string>>& env_vars,
    std::string* output,
    int* exit_code_out) {
  char output_path[] = "/tmp/test_batch_topk_benchXXXXXX";
  const int output_fd = mkstemp(output_path);
  if (output_fd == -1) {
    return false;
  }
  close(output_fd);

  struct SavedEnvVar {
    std::string key;
    bool had_value;
    std::string value;
  };

  std::vector<SavedEnvVar> saved_env_vars;
  saved_env_vars.reserve(env_vars.size());
  for (const auto& env_var : env_vars) {
    const char* existing = std::getenv(env_var.first.c_str());
    saved_env_vars.push_back(
        {env_var.first, existing != nullptr, existing ? std::string(existing) : std::string()});
    if (setenv(env_var.first.c_str(), env_var.second.c_str(), 1) != 0) {
      std::remove(output_path);
      return false;
    }
  }

  const std::string command =
      std::string("./build/bench_batch_topk > ") + output_path + " 2>&1";
  const int exit_code = std::system(command.c_str());

  for (auto it = saved_env_vars.rbegin(); it != saved_env_vars.rend(); ++it) {
    if (it->had_value) {
      setenv(it->key.c_str(), it->value.c_str(), 1);
    } else {
      unsetenv(it->key.c_str());
    }
  }

  std::ifstream input(output_path);
  if (!input) {
    std::remove(output_path);
    return false;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  *output = buffer.str();
  *exit_code_out = exit_code;
  std::remove(output_path);
  return true;
}
```

- [ ] **Step 2: Replace weak repeat/measurement smoke checks with control-surface checks**

Add:

```c++
size_t count_output_lines_with_tokens(
    const std::string& output,
    std::initializer_list<const char*> tokens) {
  size_t count = 0;
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    bool matches = true;
    for (const char* token : tokens) {
      if (line.find(token) == std::string::npos) {
        matches = false;
        break;
      }
    }
    if (matches) {
      ++count;
    }
  }
  return count;
}

bool check_benchmark_single_shape_control_contract() {
  std::string output;
  int exit_code = 0;
  if (!run_benchmark_with_env({{"BATCH_TOPK_BENCH_SEG_NUM", "128"}}, &output, &exit_code)) {
    return false;
  }
  return exit_code == 0 &&
         count_output_lines_with_tokens(output, {"seg_num=128", "latency_us="}) == 1 &&
         count_output_lines_with_tokens(output, {"seg_num=", "latency_us="}) == 1;
}

bool check_benchmark_repeat_summary_control_contract() {
  std::string output;
  int exit_code = 0;
  if (!run_benchmark_with_env({{"BATCH_TOPK_BENCH_SEG_NUM", "128"},
                               {"BATCH_TOPK_BENCH_REPEAT", "3"},
                               {"BATCH_TOPK_BENCH_PRINT_SUMMARY", "1"}},
                              &output,
                              &exit_code)) {
    return false;
  }
  return exit_code == 0 &&
         count_output_lines_with_tokens(output, {"seg_num=128", "latency_us="}) == 3 &&
         count_output_lines_with_tokens(output, {"seg_num=", "latency_us="}) == 3 &&
         count_output_lines_with_tokens(output, {"summary", "median_us", "min_us", "max_us"}) == 1;
}

bool check_benchmark_invalid_env_contract() {
  std::string output;
  int exit_code = 0;
  if (!run_benchmark_with_env({{"BATCH_TOPK_BENCH_SEG_NUM", "abc"}}, &output, &exit_code)) {
    return false;
  }
  if (exit_code == 0) {
    return false;
  }
  if (!run_benchmark_with_env({{"BATCH_TOPK_BENCH_REPEAT", "abc"}}, &output, &exit_code)) {
    return false;
  }
  return exit_code != 0;
}
```

Wire them into `main()`:

```c++
  if (!check_benchmark_single_shape_control_contract()) {
    std::fprintf(stderr, "benchmark single-shape control contract is incorrect\n");
    return 1;
  }
  if (!check_benchmark_repeat_summary_control_contract()) {
    std::fprintf(stderr, "benchmark repeat summary control contract is incorrect\n");
    return 1;
  }
  if (!check_benchmark_invalid_env_contract()) {
    std::fprintf(stderr, "benchmark invalid env contract is incorrect\n");
    return 1;
  }
```

- [ ] **Step 3: Run tests to verify red**

Run: `cmake --build build -j && ./build/test_batch_topk`

Expected: build succeeds, but `./build/test_batch_topk` exits nonzero because the benchmark driver does not yet implement the single-shape / repeat / invalid-env behavior.

- [ ] **Step 4: Commit the red control-surface tests**

```bash
git add test/test_batch_topk.cu
git commit -m "test: strengthen benchmark control contracts"
```

### Task 2: Implement Shape-Select, Repeat, And Summary In The Benchmark Driver

**Files:**
- Modify: `bench/bench_batch_topk.cu`

- [ ] **Step 1: Add tri-state integer env parsing**

Replace the current parsing helper with:

```c++
enum class EnvParseStatus {
  unset,
  ok,
  invalid,
};

EnvParseStatus parse_int_env(const char* name, int* value_out) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return EnvParseStatus::unset;
  }

  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || errno == ERANGE ||
      parsed < std::numeric_limits<int>::min() ||
      parsed > std::numeric_limits<int>::max()) {
    return EnvParseStatus::invalid;
  }

  *value_out = static_cast<int>(parsed);
  return EnvParseStatus::ok;
}
```

- [ ] **Step 2: Fail closed on invalid env values**

In `main()`, use the tri-state parser:

```c++
  int selected_seg_num = 0;
  const EnvParseStatus seg_num_status =
      parse_int_env("BATCH_TOPK_BENCH_SEG_NUM", &selected_seg_num);
  if (seg_num_status == EnvParseStatus::invalid) {
    std::fprintf(stderr, "invalid seg_num selector: %s\n", std::getenv("BATCH_TOPK_BENCH_SEG_NUM"));
    return 1;
  }
  const bool has_selected_seg_num = seg_num_status == EnvParseStatus::ok;

  int repeat_count = 1;
  const EnvParseStatus repeat_status =
      parse_int_env("BATCH_TOPK_BENCH_REPEAT", &repeat_count);
  if (repeat_status == EnvParseStatus::invalid || repeat_count <= 0) {
    std::fprintf(stderr, "invalid repeat count: %s\n", std::getenv("BATCH_TOPK_BENCH_REPEAT"));
    return 1;
  }
```

- [ ] **Step 3: Add repeat collection and summary output**

Use the already implemented pattern:

```c++
  const bool print_summary =
      std::getenv("BATCH_TOPK_BENCH_PRINT_SUMMARY") != nullptr;
```

For each selected case:

```c++
    std::vector<float> latencies;
    latencies.reserve(static_cast<size_t>(repeat_count));
    for (int repeat = 0; repeat < repeat_count; ++repeat) {
      ...
      latencies.push_back(latency_us);
      std::printf(...existing latency line...);
      if (print_stage_timing) {
        std::printf(...existing stage line...);
      }
    }
```

Print the summary only when explicitly requested:

```c++
    if (print_summary) {
      float min_us = latencies[0];
      float max_us = latencies[0];
      for (float latency_us : latencies) {
        if (latency_us < min_us) min_us = latency_us;
        if (latency_us > max_us) max_us = latency_us;
      }
      std::printf(
          "summary seg_num=%d runs=%d median_us=%.2f min_us=%.2f max_us=%.2f\n",
          benchmark_case.seg_num,
          repeat_count,
          median_latency_us(latencies),
          min_us,
          max_us);
    }
```

- [ ] **Step 4: Keep the default path unchanged**

When no env vars are set:

- all five cases still run
- each prints one latency line
- there is no summary line

- [ ] **Step 5: Run tests and benchmark to verify green**

Run: `cmake --build build -j && ./build/test_batch_topk && ./build/bench_batch_topk`

Expected:

- `./build/test_batch_topk` exits `0`
- default benchmark still prints five latency lines

Then run:

```bash
BATCH_TOPK_BENCH_SEG_NUM=128 BATCH_TOPK_BENCH_REPEAT=3 BATCH_TOPK_BENCH_PRINT_SUMMARY=1 ./build/bench_batch_topk
```

Expected:

- exactly three latency lines for `seg_num=128`
- exactly one summary line
- no output for the other four shapes

Then run:

```bash
BATCH_TOPK_BENCH_SEG_NUM=abc ./build/bench_batch_topk
```

Expected: exit code nonzero

And:

```bash
BATCH_TOPK_BENCH_REPEAT=abc ./build/bench_batch_topk
```

Expected: exit code nonzero

- [ ] **Step 6: Commit**

```bash
git add bench/bench_batch_topk.cu
git commit -m "bench: add shape-specific repeated measurement mode"
```

## Self-Review

- Spec coverage:
  - The plan covers single-shape selection, repeated runs, summary output, invalid-env failure, and preserving the default five-case benchmark path.
  - It does not change any runtime kernel logic.
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to Task N” placeholders remain.
  - Every code-changing step includes concrete code blocks and commands.
- Type consistency:
  - `EnvParseStatus`, `parse_int_env`, `median_latency_us`, `BATCH_TOPK_BENCH_SEG_NUM`, `BATCH_TOPK_BENCH_REPEAT`, and `BATCH_TOPK_BENCH_PRINT_SUMMARY` are named consistently across tasks.
