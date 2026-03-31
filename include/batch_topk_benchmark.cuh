#pragma once

#include <array>

namespace radix_topk {

struct BenchmarkCase {
  int seg_num;
  int seg_len;
  int k;
  float ref_us;
};

inline constexpr std::array<BenchmarkCase, 5> kPrimaryBenchmarkCases = {{
    {128, 10000, 50, 11.0f},
    {1500, 10000, 50, 82.0f},
    {3000, 10000, 50, 161.0f},
    {4500, 10000, 50, 241.0f},
    {6000, 10000, 50, 319.0f},
}};

}  // namespace radix_topk
