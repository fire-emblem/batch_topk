#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <cuda_fp16.h>

namespace radix_topk {

inline uint16_t normalize_half_bits(uint16_t bits) {
  const uint16_t exp_mask = 0x7c00u;
  const uint16_t mantissa_mask = 0x03ffu;
  if ((bits & exp_mask) == exp_mask && (bits & mantissa_mask) != 0) {
    return 0xffffu;
  }
  if ((bits & 0x7fffu) == 0) {
    return 0x0000u;
  }
  return bits;
}

inline uint16_t encode_half_asc(half value) {
  uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits = normalize_half_bits(bits);
  return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                          : static_cast<uint16_t>(bits ^ 0x8000u);
}

inline uint16_t encode_half_desc(half value) {
  return static_cast<uint16_t>(~encode_half_asc(value));
}

inline bool candidate_better(const Candidate& lhs, const Candidate& rhs) {
  const float lhs_value = __half2float(lhs.value);
  const float rhs_value = __half2float(rhs.value);
  const bool lhs_nan = std::isnan(lhs_value);
  const bool rhs_nan = std::isnan(rhs_value);
  if (lhs_nan != rhs_nan) {
    return rhs_nan;
  }
  if (!lhs_nan && lhs_value != rhs_value) {
    return lhs_value > rhs_value;
  }
  return lhs.index < rhs.index;
}

}  // namespace radix_topk
