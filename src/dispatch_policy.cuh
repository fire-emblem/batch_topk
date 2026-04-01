#pragma once

namespace radix_topk {

inline int histogram_ctas_per_segment(int seg_num) {
  if (seg_num <= 8) {
    return 4;
  }
  if (seg_num <= 32) {
    return 2;
  }
  return 1;
}

}  // namespace radix_topk
