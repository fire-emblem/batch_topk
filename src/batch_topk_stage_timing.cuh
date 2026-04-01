#pragma once

namespace radix_topk {

struct BatchTopkStageTiming {
  float high_byte_hist_us = 0.0f;
  float high_byte_select_us = 0.0f;
  float low_byte_hist_us = 0.0f;
  float finalize_cutoff_us = 0.0f;
  float compaction_us = 0.0f;
  float final_topk_us = 0.0f;
};

void set_batch_topk_stage_timing_sink(BatchTopkStageTiming* sink);

}  // namespace radix_topk
