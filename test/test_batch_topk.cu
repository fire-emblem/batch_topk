#include <cassert>

#include "batch_topk.cuh"

int main() {
  const size_t workspace_size =
      radix_topk::batch_topk_half_workspace_size(1, 10000, 50);
  assert(workspace_size > 0);
  return 0;
}
