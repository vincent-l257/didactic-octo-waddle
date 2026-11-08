// include/basalt/prefill/gdn70.hpp - the prompt path's gated delta rule on compute capability 7.0, in FP32 SIMT
// (src/prefill/gdn70.cu).  Same interface, layouts and state as gdn_recurrence (gdn_chunked.cu).
#pragma once

#include <cstdint>

namespace basalt::prefill::gdn70 {

/// Bytes of `work` recurrence needs for T tokens.
uint64_t work_bytes(int64_t T);
/// gdn_recurrence's contract: state [128][48][128] in/out, h [T][10240], gate and beta [T][48], y [T][48][128] and
/// the gated RMS norm y16.
void recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z, const float* gamma,
                float eps, void* work, float* y, uint16_t* y16, int64_t T, void* stream);

}  // namespace basalt::prefill::gdn70
