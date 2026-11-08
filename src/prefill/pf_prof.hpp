// src/prefill/pf_prof.hpp - BASALT_PF_PROF: per-device time between ticks on the compute stream, by label (each tick
// syncs the stream, so the profiled run is slower than a plain one).
#pragma once

#include <cuda_runtime.h>

namespace basalt::prefill::detail {

/// The time since this device's previous tick goes to `label`; no-op without BASALT_PF_PROF.
void pf_tick(cudaStream_t s, const char* label);
/// Every `every` calls on a device, log its totals.
void pf_report(int every);

}  // namespace basalt::prefill::detail
