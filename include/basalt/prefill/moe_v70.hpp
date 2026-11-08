// include/basalt/prefill/moe_v70.hpp - prompt expert products on compute capability 7.0 (src/prefill/moe_v70.cu).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::moe_v70 {

/// One job: expert slot e of the weights, n (1..kJobRows) consecutive rows from r0; input row = output row.
struct Job {
    int e, r0, n;
};
constexpr int kJobRows = 64;

/// The current device runs these kernels (compute capability 7.0).
bool enabled();
/// A layer's expert types these kernels take.
bool usable(int gu_type, int d_type);
/// A device buffer of at least n jobs on the current device (grows; nullptr when it cannot).
Job* job_buffer(size_t n);

/// gate|up (M = 2 FF rows of type gu_type, gate rows then up rows; K columns) on f16 rows x [rows][K]:
/// h[r][f] = silu(gate) * up as f16, row stride FF.  Expert e's weights at w + e * es.
void gu_swiglu(int gu_type, const Job* jobs, int nj, const void* w, size_t es, int M, int K, const uint16_t* x, uint16_t* h,
               void* stream);
/// down (M rows of IQ4_NL, K columns) on f16 rows h [rows][K]: out[r][m] as f32, row stride M.
void down(const Job* jobs, int nj, const void* w, size_t es, int M, int K, const uint16_t* h, float* out, void* stream);

}  // namespace basalt::prefill::moe_v70
