// include/basalt/kernels/q6k_f16.hpp - Q6_K projections of a verify window on fp16 tensor cores (cc 7.0).
//
// y[c][r] = sum_k w[r][k] * x[c][k] with w = fp16((q - 32) * fp16(d * sc)) dequantized from the GGUF block in
// registers, x the fp16 rounding of the fp32 activations, products summed in fp32 by mma.sync m8n8k4 (f16 inputs,
// f32 accumulate).  Every column is summed in an order fixed by the shape alone: a column's value does not depend on
// how many columns ride along.  No q8_1: the activations keep fp16 precision.
//
// q6k_f16_prep writes ncols fp32 columns of n_in values as the fp16 layout q6k_f16 reads (q6k_f16_bytes of it).
#pragma once

#include <cstddef>

namespace basalt::kernels {

/// This card and call take the fp16 path: a cc 7.0 device, Q6_K (type 14), 2 <= ncols <= 8, n_in a multiple of 512,
/// n_out >= 4096, weights 4-byte aligned.
bool q6k_f16_supported(int ggml_type, int n_in, int n_out, int ncols, const void* weights) noexcept;

std::size_t q6k_f16_bytes(int n_in, int ncols) noexcept;

void q6k_f16_prep(const float* x, void* xh, int n_in, int ncols, void* stream);

void q6k_f16(const void* weights, const void* xh, float* y, int n_in, int n_out, int ncols, void* stream);

}  // namespace basalt::kernels
