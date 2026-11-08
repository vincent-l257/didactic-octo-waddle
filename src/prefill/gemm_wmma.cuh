// src/prefill/gemm_wmma.cuh - Y[t][n] = sum_k X[t][k] W[n][k] (+ beta Y) on FP16 tensor cores (WMMA), FP32 sums.
//
// For a card whose cuBLAS has no kernels (cc 7.0 under CUDA 13). BF16 operands are rounded to FP16 while staged into
// shared memory, clamped to FP16's range. 128 x 128 tiles, 8 warps of 64 x 32, K in steps of 32 with the next step
// prefetched into registers. Needs K % 32 == 0 and X, W 16-byte aligned (gemm_wmma_fits).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace basalt::prefill::wmma_gemm {

__device__ __forceinline__ uint4 to_half8(uint4 v, const __half*) { return v; }
__device__ __forceinline__ uint4 to_half8(uint4 v, const __nv_bfloat16*) {
    uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lo = __uint_as_float(w[i] << 16), hi = __uint_as_float(w[i] & 0xffff0000u);
        lo = fminf(fmaxf(lo, -65504.0f), 65504.0f);
        hi = fminf(fmaxf(hi, -65504.0f), 65504.0f);
        const __half2 h = __floats2half2_rn(lo, hi);
        w[i] = *(const uint32_t*) &h;
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

template <class T>
__global__ void __launch_bounds__(256) gemm_wmma_kernel(const T* X, const T* W, float* Y, int nt, int nn, int nk,
                                                        int ldy, float beta) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;
    __shared__ __align__(32) __half As[BM * LD];
    __shared__ __align__(32) __half Bs[BN * LD];
    __shared__ __align__(32) float Cs[8][16 * 16];
    const int t0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / 4, wn = warp % 4;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    uint4 ra[2], rb[2];
    const uint4 zero = make_uint4(0, 0, 0, 0);
    auto load = [&](int k0) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int idx = threadIdx.x + 256 * s, r = idx >> 2, kq = (idx & 3) * 8;
            ra[s] = t0 + r < nt ? *(const uint4*) (X + (size_t) (t0 + r) * nk + k0 + kq) : zero;
            rb[s] = n0 + r < nn ? *(const uint4*) (W + (size_t) (n0 + r) * nk + k0 + kq) : zero;
        }
    };
    load(0);
    for (int k0 = 0; k0 < nk; k0 += BK) {
        __syncthreads();
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int idx = threadIdx.x + 256 * s, r = idx >> 2, kq = (idx & 3) * 8;
            *(uint4*) &As[r * LD + kq] = to_half8(ra[s], X);
            *(uint4*) &Bs[r * LD + kq] = to_half8(rb[s], W);
        }
        __syncthreads();
        if (k0 + BK < nk) load(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(a[i], As + (wm * 64 + i * 16) * LD + kk, LD);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], Bs + (wn * 32 + j * 16) * LD + kk, LD);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(Cs[warp], acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int t = t0 + wm * 64 + i * 16 + e / 16, n = n0 + wn * 32 + j * 16 + e % 16;
                if (t < nt && n < nn) {
                    float* y = Y + (size_t) t * ldy + n;
                    *y = beta != 0.0f ? Cs[warp][e] + beta * *y : Cs[warp][e];
                }
            }
            __syncwarp();
        }
#endif
}

inline bool gemm_wmma_fits(const void* X, const void* W, int64_t K) {
    return K % 32 == 0 && ((uintptr_t) X & 15) == 0 && ((uintptr_t) W & 15) == 0;
}

}  // namespace basalt::prefill::wmma_gemm
