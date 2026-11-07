// include/basalt/kernels/sm70_emu.cuh - compute capability 7.0 bodies for instructions it lacks.
//
// Only the sm_70 cubin compiles these. sm_86 and sm_120 keep their own instructions, so their code is unchanged.
// Async copies become plain copies (the commit/wait points are then no-ops). ldmatrix and the m16n8 mma shapes are
// computed by the warp from the same register fragments the instructions use.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace basalt::v70 {

__device__ __forceinline__ void* smem(uint32_t s) { return __cvta_shared_to_generic((size_t) s); }

/// cp.async of N bytes (4, 8 or 16) from `g` to shared address `s`.
template <int N> __device__ __forceinline__ void copy(uint32_t s, const void* g) {
    if constexpr (N == 16) *(uint4*) smem(s) = *(const uint4*) g;
    else if constexpr (N == 8) *(uint2*) smem(s) = *(const uint2*) g;
    else { static_assert(N == 4); *(uint32_t*) smem(s) = *(const uint32_t*) g; }
}

/// cp.async with a source size: `src` bytes are read and the rest of the N are zero.
template <int N> __device__ __forceinline__ void copy_z(uint32_t s, const void* g, int src) {
    if (src >= N) { copy<N>(s, g); return; }
    uint8_t* d = (uint8_t*) smem(s);
    const uint8_t* p = (const uint8_t*) g;
    for (int i = 0; i < N; ++i) d[i] = i < src ? p[i] : 0;
}

/// __reduce_add_sync over the lanes in `mask` (the full warp where it is used).
__device__ __forceinline__ int reduce_add(unsigned mask, int v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}
__device__ __forceinline__ unsigned reduce_add(unsigned mask, unsigned v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(mask, v, o);
    return v;
}

/// ldmatrix .m8n8 .b16: each of the first 8*R lanes gives a row address of matrix lane/8; reg i of lane t is the
/// 32-bit word t%4 of row t/4 of matrix i.
template <int R> __device__ __forceinline__ void ldmatrix(uint32_t* r, const void* row) {
    const unsigned long long a = (unsigned long long) row;
    const int t = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const unsigned long long ra = __shfl_sync(0xffffffffu, a, i * 8 + t / 4);
        r[i] = ((const uint32_t*) ra)[t % 4];
    }
}

/// ldmatrix .trans: reg i of lane t holds elements (2*(t%4), t/4) and (2*(t%4)+1, t/4) of matrix i.
template <int R> __device__ __forceinline__ void ldmatrix_trans(uint32_t* r, const void* row) {
    const unsigned long long a = (unsigned long long) row;
    const int t = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const unsigned long long r0 = __shfl_sync(0xffffffffu, a, i * 8 + 2 * (t % 4));
        const unsigned long long r1 = __shfl_sync(0xffffffffu, a, i * 8 + 2 * (t % 4) + 1);
        const uint32_t lo = ((const uint16_t*) r0)[t / 4];
        const uint32_t hi = ((const uint16_t*) r1)[t / 4];
        r[i] = lo | (hi << 16);
    }
}

template <int R> __device__ __forceinline__ void ldmatrix_s(uint32_t* r, uint32_t s) { ldmatrix<R>(r, smem(s)); }
template <int R> __device__ __forceinline__ void ldmatrix_trans_s(uint32_t* r, uint32_t s) {
    ldmatrix_trans<R>(r, smem(s));
}

/// cvt.rna.tf32.f32: round the magnitude half away from zero to the 10-bit tf32 mantissa (Inf and NaN kept).
__device__ __forceinline__ uint32_t tf32_rna(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fffe000u;   // NaN: canonical
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x1000u;
    return u & 0xffffe000u;
}

/// cvt.rn.bf16x2.f32: `hi` in the upper half, `lo` in the lower, round to nearest even.
__device__ __forceinline__ uint32_t bf16x2_rn(float hi, float lo) {
    auto rn = [](float f) -> uint32_t {
        const uint32_t u = __float_as_uint(f);
        if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fffu;   // NaN: canonical
        return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    };
    return rn(lo) | (rn(hi) << 16);
}

__device__ __forceinline__ float2 h2f(uint32_t x) { return __half22float2(*(const __half2*) &x); }
__device__ __forceinline__ float2 bf2f(uint32_t x) {
    return make_float2(__uint_as_float(x << 16), __uint_as_float(x & 0xffff0000u));
}
__device__ __forceinline__ int dp4a(uint32_t a, uint32_t b, int c) { return __dp4a((int) a, (int) b, c); }

// The m16n8 accumulators: lane (g = lane/4, t = lane%4) holds (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1). A row g's
// fragments live in lanes 4g..4g+3 and a B column n's in lanes 4n..4n+3, so the lane fetches them for step j from
// lane 4g+j (A) and lanes 4*2t+j, 4*(2t+1)+j (B).

/// mma.m16n8k16.row.col.f32.{f16,bf16}: d += A (a[4]) x B (b[2]).
template <bool BF16> __device__ __forceinline__ void mma_16816(float* d, const uint32_t* a, const uint32_t* b) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    auto cv = [](uint32_t x) { return BF16 ? bf2f(x) : h2f(x); };
    auto dot = [&](uint32_t x, uint32_t y) { const float2 p = cv(x), q = cv(y); return p.x * q.x + p.y * q.y; };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t A[4], B0[2], B1[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) A[i] = __shfl_sync(0xffffffffu, a[i], 4 * g + j);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            B0[i] = __shfl_sync(0xffffffffu, b[i], 8 * t + j);
            B1[i] = __shfl_sync(0xffffffffu, b[i], 8 * t + 4 + j);
        }
        d[0] += dot(A[0], B0[0]) + dot(A[2], B0[1]);
        d[1] += dot(A[0], B1[0]) + dot(A[2], B1[1]);
        d[2] += dot(A[1], B0[0]) + dot(A[3], B0[1]);
        d[3] += dot(A[1], B1[0]) + dot(A[3], B1[1]);
    }
}

/// mma.m16n8k32.row.col.s32.s8.s8.s32: d += A (a[4]) x B (b[2]).
__device__ __forceinline__ void mma_s8_16832(int* d, const uint32_t* a, const uint32_t* b) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t A[4], B0[2], B1[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) A[i] = __shfl_sync(0xffffffffu, a[i], 4 * g + j);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            B0[i] = __shfl_sync(0xffffffffu, b[i], 8 * t + j);
            B1[i] = __shfl_sync(0xffffffffu, b[i], 8 * t + 4 + j);
        }
        d[0] = dp4a(A[2], B0[1], dp4a(A[0], B0[0], d[0]));
        d[1] = dp4a(A[2], B1[1], dp4a(A[0], B1[0], d[1]));
        d[2] = dp4a(A[3], B0[1], dp4a(A[1], B0[0], d[2]));
        d[3] = dp4a(A[3], B1[1], dp4a(A[1], B1[0], d[3]));
    }
}

/// mma.m16n8k16.row.col.s32.s8.s8.s32: d += A (a[2]) x B (b[1]).
__device__ __forceinline__ void mma_s8_16816(int* d, const uint32_t* a, const uint32_t* b) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t A0 = __shfl_sync(0xffffffffu, a[0], 4 * g + j);
        const uint32_t A1 = __shfl_sync(0xffffffffu, a[1], 4 * g + j);
        const uint32_t B0 = __shfl_sync(0xffffffffu, b[0], 8 * t + j);
        const uint32_t B1 = __shfl_sync(0xffffffffu, b[0], 8 * t + 4 + j);
        d[0] = dp4a(A0, B0, d[0]);
        d[1] = dp4a(A0, B1, d[1]);
        d[2] = dp4a(A1, B0, d[2]);
        d[3] = dp4a(A1, B1, d[3]);
    }
}

/// mma.m16n8k8.row.col.f32.tf32.tf32.f32: d += A (a[4]) x B (b[2]); the inputs are read as tf32.
__device__ __forceinline__ void mma_tf32_1688(float* d, const uint32_t* a, const uint32_t* b) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    auto f = [](uint32_t x) { return __uint_as_float(x & 0xffffe000u); };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t A[4], B0[2], B1[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) A[i] = __shfl_sync(0xffffffffu, a[i], 4 * g + j);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            B0[i] = __shfl_sync(0xffffffffu, b[i], 8 * t + j);
            B1[i] = __shfl_sync(0xffffffffu, b[i], 8 * t + 4 + j);
        }
        d[0] += f(A[0]) * f(B0[0]) + f(A[2]) * f(B0[1]);
        d[1] += f(A[0]) * f(B1[0]) + f(A[2]) * f(B1[1]);
        d[2] += f(A[1]) * f(B0[0]) + f(A[3]) * f(B0[1]);
        d[3] += f(A[1]) * f(B1[0]) + f(A[3]) * f(B1[1]);
    }
}

}  // namespace basalt::v70

/// True in the passes that keep the sm_80+ instruction: the host pass and every image but the sm_70 cubin.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define BASALT_SM80 1
#else
#define BASALT_SM80 0
#endif
