// src/prefill/gemm_sm70.cuh - Y[t][n] = sum_k X[t][k] W[n][k] (+ beta Y) for a cc 7.0 card: FP16 tensor cores
// (mma.m8n8k4), FP32 sums.
//
// X [T, K] and W [N, K] row-major, each FP16 or BF16; a BF16 operand is rounded to FP16, clamped to its range, as it
// is staged into shared memory. Y FP32 with row stride ldy. Two shared-memory stages, the next K step prefetched into
// registers, one barrier per step; tiles in groups of row tiles for L2 reuse; the epilogue goes through shared
// memory in 16-row strips with 16-byte stores. The tile is picked from N; N <= 8 runs on CUDA cores with the operands
// taken exactly. Needs K % 32 == 0 and X, W 16-byte aligned (gemm_sm70_fits).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::prefill::sm70_gemm {

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

__device__ __forceinline__ void mma884(float* d, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}

template <int BM, int BN, int WM, int WN, int BK>
struct Tile {
    static constexpr int NW = (BM / WM) * (BN / WN), NT = NW * 32, LDS = BK + 8;
    static constexpr int MI = WM / 16, NJ = WN / 16, LDC = WN + 4;
    static constexpr int A_V = BM * BK / 8, B_V = BN * BK / 8;   // 16-byte vectors per stage
    static constexpr int A_PT = (A_V + NT - 1) / NT, B_PT = (B_V + NT - 1) / NT;
    static constexpr int PIPE = 2 * (BM + BN) * LDS * 2, EPI = NW * 16 * LDC * 4;
    static constexpr int SMEM = PIPE > EPI ? PIPE : EPI;
};

// A warp's four quad pairs take a 2 x 2 grid of its WM x WN tile. A quad pair's 8-row block i holds the warp's rows
// i*16 + {0..3, 8..11} + 4*qm (its columns likewise with qn), so the eight lanes of a 16-byte shared-memory phase hit
// distinct banks with a 16-byte row pad.
template <class TX, class TW, int BM, int BN, int WM, int WN, int BK, int MINB>
__global__ void __launch_bounds__(Tile<BM, BN, WM, WN, BK>::NT, MINB)
    gemm884_kernel(const TX* __restrict__ X, const TW* __restrict__ W, float* Y, int nt, int nn, int nk, int ldy,
                   float beta) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using C = Tile<BM, BN, WM, WN, BK>;
    constexpr int NT = C::NT, LDS = C::LDS, MI = C::MI, NJ = C::NJ, KV = BK / 8, GROUP = 8;
    extern __shared__ __align__(128) unsigned char smem[];
    __half* As = (__half*) smem;                  // [2][BM][LDS]
    __half* Bs = As + 2 * BM * LDS;               // [2][BN][LDS]

    // GROUP row tiles sweep the column tiles together
    const int tiles_n = (nn + BN - 1) / BN, tiles_m = (nt + BM - 1) / BM;
    const int bid = blockIdx.x, width = GROUP * tiles_n, m_first = bid / width * GROUP;
    const int gsz = min(tiles_m - m_first, GROUP);
    const int t0 = (m_first + (bid % width) % gsz) * BM, n0 = (bid % width) / gsz * BN;

    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = warp / (BN / WN), wn = warp % (BN / WN);
    const int q = (lane >> 2) & 3, qm = q & 1, qn = q >> 1, hi = lane >> 4, l4 = lane & 3;
    const int ar = hi * 8 + qm * 4 + l4, bc = hi * 8 + qn * 4 + l4;   // this lane's row of A, column of B

    float acc[MI][NJ][8];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.0f;

    uint4 ra[C::A_PT], rb[C::B_PT];
    const uint4 zero = make_uint4(0, 0, 0, 0);
    auto load = [&](int k0) {
#pragma unroll
        for (int s = 0; s < C::A_PT; ++s) {
            const int idx = tid + NT * s, r = idx / KV, kq = (idx % KV) * 8;
            if (C::A_V % NT == 0 || idx < C::A_V)
                ra[s] = t0 + r < nt ? __ldg((const uint4*) (X + (size_t) (t0 + r) * nk + k0 + kq)) : zero;
        }
#pragma unroll
        for (int s = 0; s < C::B_PT; ++s) {
            const int idx = tid + NT * s, r = idx / KV, kq = (idx % KV) * 8;
            if (C::B_V % NT == 0 || idx < C::B_V)
                rb[s] = n0 + r < nn ? __ldg((const uint4*) (W + (size_t) (n0 + r) * nk + k0 + kq)) : zero;
        }
    };
    auto store = [&](int buf) {
        __half* a = As + buf * BM * LDS;
        __half* b = Bs + buf * BN * LDS;
#pragma unroll
        for (int s = 0; s < C::A_PT; ++s) {
            const int idx = tid + NT * s, r = idx / KV, kq = (idx % KV) * 8;
            if (C::A_V % NT == 0 || idx < C::A_V) *(uint4*) &a[r * LDS + kq] = to_half8(ra[s], X);
        }
#pragma unroll
        for (int s = 0; s < C::B_PT; ++s) {
            const int idx = tid + NT * s, r = idx / KV, kq = (idx % KV) * 8;
            if (C::B_V % NT == 0 || idx < C::B_V) *(uint4*) &b[r * LDS + kq] = to_half8(rb[s], W);
        }
    };

    const int kt_n = nk / BK;
    load(0);
    store(0);
    __syncthreads();
    for (int kt = 0; kt < kt_n; ++kt) {
        const int cur = kt & 1;
        if (kt + 1 < kt_n) load((kt + 1) * BK);
        const __half* a = As + cur * BM * LDS + (wm * WM + ar) * LDS;
        const __half* b = Bs + cur * BN * LDS + (wn * WN + bc) * LDS;
#pragma unroll
        for (int kc = 0; kc < BK; kc += 8) {
            uint4 fa[MI], fb[NJ];
#pragma unroll
            for (int i = 0; i < MI; ++i) fa[i] = *(const uint4*) (a + i * 16 * LDS + kc);
#pragma unroll
            for (int j = 0; j < NJ; ++j) fb[j] = *(const uint4*) (b + j * 16 * LDS + kc);
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NJ; ++j) mma884(acc[i][j], fa[i].x, fa[i].y, fb[j].x, fb[j].y);
#pragma unroll
            for (int i = 0; i < MI; ++i)
#pragma unroll
                for (int j = 0; j < NJ; ++j) mma884(acc[i][j], fa[i].z, fa[i].w, fb[j].z, fb[j].w);
        }
        // the other stage was last read before the previous barrier
        if (kt + 1 < kt_n) store(cur ^ 1);
        __syncthreads();
    }

    float* cs = (float*) smem + warp * 16 * C::LDC;
    const bool vec = (ldy % 4) == 0 && (((uintptr_t) Y) & 15) == 0;
    constexpr int RV = WN / 4;   // float4 per strip row
#pragma unroll
    for (int i = 0; i < MI; ++i) {
        // accumulator e of block (i, j): row (l4 & 1) + 2 ((e >> 1) & 1) + 4 hi, column (e & 1) + 4 (e >> 2) + 2 (l4 >> 1)
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int po = (l4 & 1) + 2 * ((e >> 1) & 1) + 4 * hi, co = 4 * (e >> 2) + 2 * (l4 >> 1);
                const int r = (po >> 2) * 8 + qm * 4 + (po & 3), c = j * 16 + (co >> 2) * 8 + qn * 4 + (co & 3);
                *(float2*) &cs[r * C::LDC + c] = make_float2(acc[i][j][e], acc[i][j][e + 1]);
            }
        __syncwarp();
#pragma unroll
        for (int e = lane; e < 16 * RV; e += 32) {
            const int r = e / RV, c = (e % RV) * 4;
            const int t = t0 + wm * WM + i * 16 + r, n = n0 + wn * WN + c;
            if (t >= nt) continue;
            const float4 v = *(const float4*) &cs[r * C::LDC + c];
            float* y = Y + (size_t) t * ldy + n;
            if (vec && n + 3 < nn) {
                float4 o = v;
                if (beta != 0.0f) {
                    const float4 p = *(const float4*) y;
                    o.x += beta * p.x;
                    o.y += beta * p.y;
                    o.z += beta * p.z;
                    o.w += beta * p.w;
                }
                *(float4*) y = o;
            } else {
                const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int qq = 0; qq < 4; ++qq)
                    if (n + qq < nn) y[qq] = beta != 0.0f ? vv[qq] + beta * y[qq] : vv[qq];
            }
        }
        __syncwarp();
    }
#endif
}

// N <= NB: CUDA cores, the operands taken exactly into FP32. A warp takes RW rows; each lane strides K by 16-byte
// vectors; the W rows come through L1.
__device__ __forceinline__ void unpack8(uint4 v, const __nv_bfloat16*, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}
__device__ __forceinline__ void unpack8(uint4 v, const __half*, float* f) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p = __half22float2(*(const __half2*) &w[i]);
        f[2 * i] = p.x;
        f[2 * i + 1] = p.y;
    }
}

template <class TX, class TW, int NB, int RW>
__global__ void __launch_bounds__(256) gemm_narrow_kernel(const TX* __restrict__ X, const TW* __restrict__ W, float* Y,
                                                          int nt, int nn, int nk, int ldy, float beta) {
    const int warp = (blockIdx.x * 256 + threadIdx.x) / 32, lane = threadIdx.x % 32, t0 = warp * RW;
    if (t0 >= nt) return;
    float acc[RW][NB];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int n = 0; n < NB; ++n) acc[r][n] = 0.0f;
    for (int k = lane * 8; k < nk; k += 32 * 8) {
        float w[NB][8];
#pragma unroll
        for (int n = 0; n < NB; ++n) {
            if (n < nn) {
                unpack8(__ldg((const uint4*) (W + (size_t) n * nk + k)), W, w[n]);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) w[n][e] = 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            if (t0 + r >= nt) break;
            float x[8];
            unpack8(__ldcs((const uint4*) (X + (size_t) (t0 + r) * nk + k)), X, x);
#pragma unroll
            for (int n = 0; n < NB; ++n)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[r][n] = fmaf(x[e], w[n][e], acc[r][n]);
        }
    }
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int n = 0; n < NB; ++n)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[r][n] += __shfl_xor_sync(0xffffffffu, acc[r][n], o);
    // lane r * NB + n writes (r, n); every lane holds every sum after the butterfly
    float v = 0.0f;
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int n = 0; n < NB; ++n)
            if (lane == r * NB + n) v = acc[r][n];
    const int r = lane / NB, n = lane % NB;
    if (lane < RW * NB && t0 + r < nt && n < nn) {
        float* y = Y + (size_t) (t0 + r) * ldy + n;
        *y = beta != 0.0f ? v + beta * *y : v;
    }
}

inline bool gemm_sm70_fits(const void* X, const void* W, int64_t K) {
    return K % 32 == 0 && ((uintptr_t) X & 15) == 0 && ((uintptr_t) W & 15) == 0;
}

// The tile for a shape, and the launch through `launch(kernel, grid, block, smem_bytes, args...)` (launch_cc's
// sm_70 path opts the dynamic shared memory in above 48 KiB).
template <class TX, class TW, class Launch>
void gemm_sm70(const TX* X, const TW* W, float* Y, int T, int N, int K, int ldy, float beta, Launch&& launch) {
    auto tiled = [&](auto kernel, auto tile, int bm, int bn) {
        using C = decltype(tile);
        const unsigned blocks = (unsigned) (((T + bm - 1) / bm) * ((N + bn - 1) / bn));
        launch(kernel, dim3(blocks), dim3(C::NT), (size_t) C::SMEM, X, W, Y, T, N, K, ldy, beta);
    };
    if (N <= 8) {   // the hc inject product: memory-bound, no tensor tile fits it
        constexpr int RW = 8;
        const unsigned blocks = (unsigned) ((T + RW * 8 - 1) / (RW * 8));
        if (N <= 4)
            launch(gemm_narrow_kernel<TX, TW, 4, RW>, dim3(blocks), dim3(256), (size_t) 0, X, W, Y, T, N, K, ldy, beta);
        else
            launch(gemm_narrow_kernel<TX, TW, 8, 4>, dim3((unsigned) ((T + 31) / 32)), dim3(256), (size_t) 0, X, W, Y,
                   T, N, K, ldy, beta);
    } else if (N <= 48 && K % 64 == 0) {
        tiled(gemm884_kernel<TX, TW, 64, 48, 16, 48, 64, 3>, Tile<64, 48, 16, 48, 64>{}, 64, 48);
    } else if (N < 512) {
        tiled(gemm884_kernel<TX, TW, 128, 64, 64, 32, 32, 1>, Tile<128, 64, 64, 32, 32>{}, 128, 64);
    } else {
        tiled(gemm884_kernel<TX, TW, 256, 128, 64, 64, 32, 1>, Tile<256, 128, 64, 64, 32>{}, 256, 128);
    }
}

}  // namespace basalt::prefill::sm70_gemm
