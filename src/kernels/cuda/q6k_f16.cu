// src/kernels/cuda/q6k_f16.cu - see include/basalt/kernels/q6k_f16.hpp.
//
// A "unit" is 4 values of one row: the ql bytes (l, l + 1) at 64h + 32jlo hold j = jlo in their low nibbles and
// j = jlo + 2 in their high nibbles, the qh bytes at 32h + l their top two bits; values 128h + 32j + l and l + 1.
// Unit u = (2h + jlo) * 16 + l / 2 of a 256-value block.  The prep stores each column's activations in unit order, so
// a thread's 4 k slots of an m8n8k4 step are one 8-byte load.
#include "basalt/kernels/q6k_f16.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/log.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace basalt::kernels {
namespace {

constexpr int kRows = 8;     // rows per warp: the m8n8k4 tile
constexpr int kWords = 53;   // the 4-byte words around one 210-byte block
constexpr int kStage = (kRows * kWords + 31) / 32;

__device__ __forceinline__ unsigned ld16(const unsigned char* p) { return *reinterpret_cast<const unsigned short*>(p); }
__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2*>(&u); }
// streamed weights, not kept in L1
__device__ __forceinline__ unsigned ld_stream(const void* p) {
    unsigned v;
    asm volatile("ld.global.nc.L1::no_allocate.b32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
// two bytes into the low bytes of two 16-bit lanes
__device__ __forceinline__ unsigned spread(unsigned b) { return __byte_perm(b, 0, 0x4140); }

// the unit's (j, l..l+1) pair into lo and its (j + 2, l..l+1) pair into hi, as fp16 (q - 32) * ds
__device__ __forceinline__ void dequant(unsigned v, unsigned h, int jlo, __half2 dlo, __half2 dhi, unsigned& lo,
                                        unsigned& hi) {
    const unsigned hs = h >> (2 * jlo);
    const unsigned a = (v & 0x000F000Fu) | ((hs << 4) & 0x00300030u) | 0x64006400u;   // fp16 1024 + q
    const unsigned b = ((v >> 4) & 0x000F000Fu) | (hs & 0x00300030u) | 0x64006400u;
    const __half2 bias = __halves2half2(__ushort_as_half(0x6420), __ushort_as_half(0x6420));   // 1056
    lo = h2u(__hmul2(__hsub2(u2h(a), bias), dlo));
    hi = h2u(__hmul2(__hsub2(u2h(b), bias), dhi));
}

__device__ __forceinline__ void mma884(float (&c)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__)
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}

__global__ void q6k_f16_prep_kernel(const float* __restrict__ x, uint2* __restrict__ xh, int n_in, int ncols) {
    pdl_wait();
    const int units = n_in / 4;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncols * units) return;
    const int col = i / units, uu = i % units, kb = uu / 64, u = uu % 64;
    const int h = u >> 5, jlo = (u >> 4) & 1, lp = u & 15;
    const float* s = x + (size_t) col * n_in + kb * 256 + 128 * h + 32 * jlo + 2 * lp;
    const float2 lo = *reinterpret_cast<const float2*>(s), hi = *reinterpret_cast<const float2*>(s + 64);
    xh[(size_t) col * units + uu] = make_uint2(h2u(__float22half2_rn(lo)), h2u(__float22half2_rn(hi)));
}

// One warp: 8 rows, the blocks kb = warp, warp + KS, ... of each row (KS = warps per CTA, summed in warp order).  Each
// quadpair q of the m8n8k4 step takes units (h, lgrp) = (q / 2, q % 2); a thread's A row and B column are its index
// within the quadpair.  The warp stages each block's aligned 212 bytes per row in shared memory, the next block's words
// loaded into registers while the current one is computed.
__global__ void __launch_bounds__(512) q6k_f16_kernel(const unsigned char* __restrict__ w, const uint2* __restrict__ xh,
                                                     float* __restrict__ y, int n_in, int n_out, int ncols) {
    extern __shared__ __align__(16) unsigned smem[];
    const int ks = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned* sw = smem + warp * kRows * kWords;
    const int row0 = blockIdx.x * kRows;
    const int bpr = n_in >> 8;
    const size_t rowbytes = (size_t) bpr * 210;
    const unsigned char* wr0 = w + (size_t) row0 * rowbytes;
    unsigned goff[kStage];   // byte offsets from row0's start; rows past n_out read the last row
#pragma unroll
    for (int m = 0; m < kStage; ++m) {
        const int i = lane + 32 * m;
        const int r = min(i / kWords, n_out - 1 - row0);
        goff[m] = (unsigned) (r * rowbytes) + 4 * (i % kWords);
    }
    unsigned R[kStage];
    auto gload = [&](int kb) {
        if (kb >= bpr) return;
        const unsigned char* p = wr0 + (size_t) 210 * kb - 2 * (kb & 1);
#pragma unroll
        for (int m = 0; m < kStage; ++m)
            if (lane + 32 * m < kRows * kWords) R[m] = ld_stream(p + goff[m]);
    };
    const int qp = (lane >> 2) & 3, idx = (lane & 3) + ((lane >> 4) << 2);
    const int h = qp >> 1, lgrp = qp & 1;
    const bool cv = idx < ncols;
    float c[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    gload(warp);
    pdl_wait();
    const uint2* xp = xh + (size_t) (cv ? idx : 0) * (n_in / 4);
    for (int kb = warp; kb < bpr; kb += ks) {
        __syncwarp();
#pragma unroll
        for (int m = 0; m < kStage; ++m)
            if (lane + 32 * m < kRows * kWords) sw[lane + 32 * m] = R[m];
        __syncwarp();
        gload(kb + ks);
        const unsigned char* sa = reinterpret_cast<const unsigned char*>(sw + idx * kWords) + 2 * (kb & 1);
        const uint2* xb = xp + kb * 64;
        const float d = __half2float(__ushort_as_half((unsigned short) ld16(sa + 208)));
        __half2 ds[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) ds[j] = __float2half2_rn(d * (float) (int8_t) sa[192 + 8 * h + lgrp + 2 * j]);
#pragma unroll
        for (int jlo = 0; jlo < 2; ++jlo) {
#pragma unroll
            for (int lq = 0; lq < 8; ++lq) {
                const int lp = 8 * lgrp + lq;
                const uint2 b = cv ? __ldg(xb + (2 * h + jlo) * 16 + lp) : make_uint2(0, 0);
                unsigned a0, a1;
                dequant(spread(ld16(sa + 64 * h + 32 * jlo + 2 * lp)), spread(ld16(sa + 128 + 32 * h + 2 * lp)), jlo,
                        ds[jlo], ds[jlo + 2], a0, a1);
                mma884(c, a0, a1, b.x, b.y);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {   // the four quadpairs' sums
        c[i] += __shfl_xor_sync(0xffffffffu, c[i], 4);
        c[i] += __shfl_xor_sync(0xffffffffu, c[i], 8);
    }
    if (ks > 1) {
        float* red = reinterpret_cast<float*>(smem);
        __syncthreads();
        if (warp > 0)
#pragma unroll
            for (int i = 0; i < 8; ++i) red[((warp - 1) * 8 + i) * 32 + lane] = c[i];
        __syncthreads();
        if (warp > 0) return;
        for (int k = 0; k < ks - 1; ++k)
#pragma unroll
            for (int i = 0; i < 8; ++i) c[i] += red[(k * 8 + i) * 32 + lane];
    }
    if (lane & 12) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {   // the m8n8k4 f32 accumulator layout
        const int r = (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >> 4);
        const int cc = 2 * ((lane >> 1) & 1) + (i & 1) + 4 * (i >> 2);
        if (cc < ncols && row0 + r < n_out) y[(size_t) cc * n_out + row0 + r] = c[i];
    }
}

bool cc7_device() noexcept {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    static int ok[64] = {};
    if (ok[dev] == 0) {
        int major = 0;
        const bool yes =
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major == 7;
        cudaGetLastError();
        ok[dev] = yes ? 1 : 2;
    }
    return ok[dev] == 1;
}

// warps per CTA by shape: more warps per row where the rows alone leave the card short of warps
int warps_for(int n_in, int n_out) {
    const int bpr = n_in / 256;
    const int ks = n_out >= 12288 || n_out <= 4096 ? 2 : 1;
    return std::min(ks, bpr);
}

void fail(const char* what, cudaError_t e) {
    basalt::log::error("basalt", "q6k_f16: %s: %s", what, cudaGetErrorString(e));
    std::exit(1);
}

}  // namespace

bool q6k_f16_supported(int t, int n_in, int n_out, int ncols, const void* w) noexcept {
    return t == 14 && ncols >= 1 && ncols <= 8 && n_in > 0 && n_in % 512 == 0 && n_out >= 4096 &&
           ((uintptr_t) w & 3) == 0 && cc7_device();
}

std::size_t q6k_f16_bytes(int n_in, int ncols) noexcept { return (std::size_t) n_in * ncols * 2; }

void q6k_f16_prep(const float* x, void* xh, int n_in, int ncols, void* stream) {
    const int n = ncols * (n_in / 4);
    launch_pdl(q6k_f16_prep_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, x,
               (uint2*) xh, n_in, ncols);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fail("prep", e);
}

void q6k_f16(const void* w, const void* xh, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!q6k_f16_supported(14, n_in, n_out, ncols, w) || ((uintptr_t) xh & 7) != 0) {
        basalt::log::error("basalt", "q6k_f16: n_in %d n_out %d ncols %d not supported", n_in, n_out, ncols);
        std::exit(1);
    }
    const int ks = warps_for(n_in, n_out);
    const size_t smem = std::max((size_t) ks * kRows * kWords * 4, (size_t) (ks - 1) * 8 * 32 * 4);
    launch_pdl(q6k_f16_kernel, dim3((unsigned) ((n_out + kRows - 1) / kRows)), dim3(32 * ks), smem,
               (cudaStream_t) stream, (const unsigned char*) w, (const uint2*) xh, y, n_in, n_out, ncols);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fail("launch", e);
}

}  // namespace basalt::kernels
