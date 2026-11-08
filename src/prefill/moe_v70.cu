// src/prefill/moe_v70.cu - prompt expert products on compute capability 7.0: the weights' k-step tiles dequantized to
// f16 in shared memory (block scale folded in), f16 activations, mma.m8n8k4 with f32 accumulation.  One launch covers
// a list of jobs (an expert and up to 64 consecutive sorted rows); a row's input and output share its index.
// gate|up writes silu(gate) * up as f16 H; down writes f32 rows.  Weights are read in ggml's layout (IQ3_XXS, IQ3_S, IQ2_S
// or IQ4_XS gate|up, IQ4_NL down).
#include "basalt/prefill/moe_v70.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <mutex>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"
#include "basalt/kernels/cc_launch.hpp"

namespace basalt::prefill::moe_v70 {
namespace {

constexpr int KC = 64;                     // k per step
constexpr int NTH = 128;                   // threads per block
constexpr int LD = KC + 8;                 // smem row stride in halves
constexpr int V8 = KC / 8;

__device__ __forceinline__ int kq_row() { return (threadIdx.x & 7) | ((threadIdx.x >> 4) << 3); }
__device__ __forceinline__ int kq_part() { return (threadIdx.x >> 3) & 1; }
__device__ __forceinline__ __half2 h2bits(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
__device__ __forceinline__ uint32_t bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ void store16(__half* dst, const uint32_t* v) {
    uint4* d = reinterpret_cast<uint4*>(dst);
    d[0] = make_uint4(v[0], v[1], v[2], v[3]);
    d[1] = make_uint4(v[4], v[5], v[6], v[7]);
    d[2] = make_uint4(v[8], v[9], v[10], v[11]);
    d[3] = make_uint4(v[12], v[13], v[14], v[15]);
}

// Weight tile loaders. A thread owns one 32-weight sub-block of one row per k-step (row kq_row() of the pass, columns
// 32 kq_part() of the step): fetch() reads its raw bytes into registers, put() writes 32 f16 values (block scale folded
// in) to shared memory. ew is the expert's base, wrow the weight row (< 0: past the matrix, zeros), M x K the matrix.

// 4 grid bytes (each < 256) and 4 sign bits (low nibble of t) -> 2 half2 words of +-db * byte
__device__ __forceinline__ void g4_put(uint32_t g, uint32_t t, __half2 db, uint32_t& o0, uint32_t& o1) {
    const __half2 k1024 = h2bits(0x64006400u);
    // sign bits to bit 7 of bytes 0..3, then to the half sign bits
    const uint32_t z = ((t & 0xF) * 0x10204080u) & 0x80808080u;
    const uint32_t x0 = __byte_perm(g, 0x64646464u, 0x4140), x1 = __byte_perm(g, 0x64646464u, 0x4342);
    const uint32_t a0 = bits(__hsub2(h2bits(x0), k1024)) ^ __byte_perm(z, 0, 0x1404);
    const uint32_t a1 = bits(__hsub2(h2bits(x1), k1024)) ^ __byte_perm(z, 0, 0x3424);
    o0 = bits(__hmul2(h2bits(a0), db));
    o1 = bits(__hmul2(h2bits(a1), db));
}
// 8 lookups of 4-byte grid entries: index byte j of q0|q1 plus bit j of h as bit 8, signs bits 4j..4j+3 of sg
__device__ __forceinline__ void grid8_put(uint32_t q0, uint32_t q1, uint32_t sg, uint32_t h, __half2 db, __half* dst,
                                          const uint32_t* grid) {
    uint32_t v[16];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t q = j < 4 ? q0 : q1;
        const uint32_t idx = ((q >> (8 * (j & 3))) & 0xFF) | (((h >> j) & 1) << 8);
        g4_put(grid[idx], sg >> (4 * j), db, v[2 * j], v[2 * j + 1]);
    }
    store16(dst + kq_row() * LD + kq_part() * 32, v);
}
// ggml dequantize_row_iq3_s, the scale d (1 + 2 s) rounded to f16
__device__ __forceinline__ void iq3s_put(uint32_t q0, uint32_t q1, uint32_t sg, uint32_t h, __half2 db, __half* dst,
                                         const uint32_t* grid) {
    grid8_put(q0, q1, sg, h, db, dst, grid);
}
__device__ __forceinline__ void zero_put(__half* dst) {
    uint32_t v[16] = {};
    store16(dst + kq_row() * LD + kq_part() * 32, v);
}

// IQ3_S as stored by ggml: 110-byte blocks of 256 (2-byte aligned), read with 9 u16 loads per sub-block
struct TileIQ3S {
    static constexpr int QK = 256;
    static constexpr int GRID_WORDS = 512;
    __device__ static void load_grid(uint32_t* g) {
        for (int i = threadIdx.x; i < 512; i += NTH) g[i] = iq3s_grid[i];
    }
    uint16_t d, qh, sc;
    uint32_t q0, q1, sg;
    int ib;
    bool live;
    __device__ __forceinline__ void fetch(const uint8_t* ew, int wrow, int, int K, int k0) {
        live = wrow >= 0;
        if (!live) return;
        ib = (k0 & 255) / 32 + kq_part();
        const uint16_t* p = reinterpret_cast<const uint16_t*>(ew + ((size_t) wrow * (K / 256) + (k0 >> 8)) * 110);
        d = p[0];
        q0 = (uint32_t) p[1 + 4 * ib] | ((uint32_t) p[2 + 4 * ib] << 16);
        q1 = (uint32_t) p[3 + 4 * ib] | ((uint32_t) p[4 + 4 * ib] << 16);
        qh = p[33 + (ib >> 1)];
        sg = (uint32_t) p[37 + 2 * ib] | ((uint32_t) p[38 + 2 * ib] << 16);
        sc = p[53 + (ib >> 2)];
    }
    __device__ __forceinline__ void put(__half* dst, const uint32_t* grid) const {
        if (!live) return zero_put(dst);
        const int nib = (sc >> (8 * ((ib >> 1) & 1) + 4 * (ib & 1))) & 15;
        const __half2 db = __float2half2_rn(__half2float(__ushort_as_half(d)) * (float) (1 + 2 * nib));
        iq3s_put(q0, q1, sg, qh >> (8 * (ib & 1)), db, dst, grid);
    }
};

// IQ3_S repacked (repack_iq3s): one 16-byte record per 32-weight sub-block, [row][K / 32]:
// bytes 0-7 qs, 8-11 signs, 12 qh, 13 zero, 14-15 f16 d (1 + 2 s)
struct TileIQ3SR {
    static constexpr int QK = 256;
    static constexpr int GRID_WORDS = 512;
    __device__ static void load_grid(uint32_t* g) {
        for (int i = threadIdx.x; i < 512; i += NTH) g[i] = iq3s_grid[i];
    }
    uint4 r;
    bool live;
    __device__ __forceinline__ void fetch(const uint8_t* ew, int wrow, int, int K, int k0) {
        live = wrow >= 0;
        if (live) r = __ldg(reinterpret_cast<const uint4*>(ew) + (size_t) wrow * (K / 32) + k0 / 32 + kq_part());
    }
    __device__ __forceinline__ void put(__half* dst, const uint32_t* grid) const {
        if (!live) return zero_put(dst);
        iq3s_put(r.x, r.y, r.z, r.w & 0xFF, __half2half2(__ushort_as_half((uint16_t) (r.w >> 16))), dst, grid);
    }
};

// prmt with the selector's sign-replicate bit honoured: byte i -> 0xFF or 0x00 by its msb
__device__ __forceinline__ uint32_t prmt_sign(uint32_t a) {
    uint32_t r;
    asm("prmt.b32 %0, %1, 0, 0xBA98;" : "=r"(r) : "r"(a));
    return r;
}
// 4 nibbles (the low nibbles of 4 bytes) -> kvalues_iq4nl + 128 as bytes
__device__ __forceinline__ uint32_t iq4nl_lut(uint32_t v) {
    constexpr uint32_t T0 = 0x3F2D1801u, T1 = 0x766A5D4Fu, T2 = 0xA6998D81u, T3 = 0xF1D9C5B5u;
    const uint32_t y = v | (v >> 4);
    const uint32_t sel = __byte_perm(y, 0, 0x0020) & 0x7777u;
    const uint32_t lo = __byte_perm(T0, T1, sel), hi = __byte_perm(T2, T3, sel);
    const uint32_t m = prmt_sign(v << 4);
    return (hi & m) | (lo & ~m);
}
// 16 quant bytes + f16 d -> 32 f16 values (low nibbles elements 0-15, high 16-31)
__device__ __forceinline__ void iq4nl_put(const uint32_t (&q)[4], __half2 dd, __half* dst) {
    uint32_t v[16];
    const __half2 k1152 = h2bits(0x64806480u);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const uint32_t lo = iq4nl_lut(q[w] & 0x0F0F0F0Fu), hi = iq4nl_lut((q[w] >> 4) & 0x0F0F0F0Fu);
        v[2 * w] = bits(__hmul2(__hsub2(h2bits(__byte_perm(lo, 0x64646464u, 0x4140)), k1152), dd));
        v[2 * w + 1] = bits(__hmul2(__hsub2(h2bits(__byte_perm(lo, 0x64646464u, 0x4342)), k1152), dd));
        v[8 + 2 * w] = bits(__hmul2(__hsub2(h2bits(__byte_perm(hi, 0x64646464u, 0x4140)), k1152), dd));
        v[8 + 2 * w + 1] = bits(__hmul2(__hsub2(h2bits(__byte_perm(hi, 0x64646464u, 0x4342)), k1152), dd));
    }
    store16(dst + kq_row() * LD + kq_part() * 32, v);
}

// IQ4_NL as stored by ggml: 18-byte blocks of 32 (2-byte aligned), read with 9 u16 loads
struct TileIQ4NL {
    static constexpr int QK = 32;
    static constexpr int GRID_WORDS = 0;
    __device__ static void load_grid(uint32_t*) {}
    uint16_t d;
    uint32_t q[4];
    bool live;
    __device__ __forceinline__ void fetch(const uint8_t* ew, int wrow, int, int K, int k0) {
        live = wrow >= 0;
        if (!live) return;
        const uint16_t* p = reinterpret_cast<const uint16_t*>(ew + ((size_t) wrow * (K / 32) + k0 / 32 + kq_part()) * 18);
        d = p[0];
#pragma unroll
        for (int w = 0; w < 4; ++w) q[w] = (uint32_t) p[1 + 2 * w] | ((uint32_t) p[2 + 2 * w] << 16);
    }
    __device__ __forceinline__ void put(__half* dst, const uint32_t*) const {
        if (!live) return zero_put(dst);
        iq4nl_put(q, __half2half2(__ushort_as_half(d)), dst);
    }
};

// IQ4_NL repacked (repack_iq4nl), same size: the 16 quant bytes of every block [row][K / 32] (16-byte aligned), then
// the f16 d of every block in the same order
struct TileIQ4NLR {
    static constexpr int QK = 32;
    static constexpr int GRID_WORDS = 0;
    __device__ static void load_grid(uint32_t*) {}
    uint16_t d;
    uint32_t q[4];
    bool live;
    __device__ __forceinline__ void fetch(const uint8_t* ew, int wrow, int M, int K, int k0) {
        live = wrow >= 0;
        if (!live) return;
        const size_t b = (size_t) wrow * (K / 32) + k0 / 32 + kq_part();
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(ew) + b);
        q[0] = v.x, q[1] = v.y, q[2] = v.z, q[3] = v.w;
        d = __ldg(reinterpret_cast<const unsigned short*>(ew + (size_t) M * (K / 32) * 16) + b);
    }
    __device__ __forceinline__ void put(__half* dst, const uint32_t*) const {
        if (!live) return zero_put(dst);
        iq4nl_put(q, __half2half2(__ushort_as_half(d)), dst);
    }
};

// a u32 at a 2-byte aligned address; the alignment is the same for every thread of a step (row and expert strides are
// multiples of 4), so the branch does not diverge
__device__ __forceinline__ uint32_t ld32_a2(const uint8_t* p) {
    if (reinterpret_cast<size_t>(p) & 2) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
        return (uint32_t) h[0] | ((uint32_t) h[1] << 16);
    }
    return *reinterpret_cast<const uint32_t*>(p);
}

// IQ3_XXS as stored by ggml: 98-byte blocks of 256 (2-byte aligned): d, qs[64] grid indices, then per 32-weight
// sub-block a u32 of 4 x 7-bit sign patterns (ksigns_iq2xs: the 7 bits plus their parity as bit 7) and a 4-bit scale
// in bits 28-31; db = d (0.5 + s) 0.5 rounded to f16
struct TileIQ3XXS {
    static constexpr int QK = 256;
    static constexpr int GRID_WORDS = 256;
    __device__ static void load_grid(uint32_t* g) {
        for (int i = threadIdx.x; i < 256; i += NTH) g[i] = iq3xxs_grid[i];
    }
    uint16_t d;
    uint32_t q0, q1, aux;
    bool live;
    __device__ __forceinline__ void fetch(const uint8_t* ew, int wrow, int, int K, int k0) {
        live = wrow >= 0;
        if (!live) return;
        const int ib = (k0 & 255) / 32 + kq_part();
        const uint8_t* p = ew + ((size_t) wrow * (K / 256) + (k0 >> 8)) * 98;
        d = *reinterpret_cast<const uint16_t*>(p);
        q0 = ld32_a2(p + 2 + 8 * ib);
        q1 = ld32_a2(p + 6 + 8 * ib);
        aux = ld32_a2(p + 66 + 4 * ib);
    }
    __device__ __forceinline__ void put(__half* dst, const uint32_t* grid) const {
        if (!live) return zero_put(dst);
        const __half2 db = __float2half2_rn(__half2float(__ushort_as_half(d)) * (0.5f + (float) (aux >> 28)) * 0.5f);
        uint32_t sg = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t s7 = (aux >> (7 * l)) & 127;
            sg |= (s7 | ((__popc(s7) & 1) << 7)) << (8 * l);
        }
        grid8_put(q0, q1, sg, 0, db, dst, grid);
    }
};

// IQ2_S as stored by ggml: 82-byte blocks of 256 (2-byte aligned): d, qs[32] grid index low bytes, signs[32], qh[8]
// (2 high index bits per group of 8), scales[8] (two 4-bit scales per 32-weight sub-block, one per 16 weights);
// db = d (0.5 + s) 0.25 rounded to f16. The grid's bytes are 8, 25 or 43: shared memory holds each 8-byte entry as 8
// nibbles (0, 1, 2), which a prmt over those three bytes turns back into the entry
struct TileIQ2S {
    static constexpr int QK = 256;
    static constexpr int GRID_WORDS = 1024;
    static constexpr uint32_t VALS = 0x002B1908u;
    __device__ static void load_grid(uint32_t* g) {
        for (int i = threadIdx.x; i < 1024; i += NTH) {
            uint32_t c = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t v = (uint32_t) (iq2s_grid[i] >> (8 * j)) & 0xFF;
                c |= (v == 8 ? 0u : v == 25 ? 1u : 2u) << (4 * j);
            }
            g[i] = c;
        }
    }
    uint16_t d, qh, sc;
    uint32_t qs, sg;
    int ib;
    bool live;
    __device__ __forceinline__ void fetch(const uint8_t* ew, int wrow, int, int K, int k0) {
        live = wrow >= 0;
        if (!live) return;
        ib = (k0 & 255) / 32 + kq_part();
        const uint8_t* p = ew + ((size_t) wrow * (K / 256) + (k0 >> 8)) * 82;
        const uint16_t* p16 = reinterpret_cast<const uint16_t*>(p);
        d = p16[0];
        qs = ld32_a2(p + 2 + 4 * ib);
        sg = ld32_a2(p + 34 + 4 * ib);
        qh = p16[33 + (ib >> 1)];
        sc = p16[37 + (ib >> 1)];
    }
    __device__ __forceinline__ void put(__half* dst, const uint32_t* grid) const {
        if (!live) return zero_put(dst);
        const float df = __half2float(__ushort_as_half(d));
        const uint32_t s = (sc >> (8 * (ib & 1))) & 0xFF, h = (qh >> (8 * (ib & 1))) & 0xFF;
        const __half2 db0 = __float2half2_rn(df * (0.5f + (float) (s & 15)) * 0.25f);
        const __half2 db1 = __float2half2_rn(df * (0.5f + (float) (s >> 4)) * 0.25f);
        uint32_t v[16];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx = ((qs >> (8 * l)) & 0xFF) | (((h >> (2 * l)) & 3) << 8);
            const uint32_t c = grid[idx];
            const uint32_t t = sg >> (8 * l);
            const __half2 db = l < 2 ? db0 : db1;
            g4_put(__byte_perm(VALS, 0, c & 0xFFFF), t, db, v[4 * l], v[4 * l + 1]);
            g4_put(__byte_perm(VALS, 0, c >> 16), t >> 4, db, v[4 * l + 2], v[4 * l + 3]);
        }
        store16(dst + kq_row() * LD + kq_part() * 32, v);
    }
};

// IQ4_XS as stored by ggml: 136-byte blocks of 256 (8-byte aligned): d, scales_h, scales_l[4], qs[128] (16 per
// 32-weight sub-block, IQ4_NL nibble order); dl = d (ls - 32) rounded to f16
struct TileIQ4XS {
    static constexpr int QK = 256;
    static constexpr int GRID_WORDS = 0;
    __device__ static void load_grid(uint32_t*) {}
    uint2 hd, qa, qb;
    int ib;
    bool live;
    __device__ __forceinline__ void fetch(const uint8_t* ew, int wrow, int, int K, int k0) {
        live = wrow >= 0;
        if (!live) return;
        ib = (k0 & 255) / 32 + kq_part();
        const uint2* p = reinterpret_cast<const uint2*>(ew + ((size_t) wrow * (K / 256) + (k0 >> 8)) * 136);
        hd = p[0];
        qa = p[1 + 2 * ib];
        qb = p[2 + 2 * ib];
    }
    __device__ __forceinline__ void put(__half* dst, const uint32_t*) const {
        if (!live) return zero_put(dst);
        const int ls = ((hd.y >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xF) | (((hd.x >> (16 + 2 * ib)) & 3) << 4);
        const __half2 dl = __float2half2_rn(__half2float(__ushort_as_half((uint16_t) (hd.x & 0xFFFF))) * (float) (ls - 32));
        const uint32_t q[4] = {qa.x, qa.y, qb.x, qb.y};
        iq4nl_put(q, dl, dst);
    }
};

// one thread per 32-weight sub-block: ggml IQ3_S rows (nrows x K) -> TileIQ3SR records
// grid (ceil(nsub / 256), experts); src and dst advance by sstride / nsub records per expert
__global__ void repack_iq3s(const uint8_t* __restrict__ src, size_t sstride, uint4* __restrict__ dst, size_t nsub) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= nsub) return;
    src += blockIdx.y * sstride;
    dst += blockIdx.y * nsub;
    const int ib = (int) (i % 8);
    const uint16_t* p = reinterpret_cast<const uint16_t*>(src + (i / 8) * 110);
    const uint32_t q0 = (uint32_t) p[1 + 4 * ib] | ((uint32_t) p[2 + 4 * ib] << 16);
    const uint32_t q1 = (uint32_t) p[3 + 4 * ib] | ((uint32_t) p[4 + 4 * ib] << 16);
    const uint32_t qh = (p[33 + (ib >> 1)] >> (8 * (ib & 1))) & 0xFF;
    const uint32_t sg = (uint32_t) p[37 + 2 * ib] | ((uint32_t) p[38 + 2 * ib] << 16);
    const int nib = (p[53 + (ib >> 2)] >> (8 * ((ib >> 1) & 1) + 4 * (ib & 1))) & 15;
    const uint16_t db = __half_as_ushort(__float2half_rn(__half2float(__ushort_as_half(p[0])) * (float) (1 + 2 * nib)));
    dst[i] = make_uint4(q0, q1, sg, qh | ((uint32_t) db << 16));
}
// one thread per block: ggml IQ4_NL (nblk blocks) -> TileIQ4NLR planes
// grid (ceil(nblk / 256), experts); an expert of nblk blocks is 18 nblk bytes in both layouts
__global__ void repack_iq4nl(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t nblk) {
    const size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    src += blockIdx.y * nblk * 18;
    dst += blockIdx.y * nblk * 18;
    const uint16_t* p = reinterpret_cast<const uint16_t*>(src + i * 18);
    uint32_t q[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) q[w] = (uint32_t) p[1 + 2 * w] | ((uint32_t) p[2 + 2 * w] << 16);
    reinterpret_cast<uint4*>(dst)[i] = make_uint4(q[0], q[1], q[2], q[3]);
    reinterpret_cast<uint16_t*>(dst + nblk * 16)[i] = p[0];
}

// the job's activation rows for one k-step (rows past n up to the token tiles are zeros)
template <int JN>
struct XTile {
    static constexpr int XV = JN * KC / 8 / NTH;
    uint4 v[XV];
    int y[XV];   // input row, -1 past the job's rows (zero), -2 past its token tiles (skipped)
    __device__ __forceinline__ void rows(const Job& jb, int nt) {
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            const int j = (threadIdx.x + i * NTH) / V8;
            y[i] = j < jb.n ? jb.r0 + j : j < nt * 16 ? -1 : -2;
        }
    }
    __device__ __forceinline__ void fetch(const __half* x, int ldx, int k0) {
        const int c = threadIdx.x % V8 * 8;
#pragma unroll
        for (int i = 0; i < XV; ++i) {
            v[i] = make_uint4(0, 0, 0, 0);
            if (y[i] >= 0) v[i] = __ldg(reinterpret_cast<const uint4*>(x + (size_t) y[i] * ldx + k0 + c));
        }
    }
    __device__ __forceinline__ void put(__half* sx) const {
        const int c = threadIdx.x % V8 * 8;
#pragma unroll
        for (int i = 0; i < XV; ++i)
            if (y[i] >= -1) *reinterpret_cast<uint4*>(sx + (threadIdx.x + i * NTH) / V8 * LD + c) = v[i];
    }
};

__device__ __forceinline__ void mma_884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// block geometry: 2 x 2 warps, each WM weight rows x WN tokens; DBUF double-buffers the smem tiles (one barrier a step)
template <int WM_, int WN_, bool DBUF_>
struct Cfg {
    static constexpr int WM = WM_, WN = WN_, BM = 2 * WM, JN = 2 * WN, RT = WM / 16, TT = WN / 16, NP = BM / 64;
    static constexpr bool DBUF = DBUF_;
    static constexpr int SLD = BM + 8;   // staging row stride in floats
    static constexpr int TILE = (BM + JN) * LD;   // halves per buffer
    static constexpr size_t SMEM_TILES = sizeof(__half) * TILE * (DBUF ? 2 : 1);
    // staging chunk in tokens: the whole job when it fits over the tiles, else half (a split job's LO = JN / 2 tokens)
    static constexpr int SCH = sizeof(float) * JN * SLD <= SMEM_TILES ? JN : JN / 2;
    static constexpr size_t SMEM_STAGE = sizeof(float) * SCH * SLD;
    static constexpr size_t SMEM = SMEM_TILES > SMEM_STAGE ? SMEM_TILES : SMEM_STAGE;
};

// MODE 0: out[r][col] = sum_k W_e[col][k] x[r][k], fp32, out row stride M.
// MODE 1 (gate|up, M = 2 FF): block b covers gate rows BM/2 b .. + BM/2 and the matching up rows (FF + same); writes
//   h[r][f] = silu(gate) * up as f16, row stride FF.
// W_e = wp[e] in Tile's layout, M x K (K a multiple of KC; of 256 for IQ3_S).
template <class Tile, int MODE, class C>
__global__ void __launch_bounds__(NTH) k_moe(const Job* __restrict__ jobs, const uint8_t* const* __restrict__ wp,
                                             int M, int K, const __half* __restrict__ x,
                                             float* __restrict__ out, __half* __restrict__ hout) {
    constexpr int WR = 2, WT = 2, BM = C::BM, WM = C::WM, RT = C::RT, TT = C::TT, NP = C::NP, LO = 16 * TT, SLD = C::SLD;
    const Job jb = jobs[blockIdx.y];
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* const sbase = reinterpret_cast<__half*>(smem_raw);
    __shared__ __align__(16) uint32_t grid[Tile::GRID_WORDS > 0 ? Tile::GRID_WORDS : 1];
    Tile::load_grid(grid);
    const int n = jb.n, nt = (n + 15) / 16, warp = threadIdx.x >> 5;
    const int rg = warp % WR, tg = warp / WR;
    const bool split = nt <= TT;
    const int tb = split ? 0 : tg * TT;
    const int lane = threadIdx.x & 31, q = (lane >> 2) & 3, rl = (lane & 3) + ((lane >> 4) & 1) * 4;
    const int aoff = (rg * WM + (q >> 1) * (WM / 2) + rl) * LD;
    const int sw4 = (q & 1) << 2;
    const int boff = (BM + tb * 16 + (q & 1) * 8 * TT + (rl ^ sw4)) * LD;
    float acc[RT][TT][8];
#pragma unroll
    for (int a = 0; a < RT; ++a)
#pragma unroll
        for (int b = 0; b < TT; ++b)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[a][b][j] = 0.f;

    // this thread's weight rows, one per pass of 64
    const uint8_t* const ew = wp[jb.e];
    int wr[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p) {
        const int tr = p * 64 + kq_row();
        int wrow;
        if (MODE == 1) {
            const int ff = M / 2, f0 = blockIdx.x * (BM / 2);
            wrow = tr < BM / 2 ? f0 + tr : ff + f0 + tr - BM / 2;
        } else {
            wrow = blockIdx.x * BM + tr;
        }
        wr[p] = wrow < M ? wrow : -1;
    }

    Tile wt[NP];
    XTile<C::JN> xt;
    xt.rows(jb, nt);
#pragma unroll
    for (int p = 0; p < NP; ++p) wt[p].fetch(ew, wr[p], M, K, 0);
    xt.fetch(x, K, 0);
    __syncthreads();   // grid table
    auto put = [&](__half* s) {
#pragma unroll
        for (int p = 0; p < NP; ++p) wt[p].put(s + p * 64 * LD, grid);
        xt.put(s + BM * LD);
    };
    auto fetch = [&](int k0) {
#pragma unroll
        for (int p = 0; p < NP; ++p) wt[p].fetch(ew, wr[p], M, K, k0);
        xt.fetch(x, K, k0);
    };
    auto mma_step = [&](const __half* s) {
        if (tb >= nt) return;
        const __half* arow = s + aoff;
        const __half* bcol = s + boff;
        const int kt0 = split ? tg : 0, kts = split ? WT : 1;
#pragma unroll
        for (int i = 0; i < KC / 16; ++i) {
            const int kt = kt0 + i * kts, kk = kt * 16;
            if (kt >= KC / 16) break;
            uint4 a[RT][2], b[TT][2];
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int h = 0; h < 2; ++h) a[r][h] = *reinterpret_cast<const uint4*>(arow + r * 8 * LD + kk + h * 8);
#pragma unroll
            for (int t = 0; t < TT; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) b[t][h] = *reinterpret_cast<const uint4*>(bcol + t * 8 * LD + kk + h * 8);
#pragma unroll
            for (int k4 = 0; k4 < 4; ++k4)
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int t = 0; t < TT; ++t) {
                        const uint4& A = a[r][k4 >> 1];
                        const uint4& B = b[t][k4 >> 1];
                        mma_884(acc[r][t], k4 & 1 ? A.z : A.x, k4 & 1 ? A.w : A.y, k4 & 1 ? B.z : B.x, k4 & 1 ? B.w : B.y);
                    }
        }
    };
    if (C::DBUF) {
        put(sbase);
        __syncthreads();
        int s = 0;
        for (int k0 = 0; k0 < K; k0 += KC, s ^= 1) {
            const bool more = k0 + KC < K;
            if (more) fetch(k0 + KC);
            mma_step(sbase + s * C::TILE);
            if (more) put(sbase + (s ^ 1) * C::TILE);
            __syncthreads();
        }
    } else {
        for (int k0 = 0; k0 < K; k0 += KC) {
            put(sbase);
            __syncthreads();
            if (k0 + KC < K) fetch(k0 + KC);
            mma_step(sbase);
            __syncthreads();
        }
    }
    // staging [token][row] (row stride SLD floats) in chunks of SCH tokens; a split job's two partial sums meet in it
    float* so = reinterpret_cast<float*>(smem_raw);
    constexpr int SCH = C::SCH;
    const int r0 = (lane & 1) + ((lane >> 4) & 1) * 4, cb = lane & 2, rbase = rg * WM + (q >> 1) * (WM / 2);
    auto stage = [&](int c0, int slot0, bool add) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int t = 0; t < TT; ++t)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int row = rbase + r * 8 + r0 + ((j >> 1) & 1) * 2;
                    const int slot = slot0 + (q & 1) * 8 * TT + t * 8 + ((cb + (j & 1) + ((j >> 2) & 1) * 4) ^ sw4) - c0;
                    if (slot >= 0 && slot < SCH) {
                        float* p = so + slot * SLD + row;
                        *p = add ? *p + acc[r][t][j] : acc[r][t][j];
                    }
                }
    };
    auto emit = [&](int j0, int j1) {
        if (MODE == 1) {
            const int ff = M / 2, f0 = blockIdx.x * (BM / 2);
            for (int e = threadIdx.x; e < (j1 - j0) * (BM / 2); e += NTH) {
                const int j = e / (BM / 2), i = e % (BM / 2);
                if (f0 + i >= ff) continue;
                const float g = so[j * SLD + i], u = so[j * SLD + BM / 2 + i];
                hout[(size_t) (jb.r0 + j0 + j) * ff + f0 + i] = __float2half_rn(silu(g) * u);
            }
        } else {
            const int row0 = blockIdx.x * BM;
            for (int e = threadIdx.x; e < (j1 - j0) * BM; e += NTH) {
                const int j = e / BM, r = e % BM, col = row0 + r;
                if (col >= M) continue;
                out[(size_t) (jb.r0 + j0 + j) * M + col] = so[j * SLD + r];
            }
        }
    };
    if (split) {
        if (tg == 1) stage(0, 0, false);
        __syncthreads();
        if (tg == 0) stage(0, 0, true);
        __syncthreads();
        emit(0, n);
    } else {
        for (int c0 = 0; c0 < n; c0 += SCH) {
            if (c0) __syncthreads();
            stage(c0, tb * 16, false);
            __syncthreads();
            emit(c0, min(n, c0 + SCH));
        }
    }
}

// 2 x 2 warps of 64 weight rows x 32 tokens: 128 x 64 per block
using C = Cfg<64, 32, false>;
static_assert(C::JN == kJobRows, "a job holds a block's tokens");

template <class Tile, int MODE>
void launch(const Job* jobs, int nj, const void* const* w, int M, int K, const uint16_t* x, float* out, uint16_t* h,
            cudaStream_t s) {
    if (nj <= 0) return;
    basalt::kernels::launch_cc(k_moe<Tile, MODE, C>, dim3((unsigned) ((M + C::BM - 1) / C::BM), (unsigned) nj), dim3(NTH),
                               C::SMEM, s, jobs, (const uint8_t* const*) w, M, K, (const __half*) x, out, (__half*) h);
}

}  // namespace

bool enabled() { return basalt::kernels::sm70_current(); }
bool usable(int gu_type, int d_type) {   // IQ3_XXS, IQ3_S, IQ2_S or IQ4_XS gate|up; IQ4_NL down
    return (gu_type == 18 || gu_type == 21 || gu_type == 22 || gu_type == 23) && d_type == 20;
}

namespace {
// a per-device device buffer of at least `bytes` (grows)
void* grow(int which, size_t bytes) {
    static std::mutex mu;
    static void* buf[2][64] = {};
    static size_t cap[2][64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    if (bytes > cap[which][dev]) {
        if (buf[which][dev]) cudaFree(buf[which][dev]);
        buf[which][dev] = nullptr;
        cap[which][dev] = 0;
        if (cudaMalloc(&buf[which][dev], bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        cap[which][dev] = bytes;
    }
    return buf[which][dev];
}
}  // namespace

Job* job_buffer(size_t n) { return (Job*) grow(0, n * sizeof(Job)); }
const void** ptr_buffer(size_t n) { return (const void**) grow(1, n * sizeof(void*)); }

void gu_swiglu(int gu_type, const Job* jobs, int nj, const void* const* w, int M, int K, const uint16_t* x, uint16_t* h,
               void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    switch (gu_type) {
        case 18: launch<TileIQ3XXS, 1>(jobs, nj, w, M, K, x, nullptr, h, s); break;
        case 21: launch<TileIQ3S, 1>(jobs, nj, w, M, K, x, nullptr, h, s); break;
        case 22: launch<TileIQ2S, 1>(jobs, nj, w, M, K, x, nullptr, h, s); break;
        case 23: launch<TileIQ4XS, 1>(jobs, nj, w, M, K, x, nullptr, h, s); break;
        default: break;
    }
}
void down(const Job* jobs, int nj, const void* const* w, int M, int K, const uint16_t* h, float* out, void* stream) {
    launch<TileIQ4NL, 0>(jobs, nj, w, M, K, h, out, nullptr, (cudaStream_t) stream);
}

}  // namespace basalt::prefill::moe_v70
