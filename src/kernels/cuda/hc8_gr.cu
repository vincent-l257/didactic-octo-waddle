// src/kernels/cuda/hc8_gr.cu - the hyper-connection read on the matrices' 8-bit copies (int8, one f16 scale per 32
// values of a row; hc_fused.hpp) for every caller that does not run hc_fused_read: fused_gr_read, fused_gr_read_multi
// and gr_read hand such matrices here.  Two kernels, with the partial sums between them in the caller's scratch:
//
//   down : per slice s of SL columns (half a stream c) and row k of [q_down; q_inject]: the dot of the row with
//          R' * w_norm, R' = R + bo_prev * 2 sigmoid(inj_prev / hc) (written to R_out when apply), and the slice's
//          sum of squares of R'
//   up   : rs[c] = rsqrt(mean(R'[c]^2) + eps); lo[k] = silu(sum_s rs * part[s][k] / hc), inject[c] likewise without
//          the silu; mixed[d] = mean_c xn[c,d] * sigmoid(q_up[c*n_embd + d] . lo), xn = R' * w_norm * rs; when asked,
//          mixed's q8_1 rows too (a CTA's 32 columns are one block), bitwise native_quantize_q8_1's
//
// rs is a per-slice factor, so the down dots leave it out.  Every token's values come from the same operations in the
// same order whatever the token count.
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/hc_fused.hpp"

#include "basalt/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"
#include "basalt/kernels/cc_launch.hpp"

namespace basalt::kernels {
namespace {

constexpr int MT = kFusedGrMaxT;
constexpr int GN = 2560;               // n_embd
constexpr int GHC = 4;                 // streams
constexpr int MAX_LR = 512;
constexpr int G8 = 32;
constexpr int SL = 1280;               // columns per down slice (half a stream)
constexpr int SPC = GN / SL;           // slices per stream
constexpr int NS = GN * GHC / SL;      // slices
constexpr int SLW = SL / 256;          // 8-column words per lane and row
constexpr int RG = 16;                 // rows per down CTA
constexpr int SLP = SL + SL / 8;       // a slice in shared memory: 4 floats of padding per 32 columns

struct Hc8Gr {
    const float* R[MT];
    float* R_out[MT];
    const float* bo[MT];
    const float* inj_prev[MT];
    float* lo[MT];
    float* rs[MT];
    float* inj_out[MT];
    float* mixed[MT];
    const float* w_norm;
    const int8_t* qd;
    const uint16_t* sd;
    const int8_t* qu;
    const uint16_t* su;
    const int8_t* qi;
    const uint16_t* si;
    int sd_row, su_row, si_row;
    int T, n, hc, lr, ninj;
    bool apply, q8;
    float eps;
    void* xq;                          // mixed's q8_1 rows when q8, token t's n / 32 blocks at block t * n / 32
};

__device__ __forceinline__ float s8(uint32_t w, int j) { return (float) (int) (int8_t) (w >> (8 * j)); }
__device__ __forceinline__ float f16f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// native_quantize_q8_1's operations as its fast-math build emits them (flush-to-zero, approximate division)
__device__ __forceinline__ float abs_ftz(float a) {
    float r;
    asm("abs.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}
__device__ __forceinline__ float max_ftz(float a, float b) {
    float r;
    asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float add_ftz(float a, float b) {
    float r;
    asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float div_approx_ftz(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

// one warp, lane i holding value i of the block
__device__ __forceinline__ void q8_1_block(float xi, Q81* __restrict__ y, int lane) {
    float amax = abs_ftz(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = max_ftz(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum = add_ftz(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    }
    const float d = q8_1_finite(div_approx_ftz(amax, 127.0f));
    int8_t q = 0;
    if (amax != 0.0f) {
        const float v = roundf(div_approx_ftz(xi, d));
        q = (int8_t) (v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
    }
    y->qs[lane] = q;
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

// The scratch: part[NS][nrows][NT] (the down dots), then ss[NS][NT] (the slices' sums of squares).
static_assert((NS * (MAX_LR + GHC) + NS) <= GN * GHC, "the scratch fits n_tok * hc * n_embd floats");

// grid (NS, ceil(nrows / RG)): slice s of the rows in row group rg, on R' (R and the pending write; R_out is written
// by the up kernel, since it may be R itself); row group 0 also leaves the slice's sum of squares
template <int NT>
__global__ void __launch_bounds__(256) hc8_down2_kernel(Hc8Gr p, float* scr) {
    // padded: the lanes' 8-column reads would otherwise meet two to a bank
    __shared__ __align__(16) float xs[NT * SLP];
    __shared__ float red[8][NT];
    constexpr int RPW = RG / 8;
    const int s = blockIdx.x, rg = blockIdx.y, D = GN * GHC, nrows = p.lr + p.ninj;
    const int col0 = s * SL, c = col0 / p.n, off = col0 - c * p.n;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // this warp's rows first: their loads are in flight while the slice is staged
    uint2 wv[RPW][SLW];
    float scl[RPW][SLW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int k = min(rg * RG + warp * RPW + r, nrows - 1);
        const int8_t* q = k < p.lr ? p.qd + (size_t) k * D : p.qi + (size_t) (k - p.lr) * D;
        const uint16_t* sc = k < p.lr ? p.sd + (size_t) k * p.sd_row : p.si + (size_t) (k - p.lr) * p.si_row;
#pragma unroll
        for (int i = 0; i < SLW; ++i) {
            const int cl = 8 * (lane + 32 * i);
            wv[r][i] = *reinterpret_cast<const uint2*>(q + col0 + cl);
            scl[r][i] = f16f(sc[(col0 + cl) / G8]);
        }
    }
    float ss[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float w = p.apply ? 2.0f / (1.0f + expf(-p.inj_prev[t][c] / (float) p.hc)) : 0.0f;
        ss[t] = 0.0f;
#pragma unroll
        for (int j = 0; j < (SL / 4 + 255) / 256; ++j) {
            const int i = 4 * ((int) threadIdx.x + 256 * j);
            if (i < SL) {
                float4 v = *reinterpret_cast<const float4*>(p.R[t] + col0 + i);
                if (p.apply) {
                    const float4 b = *reinterpret_cast<const float4*>(p.bo[t] + off + i);
                    v.x = fmaf(b.x, w, v.x); v.y = fmaf(b.y, w, v.y); v.z = fmaf(b.z, w, v.z); v.w = fmaf(b.w, w, v.w);
                }
                ss[t] = fmaf(v.x, v.x, ss[t]); ss[t] = fmaf(v.y, v.y, ss[t]);
                ss[t] = fmaf(v.z, v.z, ss[t]); ss[t] = fmaf(v.w, v.w, ss[t]);
                const float4 wn = *reinterpret_cast<const float4*>(p.w_norm + col0 + i);
                *reinterpret_cast<float4*>(xs + t * SLP + i + (i >> 5) * 4) = make_float4(v.x * wn.x, v.y * wn.y, v.z * wn.z, v.w * wn.w);
            }
        }
    }
    float* part = scr;
    float* ssp = scr + (size_t) NS * nrows * NT;
    if (rg == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = ss[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) red[warp][t] = v;
        }
    }
    __syncthreads();
    if (rg == 0 && threadIdx.x < NT) {
        float v = 0.0f;
        for (int k = 0; k < 8; ++k) v += red[k][threadIdx.x];
        ssp[s * NT + threadIdx.x] = v;
    }
    // the slice's activations are read once per lane and used for all of the warp's rows
    float acc[RPW][NT];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
#pragma unroll
    for (int i = 0; i < SLW; ++i) {
        const int cl = 8 * (lane + 32 * i);
        float a[RPW][8];
#pragma unroll
        for (int r = 0; r < RPW; ++r)
#pragma unroll
            for (int b = 0; b < 4; ++b) { a[r][b] = s8(wv[r][i].x, b); a[r][4 + b] = s8(wv[r][i].y, b); }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 x0 = *reinterpret_cast<const float4*>(xs + t * SLP + cl + (cl >> 5) * 4);
            const float4 x1 = *reinterpret_cast<const float4*>(xs + t * SLP + cl + (cl >> 5) * 4 + 4);
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                float d = a[r][0] * x0.x;
                d = fmaf(a[r][1], x0.y, d); d = fmaf(a[r][2], x0.z, d); d = fmaf(a[r][3], x0.w, d);
                d = fmaf(a[r][4], x1.x, d); d = fmaf(a[r][5], x1.y, d); d = fmaf(a[r][6], x1.z, d); d = fmaf(a[r][7], x1.w, d);
                acc[r][t] = fmaf(scl[r][i], d, acc[r][t]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const int k = rg * RG + warp * RPW + r;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && k < nrows) part[((size_t) s * nrows + k) * NT + t] = v;
        }
    }
}

// Below cc 12 the up kernel takes the whole register file (its grid is about one CTA per SM): the default budget
// spills.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#define HC8_UP_BOUNDS __launch_bounds__(256, 1)
#else
#define HC8_UP_BOUNDS __launch_bounds__(256)
#endif

// grid n / 32: thread (d, c, h) takes half of row c*n + d (16-byte chunks h, h+2, ...) and writes R'[c, d] to R_out.
// Every CTA first finishes rs, lo and the injection from the down slices; block 0 also writes them out.  Q8: the
// CTA's 32 columns of mixed also as one q8_1 block per token.
template <int NT, bool Q8>
__global__ void HC8_UP_BOUNDS hc8_up2_kernel(Hc8Gr p, const float* scr) {
    __shared__ __align__(16) float slo[NT * MAX_LR];
    __shared__ float srs[NT][GHC];
    __shared__ float smix[Q8 ? NT : 1][32];
    const int nrows = p.lr + p.ninj, lr = p.lr;
    const int h = threadIdx.x & 1, c = (threadIdx.x >> 1) & 3, d = blockIdx.x * 32 + (threadIdx.x >> 3);
    const size_t row = (size_t) c * p.n + d;
    const uint4* qr = reinterpret_cast<const uint4*>(p.qu + row * lr);
    const uint16_t* sr = p.su + row * p.su_row;
    const int nch = lr / 16;
    // this thread's half row and its epilogue inputs first: their loads are in flight while lo is finished
    constexpr int NJ = MAX_LR / 32;
    uint4 wv[NJ];
    float scl[NJ];
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        const int j = 2 * jj + h;
        if (j < nch) { wv[jj] = qr[j]; scl[jj] = f16f(sr[j >> 1]); }
    }
    const float wn = p.w_norm[row];
    float xr[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        xr[t] = p.R[t][row];
        if (p.apply) {
            const float w = 2.0f / (1.0f + expf(-p.inj_prev[t][c] / (float) p.hc));
            xr[t] = fmaf(p.bo[t][d], w, xr[t]);
            if (h == 0) p.R_out[t][row] = xr[t];
        }
    }
    const float* part = scr;
    const float* ssp = scr + (size_t) NS * nrows * NT;
    if ((int) threadIdx.x < NT * 4) {
        const int t = threadIdx.x / 4, cc = threadIdx.x % 4;
        float ss = 0.0f;
        for (int j = 0; j < SPC; ++j) ss += ssp[(cc * SPC + j) * NT + t];
        const float rs = rsqrtf(ss / (float) p.n + p.eps);
        srs[t][cc] = rs;
        if (blockIdx.x == 0) p.rs[t][cc] = rs;
    }
    __syncthreads();
#pragma unroll 4
    for (int i = threadIdx.x; i < nrows * NT; i += blockDim.x) {
        const int k = i / NT, t = i % NT;
        float pv[NS];
#pragma unroll
        for (int s = 0; s < NS; ++s) pv[s] = part[((size_t) s * nrows + k) * NT + t];
        float v = 0.0f;
#pragma unroll
        for (int s = 0; s < NS; ++s) v = fmaf(srs[t][s / SPC], pv[s], v);
        if (k < lr) {
            const float x = v / (float) p.hc, lo = x / (1.0f + expf(-x));
            slo[t * lr + k] = lo;
            if (blockIdx.x == 0) p.lo[t][k] = lo;
        } else if (blockIdx.x == 0) {
            p.inj_out[t][k - lr] = v;
        }
    }
    __syncthreads();
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
        const int j = 2 * jj + h;
        if (j < nch) {
            float a[16];
            const uint32_t ws[4] = {wv[jj].x, wv[jj].y, wv[jj].z, wv[jj].w};
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 4; ++b) a[4 * u + b] = s8(ws[u], b);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4* l = reinterpret_cast<const float4*>(slo + t * lr + 16 * j);
                float dd = 0.0f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float4 x = l[u];
                    dd = fmaf(a[4 * u], x.x, dd); dd = fmaf(a[4 * u + 1], x.y, dd);
                    dd = fmaf(a[4 * u + 2], x.z, dd); dd = fmaf(a[4 * u + 3], x.w, dd);
                }
                acc[t] = fmaf(scl[jj], dd, acc[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float a = acc[t] + __shfl_xor_sync(0xffffffffu, acc[t], 1);
        const float xn = xr[t] * wn * srs[t][c];
        float m = xn * (1.0f / (1.0f + expf(-a)));
        m += __shfl_xor_sync(0xffffffffu, m, 2);
        m += __shfl_xor_sync(0xffffffffu, m, 4);
        if (h == 0 && c == 0) {
            p.mixed[t][d] = m / (float) p.hc;
            if (Q8) smix[t][threadIdx.x >> 3] = m / (float) p.hc;
        }
    }
    if (Q8) {   // warp t quantizes token t's block
        __syncthreads();
        const int w = threadIdx.x >> 5;
        if (w < NT)
            q8_1_block(smix[w][threadIdx.x & 31], static_cast<Q81*>(p.xq) + (size_t) w * (p.n / 32) + blockIdx.x,
                       threadIdx.x & 31);
    }
}

}  // namespace

namespace {
template <int NT> void launch_nt(const Hc8Gr& p, float* scr, cudaStream_t st) {
    const dim3 g((unsigned) NS, (unsigned) ((p.lr + p.ninj + RG - 1) / RG));
    basalt::kernels::launch_cc(hc8_down2_kernel<NT>, g, 256, 0, st, p, scr);
    if (p.q8)
        basalt::kernels::launch_cc(hc8_up2_kernel<NT, true>, (unsigned) (GN / 32), 256, 0, st, p, (const float*) scr);
    else
        basalt::kernels::launch_cc(hc8_up2_kernel<NT, false>, (unsigned) (GN / 32), 256, 0, st, p, (const float*) scr);
}
}  // namespace

bool hc8_gr_read_multi(const FusedGrArgs* a, int n_tok, int64_t n_embd, int64_t hc, float* scratch, void* stream) {
    if (!hc8_enabled() || n_tok < 1 || a == nullptr) return false;
    Hc8View vd, vu, vi;
    if (!hc8_view(a[0].w_down, vd)) return false;
    const bool inj = a[0].w_inject != nullptr;
    const int64_t D = n_embd * hc;
    const bool have = hc8_view(a[0].w_up, vu) && (!inj || hc8_view(a[0].w_inject, vi));
    if (!have) {
        if (vd.owned || vu.owned) {
            basalt::log::error("basalt", "hc8_gr_read: a read's matrices are not all 8-bit copies");
            std::exit(1);
        }
        return false;
    }
    if (n_tok > MT || n_embd != GN || hc != GHC || vd.cols != D || vu.rows != D || vu.cols != vd.rows ||
        vd.rows > MAX_LR || vd.rows % 16 != 0 || (inj && (vi.rows != hc || vi.cols != D))) {
        basalt::log::error("basalt", "hc8_gr_read: unsupported geometry (%d tokens, n_embd %lld, hc %lld, lr %d)", n_tok,
                     (long long) n_embd, (long long) hc, vd.rows);
        std::exit(1);
    }
    Hc8Gr p{};
    p.q8 = a[0].xq != nullptr;
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        const void* xq = p.q8 ? static_cast<char*>(a[0].xq) + (size_t) t * (n_embd / 32) * sizeof(Q81) : nullptr;
        if (x.xq != xq || !x.R || !x.w_norm || !x.lo || !x.rs || !x.mixed || (inj && !x.inject_out) ||
            x.apply != a[0].apply ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            basalt::log::error("basalt", "hc8_gr_read: invalid arguments for token %d", t);
            std::exit(1);
        }
        p.R[t] = x.R; p.R_out[t] = x.R_out; p.bo[t] = x.bo_prev; p.inj_prev[t] = x.inj_prev;
        p.lo[t] = x.lo; p.rs[t] = x.rs; p.inj_out[t] = x.inject_out; p.mixed[t] = x.mixed;
    }
    p.w_norm = a[0].w_norm;
    p.qd = vd.q; p.sd = vd.s; p.sd_row = vd.srow;
    p.qu = vu.q; p.su = vu.s; p.su_row = vu.srow;
    if (inj) { p.qi = vi.q; p.si = vi.s; p.si_row = vi.srow; }
    p.xq = a[0].xq;
    p.T = n_tok; p.n = (int) n_embd; p.hc = (int) hc; p.lr = vd.rows; p.ninj = inj ? (int) hc : 0;
    p.apply = a[0].apply; p.eps = a[0].eps;
    if (scratch == nullptr) {
        basalt::log::error("basalt", "hc8_gr_read: no scratch");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    switch (n_tok) {
        case 1: launch_nt<1>(p, scratch, st); break;
        case 2: launch_nt<2>(p, scratch, st); break;
        case 3: launch_nt<3>(p, scratch, st); break;
        case 4: launch_nt<4>(p, scratch, st); break;
        case 5: launch_nt<5>(p, scratch, st); break;
        case 6: launch_nt<6>(p, scratch, st); break;
        case 7: launch_nt<7>(p, scratch, st); break;
        default: launch_nt<8>(p, scratch, st); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "hc8_gr_read: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace basalt::kernels
