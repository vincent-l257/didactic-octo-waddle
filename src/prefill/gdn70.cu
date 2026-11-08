// src/prefill/gdn70.cu - the prompt path's gated delta rule for compute capability 7.0, in FP32 SIMT.
// Same interface and state layout as basalt::prefill::gdn_recurrence (src/prefill/gdn_chunked.cu):
//   h [T][10240] FP32 (q: qh*128, k: 2048 + qh*128, v: 4096 + head*128), gate (log decay) and beta [T][48],
//   state [key 128][value head 48][value 128] FP32 in/out, y [T][48][128] FP32 (scaled by 1/sqrt(128)),
//   y16 the gated RMS norm in FP16; value head `head` reads key head head % 16.
// gdn70_prep (sub-chunk, key head): L2-normalized FP32 q/k packet [key][k rows | q rows] and, for the key head's three
// value heads, solve = (I + L)^-1 diag(beta), mqk, prefix exp(G_i) and suffix exp(G_last - G_i).
// gdn70_rec (value head, 32-column stripe; heads 40..47 in 16-column stripes): the FP32 state in registers (owner tiles)
// and shared memory (operand copy),
// W = K S and Y = Q S, R = V - diag(prefix) W, U = solve R, out = diag(prefix) Y + mqk U, S = decay S + K^T diag(suffix) U.
#include "basalt/prefill/gdn70.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "basalt/kernels/cc_launch.hpp"
#include "basalt/log.hpp"

namespace basalt::prefill::gdn70 {
namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int CH = 32;                          // tokens per sub-chunk
constexpr int NTH = 256;                        // threads of gdn70_prep
#ifndef GDN70_RT
#define GDN70_RT 256
#endif
constexpr int RT = GDN70_RT;                    // threads of gdn70_rec
constexpr int PKT = S * 2 * CH;                 // floats of a q/k packet: [key][k_0..k_31, q_0..q_31]
constexpr int CHP = CH + 4;                     // row stride of solve and mqk (rows of a warp in distinct banks)
constexpr int CTL = 2 * CH * CHP + 2 * CH;      // floats of a control block: solve, mqk, prefix, suffix
constexpr int KQP = 2 * CH + 4;                 // shared row stride of the packet
static_assert(CTL % 4 == 0 && PKT % (4 * NTH) == 0, "float4 shares");

__device__ __forceinline__ float4 f4zero() { return make_float4(0.f, 0.f, 0.f, 0.f); }

// Block (sub-chunk, key head).
__global__ void __launch_bounds__(NTH) gdn70_prep(const float* __restrict__ h, const float* __restrict__ gate,
                                                  const float* __restrict__ beta, float eps, int T,
                                                  float* __restrict__ kq_out, float* __restrict__ ctl_out) {
    __shared__ __align__(16) float buf[2 * CH * (S + 1)];   // q, k rows; then A and T of the three value heads
    __shared__ float kk[CH][CH + 1], qk[CH][CH + 1];
    __shared__ float gs[3][CH], bs[3][CH];
    float(*qs)[S + 1] = reinterpret_cast<float(*)[S + 1]>(buf);
    float(*ks)[S + 1] = reinterpret_cast<float(*)[S + 1]>(buf + CH * (S + 1));
    const int sub = blockIdx.x, qh = blockIdx.y, nsub = gridDim.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int t0 = sub * CH;
    // warp w: rows 4w..4w+3, lane: coordinates 4 lane..4 lane+3
#pragma unroll
    for (int x = 0; x < 4; ++x) {
        const int i = warp * 4 + x, t = t0 + i;
        float4 q4 = f4zero(), k4 = f4zero();
        if (t < T) {
            const float* hr = h + (int64_t) t * C + qh * S + lane * 4;
            q4 = *reinterpret_cast<const float4*>(hr);
            k4 = *reinterpret_cast<const float4*>(hr + HK * S);
        }
        float sq = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
        float sk = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
            sk += __shfl_xor_sync(0xffffffffu, sk, o);
        }
        const float qi = rsqrtf(sq + eps), ki = rsqrtf(sk + eps);
        qs[i][lane * 4 + 0] = q4.x * qi; qs[i][lane * 4 + 1] = q4.y * qi;
        qs[i][lane * 4 + 2] = q4.z * qi; qs[i][lane * 4 + 3] = q4.w * qi;
        ks[i][lane * 4 + 0] = k4.x * ki; ks[i][lane * 4 + 1] = k4.y * ki;
        ks[i][lane * 4 + 2] = k4.z * ki; ks[i][lane * 4 + 3] = k4.w * ki;
    }
    if (warp < 3) {   // the cumulative log decay and beta of value head qh + 16 warp; rows past T: g = 0, beta = 0
        const int vh = qh + HK * warp, t = t0 + lane;
        float g = t < T ? gate[(int64_t) t * HV + vh] : 0.f;
        const float b = t < T ? beta[(int64_t) t * HV + vh] : 0.f;
#pragma unroll
        for (int o = 1; o < CH; o <<= 1) {
            const float p = __shfl_up_sync(0xffffffffu, g, o);
            if (lane >= o) g += p;
        }
        gs[warp][lane] = g;
        bs[warp][lane] = b;
    }
    __syncthreads();
    float* pk = kq_out + ((int64_t) qh * nsub + sub) * PKT;
    for (int e = tid; e < PKT; e += NTH) {
        const int r = e / (2 * CH), j = e % (2 * CH);
        pk[e] = j < CH ? ks[j][r] : qs[j - CH][r];
    }
    {   // k_i . k_j and q_i . k_j, a 2 x 2 corner of both per thread
        const int i0 = (tid / (CH / 2)) * 2, j0 = (tid % (CH / 2)) * 2;
        float k00 = 0.f, k01 = 0.f, k10 = 0.f, k11 = 0.f, q00 = 0.f, q01 = 0.f, q10 = 0.f, q11 = 0.f;
#pragma unroll 8
        for (int r = 0; r < S; ++r) {
            const float a0 = ks[i0][r], a1 = ks[i0 + 1][r], b0 = qs[i0][r], b1 = qs[i0 + 1][r];
            const float c0 = ks[j0][r], c1 = ks[j0 + 1][r];
            k00 = fmaf(a0, c0, k00); k01 = fmaf(a0, c1, k01); k10 = fmaf(a1, c0, k10); k11 = fmaf(a1, c1, k11);
            q00 = fmaf(b0, c0, q00); q01 = fmaf(b0, c1, q01); q10 = fmaf(b1, c0, q10); q11 = fmaf(b1, c1, q11);
        }
        kk[i0][j0] = k00; kk[i0][j0 + 1] = k01; kk[i0 + 1][j0] = k10; kk[i0 + 1][j0 + 1] = k11;
        qk[i0][j0] = q00; qk[i0][j0 + 1] = q01; qk[i0 + 1][j0] = q10; qk[i0 + 1][j0 + 1] = q11;
    }
    __syncthreads();
    // the q/k rows are dead: buf holds A of the three value heads
    float(*A)[CH][CH + 1] = reinterpret_cast<float(*)[CH][CH + 1]>(buf);
    for (int e = tid; e < 3 * CH * CH; e += NTH) {
        const int m = e / (CH * CH), i = (e / CH) % CH, j = e % CH;
        // no exponential above the diagonal: the gate sums can be large negatives
        const float d = i >= j ? expf(gs[m][i] - gs[m][j]) : 0.f;
        A[m][i][j] = i > j ? bs[m][i] * d * kk[i][j] : 0.f;
        float* ctl = ctl_out + ((int64_t) (qh + HK * m) * nsub + sub) * CTL;
        ctl[CH * CHP + i * CHP + j] = d * qk[i][j];
    }
    if (tid < 3 * CH) {
        const int m = tid / CH, i = tid % CH;
        float* ctl = ctl_out + ((int64_t) (qh + HK * m) * nsub + sub) * CTL;
        ctl[2 * CH * CHP + i] = expf(gs[m][i]);
        ctl[2 * CH * CHP + CH + i] = expf(gs[m][CH - 1] - gs[m][i]);
    }
    __syncthreads();
    // solve = (I + A)^-1 diag(beta) by forward substitution (the column scale passes through it), warp m for value
    // head m, lane c holding column c in registers
    if (warp < 3) {
        const int m = warp, c = lane;
        float* ctl = ctl_out + ((int64_t) (qh + HK * m) * nsub + sub) * CTL;
        const float bc = bs[m][c];
        float col[CH];
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            float v0 = i == c ? bc : 0.f, v1 = 0.f;
#pragma unroll
            for (int j = 0; j < i; ++j) {
                if (j & 1) v1 = fmaf(-A[m][i][j], col[j], v1);
                else v0 = fmaf(-A[m][i][j], col[j], v0);
            }
            col[i] = v0 + v1;
            ctl[i * CHP + c] = col[i];
        }
    }
}

template <int N> __device__ __forceinline__ void ldv(const float* p, float (&v)[N]) {
    if constexpr (N == 1) {
        v[0] = *p;
    } else if constexpr (N == 8) {
        const float4 x = *reinterpret_cast<const float4*>(p), w = *reinterpret_cast<const float4*>(p + 4);
        v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w; v[4] = w.x; v[5] = w.y; v[6] = w.z; v[7] = w.w;
    } else if constexpr (N == 4) {
        const float4 x = *reinterpret_cast<const float4*>(p);
        v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
    } else {
        static_assert(N == 2);
        const float2 x = *reinterpret_cast<const float2*>(p);
        v[0] = x.x; v[1] = x.y;
    }
}
template <int N> __device__ __forceinline__ void stv(float* p, const float (&v)[N]) {
    if constexpr (N == 1) *p = v[0];
    else if constexpr (N == 8) {
        *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
    else if constexpr (N == 4) *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    else *reinterpret_cast<float2*>(p) = make_float2(v[0], v[1]);
}

template <int CB> struct RecShared {
    float kq[S][KQP];       // the packet: [key][k rows 0..31 | q rows 0..31]
    float snap[S][CB];      // the state at the sub-chunk's start
    float part[2 * CH][CB]; // W / Y partial sums, then R (rows 0..31) and Y (rows 32..63)
    float v[CH][CB];
    float u[CH][CB];
    float ud[CH][CB];       // diag(suffix) U
    float ctl[CTL];
};

// One (value head, CB-column stripe): RG = RT / 32 column groups of CW = CB / RG columns.
template <int CB>
__device__ __forceinline__ void rec_stripe(int head, int c0, float* __restrict__ state, const float* __restrict__ h,
                                           const float* __restrict__ kqp, const float* __restrict__ ctlp,
                                           float* __restrict__ y, int T, RecShared<CB>& sm) {
    constexpr int RG = RT / 32, CW = CB / RG;
    const int tid = threadIdx.x, qh = head % HK;
    const int nsub = (T + CH - 1) / CH;
    const size_t rs = (size_t) HV * S;
    const float scale = rsqrtf((float) S);
    // owner tile of the state: keys r0 + 32 a (a < 4), columns cb..cb+CW-1; also the row / columns of the small products
    const int r0 = tid / RG, cb = (tid % RG) * CW, ri = tid / RG;
    float st[4][CW];
#pragma unroll
    for (int a = 0; a < 4; ++a) ldv(state + (r0 + 32 * a) * rs + head * S + c0 + cb, st[a]);
    float4 pk[PKT / 4 / RT], pc[(CTL / 4 + RT - 1) / RT];
    float pv[CW];
    auto fetch = [&](int sub) {
        const float4* src = reinterpret_cast<const float4*>(kqp + ((int64_t) qh * nsub + sub) * PKT);
#pragma unroll
        for (int j = 0; j < PKT / 4 / RT; ++j) pk[j] = __ldg(src + tid + j * RT);
        const float4* cs = reinterpret_cast<const float4*>(ctlp + ((int64_t) head * nsub + sub) * CTL);
#pragma unroll
        for (int j = 0; j < (CTL / 4 + RT - 1) / RT; ++j) {
            const int e = tid + j * RT;
            pc[j] = e < CTL / 4 ? __ldg(cs + e) : f4zero();
        }
        const int t = sub * CH + ri;
        if (t < T) ldv(h + (int64_t) t * C + 2 * HK * S + head * S + c0 + cb, pv);
        else
#pragma unroll
            for (int z = 0; z < CW; ++z) pv[z] = 0.f;
    };
    fetch(0);
#pragma unroll
    for (int a = 0; a < 4; ++a) stv(&sm.snap[r0 + 32 * a][cb], st[a]);
    const float* solve = sm.ctl;
    const float* mqk = sm.ctl + CH * CHP;
    const float* prefix = sm.ctl + 2 * CH * CHP;
    const float* suffix = prefix + CH;
    for (int sub = 0; sub < nsub; ++sub) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PKT / 4 / RT; ++j) {
            const int e = tid + j * RT;
            *reinterpret_cast<float4*>(&sm.kq[e / (2 * CH / 4)][(e % (2 * CH / 4)) * 4]) = pk[j];
        }
#pragma unroll
        for (int j = 0; j < (CTL / 4 + RT - 1) / RT; ++j) {
            const int e = tid + j * RT;
            if (e < CTL / 4) reinterpret_cast<float4*>(sm.ctl)[e] = pc[j];
        }
        stv(&sm.v[ri][cb], pv);
        __syncthreads();
        if (sub + 1 < nsub) fetch(sub + 1);
        {   // W = K S (rows 0..31), Y = Q S (rows 32..63): 4 x CW tiles, the keys split in two halves
            const int half = tid / (RT / 2), t7 = tid % (RT / 2), a4 = (t7 / RG) * 4, b = (t7 % RG) * CW;
            float acc[4][CW] = {};
#pragma unroll 8
            for (int r = half * (S / 2); r < (half + 1) * (S / 2); ++r) {
                const float4 k4 = *reinterpret_cast<const float4*>(&sm.kq[r][a4]);
                float sv[CW];
                ldv(&sm.snap[r][b], sv);
                const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int z = 0; z < CW; ++z) acc[x][z] = fmaf(kv[x], sv[z], acc[x][z]);
            }
            if (half) {
#pragma unroll
                for (int x = 0; x < 4; ++x) stv(&sm.part[a4 + x][b], acc[x]);
            }
            __syncthreads();
            if (!half) {
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const int row = a4 + x;
                    float p[CW], w[CW];
                    ldv(&sm.part[row][b], p);
#pragma unroll
                    for (int z = 0; z < CW; ++z) w[z] = acc[x][z] + p[z];
                    if (row < CH) {   // R = V - diag(prefix) W
                        float v[CW];
                        ldv(&sm.v[row][b], v);
                        const float g = prefix[row];
#pragma unroll
                        for (int z = 0; z < CW; ++z) w[z] = fmaf(-g, w[z], v[z]);
                    }
                    stv(&sm.part[row][b], w);
                }
            }
            __syncthreads();
        }
        const int mend = (ri | (32 / RG - 1)) + 1;   // the warp's last row + 1: solve and mqk are zero above the diagonal
        {   // U = solve R
            float u[CW] = {};
#pragma unroll 4
            for (int m = 0; m < mend; ++m) {
                const float s = solve[ri * CHP + m];
                float r[CW];
                ldv(&sm.part[m][cb], r);
#pragma unroll
                for (int z = 0; z < CW; ++z) u[z] = fmaf(s, r[z], u[z]);
            }
            stv(&sm.u[ri][cb], u);
            const float d = suffix[ri];
#pragma unroll
            for (int z = 0; z < CW; ++z) u[z] *= d;
            stv(&sm.ud[ri][cb], u);
        }
        __syncthreads();
        {   // out = diag(prefix) Y + mqk U
            const float g = prefix[ri];
            float o[CW];
            ldv(&sm.part[CH + ri][cb], o);
#pragma unroll
            for (int z = 0; z < CW; ++z) o[z] *= g;
#pragma unroll 4
            for (int m = 0; m < mend; ++m) {
                const float s = mqk[ri * CHP + m];
                float u[CW];
                ldv(&sm.u[m][cb], u);
#pragma unroll
                for (int z = 0; z < CW; ++z) o[z] = fmaf(s, u[z], o[z]);
            }
            const int t = sub * CH + ri;
#pragma unroll
            for (int z = 0; z < CW; ++z) o[z] *= scale;
            if (t < T) stv(y + ((int64_t) t * HV + head) * S + c0 + cb, o);
        }
        {   // S = decay S + K^T diag(suffix) U
            const float decay = prefix[CH - 1];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int z = 0; z < CW; ++z) st[a][z] *= decay;
#pragma unroll
            for (int i = 0; i < CH; i += 4) {
                float kv[4][4], uv[4][CW];
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const float4 k4 = *reinterpret_cast<const float4*>(&sm.kq[r0 + 32 * a][i]);
                    kv[a][0] = k4.x; kv[a][1] = k4.y; kv[a][2] = k4.z; kv[a][3] = k4.w;
                    ldv(&sm.ud[i + a][cb], uv[a]);
                }
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int a = 0; a < 4; ++a)
#pragma unroll
                        for (int z = 0; z < CW; ++z) st[a][z] = fmaf(kv[a][x], uv[x][z], st[a][z]);
            }
#pragma unroll
            for (int a = 0; a < 4; ++a) stv(&sm.snap[r0 + 32 * a][cb], st[a]);
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a) stv(state + (r0 + 32 * a) * rs + head * S + c0 + cb, st[a]);
}

// Value heads [0, H32) in 32-column stripes, then [H32, 48) in 16-column stripes: the narrow stripes fill the last
// partial wave.
constexpr int H32 = 40;
constexpr int REC_BLOCKS = H32 * (S / 32) + (HV - H32) * (S / 16);
constexpr size_t REC_SMEM = sizeof(RecShared<32>);

__global__ void __launch_bounds__(RT, 1) gdn70_rec(float* __restrict__ state, const float* __restrict__ h,
                                                    const float* __restrict__ kqp, const float* __restrict__ ctlp,
                                                    float* __restrict__ y, int T) {
    extern __shared__ __align__(16) unsigned char backing[];
    const int b = blockIdx.x;
    if (b < H32 * (S / 32))
        rec_stripe<32>(b / (S / 32), (b % (S / 32)) * 32, state, h, kqp, ctlp, y, T,
                       *reinterpret_cast<RecShared<32>*>(backing));
    else
        rec_stripe<16>(H32 + (b - H32 * (S / 32)) / (S / 16), ((b - H32 * (S / 32)) % (S / 16)) * 16, state, h, kqp,
                       ctlp, y, T, *reinterpret_cast<RecShared<16>*>(backing));
}

// the output norm over a head's 128 columns, into the FP16 copy the out projection reads (Basalt's, unchanged)
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__global__ void __launch_bounds__(S) gdn70_out_norm(const float* __restrict__ z, const float* __restrict__ gamma, float eps,
                                                    const float* __restrict__ y, uint16_t* __restrict__ y16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = oc * oc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y16[at] = __half_as_ushort(__float2half_rn(v));
}

}  // namespace

uint64_t work_bytes(int64_t T) {
    const int64_t nsub = (T + CH - 1) / CH;
    return (uint64_t) nsub * ((uint64_t) HK * PKT + (uint64_t) HV * CTL) * sizeof(float);
}

void recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z, const float* gamma,
                float eps, void* work, float* y, uint16_t* y16, int64_t T, void* stream) {
    if (T <= 0) return;
    const cudaStream_t s = (cudaStream_t) stream;
    const int nsub = (int) ((T + CH - 1) / CH);
    float* kq = static_cast<float*>(work);
    float* ctl = kq + (int64_t) HK * nsub * PKT;
    basalt::kernels::launch_cc(gdn70_prep, dim3((unsigned) nsub, HK), NTH, 0, s, h, gate, beta, eps, (int) T, kq, ctl);
    // the shared-memory opt-in is per device
    static std::atomic<bool> attr[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) { basalt::log::error("basalt", "gdn70: device %d out of range", dev); std::abort(); }
    if (!attr[dev].load(std::memory_order_acquire)) {
        if (basalt::kernels::cc_func_set_attribute(gdn70_rec, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   (int) REC_SMEM) != cudaSuccess) {
            basalt::log::error("basalt", "gdn70: shared-memory opt-in failed");
            std::abort();
        }
        attr[dev].store(true, std::memory_order_release);
    }
    basalt::kernels::launch_cc(gdn70_rec, REC_BLOCKS, RT, REC_SMEM, s, state, h, (const float*) kq, (const float*) ctl, y,
                               (int) T);
    basalt::kernels::launch_cc(gdn70_out_norm, dim3((unsigned) T, HV), S, 0, s, z, gamma, eps, (const float*) y, y16);
}

}  // namespace basalt::prefill::gdn70
